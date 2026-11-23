# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Time the transformer layers of a training step for the trace of the runtime DCP scheduler.

``"layer_times": true`` in the runtime file attaches these timers once the model is built.
Every rank then records, per microbatch, the time of all its transformer layers and of their
self-attention modules, forward and backward, and the scheduler writes the numbers of a step
into the trace record of the next one (``previous_layer_times``). The calibration skill of
the ``dynamic_usp_flops`` cost model reads them; training does not.

How a time is taken:

- Forward: a mark before and after the module (forward hooks).
- Backward: a mark when autograd reaches the module's output (a hook on the node that made
  the output) and when it reaches the module's input (a hook on the node that made the
  input). No backward hook is put on the module itself: that would wrap its output in a
  view, which the in-place dropout after attention and the feed-forward cannot modify.
- A mark is a CUDA event on the current stream, read once per step; nothing waits for the
  GPU inside a step. For a model on the CPU a mark is the host clock.

The time of a module includes the waits inside it: for the keys of the other ranks of a CP
group in attention, and for the ranks of an expert-parallel block in the feed-forward.
"""

from __future__ import annotations

import time
from functools import partial

import torch

_KINDS = ("layer_forward", "layer_backward", "attention_forward", "attention_backward")
# Microbatches kept until the scheduler collects them. A rank that reads no data (a
# tensor-parallel rank above 0) is never asked for its times: it keeps the latest ones only.
_MAX_PENDING = 256
_TIMER: "LayerTimer | None" = None


class LayerTimer:
    """Forward and backward time of the transformer layers of one model, per microbatch."""

    def __init__(self) -> None:
        self.cuda = False
        self.layers = 0
        self._records: list[dict] = []
        self._entered: dict[int, tuple] = {}

    # --------------------------------------------------------------------------- marks
    def _mark(self):
        if not self.cuda:
            return time.perf_counter()
        event = torch.cuda.Event(enable_timing=True)
        event.record()
        return event

    def _elapsed_ms(self, begin, end) -> float:
        return begin.elapsed_time(end) if self.cuda else (end - begin) * 1e3

    # --------------------------------------------------------------------------- hooks
    def attach(self, model) -> int:
        """Time every module of ``model`` that has a ``self_attention`` submodule."""
        if self.layers:
            raise RuntimeError("The layer timers are attached to one model; a second one was built")
        layers = [
            module
            for module in model.modules()
            if isinstance(getattr(module, "self_attention", None), torch.nn.Module)
        ]
        if not layers:
            raise ValueError("layer_times: the model has no transformer layer to time")
        graphs = getattr(getattr(layers[0], "config", None), "cuda_graph_impl", None)
        if graphs not in (None, "none", ""):
            raise ValueError(
                "layer_times cannot time layers that run as CUDA graphs "
                f"(cuda_graph_impl={graphs!r})"
            )
        self.cuda = any(parameter.is_cuda for parameter in layers[0].parameters())
        for layer in layers:
            for name, module in (("layer", layer), ("attention", layer.self_attention)):
                first = name == "layer" and layer is layers[0]
                module.register_forward_pre_hook(
                    partial(self._enter, name, first), with_kwargs=True
                )
                module.register_forward_hook(partial(self._leave, name), with_kwargs=True)
        self.layers = len(layers)
        return self.layers

    def _enter(self, name, first, module, args, kwargs):
        if not module.training or not torch.is_grad_enabled():
            return
        if first:
            # The first layer starts the forward pass of a new microbatch.
            del self._records[: len(self._records) - _MAX_PENDING + 1]
            self._records.append({kind: [] for kind in _KINDS})
        if not self._records:
            return
        source = args[0] if args else kwargs.get("hidden_states")
        if isinstance(source, (tuple, list)):
            source = source[0]
        # The node that made the input, as it is now: a later in-place write replaces it.
        self._entered[id(module)] = (getattr(source, "grad_fn", None), self._mark())

    def _leave(self, name, module, args, kwargs, output):
        entered = self._entered.pop(id(module), None)
        if entered is None:
            return
        record, (made_input, begin) = self._records[-1], entered
        record[f"{name}_forward"].append([begin, self._mark()])
        result = output[0] if isinstance(output, (tuple, list)) else output
        made_output = getattr(result, "grad_fn", None)
        if made_output is None or made_input is None:
            return  # Nothing to differentiate, or an input that no operation made.
        pair = [None, None]
        record[f"{name}_backward"].append(pair)
        # Autograd runs a node once all gradients of its output are known: the node of the
        # module's output starts the module's backward pass, the node of its input ends it.
        made_output.register_prehook(partial(self._backward, pair, 0))
        made_input.register_prehook(partial(self._backward, pair, 1))

    def _backward(self, pair, index, _grad_outputs):
        if pair[index] is None:
            pair[index] = self._mark()

    # ------------------------------------------------------------------------- results
    def collect(self) -> list[dict]:
        """The microbatches since the last call, in order.

        Per microbatch: ``<kind>_ms``, the sum over the layers, and ``<kind>_count``, the
        number of layers in that sum, for the kinds ``layer_forward``, ``layer_backward``,
        ``attention_forward`` and ``attention_backward``; and ``after_layers_ms``, from the
        end of the last layer's forward pass to the start of its backward pass (final norm,
        output layer and loss, both directions), None when one of the two was not marked.
        """
        if self.cuda:
            torch.cuda.synchronize()
        records, self._records = self._records, []
        self._entered.clear()
        out = []
        for record in records:
            row = {}
            for kind in _KINDS:
                pairs = [pair for pair in record[kind] if all(mark is not None for mark in pair)]
                row[f"{kind}_ms"] = sum(self._elapsed_ms(begin, end) for begin, end in pairs)
                row[f"{kind}_count"] = len(pairs)
            forward, backward = record["layer_forward"], record["layer_backward"]
            # Both lists are in forward order: their last entries are the last layer.
            turned = 0 < len(backward) == len(forward) and backward[-1][0] is not None
            row["after_layers_ms"] = (
                self._elapsed_ms(forward[-1][1], backward[-1][0]) if turned else None
            )
            out.append(row)
        return out


def attach_layer_timers(model) -> int:
    """Attach the timers of this process; returns the number of timed layers.

    ``model`` is the module of this rank, or the list of one module that training holds.
    """
    global _TIMER
    modules = list(model) if isinstance(model, (list, tuple)) else [model]
    if len(modules) != 1:
        raise ValueError(
            f"layer_times times one model per rank; this rank holds {len(modules)} model "
            "chunks (interleaved pipeline stages)"
        )
    if _TIMER is None:
        _TIMER = LayerTimer()
    return _TIMER.attach(modules[0])


def collect_layer_times() -> list[dict]:
    """Layer times of the microbatches since the last call (``LayerTimer.collect``)."""
    if _TIMER is None or not _TIMER.layers:
        raise RuntimeError(
            '"layer_times" is set in the runtime file, but no layer timer is attached: call '
            "attach_layer_timers(model) once the model is built, as "
            "megatron/training/training.py does"
        )
    return _TIMER.collect()
