# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""FLOPs-over-rate cost model: GQA attention and an MoE feed-forward under classic P2P.

One layer of a CP group of ``c`` ranks whose samples have ``n_i`` valid tokens takes, forward
and backward together, in milliseconds::

    T_layer = gamma * (3 * F_pair * W / P_attn + 3 * F_ff / P_moe) + N_x * t_x

    F_pair = 4 * H_q * d               forward FLOPs of one query-key pair
    W      = sum(n_i ** 2 / 2) / c     query-key pairs of one rank
    F_ff   = 2 * T * K * 3 * h * h_e   forward FLOPs of the MoE feed-forward of one rank
    N_x    = 2 * c, 0 at c = 1         ring steps of the P2P attention: c forward, c backward

A backward pass counts as twice the FLOPs of its forward pass, hence the factors 3. The
inputs have three sources:

- The training configuration (``options_from_config``): ``L`` layers (``num_layers``),
  ``H_q`` query heads (``num_attention_heads``) of dimension ``d`` (``kv_channels``), the
  hidden size ``h``, ``K`` experts per token (``moe_router_topk``) of width ``h_e``
  (``moe_ffn_hidden_size``).
- The planner geometry: ``T`` tokens per rank (``max_seqlen_per_dp_cp_rank``) and the
  expert-parallel size.
- The coefficient file: the rates ``P_attn`` and ``P_moe`` in TFLOPS, ``t_x`` in ms, the
  factor ``gamma`` on the compute terms and ``C_opt``, the ms added once per step.

Attention counts the valid tokens of the samples and the feed-forward the whole per-rank
buffer; padding has no term. The projections, the norms and the router have none either:
the rates and ``gamma`` absorb them.

``t_x`` is the fixed time of one ring step and is added in full. In Transformer Engine's
P2P attention a rank of a group of ``c`` ranks runs ``c`` steps forward and ``c`` backward;
a step prepares its queries, keys and values, calls the attention kernel on them and
corrects the results, and all but the first forward step are preceded by a K/V exchange
(``c - 1`` forward, ``c`` backward). A group of one rank calls the kernel once, outside that
function, and has no step. What a step costs is this work around the kernel, not the
transfer, so the term counts steps and not exchanges: in the layer times of the measured
H100 calibration (7.4, 8.8, 10.3 and 13.2 ms forward for 1, 2, 4 and 8 ranks) the forward
pass grows by 0.70 to 0.73 ms times ``c``. The transfer itself is issued before the kernel
of its step and awaited after it, and a form with ``max(exchange, kernel)`` fits the
measured H100 calibration worse.

The step time is the timeline's (``timing``): the attention of a group ends when its ranks
did, the feed-forward with the slowest rank of its expert-parallel block, no rank waits for
all others after the first forward pass, and ``C_opt`` is added once. Without expert
parallelism a group takes ``L * T_layer`` for a microbatch and starts it when its ranks are
free. With it a block takes ``L`` times the layer of its slowest group; the timeline is below
that by at most the backward attention of one layer per microbatch, which has no
feed-forward behind it.

The packaged rates are provisional: fitted to another cost model, not measured (``source``
of ``gqa_p2p_flops_h100.json``).
"""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import asdict
from functools import lru_cache
from pathlib import Path

from ..runtime.config import _unique_object
from ..runtime.timeline import layer_program
from .base import NVLINK_DOMAIN_SIZES, uses_p2p

_MODEL_NAME = "gqa_p2p_flops"
_SCOPE = "The GQA P2P FLOPs cost model"
# Raw bytes of the packaged rates ``gqa_p2p_flops_<hardware>.json``, by the ``hardware`` of
# the runtime file; ORIGIN.json records the same digests.
PACKAGED_SHA256 = {"h100": "d6bf7cf5fe5c09e1ee21ec0f012146be447534d8fed6a0ce30f44a9b536c8403"}
# Constructor options that are model sizes, by the training setting each is read from.
_SIZES = {
    "layer_count": "num_layers",
    "hidden_size": "hidden_size",
    "query_heads": "num_attention_heads",
    "head_dim": "kv_channels",
    "moe_topk": "moe_router_topk",
    "moe_ffn_hidden_size": "moe_ffn_hidden_size",
}
# The numbers of a coefficient file: name -> whether zero is a valid value.
_RATES = {
    "P_attn_tflops": False,
    "P_moe_tflops": False,
    "t_x_ms": True,
    "gamma": False,
    "C_opt_ms": True,
}
# A rate above this is one written in FLOPs per second: every prediction would be zero.
_MAX_TFLOPS = 1e6
_FORWARD, _BACKWARD = "attention:forward", "attention:backward"


def _rate(value, name, zero_allowed):
    if (
        type(value) not in (int, float)
        or not math.isfinite(value)
        or value < 0
        or (value == 0 and not zero_allowed)
    ):
        kind = "nonnegative" if zero_allowed else "positive"
        raise ValueError(f"rates.{name} must be a finite {kind} number, got {value!r}")
    if name.endswith("_tflops") and value > _MAX_TFLOPS:
        raise ValueError(
            f"rates.{name} is {value!r}; the unit is TFLOPS (1e12 FLOPs per second) and at most "
            f"{_MAX_TFLOPS:g} is accepted"
        )
    return float(value)


class GqaP2PFlopsCostModel:
    """Price a CP group from the FLOPs of its attention and feed-forward and five rates.

    The model sizes are constructor options (``options_from_config`` reads them from the
    training configuration); ``hardware`` selects the packaged rates and the NVLink domain,
    and ``calibration`` names a coefficient file that replaces the packaged one.
    """

    name = _MODEL_NAME
    # The price of a group does not depend on its ranks: every group lies inside one NVLink
    # domain (``_check_geometry``), where one ring step takes ``t_x``.
    placement_invariant = True

    @staticmethod
    def options_from_config(config) -> dict:
        """The constructor options that are settings of the training configuration."""
        experts = getattr(config, "num_moe_experts", None)
        if not experts:
            raise ValueError(
                f"{_SCOPE} prices an MoE feed-forward in every layer; this model has "
                f"num_moe_experts={experts!r}"
            )
        options = {}
        for option, setting in _SIZES.items():
            value = getattr(config, setting, None)
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{_SCOPE} reads {setting} from the training configuration, a positive "
                    f"integer; got {value!r}"
                )
            options[option] = value
        return options

    def __init__(
        self,
        geometry,
        *,
        hardware,
        layer_count,
        hidden_size,
        query_heads,
        head_dim,
        moe_topk,
        moe_ffn_hidden_size,
        calibration=None,
    ) -> None:
        self.geometry = geometry
        if not isinstance(hardware, str) or hardware not in NVLINK_DOMAIN_SIZES:
            raise ValueError(
                f"hardware must be one of {sorted(NVLINK_DOMAIN_SIZES)}, got {hardware!r}"
            )
        self.hardware = hardware
        self.nvlink_domain_size = NVLINK_DOMAIN_SIZES[hardware]
        self.sizes = dict(
            layer_count=layer_count,
            hidden_size=hidden_size,
            query_heads=query_heads,
            head_dim=head_dim,
            moe_topk=moe_topk,
            moe_ffn_hidden_size=moe_ffn_hidden_size,
        )
        for option, value in self.sizes.items():
            if type(value) is not int or value < 1:
                raise ValueError(
                    f"{option} ({_SIZES[option]} of the training configuration) must be a "
                    f"positive integer, got {value!r}"
                )
        self.layer_count = layer_count
        self._load_rates(calibration)
        self._check_geometry(geometry)
        rates = self.rates
        # Forward FLOPs of one query-key pair, and of the feed-forward of one rank.
        self.f_pair = 4 * query_heads * head_dim
        self.f_ff = 2 * geometry.tokens_per_rank * moe_topk * 3 * hidden_size * moe_ffn_hidden_size
        # Milliseconds of a forward pass: FLOPs / (TFLOPS * 1e12) seconds.
        self._pair_ms = rates["gamma"] * self.f_pair / (rates["P_attn_tflops"] * 1e9)
        feed_forward_ms = rates["gamma"] * self.f_ff / (rates["P_moe_tflops"] * 1e9)
        self.feed_forward_ms = (feed_forward_ms, 2 * feed_forward_ms)
        layer = (_FORWARD, feed_forward_ms, _BACKWARD, 2 * feed_forward_ms)
        self._timing = dict(
            execution_program=[layer_program([layer] * layer_count)],
            iteration_tail_ms=rates["C_opt_ms"],
            # Only the ranks of a CP group and of an expert-parallel block wait for each other.
            first_forward_dp_cp_sync=False,
        )

    def _load_rates(self, calibration):
        """Read the packaged rates of the hardware, or the coefficient file ``calibration``."""
        self.packaged = calibration is None
        if calibration is None:
            if self.hardware not in PACKAGED_SHA256:
                raise ValueError(
                    f"No rates of the {self.name!r} cost model are packaged for hardware "
                    f"{self.hardware!r} (packaged: {sorted(PACKAGED_SHA256)}); name a "
                    'coefficient file with "calibration"'
                )
            path = Path(__file__).with_name(f"{self.name}_{self.hardware}.json")
        elif isinstance(calibration, (str, Path)):
            path = Path(calibration)
        else:
            raise ValueError(f"calibration must be a coefficient file path, got {calibration!r}")
        try:
            raw = path.read_bytes()
            # A key given twice in one object is rejected, as in the runtime file.
            data = json.loads(raw, object_pairs_hook=_unique_object)
        except (OSError, ValueError, UnicodeError) as error:
            raise ValueError(f"Cannot read coefficient file {str(path)!r}: {error}") from error
        digest = hashlib.sha256(raw).hexdigest()
        if calibration is None and digest != PACKAGED_SHA256[self.hardware]:
            raise ValueError(f"Packaged rates {path.name!r} hash mismatch")
        self.calibration_path = str(path.resolve())
        self.calibration_sha256 = digest
        self.source_hashes = {path.name: digest}
        where = f"Coefficient file {self.calibration_path!r}"
        if (
            not isinstance(data, dict)
            or type(data.get("schema_version")) is not int
            or data["schema_version"] != 1
            or data.get("name") != self.name
        ):
            raise ValueError(f'{where} is not a {self.name!r} file with "schema_version": 1')
        if data.get("hardware") != self.hardware:
            raise ValueError(
                f"{where} holds rates of hardware {data.get('hardware')!r}; the runtime file "
                f"selects {self.hardware!r}"
            )
        rates = data.get("rates")
        if not isinstance(rates, dict) or set(rates) != set(_RATES):
            raise ValueError(f'{where}: "rates" takes exactly the keys {list(_RATES)}')
        try:
            self.rates = {name: _rate(rates[name], name, zero) for name, zero in _RATES.items()}
        except (ValueError, OverflowError) as error:
            raise ValueError(f"{where}: {error}") from error
        # Where the numbers come from: required, recorded with every plan and never read.
        self.source = data.get("source")
        if not isinstance(self.source, dict) or not self.source:
            raise ValueError(f'{where}: "source" must say how the rates were obtained')
        self.status = data.get("status")
        if self.status is not None and not isinstance(self.status, str):
            raise ValueError(f'{where}: "status" must be a string when given')

    def _check_geometry(self, geometry):
        """Every CP group of the run lies inside one NVLink domain, where ``t_x`` holds."""
        if geometry.tokens_per_rank != self.geometry.tokens_per_rank:
            raise ValueError(
                f"{_SCOPE} was built for max_seqlen_per_dp_cp_rank="
                f"{self.geometry.tokens_per_rank}, got {geometry.tokens_per_rank}"
            )
        if not geometry.align_group_starts:
            raise ValueError(f"{_SCOPE} covers size-aligned CP groups only")
        domain, sizes = self.nvlink_domain_size, sorted(geometry.allowed_cp_sizes)
        outside = [
            cp
            for cp in sizes
            if any(
                start // domain != (start + cp - 1) // domain
                for start in range(0, geometry.world_size - cp + 1, cp)
            )
        ]
        if outside:
            inside = [cp for cp in sizes if cp not in outside]
            remedy = (
                f'set "max_cp_size": {inside[-1]} in the runtime file'
                if inside
                else "no CP group size of this run avoids that"
            )
            raise ValueError(
                f"{_SCOPE} has one ring-step time, for ranks of one NVLink domain: CP groups of "
                f"{outside} ranks span more than one domain of {domain} ranks (hardware "
                f"{self.hardware!r}, DPxCP={geometry.world_size}); {remedy}"
            )

    def check_domain(self, geometry, config=None):
        """Reject a geometry or a training configuration that the formulas do not describe.

        Model sizes, the expert-parallel size and DPxCP are inputs and are not compared
        with the run in which the rates were obtained (``source`` of the coefficient file).
        """
        self._check_geometry(geometry)
        if config is None:
            return
        if not uses_p2p(config):
            raise ValueError(
                f"{_SCOPE} counts the ring steps of P2P context parallelism: "
                f"cp_comm_type='p2p', got {getattr(config, 'cp_comm_type', None)!r}"
            )
        partition = getattr(config, "cp_partition_mode", "zigzag")
        if partition != "zigzag":
            raise ValueError(
                f"{_SCOPE} gives every rank of a CP group the same share of the query-key "
                f"pairs: cp_partition_mode='zigzag', got {partition!r}"
            )
        for setting in ("tensor_model_parallel_size", "expert_tensor_parallel_size"):
            value = getattr(config, setting, None)
            if value not in (None, 1):
                raise ValueError(
                    f"{_SCOPE} takes a DPxCP rank as one GPU: {setting}=1, got {value}"
                )
        for setting in (
            "multi_latent_attention",
            "experimental_attention_variant",
            "linear_attention_type",
            "is_hybrid_model",
        ):
            value = getattr(config, setting, None)
            if value:
                raise ValueError(
                    f"{_SCOPE} prices ordinary attention in every layer; got {setting}={value!r}"
                )
        frequency = getattr(config, "moe_layer_freq", 1)
        values = frequency if isinstance(frequency, (list, tuple)) else [frequency]
        if not values or any(type(value) is not int or value != 1 for value in values):
            raise ValueError(
                f"{_SCOPE} prices an MoE feed-forward in every layer: moe_layer_freq=1, got "
                f"{frequency!r}"
            )
        recompute = getattr(config, "recompute_granularity", None)
        if recompute is not None:
            raise ValueError(
                f"{_SCOPE} counts one forward and one backward pass per layer: "
                f"recompute_granularity unset, got {recompute!r}"
            )

    def _check_ranks(self, cp, ranks):
        if ranks is None:
            return
        ranks, domain = tuple(ranks), self.nvlink_domain_size
        if (
            len(ranks) != cp
            or any(type(rank) is not int for rank in ranks)
            or ranks != tuple(range(ranks[0], ranks[0] + cp))
            or not self.geometry.supports_group(ranks[0], cp)
            or ranks[0] // domain != ranks[-1] // domain
        ):
            raise ValueError(f"{_SCOPE} prices an aligned CP group inside one NVLink domain")

    @lru_cache(maxsize=65536)
    def _attention(self, lengths, cp):
        """Forward and backward ms of the attention of one layer, ring steps included."""
        geometry = self.geometry
        if type(cp) is not int or cp not in geometry.allowed_cp_sizes:
            raise ValueError("CP size is outside the CP group sizes of this run")
        if not lengths or any(type(length) is not int or length < 1 for length in lengths):
            raise ValueError("A pack holds samples of a positive integer number of tokens")
        if sum(geometry.stored_length(length, cp) for length in lengths) > geometry.capacity(cp):
            raise ValueError("Pack exceeds the token capacity of its CP group")
        # Each rank of the zigzag partition holds 1 / cp of the pairs of every sample.
        forward = self._pair_ms * sum(length * length for length in lengths) / (2 * cp)
        # A group above one rank runs ``cp`` ring steps in either direction.
        steps = cp * self.rates["t_x_ms"] if cp > 1 else 0.0
        return forward + steps, 2 * forward + steps

    def forward(self, lengths, cp, *, ranks=None):
        """Forward attention ms per layer of one pack: its FLOPs and ``cp`` ring steps."""
        self._check_ranks(cp, ranks)
        return self._attention(tuple(lengths), cp)[0]

    def backward(self, lengths, cp, *, ranks=None):
        """Backward attention ms per layer of one pack: its FLOPs and ``cp`` ring steps."""
        self._check_ranks(cp, ranks)
        return self._attention(tuple(lengths), cp)[1]

    def program_costs(self, lengths, cp, *, ranks=None):
        """The attention times of one pack, by the names of the layer program."""
        self._check_ranks(cp, ranks)
        forward, backward = self._attention(tuple(lengths), cp)
        return {_FORWARD: forward, _BACKWARD: backward}

    guide_program_costs = program_costs

    def layer_ms(self, lengths, cp):
        """``T_layer`` of the module docstring: one layer of one pack, forward and backward."""
        return sum(self._attention(tuple(lengths), cp)) + sum(self.feed_forward_ms)

    def timing(self):
        """The layer program: attention per CP group, the feed-forward up to the EP block."""
        return self._timing

    def clear(self):
        """Drop pack prediction caches before planning another global batch."""
        self._attention.cache_clear()

    def metadata(self):
        """Report the rates, where they come from, and the model sizes they are applied to.

        With ``hardware``, ``model_sizes`` (the constructor options) and ``geometry`` a reader
        of the trace builds this model again: the calibration skill prices the recorded
        layouts that way.
        """
        return {
            "name": self.name,
            "hardware": self.hardware,
            "nvlink_domain_size": self.nvlink_domain_size,
            "source_hashes": dict(self.source_hashes),
            "calibration_path": self.calibration_path,
            "calibration_sha256": self.calibration_sha256,
            "packaged_rates": self.packaged,
            "status": self.status,
            "rates": dict(self.rates),
            "source": self.source,
            "model_sizes": dict(self.sizes),
            "geometry": asdict(self.geometry),
        }
