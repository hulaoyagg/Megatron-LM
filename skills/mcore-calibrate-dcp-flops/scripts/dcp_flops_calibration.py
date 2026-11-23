#!/usr/bin/env python3
# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Fit and check the coefficients of scheduler type dynamic_usp_flops from the trace of a run.

A trace record of the runtime DCP scheduler holds the layout that a training step executed
and the moment the step received it. The measured time of a step is the time from its record
to the next one, less the time the next step waited for its plan. The predicted time of a
step is computed by the cost model and the timeline that training uses.

A run with "layer_times": true in the runtime file also records, on every rank and for every
microbatch, the time of its transformer layers and of their attention. Then every CP group of
every microbatch gives one measured layer time, and the fit uses those: a step gives a
hundred equations instead of one. Without them the fit uses the step times.

    fit       fit gamma and C_opt (--level gamma) or the four rates (--level rates); write a file
    check     compare the predictions of a file with the measured steps and layers: PASS or FAIL
    coverage  print the data arguments of a run whose samples need every CP group size
    selftest  run fit and check on synthetic traces with known coefficients

Needs numpy and the Megatron-LM checkout; no GPU, and torch is not imported.
"""

import argparse
import hashlib
import json
import re
import sys
import tempfile
from pathlib import Path
from types import SimpleNamespace

import numpy as np

MODEL_NAME = 'gqa_p2p_flops'
RATES = ('P_attn_tflops', 'P_moe_tflops', 't_x_ms', 'gamma', 'C_opt_ms')
LEVELS = ('gamma', 'rates')
# Defaults of the command line; SKILL.md says when to change them.
SKIP_STEPS = 10  # First training steps of a trace that are not used.
OUTLIER_FACTOR = 1.5  # A step this far from the median of measured / predicted is not used.
BAND = (0.8, 1.2)  # Accepted median of predicted / measured.
MIN_STEPS = {'gamma': 20, 'rates': 40}  # Fitted from step times: one equation per step.
MIN_STEPS_LAYERS = 20  # Fitted from layer times: one equation per CP group and microbatch.
SOURCES = ('auto', 'layers', 'steps')
LAYER_KINDS = ('layer_forward', 'layer_backward', 'attention_forward', 'attention_backward')
MAX_ITERATIONS = 30
MAX_CONDITION = 1e3  # Above it the steps are too alike to tell the fitted numbers apart.
RELATIVE_STEP = 1e-4  # Of the finite differences.
MIN_INVERSE_RATE = 1e-6  # 1 / TFLOPS; the cost model accepts rates up to 1e6 TFLOPS.
PASS, FAIL, ERROR = 0, 1, 2


class Fault(Exception):
    """A problem of the input; the command ends with its message."""


# ------------------------------------------------------------------------------ the cost model
def load_package(root):
    """The scheduler package of the checkout, imported by path: Megatron and torch stay out."""
    datasets = Path(root) / 'megatron' / 'core' / 'datasets'
    if not (datasets / 'dcp_runtime_scheduler' / 'models' / f'{MODEL_NAME}.py').is_file():
        raise Fault(
            f'{root} is not a Megatron-LM checkout with the {MODEL_NAME} cost model; name the '
            'checkout with --megatron'
        )
    if str(datasets) not in sys.path:
        sys.path.insert(0, str(datasets))
    from dcp_runtime_scheduler.models.gqa_p2p_flops import GqaP2PFlopsCostModel
    from dcp_runtime_scheduler.runtime.dynamic_usp import DynamicUSPRuntimePlanner
    from dcp_runtime_scheduler.runtime.planner import RuntimePlanGeometry, decode_groups
    from dcp_runtime_scheduler.runtime.timeline import build_timeline

    return SimpleNamespace(
        Model=GqaP2PFlopsCostModel,
        Planner=DynamicUSPRuntimePlanner,
        Geometry=RuntimePlanGeometry,
        decode_groups=decode_groups,
        build_timeline=build_timeline,
    )


def coefficient_file(hardware, rates, status, source):
    """The content of a coefficient file of the cost model."""
    return dict(
        schema_version=1,
        name=MODEL_NAME,
        hardware=hardware,
        status=status,
        rates={name: float(rates[name]) for name in RATES},
        source=source,
    )


class Pricer:
    """Predicted step times of recorded layouts, by the cost model and timeline of training."""

    def __init__(self, package, metadata):
        self.package = package
        try:
            geometry = dict(metadata['geometry'])
            geometry['allowed_cp_sizes'] = tuple(geometry['allowed_cp_sizes'])
            self.geometry = package.Geometry(**geometry)
            self.hardware, self.options = metadata['hardware'], dict(metadata['model_sizes'])
        except (KeyError, TypeError, ValueError) as error:
            raise Fault(
                'the trace does not record the geometry and model sizes of its cost model '
                f'({type(error).__name__}: {error}); it was written by another version of it'
            ) from error
        self.directory = tempfile.TemporaryDirectory(prefix='dcp_flops_')

    def model(self, rates):
        """The cost model with these rates, loaded from a file as training loads it."""
        path = Path(self.directory.name) / 'rates.json'
        content = coefficient_file(self.hardware, rates, 'candidate', dict(method='candidate'))
        path.write_text(json.dumps(content))
        try:
            return self.package.Model(
                self.geometry, hardware=self.hardware, calibration=str(path), **self.options
            )
        except ValueError as error:
            raise Fault(f'the cost model rejects the coefficients: {error}') from error

    def times(self, rates, steps):
        """Predicted milliseconds of every step under ``rates``."""
        model = self.model(rates)
        timeline, out = self.package.build_timeline(model, self.geometry), []
        for step in steps:
            try:
                groups = [
                    [
                        (start, cp, None, None, model.program_costs(lengths, cp))
                        for start, cp, lengths in row
                    ]
                    for row in step['groups']
                ]
            except ValueError as error:
                raise Fault(f"step {step['number']} of {step['trace']}: {error}") from error
            out.append(timeline.evaluate(groups))
        model.clear()
        return np.array(out)


# ------------------------------------------------------------------------------------ the trace
def consume_records(path):
    """The records of the steps of one rank file, in file order, and the bytes of the file."""
    raw, records, last = path.read_bytes(), [], {}
    lines = raw.decode('utf-8', 'replace').splitlines()
    for number, line in enumerate(lines, 1):
        if line.strip():
            try:
                record = json.loads(line)
            except ValueError as error:
                if number == len(lines):
                    break  # The run was stopped while it wrote its last record.
                raise Fault(f'{path}, line {number}: {error}') from error
            if record.get('event') == 'consume':
                # A run appends to the files it finds, and its step counters start at zero.
                iterator, counter = record.get('iterator'), record.get('sequence_number')
                if isinstance(counter, int) and counter <= last.get(iterator, -1):
                    raise Fault(
                        f'{path}, line {number}: the step counter starts again: the directory '
                        'holds the records of more than one run. Give every run a trace_dir of '
                        'its own, or keep only the lines of one run in every rank file'
                    )
                last[iterator] = counter
                records.append(record)
    return records, raw


def training_iterator(records):
    """The iterator with the most steps: evaluation iterators have a few."""
    counts = {}
    for record in records:
        counts[record.get('iterator')] = counts.get(record.get('iterator'), 0) + 1
    return max(counts, key=lambda iterator: (counts[iterator], -(iterator or 0)))


def read_layer_times(files, first, world):
    """Per rank, the layer times of every training step by its sequence number; or why not."""
    if not any(record.get('previous_layer_times') for record in first):
        return None, 'the run did not record them ("layer_times": true in the runtime file does)'
    ranks = [int(re.sub(r'\D', '', path.stem) or -1) for path in files]
    if ranks != list(range(world)):
        return None, (
            f'the trace holds the files of {len(files)} ranks; layer times are read from those '
            f'of ranks 0 to {world - 1}, the DPxCP ranks of a run with one pipeline stage'
        )
    times = []
    for index, path in enumerate(files):
        records = first if index == 0 else consume_records(path)[0]
        if not records:
            return None, f'{path.name} has no step'
        train, own, before = training_iterator(records), {}, None
        for record in records:
            # A record carries the layers of the step before it, whatever iterator reads next.
            if before is not None and record.get('previous_layer_times'):
                own[before] = record['previous_layer_times']
            before = record.get('sequence_number') if record.get('iterator') == train else None
        times.append(own)
    return times, None


def layers_of_step(number, microbatches, times, layer_count):
    """Attention, layer and after-layers ms per rank and microbatch of one step; or why not."""
    rows = [own.get(number) for own in times]
    if any(row is None for row in rows):
        return None, 'a rank has no layer times for the step'
    if any(len(row) != microbatches for row in rows):
        return None, 'a rank timed another number of forward passes than the step has microbatches'
    if any(
        entry.get(f'{kind}_count') != layer_count
        for row in rows
        for entry in row
        for kind in LAYER_KINDS
    ):
        return None, f'a rank did not time all {layer_count} layers in both directions'

    def table(kinds):
        return np.array(
            [[sum(entry[f'{kind}_ms'] for kind in kinds) for entry in row] for row in rows]
        )

    after = [[entry.get('after_layers_ms') for entry in row] for row in rows]
    return (
        dict(
            attention=table(LAYER_KINDS[2:]),
            whole=table(LAYER_KINDS[:2]),
            after=np.array(after, dtype=float),  # None, where it was not marked, becomes nan.
            kinds={kind: table((kind,)) for kind in LAYER_KINDS},
        ),
        None,
    )


def own_layer_times(step, geometry, layer_count):
    """What one layer of each CP group takes by itself, from the times its ranks measured.

    The ranks of a CP group exchange K/V in every attention, so they leave it together. A
    rank that was free before the others of its group waits for them inside its first
    attention; the rank that entered last waits for nobody, and its attention time, the
    smallest of the group, is the attention time of the group. With expert parallelism a
    layer also waits for the slowest group of its block, so every rank of a block measures
    about one layer time. The ranks of that slowest group do not wait in the feed-forward:
    their layer time less their attention time is the rest of the layer, which is the same
    for every rank of the block. The own layer time of a group is its attention time plus
    that rest. Without expert parallelism a block is one rank.
    """
    world, ep, layers = geometry.world_size, geometry.ep_size, step['layers']
    groups, blocks = [], []
    for microbatch, row in enumerate(step['groups']):
        attention, whole = layers['attention'][:, microbatch], layers['whole'][:, microbatch]
        of_group = np.empty(world)
        for start, cp, _ in row:
            of_group[start : start + cp] = attention[start : start + cp].min()
        rest = np.empty(world)
        for first in range(0, world, ep):
            block = slice(first, first + ep)
            slowest = of_group[block] >= of_group[block].max()
            rest[block] = (whole[block] - attention[block])[slowest].mean()
            blocks.append(float(whole[block].max() / whole[block].min()))
        for start, cp, lengths in row:
            ranks = slice(start, start + cp)
            groups.append(
                dict(
                    cp=cp,
                    lengths=lengths,
                    ms=float(of_group[start] + rest[ranks].mean()) / layer_count,
                    spread=float(attention[ranks].max() / attention[ranks].min()),
                )
            )
    return dict(
        groups=groups,
        block_spread=blocks,
        # The rank that spends the longest in its layers, waits for other ranks included.
        inside_ms=float(layers['whole'].sum(axis=1).max()),
    )


def read_trace(directory, package, skip):
    """The training steps of one trace directory and the facts of the run that wrote it."""
    files = sorted(Path(directory).glob('rank*.jsonl'))
    if not files:
        raise Fault(
            f'{directory} holds no rank*.jsonl file; "trace_dir" of the runtime file names it'
        )
    path = files[0]
    consumed, raw = consume_records(path)
    if not consumed:
        raise Fault(f'{path} has no training step')
    train = training_iterator(consumed)
    training = [record for record in consumed if record.get('iterator') == train]
    metadata = training[0].get('model_metadata') or {}
    if metadata.get('name') != MODEL_NAME:
        raise Fault(
            f'{path} was written by cost model {metadata.get("name")!r}; this skill calibrates '
            f'scheduler type dynamic_usp_flops (cost model {MODEL_NAME!r})'
        )
    digests = {
        (record.get('model_metadata') or {}).get('calibration_sha256') for record in training
    }
    if len(digests) != 1:
        raise Fault(f'{path}: the coefficient file changed during the run')
    steps, dropped, seen = [], dict(warm_up=0, interrupted=0), 0
    for this, after in zip(consumed, consumed[1:]):
        if this.get('iterator') != train:
            continue
        seen += 1
        if seen <= skip:
            dropped['warm_up'] += 1
            continue
        # An evaluation or a replayed step between two records makes their distance no step.
        if (
            after.get('iterator') != train
            or after.get('sequence_number') != this['sequence_number'] + 1
        ):
            dropped['interrupted'] += 1
            continue
        lengths = this['global_lengths']
        interval = (after['consume']['monotonic_ns'] - this['consume']['monotonic_ns']) / 1e6
        wait = 1e3 * float(after.get('residual_wait_s') or 0.0)
        groups = [
            [
                (start, cp, tuple(lengths[i] for i in ids))
                for start, cp, ids in package.decode_groups(row)
            ]
            for row in this['rows']
        ]
        steps.append(
            dict(
                trace=str(directory),
                number=this['sequence_number'],
                groups=groups,
                interval_ms=interval,
                wait_ms=wait,
                measured_ms=interval - wait,
                recorded_ms=float(this['predicted_ms']),
            )
        )
    # The layer times of every rank, where the run recorded them.
    world = (metadata.get('geometry') or {}).get('world_size')
    layer_count = (metadata.get('model_sizes') or {}).get('layer_count')
    times, missing = (None, 'the trace does not record the geometry of its cost model')
    if world and layer_count:
        times, missing = read_layer_times(files, consumed, world)
    with_layers = 0
    for step in steps:
        step['layers'] = None
        if times is not None:
            step['layers'], why = layers_of_step(
                step['number'], len(step['groups']), times, layer_count
            )
            missing, with_layers = missing or why, with_layers + (step['layers'] is not None)
    facts = dict(
        path=str(Path(directory).resolve()),
        file=path.name,
        sha256=hashlib.sha256(raw).hexdigest(),
        training_steps=len(training),
        dropped=dropped,
        metadata=metadata,
        layer_times=dict(rank_files=len(files), steps=with_layers, missing=missing),
    )
    return steps, facts


def collect(directories, package, skip, outlier, changed=False):
    """Usable steps of all traces, a pricer for them and one report line per trace.

    ``changed``: the formulas of the cost model changed after the runs. Their layouts and
    measured times are used; that this checkout prices a layout as the run did is not asked.
    """
    steps, traces, pricer, lines = [], [], None, []
    for directory in directories:
        own, facts = read_trace(directory, package, skip)
        metadata = facts['metadata']
        if pricer is None:
            pricer, first = Pricer(package, metadata), metadata
        elif any(
            metadata.get(key) != first.get(key) for key in ('hardware', 'model_sizes', 'geometry')
        ):
            raise Fault(
                f'{directory} is a run of another model, hardware or parallel layout than {directories[0]}'
            )
        if own:
            # The script must price a layout as the run did, or its fit describes another model.
            again = pricer.times(metadata['rates'], own)
            recorded = np.array([step['recorded_ms'] for step in own])
            worst = float(np.max(np.abs(again / recorded - 1)))
            if worst > 1e-6 and not changed:
                raise Fault(
                    f'{directory}: this checkout prices the recorded layouts differently from the '
                    f'run that wrote the trace (up to {100 * worst:.3g} %); use the checkout of the run'
                )
            # Steps with a checkpoint or another one-off are far from the others: not used.
            ratio = np.array([step['measured_ms'] for step in own]) / again
            centre = float(np.median(ratio))
            keep = (ratio > 0) & (ratio <= centre * outlier) & (ratio >= centre / outlier)
            facts['dropped']['outlier'] = int((~keep).sum())
            own = [step for step, kept in zip(own, keep) if kept]
        for step in own:
            step['own'] = None
            if step['layers'] is not None:
                step['own'] = own_layer_times(step, pricer.geometry, pricer.options['layer_count'])
        facts['used'] = len(own)
        dropped = ', '.join(
            f'{count} {name.replace("_", "-")}' for name, count in facts['dropped'].items()
        )
        lines.append(
            f"trace {directory} ({facts['file']}): {facts['training_steps']} training steps, "
            f"{len(own)} used; not used: {dropped}, 1 last"
        )
        timed = sum(step['own'] is not None for step in own)
        lines.append(
            f"    layer times: of {timed} of the {len(own)} used steps, from "
            f"{facts['layer_times']['rank_files']} rank files"
            + (f" ({facts['layer_times']['missing']})" if timed < len(own) else '')
        )
        steps += own
        traces.append(facts)
    if changed:
        lines.append(
            '--model-changed: the predictions that the runs recorded are not compared with those '
            'of this checkout; the layouts and the measured times of the traces are used'
        )
    waits = [step['wait_ms'] for step in steps]
    if waits:
        lines.append(
            f'waiting for the plan, subtracted from every step: median {np.median(waits):.1f} ms, largest {max(waits):.1f} ms'
        )
    return steps, traces, pricer, lines


# -------------------------------------------------------------------------------------- the fit
def to_vector(rates):
    """(gamma / P_attn, gamma / P_moe, t_x, C_opt): the step time is linear in these on one path."""
    return np.array(
        [
            rates['gamma'] / rates['P_attn_tflops'],
            rates['gamma'] / rates['P_moe_tflops'],
            rates['t_x_ms'],
            rates['C_opt_ms'],
        ]
    )


def to_rates(vector, gamma=1.0):
    """Rates with this ``gamma`` that give the vector."""
    inverse = np.maximum(vector[:2], MIN_INVERSE_RATE * gamma)
    return dict(
        P_attn_tflops=gamma / inverse[0],
        P_moe_tflops=gamma / inverse[1],
        t_x_ms=float(vector[2]),
        gamma=float(gamma),
        C_opt_ms=float(vector[3]),
    )


def linearize(pricer, vector, steps):
    """Step times and their derivatives by the four numbers, from the timeline itself.

    The step time is a maximum of sums of terms that are each one of the numbers times a
    constant, so near ``vector`` it is ``derivatives @ vector``: the sum along the critical
    path. A finite difference of the timeline gives the derivative without knowing the path.
    """
    base = pricer.times(to_rates(vector), steps)
    columns = []
    for index, floor in enumerate((1e-9, 1e-9, 1e-6)):
        delta = max(abs(vector[index]) * RELATIVE_STEP, floor)
        moved = vector.copy()
        moved[index] += delta
        columns.append((pricer.times(to_rates(moved), steps) - base) / delta)
    columns.append(np.ones(len(steps)))  # C_opt is added once to every step.
    return base, np.stack(columns, axis=1)


def nonnegative_least_squares(matrix, target):
    """Least squares with no negative coefficient, exact for a handful of columns.

    The optimum is the ordinary least-squares solution on the columns it uses, so the best
    subset of columns whose solution has no negative entry is the optimum.
    """
    scale = np.sqrt((matrix**2).mean(axis=0))
    scale[scale == 0] = 1.0
    scaled, best = matrix / scale, (float((target**2).sum()), (), ())
    for mask in range(1, 2 ** matrix.shape[1]):
        columns = [j for j in range(matrix.shape[1]) if mask >> j & 1]
        solution = np.linalg.lstsq(scaled[:, columns], target, rcond=None)[0]
        if (solution < 0).any():
            continue
        residual = float(((scaled[:, columns] @ solution - target) ** 2).sum())
        if residual < best[0]:
            best = residual, columns, solution
    out = np.zeros(matrix.shape[1])
    out[list(best[1])] = best[2]
    return out / scale, float(np.linalg.cond(scaled))


def fit(pricer, steps, start, level):
    """Fit on ``steps`` from the rates ``start``; returns the rates and a record of the fit."""
    target = np.array([step['measured_ms'] for step in steps])
    published, vector = to_vector(start), to_vector(start)
    names = ('P_attn_tflops', 'P_moe_tflops', 't_x_ms', 'C_opt_ms')
    for iteration in range(1, MAX_ITERATIONS + 1):
        _, derivatives = linearize(pricer, vector, steps)
        if level == 'gamma':
            # One factor on the two compute terms; the ring-step time stays as published.
            compute = derivatives[:, :2] @ published[:2]
            matrix = np.stack([compute, derivatives[:, 3]], axis=1)
            (factor, constant), condition = nonnegative_least_squares(
                matrix, target - derivatives[:, 2] * published[2]
            )
            new = np.array([factor * published[0], factor * published[1], published[2], constant])
        else:
            new, condition = nonnegative_least_squares(derivatives, target)
        if condition > MAX_CONDITION:
            raise Fault(
                f'the steps are too alike to fit level {level} (condition number {condition:.0f}): '
                'they need different numbers of microbatches and, for level rates, different CP '
                'group sizes and sample lengths. Add steps, or a coverage run (command coverage)'
            )
        # A compute cost of zero cannot be a result: the measured steps do not follow the model.
        if level == 'gamma' and new[1] <= 0:
            raise Fault(
                'the fit gives gamma = 0: the measured steps do not grow with the predicted '
                'compute. Check that the trace is of a training run on GPUs'
            )
        if min(new[0], new[1]) < MIN_INVERSE_RATE:
            missing = names[0] if new[0] < MIN_INVERSE_RATE else names[1]
            raise Fault(
                f'the steps do not determine {missing} (fitted cost zero); add steps or a coverage '
                'run (command coverage), or fit --level gamma'
            )
        # On an unchanged critical path the linear prediction is the step time at the new numbers.
        gap = float(np.max(np.abs(derivatives @ new / pricer.times(to_rates(new), steps) - 1)))
        vector = new
        if gap < 1e-7:
            break
    else:
        raise Fault(
            f'the fit did not settle in {MAX_ITERATIONS} iterations (last change {gap:.2g})'
        )
    if level == 'gamma':
        factor = vector[1] / published[1]
        rates = dict(start, gamma=float(start['gamma'] * factor), C_opt_ms=float(vector[3]))
    else:
        rates = to_rates(vector)
    record = dict(
        level=level,
        steps=len(steps),
        iterations=iteration,
        condition_number=round(condition, 1),
        fitted_at_zero=[name for name, value in zip(names, vector) if value <= 0],
    )
    return rates, record


def layer_matrix(pricer, vector, groups):
    """Derivatives of the layer time of every group by (gamma / P_attn, gamma / P_moe, t_x).

    The layer time is the sum of the three numbers times constants, so it is ``matrix @
    vector[:3]`` exactly. The constants are taken from the cost model by finite differences.
    """

    def layer(point):
        model = pricer.model(to_rates(np.append(point, 0.0)))
        return np.array([model.layer_ms(group['lengths'], group['cp']) for group in groups])

    base, columns = layer(vector[:3]), []
    for index, floor in enumerate((1e-9, 1e-9, 1e-6)):
        delta = max(abs(vector[index]) * RELATIVE_STEP, floor)
        moved = vector[:3].copy()
        moved[index] += delta
        columns.append((layer(moved) - base) / delta)
    return np.stack(columns, axis=1)


def fit_layers(pricer, steps, start, level, outlier=OUTLIER_FACTOR):
    """Fit on the own layer times of the CP groups of ``steps``, then C_opt on their step times."""
    groups = [group for step in steps for group in step['own']['groups']]
    target = np.array([group['ms'] for group in groups])
    published = to_vector(start)
    matrix = layer_matrix(pricer, published, groups)
    names = ('P_attn_tflops', 'P_moe_tflops', 't_x_ms')

    def solve(matrix, target):
        if level == 'gamma':
            # One factor on the two compute terms; the ring-step time stays as published.
            compute = matrix[:, :2] @ published[:2]
            (factor,), condition = nonnegative_least_squares(
                compute[:, None], target - matrix[:, 2] * published[2]
            )
            if factor <= 0:
                raise Fault(
                    'the fit gives gamma = 0: the measured layers take no longer than their '
                    'published ring steps. Check that the trace is of a training run on GPUs'
                )
            return np.array([factor * published[0], factor * published[1], published[2]]), condition
        new, condition = nonnegative_least_squares(matrix, target)
        if condition > MAX_CONDITION:
            raise Fault(
                f'the CP groups are too alike to fit level rates (condition number '
                f'{condition:.0f}): they need different sizes and sample lengths. Add a coverage '
                'run (command coverage), or fit --level gamma'
            )
        if min(new[0], new[1]) < MIN_INVERSE_RATE:
            missing = names[0] if new[0] < MIN_INVERSE_RATE else names[1]
            raise Fault(
                f'the layer times do not determine {missing} (fitted cost zero); add a coverage '
                'run (command coverage), or fit --level gamma'
            )
        return new, condition

    # A pause inside a layer (garbage collection of the interpreter, for example) makes the
    # layers of one microbatch several times longer on every rank. Such layer times are far
    # from the others, as a step that saves a checkpoint is among the steps: not used.
    keep = np.ones(len(groups), dtype=bool)
    for _ in range(MAX_ITERATIONS):
        new, condition = solve(matrix[keep], target[keep])
        ratio = target / (matrix @ new)
        centre = float(np.median(ratio[keep]))
        again = (ratio <= centre * outlier) & (ratio >= centre / outlier)
        if (again == keep).all():
            break
        keep = again
    if level == 'gamma':
        rates = dict(start, gamma=float(start['gamma'] * new[1] / published[1]), C_opt_ms=0.0)
    else:
        rates = to_rates(np.append(new, 0.0))
    # What a step takes beyond its layers, once per step.
    beyond = np.array([step['measured_ms'] for step in steps]) - pricer.times(rates, steps)
    rates['C_opt_ms'] = max(0.0, float(beyond.mean()))
    record = dict(
        level=level,
        fitted_from='layer times',
        steps=len(steps),
        groups=int(keep.sum()),
        groups_not_used=int((~keep).sum()),
        iterations=1,
        condition_number=round(condition, 1),
        fitted_at_zero=[name for name, value in zip(names, new) if value <= 0]
        + (['C_opt_ms'] if rates['C_opt_ms'] <= 0 else []),
    )
    return rates, record


def judge_layers(pricer, rates, steps):
    """Predicted / measured of one layer of every CP group of ``steps``, overall and by size."""
    groups = [group for step in steps for group in step['own']['groups']]
    model = pricer.model(rates)
    predicted = np.array([model.layer_ms(group['lengths'], group['cp']) for group in groups])
    measured = np.array([group['ms'] for group in groups])
    ratio, sizes = predicted / measured, np.array([group['cp'] for group in groups])
    return dict(
        groups=len(groups),
        steps=len(steps),
        median=round(float(np.median(ratio)), 4),
        p10=round(float(np.quantile(ratio, 0.1)), 4),
        p90=round(float(np.quantile(ratio, 0.9)), 4),
        by_cp_size={
            int(cp): dict(
                groups=int((sizes == cp).sum()),
                median=round(float(np.median(ratio[sizes == cp])), 4),
                measured_ms=round(float(np.median(measured[sizes == cp])), 3),
            )
            for cp in sorted(set(sizes.tolist()))
        },
    )


def describe_layers(summary):
    sizes = ', '.join(
        f"CP{cp} {entry['median']:.3f} ({entry['groups']} groups, {entry['measured_ms']:.2f} ms)"
        for cp, entry in summary['by_cp_size'].items()
    )
    return (
        f"one layer of a CP group, predicted / measured, {summary['groups']} groups of "
        f"{summary['steps']} steps: median {summary['median']:.3f}, 10th to 90th percentile "
        f"{summary['p10']:.3f} to {summary['p90']:.3f}\n    by group size (groups, median "
        f"measured layer): {sizes}"
    )


def layer_facts(pricer, steps, lines):
    """What the ranks measured, beside what the cost model takes for granted."""
    geometry = pricer.geometry
    groups = [group for step in steps for group in step['own']['groups']]
    spread = [group['spread'] for group in groups if group['cp'] > 1]
    if spread:
        lines.append(
            'attention time of the ranks of one CP group, largest / smallest: median '
            f'{np.median(spread):.3f}, largest {max(spread):.3f}   (above 1: ranks that were '
            'free before their group waited for it; the smallest is the time of the group)'
        )
    if geometry.ep_size > 1:
        blocks = [value for step in steps for value in step['own']['block_spread']]
        lines.append(
            'layer time of the ranks of one expert-parallel block, largest / smallest: median '
            f'{np.median(blocks):.3f}, largest {max(blocks):.3f}   (the cost model: a block '
            'advances together, 1.0)'
        )
    measured = np.array([step['measured_ms'] for step in steps])
    inside = np.array([step['own']['inside_ms'] for step in steps])
    after = np.concatenate([step['layers']['after'].ravel() for step in steps])
    after = float(np.nanmedian(after)) if np.isfinite(after).any() else None
    microbatches = float(np.median([len(step['groups']) for step in steps]))
    lines.append(
        f'the timed layers take {100 * np.median(inside / measured):.1f} % of a step (median); '
        f'outside them a step spends {np.median(measured - inside):.1f} ms'
        + (
            ''
            if after is None
            else f', of which {after:.2f} ms per microbatch after the last layer (final norm, '
            f'output layer, loss; {microbatches:g} microbatches per step)'
        )
        + '. The cost model has one constant per step for all of it: C_opt'
    )
    return dict(
        cp_group_attention_spread=round(float(np.median(spread)), 4) if spread else None,
        share_of_step_in_layers=round(float(np.median(inside / measured)), 4),
        outside_layers_ms=round(float(np.median(measured - inside)), 2),
        after_layers_ms_per_microbatch=after,
    )


def choose_source(wanted, steps, traces, lines):
    """``layers`` when every used step has the layer times of every rank, else ``steps``."""
    if wanted == 'steps':
        return 'steps'
    without = sum(step['own'] is None for step in steps)
    if steps and not without:
        return 'layers'
    reasons = [trace['layer_times']['missing'] for trace in traces]
    reason = next((reason for reason in reasons if reason), 'no step is left')
    text = f'{without} of the {len(steps)} used steps have no layer times: {reason}'
    if wanted == 'layers':
        raise Fault(text)
    lines.append(f'fitted from step times: {text}')
    return 'steps'


# ---------------------------------------------------------------------------------- the verdict
def judge(pricer, rates, steps, band):
    """Predicted / measured of ``steps`` under ``rates``, with the verdict on its median."""
    ratio = pricer.times(rates, steps) / np.array([step['measured_ms'] for step in steps])
    median = float(np.median(ratio))
    return dict(
        steps=len(steps),
        median=round(median, 4),
        p10=round(float(np.quantile(ratio, 0.1)), 4),
        p90=round(float(np.quantile(ratio, 0.9)), 4),
        min=round(float(ratio.min()), 4),
        max=round(float(ratio.max()), 4),
        share_in_band=round(float(np.mean((ratio >= band[0]) & (ratio <= band[1]))), 3),
        band=list(band),
        verdict='PASS' if band[0] <= median <= band[1] else 'FAIL',
    )


def describe(summary):
    low, high = summary['band']
    return (
        f"predicted / measured on {summary['steps']} steps: median {summary['median']:.3f}, "
        f"10th to 90th percentile {summary['p10']:.3f} to {summary['p90']:.3f}, range "
        f"{summary['min']:.3f} to {summary['max']:.3f}; {100 * summary['share_in_band']:.0f} % of "
        f"the steps inside {low:g} to {high:g}"
    )


def show_rates(rates):
    return (
        f"P_attn {rates['P_attn_tflops']:.1f} TFLOPS, P_moe {rates['P_moe_tflops']:.2f} TFLOPS, "
        f"t_x {rates['t_x_ms']:.3f} ms, gamma {rates['gamma']:.3f}, C_opt {rates['C_opt_ms']:.1f} ms"
    )


def coverage_arguments(geometry):
    """Data arguments of a run whose samples need every CP group size the scheduler may form."""
    tokens, largest = geometry.tokens_per_rank, max(geometry.allowed_cp_sizes)
    longest = largest * tokens
    mock = dict(
        mode='distribution',
        type='lognormal',
        format='thd',
        min_seq_len=64,
        max_seq_len=longest,
        mean_seq_len=tokens,
        lognormal_sigma=1.2,
    )
    return (
        f'--use-varlen-dataset --mock-data --seq-length {longest} '
        f"--varlen-mock-dataset-config-json '{json.dumps(mock, separators=(',', ':'))}'"
    )


def sizes_formed(steps, geometry):
    """CP group sizes of the layouts, and the allowed sizes that no layout has."""
    counts = {}
    for step in steps:
        for row in step['groups']:
            for _, cp, _ in row:
                counts[cp] = counts.get(cp, 0) + 1
    missing = [cp for cp in sorted(geometry.allowed_cp_sizes) if cp not in counts]
    formed = ', '.join(f'CP{cp} x {count}' for cp, count in sorted(counts.items()))
    return formed, missing


def advise_coverage(steps, pricer, lines):
    formed, missing = sizes_formed(steps, pricer.geometry)
    lines.append(f'CP groups in these layouts: {formed}')
    if missing:
        lines.append(
            f'no layout has a CP group of {missing} ranks, which the scheduler may form: the '
            'ring-step time is not checked at those sizes. A coverage run adds them: the training '
            f'launch with these data arguments, 60 steps:\n    {coverage_arguments(pricer.geometry)}'
        )
    return missing


def compare_with_log(log, steps, skip, lines):
    """The logged time of the used steps against the time between their trace records.

    Both clocks see every part of a step, in another order: the log closes an iteration after
    its training step, the trace when the next step receives its layout. Something that
    happens once in a few steps between the two (a manual garbage collection, for example)
    is in neighbouring steps of the two clocks. So the sums over the same steps are compared,
    not the medians.
    """
    pattern = re.compile(
        r'iteration\s+(\d+)\s*/\s*\d+\s*\|.*?elapsed time per iteration \(ms\): ([0-9.]+)'
    )
    text = Path(log).read_text(errors='replace')
    logged = {int(match.group(1)): float(match.group(2)) for match in pattern.finditer(text)}
    if not logged or not steps:
        lines.append(f'{log}: no "elapsed time per iteration" line, or no used step; not compared')
        return None
    iterations = sorted(logged)
    if iterations != list(range(iterations[0], iterations[0] + len(iterations))):
        # One line per several iterations holds their mean: only typical values can be compared.
        values = [logged[iteration] for iteration in iterations if iteration > iterations[0] + skip]
        if not values:
            lines.append(
                f'{log}: no "elapsed time per iteration" line after the first {skip} steps'
            )
            return None
        ratio = float(np.median([step['interval_ms'] for step in steps]) / np.median(values))
        basis = 'medians; the log has one line per several iterations'
    else:
        # The first logged iteration is the first step of the run, whose step counter is 0.
        pairs = [
            (step['interval_ms'], logged[iterations[0] + step['number']])
            for step in steps
            if iterations[0] + step['number'] in logged
        ]
        if not pairs:
            lines.append(f'{log}: no logged iteration belongs to a used step; not compared')
            return None
        ratio = float(sum(pair[0] for pair in pairs) / sum(pair[1] for pair in pairs))
        basis = f'sums over the same {len(pairs)} steps'
    lines.append(
        f'time between trace records / logged time per iteration ({basis}): {ratio:.3f}'
        + (
            ''
            if abs(ratio - 1) <= 0.02
            else '  <- differ by more than 2 %: do not fit from this trace before this is explained'
        )
    )
    return ratio


# ------------------------------------------------------------------------------------- commands
def command_fit(args, package):
    steps, traces, pricer, lines = collect(
        args.trace, package, args.skip, args.outlier, args.changed
    )
    source = choose_source(args.source, steps, traces, lines)
    need = MIN_STEPS_LAYERS if source == 'layers' else MIN_STEPS[args.level]
    if len(steps) < need:
        raise Fault(
            f'level {args.level} from {source[:-1]} times needs at least {need} usable steps; '
            f'the traces give {len(steps)}'
        )
    start = traces[0]['metadata']['rates']
    if args.level == 'gamma' and any(trace['metadata']['rates'] != start for trace in traces):
        raise Fault(
            'the traces ran with different coefficient files; level gamma keeps the rates of one file'
        )
    training, held_out = steps[0::2], steps[1::2]
    layers = None
    if source == 'layers':
        facts = layer_facts(pricer, steps, lines)
        rates, record = fit_layers(pricer, training, start, args.level, args.outlier)
        layers = dict(judge_layers(pricer, rates, held_out), **facts)
        method = (
            'least squares on the measured layer times of the CP groups of training steps; '
            'C_opt is the mean of what the step times leave'
        )
    else:
        rates, record = fit(pricer, training, start, args.level)
        record['fitted_from'] = 'step times'
        method = 'least squares on the measured step times of training steps'
    summary = judge(pricer, rates, held_out, (args.low, args.high))
    source = dict(
        method=f'dcp_flops_calibration.py fit --level {args.level}: {method}',
        started_from=dict(
            rates=start, calibration_sha256=traces[0]['metadata'].get('calibration_sha256')
        ),
        selection=dict(
            skip=args.skip,
            outlier=args.outlier,
            fitted='even positions of the used steps',
            held_out='odd positions',
        ),
        traces=[
            {
                key: trace[key]
                for key in ('path', 'file', 'sha256', 'training_steps', 'used', 'dropped')
            }
            for trace in traces
        ],
        fit=record,
        held_out=summary,
        held_out_layers=layers,
        recipe=dict(model_sizes=pricer.options, geometry=traces[0]['metadata']['geometry']),
    )
    content = coefficient_file(pricer.hardware, rates, 'fitted', source)
    out = Path(args.out)
    out.write_text(json.dumps(content, indent=2) + '\n')
    pricer.package.Model(
        pricer.geometry, hardware=pricer.hardware, calibration=str(out), **pricer.options
    )
    lines.append(f'before: {show_rates(start)}')
    basis = (
        f'{record["groups"]} layer times of {record["steps"]} steps'
        if layers
        else f'{record["steps"]} step times, {record["iterations"]} iterations'
    )
    lines.append(
        f'after:  {show_rates(rates)}   (level {args.level}, {basis}, condition number {record["condition_number"]})'
    )
    if record.get('groups_not_used'):
        share = record['groups_not_used'] / (record['groups'] + record['groups_not_used'])
        lines.append(
            f'{record["groups_not_used"]} more layer times were not used in the fit: more than '
            f'{args.outlier:g} times away from the median of measured / fitted (a pause inside a '
            'layer, such as garbage collection; its time stays in the step times and in C_opt)'
            + (
                ''
                if share <= 0.1
                else f'  <- {100 * share:.0f} % of the layer times, more than pauses explain: '
                'the cost model misses these layers; see the ratios by group size'
            )
        )
    if record['fitted_at_zero']:
        lines.append(
            f'fitted at zero: {record["fitted_at_zero"]}: the steps show no such cost; say so when reporting'
        )
    if layers:
        lines.append('held out, ' + describe_layers(layers))
    lines.append('held out, ' + describe(summary))
    missing = advise_coverage(steps, pricer, lines) if args.level == 'rates' else []
    lines.append(f'wrote {out}')
    passed = summary['verdict'] == 'PASS'
    if passed:
        action = f'set "calibration" of the runtime file to {out}'
        if missing:
            action += '; for rates to publish, add the coverage run and fit again with both traces'
    elif args.level == 'gamma':
        action = 'fit again with --level rates'
    else:
        action = 'stop and report: the cost model does not describe these steps'
    lines.append(
        f"VERDICT: {summary['verdict']} (median {summary['median']:.3f}, accepted {args.low:g} to {args.high:g}). Next: {action}"
    )
    print('\n'.join(lines))
    return PASS if passed else FAIL


def command_check(args, package):
    steps, traces, pricer, lines = collect(
        args.trace, package, args.skip, args.outlier, args.changed
    )
    if not steps:
        raise Fault('the traces give no usable step')
    if args.calibration is None:
        rates, label = traces[0]['metadata']['rates'], 'of the run'
        if any(trace['metadata']['rates'] != rates for trace in traces):
            raise Fault(
                'the traces ran with different coefficient files; name one with --calibration'
            )
    else:
        try:
            content = json.loads(Path(args.calibration).read_text())
            rates, label = content['rates'], str(args.calibration)
            fitted_on = {
                trace.get('sha256') for trace in content.get('source', {}).get('traces', [])
            }
        except (OSError, ValueError, KeyError, AttributeError) as error:
            raise Fault(f'cannot read {args.calibration}: {error}') from error
        if fitted_on & {trace['sha256'] for trace in traces}:
            # The file was fitted on the even positions of these steps: judge the others.
            steps = steps[1::2]
            lines.append(
                'the file was fitted on this trace: judged on the steps the fit did not use'
            )
    summary = judge(pricer, rates, steps, (args.low, args.high))
    lines.append(f'coefficients {label}: {show_rates(rates)}')
    if all(step['own'] is not None for step in steps):
        layer_facts(pricer, steps, lines)
        lines.append(describe_layers(judge_layers(pricer, rates, steps)))
    lines.append(describe(summary))
    advise_coverage(steps, pricer, lines)
    if args.log:
        first = str(args.trace[0])
        compare_with_log(
            args.log, [step for step in steps if step['trace'] == first], args.skip, lines
        )
    passed = summary['verdict'] == 'PASS'
    action = (
        'keep this coefficient file'
        if passed
        else 'fit: dcp_flops_calibration.py fit --trace ... --out ...'
    )
    lines.append(
        f"VERDICT: {summary['verdict']} (median {summary['median']:.3f}, accepted {args.low:g} to {args.high:g}). Next: {action}"
    )
    print('\n'.join(lines))
    return PASS if passed else FAIL


def command_coverage(args, package):
    _, traces, pricer, _ = collect(args.trace, package, args.skip, args.outlier, args.changed)
    print(coverage_arguments(pricer.geometry))
    return PASS


# ------------------------------------------------------------------------------------- selftest
def synthetic_layer_times(pricer, model, rows, rng, noise, after, pause=0.0):
    """What the layer timers of every rank record for one step that ``model`` times.

    Every rank is free when the step starts. A CP group enters its first attention when the
    last of its ranks is free; the others wait inside that attention. With expert parallelism
    every layer then waits for the slowest attention of the block, except after the backward
    attention of the first layer, where nothing is exchanged any more: the ranks of a block
    become free at different times. With probability ``pause`` the forward pass of a
    microbatch stops for 500 ms on every rank, as when the interpreter collects garbage.
    Returns the entries per microbatch and rank, and the time until the last rank is free.
    """
    geometry, count = pricer.geometry, pricer.options['layer_count']
    world, ep = geometry.world_size, geometry.ep_size
    rest_forward, rest_backward = model.feed_forward_ms
    free, out = np.zeros(world), []
    for row in rows:
        forward, backward, begin = np.empty(world), np.empty(world), np.empty(world)
        for start, cp, lengths in row:
            ranks = slice(start, start + cp)
            forward[ranks], backward[ranks] = model.forward(lengths, cp), model.backward(
                lengths, cp
            )
            begin[ranks] = free[ranks].max()
        entries, after_this = [], np.empty(world)
        stop = 500.0 if pause and rng.random() < pause else 0.0
        for first in range(0, world, ep):
            block = slice(first, first + ep)
            # The block meets in the feed-forward of the first layer and of every layer after it.
            met = (begin[block] + forward[block]).max()
            turned = met + rest_forward + (count - 1) * (forward[block].max() + rest_forward)
            turned += stop
            for rank in range(first, first + ep):
                whole_backward = (
                    count * rest_backward + (count - 1) * backward[block].max() + backward[rank]
                )
                jitter = 1 + noise * rng.standard_normal(5)
                entries.append(
                    dict(
                        layer_forward_ms=(turned - free[rank]) * jitter[0],
                        layer_backward_ms=whole_backward * jitter[1],
                        attention_forward_ms=(begin[rank] - free[rank] + count * forward[rank])
                        * jitter[2],
                        attention_backward_ms=count * backward[rank] * jitter[3],
                        after_layers_ms=after * jitter[4],
                        **{f'{kind}_count': count for kind in LAYER_KINDS},
                    )
                )
                after_this[rank] = turned + after + whole_backward
        free = after_this
        out.append(entries)
    return out, float(free.max())


def write_trace(
    directory,
    package,
    pricer,
    run_rates,
    true_rates,
    steps=70,
    seed=0,
    noise=0.004,
    layer_times=False,
    after=0.0,
    pause=0.0,
):
    """A trace of ``steps`` training steps planned with ``run_rates`` and timed by ``true_rates``.

    With ``layer_times`` every rank writes a file with the times of its layers, a step
    spends ``after`` ms per microbatch after the last layer, and a microbatch stops for a
    pause with probability ``pause``. Returns the mean number of microbatches of a step.
    """
    rng = np.random.default_rng(seed)
    geometry = pricer.geometry
    model = pricer.model(run_rates)
    planner = package.Planner(geometry, model)
    longest = max(geometry.allowed_cp_sizes) * geometry.tokens_per_rank
    records, clock, plans = [], 10**12, []
    for number in range(steps):
        draw = rng.lognormal(
            np.log(geometry.tokens_per_rank) - 0.72, 1.2, int(rng.integers(48, 97))
        )
        lengths = [int(n) for n in np.clip(draw, 64, longest)]
        plan = planner.plan(dict(enumerate(lengths)))
        plans.append(
            dict(
                trace='selftest',
                number=number,
                groups=[
                    [(s, cp, tuple(lengths[i] for i in ids)) for s, cp, ids in row]
                    for row in plan.groups
                ],
            )
        )
        records.append(
            dict(
                event='consume',
                iterator=0,
                sequence_number=number,
                global_lengths=lengths,
                rows=plan.rows,
                predicted_ms=plan.predicted_ms,
                model_metadata=model.metadata(),
            )
        )
    truth = pricer.times(true_rates, plans)
    if layer_times:
        # One simulated execution gives the layer times of every rank and the step time.
        true_model = pricer.model(true_rates)
        timed = [
            synthetic_layer_times(pricer, true_model, plan['groups'], rng, noise, after, pause)
            for plan in plans
        ]
        truth = np.array([busy for _, busy in timed]) + true_rates['C_opt_ms']
    truth = truth * (1 + noise * rng.standard_normal(steps))
    truth[0] *= 4.0  # The first step compiles and allocates.
    truth[steps // 2] += 5e3  # A step that saves a checkpoint.
    for number, record in enumerate(records):
        wait = float(rng.uniform(0.0, 0.004))
        clock += int(wait * 1e9)
        record.update(consume=dict(monotonic_ns=clock, wall_ns=clock), residual_wait_s=wait)
        clock += int(truth[number] * 1e6)
    # An evaluation between two training steps.
    records.insert(
        steps - 6,
        dict(event='consume', iterator=1, sequence_number=0, consume=records[steps - 6]['consume']),
    )
    Path(directory).mkdir(parents=True, exist_ok=True)
    ranks = geometry.world_size if layer_times else 1
    if layer_times:
        # Every record carries the layers of the step before it, on every rank.
        previous = [None] + [
            timed[record['sequence_number']][0] if record['iterator'] == 0 else None
            for record in records[:-1]
        ]
    for rank in range(ranks):
        with (Path(directory) / f'rank{rank:05d}.jsonl').open('w') as stream:
            for position, record in enumerate(records):
                if layer_times:
                    step = previous[position]
                    record = dict(
                        record,
                        previous_layer_times=[] if step is None else [row[rank] for row in step],
                    )
                stream.write(json.dumps(record) + '\n')
    log = ''.join(
        f' iteration {n + 1:8d}/{steps:8d} | elapsed time per iteration (ms): {truth[n] + 2.0:.1f} |\n'
        for n in range(steps)
    )
    (Path(directory) / 'train.log').write_text(log)
    return float(np.mean([len(plan['groups']) for plan in plans]))


def command_selftest(args, package):
    """Fit and check on synthetic traces whose true coefficients are known."""
    geometry = dict(
        world_size=16,
        tokens_per_rank=2048,
        ep_size=8,
        allowed_cp_sizes=[1, 2, 4, 8],
        dataset_alignment=32,
        max_group_seqs=64,
        align_group_starts=True,
        max_cp_size=8,
    )
    sizes = dict(
        layer_count=12,
        hidden_size=2048,
        query_heads=32,
        head_dim=128,
        moe_topk=8,
        moe_ffn_hidden_size=768,
    )
    published = dict(P_attn_tflops=570.0, P_moe_tflops=84.0, t_x_ms=0.75, gamma=1.0, C_opt_ms=120.0)
    checks, root = [], Path(tempfile.mkdtemp(prefix='dcp_flops_selftest_'))

    def run(command, **options):
        defaults = dict(
            skip=SKIP_STEPS,
            outlier=OUTLIER_FACTOR,
            low=BAND[0],
            high=BAND[1],
            level='gamma',
            source='auto',
            calibration=None,
            log=None,
            changed=False,
        )
        try:
            return command(SimpleNamespace(**{**defaults, **options}), package), None
        except Fault as fault:
            return ERROR, str(fault)

    def expect(name, passed, detail=''):
        checks.append(passed)
        print(
            f"{'ok  ' if passed else 'FAIL'} {name}{': ' + detail if detail and not passed else ''}"
        )

    def rates_of(path):
        return json.loads(Path(path).read_text())['rates']

    def basis_of(path):
        return json.loads(Path(path).read_text())['source']['fit']['fitted_from']

    for ep in (8, 1):
        label = f'EP {ep}'
        pricer = Pricer(
            package, dict(geometry=dict(geometry, ep_size=ep), hardware='h100', model_sizes=sizes)
        )
        print(f'--- {label}: every compute term 2.2 times slower than published, C_opt 60 ms')
        slower, trace = dict(published, gamma=2.2, C_opt_ms=60.0), str(root / f'slow_ep{ep}')
        write_trace(trace, package, pricer, published, slower, seed=ep)
        code, fault = run(command_check, trace=[trace], log=str(Path(trace) / 'train.log'))
        expect(f'{label}: the published file fails the check', code == FAIL, str(fault))
        out = str(root / f'gamma_ep{ep}.json')
        code, fault = run(command_fit, trace=[trace], out=out)
        fitted = rates_of(out) if code == PASS else {}
        expect(f'{label}: level gamma passes', code == PASS, str(fault))
        expect(
            f'{label}: gamma is 2.2 within 2 %',
            code == PASS and abs(fitted['gamma'] / 2.2 - 1) < 0.02,
            str(fitted),
        )
        expect(
            f'{label}: C_opt is 60 ms within 25 ms',
            code == PASS and abs(fitted['C_opt_ms'] - 60) < 25,
            str(fitted),
        )
        expect(
            f'{label}: the published rates are kept',
            code == PASS and fitted['P_moe_tflops'] == published['P_moe_tflops'],
        )
        code, fault = run(command_check, trace=[trace], calibration=out)
        expect(f'{label}: the fitted file passes the check', code == PASS, str(fault))
        print(f'--- {label}: other rates than published')
        other = dict(P_attn_tflops=400.0, P_moe_tflops=120.0, t_x_ms=1.1, gamma=1.0, C_opt_ms=40.0)
        trace = str(root / f'other_ep{ep}')
        write_trace(trace, package, pricer, published, other, steps=130, seed=10 + ep)
        out = str(root / f'rates_ep{ep}.json')
        code, fault = run(command_fit, trace=[trace], out=out, level='rates')
        fitted = rates_of(out) if code == PASS else {}
        expect(f'{label}: level rates passes', code == PASS, str(fault))
        expect(
            f'{label}: P_moe is 120 within 3 %',
            code == PASS and abs(fitted['P_moe_tflops'] / 120 - 1) < 0.03,
            str(fitted),
        )
        expect(
            f'{label}: t_x is 1.1 ms within 10 %',
            code == PASS and abs(fitted['t_x_ms'] / 1.1 - 1) < 0.10,
            str(fitted),
        )
        expect(
            f'{label}: P_attn is 400 within 25 %',
            code == PASS and abs(fitted['P_attn_tflops'] / 400 - 1) < 0.25,
            str(fitted),
        )
        print(
            f'--- {label}: layer times of every rank, 6 ms per microbatch after the last layer, '
            'a pause of 500 ms in one microbatch of twenty'
        )
        trace = str(root / f'layers_slow_ep{ep}')
        write_trace(
            trace,
            package,
            pricer,
            published,
            slower,
            steps=50,
            seed=20 + ep,
            layer_times=True,
            after=6.0,
            pause=0.05,
        )
        out = str(root / f'layers_gamma_ep{ep}.json')
        code, fault = run(command_fit, trace=[trace], out=out)
        fitted = rates_of(out) if code == PASS else {}
        expect(
            f'{label}: level gamma is fitted from the layer times and passes',
            code == PASS and basis_of(out) == 'layer times',
            str(fault),
        )
        expect(
            f'{label}: gamma is 2.2 within 0.5 %',
            code == PASS and abs(fitted['gamma'] / 2.2 - 1) < 0.005,
            str(fitted),
        )
        not_used = (
            json.loads(Path(out).read_text())['source']['fit']['groups_not_used']
            if code == PASS
            else 0
        )
        expect(f'{label}: the layers of the paused microbatches are not used', not_used > 0)
        trace = str(root / f'layers_other_ep{ep}')
        microbatches = write_trace(
            trace,
            package,
            pricer,
            published,
            other,
            steps=50,
            seed=30 + ep,
            layer_times=True,
            after=6.0,
        )
        out = str(root / f'layers_rates_ep{ep}.json')
        code, fault = run(command_fit, trace=[trace], out=out, level='rates')
        fitted = rates_of(out) if code == PASS else {}
        expect(f'{label}: level rates passes on layer times', code == PASS, str(fault))
        expect(
            f'{label}: P_attn is 400 within 3 %, P_moe 120 within 1 %, t_x 1.1 ms within 3 %',
            code == PASS
            and abs(fitted['P_attn_tflops'] / 400 - 1) < 0.03
            and abs(fitted['P_moe_tflops'] / 120 - 1) < 0.01
            and abs(fitted['t_x_ms'] / 1.1 - 1) < 0.03,
            str(fitted),
        )
        expect(
            f'{label}: C_opt takes the 40 ms per step and the time after the layers',
            code == PASS and abs(fitted['C_opt_ms'] - 40 - 6 * microbatches) < 10,
            f'{fitted}, {microbatches:.2f} microbatches per step',
        )
        code, fault = run(command_check, trace=[trace], calibration=out)
        expect(
            f'{label}: the file fitted on layer times passes the check', code == PASS, str(fault)
        )
        out = str(root / f'layers_steps_ep{ep}.json')
        code, fault = run(command_fit, trace=[trace], out=out, source='steps')
        expect(
            f'{label}: the same trace can be fitted from step times',
            code in (PASS, FAIL) and basis_of(out) == 'step times',
            str(fault),
        )
    print('--- layer times of some ranks only')
    (Path(trace) / 'rank00003.jsonl').unlink()
    out = str(root / 'layers_incomplete.json')
    code, fault = run(command_fit, trace=[trace], out=out)
    expect(
        'without the file of a rank the fit uses the step times',
        code in (PASS, FAIL) and basis_of(out) == 'step times',
        str(fault),
    )
    code, fault = run(command_fit, trace=[trace], out=out, level='rates', source='layers')
    expect(
        '--from layers is then refused', code == ERROR and 'ranks 0 to 15' in str(fault), str(fault)
    )
    trace = str(root / 'other_ep1')
    print('--- refused inputs')
    code, fault = run(command_fit, trace=[trace], out=str(root / 'few.json'), skip=125)
    expect('too few steps are refused', code == ERROR and 'at least' in str(fault), str(fault))
    text = (Path(trace) / 'rank00000.jsonl').read_text()
    foreign = root / 'foreign'
    foreign.mkdir()
    (foreign / 'rank00000.jsonl').write_text(
        text.replace(f'"name": "{MODEL_NAME}"', '"name": "qwen3_h100_p2p"')
    )
    code, fault = run(command_check, trace=[str(foreign)])
    expect(
        'a trace of another cost model is refused',
        code == ERROR and 'qwen3_h100_p2p' in str(fault),
        str(fault),
    )
    doubled = root / 'doubled'
    doubled.mkdir()
    (doubled / 'rank00000.jsonl').write_text(text + text)
    code, fault = run(command_check, trace=[str(doubled)])
    expect(
        'a directory that two runs wrote into is refused',
        code == ERROR and 'more than one run' in str(fault),
        str(fault),
    )
    changed = root / 'changed'
    changed.mkdir()
    (changed / 'rank00000.jsonl').write_text(
        re.sub(
            r'"predicted_ms": ([0-9.]+)',
            lambda m: f'"predicted_ms": {float(m.group(1)) * 1.01}',
            text,
        )
    )
    code, fault = run(command_check, trace=[str(changed)])
    expect(
        'a trace priced by another version is refused',
        code == ERROR and 'prices the recorded layouts differently' in str(fault),
        str(fault),
    )
    print(f"selftest: {sum(checks)} of {len(checks)} checks passed (files in {root})")
    return PASS if all(checks) else FAIL


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__.split('\n')[0])
    parser.add_argument(
        '--megatron',
        default=str(Path(__file__).resolve().parents[3]),
        help='Megatron-LM checkout (default: the one that holds this script)',
    )
    commands = parser.add_subparsers(dest='command', required=True)

    def common(command, needs_trace=True):
        if needs_trace:
            command.add_argument(
                '--trace',
                action='append',
                required=True,
                help='trace directory of a run ("trace_dir"); repeat for more runs',
            )
        command.add_argument(
            '--model-changed',
            dest='changed',
            action='store_true',
            help='the formulas of the cost model changed after the runs: use the layouts and '
            'measured times of the traces without comparing their recorded predictions',
        )
        command.add_argument(
            '--skip',
            type=int,
            default=SKIP_STEPS,
            help=f'first training steps of a trace that are not used (default {SKIP_STEPS})',
        )
        command.add_argument(
            '--outlier',
            type=float,
            default=OUTLIER_FACTOR,
            help=f'a step is not used when measured / predicted is this far from the median (default {OUTLIER_FACTOR})',
        )
        command.add_argument(
            '--low',
            type=float,
            default=BAND[0],
            help=f'smallest accepted median of predicted / measured (default {BAND[0]})',
        )
        command.add_argument(
            '--high',
            type=float,
            default=BAND[1],
            help=f'largest accepted median of predicted / measured (default {BAND[1]})',
        )

    fit_command = commands.add_parser('fit', help='fit the coefficients and write a file')
    common(fit_command)
    fit_command.add_argument('--out', required=True, help='coefficient file to write')
    fit_command.add_argument(
        '--level',
        choices=LEVELS,
        default='gamma',
        help='gamma: gamma and C_opt; rates: P_attn, P_moe, t_x and C_opt (default gamma)',
    )
    fit_command.add_argument(
        '--from',
        dest='source',
        choices=SOURCES,
        default='auto',
        help='fit from the layer times of the CP groups or from the step times (default auto: '
        'layer times when the run recorded them on every rank)',
    )
    fit_command.set_defaults(function=command_fit)
    check_command = commands.add_parser('check', help='compare predictions with measured steps')
    common(check_command)
    check_command.add_argument(
        '--calibration', help='coefficient file to check (default: the one the run used)'
    )
    check_command.add_argument(
        '--log',
        help='training log of the first --trace; its time per iteration is compared with the trace',
    )
    check_command.set_defaults(function=command_check)
    coverage_command = commands.add_parser(
        'coverage', help='print the data arguments of a coverage run'
    )
    common(coverage_command)
    coverage_command.set_defaults(function=command_coverage)
    selftest_command = commands.add_parser('selftest', help='run fit and check on synthetic traces')
    selftest_command.set_defaults(function=command_selftest)
    args = parser.parse_args(argv)
    try:
        return args.function(args, load_package(args.megatron))
    except Fault as fault:
        print(f'ERROR: {fault}', file=sys.stderr)
        return ERROR


if __name__ == '__main__':
    sys.exit(main())
