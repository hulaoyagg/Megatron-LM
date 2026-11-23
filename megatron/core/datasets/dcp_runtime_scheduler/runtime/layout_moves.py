# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Refine a complete layout of the DynamicUSP planner with priced moves.

Stage 1 (``hot_switch``) chooses the microbatch count and the CP groups from one scalar weight
per sample. It never asks what the ranks that share an expert-parallel block wait for. The
refinement starts from a complete layout, keeps every CP group at its size and tries two
kinds of moves:

- ``reassign``: change which samples a CP group holds. One sample moves to another group, two
  samples trade places, or CP groups of equal size trade their whole contents.
- ``relocate``: change which ranks a CP group runs on, so that wide groups share a unit of one
  microbatch and the units finish at the same time.

A unit is an aligned run of ranks that holds whole CP groups and whole expert-parallel blocks:
one expert-parallel block when no CP group is larger. Every candidate is priced with the
injected cost model and the complete-step timeline, and is kept only when the predicted step
time drops. The planner scores the refined layout once more with the function it uses for
every other layout before it selects it (``dynamic_usp_moves``).

The runtime file of scheduler type ``dynamic_usp_flops`` switches it on with ``"stage2"``.
It is not the DynamicUSP Stage 2 of the original profile (``stage2.py``, a port of the
upstream re-packing): this module is written for this runtime and ports no upstream code.
It imports nothing of the package, so that the configuration can validate its options.
"""

from __future__ import annotations

import math
import time
from collections.abc import Mapping
from dataclasses import dataclass
from typing import NamedTuple

MOVES = ("reassign", "relocate")
# The runtime file selects the moves with one level; every level adds one kind of move.
LEVELS = ((), MOVES[:1], MOVES[:2])
# Relative margin by which a step time has to drop to count as an improvement. It is far above
# the rounding of a step time that is accumulated over thousands of additions.
TOLERANCE = 1e-9
# Runs of one cell that start an exchange, and groups of one cell that receive a sample.
HEAVY_RUNS = 2
LIGHT_TARGETS = 3


@dataclass(frozen=True)
class MoveOptions:
    """Which moves refine a layout and how long one global batch may spend on them."""

    moves: tuple[str, ...] = ()
    """Enabled kinds of moves, a subset of ``MOVES``. Empty: no refinement."""

    time_budget_s: float = 1.0
    """Seconds one global batch may spend on the refinement."""

    def __post_init__(self):
        if isinstance(self.moves, str) or any(move not in MOVES for move in self.moves):
            raise ValueError(f"stage2 moves must be taken from {list(MOVES)}, got {self.moves!r}")
        object.__setattr__(self, "moves", tuple(move for move in MOVES if move in self.moves))
        budget = self.time_budget_s
        if type(budget) not in (int, float) or not 0 < budget < math.inf:
            raise ValueError("stage2 time_budget_s must be a finite positive number of seconds")

    @classmethod
    def parse(cls, value) -> "MoveOptions":
        """Options from a level of the runtime file, a mapping of fields or an instance."""
        if value is None:
            return cls()
        if isinstance(value, cls):
            return value
        if type(value) is int:
            if not 0 <= value < len(LEVELS):
                raise ValueError(f"stage2 must be a level from 0 to {len(LEVELS) - 1}, got {value}")
            return cls(moves=LEVELS[value])
        if isinstance(value, Mapping):
            unknown = sorted(set(value) - {"moves", "time_budget_s"}, key=str)
            if unknown:
                raise ValueError(
                    f"stage2 takes the keys moves and time_budget_s, got unknown key(s) {unknown}"
                )
            options = dict(value)
            if "moves" in options and isinstance(options["moves"], (list, tuple)):
                options["moves"] = tuple(options["moves"])
            return cls(**options)
        raise ValueError(
            f"stage2 must be a level from 0 to {len(LEVELS) - 1} or an object with the keys "
            f"moves and time_budget_s, got {value!r}"
        )

    @property
    def enabled(self) -> bool:
        """Whether any move is enabled."""
        return bool(self.moves)


class _Group(NamedTuple):
    """One priced CP group. The first five fields are what ``TimelineModel`` reads."""

    start: int
    cp: int
    forward: float
    backward: float
    program: dict | None
    ids: tuple
    # Time of one microbatch for this group alone; orders candidates, never accepts one.
    time: float


def _faster(new, old) -> bool:
    """Whether the step time ``new`` is below ``old`` by more than the margin."""
    return new < old - TOLERANCE * max(1.0, abs(old))


def _segments(operations):
    """Split one phase of a timeline program into ``(cycle, repeat)`` runs."""
    operations = tuple(operations)
    runs, index, count = [], 0, len(operations)
    while index < count:
        period, repeat = 1, 1
        for length in range(1, (count - index) // 2 + 1):
            cycle = operations[index : index + length]
            times = 1
            while operations[index + times * length : index + (times + 1) * length] == cycle:
                times += 1
            if times > 1 and length * times > period * repeat:
                period, repeat = length, times
        runs.append((operations[index : index + period], repeat))
        index += period * repeat
    return tuple(runs)


class _UnitScorer:
    """Step time of a layout, computed one unit at a time.

    The operations of a timeline program join ranks of one CP group or of one expert-parallel
    block. Both lie inside one unit, so units only meet in the DPxCP synchronisation after the
    first forward pass, and a changed unit is replayed without the others. The times of a unit
    are kept relative to that synchronisation.

    Two shortcuts keep a trial cheap. Both repeat the additions of the timeline in another
    order, so a score differs from ``TimelineModel.evaluate`` by rounding only.

    - A repeated cycle of operations is applied twice and then extrapolated. That is exact
      when every CP group lies inside one expert-parallel block or covers whole blocks: each
      group or block that no other one contains is then synchronised as a whole by an
      operation of the cycle, or its parts never meet, and from the second cycle on every rank
      advances by a constant.
    - In a unit that is one expert-parallel block, every expert-parallel operation leaves all
      ranks with one time. It is kept as one number, and a CP operation followed by such a
      synchronisation adds the cost of the slowest group.
    """

    name = "unit"

    def __init__(self, timeline, unit, ep_size) -> None:
        self.timeline = timeline
        self.unit = unit
        self.count = timeline.world_size // unit
        self.blocks = tuple((first, first + ep_size) for first in range(0, unit, ep_size))
        self.whole = len(self.blocks) == 1
        self.forward, self.backward = (_segments(self._steps(p)) for p in timeline.program)
        self.sync = timeline.first_forward_dp_cp_sync
        # Per microbatch and unit: the compiled cell. Per unit: the times after every
        # microbatch, the time of the first forward pass and the time the unit finishes.
        self.cells: list[list] = []
        self.states: list[list] = []
        self.heads: list[float] = []
        self.peaks: list[float] = []
        self._pending: tuple = ()

    def _steps(self, operations):
        """Operations of one phase, a CP operation joined with a whole-unit synchronisation
        that follows it."""
        operations, steps, index = tuple(operations), [], 0
        while index < len(operations):
            kind, payload = operations[index]
            after = operations[index + 1] if index + 1 < len(operations) else ("", None)
            if self.whole and kind == "cp" and after[0] in ("ep", "ep_mean"):
                steps.append(("cp+" + after[0], (payload, after[1])))
                index += 2
            else:
                steps.append((kind, payload))
                index += 1
        return steps

    @staticmethod
    def _compile(groups):
        spans, offset = [], 0
        for group in groups:
            spans.append((offset, offset + group.cp, group.program))
            offset += group.cp
        return tuple(spans), {}

    def _means(self, cell, name):
        """Mean of one named cost over the ranks of every expert-parallel block of a cell."""
        spans, cache = cell
        durations = cache.get(name)
        if durations is None:
            per_rank = []
            for first, last, program in spans:
                per_rank.extend([program[name]] * (last - first))
            durations = cache[name] = tuple(
                sum(per_rank[first:last]) / (last - first) for first, last in self.blocks
            )
        return durations

    @staticmethod
    def _slowest(cell, name):
        """Largest value of one named cost among the groups of a cell."""
        spans, cache = cell
        value = cache.get(("slowest", name))
        if value is None:
            value = cache[("slowest", name)] = max(program[name] for _, _, program in spans)
        return value

    def _apply(self, times, cell, operations):
        """The operations of ``TimelineModel._program_phase`` on the ranks of one unit.

        ``times`` is a list with the time of every rank, or one number that all ranks share.
        A list is changed in place.
        """
        spans = cell[0]
        for kind, payload in operations:
            shared = type(times) is float
            if kind == "local":
                times = times + payload if shared else [value + payload for value in times]
            elif kind == "cp":
                if shared:
                    times = [times] * self.unit
                for first, last, program in spans:
                    value = max(times[first:last]) + program[payload]
                    times[first:last] = [value] * (last - first)
            elif kind in ("ep", "ep_mean"):
                durations = self._means(cell, payload) if kind == "ep_mean" else None
                if self.whole:
                    duration = payload if durations is None else durations[0]
                    times = (times if shared else max(times)) + duration
                    continue
                if shared:
                    times = [times] * self.unit
                for index, (first, last) in enumerate(self.blocks):
                    duration = payload if durations is None else durations[index]
                    value = max(times[first:last]) + duration
                    times[first:last] = [value] * (last - first)
            else:
                name, value = payload
                duration = value if kind == "cp+ep" else self._means(cell, value)[0]
                if shared:
                    times = times + self._slowest(cell, name) + duration
                else:
                    ends = [
                        max(times[first:last]) + program[name] for first, last, program in spans
                    ]
                    times = max(ends) + duration
        return times

    def _phase(self, times, cell, segments):
        if type(times) is not float:
            times = list(times)
        for operations, repeat in segments:
            once = self._apply(times, cell, operations)
            if repeat == 1:
                times = once
                continue
            times = self._apply(once if type(once) is float else list(once), cell, operations)
            if repeat == 2:
                continue
            if type(once) is float and type(times) is float:
                times += (repeat - 2) * (times - once)
            elif type(once) is not float and type(times) is not float:
                times = [
                    after + (repeat - 2) * (after - before) for before, after in zip(once, times)
                ]
            else:
                for _ in range(repeat - 2):
                    times = self._apply(times, cell, operations)
        return times

    def _chain(self, cells, start=0, times=None):
        """Times of one unit after every microbatch from ``start`` on.

        ``times`` are the times before microbatch ``start``. Also returns the time of the first
        forward pass when the chain begins at the first microbatch and the ranks synchronise
        after it; the later times are then relative to that synchronisation.
        """
        head, states = None, []
        for index in range(start, len(cells)):
            if index == 0:
                times = self._phase(0.0, cells[0], self.forward)
                if self.sync:
                    head, times = (times if type(times) is float else max(times)), 0.0
            else:
                times = self._phase(times, cells[index], self.forward)
            times = self._phase(times, cells[index], self.backward)
            states.append([times] * self.unit if type(times) is float else times)
        return head, states

    def _score(self, heads, peaks):
        sync = max(heads) if self.sync else 0.0
        return sync + max(peaks) + self.timeline.iteration_tail_ms

    def load(self, cells):
        """Step time of a complete layout; ``cells[microbatch][unit]`` lists a unit's groups."""
        self.cells = [[self._compile(groups) for groups in row] for row in cells]
        self.states = [[None] * (len(cells) + 1) for _ in range(self.count)]
        self.heads, self.peaks = [0.0] * self.count, [0.0] * self.count
        for unit in range(self.count):
            head, states = self._chain([row[unit] for row in self.cells])
            self.heads[unit] = head or 0.0
            self.states[unit][1:] = states
            self.peaks[unit] = max(states[-1])
        return self._score(self.heads, self.peaks)

    def trial(self, changes):
        """Step time with some cells replaced; ``commit`` makes that layout the current one."""
        first = {}
        for microbatch, unit in changes:
            first[unit] = min(first.get(unit, microbatch), microbatch)
        heads, peaks = list(self.heads), list(self.peaks)
        pending = []
        for unit, start in first.items():
            compiled = {
                microbatch: self._compile(groups)
                for (microbatch, other), groups in changes.items()
                if other == unit
            }
            cells = [compiled.get(index) or row[unit] for index, row in enumerate(self.cells)]
            # The chain resumes from the stored times before the first changed microbatch.
            head, states = self._chain(cells, start, self.states[unit][start])
            if not start:
                heads[unit] = head or 0.0
            peaks[unit] = max(states[-1])
            pending.append((unit, compiled, states))
        self._pending = pending, heads, peaks
        return self._score(heads, peaks)

    def commit(self):
        """Make the layout of the last ``trial`` the current one."""
        pending, self.heads, self.peaks = self._pending
        for unit, compiled, states in pending:
            for microbatch, cell in compiled.items():
                self.cells[microbatch][unit] = cell
            self.states[unit][len(self.states[unit]) - len(states) :] = states

    def critical(self):
        """Units that set the step time: the last to finish and the slowest first forward."""
        units = [max(range(self.count), key=self.peaks.__getitem__)]
        if self.sync:
            head = max(range(self.count), key=self.heads.__getitem__)
            if head not in units:
                units.append(head)
        return units

    def cell_time(self, groups):
        """Time of one microbatch for the groups of one unit on their own."""
        cell = self._compile(groups)
        times = self._phase(self._phase(0.0, cell, self.forward), cell, self.backward)
        return times if type(times) is float else max(times)


def _advance(timeline, state, groups, first):
    """The completion times of the ranks after one more microbatch, as ``evaluate`` steps."""
    forward, backward = timeline.program
    state = timeline._program_phase(state, groups, forward)
    if first and timeline.first_forward_dp_cp_sync:
        state = [max(state)] * len(state)
    return timeline._program_phase(state, groups, backward)


class _PrefixScorer:
    """Step time of any layout, microbatch by microbatch as ``TimelineModel.evaluate``.

    The state before every microbatch is kept, so a trial restarts at the first microbatch it
    changes. This is the fallback when the units of a program cannot be replayed separately.
    """

    name = "prefix"

    def __init__(self, timeline, unit, ep_size=None) -> None:
        self.timeline = timeline
        self.count = timeline.world_size // unit
        self.cells: list[list] = []
        self.states: list[list] = []
        self._pending: tuple = ()

    def load(self, cells):
        """Step time of a complete layout; ``cells[microbatch][unit]`` lists a unit's groups."""
        self.cells = [list(row) for row in cells]
        self.states = [[0.0] * self.timeline.world_size]
        for index, row in enumerate(self.cells):
            groups = [group for cell in row for group in cell]
            self.states.append(_advance(self.timeline, self.states[-1], groups, index == 0))
        return max(self.states[-1]) + self.timeline.iteration_tail_ms

    def trial(self, changes):
        """Step time with some cells replaced; ``commit`` makes that layout the current one."""
        start = min(microbatch for microbatch, _ in changes)
        state, states = self.states[start], []
        for index in range(start, len(self.cells)):
            groups = [
                group
                for unit, cell in enumerate(self.cells[index])
                for group in changes.get((index, unit), cell)
            ]
            state = _advance(self.timeline, state, groups, index == 0)
            states.append(state)
        self._pending = start, changes, states
        return max(state) + self.timeline.iteration_tail_ms

    def commit(self):
        """Make the layout of the last ``trial`` the current one."""
        start, changes, states = self._pending
        for (microbatch, unit), groups in changes.items():
            self.cells[microbatch][unit] = groups
        self.states[start + 1 :] = states

    def critical(self):
        """The unit that finishes last."""
        state = self.states[-1]
        width = len(state) // self.count
        peaks = [max(state[unit * width : (unit + 1) * width]) for unit in range(self.count)]
        return [max(range(self.count), key=peaks.__getitem__)]

    def cell_time(self, groups):
        """Slowest group of one unit; a stand-in for the time of the unit."""
        return max(group.time for group in groups)


class LayoutRefiner:
    """Refine complete layouts of one planner with the moves that its options enable."""

    def __init__(self, geometry, sizes, timeline, options) -> None:
        self.sizes = tuple(sorted(sizes))
        self.timeline = timeline
        self.options = options
        ep_size = geometry.ep_size
        # Smallest aligned run of ranks that holds whole CP groups and expert-parallel blocks.
        self.unit = math.lcm(self.sizes[-1], ep_size)
        self.count = geometry.world_size // self.unit
        # Runs of ranks that trade places: a whole cell, or the ranks of one group size.
        self.run_lengths = [self.unit] + [s for s in reversed(self.sizes) if s < self.unit]
        program = timeline.program
        # How often the program of a step runs every named cost of a group.
        named: dict[str, int] = {}
        for phase in program:
            for kind, payload in phase:
                if kind in ("cp", "ep_mean"):
                    named[payload] = named.get(payload, 0) + 1
        self._named = tuple(named.items())
        blocks = any(kind in ("ep", "ep_mean") for phase in program for kind, _ in phase)
        separable = not blocks or all(s % ep_size == 0 or ep_size % s == 0 for s in self.sizes)
        # A unit that is one expert-parallel block is synchronised as a whole in every
        # layer, so the place of a group inside the unit does not change any time.
        self.atomic = blocks and self.unit == ep_size
        self._scorer = (_UnitScorer if separable else _PrefixScorer)(timeline, self.unit, ep_size)
        # State of one ``refine`` call: the inputs, the layout as priced groups per microbatch
        # and unit, its step time, the moves of the current phase and their deadline.
        self.lengths: dict[int, int] = {}
        self.costs = None
        self.cells: list[list] = []
        self.step_ms = 0.0
        self.moves: frozenset[str] = frozenset()
        self.deadline = 0.0
        self.version = 0
        self.statistics: dict = {}

    # ------------------------------------------------------------------ groups
    def _time(self, forward, backward, program):
        return sum(count * program[name] for name, count in self._named)

    def _group(self, start, cp, ids):
        """Price ``ids`` as one CP group; None when the pack does not fit or has no price."""
        values = tuple(self.lengths[i] for i in ids)
        if not values or not self.costs.fits(values, cp):
            return None
        price = self.costs.pack(values, cp, start)
        if price is None:
            return None
        return _Group(start, cp, *price, tuple(ids), self._time(*price))

    def _moved(self, group, start):
        """The same group on other ranks; priced again when the price depends on the ranks."""
        if start == group.start:
            return group
        if self.costs.placement_period is None:
            return group._replace(start=start)
        return self._group(start, group.cp, group.ids)

    def _layout(self, rounds):
        """Priced groups per microbatch and unit, or None when a group is not aligned."""
        cells = []
        for row in rounds:
            units, start = [[] for _ in range(self.count)], 0
            for entry in row:
                cp = entry["cp"]
                if cp not in self.sizes or start % cp:
                    return None
                group = self._group(start, cp, entry["sample_ids"])
                if group is None:
                    return None
                units[start // self.unit].append(group)
                start += cp
            cells.append(units)
        return cells

    # ----------------------------------------------------------------- scoring
    def _expired(self):
        return time.perf_counter() >= self.deadline

    def _best(self, candidates):
        """Score ``(changes, kind)`` candidates and keep the fastest one that beats the current
        layout; False when none does."""
        best = None
        for changes, kind in candidates:
            if changes is not None:
                self.statistics["trials"][kind] += 1
                step_ms = self._scorer.trial(changes)
                if _faster(step_ms, self.step_ms if best is None else best[0]):
                    best = step_ms, changes, kind
            if self._expired():
                break
        if best is None:
            return False
        _, changes, kind = best
        # The scorer holds the last trial, which need not be the best one.
        self.step_ms = self._scorer.trial(changes)
        self._scorer.commit()
        for (microbatch, unit), groups in changes.items():
            self.cells[microbatch][unit] = groups
        self.version += 1
        self.statistics["accepted"][kind] += 1
        return True

    # ---------------------------------------------------------------- exchange
    @staticmethod
    def _runs(cell, length):
        """Aligned runs of ``length`` ranks that hold whole groups, as slices of the cell."""
        runs, index = [], 0
        while index < len(cell):
            first = cell[index]
            if first.cp > length:
                index += 1
                continue
            end = index
            while end < len(cell) and cell[end].start < first.start + length:
                end += 1
            runs.append((index, end))
            index = end
        return runs

    @staticmethod
    def _weight(cell, run):
        groups = cell[run[0] : run[1]]
        return max(group.time for group in groups), sum(group.cp * group.time for group in groups)

    def _swap(self, here, run, there, other):
        """Cells after two runs of equal length trade places; None when a group has no price."""
        cell, partner = self.cells[here[0]][here[1]], self.cells[there[0]][there[1]]
        shift = partner[other[0]].start - cell[run[0]].start
        leaving = [self._moved(group, group.start + shift) for group in cell[run[0] : run[1]]]
        arriving = [self._moved(g, g.start - shift) for g in partner[other[0] : other[1]]]
        if None in leaving or None in arriving:
            return None
        if here == there:
            (a, b), (c, d) = sorted((run, other))
            first, second = (arriving, leaving) if (a, b) == run else (leaving, arriving)
            return {here: cell[:a] + first + cell[b:c] + second + cell[d:]}
        return {
            here: cell[: run[0]] + arriving + cell[run[1] :],
            there: partner[: other[0]] + leaving + partner[other[1] :],
        }

    def _partners(self, here, run, length):
        """Exchanges of one run with the lightest run of every other cell, per kind of move."""
        cell = self.cells[here[0]][here[1]]
        pattern = tuple(group.cp for group in cell[run[0] : run[1]])
        for microbatch, row in enumerate(self.cells):
            for unit, partner in enumerate(row):
                there = (microbatch, unit)
                if there == here and self.atomic:
                    continue
                same, different = [], []
                for other in self._runs(partner, length):
                    if there == here and other == run:
                        continue
                    equal = tuple(group.cp for group in partner[other[0] : other[1]]) == pattern
                    (same if equal else different).append(other)
                # Equal sizes trade contents only; different sizes change where groups run.
                for runs, kind in ((same, "reassign"), (different, "relocate")):
                    if runs and kind in self.moves:
                        other = min(runs, key=lambda r: self._weight(partner, r))
                        yield self._swap(here, run, there, other), kind

    def _exchange(self):
        """Trade a heavy run of a critical unit with a lighter run elsewhere; False when no
        trade lowers the step time."""
        for unit in self._scorer.critical():
            for microbatch in range(len(self.cells)):
                cell, here = self.cells[microbatch][unit], (microbatch, unit)
                for length in self.run_lengths:
                    runs = sorted(
                        self._runs(cell, length),
                        key=lambda run: self._weight(cell, run),
                        reverse=True,
                    )
                    for run in runs[:HEAVY_RUNS]:
                        if self._best(self._partners(here, run, length)):
                            return True
                        if self._expired():
                            return False
        return False

    # ----------------------------------------------------------------- samples
    def _offers(self, here, index, sample):
        """Moves of one sample to, and trades with, the lightest groups of every cell."""
        cell = self.cells[here[0]][here[1]]
        group, length = cell[index], self.lengths[sample]
        rest = tuple(i for i in group.ids if i != sample)
        shrunk = self._group(group.start, group.cp, rest) if rest else None
        for microbatch, row in enumerate(self.cells):
            for unit, partner in enumerate(row):
                there = (microbatch, unit)
                targets = sorted(
                    (i for i in range(len(partner)) if there != here or i != index),
                    key=lambda i: partner[i].time,
                )[:LIGHT_TARGETS]
                moved = traded = False
                for target in targets:
                    other = partner[target]
                    if not moved and shrunk is not None:
                        grown = self._group(other.start, other.cp, other.ids + (sample,))
                        if grown is not None:
                            moved = True
                            yield self._replaced(here, index, shrunk, there, target, grown)
                    if traded:
                        continue
                    for choice in sorted(other.ids, key=self.lengths.__getitem__):
                        if self.lengths[choice] >= length:
                            break
                        mine = self._exchanged(group, sample, choice)
                        theirs = self._exchanged(other, choice, sample)
                        if mine is not None and theirs is not None:
                            traded = True
                            yield self._replaced(here, index, mine, there, target, theirs)
                            break

    def _exchanged(self, group, old, new):
        """The group with one of its samples replaced by another sample."""
        return self._group(group.start, group.cp, tuple(new if i == old else i for i in group.ids))

    def _replaced(self, here, index, group, there, target, other):
        changes = {here: list(self.cells[here[0]][here[1]])}
        changes[here][index] = group
        changes.setdefault(there, list(self.cells[there[0]][there[1]]))[target] = other
        return changes, "reassign"

    def _samples(self):
        """Take one sample out of the slowest group of a cell of a critical unit; False when
        no move or trade lowers the step time."""
        for unit in self._scorer.critical():
            for microbatch in range(len(self.cells)):
                cell, here = self.cells[microbatch][unit], (microbatch, unit)
                index = max(range(len(cell)), key=lambda i: cell[i].time)
                for sample in sorted(cell[index].ids, key=lambda i: -self.lengths[i]):
                    if self._best(self._offers(here, index, sample)):
                        return True
                    if self._expired():
                        return False
        return False

    # ------------------------------------------------------------------ repack
    def _repack(self):
        """Rebuild all cells at once: the slowest groups share cells, the cells are spread over
        the units so that the units finish together. Without ``relocate`` every cell keeps its
        group sizes and only groups of equal size trade contents; without ``reassign`` groups
        of equal size keep their order."""
        content, free = "reassign" in self.moves, "relocate" in self.moves
        positions = [(m, unit) for m in range(len(self.cells)) for unit in range(self.count)]
        groups = [group for m, unit in positions for group in self.cells[m][unit]]
        order = {id(group): index for index, group in enumerate(groups)}
        groups.sort(key=lambda g: (-g.cp, -g.time if content else 0.0, order[id(g)]))
        if free:
            bundles, bundle, used = [], [], 0
            for group in groups:
                bundle.append(group)
                used += group.cp
                if used == self.unit:
                    bundles.append((None, bundle))
                    bundle, used = [], 0
            if bundle or len(bundles) != len(positions):
                return False
            slots = {None: list(positions)}
        else:
            patterns = {p: tuple(g.cp for g in self.cells[p[0]][p[1]]) for p in positions}
            supply = {}
            for group in groups:
                supply.setdefault(group.cp, []).append(group)
            # Cells with wider groups come first: their narrower groups take the slowest
            # contents, which a wider group of the same cell outlasts anyway.
            bundles = [
                (patterns[p], [supply[cp].pop(0) for cp in patterns[p]])
                for p in sorted(positions, key=lambda p: ([-cp for cp in patterns[p]], p))
            ]
            slots = {}
            for position in positions:
                slots.setdefault(patterns[position], []).append(position)
        # The slowest cell first, each to the unit with the least time so far; then trades
        # between the slowest unit and another one. Both add the times of single cells, which
        # is cheap; the complete candidate gets one exact score.
        times = [self._scorer.cell_time(bundle) for _, bundle in bundles]
        totals, where = [0.0] * self.count, {}
        for index in sorted(range(len(bundles)), key=lambda i: (-times[i], i)):
            options = slots[bundles[index][0]]
            position = min(options, key=lambda p: (totals[p[1]], p))
            options.remove(position)
            totals[position[1]] += times[index]
            where[position] = index
        for _ in range(4 * len(bundles)):
            slow = max(range(self.count), key=totals.__getitem__)
            best = None
            for here, mine in where.items():
                for there, theirs in where.items():
                    shift = times[mine] - times[theirs]
                    if here[1] != slow or there[1] == slow or shift <= 0:
                        continue
                    if bundles[mine][0] != bundles[theirs][0]:
                        continue
                    worst = max(totals[slow] - shift, totals[there[1]] + shift)
                    if best is None or worst < best[0]:
                        best = worst, here, there, shift
            if best is None or not _faster(best[0], totals[slow]):
                break
            _, here, there, shift = best
            where[here], where[there] = where[there], where[here]
            totals[slow] -= shift
            totals[there[1]] += shift
        changes = {}
        for position, index in where.items():
            start, placed = position[1] * self.unit, []
            for group in bundles[index][1]:
                placed.append(self._moved(group, start))
                start += group.cp
            if None in placed:
                return False
            changes[position] = placed
        return self._best([(changes, "relocate" if free else "reassign")])

    # ------------------------------------------------------------------ search
    def _converge(self):
        """Apply the enabled moves until none lowers the step time; False when time ran out."""
        repacked = None
        while not self._expired():
            before = self.version
            if repacked != self.version:
                self._repack()
                repacked = self.version
            while self._exchange():
                pass
            if "reassign" in self.moves:
                while self._samples():
                    pass
            if self.version == before:
                return True
        return False

    def refine(self, rounds, lengths, costs, deadline):
        """Refine one layout until ``deadline`` (a ``time.perf_counter`` value).

        Returns the rounds and the statistics of the search. The rounds are the object that
        was passed in when no move was kept.
        """
        begin = time.perf_counter()
        self.lengths, self.costs = lengths, costs
        self.statistics = statistics = dict(
            scorer=self._scorer.name,
            accepted=dict.fromkeys(self.options.moves, 0),
            trials=dict.fromkeys(self.options.moves, 0),
            converged=False,
        )
        cells = self._layout(rounds)
        if cells is None:
            statistics["skipped"] = "the layout has a group that is not size-aligned or priced"
            return rounds, statistics
        self.cells, self.version = cells, 0
        self.step_ms = statistics["start_ms"] = self._scorer.load(cells)
        # One phase per enabled move, each adding a move to the ones before it: the result of
        # a level is the starting point of the next level.
        moves, converged = self.options.moves, True
        for index in range(len(moves)):
            self.moves = frozenset(moves[: index + 1])
            self.deadline = begin + (deadline - begin) * (index + 1) / len(moves)
            converged = self._converge() and converged
        statistics.update(
            converged=converged, end_ms=self.step_ms, elapsed_s=time.perf_counter() - begin
        )
        if not self.version:
            return rounds, statistics
        return [
            [dict(cp=group.cp, sample_ids=list(group.ids)) for cell in row for group in cell]
            for row in self.cells
        ], statistics
