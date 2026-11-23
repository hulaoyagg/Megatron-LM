# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
"""The refinement of a layout by priced moves (runtime/layout_moves.py) and its planner."""

import random
import time

import pytest

from megatron.core.datasets.dcp_runtime_scheduler.models import make_model
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp import (
    DynamicUSPRuntimePlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.dynamic_usp_moves import (
    DynamicUSPMovesPlanner,
)
from megatron.core.datasets.dcp_runtime_scheduler.runtime.layout_moves import MOVES, MoveOptions
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    validate_rows,
)

PRICE = {1: 1.0, 2: 2.0, 4: 4.0, 8: 8.0}
LAYERS, REST = 2, 1.0


class _TierModel:
    """A fixed price per layer for every group size, plus a share for every token of a rank.

    Forward attention of a group is ``PRICE[cp] + content * tokens / cp``, backward twice
    that; the rest of a layer takes ``REST`` and synchronises an expert-parallel block.
    """

    name = "tier"
    nvlink_domain_size = None
    placement_invariant = True

    def __init__(self, geometry, content=0.0) -> None:
        self.geometry, self.content = geometry, content

    def forward(self, lengths, cp, *, ranks=None):
        return PRICE[cp] + self.content * sum(lengths) / cp

    def backward(self, lengths, cp, *, ranks=None):
        return 2 * self.forward(lengths, cp)

    def program_costs(self, lengths, cp, *, ranks=None):
        return dict(att_f=self.forward(lengths, cp), att_b=self.backward(lengths, cp))

    def timing(self):
        forward = [dict(sync="cp", cost="att_f"), dict(sync="ep", duration_ms=REST)]
        backward = [dict(sync="ep", duration_ms=REST), dict(sync="cp", cost="att_b")]
        program = dict(layer_count=LAYERS, forward=forward * LAYERS, backward=backward * LAYERS)
        return dict(iteration_tail_ms=0.0, execution_program=[program])

    def clear(self):
        return None

    def check_domain(self, geometry, config=None):
        return None


def _cell(attention):
    """Forward and backward time of one microbatch for an expert-parallel block whose slowest
    group has this forward attention time."""
    return LAYERS * (attention + REST), LAYERS * (REST + 2 * attention)


def _tier_planner(world, ep, sizes, content=0.0, **moves):
    geometry = RuntimePlanGeometry(world, 64, ep, sizes, 2 * world, 64, True)
    return DynamicUSPMovesPlanner(
        geometry, _TierModel(geometry, content), moves=dict(time_budget_s=30.0, **moves)
    )


def _refine(planner, rounds, lengths):
    """Refine one layout; the refined rounds, their score and the search statistics."""
    deadline = time.perf_counter() + planner.move_options.time_budget_s
    refined, statistics = planner.refiner.refine(rounds, lengths, planner.costs, deadline)
    return refined, planner.timeline.evaluate(planner._costed(refined, lengths)), statistics


def _shape(rounds):
    return [[group["cp"] for group in row] for row in rounds]


def test_reassign_trades_contents_and_keeps_every_group_size():
    # Two expert-parallel blocks of two ranks. Block 0 holds the long samples in both
    # microbatches, so it finishes last; no sample fits a rank together with another one.
    lengths = dict(enumerate([60, 60, 20, 20, 60, 60, 20, 20]))
    rounds = [
        [dict(cp=1, sample_ids=[i]) for i in (0, 1, 2, 3)],
        [dict(cp=1, sample_ids=[i]) for i in (4, 5, 6, 7)],
    ]
    planner = _tier_planner(4, 2, (1, 2), content=0.01, moves=MOVES[:1])
    heavy, light = _cell(PRICE[1] + 0.6), _cell(PRICE[1] + 0.2)
    before = planner.timeline.evaluate(planner._costed(rounds, lengths))
    assert before == pytest.approx(heavy[0] + heavy[1] + sum(heavy))

    refined, after, statistics = _refine(planner, rounds, lengths)
    # The long samples share one microbatch: each block has a slow and a fast microbatch.
    assert after == pytest.approx(heavy[0] + heavy[1] + sum(light))
    assert _shape(refined) == _shape(rounds)
    assert statistics["accepted"]["reassign"] > 0


def test_relocate_moves_wide_groups_that_reassign_has_to_keep():
    # Block 0 runs a group of four ranks in both microbatches; its samples need four ranks.
    lengths = dict(enumerate([200, 200] + [40] * 8))
    rounds = [
        [dict(cp=4, sample_ids=[0])] + [dict(cp=1, sample_ids=[i]) for i in (2, 3, 4, 5)],
        [dict(cp=4, sample_ids=[1])] + [dict(cp=1, sample_ids=[i]) for i in (6, 7, 8, 9)],
    ]
    wide, narrow = _cell(PRICE[4]), _cell(PRICE[1])
    kept, time_kept, _ = _refine(_tier_planner(8, 4, (1, 2, 4), moves=MOVES[:1]), rounds, lengths)
    assert kept is rounds
    assert time_kept == pytest.approx(wide[0] + wide[1] + sum(wide))

    planner = _tier_planner(8, 4, (1, 2, 4), moves=MOVES)
    refined, after, statistics = _refine(planner, rounds, lengths)
    # Every block now runs one wide and one narrow microbatch.
    assert after == pytest.approx(wide[0] + wide[1] + sum(narrow))
    assert sorted(map(sorted, _shape(refined))) == [[1] * 8, [4, 4]]
    assert statistics["accepted"]["relocate"] > 0


def _batch(seed, count, longest):
    rng = random.Random(seed)
    return [min(longest, max(1, int(rng.lognormvariate(7.0, 0.9)))) for _ in range(count)]


def _models():
    h100 = RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)
    usp = RuntimePlanGeometry(32, 4096, 8, (8, 16, 32), 64, 64, True)
    # Groups of four ranks cover two expert-parallel blocks of two.
    blocks = RuntimePlanGeometry(8, 4096, 2, (1, 2, 4), 16, 64, True)
    # Blocks of three ranks cut the groups of two, four and eight: no unit replays alone.
    cut = RuntimePlanGeometry(24, 4096, 3, (1, 2, 4, 8), 48, 64, True)
    return [
        pytest.param(h100, lambda: make_model("qwen3_h100_p2p", h100), 16384, "unit", id="h100"),
        pytest.param(
            usp,
            lambda: make_model("usp_calib_v1", usp, allow_provisional=True),
            32768,
            "unit",
            id="usp",
        ),
        pytest.param(blocks, lambda: _TierModel(blocks, 1e-4), 16384, "unit", id="blocks"),
        pytest.param(cut, lambda: _TierModel(cut, 1e-4), 32768, "prefix", id="cut"),
    ]


@pytest.mark.parametrize("geometry, model, longest, scorer", _models())
def test_refined_step_times_agree_with_the_planner_score(geometry, model, longest, scorer):
    """The search prices every trial itself; the planner scores the layout it starts from and
    the layout it returns with ``TimelineModel.evaluate``. Both have to tell the same time."""
    planner = DynamicUSPMovesPlanner(geometry, model(), moves=dict(moves=MOVES, time_budget_s=2.0))
    logical = dict(enumerate(_batch(1, 96, longest)))
    alignment = geometry.dataset_alignment
    padded = {i: (n + alignment - 1) // alignment * alignment for i, n in logical.items()}
    baseline = DynamicUSPRuntimePlanner(geometry, model()).plan_padded(
        padded, logical_lengths=logical
    )
    plan = planner.plan_padded(padded, logical_lengths=logical)
    validate_rows(plan.rows, padded, geometry)
    assert plan.predicted_ms <= baseline.predicted_ms
    report = planner.last_statistics["stage2_moves"]
    assert report["moves"] == list(MOVES) and report["scorer"] == scorer
    assert "skipped" not in report and "rejected" not in report
    # The layout the moves start from is the plan of the planner without them.
    assert report["start_predicted_ms"] == pytest.approx(baseline.predicted_ms, rel=1e-12)
    assert report["start_ms"] == pytest.approx(report["start_predicted_ms"], rel=1e-9)
    if "refined_ms" in report:
        assert report["end_ms"] == pytest.approx(report["refined_ms"], rel=1e-9)
        assert report["refined_ms"] < report["start_predicted_ms"]
        assert (report["planned"], plan.predicted_ms) == ("moves", report["refined_ms"])
    else:
        assert (report["planned"], plan.rows) == ("selected", baseline.rows)


def test_every_level_starts_from_the_result_of_the_level_before():
    geometry = RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True)
    logical = dict(enumerate(_batch(3, 64, 16384)))
    padded = {i: (n + 63) // 64 * 64 for i, n in logical.items()}
    predicted = []
    for level in range(3):
        planner = DynamicUSPMovesPlanner(
            geometry,
            make_model("qwen3_h100_p2p", geometry),
            moves=dict(moves=MOVES[:level], time_budget_s=60.0),
        )
        plan = planner.plan_padded(padded, logical_lengths=logical)
        validate_rows(plan.rows, padded, geometry)
        predicted.append(plan.predicted_ms)
        report = planner.last_statistics.get("stage2_moves")
        assert (report is None) == (level == 0) == (planner.refiner is None)
        if report is not None:
            assert report["converged"] and list(report["accepted"]) == list(MOVES[:level])
    assert predicted[2] <= predicted[1] < predicted[0]


@pytest.mark.parametrize(
    "value",
    [
        3,
        -1,
        True,
        "2",
        1.0,
        [1],
        dict(moves=["narrow"]),
        dict(moves="relocate"),
        dict(time_budget_s=0),
        dict(budget=1.0),
    ],
)
def test_invalid_options_are_rejected(value):
    with pytest.raises((TypeError, ValueError)):
        MoveOptions.parse(value)


def test_levels_and_objects_select_the_moves():
    assert [MoveOptions.parse(level).moves for level in range(3)] == [(), MOVES[:1], MOVES]
    assert not MoveOptions.parse(None).enabled and MoveOptions.parse(1).enabled
    chosen = MoveOptions.parse(dict(moves=["relocate", "reassign"], time_budget_s=0.25))
    assert (chosen.moves, chosen.time_budget_s) == (MOVES, 0.25)  # In the order they run.


def test_the_moves_refine_layouts_of_one_pipeline_stage_only():
    geometry = RuntimePlanGeometry(32, 4096, 8, (8, 16, 32), 64, 64, True)
    model = make_model("usp_calib_v1", geometry, allow_provisional=True)
    assert DynamicUSPMovesPlanner(geometry, model, pipeline=(2, 1)).refiner is None  # Off.
    with pytest.raises(ValueError, match="one pipeline stage only"):
        DynamicUSPMovesPlanner(geometry, model, moves=2, pipeline=(2, 1))
