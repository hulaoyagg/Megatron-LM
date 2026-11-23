# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""The DynamicUSP planner with a refinement of its layout by priced moves.

``DynamicUSPRuntimePlanner`` chooses the count of microbatches and the CP groups (Stage 1 and
the count search) and scores the layout of the default heuristic beside its own. This planner
does the same and then refines the layout it selected (``layout_moves``): samples move between
CP groups and CP groups between ranks, every candidate priced by the cost model and the
step timeline. The refined layout is scored again with the function that scored the others
and replaces the selected one only when its predicted step time is lower.

``"stage2"`` in the runtime file of scheduler type ``dynamic_usp_flops`` selects the moves;
without it this planner is ``DynamicUSPRuntimePlanner``.
"""

from __future__ import annotations

import math
import time

from .dynamic_usp import DynamicUSPRuntimePlanner
from .layout_moves import TOLERANCE, LayoutRefiner, MoveOptions
from .planner import RuntimeScheduleError


class DynamicUSPMovesPlanner(DynamicUSPRuntimePlanner):
    """``DynamicUSPRuntimePlanner`` whose selected layout is refined by priced moves."""

    def __init__(self, geometry, cost_model, *, moves=None, **options) -> None:
        super().__init__(geometry, cost_model, **options)
        self.move_options = MoveOptions.parse(moves)
        self.refiner = None
        if self.move_options.enabled:
            if self.pipeline[0] > 1:
                raise ValueError("stage2 refines layouts of one pipeline stage only")
            self.refiner = LayoutRefiner(
                self.geometry, self.scheduler.sizes, self.timeline, self.move_options
            )

    def _make_plan(self, rounds, lengths, predicted):
        if self.refiner is not None:
            rounds, predicted = self._refined(rounds, lengths, predicted)
        return super()._make_plan(rounds, lengths, predicted)

    def _refined(self, rounds, lengths, predicted):
        """The selected layout after the moves, with its step time; the statistics say which.

        ``last_statistics["stage2_moves"]``: the moves and their time budget, the step time
        of the selected layout (``start_predicted_ms``), what the search did (``accepted``
        and ``trials`` per move, ``converged``, ``elapsed_s``), the step time of the refined
        layout by the planner's own score (``refined_ms``) and ``planned``: ``"moves"`` when
        the refined layout is the plan, else ``"selected"``.
        """
        begin = time.perf_counter()
        options = self.move_options
        refined, report = self.refiner.refine(
            rounds, lengths, self.costs, begin + options.time_budget_s
        )
        report.update(
            moves=list(options.moves),
            time_budget_s=options.time_budget_s,
            start_predicted_ms=predicted,
            planned="selected",
        )
        if refined is not rounds:
            try:
                self._validated(refined, lengths)
                value = self.timeline.evaluate(self._costed(refined, lengths))
            except RuntimeScheduleError as error:
                # A refined layout that breaks the rank-row contract is dropped, not executed.
                report["rejected"] = str(error)
            else:
                report["refined_ms"] = value
                if math.isfinite(value) and value < predicted - TOLERANCE * max(1.0, predicted):
                    rounds, predicted = refined, value
                    report["planned"] = "moves"
        report["elapsed_s"] = time.perf_counter() - begin
        self.last_statistics["stage2_moves"] = report
        return rounds, predicted
