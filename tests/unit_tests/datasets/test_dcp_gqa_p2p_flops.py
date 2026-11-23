# Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.

"""Scheduler type dynamic_usp_flops: the formulas of its cost model, its wiring and its limits."""

import hashlib
import json
import random
import statistics
import subprocess
import sys
import time
from concurrent.futures import ThreadPoolExecutor
from pathlib import Path
from types import SimpleNamespace

import pytest
import test_dcp_schedule_prefetch as harness  # The rank threads of the prefetcher tests.
import torch

from megatron.core.datasets import dcp_layer_times
from megatron.core.datasets import dcp_runtime_worker as worker
from megatron.core.datasets import dcp_schedule_prefetch as prefetch
from megatron.core.datasets.data_schedule import validate_dynamic_cp_runtime, wrap_data_iterator
from megatron.core.datasets.dcp_runtime_scheduler import models
from megatron.core.datasets.dcp_runtime_scheduler.runtime.planner import (
    RuntimePlanGeometry,
    decode_groups,
)

PACKAGE = Path(models.__file__).parents[1]
PACKAGED = json.loads((PACKAGE / 'models/gqa_p2p_flops_h100.json').read_text())
RATES = PACKAGED['rates']
# fmt: off
# The model sizes of the recipe behind the packaged rates (Qwen3-30B-A3B) on 32 H100.
LAUNCH = dict(
    cp_comm_type='p2p', min_dynamic_context_parallel_size=1, max_seqlen_per_dp_cp_rank=4096,
    num_layers=48, hidden_size=2048, num_attention_heads=32, kv_channels=128, num_moe_experts=128,
    moe_router_topk=8, moe_ffn_hidden_size=768, expert_model_parallel_size=8,
)
# fmt: on
FILE = dict(type='dynamic_usp_flops', hardware='h100', max_cp_size=8)


def _spec(tmp_path, world=32, file=(), **settings):
    """Start-up validation of a run after changes to the runtime file or to the launch."""
    runtime = {key: value for key, value in {**FILE, **dict(file)}.items() if value is not None}
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(runtime))
    config = SimpleNamespace(dynamic_cp_runtime_config=str(path), **{**LAUNCH, **settings})
    return prefetch.runtime_spec(config, world)


def _planner(tmp_path, **changes):
    return worker._initialize_planner(_spec(tmp_path, **changes))


def _layer_ms(lengths, cp, rates=RATES, **sizes):
    """T_layer as the README writes it: FLOPs / (TFLOPS * 1e12) seconds, in ms."""
    size = {**LAUNCH, **sizes}
    pairs = sum(n * n / 2 for n in lengths) / cp
    pair_flops = 4 * size['num_attention_heads'] * size['kv_channels']
    feed_forward_flops = (
        2
        * size['max_seqlen_per_dp_cp_rank']
        * size['moe_router_topk']
        * 3
        * size['hidden_size']
        * size['moe_ffn_hidden_size']
    )
    compute_s = 3 * pair_flops * pairs / (rates['P_attn_tflops'] * 1e12)
    compute_s += 3 * feed_forward_flops / (rates['P_moe_tflops'] * 1e12)
    ring_steps = 2 * cp if cp > 1 else 0
    return rates['gamma'] * compute_s * 1e3 + ring_steps * rates['t_x_ms']


def _step_ms(planner, rounds, lengths):
    rounds = [[dict(cp=cp, sample_ids=ids) for cp, ids in row] for row in rounds]
    return planner.timeline.evaluate(planner._costed(rounds, lengths))


PACKS = [((4096,), 1), ((16384,), 4), ((1000, 3000, 250), 2), ((8192,) * 4, 8), ((1,) * 64, 8)]


@pytest.mark.parametrize('lengths,cp', PACKS)
def test_layer_time_is_the_formula_with_its_ring_steps_split_by_direction(tmp_path, lengths, cp):
    model = _planner(tmp_path).model
    assert model.layer_ms(lengths, cp) == pytest.approx(_layer_ms(lengths, cp), rel=1e-12)
    # The backward pass has twice the FLOPs; the ring has cp steps forward and cp backward.
    forward, backward = model.forward(lengths, cp), model.backward(lengths, cp)
    steps = cp * RATES['t_x_ms'] if cp > 1 else 0.0
    compute = forward - steps
    assert backward == pytest.approx(2 * compute + steps, rel=1e-12)
    assert forward + backward + sum(model.feed_forward_ms) == pytest.approx(
        model.layer_ms(lengths, cp)
    )
    assert model.program_costs(lengths, cp, ranks=range(cp)) == {
        'attention:forward': forward,
        'attention:backward': backward,
    }


def test_packaged_rates_give_these_layer_times(tmp_path):
    model = _planner(tmp_path).model
    assert model.layer_ms((4096,), 1) == pytest.approx(11.36, abs=0.005)
    assert model.layer_ms((16384,), 4) == pytest.approx(19.13, abs=0.005)


@pytest.mark.parametrize(
    'sizes',
    [
        dict(hidden_size=4096),
        dict(num_attention_heads=16, kv_channels=256),
        dict(moe_router_topk=4, moe_ffn_hidden_size=1536),
        dict(max_seqlen_per_dp_cp_rank=8192),
        dict(num_layers=24),
    ],
)
def test_model_sizes_are_those_of_the_training_configuration(tmp_path, sizes):
    planner = _planner(tmp_path, world=8, **sizes)
    for lengths, cp in ((4096,), 1), ((3000, 250), 2):
        expected = _layer_ms(lengths, cp, **sizes)
        assert planner.model.layer_ms(lengths, cp) == pytest.approx(expected, rel=1e-12)
    # One microbatch of eight equal CP1 groups: L layers and the step constant.
    layers = {**LAUNCH, **sizes}['num_layers']
    step = _step_ms(planner, [[(1, [i]) for i in range(8)]], dict.fromkeys(range(8), 4096))
    assert step == pytest.approx(layers * _layer_ms((4096,), 1, **sizes) + RATES['C_opt_ms'])


# Three microbatches on the eight ranks of one node; (CP size, tokens of its one sample) in rank
# order. Ranks 5 to 7 wait for rank 4 before the CP4 group; in ranks 0 to 3 the slowest group moves.
ROUNDS = [
    [(2, 3000), (1, 1024), (1, 512), (1, 4096), (1, 512), (1, 512), (1, 512)],
    [(1, 1024), (1, 512), (2, 6000), (4, 9000)],
    [(2, 3000), (1, 512), (1, 1024), (1, 512), (1, 4096), (1, 512), (1, 512)],
]
LENGTHS = dict(enumerate(tokens for row in ROUNDS for _, tokens in row))
_IDS = iter(LENGTHS)
ROUNDS = [[(cp, [next(_IDS)]) for cp, _ in row] for row in ROUNDS]


def _groups():
    """ROUNDS as (ranks, T_layer) per microbatch."""
    for row in ROUNDS:
        start, groups = 0, []
        for cp, ids in row:
            layer = _layer_ms([LENGTHS[i] for i in ids], cp)
            groups.append((range(start, start + cp), layer))
            start += cp
        yield groups


def test_without_expert_parallelism_a_group_waits_for_its_own_ranks_only(tmp_path):
    planner = _planner(tmp_path, world=8, expert_model_parallel_size=1)
    end = [0.0] * 8
    for groups in _groups():
        for ranks, layer in groups:
            done = max(end[rank] for rank in ranks) + LAUNCH['num_layers'] * layer
            for rank in ranks:
                end[rank] = done
    expected = max(end) + RATES['C_opt_ms']
    assert _step_ms(planner, ROUNDS, LENGTHS) == pytest.approx(expected, rel=1e-12)
    # Adding up the microbatches of every rank leaves out the wait for the other ranks of a group.
    sums = [0.0] * 8
    for groups in _groups():
        for ranks, layer in groups:
            for rank in ranks:
                sums[rank] += LAUNCH['num_layers'] * layer
    assert max(sums) + RATES['C_opt_ms'] < expected


@pytest.mark.parametrize('ep', [4, 8])  # No CP group of ROUNDS spans two blocks of 4 or 8 ranks.
def test_with_expert_parallelism_a_block_advances_with_its_slowest_group(tmp_path, ep):
    planner, layers = _planner(tmp_path, world=8, expert_model_parallel_size=ep), 48
    blocks = [0.0] * (8 // ep)
    for groups in _groups():
        slowest = [0.0] * len(blocks)
        for ranks, layer in groups:
            for block in {rank // ep for rank in ranks}:
                slowest[block] = max(slowest[block], layer)
        blocks = [total + layers * layer for total, layer in zip(blocks, slowest)]
    expected = max(blocks) + RATES['C_opt_ms']
    step = _step_ms(planner, ROUNDS, LENGTHS)
    # Below the formula by at most the backward attention of one layer per microbatch boundary.
    model = planner.model
    slack = max(model.backward((LENGTHS[i],), cp) for row in ROUNDS for cp, (i,) in row)
    assert expected - (len(ROUNDS) - 1) * slack <= step <= expected * (1 + 1e-12)
    # A smaller block waits less: no expert parallelism < blocks of 4 ranks < one block of 8.
    smaller = _planner(tmp_path, world=8, expert_model_parallel_size={4: 1, 8: 4}[ep])
    assert _step_ms(smaller, ROUNDS, LENGTHS) < step - 1.0


def test_runtime_file_selects_the_cost_model_and_the_dynamic_usp_solver_plans_with_it(
    tmp_path, monkeypatch
):
    # The handler of the CPU worker refuses a process with CUDA: on a machine with GPUs the
    # imports of this test process initialize it. The handler is given no Torch state to see.
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    spec = _spec(tmp_path)
    planner = worker._initialize_planner(spec)
    solver, sizes = type(planner).__module__.rsplit('.')[-1], spec['geometry']['allowed_cp_sizes']
    assert (spec['model'], solver, sizes) == ('gqa_p2p_flops', 'dynamic_usp', (1, 2, 4, 8))
    line = 'cost model=gqa_p2p_flops, hardware=h100 (NVLink domain: 8 consecutive ranks)'
    assert line in prefetch.describe_runtime_spec(spec)
    assert planner.costs.pack((1024,), 3, 0) is None  # No price for a CP size outside the list.
    assert not planner.timeline.first_forward_dp_cp_sync and not planner.stage2_enabled
    rng = random.Random(0)
    lengths = [min(16384, max(1, int(rng.lognormvariate(7.0, 0.9)))) for _ in range(256)]
    padded = [-(-length // 64) * 64 for length in lengths]
    result = json.loads(json.dumps(worker._solve(planner, spec, lengths, padded, None)))
    rounds = [[(cp, ids) for _, cp, ids in decode_groups(row)] for row in result['rows']]
    assert sorted(i for row in rounds for _, ids in row for i in ids) == list(range(256))
    assert result['predicted_ms'] == pytest.approx(
        _step_ms(planner, rounds, dict(enumerate(lengths)))
    )
    assert result['statistics']['stage1_weight'] == 'cost_model_sample_work'
    metadata = result['model_metadata']
    assert (metadata['status'], metadata['packaged_rates']) == ('provisional', True)
    assert metadata['rates'] == RATES and metadata['nvlink_domain_size'] == 8
    # Without expert parallelism the same file plans as well.
    alone = _planner(tmp_path, expert_model_parallel_size=1)
    assert alone.plan(dict(enumerate(lengths))).microbatches >= 1


@pytest.mark.parametrize(
    'changes,phrase',
    [
        (dict(cp_comm_type='a2a'), "cp_comm_type='p2p', got 'a2a'"),
        (dict(cp_partition_mode='contiguous'), "cp_partition_mode='zigzag', got 'contiguous'"),
        (dict(tensor_model_parallel_size=2), 'tensor_model_parallel_size=1, got 2'),
        (dict(num_moe_experts=None), 'this model has num_moe_experts=None'),
        (dict(moe_layer_freq=[1, 0]), 'moe_layer_freq=1, got [1, 0]'),
        (dict(recompute_granularity='full'), "recompute_granularity unset, got 'full'"),
        (dict(experimental_attention_variant='gdn'), 'prices ordinary attention in every layer'),
        (dict(hidden_size=None), 'reads hidden_size from the training configuration'),
        (dict(file=dict(max_cp_size=None)), 'set "max_cp_size": 8 in the runtime file'),
        (dict(file=dict(hardware='gb200')), "are packaged for hardware 'gb200'"),
        (dict(file=dict(hardware=None)), "requires key(s) ['hardware']"),
        (dict(file=dict(non_attention={})), "unknown key(s) ['non_attention']"),
    ],
)
def test_start_up_names_what_the_formulas_do_not_describe(tmp_path, changes, phrase):
    with pytest.raises(ValueError) as caught:
        _spec(tmp_path, **changes)
    # Only a fault of the launch is reported as a mismatch with the training configuration.
    mismatch = 'does not match the training configuration' in str(caught.value)
    assert phrase in str(caught.value) and mismatch == ('file' not in changes)


def test_more_than_one_pipeline_stage_is_refused(tmp_path):
    path = tmp_path / 'runtime.json'
    path.write_text(json.dumps(FILE))
    config = SimpleNamespace(dynamic_cp_runtime_config=str(path), **LAUNCH)
    with pytest.raises(ValueError, match='plans for more than one pipeline stage only'):
        prefetch.runtime_spec(config, 32, pp_size=2)


def test_coefficient_file_beside_the_runtime_file_replaces_the_packaged_rates(tmp_path):
    rates = dict(RATES, gamma=1.25, t_x_ms=0.5, C_opt_ms=50.0)
    raw = json.dumps(dict(PACKAGED, status='measured', rates=rates))
    (tmp_path / 'rates.json').write_text(raw)
    spec = _spec(tmp_path, world=8, file=dict(calibration='rates.json'))
    assert spec['calibration_sha256'] == hashlib.sha256(raw.encode()).hexdigest()
    planner = worker._initialize_planner(spec)
    for lengths, cp in PACKS[:3]:
        expected = _layer_ms(lengths, cp, rates)
        assert planner.model.layer_ms(lengths, cp) == pytest.approx(expected, rel=1e-12)
    step = _step_ms(planner, [[(1, [i]) for i in range(8)]], dict.fromkeys(range(8), 4096))
    assert step == pytest.approx(48 * _layer_ms((4096,), 1, rates) + 50.0)
    metadata = planner.model.metadata()
    assert (metadata['status'], metadata['packaged_rates']) == ('measured', False)


@pytest.mark.parametrize(
    'change,phrase',
    [
        (
            dict(hardware='gb200'),
            "holds rates of hardware 'gb200'; the runtime file selects 'h100'",
        ),
        (dict(name='qwen3_h100_p2p'), "is not a 'gqa_p2p_flops' file"),
        (dict(rates={k: v for k, v in RATES.items() if k != 'gamma'}), 'takes exactly the keys'),
        (dict(rates=dict(RATES, P_attn_tflops=5.7e14)), 'the unit is TFLOPS'),
        (dict(rates=dict(RATES, gamma=0)), 'rates.gamma must be a finite positive number'),
        (dict(rates=dict(RATES, t_x_ms=True)), 'rates.t_x_ms must be a finite nonnegative number'),
        (dict(source={}), '"source" must say how the rates were obtained'),
    ],
)
def test_malformed_coefficient_file_is_refused_at_start_up(tmp_path, change, phrase):
    (tmp_path / 'rates.json').write_text(json.dumps(dict(PACKAGED, **change)))
    with pytest.raises(ValueError) as caught:
        _spec(tmp_path, file=dict(calibration='rates.json'))
    assert phrase in str(caught.value)
    assert 'does not match the training configuration' not in str(caught.value)


def test_packaged_rates_are_the_stated_fit_to_the_h100_calibration():
    """Model against model: what ``source`` of the packaged file says, checked on the 60 cases."""
    geometry = RuntimePlanGeometry(32, 4096, 8, (1, 2, 4, 8), 64, 64, True, 8)
    table = models.make_model('qwen3_h100_p2p', geometry)
    options = models.model_class('gqa_p2p_flops').options_from_config(SimpleNamespace(**LAUNCH))
    model = models.make_model('gqa_p2p_flops', geometry, hardware='h100', **options)
    pins = Path(__file__).parent / 'dcp_qwen3_h100_p2p_frozen_costs.json'
    errors = []
    for case in json.loads(pins.read_text())['cases']:
        lengths, cp = tuple(case['lengths']), case['cp']
        table_ms = sum(table.program_costs(lengths, cp).values())
        errors.append(abs(model.layer_ms(lengths, cp) / table_ms - 1))
    fit = PACKAGED['source']['fit']
    assert len(errors) == fit['cases'] == 60
    assert statistics.median(errors) == pytest.approx(fit['layer_error_median'], abs=5e-4)
    assert max(errors) == pytest.approx(fit['layer_error_max'], abs=5e-4)
    assert RATES['C_opt_ms'] == pytest.approx(table.iteration_tail_ms, abs=1e-3)


# ----------------------------------------------------------- Stage 2: priced moves
def _long_tailed(seed, count=128):
    """Sample lengths with a long tail: most are short, a few need the widest CP group."""
    rng = random.Random(seed)
    return [min(32768, max(64, int(rng.lognormvariate(6.2, 1.5)) // 16 * 16)) for _ in range(count)]


@pytest.mark.parametrize('ep', [8, 1])
def test_stage2_of_the_runtime_file_refines_the_selected_layout_by_moves(tmp_path, monkeypatch, ep):
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))  # As above: no Torch state.
    launch = dict(world=8, expert_model_parallel_size=ep)
    off, on = _spec(tmp_path, **launch), _spec(tmp_path, file=dict(stage2=2), **launch)
    line = 'Stage 2 by moves (reassign, relocate; at most 1 s per global batch)'
    assert line in prefetch.describe_runtime_spec(on)
    assert 'Stage 2' not in prefetch.describe_runtime_spec(off)
    assert (off['solver_options'], on['solver_options']) == ({}, dict(moves=2))
    base, refined = worker._initialize_planner(off), worker._initialize_planner(on)
    names = [type(planner).__name__ for planner in (base, refined)]
    assert names == ['DynamicUSPRuntimePlanner', 'DynamicUSPMovesPlanner']
    gains = []
    for seed in range(4):
        lengths = _long_tailed(seed)
        before = worker._solve(base, off, lengths, lengths, None)
        after = worker._solve(refined, on, lengths, lengths, None)
        report = after['statistics']['stage2_moves']
        assert 'stage2_moves' not in before['statistics']
        # The moves start from the layout that the planner without them selects.
        assert report['start_predicted_ms'] == pytest.approx(before['predicted_ms'], rel=1e-12)
        # The worker validated the rows; every sample is in one group of one microbatch.
        placed = [i for row in after['rows'] for _, _, ids in decode_groups(row) for i in ids]
        assert sorted(placed) == list(range(len(lengths)))
        assert len(after['rows']) == len(before['rows'])  # The count of microbatches stays.
        if report['planned'] == 'moves':
            assert after['predicted_ms'] == report['refined_ms'] < before['predicted_ms']
            rounds = [[(cp, ids) for _, cp, ids in decode_groups(row)] for row in after['rows']]
            again = _step_ms(refined, rounds, dict(enumerate(lengths)))
            assert again == pytest.approx(after['predicted_ms'], rel=1e-12)
        else:
            assert after['rows'] == before['rows']
        gains.append(after['predicted_ms'] / before['predicted_ms'] - 1)
    # Never slower. With one expert-parallel block over all ranks a microbatch takes what its
    # slowest group takes, and the moves gain little; without the blocks they gain more.
    assert max(gains) <= 0 and min(gains) < (-0.002 if ep == 1 else 0)


def test_stage2_moves_with_expert_parallel_blocks_that_cut_cp_groups(tmp_path, monkeypatch):
    """Blocks of 3 ranks and CP groups of 2, 4 or 8: no unit of the step can be replayed alone."""
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    launch = dict(world=24, expert_model_parallel_size=3)
    off, on = _spec(tmp_path, **launch), _spec(tmp_path, file=dict(stage2=2), **launch)
    base, refined = worker._initialize_planner(off), worker._initialize_planner(on)
    rng = random.Random(1)
    # Multiples of 48, the dataset alignment of 24 ranks, up to eight ranks of 4096 tokens.
    lengths = [
        min(32736, max(96, int(rng.lognormvariate(6.2, 1.5)) // 48 * 48)) for _ in range(384)
    ]
    before = worker._solve(base, off, lengths, lengths, None)
    after = worker._solve(refined, on, lengths, lengths, None)
    report = after['statistics']['stage2_moves']
    assert report['scorer'] == 'prefix' and report['start_ms'] == pytest.approx(
        before['predicted_ms']
    )
    assert after['predicted_ms'] <= before['predicted_ms']
    placed = [i for row in after['rows'] for _, _, ids in decode_groups(row) for i in ids]
    assert sorted(placed) == list(range(len(lengths)))
    if report['planned'] == 'moves':
        rounds = [[(cp, ids) for _, cp, ids in decode_groups(row)] for row in after['rows']]
        again = _step_ms(refined, rounds, dict(enumerate(lengths)))
        assert (
            again
            == pytest.approx(report['end_ms'], rel=1e-12)
            == pytest.approx(report['refined_ms'])
        )


def test_stage2_levels_and_values_that_the_runtime_file_refuses(tmp_path, monkeypatch):
    monkeypatch.setattr(worker, 'sys', SimpleNamespace(modules={}))
    lengths = _long_tailed(0)
    base = worker._initialize_planner(_spec(tmp_path, world=8))
    plan = worker._solve(base, _spec(tmp_path, world=8), lengths, lengths, None)
    # Level 0 is off: the planner refines nothing and plans as the one without the key.
    spec = _spec(tmp_path, world=8, file=dict(stage2=0))
    planner = worker._initialize_planner(spec)
    assert planner.refiner is None and 'Stage 2' not in prefetch.describe_runtime_spec(spec)
    same = worker._solve(planner, spec, lengths, lengths, None)
    assert same['rows'] == plan['rows'] and 'stage2_moves' not in same['statistics']
    # Level 1 moves samples only; an object names the moves and the seconds per global batch.
    spec = _spec(tmp_path, world=8, file=dict(stage2=1))
    assert worker._initialize_planner(spec).move_options.moves == ('reassign',)
    spec = _spec(tmp_path, world=8, file=dict(stage2=dict(moves=['relocate'], time_budget_s=0.05)))
    planner = worker._initialize_planner(spec)
    assert (planner.move_options.moves, planner.move_options.time_budget_s) == (('relocate',), 0.05)
    report = worker._solve(planner, spec, lengths, lengths, None)['statistics']['stage2_moves']
    assert report['moves'] == ['relocate'] and report['elapsed_s'] < 1.0
    for value, phrase in (
        (3, 'stage2 must be a level from 0 to 2, got 3'),
        (True, 'stage2 must be a level from 0 to 2 or an object'),
        (dict(moves=['swap']), "stage2 moves must be taken from ['reassign', 'relocate']"),
        (dict(budget=1), "got unknown key(s) ['budget']"),
        (dict(time_budget_s=0), 'stage2 time_budget_s must be a finite positive number'),
    ):
        with pytest.raises(ValueError) as caught:
            _spec(tmp_path, world=8, file=dict(stage2=value))
        assert phrase in str(caught.value) and 'runtime.json' in str(caught.value)


# ------------------------------------------------------------------ the calibration skill
SKILL_SCRIPT = (
    Path(__file__).parents[3] / 'skills/mcore-calibrate-dcp-flops/scripts/dcp_flops_calibration.py'
)


def _skill(*arguments):
    """Run the script of the calibration skill as a user does: a process without torch."""
    command = [sys.executable, str(SKILL_SCRIPT), *map(str, arguments)]
    return subprocess.run(command, capture_output=True, text=True, timeout=300)


def test_calibration_skill_recovers_known_coefficients_from_synthetic_traces():
    done = _skill('selftest')
    assert done.returncode == 0, done.stdout[-3000:] + done.stderr[-3000:]
    lines = done.stdout.splitlines()
    assert lines[-1].startswith('selftest: 42 of 42 checks passed')
    assert not [line for line in lines if line.startswith('FAIL')]  # Every check prints "ok".


def test_calibration_skill_reads_the_trace_that_the_prefetcher_writes(tmp_path, monkeypatch):
    """Eight ranks as threads train 45 steps with the real CPU worker; the script reads the trace.

    The ranks are threads of one process, so the layer times of a rank are made up here: one
    entry per microbatch of the step that the rank just ran, as its layer timers give them.
    """
    sim = harness._Ranks(monkeypatch, 1)
    sim.fault, sim.plans = None, 2
    file = dict(type='dynamic_usp_flops', hardware='h100', trace_dir=str(tmp_path / 'trace'))
    file['layer_times'] = True
    launch = dict(LAUNCH, max_seqlen_per_dp_cp_rank=1024, num_layers=12)

    def timed():
        entry = {f'{kind}_count': 12 for kind in dcp_layer_times._KINDS}
        entry.update(layer_forward_ms=40.0, layer_backward_ms=80.0 + sim.local.rank)
        entry.update(attention_forward_ms=12.0, attention_backward_ms=24.0, after_layers_ms=1.0)
        return [dict(entry) for _ in range(getattr(sim.local, 'microbatches', 0))]

    monkeypatch.setattr(prefetch, 'collect_layer_times', timed)

    def body(rank):
        sim.local.rank, rng = rank, random.Random(0)
        items = [[rng.randint(100, 6000) for _ in range(6)] for _ in range(94)]
        data = harness._Rerun(
            iterable=iter(items), items=items, replay=iter(items), replaying=False
        )
        config, groups = harness._config(
            tmp_path / f'{rank}.json', file, None, **launch
        ), sim.groups(rank)
        groups.dynamic_cp = dict.fromkeys((1, 2, 4, 8))
        validate_dynamic_cp_runtime(config, groups)
        for _ in range(45):
            wrap_data_iterator(data, config, 2, groups, prefetch_next=True)
            sim.local.microbatches = len(data._dcp_schedule_prefetcher.last_telemetry['rows'])
        data._dcp_schedule_prefetcher.close()

    with ThreadPoolExecutor(8) as pool:
        list(pool.map(body, range(8)))
    done = _skill('check', '--trace', tmp_path / 'trace')
    # The steps of this harness take CPU milliseconds, so the verdict is FAIL. What is checked:
    # the record format, and that the script prices every recorded layout as the run did.
    assert done.returncode == 1 and not done.stderr, done.stdout[-2000:] + done.stderr[-2000:]
    assert '45 training steps, ' in done.stdout and 'VERDICT: FAIL' in done.stdout
    assert 'CP groups in these layouts: ' in done.stdout
    # Every rank file gives the layer times of every used step, taken from the record after it.
    used = int(done.stdout.split(' training steps, ')[1].split(' used')[0])
    assert f'layer times: of {used} of the {used} used steps, from 8 rank files' in done.stdout
    assert 'one layer of a CP group, predicted / measured, ' in done.stdout
    first = [json.loads(line) for line in (tmp_path / 'trace/rank00000.jsonl').open()]
    first = [record for record in first if record['event'] == 'consume']
    assert first[0]['previous_layer_times'] == []  # No step ran before the first one.
    assert len(first[1]['previous_layer_times']) == len(first[0]['rows'])


# ---------------------------------------------------------------------- the layer timers
class _Attention(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.qkv, self.proj = torch.nn.Linear(16, 16), torch.nn.Linear(16, 16)

    def forward(self, hidden_states, attention_mask=None):
        time.sleep(0.003)
        return self.proj(torch.tanh(self.qkv(hidden_states))), None


class _Layer(torch.nn.Module):
    """The shape of a transformer layer: norm, attention, in-place dropout, residual, MLP."""

    def __init__(self):
        super().__init__()
        self.input_layernorm, self.self_attention = torch.nn.LayerNorm(16), _Attention()
        self.pre_mlp_layernorm, self.mlp = torch.nn.LayerNorm(16), torch.nn.Linear(16, 16)

    def forward(self, hidden_states, /, **kwargs):
        output, _ = self.self_attention(self.input_layernorm(hidden_states), attention_mask=None)
        # As the fused bias-dropout-add of Megatron: in place on the attention output.
        output = torch.nn.functional.dropout(output, p=0.1, training=self.training, inplace=True)
        hidden_states = hidden_states + output
        time.sleep(0.002)
        return hidden_states + self.mlp(self.pre_mlp_layernorm(hidden_states)), None


class _Model(torch.nn.Module):
    def __init__(self, layers=3):
        super().__init__()
        self.embedding, self.output_layer = torch.nn.Linear(8, 16), torch.nn.Linear(16, 1)
        self.layers = torch.nn.ModuleList(_Layer() for _ in range(layers))

    def forward(self, tokens):
        hidden_states = self.embedding(tokens)
        for layer in self.layers:
            hidden_states, _ = layer(hidden_states, attention_mask=None)
        time.sleep(0.004)
        return self.output_layer(hidden_states).sum()


def test_layer_timers_time_every_layer_forward_and_backward(monkeypatch):
    monkeypatch.setattr(dcp_layer_times, '_TIMER', None)
    with pytest.raises(RuntimeError, match='no layer timer is attached'):
        dcp_layer_times.collect_layer_times()
    torch.manual_seed(0)
    model = _Model()
    untimed = _Model()
    untimed.load_state_dict(model.state_dict())
    assert dcp_layer_times.attach_layer_timers([model]) == 3  # Training holds a list of one.
    assert dcp_layer_times.collect_layer_times() == []
    tokens = torch.randn(4, 8)
    for _ in range(2):  # Two microbatches of one step.
        model(tokens).backward()
    model.eval()
    with torch.no_grad():
        model(tokens)  # An evaluation is not timed.
    model.train()
    first, second = dcp_layer_times.collect_layer_times()
    assert dcp_layer_times.collect_layer_times() == []  # A step is given once.
    for entry in (first, second):
        assert [entry[f'{kind}_count'] for kind in dcp_layer_times._KINDS] == [3, 3, 3, 3]
        # The sleeps: 3 ms in attention and 2 ms in the rest of each of the three layers,
        # 4 ms between the last layer and its backward pass.
        assert 9.0 <= entry['attention_forward_ms'] <= entry['layer_forward_ms'] - 6.0
        assert 0.0 < entry['attention_backward_ms'] < entry['layer_backward_ms']
        assert entry['after_layers_ms'] >= 4.0
    # The timers change no gradient. Dropout draws the same numbers in both models.
    for net in (model, untimed):
        net.zero_grad()
        torch.manual_seed(1)
        net(tokens).backward()
    for timed, plain in zip(model.parameters(), untimed.parameters()):
        assert torch.equal(timed.grad, plain.grad)
    # A rank whose times nobody collects keeps the latest microbatches only.
    monkeypatch.setattr(dcp_layer_times, '_MAX_PENDING', 2)
    for _ in range(3):
        model(tokens).backward()
    assert len(dcp_layer_times.collect_layer_times()) == 2


@pytest.mark.skipif(not torch.cuda.is_available(), reason='CUDA events need a GPU')
def test_layer_timers_read_cuda_events_for_a_model_on_a_gpu(monkeypatch):
    monkeypatch.setattr(dcp_layer_times, '_TIMER', None)
    model = _Model().cuda()
    assert dcp_layer_times.attach_layer_timers(model) == 3 and dcp_layer_times._TIMER.cuda
    tokens = torch.randn(4, 8, device='cuda')
    for _ in range(3):
        model(tokens).backward()
    for entry in dcp_layer_times.collect_layer_times()[1:]:  # The first pass initializes CUDA.
        assert [entry[f'{kind}_count'] for kind in dcp_layer_times._KINDS] == [3, 3, 3, 3]
        # The host sleeps inside the modules; a mark is taken when the idle GPU reaches it.
        assert 9.0 <= entry['attention_forward_ms'] <= entry['layer_forward_ms'] - 6.0
        assert 0.0 < entry['attention_backward_ms'] < entry['layer_backward_ms']
        assert entry['after_layers_ms'] >= 4.0


def test_layer_timers_name_what_they_cannot_time(monkeypatch):
    monkeypatch.setattr(dcp_layer_times, '_TIMER', None)
    with pytest.raises(ValueError, match='no transformer layer to time'):
        dcp_layer_times.attach_layer_timers(torch.nn.Linear(2, 2))
    with pytest.raises(ValueError, match='this rank holds 2 model chunks'):
        dcp_layer_times.attach_layer_timers([_Model(), _Model()])
    graphs = _Model()
    graphs.layers[0].config = SimpleNamespace(cuda_graph_impl='local')
    with pytest.raises(ValueError, match="CUDA graphs .cuda_graph_impl='local'."):
        dcp_layer_times.attach_layer_timers(graphs)
    assert dcp_layer_times.attach_layer_timers(_Model(2)) == 2
    with pytest.raises(RuntimeError, match='a second one was built'):
        dcp_layer_times.attach_layer_timers(_Model())


def test_layer_times_are_a_switch_of_the_runtime_file_that_needs_a_trace(tmp_path):
    assert _spec(tmp_path)['layer_times'] is False
    with pytest.raises(ValueError, match='layer_times writes into the trace; it needs trace_dir'):
        _spec(tmp_path, file=dict(layer_times=True))
    with pytest.raises(ValueError, match='layer_times must be true or false, got 1'):
        _spec(tmp_path, file=dict(layer_times=1, trace_dir=str(tmp_path / 'trace')))
    spec = _spec(tmp_path, file=dict(layer_times=True, trace_dir=str(tmp_path / 'trace')))
    assert spec['layer_times'] is True
    assert ', layer times recorded' in prefetch.describe_runtime_spec(spec)
