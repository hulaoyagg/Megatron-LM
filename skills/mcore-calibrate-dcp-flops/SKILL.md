---
name: mcore-calibrate-dcp-flops
description: Check or fit the coefficient file of the dynamic_usp_flops runtime DCP scheduler (FLOPs-over-rate cost model for GQA with an MoE feed-forward under classic P2P) from the trace of a training run. Use when its predicted step time should match a model, hardware or software stack, after such a change, when rates for new hardware are to be published, or to see whether a coefficient file still fits. Two CPU commands on a trace; the layer timers ship with Megatron and are switched on in the runtime file; no code changes.
---

# Calibrate the `dynamic_usp_flops` cost model

Work relative to the Megatron-LM checkout. The cost model, its formulas and the
runtime file are described in
`megatron/core/datasets/dcp_runtime_scheduler/README.md`
(section "`dynamic_usp_flops` cost model"). `$S` is
`skills/mcore-calibrate-dcp-flops/scripts/dcp_flops_calibration.py`; it needs
numpy and the checkout, no GPU.

## What is fitted

The cost model has five numbers in a coefficient file. Everything else it uses
is read from the training configuration.

| Number | Meaning | Who provides it | Fit level |
| --- | --- | --- | --- |
| `P_attn_tflops` | Rate of attention | Published per hardware | `rates` |
| `P_moe_tflops` | Rate of the MoE feed-forward | Published per hardware | `rates` |
| `t_x_ms` | Time of one ring step of the P2P attention, inside an NVLink domain; a group of `c` ranks runs `2c` of them per layer | Published per hardware | `rates` |
| `gamma` | Factor on the two compute terms | Every user, for the own run | `gamma` |
| `C_opt_ms` | Time added once per step | Every user, for the own run | both |

Level `gamma` (the default) keeps the published rates and fits `gamma` and
`C_opt_ms`. Level `rates` fits the three rates and `C_opt_ms` with `gamma` 1; it
is for publishing rates and for a run that level `gamma` does not describe.

The first four numbers describe one layer of one CP group. A run with
`"layer_times": true` records the time of the layers on every rank, so every CP
group of every microbatch gives one measured layer time: about a hundred
equations per step. The fit uses them when the trace has them, and the step
times, one equation per step, when it does not.

## Rules

- Run the commands as written and report their output lines unchanged. The last
  line of `check` and `fit` is the verdict and names the next step.
- Do not add timers of your own. The layer timers ship with Megatron and are
  switched on by one key of the runtime file (step 1).
- Do not change the cost model or its formulas, or edit the constants of the
  script. Every choice of the measurement (which steps count, how they are
  split, when a fit is accepted) is made by the script.
- Change the launch only where a step below says so.
- Stop and ask the user when the script prints `ERROR`, when level `rates`
  ends in `FAIL`, and before starting any run on GPUs that the user did not
  ask for.

## Procedure

### 1. Train with a trace and layer times

Add `trace_dir` and `layer_times` to the runtime file of the training run and
train at least 40 steps (the first 10 are not used). Nothing else changes.

```json
{
  "type": "dynamic_usp_flops",
  "hardware": "h100",
  "max_cp_size": 8,
  "trace_dir": "trace/",
  "layer_times": true
}
```

The start-up line `Runtime DCP scheduler: ...` then ends with `layer times
recorded`. The key is for the calibration run; take it out again for training.

Give every run a new, empty `trace_dir`: a run appends to the rank files it
finds, and the script refuses a directory that two runs wrote into. A relative
`trace_dir` is relative to the working directory of the launch (a relative
`calibration` is relative to the runtime file). Keep the `rank*.jsonl` file of
every rank.

A run that already has a trace of this scheduler type can be used as it is.
Without layer times the commands below work from the step times: train 70
steps then, and expect less certain values (see Limits).

### 2. Check the file in use

```bash
python $S check --trace trace/
```

`VERDICT: PASS`: the coefficient file of the run is good enough; stop here.
`VERDICT: FAIL`: go on.

With `--log train.log` the command also compares the time between trace
records with the "elapsed time per iteration" of the training log. Do this the
first time a cluster or a software stack is calibrated; summed over the same
steps the two should agree within 2 %. The log needs one line per iteration
(`--log-interval 1`).

### 3. Fit `gamma` and `C_opt_ms`

```bash
python $S fit --trace trace/ --out flops_rates.json
```

`PASS`: name the file in the runtime file (`"calibration": "flops_rates.json"`,
a relative path is resolved beside the runtime file) and use it. `FAIL`: go on.
When the task is to obtain rates for a hardware, skip this step: see
[Publishing rates for a hardware](#publishing-rates-for-a-hardware).

### 4. Fit the rates

```bash
python $S fit --trace trace/ --out flops_rates.json --level rates
```

`PASS`: use the file as in step 3. `FAIL`: stop and report; the cost model does
not describe this run. Without layer times this level needs 40 usable steps.

### 5. Coverage run, when the script asks for it

A training run forms only the CP group sizes its samples need. When a size the
scheduler may form never occurs, `check` and `fit --level rates` say so and
print the data arguments of a coverage run: the same launch with mock samples
of every length up to the largest CP group.

```bash
python $S coverage --trace trace/
```

Run the training launch for as many steps as in step 1 with those arguments in
place of its own dataset arguments and `--seq-length` (the model, the
tokenizer, the parallel sizes and the batch size stay), and with another
`trace_dir`. Then fit on both traces:

```bash
python $S fit --trace trace/ --trace trace_coverage/ --out flops_rates.json --level rates
```

Fit on both traces, not on the coverage trace alone: the training trace keeps
the rates accurate for the layouts that training runs, and the coverage trace
adds the group sizes those layouts lack. For level `gamma` no coverage run is
needed.

Two things to know about a coverage run:

- With `--attention-backend flash` the printed data arguments do not start:
  FlashAttention takes no padding between the samples of a pack, so every
  sample length must be a multiple of twice DPxCP. Draw the lengths from the
  same distribution, round them, and give them as a file
  (`{"mode":"file","format":"thd","path":...}`; the mock dataset keeps the
  value of a line less one as the sample length).
- Its samples are longer, so a step has about twice the microbatches of a
  training step. `C_opt_ms` is one number per step: fitted on both traces it
  is too large for the training steps (see Limits). The three rates are not
  affected.

## Reading the output

With layer times `check` and `fit` print four more lines. Report them with the
verdict; they do not change it.

| Line | What it says |
| --- | --- |
| `attention time of the ranks of one CP group, largest / smallest` | 1.0 when the ranks of a group enter attention together. Above 1 a rank was free before the others of its group and waited for them; that is expected without expert parallelism. The script uses the smallest time of the group. |
| `layer time of the ranks of one expert-parallel block, largest / smallest` | The cost model lets a block advance together, at the pace of its slowest group: 1.0. |
| `the timed layers take N % of a step ...` | What a step spends outside its layers, and how much of that after the last layer of every microbatch. The cost model has one constant per step for all of it, `C_opt_ms`. |
| `one layer of a CP group, predicted / measured ...` | The coefficients against the measured layer of every CP group, overall and by group size. A size that is off while the others fit points at `t_x_ms`. |
| `N more layer times were not used in the fit ...` | Only `fit`: layers that a pause made several times longer, most often the garbage collection of the interpreter. Expected for a few percent of the layer times. Above a tenth the line says so: the cost model then misses a kind of layer; report the line and the ratios by group size. |

## The acceptance band

A file passes when the median of predicted / measured step time, on steps the
fit did not use, lies between 0.8 and 1.2. This default is deliberately wide.
`--low` and `--high` change it, for `check` and for `fit`; choose the band for
the experiment (for example 0.95 to 1.05 before publishing rates, or the
default while a new setup is still being brought up) and say which band was
used when reporting. The output always shows the median, the 10th and 90th
percentile and the range, so a result can be judged against another band
without running again.

## What the script does

- **Measured step time.** A trace record is written when a step receives its
  layout. The time of a step is the time from its record to the next one, less
  the time the next step waited for its plan (`residual_wait_s`). It is the
  complete loop of a step. The logged time per iteration covers the same loop
  from the end of one step to the end of the next; the two agree on average,
  and step by step up to the time of reading a batch.
- **Predicted step time.** The script imports the cost model and the timeline
  of the checkout and prices the recorded layout with them, as training does.
  It first prices every step with the coefficients of the run and stops when
  the result differs from the recorded prediction.
- **Steps that are not used.** The first 10 steps of a trace (`--skip`), a step
  with an evaluation or a replay before the next one, and a step whose
  measured / predicted is more than 1.5 times away from the median
  (`--outlier`; a checkpoint, for example). The first output line counts them.
- **Measured layer time of a CP group.** Every rank records, per microbatch,
  the time of its layers and of the attention inside them, waits included. The
  ranks of a CP group leave attention together; a rank that was free before
  the others waits for them inside its first attention, so the attention time
  of a group is the smallest of its ranks, that of the rank that entered
  last. With expert parallelism every rank of a block also waits for the
  slowest group of the block in the feed-forward; the ranks of that group do
  not wait there, so their layer time less their attention time is the rest
  of the layer, the same for the whole block. The layer time of a group by
  itself is its attention time plus that rest, divided by the number of
  layers.
- **Fit from layer times** (`--from layers`, the default when every rank
  recorded them). The layer time of a group is the sum of `gamma / P_attn`,
  `gamma / P_moe` and `t_x` times constants that the script takes from the
  cost model. One least-squares problem without negative coefficients over
  the CP groups of the steps at even positions gives the rates, or, for level
  `gamma`, the one factor on the two compute terms. A layer time more than
  1.5 times (`--outlier`) away from the median of measured / fitted is then
  left out and the fit repeated: a pause inside a layer, such as a garbage
  collection, makes the layers of one microbatch several times longer on
  every rank. `C_opt_ms` is then the mean of what the measured step times
  leave above the predicted ones; the pauses are in it.
- **Fit from step times** (`--from steps`, and when layer times are missing).
  On one critical path the step time is a sum of terms that are each one of
  `gamma / P_attn`, `gamma / P_moe`, `t_x` and `C_opt` times a constant. The
  script takes those constants as finite differences of the timeline, solves
  the least-squares problem on the steps at even positions, and repeats until
  the critical paths no longer change.
- **Verdict.** On the step times of the steps at odd positions, for both ways
  of fitting. The file records the traces, the steps used, the fit and the
  verdict under `source`.

`python $S selftest` runs all of this on synthetic traces with known
coefficients, with and without expert parallelism, layer times and pauses (15 s on a
CPU). Run it after a change to the cost model or to this script.

## Messages

| Message | Do |
| --- | --- |
| `was written by cost model ...` | The run used another scheduler type. This skill needs a trace of `dynamic_usp_flops`. |
| `prices the recorded layouts differently` | Use the checkout the run was started from. Only when the formulas of the cost model were changed on purpose after the run: add `--model-changed`, which uses the layouts and measured times of the trace as they are. |
| `needs at least N usable steps` | Train longer with the trace, or lower `--skip` when the first steps are known to be regular. |
| `the step counter starts again ... more than one run` | Two runs wrote into the directory. Run again with a new `trace_dir`. |
| `fitted from step times: ... have no layer times: ...` | The reason follows the colon. To fit from layer times, run step 1 with `"layer_times": true` and keep the `rank*.jsonl` file of every rank in the trace directory. |
| `a rank did not time all N layers in both directions` | Report it with the launch arguments: the timers do not cover this model or this configuration. Fit with `--from steps` meanwhile. |
| `the CP groups are too alike`, `the steps are too alike` | Every group has the same size, or every step the same number of microbatches. Use training data of varying length, or add a coverage run. |
| `the fit gives gamma = 0` | The measured times do not follow the predictions at all. Check that the trace is of the run that is meant. |
| `do not determine P_attn_tflops` | Add steps or a coverage run, or stay at level `gamma`. |
| `fitted at zero: [...]` | A term had no measurable cost in these steps (often `t_x_ms` when no layout had a CP group above one rank). Report it; add a coverage run before publishing. |

## Publishing rates for a hardware

Rates are published by fitting level `rates` on a reference recipe: a training
trace and a coverage trace of the same launch, both with layer times, at least
60 usable steps each (a run of 71 steps: the first 10 and the last are not
used), with a band chosen for the purpose. Level `gamma` is skipped, whatever
`check` says:

```bash
python $S coverage --trace trace/          # data arguments of the coverage run
python $S fit --trace trace/ --trace trace_coverage/ --out flops_rates.json --level rates --low 0.95 --high 1.05
```

Copy the result to
`megatron/core/datasets/dcp_runtime_scheduler/models/gqa_p2p_flops_<hardware>.json`,
record its sha256 in `PACKAGED_SHA256` of `models/gqa_p2p_flops.py` and in
`ORIGIN.json`, and keep `source` as the script wrote it. The file carries the
`hardware` of the runtime file, which names the NVLink domain (`h100`: the 8
GPUs of a node; `gb200`). Rates of another GPU with one of these domains are
used through `calibration` under that name; packaging them under a name of
their own also needs that name in the NVLink topology of the package. The
packaged `h100` file is not such a result: it is fitted to another cost model
and marked `provisional`.

## Limits

- So far the layer timers and this skill were only tried on a small number of
  machines; fused attention, HybridEP and runs on more than one node were not
  tried.
- The rates describe the recipe they were fitted on and need not carry over to
  another model or another `max_seqlen_per_dp_cp_rank`: a part of the time of
  a layer does not follow its FLOPs. Expect a `gamma` away from 1 on a recipe
  other than the one of the rates, or fit level `rates` on the own recipe.
- `C_opt_ms` is one number per step, but what a step spends outside its layers
  grows with its microbatches. A fit puts the mean into `C_opt_ms`: steps with
  more microbatches than those of the fit are predicted too short, and a fit
  that includes a coverage run predicts the training steps too long. The
  output line on the timed layers gives the size.
- The garbage collection of the interpreter can stop a rank inside a layer,
  and the other ranks wait for it. The fit leaves those layer times out; the
  step times, and so `C_opt_ms` and the spread of the verdict, keep them. With
  `--manual-gc` the collections happen between two steps.
- One pipeline stage, as the cost model. The refinement of layouts by moves
  (`"stage2"` of the runtime file) changes which layouts a run executes, not
  how this skill reads its trace.
- The verdict is about step times. It does not show that the scheduler
  chooses better layouts with one file than with another.
- Fitted from step times, the single numbers are not reliable: two runs of one
  recipe can give very different rates that both pass. Judge a fit from step
  times by its verdict only, and do not plan with its rates when layer times
  can be recorded.
- The steps are those of the training run, with samples of varying length.
  Fitting on batches of one sample length instead gave steadier values that
  predicted training steps 3 to 5 % too high in simulations.
