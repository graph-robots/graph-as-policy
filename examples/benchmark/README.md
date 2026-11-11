# benchmark — configs for `gap benchmark`

> **What:** Grid harness configs: 1-cell smoke → posvar grid → the release gate · **Needs:** `grocery` (CUDA) + LLM key · **Time:** minutes → hours

Benchmark configs from a 1-cell smoke to the release acceptance gate. The
harness expands a config into a grid of cells (`modes × families ×
variations × task_ids × seeds`), runs each cell (generate and/or execute),
and writes `summary.json` / `summary.tsv` plus collated videos under the
config's `output_dir`.

## Run it

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo   # the benchmark families plan with CuRobo
export OPENROUTER_API_KEY=...                       # codegen + VLM (or vertex)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/smoke.yaml
```

| Config | What it is |
|---|---|
| [smoke.yaml](smoke.yaml) | 1 task × 1 seed sanity check (~minutes) |
| [posvar.yaml](posvar.yaml) | grid over the posvar variation families |
| [grocery_acceptance_smoke.yaml](grocery_acceptance_smoke.yaml) | the gate's 20-trial smoke (tasks 0–1) |
| [grocery_acceptance.yaml](grocery_acceptance.yaml) | **the release gate**: the full acceptance grid, gated by the config's `gate_threshold` |

## Gate semantics

```bash
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

- `--gate` exits non-zero when `success_rate` lands below the config's
  `gate_threshold` — wire it into CI or a release checklist as-is.
- `--resume` skips cells whose results already exist in `output_dir` and
  rebuilds the summary — a long acceptance run survives interruptions.
- Cells run in parallel worker processes; spread EGL rendering across GPUs
  with `GAP_MUJOCO_EGL_DEVICES=0,1,2` and cap workers in the config.

## Reading results

```bash
uv run gap viz            # browse every cell's trace + video
column -ts$'\t' outputs/<run>/summary.tsv
```

`summary.json` carries per-task pivots (success / completion rates per
task, per seed) for anything programmatic.

## Next steps

- [grocery_fulfillment](../grocery_fulfillment/) — what the acceptance
  family actually does, and a checked-in generated graph.
