# Benchmark Grids

:::{note} Requirements
A CUDA GPU with `MUJOCO_GL=egl` for headless rendering, the `grocery` extra
(the benchmark families plan with CuRobo), and an LLM credential for code
generation (`OPENROUTER_API_KEY`, or Vertex via the config's `llm:`
block). The policy modes additionally need a running policy server.
:::

[examples/benchmark](gh-engine:examples/benchmark) collects ready-to-run
configs for `gap benchmark`, from a one-cell sanity check to the release
acceptance gate. The harness expands a config into a grid of cells
(`modes × families × variations`, times the policy axis for the policy
modes), runs each cell's `task_ids × seeds` trials (generate and/or
execute), and writes `summary.json` / `summary.tsv` plus collated videos
under the config's `output_dir`.

This page tours the example configs; [Benchmarking](../benchmarks/benchmarking.md)
is the full guide and [Benchmark Config](../reference/benchmark-config.md) the
key-by-key reference.

## Run it

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo
export OPENROUTER_API_KEY=...    # codegen + VLM (or vertex)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/smoke.yaml
```

| Config | What it is |
|---|---|
| [smoke.yaml](gh-engine:examples/benchmark/smoke.yaml) | 1 family × 1 variation × 1 mode × 1 task × 1 seed sanity check |
| [posvar.yaml](gh-engine:examples/benchmark/posvar.yaml) | the LIBERO-PosVar variation × mode ablation grid |
| [grocery_acceptance_smoke.yaml](gh-engine:examples/benchmark/grocery_acceptance_smoke.yaml) | the gate's 20-trial smoke (tasks 0–1 × 10 trials) |
| [grocery_acceptance.yaml](gh-engine:examples/benchmark/grocery_acceptance.yaml) | **the release gate**: the full acceptance grid, gated by the config's `gate_threshold` |

## Two config shapes

**Grid mode** — a `benchmark:` block declares the axes, and the harness takes
their cross product. `smoke.yaml` is the minimal case:

```yaml
task: "auto"           # per-task prompts resolved from LIBERO metadata
skills: ../../../open-robot-skills   # optional; omit to auto-discover

llm:
  provider: openrouter

benchmark:
  families: [grocery_packing]
  variations: [object]
  modes: [llm_generation]
  n_tasks: 1
  task_ids: [0]
  n_seeds: 1
  num_workers: 1
  record_video: true
  smoke: true
  output_dir: ../../benchmark_runs/smoke
```

**Suites mode** — no `benchmark:` block; each top-level `suites:` entry is one
cell, and all cells launch concurrently with a per-suite `num_workers` bound.
The acceptance gate uses this shape — see
[Grocery Fulfillment](grocery-fulfillment.md#config-anatomy) for its anatomy.

In both shapes, `skills:` may be omitted entirely: the harness auto-discovers
the open-robot-skills checkout via `$GAP_SKILLS_PATH` or a checkout next to
the GaP repo. Explicit relative paths resolve against the config file's
directory.

## Gate semantics

```bash
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

- `--gate` exits non-zero when the overall `success_rate` lands below the
  config's `gate_threshold` (default 0.90), when any cell errored, or when no
  trial ran — wire it into CI or a release checklist as-is.
- `--resume` reuses the latest run directory, skips cells whose results
  already exist, and rebuilds the merged summary — a 500-trial run survives
  interruptions.
- `--families` / `--modes` restrict a grid-mode config from the command line
  (passing them with a suites-mode config is an error); `--output-dir`
  overrides the config's output directory.

Each cell's trials run in parallel worker processes. Spread EGL rendering across GPUs with
`GAP_MUJOCO_EGL_DEVICES=0,1,2` (workers round-robin across the listed
devices) and cap concurrency with the config's `num_workers` keys.

## Policy modes

`posvar.yaml` sweeps three modes over the PosVar variations
(`pos_var`, `permutation`, `basket_swap`, `all`), 10 tasks × 50 seeds each:

- `llm_generation` — generated graphs only.
- `llm_plus_policy` — a steered hybrid graph that hands off to a VLA
  ([Steered Policy](steered-policy.md)).
- `policy_only` — the bare VLA baseline.

The policy modes run a policy **skill** (`pi05-libero` or `molmoact-libero`),
whose preset server the workers auto-boot. To instead share one **external**
server across all workers — the setup below — run it yourself and override the
skill's recipe with a `url:`; the harness then preflights that websocket and
never owns its lifecycle:

```bash
uv run gap policy serve pi05-libero --port 9100    # -> ws://127.0.0.1:9100
```

The config overrides the `pi05-libero` skill's serving recipe to point at that
external server (the override key must equal the skill name), with per-mode
worker overrides:

```yaml
policies:
  pi05-libero:                       # override: skill name == entry key
    url: ws://127.0.0.1:9100

policy_manager:
  startup_timeout_s: 900

benchmark:
  # ...
  num_workers: 8
  mode_overrides:
    llm_plus_policy:
      workflow_dir: ../steered_policy/graph   # your steered graph template
      num_workers: 4      # single shared websocket server — throttle
    policy_only:
      workflow_dir: ../steered_policy/graph
      num_workers: 4
```

Point each mode's `workflow_dir` at the graph template you want it to run —
for example [examples/steered_policy/graph_loop](gh-engine:examples/steered_policy/graph_loop),
whose `{{policy_id}}.run` tool placeholder the harness materializes per cell
(substituting the policy-skill name). The
policy modes are throttled to fewer workers than `llm_generation` because
every worker shares one inference endpoint. See
[Policies](../benchmarks/policies.md) for the full policy-serving guide.

## Reading results

```bash
uv run gap viz --root benchmark_runs                 # browse every cell's trace
column -ts$'\t' benchmark_runs/<name>/<run>/summary.tsv   # quick terminal pivot
```

- `summary.tsv` is the human-readable table; the path is printed at the end
  of every run.
- `summary.json` carries the same data programmatically: the grid echo, the
  full per-cell records (with per-task success and completion rates), and
  the mode × variation pivot matrix.
- With `record_video: true`, per-trial videos are collated into the run
  dir's `videos/` folder; open them from disk. `gap viz` browses traces and
  image assets — see [Traces](../running/traces.md).

## Next steps

- [Benchmarking](../benchmarks/benchmarking.md) — the full harness guide:
  modes, families, variations, and output layout.
- [Benchmark Config](../reference/benchmark-config.md) — every config key.
- [Grocery Fulfillment](grocery-fulfillment.md) — what the acceptance family
  actually does, with a checked-in generated graph.
- [Policies](../benchmarks/policies.md) — serving VLAs for the policy modes.
