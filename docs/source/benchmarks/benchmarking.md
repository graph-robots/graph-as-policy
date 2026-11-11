# Benchmarking

`gap benchmark` sweeps a grid of evaluation cells — pipeline ablation *modes* ×
task suite *variations* × learned *policies* — and folds every cell into one
`summary.tsv` you can eyeball, plus a pass/fail gate you can wire into CI. GaP
itself is released against this harness: the grocery acceptance config must
clear its `gate_threshold` across the full task × trial grid before any
release.

The harness adds no execution path of its own. Every cell is one native
[`launch()`](gh-engine:gap/agent/launcher.py) over `tasks × seeds`, the same engine that
runs `gap generate` + trial execution — so the *only* difference between modes
is how the workflow is produced. Seeding is deterministic and identical across
modes: seed `i` maps to the same LIBERO initial state in every cell.

:::{note} Requirements
Linux + NVIDIA GPU with EGL headless rendering (`MUJOCO_GL=egl`), an LLM API
key (OpenRouter — the default OpenAI-compatible provider — or Vertex), and the `grocery` extra for the
benchmark families (`uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo` — they
plan with CuRobo). The `llm_plus_policy` / `policy_only` modes additionally
need a running VLA policy server (see [Learned Policies](policies.md)).
:::

## Two config shapes

A benchmark YAML is a standalone pipeline config (`llm:` / `skills:` /
`policies:` / `trials:` …) plus, optionally, a `benchmark:` block. The block's
presence selects the shape:

::::{grid} 1 1 2 2

:::{grid-item-card} Grid mode — `benchmark:` block present
Sweep `modes × (family, variation) × policies`. Each cell scores one suite
over `task_ids × n_seeds`. This is the ablation shape
([posvar.yaml](gh-engine:examples/benchmark/posvar.yaml)).
:::

:::{grid-item-card} Suites mode — no `benchmark:` block
The YAML's own `suites:` list is the cell axis: one cell per hand-curated
suite entry (per-task prompts + `objects:` hints). This is the acceptance-gate
shape ([grocery_acceptance.yaml](gh-engine:examples/benchmark/grocery_acceptance.yaml)).
:::

::::

A YAML with neither a `benchmark:` block nor a `suites:` list fails to load,
and `base:` inheritance is rejected — benchmark YAMLs are standalone. Every
key for both shapes is documented in the
[benchmark config reference](../reference/benchmark-config.md).

A minimal grid config:

```yaml
task: "auto"            # prompts resolved from LIBERO metadata

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
  output_dir: ../../benchmark_runs/smoke
```

This is [smoke.yaml](gh-engine:examples/benchmark/smoke.yaml) minus two
optional keys (`skills:`, which auto-discovers your open-robot-skills checkout
when omitted, and the inert `smoke: true` marker): one cell, one task, one
seed — a minutes-long sanity check that codegen plus the in-process sim eval
work end to end.

## The three modes

Grid mode ablates *how the workflow is produced*; scoring is shared:

| Mode | Workflow source | Needs a policy server |
|---|---|---|
| `llm_generation` | Zero-shot multi-agent codegen per task (one shot — no rehearsal, no refine loop) | No |
| `llm_plus_policy` | A fixed workflow template materialized per task: LLM-authored OBB approach + a `run_policy` VLA node | Yes |
| `policy_only` | Bare-VLA baseline: home/open-gripper bring-up + `run_policy` — no perception, no OBB, no approach | Yes |

`policy_only` isolates exactly what the graph scaffolding in `llm_plus_policy`
buys over the raw VLA. Both template modes **require**
`mode_overrides.<mode>.workflow_dir` in the YAML (the template directory); a
missing template errors the whole cell. Templates carry `{{target}}` /
`{{container}}` / `{{policy_id}}` placeholders filled per task from the LIBERO
prompt — see [template placeholders](../reference/benchmark-config.md#workflow-template-placeholders).

### The policy A/B axis

Set `benchmark.policies` to a list of policy ids (entries of the top-level
`policies:` registry) and every policy-dependent mode runs once per id — a
single benchmark A/Bs registered VLAs end to end. Cell directories gain a
`/<policy_id>/` segment so two policies never overwrite each other, and the
summary pivots key rows as `mode@policy_id`. `llm_generation` ignores the axis
and runs once.

## Running a benchmark

```bash
MUJOCO_GL=egl gap benchmark examples/benchmark/smoke.yaml
MUJOCO_GL=egl gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

| Flag | Effect |
|---|---|
| `--gate` | Apply the acceptance gate; exit nonzero on FAIL |
| `--resume` | Reuse the latest run dir; skip already-scored cells |
| `--families FAMILY ...` | Restrict the grid to these families (**grid mode only**) |
| `--modes MODE ...` | Restrict the grid to these modes (**grid mode only**) |
| `--output-dir DIR` | Override the config's output directory |
| `-v` / `--verbose` | Debug logging |

Exit codes: `0` success (gate PASS), `1` run exception or gate FAIL, `2`
config load failure — or `--families`/`--modes` used on a suites-mode config,
where they have no effect.

Modes run **sequentially** (they contend on sim GPUs and the shared policy
server); tasks × seeds run in parallel *inside* each cell, bounded by
`num_workers`. In suites mode, all remaining suite cells launch
**concurrently**, each bounded by its own per-suite `num_workers`. A crashed
cell is recorded as an `error` cell and the run keeps going — the grid never
aborts.

### Gating (`--gate`)

The gate passes only when **all three** hold:

- at least one trial ran,
- **no cell errored**, and
- the trial-pooled success rate ≥ `gate_threshold` (default **0.90**).

The CLI prints `gate PASS|FAIL: success_rate=... threshold=...` and exits `1`
on FAIL — wire it into a CI job or release checklist as-is.

:::{warning}
A single errored cell fails the gate even when the pooled success rate clears
the threshold. Typical error-cell causes: a template mode missing its
`workflow_dir` override, an unreachable policy server, or a crashed
`launch()`. Check the `error` column of `summary.tsv`.
:::

### Resuming (`--resume`)

Resume is **cell-grained**. With `--resume`, the harness reuses the **latest**
timestamped run dir (`YYYYMMDD_HHMMSS`) under `output_dir` and skips every
cell whose `cell_result.json` exists; the merged summary is rebuilt over old +
new cells. Error cells are deliberately never persisted, so a resumed run
re-runs them; a corrupt `cell_result.json` logs a warning and re-runs the
cell. A 500-trial acceptance run survives interruption.

:::{note}
Resume only matches the latest stamped directory and does not re-check
persisted cells against a changed grid or threshold — keep the config stable
across resumes, or start a fresh run dir.
:::

## Outputs

Each run writes a timestamped directory under `output_dir` (grid mode) or
under the pipeline's `trials.output_dir` (suites mode):

```text
benchmark_runs/posvar/20260611_142233/
├── summary.json                 # {grid, cells, matrix}
├── summary.tsv                  # per-cell rows + pivot blocks
├── aaa_done_flag.txt            # run-complete sentinel
├── videos/                      # flat collation (record_video: true)
├── llm_generation/
│   ├── aaa_done_flag.txt        # per-mode sentinel
│   └── posvar/
│       ├── pos_var/             # one cell: launch() artifacts
│       │   └── cell_result.json # resume marker + cell metrics
│       └── ...
└── llm_plus_policy/
    └── pi05-libero/             # policy axis segment (the policy-skill name)
        └── posvar/
            ├── pos_var/
            ├── pos_var__wf/     # materialized workflow templates
            └── ...
```

Suites-mode cells live under `<run>/suites/<suite_name>/` instead (duplicate
suite names get `_02`, `_03`, … suffixes). Template modes write produced
workflows to the `__wf` *sibling* of the cell dir because `launch()` cleans
its artifact dir at start.

- **`summary.tsv`** — one row per cell (`mode`, `policy_id`, `family`,
  `variation`, `suite_name`, `success_rate`, `completion_rate`, `avg_reward`,
  `n_success`, `n_trials`, `wall_clock_s`, `error`) followed by two pivot
  blocks, `success_rate_pivot` and `completion_rate_pivot`, with one row per
  `mode@policy_id` and one column per `family/variation` (`ERR` for errored
  cells, `-` for cells not in the grid).
- **`summary.json`** — the same data programmatically: the `grid` echo, full
  per-cell records including per-task rows (`task_id`, `success_rate`,
  `completion_rate`, `avg_reward`, `n_trials`, `n_success`) and the
  produce/eval wall-clock split, plus the pivot `matrix`.
- **`completion_rate`** is partial credit — the mean fraction of sub-goals
  completed per trial. For multi-item suites (`grocery_packing`) it is
  non-zero even when no trial fully succeeds; for single-goal suites it equals
  `success_rate`.
- **`videos/`** — with `record_video: true`, every per-trial
  `.../task_NN/trial_*_{pass,fail}/video.mp4` is hard-linked (copy fallback
  across filesystems, so no extra disk in the common case) into a flat
  directory with `__`-joined descriptive names next to `summary.tsv`.

Quick looks:

```bash
column -ts$'\t' benchmark_runs/posvar/20260611_142233/summary.tsv
gap viz          # browse per-trial traces and assets
```

See [Traces](../running/traces.md) for what each trial records.

## Operational guidance

### Spread workers across GPUs

```bash
export GAP_MUJOCO_EGL_DEVICES=0,1,2
MUJOCO_GL=egl gap benchmark examples/benchmark/posvar.yaml
```

`GAP_MUJOCO_EGL_DEVICES=<csv>` round-robins worker processes across the
listed GPUs: worker *i* gets device `devices[(offset + i) % len]`. In each
spawned worker, both `CUDA_VISIBLE_DEVICES` and `MUJOCO_EGL_DEVICE_ID` are
exported to that physical id *before* any CUDA import, pinning the worker's
whole stack — perception models, planning, EGL rendering — to one GPU.
Workers share nothing: each loads its own perception models (a few GB), so
budget VRAM per worker. Listing `1,2,3` keeps GPU 0 free for a co-tenant
policy server.

In suites mode, concurrent cells are staggered: each cell's worker slots start
at an offset equal to the cumulative `num_workers` of the suites before it, so
every cell's model stack does not land on GPU 0.

### Size the trial watchdog

`trials.task_timeout_secs` hard-kills a trial that exceeds the cap and records
it as failed with `exit_code=124`; `0` (the default) disables the watchdog.
Size it from your *slowest passing* trials, not the average: the acceptance
config uses **900 s** because passing flat-box grasp trials legitimately
measured 391–537 s when planning falls back to per-pose iteration — a 600 s
cap was killing trials that were on track to succeed.

### Throttle workers on shared policy endpoints

A single websocket VLA server serializes inference across every worker that
calls it. Cranking `num_workers` past the server's throughput just queues
requests. [posvar.yaml](gh-engine:examples/benchmark/posvar.yaml) runs
`llm_generation` at 8 workers but overrides the two policy modes to
`num_workers: 4` via `mode_overrides`.

### Policy-server preflight

External policies (entries with `url:`) are never started by the harness.
Before any cell runs, grid mode probes each `url:` policy named in
`benchmark.policies` or a `mode_overrides.<mode>.policy_id` with a 5-second
websocket handshake and raises immediately if one is unreachable — failing
fast instead of erroring cells mid-sweep. A policy reached only through the
implicit `pi05-libero` template fallback (whose preset the workers auto-boot)
is not probed. Start the server first:

```bash
gap policy serve pi05-libero --port 9100    # -> ws://127.0.0.1:9100
```

Managed entries (`start_cmd:` / `preset:`) skip the probe — the
[PolicyManager](policies.md) boots them itself, waits for the port (default
`policy_manager.startup_timeout_s: 120`), and tears them down after the run.
Suites mode skips preflight entirely: each worker boots exactly what its
workflow references.

### Debugging a cell in-process

```bash
GAP_PARALLEL_INPROC=1 gap benchmark my_config.yaml -v
```

When `num_workers` is 1, `GAP_PARALLEL_INPROC=1` runs trials sequentially in
the current process instead of spawning workers — breakpoints and
monkeypatches work. This is a test/debug escape hatch only: never score real
runs with it (concurrent cells in one process share one GPU, one GIL, and one
set of process-global guard counters, which corrupts results).

## Python API

```python
import gap.benchmark

summary = gap.benchmark.run("examples/benchmark/smoke.yaml")
summary = gap.benchmark.run(cfg, gate=True, resume=True)

assert summary.ok
print(summary.success_rate, summary.n_success, summary.n_trials)
print(summary.run_dir)            # the timestamped run directory
for cell in summary.cells:        # one ModeResult per cell
    print(cell.mode, cell.variation, cell.success_rate, cell.error)
```

`gap.benchmark.run(config, *, gate=False, resume=False)` accepts a
`BenchmarkConfig` or a YAML path and returns a `BenchmarkSummary` with `ok`,
`gated`, `gate_threshold`, `success_rate`, `completion_rate`, `n_trials`,
`n_success`, `run_dir`, `cells`, and `summary` (the merged dict also written
to `summary.json`). It is synchronous (wraps `asyncio.run`); use the async
`gap.benchmark.run_benchmark(cfg, *, gate=False, resume=False)` from inside an
existing event loop. `gap.benchmark` also exports `BenchmarkConfig`
(`from_yaml`), `BenchmarkModeOverride`, `DEFAULT_GATE_THRESHOLD`,
`FAMILY_SUITES`, `KNOWN_FAMILIES`, `KNOWN_MODES`, and `POSVAR_SUITES`.

## Next steps

- [Benchmark config reference](../reference/benchmark-config.md) — every YAML
  key for both shapes, with types and defaults.
- [Learned Policies](policies.md) — registering, serving, and managing VLA
  policy servers.
- [Benchmark example walkthrough](../examples/benchmark.md) — the checked-in
  configs, from the 1-cell smoke to the release gate.
- [Traces](../running/traces.md) — inspecting per-trial artifacts with
  `gap viz`.
