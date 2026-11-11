# Troubleshooting

First stop, always: `gap check` — every NOT READY line carries a fix
hint. Second stop: the trace of the failing run.

## Install / environment

| Symptom | Cause → fix |
|---|---|
| `uv sync` → "does not appear to be a Python project" | Submodules not initialized → `git submodule update --init` (clone with `--recurse-submodules`) |
| `uv sync` can't resolve `open-robot-skills` | The sibling checkout is missing → clone open-robot-skills next to graph-as-policy (engine `uv sync` needs it for path-source metadata) |
| cuRobo build fails | `CUDA_HOME` unset → `uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo` (cuRobo JIT-compiles CUDA at install) |
| Sim window/render errors, EGL asserts | Headless GL not selected → `MUJOCO_GL=egl` (NVIDIA driver + EGL required) |
| sam3 install conflicts on numpy | Expected — the registry pyproject overrides sam3's strict pin (`numpy>=1.26`); use `uv sync`, don't hand-pip |
| Gated HF weights 401 | `export HF_TOKEN=...`, then `gap skills check --download` |
| Triton/JIT `FileNotFoundError` mentioning a compiler | Stale `$CC` pointing at a missing compiler → unset `CC` or point it at a real one (sam3 NMS JIT) |
| `gap generate` errors about credentials | No provider configured → `export OPENROUTER_API_KEY=...` (or `--provider vertex`; vertex uses gcloud ADC: `gcloud auth application-default login`) |
| Bundle NOT READY: missing deps | `uv sync --extra <bundle>` in the registry (or `pip install '<dist>[<bundle>]'`) — `gap check` prints the right one |
| Weights "unknown / not cached" | `gap skills check --download` prefetches; harmless otherwise (download on first use) |

## Validation failures (`gap run --validate-only`)

Errors cite rule codes — see the W1–W8 / S1–S11 digest in
`references/authoring-graphs.md` (full spec: `docs/runtime.md` §5).
Frequent offenders: a `$ref` to a node that isn't a producer (S5), an
`on_error` value that collides with a declared node (S9), conditional
edges missing `router_field` (S8), a subgraph input nothing upstream
binds (W8), tool names that don't exist (check `gap tools list`).

## Runtime failures

Every run writes a trace directory (default `outputs/run_<timestamp>`):

- `dag_trace.json` — per-node inputs, outputs, timing, error, and
  tool-level subcalls (`record_subcall` entries) — read this first;
- per-node assets (images, clouds, plans) referenced from the trace;
- checkpoint results — a failed postcondition with mode `warn` logs and
  continues; `raise` aborts the run.

Tools: `gap viz` (interactive browser over `outputs/`, localhost:9432);
`gap trace-diff <a> <b>` (align two runs, report where they diverge —
ideal for "it worked yesterday"). Guards: perception/planning/sim-step
call budgets come from `safety_limits` (task YAML or `GAP_MAX_*` env) —
hitting one is a graph-design smell, not a limit to raise blindly.

## Codegen failures (`gap generate`)

The pipeline retries script-fix loops on validation errors; if it still
fails, read the printed validation issues — usually a skill whose
required inputs aren't bound by the coordinator. Richer context helps:
make sure the relevant bundles are READY (`gap check`) so the planner
sees their tools, and keep instructions concrete ("pick up the alphabet
soup and place it in the basket", not "tidy up").

## Benchmarks

Claiming a success rate requires `gap benchmark <yaml> --gate` (the YAML
pins suite/tasks/trials/threshold). One green sim run is a smoke, not a
result. `--resume` continues an interrupted grid.
