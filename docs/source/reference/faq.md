# Troubleshooting and FAQ

Answers to the problems users actually hit, grouped by topic. Each entry
gives the short fix and links to the page that covers the subject in
depth. If your problem is not here, check
[Environment variables](environment-variables.md) and the
[CLI reference](cli.md) — many surprises turn out to be a default you
can override.

## Install and environment

**`uv sync` failed with "does not appear to be a Python project".**
You cloned the engine without its submodules, so the vendored packages
that `pyproject.toml` references are empty directories. Re-clone with
`git clone --recurse-submodules`, or fix an existing checkout with
`git submodule update --init --recursive`. See
[Installation](../getting-started/installation.md).

**I ran a bare `uv sync` and my simulator/perception install disappeared.**
`uv sync` is *exact*: it makes the environment match exactly what you ask
for, so a bare `uv sync` after `uv sync && uv run gap skills install sam3 grounding-dino geometry` removes the
quickstart extras again. Always repeat the `--extra` flags you want, or
use `uv sync --inexact …` for additive installs. See
[Installation](../getting-started/installation.md).

**pip downgraded numpy to 1.26 when I installed SAM3.**
SAM3's published metadata over-pins `numpy==1.26`. The engine's
`pyproject.toml` carries a `[tool.uv]` `override-dependencies` entry that
relaxes it to `numpy>=1.26`, but plain pip does not read uv overrides and
will happily downgrade your numpy. Use `uv sync` for any extras set that
includes SAM3, or pin numpy back by hand after a pip install.

**Rendering crashes on a headless machine.**
Set `MUJOCO_GL=egl` on every sim command — MuJoCo's default GL backend
needs a display. Spawned benchmark workers get this set for you, but
plain `gap run --sim …` does not. To spread workers across GPUs, set
`GAP_MUJOCO_EGL_DEVICES=0,1,2` as well. See
[Environments](../running/environments.md).

**When do model weights download? `gap skills check --download` said
"declares no weights (no prefetch())".**
`--download` runs each bundle's optional `prefetch()` function. The
perception bundles (`sam3`, `grounding-dino`) define one, so their
weights (~3.5 GB) fetch eagerly during the check. A bundle *without* a
`prefetch()` prints that message, and its weights download lazily — the
first call into the model runs HuggingFace `from_pretrained`, so the
first trial pays the fetch. Set `HF_TOKEN` before the first download:
the gated SAM3 weights sit in the default perception path. See
[Installation](../getting-started/installation.md#model-weights).

**CuRobo fails to build.**
CuRobo JIT-compiles CUDA code at install time: `CUDA_HOME` must point at
a toolkit matching your torch build's CUDA version, and pip installs need
`--no-build-isolation`. The supported recipe is
`uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo` (or `--extra curobo`
inside the open-robot-skills checkout). The first planner call per
process also pays a JIT warmup of several seconds — that is normal. See
[Skill catalog](../skills/skill-catalog.md).

**Can I bump the mujoco version?**
Not without re-validating. mujoco is pinned `==3.6.0` because the
acceptance numbers were measured on it; even minor bumps move contact
physics enough to shift grasp success rates. If you change it, re-run the
acceptance benchmark before trusting results. See
[Benchmarking](../benchmarks/benchmarking.md).

## Running graphs and sim

**What goes in `gap run --sim`?**
A `suite/task_index` pair such as `libero_object/0`. Only the `libero`
sim family is wired into the CLI; other environments are reached through
the Python API or custom connectors. See
[Execution](../running/execution.md) and
[Environments](../running/environments.md).

**`gap run` can't find a workflow in my `gap generate` output directory.**
Generation writes each task to a `task_NN` subdirectory —
`<out>/task_00/` for a single task. Point `gap run` at the task subdir,
not at the `--out` directory itself. See
[Generation](../authoring/generation.md).

**My script calls `time.sleep()` but the sim state never changes.**
Wall-clock time does not advance the simulator — only stepping does. Use
the `settle_steps` input on gripper and motion tools instead of sleeping
(e.g. `settle_steps=40` after opening the gripper).

**`except Exception` doesn't catch `GuardLimitExceeded`.**
That is intentional: `GuardLimitExceeded` inherits from `BaseException`
so skill code cannot accidentally swallow a safety guard violation — the
workflow must terminate. Guard counters are process-global but reset
automatically at the start of every `execute()`; call `reset_counters()`
yourself only if you drive tools outside the executor.
Limits are configurable via `GAP_MAX_PERCEPTION_CALLS`,
`GAP_MAX_PLANNING_CALLS`, and `GAP_MAX_SIM_STEPS`. See
[Executor](executor.md).

**My loop only runs once — the second visit to a node is skipped.**
Within a single subgraph activation, a re-entered node is silently
skipped (it is already in `completed_nodes`). Express loops as subgraph
revisits — route back to the subgraph so it re-activates — not as cycles
inside one subgraph scope. Runaway loops are stopped by
`GAP_ITERATION_CAP` (default 10000). See
[Patterns](../authoring/patterns.md).

**My run reached a failure end node and then raised `PipelineError`. Bug?**
No — `WorkflowExecutor.execute()` raises `PipelineError` whenever the
workflow terminates at a failure end node. Only the high-level
`gap.execute()` entry point converts that into a result object you can
inspect. Use `gap.execute()` unless you specifically want the raw
executor. See [API](api.md).

**Validation passed but my tool node fails at runtime with an unknown tool.**
Validation treats `tool` nodes that reference unregistered tools as
*warnings*, not errors, because connector tools are only known at run
time. A typo in a connector tool name therefore surfaces only when the
node executes. Read validation warnings, not just the error count. See
[Workflow schema](workflow-schema.md).

**What do LIBERO seeds actually control?**
Seeds index *baked* initial-state variations, not an RNG: the mapping is
`(seed - 1) % n_inits`, and `seed=0` means unseeded. Seeds 1..50 are
deterministic scene variations you can replay exactly. See
[Environments](../running/environments.md).

## Generation and LLM

**`gap generate` returns a 404 / unknown-model error on OpenRouter.**
OpenRouter model slugs are usually namespaced (e.g.
`google/gemini-3.1-flash-lite-preview`, `anthropic/claude-sonnet-4.5`). If
the bare default 404s for your account, pass `--model` or set `llm.model`
in your config YAML with the namespaced slug. Vertex serves Gemini models
only and additionally needs the `[vertex]` extra and ADC credentials. The
`GAP_VLM_*` environment variables only configure the runtime VLM
perception path, not codegen. See
[LLM providers](../authoring/llm-providers.md).

**I regenerate with temperature > 0 but get the identical graph every time.**
The codegen disk cache (active when `GAP_LLM_CACHE_DIR` or `cache_dir:`
is set) memoizes responses by prompt, so at nonzero temperature it
replays the single cached sample instead of drawing a new one. Set `GAP_LLM_NO_CACHE=1` when you want diverse regenerations. See
[Generation](../authoring/generation.md).

**`gap generate` "succeeded" but the graph fails validation.**
Generation runs an LLM fix loop over validation errors; when the loop
exhausts its retries, the residual errors are logged as a *warning* and
the pipeline still returns success. Always gate generated output with
`gap run <task_dir> --validate-only` before executing it. See
[Generation](../authoring/generation.md).

**Perception returns stale results when I re-run the same scene.**
The `perceiving-objects` skill keeps a result cache at
`<open-robot-skills>/.llm_cache/perceiving-objects`, keyed on the image,
`GAP_VLM_PROVIDER`/`GAP_VLM_MODEL`, and an internal cache version. If a
repeated scene serves outdated detections, set `GAP_PERCEPTION_CACHE=0`
or wipe that directory. Note this is a different cache from the codegen
LLM cache (`GAP_LLM_CACHE_DIR`). See
[Environment variables](environment-variables.md).

**Molmo tools fail with a `ToolError` about a base URL.**
Molmo has no hosted API — you must self-host it (e.g. vLLM serving
`allenai/Molmo2-8B`) and set `GAP_MOLMO_BASE_URL` (optionally
`GAP_MOLMO_MODEL`). The zero-GPU alternative is the `gemini-er` bundle.
See [Tool catalog](../skills/tool-catalog.md).

**I set `model:` in an agent's frontmatter but it had no effect.**
`AgentSpec` frontmatter `model:` is parsed but never consumed. Per-role
models are configured only via `composition.coordinator_model` and
`composition.subgraph_model` in the generation config. See
[Generation](../authoring/generation.md).

## Checkpoints and traces

**A checkpoint reports `passed=False` but the robot clearly did the task.**
LLM-generated perception-audit checkpoints (e.g. comparing a perceived
OBB to ground truth) can false-fail on a robot-frame vs world-frame
mismatch — the comparison is in the wrong frame, not the robot. Trust
physical checkpoints (`target_held`, `target_in_basket`-style checks on
world state) for the task verdict, and treat perception-audit failures as
triage signal. See [Checkpoints](../running/checkpoints.md).

**Checkpoints silently did nothing on my run.**
Checkpoint enforcement needs ground truth: when the connector exposes no
`world_snapshot` (all real robots, some custom connectors), enforcement
degrades to a single logged warning and every checkpoint is skipped.
Also mind the defaults: `gap.execute()` defaults to `"warn"` (failures
log but the run continues) while a hand-constructed `WorkflowExecutor`
defaults to `"off"`. Pass `checkpoints="raise"` to make failures fatal.
See [Checkpoints](../running/checkpoints.md).

**My run crashed and there is no `dag_trace.json` in the trace directory.**
The trace file is flushed only when the workflow reaches an end node — a
hard crash (or kill) before that leaves per-node `node_data` and assets
on disk but no flushed trace. The partial artifacts are still useful for
forensics; the missing file just means the run never terminated cleanly.
See [Traces](../running/traces.md).

**What do `gap trace-diff` exit codes mean?**
`0`: traces aligned and every matched node agrees. `1`: traces aligned
but verdicts disagree somewhere. `2`: zero nodes matched — the traces did
not align at all, usually because the two runs used different
`workflow.json` node names. trace-diff matches nodes by name, so it only
compares runs of the same workflow. See [Traces](../running/traces.md).

**Trace JSON shows placeholder summaries instead of my image arrays.**
Arrays with ≥ 100 elements are summarized in `dag_trace.json` to keep it
readable; real pixel data lives in the trace's `assets/` directory. This
also means viz *replay* is best-effort — replayed tools may receive
placeholder strings where large arrays were summarized. See
[Traces](../running/traces.md).

**Where is my run video? `gap viz` doesn't show it.**
`gap run --record-video` (which requires both `--sim` and tracing)
writes `run_video.mp4` into the trace directory, but the viz UI renders
only PNG/JPEG assets — open the mp4 from disk with your video player. See
[Traces](../running/traces.md).

## Skills and registries

**I passed `--skills` and my other registries vanished.**
`--skills` and `$GAP_SKILLS_PATH` are *full overrides*, not additive:
configured, project, and auto-discovered registries are all suppressed
(`gap registry list` prints a note when this happens). An invalid entry
in `$GAP_SKILLS_PATH` is a hard error, never a silent fallthrough. To
*add* a registry instead, use `gap registry add`. See
[Registries](../skills/registries.md).

**My registry isn't auto-discovered.**
Auto-discovery only picks up a sibling directory named exactly
`open-robot-skills` with populated bundle roots (or running inside such a
checkout). A freshly `gap registry init`-ed registry is empty, fails that
test, and must be added explicitly with `gap registry add` or pointed at
via `--skills`. Note that `gap registry add` *prepends* — the new
registry gets highest precedence and shadows same-named bundles
everywhere else. See [Registries](../skills/registries.md).

**I changed registry precedence but the old bundle code still runs.**
Bundle modules are imported once per process into a process-global
namespace, so precedence changes require a fresh process. Cross-registry
name shadowing is first-wins with a loud warning; a duplicate bundle name
*within* one registry is a hard error. See
[Registries](../skills/registries.md).

**`gap check` exits 0 even though half my bundles are not ready.**
By design: missing a GPU on a laptop is not an error. Pass `--strict` to
make any not-ready bundle (or zero resolved registries) fail the command
— use that in CI. See [Registries](../skills/registries.md).

**`gap skills test` skips my GPU tests / my pytest flags are ignored.**
The default registry pytest config deselects `gpu` and `llm` markers, so
GPU smokes need an explicit `gap skills test <bundle> -- -m gpu`. Flag
ordering matters: GaP's own flags go *before* the `--`, pytest arguments
after it. The command also runs the *current* interpreter's pytest with
cwd at the registry — registries with their own venv should run
`uv run pytest` there instead. See
[Testing bundles](../skills/testing-bundles.md).

**`pip install open-robot-skills` gave me no skills.**
The wheel intentionally contains no code — only dependency metadata.
Bundles are discovered from a checkout on disk, so you need the git
clone next to the engine (or registered via `gap registry add`). See
[Registries](../skills/registries.md).

**CuRobo's `solve_ik` raises `PlanningFailed` saying it's not supported
on v0.8.**
The curobo bundle straddles an API split: `solve_ik` and
`batch_grasp_feasibility` are built on the v0.7 `IKSolver` API that
curobo v0.8 removed, while `plan_grasp_motion` requires v0.8. On a
v0.8-only install, plan via `plan_to_grasp_poses` or `plan_grasp_motion`
instead. The SHA pinned in the bundle's extra is the supported install.
See [Tool catalog](../skills/tool-catalog.md).

**Does `gap.testing` have a fake connector?**
No. `gap.testing` exports exactly four names: `FakeContext`,
`ToolCallRecord`, `make_test_observation`, and `assert_graph_valid`.
`FakeContext` is fail-loud: unscripted tools and exhausted response
sequences both raise `ToolError`, so script every tool your test touches.
See [Testing bundles](../skills/testing-bundles.md).

## Benchmarks and policies

**Trials that were visibly succeeding got marked as failures.**
Check `trials.task_timeout_secs` — when it is too low, the watchdog
hard-kills trials that were still completing. The grocery acceptance
configs use 900 s because *passing* flat-box trials measured 391–537 s;
an earlier 600 s setting caused false failures. Budget for your slowest
passing trial, not the average. See
[Benchmark config](benchmark-config.md).

**The gate failed even though my success rate is above threshold.**
`--gate` fails on *any* errored cell and on a run where zero trials ran,
not just on low success rate — an infrastructure error is treated as a
red result. Fix or exclude the erroring cell; on resume, error cells always
re-run while completed cells are reused from the latest stamped
directory. See [Benchmarking](../benchmarks/benchmarking.md).

**The benchmark deleted files from my output directory.**
`launch()` cleans the artifact directory at start, so every cell must
write to a unique directory — never point two cells (or a cell and your
own files) at the same path. Generated workflows are written to a `__wf`
sibling for exactly this reason. See
[Benchmarking](../benchmarks/benchmarking.md).

**My policy benchmark can't reach the policy server.**
The harness never owns the server lifecycle for `url:` policies — start
it yourself first, e.g. `gap policy serve pi05-libero --port 9100`. Note
`gap policy serve` picks a *random* port unless you pass `--port`, and
its 900 s startup timeout exists because the first run downloads
checkpoints (`GAP_OPENPI_DIR` must point at your openpi/MolmoAct
checkout). Throttle `num_workers` when sharing one inference endpoint —
a single websocket serializes the workers anyway. See
[Policies](../benchmarks/policies.md).

**My benchmark YAML uses `base:` inheritance and is rejected.**
`base:` inheritance has been removed everywhere (benchmark YAMLs and
generation configs alike) — it is a hard error, not a deprecation.
Benchmark configs must be standalone files. See
[Benchmark config](benchmark-config.md).

## Real robots

**`robot.go_home` does nothing on my real robot.**
That is the safety gate, not a bug: `go_home` is a no-op (with a warning
log) whenever the connector config says the robot is real, because a
blind joint-space home through an unknown scene is dangerous. Real
connectors also never receive `sim.*` tools. Plan explicit, validated
motions instead. Read [Safety](../real-robots/safety.md) before running
anything.

**Why don't checkpoints run on my real robot?**
Checkpoints need privileged ground truth (`world_snapshot`), which only
simulators provide. On real hardware, enforcement silently degrades to a
single warning. Use perception-based verification in the graph itself
(e.g. a VLM yes/no check) instead. See
[Checkpoints](../running/checkpoints.md).

**My UR camera poses look identical to the wrist pose.**
The hand-eye calibration file is missing or not found: without
`GAP_UR_ZED_CALIB`, the connector degrades *silently* to an identity
extrinsic (warning log only), so every camera pose equals the wrist pose
and your perception is offset with no error raised. Set the env var to a
valid calibration before trusting any 3D output. See
[Connectors](../real-robots/connectors.md).

**`import pyzed` fails after installing the `[real]` extra.**
pyzed is not on PyPI — it ships with the ZED SDK and must be installed
manually from Stereolabs. The `[real]` extra covers the rest (e.g.
`rtde_receive` for UR). See [Connectors](../real-robots/connectors.md).

**How do I tell whether the Franka stack is hung, stuck, or dead?**
Read the heartbeat fields: `obs_age` growing means the realtime side is
hung; `joint_max_diff ≈ 0` with fresh observations means the robot is
physically stuck; `republish_hz → 0` means the GaP-side republisher died;
`obs_stale` flips after 2 s of silence. See
[Connectors](../real-robots/connectors.md).
