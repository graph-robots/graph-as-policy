# Architecture

How GaP is put together and why: the principles behind the engine, a map of
its modules, the design decisions contributors most often ask about, the
acceptance gates that defined the v1 release, and how the project is
packaged. This page adapts the release design doc
([docs/design.md](gh-engine:docs/design.md)) for a public audience; for the
user-facing vocabulary (graph, node, tool, skill, connector, checkpoint,
trace) start with [Concepts](../getting-started/concepts.md).

## Principles

GaP is a port of a working research codebase, restructured around five
explicit principles:

- **As easy to use as a Python library.** `pip install`, four lines of
  Python, and an LLM API key. The quickstart needs zero self-hosted servers
  and zero extra terminals.
- **In-process by default.** The simulator, the vision models, and IK all
  run in one Python process. Ray is an opt-in extra for scale-out
  benchmarking — never a prerequisite.
- **Plain Python, no protos.** gRPC and protobuf were deleted everywhere.
  Typed dicts + numpy arrays are the data contract. The only wire protocol
  left in the system is the msgpack bridge to real robots.
- **Two repos.** The engine ([graph-as-policy](gh-engine:.)) and the
  contributable skill library ([open-robot-skills](gh-skills:.), in the
  Anthropic Agent Skills format) are separate repos. Skills are discovered
  by filesystem path, not by pip entry points — see
  [Skill registries](../skills/registries.md).
- **Minimum viable abstraction.** Working code was ported with seam changes,
  not rewritten around speculative interfaces. There is no plugin system
  where a function call suffices, and no pluggable backend where one
  implementation is enough (the connector's IK is a deliberate example).

## The two repos

```text
┌────────────────────────── gap (engine) ──────────────────────────┐
│ agent/      instruction ─► coordinator ─► subgraph agents        │
│             ─► checkpoint agent ─► validate / fix ─► graph       │
│ runtime/    executor (super-steps, streaming, Send), validator,  │
│             tracing, policy loop, verify/ (World, checkpoints)   │
│ tools/      ToolRegistry: typed schemas, tags→guards, dispatch   │
│ connector/  sim()/real(), in-process pyroki IK, rr_launcher,     │
│             data collector                                       │
│ envs/       libero (+perturbed), franka_real, ur_zed, msgpack    │
│ benchmark/  grid harness (modes × families × seeds), --gate      │
│ viz/        FastAPI+React trial browser, replay3d, PDF render    │
└──────────────┬───────────────────────────────▲───────────────────┘
               │ discovers (by path)           │ registers tools
        ┌──────▼───────────────────────────────┴──────┐
        │ open-robot-skills (Agent Skills format)     │
        │ tools/   sam3, grounding-dino, gemini-er,   │
        │          molmo, vlm, curobo, geometry       │
        │ skills/  perceiving-objects(+variants),     │
        │          grasping-*, transporting-objects,  │
        │          tracking-objects, pi05-libero,     │
        │          molmoact-libero                    │
        └─────────────────────────────────────────────┘
```

Three flows cross this boundary:

- **execute** — graph + connector + skills → `ExecutionResult` + trace
  ([Executing graphs](../running/execution.md))
- **generate** — instruction + skill catalog → validated workflow directory
  ([Generating graphs](../authoring/generation.md))
- **benchmark** — config → grid of generate/execute cells → summary + videos
  ([Benchmarking](../benchmarks/benchmarking.md))

The dependency is one-way: open-robot-skills bundles import GaP's stable
authoring surface (`gap.NodeContext`, `gap.types`, `gap.errors`,
`gap.skills`, `gap.testing`); the engine never imports the skills repo — it
discovers bundle directories on disk and loads them.

## Module map

- **[gap/agent/](gh-engine:gap/agent)** — the LLM generation pipeline: a
  coordinator agent decides the subgraph topology, per-subgraph agents write
  the state machines and inline scripts, a checkpoint agent authors
  postcondition predicates, and a script-fix loop (≤2 attempts) repairs
  validation errors. `llm.py` is the provider layer (`openrouter` default,
  `vertex`) for generation — the runtime `vlm` tool bundle
  carries its own equivalent provider config (`GAP_VLM_*`); `launcher.py` and
  `parallel.py` run generate/execute trials (including the benchmark's
  multiprocess worker pool). See [Generating graphs](../authoring/generation.md)
  and [LLM providers](../authoring/llm-providers.md).
- **[gap/runtime/](gh-engine:gap/runtime)** — the execution engine:
  `workflow.py` (strict v3 schema parsing), `validate.py` (structural and
  type rules), `executor.py` (the super-step scheduler), `context.py`
  (`NodeContext`, the whole skill-facing surface), `tracing.py`,
  `observation_stream.py`, the policy loop (`policy.py`,
  `policy_manager.py`, `policy_presets.py`), and `verify/` (the
  `World`/`Body`/`Robot` predicate vocabulary behind checkpoints). See the
  [executor reference](../reference/executor.md).
- **[gap/tools/](gh-engine:gap/tools)** — the `ToolRegistry`: the `@tool`
  decorator, schema extraction from type hints, name-collision rejection,
  dispatch, and `guards.py` (tag-based call-count limits).
  `ray_executor.py` holds the opt-in `@ray.remote` wrappers.
- **[gap/connector/](gh-engine:gap/connector)** — the embodiment layer:
  `sim()` / `real()` factories, the `Connector` base class, in-process
  pyroki IK (`ik.py`), the robots_realtime session launcher
  (`rr_launcher.py`), the HDF5 data collector (`collector.py`), and the
  world-snapshot adapter checkpoints evaluate against.
- **[gap/envs/](gh-engine:gap/envs)** — concrete environments behind a lazy
  registry: LIBERO (plus the perturbed variant), the real Franka env, the
  UR+ZED env, and the msgpack bridge real robots speak. Factories are
  dotted paths, so importing the registry never pulls in mujoco or pyzed.
- **[gap/benchmark/](gh-engine:gap/benchmark)** — the grid harness:
  config/family expansion (`modes × families × variations × task_ids ×
  seeds`), per-cell launch, report math, and the `--gate` / `--resume`
  machinery. See [Benchmarking](../benchmarks/benchmarking.md).
- **[gap/viz/](gh-engine:gap/viz)** — the FastAPI + React trial browser
  (`gap viz`), the swimlane graph builder, the 3D replay (viser), and
  `render.py` (graph → PDF/PNG). The pre-built frontend ships in the wheel
  (see [Packaging](#packaging)).
- **[gap/cli/](gh-engine:gap/cli)** — one module per subcommand with
  lazy-import dispatch, so `gap --help` and `gap skills list` work on a
  bare install without importing heavy dependencies. See the
  [CLI reference](../reference/cli.md).
- **[gap/builder/](gh-engine:gap/builder)** — the Python authoring API
  (`Workflow`/`Subgraph`); it emits the identical on-disk artifact the LLM
  pipeline produces, through the same validation. See
  [The builder API](../authoring/builder.md).
- **[gap/skills/](gh-engine:gap/skills)** — registry resolution and
  precedence, SKILL.md frontmatter parsing, the `Skill` base class for
  stateful bundles, prompt loading, and the capability probes behind
  `gap check`.
- **[gap/testing/](gh-engine:gap/testing)** — the public test fixtures,
  exported for bundle authors and engine contributors alike: `FakeContext`,
  `ToolCallRecord`, `make_test_observation`, `assert_graph_valid`. See
  [Testing bundles](../skills/testing-bundles.md).

Three top-level modules round out the package: `types.py` (the data
vocabulary), `schema.py` (the type-name registry), and `errors.py` (the
`PipelineError` hierarchy).

## Key design decisions

### A data vocabulary instead of protos

[gap/types.py](gh-engine:gap/types.py) replaces the old protobuf message
definitions with numpy-first TypedDicts: `Se3Pose`, `OrientedBoundingBox`,
`CameraFrame` (rgb `u8[H,W,3]`, depth `f32[H,W]` in meters, intrinsics
`f64[3,3]`), `Mask`, `PointCloud`, `JointState`, `Trajectory`, `ArmState`,
`Observation`, `WorldConfig`, `GraspCandidates`. Nothing is byte-packed
in-process; conversions from simulator buffers happen once, inside the env
classes. Two conventions are load-bearing:

- **Quaternions are wxyz, scalar-first.** LIBERO's xyzw is converted at the
  env boundary, nowhere else.
- **`ArmState.proprio_state` is policy-training-exact** — its layout matches
  what learned policies were trained on and is never transformed.

[gap/schema.py](gh-engine:gap/schema.py) maps the type-name strings used by
subgraph `inputs:`/`outputs:` declarations to these definitions; the
validator and the viz frontend consult it instead of proto descriptors.

### One dispatch surface for tools

Every callable a graph can reach is registered in the `ToolRegistry` and
dispatched by name — `ctx.tool(name, **kwargs)` from scripts, or `type:
tool` nodes in the graph. The `@tool` decorator extracts typed schemas from
type hints; the same schemas drive validation and the LLM's tool catalogs,
so the model sees exactly what the validator enforces. Names are
namespaced: `robot.*`/`sim.*` are reserved for connectors, `<model>.<func>`
(e.g. `curobo.plan_to_pose`, `sam3.segment_text`) for tool bundles; the
loader rejects collisions.

Safety guards hook this single dispatch point: tools carry tags
(`perception`, `planning`, `sim_step`), and per-tag call-count limits —
from `safety_limits` in a task config or `GAP_MAX_*` environment variables
— decrement on every call. `GuardLimitExceeded` subclasses `BaseException`
on purpose, so a skill's `except Exception` cannot swallow it. Ordinary
failures are `PipelineError` subclasses (`PerceptionFailed`,
`PlanningFailed`, `GraspFailed`, …), which the executor wraps and routes
through subgraph `on_error` edges.

### The connector is the embodiment

A graph never talks to MuJoCo or a robot driver directly. The
[connector](../real-robots/connectors.md) owns the environment (sim) or the
robot link (real), and registers the `robot.*`/`sim.*` tools that form the
embodiment surface — the same tool names whether the body is LIBERO or a
physical Franka (see the
[connector tools reference](../reference/connector-tools.md)). Inverse
kinematics is built into the connector (pyroki, JAX on CPU, always
in-process) — deliberately not pluggable. Motion *planning* is a separate
concern: the `curobo` tool bundle produces collision-aware trajectories
that skills execute via `robot.execute_trajectory`. All motion tools take
`arm_id=0`, leaving the door open for multi-arm support post-v1.

### Super-step execution

The executor is a frontier scheduler: each super-step runs every ready node
concurrently, collects results, routes conditional edges on exit values,
and advances the frontier. Subgraphs execute recursively with their own
input bindings and `on_error` routing; streaming nodes publish snapshots
that downstream nodes consume via `$ref`; `Send` fan-out launches dynamic
parallel branches from router nodes; a node-visit cap guards against
runaway cycles; and `end` nodes can carry best-effort recovery tool calls
(open the gripper, go home). The semantics are specified in the
[executor reference](../reference/executor.md) and the
[workflow schema](../reference/workflow-schema.md).

### The trace is a stability guarantee

Tracing is on by default — the trace is the product, not a debug flag.
Every run writes `workflow.json`, `dag_trace.json`, and `node_data/<id>/`
with per-node inputs/outputs and extracted PNG/NPZ assets
([Traces](../running/traces.md)). This on-disk layout is one of the
project's two declared stability guarantees (the other is the
skill-authoring import surface): `gap viz`, `gap trace-diff`, and users'
own tooling depend on it, the deterministic tracing tests pin it, and
changes to it require a migration note — see
[Contributing](contributing.md).

## Release gates

v1 was defined by four acceptance gates, and the first remains the
regression bar for every release:

- **G1 — Grocery fulfillment gate.** `gap benchmark` in `llm_generation`
  mode on the grocery-fulfillment suites must clear the config's
  `gate_threshold`, with CuRobo motion planning and the
  grasping/transport skills in the prompt.
  The pinned config is
  [examples/benchmark/grocery_acceptance.yaml](gh-engine:examples/benchmark/grocery_acceptance.yaml);
  `gap benchmark --gate` exits non-zero below threshold.
- **G2 — Correct generation.** `gap.agent.generate` on grocery-fulfillment
  instructions produces graphs that pass the equivalence suite and execute
  to success — the *generated* code clears G1, not just hand-written
  graphs.
- **G3 — Steered policy works.** The hover-then-handover example (perceive
  → approach above the target → hand control to a learned policy) runs end
  to end — see [Steered policy](../examples/steered-policy.md).
- **G4 — Quickstart is one command** on the stated hardware floor: **1×
  NVIDIA RTX 4090 (≥24 GB VRAM) + Linux + EGL**, an LLM API key, the two
  repos cloned side by side, and a one-time multi-GB model-weight download
  on first use (`HF_TOKEN` for the gated SAM3 weights). The floor is stated
  up front — no pretending it runs on a laptop CPU.

## Packaging

- **Distribution `graph-as-policy`, import `gap`**, console script `gap`,
  Python ≥ 3.10, MIT license.
- **Core dependencies are lean** — numpy/scipy, FastAPI/uvicorn, httpx (the
  OpenRouter / OpenAI-compatible LLM client), pyroki + JAX (CPU) for
  in-process IK, msgpack, imaging libraries. No grpcio, no protobuf, no torch
  in core.
- **Extras** gate the heavy stacks: `[libero]` (MuJoCo/robosuite/the posvar
  LIBERO fork — `mujoco==3.6.0` is pinned exactly because the acceptance
  numbers were measured on it), `[ray]`, `[real]`, `[vertex]`, `[policy]`.
  open-robot-skills bundles install one at a time (`gap skills install`),
  never as an extra of this package, so the two repos do not depend on each
  other in a cycle. See
  [Installation](../getting-started/installation.md).
- **uv-first.** The committed `uv.lock` pins the exact environment the
  measured results were produced on; `[tool.uv.sources]` resolves the
  vendored submodules and the side-by-side open-robot-skills checkout. pip
  is supported with the submodules named explicitly.
- **The viz frontend ships pre-built.** `gap/viz/frontend/dist/` is checked
  into the repo and included in the wheel as package data, so pip users get
  `gap viz` without Node. CI asserts the wheel contains `dist/` and leaks
  no `frontend/src/` or `node_modules` — see
  [Contributing](contributing.md#what-ci-enforces).
- **Pinned submodules** under `third_party/`: robots_realtime, the
  Variational-Automation-Benchmark fork, robosuite, and LIBERO-PRO.

## Where next

- [Contributing](contributing.md) — dev setup, test markers, what CI
  enforces
- [Roadmap](roadmap.md) — the v1 line and the named post-v1 cuts
- [Workflow schema](../reference/workflow-schema.md) and
  [executor semantics](../reference/executor.md) — the normative specs
- [Authoring bundles](../skills/authoring-bundles.md) — extending the
  skills repo instead of the engine
