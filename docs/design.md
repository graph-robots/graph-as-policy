# graph-as-policy + open-robot-skills: Design

The architecture of GaP and open-robot-skills: what the pieces are, how they fit, and
the contracts (graph schema, tool layer, connector, skill format, testing)
that the two repos hold stable.

## 1. Overview & principles

**GaP** ("graph-as-policy"): a natural-language task is compiled by an LLM agent pipeline into a typed, verified execution graph of robot skills; the graph — not a monolithic policy — is what runs, on simulators or real robots.

```python
import gap
conn = gap.connector.sim("libero", task="libero_object/0")   # one process, no terminals
result = gap.execute(graph, conn)    # graph = dir | dict | builder.Workflow; skills auto-discovered
g = gap.agent.generate("pick up the soup can and put it in the basket")
gap.benchmark.run("examples/benchmark/smoke.yaml")
gap.viz.serve("outputs/")
```

Principles (user-set):
- **As easy to use as `models`**: pip install + 4 lines + `OPENROUTER_API_KEY`. Quickstart needs zero self-hosted servers, zero extra terminals.
- **In-process by default**: env + models in one Python process. Ray is opt-in scale-out, never a prerequisite.
- **Plain Python, no protos**: gRPC/protobuf deleted everywhere. Typed dicts + numpy are the data contract. The only wire protocol left is the msgpack bridge to real robots.
- **Two repos**: `gap` (engine) and `open-robot-skills` (contributable skill library in Anthropic Agent Skills format, discovered by path).
- **No over-engineering**: port working code with seam changes; minimum viable abstraction.

Distribution name `graph-as-policy`, import `gap`, Python ≥3.10 (isaaclab pin gone). Repos live side by side under `/home/kych/graph-as-policy/{gap,open-robot-skills}`.

### 1.1 Acceptance criteria (release gates)

- **G1 — Grocery fulfillment ≥90%.** `gap benchmark` in `llm_generation` mode on the variational-automation (posvar) grocery-fulfillment suites (`libero_object_all_variance`, `grocery_packing` families) achieves **>90% success** with CuRobo-only motion planning + grasping/transport skills in the prompt (the **curobo tool bundle is required for the gate**, even though the quickstart stays curobo-free).
- **G2 — Correct generation.** `gap.agent.generate` on grocery-fulfillment instructions produces graphs that pass the equivalence suite and execute to success (the generated code, not just hand-ported graphs, clears G1).
- **G3 — Steered policy works.** The hover-then-handover example (perceive → approach above target → hand control to the learned policy) runs end to end.
- **G4 — Quickstart is one command** on the stated hardware floor: **1× NVIDIA RTX 4090 (≥24 GB VRAM) + Linux + EGL**, `OPENROUTER_API_KEY` (+ `HF_TOKEN` for gated weights), the two repos cloned side by side, and a one-time several-GB weight download (`gap skills check --download`). The README states this floor up front — no pretending it runs on a laptop CPU.

## 2. Concepts

| Concept | Definition |
|---|---|
| **Graph / workflow** | v3 JSON: top-level DAG of nodes + named subgraphs (each its own DAG). The policy artifact. |
| **Node** | `tool` \| `script` \| `router` \| `subgraph` \| `noop` \| `end`. Edges + conditional edges route on exit values. |
| **Tool** | Unit of *execution*: one typed callable, dispatched by name via the ToolRegistry (`ctx.tool(name, **kw)` or `type: tool` nodes). |
| **Skill** | Unit of *packaging/discovery/LLM context*: an Agent Skills bundle (SKILL.md + scripts/prompts/references). |
| **Connector** | The embodiment: owns an env (sim) or robot link (real); registers `robot.*`/`sim.*` tools; exposes benchmark hooks and world snapshots. |
| **Checkpoint** | LLM-authored postcondition predicate attached to a subgraph (`validate=True`), evaluated against sim ground truth at subgraph exit. |
| **Trial / trace** | One execution with tracing: `workflow.json`, `dag_trace.json`, `node_data/<id>/` (inputs/outputs + PNG/NPZ assets). |

**Two distributions, one workspace.** graph-as-policy is a uv workspace with two pip distributions: **`gap-core`** is the bundle-author surface (`gap_core.tools`/`types`/`errors`/`schema`/`skills`/`rpc`, ~150 MB installed) — the stable, narrow API that open-robot-skills bundles depend on — and **`graph-as-policy`** is the runtime (`gap.runtime`/`agent`/`connector`/`cli`/`builder`/`viz`/`benchmark`) that consumes gap-core and ships the agent/executor/viz stack. Bundles depend only on `gap-core`, so the heavy runtime stack (fastapi, JAX, MuJoCo, pyroki, opencv) stays out of bundle venvs; bundle authors and CI install just `gap-core`, while end users running graphs install `graph-as-policy` (which pulls gap-core transitively).

**Tools vs skills — a first-class split, mirrored in the repo layout.** open-robot-skills has two top-level categories:
- **Tools** (`open-robot-skills/tools/<bundle>/`) = *what the robot can compute*: model-backed callables with no task strategy. **Tool bundles are named after the model**: `sam3`, `grounding-dino`, `gemini-er`, `molmo`, `vlm` (generic API VLM), `curobo` (motion planning), `geometry` (pure math). A tool bundle exposes typed functions via `@tool` in `tools.py`; its SKILL.md documents when to call them. (IK is NOT a tool bundle — it's built into the connector, see §7.)
- **Skills** (`open-robot-skills/skills/<bundle>/`) = *what the robot can do*: manipulation strategies that own subgraphs in generated graphs — perceive an object, grasp, transport, track, run a learned policy. A learned policy is itself a skill, one bundle per model checkpoint (`pi05-libero`, `molmoact-libero`), so the coordinator picks the model by its capability description rather than via a generic runner + `policy_id`. Skills keep capability names. **All skills are flat** — no atomic/composite distinction; a skill bundles LLM guidance (SKILL.md) + canonical scripts, and *may* also expose a callable via `tools.py` when it is invocable as a single unit (pi05-libero, molmoact-libero, tracking-objects).

So tools come from exactly two registries: **connector tools** (`robot.*`, `sim.*` — embodiment surface shipped by GaP core, not in open-robot-skills) and **tool bundles** (`<model>.<func>`, e.g. `curobo.plan_to_pose`, `sam3.segment_text`). Model-named prefixes deliberately match the old gRPC service short names, so the tool-name migration is mostly `Method` → `snake_case` with the prefix unchanged. Graph `script` nodes are per-graph generated code — neither tool nor skill; they call tools.

## 3. Architecture

```
┌────────────────────────── gap (engine) ──────────────────────────┐
│ agent/      instruction ──► coordinator ──► subgraph agents      │
│             ──► checkpoint agent ──► validate/fix ──► graph      │
│ runtime/    executor (super-steps, streaming, Send), validate,   │
│             tracing, policy loop, verify/ (World, checkpoints)   │
│ tools/      ToolRegistry: schemas, tags→guards, dispatch         │
│ connector/  sim()/real(), env registry, rr_launcher, collector   │
│ envs/       libero(+perturbed), franka_real, ur_zed, msgpack     │
│ benchmark/  grid harness (modes × families × seeds)              │
│ viz/        FastAPI+React trial browser, replay3d, PDF render    │
└──────────────┬───────────────────────────────▲───────────────────┘
               │ discovers (path)              │ registers tools
        ┌──────▼───────────────────────────────┴──────┐
        │ open-robot-skills (Agent Skills-format, contributable)│
        │ tools/   sam3, grounding-dino, gemini-er,      │
        │          molmo*, vlm, curobo, geometry         │
        │ skills/  perceiving-objects(-oneshot/-multiview│
        │          /-parts), grasping-with-planner,      │
        │          grasping-direct-ik, grasping-short-axis,│
        │          transporting-objects, tracking-objects,│
        │          pi05-libero, molmoact-libero          │
        └──────────────────────────────────────────────────┘
```

Three flows: **execute** (graph + connector + skills → ExecutionResult + trace), **generate** (instruction + skill catalog → graph dir), **benchmark** (config → grid of generate/execute cells → summary + videos).

## 4. Graph model (v3 — unchanged)

The v3 schema ports as-is from the dev tree's `runtime/workflow.py`: `version: 3`, `nodes`, `edges`, `conditional_edges` (router_field → mapping), `subgraphs` (with `inputs`/`outputs`, `exit.success_values`, `on_error`), streaming nodes (`streaming: true`, consumed via `{"$ref": "<node>"}` snapshots), Send dynamic fan-out from router nodes, `noop` exit markers, virtual START/END. Dataflow via `{"$ref": "node.field.subfield"}`; resolution already walks dicts/lists/attributes — one-line reorder (dict-key before attr). Recovery actions on `end` nodes become `ToolCall{tool, inputs}` (was proto `ServiceCall`) — used for gripper-open / go-home cleanup. Tool names are opaque strings to the schema, so renaming services→tools does **not** bump the version. Legacy v2 node types stay rejected with the existing migration error.

## 5. Data vocabulary (`gap/types.py`, ~150 lines — replaces protos)

Numpy-first (no byte packing in-process; conversions from sim buffers happen once, inside env classes):

- Geometry: `Vec3`, `Quaternion` (**wxyz scalar-first; LIBERO's xyzw converted at the env boundary — documented**), `Se3Pose{position, rotation}`, `OrientedBoundingBox{center, extent(half), orientation}`, `BoundingBox2D`.
- Sensor: `CameraFrame{name, rgb: u8[H,W,3], depth: f32[H,W] (meters), intrinsics: f64[3,3], pose: Se3Pose}`, `Mask: u8[H,W]`, `PointCloud{points: f32[N,3], colors?: f32[N,3]}`.
- Robot: `JointState{positions, names?}`, `Trajectory{waypoints: list[JointState]}`, `GripperState{position}`, `ArmState{joint_state, gripper_fraction, ee_pose, gripper_qpos, proprio_state}` — `proprio_state` layout is policy-training-exact (e.g. openpi-LIBERO `[eef_pos(3), axisangle(3), gripper_qpos(2)]`); never transformed.
- Aggregate: `Observation{cameras: list[CameraFrame], arms: list[ArmState]}`.
- Planning: `CollisionMesh{name, vertices, faces, pose}`, `WorldConfig{meshes}`, `GraspCandidates{poses, scores}`.

`gap/schema.py` (~50 lines): registry mapping type-name strings (used by subgraph `inputs:`/`outputs:` declarations and the viz frontend) → TypedDict definitions; the validator and `viz/graph_builder.py` consult it instead of proto descriptors.

## 6. Tool layer (`gap/tools/`)

Ported ToolRegistry (the dev tree's `tools/_registry.py`) minus the GrpcAdapter:
- `@tool(name="geometry.filter_and_compute_obb", summary=..., tags=("perception",))` on type-hinted functions; schemas via existing `extract_schema` TypedDict introspection (drives both validation and LLM catalogs).
- Names: `robot.*`/`sim.*` reserved for connectors; `<model>.<func>` for tool bundles (a skill may also expose `<skill>.<func>` when callable as a unit); loader rejects collisions. Model-named prefixes match the old service short names (`sam3.SegmentText` → `sam3.segment_text`, `geometry_svc.FilterAndComputeOBB` → `geometry.filter_and_compute_obb`) — the migration is prefix-preserving snake_casing for most call sites. The two legacy pure tools (`geometry.iou`, `geometry.pose_distance`) already carry the right prefix and fold into the geometry bundle.
- **Guards** (safety limits): today enforced in `ctx.call` keyed on gRPC service-name prefixes (the dev tree's guards module) — re-keyed on **tool tags** (`perception`/`planning`/`sim_step`) at `ctx.tool` dispatch; limits from task.yaml `safety_limits` with `GAP_MAX_*` env fallback; `GuardLimitExceeded(BaseException)` stays uncatchable by skills.
- Errors: skills raise `PipelineError` subclasses (`PerceptionFailed`, `PlanningFailed`, `GraspFailed`, `ValidationFailed`, …); executor wraps into `NodeExecutionError` and routes subgraph `on_error`.

`ctx` (NodeContext, ported) — the whole skill-facing surface: `ctx.tool(name, **kw)` (sole dispatch; `ctx.call`/`ctx.service` die with gRPC), `ctx.publish(value)` (streaming), `ctx.cancel_token.raise_if_set()`, `observation_stream: ObservationStream` injected param with `.latest(timeout)`. Trace hooks and registries stay private.

## 7. Connector design (`gap/connector/`, `gap/envs/`)

```python
def sim(env="libero", *, task="libero_object/0", cameras=None, headless=True, seed=None,
        record_video=False, **env_kwargs) -> SimConnector
def real(robot, *, rr_config=None, rr_autostart=True, **robot_kwargs) -> RealConnector

class Connector:                               # context manager
    config: EnvConfig                          # arm_dof, num_arms, action_mode, control_freq,
                                               # home_joints, tcp_offset, tcp_rotation_z,
                                               # urdf_path, default_cameras, is_real
    capabilities: Capabilities                 # reset / success_check / video / world_state
    def get_observation(self) -> Observation
    def close(self)
class SimConnector(Connector):
    def reset(self, seed=None); def check_success(self) -> tuple[bool, float]
    def world_snapshot(self) -> World          # checkpoint evaluation (ground truth)
    def start_video(self); def save_video(self, path) -> Path
class RealConnector(Connector):
    def wait_ready(self, timeout_s=60)
```

Registered connector tools (absorbing the four gRPC servicer method bodies from `services/sim_bridge/server.py` — GoToPose IK orchestration, gripper settle loops, trajectory execution): `robot.get_observation`, `robot.get_ee_pose`, `robot.go_to_pose`, `robot.go_to_pose_cartesian`, `robot.move_to_joints`, `robot.execute_trajectory`, `robot.go_home`, `robot.solve_ik`, `robot.open_gripper`/`close_gripper`/`get_gripper`, and sim-only `sim.reset`, `sim.check_success`, `sim.step`, `sim.apply_policy_action` (envs that support direct VLA actions). All take `arm_id=0` (multi-arm-ready); bimanual `*Both` variants are post-v1.

- **Env registry** (`gap/envs/registry.py`) replaces the 280-line if/elif (`server.py:463-742`): `register_env(name, "gap.envs.libero_env:make_env", prefix=False)`; factories are lazy dotted paths (registry imports without mujoco/pyzed); factory returns `(BaseEnv, EnvConfig)`. `GAP_*` env vars replace `VOS_*`.
- **IK — built into the connector, always in-process, deliberately simple.** `gap/connector/ik.py` ports `ik_backend.py`'s pyroki path (JAX, in-process — pyroki ships as a GaP core dependency); it powers `robot.go_to_pose`/`go_to_pose_cartesian`/`robot.solve_ik`. No pluggable backend, no `ik_backend:` config (the YAM 6-DOF case that needed CuRobo-IK is cut). **Motion planning is a separate concern**: the `curobo` tool bundle produces collision-aware trajectories that skills execute via `robot.execute_trajectory` — it is not IK plumbing.
- **Franka (streamlined)**: `robots_realtime` vendored as a submodule, **never imported** (own pinned env, realtime loops). `rr_launcher.py` (~100L) spawns `uv run --directory third_party/robots_realtime rr-session <config>` in a process group with log tee + kill-on-close. Order: `FrankaRealEnv` binds the msgpack server (port 9000) pre-seeded with hold-home → spawn client (it retries until the server is up) → `wait_ready()` blocks on first RGB, surfacing the existing `_diagnose_missing_rgb` diagnostics on timeout. `rr_autostart=False` restores the two-terminal debug flow. GoHome safety keys off `config.is_real`. The 50 Hz republish + heartbeat threads port unchanged.
- **UR+ZED**: `ur_zed_env.py` from master, dexnet `sys.path` hacks removed, `pyzed`/`rtde_receive` imports lazy with pip-hint errors. Perception-only (no motion tools registered).
- **Data collector** (`gap/connector/collector.py`, NEW ~150L): wraps env stepping to record synchronized obs/action/reward per control step → HDF5 (LeRobot-convertible). Nothing equivalent exists today (only video frames) — required by the collect_and_train example.

## 8. open-robot-skills repo design

```
open-robot-skills/
├── README.md                 # contribution guide, catalog table, format spec pointer
├── pyproject.toml            # ONE distribution; each bundle = one extra (see §14) + uv.lock
├── .claude-plugin/marketplace.json   # optional: doubles as a Claude Code plugin marketplace
├── tests/
├── tools/                    # model-backed callables, NAMED BY MODEL (former gRPC services)
│   ├── sam3/                 #   segmentation (segment_text/point/box) + tracker
│   ├── grounding-dino/       #   open-vocabulary detection (detect)
│   ├── gemini-er/            #   Gemini Robotics-ER 2D detection/spatial reasoning via the
│   │                         #   google-genai SDK (API-based — no self-hosting). NOTE: the dev
│   │                         #   tree's scripts call a gemini_er.v1 service that has NO impl
│   │                         #   in-tree; this bundle implements detect() directly on the SDK
│   │                         #   (usage extracted from perceive_gemini_er.py)
│   ├── molmo/                #   pointing VLM (point_prompt/query/query_yes_no). OPTIONAL:
│   │                         #   self-hosted vLLM; the generic API VLM can't point — gemini-er is the
│   │                         #   API-based alternative for detection
│   ├── vlm/                  #   generic API VLM (query/query_yes_no); provider = openrouter
│   │                         #   default / vertex; zero VRAM
│   ├── curobo/               #   collision-aware MOTION PLANNING (plan_to_grasp_poses,
│   │                         #   plan_with_grasped_object, plan_linear, plan_to_pose…)
│   │                         #   — REQUIRED for the G1 gate; CUDA-JIT documented.
│   │                         #   (IK itself lives in the connector, not here.)
│   └── geometry/             #   pure math, no model: mask_to_world_points,
│                             #   filter_and_compute_obb, top_down_grasp_candidates,
│                             #   build_world, iou, pose_distance
└── skills/                   # manipulation strategies — ALL FLAT (no atomic/composite)
    ├── perceiving-objects/             # perception_single: DINO + VLM letter-select + SAM3
    ├── perceiving-objects-oneshot/     # ← perception_any (refactor/progress ONLY, not on backup):
    │                                   #   DINO + one-shot VLM set-of-marks + SAM3; clean
    │                                   #   not_found for clean-all-items loops — the grocery
    │                                   #   workhorse and the steered-policy perceiver
    ├── perceiving-object-parts/
    ├── grasping-with-planner/          # needs the curobo tool bundle
    ├── grasping-direct-ik/
    ├── grasping-short-axis/            # ← grasp_short_axis (refactor/progress): deterministic
    │                                   #   handle/short-axis grasp; deps (curobo+geometry)
    │                                   #   already required for G1
    ├── transporting-objects/
    ├── tracking-objects/               # exposes its callable as a tool (streaming)
    ├── pi05-libero/                    # learned-policy skill: openpi π0.5 LIBERO checkpoint.
    │                                   #   Owns its serving preset (skill name == preset == policy
    │                                   #   id); exposes its callable as a tool (openpi websocket);
    │                                   #   gripper-cycle + VLM termination
    └── molmoact-libero/                # learned-policy skill: MolmoAct LIBERO checkpoint (same task
                                        #   family — the MolmoAct alternative to pi05-libero)
```

Both categories use the same Agent Skills bundle format (SKILL.md + resources); the folder conveys the kind. Each is one directory = one bundle = one PR for contributors.

**Frontmatter** = Agent Skills spec core (`name` ≤64 lowercase-hyphen == dirname; `description` ≤1024, third-person "use when…"; optional `license`, `compatibility: "requires gap>=0.1"` — loader warns on mismatch; `metadata` for category/tags) **plus all GaP extensions nested under one `gap:` key** so spec fields are never overloaded (notably `allowed-tools`, whose Claude Code semantics differ):

```yaml
---
name: perceiving-objects
description: Detect and localize a named object in the workspace as an oriented bounding
  box + mask + point cloud, using open-vocabulary detection, VLM disambiguation and
  segmentation. Use when a manipulation task needs to find a target object.
compatibility: requires gap>=0.1
metadata: {category: perception, author: gap-team}
gap:
  allowed_tools: [robot.get_observation, grounding-dino.detect,
                  vlm.query, sam3.segment_box,
                  geometry.mask_to_world_points, geometry.filter_and_compute_obb]
  exit_conditions: {found: "...", not_found: "..."}
  produces_outputs: {"<name>_obb": OrientedBoundingBox, "<name>_mask": Mask}
  canonical_scripts: [{perceive_dino_vlm: scripts/perceive_dino_vlm.py}]
  prompts: {vlm_select_box: prompts/vlm_select_box.md}
  references: [{title: ..., path: references/perception_pipeline_invariants.md}]
  hard_rules: [...]
  streaming: false
---
```
Tool bundles use the same format with `gap.tools:` instead (exposed functions + one-line summaries; no exit_conditions/canonical_scripts). The legacy `runtime.shape` (atomic|composite) field is **gone** — the tools/ vs skills/ folder split replaces it, and formerly-"atomic" skills simply expose their callable in `tools.py`. Conversion from the legacy frontmatter is otherwise mechanical (union of fields verified across all 11 bundles); `composes` (service FQNs) is dropped. Dependencies: **declared once, as the bundle's extra in open-robot-skills/pyproject.toml** (extra name == bundle name); the SKILL.md body shows the human-readable install line (`pip install -e "open-robot-skills[sam3]"`) with rationale, per the spec. No per-bundle requirements.txt.

**Skill-authoring contract** (the stable import surface; everything else internal):
```python
from gap import NodeContext, CancelToken
from gap.types import Se3Pose, OrientedBoundingBox, Mask, Observation, ...
from gap.errors import PipelineError, PerceptionFailed, PlanningFailed, GraspFailed, ...
from gap.skills import tool, Skill, SkillMeta, load_prompt
from gap.testing import FakeContext, make_test_observation     # unit-test without a robot
```
- Bundle tools live in `tools.py`, lazy-load model weights on first call (module-level cached loader), honor per-bundle `device` config.
- `load_prompt(__package__, name, **vars)` + synthetic packages kept (legacy prefix → `gap_skills.*`) — skill scripts depend on bundle-relative prompt loading.
- Class-based stateful tools (`Skill` base, one instance per workflow — pi05-libero, molmoact-libero, tracking-objects) keep working. The policy skills subclass `gap.runtime.policy_skill.PolicyLoopSkill`, which wraps the unchanged closed-loop body `gap.runtime.policy.run_policy_loop`; each sets `preset` to its bundle name (== preset == policy id) and takes no `policy_id` argument.
- open-robot-skills has a **one-way runtime dependency on GaP** (ctx/types/errors/load_prompt — verified: every existing skill script imports these). "Standalone" means discovery and contribution are path-based (`gap.skills.find_skills_path`: explicit path > `GAP_SKILLS_PATH` > the side-by-side checkout), not pip-coupled.

Cut from the dev tree: grasp_moe, grasp_multi (graspgen), bimanual_crate_lift (yam).

**Catalog & progressive disclosure** map 1:1 onto the existing compose flow, now with the cleaner split: the **coordinator** sees the *skills* catalog (name+description) and assigns each subgraph to a skill; each **subgraph agent** gets that skill's full SKILL.md plus the schemas of its `gap.allowed_tools` (drawn from connector tools + tool bundles); scripts/references load on demand. Tool bundles never own subgraphs — they appear only through the flat tool catalog.

## 9. Execution engine (`gap/runtime/`)

- **Executor** ports as-is (super-step frontier scheduler, subgraph recursion, streaming slots + cancel grace, Send fan-out, node_visit_cap cycle guard, recovery on end nodes). Constructor takes `connector` + `tool_registry` (+ skills registry); the `executor_factory.__new__` hack and `_activate_direct_real_runtime` die.
- **Validation** (`validate.py`): structural rules unchanged; type checking via `gap/schema.py` + TypedDict introspection (proto descriptor walk deleted).
- **Tracing** (`tracing.py`): same on-disk layout — `workflow.json`, `dag_trace.json`, `node_data/<id>/` with PNG/NPZ assets — with asset extraction switched from proto walking to dict/numpy structural checks (`_is_image/_is_mask/_is_depth/_is_pointcloud`). The layout is a **stability guarantee** (viz + users depend on it); a golden-trace parity test replays a recorded trial through the new extractor.
- **Observation stream**: background thread polls `connector.get_observation` at configured Hz; `.latest()` reads recorded into trace.
- **Policy loop** (`policy.py` + `policy_manager.py`, backup branch versions with the gripper-cycle detector): policies reached via openpi websocket; `policies:` config supports **external `url:`** and **managed `start_cmd:` (`{port}` placeholder, env overrides, preflight validation)**.
- **Policy serving presets** (the clean serving story G3 needs — a user must be able to reproduce it, not just read a config hook): a small preset registry (`gap/runtime/policy_presets.py`) of named entries `{checkpoint URI + download step, start_cmd template, env}`. v1 ships **`pi05-libero`** (openpi recipe: checkpoint from openpi-assets + `serve_policy.py` start_cmd; docker optional) and **`molmoact-libero`** (covers the benchmark's policy A/B axis). `gap policy serve <preset>` is the one-command path (downloads checkpoint if missing, spawns via the existing policy_manager managed mode); task.yaml `policies:` entries may reference `preset: pi05-libero` instead of spelling out start_cmd. `examples/steered_policy/` uses the preset, so G3 is reproducible end to end. Weight ownership split: `gap skills check --download` owns bundle weights; `gap policy serve` owns policy checkpoints.
- **Verification (checkpoints) — ships in v1.** Extracted from the dev tree's rehearsal eval library into `gap/runtime/verify/`: `World`/`Body`/`Robot` views (poses, AABBs, contacts frozenset, cavity bounds; `is_grasped/is_in/is_on/is_above/eventually/always`), `Checkpoint`, `evaluate_checkpoint`, `load_checkpoints` — all generic; only the Isaac factory drops. NEW: `LiberoWorldAdapter` (~150-200L; object poses from env ground truth, contacts from MuJoCo buffers, AABBs precomputed at reset) and a checkpoint hook (~100L) on the existing `subgraph_exit_hook`/`SubgraphExitEvent` seam, loading `<workflow_dir>/checkpoints/<sg>.py` and evaluating `validate=True` predicates against `connector.world_snapshot()`. `gap.execute(..., checkpoints="off"|"warn"|"raise")`; real connectors lack `world_state` → skipped with a log line.

## 10. Agent library (`gap/agent/`)

Pipeline ports as-is: coordinator (topology + subgraph declarations) → per-subgraph agents (state machines + inline scripts, retry loops) → checkpoint agent (postconditions; now behind a config flag, default ON) → graph validation → LLM script-fix loop (≤2 attempts). Reviewer agent deleted (dead). Prompts: tool catalogs rendered from ToolRegistry TypedDict schemas (proto2prompt, `gen/prompt`, `make prompt` die); dev-tree builder references renamed; `gap:` frontmatter consumed.

**Generic LLM provider layer** (`llm.py`):
```python
class LLMClient(Protocol):
    async def complete(self, *, system, messages, **gen_kw) -> str
    async def complete_with_tools(self, *, system, messages, tools, tool_handler, **gen_kw) -> str
def make_llm(cfg: LlmConfig) -> LLMClient      # cfg.provider: openrouter | vertex
```
- `openrouter` (default): httpx OpenAI-compatible chat-completions path against `https://openrouter.ai/api/v1`, model default `gemini-3.1-flash-lite-preview`, native tool-use loop via the OpenAI tools API (the checkpoint agent's meta-tools need `complete_with_tools`); point `endpoint:`/`base_url` at any other OpenAI-compatible server (native OpenAI, a local vLLM) to reuse the same path.
- `vertex`: google-genai for gemini models only; `[vertex]` extra.
- Disk response cache kept across providers; GRPO/logprobs path dropped. Auth via standard env vars.

`gap.agent.generate(instruction, *, skills, model=None, provider=None, out_dir=None, config=None) -> GeneratedGraph{path, workflow, code}` (+ `generate_sync`). The same provider layer serves the `vlm` tool bundle — quickstart perception (letter-based selection, deliberately coordinate-free) verified against the hosted VLM.

Graph authoring by humans is first-class: `gap.builder` (WorkflowSpec/Subgraph, ports as-is — prompt-referenced API names stay stable) is documented alongside `gap generate`.

## 11. Benchmark (`gap/benchmark/`)

Grid harness from the progress worktree: `modes × families × variations × task_ids × seeds`, sequential modes (GPU contention), per-cell `launch()`, summary.json/tsv with success_rate / completion_rate / per-task pivots, hard-linked video collation. Modes: `llm_generation`, `llm_plus_policy`, `policy_only` (unimplemented `monolithic` dropped). Families — all verified supported by the public posvar fork (`https://github.com/ehehee/Variational-Automation-Benchmark`, submodule, pin SHA): `posvar` (pos_var / permutation / basket_swap / all), `libero`, `libero_pro`, `grocery_packing`.

**The G1 acceptance run** is a pinned benchmark config (`examples/benchmark/grocery_acceptance.yaml`: curobo-only planning, grasping/transport skills in prompt, curated per-task object hints): `llm_generation` mode over the grocery/all-variance suites, threshold **≥90% success** (trial count configurable; 500 for the full gate, a 20-trial smoke for nightlies). `gap benchmark --gate examples/benchmark/grocery_acceptance.yaml` exits non-zero below threshold — this is the regression bar for every release.

**Parallel trials** (real rework, ~500-700 LOC): `compose/parallel.py` keeps the mp.spawn worker pool, trial collection, task-reuse, EGL spread (`GAP_MUJOCO_EGL_DEVICES`), **and the per-trial `trial_max_seconds` watchdog** (a hung trial must not stall a 500-trial gate), but deletes Ray-Serve port plumbing + services.yaml generation; each worker builds its own in-process connector + tool registry inside `worker_setup()` (models aren't picklable — construct post-fork). VRAM scales with workers → default `num_workers=1`; `[ray]` extra adds shared skill actors (`gap/tools/ray_executor.py`, thin `@ray.remote` wrappers over the same functions). **Benchmark resume**: cells are independent — `gap benchmark --resume` skips cells whose results already exist in the output dir and rebuilds the summary, so the 500-trial G1 gate survives interruptions.

## 12. Viz (`gap/viz/`)

FastAPI + React 19 trial browser ports as-is (backup branch's redesigned swimlane `graph_builder.py`); proto schema introspection replaced by the `gap/schema.py` registry (~30-line change); frontend untouched, `dist/` checked in as package data (no node needed for pip users). Live-ish by construction (rescans outputs per request). `replay3d` kept (viser). `render.py` (from `scripts/render_graph.py`): matplotlib v3-graph → paper-ready PDF/PNG. `trace_diff` kept as a dev tool — it is the parity instrument for the de-proto migration itself.

## 13. CLI

`gap run <graph> [--sim ENV:TASK | --real {franka,ur_zed}] [--skills PATH] [--validate-only] [--no-trace]` — **tracing is ON by default** (dev defaulted off, which left `gap viz` empty; the trace is the product), `gap generate "<instruction>" [--skills] [--provider] [--model] [--out]`, `gap benchmark <config.yaml> [--gate] [--resume]`, `gap viz [--root] [--port]`, `gap skills list|check [--download]|new <name>`, `gap policy serve <preset>`, `gap trace-diff A B`. Lazy-import dispatch pattern kept. No `gap serve`.

## 14. Packaging

- **GaP core deps**: numpy, scipy, pyyaml, httpx, fastapi, uvicorn, pillow, opencv-python-headless, robot_descriptions, yourdfpy, **pyroki + jax (CPU — the connector's in-process IK)**, matplotlib, h5py, msgpack(+numpy), viser. No grpcio/protobuf/protoc/cargo anywhere.
- **Extras**: `[libero]` mujoco/robosuite/posvar-fork/bddl/robomimic/imageio[ffmpeg]; `[ray]`; `[real]` ur-rtde (pyzed = documented manual ZED SDK install, lazy import); `[vertex]` google-genai; `[dev]` pytest/ruff/mypy.
- **open-robot-skills dependency mechanism (clean by construction)**: open-robot-skills is one pip distribution; **each bundle = one extra** (extra name == bundle name: `[sam3]`, `[grounding-dino]`, `[curobo]`, `[gemini-er]`, `[pi05-libero]`, `[molmoact-libero]` — the last two both pull `openpi-client`, …) — installed one bundle at a time with `gap skills install`, never as a meta-extra of the engine. Non-PyPI deps (the sam3 fork, nvidia-curobo) are **pinned `git+https` entries inside those extras** — one resolver run surfaces cross-bundle conflicts at install time, and CI installs `[all]` to prove co-installability (dev's docker venv already proved these deps coexist). `uv.lock` pins the exact G1-gate environment. **Ownership split: pip owns code; `gap skills check` only verifies** (import probe + weight presence per bundle, mapping bundle→extra by name); `--download` prefetches **weights only** (HF_TOKEN documented) — nothing ever pip-installs behind the user's back. Quickstart is literally `pip install -e gap -e "open-robot-skills[quickstart]"` → `gap skills check --download` → `gap run …`. The one documented wart: curobo's CUDA JIT (`--no-build-isolation`, `CUDA_HOME`) lives on that extra alone. P2 task: diff the dev tree's vendored sam3 against upstream — if patched, publish the fork (or vendor under `tools/sam3/_vendor/`) and pin that.
- Submodules: `robots_realtime`, `Variational-Automation-Benchmark` (both pinned). `py.typed` shipped. uv-first docs, pip supported.

## 15. Testing strategy

**Design principle — every LLM boundary has a recorded artifact.** The pipeline is split so the deterministic suite exercises *everything* (parsing, validation, execution, tracing, viz) against canned LLM outputs and golden graphs, while `llm`-marked tests regenerate those artifacts live and check *equivalence*, not byte-equality (per the original requirement: "LLM needed for equivalence generation, LLM not needed" otherwise).

### 15.1 Markers & where suites run

| Marker | Needs | Runs |
|---|---|---|
| *(none)* | nothing (CPU, no heavy deps) | locally + PR gate, 3.10/3.11/3.12 |
| `sim` | `[libero]` (mujoco/EGL) | nightly GPU runner; locally on demand |
| `gpu` | model weights (torch/sam3/dino) | nightly GPU runner |
| `llm` | `OPENROUTER_API_KEY` (vertex variant opt-in) | nightly, small token budget |
| `real` | hardware | manual checklist before release |

`pyproject`: `addopts = "-m 'not llm and not gpu and not sim and not real'"`.

### 15.2 `gap.testing` (public module — same fixtures we use, exported for contributors)

- `FakeContext(tool_responses={name: value | callable | list[values]}, record=True)` — NodeContext stand-in; scripted per-tool responses (lists pop sequentially), full call log for assertions; raises configured `PipelineError`s to test error paths.
- `FakeConnector(observations=[...], world_snapshots=[...], config=EnvConfig(...))` — implements the Connector ABC; scripted observation/world sequences; registers standard `robot.*`/`sim.*` tools backed by a kinematic stub.
- `make_test_observation(objects=[("cube", pose, size)], camera=...)` — geometrically *consistent* synthetic data: rgb with rendered colored boxes, depth/intrinsics/camera-pose that reproject correctly, so perception-math tests (mask→points→OBB) verify real numerics, not mocks.
- `connector_contract_suite(connector_factory)` — parametrized ABC-compliance tests any connector must pass: observation shapes/dtypes (`u8[H,W,3]`, `f32[H,W]`, intrinsics 3×3), tool names registered, capabilities consistent with type (Sim ⇒ reset/success/world_state), `reset(seed)` determinism for sims, `close()` idempotent. Exported so third-party connectors (the cable/ROS-style path) get a conformance test for free.
- `assert_graph_valid(graph)`, `golden_trace(tmp_path)` helpers; `gap/testing/equivalence.py` (below).

### 15.3 GaP unit tests by subsystem (deterministic suite)

- **types/schema** (new): quaternion wxyz↔xyzw boundary conversion; numpy dtype/shape invariants; `gap/schema.py` registry resolves every type-name string used by shipped graphs' `inputs:`/`outputs:` declarations; unknown-type error message.
- **tools** (new): `@tool` registration + TypedDict schema extraction (port the dev tree's schema tests); name-collision rejection (`robot.*` reserved); dispatch through stub tools incl. kwargs filtering; **guards**: tag→category classification, limit decrement, `GuardLimitExceeded` escapes a skill's bare `except Exception`; `NodeExecutionError` wrapping preserves cause.
- **runtime/workflow** (ported + extended): v3 parse suite (`test_workflow_v3`), strict-key/legacy-type rejection; **Ref resolution** on plain dicts/lists/nesting/negative indices/streaming-slot snapshot reads, fuzzed over `$ref` paths harvested from all shipped graphs; validator rules W1–W8/S1–S11 (ported — they're proto-free already except type lookup, which now hits the registry); recovery `ToolCall` parse.
- **executor** (ported + new on `FakeContext`/stub tools): super-step fan-out and first-error cancellation; conditional routing on exit values incl. `on_error`; subgraph input binding + output refs; streaming (publish → downstream `latest()`, cancel-grace teardown, no-outgoing-edge rule); Send dynamic fan-out collects list; `node_visit_cap` cycle guard; recovery actions run best-effort on end nodes; `SubgraphExitEvent` payload correctness.
- **tracing**: **golden-trace parity** — fixture trial recorded from the dev tree (workflow + node outputs as dicts/arrays) replayed through the new extractor; assert identical `dag_trace.json` structure and identical `node_data/` asset filenames/formats. Per-type extractor units (image/mask/depth/cloud, large-array summarization `<ndarray shape=… >`).
- **verify/checkpoints**: World/Body/Robot fixture vocabulary tests ported from eval_lib's coverage (`is_grasped/is_in/is_on/is_above/eventually/always`, cavity bounds); `evaluate_checkpoint` arity-2 predicates + outputs dict + eval-error capture + diagnostics; `load_checkpoints` sidecar exec isolation; hook integration: `FakeConnector` with scripted enter/exit world snapshots → warn vs raise modes; transition checks (held-after-grasp, released-after-place).
- **connector/envs**: env registry resolve/prefix/default + lazy factories (**import the registry with mujoco absent** — must not raise); `EnvConfig` application (tcp offset/rotation, home joints); msgpack bridge round-trip against a mocked rr client speaking recorded protocol frames (obs in, 50 Hz action frames out, heartbeat staleness detection); `rr_launcher` lifecycle on a fake script (spawn, log tee, process-group kill); data collector (HDF5 schema, obs/action length sync, episode boundaries).
- **agent (no LLM)**: prompt-assembly snapshots from a stub skill catalog (coordinator/subgraph/checkpoint prompts contain the right tool schemas and SKILL.md bodies); **response parsing on canned LLM outputs** (coordinator WorkflowSpec extraction, subgraph + inline-script extraction, missing-capability block); script-fix loop driven by canned validation errors; builder→JSON round-trip; every golden graph passes `validate_workflow`.
- **llm providers**: mocked-HTTP contract tests per provider — request shaping (system/messages/max_tokens), streaming assembly, the tool-use loop via the OpenAI tools API (openrouter), vertex gemini routing, 429 retry/backoff, disk-cache hit/miss keys.
- **benchmark** (ported from progress + extended): config/family expansion (posvar×4, libero, libero_pro, grocery), mode×policy axis collapse, report math (success_rate, trial-weighted completion_rate, per-task pivots) on synthetic results, TSV/JSON writers.
- **viz**: `graph_builder` on golden graphs → WorkflowGraph snapshot (swimlanes, control vs data edges); `trial_loader` discovery on a fixture outputs tree; FastAPI endpoints via TestClient (trials list, node data, assets); `render.py` smoke (PDF bytes non-empty) on each golden graph.
- **CLI**: every subcommand `--help` without heavy imports (lazy-dispatch regression); arg→config plumbing.

### 15.3b Branch-content coverage (the G1–G3 features get first-class tests)

- **perceiving-objects-oneshot** (perception_any): script units on `FakeContext` — canned DINO detections + set-of-marks VLM answers including the `"none"` → `not_found` path (the clean-all-items termination G1 depends on); frontmatter/schema validation like every bundle.
- **gemini-er**: mocked google-genai SDK tests (request shaping, bbox parsing, no-detection path); tool schema extraction; an opt-in live-API smoke (`llm` marker).
- **Steered policy**: `graph_obb_policy_loop`/`_grasp` validate in the default suite (golden graphs); `sim`+policy-server test executes the hover→handover path with a scripted policy stub (FakeContext-level unit: approach_above output feeds run_policy inputs; gripper-cycle termination unit ports from policy.py tests).
- **Grocery generation**: golden grocery graphs in the equivalence corpus (G2); `llm` suite generates K grocery graphs and gates on structural equivalence + sim execution.
- **G1 gate**: the acceptance benchmark config itself is exercised by a 1-cell smoke in nightly (`sim`+`gpu`+`llm`); the full 500-trial gate is a manual release step.

### 15.4 `llm`-marked suite (equivalence generation)

- **Graph equivalence, not text equality** — `gap/testing/equivalence.py`: normalize both graphs (canonical node ordering; name-insensitive DAG isomorphism per subgraph; compare the *skill multiset*, exit-condition wiring, and top-level topology), then a behavioral gate (generated graph validates and, under `sim`, executes to success). LLM output is stochastic → tests generate K graphs per golden instruction and assert a pass-rate threshold (e.g. ≥2/3 structurally equivalent + executable), recorded per-model so regressions are visible without flaking CI.
- Golden instruction set: the quickstart task + 2–3 libero tasks with checked-in reference graphs (regenerated by a documented `make goldens` script).
- Checkpoint-agent tool-use regression per provider (openrouter in nightly; vertex opt-in): generated sidecars load and their predicates evaluate against fixture Worlds.
- VLM probe: letter-based box selection on a fixture image with known answer.

### 15.5 open-robot-skills repo tests

- **Repo-level (CI, no GPU)**: every bundle in both `tools/` and `skills/` — frontmatter spec validation (name==dirname, description ≤1024 + "use when" heuristic, `gap:` block against a published JSON schema — tool bundles must declare `gap.tools`, skills must declare exit_conditions; `compatibility` parses); all referenced paths exist (canonical_scripts/prompts/references/examples); **the pyproject extras table covers every bundle (extra name == bundle name) and every bundle's own environment resolves**; catalog load + tool-name collision check; schema extraction on every declared tool; **lazy-import proof**: importing every `tools.py` must not pull torch/transformers into `sys.modules`.
- **Per-bundle units** (using `gap.testing`): skill scripts run against `FakeContext` with canned tool responses (e.g. perceiving-objects: canned detections + VLM letter answer + mask + synthetic depth → assert OBB/mask outputs and exit condition; error path → `not_found`); tool bundles' pure-math layers CPU-tested on `make_test_observation` geometry (geometry-bundle OBB/points numerics); model-touching paths behind `gpu` (one tiny smoke input per tool bundle). Connector IK tested in GaP core (pyroki solve vs known Franka poses, CPU JAX).
- **Cross-repo nightly**: quickstart end-to-end (`sim`+`gpu`+`llm`); recorded-observation replay proxies for hardware examples (cable: stored ZED frames through the perception graph).

### 15.6 What ports vs what's new

Ported from dev tree: `tests/runtime` (11 files), `tests/builder`, `tests/compose` (config), `tests/sandbox` (predicates), `tests/benchmark` (progress). Dropped with their subsystems: `tests/rehearse`, `tests/scene_spec`. New (this design): tools/guards, types/schema, tracing parity, verify/checkpoints, connector contract + msgpack/rr_launcher/collector, agent canned-response suite, llm provider mocks, equivalence harness, viz endpoint tests, open-robot-skills repo suite.

## 16. Examples

| Example | Content | Key deps |
|---|---|---|
| `libero_quickstart/` | **A NEW graph, authored for v1** — no curobo-free LIBERO graph exists anywhere in the dev tree (every `graph_cartesian_obb` variant calls `curobo.PlanToGraspPoses` via `grasp_curobo_obb`). Base it on graph_cartesian_obb's topology but swap the grasp subgraph to a grasping-direct-ik-style top-down descend. **Fallback decided**: if direct-IK grasp success is poor on libero_object/0 (<~80% in P4 testing), the quickstart documents the curobo install (CUDA-JIT toolchain) rather than shipping a flaky demo — G4's "one command" then includes that install step. | `[libero]` + sam3/grounding-dino/geometry bundles + `OPENROUTER_API_KEY` (vlm) |
| `grocery_fulfillment/` | the G1 flagship: posvar all-variance / grocery_packing tasks via `gap generate` + the acceptance benchmark configs (ported `grocery_packing*.yaml`); uses perceiving-objects-oneshot + grasping-with-planner (curobo) + transporting-objects | `[libero]`, curobo backend |
| `steered_policy/` | G3: `graph_obb_policy_loop` (+ `_grasp` variant) — perceive (perceiving-objects-oneshot) → `approach_above` hover over target → **handover to the learned-policy skill** (`{{policy_id}}.run`, e.g. `pi05-libero.run`; no `policy_id` input; gripper-cycle termination). The launcher auto-boots the skill's preset (downloads the checkpoint + spawns the server); `gap policy serve pi05-libero` runs it by hand | `[libero]`, `[pi05-libero]` (or run `gap policy serve pi05-libero`) |
| `cable_ur/` | master's cable example; standalone-connector showcase (perception-only) | `[real]`, ZED SDK |
| `real_franka_pick_place/` | `pick_and_place_jello` — **v2 schema, migrate to v3** | `[real]`, robots_realtime |
| `collect_and_train/` | N× execute + data collector → HDF5/LeRobot → external training recipe → eval reusing the steered_policy graph | `[libero]`, policy server |
| `viz_walkthrough/` | quickstart with tracing → `gap viz` → `gap.viz.render` PDF | — |
| `benchmark/` | smoke.yaml (1×1), posvar.yaml, grocery acceptance configs, full.yaml | `[libero]` |

Minimal task.yaml (config.py pruned): `task`, `suites`, `trials`, `environment.cameras`, `safety_limits`, `llm`, `policies`(+`policy_manager`), `skills` path. Deleted: `ray_serve`, `services`, `api_servers`, `base:` platform inheritance (the platform.yaml concept dies with services).

## 17. Safety, licensing, hygiene

`docs/safety.md` real-robot notice (E-stop, workspace clearance, hold-home preseed rationale). License audit: CuRobo (NVIDIA source-available, non-commercial terms — optional never-vendored dep), sam3/robosuite/posvar-fork/LIBERO licenses attributed in both READMEs. IP scrub before first commit: grep legacy project names and internal URLs; delete stray media (screenshots/mp4/MUJOCO_LOG present on the backup branch). Clean history: single initial commit per repo. CONTRIBUTING.md + issue templates both repos.

## 18. Deliberately post-v1 (state in README roadmap — name the cuts, don't hide them)

- **Learned grasp planners**: graspgen was deployed in dev with a shipped example (`graph_curobo_graspgen`); m2t2/graspnet were implemented. **v1 ships geometric grasping only** (OBB top-down candidates + curobo/direct-IK); learned grasp generation returns post-v1.
- **Non-pick-place domains**: `turn_on_stove` (articulated manipulation, 5 graph variants), `usb_insertion` (8 variants — depends on 11 master-only legacy skills: insert, align_port, find_contact, verify_gripper_grasp, …), `full_popcorn_cycle` long-horizon graphs, the droid_sim Isaac env; real-Franka examples shrink 6 → 1 for v1.
- Execution-feedback graph repair (the refine loop), remote model-serving tier, bimanual/YAM, Isaac rehearsal pipeline, cross-machine sim.
- Pointing: gemini-er ships in v1 as the API-based option; self-hosted Molmo stays optional.

---

