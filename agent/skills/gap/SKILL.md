---
name: gap
description: Program robots with GaP (graph-as-policy) — compile natural-language
  tasks into typed, verified robot skill graphs and run them on simulators or
  real robots. Use when the user mentions GaP or graph-as-policy, robot
  manipulation, robot skills, robot tools or capabilities, skill registries,
  open-robot-skills, LIBERO or MuJoCo manipulation sims, pick-and-place,
  grasping or perception strategies, generating or hand-authoring robot task
  graphs, robot benchmarks, or asks what a robot can do. Covers searching
  registries (gap registry / gap skills / gap tools), capability checks (gap
  check), running graphs in sim (gap run --sim), generating graphs from
  language (gap generate), authoring graphs with gap.builder, creating skill
  bundles and registries with scaffolded unit tests (gap skills new / gap
  skills test / gap registry init), debugging traces (gap viz, gap
  trace-diff), and safety-gated real-robot runs.
license: MIT
metadata:
  category: robotics
  tags: [robotics, manipulation, simulation, skills]
---

# graph-as-policy — GaP

GaP compiles robot tasks into **typed, verified skill graphs** and executes
them. The policy *is* the graph: nodes call tools (perception models,
planners, robot motions) or run skill scripts; postcondition checkpoints
verify progress against the simulator state. Two kinds of repos exist:

- the **engine** (`graph-as-policy`, imports as `gap`) — CLI, runtime,
  builder, codegen, connectors (LIBERO sim / real Franka & UR);
- **skill registries** — directories of bundles (`skills/<name>/SKILL.md`
  strategies, `tools/<name>/SKILL.md` model-backed tools).
  [open-robot-skills](https://github.com/graph-robots/open-robot-skills) is
  the canonical public registry, normally cloned side-by-side with the
  engine and auto-discovered. Any number of registries layer by precedence.

**Ground truth:** `gap <command> --help` is authoritative;
`references/cli.md` here is a generated snapshot. When working inside a
checkout, prefer its `docs/` (design.md, runtime.md, skills.md, safety.md)
over the digests in this skill.

## SAFETY — standing rules

- **NEVER run `gap run --real ...` or `gap.connector.real(...)` unless the
  human has explicitly confirmed that specific run in this conversation.**
  Real-robot work requires a human with the E-stop in hand and
  `docs/safety.md` read. Do not weaken this gate, ever.
- Sim-first, always: validate (`gap run <graph> --validate-only`), then sim
  (`--sim`), and only then discuss real hardware.
- Use `--checkpoints raise` on anything real; `warn` (the default) in sim.
- Never claim a task succeeded without evidence: checkpoint pass in the
  trace, `sim.check_success`, or `gap benchmark --gate`.
- Never commit API keys or tokens; set them as env vars.

## 0. Bootstrap / install check

Always start by finding out what exists and what can run:

```bash
uv run gap check            # inside a checkout (or just `gap check` if on PATH)
gap check --format json     # machine-readable; parse this when deciding
```

If the CLI is missing, set up from scratch (Linux; GPU needed for sim):

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # side-by-side
cd graph-as-policy
uv sync && uv run gap skills install sam3 grounding-dino geometry        # engine + sim + sam3/grounding-dino/geometry
uv run gap skills check --download   # verify bundles + prefetch model weights
```

Gotchas: cloning without `--recurse-submodules` breaks `uv sync`
("does not appear to be a Python project") — run
`git submodule update --init`; `gap skills install curobo` needs `CUDA_HOME` set
(cuRobo compiles CUDA at install); sim runs want `MUJOCO_GL=egl`.

Env vars that matter: `OPENROUTER_API_KEY` (default codegen provider) /
Vertex via gcloud ADC + `GOOGLE_CLOUD_PROJECT`;
`GAP_LLM_PROVIDER` + `GAP_LLM_MODEL` pin a non-default provider per
shell (vertex also needs `uv run --extra vertex`); `GAP_SKILLS_PATH`
(colon-separated registry roots); `HF_TOKEN` (gated weights);
`GAP_LLM_CACHE_DIR`, `GAP_LLM_NO_CACHE`.

## 1. Discover what the robot can do

```bash
gap registry list                      # active registries, precedence order
gap skills list                        # every bundle (skills + tools) + registry
gap skills table --format markdown     # paste-ready catalog
gap tools list                         # flat tool catalog with live schemas
gap tools show geometry.compute_obb    # full input/output schema + runnability
gap check                              # what is operational HERE + fix hints
```

`gap check` is the decision input: a bundle is READY (deps importable,
declared GPU/env requirements met) or NOT READY with a fix hint; each
skill is READY or BLOCKED by the tool bundles it needs. Apply the fix
hints (`uv sync --extra X`, `export KEY=...`) before attempting tasks that
need those bundles. Resolution precedence for registries: `--skills`
flags > `$GAP_SKILLS_PATH` > project `pyproject [tool.gap].registries` >
`~/.config/gap/registries.toml` > the side-by-side open-robot-skills
checkout. Details: `references/registries.md`.

## 2. Run an existing graph in sim

```bash
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
    --sim libero_object/0 --checkpoints warn
```

- `--validate-only` first when in doubt — structural validation without a
  robot.
- `--inputs key=value` binds top-level workflow inputs.
- `--record-video` (sim only) saves `<trace-dir>/run_video.mp4` of the run.
- Every run writes a trace (`dag_trace.json` + per-node assets) under
  `--trace-dir` (default `outputs/run_<timestamp>`). Read it for node
  inputs/outputs, checkpoint results, and failures.
- `gap viz` serves an interactive trial browser at `localhost:9432`.
- The repo's `examples/` each have a README: `libero_quickstart` (start
  here), `grocery_fulfillment` (benchmark-gated), `steered_policy`,
  `collect_and_train`, `cable_ur`, `real_franka_pick_place`.

Programmatic equivalent:

```python
import gap
conn = gap.connector.sim("libero", task="libero_object/0")
result = gap.execute("examples/libero_quickstart/graph", conn,
                     checkpoints="warn")
print(result.success, result.exit_status, result.trace_path)
```

## 3. Compile a graph from language

**When this skill is running inside Claude Code, *you* are the codegen
pipeline — do NOT call `gap generate`.** Turning a task into a graph is a
§4 hand-authoring job: read the task, pick skills from the active
registries, build the nodes with `gap.builder`, attach checkpoints, and
drive `gap run --validate-only` to a clean pass. You are a stronger model
than the one `gap generate` would dispatch to, and you stay in the loop to
fix validation errors — so author directly, don't shell out.

`gap generate` is the **headless** path: benchmark grids, cron, or
`gap.agent.generate_sync()` called from code — any run with no interactive
model present. It dispatches a coordinator → per-subgraph → checkpoint
pipeline to the configured provider's API (`--provider`/`--model`; see
`gap check` for what's configured). Reach for it only when there is no
Claude in the loop.

```bash
# headless only — NOT the path to use from inside Claude Code:
uv run gap generate "pick up the alphabet soup and put it in the basket" \
    --provider openrouter --out outputs/soup
```

## 4. Author a graph (gap.builder) — the default path

This is how you turn a task into a graph. The loop:

1. **Decompose** the task into one subgraph per skill it needs
   (`gap skills list`). Canonical pick-and-place shape: perceive target →
   perceive container → grasp → transport.
2. **Read the contracts first** — don't guess field names. `gap tools show
   <tool>` gives exact input/output fields (what you bind with `Ref`); each
   chosen skill's `SKILL.md` gives its recommended inner state flow and
   canonical scripts. Mirror them — they are the contract the skill was
   validated with.
3. **Build** the `Subgraph`s and the top-level `Workflow`, then write a
   script file for every `type="script"` node.
4. **Attach `validate=True` checkpoints** per subgraph so success is
   verified against sim ground truth, not assumed (a grasp/place with no
   checkpoint is unverified).
5. **Validate → fix → repeat** (`gap run --validate-only`) until clean,
   then sim.

```python
from gap.builder import Workflow, Subgraph, Ref

sg = Subgraph(name="grasp_sg", skill="grasping-direct-ik")
sg.add_input("target_obb", type_name="OrientedBoundingBox")
sg.add_node("open", type="tool", tool="robot.open_gripper")
sg.add_node("grasp", type="tool", tool="geometry.top_down_grasp_candidates",
            inputs={"obb": Ref("in.target_obb")})
sg.add_exit("grasped")
sg.set_on_error("failed")
sg.add_edge("START", "open"); sg.add_edge("open", "grasp")
sg.add_edge("grasp", "grasped"); sg.add_edge("grasped", "END")

wf = Workflow(name="my_task")
wf.add_subgraph(sg)
wf.add_node("grasp_node", type="subgraph", ref="grasp_sg")
wf.add_node("done", type="end", status="success")
wf.add_edge("START", "grasp_node")
wf.add_conditional_edges("grasp_node", {"grasped": "done", "failed": "done"},
                         router_field="exit")
wf.save("my_graph/workflow.json")
```

Iterate against the validator until clean — failures cite rule codes
(workflow W1–W8, subgraph S1–S11):

```bash
uv run gap run my_graph --validate-only
```

Wire tool names exactly as `gap tools list` reports them; bind dataflow
with `Ref("node.field")` / `Ref("in.<input>")`. Add postcondition
checkpoints (`sg.add_checkpoint(...)`) so success is verified, not
assumed. Full surface + rule digest: `references/authoring-graphs.md`.

## 5. Author a new skill or tool bundle

```bash
uv run gap skills new my-skill --kind skill          # or --kind tool
# targets the highest-precedence registry; --registry NAME to choose
```

The scaffold includes SKILL.md, `scripts/example.py` (or `tools.py`), and
a unit-test skeleton `tests/test_my_skill.py`. Then:

1. Fill SKILL.md: `name` == dirname; description **must contain a
   "Use when …" sentence** (it is the planner's entire view of the
   bundle); GaP extensions only under the `gap:` key; declare
   `gap.requires:` (`{gpu: true, env: [KEY], env_any: [...], weights: true}`,
   or `{}`) so `gap check` can vouch for it.
2. Implement. Skill scripts: `def run(ctx, *, typed_kwargs) -> TypedDict`,
   call tools via `ctx.tool("bundle.fn", ...)`, load prompt templates via
   `load_prompt(__package__, name, **vars)`. Tool bundles:
   `@tool(name="bundle.fn", summary=..., tags=...)` with **lazy heavy
   imports** — importing `tools.py` must never import torch (test-enforced).
3. Declare one pip extra named after the bundle in the registry's
   `pyproject.toml` (`[]` when no deps) and run `uv lock`.
4. Test CPU-only with `gap.testing.FakeContext` (canned tool responses,
   call assertions) and `make_test_observation` (synthetic RGB-D);
   GPU/LLM smokes go behind the `gpu`/`llm` markers (deselected by
   default).
5. Verify: `gap skills check && gap skills test my-skill && gap check`.

Full contract, test patterns, PR checklist: `references/authoring-bundles.md`.

## 6. Create / wire registries

```bash
gap registry init ~/my-lab-skills --name my-lab-skills --add   # new registry
gap registry add lab ~/existing-skills                         # layer existing
gap registry list                                              # inspect
```

Registries are local directories (clone remote ones yourself). Multiple
registries merge; earlier entries shadow same-named bundles (loud
warning) — fork-and-shadow one bundle instead of forking the whole
public registry. Projects pin their set in `pyproject.toml`
`[tool.gap].registries = ["./skills", "../open-robot-skills"]`.

## 7. Debug failures

- Validation errors cite W*/S* rule codes — fix the graph, re-run
  `--validate-only` (digest table in `references/authoring-graphs.md`).
- Runtime: read `dag_trace.json` in the trace dir (per-node
  inputs/outputs/subcalls), or browse with `gap viz`.
- Two runs disagree? `gap trace-diff <trace_a> <trace_b>`.
- Install/runtime gotchas (submodules, CUDA_HOME, EGL, numpy pin, keys,
  weights): `references/troubleshooting.md`. `gap check`'s fix hints are
  the first stop.

## 8. Real robots (human-gated)

Only after ALL of: the human explicitly confirmed this exact command in
this conversation; the same graph passed in sim; `--validate-only` is
clean; `--checkpoints raise`; the human has read `docs/safety.md`
(E-stop, clear workspace; `robot.go_home` is disabled on real robots by
design). Reference setups: `examples/real_franka_pick_place/`
(Franka + Robotiq via robots_realtime), `examples/cable_ur/`
(UR + ZED, perception-only).

## References

- `references/cli.md` — generated `--help` for every command
- `references/registries.md` — the registry model + config files
- `references/authoring-graphs.md` — builder API, workflow v3, W*/S* rules
- `references/authoring-bundles.md` — SKILL.md contract, tests, PR checklist
- `references/troubleshooting.md` — failure → fix table, trace anatomy
- In a checkout: `docs/design.md`, `docs/runtime.md`, `docs/skills.md`,
  `docs/safety.md`, `examples/*/README.md`
