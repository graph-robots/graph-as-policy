# Authoring skill-registry bundles

How to write, package, test, and contribute a bundle to a **skill
registry** — the canonical public one being
[open-robot-skills](../../open-robot-skills). Bundles use the
[Anthropic Agent Skills](https://agentskills.io/specification) directory
format (`SKILL.md` + resources) and are discovered **by path** — never by
pip entry-points.

---

## 0. Registries

A registry is any local directory with a `tools/` and/or `skills/`
bundle root. open-robot-skills is the example, not a special case:
scaffold your own with `gap registry init <path>`, layer it with
`gap registry add <name> <path>`, and inspect the active set with
`gap registry list`. Multiple registries are active simultaneously,
resolved in precedence order:

1. explicit `--skills PATH` flags / `skills=` arguments (repeatable —
   full override);
2. `$GAP_SKILLS_PATH` — an OS-pathsep-separated list (a single path
   keeps its historical meaning) — also a full override;
3. the nearest `pyproject.toml` declaring
   `[tool.gap].registries = ["./skills", "../open-robot-skills"]`
   (paths relative to that file; walking up from the cwd);
4. user config `~/.config/gap/registries.toml`, managed by
   `gap registry add/remove`;
5. an `open-robot-skills` checkout next to the GaP checkout
   (auto-discovered — the documented side-by-side layout).

The merged catalog is the union; a bundle name claimed by a
higher-precedence registry **shadows** same-named bundles below it
(first wins, loud warning) — the supported way to override one public
bundle with a lab fork. Within one registry a duplicate name is a hard
error. Registries are local paths in this release; clone remote ones
yourself before `gap registry add`.

---

## 1. Tools vs skills — the two bundle kinds

The folder a bundle lives in conveys its kind; both use the same format.

| | `tools/<bundle>/` | `skills/<bundle>/` |
|---|---|---|
| **What it is** | *what the robot can compute*: model-backed callables with no task strategy | *what the robot can do*: a manipulation strategy that owns subgraphs in generated graphs |
| **Naming** | named after the **model** (`sam3`, `grounding-dino`, `gemini-er`, `molmo`, `vlm`, `curobo`, `geometry`) | named after the **capability** (`perceiving-objects`, `grasping-direct-ik`, `transporting-objects`, …) |
| **Exposes** | typed functions via `@tool` in `tools.py` (`sam3.segment_box`, `curobo.plan_to_pose`) | LLM guidance (SKILL.md body) + canonical scripts under `scripts/`; *may* also expose a callable via `tools.py`/`skill.py` when invocable as a single unit (`pi05-libero`, `molmoact-libero`, `tracking-objects`) |
| **Appears in graphs as** | `type: tool` nodes (and `ctx.tool(...)` calls inside scripts) | the `skill:` field of a subgraph; its scripts as `type: script` nodes |
| **LLM context** | flat tool catalog (name + summary + typed schema) | the coordinator sees name+description; the subgraph agent gets the full SKILL.md plus the schemas of its `gap.allowed_tools` |

Tool names come from exactly two registries: **connector tools**
(`robot.*` / `sim.*` — shipped by GaP core, reserved prefixes) and **bundle
tools** (`<bundle>.<func>`). The loader rejects collisions. All skills are
flat — there is no atomic/composite distinction.

## 2. Repo layout

```
open-robot-skills/
├── pyproject.toml            # ONE distribution; each bundle = one extra (§6)
├── .claude-plugin/marketplace.json
├── tests/                    # repo-level + per-bundle tests (CPU by default)
├── tools/
│   └── <bundle>/
│       ├── SKILL.md          # frontmatter + body (required)
│       ├── tools.py          # @tool functions (lazy model loading, §5)
│       └── _impl.py …        # private helpers (underscore = not discovered)
└── skills/
    └── <bundle>/
        ├── SKILL.md
        ├── scripts/          # canonical scripts (type: script states)
        ├── prompts/          # templates loaded via load_prompt at runtime
        ├── references/       # long-form rationale, lazy-loaded by the agent
        └── examples/         # example subgraphs / invocations
```

One bundle = one directory = one pyproject extra = one PR.

## 3. SKILL.md format

Frontmatter is YAML between `---` delimiters; everything after the closing
`---` is the **body**, which is handed verbatim to the subgraph agent (for
skills) or shown as tool-bundle documentation. The frontmatter follows the
Agent Skills spec at the top level, with **all GaP extensions nested under
one `gap:` key** so spec fields are never overloaded.

### 3.1 Spec-core fields (top level)

| Field | Required | Constraint (enforced by the loader) |
|---|---|---|
| `name` | yes | must equal the bundle directory name; ≤64 chars; lowercase letters/digits with single hyphens |
| `description` | yes | ≤1024 chars; third-person, ends with a "Use when …" sentence — this is the coordinator's entire view of the bundle |
| `license` | no | SPDX-ish string |
| `compatibility` | no | e.g. `requires gap>=0.1`; the loader warns when the installed GaP doesn't satisfy it |
| `metadata` | no | free-form mapping; the loader reads `metadata.category` and `metadata.tags` for catalog grouping |

Legacy keys are hard errors with migration hints: `runtime`/`shape`/
`composes`/`category`/`tags`/`contract` at the top level are removed (the
tools-vs-skills folder split replaces `shape`; `category`/`tags` move under
`metadata:`), and any GaP-extension key found at the top level instead of
under `gap:` is rejected.

### 3.2 The `gap:` extension block — every field the loader consumes

| Field | Used by | Meaning |
|---|---|---|
| `allowed_tools` | prompt assembler | list of flat tool names this skill's scripts may call; the subgraph agent receives exactly these schemas |
| `exit_conditions` | coordinator + validator | `{exit_name: meaning}`; the coordinator wires conditional edges against these, and generated subgraphs must declare exactly this set (`success_values ∪ {on_error}`) |
| `produces_outputs` | coordinator | `{output_name: type_name}` the skill expects callers to bind; names may contain `<name>` for substitution (e.g. `"<name>_obb": OrientedBoundingBox`) |
| `required_inputs` | coordinator | `{input_name: type_name}` the subgraph must declare and bind from upstream |
| `canonical_scripts` | subgraph agent + registry | list of `- logical_name: scripts/file.py`; discovered with full typed schemas so the agent can emit them as `type: script` states |
| `prompts` | `load_prompt` | `{logical_name: prompts/file.md}` templates loaded at script runtime |
| `references` | codegen agent | list of `{title, path}` long-form docs, lazy-loaded on demand |
| `examples` | codegen agent | list of `{title, path}` example invocations/subgraphs |
| `errors` | prompt assembler | list of error strings the skill can raise (e.g. `"NOT_FOUND: …"`) |
| `tips` | prompt assembler | free-form authoring tips string |
| `hard_rules` | prompt assembler | inline rules or anchored refs into the bundle's own `references/` (e.g. `perception_pipeline_invariants.md#emit-both-obb-and-mask`) |
| `streaming` | validator (rule S4) | `true` iff the bundle's callable streams via `ctx.publish`; checked against `streaming: true` nodes |
| `tools` | catalog docs | **tool bundles**: list of `- name: one-line summary` for each `@tool` in `tools.py` (documentation; authoritative schemas come from the registry) |
| `requires` | `gap check`, `gap skills check` | operational requirements: `{gpu: true, env: [VARS…], env_any: [A, B], weights: true, connector: [motion.plan_joint, …]}` (all keys optional; unknown keys rejected). `requires: {}` = explicitly nothing. `gap check` derives per-bundle readiness — and per-skill runnability — from this. `connector` lists flat tool names a richer run-time connector registers (beyond the generic `robot.*`/`sim.*` and beyond any bundle's `gap.tools`): they count as resolvable in `allowed_tools`, and `gap check` reports the skill as READY "— needs a connector providing …" rather than blocking on what it cannot probe. Mandatory for tool bundles in the canonical registry (test-enforced) |

`params`/`outputs` are **not** frontmatter — they come from Python
introspection of the bundle's callables and scripts.

Bundles that download weights may also define a **filesystem-only**
`weights_cached() -> bool | None` next to `prefetch()` in `tools.py` —
`gap check` reports the cache state without downloading; absent hook →
"unknown".

### 3.3 Example (skill bundle)

```yaml
---
name: perceiving-objects
description: Fast single-path 3D object perception. … Use when a
  manipulation workflow needs to localize one named object quickly.
license: MIT
compatibility: requires gap>=0.1
metadata: {category: perception, tags: [perception, dino, vlm, sam3]}
gap:
  allowed_tools:
    - robot.get_observation
    - grounding-dino.detect
    - vlm.query
    - sam3.segment_box
    - geometry.mask_to_world_points
    - geometry.filter_and_compute_obb
  exit_conditions:
    found: Target detected; OBB and mask bound in subgraph outputs.
    not_found: Target not visible in any view.
  produces_outputs:
    "<name>_obb": OrientedBoundingBox
    "<name>_mask": Mask
  canonical_scripts:
    - perceive_dino_vlm: scripts/perceive_dino_vlm.py
  prompts: {vlm_pairwise: prompts/vlm_pairwise.md}
  references:
    - title: Perception pipeline invariants
      path: references/perception_pipeline_invariants.md
  streaming: false
---
```

Tool bundles use the same shape with `gap.tools:` instead of
`exit_conditions`/`canonical_scripts`.

## 4. The authoring contract (stable import surface)

Bundle code imports **only** from these modules; everything else in GaP is
internal and may change without notice:

```python
from gap import NodeContext, CancelToken
from gap.types import (Se3Pose, OrientedBoundingBox, Mask, PointCloud,
                       Observation, CameraFrame, Trajectory, ...)
from gap.errors import (PipelineError, PerceptionFailed, PlanningFailed,
                        GraspFailed, ValidationFailed, ToolError)
from gap.skills import tool, Skill, SkillMeta, load_prompt
from gap.testing import FakeContext, make_test_observation   # tests only
```

- **Scripts** define `run(ctx: NodeContext, ...) -> Output` with full type
  annotations (`Output` a TypedDict). They call tools exclusively through
  `ctx.tool(name, **kwargs)` and raise `PipelineError` subclasses on
  failure.
- **Tool functions** are plain typed functions decorated with
  `@tool(name="<bundle>.<func>", summary="...", tags=(...))`. Tags drive
  the guard limits (`perception` / `planning` / `sim_step`).
- **Stateful callable skills** subclass `gap.skills.Skill` and define
  `run(self, ctx, ...)`; the runtime keeps one instance per workflow
  execution (state persists across visits, discarded at the end).
- **Prompts**: `load_prompt(__package__, "vlm_pairwise", n=3, labels=...)`
  renders `prompts/vlm_pairwise.md` with `{{ var }}` substitutions and
  `{% if var %}…{% endif %}` blocks. This works because the registry loads
  canonical scripts under a synthetic package rooted at the bundle dir
  (the loader walks up to `SKILL.md`). Ad-hoc LLM-emitted scripts have no
  bundle and cannot use it — by design.
- **Streaming skills** declare `gap.streaming: true`, loop with
  `ctx.publish(snapshot)` per iteration, and check
  `ctx.cancel_token.raise_if_set()` so teardown drains promptly.

## 5. Lazy model loading (mandatory for tool bundles)

Importing a bundle's `tools.py` must **never** pull torch / transformers /
CUDA — the repo-level test suite asserts this. The pattern, used by every
shipped bundle:

```python
_load_lock = threading.Lock()
_model = None

def _get_model():
    global _model
    with _load_lock:
        if _model is None:
            from transformers import AutoModel        # import INSIDE the loader
            _model = AutoModel.from_pretrained(os.environ.get("GAP_X_MODEL", DEFAULT))
            _model = _model.to(os.environ.get("GAP_X_DEVICE", "cuda")).eval()
    return _model

@tool(name="my-model.detect", summary="...", tags=("perception",))
def detect(rgb: np.ndarray, text: str) -> DetectResult:
    model = _get_model()                              # weights load on FIRST CALL
    ...
```

Per-bundle device/model knobs are environment variables with a `GAP_`
prefix, documented in the SKILL.md body.

## 6. Dependencies: one pip extra per bundle

open-robot-skills is a single distribution whose wheel ships no code — bundles are
read from the checkout by path. The pyproject exists for the **dependency
mechanism**: each bundle declares its deps as an extra whose name equals the
bundle name, so one resolver run surfaces cross-bundle conflicts:

```toml
[project.optional-dependencies]
my-model = ["torch>=2.7", "some-model @ git+https://github.com/...@<sha>"]
```

```bash
uv sync --extra my-model      # the install line your SKILL.md shows
                              # (pip: pip install -e "open-robot-skills[my-model]")
```

Run `uv lock` after editing the extras so the committed lockfile stays in
sync.

Bundles install one at a time (`gap skills install <bundle>`, or `--all`); there are no meta-extras. Non-PyPI deps are
pinned `git+https` entries inside the extra. **uv/pip own code installs;
`gap skills check` only verifies** (import probe + weight presence per
bundle), and `gap skills check --download` prefetches **model weights
only** (set `HF_TOKEN` for gated repos). Nothing ever installs behind
the user's back.

## 7. Testing with `gap.testing`

`gap.testing` exports the same fakes GaP's own suite uses, so a bundle is
unit-testable without a robot, a GPU, or an LLM:

```python
from gap.errors import PerceptionFailed
from gap.testing import FakeContext, make_test_observation

def test_perceive_finds_target():
    obs, ground_truth = make_test_observation(
        objects=[("cube", (0.0, 0.0, 0.03), (0.06, 0.06, 0.06))])
    ctx = FakeContext(tool_responses={
        "robot.get_observation": obs,
        "grounding-dino.detect": {"detections": [{"box": [10, 10, 60, 60],
                                                  "label": "cube", "score": 0.9}]},
        "vlm.query": ["A"],                     # list = scripted answers, pop in order
        "sam3.segment_box": {"mask": ground_truth["cube"]["mask"]},
    })
    out = perceive_dino_vlm.run(ctx, object_name="cube")
    assert ctx.call_count("grounding-dino.detect") == 1
    assert out["obb"]["extent"][2] > 0

def test_perceive_not_found():
    def _no_detections(**kwargs):
        raise PerceptionFailed("nothing detected")   # exercise the error path
    ctx = FakeContext(tool_responses={"grounding-dino.detect": _no_detections})
    ...
```

- `FakeContext(tool_responses={name: value | callable | [values]})` — a
  `NodeContext` stand-in. Plain values return on every call; callables are
  invoked with the call kwargs (raise inside one to exercise error paths);
  lists pop front-to-back. Unscripted tools raise `ToolError` so tests fail
  loudly. The full call log lands in `ctx.calls`
  (plus `ctx.calls_to(name)` / `ctx.call_count(name)` helpers), and
  `ctx.published` collects streaming publishes.
- `make_test_observation(objects=[(name, center_xyz, size_xyz)], ...)` —
  returns `(observation, ground_truth)` with geometrically *consistent*
  synthetic data: RGB with rendered boxes plus depth/intrinsics/camera pose
  that reproject correctly, so perception math (mask → points → OBB) is
  tested on real numerics, not mocks.
- `assert_graph_valid(graph)` — loader + structural validation for example
  subgraphs.

Mark anything that loads weights with `@pytest.mark.gpu` and anything that
hits an LLM API with `@pytest.mark.llm` — the default suite must stay green
on a CPU-only machine (`pytest tests -q` deselects both).

## 8. Scaffolding a new bundle

```bash
gap skills new my-model --kind tool       # targets the highest-precedence registry
gap skills new doing-things --kind skill --registry my-lab-skills
```

This creates the bundle directory with a TODO-annotated `SKILL.md` and
either a `tools.py` stub (tool) or `scripts/example.py` + `prompts/` +
`references/` (skill), **plus a unit-test skeleton**
`tests/test_<name>.py` in the owning registry (a `tests/conftest.py`
with the standard session fixtures is created when missing).
`gap skills list` shows the bundle immediately;
`gap skills test <name>` runs its tests. No registry configured? Create
one first: `gap registry init <path> --add`.

## 8.1 Verifying: `gap skills check`, `gap skills test`, `gap check`

```bash
gap skills check          # per-bundle PASS/WARN/FAIL; non-zero exit on FAIL
gap skills test my-model  # the bundle's unit tests, run from its registry
gap skills test           # every active registry's full tests/ suite
gap check                 # capability report: can each bundle run HERE?
gap skills table --format markdown   # catalog table, paste-ready for READMEs
```

`gap skills check` runs two layers per bundle: **format validation** (the
rules live engine-side in `gap.skills.validate`, so third-party
registries get the same checker the open-robot-skills test suite
enforces — frontmatter shape per kind, every referenced
`canonical_scripts`/`prompts`/`references`/`examples` path exists,
`gap.allowed_tools` resolve against connector tools + all declared bundle
tools, `produces_outputs`/`required_inputs` type names, the
one-pip-extra-per-bundle convention, and `gap.requires` consistency) and
an **import probe** (each bundle registered individually; ImportErrors
map to the owning registry's install line — `uv sync --extra <bundle>`
or `pip install "<dist>[<bundle>]"`). `--download` additionally runs each
bundle's optional `prefetch()` to fetch model weights.

`gap skills test` invokes pytest with the owning registry's own config
(cwd at the registry root, `sys.executable -m pytest`): exact file
`tests/test_<name with - → _>.py` when present, `-k` fallback otherwise;
pytest args pass through after `--`
(`gap skills test sam3 -- -m gpu -x`). Registries with their own
separate venv should run `uv run pytest` there instead.

`gap check` answers the *operational* question — deps importable,
declared `gap.requires` met (GPU via nvidia-smi, env vars), weights
cached — and rolls it up per skill (a skill is blocked by exactly the
not-ready bundles owning its `allowed_tools`; `robot.*`/`sim.*` are
connector-satisfied). `gap skills check` = "is the bundle well-formed";
`gap check` = "can it run here".

## 9. Contribution checklist

Before opening the PR (one bundle per PR):

- [ ] `SKILL.md` frontmatter passes the loader: `name` == directory name,
      description ≤1024 chars ending in "Use when …", GaP extensions under
      `gap:`.
- [ ] Skills declare `gap.exit_conditions` (+ `produces_outputs` /
      `required_inputs` as applicable); tool bundles declare `gap.tools`
      with one line per exposed function.
- [ ] `gap.requires` declared (`{}` when the bundle needs nothing);
      gpu-tagged bundles say `gpu: true`; weight-downloading bundles
      consider a `weights_cached()` hook. `gap check` shows the bundle
      READY (or honestly NOT READY with the right hints).
- [ ] Every path referenced in frontmatter
      (`canonical_scripts`/`prompts`/`references`/`examples`) exists.
- [ ] `tools.py` imports clean without torch/transformers in `sys.modules`
      (lazy loading, §5); model/device knobs are `GAP_*` env vars
      documented in the body.
- [ ] Dependencies are one extra in `pyproject.toml` (extra name == bundle
      name); the SKILL.md body shows the `gap skills install <bundle>` install
      line; `uv lock` is updated and the bundle's own environment still resolves.
- [ ] Tool names are `<bundle>.<func>` — the `robot.*`/`sim.*` prefixes are
      reserved and will be rejected.
- [ ] Unit tests on `FakeContext`/`make_test_observation` cover the happy
      path and at least one failure exit; GPU/LLM paths are marked.
- [ ] `pytest tests -q` (open-robot-skills) and `pytest tests -q` (gap) pass;
      `ruff check` is clean.
