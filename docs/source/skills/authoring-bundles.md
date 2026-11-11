# Authoring Skill Bundles

This guide shows how to write and package a bundle for a skill registry —
the canonical public one being
[open-robot-skills](gh-skills:README.md). Bundles use the
[Anthropic Agent Skills](https://agentskills.io/specification) directory
format (`SKILL.md` + resources) and are discovered **by path** — never by
pip entry-points. For how registries are resolved and layered, see
[Skill Registries](registries.md); for testing your bundle, see
[Testing Skill Bundles](testing-bundles.md).

:::{note}
**What is `gap_core`?** As of the `gap-core` / `gap-runtime` workspace
split, bundle authors depend only on **`graph-as-policy-core`** (~150 MB)
— a slim package with no fastapi, JAX, or MuJoCo. The stable authoring
surface (`@tool`, types, errors, `Skill`/`SkillMeta`/`Param`/`Serving`)
lives under `gap_core.*`; the heavy runtime (`gap.execute`,
`gap.connector`, `gap.agent`) stays on the full `graph-as-policy`
distribution. Your bundle's `pyproject.toml` should depend on
`graph-as-policy-core` only.
:::

## Tools vs skills — the two bundle kinds

The folder a bundle lives in conveys its kind (`tools/` vs `skills/` —
kind is never declared in frontmatter); both use the same `SKILL.md`
format.

| | `tools/<bundle>/` | `skills/<bundle>/` |
|---|---|---|
| **What it is** | *what the robot can compute*: model-backed callables with no task strategy | *what the robot can do*: a manipulation strategy that owns subgraphs in generated graphs |
| **Naming** | named after the **model** (`sam3`, `grounding-dino`, `gemini-er`, `molmo`, `vlm`, `curobo`, `geometry`) | named after the **capability** (`perceiving-objects`, `grasping-direct-ik`, `transporting-objects`, …) |
| **Exposes** | typed functions via `@tool` in `tools.py` (`sam3.segment_box`, `curobo.plan_to_pose`) | LLM guidance (SKILL.md body) + canonical scripts under `scripts/`; *may* also expose a callable via `tools.py`/`skill.py` when invocable as a single unit (`pi05-libero`, `molmoact-libero`, `tracking-objects`) |
| **Appears in graphs as** | `type: tool` nodes (and `ctx.tool(...)` calls inside scripts) | the `skill:` field of a subgraph; its scripts as `type: script` nodes |
| **LLM context** | flat tool catalog (name + summary + typed schema) | the coordinator sees name + description; the subgraph agent gets the full SKILL.md plus the schemas of its `gap.allowed_tools` |

Tool names come from exactly two sources: **connector tools**
(`robot.*` / `sim.*` — shipped by GaP core; these prefixes are reserved
and `@tool` rejects them — see [Connector Tools](../reference/connector-tools.md))
and **bundle tools** (`<bundle>.<func>`). The loader rejects name
collisions. All skills are flat — there is no atomic/composite
distinction. Browse what already exists in the
[skill catalog](skill-catalog.md) and [tool catalog](tool-catalog.md).

## Repo layout

```text
open-robot-skills/
├── pyproject.toml            # ONE distribution; each bundle = one extra
├── .claude-plugin/marketplace.json
├── tests/                    # repo-level + per-bundle tests (CPU by default)
├── tools/
│   └── <bundle>/
│       ├── SKILL.md          # frontmatter + body (required)
│       ├── tools.py          # @tool functions (lazy model loading)
│       └── _impl.py …        # private helpers (underscore = not discovered)
└── skills/
    └── <bundle>/
        ├── SKILL.md
        ├── scripts/          # canonical scripts (type: script states)
        ├── prompts/          # templates loaded via load_prompt at runtime
        ├── references/       # long-form rationale, lazy-loaded by the agent
        └── examples/         # example subgraphs / invocations
```

One bundle = one directory = one pyproject extra = one PR. Directories
without a `SKILL.md` and underscore-prefixed directories are skipped by
discovery.

## SKILL.md format

Frontmatter is YAML between `---` delimiters; everything after the closing
`---` is the **body**, which is handed verbatim to the subgraph agent (for
skills) or shown as tool-bundle documentation. The frontmatter follows the
Agent Skills spec at the top level, with **all GaP extensions nested under
one `gap:` key** so spec fields are never overloaded.

### Spec-core fields (top level)

| Field | Required | Constraint (enforced by the loader) |
|---|---|---|
| `name` | yes | must equal the bundle directory name; ≤64 chars; lowercase letters/digits with single hyphens (`^[a-z0-9]+(?:-[a-z0-9]+)*$`) |
| `description` | yes | ≤1024 chars; third-person, with a "Use when …" sentence — this is the coordinator's entire view of the bundle. A missing usage cue is a validation warning engine-side (the bundle reports WARN in `gap skills check`) |
| `license` | no | SPDX-ish string |
| `compatibility` | no | e.g. `requires gap>=0.1`; the loader warns (never errors) when the installed GaP doesn't satisfy it |
| `metadata` | no | free-form mapping; the loader reads `metadata.category` and `metadata.tags` for catalog grouping |

Legacy keys are hard errors with migration hints: `runtime`, `shape`,
`composes`, `category`, `tags`, and `contract` at the top level are
removed (the tools-vs-skills folder split replaces `shape`;
`category`/`tags` move under `metadata:`), and any GaP-extension key
found at the top level instead of under `gap:` is rejected.

### The `gap:` extension block — every field the loader consumes

| Field | Used by | Meaning |
|---|---|---|
| `allowed_tools` | prompt assembler | list of flat tool names this skill's scripts may call; the subgraph agent receives exactly these schemas |
| `exit_conditions` | coordinator + validator | `{exit_name: meaning}`; the coordinator wires conditional edges against these, and generated subgraphs must declare exactly this set |
| `produces_outputs` | coordinator | `{output_name: type_name}` the skill expects callers to bind; names may contain `<name>` for substitution (e.g. `"<name>_obb": OrientedBoundingBox`) |
| `required_inputs` | coordinator | `{input_name: type_name}` the subgraph must declare and bind from upstream |
| `canonical_scripts` | subgraph agent + registry | list of `- logical_name: scripts/file.py`; discovered with full typed schemas so the agent can emit them as `type: script` states |
| `prompts` | `load_prompt` | `{logical_name: prompts/file.md}` templates loaded at script runtime |
| `references` | codegen agent | list of `{title, path}` entries (or bare path strings) — long-form docs, lazy-loaded on demand |
| `examples` | codegen agent | list of `{title, path}` entries (or bare path strings) — example invocations/subgraphs |
| `errors` | prompt assembler | list of error strings the skill can raise (e.g. `"NOT_FOUND: …"`) |
| `tips` | prompt assembler | free-form authoring tips |
| `hard_rules` | prompt assembler | inline rules or anchored refs into the bundle's own `references/` (e.g. `perception_pipeline_invariants.md#emit-both-obb-and-mask`) |
| `streaming` | validator | `true` iff the bundle's callable streams via `ctx.publish`; checked against `streaming: true` nodes in graphs |
| `tools` | catalog docs | **tool bundles**: list of `- name: one-line summary` for each `@tool` in `tools.py` (documentation; authoritative schemas come from the registry). Names must be namespaced `<bundle>.<func>` |
| `requires` | `gap check` | operational requirements: `{gpu: true, env: [VARS…], env_any: [A, B], weights: true}` — exactly these four keys, all optional; unknown keys are rejected (a typo would silently disable probes). `requires: {}` = explicitly nothing |

`params`/`outputs` are **not** frontmatter — they come from Python
introspection of the bundle's callables and scripts.

:::{important}
In open-robot-skills, `gap.requires` is **mandatory on every tool
bundle** — declare `requires: {}` when the bundle genuinely needs
nothing. A tool bundle without the block has `meta.requires is None` and
silently reports as unconditionally ready in `gap check`; the registry's
[test suite](gh-skills:tests/test_requires_declared.py) rejects that, and
also requires gpu-tagged bundles to declare `requires: {gpu: true}`.
:::

### Example (skill bundle)

Abridged from the real
[skills/perceiving-objects](gh-skills:skills/perceiving-objects):

```yaml
---
name: perceiving-objects
description: >
  Fast single-path 3D object perception. Runs Grounding-DINO broad
  detection, a pairwise VLM crop tournament to identify the target box,
  SAM3 box segmentation, and depth back-projection to a world-frame point
  cloud ... Use when a manipulation workflow needs to localize one named
  object quickly.
license: MIT
compatibility: requires gap>=0.1
metadata:
  category: perception
  tags: [perception, dino, vlm, sam3, single-method]
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
  prompts:
    vlm_pairwise: prompts/vlm_pairwise.md
---
```

Tool bundles use the same shape with `gap.tools:` instead of
`exit_conditions`/`canonical_scripts`.

## Authoring a policy bundle

A third bundle kind lives under `<registry>/policies/<name>/` and wraps
a learned policy server (VLA, diffusion, IL) behind the connector. Pick
this kind when the deliverable is *weights + a serving entrypoint*
rather than scripts or `@tool` callables. The SKILL.md uses the usual
shape with a `gap.serving:` block declaring how to launch the server:

```yaml
gap:
  serving:
    command: "uv run python -m my_policy.serve --port {port}"
    protocol: websocket            # or stdio-msgpack
    env: [HF_TOKEN, GAP_DEVICE]
    requires_gpu: true
    weights_uri: "hf://my-org/my-policy-libero@v0.3"
```

`gap.execute` spawns the command, dials the protocol, and routes
observations/actions through the connector. See
[policies/pi05-libero](gh-skills:policies/pi05-libero) and
[policies/molmoact-libero](gh-skills:policies/molmoact-libero) in
open-robot-skills for working examples.

## The authoring contract (stable import surface)

Bundle code imports **only** from these modules; everything else in GaP is
internal and may change without notice (see the [API reference](../reference/api.md)):

```python
from gap import NodeContext, CancelToken
from gap_core.types import (Se3Pose, OrientedBoundingBox, Mask, PointCloud,
                            Observation, CameraFrame, Trajectory)
from gap_core.errors import (PipelineError, PerceptionFailed, PlanningFailed,
                             GraspFailed, ValidationFailed, ToolError)
from gap_core.tools import tool
from gap_core.skills import Skill, SkillMeta, load_prompt
from gap.testing import FakeContext, make_test_observation   # tests only
```

- **Scripts** define `run(ctx: NodeContext, ...) -> Output` with full type
  annotations (`Output` a TypedDict). They call tools exclusively through
  `ctx.tool(name, **kwargs)` and raise `PipelineError` subclasses on
  failure.
- **Tool functions** are plain typed functions decorated with
  `@tool(name="<bundle>.<func>", summary="...", tags=(...))`. Tags drive
  the call-count guard limits: a `perception` / `planning` / `sim_step`
  tag puts the tool under the corresponding
  `GAP_MAX_PERCEPTION_CALLS` / `GAP_MAX_PLANNING_CALLS` / `GAP_MAX_SIM_STEPS`
  budget (see [Environment Variables](../reference/environment-variables.md)).
- **Stateful callable skills** subclass `gap_core.skills.Skill` and define
  `run(self, ctx, ...)`. The runtime keeps one instance per skill per
  workflow execution — instance attributes persist across visits to the
  state and are discarded when the execution ends. Function-style
  `run(ctx, ...)` callables work unchanged; subclass only for genuine
  state needs (replan caches, trackers accumulating evidence). At most
  one `Skill` subclass per module — two is a `ValueError`.
- **Prompts**: `load_prompt(__package__, "vlm_pairwise", n=3, labels=...)`
  renders `prompts/vlm_pairwise.md` with `{{ var }}` substitutions and
  non-nested `{% if var %}…{% endif %}` blocks (plain regex substitution;
  no jinja dependency — an undefined `{{ var }}` raises `KeyError`). This
  works because the registry loads canonical scripts under a synthetic
  package rooted at the bundle dir (the loader walks up to `SKILL.md`).
  Ad-hoc LLM-emitted scripts have no bundle and cannot use it — by
  design.
- **Streaming skills** declare `gap.streaming: true`, loop with
  `ctx.publish(snapshot)` per iteration, and check
  `ctx.cancel_token.raise_if_set()` so teardown drains promptly.

## Lazy model loading (mandatory for tool bundles)

Importing a bundle's `tools.py` must **never** pull torch / transformers /
CUDA — discovery imports every bundle, and `gap check` relies on imports
being cheap. The open-robot-skills suite
[asserts](gh-skills:tests/test_bundles_load.py) that after loading every
`tools.py`, none of `torch`, `transformers`, `curobo`, `sam3` is in
`sys.modules`. The pattern, used by every shipped bundle:

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

## Dependencies: one pip extra per bundle

open-robot-skills is a single distribution whose wheel ships no code —
bundles are read from the checkout by path. The pyproject exists for the
**dependency mechanism**: each bundle declares its deps as an extra whose
name equals the bundle name (an empty list `[]` when it has none), so one
resolver run surfaces cross-bundle conflicts:

```toml
[project.optional-dependencies]
my-model = ["torch>=2.7", "some-model @ git+https://github.com/...@<sha>"]
```

```bash
uv sync --extra my-model      # the install line your SKILL.md shows
                              # (pip: pip install -e "open-robot-skills[my-model]")
```

Run `uv lock` after editing the extras so the committed lockfile stays in
sync. Bundles install one at a time (`gap skills install <bundle>`, or `--all`); there are no meta-extras. A missing extra is
a warning in `gap skills check` and a hard test failure in
open-robot-skills.

**uv/pip own code installs; `gap skills check` only verifies.** Nothing
ever installs behind the user's back.

## Weights: `weights_cached()` and `prefetch()` hooks

Bundles that download model weights declare `requires: {weights: true}`
and may define two optional module-level hooks in `tools.py` (or
`skill.py`):

- `weights_cached() -> bool | None` — a **filesystem-only** probe `gap
  check` calls to report cache state without downloading anything and
  without importing torch. Return `True`/`False`, or `None` for
  "unknown"; absent hook → "unknown". See
  [tools/grounding-dino/tools.py](gh-skills:tools/grounding-dino/tools.py)
  for a real one (checks the Hugging Face cache for the model's
  `config.json`).
- `prefetch()` — an explicit download step. `gap skills check --download`
  runs each bundle's `prefetch()` after the format checks; bundles
  without the hook are reported as `declares no weights (no prefetch())`,
  and a hook that raises marks the run failed.

:::{note}
No bundle currently shipped in open-robot-skills defines `prefetch()`,
so today weights download lazily at the first model call rather than
during `--download`. Missing weights never fail `gap check` — the cache
state is an informational line and the bundle stays READY. Set
`HF_TOKEN` before the first call for gated Hugging Face repos.
:::

## Scaffolding a new bundle

```bash
gap skills new my-model --kind tool       # targets the highest-precedence registry
gap skills new doing-things --kind skill --registry my-lab-skills
```

This creates the bundle directory with a TODO-annotated `SKILL.md` and
either a `tools.py` stub (tool) or `scripts/example.py` + `prompts/` +
`references/` (skill), **plus a unit-test skeleton**
`tests/test_<name_with_underscores>.py` in the owning registry (a
`tests/conftest.py` with the standard session fixtures is created when
missing; an existing test file is never overwritten). It refuses to
overwrite an existing bundle directory, and prints the next steps:
declare the pip extra, then `gap skills check`, then
`gap skills test <name>`.

No registry configured? `gap skills new` exits with a recipe — create one
first with `gap registry init <path> --add`, or pass a target with
`--skills <path>` (see [Skill Registries](registries.md)).

## Verifying

```bash
gap skills check          # per-bundle PASS/WARN/FAIL; non-zero exit on FAIL
gap skills test my-model  # the bundle's unit tests, run from its registry
gap check                 # capability report: can each bundle run HERE?
gap skills table --format markdown   # catalog table, paste-ready for READMEs
```

`gap skills check` runs two layers per bundle: **format validation** (the
rules live engine-side in [gap/skills/validate.py](gh-engine:gap/skills/validate.py),
so third-party registries get the same checker the open-robot-skills test
suite enforces — frontmatter shape per kind, every referenced
`canonical_scripts`/`prompts`/`references`/`examples` path exists,
`gap.allowed_tools` resolve against connector tools plus all declared
bundle tools, `produces_outputs`/`required_inputs` type names, the
one-pip-extra-per-bundle convention, and `gap.requires` consistency) and
an **import probe** (each bundle registered individually so one broken
bundle doesn't mask the rest; ImportErrors map to the owning registry's
install line — `uv sync --extra <bundle>` or
`pip install "<dist>[<bundle>]"`).

`gap check` answers the *operational* question — deps importable,
declared `gap.requires` met (GPU via nvidia-smi, env vars), weights
cached — and rolls it up per skill (a skill is blocked by exactly the
not-ready bundles owning its `allowed_tools`; `robot.*`/`sim.*` are
connector-satisfied). In short: `gap skills check` = "is the bundle
well-formed"; `gap check` = "can it run here". See the
[CLI reference](../reference/cli.md) for all flags, and
[Testing Skill Bundles](testing-bundles.md) for `gap skills test`.

## Contribution checklist

Before opening the PR (one bundle = one directory = one PR):

- [ ] `SKILL.md` frontmatter passes the loader: `name` == directory name,
      description ≤1024 chars with a "Use when …" sentence, GaP
      extensions under `gap:`.
- [ ] Skills declare `gap.exit_conditions` (+ `produces_outputs` /
      `required_inputs` as applicable); tool bundles declare `gap.tools`
      with one line per exposed function.
- [ ] `gap.requires` declared (`{}` when the bundle needs nothing);
      gpu-tagged bundles say `gpu: true`; weight-downloading bundles
      consider a `weights_cached()` hook. `gap check` shows the bundle
      READY (or honestly NOT READY with the right hints).
- [ ] Every path referenced in frontmatter
      (`canonical_scripts`/`prompts`/`references`/`examples`) exists.
- [ ] `tools.py` imports clean without torch/transformers in
      `sys.modules` (lazy loading); model/device knobs are `GAP_*` env
      vars documented in the body.
- [ ] Dependencies are one extra in `pyproject.toml` (extra name ==
      bundle name); the SKILL.md body shows the
      `gap skills install <bundle>` install line; `uv lock` is updated and
      the bundle's own environment still resolves.
- [ ] Tool names are `<bundle>.<func>` — the `robot.*`/`sim.*` prefixes
      are reserved and will be rejected.
- [ ] Unit tests on `FakeContext`/`make_test_observation` cover the happy
      path and at least one failure exit; GPU/LLM paths are marked (see
      [Testing Skill Bundles](testing-bundles.md)).
- [ ] `gap skills check`, `gap skills test <bundle>`, and the registry's
      full `pytest tests -q` pass; `ruff check` is clean.
