# Agent Quickstart

:::{note}
**Requirements:** same as the [LIBERO Quickstart](libero-quickstart.md) — a
GPU, the `quickstart` extra, and an LLM API key — plus
[Claude Code](https://claude.com/claude-code) (or another coding agent).
Budget about 10 minutes.
:::

This walkthrough has Claude Code, armed with one skill, drive GaP
end-to-end: it generates a pick-and-place graph from a single sentence,
validates it, and the graph then runs in LIBERO sim. Every command and
output below comes from a real recorded session — **the generated graph put
the cream cheese in the basket in 75 s, on video.**

Two agent layers are involved — don't conflate them:

```text
Claude Code ──(the `gap` skill)──▶ drives the CLI: gap check / generate / run / skills …
                                         │
                                         ▼
                            gap's own codegen agent (`gap generate`)
                                         │  reads bundle SKILL.mds from disk
                                         ▼
                       skill registries (e.g. ../open-robot-skills)
```

Claude Code needs **only the `gap` skill**. The robot bundles in
[open-robot-skills](gh-skills:.) are consumed by GaP's *internal* codegen
agent as files on disk — they do not need to be installed into Claude Code.

## 0. Prerequisites (once)

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # side-by-side
cd graph-as-policy
uv sync && uv run gap skills install sam3 grounding-dino geometry            # engine + sim + perception bundles
uv run gap skills check --download    # validate every bundle
```

Model weights download from HuggingFace at the first model call;
`--download` additionally runs each bundle's optional `prefetch()` hook
where a bundle defines one. See
[Installation](../getting-started/installation.md) for details.

Pick an LLM provider for generation and for the VLM perception bundle
(`gap check` tells you what is configured):

```bash
export OPENROUTER_API_KEY=...         # simplest: one key drives both
# or vertex: gcloud auth application-default login, then
#   export GAP_VLM_PROVIDER=vertex GAP_VLM_PROJECT_ID=<proj> \
#          GAP_VLM_REGION=global GAP_VLM_MODEL=gemini-3.1-pro-preview
#   and pass a config YAML with the same llm: settings to gap generate
```

Provider details: [LLM providers](../authoring/llm-providers.md) for
generation, [LIBERO Quickstart](libero-quickstart.md#vlm-provider) for the
perception VLM.

## 1. Install the agent skill (once)

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap
```

Verify in a new Claude Code session: ask *"What robot skills are
available?"* — `gap` should be listed. The skill content lives at
[agent/skills/gap](gh-engine:agent/skills/gap) in the engine repo; install
options for other agents (Cursor, Codex CLI, …) are in
[agent/INSTALL.md](gh-engine:agent/INSTALL.md). The full tour of what the
skill teaches the agent — including its safety gates and troubleshooting
playbook — is in [Claude Code](../agents/claude-code.md).

## 2. Ask Claude Code for a graph

Prompt used in the recorded session:

> Using your gap skill: generate a robot task graph for "pick up the
> cream cheese and put it in the basket" with output directory
> outputs/agent_gen_test. Then validate the generated graph without
> running any simulator or real robot.

What the agent does, all taught by the skill — you can run the same three
commands by hand:

```bash
uv run gap check                                   # capability report first
uv run gap generate "pick up the cream cheese and put it in the basket" \
    --out outputs/agent_gen_test                   # add --config <yaml> for vertex
uv run gap run outputs/agent_gen_test/task_00 --validate-only
```

Expected ending: `OK: 0 errors` and a workflow directory:

```text
outputs/agent_gen_test/task_00/
├── workflow.json      # 4 subgraphs: perceive_target, perceive_container,
│                      #   grasp_target (grasping-with-planner),
│                      #   place_target (transporting-objects)
├── scripts/           # the canonical skill scripts the graph calls
├── checkpoints/       # generated postcondition predicates
└── agent_traces/      # the codegen agents' own transcripts
```

:::{important}
Generated output lands in `<out>/task_00/` — point `gap run` at the task
subdirectory, not at `--out` itself.
:::

In the recorded session the agent's first `gap generate` hit a missing
vertex SDK — it read the error, ran `uv sync --inexact --extra vertex`, and
retried successfully. That self-repair is the skill's troubleshooting
playbook working as intended.

How the generation pipeline itself works (coordinator, per-subgraph agents,
validation loop) is covered in [Generation](../authoring/generation.md).

## 3. Run it in sim (with video)

The cream-cheese scene is task **1** of the `libero_object_all_variance`
suite (classic LIBERO numbering):

```bash
MUJOCO_GL=egl uv run gap run outputs/agent_gen_test/task_00 \
    --sim libero_object_all_variance/1 \
    --record-video --trace-dir outputs/agent_gen_sim
```

Recorded result:

```text
SUCCESS (exit=success, 75.0s)
trace: outputs/agent_gen_sim
checkpoint: ... name='target_held'      passed=True   ← gripper held the cream cheese
checkpoint: ... name='target_in_basket' passed=True   ← it ended up in the basket
video: outputs/agent_gen_sim/run_video.mp4 (506 frames)
```

Frames from the recorded run:

::::{grid} 2
:gutter: 2

:::{grid-item-card} Start — perceive the scene
```{image} ../_static/agent-run-start.png
:alt: Initial LIBERO scene with cream cheese and basket
```
:::

:::{grid-item-card} Grasp — planner trajectory to the cream cheese
```{image} ../_static/agent-run-grasp.png
:alt: The arm descending onto the cream cheese
```
:::

:::{grid-item-card} Transport — lift and carry over the basket
```{image} ../_static/agent-run-transport.png
:alt: The arm carrying the cream cheese toward the basket
```
:::

:::{grid-item-card} End — released in the basket, success
```{image} ../_static/agent-run-end.png
:alt: The cream cheese placed in the basket
```
:::

::::

You may also see generated *perception-audit* checkpoints report
`passed=False` (e.g. `target_obb_matches_truth`): those compare the
robot-frame perception OBB against the world-frame privileged pose — a
frame mismatch in the generated predicate, not a task failure. The physical
checkpoints (`target_held`, `target_in_basket`) are the ground truth.
Checkpoints run in `warn` mode by default, so the run continues either way;
see [Checkpoints](../running/checkpoints.md).

## 4. Watch and debug

```bash
uv run gap viz          # browse trials at localhost:9432 —
                        # graph swimlane, per-node I/O, masks, plans
```

The trace browser renders the graph swimlane, per-node resolved inputs and
outputs, and the image assets each node produced (camera frames, perception
masks). Watch the recorded run by opening
`outputs/agent_gen_sim/run_video.mp4` from disk with any video player, read
the raw execution record in `outputs/agent_gen_sim/dag_trace.json`, and
diff two runs with `gap trace-diff <a> <b>`. More in
[Traces](../running/traces.md).

## 5. Iterate

- Ask the agent to fix or extend the graph (it edits `workflow.json` /
  scripts and re-validates), or hand-author with `gap.builder` —
  see [Build a Graph](build-a-graph.md) and the
  [Builder guide](../authoring/builder.md).
- Missing a capability? `gap skills new my-skill --kind skill` scaffolds a
  bundle **with a unit test**; `gap skills test my-skill` runs it
  ([Authoring bundles](../skills/authoring-bundles.md),
  [Testing bundles](../skills/testing-bundles.md)). Your own registries
  layer over the public one: `gap registry init`
  ([Registries](../skills/registries.md)).
- Claiming success rates needs `gap benchmark <yaml> --gate`
  ([Benchmark](benchmark.md)), not one green run.

:::{warning}
The skill hard-gates real-robot commands — an agent will not run
`gap run --real ...` without your explicit confirmation, sim-first. Read
[Safety](../real-robots/safety.md) before any hardware work.
:::
