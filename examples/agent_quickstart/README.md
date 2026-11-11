# Drive GaP with an AI coding agent (Claude Code)

The step-by-step loop this example documents was run for real — every
command and output below comes from an actual session: **Claude Code,
armed with one skill, generated a pick-and-place graph from a sentence,
validated it, and the graph then ran in LIBERO sim and put the cream
cheese in the basket (75 s, on video).**

Two agent layers are involved — don't conflate them:

```
Claude Code ──(the `gap` skill)──▶ drives the CLI: gap check / generate / run / skills …
                                         │
                                         ▼
                            GaP's own codegen agent (`gap generate`)
                                         │  reads bundle SKILL.mds from disk
                                         ▼
                       skill registries (e.g. ../open-robot-skills)
```

Claude Code needs **only the `gap` skill**. The robot bundles in
[open-robot-skills](https://github.com/graph-robots/open-robot-skills)
are consumed by GaP's *internal* codegen agent as files on disk — they
do not need to be installed into Claude Code.

## 0. Prerequisites (once)

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # side-by-side
cd graph-as-policy
uv sync && uv run gap skills install sam3 grounding-dino geometry            # engine + sim + perception bundles
uv run gap skills check --download    # verify bundles + prefetch weights (~3.5 GB)
```

Pick an LLM provider for generation and the VLM perception bundle
(`gap check` tells you what is configured):

```bash
export OPENROUTER_API_KEY=...         # simplest: one key drives both
```

Or Vertex AI (no API key — uses gcloud ADC). Set once per shell (or
`.bashrc`) and the bare `gap generate "<task>"` works with no flags:

```bash
gcloud auth application-default login                  # once
export GOOGLE_CLOUD_PROJECT=<your-project>
export GAP_LLM_PROVIDER=vertex GAP_LLM_MODEL=<gemini-model>   # codegen
export GAP_VLM_PROVIDER=vertex GAP_VLM_PROJECT_ID=$GOOGLE_CLOUD_PROJECT \
       GAP_VLM_REGION=global GAP_VLM_MODEL=<gemini-model>     # perception bundle
```

One uv gotcha: include the SDK on every `uv run` — a plain `uv run`
re-syncs the project venv and prunes it again:

```bash
uv run --extra vertex gap generate "<task>"
```

## 1. Install the agent skill (once)

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap
```

Verify in a new Claude Code session: ask *"What robot skills are
available?"* — `gap` should be listed. Working inside this checkout?
The skill content lives at [agent/skills/gap/](../../agent/skills/gap/);
other agents (Cursor, Codex, …): [agent/INSTALL.md](../../agent/INSTALL.md).

## 2. Ask Claude Code for a graph

Prompt used in the recorded session:

> Using your gap skill: generate a robot task graph for "pick up the
> cream cheese and put it in the basket" with output directory
> outputs/agent_gen_test. Then validate the generated graph without
> running any simulator or real robot.

What the agent does (all taught by the skill — you can run the same
three commands by hand):

```bash
uv run gap check                                   # capability report first
uv run gap generate "pick up the cream cheese and put it in the basket" \
    --out outputs/agent_gen_test
# vertex variant (verified):
#   GOOGLE_CLOUD_PROJECT=<proj> uv run --extra vertex gap generate "..." \
#       --provider vertex --model <gemini-model> --out outputs/agent_gen_test
uv run gap run outputs/agent_gen_test/task_00 --validate-only
```

Expected ending: `OK: 0 errors` and a workflow directory:

```
outputs/agent_gen_test/task_00/
├── workflow.json      # 4 subgraphs: perceive_target, perceive_container,
│                      #   grasp_target (grasping-with-planner),
│                      #   place_target (transporting-objects)
├── scripts/           # the canonical skill scripts the graph calls
├── checkpoints/       # generated postcondition predicates
└── agent_traces/      # the codegen agents' own transcripts
```

In the recorded session the agent's first `gap generate` hit a missing
vertex SDK — it read the error, ran `uv sync --inexact --extra vertex`,
and retried successfully. That self-repair is the skill's
troubleshooting playbook working as intended.

## 3. Run it in sim (with video)

The cream-cheese scene is task **1** of the `libero_object_all_variance`
suite (classic LIBERO numbering):

```bash
MUJOCO_GL=egl uv run gap run outputs/agent_gen_test/task_00 \
    --sim libero_object_all_variance/1 \
    --record-video --trace-dir outputs/agent_gen_sim
```

Recorded result:

```
SUCCESS (exit=success, 75.0s)
trace: outputs/agent_gen_sim
checkpoint: ... name='target_held'      passed=True   ← gripper held the cream cheese
checkpoint: ... name='target_in_basket' passed=True   ← it ended up in the basket
video: outputs/agent_gen_sim/run_video.mp4 (506 frames)
```

You may also see generated *perception-audit* checkpoints report
`passed=False` (e.g. `target_obb_matches_truth`): those compare the
robot-frame perception OBB against the world-frame privileged pose — a
frame mismatch in the generated predicate, not a task failure. The
physical checkpoints (`target_held`, `target_in_basket`) are the ground
truth. Checkpoints run in `warn` mode by default, so the run continues
either way.

## 4. Watch and debug

```bash
uv run gap viz          # browse trials at localhost:9432 —
                        # video, graph swimlane, per-node I/O, masks, plans
```

or open the mp4 directly, read `outputs/agent_gen_sim/dag_trace.json`,
and diff two runs with `gap trace-diff <a> <b>`.

## 5. Iterate

- Ask the agent to fix or extend the graph (it edits `workflow.json` /
  scripts and re-validates), or hand-author with `gap.builder`
  ([build_a_graph](../build_a_graph/)).
- Missing a capability? `gap skills new my-skill --kind skill` scaffolds
  a bundle **with a unit test**; `gap skills test my-skill` runs it.
  Your own registries layer over the public one: `gap registry init`.
- Claiming success rates needs `gap benchmark <yaml> --gate`
  ([benchmark](../benchmark/)), not one green run.

**Safety:** the skill hard-gates real-robot commands — an agent will not
run `gap run --real ...` without your explicit confirmation, sim-first.
Read [docs/safety.md](../../docs/safety.md) before any hardware work.
