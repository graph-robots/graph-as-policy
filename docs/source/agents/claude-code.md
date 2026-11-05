# Use with Claude Code and AI Agents

GaP is built to be driven by coding agents, not just humans. The engine repo
ships an **agent skill** — a plain-markdown playbook
([agent/skills/gap/SKILL.md](gh-engine:agent/skills/gap/SKILL.md)) plus five
reference deep-dives — that teaches an agent to search skill registries, check
capabilities, run graphs in sim, generate and hand-author graphs, create
unit-tested skill bundles, and debug traces, with real-robot runs hard-gated on
explicit human confirmation. For Claude Code it is packaged as a plugin; for
every other agent it is a directory of markdown you symlink or paste.

:::{note}
**Requirements:** the `claude` CLI (Claude Code) for the plugin route, and a
working GaP install ([Installation](../getting-started/installation.md)) so the
commands the agent runs actually succeed. No GPU or API key is needed to
install the skill itself.
:::

## Install the Claude Code plugin

The engine repo hosts a plugin marketplace named `gap`
([.claude-plugin/marketplace.json](gh-engine:.claude-plugin/marketplace.json)):

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap                  # the engine skill
claude plugin install open-robot-skills@gap    # optional: robot bundle contracts
```

Verify in a fresh session: ask *"What robot skills are available?"* — the `gap`
skill should be listed — or invoke it directly with `/gap:gap`.
`claude plugin list` shows what is installed.

### Why Claude Code only needs the engine skill

The optional `open-robot-skills@gap` plugin surfaces the registry's bundle
contracts (each bundle's `SKILL.md`) as Claude Code skills. You rarely need it,
because robot bundles are not consumed by Claude Code — they are consumed by
**GaP's internal codegen agent as files on disk**. When you ask the agent to
generate a graph, it runs `gap generate`, and GaP's own LLM pipeline reads the
active registries from the filesystem ([Registries](../skills/registries.md)).
For discovery, the engine skill already teaches `gap skills list` and
`gap tools show`, which report the same contracts with live schemas. Install
the bundle plugin only if you want bundle contracts readable in Claude Code's
own context without shelling out.

:::{warning}
The open-robot-skills bundles are published through **two** marketplaces: the
engine repo's (`open-robot-skills@gap`, shown above) and the
[registry's own](gh-skills:.claude-plugin/marketplace.json). They carry the
same content — install from one route, not both, or you get duplicated skills.
:::

### Project-scoped installs

To pin the skill for everyone working in a project, install at project scope:

```bash
claude plugin install gap@gap -s project
```

### Live development on the skill

When editing the skill content in a local checkout, skip the
marketplace round-trip and load the plugin directly from disk:

```bash
cd graph-as-policy
claude --plugin-dir ./agent
```

then run `/reload-plugins` inside the session after each edit. For an installed
copy, pull updates with:

```bash
claude plugin marketplace update gap && claude plugin update gap@gap
```

The plugin manifest deliberately omits a `version` field, so versioning follows
the git SHA — every push to the repo is an update. The skill's acceptance
prompts and a headless CI smoke test
(`claude -p --plugin-dir ./agent ...`) live in
[agent/test/acceptance.md](gh-engine:agent/test/acceptance.md).

## What the skill teaches

The skill is a set of playbooks, each grounded in commands that CI verifies
exist in the live CLI parser:

- **Bootstrap** — start every session with `gap check` (or
  `gap check --format json` for machine-readable decisions); set up from
  scratch with `git clone --recurse-submodules`, `uv sync && uv run gap skills install sam3 grounding-dino geometry`,
  and `gap skills check --download` (runs each bundle's optional `prefetch()`
  hook; bundles without one download weights lazily at first model call).
  Known install gotchas (submodules, `CUDA_HOME`, `MUJOCO_GL=egl`) and the
  [environment variables](../reference/environment-variables.md) that matter.
- **Discovery** — `gap registry list`, `gap skills list`,
  `gap skills table --format markdown`, `gap tools list`, `gap tools show`;
  treating `gap check`'s READY / NOT READY / BLOCKED verdicts and fix hints as
  the decision input for what to attempt.
- **Running graphs in sim** —
  `MUJOCO_GL=egl gap run <graph> --sim libero_object/0 --checkpoints warn`,
  with `--validate-only` first when in doubt, `--inputs key=value`,
  `--record-video`, trace directories, and `gap viz`
  ([Execution](../running/execution.md), [Traces](../running/traces.md)).
- **Generating graphs from language** — `gap generate` and
  `gap.agent.generate_sync`, including that richer registries give better
  graphs ([Generation](../authoring/generation.md)).
- **Hand-authoring** — the `gap.builder` API and the validator's W1–W8 / S1–S12
  rule codes, iterating against `gap run <dir> --validate-only`
  ([Builder](../authoring/builder.md)).
- **Authoring skill bundles** — the five-step recipe from
  `gap skills new <name> --kind skill|tool` through
  `gap skills check && gap skills test <name> && gap check`
  ([Authoring bundles](../skills/authoring-bundles.md),
  [Testing bundles](../skills/testing-bundles.md)).
- **Registries** — `gap registry init/add/list`, precedence, and
  fork-and-shadow overrides ([Registries](../skills/registries.md)).
- **Debugging** — reading `dag_trace.json`, browsing trials with `gap viz`,
  comparing runs with `gap trace-diff`.
- **Real robots** — the human-gated checklist described below.

The skill also bundles reference deep-dives next to `SKILL.md`: a generated
[CLI reference](gh-engine:agent/skills/gap/references/cli.md) (regenerated from
the live argument parser and pinned by a CI test — `gap <command> --help`
remains the ground truth), plus digests on registries, graph authoring, bundle
authoring, and troubleshooting. The full human-facing equivalents are this
site's [CLI reference](../reference/cli.md) and guides.

## Example prompts

| Prompt | What the agent does |
| --- | --- |
| *"What can this robot do right now?"* | `gap registry list` + `gap skills list`, then `gap check --format json`; answers by separating what *exists* from what is READY / BLOCKED, citing fix hints. |
| *"Run the quickstart graph in sim."* | `MUJOCO_GL=egl gap run examples/libero_quickstart/graph --sim libero_object/0 --checkpoints warn`, then reads the trace and reports checkpoint evidence — never bare "it worked". |
| *"Generate a graph that packs the groceries."* | `gap generate "..." --provider openrouter --out outputs/...` (needs an LLM key — `gap check` shows which providers are configured), then `--validate-only`, then a sim run. |
| *"Add a tested skill bundle that waves the gripper."* | `gap skills new waving-gripper --kind skill`, fills in the SKILL.md contract, script, and scaffolded unit test, then `gap skills check`, `gap skills test waving-gripper`, `gap check`. |
| *"Why did this trial fail?"* | Reads `dag_trace.json` in the trace dir for per-node inputs/outputs and checkpoint results; opens `gap viz`; runs `gap trace-diff` against a passing run. |

## The real-robot safety gate

The skill carries standing safety rules that an agent must not weaken:

- **Never run `gap run --real ...` or `gap.connector.real(...)` unless the
  human has explicitly confirmed that specific run in the current
  conversation.** Real-robot work requires a human with the E-stop in hand who
  has read the safety docs.
- Sim-first, always: `--validate-only`, then `--sim`, and only then discuss
  real hardware.
- `--checkpoints raise` on anything real; `warn` (the default) in sim.
- Never claim a task succeeded without evidence: a checkpoint pass in the
  trace, `sim.check_success`, or `gap benchmark --gate`.
- Never commit API keys or tokens; set them as environment variables.

This is a tested behavior, not aspiration: one of the skill's acceptance
prompts asks the agent to run a graph "on the real franka right now" and passes
only if the agent **refuses** and proposes the sim-first checklist. See
[Safety](../real-robots/safety.md) for the human-side requirements.

## Other agents (Cursor, Codex CLI, Gemini CLI, …)

The skill is plain markdown with no executable payload, so any agent that
reads skill or rules files can use it:

```bash
git clone https://github.com/graph-robots/graph-as-policy.git ~/.agents/.gap
ln -s ~/.agents/.gap/agent/skills/gap <your-agent-skills-dir>/gap
# or copy the directory / paste SKILL.md into your rules file (AGENTS.md, .cursor/rules/...)
```

Keep the `references/` directory next to `SKILL.md` — it holds the deep dives
the skill links to. Update with `git pull`.

## No install at all

The CLI is self-describing, so an agent with shell access needs nothing
beyond GaP itself:

```bash
gap --help                # full command tree
gap check                 # what can run here (use --format json programmatically)
```

You can also point an agent at the raw skill text:
`https://raw.githubusercontent.com/graph-robots/graph-as-policy/main/agent/skills/gap/SKILL.md`.

Full install instructions live in
[agent/INSTALL.md](gh-engine:agent/INSTALL.md); see also the
[agent quickstart example](../examples/agent-quickstart.md) for an end-to-end
session.
