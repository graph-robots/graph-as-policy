# GaP — Graph-as-Policy

:::{admonition} 🧪 GaP Beta Code release (1 July 2026)
:class: important

**GaP is under active development and now in beta testing.** Please send
comments and suggestions to [kych@berkeley.edu](mailto:kych@berkeley.edu) — we
plan to release an updated version by **1 Aug 2026**. Expect rough edges: APIs,
the workflow schema, and skill interfaces may change without notice between
releases.
:::

:::{admonition} 🆕 Self-learning pipeline released (Nov 2026)
:class: tip

The loop that self-improves a GaP graph in simulation is open source:
[**agentic-self-learning**](https://github.com/graph-robots/agentic-self-learning).
A coding agent writes a graph policy, runs it on RoboSimStudio tasks, reads
structured simulation feedback, and rewrites it. Paper:
[*A Controlled Study of Models, Skills, Feedback, and Curriculum in Agentic
Robotics*](https://arxiv.org/abs/2611.09030) ·
[project page](https://graph-robots.github.io/agentic-self-learning/).
:::

**The graph is the policy.** GaP is a multi-agent coding harness that compiles
a natural-language task into a typed, verified computation graph of modular
skills, self-improves it in simulation, and runs the graph — not a black-box
policy — on simulators and real robots. It targets **Variational Automation
(VA)**: tasks a robot must perform *persistently and reliably across many
varying instances* (objects vary in geometry and pose), not just solve once.

<p><em>GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness for Variational Automation</em></p>

<div class="gap-real-grid">
  <figure>
    <video src="../_static/grocery_packing_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Pack grocery items — Franka, 10×.</figcaption>
  </figure>
  <figure>
    <video src="../_static/popcorn_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Make popcorn — long-horizon stove manipulation, 16×.</figcaption>
  </figure>
  <figure>
    <video src="../_static/tool_packing_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>Pack tools &amp; chargers into tagged bins.</figcaption>
  </figure>
  <figure>
    <video src="../_static/usb_insertion_real.mp4" autoplay loop muted playsinline></video>
    <figcaption>USB-C cable insertion — UR5 with force feedback.</figcaption>
  </figure>
</div>

*Real-robot rollouts of graphs generated from one-sentence task descriptions.*

The [Overview](../getting-started/overview.md) has the full story — what
Variational Automation is, why the policy is a graph, and what you need to
run GaP.

## Hello world — your first run

The fastest path is the **[15-minute tour](../getting-started/quickstart.md)**:
clone the two repos side by side, sync, install the skill bundles, set one LLM
key, and run the hero LIBERO graph end-to-end with the trace open.

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
git clone https://github.com/graph-robots/open-robot-skills.git   # sibling, auto-discovered
cd graph-as-policy
uv sync                                   # engine + LIBERO sim
uv run gap skills install --all           # per-bundle venvs (sam3, cuRobo, vlm, …)
export HF_TOKEN=...                        # gated SAM3 weights
export OPENROUTER_API_KEY=...             # codegen + in-graph VLM (openrouter is the default)
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph --sim libero_object_all_variance/0
uv run gap viz                            # browse the trial at localhost:9432
```

::::{grid} 1 2 3 3
:gutter: 3

:::{grid-item-card} 🚀 Installation
:link: ../getting-started/installation
:link-type: doc
`uv sync` + `gap skills install --all` — engine, sim, perception, motion planning, each in its own venv.
:::

:::{grid-item-card} ⏱️ The 15-minute tour
:link: ../getting-started/quickstart
:link-type: doc
Zero to a verified rollout on LIBERO — then a packing loop with a real backward edge.
:::

:::{grid-item-card} 🧪 Examples
:link: ../examples/index
:link-type: doc
Eleven examples, from the end-to-end quickstart to the release gate and real robots.
:::

:::{grid-item-card} 🧩 Skill catalog
:link: ../skills/skill-catalog
:link-type: doc
Contributable perception, grasping, transport, tracking, and policy skills.
:::

:::{grid-item-card} ⌨️ CLI reference
:link: ../reference/cli
:link-type: doc
`gap run`, `generate`, `check`, `registry`, `skills`, `benchmark`, `viz`, and more.
:::

:::{grid-item-card} 🤖 Use with Claude Code
:link: ../agents/claude-code
:link-type: doc
One plugin install teaches AI coding agents to drive the whole workflow.
:::
::::

Read more in [Overview](../getting-started/overview.md), or jump straight to
[Installation](../getting-started/installation.md).

```{toctree}
:hidden:
:caption: Getting Started
:maxdepth: 1

../getting-started/overview
../getting-started/installation
../getting-started/quickstart
../getting-started/concepts
```

```{toctree}
:hidden:
:caption: Examples
:maxdepth: 1

../examples/index
```

```{toctree}
:hidden:
:caption: Running Graphs
:maxdepth: 1

../running/execution
../running/environments
../running/checkpoints
../running/traces
```

```{toctree}
:hidden:
:caption: Generating & Authoring
:maxdepth: 1

../authoring/generation
../authoring/llm-providers
../authoring/builder
../authoring/patterns
```

```{toctree}
:hidden:
:caption: Skills & Tools
:maxdepth: 1

../skills/registries
../skills/skill-catalog
../skills/tool-catalog
../skills/authoring-bundles
../skills/testing-bundles
```

```{toctree}
:hidden:
:caption: Benchmarks & Policies
:maxdepth: 1

../benchmarks/benchmarking
../benchmarks/policies
```

```{toctree}
:hidden:
:caption: Real Robots
:maxdepth: 1

../real-robots/connectors
../real-robots/safety
```

```{toctree}
:hidden:
:caption: AI Agents
:maxdepth: 1

../agents/claude-code
```

```{toctree}
:hidden:
:caption: Reference
:maxdepth: 1

../reference/cli
../reference/api
../reference/workflow-schema
../reference/executor
../reference/connector-tools
../reference/environment-variables
../reference/benchmark-config
../reference/faq
```

```{toctree}
:hidden:
:caption: Developers
:maxdepth: 1

../developers/architecture
../developers/contributing
../developers/roadmap
```
