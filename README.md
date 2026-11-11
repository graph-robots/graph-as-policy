<div align="center">

# Graph-as-Policy
<h3 align="center">
  <strong>🧪 BETA CODE RELEASE</strong> | <em>Beta</em> | July 1, 2026
</h3>



**CoRL 2026** · [Paper (arXiv)](https://arxiv.org/abs/2607.05369) · [Project page](https://graph-robots.github.io/gap/)

🆕 **Nov 2026 — the self-learning pipeline is open source:**
[**agentic-self-learning**](https://github.com/graph-robots/agentic-self-learning)
· [Paper (arXiv)](https://arxiv.org/abs/2611.09030)
· [Project page](https://graph-robots.github.io/agentic-self-learning/)

**The graph is the policy.**
GaP targets *Variational Automation* — tasks a robot must perform persistently
and reliably across many varying instances (objects vary in geometry and pose),
not just solve once. It compiles a natural-language task into a typed, verified
computation graph of modular skills, self-improves it in simulation, and runs
the graph — not a black-box policy — on simulators and real robots.

[![License: Apache 2.0](https://img.shields.io/badge/license-Apache--2.0-green.svg)](LICENSE)
[![Python 3.10+](https://img.shields.io/badge/python-3.10%2B-blue.svg)](pyproject.toml)
[![Skills: open-robot-skills](https://img.shields.io/badge/skills-open--robot--skills-orange.svg)](https://github.com/graph-robots/open-robot-skills)
[![Docs](https://img.shields.io/badge/docs-quickstart-0f766e.svg)](https://graph-robots.github.io/graph-as-policy/getting-started/quickstart.html)

![GaP on real robots — grocery packing, popcorn, tool packing, USB-C insertion](docs/assets/real_robots_strip.gif)

<sub>GaP running on real robots — graphs generated from natural-language
tasks, refined in sim, executed on hardware. Left→right: grocery packing
(Franka, 10×), popcorn (long-horizon stove manipulation, 16×), tool & charger
packing into tagged bins, and USB-C insertion (UR5 with force feedback).</sub>

</div>

```python
import gap

conn   = gap.connector.sim("libero", task="libero_object/0")
result = gap.execute("examples/libero_quickstart/graph", conn)
graph  = gap.agent.generate_sync("pick the soup can and put it in the basket")
print(graph)                  # the policy, as a graph
gap.viz.serve("outputs")      # browse the recorded trial
```

Skills live in the sibling
[**open-robot-skills**](https://github.com/graph-robots/open-robot-skills)
repo (Anthropic Agent Skills format, contributable) and are discovered
by path — clone the two repos side by side and every command finds them.

## Quickstart

**Requirements:** **1× NVIDIA RTX 4090 (≥24 GB VRAM, Linux + EGL)**, an
LLM API key (OpenRouter / Vertex), and
[uv](https://docs.astral.sh/uv/). First run downloads ~3.5 GB of model
weights; the gated SAM3 weights are part of the default perception path,
so in practice you also need `HF_TOKEN` (a free
[HuggingFace token](https://huggingface.co/settings/tokens) with SAM3
access) before the first run.

```bash
git clone --recurse-submodules https://github.com/graph-robots/graph-as-policy.git
#   ^ submodules are required (vendored sim stack) — if you already cloned
#     plain, run: git submodule update --init --recursive
git clone https://github.com/graph-robots/open-robot-skills.git    # sibling, auto-discovered
cd graph-as-policy
uv sync                                  # engine + LIBERO sim (now baseline)
# `uv sync --extra vertex` instead if you'll use --provider vertex for codegen
uv run gap skills install --all          # per-bundle venvs (sam3, cuRobo, vlm, …)
#   --all includes the heavyweight learned-policy bundles; for one example,
#   `gap skills install --workflow <graph-dir>` installs just what it uses

export HF_TOKEN=...                      # for the gated SAM3 weights
uv run gap skills check --download       # weight prefetch (SAM3 + GDINO) + capability gate

# Pick one LLM provider for codegen + the in-graph VLM. openrouter is the
# default; for vertex, see `docs/source/authoring/llm-providers.md`.
export OPENROUTER_API_KEY=...
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                           # browse the trial at localhost:9432
```

See the **[15-minute tour](https://graph-robots.github.io/graph-as-policy/getting-started/quickstart.html)**
for the full walkthrough (clone → run → generate → trace).

## Examples

| Example | What it shows |
|---|---|
| [libero_quickstart](examples/libero_quickstart/) | Hero: vision → OBB grasp → transport, ground-truth verified |
| [grocery_packing](examples/grocery_packing/) | Pack every item with a loop: a graph with a real backward edge |
| [generate_a_graph](examples/generate_a_graph/) | Instruction → validated workflow dir; all LLM providers |
| [build_a_graph](examples/build_a_graph/) | Full Python authoring with `gap.builder`: checkpoints, recovery |
| [agent_quickstart](examples/agent_quickstart/) | Claude Code + one skill: sentence → graph → sim success |
| [grocery_fulfillment](examples/grocery_fulfillment/) | Flagship acceptance family; graphs LLM-generated per task |
| [benchmark](examples/benchmark/) | Grid harness: smoke → posvar → release gate |
| [steered_policy](examples/steered_policy/) | Perceive + hover, then hand to a learned VLA policy |
| [collect_and_train](examples/collect_and_train/) | Graph as scripted expert → dataset → train → policy node |
| [cable_ur](examples/cable_ur/) | Real UR + ZED perception (motion-disabled) |
| [real_franka_pick_place](examples/real_franka_pick_place/) | Real Franka pick-place via robots_realtime |

Full gallery with install needs and time estimates in
**[examples/README.md](examples/README.md)**. Real-robot examples
— read [docs/safety.md](docs/safety.md) first.

## Self-learning pipeline

The loop that self-improves a graph in simulation is released as
**[agentic-self-learning](https://github.com/graph-robots/agentic-self-learning)**.
A coding agent (Claude Code or Codex) writes a GaP `workflow.json` policy,
runs it on [RoboSimStudio](https://github.com/robosimstudio/robosimstudio)
tasks, reads structured simulation feedback, and rewrites the graph. Every
feedback condition from the study (state, relations, or visual feedback; how
space is described; per-episode, per-motion, or per-node timing) is a config
setting, and skills from
[open-robot-skills](https://github.com/graph-robots/open-robot-skills) plug in
as the agent's skill library.

It accompanies *A Controlled Study of Models, Skills, Feedback, and Curriculum
in Agentic Robotics* ([arXiv](https://arxiv.org/abs/2611.09030) ·
[project page](https://graph-robots.github.io/agentic-self-learning/)).

## Documentation

Full docs site: **<https://graph-robots.github.io/graph-as-policy/>**

- [Quickstart (15-min tour)](https://graph-robots.github.io/graph-as-policy/getting-started/quickstart.html) — clone → run → generate → trace
- [Runtime & schema](https://graph-robots.github.io/graph-as-policy/reference/workflow-schema.html) — workflow JSON, executor semantics, checkpoints
- [Skill authoring](https://graph-robots.github.io/graph-as-policy/skills/authoring-bundles.html) — Agent Skills format, `gap.requires:` frontmatter
- [LLM providers](https://graph-robots.github.io/graph-as-policy/authoring/llm-providers.html) — openrouter / vertex
- [Skill registries](https://graph-robots.github.io/graph-as-policy/skills/registries.html) — `--skills`, `$GAP_SKILLS_PATH`, `gap registry …`
- [CLI reference](https://graph-robots.github.io/graph-as-policy/reference/cli.html) — every `gap` verb
- [Safety](https://graph-robots.github.io/graph-as-policy/real-robots/safety.html) — required reading before any real-robot example

## Use with Claude Code & AI agents

GaP ships an agent skill ([`agent/`](agent/)) that teaches AI coding
agents the workflows above:

```bash
claude plugin marketplace add graph-robots/graph-as-policy
claude plugin install gap@gap
```

Then ask: *"what can this robot do right now?"*, *"run the quickstart graph
in sim"*, *"generate a graph that packs the groceries"*, *"why did this
trial fail?"*. Real-robot commands are gated on explicit human
confirmation. Other agents (Cursor, Codex, …) and the no-install path:
[agent/INSTALL.md](agent/INSTALL.md).

## Citation

If you use GaP in your research, please cite our CoRL 2026 paper:

```bibtex
@inproceedings{chen2026gap,
  title     = {GaP: A Graph-as-Policy Multi-Agent Self-Learning Harness
               for Variational Automation (VA) Tasks},
  author    = {Chen, Kaiyuan and Xie, Shuangyu and Fu, Letian and Yu, Justin and
               Pacini, William and Bajamahal, Sandeep and Kim, Hudson and
               Drake, Jaimyn and Kim, Daehwa and Xue, Haoru and Francis, Jonathan and
               Juette, Christian and Schaldenbrand, Peter and Seker, Muhammet Yunus and
               Wickramarachchi, Ruwan and Yoo, Uksang and Wang, Guanzhi and
               Murali, Adithyavairavan and Sundaralingam, Balakumar and
               Sastry, S. Shankar and Huang, Spencer and Zhu, Yuke and
               Fan, Linxi and Goldberg, Ken},
  booktitle = {Conference on Robot Learning (CoRL)},
  year      = {2026},
  url       = {https://arxiv.org/abs/2607.05369}
}
```

If you use the self-learning pipeline
([agentic-self-learning](https://github.com/graph-robots/agentic-self-learning)),
please also cite:

```bibtex
@misc{chen2026controlled,
  title         = {A Controlled Study of Models, Skills, Feedback, and Curriculum
                   in Agentic Robotics},
  author        = {Chen, Kaiyuan and Zhu, Xinyue and Xie, Shuangyu and Yoo, Uksang and
                   Luong, Thi Thu Phuong and Gualtieri, Marcus and Juette, Christian and
                   Ren, Liu and Huang, Spencer and Goldberg, Ken},
  year          = {2026},
  eprint        = {2611.09030},
  archivePrefix = {arXiv},
  primaryClass  = {cs.RO},
  url           = {https://arxiv.org/abs/2611.09030}
}
```

## License & attribution

graph-as-policy and open-robot-skills are Apache-2.0-licensed; they stand on third-party
work — LIBERO/LIBERO-PRO, Variational-Automation-Benchmark, robosuite,
robots_realtime, pyroki, SAM3, Grounding DINO, and (optionally) NVIDIA
cuRobo. Full attribution table: **[NOTICE.md](NOTICE.md)**.
