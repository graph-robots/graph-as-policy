# Quickstart: The 15-Minute Tour

Zero to a verified rollout, with the trace open. You will:

- run the checked-in quickstart graph on the LIBERO simulator — vision
  models, IK, and the sim in one process;
- read the recorded trace, the artifact everything else revolves around;
- browse it in `gap viz`;
- generate your own graph from one sentence;
- loop the same machinery: pack *every* item on the table, through a
  graph with a real backward edge.

::::{grid} 1 1 2 2
:gutter: 3

:::{grid-item}
![quickstart rollout on LIBERO: perceive, grasp, transport](../_static/quickstart_rollout.gif)

*"pick up the soup can and put it in the basket" — live rollout*
:::

:::{grid-item}
![the executed quickstart workflow graph](../_static/quickstart_graph.png)

*the graph that ran it*
:::
::::

:::{note} Requirements
Linux + an NVIDIA GPU with EGL (~10 GB VRAM covers this quickstart;
[Installation](installation.md) recommends ≥24 GB for the full stack), the
[two repos installed](installation.md) with
`uv sync && uv run gap skills install --all`, and an LLM API key
(`export OPENROUTER_API_KEY=...`, or
[another provider](../authoring/llm-providers.md)). The first run
downloads ~3.5 GB of model weights, and the gated SAM3 weights are in
this quickstart's perception path — so set `HF_TOKEN` first; see
[Model weights](installation.md#model-weights). The closing packing
segment (minute 12–15) additionally needs the CuRobo planning stack:
`uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo`.
:::

## Minute 0–6: run the quickstart graph

LIBERO sim + Grounding DINO + SAM3 + a hosted VLM + in-process IK,
executing a perceive → grasp → transport graph:

```bash
export OPENROUTER_API_KEY=...          # or another provider

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
```

`MUJOCO_GL=egl` selects headless GPU rendering and is required on every
sim command; `--sim SUITE/TASK` picks the seeded LIBERO task variation.

While it runs (~25–55 s per trial, measured on an A100 — the **first**
trial of a session runs minutes longer while the vision models load cold;
steady-state kicks in from the second trial), what you are
watching in the log: perception (Grounding DINO proposes boxes, a hosted
VLM picks the right one, SAM3 segments it), geometry (mask + depth →
oriented bounding box → top-down grasp candidates), then motion (align,
descend, close, transport). That one command, in one process:

- launched LIBERO (MuJoCo + EGL) on the seeded task variation;
- found the can and the basket with Grounding DINO + SAM3, disambiguated
  by a hosted VLM;
- fused masks + depth into oriented bounding boxes and derived a top-down
  grasp;
- executed perceive → grasp → transport with in-process IK;
- verified `target_held` against simulator ground truth at the subgraph
  exit;
- recorded the full trace to `outputs/`.

Two flags worth knowing now: `--validate-only` checks a graph (structure
plus skill registry) without touching a sim, and
`--checkpoints off|warn|raise` (default `warn`) controls how ground-truth
postcondition checkpoints are enforced at subgraph exits — see
[Checkpoints](../running/checkpoints.md).

## Reading the trace

When it finishes, the **trace** is in `outputs/run_<timestamp>/`:

```text
outputs/run_<timestamp>/
├── workflow.json        # the graph that ran (self-contained copy)
├── dag_trace.json       # every node visit: timings, exits, errors, routing
├── node_data/<node>/    # per-node inputs/outputs + extracted assets
└── scripts/             # the script nodes, as executed
```

`node_data/` is where debugging lives: the perception node's directory
holds the camera frames it saw (PNG), the masks and point clouds it
produced (NPZ), and the VLM exchange; the grasp nodes record the poses
they targeted. If a run fails, the failing node's directory shows exactly
what it was looking at when it failed. The full layout is specified in
[Traces](../running/traces.md) — it is a stability guarantee, safe to
build tooling against.

A rollout video is recorded automatically for sim runs: the run is
saved as `<trace-dir>/run_video.mp4` alongside the JSON for easy sharing
(plus per-camera videos when the env buffers them). Open it from disk with
any player. Pass `--no-video` if you want to skip rendering.

## Minute 6–8: browse it

```bash
uv run gap viz                        # browse the recorded trial at localhost:9432
```

Click into the trial: the graph renders as swimlanes (one per subgraph)
with the executed route highlighted; clicking a node shows its inputs,
outputs, timings, and assets — the same `node_data/` you just saw, with
images inline. When two runs disagree,
`gap trace-diff <trial_a> <trial_b>` diffs them structurally.

## Minute 8–12: generate your own

```bash
uv run gap generate "pick up the alphabet soup can and place it in the basket"
```

The agent pipeline — coordinator → per-subgraph agents → checkpoint
agent — picks skills from the open-robot-skills catalog, writes the
subgraphs and their scripts, attaches ground-truth checkpoints, and
validates the result; on validation errors a script-fix loop repairs its
own output (you'll see those attempts in the log). The compiled policy
prints in the terminal as a box-drawing graph, and the artifact lands in
`outputs/generated_<timestamp>/task_00/` — the same shape as the
quickstart graph's, so you already know how to read it. Run it:

```bash
MUJOCO_GL=egl uv run gap run outputs/generated_<timestamp>/task_00 \
  --sim libero_object_all_variance/0
```

:::{note}
`gap run` must target the `task_00/` subdirectory, not the `--out`
directory itself — generation writes one folder per task.
:::

[Generation](../authoring/generation.md) covers the pipeline, providers,
and config in depth.

## Minute 12–15: loop it — pack every item

One pick is a straight line through the graph. The packing example runs
the same perceive → grasp → transport machinery **in a loop**: transport
routes *back* to perception, and the graph keeps picking until a VLM
confirms every grocery item is in the basket.

<figure>
  <video src="../_static/grocery_packing_sim.mp4" autoplay loop muted playsinline width="70%"></video>
  <figcaption>The packing loop on the pack-all suite
  (<code>libero_object_packing/0</code>), 2×: one perceive → grasp →
  transport pass per item, until the VLM completion check reports the
  table clear.</figcaption>
</figure>

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo   # one-time: adds CuRobo planning

MUJOCO_GL=egl uv run gap run examples/grocery_packing/packing_graph \
  --sim libero_object_packing/0
```

Watch the first pass land and the route bend backward:
`transport --placed--> perceive_next` resolves to an already-completed
node, so the executor resets and re-runs the loop body — re-perceiving
*both* the next item and the basket every iteration. Termination is
**unprivileged**: a per-pass VLM completion check, not a simulator
verdict, decides when the table is clear, so the same policy runs
unchanged on a real robot. Seeing the backward edge fire once is the
point of this segment; a full pack takes roughly a minute per item (plus
a one-time ~40 s CuRobo kernel JIT on the first planning call), so let
it finish in the background and open `gap viz` after — `perceive_next`
is visited once per object.

[Grocery Packing](../examples/grocery-packing.md) dissects the loop in
depth: the subgraphs, the backward-edge (subgraph-revisit) semantics,
the layered termination signals, and the `gap generate` recipe that
reproduces this same graph from one sentence.

## Where next

| You want to | Go to |
|---|---|
| See everything GaP can do | [Examples gallery](../examples/index.md) |
| Understand the vocabulary precisely | [Concepts](concepts.md) |
| Author graphs in Python | [Build a graph](../examples/build-a-graph.md) · [The builder API](../authoring/builder.md) |
| Write or contribute a skill | [Authoring bundles](../skills/authoring-bundles.md) |
| Make a success-rate claim | [Benchmarking](../benchmarks/benchmarking.md) — one green run is not a number |
| Move a real robot | [Safety](../real-robots/safety.md) **first**, then [Franka pick & place](../examples/real-franka-pick-place.md) |
