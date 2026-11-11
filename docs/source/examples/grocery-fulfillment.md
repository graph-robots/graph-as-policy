# Grocery Fulfillment

:::{note} Requirements
A CUDA GPU (MuJoCo EGL rendering + CuRobo planning + perception weights), the
`grocery` dependency extra, and an LLM credential for graph generation. The
configs pin Vertex with `gemini-3.1-pro-preview`; the default `openrouter`
provider works by switching the config's `llm:` block.
:::

This is GaP's flagship acceptance family — and its release gate. The task:
pick a *described* grocery item ("the alphabet soup can with a blue and yellow
label") and place it in the basket, under the benchmark's baked pose,
permutation, and basket-swap variations. Every graph is **generated per task**
by `gap.agent.generate` (the `llm_generation` benchmark mode); nothing is
hand-written. The full config doubles as the release gate: `--gate` makes
`gap benchmark` exit non-zero when a cell falls below the config's
`gate_threshold`.

:::{figure} ../_static/sim_grocery_variation.gif
:alt: Sampled initial arrangements of the grocery fulfillment benchmark
:width: 60%

Each instance re-samples the object arrangement and basket placement — the
generated graph must re-perceive and adapt every episode.
:::

The example lives at [examples/grocery_fulfillment](gh-engine:examples/grocery_fulfillment);
the configs it runs live in [examples/benchmark](gh-engine:examples/benchmark)
(see [Benchmark Grids](benchmark.md)).

## Run it

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo   # quickstart set + CuRobo planning
uv run gap skills check --download

# the 20-trial smoke (tasks 0-1)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance_smoke.yaml --gate

# the full acceptance gate (interruptible; --resume continues)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

`gap skills check --download` verifies the skill bundles and runs any
per-bundle prefetch hooks; the current bundles do not define one, so model
weights download lazily on the first model call instead. Expect the first
trial of a fresh checkout to be slow for that reason.

## Config anatomy

[grocery_acceptance.yaml](gh-engine:examples/benchmark/grocery_acceptance.yaml)
has no `benchmark:` block, so it runs in *suites mode*: each `suites:` entry is
one independent cell, and all cells launch concurrently — one suite per task:

```yaml
task: "auto"            # per-task prompts resolved from LIBERO metadata

llm:
  provider: vertex
  model: gemini-3.1-pro-preview
  project_id: bc-y7-06

gate_threshold: 0.90

suites:
  - suite_name: libero_object_all_variance
    task_ids: [0]
    task_prompts:
      0: "Pick up the alphabet soup can and put it in the basket."
    objects:
      target: "alphabet soup can with a blue and yellow label"
      expected_label: "Alphabet Soup"
      shape_hint: "metal cylinder about 7 cm wide, wrapped in a blue-and-yellow label"
    num_workers: 1
  # ... 9 more suites, one per task

trials:
  trials_per_generation: 50
  task_ids: [0, 1, 2, 3, 4, 5, 6, 7, 8, 9]
  num_workers: 25
  regenerate_code_per_trial: false
  record_video: true
  task_timeout_secs: 900
  enable_tracing: false
  output_dir: ../../benchmark_runs/grocery_acceptance
```

The parts worth knowing:

- **`objects:` hints are load-bearing.** Each task's
  `target` / `expected_label` / `shape_hint` block is appended to the code
  generation prompt so the generated perception scripts get an unambiguous
  description of the LIBERO grocery asset. Drop them and the generated
  graphs misidentify look-alike items.
- **`task_timeout_secs: 900`, not lower.** Flat-box grasp planning
  legitimately runs long when the goalset planner falls back to per-pose
  iteration — a tight timeout kills trials that are on track to succeed.
- **`regenerate_code_per_trial: false`** generates one graph per task and
  reuses it across that task's trials; seeds select the suite's baked
  initial-state variations.
- **`skills:` is optional.** Omit it to auto-discover the open-robot-skills
  checkout (`$GAP_SKILLS_PATH`, or a checkout next to the GaP repo); explicit
  relative paths resolve against the config file's directory.

[grocery_acceptance_smoke.yaml](gh-engine:examples/benchmark/grocery_acceptance_smoke.yaml)
is the same recipe shrunk to tasks 0–1 × 10 trials — same prompts and
`objects:` blocks, smaller worker budget. Run it before committing to the
full gate.

## What gets generated

[sample_generated_graph/](gh-engine:examples/grocery_fulfillment/sample_generated_graph)
is a `gap.agent.generate` output for task 0 ("Pick the blue
and yellow alphabet soup can and place it in the basket"). The coordinator
decomposed the task into four subgraphs:

| Subgraph | Skill | What it does |
|---|---|---|
| `target_sg` | `perceiving-objects` | observe → DINO + VLM-tournament + SAM3 perception script → `geometry.filter_and_compute_obb` for the soup can |
| `container_sg` | `perceiving-objects` | same pipeline aimed at the basket |
| `grasp_sg` | `grasping-with-planner` | open gripper → top-down grasp candidates → axis-locked straight-Z descend (collision-aware planner fallback) → observe → close |
| `transport_sg` | `transporting-objects` | compute drop pose → waypoint move above basket → descend and release |

The subgraph agents authored the inner state machines plus the helper scripts
under `scripts/` (perception, grasp descend, drop-pose computation), and a
checkpoint agent attached `validate=True` postconditions that
[checkpoint enforcement](../running/checkpoints.md) checks against
simulator ground truth at every subgraph exit.

The grasp script (`scripts/grasp_sg/grasp_descend_linear.py`) carries the same
tuned recipe as [Grocery Packing](grocery-packing.md)'s `grasp_move.py`,
distilled to the single-object case: rise, translate in XY over the object,
then a pure-vertical, orientation-locked descend via
`curobo.plan_directed_linear` (`allowed_axes=["Z"]`,
`orientation_mode="LOCK"`) — with a hand-off to the collision-aware CuRobo
planner when the straight-line solve is infeasible. Unlike the planner's
goalset, the straight-Z fast path also grips a *flat* object (the cream-cheese
box) by simply lowering onto it, and its grip depth is floored above the
object base so the fingers never strike the table.

Run the sample graph directly — no generation needed (still needs the
perception VLM credential):

```bash
MUJOCO_GL=egl uv run gap run examples/grocery_fulfillment/sample_generated_graph \
  --sim libero_object_all_variance/0 --checkpoints warn
```

<figure>
  <video src="../_static/grocery_fulfillment_sim.mp4" autoplay loop muted playsinline width="70%"></video>
  <figcaption>The sample graph running on <code>libero_object_all_variance/0</code>:
  perceive the described can and the basket, straight-Z grasp, transport,
  release — recorded by <code>gap run</code>'s default video capture.</figcaption>
</figure>

### Checkpoint sidecars

Each subgraph gets a generated sidecar under `checkpoints/` ("Do not edit" —
the harness re-executes the original builder block to capture the predicate
lambdas with their closures). The perception subgraphs verify their OBBs
against the privileged object poses; from
[checkpoints/grasp_sg.py](gh-engine:examples/grocery_fulfillment/sample_generated_graph/checkpoints/grasp_sg.py):

```python
sg.add_checkpoint('grasp_pose_above_table',
    predicate=lambda w, o: o['grasp_pose']['position']['z'] > 0.01,
    rationale='planned grasp z-height is above the tabletop', validate=True)
sg.add_checkpoint('target_held',
    predicate=lambda w: w.body('alphabet soup').is_grasped(),
    rationale='the alphabet soup can is held by the robot gripper', validate=True)
```

and from `checkpoints/transport_sg.py`:

```python
sg.add_checkpoint('target_in_container',
    predicate=lambda w: w.body('alphabet soup').is_in(w.body('basket')),
    rationale='alphabet soup settled inside the basket after release', validate=True)
```

Predicates take the privileged world snapshot `w`
(`w.body(name).position` / `.is_grasped()` / `.is_in()` / `.interior_lower` /
`.interior_upper`) and optionally the subgraph's bound outputs `o`.
`gap.execute(..., checkpoints="warn")` logs violations;
`checkpoints="raise"` turns them into failures. The sidecars also double as a
readable record of the `gap.builder` API the generator emits — see
[Builder](../authoring/builder.md).

## Next steps

- [Grocery Packing](grocery-packing.md) — the pack-everything loop whose
  grasp/transport recipes this family shares.
- [Benchmark Grids](benchmark.md) — all benchmark configs, gate semantics,
  and resume behavior.
- [Generate a Graph](generate-a-graph.md) — the generation pipeline this
  benchmark drives at grid scale, run once interactively.
- [Benchmarking](../benchmarks/benchmarking.md) — the full benchmarking
  guide.
- [Checkpoints](../running/checkpoints.md) — how `validate=True`
  postconditions are enforced at runtime.
