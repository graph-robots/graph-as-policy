# Grocery Packing

:::{note} Requirements
A CUDA GPU (MuJoCo EGL rendering + CuRobo planning + perception weights), the
`grocery` dependency extra, and a VLM credential for perception and the
completion check (`GAP_VLM_PROVIDER` / `GAP_VLM_MODEL` + the provider's
credentials).
:::

Where the [quickstart](libero-quickstart.md) picks one described object, this
example **loops**: perceive the next object on the table, grasp it with the
planner recipe, transport it into the basket, then route **back** to
perception and repeat — until every grocery item is in the basket. It is the
executor's reference example for a graph with a real **backward edge**.

The example lives at [examples/grocery_packing](gh-engine:examples/grocery_packing) —
a self-contained static v3 graph (`packing_graph/workflow.json` plus the
`scripts/` it references). There is no build step: load and run it like any
other graph.

<figure>
  <video src="../_static/grocery_packing_sim.mp4" autoplay loop muted playsinline width="70%"></video>
  <figcaption>The packing loop on the pack-all suite
  (<code>libero_object_packing/0</code>), 2×: one perceive → grasp → transport
  pass per item, until the VLM completion check reports the table clear.</figcaption>
</figure>

## The loop

```text
START ──→ perceive_next ──found──→ grasp ──grasped──→ transport ──┐
            │  │ none                 │ failed           │ blocked │ placed
  not_found │  ↓                      ↓                  │         │
            ↓  done (clean           abort ←─────────────┘         │
          abort  success exit)                                     │
            └──────────────── perceive_next ←──────────────────────┘
                    THE BACKWARD EDGE (transport → perceive_next)
```

`transport --placed--> perceive_next` is a genuine cycle. The executor treats
a conditional edge that resolves to an already-completed node as a loop: it
resets the loop body (`perceive_next`, `grasp`, `transport`) and re-runs it.
The cross-subgraph store keeps the most-recent producer, so each iteration
grasps the freshly-perceived `target_obb` and places into the
freshly-perceived `container_obb` — the loop head re-localizes **both** the
next item and the basket every pass. `GAP_ITERATION_CAP` bounds runaway
loops; see [Graph patterns](../authoring/patterns.md) for the loop
(subgraph-revisit) semantics.

## The subgraphs

- **perceive_next_sg** is the **loop head**, and it localizes *both* ends of
  the pick on **every iteration**: `get_observation → exterior_view →
  perceive("basket") → filter_and_compute_obb → perceive("grocery item") →
  decide`. The basket is re-perceived each pass alongside the next item, so
  the place pose always reflects the current scene. Two things keep the item
  pick honest:
  - The perceive node passes an `object_description` — *"a packaged grocery
    product such as a can, box, carton, jar, or bottle; never the wicker
    basket or storage container"* — so the VLM pairwise tournament prefers a
    real item over the basket whenever one is on the table.
  - `decide` (`route_next_object.py`) is the loop's stop signal — a per-pass
    **VLM completion check**, not a privileged simulator verdict (see below).
- **grasp_sg** — `open → top_down_grasp_candidates → grasp_move → observe →
  close`. `grasp_move.py` is the tuned approach recipe: rise to a hover
  height, translate in XY over the object, then descend straight down onto it
  with cuRobo's **axis-constrained** linear plan (`plan_directed_linear`,
  `allowed_axes=["Z"]`, `orientation_mode="LOCK"`) — a guaranteed vertical
  drop, not a curved IK path.
- **transport_sg** — `transport_move → release`. `transport_move.py` lifts,
  translates in XY over the basket, then does the same axis-locked straight-Z
  descend *into* the basket; `place_release.py` opens the gripper and
  retracts linearly.

## Termination — unprivileged, VLM-verified

`route_next_object.py` needs no privileged simulator signal — the same policy
runs unchanged on a real robot. Signals are layered:

1. **VLM completion check (primary).** Every pass, the exterior frame goes to
   the VLM: *"have ALL the grocery items been placed inside the basket?"* A
   confident YES → `none` → done (success). The check only ever forces a
   STOP — a NO (or an unavailable VLM) never forces the loop to continue, so
   the guards below still guarantee termination.
2. **Env verdict (secondary, sim-only backstop).** `sim.check_success` is
   polled, wrapped so a non-sim connector falls through cleanly.
3. **No-progress guard.** Re-perceiving the *same* target (cloud centroid
   within 3 cm) three passes in a row means the last grasp+transport cycle
   changed nothing → `none`. A pass budget backstops pathological
   alternation.
4. **Perception.** Otherwise: an item was returned → `found` (grasp it);
   nothing → `none`.

`none` is a normal exit (not `on_error`), so finishing never looks like a
failure.

## Run it

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo
uv run gap skills check --download

# Runs against the VAB pack-all suite; a run video is recorded by default
# to <trace-dir>/run_video.mp4 (pass --no-video to skip it).
MUJOCO_GL=egl uv run gap run examples/grocery_packing/packing_graph \
    --sim libero_object_packing/0
uv run gap viz     # browse the trace — perceive_next is visited once per object
```

The default task is the variational-automation pack-all suite
(`libero_object_packing`): when an object settles inside the basket the env
marks it *delivered* and removes it from the scene, so the next
`perceive_next` pass finds the next object. The graph stays a pure policy
(perceive / grasp / transport); the *benchmark* owns the bookkeeping and the
score. On the default arrangement the loop delivers the items one per pass
and exits cleanly the iteration after the last delivery.

## Generate it yourself

The static graph above is also what `gap generate` produces from the task
sentence — the `perceiving-next-item` / `grasping-with-planner` /
`transporting-objects` skills carry the same tuned recipes as this example's
scripts, so the generated loop matches this one in structure and behavior:

```bash
uv run gap generate "Pick all the objects and place them in the basket" \
    --provider vertex --model gemini-3.1-pro-preview
MUJOCO_GL=egl uv run gap run outputs/generated_<timestamp>/task_00 \
    --sim libero_object_packing/0
```

:::{admonition} Benchmarking note
:class: note

`gap benchmark` applies per-trial safety guards sized for a *single* pick
(`max_perception_calls` etc.). A pack-all loop does several times the work —
raise the limits in the benchmark YAML or the workers kill mid-loop episodes:

```yaml
safety_limits: {max_perception_calls: 400, max_planning_calls: 200, max_sim_steps: 40000}
```
:::

## Next steps

- [Build a Graph](build-a-graph.md) — the single-object version of this
  graph, authored step by step.
- [Grocery Fulfillment](grocery-fulfillment.md) — the LLM-generated
  acceptance benchmark whose grasp/transport recipes this example mirrors.
- [Graph patterns](../authoring/patterns.md) — the loop (backward-edge)
  semantics the executor implements.
