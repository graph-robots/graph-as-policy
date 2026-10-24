# grocery_packing — pack EVERY object into the basket with a loop

> **What:** a pick-and-place graph with a real **backward edge** — loop until
> every grocery item is in the basket · **Needs:** `uv sync` + `gap skills
> install --all` (CUDA) + GPU + LLM key · **Time:** ~min/item · **Artifact:**
> a static `workflow.json` (v3) you run directly.

Where [build_a_graph](../build_a_graph/) picks one described object, this
example **loops**: perceive the next object on the table, grasp it with the
planner recipe, transport it into the basket, then route **back** to perception
and repeat — until the benchmark reports every item delivered.

This is a self-contained static graph: `packing_graph/workflow.json` plus the
`scripts/` it references. There is no build step — load and run it like any
other v3 graph.

```
packing_graph/
├── workflow.json            # the v3 graph: 3 subgraphs + done/abort ends
└── scripts/
    ├── perceive_dino_vlm.py # DINO + VLM tournament + SAM3 → world-frame cloud
    ├── exterior_view.py     # keep only the agentview cam (drop the wrist)
    ├── route_next_object.py # loop control: all-packed → done, else found/none
    ├── grasp_move.py        # VAB-style grasp approach (rise → XY → straight-Z)
    ├── transport_move.py    # VAB-style transport into the basket (straight-Z)
    └── place_release.py     # open gripper + linear retract
```

## The loop

```
START ──→ perceive_next ──found──→ grasp ──grasped──→ transport ──┐
            │  │ none                 │ failed           │ blocked │ placed
  not_found │  ↓                      ↓                  │         │
            ↓  done (clean           abort ←─────────────┘         │
          abort  success exit)                                     │
            └──────────────── perceive_next ←──────────────────────┘
                    THE BACKWARD EDGE (transport → perceive_next)
```

`transport --placed--> perceive_next` is a genuine cycle. The executor treats a
conditional edge that resolves to an already-completed node as a loop: it resets
the loop body (`perceive_next`, `grasp`, `transport`) and re-runs it. The
cross-subgraph store keeps the most-recent producer, so each iteration grasps
the freshly-perceived `target_obb` and places into the freshly-perceived
`container_obb` — the loop head re-localizes **both** the next item and the
basket every pass. `GAP_ITERATION_CAP` bounds runaway loops. See
`docs/runtime.md` §7.2 for the back-edge semantics.

## The subgraphs

- **perceive_next_sg** (`perceiving-objects`) is the **loop head**, and it
  localizes *both* ends of the pick on **every iteration**:
  `get_observation → exterior_view → perceive("basket") →
  filter_and_compute_obb → perceive("grocery item") → decide`. There is no
  separate run-once container subgraph — the basket (`container_obb`) is
  re-perceived each pass alongside the next item, so the place pose always
  reflects the current scene (at the cost of one extra perceive per
  iteration). Two things keep the item pick honest:
  - The perceive node passes an `object_description` — *"a packaged grocery
    product such as a can, box, carton, jar, or bottle; never the wicker basket
    or storage container"* — so the VLM pairwise tournament prefers a real item
    over the basket whenever one is on the table.
  - `decide` (`route_next_object.py`) is the loop's stop signal — a per-pass
    **VLM completion check**, not a privileged simulator verdict (see below).

- **grasp_sg** — `open → top_down_grasp_candidates → grasp_move → observe →
  close`. `grasp_move.py` is the **VAB approach recipe**: rise to a hover height
  (straight-line cartesian), translate in XY over the object (cartesian), then
  descend straight down onto it with cuRobo's **axis-constrained** linear plan
  (`plan_directed_linear`, `allowed_axes=["Z"]`, `orientation_mode="LOCK"`) — a
  guaranteed vertical drop, not a curved IK path.

- **transport_sg** — `transport_move → release`. `transport_move.py` lifts,
  translates in XY over the basket, then does the same axis-locked straight-Z
  descend *into* the basket; `place_release.py` opens the gripper and retracts
  linearly.

## Termination — unprivileged, VLM-verified

`route_next_object.py` needs no privileged simulator signal — the same policy
runs unchanged on a real robot. Signals are layered:

1. **VLM completion check (primary).** Every pass, the exterior frame goes to
   the VLM: *"have ALL the grocery items been placed inside the basket?"* A
   confident YES → **`none` → done (success)**. The check only ever forces a
   STOP — a NO (or an unavailable VLM) never forces the loop to continue, so
   the guards below still guarantee termination.
2. **Env verdict (secondary, sim-only backstop).** `sim.check_success` is
   polled; `task_completed` reads the success the env cached in its own
   `step()` — it never re-evaluates the stateful `pack_all_into` predicate
   (which teleports delivered objects on each evaluation). Wrapped so a
   non-sim connector falls through cleanly.
3. **No-progress guard.** Re-perceiving the *same* target (cloud centroid
   within 3 cm) three passes in a row means the last grasp+transport cycle
   changed nothing (a delivery removes its item) → `none`. A 30-pass budget
   backstops pathological alternation. Guard state is keyed per trace dir so
   benchmark workers never leak one trial's state into the next.
4. **Perception.** Otherwise: an item was returned → `found` (grasp it);
   nothing → `none`.

`none` is a normal exit (not `on_error`), so finishing never looks like a
failure.

## The packing benchmark (teleport-on-delivery)

The default task is the **Variational-Automation-Benchmark** pack-all suite
(`libero_object_packing`). The moment an object settles inside the basket, the
`pack_all_into` predicate marks it *delivered* and the env **teleports it to a
graveyard pose** (out of the scene), scoring `completion_rate = delivered /
total`. Each delivered object disappears from the table, so the next
`perceive_next` finds the *next* object; when all are packed `task_completed`
fires and the loop exits. The graph stays a pure policy (perceive / grasp /
transport); the *benchmark* owns the teleport and the score.

## Execute it

Needs the `perceiving-objects` weights, CuRobo (`CUDA_HOME=/usr/local/cuda uv run gap skills install curobo`), a
VLM credential, and the executor's back-edge support (in this checkout):

```bash
# Runs against the VAB pack-all suite; a run video is recorded by default
# to <trace-dir>/run_video.mp4 (pass --no-video to skip it).
MUJOCO_GL=egl uv run gap run examples/grocery_packing/packing_graph \
    --sim libero_object_packing/0
uv run gap viz     # browse the recorded trace — perceive_next is visited once per object
```

The `open-robot-skills` checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
sibling checkout); pass `--skills /path/to/open-robot-skills` to override. The
VLM provider/model come from `GAP_VLM_PROVIDER` / `GAP_VLM_MODEL` (+ the
provider's credentials).

On the default arrangement the loop delivers the items one per pass and
exits cleanly the iteration after the last delivery. As with the other
CuRobo examples, the first run of a session pays one-time costs (cold
vision-model loads + CuRobo's CUDA-kernel JIT, ~40 s) before per-iteration
timing settles.

## Generate it yourself

The static graph above is also what `gap generate` produces from the task
sentence — the `perceiving-next-item` / `grasping-with-planner` /
`transporting-objects` skills carry the same tuned recipes as this example's
scripts, so the generated loop matches this one in structure and behavior:

```bash
# LLM codegen (needs an LLM credential; Vertex shown — OpenRouter also works):
export GAP_LLM_PROVIDER=vertex GAP_LLM_MODEL=gemini-3.1-pro-preview \
       GOOGLE_CLOUD_PROJECT=<your-project>       # + `gcloud auth application-default login`
uv run gap generate "Pick all the objects and place them in the basket" \
    --provider vertex --model gemini-3.1-pro-preview

# Run the generated graph on the same suite:
MUJOCO_GL=egl uv run gap run outputs/generated_<timestamp>/task_00 \
    --sim libero_object_packing/0
```
**Benchmarking note:** `gap benchmark` applies per-trial safety guards sized
for a *single* pick (`max_perception_calls: 50` etc.). A six-item pack-all
loop does ~6× the work — raise them in the benchmark YAML or the workers kill
mid-loop episodes:

```yaml
safety_limits: {max_perception_calls: 400, max_planning_calls: 200, max_sim_steps: 40000}
```

## Where to go next

- [build_a_graph](../build_a_graph/) — the single-object version of this graph.
- [grocery_fulfillment](../grocery_fulfillment/) — the LLM-generated acceptance
  benchmark whose grasp/transport recipes this example mirrors.
- `docs/runtime.md` §7.2 — the loop (backward-edge) semantics the executor
  implements.
