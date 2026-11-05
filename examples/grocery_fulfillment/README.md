# grocery_fulfillment — the acceptance benchmark

> **What:** The flagship acceptance family: graphs LLM-generated per task, nothing hand-written · **Needs:** `grocery` (CUDA) + LLM key · **Time:** minutes (smoke)

The flagship task family: pick a *described* grocery item and place it in the
basket, under the variational-automation benchmark's baked pose /
permutation / basket-swap variations. The graphs are **generated per task**
by `gap.agent.generate` (`llm_generation` mode); nothing is hand-written.
The full config doubles as the release gate: `--gate` makes `gap benchmark`
exit non-zero when a cell falls below the config's `gate_threshold`.

## Run it

```bash
# add --extra vertex: the configs below pin the vertex provider, whose SDK
# (google-genai) is engine-side for graph generation
uv sync --extra vertex && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo
uv run gap skills check --download

# the 20-trial smoke (tasks 0-1)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance_smoke.yaml --gate

# the full acceptance gate (interruptible; --resume continues)
MUJOCO_GL=egl uv run gap benchmark examples/benchmark/grocery_acceptance.yaml --gate --resume
```

You also need an LLM credential for graph generation — the configs pin
Vertex with `gemini-3.1-pro-preview`; `openrouter` works by switching the
config's `llm:` block.

## The recipe (ported verbatim)

[`examples/benchmark/grocery_acceptance.yaml`](../benchmark/grocery_acceptance.yaml):
one suite per task with per-task curated object descriptors
(`target` / `expected_label` / `shape_hint` — these hints are appended to
the codegen prompt and are load-bearing), a generous per-trial timeout
(flat-box grasp planning legitimately runs long when the goalset planner
falls back to per-pose iteration), video on.

## The sample graph

[`sample_generated_graph/`](sample_generated_graph) is a
`gap.agent.generate` output for task 0 ("Pick the blue and yellow alphabet
soup can and place it in the basket"): the coordinator decomposed the task
into two `perceiving-objects` subgraphs (target, then basket) →
`grasping-with-planner` → `transporting-objects`, the subgraph agents
authored the inner state machines + scripts, and the checkpoint agent
attached `validate=True` postconditions that
`gap.execute(checkpoints="warn"|"raise")` enforces against simulator ground
truth at every subgraph exit — perception OBBs within tolerance of the
privileged object poses, held-after-grasp, drop pose inside the basket
cavity, and settled-in-basket.

Its grasp subgraph (`grasp_sg`) carries the same tuned grasp recipe as
[grocery_packing](../grocery_packing/)'s `grasp_move.py`, distilled to the
single-object case (`scripts/grasp_sg/grasp_descend_linear.py`):

```
open → top_down_grasp_candidates → grasp_descend_linear → observe → close
```

- **Fast path (default):** rise, translate in XY over the object (rotating
  to the grasp yaw), then an **axis-locked straight-Z descend** onto it via
  `curobo.plan_directed_linear` (`allowed_axes=["Z"]`,
  `orientation_mode="LOCK"`) — pure-vertical with zero lateral drift. Unlike
  the planner's goalset, it also grips a *flat* object (e.g. the
  cream-cheese box) by simply lowering onto it.
- **Fallback:** if the straight-line solve is infeasible — a far-edge item
  where the fixed top-down wrist has no IK — it hands off to the
  collision-aware CuRobo planner, which searches the whole candidate fan
  for a reachable, collision-free wrist.
- **Depth:** a shallow grip near the perceived top, floored a hair above
  the object base so the fingers never strike the table — flat boxes still
  get a real mid-height grip, tall cartons don't get toppled by the palm.

Run it directly — no generation needed (still needs the perception VLM
credential):

```bash
MUJOCO_GL=egl uv run gap run examples/grocery_fulfillment/sample_generated_graph \
  --sim libero_object_all_variance/0 --checkpoints warn
```

A clean run perceives the can and the basket, grasps, places, and passes
all seven checkpoints. The first run of a session pays one-time costs —
cold vision-model loads plus CuRobo's CUDA-kernel JIT (~40 s) — so expect
a few extra minutes before per-node timing settles.

## Next steps

- [benchmark](../benchmark/) — all benchmark configs, the gate semantics,
  and resume behavior.
- [grocery_packing](../grocery_packing/) — the pack-everything loop whose
  grasp/transport recipes this family shares.
- [generate_a_graph](../generate_a_graph/) — the generation pipeline this
  benchmark drives at grid scale.
