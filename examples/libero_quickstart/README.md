# libero_quickstart — pick the soup can into the basket

> **What:** The end-to-end hero: real vision → OBB grasp → transport, ground-truth verified · **Needs:** `quickstart` + GPU + LLM key · **Time:** ~25–55 s/trial

The end-to-end quickstart for GaP: a static, fully-authored workflow graph
that perceives a target object and a container with real vision models
(Grounding DINO + SAM3 + a hosted VLM), derives a top-down grasp from the
fused 3D oriented bounding box, picks the object with a direct-IK
align-then-descend grasp, and places it in the basket — verified by the
sim's own success predicate and a ground-truth `target_held` checkpoint.

## Run it

From the GaP checkout (open-robot-skills cloned next to it):

```bash
uv sync && uv run gap skills install sam3 grounding-dino geometry            # engine + LIBERO sim + perception models
uv run gap skills check --download    # verify bundles + prefetch weights (~3.5 GB)
export OPENROUTER_API_KEY=...          # VLM provider — alternatives below

MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph \
  --sim libero_object_all_variance/0
uv run gap viz                        # browse the recorded trial
```

- The open-robot-skills checkout is auto-discovered (`$GAP_SKILLS_PATH` or the
  checkout next to the graph-as-policy checkout); `--skills PATH` overrides.
- `--validate-only` checks the graph without executing.
- `--checkpoints warn` (default) evaluates the ground-truth postcondition
  checkpoints in `graph/checkpoints/` after each subgraph (sim only) —
  `grasp_sg.target_held` verifies via privileged contacts that the gripper
  actually holds the can after `close`.
- Traces (per-node inputs/outputs, every model call, rendered frames) land
  in `outputs/run_<timestamp>/`.

## Two variants

| Graph | Grasp strategy | Needs |
|---|---|---|
| `graph/` (**default**) | direct-IK align-then-descend (`grasping-direct-ik` skill recipe): pre-rotate to the grasp yaw at a safe height above the OBB, descend straight down, close | `--extra quickstart` |
| `graph_planner/` | CuRobo goal-set planning over the same grasp candidates (`grasping-with-planner`): reconstruct a collision world from RGB-D, plan a collision-free joint trajectory to the best reachable candidate | `--extra grocery` (CUDA build, see below) |

`graph_planner/` is the faithful port of the dev tree's
`libero_pro/graph_cartesian_obb` example. The default `graph/` replaces only
its grasp subgraph with the planner-free recipe so the quickstart needs no
CuRobo install; perception and transport are identical.

`gap skills install sam3 grounding-dino geometry` pulls torch, torchvision, SAM3 (pinned git SHA),
Grounding DINO (transformers), and the CPU geometry deps — pinned by the
committed `uv.lock`. Model weights (`facebook/sam3`,
`IDEA-Research/grounding-dino-base`) download from HuggingFace on first
call. (pip equivalent: see the [main README](../../README.md#installation-details).)

## What can go wrong

When a trial fails, the trace almost always points at perception, not the
grasp mechanics:

- **Target mis-identification** — the DINO + VLM disambiguation picks the
  wrong detection, so the grasp executes cleanly at a wrong-object
  location. Shows up as a `target_held` checkpoint failure.
- **Container perception** — a degenerate or oversized basket OBB shifts
  the drop pose, so a held object is placed next to (or on the rim of) the
  basket. The planner variant shares the same perception subgraphs, so
  CuRobo does not recover these.

Wall-clock is ~25–55 s per trial on one A100 (models stay resident across
trials in one process; the **first** trial of a session pays the cold
model loads and can take a few minutes end-to-end).

## VLM provider

The perception pipeline disambiguates DINO detections with a hosted VLM
(`vlm.query`). The default provider is OpenRouter:

```bash
export OPENROUTER_API_KEY=...                  # default provider; model override:
export GAP_VLM_MODEL=gemini-3.1-pro-preview    # optional (unset = gemini-3.1-flash-lite-preview)
```

Alternative — Vertex AI with application-default credentials (what the
results above were measured with):

```bash
gcloud auth application-default login
export GAP_VLM_PROVIDER=vertex
export GAP_VLM_PROJECT_ID=<your-project>
export GAP_VLM_REGION=global
export GAP_VLM_MODEL=gemini-3.1-pro-preview
```

Any OpenAI-compatible endpoint (e.g. a local vLLM) also works — point
`GAP_VLM_BASE_URL` at it (with `GAP_VLM_MODEL`); the default `openrouter`
provider already speaks the OpenAI-compatible protocol.

## The planner variant

To run `graph_planner/`, install CuRobo first (CUDA build), then point
`gap run` at it:

```bash
uv sync && CUDA_HOME=/usr/local/cuda uv run gap skills install sam3 grounding-dino geometry curobo
MUJOCO_GL=egl uv run gap run examples/libero_quickstart/graph_planner \
  --sim libero_object_all_variance/0
```

## Graph anatomy

```
START → target → container → grasp → transport → done
            ↘ not_found      ↘ failed   ↘ blocked → abort (open gripper, go home)
```

- **target_sg** (`perceiving-objects`): a single DINO detect + pairwise-VLM
  crop tournament + SAM3 perception path on one observation, back-projected
  to a world-frame cloud, then `geometry.filter_and_compute_obb` fits the
  target's oriented bounding box.
- **container_sg** (same bundle): DINO+VLM perception of the basket →
  container OBB.
- **grasp_sg** (`grasping-direct-ik` in `graph/`): `robot.open_gripper` →
  `geometry.top_down_grasp_candidates(obb)` → compute the align pose 15 cm
  above the OBB top → `robot.go_to_pose` (rotate at altitude) →
  `robot.go_to_pose` straight down to `candidates.poses[0]` →
  `robot.close_gripper`.
  In `graph_planner/` this is instead: approach above the target →
  re-observe → `geometry.build_world_config` (collision scene from RGB-D,
  target carved out by its SAM3 mask) → `curobo.plan_to_grasp_poses` over
  the full candidate set → `robot.execute_trajectory`.
- **transport_sg** (`transporting-objects`): drop pose from the container
  OBB → lift + lateral `robot.go_to_pose` waypoints → descend, open,
  `robot.go_home`.

Scripts under `graph*/scripts/` are per-workflow copies of the canonical
bundle scripts in open-robot-skills (`skills/<bundle>/scripts/`); the subgraph's
`skill:` field names the owning bundle so scripts import under its package
(prompt loading, sibling imports).

## Next steps

- [build_a_graph](../build_a_graph/) — author this same graph in Python
  with `gap.builder`.
- [generate_a_graph](../generate_a_graph/) — let the LLM pipeline author it
  from one instruction.
- [grocery_fulfillment](../grocery_fulfillment/) — the generated-graph
  acceptance benchmark built on the same skills.
