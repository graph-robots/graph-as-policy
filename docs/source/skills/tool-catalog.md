# Tool Catalog

The [open-robot-skills](gh-skills:.) registry ships seven tool bundles
under `tools/`. Each bundle is a directory with a `SKILL.md` (frontmatter
declares the tool list and a `gap.requires` capability block) and a
`tools.py` of `@tool`-decorated functions. Heavy model code lives in
private modules and is imported lazily — importing a bundle never pulls
torch. Tool I/O uses numpy arrays plus the `gap.types` types
(`Se3Pose`, `JointState`, `Trajectory`, `WorldConfig`, `Mask`,
`BoundingBox2D`, `OrientedBoundingBox`, `PointCloud`, ...); failures
raise `gap.errors.ToolError`, `PlanningFailed`, or `PerceptionFailed`.

Each bundle declares a pip extra named after itself, so installs are
one resolver run:

```bash
uv run gap skills install sam3 grounding-dino geometry
```

:::{note} Requirements
Requirements vary per bundle: `sam3` and `grounding-dino` need a CUDA
GPU and downloadable weights, `curobo` needs a CUDA toolkit at install
time, `gemini-er` and `vlm` need API keys, `molmo` needs a self-hosted
vLLM endpoint, and `geometry` needs nothing. `gap check` reports which
bundles are runnable from each bundle's `gap.requires` declaration.
:::

| Bundle | Category | Tools | `gap.requires` | Env vars (default) |
|---|---|---|---|---|
| [sam3](#sam3) | perception | 6 | `gpu`, `weights` | `GAP_SAM3_DEVICE` (`cuda`) |
| [grounding-dino](#grounding-dino) | perception | 1 | `gpu`, `weights` | `GAP_DINO_DEVICE` (`cuda`), `GAP_DINO_MODEL` (`IDEA-Research/grounding-dino-base`) |
| [gemini-er](#gemini-er) | perception | 1 | `env_any: [GOOGLE_API_KEY, GEMINI_API_KEY]` | `GAP_GEMINI_ER_MODEL` (`gemini-robotics-er-1.5-preview`) |
| [molmo](#molmo) | perception | 3 | `env: [GAP_MOLMO_BASE_URL]` | `GAP_MOLMO_BASE_URL` (required), `GAP_MOLMO_MODEL` (`allenai/Molmo2-8B`) |
| [vlm](#vlm) | perception | 2 | `env_any: [OPENROUTER_API_KEY, GAP_VLM_API_KEY, GAP_VLM_PROJECT_ID]` | `GAP_VLM_*` — see below |
| [curobo](#curobo) | planning | 10 | `gpu` | — (`CUDA_HOME` at install time) |
| [geometry](#geometry) | perception | 18 | none (CPU, no weights) | — |

## sam3

Segment Anything 3: text-, point-, and box-prompted instance
segmentation plus a stateful streaming video tracker.
Source: [tools/sam3](gh-skills:tools/sam3).

| Tool | Signature | Returns |
|---|---|---|
| `sam3.segment_text` | `(image, query, max_results=5)` | `{masks, scores, boxes}` score-sorted best-first; `max_results <= 0` returns everything |
| `sam3.segment_point` | `(image, pixel_x, pixel_y)` | `{masks, scores}` (multimask, best-first) |
| `sam3.segment_box` | `(image, box, pixel_x=0.0, pixel_y=0.0, use_point=False)` | `{masks, scores}`; with `use_point=True` the pixel refines the box prompt |
| `sam3.tracker_init` | `(image, text="", box=None, pixel_x=0.0, pixel_y=0.0, use_point=False, object_name="")` | `{tracker_id, initial_mask, initial_box, score, object_present}` |
| `sam3.tracker_update` | `(tracker_id, image)` | `{mask, box, confidence, object_present}` |
| `sam3.tracker_close` | `(tracker_id)` | `{closed}` — idempotent |

**Requirements.** `gap.requires: {gpu, weights}`. Extra `sam3` installs
torch, torchvision, the upstream `sam3` package pinned to `b26a5f3`,
plus `decord2` and `pycocotools`. Weights download on the first model
build. `GAP_SAM3_DEVICE` selects the device (default `cuda`); the image
model also runs on `cpu` (slowly), the video tracker is CUDA-only in
practice. The upstream package's metadata pins `numpy==1.26` — the
repo's `uv.lock` relaxes it, but plain pip may downgrade numpy.

**Behavior.**

- Tracker prompt precedence is **box > point > text**; a point prompt is
  converted to a small box (10% of image). No prompt at all raises
  `ToolError`; a prompt that finds nothing returns
  `object_present=False` with an empty `tracker_id` (no exception).
- **Sessions and TTL**: tracker sessions idle longer than 120 s are
  evicted lazily on the next tracker call; an unknown or evicted
  `tracker_id` raises `ToolError`.
- **Drift handling** in `tracker_update`: a mask-area jump > 1.5× the
  running median, or confidence < 0.30, holds the **last good mask**
  with `confidence=0.0` and `object_present=True` ("drifting — skip
  this frame"); after 5 consecutive drift hits `object_present=False`
  and the caller should re-init.
- The video tracker JIT-compiles Triton NMS kernels via the `CC` env
  var; the bundle forces `CC` to a real compiler before tracker use
  (a stale `CC` otherwise surfaces as `FileNotFoundError` in
  `tracker_init`).

## grounding-dino

Zero-shot 2D detection (`IDEA-Research/grounding-dino-base` via
transformers). Source: [tools/grounding-dino](gh-skills:tools/grounding-dino).

| Tool | Signature | Returns |
|---|---|---|
| `grounding-dino.detect` | `(image, query, box_threshold=0.20, text_threshold=0.20)` | `{detections: [{box, label, score}]}` |

**Requirements.** `gap.requires: {gpu, weights}`. Extra
`grounding-dino` installs `torch>=2.7` and `transformers>=4.41`; weights
download from HuggingFace on first call. `GAP_DINO_DEVICE` (default
`cuda`; CPU works but is slow) and `GAP_DINO_MODEL` (default
`IDEA-Research/grounding-dino-base`). The module also exposes a
`weights_cached()` probe so `gap check` can report weight presence
without downloading.

**Behavior.** The text encoder expects period-terminated phrases
(`"red cube. green cube."`); a missing final period is appended
automatically, but separate multiple objects yourself. The 0.20/0.20
defaults are deliberately low for recall; zero or negative thresholds
fall back to the defaults. `label` strings are the matched text spans,
not your full query. Empty `detections` means not-found — don't retry
blindly with the same prompt.

## gemini-er

Hosted open-vocabulary 2D detection on Gemini Robotics-ER. Zero GPU.
Source: [tools/gemini-er](gh-skills:tools/gemini-er).

| Tool | Signature | Returns |
|---|---|---|
| `gemini-er.detect` | `(image, query, model=None)` | `{detections: [{box, label, score}]}` |

**Requirements.** `gap.requires: {env_any: [GOOGLE_API_KEY,
GEMINI_API_KEY]}`. Extra `gemini-er` installs `google-genai`.
`GAP_GEMINI_ER_MODEL` overrides the model (default
`gemini-robotics-er-1.5-preview`).

**Behavior.** The model emits Gemini's `box_2d` convention
(`[ymin, xmin, ymax, xmax]` normalized 0–1000); the tool converts to
pixel-space `{x1, y1, x2, y2}` and clamps to image bounds. `score`
defaults to 1.0 when the model reports none. No match (or unparseable
output) returns an empty list, never an error. This is the hosted,
GPU-free alternative to `grounding-dino.detect` and
`molmo.point_prompt`.

## molmo

Visual pointing and Q&A against a **self-hosted** Molmo vLLM endpoint
(OpenAI-compatible). The bundle itself is zero-GPU (httpx client only),
but Molmo has no hosted API. Source: [tools/molmo](gh-skills:tools/molmo).

| Tool | Signature | Returns |
|---|---|---|
| `molmo.point_prompt` | `(image, query)` | `{pixel_x, pixel_y, found}` — first detected point; `found=False` with zeros when unparseable |
| `molmo.query` | `(prompt, image=None)` | `{text}` |
| `molmo.query_yes_no` | `(prompt, image=None)` | `{answer, text}` |

**Requirements.** `gap.requires: {env: [GAP_MOLMO_BASE_URL]}`; no local
deps. `GAP_MOLMO_MODEL` selects the served model (default
`allenai/Molmo2-8B`). The `SKILL.md` ships a vLLM hosting recipe:

```bash
CUDA_VISIBLE_DEVICES=0 PYTHONNOUSERSITE=1 \
  python -m vllm.entrypoints.openai.api_server \
  --model allenai/Molmo2-8B --trust-remote-code --dtype bfloat16 \
  --port 8122 --gpu-memory-utilization 0.5 \
  --max-model-len 4096 --max-num-batched-tokens 4096

export GAP_MOLMO_BASE_URL=http://127.0.0.1:8122/v1
```

On a dedicated GPU, `--gpu-memory-utilization 0.85
--max-num-batched-tokens 8192` gives roughly 2× perception throughput.

**Behavior.** `point_prompt` sends the canonical `"Point at <query>"`
prompt and parses all four Molmo point output formats (Molmo2
`<points coords=...>`, Molmo1 `<point x= y=>`, legacy `<points x1=
y1=>`, plain `x, y`), converting normalized coordinates to pixels.
`query_yes_no` coerces with the verbatim substring rule: `answer` is
true iff `"yes"` appears in the lowercased reply. A backend unreachable
after 3 retries (exponential backoff) raises `ToolError`.

## vlm

Free-form and yes/no visual Q&A against a hosted VLM, behind a
two-provider switch. Zero GPU. Source: [tools/vlm](gh-skills:tools/vlm).

| Tool | Signature | Returns |
|---|---|---|
| `vlm.query` | `(prompt, image=None, images=None, provider=None, model=None)` | `{text}` |
| `vlm.query_yes_no` | `(prompt, image=None, images=None, provider=None, model=None)` | `{answer, text}` |

**Requirements.** `gap.requires: {env_any: [OPENROUTER_API_KEY,
GAP_VLM_API_KEY, GAP_VLM_PROJECT_ID]}` — one provider must be
configured. Six env vars:

| Env var | Meaning | Default |
|---|---|---|
| `GAP_VLM_PROVIDER` | `openrouter` \| `vertex` | `openrouter` |
| `GAP_VLM_MODEL` | Model name (OpenRouter accepts Gemini slugs with or without the `google/` prefix) | `gemini-3.1-flash-lite-preview` |
| `GAP_VLM_BASE_URL` | OpenAI-compatible endpoint (openrouter provider) | `https://openrouter.ai/api/v1` |
| `GAP_VLM_API_KEY` | API key for the openrouter provider (or `OPENROUTER_API_KEY`) | — |
| `GAP_VLM_PROJECT_ID` | GCP project (vertex provider) | — |
| `GAP_VLM_REGION` | Vertex region | `global` |

The `openrouter` provider talks to OpenRouter's OpenAI-compatible
chat-completions API (`OPENROUTER_API_KEY`, or `GAP_VLM_API_KEY`), and
points at any other OpenAI-compatible server (e.g. local vLLM) via
`GAP_VLM_BASE_URL`; the `vertex` provider routes GEMINI models through
google-genai and needs the engine's vertex extra
(`pip install "graph-as-policy[vertex]"`). An unknown provider name
raises `ToolError` listing the valid choices.

**Behavior.** All providers pin temperature 0.0 and max_tokens 1024;
there is no system-prompt knob by design. `images=` accepts multiple
context images. `query_yes_no` appends an explicit "Answer with the
single word YES or NO first" instruction and coerces via the first
standalone `yes`/`no` word (word-boundary regex), falling back to the
legacy substring check — without the instruction, affirmative prose
containing no literal "yes" gets silently mislabeled as `False` (the
documented cream-cheese failure mode that `molmo.query_yes_no`, which
keeps the verbatim substring rule, is still subject to).

## curobo

NVIDIA cuRobo motion planning: collision-free trajectories, attached-
object transport, constrained linear moves, IK, batch feasibility, and
trajectory validation. Source: [tools/curobo](gh-skills:tools/curobo).
Trajectories in/out are GaP `Trajectory` dicts; worlds are `WorldConfig`
dicts built with `geometry.build_world_config`.

**Requirements.** `gap.requires: {gpu: true}`. cuRobo JIT-compiles CUDA
extensions at install time — build isolation must be off and `CUDA_HOME`
must match your torch build:

```bash
export CUDA_HOME=/usr/local/cuda
uv sync --extra curobo    # (pip: pip install -e ".[curobo]" --no-build-isolation)
```

The extra pins `nvidia-curobo` to NVlabs SHA `4ea7736` (the "cuRoboV2
research release") plus `cuda-core[cu12]>=0.7`, `torch>=2.7`, and
`trimesh`. The first planner call per process pays JIT/warmup latency;
planner instances are cached (recreating them per call corrupts CUDA
graph state), and GPU access is serialized by a module lock.

**Signatures** (planning failure returns `success=False`; infrastructure
errors raise `PlanningFailed`):

```python
curobo.plan_to_grasp_poses(
    world_config, start_joint_position, grasp_poses: list[Se3Pose],
    robot_file="franka.yml", max_attempts=8, use_cuda_graph=False,
    position_threshold=0.01, rotation_threshold=0.05, position_threshold_z=0.01,
    grasp_pose_is_fingertip=True, grasp_z_clearance=0.005, num_ik_seeds=128,
    relax_orientation=False, use_grasp_approach=False, grasp_approach_offset=0.03,
    grasp_approach_linear_axis=2, grasp_approach_tstep_fraction=0.7,
    use_world_collision=True, robot_collision_sphere_buffer=-0.01,
    collision_activation_distance=0.001, ignore_obstacle_names=None,
    debug_out_dir=None,
) -> {"success", "trajectory", "goalset_index"}

curobo.plan_with_grasped_object(
    world_config, start_joint_position, target_pose, object_name,
    robot_file="franka.yml", max_attempts=8, use_cuda_graph=False,
    position_threshold=0.05, rotation_threshold=0.1, position_threshold_z=0.05,
    num_ik_seeds=128, use_world_collision=True,
    robot_collision_sphere_buffer=-0.01, collision_activation_distance=0.01,
    surface_sphere_radius=0.001, link_name="attached_object",
    remove_obstacles_from_world=False, debug_out_dir=None,
) -> {"success", "trajectory"}

curobo.plan_linear(
    start_pose, end_pose, start_joint_position,
    robot_file="franka.yml", hold_vec_weight=None,
) -> {"success", "trajectory", "failure_reason"}

curobo.plan_directed_linear(
    start_joint_position, start_pose=None, target_pose=None,
    allowed_axes=None, explicit_direction=None, distance=None,
    endpoint_mode="PROJECT_TO_TARGET", orientation_mode="LOCK",
    orientation_target=None, robot_file="franka.yml",
) -> {"success", "trajectory", "failure_reason"}

curobo.plan_grasp_motion(
    start_joint_position, grasp_pose, approach_axis="y",
    approach_distance=0.12, approach_in_tool_frame=False,
    lift_axis="y", lift_distance=0.20, lift_in_tool_frame=False,
    robot_file="franka.yml",
) -> {"success", "approach_trajectory", "grasp_trajectory",
      "lift_trajectory", "failure_reason"}

curobo.plan_to_pose(
    target_pose, start_joint_position, robot_file="franka.yml",
    tcp_offset=None, world_config=None, position_threshold=0.005,
    rotation_threshold=0.05, max_attempts=4,
) -> {"success", "trajectory"}

curobo.solve_ik(
    target_pose, seed_config=None, robot_file="franka.yml",
    tcp_offset=None, num_seeds=32, position_threshold=0.005,
    rotation_threshold=0.05,
) -> {"success", "joint_config"}

curobo.batch_grasp_feasibility(
    world_config, start_state, grasp_poses, grasp_pose_is_fingertip=True,
    approach_offset_m=0.10, num_corridor_samples=5, num_ik_seeds=32,
    position_threshold=0.005, rotation_threshold=0.05,
    collision_activation_distance=0.01, robot_collision_sphere_buffer=None,
    robot_file="franka.yml", ignore_obstacle_names=None,
) -> {"feasible", "grasp_ik_ok", "approach_ik_ok",
      "corridor_collision_fraction"}    # all aligned with input order

curobo.validate_joint_trajectory_robot(
    world_config, trajectory, robot_file="franka.yml", use_cuda_graph=False,
    robot_collision_sphere_buffer=-0.01, collision_activation_distance=0.01,
    ignore_obstacle_names=None,
) -> {"success", "failure_reason", "first_collision_waypoint",  # -1 if none
      "collision_status_detail"}

curobo.validate_joint_trajectory_grasped(
    world_config, trajectory, object_name, robot_file="franka.yml",
    use_cuda_graph=False, robot_collision_sphere_buffer=-0.01,
    collision_activation_distance=0.01, surface_sphere_radius=0.001,
    link_name="attached_object", remove_obstacles_from_world=False,
) -> {"success", "failure_reason", "first_collision_waypoint",
      "collision_status_detail"}
```

**Behavior.**

- **Version split.** The planners (`plan_to_grasp_poses`,
  `plan_with_grasped_object`, `plan_linear`, `plan_directed_linear`,
  `plan_to_pose`) target the cuRobo **v0.8** MotionPlanner API, and
  `plan_grasp_motion` requires v0.8. `solve_ik` and
  `batch_grasp_feasibility` are built on the **v0.7** IKSolver API that
  v0.8 removed — on a v0.8-only install they raise `PlanningFailed`;
  plan via the goalset tools instead. The validators use the v0.7
  MotionGen path.
- **Frames and the fingertip offset.** Grasp/target poses are in the
  robot-base frame. With `grasp_pose_is_fingertip=True` (the default)
  grasp positions are fingertip-pad centers, converted solver-side to
  the `panda_hand` frame with the 0.1029 m offset along hand Z.
- Always pass the grasp target's mesh name in `ignore_obstacle_names`
  for `plan_to_grasp_poses` / `batch_grasp_feasibility` — closing on the
  target is not a collision.
- `robot_collision_sphere_buffer=-0.01` is intentionally negative: it
  shrinks the robot collision spheres 1 cm to reduce IK failures
  against dense perception meshes.
- `use_cuda_graph` must stay `False` for the validators, and
  `validate_joint_trajectory_grasped` always invalidates the planner
  cache afterward so the attachment cannot leak — expect the next
  planning call to re-create the planner.
- Set `debug_out_dir` on the grasp/transport planners to dump world and
  robot-sphere OBJ/PLY artifacts on failure; plan results (success and
  failure) are appended to an `events.jsonl` log in the same place. The
  value is treated as a trace directory — artifacts land under
  `<debug_out_dir>/data/gap/skills/curobo/`; with no value the fallback
  is `./curobo_debug*` in the working directory.

## geometry

Pure-math CPU geometry: back-projection, transforms, OBB fitting, grasp
derivation, and collision-world reconstruction. No GPU, no weights.
Source: [tools/geometry](gh-skills:tools/geometry). Extra `geometry`
installs `open3d>=0.18` and `scikit-learn>=1.3` (cv2/scipy come with GaP
core); the module imports lazily, so light tools work without the extra.

| Tool | Signature | Returns |
|---|---|---|
| `geometry.depth_to_point_cloud` | `(depth, intrinsics)` | `{points}` |
| `geometry.mask_to_world_points` | `(mask, depth, intrinsics, camera_pose)` | `{points}` — keeps depths in [0.015, 20.0] m |
| `geometry.pixel_to_world_point` | `(pixel_x, pixel_y, depth, intrinsics, camera_pose)` | `{point}`; `ToolError` on OOB pixel / invalid depth |
| `geometry.transform_points` | `(points, transform)` | `{points}` |
| `geometry.exclude_robot_points` | `(points, joint_positions, distance_threshold=0.05)` | `{points}` — 7-DOF Franka only; other arms pass through |
| `geometry.filter_noise` | `(points, eps=0.005, min_samples=10)` | `{points}` — returns the original cloud if DBSCAN labels everything noise |
| `geometry.compute_obb` | `(points)` | `{obb}`; `PerceptionFailed` on < 4 points |
| `geometry.filter_and_compute_obb` | `(points, eps=0.005, min_samples=10)` | `{obb}` |
| `geometry.top_down_grasp_from_obb` | `(obb, z_offset=0.0)` | `{pose}` |
| `geometry.top_down_grasp_candidates` | `(obb, z_offset=-0.04)` | `{candidates: {poses}}` |
| `geometry.select_top_down_grasp` | `(grasp_poses, gripper_position=None)` | `{pose}` |
| `geometry.front_grasp_from_obb` | `(obb, approach_offset=0.08, approach_hint=None, z_offset=0.0)` | `{grasp_pose, pre_grasp_pose, approach_direction, slide_axis}`; `PlanningFailed` when the approach is near-vertical |
| `geometry.build_world_config` | `(cameras, object_masks=None, voxel_size=0.005, noise_eps=0.01, noise_min_samples=5, table_z_threshold=0.0, mesh_alpha=0.03, robot_joint_state=None, robot_distance_threshold=0.15, robot_file="franka.yml", target_obb=None, target_obb_name="target")` | `{config, mesh_names}` |
| `geometry.rotate_quat_z90` | `(quat)` | `{quat}` |
| `geometry.compute_drop_position` | `(container_obb, clearance=0.05, object_z_extent=0.0)` | `{position}` |
| `geometry.compute_xy_distance` | `(point_a, point_b)` | `{distance}` |
| `geometry.iou` | `(box_a, box_b)` — `[x1, y1, x2, y2]` boxes | `{iou}` |
| `geometry.pose_distance` | `(a, b)` — `[x, y, z]` points | `{distance}` |

**Behavior.**

- **OBB `extent` holds HALF-extents** (gap.types convention).
  `compute_obb` is upright-only — rotation only around world Z — and
  extents use the 2nd/98th percentile of points, not strict min/max.
  Single-camera 2.5D clouds bias OBB centers a few cm toward the camera.
- `top_down_grasp_candidates` defaults to `z_offset=-0.04` (fingertip
  4 cm below the OBB top); with `0.0` the fingers close *above* the
  object — a silent empty grip. Grasp Z is clamped to −0.05 m (the
  table-clearance floor; the LIBERO table top is at world z = 0).
  `poses[0]` is the canonical primary, `poses[1]` its 90°-rotated
  alternate, followed by the enriched fan for a planner goalset.
- `top_down_grasp_from_obb` yaw is **not** derived from the OBB —
  fingers may close across the wide axis; use the candidate fan when
  orientation matters.
- `build_world_config`: table removal only runs when
  `table_z_threshold != 0` (typical −0.01); robot-point exclusion is
  Franka-only and skips non-7-DOF joint states; prefer explicit
  `object_masks` over the `target_obb` projection fallback (the
  projection is a corner-AABB approximation inflated by 2 cm). The
  excluded target is returned in `mesh_names` so planners can
  `ignore_obstacle_names` it.
- Output dicts are wrappers: bind `Ref("node.obb")`, `Ref("node.points")`,
  `Ref("node.pose")`, `Ref("node.position")` — never a bare `Ref("node")`.
  Cloud-consuming tools take `points=`, not `point_cloud=`.

## The canonical perception-to-grasp chain

The tool bundles compose into one recipe that every grasp workflow in
the [skill catalog](skill-catalog.md) instantiates:

```text
grounding-dino.detect / gemini-er.detect / molmo.point_prompt   # find the object in 2D
        │
        ▼
sam3.segment_box                       # pixel-accurate mask inside the detection box
        │
        ▼
geometry.mask_to_world_points          # back-project mask + depth to a world cloud
        │
        ▼
geometry.filter_and_compute_obb        # DBSCAN filter + upright OBB fit
        │
        ├──▶ geometry.top_down_grasp_candidates    # grasp goalset from the OBB
        │
        └──▶ geometry.build_world_config           # collision world (target excluded by mask)
                      │
                      ▼
curobo.plan_to_grasp_poses             # goalset plan; pass the target mesh name
                                       # in ignore_obstacle_names
```

Pick the detector for your platform — `grounding-dino` (local GPU),
`gemini-er` (hosted API, zero GPU), or `molmo.point_prompt`
(self-hosted pointing) — and the rest of the chain is identical. The
`vlm` tools slot in wherever semantic judgment is needed: target
disambiguation among detections, checkpoint verification, and policy
termination.

## See also

- [Skill catalog](skill-catalog.md) — the skill bundles that orchestrate
  these tools into subgraphs.
- [Skill registries](registries.md) — discovery, `gap check`, and the
  `gap tools` commands.
- [Authoring bundles](authoring-bundles.md) — writing your own tool
  bundle.
- [Connector tools](../reference/connector-tools.md) — the `robot.*` /
  `sim.*` tools the connector provides alongside these.
- [Environment variables](../reference/environment-variables.md) — the
  full env var reference.
