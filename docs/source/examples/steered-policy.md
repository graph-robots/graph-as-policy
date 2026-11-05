# Steered Policy

:::{note} Requirements
A CUDA GPU, the `quickstart` and `policy` extras, an
[openpi](https://github.com/Physical-Intelligence/openpi) checkout at
`$GAP_OPENPI_DIR` to serve the π0.5 LIBERO checkpoint, and a VLM credential
for the perception and VLM-termination steps.
:::

[examples/steered_policy](gh-engine:examples/steered_policy) contains two
hybrid graphs that *steer* a learned VLA policy with geometric perception.
The division of labor: OBB perception localizes the target — robust even on
perturbed, out-of-distribution layouts — and a Cartesian approach
pre-positions the end-effector safely **above** the object. Only then is
control handed to the closed-loop VLA, which now starts from a pose close to
its training distribution instead of from wherever a perturbed reset left it.

- **`graph_loop/`** — the clean-all-items loop. `capture` records the
  episode-start arm pose; each iteration resets to it, perceives the next
  item (`perceiving-objects-oneshot` — a VLM answer of "none" cleanly ends
  the loop), approaches above its OBB, then the policy skill's `.run` tool
  (e.g. `pi05-libero.run`) picks *and places* the item, terminating when the
  commanded gripper completes a full open → close → open cycle.
- **`graph_grasp/`** — VLA-grasp + geometric-place split. The policy does
  only the dexterous grasp (terminated by a VLM held-and-lifted check), then
  `place_above_basket.py` lifts straight up, transports high, descends over
  the basket OBB, and releases. The VLA does the grasp it generalizes well;
  a geometric skill does the precise placement it does not.

These graphs are also what the benchmark's `llm_plus_policy` mode runs at
grid scale — see [Benchmark Grids](benchmark.md#policy-modes).

## Run it

```bash
uv sync --extra policy && uv run gap skills install sam3 grounding-dino geometry   # sim + perception + openpi websocket client
```

Serve a policy first. `gap policy serve` spawns the preset's server command
and blocks until Ctrl-C; the first run downloads the checkpoint (which is why
the default startup timeout is 900 s):

```bash
uv run gap policy serve pi05-libero --port 9100
```

Then run a graph. The workflows carry a `{{policy_id}}` placeholder that the
benchmark harness materializes per cell; for a standalone run, template it
first (for example, sed the placeholder in `workflow.json` to a policy-skill
name such as `pi05-libero`, which forms the `pi05-libero.run` tool) and pass
any remaining inputs with `--inputs`. The launcher auto-boots the named
skill's preset server, so the explicit `gap policy serve` above is only needed
when you want to share one server across runs:

```bash
MUJOCO_GL=egl uv run gap run examples/steered_policy/graph_loop \
    --sim libero_object_all_variance/0
```

See [Policies](../benchmarks/policies.md) for the full policy-serving guide,
including the per-checkpoint policy skills, custom checkpoints, and the
`policies:` overrides that point a skill at an external websocket server.

## Graph anatomy

`graph_loop`'s top level is a five-node loop with two end nodes:

```text
START → capture → target → approach → run → reset → target → ...
                     │
                     └─ not_found → done (success)
        any failure ────────────────→ abort (failure)
```

- `capture` records the episode-start joint configuration
  (`observe.arm_states.0.joint_state`) so every iteration's `reset` returns
  the arm to the same in-distribution starting pose
  (`robot.move_to_joints` with `tolerance: 0.005`, `max_steps: 800`, after
  `robot.open_gripper` with `settle_steps: 20`).
- `target` runs the `perceiving-objects-oneshot` skill: one DINO detection
  pass, a letter-labeled set-of-marks overlay, one `vlm.query`, then SAM
  segmentation and `geometry.filter_and_compute_obb`. Any failure — or the
  VLM answering "none" — routes to the subgraph's `on_error: "not_found"`
  exit, which the top level maps to `done`. **The error exit encodes loop
  success**: no items left means the table is clean.
- `abort` is an end node with a `recovery` tool list
  (`robot.open_gripper` with `settle_steps: 40`) so a failed run never ends
  holding an object.

### The policy node

The `run` subgraph declares a graph-scoped `ObservationStream` input and
calls the policy skill's `.run` tool. The template carries a `{{policy_id}}`
placeholder that the harness substitutes with a policy-**skill** name (e.g.
`pi05-libero`) to form the concrete tool — the skill owns its model, so there
is **no `policy_id` input**:

```json
{
  "type": "tool",
  "tool": "{{policy_id}}.run",
  "inputs": {
    "observation_stream": {"$ref": "in.observation_stream"},
    "prompt": "pick up the object and place it in the basket",
    "termination_prompt": "",
    "gripper_cycle_termination": true,
    "max_windows": 70,
    "replan_every": 5,
    "term_period": 6,
    "arm_id": 0,
    "vlm_camera": 0,
    "settle_steps": 10
  }
}
```

Three termination mechanisms are available, checked while the loop replans
every `replan_every` steps up to `max_windows` windows:

| Mechanism | Trigger |
|---|---|
| `gripper_cycle_termination: true` | the commanded gripper completes one open → close → open grasp/release cycle |
| `termination_prompt: "..."` | a VLM, queried every `term_period` windows on camera `vlm_camera`, answers yes |
| `max_windows` | hard budget exhausted |

`graph_grasp` uses the VLM mechanism for its grasp stage —
`termination_prompt: "Is the object held in the gripper and lifted off the
surface?"` — because a dedicated gripper-*grasp* detector
(`gripper_grasp_termination`) existed only on a dev branch; the ported engine
exposes cycle detection, VLM termination, and `max_windows`.

## Hand-off tuning

The scripts encode hard-won rules for keeping the VLA in-distribution at
hand-off; their docstrings carry the full rationale.

`approach_above_target.py` (identical in both graphs):

- **Preserve the current end-effector rotation.** Forcing a top-down
  quaternion pushes the π0.5 proprio state out of distribution; the policy
  was trained from the env's natural reset rotation.
- **Approach height** is `max(obb_top + 0.12, 0.30)` — above the perturbed
  object, within the reachable envelope.
- **Let the move settle**: `move_tolerance: 0.003` with
  `move_max_steps: 400`. The `go_to_pose` defaults (0.01 rad tolerance, 120
  steps) stop early at ~0.009 rad, leaving the rotation a few degrees off
  the commanded one and drifting the hand-off proprio.

`place_above_basket.py` (`graph_grasp` only):

- **Lift straight up first** — keep the grasp XY, raise Z. A combined
  lift+translate drags the held item low across the floor and through other
  items.
- Transport at `max(lift_z=0.40, basket_top + 0.25)`, descend to
  `basket_top + 0.10`, then `robot.open_gripper` with `settle_steps: 40`.

## Next steps

- [Policies](../benchmarks/policies.md) — serving presets, custom
  `start_cmd` servers, and the policy registry.
- [Benchmark Grids](benchmark.md) — running these graphs as the
  `llm_plus_policy` / `policy_only` benchmark modes.
- [Collect and Train](collect-and-train.md) — train the policy these graphs
  steer, from data collected by a GaP graph.
