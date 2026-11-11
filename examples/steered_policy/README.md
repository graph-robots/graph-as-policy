# Steered policy — hover, then hand over to the VLA

> **What:** Hybrid graphs: perceive + hover above the target, then hand control to a VLA policy · **Needs:** `quickstart` + `policy` + LLM key · **Time:** ~min/trial

Two hybrid graphs that *steer* a learned policy with geometric perception:
OBB perception localizes the target (robust on perturbed layouts), a
Cartesian approach pre-positions the end-effector safely **above** the
object, and only then is control handed to the closed-loop VLA policy —
which starts from a pose close to its training distribution.

- **`graph_loop/`** — the clean-all-items loop: `capture` records the
  episode-start arm pose; each iteration `reset`s to it, perceives the next
  item (`perceiving-objects-oneshot`, whose VLM "none" answer cleanly ends
  the loop), approaches above its OBB, then the policy skill's `.run` tool
  (e.g. `pi05-libero.run`) picks *and places* it, terminating on a full
  gripper open→close→open cycle.
- **`graph_grasp/`** — VLA-grasp + geometric-place split: the policy does
  only the dexterous grasp (terminated by a VLM held-and-lifted check; the
  dev tree used a gripper *grasp* detector that the ported engine does not
  carry — see the note below), then `place_above_basket.py` lifts straight
  up, transports high, descends over the basket OBB, and releases.

## Run it

```bash
uv sync --extra policy && uv run gap skills install sam3 grounding-dino geometry   # + the openpi websocket client
```

The graph references a policy **skill** (e.g. `pi05-libero`), whose preset
server the launcher boots automatically (downloads the checkpoint on first
use; needs an [openpi](https://github.com/Physical-Intelligence/openpi)
checkout on `$GAP_OPENPI_DIR`). To run that server by hand instead — e.g. to
share it across runs:

```bash
uv run gap policy serve pi05-libero --port 9100
```

Then, with `{{policy_id}}` materialized to a policy-skill name such as
`pi05-libero` (forming the `pi05-libero.run` tool; the benchmark harness does
this per cell; standalone, sed the placeholder or template the workflow):

```bash
MUJOCO_GL=egl uv run gap run examples/steered_policy/graph_loop \
    --sim libero_object_all_variance/0
```

## Notes

- The policy node names the skill's `.run` tool (e.g. `pi05-libero.run`) and
  carries **no `policy_id`** — the skill owns its model. It reads the
  graph-scoped `observation_stream` and terminates on `gripper_cycle` / VLM /
  `max_windows`. `molmoact-libero` is the MolmoAct alternative for the same
  task family.
- `graph_grasp` delta vs the dev tree: `gripper_grasp_termination` (detect
  open→close-and-hold) existed only on a dev branch variant of the policy
  loop; the faithful ported engine exposes cycle detection + VLM
  termination, so the grasp stage uses
  `termination_prompt: "Is the object held in the gripper and lifted off the surface?"`.
- The approach scripts deliberately preserve the current end-effector
  rotation and use a tight `move_tolerance` (0.003) with a generous step
  budget so the hand-off proprio stays in-distribution — see the script
  docstrings for the tuning rationale.
