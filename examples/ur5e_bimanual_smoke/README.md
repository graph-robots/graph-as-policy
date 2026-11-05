# ur5e_bimanual_smoke — bring-up check for the dionysus rig

Per arm: `go_home`, a straight 8 cm descent and rise holding the home
orientation, open/close/open the gripper; then one observation of all four
cameras. Nothing in the scene is touched — clear the table under both home
poses first.

```bash
# plans only, drawn in viser; nothing moves
GAP_UR_DRY_RUN=1 gap run examples/ur5e_bimanual_smoke/graph --real ur5e_bimanual
# real, approving each motion in viser (http://localhost:8080)
GAP_UR_CONFIRM=1 gap run examples/ur5e_bimanual_smoke/graph --real ur5e_bimanual
```

Stops: the viser STOP button, `Space` in this terminal, `rig-stop` from any
terminal, or the pendant e-stop.
