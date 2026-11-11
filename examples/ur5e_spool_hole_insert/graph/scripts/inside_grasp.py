"""Inside-out grasp: close the gripper, go straight down into the hole, open it.

The closed 2F-85 goes in as a probe along the hole's axis. Opened inside a hole
smaller than its stroke, the fingers stop when their outsides meet the wall,
short of fully open: that stall is the grasp. Fully open means they missed the
wall, so it lets go, backs out and fails.

With ``lift_m > 0`` it then lifts the part that far, holds (2 s), sets it back down and
lets go by closing, so the scene ends as it started. Every leg is a straight
line holding the full orientation (``lock``), and every motion waits for the
operator's approval when ``GAP_UR_CONFIRM=1``.
"""

import logging
from typing import TypedDict

import numpy as np

from gap import NodeContext

logger = logging.getLogger(__name__)

GRASP_MAX_OPEN = 0.9  # open fraction reached when nothing stopped the fingers
AT_PRE_TOL_M = 0.005
LEG_TOL_RAD = 0.005  # settle tolerance of the legs in and out of the hole


class Output(TypedDict):
    inserted_m: float
    grip_open_fraction: float
    lifted_m: float


def _xyz(pose: dict) -> np.ndarray:
    p = pose["position"]
    return np.array([float(p["x"]), float(p["y"]), float(p["z"])])


def _shifted(pose: dict, dz: float) -> dict:
    out = {"position": dict(pose["position"]), "rotation": dict(pose["rotation"])}
    out["position"]["z"] = float(out["position"]["z"]) + float(dz)
    return out


def _line(ctx: NodeContext, end: dict, arm_id: int, what: str, tolerance: float = 0.003) -> None:
    here = _xyz(ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"])
    n = max(12, int(np.ceil(np.linalg.norm(_xyz(end) - here) / 0.01)))  # a waypoint per cm keeps long legs straight
    plan = ctx.tool("motion.plan_linear", end=end, arm_id=arm_id, orientation="lock", tolerance=tolerance,
                    num_waypoints=n)
    if not plan.get("planned") or not plan.get("trajectory"):
        raise RuntimeError(f"no straight path {what}: {plan.get('reason')}")
    ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"], arm_id=arm_id, tolerance=LEG_TOL_RAD)


def run(ctx: NodeContext, pre_pose: dict, insert_pose: dict, arm_id: int = 1, lift_m: float = 0.03,
        hold_steps: int = 100) -> Output:
    here = _xyz(ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"])
    off = float(np.linalg.norm(here - _xyz(pre_pose)))
    if off > AT_PRE_TOL_M:
        raise RuntimeError(f"arm {arm_id} is {off * 1e3:.1f} mm from the pre-insert pose (> {AT_PRE_TOL_M * 1e3:.0f} mm)")
    dry = bool(ctx.tool("robot.describe_gripper", arm_id=arm_id).get("dry_run"))

    ctx.tool("robot.close_gripper", settle_steps=15, arm_id=arm_id)
    _line(ctx, insert_pose, arm_id, "into the hole")
    bottom_pose = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    inserted = float(_xyz(pre_pose)[2] - _xyz(bottom_pose)[2])
    logger.info("[grasp] arm %d: TCP at (%.3f, %.3f, %.3f), %.1f mm below the pre-insert pose",
                arm_id, *_xyz(bottom_pose), inserted * 1e3)

    grip = float(ctx.tool("robot.open_gripper", settle_steps=20, arm_id=arm_id)["position"])
    if grip >= GRASP_MAX_OPEN and not dry:
        logger.warning("[grasp] fingers opened to %.2f: they missed the wall; backing out", grip)
        ctx.tool("robot.close_gripper", settle_steps=15, arm_id=arm_id)
        _line(ctx, pre_pose, arm_id, "back out")
        raise RuntimeError(f"no inside grasp: the gripper opened to {grip:.2f} without meeting the wall")
    logger.info("[grasp] arm %d: fingers stopped at open fraction %.2f against the wall%s",
                arm_id, grip, " (dry run: no contact simulated)" if dry else "")

    lifted = 0.0
    if lift_m > 0:
        _line(ctx, _shifted(bottom_pose, lift_m), arm_id, "lifting the part")
        ctx.tool("robot.wait_steps", steps=int(hold_steps))
        still = float(ctx.tool("robot.get_gripper", arm_id=arm_id)["position"])
        logger.info("[grasp] lifted %.0f mm, gripper at %.2f", lift_m * 1e3, still)
        if still >= GRASP_MAX_OPEN and not dry:
            # The fingers opened fully: the part slipped off. Do not lower open fingers
            # onto wherever it landed.
            ctx.tool("robot.close_gripper", settle_steps=15, arm_id=arm_id)
            raise RuntimeError(f"the part dropped during the lift (gripper opened to {still:.2f})")
        _line(ctx, bottom_pose, arm_id, "setting the part back down")
        lifted = float(lift_m)

    ctx.tool("robot.close_gripper", settle_steps=15, arm_id=arm_id)  # contracting lets go
    _line(ctx, pre_pose, arm_id, "back out of the hole", tolerance=0.005)
    ctx.tool("robot.open_gripper", settle_steps=10, arm_id=arm_id)
    return {"inserted_m": inserted, "grip_open_fraction": grip, "lifted_m": lifted}
