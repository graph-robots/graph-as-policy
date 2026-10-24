"""Rig bring-up: every arm homes, dips a few centimetres and back, cycles its gripper.

Each move goes through the connector's safety gate (and viser approval under
GAP_UR_CONFIRM=1). The descent is a straight line holding the home orientation,
so the TCP only ever moves vertically above wherever home puts it.
"""

import logging
from typing import TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)


class Output(TypedDict):
    report: dict


def run(ctx: NodeContext, arm_ids: list, descend_m: float = 0.08, gripper: bool = True) -> Output:
    report = {}
    for arm_id in arm_ids:
        arm_id = int(arm_id)
        ctx.tool("robot.go_home", arm_id=arm_id)
        start = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
        down = {"position": dict(start["position"]), "rotation": dict(start["rotation"])}
        down["position"]["z"] = float(down["position"]["z"]) - float(descend_m)
        plan = ctx.tool("motion.plan_linear", end=down, arm_id=arm_id, orientation="lock",
                        tolerance=0.004, num_waypoints=10)
        if not plan.get("planned"):
            raise RuntimeError(f"arm {arm_id}: descent not planned: {plan.get('reason')}")
        ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"], arm_id=arm_id)
        reached = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]["position"]
        up = ctx.tool("motion.plan_linear", end=start, arm_id=arm_id, orientation="lock",
                      tolerance=0.004, num_waypoints=10)
        if not up.get("planned"):
            raise RuntimeError(f"arm {arm_id}: rise not planned: {up.get('reason')}")
        ctx.tool("robot.execute_trajectory", trajectory=up["trajectory"], arm_id=arm_id)
        grip = {}
        if gripper:
            grip["open"] = ctx.tool("robot.open_gripper", settle_steps=10, arm_id=arm_id)["position"]
            grip["closed"] = ctx.tool("robot.close_gripper", settle_steps=10, arm_id=arm_id)["position"]
            grip["reopened"] = ctx.tool("robot.open_gripper", settle_steps=10, arm_id=arm_id)["position"]
        dz = float(start["position"]["z"]) - float(reached["z"])
        report[f"arm{arm_id}"] = {"descended_m": dz, "descent_error_m": abs(dz - float(descend_m)), "gripper": grip}
        logger.info("[smoke] arm %d: descended %.4f m (asked %.3f), gripper %s", arm_id, dz, descend_m, grip)
    return {"report": report}
