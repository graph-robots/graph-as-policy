"""After any failure: close the gripper, rise straight up, open it. Nothing else moves.

A failure may leave the fingers inside the hole, possibly opened against its
wall (an inside grasp): closing them first lets go of the part, and rising
straight up brings them out without touching the rim. Closing or opening in
free space above the part is harmless. No go-home sweep.
"""

import logging
from typing import TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)


class Output(TypedDict):
    rose: bool


def run(ctx: NodeContext, arm_id: int = 1, rise_m: float = 0.06) -> Output:
    ctx.tool("robot.close_gripper", settle_steps=15, arm_id=arm_id)
    pose = ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"]
    end = {"position": dict(pose["position"]), "rotation": dict(pose["rotation"])}
    end["position"]["z"] = float(end["position"]["z"]) + float(rise_m)
    plan = ctx.tool("motion.plan_linear", end=end, arm_id=arm_id, orientation="lock", tolerance=0.01, num_waypoints=8)
    rose = bool(plan.get("planned") and plan.get("trajectory"))
    if rose:
        ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"], arm_id=arm_id)
    else:
        logger.warning("[retreat] rise not planned (%s); holding where it is", plan.get("reason"))
    ctx.tool("robot.open_gripper", settle_steps=10, arm_id=arm_id)
    return {"rose": rose}
