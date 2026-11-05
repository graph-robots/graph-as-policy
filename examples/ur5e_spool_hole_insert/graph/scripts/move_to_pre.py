"""Open the gripper and go to the pre-insert pose (6 cm above the hole).

Open, because the wrist camera looks past the fingertips: closed fingers would
hide the hole from the close-range look the next step takes. A joint-space move
(the transit), and it must land within 5 mm, or nothing follows.
"""

import logging
from typing import TypedDict

from gap import NodeContext

logger = logging.getLogger(__name__)


class Output(TypedDict):
    position_error_m: float


def run(ctx: NodeContext, pre_pose: dict, arm_id: int = 1) -> Output:
    ctx.tool("robot.open_gripper", settle_steps=10, arm_id=arm_id)
    plan = ctx.tool("motion.plan_joint", pose=pre_pose, arm_id=arm_id, orientation="lock", num_waypoints=30)
    if not plan.get("planned") or not plan.get("trajectory"):
        raise RuntimeError(f"no joint path to the pre-insert pose: {plan.get('reason')}")
    err = float(plan.get("position_error_m") or 0.0)
    if err > 0.005:
        raise RuntimeError(f"pre-insert pose only reachable to {err * 1e3:.1f} mm")
    ctx.tool("robot.execute_trajectory", trajectory=plan["trajectory"], arm_id=arm_id)
    logger.info("[approach] arm %d at the pre-insert pose", arm_id)
    return {"position_error_m": err}
