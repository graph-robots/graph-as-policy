"""Turn a perceived hole into two TCP poses: 6 cm above it, and inside it.

The gripper goes in closed, as a probe, along the hole's axis: tool z (the
approach axis) points down the axis, and the roll is the arm's current one, so
the wrist does not turn for a round hole. The closed 2F-85 fingertips are taken
as a 22 mm-radius envelope (conservative); a hole that does not clear it by
3 mm all round is refused rather than tried. 25 mm in by default, so the
fingers have wall to open against; it stops 1 cm above the hole's bottom
whatever was asked.
"""

import logging
from typing import TypedDict

import numpy as np
from gap_core.types import pose_to_matrix
from scipy.spatial.transform import Rotation

from gap import NodeContext

logger = logging.getLogger(__name__)

FINGER_HALF_EXTENT_M = 0.022
MIN_RADIAL_CLEARANCE_M = 0.003
MIN_BOTTOM_CLEARANCE_M = 0.010


class Output(TypedDict):
    pre_pose: dict
    insert_pose: dict
    insert_m: float
    clearance_m: float


def _v(d: dict) -> np.ndarray:
    return np.array([float(d["x"]), float(d["y"]), float(d["z"])])


def _pose(p: np.ndarray, R: np.ndarray) -> dict:
    x, y, z, w = Rotation.from_matrix(R).as_quat()
    return {"position": {"x": float(p[0]), "y": float(p[1]), "z": float(p[2])},
            "rotation": {"w": float(w), "x": float(x), "y": float(y), "z": float(z)}}


def run(ctx: NodeContext, hole: dict, arm_id: int = 1, standoff_m: float = 0.06,
        insert_depth_m: float = 0.025) -> Output:
    center, axis = _v(hole["center"]), _v(hole["axis"])
    axis /= np.linalg.norm(axis)
    clearance = float(hole["radius_m"]) - FINGER_HALF_EXTENT_M
    if clearance < MIN_RADIAL_CLEARANCE_M:
        raise RuntimeError(f"hole radius {hole['radius_m'] * 1e3:.1f} mm leaves {clearance * 1e3:.1f} mm around the "
                           f"closed fingers (< {MIN_RADIAL_CLEARANCE_M * 1e3:.0f} mm): not trying")
    insert_m = min(float(insert_depth_m), float(hole["depth_m"]) - MIN_BOTTOM_CLEARANCE_M)
    if insert_m < 0.005:
        raise RuntimeError(f"hole only {hole['depth_m'] * 1e3:.0f} mm deep: nothing to insert into")

    current = pose_to_matrix(ctx.tool("robot.get_ee_pose", arm_id=arm_id)["pose"])
    z = -axis  # approach: down the hole
    x = current[:3, 0] - (current[:3, 0] @ z) * z  # keep the current roll
    if np.linalg.norm(x) < 1e-6:
        x = np.array([1.0, 0.0, 0.0]) - z[0] * z
    x /= np.linalg.norm(x)
    R = np.column_stack([x, np.cross(z, x), z])
    pre = _pose(center + axis * float(standoff_m), R)
    ins = _pose(center - axis * insert_m, R)
    logger.info("[plan] arm %d: pre (%.3f, %.3f, %.3f), insert %.1f mm to z=%.3f, radial clearance %.1f mm",
                arm_id, *(center + axis * float(standoff_m)), insert_m * 1e3, ins["position"]["z"], clearance * 1e3)
    return {"pre_pose": pre, "insert_pose": ins, "insert_m": insert_m, "clearance_m": clearance}
