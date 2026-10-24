"""Analytic UR5e inverse kinematics + RSS-style joint/linear planning.

Closed-form IK for the UR's spherical-free 6R chain (Hawkins, "Analytic Inverse
Kinematics for the Universal Robots UR-5/UR-10 Arms", 2013): up to 8 solutions,
each verified by forward kinematics, the one nearest the seed (per-joint 2π
unwrapped inside the limits) returned. No GPU, no JAX, no optimisation — the
answer is exact for the nominal DH model, which is also what the rig's hand-eye
calibration and the controller's ``actual_TCP_pose`` agree with to ~1 mm.

Frames: every pose in or out is a **world-frame TCP pose** (``Se3Pose``, wxyz).
Each arm carries its full SE(3) base (``world_T_base`` — the dionysus bases are
tilted ~45°, which a translation-only ``arm_bases`` cannot express) and its
flange->TCP offset.

Planning mirrors RoboSimStudio's ``motion.plan_joint`` / ``motion.plan_linear``
so graphs learned in sim call the same verbs on the real arm:

- ``plan_joint``: solve the pose once, ease there in joint space (3t²-2t³).
- ``plan_linear``: per-waypoint IK along a straight TCP line, orientation held
  (``hold_rotation``) or approach-axis-only (roll chosen for continuity);
  refuses (``None`` + ``last_refusal``) on an unreachable waypoint, a branch
  flip between waypoints, or a final miss beyond ``tolerance``.
"""

from __future__ import annotations

import logging
import math
from dataclasses import dataclass
from typing import Any

import numpy as np

logger = logging.getLogger(__name__)

D = np.array([0.1625, 0.0, 0.0, 0.1333, 0.0997, 0.0996])
A = np.array([0.0, -0.425, -0.3922, 0.0, 0.0, 0.0])
ALPHA = np.array([math.pi / 2, 0.0, 0.0, math.pi / 2, -math.pi / 2, 0.0])

#: Max per-joint jump between consecutive linear-plan waypoints [rad].
MAX_WAYPOINT_STEP = 0.35


def _dh(theta: float, d: float, a: float, alpha: float) -> np.ndarray:
    ct, st = math.cos(theta), math.sin(theta)
    ca, sa = math.cos(alpha), math.sin(alpha)
    return np.array([[ct, -st * ca, st * sa, a * ct],
                     [st, ct * ca, -ct * sa, a * st],
                     [0.0, sa, ca, d],
                     [0.0, 0.0, 0.0, 1.0]])


def fk_frames(q) -> list[np.ndarray]:
    q = np.asarray(q, dtype=np.float64).reshape(-1)[:6]
    T = np.eye(4)
    out = []
    for i in range(6):
        T = T @ _dh(float(q[i]), D[i], A[i], ALPHA[i])
        out.append(T.copy())
    return out


def fk(q) -> np.ndarray:
    """Flange in the controller base frame."""
    return fk_frames(q)[-1]


def ik_flange(T: np.ndarray, tol: float = 1e-6) -> list[np.ndarray]:
    """All analytic solutions (each in (-π, π]) for a base-frame flange pose."""
    T = np.asarray(T, dtype=np.float64)
    d1, a2, a3, d4, d5, d6 = D[0], A[1], A[2], D[3], D[4], D[5]
    sols: list[np.ndarray] = []
    p05 = T @ np.array([0.0, 0.0, -d6, 1.0])
    r = math.hypot(p05[0], p05[1])
    if r < abs(d4):
        return []
    psi = math.atan2(p05[1], p05[0])
    phi = math.acos(np.clip(d4 / r, -1.0, 1.0))
    for t1 in (psi + phi + math.pi / 2, psi - phi + math.pi / 2):
        s1, c1 = math.sin(t1), math.cos(t1)
        c5 = (T[0, 3] * s1 - T[1, 3] * c1 - d4) / d6
        if abs(c5) > 1.0 + 1e-9:
            continue
        a5 = math.acos(np.clip(c5, -1.0, 1.0))
        for t5 in (a5, -a5):
            s5 = math.sin(t5)
            if abs(s5) < 1e-9:
                t6 = 0.0  # wrist singularity: q6 is free; any value is a solution
            else:
                t6 = math.atan2((-T[0, 1] * s1 + T[1, 1] * c1) / s5, (T[0, 0] * s1 - T[1, 0] * c1) / s5)
            T01 = _dh(t1, d1, 0.0, math.pi / 2)
            T45 = _dh(t5, d5, 0.0, -math.pi / 2)
            T56 = _dh(t6, d6, 0.0, 0.0)
            T14 = np.linalg.inv(T01) @ T @ np.linalg.inv(T45 @ T56)
            p13 = T14 @ np.array([0.0, -d4, 0.0, 1.0]) - np.array([0.0, 0.0, 0.0, 1.0])
            n13 = float(np.linalg.norm(p13[:3]))
            c3 = (n13**2 - a2**2 - a3**2) / (2.0 * a2 * a3)
            if abs(c3) > 1.0 + 1e-9:
                continue
            a3_ = math.acos(np.clip(c3, -1.0, 1.0))
            for t3 in (a3_, -a3_):
                t2 = -math.atan2(p13[1], -p13[0]) + math.asin(np.clip(a3 * math.sin(t3) / n13, -1.0, 1.0))
                T12 = _dh(t2, 0.0, a2, 0.0)
                T23 = _dh(t3, 0.0, a3, 0.0)
                T34 = np.linalg.inv(T12 @ T23) @ T14
                t4 = math.atan2(T34[1, 0], T34[0, 0])
                q = np.array([t1, t2, t3, t4, t5, t6])
                q = (q + math.pi) % (2.0 * math.pi) - math.pi
                if np.abs(fk(q) - T).max() < tol * 100:
                    sols.append(q)
    return sols


# ── pose helpers (wxyz) ───────────────────────────────────────────────────────


def quat_wxyz_to_matrix(q) -> np.ndarray:
    w, x, y, z = (float(v) for v in q)
    n = math.sqrt(w * w + x * x + y * y + z * z)
    w, x, y, z = w / n, x / n, y / n, z / n
    return np.array([[1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)],
                     [2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)],
                     [2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)]])


def matrix_to_quat_wxyz(R: np.ndarray) -> np.ndarray:
    from scipy.spatial.transform import Rotation

    x, y, z, w = Rotation.from_matrix(np.asarray(R)[:3, :3]).as_quat()
    q = np.array([w, x, y, z])
    return -q if q[0] < 0 else q


def pose_to_matrix(pose: dict) -> np.ndarray:
    p, r = pose["position"], pose.get("rotation") or {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}
    T = np.eye(4)
    T[:3, :3] = quat_wxyz_to_matrix([r["w"], r["x"], r["y"], r["z"]])
    T[:3, 3] = [float(p["x"]), float(p["y"]), float(p["z"])]
    return T


def matrix_to_pose(T: np.ndarray) -> dict:
    q = matrix_to_quat_wxyz(T)
    return {"position": {"x": float(T[0, 3]), "y": float(T[1, 3]), "z": float(T[2, 3])},
            "rotation": {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}}


def _rot_z(phi: float) -> np.ndarray:
    T = np.eye(4)
    c, s = math.cos(phi), math.sin(phi)
    T[:2, :2] = [[c, -s], [s, c]]
    return T


def rotation_angle(Ra: np.ndarray, Rb: np.ndarray) -> float:
    c = (np.trace(np.asarray(Ra)[:3, :3].T @ np.asarray(Rb)[:3, :3]) - 1.0) / 2.0
    return float(math.acos(np.clip(c, -1.0, 1.0)))


# ── per-arm model ─────────────────────────────────────────────────────────────


@dataclass
class URArm:
    name: str
    world_T_base: np.ndarray
    tcp_offset: np.ndarray  # flange -> TCP translation, flange frame
    lower: np.ndarray
    upper: np.ndarray
    home_q: np.ndarray | None = None

    def tcp_world(self, q) -> np.ndarray:
        T = self.world_T_base @ fk(q)
        out = T.copy()
        out[:3, 3] = T[:3, 3] + T[:3, :3] @ self.tcp_offset
        return out

    def link_points_world(self, q) -> np.ndarray:
        pts = [self.world_T_base[:3, 3]]
        pts += [(self.world_T_base @ F)[:3, 3] for F in fk_frames(q)]
        pts.append(self.tcp_world(q)[:3, 3])
        return np.asarray(pts)

    def flange_target(self, world_T_tcp: np.ndarray, tcp_offset: np.ndarray | None = None) -> np.ndarray:
        off = self.tcp_offset if tcp_offset is None else np.asarray(tcp_offset, dtype=np.float64)
        world_T_flange = world_T_tcp.copy()
        world_T_flange[:3, 3] = world_T_tcp[:3, 3] - world_T_tcp[:3, :3] @ off
        return np.linalg.inv(self.world_T_base) @ world_T_flange

    def nearest(self, sols: list[np.ndarray], seed: np.ndarray) -> np.ndarray | None:
        """Unwrap each solution to the 2π branch nearest ``seed`` (inside limits), pick the closest."""
        best, best_cost = None, math.inf
        w = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
        for s in sols:
            q = s.copy()
            for j in range(6):
                k = round((seed[j] - q[j]) / (2 * math.pi))
                cand = [q[j] + 2 * math.pi * (k + dk) for dk in (-1, 0, 1)]
                cand = [c for c in cand if self.lower[j] - 1e-9 <= c <= self.upper[j] + 1e-9]
                if not cand:
                    q = None
                    break
                q[j] = min(cand, key=lambda c: abs(c - seed[j]))
            if q is None:
                continue
            cost = float(np.sum(w * (q - seed) ** 2))
            if cost < best_cost:
                best, best_cost = q, cost
        return best


class URAnalyticIK:
    """IK backend over one or more UR5e arms (GaP ``Connector.ik`` surface).

    Args:
        arms: ``URArm`` per ``arm_id`` (list index = arm_id).
        roll_samples: Roll candidates about the approach axis in ``axis`` mode.
        waypoints: Default waypoint count for ``plan_linear``.
        tolerance: Default ``plan_linear`` final-position gate [m] (RSS default 0.03).
    """

    trajectory_needs_joint_reverse = False
    supports_world_aware_plan = False
    honours_roll = True

    def __init__(self, arms: list[URArm], roll_samples: int = 72, waypoints: int = 40,
                 joint_waypoints: int = 30, tolerance: float = 0.03) -> None:
        self.arms = list(arms)
        self.roll_samples = int(roll_samples)
        self.waypoints = int(waypoints)
        self.joint_waypoints = int(joint_waypoints)
        self.tolerance = float(tolerance)
        self.last_refusal = ""
        self.last_position_error: float | None = None

    def arm(self, arm_id: int) -> URArm:
        if not 0 <= int(arm_id) < len(self.arms):
            raise ValueError(f"arm_id {arm_id} out of range (have {len(self.arms)} arms)")
        return self.arms[int(arm_id)]

    def _seed(self, arm: URArm, seed) -> np.ndarray:
        if seed is not None:
            return np.asarray(seed, dtype=np.float64).reshape(-1)[:6]
        return arm.home_q.copy() if arm.home_q is not None else np.zeros(6)

    # ── single pose ───────────────────────────────────────────────────────

    def solve(self, world_T_tcp: np.ndarray, arm_id: int = 0, seed=None, hold_rotation: bool = True,
              tcp_offset=None, roll_window: float | None = None) -> tuple[np.ndarray | None, float, float]:
        """(q | None, position_error_m, rotation_error_rad) for a world TCP pose.

        ``hold_rotation=False`` holds only the approach (tool z) axis and picks the
        roll whose solution is nearest the seed; ``roll_window`` limits the roll
        search to ±window around the requested roll (continuity in linear plans).
        """
        arm = self.arm(arm_id)
        seed = self._seed(arm, seed)
        if hold_rotation:
            rolls = [0.0]
        elif roll_window is not None:
            n = max(3, int(self.roll_samples * roll_window / math.pi))
            rolls = list(np.linspace(-roll_window, roll_window, n))
        else:
            rolls = list(np.linspace(-math.pi, math.pi, self.roll_samples, endpoint=False))
        best, best_cost = None, math.inf
        w = np.array([2.0, 2.0, 1.5, 1.0, 1.0, 0.5])
        for phi in rolls:
            T = world_T_tcp @ _rot_z(phi) if phi else world_T_tcp
            q = arm.nearest(ik_flange(arm.flange_target(T, tcp_offset)), seed)
            if q is None:
                continue
            cost = float(np.sum(w * (q - seed) ** 2)) + 1e-3 * phi * phi
            if cost < best_cost:
                best, best_cost = q, cost
        if best is None:
            return None, math.inf, math.inf
        got = arm.tcp_world(best) if tcp_offset is None else None
        if got is None:
            got = arm.world_T_base @ fk(best)
            got[:3, 3] = got[:3, 3] + got[:3, :3] @ np.asarray(tcp_offset, dtype=np.float64)
        pos_err = float(np.linalg.norm(got[:3, 3] - world_T_tcp[:3, 3]))
        if hold_rotation:
            rot_err = rotation_angle(got, world_T_tcp)
        else:
            rot_err = float(math.acos(np.clip(got[:3, 2] @ world_T_tcp[:3, 2], -1.0, 1.0)))
        return best, pos_err, rot_err

    def solve_ik(self, pose: dict, arm_id: int = 0, seed_joints=None, tcp_offset=None) -> list[float] | None:
        """GaP ``Connector.ik`` contract: full pose; None when unreachable."""
        q, pos_err, rot_err = self.solve(pose_to_matrix(pose), arm_id, seed_joints, True, tcp_offset)
        if q is None or pos_err > 1e-4 or rot_err > 1e-3:
            return None
        return [float(v) for v in q]

    def tcp_pose(self, q, arm_id: int = 0) -> dict:
        return matrix_to_pose(self.arm(arm_id).tcp_world(q))

    # ── plans ─────────────────────────────────────────────────────────────

    def plan_joint(self, pose: dict, arm_id: int = 0, seed_joints=None, num_waypoints: int = 0,
                   hold_rotation: bool = False, tcp_offset=None) -> dict:
        """Solve once, ease in joint space (RSS ``plan_joint`` semantics)."""
        arm = self.arm(arm_id)
        seed = self._seed(arm, seed_joints)
        q, pos_err, rot_err = self.solve(pose_to_matrix(pose), arm_id, seed, hold_rotation, tcp_offset)
        if q is None:
            return {"waypoints": [], "position_error_m": math.inf, "rotation_error_rad": math.inf,
                    "joint_config": [], "reason": "pose is outside the arm's reach (no IK solution)"}
        steps = max(1, int(num_waypoints) or self.joint_waypoints)
        wps = []
        for i in range(1, steps + 1):
            t = i / steps
            s = t * t * (3.0 - 2.0 * t)
            wps.append({"positions": [float(v) for v in seed + s * (q - seed)]})
        return {"waypoints": wps, "position_error_m": pos_err,
                "rotation_error_rad": rot_err if hold_rotation else 0.0,
                "joint_config": [float(v) for v in q], "reason": ""}

    def plan_linear(self, start_world_pose: dict, end_world_pose: dict, *, arm_id: int = 0, tcp_offset=None,
                    seed_joints=None, num_waypoints: int = 0, hold_rotation: bool = False,
                    tolerance: float | None = None) -> dict | None:
        """Straight TCP line with per-waypoint IK; ``None`` + ``last_refusal`` when it cannot hold."""
        self.last_refusal = ""
        self.last_position_error = None
        arm = self.arm(arm_id)
        seed = self._seed(arm, seed_joints)
        T0 = pose_to_matrix(start_world_pose)
        T1 = pose_to_matrix(end_world_pose)
        tol = self.tolerance if tolerance is None else float(tolerance)
        steps = max(2, int(num_waypoints) or self.waypoints)
        q_prev = seed
        wps: list[dict] = []
        T_last = None
        for i in range(1, steps + 1):
            t = i / steps
            T = T1.copy()
            T[:3, 3] = T0[:3, 3] + t * (T1[:3, 3] - T0[:3, 3])
            q = None
            if not hold_rotation:
                # "axis" keeps the line's starting roll about the approach axis wherever it is reachable, and
                # only searches nearby rolls when it is not. Picking, per waypoint, the roll nearest the
                # previous joints let the roll walk 3 degrees per waypoint on a vertical descent (+42 degrees
                # over two 8-waypoint lines on the dionysus left arm, 2026-10-20), and the jaws came down across
                # the object they were aimed square to.
                z = T[:3, 2]
                x0 = T0[:3, 0] - (T0[:3, 0] @ z) * z
                if np.linalg.norm(x0) > 1e-6:
                    x0 /= np.linalg.norm(x0)
                    keep = T.copy()
                    keep[:3, :3] = np.column_stack([x0, np.cross(z, x0), z])
                    q, perr, _ = self.solve(keep, arm_id, q_prev, True, tcp_offset)
                    if q is not None and (perr > tol or np.abs(q - q_prev).max() > MAX_WAYPOINT_STEP):
                        q = None
            if q is not None:
                pass
            elif hold_rotation or i == 1:
                q, perr, _ = self.solve(T, arm_id, q_prev, hold_rotation, tcp_offset)
            else:
                # Keep the roll continuous: search only near the previous waypoint's roll.
                prev_T = arm.tcp_world(q_prev) if tcp_offset is None else T_last
                ref = T.copy()
                if prev_T is not None:
                    # Express the requested frame with the previous roll about the approach axis.
                    z = T[:3, 2]
                    x_prev = prev_T[:3, 0] - (prev_T[:3, 0] @ z) * z
                    if np.linalg.norm(x_prev) > 1e-6:
                        x_prev /= np.linalg.norm(x_prev)
                        ref[:3, :3] = np.column_stack([x_prev, np.cross(z, x_prev), z])
                q, perr, _ = self.solve(ref, arm_id, q_prev, False, tcp_offset, roll_window=math.radians(15))
            if q is None:
                self.last_refusal = f"waypoint {i}/{steps} of the line is out of reach"
                return None
            if np.abs(q - q_prev).max() > MAX_WAYPOINT_STEP:
                self.last_refusal = (f"waypoint {i}/{steps} needs a {np.abs(q - q_prev).max():.2f} rad joint jump "
                                     "(wrist/elbow branch flip or singularity on the line)")
                return None
            wps.append({"positions": [float(v) for v in q]})
            q_prev = q
            T_last = arm.tcp_world(q)
        self.last_position_error = float(np.linalg.norm(T_last[:3, 3] - T1[:3, 3]))
        if self.last_position_error > tol:
            self.last_refusal = f"line ends {self.last_position_error * 1000:.1f} mm from the target (> {tol * 1000:.0f} mm)"
            return None
        return {"waypoints": wps}


def arms_from_rig_wire(rig: dict) -> list[URArm]:
    """``URArm`` list (by arm_id) from the rig description the controller sends."""
    out: dict[int, URArm] = {}
    for name, a in rig["arms"].items():
        out[int(a["arm_id"])] = URArm(
            name=name,
            world_T_base=np.asarray(a["world_T_base"], dtype=np.float64),
            tcp_offset=np.asarray(a["tcp_offset"], dtype=np.float64),
            lower=np.asarray(a["joint_lower"], dtype=np.float64),
            upper=np.asarray(a["joint_upper"], dtype=np.float64),
            home_q=np.asarray(a["home_q"], dtype=np.float64),
        )
    return [out[i] for i in sorted(out)]


def load_rig_arms(rig_yaml: str | None = None) -> tuple[list[URArm], dict]:
    """Same, read straight from the controllers' ``rig.yaml`` (offline / tests)."""
    from pathlib import Path

    import yaml

    from gap.connector.rr_launcher import DEFAULT_RR_DIR

    path = Path(rig_yaml) if rig_yaml else DEFAULT_RR_DIR / "configs" / "ur5e_bimanual" / "rig.yaml"
    cfg = yaml.safe_load(path.read_text())
    root = path.parent

    def find(calib: str, parent: str, child: str) -> np.ndarray:
        data = yaml.safe_load((root / calib).read_text())
        for rec in data["transforms"]:
            if rec["parent_frame"] == parent and rec["child_frame"] == child:
                t, r = rec["translation"], rec["rotation"]
                T = np.eye(4)
                T[:3, :3] = quat_wxyz_to_matrix([r["w"], r["x"], r["y"], r["z"]])
                T[:3, 3] = [t["x"], t["y"], t["z"]]
                return T
        raise KeyError(f"{calib}: {parent} -> {child}")

    wire: dict[str, Any] = {"surface_z": float(cfg.get("surface_z", 0.0)), "arms": {},
                            "workspace_min": cfg["workspace"]["min"], "workspace_max": cfg["workspace"]["max"],
                            "clearance": cfg.get("clearance", {}), "cameras": {}}
    for name, a in cfg["arms"].items():
        lim = a.get("joint_limits", [-2 * math.pi, 2 * math.pi])
        wire["arms"][name] = {
            "arm_id": int(a["arm_id"]),
            "ip": str(a["ip"]),
            "world_T_base": find(a["calib"], "checkerboard", a["base_frame"]).tolist(),
            "tcp_offset": a["tcp_offset"], "home_q": a["home_q"],
            "home_gripper": a.get("home_gripper", 0.0),
            "joint_lower": a.get("joint_lower", [lim[0]] * 6), "joint_upper": a.get("joint_upper", [lim[1]] * 6),
        }
    return arms_from_rig_wire(wire), wire


__all__ = ["URAnalyticIK", "URArm", "arms_from_rig_wire", "fk", "ik_flange", "load_rig_arms",
           "matrix_to_pose", "pose_to_matrix"]
