"""UR5eBimanualConnector — GaP's real connector for the dionysus bimanual UR5e rig.

On top of :class:`~gap.connector.real.RealConnector`, it provides what graphs
learned in RoboSimStudio call and the stock connector lacks, with the same
argument names and result shapes as RSS's ``ConnectorNewton``:

- ``motion.plan_joint`` / ``motion.plan_linear`` / ``motion.describe_planner``
  (analytic UR5e IK, :mod:`gap.connector.ur_ik`);
- ``robot.describe_gripper`` / ``describe_arm`` / ``describe_workspace`` /
  ``grasp_frame`` / ``wait_steps`` (Robotiq 2F-85 numbers from RSS's
  ``robotiq_2f85_menagerie`` spec; heights from the rig calibration);
- ``arm_id`` on ``robot.execute_trajectory`` / ``robot.move_to_joints``, and a
  ``robot.go_home(arm_id)`` that really moves (rise, then a slow joint move);
- ``geometry.mask_to_world_points`` in-process (GaP then does not boot the
  geometry bundle, whose open3d pin has no aarch64 wheel).

**Every motion goes through one gate** (:meth:`_guarded_execute`): the joint path
is densified and checked against joint limits, the TCP workspace box, a table
floor for the TCP and the wrist/forearm frames, and the distance to the other
arm. A plan that fails is drawn red in viser and refused (``planned: false`` from
the planners, ``ToolError`` from the movers). With ``GAP_UR_CONFIRM=1`` each
motion also waits for **Approve** in viser. Stops (latch, viser STOP, space key,
stale stream) raise ``RigStopped`` from the env, which unwinds the graph.
"""

from __future__ import annotations

import logging
import math
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.types import make_pose

from gap.connector.real import RealConnector
from gap.connector.ur_ik import (
    URAnalyticIK,
    arms_from_rig_wire,
    matrix_to_pose,
    matrix_to_quat_wxyz,
    pose_to_matrix,
)

logger = logging.getLogger(__name__)

# Robotiq 2F-85 (RoboSimStudio robotiq_2f85_menagerie, measured there).
GRIPPER = {
    "name": "robotiq_2f85",
    "span": 0.085,
    "pad_half_span": 0.04897,
    "pad_clearance": 0.004,
    "pad_thickness": 0.0375,
    "tip_reach": 0.00442,
    "housing_half": 0.0424,
    "housing_rise": 0.0597,
}
MARGIN = 0.02  # RSS robot_tools.MARGIN
_DENSE_STEP = 0.05  # rad, gate resolution between waypoints


def _seg_seg_dist(p1, q1, p2, q2) -> float:
    """Minimum distance between segments p1q1 and p2q2."""
    d1, d2, r = q1 - p1, q2 - p2, p1 - p2
    a, e, f = d1 @ d1, d2 @ d2, d2 @ r
    if a < 1e-12 and e < 1e-12:
        return float(np.linalg.norm(r))
    if a < 1e-12:
        s, t = 0.0, np.clip(f / e, 0.0, 1.0)
    else:
        c = d1 @ r
        if e < 1e-12:
            t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
        else:
            b = d1 @ d2
            den = a * e - b * b
            s = np.clip((b * f - c * e) / den, 0.0, 1.0) if den > 1e-12 else 0.0
            t = (b * s + f) / e
            if t < 0.0:
                t, s = 0.0, np.clip(-c / a, 0.0, 1.0)
            elif t > 1.0:
                t, s = 1.0, np.clip((b - c) / a, 0.0, 1.0)
    return float(np.linalg.norm((p1 + d1 * s) - (p2 + d2 * t)))


def _polyline_dist(P: np.ndarray, Q: np.ndarray) -> float:
    return min(_seg_seg_dist(P[i], P[i + 1], Q[j], Q[j + 1]) for i in range(len(P) - 1) for j in range(len(Q) - 1))


class UR5eBimanualConnector(RealConnector):
    """Real connector for the bimanual UR5e rig (see module docstring)."""

    honours_roll = True
    """The planners solve the whole orientation for ``orientation="axis"`` too, as the twin does: the dionysus
    embodiment declares ``honour_roll`` (r2s2rn ``r2s2rn_twin/dionysus.py``), so RoboSimStudio holds the roll there.
    Until 2026-10-22 ``axis`` freed it here: the gasket policy's pregrasp (50/50 at the exact rotation in the twin)
    ended 60 degrees off its heading on the rig while reporting 0, and its locked descent was refused for a 1.04 rad
    wrist jump on the first waypoint."""

    def __init__(self, env: Any, config: Any, *, rig_wire: dict, camera_names: list[str] | None = None,
                 rr_session: Any | None = None, close_axis_tcp=(1.0, 0.0, 0.0)) -> None:
        self.rig = rig_wire
        ik = URAnalyticIK(arms_from_rig_wire(rig_wire))
        super().__init__(env, config, camera_names=camera_names, ik=ik, rr_session=rr_session, motion_enabled=True)
        self.surface_z = float(rig_wire.get("surface_z", 0.0))
        self.ws_min = np.asarray(rig_wire.get("workspace_min", [-1, -1, 0]), dtype=np.float64)
        self.ws_max = np.asarray(rig_wire.get("workspace_max", [1, 1, 1]), dtype=np.float64)
        clr = rig_wire.get("clearance") or {}
        self.tcp_floor = self.surface_z + float(clr.get("tcp_above_surface", 0.005))
        self.link_floor = self.surface_z + float(clr.get("links_above_surface", 0.04))
        self.inter_arm = float(clr.get("inter_arm", 0.12))
        self.approach_axis = np.array([0.0, 0.0, 1.0])
        self.close_axis = np.asarray(close_axis_tcp, dtype=np.float64)
        self._plans: list[dict] = []

    # ── registry ──────────────────────────────────────────────────────────

    def _register_robot_tools(self, reg: ToolRegistry) -> None:
        rc = reg.register_callable
        rc("robot.get_observation", self.get_observation,
           summary="Capture the current observation: all cameras + arm states.")
        rc("robot.get_camera_pose", self.get_camera_pose, summary="Get one camera's world pose by name.")
        rc("robot.get_ee_pose", self._tool_get_ee_pose, summary="Get the TCP pose in world frame.")
        rc("robot.get_gripper", self._tool_get_gripper, summary="Get the gripper open fraction (0 closed, 1 open).")
        rc("robot.get_wrench", self._tool_get_wrench,
           summary="The wrist force/torque sensor at the TCP, world frame (N, N m).")
        rc("robot.get_gripper_pose", self._tool_get_gripper_pose, summary="Get the TCP pose in world frame.")
        rc("robot.go_to_pose", self.go_to_pose, summary="Move the TCP to a world-frame pose via IK.",
           tags=("sim_step",))
        rc("robot.go_to_pose_cartesian", self.go_to_pose_cartesian,
           summary="Move the TCP along a straight line to a world-frame pose.", tags=("sim_step",))
        rc("robot.move_to_joints", self._tool_move_to_joints, summary="Move one arm to a joint configuration.",
           tags=("sim_step",))
        rc("robot.execute_trajectory", self._tool_execute_trajectory,
           summary="Execute a joint trajectory on one arm (time-parameterised, gated).", tags=("sim_step",))
        rc("robot.go_home", self.go_home, summary="Raise, then move one arm (or all, arm_id=-1) home.",
           tags=("sim_step",))
        rc("robot.open_gripper", self.open_gripper, summary="Open the gripper.", tags=("sim_step",))
        rc("robot.close_gripper", self.close_gripper, summary="Close the gripper.", tags=("sim_step",))
        rc("robot.set_grip", self.set_grip, summary="The jaws to a width, a preset or a fraction closed.",
           tags=("sim_step",))
        rc("robot.forward_kinematics", self.forward_kinematics,
           summary="TCP poses for joint vectors (pure kinematics), and the joints now.", tags=("planning",))
        rc("robot.solve_ik", self._tool_solve_ik, summary="Solve IK for a world-frame TCP pose.",
           tags=("planning",))
        rc("robot.describe_gripper", self.describe_gripper, summary="The hand's geometry and conventions.")
        rc("robot.describe_arm", self.describe_arm, summary="Joints, limits, home, TCP, solver of one arm.")
        rc("robot.describe_workspace", self.describe_workspace,
           summary="Surface height, transport height and clearances for one arm.")
        rc("robot.grasp_frame", self.grasp_frame,
           summary="A grasp orientation from this hand's axes (both half-turn variants).")
        rc("robot.wait_steps", self.wait_steps, summary="Hold for N control steps.", tags=("sim_step",))
        rc("motion.plan_joint", self.plan_joint, summary="Solve a pose once and ease there in joint space.",
           tags=("planning",))
        rc("motion.plan_linear", self.plan_linear, summary="A straight TCP line, per-waypoint IK.",
           tags=("planning",))
        rc("motion.describe_planner", self.describe_planner, summary="What is planning (no collision world).")
        rc("geometry.mask_to_world_points", self.mask_to_world_points,
           summary="Back-project a mask's valid depth pixels to world points.", tags=("perception",))

    # ── state helpers ─────────────────────────────────────────────────────

    def _q(self, arm_id: int) -> np.ndarray:
        q = self._current_joints(arm_id)
        if q is None:
            raise ToolError("robot", f"no joint state for arm {arm_id}")
        return np.asarray(q, dtype=np.float64)

    def _arm_name(self, arm_id: int) -> str:
        return self.ik.arm(arm_id).name

    def _resolve_arm(self, arm_id: int, role: str = "") -> int:
        if role:
            names = {"work": 0, "support": 1}
            if role in names:
                return names[role]
            for i, a in enumerate(self.ik.arms):
                if a.name == role:
                    return i
        return int(arm_id)

    def _tcp_world(self, arm_id: int) -> np.ndarray:
        return self.ik.arm(arm_id).tcp_world(self._q(arm_id))

    # ── safety gate ───────────────────────────────────────────────────────

    def _violation(self, arm_id: int, q: np.ndarray, others: dict[int, np.ndarray]) -> tuple[float, str]:
        """Largest constraint violation [m or rad] at ``q`` (<= 0 means OK) and its name."""
        arm = self.ik.arm(arm_id)
        worst, why = -math.inf, ""

        def bump(v: float, name: str) -> None:
            nonlocal worst, why
            if v > worst:
                worst, why = v, name

        bump(float(np.max(np.r_[arm.lower - q, q - arm.upper])), "joint limit")
        tcp = arm.tcp_world(q)[:3, 3]
        bump(float(np.max(np.r_[self.ws_min - tcp, tcp - self.ws_max])), "TCP outside the workspace box")
        bump(self.tcp_floor - float(tcp[2]), "TCP below the table floor")
        pts = arm.link_points_world(q)
        bump(self.link_floor - float(pts[2:-1, 2].min()), "wrist/forearm below the table floor")
        for oid, oq in others.items():
            d = _polyline_dist(pts[1:], self.ik.arm(oid).link_points_world(oq)[1:])
            bump(self.inter_arm - d, f"within {self.inter_arm:.2f} m of arm {oid}")
        return worst, why

    def _gate(self, arm_id: int, waypoints: list) -> tuple[bool, str, np.ndarray]:
        """Check the densified joint path from the current joints through ``waypoints``."""
        q0 = self._q(arm_id)
        others = {}
        for i in range(len(self.ik.arms)):
            if i != arm_id:
                try:
                    others[i] = self._q(i)
                except ToolError:
                    pass
        path = [q0] + [np.asarray(w, dtype=np.float64)[:6] for w in waypoints]
        dense = [q0]
        for a, b in zip(path[:-1], path[1:], strict=True):
            n = max(1, int(math.ceil(float(np.abs(b - a).max()) / _DENSE_STEP)))
            dense += [a + (b - a) * (k / n) for k in range(1, n + 1)]
        tcp_path = np.array([self.ik.arm(arm_id).tcp_world(q)[:3, 3] for q in dense])
        prev, _ = self._violation(arm_id, q0, others)
        for k, q in enumerate(dense[1:], start=1):
            v, why = self._violation(arm_id, q, others)
            # A violated start may be left, never made worse.
            if v > max(prev, 0.0) + 1e-4:
                return False, f"{why} (by {v * 1000:.0f} mm/mrad at step {k}/{len(dense) - 1})", tcp_path
            prev = v
        return True, "", tcp_path

    def _show_plan(self, arm_id: int, tcp_path: np.ndarray, ok: bool, label: str) -> None:
        upd = getattr(self.env, "viz_update", None)
        if upd is None:
            return
        self._plans = (self._plans + [{"tcp_path": tcp_path.astype(float).tolist(), "ok": bool(ok),
                                       "label": f"arm{arm_id} {label}"}])[-6:]
        upd(plans=self._plans)

    def _guarded_execute(self, arm_id: int, waypoints: list, label: str, tolerance: float = 0.01,
                         stop_force_n: float = 0.0, speed_scale: float = 1.0) -> dict:
        self.env.check_stop()
        ok, reason, tcp_path = self._gate(arm_id, waypoints)
        self._show_plan(arm_id, tcp_path, ok, label)
        if not ok:
            raise ToolError("robot.execute_trajectory", f"{label} refused by the safety gate: {reason}")
        dist = float(np.linalg.norm(tcp_path[-1] - tcp_path[0])) if len(tcp_path) else 0.0
        summary = (f"arm {arm_id} ({self._arm_name(arm_id)}): {label}, {len(waypoints)} waypoints, "
                   f"TCP travels {dist * 100:.1f} cm to z={tcp_path[-1][2]:.3f}"
                   + (f", stops at {stop_force_n:.0f} N" if stop_force_n > 0 else ""))
        if not self.env.request_approval(summary):
            raise ToolError("robot.execute_trajectory", f"{label} rejected by the operator")
        res = self.env.stream_joint_trajectory([list(map(float, w[:6])) for w in waypoints],
                                               settle_tolerance=max(tolerance, 0.005), arm_id=arm_id,
                                               stop_force_n=float(stop_force_n), speed_scale=float(speed_scale))
        return res or {}

    # ── movers (all routed through the gate) ──────────────────────────────

    def move_to_joints(self, target_joints, tolerance: float = 0.01, max_steps: int = 120, arm_id: int = 0,
                       settle_steps: int | None = None) -> None:
        self._guarded_execute(arm_id, [list(target_joints)[:6]], "move_to_joints", tolerance)

    def stream_trajectory(self, waypoints, *, settle_tolerance: float = 0.01, settle_max_steps: int = 60,
                          arm_id: int = 0) -> bool:
        self._guarded_execute(arm_id, waypoints, "trajectory", settle_tolerance)
        return True

    def _tool_execute_trajectory(self, trajectory: Any, subsample: int = 0, tolerance: float = 0.0,
                                 max_steps_per_waypoint: int = 0, arm_id: int = 0, stop_force_n: float = 0.0,
                                 speed_scale: float = 1.0) -> dict:
        """Stream a planned trajectory; with ``stop_force_n`` it stops early, holding, on a contact force that size."""
        if not trajectory or not trajectory.get("waypoints"):
            raise ToolError("robot.execute_trajectory", "trajectory required")
        wps = [w["positions"] if isinstance(w, dict) else w for w in trajectory["waypoints"]]
        res = self._guarded_execute(int(arm_id), wps, "execute_trajectory", tolerance if tolerance > 0 else 0.01,
                                    stop_force_n=float(stop_force_n), speed_scale=float(speed_scale))
        out = {"final_tracking_error_rad": float(res.get("final_tracking_error_rad", 0.0))}
        for k in ("stopped_on_force", "force_n", "progress"):
            if k in res:
                out[k] = res[k]
        return out

    def _tool_get_wrench(self, arm_id: int = 0) -> dict:
        """The wrist sensor's force and torque at the TCP, turned from the arm's base frame into the world's."""
        self.env._update_from_network()
        f = None if self.env.dry_run else self.env._force(self._arm_name(int(arm_id)))
        if f is None:
            return {"force": [0.0, 0.0, 0.0], "torque": [0.0, 0.0, 0.0], "frame": "world", "available": False}
        R = np.asarray(self.ik.arm(int(arm_id)).world_T_base, dtype=float)[:3, :3]
        return {"force": (R @ f[:3]).round(3).tolist(), "torque": (R @ f[3:6]).round(4).tolist(), "frame": "world",
                "available": True}

    def _tool_move_to_joints(self, joint_config: Any, tolerance: float = 0.0, max_steps: int = 0,
                             arm_id: int = 0) -> None:
        pos = joint_config.get("positions") if isinstance(joint_config, dict) else joint_config
        if pos is None or len(pos) < 6:
            raise ToolError("robot.move_to_joints", "joint_config needs 6 positions")
        self.move_to_joints(list(pos)[:6], tolerance=tolerance if tolerance > 0 else 0.01, arm_id=int(arm_id))

    def go_home(self, arm_id: int = -1) -> dict:
        """Rise 5 cm, then ease to the rig home posture — one arm, or all (``-1``)."""
        ids = list(range(len(self.ik.arms))) if int(arm_id) < 0 else [int(arm_id)]
        out = {}
        for i in ids:
            arm = self.ik.arm(i)
            q = self._q(i)
            T = arm.tcp_world(q)
            up = T.copy()
            up[2, 3] = min(T[2, 3] + 0.05, self.ws_max[2] - 0.01)
            if up[2, 3] > T[2, 3] + 0.01:
                plan = self.ik.plan_linear(matrix_to_pose(T), matrix_to_pose(up), arm_id=i, seed_joints=q,
                                           num_waypoints=8, hold_rotation=True, tolerance=0.01)
                if plan is not None:
                    self._guarded_execute(i, [w["positions"] for w in plan["waypoints"]], "go_home rise")
                    q = self._q(i)
            home = arm.home_q
            n = 30
            wps = [list(q + (home - q) * (t * t * (3 - 2 * t))) for t in (k / n for k in range(1, n + 1))]
            self._guarded_execute(i, wps, "go_home")
            out[self._arm_name(i)] = "home"
        return out

    # ── planners (RSS motion.* semantics) ─────────────────────────────────

    def plan_joint(self, pose: dict, arm_id: int = 0, num_waypoints: int = 0, orientation: str = "axis") -> dict:
        if orientation not in ("axis", "lock"):
            raise ValueError(f"orientation must be 'axis' or 'lock', got {orientation!r}")
        plan = self.ik.plan_joint(pose, arm_id=int(arm_id), seed_joints=self._q(int(arm_id)),
                                  num_waypoints=int(num_waypoints),
                                  hold_rotation=(orientation == "lock" or self.honours_roll))
        result = {
            "trajectory": {"waypoints": plan["waypoints"]} if plan["waypoints"] else None,
            "planned": bool(plan["waypoints"]),
            "reason": plan.get("reason", ""),
            "position_error_m": float(plan["position_error_m"]),
            "rotation_error_rad": float(plan["rotation_error_rad"]),
            "joint_config": plan["joint_config"],
        }
        return self._gate_plan(int(arm_id), result, "plan_joint")

    def plan_linear(self, end: dict, start: dict | None = None, arm_id: int = 0, num_waypoints: int = 0,
                    axis: str = "", orientation: str = "axis", tolerance: float = 0.0,
                    world_config: dict | None = None, attached_object: dict | None = None,
                    allow_start_contact: bool = False, allow_goal_contact: bool = False,
                    contact_margin: float = 0.005, seed_joints: list | None = None) -> dict:
        if orientation not in ("axis", "lock"):
            raise ValueError(f"orientation must be 'axis' or 'lock', got {orientation!r}")
        if world_config is not None or attached_object is not None or allow_start_contact or allow_goal_contact:
            return {"trajectory": None, "planned": False, "position_error_m": None,
                    "reason": "this real-arm planner has no collision world; it cannot honour world_config, "
                              "attached_object or contact allowances"}
        i = int(arm_id)
        seed = np.asarray(seed_joints, dtype=np.float64) if seed_joints is not None else self._q(i)
        start = start or matrix_to_pose(self.ik.arm(i).tcp_world(seed))
        plan = self.ik.plan_linear(start, end, arm_id=i, seed_joints=seed, num_waypoints=int(num_waypoints),
                                   hold_rotation=(orientation == "lock" or self.honours_roll),
                                   tolerance=float(tolerance) if tolerance and tolerance > 0 else None)
        if plan is None:
            result = {"trajectory": None, "planned": False, "reason": self.ik.last_refusal or "no line",
                      "position_error_m": self.ik.last_position_error}
        else:
            result = {"trajectory": plan, "planned": True, "reason": "",
                      "position_error_m": self.ik.last_position_error}
        if seed_joints is not None:
            return result  # a probe from a posture the arm is not in: the gate runs at execution
        return self._gate_plan(i, result, "plan_linear")

    def _gate_plan(self, arm_id: int, result: dict, label: str) -> dict:
        traj = result.get("trajectory")
        if not traj:
            return result
        ok, reason, tcp_path = self._gate(arm_id, [w["positions"] for w in traj["waypoints"]])
        self._show_plan(arm_id, tcp_path, ok, label)
        if not ok:
            result.update(trajectory=None, planned=False, reason=f"refused by the safety gate: {reason}")
        return result

    def describe_planner(self) -> dict:
        return {"backend": "URAnalyticIK", "world_aware": False, "honours_roll": self.honours_roll}

    # ── descriptions / frames ─────────────────────────────────────────────

    def describe_gripper(self, arm_id: int = 0, role: str = "") -> dict:
        i = self._resolve_arm(arm_id, role)
        g = GRIPPER
        return {
            "arm": self._arm_name(i), "arm_id": i, "role": "work" if i == 0 else "support",
            "gripper": g["name"], "kind": "geared", "dof": 1,
            "span_m": g["span"], "open_footprint_m": 2.0 * g["pad_half_span"],
            "max_grasp_width_m": max(0.0, g["span"] - 2.0 * g["pad_clearance"]), "min_grasp_width_m": 0.0,
            "command": {"open": 0.0, "closed": 1.0, "convention": "0 = open, 1 = fully closed",
                        "observed_convention": "robot.get_gripper reports 1.0 = OPEN"},
            "travel_is_linear": False,
            "width_fit": {"source": "span", "at_open_m": g["span"], "at_closed_m": 0.0},
            "finger": {"stated": True, "reach_m": g["tip_reach"], "pad_thickness_m": g["pad_thickness"],
                       "clearance_m": g["pad_clearance"]},
            "housing": {"stated": True, "half_width_m": g["housing_half"], "rise_m": g["housing_rise"]},
            "approach_axis": dict(zip("xyz", map(float, self.approach_axis), strict=True)),
            "close_axis": dict(zip("xyz", map(float, self.close_axis), strict=True)),
            "finger_bodies": [["left_pad"], ["right_pad"]],
            # Dry run: the gripper is simulated and touches nothing, so contact checks cannot pass.
            "dry_run": bool(getattr(self.env, "dry_run", False)),
        }

    def set_grip(self, preset: str = "", width_m: float = -1.0, object_width_m: float = -1.0,
                 squeeze_m: float = 0.004, fraction_closed: float = -1.0, arm_id: int = 0, role: str = "",
                 ramp_steps: int = 24, settle_steps: int = 0) -> dict:
        """RoboSimStudio's ``robot.set_grip`` on the real hand, so a graph learned in the twin runs here.

        Same arguments and presets as the twin's tool (exactly one of ``preset``, ``width_m``,
        ``object_width_m``, ``fraction_closed``; ``pinch``/``grip`` take ``object_width_m``). The
        2F-85's vendor position is taken as linear in the opening (``width_fit`` in
        :meth:`describe_gripper`), so a width is ``width / span`` open. ``ramp_steps`` is accepted
        and unused: the hand's own controller ramps.
        """
        i = self._resolve_arm(arm_id, role)
        span = float(GRIPPER["span"])
        relative = preset in ("pinch", "grip")
        if relative and object_width_m < 0.0:
            raise ValueError(f"grip preset {preset!r} is relative to an object: give object_width_m with it")
        given = [bool(preset), width_m >= 0.0, object_width_m >= 0.0 and not relative, fraction_closed >= 0.0]
        if sum(given) != 1:
            raise ValueError("give exactly one of preset, width_m, object_width_m, fraction_closed"
                             " (object_width_m may accompany preset 'pinch' or 'grip')")
        if fraction_closed >= 0.0:
            requested = span * (1.0 - float(np.clip(fraction_closed, 0.0, 1.0)))
        elif preset:
            table = {"open": span, "close": 0.0, "pinch": object_width_m + 2.0 * GRIPPER["pad_clearance"],
                     "grip": object_width_m - squeeze_m}
            if preset not in table:
                raise ValueError(f"unknown grip preset {preset!r}; have {sorted(table)}")
            requested = table[preset]
        elif object_width_m >= 0.0:
            requested = object_width_m - squeeze_m
        else:
            requested = float(width_m)
        open_fraction = float(np.clip(requested / span, 0.0, 1.0))
        self.set_gripper(open_fraction, arm_id=i)
        if settle_steps > 0:
            with self._quiet_motion():
                self._settle(int(settle_steps))
        reason = "wider than the span" if requested > span else ("narrower than the jaws close" if requested < 0 else "")
        return {"arm": self._arm_name(i), "requested_width_m": requested, "command": 1.0 - open_fraction,
                "width_m": open_fraction * span, "clamped": bool(reason), "reason": reason,
                "position": self.get_gripper_fraction(arm_id=i)}

    def forward_kinematics(self, joint_configs: list | None = None, arm_id: int = 0, role: str = "") -> dict:
        """RoboSimStudio's ``robot.forward_kinematics``: the TCP pose (world frame) for each joint vector given.

        Pure kinematics on the chain the planner and the safety gate use; nothing moves. Returns
        ``joint_config`` (the arm's joints now) and ``poses``, one per vector, in the order given.
        """
        i = self._resolve_arm(arm_id, role)
        arm = self.ik.arm(i)
        poses = [matrix_to_pose(arm.tcp_world(np.asarray(q, dtype=np.float64)[:6])) for q in joint_configs or []]
        return {"arm": arm.name, "joint_config": self._q(i)[:6].tolist(), "poses": poses}

    def describe_arm(self, arm_id: int = 0, role: str = "") -> dict:
        i = self._resolve_arm(arm_id, role)
        arm = self.ik.arm(i)
        home = matrix_to_pose(arm.tcp_world(arm.home_q))
        return {
            "arm": arm.name, "arm_id": i, "role": "work" if i == 0 else "support",
            "arms": [{"arm_id": k, "name": a.name, "role": "work" if k == 0 else "support"}
                     for k, a in enumerate(self.ik.arms)],
            "dof": 6,
            "joint_limits": {"lower": arm.lower.tolist(), "upper": arm.upper.tolist()},
            "home_q": arm.home_q.tolist(),
            "home_pose": home,
            "tcp_offset": dict(zip("xyz", map(float, arm.tcp_offset), strict=True)),
            "tcp_rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0},
            "solver": {"name": "URAnalyticIK", "honours_roll": self.honours_roll},
        }

    def describe_workspace(self, arm_id: int = 0, role: str = "") -> dict:
        i = self._resolve_arm(arm_id, role)
        arm = self.ik.arm(i)
        g = GRIPPER
        home_z = float(arm.tcp_world(arm.home_q)[2, 3])
        floor = self.surface_z + g["tip_reach"] + MARGIN
        transport_z = max(home_z, floor)
        return {
            "arm": arm.name, "arm_id": i,
            "surface_z": self.surface_z,
            "transport_z": transport_z,
            "transport_above_surface_m": transport_z - self.surface_z,
            "transport_z_source": "home_tcp" if home_z >= floor else "surface_clearance",
            "align_clearance_m": g["tip_reach"] + g["housing_rise"] + g["pad_thickness"] + MARGIN,
            "pregrasp_standoff_m": max(2.0 * g["tip_reach"], 0.10),
            "margin_m": MARGIN,
            "envelope_stated": True,
            "workspace_min": self.ws_min.tolist(),
            "workspace_max": self.ws_max.tolist(),
        }

    def grasp_frame(self, approach: dict | None = None, close_heading_deg: float | None = None,
                    arm_id: int = 0, role: str = "") -> dict:
        i = self._resolve_arm(arm_id, role)
        down = np.array([approach["x"], approach["y"], approach["z"]] if approach else [0.0, 0.0, -1.0], float)
        n = float(np.linalg.norm(down))
        if n < 1e-9:
            raise ValueError("approach must be a non-zero direction")
        down /= n
        yaw = math.radians(float(close_heading_deg or 0.0))
        heading = np.array([math.cos(yaw), math.sin(yaw), 0.0])
        heading -= (heading @ down) * down
        if np.linalg.norm(heading) < 1e-6:
            heading = np.cross(down, [1.0, 0.0, 0.0])
            if np.linalg.norm(heading) < 1e-6:
                heading = np.cross(down, [0.0, 1.0, 0.0])
        heading /= np.linalg.norm(heading)

        def basis(a, c):
            a = np.asarray(a, float) / np.linalg.norm(a)
            c = np.asarray(c, float) - (np.asarray(c, float) @ a) * a
            c /= np.linalg.norm(c)
            return np.column_stack([a, c, np.cross(a, c)])

        local = basis(self.approach_axis, self.close_axis)

        def quat(h):
            R = basis(down, h) @ local.T
            q = matrix_to_quat_wxyz(R)
            return {"w": float(q[0]), "x": float(q[1]), "y": float(q[2]), "z": float(q[3])}

        return {"arm": self._arm_name(i), "rotation": quat(heading), "rotation_flipped": quat(-heading),
                "approach": dict(zip("xyz", map(float, down), strict=True)), "close_axis": dict(zip("xyz", map(float, heading), strict=True))}

    def wait_steps(self, steps: int = 30, timeout_s: float = 0.0) -> dict:
        n = max(0, int(steps))
        for _ in range(n):
            self.env._step_once()
        return {"steps": n}

    # ── perception ────────────────────────────────────────────────────────

    def mask_to_world_points(self, mask: Any, depth: Any, intrinsics: Any, camera_pose: Any) -> dict:
        m = np.asarray(mask) > 0
        d = np.asarray(depth, dtype=np.float32)
        if d.ndim == 3:
            d = d[..., 0]
        K = np.asarray(intrinsics, dtype=np.float64).reshape(3, 3)
        T = pose_to_matrix(camera_pose) if isinstance(camera_pose, dict) else np.asarray(camera_pose, float)
        if m.shape != d.shape:
            raise ToolError("geometry.mask_to_world_points", f"mask {m.shape} and depth {d.shape} differ")
        valid = m & np.isfinite(d) & (d >= 0.015) & (d <= 20.0)
        v, u = np.nonzero(valid)
        z = d[valid].astype(np.float64)
        pc = np.stack([(u - K[0, 2]) * z / K[0, 0], (v - K[1, 2]) * z / K[1, 1], z], axis=1)
        pw = (pc @ T[:3, :3].T + T[:3, 3]).astype(np.float32)
        upd = getattr(self.env, "viz_update", None)
        if upd is not None and len(pw):
            upd(points=[{"xyz": pw, "color": [255, 200, 0]}])
        return {"points": {"points": pw.reshape(-1, 3)}}

    # ── poses in world frame (arm_bases is None: env already reports world) ──

    def get_ee_pose(self, arm_id: int = 0):
        obs = self.env.get_observation()
        cart = obs.get(f"robot_cartesian_pos_{arm_id}")
        if cart is None:
            T = self._tcp_world(arm_id)
            return make_pose(T[:3, 3], matrix_to_quat_wxyz(T[:3, :3]))
        cart = np.asarray(cart, dtype=np.float64)
        return make_pose(cart[:3], cart[3:7])


__all__ = ["UR5eBimanualConnector"]
