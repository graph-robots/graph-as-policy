"""Connector base — the absorbed sim_bridge servicer logic, as plain Python.

The original research codebase ran four gRPC servicers (SimBridge,
Observation, Gripper, RobotControl) over one shared ``SimState``. This module
absorbs the *method bodies* of those servicers into a plain-Python
:class:`Connector`: the proto request/response packing becomes ordinary
arguments and :mod:`gap.types` TypedDicts, while every control loop —
convergence stepping, settle counts, velocity-mode action synthesis, TCP
offset/rotation application — is kept verbatim.

A connector owns one environment instance (built by ``gap.envs``) and exposes:

- ``tool_registry`` — a fresh :class:`gap.tools.ToolRegistry` with the
  connector's ``robot.*`` (and subclass ``sim.*``) tools registered;
- ``get_observation()`` — env obs dict → :class:`gap.types.Observation`;
- ``capabilities`` — what this backend can do (reset/success/video/world);
- ``close()`` + context-manager support.

Quaternion convention: gap is wxyz scalar-first throughout. Env classes
already convert sim-native xyzw quaternions to wxyz inside their obs dict
(``robot_cartesian_pos_*[3:7]``, camera ``pose[3:]``, object poses) — the
connector preserves that and never re-encodes image buffers.
"""

from __future__ import annotations

import contextlib
import logging
import math
from collections.abc import Callable
from dataclasses import dataclass
from typing import Any

import numpy as np
from gap_core.errors import ToolError
from gap_core.tools import ToolRegistry
from gap_core.types import (
    ArmState,
    CameraFrame,
    Observation,
    Se3Pose,
    Trajectory,
    make_pose,
)

from gap import env_config

logger = logging.getLogger(__name__)

# ---------------------------------------------------------------------------
# Free-space OSC servo budget (``go_to_pose_cartesian`` fast path).
#
# These are the numbers the authoring agent has to plan against, so they are
# named here rather than spelled inline: the tool catalog it reads interpolates
# them into the motion tools' summaries, and ``_servo_to_pose`` takes them as
# its defaults. One definition, so the prompt can never quote a stale budget.
# ---------------------------------------------------------------------------

#: Hard cap on control ticks for one servo'd leg. Exhausting it is
#: ``not_converged``: the arm kept making progress but ran out of budget.
OSC_MAX_TICKS = 200

#: Patience: consecutive ticks with no improvement in position error before
#: the leg is declared ``stalled``. Much shorter than :data:`OSC_MAX_TICKS`,
#: because a servo that has stopped closing the gap will not restart on its
#: own — it is in contact, at a joint limit, or commanding into a wall.
OSC_STALL_TICKS = 25

#: Convergence tolerances for a servo'd leg.
OSC_POS_TOL_M = 0.005
OSC_ROT_TOL_RAD = 0.05

# Franka Panda home joint configuration (radians)
_FRANKA_HOME_JOINTS = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]

# DROID Sim Evals home (matches third_party/droid-sim-evals nvidia_droid.py
# init_state). Differs from canonical Franka home in joint7=0 (vs π/4).
_DROID_HOME_JOINTS = [
    0.0,
    -math.pi / 5,      # -0.6283
    0.0,
    -4 * math.pi / 5,  # -2.5133
    0.0,
    3 * math.pi / 5,   # 1.8850
    0.0,
]


@dataclass(frozen=True)
class Capabilities:
    """What a connector backend can do.

    ``gap.execute`` and the benchmark harness branch on these instead of
    isinstance checks.
    """

    reset: bool = False
    success_check: bool = False
    video: bool = False
    world_state: bool = False


def _as_vec3(value: Any) -> np.ndarray | None:
    """Coerce a Vec3-ish value (dict / sequence / ndarray) to float64 (3,)."""
    if value is None:
        return None
    if isinstance(value, dict):
        return np.array(
            [float(value["x"]), float(value["y"]), float(value["z"])], dtype=np.float64
        )
    return np.asarray(value, dtype=np.float64).reshape(3)


def _vec3_dict(value: Any) -> dict[str, float]:
    """Render a Vec3-ish value as the ``{"x", "y", "z"}`` wire shape."""
    v = _as_vec3(value)
    if v is None:
        v = np.zeros(3)
    return {"x": float(v[0]), "y": float(v[1]), "z": float(v[2])}


def _unit(v: np.ndarray) -> np.ndarray:
    """Normalize, leaving a degenerate vector alone rather than dividing by 0."""
    n = float(np.linalg.norm(v))
    return v if n < 1e-12 else v / n


def _as_positions(value: Any) -> list[float]:
    """Coerce a JointState-ish value (dict with 'positions' / sequence) to list."""
    if isinstance(value, dict):
        value = value.get("positions", [])
    return [float(v) for v in np.asarray(value, dtype=np.float64).reshape(-1)]


class Connector:
    """Base connector: shared robot-control + observation logic.

    Args:
        env: An environment instance with the pinned env surface
            (``reset``/``step``/``get_observation`` → sim-native obs dict,
            ``task_completed``/``compute_reward``, video methods).
        config: An ``gap.envs.registry.EnvConfig`` (duck-typed — anything
            exposing ``arm_dof``, ``num_arms``, ``action_mode``,
            ``control_freq``, ``home_joints``, ``tcp_offset``,
            ``tcp_rotation_z``, ``arm_bases``, ``robot_urdf_path``,
            ``default_cameras``, ``is_real``).
        camera_names: Camera override; defaults to
            ``config.default_cameras``.
        ik: Optional pre-built IK backend. Defaults to a
            :class:`gap.connector.ik.CuRoboBackend` built from *config* (GPU,
            v0.8 MotionPlanner — linear motion with IK fallback via
            ``plan_to_pose``). Pass an explicit
            :class:`gap.connector.ik.PyRokiBackend` to opt into the in-process
            CPU-JAX path (e.g. for YAM/6-DOF or no-GPU machines).
    """

    def __init__(
        self,
        env: Any,
        config: Any,
        *,
        camera_names: list[str] | None = None,
        ik: Any | None = None,
    ) -> None:
        self.env = env
        self.config = config

        # --- SimState fields, populated from EnvConfig ---------------------
        self.action_mode: str = getattr(config, "action_mode", "absolute_joints")
        self.control_freq: float = float(getattr(config, "control_freq", 20.0))
        self.camera_names: list[str] = list(
            camera_names
            or getattr(config, "default_cameras", None)
            or ["agentview", "robot0_eye_in_hand"]
        )
        self._gripper_fraction: float = 1.0  # open
        self._arm_dof: int = int(getattr(config, "arm_dof", 7))
        self.num_arms: int = int(getattr(config, "num_arms", 1) or 1)
        configured_names = getattr(config, "joint_names", None)
        self._joint_names: tuple[str, ...] = tuple(
            configured_names or (f"joint{i + 1}" for i in range(self._arm_dof))
        )
        configured_limits = getattr(config, "joint_limits", None)
        self._joint_limits: np.ndarray | None = (
            np.asarray(configured_limits, dtype=np.float64).reshape(-1, 2)
            if configured_limits is not None else None
        )
        home = getattr(config, "home_joints", None)
        self._home_joints: list[float] | None = list(home) if home is not None else None
        self._robot_urdf_path: str | None = getattr(config, "robot_urdf_path", None)
        # TCP offset from EE link to tool tip (meters, in EE frame)
        self._tcp_offset: np.ndarray | None = _as_vec3(getattr(config, "tcp_offset", None))
        # TCP rotation from EE link to tool tip (scipy Rotation or None)
        tcp_rot_z = getattr(config, "tcp_rotation_z", None)
        if tcp_rot_z is not None:
            from scipy.spatial.transform import Rotation as _Rot

            self._tcp_rotation = _Rot.from_euler("z", float(tcp_rot_z))
        else:
            self._tcp_rotation = None
        # Per-arm base positions in world frame (bimanual)
        bases = getattr(config, "arm_bases", None)
        self._arm_bases: list[tuple[float, float, float]] | None = (
            [tuple(float(v) for v in b) for b in bases] if bases is not None else None
        )
        self.is_real: bool = bool(getattr(config, "is_real", False))
        # Opt-in policy-like OSC Cartesian servo for straight-line moves
        # (``go_to_pose_cartesian``): streams clamped OSC deltas through the
        # env's ``apply_policy_action`` with NO IK/planning. Off by default —
        # it drops collision-checking on the servo'd segment, so enable only
        # where the straight-line path is known clear. Falls back to the
        # cuRobo linear plan on stall/non-convergence.
        self._servo_enabled = env_config.libero_servo()
        self._cam_suspend_depth = 0  # reentrancy guard for _quiet_motion

        # Pluggable IK backend (cuRobo by default; lazy). Pass ``ik=`` to
        # opt into PyRoKi or any other backend implementing the same surface.
        self._ik = ik

        # Step-callback seam: the DataCollector (and anything else that
        # wants synchronized per-control-step samples) registers a callable
        # ``fn(action, obs, reward, done)`` here. Fired by ``step_once`` and
        # the connector-owned move loops.
        self._step_callbacks: list[Callable[[np.ndarray | None, dict, float, bool], None]] = []

        self._tool_registry: ToolRegistry | None = None
        self._closed = False

    # ------------------------------------------------------------------
    # IK backend (lazy)
    # ------------------------------------------------------------------

    @property
    def ik(self) -> Any:
        if self._ik is None:
            from gap.connector.ik import CuRoboBackend

            urdf_path = self._robot_urdf_path
            # "panda_description" is a robot_descriptions name, not a path.
            kwargs: dict[str, Any] = {}
            if urdf_path and not urdf_path.endswith((".urdf", ".xacro")):
                kwargs["robot_urdf"] = urdf_path
            elif urdf_path:
                kwargs["robot_urdf_path"] = urdf_path
            try:
                self._ik = CuRoboBackend(
                    arm_dof=self._arm_dof,
                    tcp_offset=self._tcp_offset,
                    tcp_rotation=self._tcp_rotation,
                    home_joints=self._home_joints,
                    arm_bases=self._arm_bases,
                    **kwargs,
                )
                logger.info(
                    "Connector.ik: using cuRobo backend (default; arm_dof=%d)",
                    self._arm_dof,
                )
            except NotImplementedError as e:
                # cuRobo doesn't bundle this arm (e.g. YAM 6-DOF). Surface a
                # clear pointer rather than silently auto-falling-back: per
                # the design decision, PyRoKi is opt-in only.
                raise RuntimeError(
                    f"Default IK backend (cuRobo) cannot serve this config: {e} "
                    f"Construct the connector with ik=PyRokiBackend(...) to "
                    f"opt into the in-process PyRoKi path."
                ) from e
        return self._ik

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        env = self.env
        return Capabilities(
            reset=hasattr(env, "reset"),
            success_check=hasattr(env, "task_completed"),
            video=hasattr(env, "enable_video_capture"),
            world_state=False,
        )

    # ------------------------------------------------------------------
    # Tool registry
    # ------------------------------------------------------------------

    @property
    def tool_registry(self) -> ToolRegistry:
        """A fresh ToolRegistry with this connector's tools registered.

        Built once per connector. Holds exactly the connector-owned
        ``robot.*`` / ``sim.*`` tools; skill-bundle ``@tool`` registrations
        are drained into it by ``gap.execute(..., skills=...)``.
        """
        if self._tool_registry is None:
            reg = ToolRegistry()
            self._register_robot_tools(reg)
            self._register_extra_tools(reg)
            self._tool_registry = reg
        return self._tool_registry

    def _register_robot_tools(self, reg: ToolRegistry) -> None:
        rc = reg.register_callable
        # Getters — no guard tag.
        rc("robot.get_observation", self.get_observation,
           summary="Capture the current observation: all cameras + arm states.")
        rc("robot.get_camera_pose", self.get_camera_pose,
           summary="Get one camera's world pose by name.")
        rc("robot.get_ee_pose", self._tool_get_ee_pose,
           summary="Get the end-effector pose in world frame.")
        rc("robot.get_gripper", self._tool_get_gripper,
           summary="Get the gripper open fraction (0 closed, 1 open).")
        rc("robot.get_gripper_pose", self._tool_get_gripper_pose,
           summary="Get the gripper (end-effector) pose in world frame.")
        rc("robot.get_joint_state", self.get_joint_state,
           summary="Get named arm joints with limits, normalized progress and margin.")
        # Embodiment description — what hand/arm/table this actually is, so
        # skills can stay written against "the hand" rather than a Panda.
        # Registered as defaults: a simulator connector that knows its own hand
        # (RoboSimStudio's does) registers the same names and replaces these.
        rc("robot.describe_gripper", self.describe_gripper,
           summary="Describe the hand: jaw span, closing axis, fingertip geometry.", default=True)
        rc("robot.describe_arm", self.describe_arm,
           summary="Describe the arm: DOF, TCP offset, home pose, solver behaviour.", default=True)
        rc("robot.describe_workspace", self.describe_workspace,
           summary="Describe the work surface height, transport height and clearances.", default=True)
        rc("robot.grasp_frame", self.grasp_frame,
           summary="Compose the wrist rotation for an approach direction "
                   "plus a jaw-closing heading, in THIS hand's axes.", default=True)
        # Control — sim_step guard tag.
        rc("robot.go_to_pose", self.go_to_pose,
           summary="Move the end-effector to a world-frame pose via IK. "
                   "BLOCKING: returns only once the arm converged, and raises "
                   "if it did not. Pose is the connector-public TCP pose in "
                   "connector-world coordinates — the same frame "
                   "robot.get_ee_pose reports; never pre-apply a TCP offset.",
           tags=("sim_step",))
        rc("robot.go_to_pose_cartesian", self.go_to_pose_cartesian,
           summary="Move the end-effector along a straight Cartesian line. "
                   "BLOCKING and collision-UNAWARE on the servo'd fast path, "
                   "so use it only for a leg you know is clear. Raises "
                   f"'stalled' after {OSC_STALL_TICKS} ticks with no progress "
                   f"in position error, or 'not_converged' after "
                   f"{OSC_MAX_TICKS} ticks; tolerances are "
                   f"{OSC_POS_TOL_M} m / {OSC_ROT_TOL_RAD} rad. Either error "
                   "reports target, achieved and residual in "
                   "connector-world coordinates. Split a long move into "
                   "short legs rather than raising the budget.",
           tags=("sim_step",))
        if self.supports_joint_position_control():
            rc("robot.move_to_joints", self._tool_move_to_joints,
               summary="Move the arm to a joint configuration, blocking until converged.",
               tags=("sim_step",))
            rc("robot.execute_trajectory", self._tool_execute_trajectory,
               summary="Execute a joint trajectory waypoint-by-waypoint.",
               tags=("sim_step",))
            rc("robot.go_home", self.go_home,
               summary="Move all arms to the home joint configuration.",
               tags=("sim_step",))
        rc("robot.open_gripper", self.open_gripper,
           summary="Open the gripper and let physics settle.",
           tags=("sim_step",))
        rc("robot.close_gripper", self.close_gripper,
           summary="Close the gripper and let physics settle.",
           tags=("sim_step",))
        rc("robot.set_grip", self.set_grip,
           summary="Command a partial jaw opening (0 closed, 1 open).",
           tags=("sim_step",), default=True)  # a simulator connector may supply its own
        rc("robot.wait_steps", self.wait_steps,
           summary="Hold the current command for N control steps and let physics settle.",
           tags=("sim_step",), default=True)  # a simulator connector may supply its own
        # Planning.
        rc("robot.solve_ik", self._tool_solve_ik,
           summary="Solve IK for a world-frame pose; returns a joint configuration.",
           tags=("planning",))

    def supports_joint_position_control(self) -> bool:
        """Whether this connector can honestly execute joint targets."""
        return (
            hasattr(self.env, "move_to_joints_blocking")
            or (
                self.action_mode in {"velocity_joints", "absolute_joints"}
                and hasattr(self.env, "handle")
            )
        )

    def joint_position_control_reason(self) -> str:
        if self.supports_joint_position_control():
            return "available"
        return (
            f"unsupported by {type(self.env).__name__}: controller action_mode="
            f"{self.action_mode!r} does not accept joint-position commands"
        )

    def _register_extra_tools(self, reg: ToolRegistry) -> None:
        """Subclass hook for ``sim.*`` / backend-specific tools."""

    # ------------------------------------------------------------------
    # Step-callback seam (DataCollector et al.)
    # ------------------------------------------------------------------

    def add_step_callback(
        self, fn: Callable[[np.ndarray | None, dict, float, bool], None]
    ) -> None:
        """Register ``fn(action, obs, reward, done)``, fired once per sim step.

        ``action`` is the low-level action applied for the step (``None``
        when the env stepped internally, e.g. via its own blocking-move
        loop); ``obs`` is the env's structured obs dict.
        """
        self._step_callbacks.append(fn)

    def remove_step_callback(self, fn: Callable) -> None:
        if fn in self._step_callbacks:
            self._step_callbacks.remove(fn)

    def _emit_step(
        self, action: np.ndarray | None, obs: dict, reward: float, done: bool
    ) -> None:
        for fn in self._step_callbacks:
            try:
                fn(action, obs, reward, done)
            except Exception:
                logger.warning("step callback %r failed", fn, exc_info=True)

    # ------------------------------------------------------------------
    # Frame helpers (verbatim from SimState)
    # ------------------------------------------------------------------

    def world_pose_to_base_frame(self, pose: Se3Pose, arm_id: int) -> Se3Pose:
        """Transform a world-frame Se3Pose into the specified arm's base frame.

        For YAM the arms are mounted vertically (identity base rotation), so
        only a translational offset is needed.
        """
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pose
        bx, by, bz = self._arm_bases[arm_id]
        p = pose["position"]
        return {
            "position": {"x": p["x"] - bx, "y": p["y"] - by, "z": p["z"] - bz},
            "rotation": dict(pose["rotation"]),
        }

    def _base_frame_to_world(self, pos: np.ndarray, arm_id: int) -> np.ndarray:
        """Translate a base-frame position to world frame."""
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pos
        base = np.array(self._arm_bases[arm_id])
        return pos + base

    # ------------------------------------------------------------------
    # Observation assembly (absorbed ObservationServicer)
    # ------------------------------------------------------------------

    def _build_camera_frame(self, cam_data: dict) -> CameraFrame:
        frame: dict[str, Any] = {
            "name": cam_data.get("name", ""),
            "rgb": cam_data["images"]["rgb"],
        }
        # Depth and intrinsics may not be available (e.g. sim without depth)
        depth = cam_data.get("images", {}).get("depth")
        if depth is not None:
            d = np.asarray(depth, dtype=np.float32)
            # Sim backends (robosuite/MuJoCo) emit depth as [H, W, 1];
            # gap.types.CameraFrame mandates float32 [H, W] — squeeze the
            # trailing channel once, here at the boundary.
            if d.ndim == 3 and d.shape[-1] == 1:
                d = d[..., 0]
            frame["depth"] = d
        intrinsics = cam_data.get("intrinsics")
        if intrinsics is not None:
            frame["intrinsics"] = np.asarray(intrinsics)
        pose = cam_data.get("pose")
        if pose is not None:
            # Env convention: [x, y, z, qw, qx, qy, qz] (wxyz, converted from
            # sim-native xyzw inside the env class).
            frame["pose"] = make_pose(pose[:3], pose[3:])
        return frame  # type: ignore[return-value]

    def _build_observation(self, obs: dict) -> Observation:
        """Assemble a gap Observation from the env's structured obs dict.

        Verbatim port of ``SimState.build_observation_response`` minus the
        proto packing: rgb/depth stay numpy (no byte re-encoding), and the
        quaternions in ``robot_cartesian_pos_*`` / camera ``pose`` arrive
        wxyz from the env class.
        """
        cameras: list[CameraFrame] = []
        for cam_key in self.camera_names:
            if cam_key in obs and "images" in obs[cam_key]:
                cam = obs[cam_key]
                cam["name"] = cam_key
                cameras.append(self._build_camera_frame(cam))

        arms: list[ArmState] = []
        num_arms = getattr(self, "num_arms", 1)
        dof = self._arm_dof
        for arm_id in range(num_arms):
            joint_pos = obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1))
            joint_pos = np.asarray(joint_pos)
            joints = joint_pos[:dof] if len(joint_pos) >= dof else joint_pos
            gripper_frac = float(joint_pos[dof]) if len(joint_pos) > dof else 1.0

            cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
            ee_pos = cart[:3].copy()
            ee_quat = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
            # FK returns base-frame; convert to world frame
            ee_pos = self._base_frame_to_world(ee_pos, arm_id)

            # Surface the simulator's raw per-finger gripper qpos when
            # available. LIBERO π-series checkpoints consume this verbatim
            # as the last 2 entries of their state vector.
            gripper_qpos: list[float] = []
            if hasattr(self.env, "handle"):
                try:
                    raw = self.env._current_obs.get(  # type: ignore[attr-defined]
                        f"robot{arm_id}_gripper_qpos"
                    )
                    if raw is not None:
                        gripper_qpos = [float(x) for x in np.asarray(raw).reshape(-1)]
                except Exception:
                    gripper_qpos = []

            # Raw VLA proprioception state (policy-training-exact layout).
            proprio = obs.get(f"robot_proprio_pi05_libero_{arm_id}")

            arm: dict[str, Any] = {
                "joint_state": {"positions": np.asarray(joints, dtype=np.float64)},
                "gripper_fraction": gripper_frac,
                "ee_pose": make_pose(ee_pos, ee_quat),
            }
            if gripper_qpos:
                arm["gripper_qpos"] = np.asarray(gripper_qpos, dtype=np.float64)
            if proprio is not None:
                arm["proprio_state"] = np.asarray(proprio, dtype=np.float64).reshape(-1)
            arms.append(arm)  # type: ignore[arg-type]

        return {"cameras": cameras, "arms": arms}

    def get_observation(self) -> Observation:
        """Capture the current observation (all cameras + all arm states)."""
        obs = self.env.get_observation()
        return self._build_observation(obs)

    def get_camera_pose(self, camera_name: str) -> dict:
        """Get one camera's world pose. Raises ToolError when unknown."""
        obs = self.env.get_observation()
        if camera_name in obs and "pose" in obs[camera_name]:
            pose = obs[camera_name]["pose"]
            return {"pose": make_pose(pose[:3], pose[3:])}
        raise ToolError("robot.get_camera_pose", f"Camera '{camera_name}' not found")

    # ------------------------------------------------------------------
    # EE pose / gripper getters (verbatim from SimState)
    # ------------------------------------------------------------------

    def get_ee_pose(self, arm_id: int = 0) -> Se3Pose:
        obs = self.env.get_observation()
        cart = np.asarray(obs.get(f"robot_cartesian_pos_{arm_id}", np.zeros(8)))
        ee_pos = cart[:3].copy()
        ee_quat = cart[3:7] if len(cart) >= 7 else np.array([1.0, 0.0, 0.0, 0.0])
        # FK returns base-frame; convert to world frame for public API
        ee_pos = self._base_frame_to_world(ee_pos, arm_id)
        return make_pose(ee_pos, ee_quat)

    def _tool_get_ee_pose(self, arm_id: int = 0) -> dict:
        return {"pose": self.get_ee_pose(arm_id=arm_id)}

    def _tool_get_gripper_pose(self, arm_id: int = 0) -> dict:
        return {"pose": self.get_ee_pose(arm_id=arm_id)}

    def get_gripper_fraction(self, arm_id: int = 0) -> float:
        obs = self.env.get_observation()
        dof = self._arm_dof
        joint_pos = np.asarray(obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1)))
        return float(joint_pos[dof]) if len(joint_pos) > dof else self._gripper_fraction

    def _tool_get_gripper(self, arm_id: int = 0) -> dict:
        return {"position": self.get_gripper_fraction(arm_id=arm_id)}

    def get_joint_state(
        self, arm_id: int = 0, near_limit_threshold: float = 0.05
    ) -> dict:
        """Return simulator-neutral named joint diagnostics for one arm."""
        values = self._current_joints(arm_id)
        if values is None:
            raise ToolError("robot.get_joint_state", f"arm {arm_id} joint state unavailable")
        threshold = max(0.0, float(near_limit_threshold))
        rows: list[dict[str, Any]] = []
        for index, position in enumerate(values):
            name = (
                self._joint_names[index]
                if index < len(self._joint_names) else f"joint{index + 1}"
            )
            row: dict[str, Any] = {
                "name": name, "position": float(position), "limit": None,
                "progress": None, "margin": None, "distance_to_limit": None,
                "near_limit": False,
            }
            if self._joint_limits is not None and index < len(self._joint_limits):
                lower, upper = (float(v) for v in self._joint_limits[index])
                span = upper - lower
                if np.isfinite(lower) and np.isfinite(upper) and span > 0:
                    progress = (float(position) - lower) / span
                    margin = min(progress, 1.0 - progress)
                    row.update({
                        "limit": {"lower": lower, "upper": upper},
                        "progress": float(progress),
                        "margin": float(margin),
                        "distance_to_limit": float(
                            min(float(position) - lower, upper - float(position))
                        ),
                        "near_limit": bool(margin <= threshold + 1e-12),
                    })
            rows.append(row)
        return {
            "arm_id": int(arm_id), "near_limit_threshold": threshold,
            "limits_available": self._joint_limits is not None, "joints": rows,
        }

    def _joint_limit_diagnostic(self, arm_id: int = 0) -> str:
        state = self.get_joint_state(arm_id=arm_id)
        limited = [row for row in state["joints"] if row["margin"] is not None]
        if not limited:
            return "joint limits unavailable"
        limited.sort(key=lambda row: row["margin"])
        rows = [row for row in limited if row["near_limit"]] or limited[:2]
        fields = []
        for row in rows[:3]:
            limit = row["limit"]
            side = "lower" if row["progress"] <= 0.5 else "upper"
            fields.append(
                f"{row['name']}={row['position']:.4f}, "
                f"{side}={limit[side]:.4f}, margin={row['margin']:.3f}"
            )
        return "; ".join(fields)

    # ------------------------------------------------------------------
    # Low-level sim operations (verbatim from SimState)
    # ------------------------------------------------------------------

    def move_to_joints(
        self,
        target_joints: list[float],
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
        settle_steps: int | None = None,
    ) -> None:
        """Move to target joints, blocking until converged, then settle.

        ``max_steps == 0`` is a streaming fire-and-forget path: publish the
        joint target once and return without polling or settling. Only
        meaningful on envs whose ``move_to_joints_blocking`` honors the same
        convention (e.g. the real Franka env).
        """
        dof = self._arm_dof
        target = np.array(target_joints[:dof], dtype=np.float64)

        if hasattr(self.env, "move_to_joints_blocking"):
            self.env.move_to_joints_blocking(
                target, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id
            )
            if max_steps == 0:
                return
            # Let physics settle after convergence (YAM needs this; Panda doesn't)
            if settle_steps is None:
                settle_steps = 30 if dof == 6 else 0
            self._settle(settle_steps)
            return

        # Manual convergence loop (single-arm fallback for sim envs)
        for step in range(max_steps):
            obs = self.env.get_observation()
            current = np.asarray(
                obs.get(f"robot_joint_pos_{arm_id}", np.zeros(dof + 1))
            )[:dof]
            if np.linalg.norm(current - target) < tolerance and step > 0:
                break

            if self.action_mode == "velocity_joints":
                delta = (target - current) * self.control_freq
                action = np.concatenate([delta, [1.0 - self._gripper_fraction * 2.0]])
            else:
                gv = 1.0 - self._gripper_fraction * 2.0
                action = np.concatenate([target, [gv, gv]])[: dof + 1]

            self.env.handle.step(action)
            self._emit_step(
                action,
                obs,
                float(self.env.compute_reward() or 0.0),
                bool(getattr(self.env, "_current_done", False)),
            )

    def stream_trajectory(
        self,
        waypoints: list[list[float]],
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        """Feed-forward stream a joint trajectory (1 sim tick per waypoint).

        Returns True if the env supports streaming and the call ran; False
        if the caller should fall back to per-waypoint ``move_to_joints``.
        """
        if not hasattr(self.env, "stream_joint_trajectory"):
            return False
        dof = self._arm_dof
        wps = [list(np.asarray(w, dtype=np.float64)[:dof]) for w in waypoints]
        self.env.stream_joint_trajectory(
            wps,
            settle_tolerance=settle_tolerance,
            settle_max_steps=settle_max_steps,
            arm_id=arm_id,
        )
        return True

    def step_once(self) -> tuple[dict, float, bool, bool, dict]:
        """Execute one sim step holding current position + current gripper.

        Calls ``env._step_once()`` which advances physics with zero joint
        velocity and the current gripper fraction (set via
        :meth:`set_gripper`). Returns ``(obs, reward, terminated, truncated,
        info)``.
        """
        self.env._step_once()
        obs = self.env.get_observation()

        reward = self.env.compute_reward()
        done = getattr(self.env, "_current_done", False)
        truncated = self.env._sim_step_count >= self.env.max_steps
        self._emit_step(None, obs, float(reward or 0.0), bool(done))
        info: dict = {}
        latency_fn = getattr(self.env, "get_latency_info", None)
        if latency_fn is not None:
            try:
                info.update(latency_fn())
            except Exception:
                pass
        return obs, reward, bool(done), truncated, info

    def _settle(self, n: int) -> None:
        """Advance ``n`` hold steps (physics + video frame only). Skips the
        per-step ``get_observation`` + ``compute_reward`` + callback emit that
        ``step_once`` does — a hold-in-place settle needs none of it — UNLESS a
        step callback (e.g. a DataCollector) is attached, in which case each
        step is recorded via the full ``step_once`` path."""
        if self._step_callbacks or not hasattr(self.env, "_step_once"):
            for _ in range(int(n)):
                self.step_once()
        else:
            for _ in range(int(n)):
                self.env._step_once()

    def set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Set gripper open fraction (0 closed, 1 open). Local call."""
        self._gripper_fraction = float(np.clip(fraction, 0.0, 1.0))
        if hasattr(self.env, "_set_gripper"):
            self.env._set_gripper(self._gripper_fraction, arm_id=arm_id)

    # ------------------------------------------------------------------
    # Gripper (absorbed GripperServicer)
    # ------------------------------------------------------------------

    @contextlib.contextmanager
    def _quiet_motion(self):
        """Suspend the env's per-step camera rendering for a motion/settle
        segment (``GAP_LIBERO_MOTION_RENDER=0``), restoring + refreshing on exit
        so the next perception read sees fresh frames. Reentrant; a no-op when
        the env doesn't support it or rendering-in-motion is on."""
        env = self.env
        setf = getattr(env, "set_cameras_active", None)
        if (
            setf is None
            or getattr(env, "_motion_render", True)
            or self._cam_suspend_depth > 0
            or self._step_callbacks  # a DataCollector needs per-step frames
        ):
            self._cam_suspend_depth += 1
            try:
                yield
            finally:
                self._cam_suspend_depth -= 1
            return
        self._cam_suspend_depth += 1
        setf(False)
        try:
            yield
        finally:
            setf(True)
            self._cam_suspend_depth -= 1
            try:
                env.refresh_camera_obs()
            except Exception:
                logger.debug("refresh_camera_obs failed", exc_info=True)

    def open_gripper(self, settle_steps: int = 40, arm_id: int = 0) -> dict:
        """Open the gripper and run the settle loop (default 40 steps)."""
        settle = settle_steps if settle_steps > 0 else 40
        self.set_gripper(1.0, arm_id=arm_id)
        with self._quiet_motion():
            self._settle(settle)
        position = self.get_gripper_fraction(arm_id=arm_id)
        logger.info(
            "robot.open_gripper arm=%d settle=%d -> position=%.4f",
            arm_id, settle, position,
        )
        return {"position": position}

    def close_gripper(self, settle_steps: int = 60, arm_id: int = 0) -> dict:
        """Close the gripper and run the settle loop (default 60 steps)."""
        settle = settle_steps if settle_steps > 0 else 60
        self.set_gripper(0.0, arm_id=arm_id)
        with self._quiet_motion():
            self._settle(settle)
        position = self.get_gripper_fraction(arm_id=arm_id)
        logger.info(
            "robot.close_gripper arm=%d settle=%d -> position=%.4f",
            arm_id, settle, position,
        )
        return {"position": position}

    # ------------------------------------------------------------------
    # Embodiment description
    # ------------------------------------------------------------------
    #
    # Skills are written against "the hand", not against a Panda. These four
    # tools are how a skill asks what it is actually holding, so the same
    # grasp logic survives a change of gripper, arm, or table height. They
    # are pure description — no privilege, no physics step — which is why
    # they live here and not in the sim connector.

    def _gripper_spec(self) -> Any:
        from gap.envs.registry import GripperSpec

        return getattr(self.config, "gripper", None) or GripperSpec()

    def _workspace_spec(self) -> Any:
        from gap.envs.registry import WorkspaceSpec

        return getattr(self.config, "workspace", None) or WorkspaceSpec()

    def describe_gripper(self, arm_id: int = 0) -> dict:
        """What the hand is: jaw span, closing axis, fingertip geometry."""
        g = self._gripper_spec()
        closed = g.width_at_closed_m
        payload: dict[str, Any] = {
            "name": g.name,
            "arm_id": int(arm_id),
            "type": "parallel_jaw",
            "span_m": float(g.span_m),
            "max_grasp_width_m": float(g.span_m),
            "min_grasp_width_m": float(g.min_grasp_width_m),
            "close_axis": _vec3_dict(g.close_axis),
            "approach_axis": _vec3_dict(g.approach_axis),
            "width_fit": {
                # "span" says these are the declared span and a zero, not a
                # calibration — verifying-grasps keys its wording off this.
                "source": "measured" if closed is not None else "span",
                "at_open_m": float(g.span_m),
                "at_closed_m": float(closed or 0.0),
            },
        }
        reach = g.finger_reach_m
        payload["finger"] = {
            "stated": reach is not None,
            "reach_m": float(reach or 0.0),
            "clearance_m": float(g.finger_clearance_m),
        }
        return payload

    def describe_arm(self, arm_id: int = 0) -> dict:
        """What the arm is: DOF, TCP, home, and what its solver honours."""
        off = self._default_tcp_offset()
        base = None
        if self._arm_bases is not None and arm_id < len(self._arm_bases):
            base = _vec3_dict(self._arm_bases[arm_id])
        name = f"arm{arm_id}" if self.num_arms > 1 else "arm"
        return {
            "arm_id": int(arm_id),
            "arm": name,
            "name": name,
            "dof": int(self._arm_dof),
            "num_arms": int(self.num_arms),
            "tcp_offset": _vec3_dict(off),
            "home_joints": list(self._home_joints) if self._home_joints else None,
            "base_position": base,
            "control_freq": float(self.control_freq),
            # go_to_pose solves full-SE(3) IK, so a commanded wrist roll is
            # reached rather than dropped. Connectors that only servo
            # position + approach must report False here.
            "solver": {"honours_roll": True, "blocking": True},
            "joint_position_control": {
                "available": self.supports_joint_position_control(),
                "reason": self.joint_position_control_reason(),
            },
            "rot_key": "rotation",
        }

    def _workspace_payload(self) -> dict:
        """Subclass hook: the sim connector measures ``surface_z`` instead."""
        w = self._workspace_spec()
        surface_z = float(w.surface_z)
        transport_z = w.transport_z
        return {
            "surface_z": surface_z,
            "transport_z": float(transport_z) if transport_z is not None else surface_z + 0.25,
            "align_clearance_m": float(w.align_clearance_m),
            "source": "config",
        }

    def describe_workspace(self, arm_id: int = 0) -> dict:
        """Where the work surface is and how high to carry things over it."""
        payload = self._workspace_payload()
        payload["arm_id"] = int(arm_id)
        return payload

    def grasp_frame(
        self,
        approach: Any | None = None,
        close_heading_deg: float = 0.0,
        arm_id: int = 0,
    ) -> dict:
        """The wrist rotation that points the hand *approach* and closes along
        *close_heading_deg*.

        Asked for rather than composed, because "closing heading" only means
        something once you know which way this particular hand's jaws travel.
        ``approach`` defaults to straight down; ``close_heading_deg`` is a
        compass bearing in the world XY plane (0 = +x, 90 = +y).

        The heading is NOT the object's long-axis heading. They are ninety
        degrees apart, and confusing them closes the fingers down the length
        of a handle instead of across it.
        """
        from scipy.spatial.transform import Rotation as _R

        g = self._gripper_spec()
        e_close = _unit(_as_vec3(g.close_axis))
        e_appr = _unit(_as_vec3(g.approach_axis))
        e_close = _unit(e_close - float(e_close @ e_appr) * e_appr)

        w_appr = _unit(_as_vec3(approach) if approach is not None else np.array([0.0, 0.0, -1.0]))
        heading = math.radians(float(close_heading_deg))
        w_close = np.array([math.cos(heading), math.sin(heading), 0.0])
        w_close = w_close - float(w_close @ w_appr) * w_appr
        if float(np.linalg.norm(w_close)) < 1e-8:
            # Heading is parallel to the approach: the roll is unconstrained,
            # so pick any perpendicular rather than emitting a NaN rotation.
            fallback = np.array([1.0, 0.0, 0.0])
            if abs(float(fallback @ w_appr)) > 0.9:
                fallback = np.array([0.0, 1.0, 0.0])
            w_close = fallback - float(fallback @ w_appr) * w_appr
        w_close = _unit(w_close)

        ee = np.column_stack([e_close, e_appr, np.cross(e_close, e_appr)])
        world = np.column_stack([w_close, w_appr, np.cross(w_close, w_appr)])
        x, y, z, w = _R.from_matrix(world @ ee.T).as_quat()
        return {
            "rotation": {"x": float(x), "y": float(y), "z": float(z), "w": float(w)},
            "approach": _vec3_dict(w_appr),
            "close_axis_world": _vec3_dict(w_close),
            "close_heading_deg": float(close_heading_deg),
            "arm_id": int(arm_id),
        }

    def wait_steps(self, steps: int = 20, arm_id: int = 0) -> dict:
        """Hold the current command for *steps* control steps and let physics
        settle. The honest way to wait for a released object to come to rest."""
        n = max(0, int(steps))
        with self._quiet_motion():
            self._settle(n)
        return {"steps": n}

    def set_grip(self, fraction: float, settle_steps: int = 0, arm_id: int = 0) -> dict:
        """Command a partial jaw opening (0 closed, 1 open).

        ``open_gripper``/``close_gripper`` are the ends of this range; this is
        for the middle — a pre-shape wider than the object but narrower than
        the neighbour it must not sweep.
        """
        self.set_gripper(fraction, arm_id=arm_id)
        if settle_steps > 0:
            with self._quiet_motion():
                self._settle(int(settle_steps))
        return {"position": self.get_gripper_fraction(arm_id=arm_id)}

    # ------------------------------------------------------------------
    # IK orchestration (absorbed RobotControlServicer)
    # ------------------------------------------------------------------

    def _default_tcp_offset(self) -> np.ndarray:
        """Fallback TCP offset used when the caller doesn't provide one.

        Prefer the value the env config supplied (``self._tcp_offset`` —
        franka_real uses -0.157 m for the Robotiq tool, libero uses -0.1 m
        for the panda hand). Fall back to the historical Panda fingertip
        default of -0.1 m along Z only when no config offset is configured.
        YAM's fixed TCP is applied inside the backend itself.
        """
        if self._tcp_offset is not None:
            return np.asarray(self._tcp_offset, dtype=np.float64).reshape(3)
        return np.array([0.0, 0.0, -0.1], dtype=np.float64)

    def _current_joints(self, arm_id: int) -> list[float] | None:
        """Read the current joint positions (simulator order) for seeding IK."""
        if self.env is None:
            return None
        obs = self.env.get_observation()
        cur = obs.get(f"robot_joint_pos_{arm_id}")
        if cur is None:
            return None
        return list(np.asarray(cur[: self._arm_dof], dtype=np.float64))

    def _solve_ik(
        self,
        pose: Se3Pose,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
    ) -> list[float] | None:
        """Delegate to the configured IK backend.

        Returns joint positions in simulator-native order, or None.
        """
        if self.ik is None:
            raise RuntimeError("IK backend not configured")
        off = _as_vec3(tcp_offset)
        if off is None:
            off = self._default_tcp_offset()
        seed = self._current_joints(arm_id)
        joints = self.ik.solve_ik(
            pose,
            arm_id=arm_id,
            seed_joints=seed,
            tcp_offset=off,
        )
        if joints is None:
            logger.warning("_solve_ik FAILED for arm=%d", arm_id)
            return None
        p = pose["position"]
        logger.info(
            "_solve_ik arm=%d tcp=(%.4f,%.4f,%.4f) joints=[%s]",
            arm_id, p["x"], p["y"], p["z"],
            ", ".join(f"{j:.4f}" for j in joints),
        )
        return joints

    def _tool_solve_ik(
        self,
        pose: Se3Pose,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
    ) -> dict:
        """Tool wrapper: raises ToolError on failure (source: INVALID_ARGUMENT)."""
        joints = self._solve_ik(pose, tcp_offset, arm_id=arm_id)
        if joints is None:
            raise ToolError("robot.solve_ik", "Target pose is unreachable")
        return {"joint_config": {"positions": list(joints)}, "success": True}

    def _execute_trajectory(
        self,
        trajectory: Trajectory,
        subsample: int,
        tolerance: float,
        max_steps_per_wp: int,
        arm_id: int = 0,
        reverse_joints: bool = False,
    ) -> None:
        """Execute a joint trajectory waypoint-by-waypoint.

        Args:
            reverse_joints: If True, reverse joint order per waypoint (needed
                for PyRoKI which returns joints in reversed URDF parse order
                for 6-DOF arms).

        Forces the env into ``closed_loop`` joint motion mode for the
        duration of the trajectory: qpos-teleport per waypoint bypasses
        contact integration, which is fine for contact-free single moves but
        breaks the descent into the target object. Single moves keep their
        configured mode.
        """
        saved_mode = getattr(self.env, "_joint_motion_mode", None)
        if saved_mode is not None and saved_mode != "closed_loop":
            self.env._joint_motion_mode = "closed_loop"
        try:
            subsample = max(subsample, 1)
            waypoints = trajectory["waypoints"]
            total = len(waypoints)
            # Streaming path: ignore `subsample` — the dense interpolation IS
            # the control sequence; one final settle on the last waypoint
            # replaces the per-waypoint convergence loop.
            if getattr(self.env, "_stream_enabled", True) and hasattr(
                self.env, "stream_joint_trajectory"
            ):
                wps_all: list[list[float]] = []
                for waypoint in waypoints:
                    joints = _as_positions(waypoint)
                    if reverse_joints:
                        joints.reverse()
                    wps_all.append(joints)
                self.stream_trajectory(
                    wps_all,
                    settle_tolerance=tolerance,
                    settle_max_steps=max(max_steps_per_wp * 4, 60),
                    arm_id=arm_id,
                )
                logger.info(
                    "robot.execute_trajectory streamed %d waypoints (subsample ignored)",
                    total,
                )
                return
            # Per-waypoint blocking fallback.
            executed = 0
            for i, waypoint in enumerate(waypoints):
                if i % subsample != 0 and i != total - 1:
                    continue
                joints = _as_positions(waypoint)
                if reverse_joints:
                    joints.reverse()
                executed += 1
                self.move_to_joints(
                    joints, tolerance=tolerance, max_steps=max_steps_per_wp, arm_id=arm_id,
                )
            logger.info(
                "robot.execute_trajectory: executed %d/%d waypoints", executed, total,
            )
        finally:
            if saved_mode is not None:
                self.env._joint_motion_mode = saved_mode

    # ------------------------------------------------------------------
    # High-level motion tools (absorbed RobotControlServicer handlers)
    # ------------------------------------------------------------------

    def go_to_pose(
        self,
        pose: Se3Pose,
        tolerance: float = 0.0,
        max_steps: int = 0,
        tcp_offset: Any | None = None,
        arm_id: int = 0,
        z_approach: float = 0.0,
        nonblocking: bool = False,
    ) -> None:
        """Move the end-effector to a world-frame pose via IK.

        ``tolerance``/``max_steps`` <= 0 take the source defaults (0.01 rad /
        120 steps for Panda; 0.03 / 300 for 6-DOF arms). ``z_approach > 0``
        first moves to a point that far above the target.
        """
        if not pose:
            raise ToolError("robot.go_to_pose", "pose required")
        # NB: ``go_to_pose`` is NOT servo'd. It is the contact-rich grasp
        # descend (and precision hand-off) path, which needs cuRobo's IK +
        # controlled approach; an OSC straight-line servo onto the object
        # stalls on contact and misses the grasp (measured reward 1.0 -> 0.0).
        # Only the free-space straight legs (``go_to_pose_cartesian``) servo.

        if tcp_offset is None:
            # Use the env's configured TCP (franka_real: -0.157, libero: -0.1)
            # rather than a hardcoded Panda default — the latter silently
            # shifts the IK target by ~5.7 mm on the real Robotiq stack.
            off = self._default_tcp_offset()
            tcp_offset = {"x": float(off[0]), "y": float(off[1]), "z": float(off[2])}
        is_yam = self._arm_dof == 6
        tolerance = tolerance if tolerance > 0 else (0.03 if is_yam else 0.01)
        # nonblocking=True forces max_steps=0, which makes move_to_joints_blocking
        # publish the command once and return. Each new go_to_pose call
        # supersedes the last.
        if nonblocking:
            max_steps = 0
        else:
            max_steps = max_steps if max_steps > 0 else (300 if is_yam else 120)

        with self._quiet_motion():
            # z_approach: first move above target
            if z_approach > 0 and pose.get("position"):
                p = pose["position"]
                approach_pose: Se3Pose = {
                    "position": {"x": p["x"], "y": p["y"], "z": p["z"] + z_approach},
                    "rotation": dict(pose["rotation"]),
                }
                joints = self._solve_ik(approach_pose, tcp_offset, arm_id=arm_id)
                if joints is None:
                    raise ToolError("robot.go_to_pose", "IK failed for approach pose")
                self.move_to_joints(joints, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id)

            joints = self._solve_ik(pose, tcp_offset, arm_id=arm_id)
            if joints is None:
                raise ToolError("robot.go_to_pose", "IK failed for target pose")
            self.move_to_joints(joints, tolerance=tolerance, max_steps=max_steps, arm_id=arm_id)

        # Post-move verification: log actual EE position vs target. Skip in
        # nonblocking mode — the arm hasn't reached the target yet.
        if not nonblocking:
            actual_ee = self.get_ee_pose(arm_id=arm_id)
            tp = pose["position"]
            ap = actual_ee["position"]
            err = (
                (tp["x"] - ap["x"]) ** 2
                + (tp["y"] - ap["y"]) ** 2
                + (tp["z"] - ap["z"]) ** 2
            ) ** 0.5
            logger.info(
                "go_to_pose arm=%d target=(%.4f,%.4f,%.4f) actual_ee=(%.4f,%.4f,%.4f) err=%.4f",
                arm_id, tp["x"], tp["y"], tp["z"], ap["x"], ap["y"], ap["z"], err,
            )

    def _servo_to_pose(
        self,
        pose: Se3Pose,
        *,
        arm_id: int = 0,
        pos_tol: float = OSC_POS_TOL_M,
        rot_tol: float = OSC_ROT_TOL_RAD,
        max_ticks: int = OSC_MAX_TICKS,
        stall_ticks: int = OSC_STALL_TICKS,
        tool_name: str = "robot.go_to_pose_cartesian",
    ) -> None:
        """Drive the EE to ``pose`` (world frame) by streaming clamped OSC
        deltas through the env's ``apply_policy_action`` — the policy's own
        actuator, NO IK/planning. Policy-like and fast, but collision-UNAWARE,
        so it is used only for known-safe straight-line segments; on stall or
        non-convergence it raises :class:`ToolError` so the caller can fall
        back to the collision-aware cuRobo linear plan.

        Frames: ``get_ee_pose`` and the target are both world-frame, and the
        LIBERO Franka base is axis-aligned with world, so a world-frame
        position delta is exactly the OSC ``control_delta`` the controller
        expects (output_max 0.05 m / 0.5 rad per step). Orientation uses
        robosuite's own ``orientation_error`` to match the OSC convention.
        """
        apply = getattr(self.env, "apply_policy_action", None)
        if apply is None:
            raise ToolError(
                "robot.go_to_pose_cartesian",
                "servo path requires env.apply_policy_action (OSC passthrough)",
            )
        from robosuite.utils.control_utils import orientation_error
        from robosuite.utils.transform_utils import quat2mat

        def _mat(q: Any) -> np.ndarray:
            if not q:
                return np.eye(3)
            return quat2mat(
                np.array([q["x"], q["y"], q["z"], q["w"]], dtype=np.float64)
            )

        tp = pose["position"]
        target_pos = np.array(
            [float(tp["x"]), float(tp["y"]), float(tp["z"])], dtype=np.float64
        )
        rot = pose.get("rotation")
        target_mat = _mat(rot) if rot else None  # None → hold current orientation
        gcmd = 1.0 - 2.0 * float(getattr(self.env, "_gripper_fraction", 1.0))
        out_p, out_r = 0.05, 0.5  # OSC output_max (m, rad)

        def motion_record(
            status: str,
            *,
            achieved: Se3Pose | None = None,
            position_error: float | None = None,
            rotation_error: float | None = None,
            ticks: int = 0,
        ) -> dict[str, Any]:
            record = {
                "tool": tool_name,
                "status": status,
                "target": pose,
                "achieved": achieved,
                "position_error_m": position_error,
                "rotation_error_rad": rotation_error,
                "ticks": int(ticks),
                "max_ticks": int(max_ticks),
                "stall_patience_ticks": int(stall_ticks),
                "frame": "connector_world",
            }
            self._last_motion_diagnostic = record
            return record

        motion_record("running")
        best = float("inf")
        stuck = 0
        pos_err = float("inf")
        rot_err = float("inf")
        cur: Se3Pose | None = None
        for tick in range(1, int(max_ticks) + 1):
            cur = self.get_ee_pose(arm_id=arm_id)
            cp = cur["position"]
            cur_pos = np.array(
                [float(cp["x"]), float(cp["y"]), float(cp["z"])], dtype=np.float64
            )
            dpos = target_pos - cur_pos
            dori = (
                orientation_error(target_mat, _mat(cur.get("rotation")))
                if target_mat is not None
                else np.zeros(3)
            )
            pos_err = float(np.linalg.norm(dpos))
            rot_err = float(np.linalg.norm(dori))
            if pos_err < pos_tol and rot_err < rot_tol:
                motion_record(
                    "reached",
                    achieved=cur,
                    position_error=pos_err,
                    rotation_error=rot_err,
                    ticks=tick - 1,
                )
                return
            action = np.empty(7, dtype=np.float64)
            action[:3] = np.clip(dpos / out_p, -1.0, 1.0)
            action[3:6] = np.clip(dori / out_r, -1.0, 1.0)
            action[6] = gcmd
            apply(action)
            emit_step = getattr(self, "_emit_step", None)
            if callable(emit_step):
                try:
                    obs = self.env.get_observation()
                except Exception:
                    obs = {}
                try:
                    reward = float(self.env.compute_reward() or 0.0)
                except Exception:
                    reward = 0.0
                emit_step(
                    action, obs, reward,
                    bool(getattr(self.env, "_current_done", False)),
                )
            if pos_err < best - 1e-4:
                best, stuck = pos_err, 0
            else:
                stuck += 1
                if stuck >= int(stall_ticks):
                    diagnostic_fn = getattr(self, "_joint_limit_diagnostic", None)
                    diagnostic = (
                        diagnostic_fn(arm_id)
                        if callable(diagnostic_fn)
                        else "joint state unavailable"
                    )
                    motion_record(
                        "stalled",
                        achieved=cur,
                        position_error=pos_err,
                        rotation_error=rot_err,
                        ticks=tick,
                    )
                    tp_text = ",".join(f"{value:.4f}" for value in target_pos)
                    cp_text = ",".join(f"{value:.4f}" for value in cur_pos)
                    raise ToolError(
                        tool_name,
                        f"OSC servo stalled after {tick}/{max_ticks} ticks "
                        f"(patience={stall_ticks}): target=({tp_text}), "
                        f"achieved=({cp_text}), pos_err={pos_err:.4f} m, "
                        f"rot_err={rot_err:.4f} rad; {diagnostic}",
                    )
        diagnostic_fn = getattr(self, "_joint_limit_diagnostic", None)
        diagnostic = (
            diagnostic_fn(arm_id)
            if callable(diagnostic_fn)
            else "joint state unavailable"
        )
        motion_record(
            "not_converged",
            achieved=cur,
            position_error=pos_err,
            rotation_error=rot_err,
            ticks=int(max_ticks),
        )
        tp_text = ",".join(f"{value:.4f}" for value in target_pos)
        cp_text = ",".join(
            f"{value:.4f}" for value in (
                cur_pos if cur is not None else np.full(3, np.nan)
            )
        )
        raise ToolError(
            tool_name,
            f"OSC servo did not converge in {max_ticks} ticks: "
            f"target=({tp_text}), achieved=({cp_text}), "
            f"pos_err={pos_err:.4f} m, rot_err={rot_err:.4f} rad; {diagnostic}",
        )

    def go_to_pose_cartesian(self, pose: Se3Pose, arm_id: int = 0) -> None:
        """Move along a straight Cartesian line to ``pose``.

        The target uses the same connector-public end-effector / TCP pose and
        connector-world frame returned by :meth:`get_ee_pose`. Backend-specific
        IK-link offsets are an implementation detail and must not be applied by
        graph code.
        """
        if not pose:
            raise ToolError("robot.go_to_pose_cartesian", "pose required")
        # Policy-like OSC servo fast-path (opt-in via GAP_LIBERO_SERVO): stream
        # clamped OSC deltas with NO IK/planning for this straight-line move.
        # Falls back to the collision-aware cuRobo linear plan on stall.
        if self._servo_enabled and hasattr(self.env, "apply_policy_action"):
            start_joints = self._current_joints(arm_id)  # pre-servo config
            try:
                with self._quiet_motion():
                    self._servo_to_pose(pose, arm_id=arm_id)
                return
            except ToolError as exc:
                logger.warning(
                    "[servo] cartesian servo failed (%s); restoring pre-servo "
                    "config and falling back to cuRobo linear plan", exc,
                )
                # A stalled servo may have left the arm partway — and possibly
                # in collision, which makes the cuRobo fallback fail with a
                # "start state in collision" (e.g. a cluttered grasp leg in
                # grocery_packing). Return to the pre-servo config so the
                # planner solves from the same state it would have without the
                # servo attempt.
                if start_joints is not None:
                    try:
                        with self._quiet_motion():
                            self.move_to_joints(
                                start_joints, tolerance=0.02, max_steps=200,
                                arm_id=arm_id,
                            )
                    except Exception:
                        logger.debug("[servo] restore-to-start failed", exc_info=True)
        if self.ik is None:
            raise ToolError("robot.go_to_pose_cartesian", "IK backend not configured")

        off = self._default_tcp_offset()
        is_yam = self._arm_dof == 6
        ee_pose = self.get_ee_pose(arm_id=arm_id)
        # Pass the current joint config as the linear seed — cuRobo's
        # MotionPlanner needs the actual start_config for FK; PyRoKi
        # accepts and ignores it.
        seed = self._current_joints(arm_id)
        trajectory = self.ik.plan_linear(
            ee_pose, pose, arm_id=arm_id, tcp_offset=off, seed_joints=seed,
        )
        if trajectory is None:
            raise ToolError("robot.go_to_pose_cartesian", "linear plan failed")
        tolerance = 0.03 if is_yam else 0.01
        max_steps = 200 if is_yam else 120
        with self._quiet_motion():
            self._execute_trajectory(
                trajectory, 1, tolerance, max_steps,
                arm_id=arm_id,
                reverse_joints=self.ik.trajectory_needs_joint_reverse,
            )

    def _tool_execute_trajectory(
        self,
        trajectory: Trajectory,
        subsample: int = 0,
        tolerance: float = 0.0,
        max_steps_per_waypoint: int = 0,
    ) -> None:
        """Execute a joint trajectory (source ExecuteJointTrajectory defaults)."""
        if not trajectory or not trajectory.get("waypoints"):
            raise ToolError("robot.execute_trajectory", "trajectory required")
        subsample = subsample if subsample > 0 else 1
        tolerance = tolerance if tolerance > 0 else 0.01
        max_steps = max_steps_per_waypoint if max_steps_per_waypoint > 0 else 120
        self._execute_trajectory(trajectory, subsample, tolerance, max_steps)

    def _tool_move_to_joints(
        self,
        joint_config: Any,
        tolerance: float = 0.0,
        max_steps: int = 0,
    ) -> None:
        """Move the arm to a joint configuration (source MoveToJoints defaults)."""
        positions = _as_positions(joint_config) if joint_config is not None else []
        if not positions:
            raise ToolError("robot.move_to_joints", "joint_config required")
        tolerance = tolerance if tolerance > 0 else 0.01
        max_steps = max_steps if max_steps > 0 else 120
        self.move_to_joints(positions, tolerance=tolerance, max_steps=max_steps)

    def go_home(self) -> None:
        """Move every arm to the home joint configuration."""
        # Safety: skip GoHome on real robot suites to avoid collisions
        if self.is_real:
            logger.warning("go_home skipped on real robot suite (safety)")
            return
        home = self._home_joints or _FRANKA_HOME_JOINTS
        with self._quiet_motion():
            for arm_id in range(self.num_arms):
                self.move_to_joints(home, tolerance=0.01, max_steps=200, arm_id=arm_id)

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Close the underlying environment (idempotent)."""
        if self._closed:
            return
        self._closed = True
        env = self.env
        if env is None:
            return
        try:
            if hasattr(env, "close"):
                env.close()
            elif hasattr(env, "handle"):
                env.handle.env.close()
        except Exception:
            logger.debug("env close failed", exc_info=True)

    def __enter__(self) -> Connector:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.close()


__all__ = ["Capabilities", "Connector"]
