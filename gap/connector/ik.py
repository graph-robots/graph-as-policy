"""In-process PyRoKI inverse kinematics for the connector layer.

This module merges two pieces of the source research codebase into one
plain-Python unit (no gRPC):

- the PyRoKI *solver* setup from the dev tree's pyroki service (and its
  ``pyroki_snippets``): basic IK, velocity-cost IK seeded with the previous
  configuration, and linear Cartesian planning by waypoint-interpolated IK;
- the *frame/TCP math* from ``services/sim_bridge/ik_backend.py``'s
  ``PyRoKIBackend``: the Franka path (apply the configured TCP offset and
  optional TCP rotation, solve for ``panda_hand``) plus the 6-DOF path's
  joint-order reversal and base-frame conversion quirks.

The numerics are kept identical to the source: same cost weights
(pos 50 / ori 10), same trust-region config, same refinement iteration
counts and jump threshold.

pyroki is CPU-JAX — the first solve per robot model JIT-compiles (a few
seconds); subsequent solves are milliseconds.
"""

from __future__ import annotations

import logging
from pathlib import Path

import numpy as np
from gap_core.types import Se3Pose, Trajectory, make_pose
from scipy.spatial.transform import Rotation, Slerp

logger = logging.getLogger(__name__)

# YAM TCP offset from link_6 to gripper tip (meters, in link_6 frame).
# Matches MuJoCo's `grasp_site` (pos="0 0 0.1347") on link_6 — the +Z axis
# points toward the gripper fingers, so the tip is at +0.1347 m along link_6 z.
_YAM_TCP_OFFSET = np.array([0.0, 0.0, 0.1347], dtype=np.float64)


# ---------------------------------------------------------------------------
# Robot model loading (cached)
# ---------------------------------------------------------------------------

_ROBOT_CACHE: dict[str, object] = {}


def load_robot(
    robot_urdf: str = "panda_description",
    robot_urdf_path: str | None = None,
):
    """Load (and cache) a ``pyroki.Robot`` model.

    Mirrors ``PyRoKIServicer._load_solver``: a local URDF path wins when it
    exists on disk; otherwise the named ``robot_descriptions`` entry is
    loaded (default ``panda_description``).
    """
    import pyroki as pk

    key = robot_urdf_path if robot_urdf_path and Path(robot_urdf_path).exists() else robot_urdf
    cached = _ROBOT_CACHE.get(str(key))
    if cached is not None:
        return cached

    if robot_urdf_path and Path(robot_urdf_path).exists():
        import yourdfpy

        urdf_path = Path(robot_urdf_path)
        mesh_dir = str(urdf_path.parent / "assets")
        logger.info("Loading robot URDF from path %r with PyRoKI...", robot_urdf_path)
        urdf = yourdfpy.URDF.load(
            str(urdf_path),
            mesh_dir=mesh_dir,
            build_collision_scene_graph=False,
            load_collision_meshes=False,
        )
    else:
        from robot_descriptions.loaders.yourdfpy import load_robot_description

        logger.info("Loading robot URDF %r with PyRoKI...", robot_urdf)
        urdf = load_robot_description(robot_urdf)
    robot = pk.Robot.from_urdf(urdf)
    logger.info(
        "PyRoKI loaded: robot=%r, actuated_joints=%d",
        robot_urdf_path or robot_urdf,
        robot.joints.num_actuated_joints,
    )
    _ROBOT_CACHE[str(key)] = robot
    return robot


# ---------------------------------------------------------------------------
# JAX solves (verbatim from the dev tree's pyroki service snippets)
# ---------------------------------------------------------------------------


# The JAX solves live in gap.connector._ik_jax (verbatim from the source
# pyroki_snippets — @jdc.jit needs module-level jax imports for annotation
# resolution). Imported lazily so `import gap.connector` stays light.


def solve_ik_basic(
    robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
) -> np.ndarray:
    """Solve the basic IK problem (no seed).

    Returns the joint configuration, shape ``(num_actuated_joints,)``.
    """
    from gap.connector import _ik_jax

    return _ik_jax.solve_ik(
        robot=robot,
        target_link_name=target_link_name,
        target_wxyz=np.asarray(target_wxyz, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
    )


def solve_ik_vel_cost(
    robot,
    target_link_name: str,
    target_wxyz: np.ndarray,
    target_position: np.ndarray,
    prev_cfg: np.ndarray,
    initial_cfg: np.ndarray | None = None,
) -> np.ndarray:
    """Velocity-cost IK: prefer solutions near ``prev_cfg``.

    Keeps the solver on the IK branch closest to the arm's current
    configuration (no elbow flips between solves).
    """
    from gap.connector import _ik_jax

    return _ik_jax.solve_ik_vel_cost(
        robot=robot,
        target_link_name=target_link_name,
        target_wxyz=np.asarray(target_wxyz, dtype=np.float64),
        target_position=np.asarray(target_position, dtype=np.float64),
        prev_cfg=np.asarray(prev_cfg, dtype=np.float64),
        initial_cfg=initial_cfg,
    )


# ---------------------------------------------------------------------------
# Linear Cartesian planning (verbatim from the dev tree's pyroki service)
# ---------------------------------------------------------------------------


def slerp_quaternions(
    q_start: np.ndarray,  # wxyz format
    q_end: np.ndarray,  # wxyz format
    num_steps: int,
) -> np.ndarray:
    """SLERP interpolation between two wxyz quaternions.

    Returns an array of shape ``(num_steps, 4)`` in wxyz format.
    """
    # scipy uses xyzw format, so convert
    r_start = Rotation.from_quat([q_start[1], q_start[2], q_start[3], q_start[0]])
    r_end = Rotation.from_quat([q_end[1], q_end[2], q_end[3], q_end[0]])

    key_rots = Rotation.concatenate([r_start, r_end])
    key_times = [0, 1]
    slerp = Slerp(key_times, key_rots)

    times = np.linspace(0, 1, num_steps)
    interp_rots = slerp(times)

    quats_xyzw = interp_rots.as_quat()
    quats_wxyz = np.column_stack(
        [quats_xyzw[:, 3], quats_xyzw[:, 0], quats_xyzw[:, 1], quats_xyzw[:, 2]]
    )
    return quats_wxyz


def plan_trajectory_linear_ik(
    robot,
    target_link_name: str,
    start_pos: np.ndarray,
    start_wxyz: np.ndarray,
    end_pos: np.ndarray,
    end_wxyz: np.ndarray,
    num_waypoints: int = 20,
    use_prev_cfg: bool = True,
    jump_threshold: float = 0.5,
    ik_refinement_iters: int = 15,
) -> np.ndarray:
    """Plan a trajectory by linear interpolation + IK at each waypoint.

    Returns an array of shape ``(num_waypoints, num_joints)``.
    """
    # Linear interpolation for positions
    positions = np.linspace(start_pos, end_pos, num_waypoints)

    # SLERP for orientations
    orientations = slerp_quaternions(start_wxyz, end_wxyz, num_waypoints)

    # Solve IK for each waypoint
    trajectory = []
    prev_cfg = None
    jump_warnings = []

    for i, (pos, wxyz) in enumerate(zip(positions, orientations, strict=False)):
        if use_prev_cfg and prev_cfg is not None:
            # Use velocity-cost IK to stay close to previous solution
            for _ in range(ik_refinement_iters):
                cfg = solve_ik_vel_cost(
                    robot=robot,
                    target_link_name=target_link_name,
                    target_wxyz=wxyz,
                    target_position=pos,
                    prev_cfg=prev_cfg,
                )
                if np.allclose(cfg, prev_cfg, atol=1e-3):
                    break
                else:
                    prev_cfg = cfg
        else:
            cfg = solve_ik_basic(
                robot=robot,
                target_link_name=target_link_name,
                target_wxyz=wxyz,
                target_position=pos,
            )
        cfg = np.array(cfg)

        # Check for large joint jumps
        if prev_cfg is not None:
            joint_diff = np.abs(cfg - prev_cfg)
            large_jumps = np.where(joint_diff > jump_threshold)[0]
            if len(large_jumps) > 0:
                for joint_idx in large_jumps:
                    jump_warnings.append(
                        f"  Waypoint {i}: joint {joint_idx} jumped "
                        f"{np.degrees(joint_diff[joint_idx]):.1f} deg "
                        f"({joint_diff[joint_idx]:.3f} rad)"
                    )

        trajectory.append(cfg)
        prev_cfg = cfg

    if jump_warnings:
        logger.warning(
            "%d large joint jump(s) detected (threshold: %.1f deg):\n%s",
            len(jump_warnings),
            np.degrees(jump_threshold),
            "\n".join(jump_warnings),
        )

    return np.array(trajectory)


# ---------------------------------------------------------------------------
# Pose helpers (gap.types Se3Pose <-> numpy)
# ---------------------------------------------------------------------------


def _pose_to_numpy(pose: Se3Pose) -> tuple[np.ndarray, np.ndarray]:
    """Extract (position, wxyz quaternion) arrays from an Se3Pose."""
    p = pose["position"]
    r = pose["rotation"]
    position = np.array([p["x"], p["y"], p["z"]], dtype=np.float64)
    wxyz = np.array([r["w"], r["x"], r["y"], r["z"]], dtype=np.float64)
    return position, wxyz


def _trajectory_from_array(arr: np.ndarray) -> Trajectory:
    return {
        "waypoints": [
            {"positions": np.asarray(row, dtype=np.float64)} for row in np.asarray(arr)
        ]
    }


# ---------------------------------------------------------------------------
# Backend — PyRoKIBackend's frame/TCP math, in-process
# ---------------------------------------------------------------------------


class PyRokiBackend:
    """In-process IK backend with the source ``PyRoKIBackend`` semantics.

    For **Panda** (7-DOF): pre-applies the configured TCP offset (optionally
    respecting a TCP-frame rotation, e.g. Robotiq mounted at pi/4) and solves
    for the ``panda_hand`` link. The per-call ``tcp_offset`` argument is
    accepted for interface parity but — exactly as in the source backend —
    the *configured* offset governs the Panda path (``None`` for LIBERO:
    targets are panda_hand-frame poses already).

    For **6-DOF (YAM)**: converts world-frame TCP target to base-frame
    ``link_6`` target, reverses seed joint order before the solve, and
    reverses returned joint order (PyRoKI parses the YAM URDF in
    joint6→joint1 order).
    """

    def __init__(
        self,
        *,
        arm_dof: int = 7,
        tcp_offset: np.ndarray | None = None,
        tcp_rotation: Rotation | None = None,
        home_joints: list[float] | None = None,
        arm_bases: list[tuple[float, float, float]] | None = None,
        robot_urdf: str = "panda_description",
        robot_urdf_path: str | None = None,
        target_link: str | None = None,
    ) -> None:
        self._arm_dof = int(arm_dof)
        self._tcp_offset = (
            np.asarray(tcp_offset, dtype=np.float64) if tcp_offset is not None else None
        )
        self._tcp_rotation = tcp_rotation
        self._home_joints = list(home_joints) if home_joints is not None else None
        self._arm_bases = list(arm_bases) if arm_bases is not None else None
        self._robot_urdf = robot_urdf
        self._robot_urdf_path = robot_urdf_path
        self._target_link = target_link or ("panda_hand" if self._arm_dof == 7 else "link_6")
        self._robot = None  # lazy

    # -- model -------------------------------------------------------------

    @property
    def robot(self):
        if self._robot is None:
            self._robot = load_robot(self._robot_urdf, self._robot_urdf_path)
        return self._robot

    @property
    def trajectory_needs_joint_reverse(self) -> bool:
        return self._arm_dof == 6

    def _is_yam(self) -> bool:
        return self._arm_dof == 6

    # -- frames ------------------------------------------------------------

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

    # -- IK ------------------------------------------------------------------

    def solve_ik(
        self,
        target_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        seed_joints: list[float] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> list[float] | None:
        """Solve IK. Returns joints in simulator-native order, or None on failure."""
        if self._is_yam():
            return self._solve_yam(target_world_pose, arm_id, seed_joints)
        return self._solve_panda(target_world_pose, tcp_offset, seed_joints)

    # --- Panda path --------------------------------------------------------

    def _panda_tcp_to_link(self, pose: Se3Pose) -> Se3Pose:
        """Convert a TCP-frame target into a ``panda_hand``-link target.

        gap stores ``tcp_offset`` in the *negated* link-to-TCP convention
        (e.g. ``(0, 0, -0.097)`` for the Franka grip_site). Adding
        ``R_link @ tcp_offset`` to the TCP target yields the link position
        whose tool tip, after FK, lands at the requested TCP. Identity
        operation when no offset / rotation is configured.
        """
        if self._tcp_offset is None and self._tcp_rotation is None:
            return pose
        q = pose["rotation"]
        R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        if self._tcp_rotation is not None:
            R_link = R @ self._tcp_rotation.inv().as_matrix()
            link_quat_xyzw = Rotation.from_matrix(R_link).as_quat()
            link_rot = {
                "x": float(link_quat_xyzw[0]),
                "y": float(link_quat_xyzw[1]),
                "z": float(link_quat_xyzw[2]),
                "w": float(link_quat_xyzw[3]),
            }
        else:
            R_link = R
            link_rot = dict(pose["rotation"])
        if self._tcp_offset is None:
            return {"position": dict(pose["position"]), "rotation": link_rot}
        tcp_world = R_link @ self._tcp_offset
        p = pose["position"]
        return {
            "position": {
                "x": p["x"] + float(tcp_world[0]),
                "y": p["y"] + float(tcp_world[1]),
                "z": p["z"] + float(tcp_world[2]),
            },
            "rotation": link_rot,
        }

    def _solve_panda(
        self,
        pose: Se3Pose,
        tcp_offset: np.ndarray | None,
        seed_joints: list[float] | None = None,
    ) -> list[float] | None:
        """Panda: rotate panda_hand orientation if a TCP rotation is configured
        (Robotiq at pi/4), shift target from tool tip to panda_hand link, and
        solve.

        ``seed_joints`` becomes the velocity-cost solve's ``prev_cfg`` so the
        solver stays on the IK branch closest to the arm's current
        configuration; without it the basic solve runs from the internal
        default seed (which can jump branches between phases).
        """
        del tcp_offset  # configured offset governs; per-call kwarg ignored
        solve_pose = self._panda_tcp_to_link(pose)

        target_position, target_wxyz = _pose_to_numpy(solve_pose)
        n_actuated = self.robot.joints.num_actuated_joints
        try:
            if seed_joints is not None:
                # The ``panda_description`` URDF exposes 8 actuated joints
                # (panda_joint1..7 + panda_finger_joint1). The simulator only
                # tracks the 7 arm joints, so pad with a neutral finger qpos
                # to match the velocity-cost solver's expected prev_cfg shape.
                seed = list(seed_joints)[: self._arm_dof]
                seed = seed + [0.0] * (n_actuated - len(seed))
                cfg = solve_ik_vel_cost(
                    robot=self.robot,
                    target_link_name=self._target_link,
                    target_wxyz=target_wxyz,
                    target_position=target_position,
                    prev_cfg=np.asarray(seed, dtype=np.float64),
                )
            else:
                cfg = solve_ik_basic(
                    robot=self.robot,
                    target_link_name=self._target_link,
                    target_wxyz=target_wxyz,
                    target_position=target_position,
                )
        except Exception:
            logger.exception("IK solve failed")
            return None
        joints = [float(v) for v in np.asarray(cfg)]
        # Drop the finger joint pyroki returns; the arm controller only
        # consumes the arm joints.
        return joints[: self._arm_dof]

    # --- YAM path ----------------------------------------------------------

    def _solve_yam(
        self,
        pose: Se3Pose,
        arm_id: int,
        seed_joints: list[float] | None,
    ) -> list[float] | None:
        """YAM: world→base frame, TCP→link_6 subtraction, reversed seed."""
        base_frame_pose = self.world_pose_to_base_frame(pose, arm_id)
        q = base_frame_pose["rotation"]
        R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        tcp_in_base = R @ _YAM_TCP_OFFSET
        bp = base_frame_pose["position"]
        link6_pose: Se3Pose = {
            "position": {
                "x": bp["x"] - float(tcp_in_base[0]),
                "y": bp["y"] - float(tcp_in_base[1]),
                "z": bp["z"] - float(tcp_in_base[2]),
            },
            "rotation": dict(base_frame_pose["rotation"]),
        }
        # Seed IK in reversed order (PyRoKI parses YAM URDF as joint6→joint1)
        if seed_joints is not None:
            s = list(seed_joints)[: self._arm_dof]
            s.reverse()
            seed = s
        else:
            home = self._home_joints or [0.0, 1.047, 1.047, 0.0, 0.0, 0.0]
            seed = list(reversed(home))
        target_position, target_wxyz = _pose_to_numpy(link6_pose)
        try:
            cfg = solve_ik_vel_cost(
                robot=self.robot,
                target_link_name=self._target_link,
                target_wxyz=target_wxyz,
                target_position=target_position,
                prev_cfg=np.asarray(seed, dtype=np.float64),
            )
        except Exception:
            logger.exception("IK solve failed")
            return None
        joints = [float(v) for v in np.asarray(cfg)]
        joints.reverse()  # to joint1→joint6 (simulator order)
        return joints

    # --- Linear plan -------------------------------------------------------

    def plan_linear(
        self,
        start_world_pose: Se3Pose,
        end_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        tcp_offset: np.ndarray | None = None,
        seed_joints: list[float] | None = None,
        num_waypoints: int = 40,
        ik_refinement_iters: int = 40,
        jump_threshold: float = 0.5,
    ) -> Trajectory | None:
        """Plan a Cartesian straight-line trajectory in TCP frame.

        Both ``start_world_pose`` and ``end_world_pose`` are TCP-frame poses:
        the backend converts them to the IK link frame using ``self._tcp_offset``
        before per-waypoint IK, matching ``solve_ik``'s contract. ``seed_joints``
        is accepted for interface parity with CuRoboBackend (PyRoKi's per-
        waypoint solver chains its own previous-config seed).

        Waypoints are in backend-native joint order; see
        ``trajectory_needs_joint_reverse``. Defaults match the source's
        ``PyRoKIPlanRequest(num_waypoints=40, ik_refinement_iters=40)``.
        """
        del seed_joints  # unused; PyRoKi seeds per-waypoint internally
        del tcp_offset  # configured offset governs; per-call kwarg ignored
        if self._is_yam():
            start_pose = self.world_pose_to_base_frame(start_world_pose, arm_id)
            end_pose = self.world_pose_to_base_frame(end_world_pose, arm_id)
            adjusted = []
            for bp in (start_pose, end_pose):
                q = bp["rotation"]
                R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
                off = R @ _YAM_TCP_OFFSET
                p = bp["position"]
                adjusted.append(
                    make_pose(
                        (p["x"] - off[0], p["y"] - off[1], p["z"] - off[2]),
                        (q["w"], q["x"], q["y"], q["z"]),
                    )
                )
            start_pose, end_pose = adjusted
        else:
            # Panda: convert both endpoints from TCP frame to panda_hand link
            # frame using the configured TCP offset / rotation. Matches
            # _solve_panda's per-call shift so solve_ik and plan_linear share
            # one user-facing convention (the caller passes TCP poses).
            start_pose = self._panda_tcp_to_link(start_world_pose)
            end_pose = self._panda_tcp_to_link(end_world_pose)

        start_position, start_wxyz = _pose_to_numpy(start_pose)
        end_position, end_wxyz = _pose_to_numpy(end_pose)
        try:
            sol_traj = plan_trajectory_linear_ik(
                robot=self.robot,
                target_link_name=self._target_link,
                start_pos=start_position,
                start_wxyz=start_wxyz,
                end_pos=end_position,
                end_wxyz=end_wxyz,
                num_waypoints=num_waypoints,
                jump_threshold=jump_threshold,
                ik_refinement_iters=ik_refinement_iters,
            )
        except Exception:
            logger.exception("Linear planning failed")
            return None
        return _trajectory_from_array(np.asarray(sol_traj))

    def supports_world_aware_plan(self) -> bool:
        return False

    def plan_to_pose(self, *args, **kwargs):
        raise NotImplementedError(
            "PyRokiBackend does not support collision-aware single-pose planning; "
            "use the open-robot-skills curobo bundle for world-aware planning."
        )


# ---------------------------------------------------------------------------
# Backend — cuRobo v0.8 (GPU): linear motion, plus its own InverseKinematics
# ---------------------------------------------------------------------------


_CUROBO_INSTALL_HINT = (
    "cuRobo bundle is not importable (gap_skills.tools.curobo._curobo_impl). "
    "Install with:  pip install -e 'open-robot-skills[curobo]' "
    "--no-build-isolation  with CUDA_HOME set, or construct the connector "
    "with ik=PyRokiBackend(...) to opt back into the in-process PyRoKi path."
)


class CuRoboBackend:
    """cuRobo-backed IK + linear-motion backend (default in the connector).

    Mirrors :class:`PyRokiBackend`'s public surface so the connector can
    swap them through the ``ik=`` kwarg without touching call sites.

    ``plan_linear``
        Tries the v0.8 ``MotionPlanner``-based cartesian linear plan
        (``_curobo_impl.plan_linear`` → ``plan_directed_linear``). On failure
        — the linear-constraint plan has no feasible straight-line solution —
        falls back to ``_curobo_impl.plan_to_pose``, the v0.8 single-pose
        collision-aware planner. Both reuse the same cached MotionPlanner
        machinery and already report ``(success, trajectory, ...)``.

    ``solve_ik``
        Solves through cuRobo's own ``InverseKinematics`` (thirty-two seeds
        on the GPU, the one nearest the seed kept), built once per backend
        from ``robot_file``; ``plan_to_pose`` is the collision-aware planner,
        not an IK.

    cuRobo is CUDA-only. The implementation module is imported lazily, so
    constructing this backend on a CPU-only box succeeds — only an actual
    ``solve_ik`` / ``plan_linear`` call hits the import. The error surfaced
    on import failure points the user at the install line and the opt-in
    PyRoKi fallback.

    Robot coverage: 7-DOF Franka via the bundled ``franka.yml``. Pass
    ``robot_file=`` for other v0.8 robot configs. 6-DOF (YAM) is not yet
    bundled in cuRobo and raises ``NotImplementedError`` in ``__init__`` —
    construct the connector with ``ik=PyRokiBackend(...)`` for YAM.
    """

    def __init__(
        self,
        *,
        arm_dof: int = 7,
        tcp_offset: np.ndarray | None = None,
        tcp_rotation: Rotation | None = None,
        home_joints: list[float] | None = None,
        arm_bases: list[tuple[float, float, float]] | None = None,
        robot_file: str = "franka.yml",
        robot_urdf: str | None = None,
        robot_urdf_path: str | None = None,
        target_link: str | None = None,
    ) -> None:
        if int(arm_dof) != 7:
            raise NotImplementedError(
                f"CuRoboBackend currently supports 7-DOF Franka only "
                f"(arm_dof=7); got arm_dof={arm_dof}. For YAM/6-DOF "
                f"construct the connector with ik=PyRokiBackend(...) "
                f"explicitly."
            )
        self._arm_dof = int(arm_dof)
        self._tcp_offset = (
            np.asarray(tcp_offset, dtype=np.float64) if tcp_offset is not None else None
        )
        self._tcp_rotation = tcp_rotation
        self._home_joints = list(home_joints) if home_joints is not None else None
        self._arm_bases = list(arm_bases) if arm_bases is not None else None
        self._robot_file = robot_file
        # Opt-in cuRobo CUDA-graph capture for the v0.8 pose planner. After a
        # one-time capture warmup, repeated same-config plans run far faster.
        # Valid for the free-space (world=off) IK path used by go_to_pose
        # (fixed shapes). ``GAP_CUROBO_CUDA_GRAPH=1`` enables it.
        from gap import env_config
        self._use_cuda_graph = env_config.curobo_cuda_graph()
        # robot_urdf / robot_urdf_path / target_link accepted for interface
        # parity with PyRokiBackend but unused: cuRobo resolves the model from
        # robot_file. Stash for diagnostics.
        self._robot_urdf = robot_urdf
        self._robot_urdf_path = robot_urdf_path
        self._target_link = target_link

    # -- model -------------------------------------------------------------

    @staticmethod
    def _import_impl():
        """Lazy import of the curobo bundle's impl module.

        Raises ImportError with an actionable message if the bundle isn't
        installed or CUDA is missing.
        """
        try:
            from gap_skills.tools.curobo import _curobo_impl  # noqa: PLC0415
            return _curobo_impl
        except ImportError as e:
            raise ImportError(f"{_CUROBO_INSTALL_HINT} ({e})") from e

    @property
    def trajectory_needs_joint_reverse(self) -> bool:
        return False  # Franka kinematic order matches simulator order

    def supports_world_aware_plan(self) -> bool:
        return True

    # -- frames ------------------------------------------------------------

    def world_pose_to_base_frame(self, pose: Se3Pose, arm_id: int) -> Se3Pose:
        """Same arm-base translation PyRokiBackend uses."""
        if self._arm_bases is None or arm_id >= len(self._arm_bases):
            return pose
        bx, by, bz = self._arm_bases[arm_id]
        p = pose["position"]
        return {
            "position": {"x": p["x"] - bx, "y": p["y"] - by, "z": p["z"] - bz},
            "rotation": dict(pose["rotation"]),
        }

    def _tcp_to_link(self, pose: Se3Pose) -> Se3Pose:
        """Convert a TCP-frame target into a ``panda_hand``-link target.

        Mirrors :meth:`PyRokiBackend._panda_tcp_to_link`: gap stores
        ``tcp_offset`` in the *negated* link-to-TCP convention (e.g.
        ``(0, 0, -0.097)`` for the Franka grip_site). Adding
        ``R_link @ tcp_offset`` to the TCP target yields the link position
        whose tool tip, after FK, lands at the requested TCP. When a TCP
        rotation is configured (Robotiq at pi/4), apply
        ``R_link = R_world @ R_tcp.inv()`` first. Identity operation when
        no offset / rotation is configured.

        After this conversion we pass ``tcp_offset=None`` to
        ``_curobo_impl.plan_to_pose`` / ``plan_linear`` so the impl performs
        no further offset math — keeping a single sign convention.
        """
        if self._tcp_offset is None and self._tcp_rotation is None:
            return pose
        q = pose["rotation"]
        R = Rotation.from_quat([q["x"], q["y"], q["z"], q["w"]]).as_matrix()
        if self._tcp_rotation is not None:
            R_link = R @ self._tcp_rotation.inv().as_matrix()
            link_quat_xyzw = Rotation.from_matrix(R_link).as_quat()
            link_rot = {
                "x": float(link_quat_xyzw[0]),
                "y": float(link_quat_xyzw[1]),
                "z": float(link_quat_xyzw[2]),
                "w": float(link_quat_xyzw[3]),
            }
        else:
            R_link = R
            link_rot = dict(pose["rotation"])
        if self._tcp_offset is None:
            return {"position": dict(pose["position"]), "rotation": link_rot}
        tcp_world = R_link @ self._tcp_offset
        p = pose["position"]
        return {
            "position": {
                "x": p["x"] + float(tcp_world[0]),
                "y": p["y"] + float(tcp_world[1]),
                "z": p["z"] + float(tcp_world[2]),
            },
            "rotation": link_rot,
        }

    def _pose_for_curobo(
        self, pose: Se3Pose, arm_id: int
    ) -> tuple[np.ndarray, np.ndarray]:
        """TCP-frame world Se3Pose -> (position xyz, quaternion wxyz) in robot
        base frame at the IK link.

        Pipeline: world → base (``world_pose_to_base_frame``) → TCP→link
        (``_tcp_to_link``). The impl receives a pure link-frame target and
        is called with ``tcp_offset=None`` — sign convention lives entirely
        in :meth:`_tcp_to_link`.
        """
        base = self.world_pose_to_base_frame(pose, arm_id)
        link = self._tcp_to_link(base)
        return _pose_to_numpy(link)

    def _resolve_seed(
        self, seed_joints: list[float] | None
    ) -> np.ndarray:
        """Coerce the seed/home/zero fallback to a (arm_dof,) float64 array."""
        if seed_joints is not None:
            arr = np.asarray(list(seed_joints)[: self._arm_dof], dtype=np.float64)
        elif self._home_joints is not None:
            arr = np.asarray(self._home_joints[: self._arm_dof], dtype=np.float64)
        else:
            arr = np.zeros(self._arm_dof, dtype=np.float64)
        if arr.shape[0] < self._arm_dof:
            arr = np.concatenate(
                [arr, np.zeros(self._arm_dof - arr.shape[0], dtype=np.float64)]
            )
        return arr

    # -- IK ----------------------------------------------------------------

    def _ik(self):
        """cuRobo's IK solver for ``robot_file``, built once and kept.

        Thirty-two parallel seeds, self-collision aware, no world: the same
        problem ``plan_to_pose`` was standing in for at thirty-seven
        milliseconds a call, at a few. Built lazily so a backend constructed
        for its frames alone never touches the GPU.
        """
        ik = getattr(self, "_ik_solver", None)
        if ik is None:
            try:
                from curobo.inverse_kinematics import (  # noqa: PLC0415
                    InverseKinematics,
                    InverseKinematicsCfg,
                )
            except ImportError as e:
                raise ImportError(f"{_CUROBO_INSTALL_HINT} ({e})") from e
            config = InverseKinematicsCfg.create(
                robot=self._robot_file,
                num_seeds=32,
                use_cuda_graph=bool(self._use_cuda_graph),
            )
            ik = InverseKinematics(config)
            self._ik_solver = ik
        return ik

    def solve_ik(
        self,
        target_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        seed_joints: list[float] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> list[float] | None:
        """Solve IK for a TCP-frame world pose with cuRobo's ``InverseKinematics``.

        ``_pose_for_curobo`` does the world→base + TCP→link conversion using
        ``self._tcp_offset`` / ``self._tcp_rotation``, so the solver sees a
        pure link-frame target — all offset math lives in the backend, with a
        single sign convention. The per-call ``tcp_offset`` kwarg is accepted
        for interface parity with PyRokiBackend but is ignored (configured
        offset governs).

        The seed (``seed_joints``, else home, else zero) is the solver's
        ``current_state``, so of the seeds that converge the one nearest the
        arm's posture is returned. Returns simulator-order joints or None on
        failure (matches PyRokiBackend's contract).
        """
        del tcp_offset  # configured offset governs; per-call kwarg ignored
        target_pos, target_quat_wxyz = self._pose_for_curobo(target_world_pose, arm_id)
        seed_arr = self._resolve_seed(seed_joints)
        try:
            import torch  # noqa: PLC0415
            from curobo.types import GoalToolPose, JointState, Pose  # noqa: PLC0415

            ik = self._ik()
            device = getattr(ik.device_cfg, "device", "cuda")
            as_row = lambda a: torch.as_tensor(  # noqa: E731
                np.asarray(a, dtype=np.float32).reshape(1, -1), device=device
            )
            goal = Pose(position=as_row(target_pos), quaternion=as_row(target_quat_wxyz))
            seed = JointState.from_position(as_row(seed_arr[: self._arm_dof]), joint_names=list(ik.joint_names))
            result = ik.solve_pose(
                GoalToolPose.from_poses({ik.tool_frames[0]: goal}, num_goalset=1),
                current_state=seed,
            )
        except Exception:
            logger.exception("CuRoboBackend.solve_ik: InverseKinematics raised")
            return None
        if not bool(result.success.reshape(-1)[0].item()):
            logger.warning("CuRoboBackend.solve_ik: no IK solution")
            return None
        joints = np.asarray(result.js_solution.position.detach().cpu().numpy(), dtype=np.float64).reshape(-1)
        return [float(v) for v in joints[: self._arm_dof]]

    # --- Linear plan + IK fallback ----------------------------------------

    def plan_linear(
        self,
        start_world_pose: Se3Pose,
        end_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        tcp_offset: np.ndarray | None = None,
        seed_joints: list[float] | None = None,
        num_waypoints: int = 40,
        ik_refinement_iters: int = 40,
        jump_threshold: float = 0.5,
    ) -> Trajectory | None:
        """Plan a TCP-frame straight-line cartesian motion, falling back to
        IK on failure.

        ``start_world_pose`` and ``end_world_pose`` are TCP-frame poses
        (matching ``solve_ik``'s contract). ``_pose_for_curobo`` applies the
        world→base + TCP→link conversion before handing off to the impl,
        whose internals plan in the IK link frame. The per-call
        ``tcp_offset`` kwarg is ignored (configured offset governs).

        Step 1 — linear: call ``_curobo_impl.plan_linear`` (delegates to v0.8
        ``plan_directed_linear`` with all three axes free, orientation locked
        at the target). Returns ``(success, traj, failure_reason)``.

        Step 2 — IK fallback: if linear couldn't find a path, call
        ``_curobo_impl.plan_to_pose`` for the endpoint. Same MotionPlanner
        machinery without the per-axis hold constraint, so it can route
        around obstacles the straight line couldn't.

        Returns None only when both stages fail. ``num_waypoints`` /
        ``ik_refinement_iters`` / ``jump_threshold`` are accepted for
        interface parity with PyRokiBackend but ignored — cuRobo's
        MotionPlanner controls its own interpolation density and seeding.
        """
        del num_waypoints, ik_refinement_iters, jump_threshold  # interface parity
        del tcp_offset  # configured offset governs; per-call kwarg ignored
        impl = self._import_impl()
        start_pos, start_wxyz = self._pose_for_curobo(start_world_pose, arm_id)
        end_pos, end_wxyz = self._pose_for_curobo(end_world_pose, arm_id)
        seed_arr = self._resolve_seed(seed_joints)

        # ── Step 1: linear cartesian plan ───────────────────────────────
        success = False
        traj_arr: np.ndarray | None = None
        failure_reason = "not_attempted"
        try:
            success, traj_arr, failure_reason = impl.plan_linear(
                start_pose=(start_pos, start_wxyz),
                end_pose=(end_pos, end_wxyz),
                start_joint_position=seed_arr,
                robot_file=self._robot_file,
            )
        except Exception as e:
            logger.exception("CuRoboBackend.plan_linear: plan_linear raised")
            failure_reason = f"exception:{e}"

        if success and traj_arr is not None and len(traj_arr) > 0:
            arr = np.asarray(traj_arr)
            return _trajectory_from_array(arr[:, : self._arm_dof])

        logger.info(
            "CuRoboBackend.plan_linear failed (reason=%s); "
            "falling back to plan_to_pose for the endpoint.",
            failure_reason,
        )

        # ── Step 2: IK fallback via single-pose plan_to_pose ────────────
        try:
            success_ik, traj_ik = impl.plan_to_pose(
                target_position=end_pos,
                target_quat_wxyz=end_wxyz,
                start_joint_position=seed_arr,
                robot_file=self._robot_file,
                tcp_offset=None,  # already applied in _pose_for_curobo
                use_cuda_graph=self._use_cuda_graph,
            )
        except Exception:
            logger.exception(
                "CuRoboBackend.plan_linear: plan_to_pose fallback raised"
            )
            return None
        if not success_ik or traj_ik is None or len(traj_ik) == 0:
            logger.warning(
                "CuRoboBackend.plan_linear: linear failed AND plan_to_pose "
                "fallback failed; returning None."
            )
            return None
        return _trajectory_from_array(np.asarray(traj_ik)[:, : self._arm_dof])

    def plan_to_pose(
        self,
        target_world_pose: Se3Pose,
        *,
        arm_id: int = 0,
        seed_joints: list[float] | None = None,
        tcp_offset: np.ndarray | None = None,
    ) -> Trajectory | None:
        """Expose cuRobo's v0.8 single-pose collision-aware planner directly.

        ``target_world_pose`` is a TCP-frame pose. Per-call ``tcp_offset`` is
        ignored (configured offset governs); offset math lives in
        ``_pose_for_curobo``.
        """
        del tcp_offset  # configured offset governs; per-call kwarg ignored
        impl = self._import_impl()
        target_pos, target_quat_wxyz = self._pose_for_curobo(target_world_pose, arm_id)
        seed_arr = self._resolve_seed(seed_joints)
        try:
            success, traj = impl.plan_to_pose(
                target_position=target_pos,
                target_quat_wxyz=target_quat_wxyz,
                start_joint_position=seed_arr,
                robot_file=self._robot_file,
                tcp_offset=None,  # already applied in _pose_for_curobo
                use_cuda_graph=self._use_cuda_graph,
            )
        except Exception:
            logger.exception("CuRoboBackend.plan_to_pose raised")
            return None
        if not success or traj is None or len(traj) == 0:
            return None
        return _trajectory_from_array(np.asarray(traj)[:, : self._arm_dof])


# ---------------------------------------------------------------------------
# Standalone conveniences (default Franka backend)
# ---------------------------------------------------------------------------

_DEFAULT_BACKEND: PyRokiBackend | None = None


def _default_backend() -> PyRokiBackend:
    global _DEFAULT_BACKEND
    if _DEFAULT_BACKEND is None:
        _DEFAULT_BACKEND = PyRokiBackend()
    return _DEFAULT_BACKEND


def solve_ik(
    target_pose: Se3Pose,
    seed_joints: list[float] | None = None,
    tcp_offset: np.ndarray | None = None,
    *,
    robot_urdf_path: str | None = None,
    arm_id: int = 0,
) -> list[float] | None:
    """Solve IK against the default Franka model (or a custom URDF path).

    ``target_pose`` is a world-frame :class:`gap.types.Se3Pose` for the
    ``panda_hand`` link (no TCP offset is configured on the default
    backend — matching the source LIBERO setup).
    """
    if robot_urdf_path:
        backend = PyRokiBackend(robot_urdf_path=robot_urdf_path)
    else:
        backend = _default_backend()
    return backend.solve_ik(
        target_pose, arm_id=arm_id, seed_joints=seed_joints, tcp_offset=tcp_offset
    )


def plan_linear(
    start_pose: Se3Pose,
    end_pose: Se3Pose,
    *,
    num_waypoints: int = 40,
    ik_refinement_iters: int = 40,
    robot_urdf_path: str | None = None,
    arm_id: int = 0,
) -> Trajectory | None:
    """Linear Cartesian plan against the default Franka model."""
    if robot_urdf_path:
        backend = PyRokiBackend(robot_urdf_path=robot_urdf_path)
    else:
        backend = _default_backend()
    return backend.plan_linear(
        start_pose,
        end_pose,
        arm_id=arm_id,
        num_waypoints=num_waypoints,
        ik_refinement_iters=ik_refinement_iters,
    )


__all__ = [
    "CuRoboBackend",
    "PyRokiBackend",
    "load_robot",
    "plan_linear",
    "plan_trajectory_linear_ik",
    "slerp_quaternions",
    "solve_ik",
    "solve_ik_basic",
    "solve_ik_vel_cost",
]
