"""RealConnector — real-hardware backends + the ``real()`` factory.

Two robots:

- ``franka``: :class:`gap.envs.franka_real_env.FrankaRealEnv`. The env
  binds the msgpack server (pre-seeded with a hold-home command) and the
  vendored robots_realtime client connects back to it.
  ``rr_autostart=True`` (default) spawns that client via
  :class:`gap.connector.rr_launcher.RRSession`; ``rr_autostart=False``
  restores the two-terminal debug flow (run ``uv run --directory
  third_party/robots_realtime rr-session
  configs/franka/franka_robotiq_client.yaml`` yourself).
- ``ur_zed``: :class:`gap.envs.ur_zed_env.URZedEnv`. Direct pyzed capture
  + read-only RTDE joint state. **Perception-only**: the UR side has no
  motion interface wired (RTDE receive only), so the connector registers
  exclusively observation/camera tools — no ``robot.go_to_pose`` /
  gripper / trajectory tools exist in its registry, making accidental
  motion structurally impossible.

No ``sim.*`` tools are ever registered on a real connector — there is no
reset, no scripted success check, and no ground-truth world state on
hardware (``capabilities`` reports all False).

Safety: ``robot.go_home`` is guarded in :meth:`gap.connector.core.
Connector.go_home` — with ``config.is_real`` set (both real envs set it)
the call logs a warning and returns without moving, the direct port of
the source server's GoHome guard for real suites.

    import gap
    conn = gap.connector.real("franka")
    result = gap.execute(graph, conn)   # open-robot-skills auto-discovered
"""

from __future__ import annotations

import logging
import time
from pathlib import Path
from typing import Any

from gap_core.tools import ToolRegistry

from gap.connector.core import Capabilities, Connector

logger = logging.getLogger(__name__)

#: Default rr-session config for the Franka: ViserTeleop client agent
#: (FrankaOscClientCartesianAgent, robotiq_gripper=true, client_port 9000)
#: + RobotNode + ZED CameraNode. Resolved by rr-session relative to the
#: submodule checkout.
DEFAULT_FRANKA_RR_CONFIG = "configs/franka/franka_robotiq_client.yaml"

#: rr-session configs for the dionysus bimanual UR5e rig (relative to the
#: robots_realtime checkout). The dry-run one opens the arms read-only.
DEFAULT_UR5E_RR_CONFIG = "configs/ur5e_bimanual/gap_client.yaml"
DEFAULT_UR5E_RR_DRYRUN_CONFIG = "configs/ur5e_bimanual/gap_client_dryrun.yaml"

#: Observation/camera tool subset registered for perception-only robots.
_OBSERVATION_TOOL_PREFIXES = (
    "robot.get_observation",
    "robot.get_camera_pose",
    "robot.get_ee_pose",
    "robot.get_gripper",
    "robot.get_gripper_pose",
)


class RealConnector(Connector):
    """Connector over a real-hardware environment.

    Args:
        env: Real env instance (FrankaRealEnv / URZedEnv).
        config: The env's ``EnvConfig`` (``is_real=True``).
        camera_names: Camera override; defaults to ``config.default_cameras``.
        ik: Optional pre-built IK backend.
        rr_session: Optional :class:`~gap.connector.rr_launcher.RRSession`
            whose lifetime this connector owns (terminated on ``close()``).
        motion_enabled: When False (perception-only robots), register only
            the observation/camera tools — no motion, gripper, trajectory,
            or IK tools.
    """

    def __init__(
        self,
        env: Any,
        config: Any,
        *,
        camera_names: list[str] | None = None,
        ik: Any | None = None,
        rr_session: Any | None = None,
        motion_enabled: bool = True,
    ) -> None:
        super().__init__(env, config, camera_names=camera_names, ik=ik)
        self._rr_session = rr_session
        self._motion_enabled = bool(motion_enabled)

    # ------------------------------------------------------------------
    # Capabilities
    # ------------------------------------------------------------------

    @property
    def capabilities(self) -> Capabilities:
        """Real hardware: no scripted reset, success check, video, or world state.

        The env classes expose ``reset``/``enable_video_capture`` shims for
        interface parity, but on hardware "reset" only means "wait for an
        observation" and there is no ground truth — the benchmark harness
        must not branch into sim-only flows here.
        """
        return Capabilities(
            reset=False, success_check=False, video=False, world_state=False,
        )

    # ------------------------------------------------------------------
    # Tool registration
    # ------------------------------------------------------------------

    def _register_robot_tools(self, reg: ToolRegistry) -> None:
        if self._motion_enabled:
            super()._register_robot_tools(reg)
            return
        # Perception-only (ur_zed): observation/camera getters exclusively.
        rc = reg.register_callable
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

    # No _register_extra_tools override: the base hook is a no-op, so no
    # sim.* tools are ever registered on a real connector.

    # ------------------------------------------------------------------
    # Readiness
    # ------------------------------------------------------------------

    def wait_ready(self, timeout_s: float = 60.0, poll_s: float = 0.5) -> None:
        """Block until the first RGB observation arrives from the hardware.

        Polls ``env.get_observation()`` until the primary camera carries an
        RGB frame. On timeout, raises ``TimeoutError`` carrying the env's
        ``_diagnose_missing_rgb`` diagnosis (which wire stage is silent)
        when the env provides one.
        """
        cam_name = self.camera_names[0] if self.camera_names else None
        deadline = time.monotonic() + timeout_s
        while time.monotonic() < deadline:
            try:
                obs = self.env.get_observation()
            except Exception:
                logger.debug("wait_ready: get_observation failed", exc_info=True)
                obs = {}
            for key in ([cam_name] if cam_name else list(obs.keys())):
                cam = obs.get(key) if key else None
                if isinstance(cam, dict) and cam.get("images", {}).get("rgb") is not None:
                    logger.info("real connector ready: camera %r streaming", key)
                    return
            time.sleep(poll_s)

        diagnosis = ""
        diagnose = getattr(self.env, "_diagnose_missing_rgb", None)
        if callable(diagnose):
            try:
                diagnosis = " " + diagnose(cam_name or "")
            except Exception:
                pass
        rr_hint = ""
        if self._rr_session is not None and self._rr_session.poll() is not None:
            rr_hint = (
                f" rr-session exited with code {self._rr_session.poll()}"
                f" (see {self._rr_session.log_path})."
            )
        raise TimeoutError(
            f"No RGB observation from the real robot within {timeout_s:.0f}s."
            f"{rr_hint}{diagnosis}"
        )

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def close(self) -> None:
        """Terminate the owned rr-session (if any), then close the env."""
        if self._closed:
            return
        if self._rr_session is not None:
            try:
                self._rr_session.terminate()
            except Exception:
                logger.warning("rr-session terminate failed", exc_info=True)
        super().close()


# ---------------------------------------------------------------------------
# Factory
# ---------------------------------------------------------------------------


def real(
    robot: str = "franka",
    *,
    cameras: list[str] | None = None,
    rr_config: str | Path | None = None,
    rr_autostart: bool = True,
    rr_log_path: str | Path | None = None,
    port: int | None = None,
    wait_timeout_s: float = 60.0,
    **env_kwargs: Any,
) -> RealConnector:
    """Build a :class:`RealConnector` for real hardware.

    Args:
        robot: ``"franka"`` (robots_realtime msgpack bridge, full motion)
            or ``"ur_zed"`` (UR + ZED, perception-only).
        cameras: Camera-name override (default: the env's
            ``EnvConfig.default_cameras``).
        rr_config: Franka only — rr-session yaml, resolved relative to the
            ``third_party/robots_realtime`` checkout. Defaults to
            :data:`DEFAULT_FRANKA_RR_CONFIG`.
        rr_autostart: Franka only — spawn the rr-session client
            automatically. Pass False to drive it from a second terminal
            (debug flow); the connector then still blocks in
            ``wait_ready`` until your client connects.
        rr_log_path: Franka only — rr-session log tee destination.
        port: Franka only — msgpack server port (default 9000).
        wait_timeout_s: Seconds ``wait_ready`` blocks for the first RGB
            observation before raising.
        **env_kwargs: Extra env-factory kwargs (e.g. ``host=`` for franka;
            ``robot_ip=``, ``calibration_path=`` for ur_zed).

    Order of operations for franka (matters): the env constructor binds
    the msgpack server and pre-seeds a hold-home action *before* the
    rr-session client is spawned — the client retries until the server is
    up, and its very first action request must see a valid hold command
    (an empty reply makes it fall back to its Viser IK gizmo and jolt the
    arm).
    """
    from gap.envs.registry import resolve

    if robot == "franka":
        factory, key = resolve("franka_real")
        if port is not None:
            env_kwargs["port"] = port
        env, config = factory(key, 0, camera_names=cameras, **env_kwargs)

        rr_session = None
        if rr_autostart:
            from gap.connector.rr_launcher import RRSession

            rr_session = RRSession(
                rr_config or DEFAULT_FRANKA_RR_CONFIG, log_path=rr_log_path,
            )
        conn = RealConnector(
            env, config, camera_names=cameras, rr_session=rr_session,
        )
        try:
            conn.wait_ready(timeout_s=wait_timeout_s)
        except BaseException:
            conn.close()
            raise
        return conn

    if robot == "ur_zed":
        if rr_config is not None or port is not None:
            raise ValueError(
                "rr_config/port only apply to robot='franka' — ur_zed has no "
                "rr-session (direct pyzed + RTDE capture)"
            )
        factory, key = resolve("ur_zed")
        env, config = factory(key, 0, camera_names=cameras, **env_kwargs)
        # Perception-only: no motion tools (see RealConnector docstring).
        conn = RealConnector(
            env, config, camera_names=cameras, motion_enabled=False,
        )
        try:
            conn.wait_ready(timeout_s=wait_timeout_s)
        except BaseException:
            conn.close()
            raise
        return conn

    if robot == "ur5e_bimanual":
        from gap.connector.rr_launcher import DEFAULT_RR_DIR, RRSession
        from gap.connector.ur5e_bimanual import UR5eBimanualConnector

        factory, key = resolve("ur5e_bimanual_real")
        if port is not None:
            env_kwargs["port"] = port
        env, config = factory(key, 0, camera_names=cameras, **env_kwargs)
        rr_session = None
        try:
            if rr_autostart:
                cfg = rr_config or (DEFAULT_UR5E_RR_DRYRUN_CONFIG if env.dry_run else DEFAULT_UR5E_RR_CONFIG)
                binary = DEFAULT_RR_DIR / ".venv" / "bin" / "rr-session"
                command = [str(binary), str(cfg), "--no-tui"] if binary.exists() else None
                rr_session = RRSession(cfg, log_path=rr_log_path, command=command)
            rig_wire = env.wait_for_rig(timeout_s=wait_timeout_s)
        except BaseException:
            if rr_session is not None:
                rr_session.terminate()
            env.close()
            raise
        conn = UR5eBimanualConnector(env, config, rig_wire=rig_wire, camera_names=cameras, rr_session=rr_session)
        try:
            conn.wait_ready(timeout_s=wait_timeout_s)
        except BaseException:
            conn.close()
            raise
        return conn

    raise ValueError(f"unknown real robot {robot!r}; expected 'franka', 'ur_zed' or 'ur5e_bimanual'")


__all__ = ["DEFAULT_FRANKA_RR_CONFIG", "DEFAULT_UR5E_RR_CONFIG", "RealConnector", "real"]
