"""UR5eBimanualRealEnv — the dionysus bimanual UR5e rig via robots_realtime.

The same split as :mod:`gap.envs.franka_real_env`: this env binds the msgpack
server; the ``rr-session`` client (``configs/ur5e_bimanual/gap_client.yaml`` in
``third_party/robots_realtime``) connects back, sends observations and receives
the latest action. On the client side, ``URBimanualClientAgent`` forwards
per-arm joint targets to ``URRegisterArm`` drivers (RTDE registers -> the vendor
"Default program"), and hosts the viser view with the STOP button.

What is different from the Franka env, and why:

- **Two arms, four RGB-D cameras.** ``arm_id`` follows ``rig.yaml`` (0 = right,
  the work arm; 1 = left). Every camera arrives with a calibrated camera->world
  pose (wrist cameras composed with forward kinematics by the client), so
  observations are already in the world frame and ``EnvConfig.arm_bases`` is None.
- **Time-parameterised streaming.** A trajectory is resampled along its
  max-norm joint path with a trapezoidal velocity profile (``max_joint_vel``,
  ``max_joint_acc``) and streamed at 50 Hz; the driver adds its own velocity cap
  and lead limit underneath.
- **Moves fail loudly.** A move that does not converge holds the arm and raises;
  nothing silently returns with a target still commanded.
- **A move can stop on contact.** The drivers carry the wrist's force/torque
  sensor (``tcp_force``, RTDE ``actual_TCP_force``). A trajectory streamed with
  ``stop_force_n`` tares the force first and, when it has changed by more than
  that, holds the arm where it is and returns ``stopped_on_force``. Without it a
  blocked move still "converges": r2s2rn's gasket press (2026-10-24) stalled 6 mm
  short on a bolt boss within the 0.01 rad settle tolerance, and kept pushing
  until the arm protective-stopped. ``speed_scale`` slows such a move so the
  50 Hz check catches the contact early.
- **Stops are not exceptions a graph can catch.** The latch file, a STOP from the
  viser button, the space key, or a stale observation stream during motion
  raise :class:`RigStopped` (a ``BaseException``), which unwinds the whole graph
  without running its recovery motions; ``gap run``'s ``finally`` then closes the
  connector, which holds and REST-stops both arms.
- **Dry run** (``GAP_UR_DRY_RUN=1``): nothing is ever sent; a shadow joint state
  advances as if each move succeeded, so a whole graph's plans can be reviewed
  in viser against the live cameras before anything moves.
"""

from __future__ import annotations

import json
import logging
import math
import os
import sys
import threading
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

import numpy as np

from gap.envs.base_env import BaseEnv
from gap.envs.msgpack_bridge import MsgpackNumpyServer, start_server_in_background

logger = logging.getLogger(__name__)

DEFAULT_CAMERAS = ("top", "bottom", "right_wrist", "left_wrist")
DEFAULT_ARMS = {0: "right", 1: "left"}
LATCH_PATH = Path(os.environ.get("DIONYSUS_STOP_LATCH", "/dev/shm/dionysus_rig_stop"))
_MAX_VIZ_POINTS = 3000


class RigStopped(BaseException):
    """The rig was stopped (latch / STOP / stale stream). Unwinds the graph."""


def _g(d: Any, key: str) -> Any:
    if not isinstance(d, dict):
        return None
    if key in d:
        return d[key]
    return d.get(key.encode())


def _to_str_keys(obj: Any) -> Any:
    if isinstance(obj, dict):
        return {(k.decode() if isinstance(k, bytes) else k): _to_str_keys(v) for k, v in obj.items()}
    if isinstance(obj, list):
        return [_to_str_keys(v) for v in obj]
    if isinstance(obj, bytes):
        try:
            return obj.decode()
        except UnicodeDecodeError:
            return obj
    return obj


def _latch_active() -> dict | None:
    try:
        raw = LATCH_PATH.read_text()
    except FileNotFoundError:
        return None
    except OSError:
        return {"reason": "latch unreadable"}
    try:
        return json.loads(raw) if raw else {"reason": "latched"}
    except ValueError:
        return {"reason": "latched"}


def _latch_set(reason: str) -> None:
    if LATCH_PATH.exists():
        return
    try:
        LATCH_PATH.write_text(json.dumps({"t": time.time(), "reason": reason, "source": "gap", "pid": os.getpid()}))
    except OSError:
        logger.exception("could not write the stop latch %s", LATCH_PATH)


def _rest_stop(ip: str) -> str:
    try:
        req = Request(f"http://{ip}/universal-robots/robot-api/program/v1/state",
                      data=json.dumps({"action": "stop"}).encode(), method="PUT")
        req.add_header("Content-Type", "application/json")
        with urlopen(req, timeout=1.5) as r:  # noqa: S310 - robot LAN
            return r.read().decode(errors="replace")[:120]
    except Exception as e:  # noqa: BLE001
        return f"FAILED {type(e).__name__}: {e}"


def trapezoid_samples(path: np.ndarray, vmax: float, amax: float, dt: float) -> np.ndarray:
    """Resample a joint polyline (N,6) at ``dt`` with a trapezoidal speed profile.

    The path parameter is cumulative max-norm joint distance, so every joint's
    speed stays <= ``vmax`` and acceleration <= ``amax``.
    """
    path = np.asarray(path, dtype=np.float64)
    seg = np.abs(np.diff(path, axis=0)).max(axis=1) if len(path) > 1 else np.zeros(0)
    S = float(seg.sum())
    if S < 1e-9:
        return path[-1:].copy()
    cum = np.concatenate([[0.0], np.cumsum(seg)])
    t_acc = vmax / amax
    s_acc = 0.5 * amax * t_acc**2
    if 2 * s_acc >= S:
        t_acc = math.sqrt(S / amax)
        vpk, t_flat = amax * t_acc, 0.0
    else:
        vpk, t_flat = vmax, (S - 2 * s_acc) / vmax
    T = 2 * t_acc + t_flat
    ts = np.arange(dt, T + dt, dt)
    s = np.empty_like(ts)
    for i, t in enumerate(ts):
        if t <= t_acc:
            s[i] = 0.5 * amax * t * t
        elif t <= t_acc + t_flat:
            s[i] = 0.5 * amax * t_acc**2 + vpk * (t - t_acc)
        else:
            td = min(t, T) - t_acc - t_flat
            s[i] = 0.5 * amax * t_acc**2 + vpk * t_flat + vpk * td - 0.5 * amax * td * td
    s = np.clip(s, 0.0, S)
    s[-1] = S
    out = np.empty((len(s), path.shape[1]))
    for j in range(path.shape[1]):
        out[:, j] = np.interp(s, cum, path[:, j])
    return out


class UR5eBimanualRealEnv(BaseEnv):
    """Real bimanual UR5e env over the robots_realtime msgpack bridge."""

    def __init__(
        self,
        camera_names: list[str] | None = None,
        port: int = 9000,
        host: str = "127.0.0.1",
        dry_run: bool | None = None,
        confirm: bool | None = None,
        max_joint_vel: float | None = None,
        max_joint_acc: float = 0.6,
        control_hz: float = 50.0,
        stale_after_s: float = 0.3,
        settle_timeout_s: float = 15.0,
        approval_timeout_s: float = 300.0,
        key_stop: bool | None = None,
    ) -> None:
        super().__init__()
        self.camera_names = list(camera_names or DEFAULT_CAMERAS)
        env = os.environ
        self.dry_run = bool(int(env.get("GAP_UR_DRY_RUN", "0"))) if dry_run is None else bool(dry_run)
        self.confirm = bool(int(env.get("GAP_UR_CONFIRM", "0"))) if confirm is None else bool(confirm)
        self.max_joint_vel = float(env.get("GAP_UR_MAX_JOINT_VEL", "0.20")) if max_joint_vel is None \
            else float(max_joint_vel)
        self.max_joint_acc = float(max_joint_acc)
        self.dt = 1.0 / float(control_hz)
        self.stale_after_s = float(stale_after_s)
        self.settle_timeout_s = float(settle_timeout_s)
        self.approval_timeout_s = float(approval_timeout_s)

        self.arm_names: dict[int, str] = dict(DEFAULT_ARMS)
        self.rig_wire: dict | None = None
        # Arm IPs come only from the live rig description the client sends, so a
        # rehearsal against the fake rig can never REST-stop the real arms.
        self.ips: dict[str, str] = {}
        self._drove: set[str] = set()  # arms that were sent a motion or gripper target

        self._lock = threading.Lock()
        self._target_q: dict[str, np.ndarray] = {}
        self._target_grip: dict[str, float] = {}
        self._meas: dict[str, dict] = {}
        self._virt_q: dict[str, np.ndarray] = {}
        self._virt_grip: dict[str, float] = {}
        self._cams: dict[str, dict] = {}
        self._obs: dict[str, Any] = {}
        self._last_obs_t = 0.0
        self.stop_reason: str | None = None
        self._ack: dict | None = None
        self._approval_seq = 0
        self._viz: dict[str, Any] = {"seq": 0, "status": "connected", "plans": [], "targets": [], "points": [],
                                     "awaiting_approval": None, "node": ""}
        self._fk = None  # list[URArm] once the rig description arrives
        self._sim_step_count = 0
        self.max_steps = 999999
        self._closed = False

        self.server = MsgpackNumpyServer(host=host, port=port)
        self.server.latest_action = {"timestamp": time.time(), "viz": self._viz}
        self.server.on_observation = self._on_frame
        start_server_in_background(self.server)

        self._stop_evt = threading.Event()
        self._republish = threading.Thread(target=self._republish_loop, daemon=True, name="UR5eRepublish")
        self._republish.start()
        self._key_thread = None
        if (key_stop if key_stop is not None else env.get("GAP_UR_KEYSTOP", "1") == "1") and sys.stdin.isatty():
            self._key_thread = threading.Thread(target=self._key_loop, daemon=True, name="UR5eKeyStop")
            self._key_thread.start()
        logger.warning("[UR5eRealEnv] bridge on %s:%d dry_run=%s confirm=%s vmax=%.2f rad/s",
                       host, port, self.dry_run, self.confirm, self.max_joint_vel)

    # ── wire ──────────────────────────────────────────────────────────────

    def _republish_loop(self) -> None:
        try:
            while not self._stop_evt.is_set():
                action: dict[str, Any] = {"timestamp": time.time()}
                with self._lock:
                    if not self.dry_run and self.stop_reason is None:
                        for name, q in self._target_q.items():
                            action[name] = {"joint_pos": q.astype(np.float32).tolist(),
                                            "gripper": float(self._target_grip.get(name, 1.0))}
                    action["viz"] = self._viz
                self.server.latest_action = action
                time.sleep(self.dt)
        except Exception:  # noqa: BLE001 - make a dead republisher loud; the client then holds
            logger.exception("[UR5eRealEnv] REPUBLISHER CRASHED — the client will hold (stale actions)")

    def _key_loop(self) -> None:
        import select
        import termios
        import tty

        fd = sys.stdin.fileno()
        try:
            old = termios.tcgetattr(fd)
        except termios.error:
            return
        try:
            tty.setcbreak(fd)
            while not self._stop_evt.is_set():
                r, _, _ = select.select([sys.stdin], [], [], 0.2)
                if r:
                    ch = sys.stdin.read(1)
                    if ch in (" ", "s", "S"):
                        self.emergency_stop("space key in the gap terminal")
        except Exception:  # noqa: BLE001
            logger.debug("key listener stopped", exc_info=True)
        finally:
            try:
                termios.tcsetattr(fd, termios.TCSADRAIN, old)
            except termios.error:
                pass

    def _on_frame(self, msg: dict) -> None:
        """Every received frame (server thread): the fields a client sends only
        now and then — rig description, STOP, approval answers — and freshness."""
        if not isinstance(msg, dict):
            return
        self._last_obs_t = time.time()
        rig = _g(msg, "rig")
        if rig is not None and self.rig_wire is None:
            self._adopt_rig(_to_str_keys(rig))
        stop = _g(msg, "stop")
        if stop is not None and self.stop_reason is None:
            self.stop_reason = str(_to_str_keys(stop).get("reason", "client stop"))
            self.hold_all()
            logger.error("[UR5eRealEnv] client reports STOP: %s", self.stop_reason)
        ack = _g(msg, "viz_ack")
        if ack is not None:
            self._ack = _to_str_keys(ack)
        # Camera images ride on only some frames (each camera at most every camera_period_s, and only a new
        # picture), so they are kept here, as they arrive, rather than read from whatever frame happens to be the
        # latest when an observation is asked for: that served a wrist picture from a previous move to an
        # eye-in-hand servo, which then corrected for where the hand had been (r2s2rn gasket run007, 2026-10-20).
        for cam in self.camera_names:
            entry = self._camera_entry(_g(msg, cam))
            if entry is not None:
                self._cams[cam] = entry

    def _update_from_network(self) -> None:
        msg = self.server.latest_observation
        if not isinstance(msg, dict):
            return
        for name in self.arm_names.values():
            a = _g(msg, name)
            if not isinstance(a, dict):
                continue
            jp = _g(a, "joint_pos")
            if jp is None:
                continue
            meas = {
                "q": np.asarray(jp, dtype=np.float64).reshape(-1)[:6],
                "grip": float(np.asarray(jp).reshape(-1)[6]) if np.asarray(jp).size > 6 else 1.0,
                "tcp": np.asarray(_g(a, "tcp_world"), dtype=np.float64) if _g(a, "tcp_world") is not None else None,
                "driving": int(_g(a, "driving") or 0),
                "latched": int(_g(a, "latched") or 0),
                "force": np.asarray(_g(a, "tcp_force"), dtype=np.float64).reshape(-1)[:6]
                if _g(a, "tcp_force") is not None else None,
            }
            with self._lock:
                self._meas[name] = meas
                if name not in self._target_q:  # seed: the first command is "stay here"
                    self._target_q[name] = meas["q"].copy()
                    self._target_grip[name] = meas["grip"]
                if name not in self._virt_q:
                    self._virt_q[name] = meas["q"].copy()
                    self._virt_grip[name] = meas["grip"]
        for cam in self.camera_names:
            entry = self._camera_entry(_g(msg, cam))
            if entry is not None:
                self._cams[cam] = entry

    @staticmethod
    def _camera_entry(c: Any) -> dict[str, Any] | None:
        """One camera's block of a client frame as an observation entry, or None when the frame has no picture of it."""
        if not isinstance(c, dict):
            return None
        images = _g(c, "images") or {}
        rgb = _g(images, "rgb")
        if rgb is None:
            return None
        entry: dict[str, Any] = {"images": {"rgb": np.asarray(rgb)}}
        depth = _g(c, "depth_data")
        if depth is not None:
            entry["images"]["depth"] = np.asarray(depth, dtype=np.float32)
        intr = _g(_g(c, "intrinsics") or {}, "left") or {}
        K = _g(intr, "intrinsics_matrix")
        if K is not None:
            entry["intrinsics"] = np.asarray(K, dtype=np.float32)
        pose = _g(c, "pose")
        if pose is not None:
            entry["pose"] = np.asarray(pose, dtype=np.float32)
        entry["timestamp"] = float(_g(c, "timestamp") or 0.0)
        return entry

    def _adopt_rig(self, rig: dict, live: bool = True) -> None:
        from gap.connector.ur_ik import arms_from_rig_wire

        self.rig_wire = rig
        self.arm_names = {int(a["arm_id"]): name for name, a in rig["arms"].items()}
        if live:
            self.ips = {name: str(a["ip"]) for name, a in rig["arms"].items() if a.get("ip")}
        self._fk = arms_from_rig_wire(rig)
        logger.info("[UR5eRealEnv] rig adopted: arms=%s surface_z=%.3f", self.arm_names, rig.get("surface_z", 0.0))

    def wait_for_rig(self, timeout_s: float = 30.0) -> dict:
        deadline = time.time() + timeout_s
        while time.time() < deadline:
            self._update_from_network()
            if self.rig_wire is not None and len(self._meas) == len(self.arm_names):
                return self.rig_wire
            time.sleep(0.1)
        raise TimeoutError("no rig description / arm states from rr-session")

    # ── observation ───────────────────────────────────────────────────────

    def _q(self, name: str) -> np.ndarray | None:
        src = self._virt_q if self.dry_run else {k: v["q"] for k, v in self._meas.items()}
        q = src.get(name)
        return None if q is None else np.asarray(q, dtype=np.float64)

    def _grip(self, name: str) -> float:
        if self.dry_run:
            return float(self._virt_grip.get(name, 1.0))
        return float(self._meas.get(name, {}).get("grip", 1.0))

    def get_observation(self) -> dict[str, Any]:
        self._update_from_network()
        obs: dict[str, Any] = {}
        for arm_id, name in self.arm_names.items():
            q = self._q(name)
            if q is None:
                continue
            g = self._grip(name)
            obs[f"robot_joint_pos_{arm_id}"] = np.concatenate([q, [g]]).astype(np.float32)
            tcp = None
            if self._fk is not None and arm_id < len(self._fk):
                from gap.connector.ur_ik import matrix_to_quat_wxyz

                T = self._fk[arm_id].tcp_world(q)
                tcp = np.concatenate([T[:3, 3], matrix_to_quat_wxyz(T[:3, :3])])
            elif self._meas.get(name, {}).get("tcp") is not None:
                tcp = self._meas[name]["tcp"]
            if tcp is not None:
                obs[f"robot_cartesian_pos_{arm_id}"] = np.concatenate([tcp, [g]]).astype(np.float32)
        for cam, entry in self._cams.items():
            obs[cam] = entry
        self._obs = obs
        return obs

    # ── stop ──────────────────────────────────────────────────────────────

    def check_stop(self) -> None:
        """Raise :class:`RigStopped` if anything has stopped the rig."""
        if self.stop_reason is None:
            latch = _latch_active()
            if latch:
                self.stop_reason = f"latch: {latch.get('reason')}"
        if self.stop_reason is not None:
            raise RigStopped(self.stop_reason)

    def hold_all(self) -> None:
        with self._lock:
            for name, m in self._meas.items():
                self._target_q[name] = m["q"].copy()
                self._target_grip[name] = m["grip"]

    def emergency_stop(self, reason: str) -> None:
        """Layer 1 (HOLD + latch) and Layer 2 (REST program stop) from the GaP side."""
        first = self.stop_reason is None
        self.stop_reason = self.stop_reason or reason
        self.hold_all()
        _latch_set(reason)
        if first:
            logger.error("[UR5eRealEnv] EMERGENCY STOP: %s", reason)
            for name, ip in self.ips.items():
                threading.Thread(target=lambda n=name, i=ip: logger.error(
                    "[UR5eRealEnv] REST stop %s: %s", n, _rest_stop(i)), daemon=True).start()

    # ── viz / approval ────────────────────────────────────────────────────

    def viz_update(self, **fields: Any) -> None:
        with self._lock:
            viz = dict(self._viz)
            for k, v in fields.items():
                if k == "points" and v:
                    v = [dict(p, xyz=np.asarray(p["xyz"], dtype=np.float32)[
                        :: max(1, len(p["xyz"]) // _MAX_VIZ_POINTS)].tolist()) for p in v]
                viz[k] = v
            viz["seq"] = int(viz.get("seq", 0)) + 1
            self._viz = viz

    def request_approval(self, summary: str) -> bool:
        """Block until the operator approves/rejects in viser (``confirm`` mode only)."""
        if not self.confirm or self.dry_run:
            return True
        self._approval_seq += 1
        aid = self._approval_seq
        self._ack = None
        self.viz_update(awaiting_approval={"id": aid, "summary": summary}, status=f"awaiting approval #{aid}")
        logger.warning("[UR5eRealEnv] waiting for viser approval #%d: %s", aid, summary)
        deadline = time.time() + self.approval_timeout_s
        try:
            while time.time() < deadline:
                self._update_from_network()
                self.check_stop()
                ack = self._ack
                if ack is not None and int(ack.get("id", -1)) == aid:
                    return bool(ack.get("approved"))
                time.sleep(0.05)
            logger.error("[UR5eRealEnv] approval #%d timed out — rejecting", aid)
            return False
        finally:
            self.viz_update(awaiting_approval=None, status="running")

    # ── motion ────────────────────────────────────────────────────────────

    def _fresh(self) -> bool:
        return time.time() - self._last_obs_t <= self.stale_after_s

    def stream_joint_trajectory(self, waypoints: list[list[float]], *, settle_tolerance: float = 0.01,
                                settle_max_steps: int = 60, arm_id: int = 0, stop_force_n: float = 0.0,
                                speed_scale: float = 1.0) -> dict:
        """Time-parameterise and stream; then wait for convergence (raises if it does not).

        With ``stop_force_n`` > 0 the wrist force is tared first, and the move ends early, holding the arm where
        it is, once the force has changed by more than that many newtons (``stopped_on_force`` in the result).
        ``speed_scale`` (0.01 to 1) scales the joint speed and acceleration limits."""
        self._update_from_network()
        self.check_stop()
        name = self.arm_names[int(arm_id)]
        q0 = self._q(name)
        if q0 is None:
            raise RuntimeError(f"no joint state for arm {arm_id} ({name})")
        path = np.vstack([q0] + [np.asarray(w, dtype=np.float64)[:6] for w in waypoints])
        k = float(np.clip(speed_scale, 0.01, 1.0))
        samples = trapezoid_samples(path, self.max_joint_vel * k, self.max_joint_acc * k, self.dt)
        guarded = float(stop_force_n) > 0.0
        if self.dry_run:
            with self._lock:
                self._virt_q[name] = samples[-1].copy()
            time.sleep(min(0.2, 0.01 * len(samples)))
            self._sim_step_count += len(samples)
            return {"final_tracking_error_rad": 0.0, "dry_run": True, **({"stopped_on_force": False} if guarded else {})}
        if not self._fresh():
            raise RigStopped("observation stream stale before motion")
        f0 = self._tare_force(name) if guarded else None
        self._drove.add(name)
        t_next = time.perf_counter()
        peak = 0.0
        for i, q in enumerate(samples):
            self._update_from_network()
            self.check_stop()
            if not self._fresh():
                self.emergency_stop("observation stream stale during motion")
                raise RigStopped(self.stop_reason or "stale")
            if f0 is not None:
                f = self._force(name)
                df = float(np.linalg.norm(f[:3] - f0)) if f is not None else math.inf
                peak = max(peak, df)
                if df > stop_force_n:
                    here = self._q(name)
                    with self._lock:
                        self._target_q[name] = here.copy()
                    logger.info("[UR5eRealEnv] %s stopped on force: %.1f N after %d of %d samples",
                                name, df, i, len(samples))
                    return {"stopped_on_force": True, "force_n": round(df, 2), "progress": i / len(samples),
                            "final_tracking_error_rad": float(np.abs(here - samples[-1]).max())}
            with self._lock:
                self._target_q[name] = q
            self._sim_step_count += 1
            t_next += self.dt
            time.sleep(max(0.0, t_next - time.perf_counter()))
        out = self._settle(name, samples[-1], max(settle_tolerance, 0.005),
                           max(self.settle_timeout_s, 3.0 * len(samples) * self.dt))
        if f0 is not None:
            out.update(stopped_on_force=False, force_n=round(peak, 2))
        return out

    def _force(self, name: str) -> np.ndarray | None:
        with self._lock:
            m = self._meas.get(name) or {}
        return None if m.get("force") is None else m["force"].copy()

    def _tare_force(self, name: str, n: int = 10) -> np.ndarray:
        """The force now (mean of *n* stream frames, base frame), what a guarded move measures against."""
        got = []
        for _ in range(n):
            self._update_from_network()
            f = self._force(name)
            if f is not None:
                got.append(f[:3])
            time.sleep(self.dt)
        if len(got) < n // 2:
            raise RuntimeError(f"no wrist force from arm {name}: a force-guarded move is refused")
        return np.mean(got, axis=0)

    def _settle(self, name: str, q_final: np.ndarray, tol: float, timeout: float) -> dict:
        deadline = time.time() + timeout
        err = math.inf
        while time.time() < deadline:
            self._update_from_network()
            self.check_stop()
            q = self._q(name)
            if q is not None:
                err = float(np.abs(q - q_final).max())
                if err < tol:
                    return {"final_tracking_error_rad": err}
            time.sleep(0.02)
        self.hold_all()
        raise RuntimeError(f"arm {name} did not reach its target within {timeout:.1f}s "
                           f"(max joint error {err:.4f} rad) — holding where it is")

    def move_to_joints_blocking(self, joints, *, tolerance: float = 0.01, max_steps: int = 350,
                                arm_id: int = 0) -> None:
        target = np.asarray(joints, dtype=np.float64).reshape(-1)[:6]
        if max_steps == 0:
            # Streaming servo: only small steps are accepted without a time law.
            self._update_from_network()
            self.check_stop()
            name = self.arm_names[int(arm_id)]
            q = self._q(name)
            if q is None or float(np.abs(target - q).max()) > 0.05:
                raise RuntimeError("fire-and-forget joint target more than 0.05 rad away — refused")
            with self._lock:
                if self.dry_run:
                    self._virt_q[name] = target
                else:
                    self._target_q[name] = target
                    self._drove.add(name)
            return
        self.stream_joint_trajectory([target.tolist()], settle_tolerance=tolerance, arm_id=arm_id)

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        """Command the gripper; return when it reached the command or stalled (object)."""
        self._update_from_network()
        self.check_stop()
        name = self.arm_names[int(arm_id)]
        frac = float(np.clip(fraction, 0.0, 1.0))
        if self.dry_run:
            with self._lock:
                # A dry-run close pretends to hold something, so grasp checks pass.
                self._virt_grip[name] = frac if frac > 0.5 else 0.4
            return
        with self._lock:
            self._target_grip[name] = frac
            self._drove.add(name)
        t0 = time.time()
        last, last_change = self._grip(name), time.time()
        while time.time() - t0 < 3.0:
            time.sleep(0.05)
            self._update_from_network()
            self.check_stop()
            g = self._grip(name)
            if abs(g - frac) < 0.03:
                return
            if abs(g - last) > 0.005:
                last, last_change = g, time.time()
            elif time.time() - t0 > 0.6 and time.time() - last_change > 0.3:
                return  # stalled on an object (or at its limit)

    def _step_once(self) -> None:
        self._update_from_network()
        self.check_stop()
        self._sim_step_count += 1
        time.sleep(self.dt)

    # ── BaseEnv ───────────────────────────────────────────────────────────

    def reset(self, *, seed: int | None = None, options: dict | None = None):
        self.wait_for_rig()
        self._sim_step_count = 0
        return self.get_observation(), {}

    def step(self, action):
        self._sim_step_count += 1
        return self.get_observation(), 0.0, False, False, {}

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def _diagnose_missing_rgb(self, cam_name: str) -> str:
        msg = self.server.latest_observation
        if msg is None:
            return "(rr-session not connected to the bridge yet)"
        keys = [k.decode() if isinstance(k, bytes) else str(k) for k in msg]
        return f"(wire keys={keys}; camera {cam_name!r} has not delivered RGB — CameraNode up? recording_studio camera stage OFF?)"

    def close(self) -> None:
        """HOLD both arms, REST-stop their programs, stop the bridge."""
        if self._closed:
            return
        self._closed = True
        try:
            self.hold_all()
            time.sleep(3 * self.dt)
            if not self.dry_run:
                for name, ip in self.ips.items():
                    if name not in self._drove:
                        continue  # never driven here (e.g. a read-only arm): leave its program alone
                    logger.info("[UR5eRealEnv] close: REST stop %s -> %s", name, _rest_stop(ip))
        finally:
            self._stop_evt.set()
            try:
                self.server.stop()
            except Exception:  # noqa: BLE001
                logger.debug("msgpack server stop failed", exc_info=True)


def make_env(suite_name: str = "ur5e_bimanual_real", task_id: int = 0, camera_names: list[str] | None = None,
             enable_render: bool = False, *, port: int = 9000, host: str = "127.0.0.1", rig_yaml: str | None = None,
             **extra: Any):
    """Build the env + its EnvConfig (registry contract). Home/TCP come from rig.yaml."""
    from gap.connector.ur_ik import load_rig_arms
    from gap.envs.registry import EnvConfig

    arms, wire = load_rig_arms(rig_yaml)
    env = UR5eBimanualRealEnv(camera_names=camera_names, port=port, host=host, **extra)
    env._adopt_rig(wire, live=False)  # offline copy (no IPs); the client's live one replaces it
    env.rig_wire = None  # still wait for the live one in reset()/wait_for_rig()
    config = EnvConfig(
        arm_dof=6,
        num_arms=len(arms),
        action_mode="absolute_joints",
        control_freq=50.0,
        home_joints=tuple(float(v) for v in arms[0].home_q),
        tcp_offset=tuple(float(v) for v in arms[0].tcp_offset),
        tcp_rotation_z=None,
        arm_bases=None,  # poses are already world-frame (full SE(3) bases live in the IK)
        robot_urdf_path="ur5e_description",
        default_cameras=tuple(env.camera_names),
        is_real=True,
    )
    return env, config


__all__ = ["RigStopped", "UR5eBimanualRealEnv", "make_env", "trapezoid_samples"]
