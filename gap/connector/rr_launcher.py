"""RRSession — subprocess launcher for the robots_realtime client.

robots_realtime is vendored as a pinned submodule at
``third_party/robots_realtime`` and is **never imported** by gap: it pins
its own environment (realtime loops, hardware SDKs, its own jax/pyroki
revisions). Instead this module spawns its ``rr-session`` entry point via
``uv run --directory <submodule> rr-session <config>`` in its own process
group, tees its output to a log file, and guarantees the whole group dies
with :meth:`RRSession.terminate` (SIGTERM → grace → SIGKILL) — rr-session
forks node subprocesses (robot, camera, teleop), and killing only the
direct child would leave the realtime robot loop running headless.

    from gap.connector.rr_launcher import RRSession

    with RRSession("configs/franka/franka_robotiq_client.yaml") as rr:
        ...  # FrankaRealEnv's msgpack server receives the client
"""

from __future__ import annotations

import logging
import os
import signal
import subprocess
import threading
import time
from collections.abc import Sequence
from pathlib import Path

logger = logging.getLogger(__name__)

#: Default submodule checkout the ``uv run`` is rooted at.
DEFAULT_RR_DIR = Path(__file__).resolve().parents[2] / "third_party" / "robots_realtime"


class RRSession:
    """Run ``rr-session <config>`` as a managed child process group.

    Args:
        config: Session yaml, resolved by rr-session relative to *cwd*
            (e.g. ``configs/franka/franka_robotiq_client.yaml``).
        cwd: robots_realtime checkout to ``uv run --directory`` into.
            Defaults to the vendored submodule.
        log_path: File the child's combined stdout/stderr is teed to.
            Defaults to ``/tmp/gap_rr_session_<pid>.log``.
        command: Full argv override (e.g. the checkout's own venv
            ``rr-session`` binary, which avoids ``uv run`` re-syncing the
            lockfile). The child still runs in *cwd*, so the session yaml's
            relative paths resolve.
        env: Child environment; default is this process's minus the
            variables that would leak this interpreter into the child
            (``PYTHONPATH``, ``VIRTUAL_ENV``, ``PYTHONHOME``).
    """

    def __init__(
        self,
        config: str | Path,
        *,
        cwd: str | Path | None = None,
        log_path: str | Path | None = None,
        command: Sequence[str] | None = None,
        env: dict[str, str] | None = None,
    ) -> None:
        self.config = str(config)
        self.cwd = Path(cwd) if cwd is not None else DEFAULT_RR_DIR
        if command is None:
            if not self.cwd.is_dir():
                raise FileNotFoundError(
                    f"robots_realtime checkout not found at {self.cwd} — "
                    f"run `git submodule update --init third_party/robots_realtime`"
                )
            # --no-tui: stdout is a pipe, the Rich TUI would garble the log.
            command = [
                "uv", "run", "--directory", str(self.cwd),
                "rr-session", self.config, "--no-tui",
            ]
        self.command = list(command)

        if env is None:
            env = {k: v for k, v in os.environ.items()
                   if k not in ("PYTHONPATH", "VIRTUAL_ENV", "PYTHONHOME")}
        self._proc = subprocess.Popen(
            self.command,
            cwd=str(self.cwd) if self.cwd.is_dir() else None,
            env=env,
            stdout=subprocess.PIPE,
            stderr=subprocess.STDOUT,
            # Own process group (setsid): rr-session forks node children;
            # terminate() signals the whole group.
            start_new_session=True,
        )
        self.log_path = Path(
            log_path if log_path is not None
            else f"/tmp/gap_rr_session_{self._proc.pid}.log"
        )
        logger.info(
            "rr-session spawned: pid=%d pgid=%d cmd=%s log=%s",
            self._proc.pid, os.getpgid(self._proc.pid), self.command, self.log_path,
        )

        self._tee_thread = threading.Thread(target=self._tee, daemon=True)
        self._tee_thread.start()

    # ------------------------------------------------------------------

    def _tee(self) -> None:
        """Copy child stdout/stderr lines to the log file (+ debug logger)."""
        assert self._proc.stdout is not None
        try:
            with open(self.log_path, "ab") as log:
                for line in self._proc.stdout:
                    log.write(line)
                    log.flush()
                    logger.debug("[rr-session] %s", line.rstrip().decode(errors="replace"))
        except Exception:
            logger.debug("rr-session log tee stopped", exc_info=True)

    # ------------------------------------------------------------------

    def poll(self) -> int | None:
        """Child's exit code, or None while it is still running."""
        return self._proc.poll()

    @property
    def pid(self) -> int:
        return self._proc.pid

    def terminate(self, grace_s: float = 12.0) -> None:
        """Kill the whole process group: SIGTERM, wait *grace_s*, SIGKILL."""
        if self._proc.poll() is not None:
            return
        try:
            pgid = os.getpgid(self._proc.pid)
        except ProcessLookupError:
            return

        logger.info("rr-session terminate: SIGTERM pgid=%d", pgid)
        try:
            os.killpg(pgid, signal.SIGTERM)
        except ProcessLookupError:
            return

        deadline = time.monotonic() + grace_s
        while time.monotonic() < deadline:
            if self._proc.poll() is not None:
                break
            time.sleep(0.05)

        if self._proc.poll() is None:
            logger.warning("rr-session did not exit in %.1fs; SIGKILL pgid=%d", grace_s, pgid)
        # Always sweep the group with SIGKILL: even when the direct child
        # exited on SIGTERM, forked grandchildren may linger.
        try:
            os.killpg(pgid, signal.SIGKILL)
        except ProcessLookupError:
            pass
        try:
            self._proc.wait(timeout=5.0)
        except subprocess.TimeoutExpired:
            logger.error("rr-session pid=%d unkillable?", self._proc.pid)

    # ------------------------------------------------------------------

    def __enter__(self) -> RRSession:
        return self

    def __exit__(self, exc_type, exc, tb) -> None:
        self.terminate()


__all__ = ["DEFAULT_RR_DIR", "RRSession"]
