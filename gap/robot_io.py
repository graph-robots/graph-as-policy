"""Low-level robot I/O shared by simulator and hardware graph runners.

The graph runtime owns kinematics, planning, and command convergence.  An I/O
driver only measures joints and cameras and applies one position target for one
control interval.  Task state, simulator handles, and robot models stay outside
this interface.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Protocol, runtime_checkable

import numpy as np


@dataclass(frozen=True)
class CameraSample:
    """One camera capture with calibration in the graph's reference frame."""

    rgb: np.ndarray
    depth_m: np.ndarray | None
    intrinsics: np.ndarray
    pose: np.ndarray
    timestamp_s: float


@dataclass(frozen=True)
class RobotSample:
    """Measured state; camera captures may be older than the joint sample."""

    sequence: int
    timestamp_s: float
    joint_positions: np.ndarray
    joint_velocities: np.ndarray
    cameras: dict[str, CameraSample]


@dataclass(frozen=True)
class JointPositionCommand:
    """Absolute targets in ``RobotIO.joint_names`` order.

    Gains are optional because some position controllers keep them fixed.  A
    driver must reject supplied gains if it cannot apply them.
    """

    positions: np.ndarray
    kp: np.ndarray | None = None
    kd: np.ndarray | None = None


@runtime_checkable
class RobotIO(Protocol):
    """The only live robot boundary used by a perception graph runner.

    ``exchange`` applies one whole-robot target for one control interval and
    returns a sample newer than the one available before the call.  Drivers
    that cannot step their own clock, such as hardware, wait for the next
    measured sample.  Cameras may run at a lower rate and retain their own
    timestamps.  Gripper joints are included in the named joint vector.
    """

    @property
    def joint_names(self) -> tuple[str, ...]: ...

    @property
    def control_period_s(self) -> float: ...

    def observe(self) -> RobotSample: ...

    def exchange(self, command: JointPositionCommand) -> RobotSample: ...


def validate_command(command: JointPositionCommand, joint_count: int) -> None:
    """Reject ambiguous or unsafe numeric payloads before a driver writes."""
    if joint_count <= 0:
        raise ValueError("joint_count must be positive")
    for name in ("positions", "kp", "kd"):
        value = getattr(command, name)
        if value is None:
            continue
        array = np.asarray(value, dtype=np.float64)
        if array.shape != (joint_count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {joint_count} finite values")
        if name != "positions" and np.any(array < 0.0):
            raise ValueError(f"{name} must be nonnegative")


def validate_sample(sample: RobotSample, joint_count: int) -> None:
    """Check freshness-independent shape and numeric invariants at the seam."""
    if sample.sequence < 0 or not np.isfinite(sample.timestamp_s):
        raise ValueError("robot sample needs a valid sequence and timestamp")
    for name in ("joint_positions", "joint_velocities"):
        array = np.asarray(getattr(sample, name), dtype=np.float64)
        if array.shape != (joint_count,) or not np.isfinite(array).all():
            raise ValueError(f"{name} must contain {joint_count} finite values")
    for name, camera in sample.cameras.items():
        rgb = np.asarray(camera.rgb)
        depth = None if camera.depth_m is None else np.asarray(camera.depth_m)
        if rgb.ndim != 3 or rgb.shape[-1] != 3 or rgb.dtype != np.uint8:
            raise ValueError(f"camera {name!r} needs uint8 HxWx3 RGB")
        if depth is not None and (depth.shape != rgb.shape[:2] or depth.dtype != np.float32):
            raise ValueError(f"camera {name!r} needs float32 depth matching RGB")
        if np.asarray(camera.intrinsics).shape != (3, 3):
            raise ValueError(f"camera {name!r} needs 3x3 intrinsics")
        if np.asarray(camera.pose).shape != (4, 4):
            raise ValueError(f"camera {name!r} needs a 4x4 pose")
        if not np.isfinite(camera.timestamp_s):
            raise ValueError(f"camera {name!r} needs a finite timestamp")
