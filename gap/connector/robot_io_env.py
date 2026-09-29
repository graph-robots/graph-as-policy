"""Adapt joint-and-camera I/O to the existing GaP connector environment API.

The adapter owns no simulator, task, or kinematic model.  A separate FK
function and gripper mappings translate raw joint measurements for the graph's
existing observation schema.  All control paths end in ``RobotIO.exchange``.
"""

from __future__ import annotations

from collections.abc import Callable
from dataclasses import dataclass

import numpy as np
from gap_core.types import matrix_to_pose

from gap.robot_io import (
    JointPositionCommand,
    RobotIO,
    RobotSample,
    validate_command,
    validate_sample,
)


@dataclass(frozen=True)
class ArmLayout:
    """Pure mapping between one arm and the driver's whole-robot joint order."""

    joint_names: tuple[str, ...]
    gripper_joint_names: tuple[str, ...] = ()
    gripper_fraction: Callable[[np.ndarray], float] | None = None
    gripper_target: Callable[[float], np.ndarray] | None = None


class RobotIOEnv:
    """Compatibility facade for the paper-release ``Connector``.

    ``fk`` is a pure kinematic-model lookup.  It must not read the live
    simulator.  The driver itself supplies only joint measurements, cameras,
    and an atomic position-target exchange.
    """

    def __init__(
        self,
        io: RobotIO,
        arms: tuple[ArmLayout, ...],
        fk: Callable[[int, np.ndarray], np.ndarray],
    ) -> None:
        if not arms:
            raise ValueError("at least one arm is required")
        self.io = io
        self.arms = arms
        self.fk = fk
        self._indices = tuple(
            np.asarray([io.joint_names.index(name) for name in arm.joint_names], dtype=np.int64)
            for arm in arms
        )
        self._gripper_indices = tuple(
            np.asarray([io.joint_names.index(name) for name in arm.gripper_joint_names], dtype=np.int64)
            for arm in arms
        )
        used = [int(index) for group in (*self._indices, *self._gripper_indices) for index in group]
        if len(used) != len(set(used)):
            raise ValueError("arm and gripper joint mappings must be disjoint")
        for arm in arms:
            if bool(arm.gripper_joint_names) != bool(arm.gripper_fraction and arm.gripper_target):
                raise ValueError("gripper mappings require both conversion functions")
        # Connector construction may happen before a simulator owner thread
        # starts pumping its bus. Defer the first read until graph execution.
        self._sample: RobotSample | None = None
        self._target: np.ndarray | None = None
        self._sim_step_count = 0
        self.max_steps = 999999
        self._current_done = False

    def _checked(self, sample: RobotSample) -> RobotSample:
        validate_sample(sample, len(self.io.joint_names))
        return sample

    def _ensure_sample(self) -> RobotSample:
        if self._sample is None:
            self._sample = self._checked(self.io.observe())
            self._target = np.asarray(self._sample.joint_positions, dtype=np.float64).copy()
        return self._sample

    def _exchange(self) -> RobotSample:
        before = self._ensure_sample().sequence
        assert self._target is not None
        command = JointPositionCommand(self._target.copy())
        validate_command(command, len(self.io.joint_names))
        sample = self._checked(self.io.exchange(command))
        if sample.sequence <= before:
            raise RuntimeError("robot command returned without a fresh joint sample")
        self._sample = sample
        self._sim_step_count += 1
        return sample

    def get_observation(self) -> dict:
        measured = (
            self._ensure_sample()
            if self._sample is None
            else self._checked(self.io.observe())
        )
        assert self._sample is not None
        if measured.sequence < self._sample.sequence:
            raise RuntimeError("robot observation sequence moved backward")
        self._sample = measured
        sample = self._sample
        out: dict = {}
        for arm_id, (arm, indices, gripper_indices) in enumerate(
            zip(self.arms, self._indices, self._gripper_indices, strict=True)
        ):
            joints = np.asarray(sample.joint_positions[indices], dtype=np.float64)
            fraction = (
                float(arm.gripper_fraction(sample.joint_positions[gripper_indices]))
                if arm.gripper_fraction is not None
                else 1.0
            )
            if not 0.0 <= fraction <= 1.0:
                raise ValueError("measured gripper open fraction must be in [0, 1]")
            out[f"robot_joint_pos_{arm_id}"] = np.append(joints, fraction)
            out[f"robot_joint_vel_{arm_id}"] = np.asarray(
                sample.joint_velocities[indices], dtype=np.float64
            )
            pose = matrix_to_pose(np.asarray(self.fk(arm_id, joints), dtype=np.float64))
            out[f"robot_cartesian_pos_{arm_id}"] = np.asarray(
                [*pose["position"].values(), *pose["rotation"].values()], dtype=np.float64
            )
        for name, camera in sample.cameras.items():
            pose = matrix_to_pose(np.asarray(camera.pose, dtype=np.float64))
            images = {"rgb": camera.rgb}
            if camera.depth_m is not None:
                images["depth"] = camera.depth_m
            out[name] = {
                "images": images,
                "intrinsics": np.asarray(camera.intrinsics),
                "pose": np.asarray(
                    [*pose["position"].values(), *pose["rotation"].values()], dtype=np.float64
                ),
                "timestamp": camera.timestamp_s,
            }
        return out

    def move_to_joints_blocking(
        self,
        target,
        *,
        tolerance: float = 0.01,
        max_steps: int = 120,
        arm_id: int = 0,
    ) -> None:
        indices = self._indices[int(arm_id)]
        values = np.asarray(target, dtype=np.float64)
        if values.shape != (len(indices),) or not np.isfinite(values).all():
            raise ValueError("arm target has invalid shape or nonfinite positions")
        self._ensure_sample()
        assert self._target is not None
        self._target[indices] = values
        for step in range(max(1, int(max_steps))):
            sample = self._exchange()
            if max_steps == 0:
                return
            if step > 0 and np.linalg.norm(sample.joint_positions[indices] - values) < tolerance:
                return

    def stream_joint_trajectory(
        self,
        waypoints,
        *,
        settle_tolerance: float = 0.01,
        settle_max_steps: int = 60,
        arm_id: int = 0,
    ) -> bool:
        waypoints = list(waypoints)
        for waypoint in waypoints:
            self.move_to_joints_blocking(waypoint, max_steps=0, arm_id=arm_id)
        if waypoints:
            self.move_to_joints_blocking(
                waypoints[-1],
                tolerance=settle_tolerance,
                max_steps=settle_max_steps,
                arm_id=arm_id,
            )
        return True

    def stream_dual_trajectory(
        self,
        tracks: dict[int, list[list[float]]],
        *,
        settle_tolerance: float = 0.005,
        settle_max_steps: int = 60,
    ) -> bool:
        """Write every arm's next target in one whole-robot command."""
        tracks = {int(arm): list(rows) for arm, rows in tracks.items() if rows}
        if not tracks:
            return True
        self._ensure_sample()
        assert self._target is not None
        length = max(len(rows) for rows in tracks.values())
        for step in range(length):
            for arm_id, rows in tracks.items():
                indices = self._indices[arm_id]
                values = np.asarray(rows[min(step, len(rows) - 1)], dtype=np.float64)
                if values.shape != (len(indices),) or not np.isfinite(values).all():
                    raise ValueError("arm target has invalid shape or nonfinite positions")
                self._target[indices] = values
            self._exchange()
        for _ in range(max(0, int(settle_max_steps))):
            sample = self._exchange()
            if all(
                np.linalg.norm(sample.joint_positions[self._indices[arm_id]] - self._target[self._indices[arm_id]])
                < settle_tolerance
                for arm_id in tracks
            ):
                break
        return True

    def _set_gripper(self, fraction: float, arm_id: int = 0) -> None:
        self._ensure_sample()
        assert self._target is not None
        arm = self.arms[int(arm_id)]
        indices = self._gripper_indices[int(arm_id)]
        if arm.gripper_target is None:
            raise ValueError("arm has no gripper")
        if not np.isfinite(fraction) or not 0.0 <= fraction <= 1.0:
            raise ValueError("gripper open fraction must be in [0, 1]")
        values = np.asarray(arm.gripper_target(float(fraction)), dtype=np.float64)
        if values.shape != (len(indices),) or not np.isfinite(values).all():
            raise ValueError("gripper target has invalid shape or nonfinite positions")
        self._target[indices] = values

    def _step_once(self) -> None:
        self._exchange()

    def compute_reward(self) -> float:
        return 0.0

    def task_completed(self) -> bool:
        return False

    def close(self) -> None:
        """The runner owns the driver lifetime; connector close is harmless."""
