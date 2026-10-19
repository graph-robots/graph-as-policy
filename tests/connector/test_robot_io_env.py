"""Exercise the paper-release connector over a simulator-independent I/O seam."""

from __future__ import annotations

import numpy as np
import pytest

from gap.connector.core import Connector
from gap.connector.robot_io_env import ArmLayout, RobotIOEnv
from gap.robot_io import CameraSample, JointPositionCommand, RobotSample


class FakeRobotIO:
    joint_names = ("left_0", "left_grip", "right_0")
    control_period_s = 0.02

    def __init__(self) -> None:
        self.observations = 0
        self.sequence = 0
        self.positions = np.array([0.1, 0.04, -0.2])
        self.commands: list[np.ndarray] = []
        self.stale = False

    def observe(self) -> RobotSample:
        self.observations += 1
        camera = CameraSample(
            rgb=np.zeros((2, 3, 3), dtype=np.uint8),
            depth_m=np.ones((2, 3), dtype=np.float32),
            intrinsics=np.eye(3),
            pose=np.eye(4),
            timestamp_s=10.0,
        )
        return RobotSample(
            sequence=self.sequence,
            timestamp_s=10.0 + self.sequence * self.control_period_s,
            joint_positions=self.positions.copy(),
            joint_velocities=np.array([0.01, 0.0, -0.02]),
            cameras={"front": camera},
        )

    def exchange(self, command: JointPositionCommand) -> RobotSample:
        self.commands.append(np.asarray(command.positions).copy())
        self.positions = np.asarray(command.positions).copy()
        if not self.stale:
            self.sequence += 1
        return self.observe()


def _env(io: FakeRobotIO) -> RobotIOEnv:
    return RobotIOEnv(
        io,
        (
            ArmLayout(
                ("left_0",),
                ("left_grip",),
                gripper_fraction=lambda q: float(q[0] / 0.04),
                gripper_target=lambda fraction: np.array([0.04 * fraction]),
            ),
            ArmLayout(("right_0",)),
        ),
        fk=lambda arm_id, joints: np.array(
            [[1, 0, 0, float(joints[0])], [0, 1, 0, float(arm_id)], [0, 0, 1, 0], [0, 0, 0, 1]],
            dtype=np.float64,
        ),
    )


def test_joint_and_camera_observation_uses_measured_state_and_pure_fk() -> None:
    env = _env(FakeRobotIO())
    obs = env.get_observation()
    np.testing.assert_allclose(obs["robot_joint_pos_0"], [0.1, 1.0])
    np.testing.assert_allclose(obs["robot_joint_vel_1"], [-0.02])
    np.testing.assert_allclose(obs["robot_cartesian_pos_1"][:3], [-0.2, 1.0, 0.0])
    np.testing.assert_array_equal(obs["front"]["images"]["rgb"], np.zeros((2, 3, 3), np.uint8))
    assert obs["front"]["timestamp"] == 10.0


def test_initial_observation_is_deferred_until_the_driver_is_running() -> None:
    io = FakeRobotIO()
    env = _env(io)
    assert io.observations == 0
    env.get_observation()
    assert io.observations == 1


def test_commands_are_atomic_and_preserve_other_arm_and_gripper_targets() -> None:
    io = FakeRobotIO()
    env = _env(io)
    env._set_gripper(0.5)
    env.move_to_joints_blocking([0.3], max_steps=0, arm_id=0)
    np.testing.assert_allclose(io.commands[-1], [0.3, 0.02, -0.2])
    env.move_to_joints_blocking([0.7], max_steps=0, arm_id=1)
    np.testing.assert_allclose(io.commands[-1], [0.3, 0.02, 0.7])
    assert env._sim_step_count == 2


def test_command_requires_a_new_joint_sample() -> None:
    io = FakeRobotIO()
    env = _env(io)
    io.stale = True
    with pytest.raises(RuntimeError, match="fresh joint sample"):
        env.move_to_joints_blocking([0.3], max_steps=0)


def test_existing_connector_solves_pose_above_robot_io() -> None:
    class PureIK:
        def solve_ik(self, pose, *, arm_id=0, seed_joints=None, tcp_offset=None):
            return [0.35]

    class Config:
        arm_dof = 1
        num_arms = 2
        action_mode = "absolute_joints"
        control_freq = 50.0
        home_joints = [0.0]
        default_cameras = ("front",)
        is_real = True

    io = FakeRobotIO()
    connector = Connector(_env(io), Config(), ik=PureIK())
    assert connector.get_observation()["cameras"][0]["name"] == "front"
    connector.go_to_pose(
        {"position": {"x": 0.35, "y": 0.0, "z": 0.0},
         "rotation": {"w": 1.0, "x": 0.0, "y": 0.0, "z": 0.0}},
        max_steps=3,
    )
    assert io.commands
    np.testing.assert_allclose(io.commands[-1], [0.35, 0.04, -0.2])
