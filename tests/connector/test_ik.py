"""In-process IK backend parity tests.

PyRoKi: CPU-JAX, default-installed historically; now opt-in via the
``[pyroki]`` extra. Ground truth comes from pyroki's own forward kinematics
of the ``panda_description`` URDF — solving back to a known FK pose must
land within millimeters/centiradians.

CuRobo: the new default connector backend (GPU; v0.8 MotionPlanner). Tests
are gated on the bundle being importable through the synthetic
``gap_skills.tools.curobo`` namespace (set up by ``gap.skills.load_skills``),
which in turn requires CUDA / nvidia-curobo / torch in the runtime venv.
"""

from __future__ import annotations

from pathlib import Path

import numpy as np
import pytest
from gap_core.types import make_pose

from gap.connector import ik as gik

_HOME = [0.0, -0.785, 0.0, -2.356, 0.0, 1.571, 0.785]


def _curobo_available() -> bool:
    """True when cuRobo v0.8 is actually runnable.

    The impl module imports fine even without nvidia-curobo v0.8 (it falls
    back to a no-op stub), so we additionally probe ``_V2_AVAILABLE`` — the
    flag the impl flips on after its v0.8 imports succeed. Any failure
    (no checkout, no CUDA, no nvidia-curobo, v0.7-only install) is treated
    as "not available".
    """
    skills_root = (
        Path(__file__).resolve().parents[3] / "open-robot-skills"
    )
    if not skills_root.exists():
        return False
    try:
        from gap.skills import load_skills

        load_skills(skills_root)
        from gap_skills.tools.curobo import _curobo_impl

        return bool(getattr(_curobo_impl, "_V2_AVAILABLE", False))
    except Exception:
        return False


_CUROBO_SKIP_REASON = (
    "curobo bundle not importable — install with `CUDA_HOME=/usr/local/cuda uv run gap skills install curobo` "
    "(needs CUDA + nvidia-curobo) or skip the cuRobo backend tests."
)


def _pyroki_available() -> bool:
    """PyRoKi moved to the ``[pyroki]`` opt-in extra; tests that drive the
    ``panda_description`` URDF need it. Skip when not installed."""
    try:
        import pyroki  # noqa: F401
    except ImportError:
        return False
    return True


_PYROKI_SKIP_REASON = (
    "pyroki not installed — opt in via `uv sync --extra pyroki` to run the "
    "PyRoKi backend tests."
)


pytestmark_pyroki = pytest.mark.skipif(
    not _pyroki_available(), reason=_PYROKI_SKIP_REASON
)


@pytest.fixture(scope="module")
def robot():
    if not _pyroki_available():
        pytest.skip(_PYROKI_SKIP_REASON)
    return gik.load_robot("panda_description")


def _fk_panda_hand(robot, joints7: list[float]) -> tuple[np.ndarray, np.ndarray]:
    """(position, wxyz) of panda_hand at the given 7-dof arm config."""
    import jax.numpy as jnp

    cfg = np.zeros(robot.joints.num_actuated_joints)
    cfg[:7] = joints7
    Ts = np.asarray(robot.forward_kinematics(jnp.array(cfg)))
    row = Ts[robot.links.names.index("panda_hand")]
    return row[4:7].copy(), row[:4].copy()  # wxyz_xyz layout


def _pose_error(robot, joints7, target_pos, target_wxyz) -> tuple[float, float]:
    pos, wxyz = _fk_panda_hand(robot, joints7)
    pos_err = float(np.linalg.norm(pos - target_pos))
    # angle between quaternions (sign-insensitive)
    dot = abs(float(np.dot(wxyz, target_wxyz)))
    ang_err = 2.0 * float(np.arccos(min(dot, 1.0)))
    return pos_err, ang_err


def test_robot_model_cached(robot):
    assert gik.load_robot("panda_description") is robot
    assert robot.joints.num_actuated_joints == 8  # 7 arm + finger


def test_solve_home_pose_near_zero_error(robot):
    """Solve back to the FK pose of the Franka home config."""
    target_pos, target_wxyz = _fk_panda_hand(robot, _HOME)
    pose = make_pose(target_pos, target_wxyz)

    joints = gik.solve_ik(pose, seed_joints=list(_HOME))
    assert joints is not None and len(joints) == 7

    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3, f"position error {pos_err:.4f} m"
    assert ang_err < 0.05, f"orientation error {ang_err:.4f} rad"
    # Seeded with home, the solution stays on the home branch.
    assert np.linalg.norm(np.asarray(joints) - np.asarray(_HOME)) < 0.2


def test_solve_without_seed_basic_path(robot):
    target_pos, target_wxyz = _fk_panda_hand(robot, _HOME)
    joints = gik.solve_ik(make_pose(target_pos, target_wxyz))
    assert joints is not None and len(joints) == 7
    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3
    assert ang_err < 0.05


def test_fk_solve_roundtrip(robot):
    """FK a perturbed config, solve back, FK again → same pose."""
    perturbed = list(np.asarray(_HOME) + [0.2, -0.1, 0.15, 0.1, -0.2, 0.1, -0.15])
    target_pos, target_wxyz = _fk_panda_hand(robot, perturbed)

    joints = gik.solve_ik(make_pose(target_pos, target_wxyz), seed_joints=list(_HOME))
    assert joints is not None
    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3, f"position error {pos_err:.4f} m"
    assert ang_err < 0.05, f"orientation error {ang_err:.4f} rad"


def test_backend_tcp_offset_shifts_target(robot):
    """A configured TCP offset solves for the link pose shifted along the
    tool axis — verified against FK of the returned joints."""
    offset = np.array([0.0, 0.0, -0.107])
    backend = gik.PyRokiBackend(tcp_offset=offset)
    # Tool-tip target: hand pose displaced by R @ offset (downward-facing).
    hand_pos, hand_wxyz = _fk_panda_hand(robot, _HOME)
    from scipy.spatial.transform import Rotation

    R = Rotation.from_quat(
        [hand_wxyz[1], hand_wxyz[2], hand_wxyz[3], hand_wxyz[0]]
    ).as_matrix()
    tip_pos = hand_pos - R @ offset  # link target == tip + R@offset == hand
    joints = backend.solve_ik(
        make_pose(tip_pos, hand_wxyz), seed_joints=list(_HOME)
    )
    assert joints is not None
    pos_err, ang_err = _pose_error(robot, joints, hand_pos, hand_wxyz)
    assert pos_err < 5e-3
    assert ang_err < 0.05


def test_plan_linear_tracks_endpoints(robot):
    start_pos, start_wxyz = _fk_panda_hand(robot, _HOME)
    end_pos = start_pos + np.array([0.0, 0.0, 0.08])

    backend = gik.PyRokiBackend()
    traj = backend.plan_linear(
        make_pose(start_pos, start_wxyz),
        make_pose(end_pos, start_wxyz),
        num_waypoints=5,
        ik_refinement_iters=3,
    )
    assert traj is not None
    waypoints = traj["waypoints"]
    assert len(waypoints) == 5
    # Endpoint FK error (waypoints are 8-dof backend-native; arm = first 7).
    last = list(np.asarray(waypoints[-1]["positions"])[:7])
    pos_err, ang_err = _pose_error(robot, last, end_pos, start_wxyz)
    assert pos_err < 1e-2
    assert ang_err < 0.1
    # Smoothness: no waypoint-to-waypoint joint jump beyond the source's
    # 0.5 rad threshold.
    arrs = [np.asarray(w["positions"])[:7] for w in waypoints]
    for a, b in zip(arrs, arrs[1:], strict=False):
        assert np.max(np.abs(b - a)) < 0.5


# ---------------------------------------------------------------------------
# CuRobo backend (the new default). Requires GPU + curobo + nvidia-curobo,
# so every test below is gated on _curobo_available().
# ---------------------------------------------------------------------------


pytestmark_curobo = pytest.mark.skipif(
    not _curobo_available(), reason=_CUROBO_SKIP_REASON
)


@pytest.fixture(scope="module")
def curobo_backend():
    """A CuRoboBackend constructed for 7-DOF Franka."""
    if not _curobo_available():
        pytest.skip(_CUROBO_SKIP_REASON)
    return gik.CuRoboBackend(arm_dof=7, home_joints=list(_HOME))


@pytestmark_curobo
def test_curobo_solve_ik_home_pose(robot, curobo_backend):
    """cuRobo's InverseKinematics reaches the Franka home pose from its own seed."""
    target_pos, target_wxyz = _fk_panda_hand(robot, _HOME)
    joints = curobo_backend.solve_ik(
        make_pose(target_pos, target_wxyz), seed_joints=list(_HOME)
    )
    assert joints is not None and len(joints) == 7
    pos_err, ang_err = _pose_error(robot, joints, target_pos, target_wxyz)
    assert pos_err < 5e-3, f"position error {pos_err:.4f} m"
    assert ang_err < 0.05, f"orientation error {ang_err:.4f} rad"


@pytestmark_curobo
def test_curobo_plan_linear_tracks_endpoints(robot, curobo_backend):
    """Short cartesian line: cuRobo returns a trajectory whose endpoint FK
    matches the requested EE pose within the same 1cm/0.1rad envelope."""
    start_pos, start_wxyz = _fk_panda_hand(robot, _HOME)
    end_pos = start_pos + np.array([0.0, 0.0, 0.08])
    traj = curobo_backend.plan_linear(
        make_pose(start_pos, start_wxyz),
        make_pose(end_pos, start_wxyz),
        seed_joints=list(_HOME),
    )
    assert traj is not None
    waypoints = traj["waypoints"]
    assert len(waypoints) >= 2
    last = list(np.asarray(waypoints[-1]["positions"])[:7])
    pos_err, ang_err = _pose_error(robot, last, end_pos, start_wxyz)
    assert pos_err < 1e-2, f"position error {pos_err:.4f} m"
    assert ang_err < 0.1, f"orientation error {ang_err:.4f} rad"


@pytestmark_curobo
def test_curobo_plan_linear_falls_back_to_plan_to_pose(
    robot, curobo_backend, monkeypatch
):
    """When ``plan_linear`` returns success=False, the backend must call
    ``plan_to_pose`` for the endpoint and still return a non-None trajectory
    whose last waypoint lands at the requested pose.

    Forces the failure by monkey-patching ``_curobo_impl.plan_linear`` to
    return ``(False, None, "forced_failure")`` — exercises the fallback path
    deterministically without depending on planner geometry.
    """
    from gap_skills.tools.curobo import _curobo_impl

    start_pos, start_wxyz = _fk_panda_hand(robot, _HOME)
    end_pos = start_pos + np.array([0.0, 0.0, 0.06])

    fallback_called = {"count": 0}
    real_plan_to_pose = _curobo_impl.plan_to_pose

    def stub_plan_linear(**_kwargs):
        return False, None, "forced_failure"

    def counted_plan_to_pose(*args, **kwargs):
        fallback_called["count"] += 1
        return real_plan_to_pose(*args, **kwargs)

    monkeypatch.setattr(_curobo_impl, "plan_linear", stub_plan_linear)
    monkeypatch.setattr(_curobo_impl, "plan_to_pose", counted_plan_to_pose)

    traj = curobo_backend.plan_linear(
        make_pose(start_pos, start_wxyz),
        make_pose(end_pos, start_wxyz),
        seed_joints=list(_HOME),
    )
    assert traj is not None, "linear failed AND fallback failed"
    assert fallback_called["count"] == 1, "plan_to_pose fallback did not fire"
    waypoints = traj["waypoints"]
    assert len(waypoints) >= 1
    last = list(np.asarray(waypoints[-1]["positions"])[:7])
    pos_err, _ang_err = _pose_error(robot, last, end_pos, start_wxyz)
    assert pos_err < 1e-2, f"fallback endpoint error {pos_err:.4f} m"


@pytestmark_curobo
def test_curobo_backend_rejects_six_dof():
    """YAM/6-DOF must surface the explicit "use PyRokiBackend" guidance
    instead of producing a broken backend."""
    with pytest.raises(NotImplementedError, match="ik=PyRokiBackend"):
        gik.CuRoboBackend(arm_dof=6)
