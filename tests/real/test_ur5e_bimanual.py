"""Hardware-free tests for the dionysus bimanual UR5e connector.

A ``FakeRig`` thread plays the robots_realtime client: it connects to the env's
msgpack server, sends both arms' joints (which track the received targets at a
capped speed, like the real driver), four small RGB-D cameras and the rig
description, and records every action it receives.
"""

from __future__ import annotations

import math
import socket
import struct
import threading
import time

import msgpack
import msgpack_numpy
import numpy as np
import pytest

from gap.connector import ur_ik
from gap.connector.ur_ik import URAnalyticIK, load_rig_arms, matrix_to_pose, pose_to_matrix
from gap.envs import ur5e_bimanual_real_env as envmod

from .conftest import free_port

msgpack_numpy.patch()

ARMS, WIRE = load_rig_arms()
NAMES = {0: "right", 1: "left"}


# ── IK ────────────────────────────────────────────────────────────────────────


def test_ik_roundtrip_and_nearest_seed():
    rng = np.random.default_rng(3)
    ik = URAnalyticIK(ARMS)
    for _ in range(200):
        q = ARMS[0].home_q + rng.uniform(-0.6, 0.6, 6)
        pose = matrix_to_pose(ARMS[0].tcp_world(q))
        sol = ik.solve_ik(pose, 0, seed_joints=q + 0.01)
        assert sol is not None
        assert np.abs(np.array(sol) - q).max() < 1e-6  # nearest solution is the generating one
    far = matrix_to_pose(np.eye(4) @ np.diag([1, 1, 1, 1]))
    far["position"] = {"x": 5.0, "y": 5.0, "z": 5.0}
    assert ik.solve_ik(far, 0) is None


def test_axis_line_keeps_the_starting_roll():
    """A vertical descent in "axis" mode keeps the hand's heading (it walked 42 degrees before 2026-10-20)."""
    ik = URAnalyticIK(ARMS)
    arm = ARMS[1]
    start = np.eye(4)
    yaw = np.radians(-75.0)
    # Top-down (tool z = world -z) over the mat, the jaws at -75 degrees: the failed real run's hover.
    x = np.array([np.cos(yaw), np.sin(yaw), 0.0])
    z = np.array([0.0, 0.0, -1.0])
    start[:3, :3] = np.column_stack([x, np.cross(z, x), z])
    start[:3, 3] = [0.009, -0.0855, 0.1022]
    q0, perr, _ = ik.solve(start, 1, arm.home_q, hold_rotation=True)
    assert q0 is not None and perr < 1e-6
    seed, here = q0, start
    yaws = []
    for z_end in (0.0422, -0.0061):
        end = here.copy()
        end[:3, 3] = [here[0, 3], here[1, 3], z_end]
        plan = ik.plan_linear(matrix_to_pose(here), matrix_to_pose(end), arm_id=1, seed_joints=seed, num_waypoints=8)
        assert plan is not None, ik.last_refusal
        for wp in plan["waypoints"]:
            T = arm.tcp_world(np.asarray(wp["positions"]))
            yaws.append(np.degrees(np.arctan2(T[1, 0], T[0, 0])))
        seed = np.asarray(plan["waypoints"][-1]["positions"])
        here = arm.tcp_world(seed)
    assert max(abs(y + 75.0) for y in yaws) < 0.5, yaws


def test_axis_mode_keeps_the_approach_axis():
    ik = URAnalyticIK(ARMS)
    T = ARMS[1].tcp_world(ARMS[1].home_q)
    T2 = T.copy()
    T2[:3, :3] = T[:3, :3] @ np.array([[0, -1, 0], [1, 0, 0], [0, 0, 1.0]])  # roll 90 deg about tool z
    q, perr, rerr = ik.solve(T2, 1, ARMS[1].home_q, hold_rotation=False)
    assert q is not None and perr < 1e-6 and rerr < 1e-6
    # The free roll lets the wrist stay near the seed instead of turning 90 degrees.
    assert abs(q[5] - ARMS[1].home_q[5]) < 0.2


def test_trapezoid_respects_limits():
    path = np.array([np.zeros(6), np.r_[0.5, 0, 0, 0, 0, 0], np.r_[0.5, -0.3, 0.2, 0, 0, 0]])
    dt, vmax, amax = 0.02, 0.2, 0.6
    s = envmod.trapezoid_samples(path, vmax, amax, dt)
    assert np.allclose(s[-1], path[-1])
    v = np.diff(np.vstack([path[:1], s]), axis=0) / dt
    assert np.abs(v).max() <= vmax + 1e-6
    a = np.diff(v, axis=0) / dt
    assert np.percentile(np.abs(a), 95) <= amax * 1.05 + 1e-6


# ── fake rig over the real wire protocol ─────────────────────────────────────


class FakeRig(threading.Thread):
    def __init__(self, port: int, vmax: float = 1.0):
        super().__init__(daemon=True)
        self.port = port
        self.vmax = vmax
        self.q = {n: ARMS[i].home_q.copy() for i, n in NAMES.items()}
        self.grip = {n: 1.0 for n in NAMES.values()}
        self.actions: list[dict] = []
        self.send = True
        self.stop_reason = None
        self.ack = None
        self.auto_ack = None  # True/False -> answer approval requests
        self.floor_z = None  # world z of a stiff floor the TCP presses on: the wrist force rises below it
        self.floor_k = 5000.0  # N/m
        self.send_force = True
        self._halt = threading.Event()
        self.start()

    def _obs(self) -> dict:
        # Like the real client: the rig description about once a second, and an
        # approval answer exactly once (the env must catch both on any frame).
        o: dict = {"timestamp": time.time()}
        if time.time() - getattr(self, "_rig_t", 0.0) > 1.0:
            o["rig"] = WIRE
            self._rig_t = time.time()
        for i, n in NAMES.items():
            T = ARMS[i].tcp_world(self.q[n])
            o[n] = {"joint_pos": np.r_[self.q[n], self.grip[n]].astype(np.float32),
                    "tcp_world": np.r_[T[:3, 3], ur_ik.matrix_to_quat_wxyz(T[:3, :3])].astype(np.float32),
                    "setpoint": self.q[n].astype(np.float32), "driving": 1, "latched": 0,
                    "safety_mode": 1, "runtime_state": 2}
            if self.send_force:  # base frame, like RTDE's actual_TCP_force, with a 2 N offset as on the rig
                fz = self.floor_k * max(0.0, self.floor_z - T[2, 3]) if self.floor_z is not None else 0.0
                f = ARMS[i].world_T_base[:3, :3].T @ np.array([0.0, 0.0, fz]) + np.array([1.5, -2.0, 0.3])
                o[n]["tcp_force"] = np.r_[f, 0.0, 0.0, 0.0].astype(np.float32)
        K = np.array([[50.0, 0, 16], [0, 50.0, 12], [0, 0, 1]], dtype=np.float32)
        for cam in ("top", "bottom", "right_wrist", "left_wrist"):
            pose = np.eye(4)
            pose[:3, :3] = np.diag([1.0, -1.0, -1.0])  # looking straight down
            pose[:3, 3] = [0.0, 0.3, 0.8]
            o[cam] = {"images": {"rgb": np.full((24, 32, 3), 100, np.uint8)},
                      "depth_data": np.full((24, 32), 0.8, np.float32),
                      "intrinsics": {"left": {"intrinsics_matrix": K}},
                      "pose": np.r_[pose[:3, 3], ur_ik.matrix_to_quat_wxyz(pose[:3, :3])].astype(np.float32),
                      "timestamp": time.time()}
        if self.stop_reason:
            o["stop"] = {"reason": self.stop_reason}
        if self.ack is not None:
            o["viz_ack"] = self.ack
            self.sent_acks = getattr(self, "sent_acks", set()) | {self.ack["id"]}
            self.ack = None
        return o

    def run(self):
        deadline = time.time() + 10
        while True:
            try:
                sock = socket.create_connection(("127.0.0.1", self.port), timeout=5)
                break
            except OSError:
                if time.time() > deadline:
                    return
                time.sleep(0.05)

        def rx(n):
            b = b""
            while len(b) < n:
                c = sock.recv(n - len(b))
                if not c:
                    raise ConnectionError
                b += c
            return b

        dt = 0.01
        try:
            while not self._halt.is_set():
                if not self.send:
                    time.sleep(dt)
                    continue
                payload = msgpack.packb(self._obs(), use_bin_type=True)
                sock.sendall(struct.pack("!I", len(payload)) + payload)
                (n,) = struct.unpack("!I", rx(4))
                act = msgpack.unpackb(rx(n), raw=True)
                self.actions.append(act)
                viz = act.get(b"viz") or {}
                wait = viz.get(b"awaiting_approval")
                if wait and self.auto_ack is not None and int(wait[b"id"]) not in getattr(self, "sent_acks", set()):
                    self.ack = {"id": int(wait[b"id"]), "approved": bool(self.auto_ack)}
                for n_ in NAMES.values():
                    a = act.get(n_.encode())
                    if a is None:
                        continue
                    tgt = np.asarray(a[b"joint_pos"], dtype=np.float64)
                    step = np.clip(tgt - self.q[n_], -self.vmax * dt, self.vmax * dt)
                    self.q[n_] = self.q[n_] + step
                    self.grip[n_] = float(np.clip(a[b"gripper"], 0.0, 1.0)) * 0.6 + self.grip[n_] * 0.4
                time.sleep(dt)
        except (ConnectionError, OSError):
            pass
        finally:
            sock.close()

    def halt(self):
        self._halt.set()


@pytest.fixture
def rig(tmp_path, monkeypatch):
    monkeypatch.setattr(envmod, "LATCH_PATH", tmp_path / "latch")
    monkeypatch.setattr(envmod, "_rest_stop", lambda ip: "faked")

    def build(**kw):
        from gap.connector.ur5e_bimanual import UR5eBimanualConnector

        port = free_port()
        kw.setdefault("dry_run", False)
        kw.setdefault("confirm", False)
        kw.setdefault("key_stop", False)
        kw.setdefault("max_joint_vel", 0.8)
        env, config = envmod.make_env(port=port, **kw)
        fake = FakeRig(port)
        wire = env.wait_for_rig(timeout_s=10)
        conn = UR5eBimanualConnector(env, config, rig_wire=wire)
        conn.wait_ready(timeout_s=10)
        built.append((conn, fake))
        return conn, fake

    built: list = []
    yield build
    for conn, fake in built:
        fake.halt()
        conn.close()


def test_observation_contract(rig):
    conn, _ = rig()
    obs = conn.get_observation()
    assert len(obs["arms"]) == 2 and len(obs["cameras"]) == 4
    cam = obs["cameras"][0]
    assert cam["rgb"].dtype == np.uint8 and cam["rgb"].shape[-1] == 3
    assert cam["depth"].dtype == np.float32 and cam["depth"].shape == cam["rgb"].shape[:2]
    assert cam["intrinsics"].shape == (3, 3)
    assert set(cam["pose"]["rotation"]) == {"w", "x", "y", "z"}
    ee = conn.get_ee_pose(0)
    T = ARMS[0].tcp_world(ARMS[0].home_q)
    assert abs(ee["position"]["z"] - T[2, 3]) < 1e-4  # world frame, not base frame


def test_registry_covers_the_example_graph(rig):
    conn, _ = rig()
    reg = conn.tool_registry
    used = ["robot.get_observation", "robot.go_home", "robot.open_gripper", "robot.close_gripper",
            "robot.get_gripper", "robot.get_ee_pose", "robot.execute_trajectory", "robot.go_to_pose",
            "robot.go_to_pose_cartesian", "robot.describe_gripper", "robot.describe_workspace",
            "robot.grasp_frame", "robot.wait_steps", "motion.plan_joint", "motion.plan_linear",
            "geometry.mask_to_world_points"]
    assert [n for n in used if n not in reg] == []


def test_plan_and_execute_descend(rig):
    conn, fake = rig()
    T = conn.ik.arm(0).tcp_world(fake.q["right"])
    T[2, 3] -= 0.10
    plan = conn.plan_linear(end=matrix_to_pose(T), orientation="lock", tolerance=0.004, num_waypoints=8)
    assert plan["planned"], plan["reason"]
    res = conn._tool_execute_trajectory(plan["trajectory"], arm_id=0)
    assert res["final_tracking_error_rad"] < 0.01
    got = conn.ik.arm(0).tcp_world(fake.q["right"])
    assert np.linalg.norm(got[:3, 3] - T[:3, 3]) < 0.01
    assert np.allclose(fake.q["left"], ARMS[1].home_q, atol=1e-6)  # the other arm held


def test_force_guard_stops_on_contact_and_holds(rig):
    # r2s2rn's gasket press, 2026-10-24: a press stalled on a bolt boss "converged" within the settle tolerance
    # and pushed on until the arm protective-stopped
    conn, fake = rig()
    T = conn.ik.arm(0).tcp_world(fake.q["right"])
    floor = T[2, 3] - 0.02
    fake.floor_z = floor
    T[2, 3] -= 0.05
    plan = conn.plan_linear(end=matrix_to_pose(T), orientation="lock", tolerance=0.004, num_waypoints=10)
    assert plan["planned"], plan["reason"]
    res = conn._tool_execute_trajectory(plan["trajectory"], arm_id=0, stop_force_n=10.0, speed_scale=0.2)
    assert res["stopped_on_force"] and res["force_n"] > 10.0 and res["progress"] < 1.0
    z = conn.ik.arm(0).tcp_world(fake.q["right"])[2, 3]
    assert floor - 0.006 < z < floor  # 2 mm deep at 10 N, plus what the 50 Hz check lets through
    time.sleep(0.3)
    assert abs(conn.ik.arm(0).tcp_world(fake.q["right"])[2, 3] - z) < 0.001  # and it stays there
    w = conn._tool_get_wrench(arm_id=0)
    assert w["available"] and w["force"][2] > 10.0  # the floor pushes up, in the world frame


def test_force_guard_needs_the_force(rig):
    conn, fake = rig()
    fake.send_force = False
    T = conn.ik.arm(0).tcp_world(fake.q["right"])
    T[2, 3] -= 0.02
    plan = conn.plan_linear(end=matrix_to_pose(T), orientation="lock", tolerance=0.004, num_waypoints=4)
    with pytest.raises(RuntimeError, match="force-guarded move is refused"):
        conn._tool_execute_trajectory(plan["trajectory"], arm_id=0, stop_force_n=10.0)
    assert np.allclose(fake.q["right"], ARMS[0].home_q, atol=1e-6)  # nothing moved
    res = conn._tool_execute_trajectory(plan["trajectory"], arm_id=0)  # unguarded moves are unchanged
    assert "stopped_on_force" not in res


def test_plan_joint_axis_holds_the_roll_as_the_twin_does(rig):
    # run004 of the gasket step, 2026-10-22: a pregrasp turned 65 deg about the vertical came back 60 deg short
    # with rotation_error 0, and the locked descent after it was refused for a 1.04 rad wrist jump.
    conn, fake = rig()
    T = conn.ik.arm(0).tcp_world(fake.q["right"])
    turned = T.copy()
    c, s = math.cos(math.radians(65.0)), math.sin(math.radians(65.0))
    turned[:3, :3] = np.array([[c, -s, 0.0], [s, c, 0.0], [0.0, 0.0, 1.0]]) @ T[:3, :3]
    turned[2, 3] -= 0.05
    plan = conn.plan_joint(matrix_to_pose(turned), arm_id=0)
    assert plan["planned"], plan["reason"]
    conn._tool_execute_trajectory(plan["trajectory"], arm_id=0)
    got = conn.ik.arm(0).tcp_world(fake.q["right"])
    assert ur_ik.rotation_angle(got, turned) < 1e-3
    down = got.copy()
    down[2, 3] -= 0.05
    assert conn.plan_linear(end=matrix_to_pose(down), orientation="lock", num_waypoints=8)["planned"]


def test_gate_refuses_table_and_other_arm(rig):
    conn, fake = rig()
    T = conn.ik.arm(0).tcp_world(fake.q["right"])
    below = T.copy()
    below[2, 3] = -0.05
    plan = conn.plan_joint(matrix_to_pose(below), arm_id=0, orientation="lock")
    assert not plan["planned"] and "safety gate" in plan["reason"]
    # Into the left arm's space.
    Tl = conn.ik.arm(1).tcp_world(fake.q["left"])
    near = T.copy()
    near[:3, 3] = Tl[:3, 3] + np.array([0.0, 0.05, 0.0])
    plan = conn.plan_joint(matrix_to_pose(near), arm_id=0, orientation="lock")
    assert not plan["planned"]
    q_below, _, _ = conn.ik.solve(below, 0, fake.q["right"], True)
    assert q_below is not None  # reachable — only the gate stands in the way
    with pytest.raises(Exception, match="safety gate"):
        conn.move_to_joints(list(q_below), arm_id=0)
    assert np.allclose(fake.q["right"], ARMS[0].home_q, atol=1e-6)


def test_go_home_and_gripper(rig):
    conn, fake = rig()
    q_away = ARMS[0].home_q + np.array([0.15, 0.0, 0.0, 0.0, 0.0, 0.1])
    conn.move_to_joints(list(q_away), arm_id=0)
    assert np.abs(fake.q["right"] - q_away).max() < 0.01
    conn.go_home(arm_id=0)
    assert np.abs(fake.q["right"] - ARMS[0].home_q).max() < 0.01
    conn.close_gripper(settle_steps=2, arm_id=0)
    assert conn.get_gripper_fraction(0) < 0.2


def test_the_twins_set_grip_and_forward_kinematics(rig):
    conn, fake = rig()
    assert all(n in conn.tool_registry for n in ("robot.set_grip", "robot.forward_kinematics"))
    out = conn.set_grip(width_m=0.036, arm_id=1)
    assert abs(out["command"] - (1.0 - 0.036 / 0.085)) < 1e-9
    assert abs(conn.get_gripper_fraction(1) - 0.036 / 0.085) < 0.05
    with pytest.raises(ValueError):
        conn.set_grip(width_m=0.03, preset="open")
    q = ARMS[1].home_q.copy()
    fk = conn.forward_kinematics(joint_configs=[q.tolist(), (q + 0.1).tolist()], arm_id=1)
    T = ARMS[1].tcp_world(q)
    assert abs(fk["poses"][0]["position"]["z"] - T[2, 3]) < 1e-9 and len(fk["poses"]) == 2
    assert np.allclose(fk["joint_config"], fake.q["left"], atol=1e-6)  # nothing moved


def test_grasp_frame_axes(rig):
    conn, _ = rig()
    f = conn.grasp_frame(approach={"x": 0, "y": 0, "z": -1}, close_heading_deg=30.0)
    R = pose_to_matrix({"position": {"x": 0, "y": 0, "z": 0}, "rotation": f["rotation"]})[:3, :3]
    assert np.allclose(R[:, 2], [0, 0, -1], atol=1e-9)  # tool z = approach
    assert np.allclose(R[:, 0], [math.cos(math.radians(30)), math.sin(math.radians(30)), 0], atol=1e-9)
    Rf = pose_to_matrix({"position": {"x": 0, "y": 0, "z": 0}, "rotation": f["rotation_flipped"]})[:3, :3]
    assert np.allclose(Rf[:, 0], -R[:, 0], atol=1e-9)
    g = conn.describe_gripper()
    assert abs(g["max_grasp_width_m"] - 0.077) < 1e-9 and g["finger"]["stated"]
    w = conn.describe_workspace()
    assert w["surface_z"] == 0.0 and w["transport_z"] > 0.3


def test_mask_to_world_points(rig):
    conn, _ = rig()
    cam = conn.get_observation()["cameras"][0]
    mask = np.zeros(cam["depth"].shape, np.uint8)
    mask[10:14, 10:20] = 255
    pts = conn.mask_to_world_points(mask, cam["depth"], cam["intrinsics"], cam["pose"])["points"]["points"]
    assert pts.shape == (40, 3)
    assert np.allclose(pts[:, 2], 0.0, atol=1e-5)  # camera 0.8 m above, depth 0.8 m -> table


def test_latch_stops_midway(rig):
    conn, fake = rig()
    q_far = ARMS[0].home_q + np.array([0.6, 0.0, 0.0, 0.0, 0.0, 0.0])

    def latch_soon():
        time.sleep(0.4)
        envmod.LATCH_PATH.write_text('{"reason": "pytest"}')

    threading.Thread(target=latch_soon, daemon=True).start()
    with pytest.raises(envmod.RigStopped):
        conn.move_to_joints(list(q_far), arm_id=0)
    n = len(fake.actions)
    time.sleep(0.3)
    later = fake.actions[n + 3:]  # skip replies already in flight when the stop landed
    assert later and all(b"right" not in a for a in later)  # nothing commanded after the stop
    assert np.abs(fake.q["right"] - q_far).max() > 0.1


def test_client_stop_and_stale_stream(rig):
    conn, fake = rig()
    q_far = ARMS[1].home_q + np.array([0.0, 0.0, 0.0, 0.0, 0.0, 0.8])

    def go_stale():
        time.sleep(0.3)
        fake.send = False

    threading.Thread(target=go_stale, daemon=True).start()
    with pytest.raises(envmod.RigStopped):
        conn.move_to_joints(list(q_far), arm_id=1)


def test_dry_run_sends_nothing(rig):
    conn, fake = rig(dry_run=True)
    q_away = ARMS[0].home_q + np.array([0.2, 0.0, 0.0, 0.0, 0.0, 0.0])
    conn.move_to_joints(list(q_away), arm_id=0)
    assert np.allclose(conn._q(0), q_away)  # shadow state advanced
    assert np.allclose(fake.q["right"], ARMS[0].home_q)  # real arm never moved
    assert all(b"right" not in a and b"left" not in a for a in fake.actions)


def test_approval(rig):
    conn, fake = rig(confirm=True)
    q_away = ARMS[0].home_q + np.array([0.1, 0.0, 0.0, 0.0, 0.0, 0.0])
    fake.auto_ack = False
    with pytest.raises(Exception, match="rejected"):
        conn.move_to_joints(list(q_away), arm_id=0)
    assert np.allclose(fake.q["right"], ARMS[0].home_q, atol=1e-6)
    fake.auto_ack = True
    fake.ack = None
    conn.move_to_joints(list(q_away), arm_id=0)
    assert np.abs(fake.q["right"] - q_away).max() < 0.01


def test_smoke_graph_end_to_end(rig, tmp_path):
    from gap.runtime.execute import execute

    conn, fake = rig()
    result = execute("examples/ur5e_bimanual_smoke/graph", conn, skills=["../open-robot-skills"],
                     trace_dir=str(tmp_path / "trace"))
    assert result.success, result.exit_status
    for i, n in NAMES.items():
        assert np.abs(fake.q[n] - ARMS[i].home_q).max() < 0.01  # back home after the dip
