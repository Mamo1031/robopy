"""``RakudaRobot`` end to end on two simulated buses.

Nothing is faked below ``RakudaRobot``: the pair system, both arms, the
leader current loop on its real control thread, the alignment ramp and the
hold all run against ``SimulatedDynamixelBus`` instances on the wall clock,
exactly as ``examples/robot/rakuda_bilateral.py --sim`` runs them.  The
identification file is stood in for by a stub ``rakuda_gravity`` module, so
``ensure_bilateral_running()`` takes the real path (lazy import,
``load_bilateral_setup(gravity_model_path)``) with a setup the test controls;
``test_gravity_end_to_end`` runs the real module from identification on.

The simulated leader arm joints carry a constant gravity current and a
static friction above it, so they hold themselves while torque is off
(connect, preflight) and stay put once the loop commands exactly the gravity
term; the follower starts ``FOLLOWER_OFFSET`` counts away so the alignment
ramp of ``start_bilateral()`` is exercised too.
"""

from __future__ import annotations

import logging
import sys
import time
import types
from typing import Any, Dict, Iterator, List, Sequence, cast

import numpy as np
import pytest
from numpy.typing import NDArray

from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaArmObs,
    RakudaBilateralParams,
    RakudaConfig,
    RakudaObs,
    RakudaSensorParams,
)
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelMotor
from robopy.motor.dynamixel_control_table import OperatingMode, XControlTable
from robopy.motor.sim_dynamixel_bus import SimulatedDynamixelBus
from robopy.robots.rakuda import rakuda_pair_sys
from robopy.robots.rakuda.rakuda_leader_control import LoopState
from robopy.robots.rakuda.rakuda_pair_sys import BilateralSetup, RakudaPairSys
from robopy.robots.rakuda.rakuda_robot import RakudaRobot

from .conftest import make_follower_bus, make_leader_bus

J = RAKUDA_ARM_JOINT_NAMES
J_INDEX = [RAKUDA_JOINT_NAMES.index(name) for name in J]
GRIP_INDEX = [RAKUDA_JOINT_NAMES.index(name) for name in RAKUDA_GRIPPER_JOINT_NAMES]
LEADER_PORT = "sim-leader"
FOLLOWER_PORT = "sim-follower"
PARAMS = RakudaBilateralParams(allow_uncompensated=True)
#: Every simulated joint starts here (``SimulatedJoint`` default).
HOME = 2048
#: The leader plant's holding current on ``J`` and the model that predicts it.
GRAVITY_MA = 20.0
#: What the loop commands: ``gravity_scale`` (0.9) times the prediction, in XC330 raw
#: units (1 mA/LSB), with zero feedback (default gains) and no barrier at home.
EXPECTED_GOAL_CURRENT = 18
#: The leader plant balances exactly that command; its static friction is twice it, so
#: neither gravity alone (torque off) nor the command (torque on) moves a joint.
LEADER_PLANT_GRAVITY_MA = float(EXPECTED_GOAL_CURRENT)
LEADER_PLANT_COULOMB_MA = 2.0 * LEADER_PLANT_GRAVITY_MA
#: Where the follower's arm joints start: the alignment ramp has to close this gap.
FOLLOWER_OFFSET = 100
RANGE = (HOME - 1024, HOME + 1024)


class WallClock:
    """The real monotonic clock in the shape the simulated bus expects."""

    monotonic_ns = staticmethod(time.monotonic_ns)
    sleep = staticmethod(time.sleep)


class ConstantGravity:
    """``GravityModel`` predicting ``GRAVITY_MA`` on ``J`` and zero elsewhere."""

    def predict_ma(self, q_counts: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
        out = np.zeros(len(RAKUDA_JOINT_NAMES), dtype=np.float64)
        out[J_INDEX] = GRAVITY_MA
        return out


class SimBuses:
    """``bus_factory`` of the robot: one simulated bus per port, kept across reconnects."""

    def __init__(self) -> None:
        self.leader = make_leader_bus(clock=WallClock(), auto_step=True, port=LEADER_PORT)
        self.follower = make_follower_bus(clock=WallClock(), auto_step=True, port=FOLLOWER_PORT)
        for name in J:
            joint = self.leader.joint(name)
            joint.gravity_ma = LEADER_PLANT_GRAVITY_MA
            joint.coulomb_ma = LEADER_PLANT_COULOMB_MA
            self.follower.joint(name).position_counts = HOME + FOLLOWER_OFFSET
        for bus in (self.leader, self.follower):
            for name in bus.motors:
                bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
        self.by_port = {LEADER_PORT: self.leader, FOLLOWER_PORT: self.follower}
        self.requests: List[tuple[str, int]] = []

    def __call__(self, port: str, motors: Dict[str, DynamixelMotor]) -> DynamixelBus:
        self.requests.append((port, len(motors)))
        return cast(DynamixelBus, self.by_port[port])


def make_setup() -> BilateralSetup:
    """A complete, validated setup for ``J`` with the constant gravity model."""
    return BilateralSetup(
        current_sign={name: 1 for name in J},
        joint_range_counts={name: RANGE for name in J},
        drive_mode={name: 0 for name in J},
        gravity=ConstantGravity(),
        gravity_validated=True,
        source="e2e-sim",
    )


def make_config() -> RakudaConfig:
    """The task's configuration; no sensors so ``RakudaRobot`` opens no camera."""
    return RakudaConfig(
        leader_port=LEADER_PORT,
        follower_port=FOLLOWER_PORT,
        sensors=RakudaSensorParams(),
        bilateral=PARAMS,
    )


def writes(bus: SimulatedDynamixelBus, item: XControlTable) -> List[Dict[str, int]]:
    """Every host write of ``item``, in order."""
    return [values for name, values in bus.instruction_log if name == item.name]


def assert_leader_joints(bus: SimulatedDynamixelBus, *, mode: int, torque: int) -> None:
    for name in J:
        regs = bus.registers(name)
        assert (regs.operating_mode, regs.torque_enable) == (mode, torque), name


def assert_recording(obs: RakudaObs, frames: int) -> None:
    """Every recorded array is present, shaped for ``frames`` frames and stamped."""
    arms = obs.arms
    for field in RakudaArmObs.ARRAY_FIELDS:
        array = getattr(arms, field)
        assert array is not None, field
        assert array.dtype == np.float32, field
        expected = (frames,) if field.endswith("_time_s") else (frames, len(RAKUDA_JOINT_NAMES))
        assert array.shape == expected, field
    assert arms.leader_t_ns is None and arms.follower_t_ns is None
    frame_t = cast(NDArray[np.float32], arms.frame_time_s)
    leader_t = cast(NDArray[np.float32], arms.leader_time_s)
    follower_t = cast(NDArray[np.float32], arms.follower_time_s)
    assert frame_t[0] >= 0.0 and np.all(np.diff(frame_t) >= 0.0)
    assert np.all(leader_t <= frame_t) and np.all(follower_t <= frame_t)
    # The pair consumed first may predate ``t0`` by at most a few cycles.
    assert leader_t[0] > -0.1 and follower_t[0] > -0.1
    assert np.all(np.diff(leader_t) >= 0.0) and np.all(np.diff(follower_t) >= 0.0)
    # The compensated leader joints never move and its grippers are held by the loop.
    # The follower copies the arm joints to the count; its grippers, which chase the
    # leader's own gripper hold, have caught up by the last frame.
    assert np.all(arms.leader[:, J_INDEX] == HOME)
    assert np.all(arms.leader[:, GRIP_INDEX] == LEADER_GRIP_HOLD_POSITION)
    assert np.all(np.abs(arms.follower[:, J_INDEX] - arms.leader[:, J_INDEX]) <= 2)
    assert np.all(np.abs(arms.follower[-1] - arms.leader[-1]) <= 2)
    # The sim reports the commanded current: the settled gravity term on ``J``, 0 elsewhere.
    last_current = cast(NDArray[np.float32], arms.leader_current)[-1]
    assert last_current[J_INDEX].tolist() == [float(EXPECTED_GOAL_CURRENT)] * len(J)
    assert np.count_nonzero(last_current) == len(J)
    assert obs.sensors is not None
    assert (obs.sensors.cameras, obs.sensors.tactile, obs.sensors.audio) == ({}, {}, {})


@pytest.fixture
def buses(monkeypatch: pytest.MonkeyPatch) -> SimBuses:
    """The two buses, the dotconfig kept out, and the stub identification module installed."""
    monkeypatch.setattr(rakuda_pair_sys, "apply_rakuda_dotconfig", lambda cfg: cfg)
    buses = SimBuses()
    gravity_module = types.ModuleType(rakuda_pair_sys._GRAVITY_MODULE)
    loaded: List[str] = []

    def load_bilateral_setup(path: str) -> BilateralSetup:
        loaded.append(str(path))
        return make_setup()

    setattr(gravity_module, "load_bilateral_setup", load_bilateral_setup)
    monkeypatch.setitem(sys.modules, rakuda_pair_sys._GRAVITY_MODULE, gravity_module)
    setattr(buses, "loaded_paths", loaded)
    return buses


@pytest.fixture
def robots() -> Iterator[List[RakudaRobot]]:
    """Every robot a test builds; the loops are stopped at teardown whatever happened."""
    made: List[RakudaRobot] = []
    yield made
    for robot in made:
        pair = robot.robot_system
        RakudaPairSys.stop_bilateral(pair)  # never leave a control thread behind
        RakudaPairSys._unregister_atexit(pair)
        try:
            robot.disconnect()
        except Exception:
            pass


def make_robot(buses: SimBuses, robots: List[RakudaRobot]) -> RakudaRobot:
    robot = RakudaRobot(make_config(), bus_factory=buses)
    robots.append(robot)
    return robot


def test_record_hold_and_reconnect(
    buses: SimBuses, robots: List[RakudaRobot], caplog: pytest.LogCaptureFixture
) -> None:
    caplog.set_level(logging.INFO)
    loaded_paths: Sequence[str] = getattr(buses, "loaded_paths")

    # -- construction and connect: state A on the leader's J, follower all on ----
    robot = make_robot(buses, robots)
    assert buses.requests == [
        (LEADER_PORT, len(RAKUDA_JOINT_NAMES)),
        (FOLLOWER_PORT, len(RAKUDA_JOINT_NAMES)),
    ]
    robot.connect()
    assert robot.is_connected
    assert_leader_joints(buses.leader, mode=OperatingMode.POSITION, torque=0)
    assert all(buses.follower.registers(n).present_position == HOME + FOLLOWER_OFFSET for n in J)
    assert all(buses.leader.registers(n).torque_enable == 1 for n in RAKUDA_GRIPPER_JOINT_NAMES)
    assert all(buses.follower.registers(n).torque_enable == 1 for n in RAKUDA_JOINT_NAMES)
    assert not robot.robot_system.bilateral_active and loaded_paths == []

    # -- record_parallel starts the loop through the identification file ---------
    obs = robot.record_parallel(max_frame=20, fps=10)
    assert loaded_paths == [PARAMS.gravity_model_path]
    assert_recording(obs, 20)
    pair = robot.robot_system
    loop = pair._leader_loop
    assert loop is not None and loop.running and pair.bilateral_active
    # The control thread owns both buses: J in current mode at the gravity term,
    # the follower copying the leader in position mode.
    assert_leader_joints(buses.leader, mode=OperatingMode.CURRENT, torque=1)
    assert all(buses.leader.registers(n).goal_current == EXPECTED_GOAL_CURRENT for n in J)
    assert_leader_joints(buses.follower, mode=OperatingMode.POSITION, torque=1)
    # The alignment ramp closed the gap before the loop started copying positions.
    assert all(buses.follower.registers(n).goal_position == HOME for n in J)
    assert all(abs(buses.follower.registers(n).present_position - HOME) <= 2 for n in J)

    report = robot.control_report()
    assert report["mode"] == "leader_current"
    assert report["state"] == LoopState.RUNNING.value
    assert report["current_sign"] == {name: 1 for name in J}
    assert report["joints"] == list(J)
    assert report["setup"] == {
        "source": "e2e-sim",
        "gravity_validated": True,
        "uncompensated": False,
    }
    assert report["faults"] == [] and report["hold"] is None
    assert report["preflight"]["stale_current_mode_recovered"] == []
    assert report["record"]["frames"] == 20
    assert report["record"]["frames_requested"] == 20
    assert report["record"]["terminated_by"] == "max_frame"
    assert report["record"]["worker_error"] is None
    assert report["sampler"]["frames"] == 20
    assert report["teleop_hz_effective"] > 0.0
    assert report["measured"]["cycles"] > 0 and report["measured"]["read_fail_count"] == 0
    assert report["follower_io"] == {
        **report["follower_io"],
        "attached": True,
        "failures": 0,
        "lost": False,
    }
    assert report["ports"] == {"leader": LEADER_PORT, "follower": FOLLOWER_PORT}
    assert report["motors"]["names"] == list(RAKUDA_JOINT_NAMES)

    # -- disconnect holds both arms and keeps torque (hold_on_disconnect) --------
    buses.leader.instruction_log.clear()
    buses.follower.instruction_log.clear()
    robot.disconnect()
    assert loop.state is LoopState.HELD and not loop.thread_alive and loop.hold_verified
    assert not pair.bilateral_active
    assert_leader_joints(buses.leader, mode=OperatingMode.POSITION, torque=1)
    for name in J:
        regs = buses.leader.registers(name)
        assert abs(regs.goal_position - regs.present_position) <= 5, name
    assert all(buses.leader.registers(n).torque_enable == 1 for n in RAKUDA_GRIPPER_JOINT_NAMES)
    assert all(buses.follower.registers(n).torque_enable == 1 for n in RAKUDA_JOINT_NAMES)
    # The hold's own off/on on J is the last leader TORQUE_ENABLE traffic; the
    # follower is not written at all.
    assert writes(buses.leader, XControlTable.TORQUE_ENABLE)[-1] == {name: 1 for name in J}
    assert writes(buses.follower, XControlTable.TORQUE_ENABLE) == []
    assert not buses.leader.port_handler.is_open and not buses.follower.port_handler.is_open
    held = robot.control_report()
    assert held["mode"] == "leader_current" and held["state"] == LoopState.HELD.value
    assert held["hold"]["verified"] is True and held["hold"]["none"] == []
    assert held["faults"] == []

    # -- reconnect over the held arms (state B) and record again ------------------
    again = make_robot(buses, robots)
    buses.leader.instruction_log.clear()
    buses.follower.instruction_log.clear()
    again.connect()
    assert again.is_connected
    # Held joints are left as found: no TORQUE_ENABLE or mode write reaches J.
    assert_leader_joints(buses.leader, mode=OperatingMode.POSITION, torque=1)
    for values in writes(buses.leader, XControlTable.TORQUE_ENABLE):
        assert not set(values) & set(J), values
    assert writes(buses.leader, XControlTable.OPERATING_MODE) == []
    assert writes(buses.follower, XControlTable.TORQUE_ENABLE) == []

    obs2 = again.record_parallel(max_frame=5, fps=10)
    assert loaded_paths == [PARAMS.gravity_model_path] * 2
    assert_recording(obs2, 5)
    loop2 = again.robot_system._leader_loop
    assert loop2 is not None and loop2 is not loop and loop2.running
    report2 = again.control_report()
    assert report2["mode"] == "leader_current"
    assert report2["state"] == LoopState.RUNNING.value
    assert report2["preflight"]["stale_current_mode_recovered"] == []
    assert report2["record"]["frames"] == 5 and report2["faults"] == []

    again.disconnect()
    assert loop2.state is LoopState.HELD and loop2.hold_verified
    assert_leader_joints(buses.leader, mode=OperatingMode.POSITION, torque=1)
    assert all(buses.follower.registers(n).torque_enable == 1 for n in RAKUDA_JOINT_NAMES)

    # The whole life cycle ran clean: no warning, error or fault anywhere.
    assert [r.getMessage() for r in caplog.records if r.levelno >= logging.WARNING] == []
