"""``connect()``/``disconnect()`` of the Rakuda arms on simulated buses (spec D13/D14/D15/D25/D35).

Every test builds a ``RakudaLeader``/``RakudaFollower`` over a
``SimulatedDynamixelBus`` injected through ``bus_factory``; the bus's
``instruction_log`` gives the exact order of host writes.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any, Dict, List, Sequence, Tuple, cast
from unittest.mock import MagicMock, call

import dynamixel_sdk as dxl
import pytest

from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_CONTROLTABLE_VALUES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_HEAD_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaBilateralParams,
    RakudaConfig,
)
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelCommError
from robopy.motor.dynamixel_control_table import OperatingMode, XControlTable
from robopy.motor.sim_dynamixel_bus import SimulatedDynamixelBus
from robopy.robots.rakuda import rakuda_ports
from robopy.robots.rakuda.rakuda_arm import BusFactory, ConnectState, RakudaArm
from robopy.robots.rakuda.rakuda_follower import RakudaFollower
from robopy.robots.rakuda.rakuda_leader import RakudaLeader
from robopy.robots.rakuda.rakuda_pair_sys import RakudaPairSys

from .conftest import make_follower_bus, make_leader_bus

GRIPPERS = RAKUDA_GRIPPER_JOINT_NAMES
ARM = RAKUDA_ARM_JOINT_NAMES
JOINT = "r_arm_sh_pitch1"  # an arm joint: current-capable and in the default current_joints
NOT_ARM = ("torso_yaw",) + RAKUDA_HEAD_JOINT_NAMES + GRIPPERS
BILATERAL = RakudaBilateralParams()


# --- helpers -------------------------------------------------------------------


def factory(bus: SimulatedDynamixelBus) -> BusFactory:
    """A ``bus_factory`` handing the arm a prepared simulated bus (a DynamixelBus drop-in)."""
    return lambda port, motors: cast(DynamixelBus, bus)


def config(**kwargs: Any) -> RakudaConfig:
    return RakudaConfig(leader_port="sim", follower_port="sim", **kwargs)


def leader(bus: SimulatedDynamixelBus, **kwargs: Any) -> RakudaLeader:
    return RakudaLeader(config(**kwargs), bus_factory=factory(bus))


def follower(bus: SimulatedDynamixelBus, **kwargs: Any) -> RakudaFollower:
    return RakudaFollower(config(**kwargs), bus_factory=factory(bus))


def writes(bus: SimulatedDynamixelBus, item: XControlTable) -> List[Dict[str, int]]:
    """Every host write of ``item``, in order."""
    return [values for name, values in bus.instruction_log if name == item.name]


def torque_writes(bus: SimulatedDynamixelBus, motor: str) -> List[int]:
    """The ``TORQUE_ENABLE`` values written to ``motor``, in order."""
    return [v[motor] for v in writes(bus, XControlTable.TORQUE_ENABLE) if motor in v]


def items_for(bus: SimulatedDynamixelBus, motor: str) -> List[Tuple[str, int]]:
    """``(item, value)`` of every host write that named ``motor``, in order."""
    return [(name, values[motor]) for name, values in bus.instruction_log if motor in values]


def enter_current_mode(
    bus: SimulatedDynamixelBus, names: Sequence[str], *, goal_current: int
) -> None:
    """Mode 0 + torque on + ``GOAL_CURRENT``: how a killed loop leaves a joint (state D)."""
    bus.torque_disabled(list(names))
    bus.write_with_readback(
        XControlTable.OPERATING_MODE, {n: int(OperatingMode.CURRENT) for n in names}
    )
    bus.torque_enabled(list(names))
    bus.sync_write(XControlTable.GOAL_CURRENT, {n: goal_current for n in names})
    bus.instruction_log.clear()


def messages(caplog: pytest.LogCaptureFixture, level: int) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == level]


def assert_gripper_setup(bus: SimulatedDynamixelBus, *, current_limit: int, goal: int) -> None:
    for name in GRIPPERS:
        regs = bus.registers(name)
        assert regs.torque_enable == 1
        assert regs.operating_mode == OperatingMode.CURRENT_BASED_POSITION
        assert regs.current_limit == current_limit
        assert regs.goal_current == goal


# --- state A: the conventional connect ---------------------------------------------


class TestConventionalConnect:
    def test_leader_enables_grippers_only_and_holds_them(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        arm = leader(bus_l)
        arm.connect()

        assert arm.is_connected
        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == [{"r_arm_grip": 1, "l_arm_grip": 1}]
        assert writes(bus_l, XControlTable.GOAL_POSITION) == [
            {n: LEADER_GRIP_HOLD_POSITION for n in GRIPPERS}
        ]
        # Torque on, then the hold goal: the same order as before this change.
        names = [name for name, _ in bus_l.instruction_log]
        assert names.index("TORQUE_ENABLE") < names.index("GOAL_POSITION")
        for name in ARM + ("torso_yaw",) + RAKUDA_HEAD_JOINT_NAMES:
            assert bus_l.registers(name).torque_enable == 0
            assert bus_l.registers(name).operating_mode == OperatingMode.POSITION
        assert_gripper_setup(
            bus_l,
            current_limit=RAKUDA_CONTROLTABLE_VALUES.LEADER_GRIP_CURRENT_LIMIT,
            goal=RAKUDA_CONTROLTABLE_VALUES.LEADER_GRIP_GOAL_CURRENT,
        )
        # Nothing was written to EEPROM: the grippers already matched.
        assert writes(bus_l, XControlTable.OPERATING_MODE) == []
        assert writes(bus_l, XControlTable.CURRENT_LIMIT) == []

    def test_leader_enabled_list_is_switched_on_with_the_grippers(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        arm = leader(bus_l, leader_torque_enabled=["head_yaw"])
        arm.connect()

        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == [
            {"head_yaw": 1, "r_arm_grip": 1, "l_arm_grip": 1}
        ]
        assert bus_l.registers("head_yaw").torque_enable == 1

    def test_follower_enables_all_seventeen_in_one_write(
        self, bus_f: SimulatedDynamixelBus
    ) -> None:
        arm = follower(bus_f)
        arm.connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == [{n: 1 for n in RAKUDA_JOINT_NAMES}]
        assert all(bus_f.registers(n).torque_enable == 1 for n in RAKUDA_JOINT_NAMES)
        assert_gripper_setup(
            bus_f,
            current_limit=RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_CURRENT_LIMIT,
            goal=RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_GOAL_CURRENT,
        )
        assert writes(bus_f, XControlTable.GOAL_POSITION) == []

    def test_follower_list_limits_the_torque_on(self, bus_f: SimulatedDynamixelBus) -> None:
        arm = follower(bus_f, follower_torque_enabled=["torso_yaw", "head_yaw"])
        arm.connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == [{"torso_yaw": 1, "head_yaw": 1}]

    def test_connect_twice_is_a_no_op(self, bus_l: SimulatedDynamixelBus) -> None:
        arm = leader(bus_l)
        arm.connect()
        count = bus_l.transaction_count
        arm.connect()
        assert bus_l.transaction_count == count

    def test_disconnect_switches_torque_off_by_default(self, bus_l: SimulatedDynamixelBus) -> None:
        arm = leader(bus_l)
        arm.connect()
        bus_l.instruction_log.clear()

        arm.disconnect()

        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == [{n: 0 for n in RAKUDA_JOINT_NAMES}]
        assert not arm.is_connected
        assert not bus_l.port_handler.is_open

    def test_disconnect_can_leave_torque_on(self, bus_f: SimulatedDynamixelBus) -> None:
        arm = follower(bus_f)
        arm.connect()
        bus_f.instruction_log.clear()

        arm.disconnect(torque_off=False)

        assert bus_f.instruction_log == []
        assert all(bus_f.registers(n).torque_enable == 1 for n in RAKUDA_JOINT_NAMES)
        assert not bus_f.port_handler.is_open

    def test_set_port_before_connect_only(self, bus_l: SimulatedDynamixelBus) -> None:
        arm = leader(bus_l)
        arm.set_port("/dev/ttyUSB9")
        assert arm.port == "/dev/ttyUSB9"
        assert bus_l.port_handler.port_name == "/dev/ttyUSB9"

        arm.connect()
        with pytest.raises(RuntimeError):
            arm.set_port("/dev/ttyUSB8")

    def test_default_bus_factory_is_the_real_bus(self) -> None:
        arm = RakudaLeader(config())
        assert isinstance(arm.motors, DynamixelBus)
        assert not arm.is_connected


# --- D13: model check before anything is written -------------------------------------


class TestVerifyModels:
    def test_mismatch_refuses_before_any_write(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus_l.registers("r_arm_grip").set(XControlTable.MODEL_NUMBER, 1240)
        arm = leader(bus_l)

        with pytest.raises(ConnectionError, match="r_arm_grip .*1240"):
            arm.connect()

        assert bus_l.instruction_log == []
        assert not arm.is_connected
        assert not bus_l.port_handler.is_open


# --- D25: operating-mode recovery ---------------------------------------------------


class TestModeRecovery:
    def test_state_c_is_restored_to_position_mode_torque_off(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus_l.torque_disabled([JOINT])
        bus_l.write_with_readback(XControlTable.OPERATING_MODE, {JOINT: int(OperatingMode.CURRENT)})
        bus_l.registers(JOINT).set(XControlTable.PRESENT_POSITION, 1500)
        bus_l.power_cycle()
        bus_l.instruction_log.clear()
        assert bus_l.registers(JOINT).operating_mode == OperatingMode.CURRENT

        leader(bus_l).connect()

        regs = bus_l.registers(JOINT)
        assert regs.operating_mode == OperatingMode.POSITION
        assert regs.torque_enable == 0
        assert regs.goal_position == 1500
        seq = items_for(bus_l, JOINT)
        assert seq.index(("OPERATING_MODE", 3)) < seq.index(("GOAL_POSITION", 1500))
        assert ("TORQUE_ENABLE", 1) not in seq
        assert any(
            JOINT in m and "restoring position mode" in m for m in messages(caplog, logging.WARNING)
        )
        # Every other joint is untouched.
        for name in RAKUDA_JOINT_NAMES:
            if name not in GRIPPERS and name != JOINT:
                assert items_for(bus_l, name) == []

    def test_state_c_on_the_follower_is_torqued_on_after_its_goal(
        self, bus_f: SimulatedDynamixelBus
    ) -> None:
        bus_f.registers(JOINT).set(XControlTable.OPERATING_MODE, int(OperatingMode.CURRENT))
        bus_f.registers(JOINT).set(XControlTable.PRESENT_POSITION, 900)

        follower(bus_f).connect()

        regs = bus_f.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable, regs.goal_position) == (3, 1, 900)
        seq = items_for(bus_f, JOINT)
        assert seq.index(("GOAL_POSITION", 900)) < seq.index(("TORQUE_ENABLE", 1))  # H1

    @pytest.mark.parametrize("mode", [OperatingMode.CURRENT, OperatingMode.VELOCITY])
    def test_state_d_is_held_then_conventional_connect_refuses(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture, mode: OperatingMode
    ) -> None:
        enter_current_mode(bus_l, [JOINT], goal_current=50)
        bus_l.registers(JOINT).set(XControlTable.OPERATING_MODE, int(mode))
        bus_l.registers(JOINT).set(XControlTable.PRESENT_POSITION, 1500)
        arm = leader(bus_l)

        with pytest.raises(ConnectionError, match=rf"previous session left \['{JOINT}'\]"):
            arm.connect()

        regs = bus_l.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable, regs.goal_position) == (3, 1, 1500)
        assert torque_writes(bus_l, JOINT) == [0, 1]
        critical = messages(caplog, logging.CRITICAL)
        assert any(f"{JOINT}=50 mA" in m and "held now" in m for m in critical)
        assert not bus_l.port_handler.is_open

    def test_state_d_is_held_and_bilateral_connect_completes(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        enter_current_mode(bus_l, [JOINT], goal_current=50)
        bus_l.registers(JOINT).set(XControlTable.PRESENT_POSITION, 1500)
        arm = leader(bus_l, bilateral=BILATERAL)

        arm.connect()

        assert arm.is_connected
        regs = bus_l.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable, regs.goal_position) == (3, 1, 1500)
        assert torque_writes(bus_l, JOINT) == [0, 1]  # the hold only; the policy left it alone

    def test_unexpected_mode_with_torque_off_is_restored(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        bus_l.registers(JOINT).set(
            XControlTable.OPERATING_MODE, int(OperatingMode.EXTENDED_POSITION)
        )
        bus_l.registers(JOINT).set(XControlTable.PRESENT_POSITION, 4096 + 100)

        leader(bus_l).connect()

        regs = bus_l.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable, regs.goal_position) == (3, 0, 100)

    def test_head_in_another_mode_is_only_warned_about(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus_l.registers("head_yaw").set(XControlTable.OPERATING_MODE, int(OperatingMode.CURRENT))

        leader(bus_l).connect()

        assert bus_l.registers("head_yaw").operating_mode == OperatingMode.CURRENT
        assert items_for(bus_l, "head_yaw") == []
        assert any("head OPERATING_MODE" in m for m in messages(caplog, logging.WARNING))

    def test_joint_that_stays_out_of_position_mode_refuses_the_connect(
        self, bus_l: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # The hold leaves a joint whose OPERATING_MODE write is not confirmed torque-off
        # in mode 0; the invariant "every current-capable joint is in mode 3" then fails.
        enter_current_mode(bus_l, [JOINT], goal_current=50)
        arm = leader(bus_l, bilateral=BILATERAL)

        def unconfirmed(*args: Any, **kwargs: Any) -> None:
            raise DynamixelCommError("still differs", dxl.COMM_NOT_AVAILABLE)

        monkeypatch.setattr(bus_l, "write_with_readback", unconfirmed)
        with pytest.raises(ConnectionError, match="could not be returned to position mode"):
            arm.connect()

        assert not arm.is_connected
        regs = bus_l.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable) == (OperatingMode.CURRENT, 0)
        assert torque_writes(bus_l, JOINT) == [0]  # never torque-enabled in current mode
        assert not bus_l.port_handler.is_open

    def test_joint_the_hold_left_off_refuses_the_connect(
        self, bus_f: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # State D on the follower (every joint wanted): the mode change to 3 is
        # confirmed but no position can be read, so the hold leaves the joint
        # torque-off with no GOAL_POSITION written after the mode change (D27 H1).
        # It reads mode 3 / torque 0 again afterwards, but the policy must not
        # switch it on: the connect is refused instead.
        enter_current_mode(bus_f, [JOINT], goal_current=50)
        arm = follower(bus_f)
        original_read = bus_f.sync_read

        def no_block(*args: Any, **kwargs: Any) -> Any:
            raise DynamixelCommError("State block read: no complete response", dxl.COMM_RX_TIMEOUT)

        def no_position(item: Any, names: List[str]) -> Dict[str, Any]:
            if item == XControlTable.PRESENT_POSITION:
                return {}
            return original_read(item, names)

        monkeypatch.setattr(bus_f, "read_state_block", no_block)
        monkeypatch.setattr(bus_f, "sync_read", no_position)
        with pytest.raises(ConnectionError, match=rf"cannot hold \['{JOINT}'\]: no position"):
            arm.connect()

        assert not arm.is_connected
        regs = bus_f.registers(JOINT)
        assert (regs.operating_mode, regs.torque_enable) == (OperatingMode.POSITION, 0)
        assert torque_writes(bus_f, JOINT) == [0]
        assert writes(bus_f, XControlTable.GOAL_POSITION) == []
        assert not bus_f.port_handler.is_open


# --- D14: gripper EEPROM only where it differs -----------------------------------------


class TestGripperEeprom:
    def test_equal_values_are_not_rewritten(self, bus_f: SimulatedDynamixelBus) -> None:
        follower(bus_f).connect()

        assert writes(bus_f, XControlTable.OPERATING_MODE) == []
        assert writes(bus_f, XControlTable.CURRENT_LIMIT) == []
        assert writes(bus_f, XControlTable.GOAL_CURRENT) == [
            {n: RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_GOAL_CURRENT for n in GRIPPERS}
        ]

    def test_differing_current_limit_is_written_with_readback(
        self, bus_f: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        for name in GRIPPERS:
            bus_f.registers(name).set(XControlTable.CURRENT_LIMIT, 20)  # as the real follower reads

        follower(bus_f).connect()

        assert writes(bus_f, XControlTable.CURRENT_LIMIT) == [{n: 128 for n in GRIPPERS}]
        assert writes(bus_f, XControlTable.OPERATING_MODE) == []
        assert all(bus_f.registers(n).current_limit == 128 for n in GRIPPERS)
        assert any(
            "CURRENT_LIMIT" in m and "20" in m and "128" in m
            for m in messages(caplog, logging.WARNING)
        )
        # Torque was off: no TORQUE_ENABLE=0 was needed.
        assert 0 not in torque_writes(bus_f, "l_arm_grip") + torque_writes(bus_f, "r_arm_grip")

    def test_only_the_torqued_gripper_is_switched_off_first(
        self, bus_f: SimulatedDynamixelBus
    ) -> None:
        for name in GRIPPERS:
            bus_f.registers(name).set(XControlTable.CURRENT_LIMIT, 20)
        bus_f.torque_enabled(["l_arm_grip"])
        bus_f.instruction_log.clear()

        follower(bus_f).connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE)[0] == {"l_arm_grip": 0}
        assert torque_writes(bus_f, "l_arm_grip") == [0, 1]
        assert torque_writes(bus_f, "r_arm_grip") == [1]
        assert all(bus_f.registers(n).current_limit == 128 for n in GRIPPERS)

    def test_wrong_operating_mode_is_corrected(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus_l.registers("r_arm_grip").set(XControlTable.OPERATING_MODE, int(OperatingMode.POSITION))

        leader(bus_l).connect()

        assert writes(bus_l, XControlTable.OPERATING_MODE) == [{"r_arm_grip": 5}]
        assert bus_l.registers("r_arm_grip").operating_mode == OperatingMode.CURRENT_BASED_POSITION
        assert any("gripper OPERATING_MODE" in m for m in messages(caplog, logging.WARNING))

    def test_unconfirmed_write_is_a_warning_not_a_failure(
        self,
        bus_f: SimulatedDynamixelBus,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        bus_f.registers("l_arm_grip").set(XControlTable.CURRENT_LIMIT, 20)

        def unconfirmed(*args: Any, **kwargs: Any) -> None:
            raise DynamixelCommError("still differs", dxl.COMM_NOT_AVAILABLE)

        monkeypatch.setattr(bus_f, "write_with_readback", unconfirmed)
        arm = follower(bus_f)
        arm.connect()

        assert arm.is_connected
        assert any("write not confirmed" in m for m in messages(caplog, logging.WARNING))

    @pytest.mark.parametrize("mode", [OperatingMode.CURRENT, OperatingMode.VELOCITY])
    def test_gripper_left_in_current_mode_is_never_torque_enabled(
        self,
        bus_l: SimulatedDynamixelBus,
        monkeypatch: pytest.MonkeyPatch,
        mode: OperatingMode,
    ) -> None:
        # A gripper left in mode 0 (a current-mode experiment) whose OPERATING_MODE
        # write is not confirmed must not get TORQUE_ENABLE=1: with GOAL_CURRENT set
        # it would drive continuously and ignore the hold GOAL_POSITION.
        bus_l.registers("r_arm_grip").set(XControlTable.OPERATING_MODE, int(mode))

        def unconfirmed(*args: Any, **kwargs: Any) -> None:
            raise DynamixelCommError("still differs", dxl.COMM_NOT_AVAILABLE)

        monkeypatch.setattr(bus_l, "write_with_readback", unconfirmed)
        arm = leader(bus_l)

        with pytest.raises(ConnectionError, match=rf"r_arm_grip.*OPERATING_MODE.*{int(mode)}"):
            arm.connect()

        assert not arm.is_connected
        assert bus_l.registers("r_arm_grip").operating_mode == mode
        assert bus_l.registers("r_arm_grip").torque_enable == 0
        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == []
        assert writes(bus_l, XControlTable.GOAL_POSITION) == []
        assert not bus_l.port_handler.is_open

    def test_gripper_in_position_mode_with_unconfirmed_write_still_connects(
        self,
        bus_l: SimulatedDynamixelBus,
        caplog: pytest.LogCaptureFixture,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        # Mode 3 is not the wanted 5, but torque-on in it only holds a position:
        # the unconfirmed write stays a warning (D14).
        bus_l.registers("r_arm_grip").set(XControlTable.OPERATING_MODE, int(OperatingMode.POSITION))

        def unconfirmed(*args: Any, **kwargs: Any) -> None:
            raise DynamixelCommError("still differs", dxl.COMM_NOT_AVAILABLE)

        monkeypatch.setattr(bus_l, "write_with_readback", unconfirmed)
        arm = leader(bus_l)
        arm.connect()

        assert arm.is_connected
        assert bus_l.registers("r_arm_grip").torque_enable == 1
        assert any(
            "OPERATING_MODE write not confirmed" in m for m in messages(caplog, logging.WARNING)
        )

    def test_arm_joint_is_rejected(self, bus_l: SimulatedDynamixelBus) -> None:
        arm = leader(bus_l)
        with pytest.raises(AssertionError):
            arm._write_gripper_eeprom([JOINT])
        assert bus_l.instruction_log == []


# --- D15/D35: torque policy at connect --------------------------------------------------


class TestTorquePolicy:
    def test_bilateral_leader_never_touches_the_current_joints(
        self, bus_l: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        arm = leader(bus_l, leader_torque_enabled=list(RAKUDA_JOINT_NAMES), bilateral=BILATERAL)
        arm.connect()

        for name in ARM:
            assert bus_l.registers(name).torque_enable == 0
            assert torque_writes(bus_l, name) == []
        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == [{n: 1 for n in NOT_ARM}]
        warnings = [m for m in messages(caplog, logging.WARNING) if "leader_torque_enabled" in m]
        assert len(warnings) == 1
        assert all(name in warnings[0] for name in ARM)

    def test_bilateral_leader_switches_off_unwanted_non_current_joints(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        bus_l.torque_enabled(["head_yaw", JOINT])
        bus_l.instruction_log.clear()

        leader(bus_l, bilateral=BILATERAL).connect()

        assert torque_writes(bus_l, "head_yaw") == [0]
        assert torque_writes(bus_l, JOINT) == []
        assert bus_l.registers(JOINT).torque_enable == 1  # left as found: the loop owns it

    def test_follower_list_missing_current_joints_is_rejected_at_construction(
        self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        monkeypatch.chdir(tmp_path)
        with pytest.raises(ValueError, match="missing"):
            RakudaPairSys(config(follower_torque_enabled=[], bilateral=BILATERAL))

    def test_conventional_leader_refuses_a_held_unwanted_joint(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        bus_l.torque_enabled([JOINT])
        bus_l.instruction_log.clear()
        arm = leader(bus_l)

        with pytest.raises(ConnectionError) as info:
            arm.connect()

        assert JOINT in str(info.value)
        assert "release --port sim --side leader" in str(info.value)
        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == []
        assert bus_l.registers(JOINT).torque_enable == 1

    def test_conventional_leader_keeps_a_held_wanted_joint_on(
        self, bus_l: SimulatedDynamixelBus
    ) -> None:
        bus_l.torque_enabled([JOINT])
        bus_l.instruction_log.clear()
        arm = leader(bus_l, leader_torque_enabled=[JOINT])

        arm.connect()

        assert torque_writes(bus_l, JOINT) == []
        assert writes(bus_l, XControlTable.TORQUE_ENABLE) == [{"r_arm_grip": 1, "l_arm_grip": 1}]

    def test_leader_empty_list_still_holds_the_grippers(self, bus_l: SimulatedDynamixelBus) -> None:
        leader(bus_l, leader_torque_enabled=[]).connect()

        assert all(bus_l.registers(n).torque_enable == 1 for n in GRIPPERS)
        assert writes(bus_l, XControlTable.GOAL_POSITION) == [
            {n: LEADER_GRIP_HOLD_POSITION for n in GRIPPERS}
        ]

    def test_bilateral_follower_keeps_the_policy_of_the_conventional_mode(
        self, bus_f: SimulatedDynamixelBus
    ) -> None:
        follower(bus_f, bilateral=BILATERAL).connect()
        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == [{n: 1 for n in RAKUDA_JOINT_NAMES}]


class TestFollowerDiffSwitching:
    def test_all_on_means_no_torque_write(self, bus_f: SimulatedDynamixelBus) -> None:
        bus_f.torque_enabled()
        bus_f.instruction_log.clear()

        follower(bus_f).connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == []

    def test_only_the_off_joints_are_switched_on(self, bus_f: SimulatedDynamixelBus) -> None:
        bus_f.torque_enabled([JOINT, "l_arm_grip"])
        bus_f.instruction_log.clear()

        follower(bus_f).connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == [
            {n: 1 for n in RAKUDA_JOINT_NAMES if n not in (JOINT, "l_arm_grip")}
        ]

    def test_unwanted_on_joint_is_switched_off_before_the_wanted_go_on(
        self, bus_f: SimulatedDynamixelBus
    ) -> None:
        bus_f.torque_enabled(["head_yaw"])
        bus_f.instruction_log.clear()

        follower(bus_f, follower_torque_enabled=["torso_yaw"]).connect()

        assert writes(bus_f, XControlTable.TORQUE_ENABLE) == [{"head_yaw": 0}, {"torso_yaw": 1}]


# --- RakudaPairSys: policy at construction, auto ports, disconnect -------------------------


class TestPairSys:
    @pytest.fixture(autouse=True)
    def _isolated_dotfiles(self, tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> None:
        monkeypatch.chdir(tmp_path)

    def test_torque_policy_drives_the_write_filters(self) -> None:
        pair = RakudaPairSys(config(leader_torque_enabled=[]))

        assert pair.torque_policy.leader == frozenset(GRIPPERS)
        assert pair._leader_torque_enabled == set(GRIPPERS)
        assert pair._follower_torque_enabled == set(RAKUDA_JOINT_NAMES)
        assert not pair.is_connected

    def test_auto_ports_are_resolved_and_set_before_opening(
        self, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        cfg = RakudaConfig(leader_port="auto", follower_port="/dev/ttyUSB0")
        pair = RakudaPairSys(cfg)
        assert pair.leader.port == "auto"
        asked: List[Tuple[str, str]] = []

        def fake_resolve(leader_port: str, follower_port: str) -> Tuple[str, str]:
            asked.append((leader_port, follower_port))
            return "/dev/ttyUSB1", follower_port

        monkeypatch.setattr(rakuda_ports, "resolve_ports", fake_resolve)
        leader_arm = MagicMock(port="auto")
        follower_arm = MagicMock(port="/dev/ttyUSB0")
        pair._leader, pair._follower = leader_arm, follower_arm

        pair.connect()

        assert asked == [("auto", "/dev/ttyUSB0")]
        assert leader_arm.mock_calls == [call.set_port("/dev/ttyUSB1"), call.connect()]
        assert follower_arm.mock_calls == [call.connect()]
        assert pair.config.leader_port == "/dev/ttyUSB1"
        assert pair.config.follower_port == "/dev/ttyUSB0"
        assert cfg.leader_port == "auto"  # the caller's object is not mutated
        assert pair.is_connected

    def test_fixed_ports_do_not_scan(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def must_not_scan(*args: Any) -> Tuple[str, str]:
            raise AssertionError("resolve_ports must not be called")

        monkeypatch.setattr(rakuda_ports, "resolve_ports", must_not_scan)
        pair = RakudaPairSys(RakudaConfig(leader_port="/dev/ttyUSB1", follower_port="/dev/ttyUSB0"))
        leader_arm = MagicMock()
        pair._leader, pair._follower = leader_arm, MagicMock()

        pair.connect()

        assert leader_arm.mock_calls == [call.connect()]

    def test_detection_failure_is_a_connection_error(self, monkeypatch: pytest.MonkeyPatch) -> None:
        def no_bus(*args: Any) -> Tuple[str, str]:
            raise OSError("no Rakuda bus answered")

        monkeypatch.setattr(rakuda_ports, "resolve_ports", no_bus)
        pair = RakudaPairSys(RakudaConfig(leader_port="auto", follower_port="auto"))
        leader_arm = MagicMock()
        pair._leader, pair._follower = leader_arm, MagicMock()

        with pytest.raises(ConnectionError, match="no Rakuda bus answered"):
            pair.connect()
        leader_arm.connect.assert_not_called()

    @pytest.mark.parametrize(
        ("kwargs", "torque_off"),
        [
            ({}, True),
            ({"bilateral": BILATERAL}, False),
            ({"bilateral": BILATERAL, "hold_on_disconnect": False}, True),
            ({"hold_on_disconnect": True}, False),
        ],
    )
    def test_disconnect_follows_hold_on_disconnect(
        self, kwargs: Dict[str, Any], torque_off: bool
    ) -> None:
        pair = RakudaPairSys(config(**kwargs))
        leader_arm, follower_arm = MagicMock(), MagicMock()
        pair._leader, pair._follower = leader_arm, follower_arm

        pair.disconnect()

        leader_arm.disconnect.assert_called_once_with(torque_off=torque_off)
        follower_arm.disconnect.assert_called_once_with(torque_off=torque_off)


# --- fixtures ---------------------------------------------------------------------


@pytest.fixture
def bus_l() -> SimulatedDynamixelBus:
    return make_leader_bus()


@pytest.fixture
def bus_f() -> SimulatedDynamixelBus:
    return make_follower_bus()


def test_connect_state_torque_on() -> None:
    state = ConnectState(
        mode={"a": 3, "b": 3}, torque={"a": 1, "b": 0}, goal_current={}, classification={}
    )
    assert state.torque_on == frozenset({"a"})


def test_arm_class_is_abstract() -> None:
    with pytest.raises(TypeError):
        RakudaArm(config(), "sim")  # type: ignore[abstract]
