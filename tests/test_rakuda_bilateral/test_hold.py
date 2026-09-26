"""``hold_joints`` and the control-module vocabulary."""

from __future__ import annotations

import logging
from dataclasses import FrozenInstanceError, dataclass
from types import SimpleNamespace
from typing import Any, Dict, List, Sequence, Tuple

import dynamixel_sdk as dxl
import numpy as np
import pytest
from numpy.typing import NDArray

from robopy.config.robot_config.rakuda_config import RAKUDA_ARM_JOINT_NAMES
from robopy.motor.dynamixel_bus import DynamixelCommError, DynamixelTimeoutError
from robopy.motor.dynamixel_control_table import OperatingMode, XControlTable
from robopy.motor.sim_dynamixel_bus import SimClock, SimulatedDynamixelBus
from robopy.robots.rakuda import rakuda_leader_control as control
from robopy.robots.rakuda.rakuda_leader_control import (
    HOLD_BLOCK_READ_TIMEOUT_S,
    HOLD_BUDGET_S,
    HOLD_MODE_SETTLE_S,
    BilateralError,
    BilateralNotReady,
    ConfigureError,
    ControlBus,
    FollowerLost,
    HoldFailed,
    HoldParams,
    HoldReport,
    LoopFault,
    LoopState,
    LoopStopped,
    PositionSnapshot,
    hold_joints,
    wrap_delta_counts,
)

from .conftest import FakeSdk, make_bus, make_leader_bus

ARM = RAKUDA_ARM_JOINT_NAMES  # the 12 arm joints
J = ARM[:3]  # r_arm_sh_pitch1, r_arm_sh_roll, r_arm_sh_pitch2
PARAMS = HoldParams()
NS = 1_000_000_000


@dataclass(frozen=True)
class ArmStateLike:
    """The three fields of ``RakudaArmState`` that :class:`PositionSnapshot` needs."""

    names: Tuple[str, ...]
    position: NDArray[np.int32]
    t_end_ns: int


def enter_current_mode(bus: SimulatedDynamixelBus, names: Sequence[str]) -> None:
    """Torque off -> mode 0 -> torque on, as ``configure()`` leaves the joints; log cleared."""
    bus.torque_disabled(list(names))
    bus.write_with_readback(
        XControlTable.OPERATING_MODE, {name: OperatingMode.CURRENT for name in names}
    )
    bus.torque_enabled(list(names))
    bus.instruction_log.clear()


def snapshot(names: Sequence[str], positions: Sequence[int], *, t_end_ns: int) -> PositionSnapshot:
    return ArmStateLike(tuple(names), np.array(positions, dtype=np.int32), t_end_ns)


def writes(bus: SimulatedDynamixelBus, item: XControlTable) -> List[Dict[str, int]]:
    return [values for name, values in bus.instruction_log if name == item.name]


def critical_messages(caplog: pytest.LogCaptureFixture) -> List[str]:
    return [r.getMessage() for r in caplog.records if r.levelno == logging.CRITICAL]


def assert_held(bus: SimulatedDynamixelBus, names: Sequence[str], goal: int = 2048) -> None:
    for name in names:
        registers = bus.registers(name)
        assert registers.operating_mode == OperatingMode.POSITION, name
        assert registers.torque_enable == 1, name
        assert registers.goal_position == goal, name


def assert_off(bus: SimulatedDynamixelBus, names: Sequence[str]) -> None:
    for name in names:
        assert bus.registers(name).torque_enable == 0, name
    for values in writes(bus, XControlTable.TORQUE_ENABLE):
        for name in names:
            assert values.get(name, 0) == 0, f"TORQUE_ENABLE=1 written to {name}"


@pytest.fixture
def clock() -> SimClock:
    return SimClock()


@pytest.fixture
def bus(clock: SimClock) -> SimulatedDynamixelBus:
    bus = make_leader_bus(clock=clock)
    enter_current_mode(bus, J)
    return bus


# --- vocabulary --------------------------------------------------------------------


class TestVocabulary:
    def test_loop_states(self) -> None:
        assert [s.name for s in LoopState] == [
            "IDLE",
            "CONFIGURED",
            "RUNNING",
            "HELD",
            "FAULT",
            "RELEASED",
        ]

    def test_exception_hierarchy(self) -> None:
        for cls in (LoopStopped, BilateralNotReady, FollowerLost, ConfigureError, HoldFailed):
            assert issubclass(cls, BilateralError)
        assert issubclass(BilateralError, RuntimeError)

    def test_loop_fault_is_frozen(self) -> None:
        fault = LoopFault("overheat", "r_arm_sh_roll 71 C", 5, 31.5, LoopState.FAULT)
        with pytest.raises(FrozenInstanceError):
            fault.reason = "x"  # type: ignore[misc]
        assert fault.hold_window_ms == 31.5

    def test_loop_stopped_carries_state_and_fault(self) -> None:
        fault = LoopFault("overheat", "r_arm_sh_roll 71 C", 5, None, LoopState.FAULT)
        faulted = LoopStopped(LoopState.FAULT, fault)
        assert (faulted.state, faulted.fault) == (LoopState.FAULT, fault)
        assert str(faulted) == "LeaderCurrentLoop is FAULT (overheat: r_arm_sh_roll 71 C)"
        held = LoopStopped(LoopState.HELD)
        assert held.fault is None
        assert str(held) == "LeaderCurrentLoop is HELD (stopped by operator)"
        assert str(LoopStopped(LoopState.RELEASED)) == "LeaderCurrentLoop is RELEASED (not running)"
        with pytest.raises(BilateralError):
            raise faulted

    def test_hold_budget_and_constants(self) -> None:
        assert HOLD_BUDGET_S == 3.0
        assert HOLD_BLOCK_READ_TIMEOUT_S == 0.04
        assert HOLD_MODE_SETTLE_S == 0.005

    def test_wrap_delta(self) -> None:
        assert wrap_delta_counts(104, 104) == 0
        assert wrap_delta_counts(2048, 2298) == -250
        assert wrap_delta_counts(2298, 2048) == 250
        assert wrap_delta_counts(10, 4090) == 16  # across 0
        assert wrap_delta_counts(4090, 10) == -16
        assert wrap_delta_counts(0, 2048) == -2048


class TestHoldParams:
    def test_defaults(self) -> None:
        assert PARAMS == HoldParams(
            read_retries=3,
            max_snapshot_age_s=0.06,
            max_jump_counts=200,
            profile_velocity=40,
            readback_attempts=5,
        )

    @pytest.mark.parametrize(
        "field, value",
        [
            ("read_retries", 0),
            ("max_snapshot_age_s", -0.1),
            ("max_jump_counts", -1),
            ("profile_velocity", -1),
            ("profile_velocity", 40_000),
            ("readback_attempts", 0),
        ],
    )
    def test_validation(self, field: str, value: float) -> None:
        overrides: Dict[str, Any] = {field: value}
        with pytest.raises(ValueError, match=field):
            HoldParams(**overrides)

    def test_from_bilateral_params_reads_by_attribute_name(self) -> None:
        params = SimpleNamespace(
            hold_read_retries=2,
            hold_max_snapshot_age_s=None,
            effective_hold_max_snapshot_age_s=0.05,
            hold_max_jump_counts=150,
            hold_profile_velocity=60,
            control_hz=60,
        )
        assert HoldParams.from_bilateral_params(params) == HoldParams(
            read_retries=2, max_snapshot_age_s=0.05, max_jump_counts=150, profile_velocity=60
        )
        with pytest.raises(AttributeError):
            HoldParams.from_bilateral_params(SimpleNamespace(hold_read_retries=2))


class TestProtocols:
    def test_both_buses_satisfy_control_bus(self, sdk: FakeSdk) -> None:
        real: ControlBus = make_bus()  # checked statically by mypy
        sim: ControlBus = make_leader_bus()
        assert set(real.motors) == {"a", "b", "c"}
        assert len(sim.motors) == 17

    def test_numpy_snapshot_like_rakuda_arm_state(
        self, bus: SimulatedDynamixelBus, clock: SimClock
    ) -> None:
        # Multi-turn values as current mode reports them; the hold reduces them to one turn.
        positions = np.array([4096 + 2040, 2049, -4096 + 2050], np.int32)
        state: PositionSnapshot = ArmStateLike(tuple(J), positions, clock.monotonic_ns())
        assert int(state.position[0]) == 6136

        report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)
        assert report.sag_counts == {J[0]: 8, J[1]: -1, J[2]: -2}


# --- the hold sequence -------------------------------------------------------------


class TestHoldSequence:
    def test_goal_is_written_before_and_after_torque_on(self, bus: SimulatedDynamixelBus) -> None:
        report = hold_joints(bus, J, PARAMS)

        items = [name for name, _ in bus.instruction_log]
        assert items == [
            "TORQUE_ENABLE",  # 1. off
            "OPERATING_MODE",  # 2. mode 3
            "PROFILE_VELOCITY",  # 3.
            "GOAL_POSITION",  # 5. before torque on
            "TORQUE_ENABLE",  # 6. on
            "GOAL_POSITION",  # 7. after torque on
        ]
        goal = {name: 2048 for name in J}
        assert bus.instruction_log[0] == ("TORQUE_ENABLE", {name: 0 for name in J})
        assert bus.instruction_log[1] == ("OPERATING_MODE", {name: 3 for name in J})
        assert bus.instruction_log[3] == ("GOAL_POSITION", goal)
        assert bus.instruction_log[4] == ("TORQUE_ENABLE", {name: 1 for name in J})
        assert bus.instruction_log[5] == ("GOAL_POSITION", goal)
        assert bus.rejected_writes == []
        assert report.verified

    def test_current_mode_joints_end_in_position_mode_holding_where_they_are(
        self, bus: SimulatedDynamixelBus
    ) -> None:
        for name in J:
            assert bus.registers(name).operating_mode == OperatingMode.CURRENT
            assert bus.registers(name).torque_enable == 1
        bus.registers(J[1]).set(XControlTable.PRESENT_POSITION, 4200)  # multi-turn (mode 0)

        report = hold_joints(bus, J, PARAMS)

        assert_held(bus, [J[0], J[2]], 2048)
        assert_held(bus, [J[1]], 104)  # mode 3 normalised the reading; the goal followed
        assert bus.registers(J[1]).present_position == 104
        assert report.source == {name: "block_read" for name in J}
        assert report.goal_counts == {J[0]: 2048, J[1]: 104, J[2]: 2048}
        assert report.none == ()
        assert report.verified

    def test_goal_current_and_watchdog_are_never_touched(self, bus: SimulatedDynamixelBus) -> None:
        hold_joints(bus, J, PARAMS)
        assert writes(bus, XControlTable.GOAL_CURRENT) == []
        assert writes(bus, XControlTable.BUS_WATCHDOG) == []

    def test_profile_velocity_is_restored_after_the_mode_change(
        self, bus: SimulatedDynamixelBus
    ) -> None:
        for name in J:
            bus.registers(name).set(XControlTable.PROFILE_VELOCITY, 123)

        hold_joints(bus, J, HoldParams(profile_velocity=40))

        # The 0 -> 3 mode change zeroes PROFILE_VELOCITY; the write after it restores a value.
        assert bus.instruction_log[1][0] == "OPERATING_MODE"
        assert bus.instruction_log[2] == ("PROFILE_VELOCITY", {name: 40 for name in J})
        for name in J:
            assert bus.registers(name).profile_velocity == 40

    def test_window_is_measured_with_the_given_clock(self, clock: SimClock) -> None:
        bus = make_leader_bus(clock=clock)
        bus.write_duration_s = 0.001
        bus.read_duration_s = 0.002
        enter_current_mode(bus, J[:2])

        report = hold_joints(bus, J[:2], PARAMS, clock=clock.monotonic_ns)

        # torque off (1) + mode write (1) + settle (5) + 2 read-backs (4) + profile (1)
        # + block read (2) + goal (1) + torque on (1) = 16 ms; steps 7-8 are outside the window.
        assert report.window_ms == pytest.approx(16.0)
        assert clock.elapsed_ns > 16_000_000

    def test_report_fields(self, bus: SimulatedDynamixelBus, clock: SimClock) -> None:
        state = snapshot(J, [2040, 2048, 2058], t_end_ns=clock.monotonic_ns() - 10_000_000)

        report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)

        assert isinstance(report, HoldReport)
        assert report.source == {name: "block_read" for name in J}
        assert report.goal_counts == {name: 2048 for name in J}
        assert report.none == ()
        assert report.sag_counts == {J[0]: 8, J[1]: 0, J[2]: -10}
        assert report.window_ms == pytest.approx(HOLD_MODE_SETTLE_S * 1e3)
        assert report.goal_rewritten_at_torque_on is False
        assert report.verified is True
        assert report.notes == ()
        with pytest.raises(FrozenInstanceError):
            report.verified = False  # type: ignore[misc]

    def test_snapshot_only_matters_for_the_joints_it_names(
        self, bus: SimulatedDynamixelBus
    ) -> None:
        state = snapshot(["torso_yaw", J[2]], [1000, 2000], t_end_ns=NS)
        report = hold_joints(bus, J, PARAMS, state, clock=lambda: NS)
        assert report.sag_counts == {J[2]: 48}
        assert report.verified

    def test_second_call_re_holds_without_error(self, bus: SimulatedDynamixelBus) -> None:
        first = hold_joints(bus, J, PARAMS)
        bus.instruction_log.clear()

        second = hold_joints(bus, J, PARAMS)

        assert second.goal_counts == first.goal_counts
        assert second.verified
        assert_held(bus, J)
        assert writes(bus, XControlTable.OPERATING_MODE) == [{name: 3 for name in J}]
        assert bus.rejected_writes == []

    def test_argument_errors_before_the_bus_is_touched(self, bus: SimulatedDynamixelBus) -> None:
        transactions = bus.transaction_count
        with pytest.raises(ValueError, match="at least one joint"):
            hold_joints(bus, [], PARAMS)
        with pytest.raises(ValueError, match="nope"):
            hold_joints(bus, [J[0], "nope"], PARAMS)
        assert bus.instruction_log == []
        assert bus.transaction_count == transactions

    def test_duplicate_names_are_held_once(self, bus: SimulatedDynamixelBus) -> None:
        report = hold_joints(bus, [J[0], J[0], J[1]], PARAMS)
        assert list(report.source) == [J[0], J[1]]
        assert bus.instruction_log[0] == ("TORQUE_ENABLE", {J[0]: 0, J[1]: 0})


# --- fallbacks: silent joints, snapshot age, jump check --------------------------------


class TestFallbacks:
    def test_all_silent_with_a_fresh_snapshot_holds_at_the_snapshot(
        self, bus: SimulatedDynamixelBus, clock: SimClock
    ) -> None:
        bus.silent = set(J)
        state = snapshot(J, [4200, 4200, 4200], t_end_ns=clock.monotonic_ns() - 50_000_000)

        report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)

        assert report.source == {name: "snapshot" for name in J}
        assert report.goal_counts == {name: 104 for name in J}
        assert report.none == ()
        assert report.sag_counts == {name: 0 for name in J}
        assert_held(bus, J, 104)
        assert writes(bus, XControlTable.GOAL_POSITION) == [{name: 104 for name in J}] * 2
        # Nothing could be read back: not verified, no third write, nothing raised.
        assert report.verified is False
        assert report.notes == (
            f"{J[0]}: OPERATING_MODE=3 not confirmed (no answer)",
            f"{J[1]}: OPERATING_MODE=3 not confirmed (no answer)",
            f"{J[2]}: OPERATING_MODE=3 not confirmed (no answer)",
            "GOAL_POSITION unreadable right after torque-on",
            "TORQUE_ENABLE/GOAL_POSITION read-back failed; hold not verified",
        )

    def test_all_silent_with_a_stale_snapshot_leaves_torque_off(
        self, bus: SimulatedDynamixelBus, clock: SimClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus.silent = set(J)
        state = snapshot(J, [4200, 4200, 4200], t_end_ns=clock.monotonic_ns() - 200_000_000)

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)

        assert report.source == {name: "none" for name in J}
        assert report.none == tuple(J)
        assert report.goal_counts == {}
        assert report.verified is False
        assert_off(bus, J)
        assert writes(bus, XControlTable.GOAL_POSITION) == []
        assert any("snapshot too old (200 ms > 60 ms)" == note for note in report.notes)
        assert critical_messages(caplog) == [
            f"cannot hold {', '.join(J)}: no position reading; torque left OFF, "
            "use the power switch"
        ]

    def test_snapshot_age_is_taken_at_the_start_of_the_hold(
        self, bus: SimulatedDynamixelBus, clock: SimClock
    ) -> None:
        # The failing reads burn many times max_snapshot_age_s of simulated
        # time; the age check must not count that time against the snapshot.
        bus.silent = set(J)
        state = snapshot(J, [2048, 2048, 2048], t_end_ns=clock.monotonic_ns() - 60_000_000)
        report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)
        assert clock.elapsed_ns > 10 * int(PARAMS.max_snapshot_age_s * NS)
        assert report.source == {name: "snapshot" for name in J}

    def test_one_silent_joint_is_left_off_and_the_others_held(
        self, bus: SimulatedDynamixelBus, caplog: pytest.LogCaptureFixture
    ) -> None:
        bus.silent = {J[1]}

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS)

        # The silent joint is excluded from the block read, so the others keep "block_read".
        assert report.source == {J[0]: "block_read", J[1]: "none", J[2]: "block_read"}
        assert report.none == (J[1],)
        assert report.goal_counts == {J[0]: 2048, J[2]: 2048}
        assert report.verified is False
        assert_held(bus, [J[0], J[2]])
        assert_off(bus, [J[1]])
        assert writes(bus, XControlTable.TORQUE_ENABLE) == [
            {name: 0 for name in J},
            {J[0]: 1, J[2]: 1},
        ]
        assert writes(bus, XControlTable.GOAL_POSITION) == [{J[0]: 2048, J[2]: 2048}] * 2
        assert critical_messages(caplog) == [
            f"cannot hold {J[1]}: no position reading; torque left OFF, use the power switch"
        ]
        assert f"{J[1]}: OPERATING_MODE=3 not confirmed (no answer)" in report.notes

    def test_reading_far_from_the_snapshot_is_reread_once_then_distrusted(
        self,
        bus: SimulatedDynamixelBus,
        monkeypatch: pytest.MonkeyPatch,
        caplog: pytest.LogCaptureFixture,
    ) -> None:
        state = snapshot(J, [2298, 2048, 2048], t_end_ns=NS)  # J[0]: 250 counts off
        reads: List[List[str]] = []
        original = bus.read_state_block

        def spy(names: Sequence[str], **kwargs: Any) -> Any:
            reads.append(list(names))
            return original(names, **kwargs)

        monkeypatch.setattr(bus, "read_state_block", spy)

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS, state, clock=lambda: NS)

        assert reads == [list(J), [J[0]]]  # the block read, then one bounded re-read
        assert report.source == {J[0]: "none", J[1]: "block_read", J[2]: "block_read"}
        assert report.none == (J[0],)
        assert J[0] not in report.sag_counts
        assert_off(bus, [J[0]])
        assert_held(bus, [J[1], J[2]])
        assert [m for m in critical_messages(caplog) if "2298" in m and "2048" in m]
        assert critical_messages(caplog)[-1].startswith(f"cannot hold {J[0]}")

    def test_reread_that_agrees_with_the_snapshot_is_used(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        state = snapshot(J, [2298, 2048, 2048], t_end_ns=NS)
        original = bus.read_state_block

        def settle_then_read(names: Sequence[str], **kwargs: Any) -> Any:
            if list(names) == [J[0]]:
                bus.registers(J[0]).set(XControlTable.PRESENT_POSITION, 2290)
            return original(names, **kwargs)

        monkeypatch.setattr(bus, "read_state_block", settle_then_read)

        report = hold_joints(bus, J, PARAMS, state, clock=lambda: NS)

        assert report.source == {J[0]: "single_read", J[1]: "block_read", J[2]: "block_read"}
        assert report.goal_counts[J[0]] == 2290
        assert report.sag_counts[J[0]] == -8
        assert report.verified
        assert_held(bus, [J[0]], 2290)

    def test_jump_check_needs_a_snapshot(self, bus: SimulatedDynamixelBus) -> None:
        bus.registers(J[0]).set(XControlTable.PRESENT_POSITION, 3000)
        report = hold_joints(bus, J, PARAMS)  # no snapshot: no reference, nothing to distrust
        assert report.source[J[0]] == "block_read"
        assert_held(bus, [J[0]], 3000)

    def test_stale_snapshot_does_not_veto_a_fresh_reading(
        self, bus: SimulatedDynamixelBus, clock: SimClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        # A snapshot older than max_snapshot_age_s is no reference (the jump bound
        # only holds for a fresh one): the fresh reading is held, not vetoed.
        bus.registers(J[0]).set(XControlTable.PRESENT_POSITION, 2348)  # 300 counts off
        state = snapshot(J, [2048, 2048, 2048], t_end_ns=clock.monotonic_ns() - 500_000_000)

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS, state, clock=clock.monotonic_ns)

        assert report.source == {name: "block_read" for name in J}
        assert report.none == ()
        assert report.sag_counts == {J[0]: 300, J[1]: 0, J[2]: 0}
        assert report.verified
        assert_held(bus, [J[0]], 2348)
        assert "snapshot too old (500 ms > 60 ms)" in report.notes
        assert critical_messages(caplog) == []

    def test_block_read_is_retried_before_single_reads(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        attempts = {"n": 0}
        original = bus.read_state_block

        def flaky(names: Sequence[str], **kwargs: Any) -> Any:
            attempts["n"] += 1
            if attempts["n"] < 3:
                raise DynamixelCommError(
                    "State block read: no complete response", dxl.COMM_RX_TIMEOUT
                )
            return original(names, **kwargs)

        monkeypatch.setattr(bus, "read_state_block", flaky)

        report = hold_joints(bus, J, HoldParams(read_retries=3))

        assert attempts["n"] == 3
        assert report.source == {name: "block_read" for name in J}
        assert report.verified


# --- step 2: a joint that provably did not reach mode 3 -----------------------------


class TestModeChangeFailure:
    def test_joint_still_in_mode_0_is_never_torque_enabled(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        stuck = J[2]
        original = bus.sync_write

        def lose_mode_write(item: Any, values: Dict[str, Any]) -> None:
            # The motor answers reads but never takes the OPERATING_MODE write.
            if item is XControlTable.OPERATING_MODE:
                values = {name: v for name, v in values.items() if name != stuck}
            original(item, values)

        monkeypatch.setattr(bus, "sync_write", lose_mode_write)

        report = hold_joints(bus, J, PARAMS)

        assert bus.registers(stuck).operating_mode == OperatingMode.CURRENT
        assert_off(bus, [stuck])
        assert_held(bus, [J[0], J[1]])
        assert report.source == {J[0]: "block_read", J[1]: "block_read", stuck: "none"}
        assert report.none == (stuck,)
        assert f"{stuck}: OPERATING_MODE=3 not confirmed by read-back; torque left off" in (
            report.notes
        )
        assert writes(bus, XControlTable.PROFILE_VELOCITY) == [{J[0]: 40, J[1]: 40}]

    def test_group_readback_failure_costs_no_legacy_read(
        self, bus: SimulatedDynamixelBus, clock: SimClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        # Every read after a failed group read-back is bounded: no sync_read at all
        # inside the window, and the responsive joints are confirmed as one group.
        stuck = J[2]
        original = bus.sync_write
        legacy_reads: List[str] = []

        def lose_mode_write(item: Any, values: Dict[str, Any]) -> None:
            if item is XControlTable.OPERATING_MODE:
                values = {name: v for name, v in values.items() if name != stuck}
            original(item, values)

        def spy(item: Any, names: List[str]) -> Dict[str, Any]:
            legacy_reads.append(item.name)
            return SimulatedDynamixelBus.sync_read(bus, item, names)

        monkeypatch.setattr(bus, "sync_write", lose_mode_write)
        monkeypatch.setattr(bus, "sync_read", spy)

        report = hold_joints(bus, J, PARAMS, clock=clock.monotonic_ns)

        assert legacy_reads == ["GOAL_POSITION", "TORQUE_ENABLE", "GOAL_POSITION"]  # steps 7-8
        assert report.none == (stuck,)
        # Settle of the first group round + readback_attempts rounds on the three
        # responsive joints + one locating round per responsive joint: 5 ms each.
        rounds = 1 + PARAMS.readback_attempts + len(J)
        assert report.window_ms == pytest.approx(rounds * HOLD_MODE_SETTLE_S * 1e3)


# --- the torque-off window and the hold budget with silent joints -----------------------


class TestBoundedWindow:
    """Spec 6.3.3/6.8.3: a dead joint must not stall the others torque-off for seconds."""

    def test_one_silent_joint_of_twelve_keeps_the_window_bounded(self, clock: SimClock) -> None:
        bus = make_leader_bus(clock=clock)
        enter_current_mode(bus, ARM)
        bus.silent = {ARM[5]}

        report = hold_joints(bus, ARM, PARAMS, clock=clock.monotonic_ns)

        assert report.none == (ARM[5],)
        assert report.source[ARM[0]] == "block_read"
        assert_held(bus, [name for name in ARM if name != ARM[5]])
        # Group round: 5 ms settle + one 40 ms read-back timeout of the silent joint;
        # classification: one 40 ms bounded read of it; group confirmation of the
        # eleven others: 5 ms settle.  No legacy 10-retry read (0.37 s) anywhere.
        timeout_ms = HOLD_BLOCK_READ_TIMEOUT_S * 1e3
        settle_ms = HOLD_MODE_SETTLE_S * 1e3
        assert report.window_ms == pytest.approx(2 * settle_ms + 2 * timeout_ms)
        assert report.window_ms <= 100.0

    @pytest.mark.parametrize("with_snapshot", [True, False])
    def test_fully_silent_twelve_joint_hold_finishes_within_the_budget(
        self, clock: SimClock, with_snapshot: bool
    ) -> None:
        bus = make_leader_bus(clock=clock)
        enter_current_mode(bus, ARM)
        bus.silent = set(ARM)
        state = (
            snapshot(ARM, [2048] * len(ARM), t_end_ns=clock.monotonic_ns() - 10_000_000)
            if with_snapshot
            else None
        )

        report = hold_joints(bus, ARM, PARAMS, state, clock=clock.monotonic_ns)

        assert clock.elapsed_ns < HOLD_BUDGET_S * NS
        assert report.verified is False
        if with_snapshot:
            assert report.source == {name: "snapshot" for name in ARM}
            assert report.none == ()
        else:
            assert report.none == tuple(ARM)
        # The window itself: two bounded timeouts per silent joint plus one settle.
        timeout_ms = HOLD_BLOCK_READ_TIMEOUT_S * 1e3
        assert report.window_ms == pytest.approx(
            HOLD_MODE_SETTLE_S * 1e3 + 2 * len(ARM) * timeout_ms
        )


# --- steps 7-8: torque-on side effects and verification -------------------------------


class TestTorqueOnAndVerification:
    def test_firmware_rewrite_at_torque_on_is_detected_and_undone(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        original = bus.torque_enabled

        def rewriting_torque_on(names: List[str] | None = None) -> None:
            original(names)
            bus.registers(J[0]).set(XControlTable.GOAL_POSITION, 100)

        monkeypatch.setattr(bus, "torque_enabled", rewriting_torque_on)

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS)

        assert report.goal_rewritten_at_torque_on is True
        assert report.verified is True
        assert_held(bus, J, 2048)
        assert len(writes(bus, XControlTable.GOAL_POSITION)) == 2
        assert any("firmware rewrote GOAL_POSITION at torque-on" in m for m in caplog.messages)

    def test_readback_mismatch_triggers_one_more_write(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = bus.sync_read
        torque_reads = {"n": 0}

        def clobber_once(item: Any, names: List[str]) -> Dict[str, Any]:
            if item is XControlTable.TORQUE_ENABLE:
                torque_reads["n"] += 1
                if torque_reads["n"] == 1:
                    bus.registers(J[1]).set(XControlTable.GOAL_POSITION, 7)
            return original(item, names)

        monkeypatch.setattr(bus, "sync_read", clobber_once)

        report = hold_joints(bus, J, PARAMS)

        assert len(writes(bus, XControlTable.GOAL_POSITION)) == 3
        assert report.verified is True
        assert report.notes == ()
        assert_held(bus, J, 2048)

    def test_persistent_mismatch_is_reported_not_raised(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch, caplog: Any
    ) -> None:
        original = bus.sync_read

        def always_clobber(item: Any, names: List[str]) -> Dict[str, Any]:
            if item is XControlTable.TORQUE_ENABLE:
                bus.registers(J[1]).set(XControlTable.GOAL_POSITION, 7)
            return original(item, names)

        monkeypatch.setattr(bus, "sync_read", always_clobber)

        with caplog.at_level(logging.INFO, logger=control.__name__):
            report = hold_joints(bus, J, PARAMS)

        assert len(writes(bus, XControlTable.GOAL_POSITION)) == 3
        assert report.verified is False
        assert report.none == ()
        assert report.notes == (f"{J[1]}: GOAL_POSITION=7 != 2048",)
        assert any(m.startswith("hold: not verified") for m in critical_messages(caplog))


# --- HoldFailed ------------------------------------------------------------------------


class TestHoldFailed:
    def test_unexpected_exception_becomes_hold_failed_with_torque_left_off(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        original = bus.sync_write

        def broken_goal_write(item: Any, values: Any) -> None:
            if item is XControlTable.GOAL_POSITION:
                raise RuntimeError("boom")
            original(item, values)

        monkeypatch.setattr(bus, "sync_write", broken_goal_write)

        with pytest.raises(HoldFailed, match="hold aborted at GOAL_POSITION: boom") as info:
            hold_joints(bus, J, PARAMS)

        assert isinstance(info.value.__cause__, RuntimeError)
        assert isinstance(info.value, BilateralError)
        assert_off(bus, J)  # nothing torque-enabled without its goal
        for name in J:
            assert bus.registers(name).operating_mode == OperatingMode.POSITION

    def test_transmit_failure_of_torque_on_is_hold_failed(
        self, bus: SimulatedDynamixelBus, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        def dead_torque_on(names: List[str] | None = None) -> None:
            raise DynamixelCommError("Failed to sync write TORQUE_ENABLE.", dxl.COMM_TX_FAIL)

        monkeypatch.setattr(bus, "torque_enabled", dead_torque_on)

        with pytest.raises(HoldFailed, match="TORQUE_ENABLE=1"):
            hold_joints(bus, J, PARAMS)

    def test_a_silent_bus_does_not_raise(self, bus: SimulatedDynamixelBus) -> None:
        bus.silent = set(bus.motors)
        report = hold_joints(bus, J, PARAMS)
        assert report.none == tuple(J)
        assert report.verified is False

    def test_hold_retried_after_a_stalled_serial_write_succeeds(self, sdk: FakeSdk) -> None:
        # A blocked USB write during step 1 raises out of the SDK with
        # port.is_using still set; the retry hold (from _run's finally or
        # stop()) must not then fail with COMM_PORT_BUSY on every transmit.
        real = make_bus()
        names = list(real.motors)
        for motor_id in (1, 2, 3):
            sdk.set_state(motor_id, position=2048, velocity=0, current=0)
            sdk.set_register(motor_id, XControlTable.TORQUE_ENABLE, 1)
        sdk.stall_writes = 1

        with pytest.raises(HoldFailed, match="hold aborted at TORQUE_ENABLE=0") as info:
            hold_joints(real, names, PARAMS)
        assert isinstance(info.value.__cause__, DynamixelTimeoutError)
        assert real.port_handler.is_using is False

        report = hold_joints(real, names, PARAMS)

        assert report.verified
        assert report.none == ()
        for motor_id in (1, 2, 3):
            assert (
                sdk.get_register(motor_id, XControlTable.OPERATING_MODE) == OperatingMode.POSITION
            )
            assert sdk.get_register(motor_id, XControlTable.TORQUE_ENABLE) == 1
            assert sdk.get_register(motor_id, XControlTable.GOAL_POSITION) == 2048
