"""Shared fixtures for the Rakuda bilateral tests.

Nothing here touches a serial port.  ``FakeSdk`` stands in for the three
``dynamixel_sdk`` classes that :class:`DynamixelBus` instantiates
(``PortHandler``, ``GroupSyncRead``, ``GroupSyncWrite``); it keeps a register
image per motor ID, records every SDK call in order, and returns scripted
``COMM_*`` codes.  ``FakeClock`` replaces ``time.monotonic_ns``/``time.sleep``
inside ``robopy.motor.dynamixel_bus`` so timing assertions are exact.
"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple

import dynamixel_sdk as dxl
import numpy as np
import pytest
import serial
from numpy.typing import NDArray

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_CONTROLTABLE_VALUES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaArmState,
    RakudaBilateralParams,
)
from robopy.motor import dynamixel_bus
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelMotor
from robopy.motor.dynamixel_control_table import (
    CURRENT_UNIT_MA,
    STATE_BLOCK_START_ADDRESS,
    OperatingMode,
    XControlTable,
    encode_value,
)
from robopy.motor.sim_dynamixel_bus import SimClock, SimulatedDynamixelBus
from robopy.robots.rakuda.rakuda_arm import RakudaArm
from robopy.robots.rakuda.rakuda_control_laws import LawOutput
from robopy.robots.rakuda.rakuda_follower import RakudaFollower
from robopy.robots.rakuda.rakuda_leader import RakudaLeader
from robopy.robots.rakuda.rakuda_leader_control import FollowerPositionIO, LeaderCurrentLoop

DEFAULT_MOTORS: Tuple[Tuple[str, int, str], ...] = (
    ("a", 1, "xc330-t288"),
    ("b", 2, "xc330-t288"),
    ("c", 3, "xm430-w350"),
)


class FakeClock:
    """A monotonic clock that only moves when told to."""

    def __init__(self, start_ns: int = 1_000_000_000) -> None:
        self.now_ns = start_ns
        self.start_ns = start_ns

    def monotonic_ns(self) -> int:
        return self.now_ns

    def monotonic(self) -> float:
        return self.now_ns / 1e9

    def advance(self, seconds: float) -> None:
        self.now_ns += int(round(seconds * 1e9))

    def sleep(self, seconds: float) -> None:
        self.advance(seconds)

    @property
    def elapsed_ns(self) -> int:
        return self.now_ns - self.start_ns


class FakeSerial:
    """Just enough of ``serial.Serial`` for ``DynamixelBus.open()`` and its reads.

    ``input_buffer`` holds status packets (``{id: bytes}``) that a motor sent
    after the SDK had already given up on a transaction; like the real
    ``readRx`` the fake ``GroupSyncRead`` accepts such a packet for a matching
    ID as if it were fresh, unless ``reset_input_buffer`` dropped it first.
    """

    def __init__(self, sdk: FakeSdk) -> None:
        self.sdk = sdk
        self.write_timeout: float | None = None
        self.input_buffer: Dict[int, List[int]] = {}

    def reset_input_buffer(self) -> None:
        self.sdk.calls.append(("Serial", "reset_input_buffer", None))
        self.input_buffer.clear()


class FakePortHandler:
    """Stand-in for ``dynamixel_sdk.PortHandler``."""

    def __init__(self, port_name: str, sdk: FakeSdk) -> None:
        self.sdk = sdk
        self.port_name = port_name
        self.is_open = False
        self.baudrate = 0
        self.ser: FakeSerial | None = None
        self.packet_timeout_ms = 0.0
        # Like the SDK: set by txPacket, cleared by the packet handler's
        # rxPacket (reads) or right after the write (SyncWrite).
        self.is_using = False

    def openPort(self) -> bool:  # noqa: N802 - SDK spelling
        if self.sdk.open_fails:
            return False
        self.is_open = True
        self.ser = FakeSerial(self.sdk) if self.sdk.ports_have_serial else None
        return True

    def closePort(self) -> None:  # noqa: N802
        self.is_open = False

    def setBaudRate(self, baudrate: int) -> bool:  # noqa: N802
        self.baudrate = baudrate
        return True

    def getPortName(self) -> str:  # noqa: N802
        return self.port_name

    def setPacketTimeoutMillis(self, msec: float) -> None:  # noqa: N802
        self.packet_timeout_ms = float(msec)
        self.sdk.calls.append(("PortHandler", "setPacketTimeoutMillis", float(msec)))


class FakeGroupSyncRead:
    """Stand-in for ``dynamixel_sdk.GroupSyncRead`` (protocol 2.0 semantics)."""

    def __init__(self, port: FakePortHandler, ph: Any, start_address: int, data_length: int):
        self.port = port
        self.sdk = port.sdk
        self.start_address = start_address
        self.data_length = data_length
        self.data_dict: Dict[int, List[int]] = {}
        self.last_result = False
        self.sdk.read_groups.append(self)

    def addParam(self, dxl_id: int) -> bool:  # noqa: N802
        if dxl_id in self.data_dict:
            return False
        self.data_dict[dxl_id] = []
        return True

    def clearParam(self) -> None:  # noqa: N802
        self.data_dict.clear()

    def txPacket(self) -> int:  # noqa: N802
        self.sdk.calls.append(("GroupSyncRead", "txPacket", self.start_address))
        return self._tx()

    def rxPacket(self) -> int:  # noqa: N802
        self.sdk.calls.append(("GroupSyncRead", "rxPacket", self.start_address))
        return self._rx()

    def txRxPacket(self) -> int:  # noqa: N802
        self.sdk.calls.append(("GroupSyncRead", "txRxPacket", self.start_address))
        result = self._tx()
        if result != dxl.COMM_SUCCESS:
            return result
        return self._rx()

    def _tx(self) -> int:
        if not self.data_dict:
            return dxl.COMM_NOT_AVAILABLE
        if self.port.is_using:
            return dxl.COMM_PORT_BUSY
        self.port.is_using = True
        # A stalled serial write escapes the SDK with ``is_using`` still set.
        self.sdk.maybe_stall()
        if self.sdk.tx_results:
            result = self.sdk.tx_results.pop(0)
            if result != dxl.COMM_SUCCESS:
                self.port.is_using = False
            return result
        return dxl.COMM_SUCCESS

    def _rx(self) -> int:
        self.last_result = False
        self.port.is_using = False
        if not self.data_dict:
            return dxl.COMM_NOT_AVAILABLE
        ids = list(self.data_dict)
        first_silent = next(
            (i for i, dxl_id in enumerate(ids) if dxl_id in self.sdk.silent_ids), None
        )
        if self.sdk.rx_results:
            result = self.sdk.rx_results.pop(0)
        elif first_silent is not None:
            result = dxl.COMM_RX_TIMEOUT
        else:
            result = dxl.COMM_SUCCESS
        # ``None`` before ``open()`` or for a serial stand-in without a buffer.
        input_buffer: Dict[int, List[int]] | None = getattr(self.port.ser, "input_buffer", None)
        if result != dxl.COMM_SUCCESS:
            # The real SDK busy-polls the port until its packet timeout expires.
            self.sdk.clock.advance(self.port.packet_timeout_ms / 1e3)
            if first_silent is not None and input_buffer is not None:
                # The SDK gave up at the silent ID; the motors behind it still
                # answer, and their packets stay queued in the input buffer.
                for dxl_id in ids[first_silent + 1 :]:
                    if dxl_id not in self.sdk.silent_ids:
                        input_buffer[dxl_id] = self._fresh_block(dxl_id)
            return result
        for dxl_id in ids:
            queued = None if input_buffer is None else input_buffer.pop(dxl_id, None)
            self.data_dict[dxl_id] = self._fresh_block(dxl_id) if queued is None else queued
        self.sdk.clock.advance(self.sdk.read_duration_s)
        self.last_result = True
        return dxl.COMM_SUCCESS

    def _fresh_block(self, dxl_id: int) -> List[int]:
        memory = self.sdk.memory(dxl_id)
        length = self.data_length - 1 if dxl_id in self.sdk.short_ids else self.data_length
        return list(memory[self.start_address : self.start_address + length])

    def isAvailable(self, dxl_id: int, address: int, data_length: int) -> bool:  # noqa: N802
        if not self.last_result or dxl_id not in self.data_dict:
            return False
        if dxl_id in self.sdk.unavailable_ids:
            return False
        if address < self.start_address:
            return False
        return self.start_address + self.data_length - data_length >= address

    def getData(self, dxl_id: int, address: int, data_length: int) -> int:  # noqa: N802
        if not self.isAvailable(dxl_id, address, data_length):
            return 0
        start = address - self.start_address
        chunk = self.data_dict[dxl_id][start : start + data_length]
        return int.from_bytes(bytes(chunk), "little")


class FakeGroupSyncWrite:
    """Stand-in for ``dynamixel_sdk.GroupSyncWrite``."""

    def __init__(self, port: FakePortHandler, ph: Any, start_address: int, data_length: int):
        self.port = port
        self.sdk = port.sdk
        self.start_address = start_address
        self.data_length = data_length
        self.data_dict: Dict[int, List[int]] = {}
        self.sdk.write_groups.append(self)

    def addParam(self, dxl_id: int, data: Sequence[int]) -> bool:  # noqa: N802
        if dxl_id in self.data_dict or len(data) > self.data_length:
            return False
        self.data_dict[dxl_id] = list(data)
        return True

    def clearParam(self) -> None:  # noqa: N802
        self.data_dict.clear()

    def txPacket(self) -> int:  # noqa: N802
        self.sdk.calls.append(("GroupSyncWrite", "txPacket", self.start_address))
        if not self.data_dict:
            return dxl.COMM_NOT_AVAILABLE
        if self.port.is_using:
            return dxl.COMM_PORT_BUSY
        self.port.is_using = True
        # A stalled serial write escapes the SDK with ``is_using`` still set.
        self.sdk.maybe_stall()
        self.port.is_using = False
        self.sdk.writes.append(
            (self.start_address, {dxl_id: list(data) for dxl_id, data in self.data_dict.items()})
        )
        result = self.sdk.write_results.pop(0) if self.sdk.write_results else dxl.COMM_SUCCESS
        if result != dxl.COMM_SUCCESS:
            return result
        for dxl_id, data in self.data_dict.items():
            remaining = self.sdk.rejected_writes.get(dxl_id, 0)
            if remaining > 0:
                # The motor ignores this write (e.g. EEPROM while torque is on).
                self.sdk.rejected_writes[dxl_id] = remaining - 1
                continue
            memory = self.sdk.memory(dxl_id)
            memory[self.start_address : self.start_address + len(data)] = bytes(data)
        return dxl.COMM_SUCCESS


class FakeSdk:
    """Scripted ``dynamixel_sdk`` layer plus a register image per motor ID.

    Attributes:
        calls: Every SDK call as ``(class, method, detail)``, in order.
        writes: Every transmitted SyncWrite as ``(address, {id: bytes})``.
        silent_ids: IDs that never answer a read (``COMM_RX_TIMEOUT`` after the
            port's packet timeout has elapsed on the fake clock).
        unavailable_ids: IDs whose read "succeeds" but ``isAvailable`` is False.
        short_ids: IDs whose status packet carries one byte less than requested
            (``isAvailable`` still True, like the real SDK).
        rejected_writes: ``{id: n}``: the next ``n`` writes to that ID are ignored.
        tx_results / rx_results / write_results: Scripted result queues; an
            empty queue means ``COMM_SUCCESS`` (subject to ``silent_ids``).
        read_duration_s: Fake time consumed by a successful ``rxPacket``.
        stall_writes: Number of upcoming ``txPacket`` calls (either group kind)
            that raise ``serial.SerialTimeoutException`` like a blocked USB write.
    """

    def __init__(self, clock: FakeClock) -> None:
        self.clock = clock
        self.calls: List[Tuple[str, str, Any]] = []
        self.writes: List[Tuple[int, Dict[int, List[int]]]] = []
        self.registers: Dict[int, bytearray] = {}
        self.silent_ids: set[int] = set()
        self.unavailable_ids: set[int] = set()
        self.short_ids: set[int] = set()
        self.rejected_writes: Dict[int, int] = {}
        self.tx_results: List[int] = []
        self.rx_results: List[int] = []
        self.write_results: List[int] = []
        self.read_duration_s = 0.0
        self.stall_writes = 0
        self.open_fails = False
        self.ports_have_serial = True
        self.ports: List[FakePortHandler] = []
        self.read_groups: List[FakeGroupSyncRead] = []
        self.write_groups: List[FakeGroupSyncWrite] = []

    # -- monkeypatch entry point -------------------------------------------

    def install(self, monkeypatch: pytest.MonkeyPatch) -> None:
        """Routes ``DynamixelBus`` at the fakes and the fake clock."""
        monkeypatch.setattr(dynamixel_bus.dxl, "PortHandler", self._make_port)
        monkeypatch.setattr(dynamixel_bus.dxl, "GroupSyncRead", FakeGroupSyncRead)
        monkeypatch.setattr(dynamixel_bus.dxl, "GroupSyncWrite", FakeGroupSyncWrite)
        monkeypatch.setattr(dynamixel_bus.time, "monotonic_ns", self.clock.monotonic_ns)
        monkeypatch.setattr(dynamixel_bus.time, "sleep", self.clock.sleep)

    def _make_port(self, port_name: str) -> FakePortHandler:
        port = FakePortHandler(port_name, self)
        self.ports.append(port)
        return port

    def maybe_stall(self) -> None:
        if self.stall_writes > 0:
            self.stall_writes -= 1
            raise serial.SerialTimeoutException("Write timeout")

    # -- register image ------------------------------------------------------

    def memory(self, motor_id: int) -> bytearray:
        return self.registers.setdefault(motor_id, bytearray(256))

    def set_register(self, motor_id: int, item: XControlTable, value: int) -> None:
        control_item = item.value
        word = encode_value(value, control_item.dtype)
        self.set_bytes(
            motor_id, control_item.address, word.to_bytes(control_item.num_bytes, "little")
        )

    def get_register(self, motor_id: int, item: XControlTable) -> int:
        control_item = item.value
        raw = self.memory(motor_id)[
            control_item.address : control_item.address + control_item.num_bytes
        ]
        return int.from_bytes(raw, "little")

    def set_bytes(self, motor_id: int, address: int, data: bytes) -> None:
        self.memory(motor_id)[address : address + len(data)] = data

    def set_state(self, motor_id: int, *, position: int, velocity: int, current: int) -> None:
        block = (
            current.to_bytes(2, "little", signed=True)
            + velocity.to_bytes(4, "little", signed=True)
            + position.to_bytes(4, "little", signed=True)
        )
        self.set_bytes(motor_id, STATE_BLOCK_START_ADDRESS, block)

    # -- inspection ----------------------------------------------------------

    def method_calls(self) -> List[str]:
        return [method for _, method, _ in self.calls]

    def packet_timeouts_ms(self) -> List[float]:
        return [detail for _, method, detail in self.calls if method == "setPacketTimeoutMillis"]


def make_bus(
    motors: Sequence[Tuple[str, int, str]] = DEFAULT_MOTORS,
    port: str = "/dev/fake0",
) -> DynamixelBus:
    """A ``DynamixelBus`` over the installed fake SDK (call after ``install``)."""
    return DynamixelBus(
        port=port,
        motors={name: DynamixelMotor(motor_id, name, model) for name, motor_id, model in motors},
    )


@pytest.fixture
def clock() -> FakeClock:
    return FakeClock()


@pytest.fixture
def sdk(clock: FakeClock, monkeypatch: pytest.MonkeyPatch) -> FakeSdk:
    fake = FakeSdk(clock)
    fake.install(monkeypatch)
    for _, motor_id, model in DEFAULT_MOTORS:
        fake.set_register(
            motor_id,
            XControlTable.MODEL_NUMBER,
            DynamixelMotor(motor_id, "tmp", model).model_number,
        )
    return fake


@pytest.fixture
def bus(sdk: FakeSdk) -> DynamixelBus:
    return make_bus()


# --- Simulated Rakuda buses (spec §10.1) ---------------------------------------
#
# The 17 motor names / IDs / models come straight from the arm classes, so a
# change there is reflected here without a second table.


def _rakuda_motors(arm_class: type[RakudaArm]) -> Dict[str, DynamixelMotor]:
    """The motor table an arm class declares, without building a config or a port."""
    return object.__new__(arm_class)._create_motors()


def _make_rakuda_bus(
    motors: Dict[str, DynamixelMotor],
    *,
    port: str,
    gripper_current_limit: int,
    clock: Any | None,
    auto_step: bool,
) -> SimulatedDynamixelBus:
    bus = SimulatedDynamixelBus(motors, port=port, clock=clock, auto_step=auto_step)
    # As a previously connected Rakuda leaves them: grippers in current-based
    # position mode with the gripping-force limit; every arm joint at factory
    # defaults (position mode, full CURRENT_LIMIT, DRIVE_MODE 0).
    for name in RAKUDA_GRIPPER_JOINT_NAMES:
        registers = bus.registers(name)
        registers.set(XControlTable.OPERATING_MODE, OperatingMode.CURRENT_BASED_POSITION)
        registers.set(XControlTable.CURRENT_LIMIT, gripper_current_limit)
    return bus


def make_leader_bus(
    *,
    clock: Any | None = None,
    auto_step: bool = False,
    port: str = "sim://leader",
) -> SimulatedDynamixelBus:
    """A simulated mini-Rakuda leader: 16 x XC330-T288 + torso_yaw XM430-W350."""
    return _make_rakuda_bus(
        _rakuda_motors(RakudaLeader),
        port=port,
        gripper_current_limit=RAKUDA_CONTROLTABLE_VALUES.LEADER_GRIP_CURRENT_LIMIT,
        clock=clock,
        auto_step=auto_step,
    )


def make_follower_bus(
    *,
    clock: Any | None = None,
    auto_step: bool = False,
    port: str = "sim://follower",
) -> SimulatedDynamixelBus:
    """A simulated Rakuda follower: XM540-W270 on IDs 1-4/27, XM430-W350 elsewhere."""
    return _make_rakuda_bus(
        _rakuda_motors(RakudaFollower),
        port=port,
        gripper_current_limit=RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_CURRENT_LIMIT,
        clock=clock,
        auto_step=auto_step,
    )


# --- Leader loop helpers (spec §10.2 test_leader_loop.py) ----------------------
#
# The loop is tested against a stub law so these tests do not depend on
# ``BilateralLaw``; only the ``LawOutput`` container is shared.


class StubLaw:
    """A ``ControlLaw`` returning a settable constant current per joint.

    Attributes:
        current_ma: The current returned by ``compute`` (joint convention, mA).
        gate: The gate value returned by ``compute``.
        error: When set, ``compute`` raises it.
        resets: ``(leader.seq, engaged)`` per ``reset`` call.
        engages: Number of ``engage`` calls (``re_engage()``).
        calls: ``(leader.seq, follower given, follower_age_s, dt)`` per ``compute`` call.
    """

    def __init__(
        self,
        joints: Sequence[str],
        *,
        current_ma: float = 0.0,
        gate: float = 1.0,
        joint_range_counts: Mapping[str, Tuple[int, int]] | None = None,
    ) -> None:
        self.joints: Tuple[str, ...] = tuple(joints)
        self.current_ma = np.full(len(self.joints), float(current_ma))
        self.gate = gate
        self.ramp = 0.0
        self.error: Exception | None = None
        self.resets: List[Tuple[int, bool]] = []
        self.engages = 0
        self.calls: List[Tuple[int, bool, float | None, float]] = []
        if joint_range_counts is not None:
            self.joint_range_counts: Dict[str, Tuple[int, int]] = dict(joint_range_counts)

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        self.resets.append((leader.seq, engaged))

    def engage(self) -> None:
        self.engages += 1

    def gravity_term(self, leader: RakudaArmState) -> NDArray[np.float64]:
        """What ``configure()`` writes at torque-on: the constant current."""
        return self.current_ma.copy()

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        if self.error is not None:
            raise self.error
        self.calls.append((leader.seq, follower is not None, follower_age_s, dt))
        zeros = np.zeros(len(self.joints))
        return LawOutput(
            current_ma=self.current_ma.copy(),
            gravity_ma=self.current_ma.copy(),
            barrier_ma=zeros,
            feedback_ma=zeros.copy(),
            gate=self.gate,
            ramp=self.ramp,
        )


@dataclass
class LoopHarness:
    """A leader loop over simulated buses sharing one clock."""

    clock: SimClock
    bus: SimulatedDynamixelBus
    law: StubLaw
    params: RakudaBilateralParams
    loop: LeaderCurrentLoop
    follower_bus: SimulatedDynamixelBus | None
    follower: FollowerPositionIO | None

    @property
    def joints(self) -> Tuple[str, ...]:
        return self.law.joints

    def run(self, cycles: int = 1) -> None:
        """Runs ``cycles`` cycles, moving the clock to each next deadline like the scheduler."""
        for _ in range(cycles):
            self.loop.run_once(self.clock.monotonic_ns())
            deadline = self.loop.next_deadline_ns
            if deadline is not None and deadline > self.clock.now_ns:
                self.clock.now_ns = deadline


def make_loop(
    joints: Sequence[str] = RAKUDA_ARM_JOINT_NAMES,
    *,
    params: RakudaBilateralParams | None = None,
    current_ma: float = 0.0,
    with_follower: bool = False,
    gripper_hold: Mapping[str, int] | None = None,
    joint_range_counts: Mapping[str, Tuple[int, int]] | None = None,
    main_alive: Callable[[], bool] | None = None,
    current_sign: Mapping[str, int] | None = None,
) -> LoopHarness:
    """A ``LeaderCurrentLoop`` on a simulated leader (and optionally follower) bus.

    ``RETURN_DELAY_TIME`` is 0 on every motor, as on the real arms after
    spec §15.7, so a default preflight has no warnings.
    """
    clock = SimClock()
    params = RakudaBilateralParams(current_joints=tuple(joints)) if params is None else params
    bus = make_leader_bus(clock=clock)
    for name in bus.motors:
        bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
    law = StubLaw(joints, current_ma=current_ma, joint_range_counts=joint_range_counts)
    units = {name: CURRENT_UNIT_MA[bus.motors[name].model_name] for name in joints}
    sign = {name: 1 for name in joints} if current_sign is None else dict(current_sign)
    follower_bus: SimulatedDynamixelBus | None = None
    follower: FollowerPositionIO | None = None
    if with_follower:
        follower_bus = make_follower_bus(clock=clock)
        for name in follower_bus.motors:
            follower_bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
        follower_bus.torque_enabled()
        follower_bus.instruction_log.clear()
        follower = FollowerPositionIO(
            follower_bus, RAKUDA_JOINT_NAMES, params.follower_read_timeout_s
        )
    loop = LeaderCurrentLoop(
        bus,
        joints,
        params,
        law,
        units,
        sign,
        follower,
        clock=clock.monotonic_ns,
        sleep=clock.sleep,
        main_alive=(lambda: True) if main_alive is None else main_alive,
        gripper_hold=gripper_hold,
    )
    return LoopHarness(clock, bus, law, params, loop, follower_bus, follower)
