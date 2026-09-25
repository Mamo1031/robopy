"""A simulated DYNAMIXEL X-series bus for running the Rakuda control stack without hardware.

:class:`SimulatedDynamixelBus` is a drop-in for
:class:`robopy.motor.dynamixel_bus.DynamixelBus` in every method the control,
identification and connection code calls: ``open``/``close``/``set_port``,
``sync_read``/``sync_write``/``read``/``write``, ``torque_enabled``/
``torque_disabled``, ``read_state_block``, ``write_goal_current_raw``,
``write_with_readback``, ``read_diagnostics``, ``read_model_numbers`` and
``verify_models``.  Errors, messages and timing contracts follow the real bus.

Each simulated motor keeps a byte-addressed register image (the X-series
control table) and a toy plant.  The firmware behaviours that bite naive
control code are reproduced on purpose:

* addresses below :data:`EEPROM_END_ADDRESS` are EEPROM: a write is silently
  ignored while ``TORQUE_ENABLE`` is 1 (so ``write_with_readback`` raises), and
  they survive :meth:`SimulatedDynamixelBus.power_cycle`;
* an ``OPERATING_MODE`` change re-initialises ``GOAL_CURRENT`` per
  :data:`MODE_CHANGE_GOAL_CURRENT`, resets ``PROFILE_ACCELERATION``/
  ``PROFILE_VELOCITY`` and the position gains, and switching to position mode
  (3) normalises the multi-turn ``PRESENT_POSITION`` into ``0..4095``;
* ``DRIVE_MODE`` bit 0 (Reverse Mode) flips the direction in which a positive
  ``GOAL_CURRENT`` moves the joint;
* out-of-range values (``GOAL_CURRENT`` beyond ``CURRENT_LIMIT``,
  ``GOAL_POSITION`` outside the position limits in mode 3, ...) are rejected
  without any error reaching the host, exactly like a Data Range error on a
  SyncWrite;
* a motor listed in :attr:`SimulatedDynamixelBus.silent` never answers: the
  deadline-bounded reads advance the clock by their timeout and raise
  :class:`DynamixelTimeoutError`, the legacy ``sync_read`` burns its ten
  retries and raises :class:`DynamixelCommError`.

**This is a toy, not a dynamics model.**  ``PRESENT_CURRENT`` mirrors the
commanded ``GOAL_CURRENT`` (in current mode with torque on) instead of being
measured, the position-mode servo is a first-order lag, and gravity is an
optional per-joint "holding current" (or, through
:meth:`SimulatedDynamixelBus.set_coupled_gravity`, a function of the whole
pose).  Gain stability and identification
accuracy can only be judged on the real arm.
"""

from __future__ import annotations

import logging
import math
import threading
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Dict, List, Literal, Mapping, Sequence, Tuple, Type

import dynamixel_sdk as dxl

from .dynamixel_bus import (
    NUM_READ_RETRY,
    READBACK_TIMEOUT_S,
    DiagnosticReading,
    DynamixelCommError,
    DynamixelMotor,
    DynamixelTimeoutError,
    MotorStateReading,
)
from .dynamixel_control_table import (
    CURRENT_LIMIT_MAX_RAW,
    CURRENT_UNIT_MA,
    EEPROM_END_ADDRESS,
    ControlItem,
    OperatingMode,
    XControlTable,
    cast_value,
    encode_value,
)

__all__ = [
    "LEGACY_READ_TIMEOUT_S",
    "MAX_MULTI_TURN_COUNTS",
    "MODE_CHANGE_GOAL_CURRENT",
    "POWER_ON_GOAL_CURRENT",
    "TORQUE_ON_RESETS_GOAL_CURRENT",
    "VELOCITY_UNIT_COUNTS_PER_S",
    "CoupledGravity",
    "GoalCurrentPolicy",
    "SimClock",
    "SimulatedDynamixelBus",
    "SimulatedJoint",
    "SimulatedMotorRegisters",
]

logger = logging.getLogger(__name__)

GoalCurrentPolicy = Literal["limit", "zero", "keep"]
#: ``fn({motor: position_counts}) -> {motor: holding current}``; see
#: :meth:`SimulatedDynamixelBus.set_coupled_gravity`.
CoupledGravity = Callable[[Mapping[str, float]], Mapping[str, float]]

#: What ``GOAL_CURRENT`` becomes when ``OPERATING_MODE`` changes: ``"limit"``
#: (the value of ``CURRENT_LIMIT``, the worst case that ``configure()`` guards
#: against by reading ``GOAL_CURRENT`` back), ``"zero"`` or ``"keep"``.  Not
#: measured on the real firmware, to be checked during hardware bring-up.
MODE_CHANGE_GOAL_CURRENT: GoalCurrentPolicy = "limit"
#: When True, a ``TORQUE_ENABLE`` 0 -> 1 transition also sets ``GOAL_CURRENT``
#: to ``CURRENT_LIMIT``; the loop's post-torque-on read-back must catch
#: that.  Default False (the e-Manual documents no such reset).
TORQUE_ON_RESETS_GOAL_CURRENT: bool = False
#: ``GOAL_CURRENT`` after a power cycle: ``"zero"`` or ``"limit"``.
POWER_ON_GOAL_CURRENT: GoalCurrentPolicy = "zero"

#: Approximate SDK packet timeout of one legacy ``sync_read`` attempt on a
#: 17-motor bus (ten retries of about 36.6 ms).
LEGACY_READ_TIMEOUT_S: float = 0.0366
#: One raw ``PRESENT_VELOCITY`` count is 0.229 rpm.
VELOCITY_UNIT_COUNTS_PER_S: float = 0.229 * 4096.0 / 60.0
#: Range of ``GOAL_POSITION`` in the multi-turn modes (4 and 5).
MAX_MULTI_TURN_COUNTS: int = 1_048_575

_SUBSTEP_S = 0.001
_MAX_CATCH_UP_S = 1.0
_POSITION_MODES = (
    OperatingMode.POSITION,
    OperatingMode.EXTENDED_POSITION,
    OperatingMode.CURRENT_BASED_POSITION,
)
_ENVIRONMENT_ITEMS = (XControlTable.PRESENT_INPUT_VOLTAGE, XControlTable.PRESENT_TEMPERATURE)


class SimClock:
    """A monotonic clock that only moves when told to.

    It has the same surface as the ``FakeClock`` used by the bus tests, so a
    single clock can be shared between a simulated bus and a control loop.
    """

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


@dataclass
class SimulatedJoint:
    """Toy plant behind one simulated motor, in encoder counts.

    In current mode (torque on) the net drive is ``direction * GOAL_CURRENT``
    minus gravity, contact and friction, all expressed as milliamps; the
    acceleration is ``counts_per_s2_per_ma`` times that net current minus
    viscous damping.  Commanding exactly the gravity current therefore holds
    the joint still, which is what the gravity-compensation tests need.  With
    torque off the joint is driven by gravity and contact alone.  In the
    position modes (3/4/5, torque on) the position tracks ``GOAL_POSITION`` as
    a first-order lag, rate-limited by ``PROFILE_VELOCITY`` when non-zero.

    Attributes:
        position_counts: Current position; multi-turn (unbounded) until an
            ``OPERATING_MODE`` 3 switch or a power cycle normalises it.
        velocity_counts_per_s: Current velocity.
        counts_per_s2_per_ma: Acceleration per milliamp of net current.
        damping_per_s: Viscous damping coefficient.
        coulomb_ma: Static/kinetic friction as an equivalent current.
        gravity_ma: Current, in the count frame (before ``DRIVE_MODE``
            reversal), that exactly balances gravity; a constant or a function
            of ``position_counts``.  Positive means gravity pulls toward
            smaller counts.
        contact_lower_counts: A wall below this position pushes back with
            ``contact_stiffness_ma_per_count`` per count of penetration.
        contact_upper_counts: A wall above this position, likewise.
        contact_stiffness_ma_per_count: Stiffness of both walls.
        position_time_constant_s: Time constant of the position-mode lag.
    """

    position_counts: float = 2048.0
    velocity_counts_per_s: float = 0.0
    counts_per_s2_per_ma: float = 100.0
    damping_per_s: float = 10.0
    coulomb_ma: float = 0.0
    gravity_ma: float | Callable[[float], float] = 0.0
    contact_lower_counts: float | None = None
    contact_upper_counts: float | None = None
    contact_stiffness_ma_per_count: float = 5.0
    position_time_constant_s: float = 0.02

    def gravity_at(self, position_counts: float) -> float:
        """Gravity holding current at ``position_counts``."""
        if callable(self.gravity_ma):
            return float(self.gravity_ma(position_counts))
        return float(self.gravity_ma)

    def contact_at(self, position_counts: float) -> float:
        """Reaction current of the contact walls at ``position_counts``."""
        reaction = 0.0
        if self.contact_upper_counts is not None and position_counts > self.contact_upper_counts:
            reaction -= self.contact_stiffness_ma_per_count * (
                position_counts - self.contact_upper_counts
            )
        if self.contact_lower_counts is not None and position_counts < self.contact_lower_counts:
            reaction += self.contact_stiffness_ma_per_count * (
                self.contact_lower_counts - position_counts
            )
        return reaction


@dataclass(frozen=True)
class _ModelDefaults:
    firmware_version: int
    temperature_limit: int
    position_p_gain: int


_MODEL_DEFAULTS: Dict[str, _ModelDefaults] = {
    "xc330-t288": _ModelDefaults(49, 70, 900),
    "xm430-w350": _ModelDefaults(46, 80, 800),
    "xm540-w270": _ModelDefaults(46, 80, 800),
}
_FALLBACK_DEFAULTS = _ModelDefaults(46, 80, 800)
_FALLBACK_CURRENT_UNIT_MA = 2.69
_FALLBACK_CURRENT_LIMIT_MAX = 1193
_DEFAULT_VOLTAGE_DV = 118
_DEFAULT_TEMPERATURE_C = 40


class SimulatedMotorRegisters:
    """Register image and plant of one simulated motor, for test assertions.

    :meth:`get`/:meth:`set` are the *physical* side of the register file: they
    bypass the EEPROM/torque/range rules that host writes go through, so tests
    can seed any state (a wrong ``MODEL_NUMBER``, a high temperature, a
    multi-turn ``PRESENT_POSITION``).  Setting ``PRESENT_POSITION`` or
    ``PRESENT_VELOCITY`` moves the plant; ``PRESENT_CURRENT`` and ``MOVING``
    are derived and overwritten on the next read.
    """

    def __init__(self, motor: DynamixelMotor, joint: SimulatedJoint) -> None:
        self.motor = motor
        self.joint = joint
        self.image = bytearray(256)
        self.current_unit_ma: float = CURRENT_UNIT_MA.get(
            motor.model_name, _FALLBACK_CURRENT_UNIT_MA
        )
        self.current_limit_max: int = CURRENT_LIMIT_MAX_RAW.get(
            motor.model_name, _FALLBACK_CURRENT_LIMIT_MAX
        )
        self._defaults = _MODEL_DEFAULTS.get(motor.model_name, _FALLBACK_DEFAULTS)
        self.reset_eeprom()
        self.set(XControlTable.PRESENT_INPUT_VOLTAGE, _DEFAULT_VOLTAGE_DV)
        self.set(XControlTable.PRESENT_TEMPERATURE, _DEFAULT_TEMPERATURE_C)
        self.reset_ram(goal_current=0)

    # -- raw access ---------------------------------------------------------

    def get(self, item: XControlTable) -> int:
        """Decoded value of ``item`` (signed for signed items)."""
        control_item: ControlItem = item.value
        raw = int.from_bytes(
            self.image[control_item.address : control_item.address + control_item.num_bytes],
            "little",
        )
        return cast_value(raw, control_item.dtype)

    def set(self, item: XControlTable, value: int) -> None:
        """Stores ``value`` without any firmware rule; see the class docstring."""
        if item is XControlTable.PRESENT_POSITION:
            self.joint.position_counts = float(value)
        elif item is XControlTable.PRESENT_VELOCITY:
            self.joint.velocity_counts_per_s = value * VELOCITY_UNIT_COUNTS_PER_S
        control_item: ControlItem = item.value
        word = encode_value(int(value), control_item.dtype)
        self.image[control_item.address : control_item.address + control_item.num_bytes] = (
            word.to_bytes(control_item.num_bytes, "little")
        )

    def reset_eeprom(self) -> None:
        """Factory EEPROM: position mode, no offset, full current limit."""
        self.set(XControlTable.MODEL_NUMBER, self.motor.model_number)
        self.set(XControlTable.FIRMWARE_VERSION, self._defaults.firmware_version)
        self.set(XControlTable.ID, self.motor.id)
        self.set(XControlTable.BAUD_RATE, 3)  # 1 Mbps
        self.set(XControlTable.RETURN_DELAY_TIME, 250)
        self.set(XControlTable.DRIVE_MODE, 0)
        self.set(XControlTable.OPERATING_MODE, OperatingMode.POSITION)
        self.set(XControlTable.HOMING_OFFSET, 0)
        self.set(XControlTable.TEMPERATURE_LIMIT, self._defaults.temperature_limit)
        self.set(XControlTable.CURRENT_LIMIT, self.current_limit_max)
        self.set(XControlTable.MAX_POSITION_LIMIT, 4095)
        self.set(XControlTable.MIN_POSITION_LIMIT, 0)
        self.set(XControlTable.SHUTDOWN, 0x34)

    def reset_ram(self, *, goal_current: int) -> None:
        """Power-on RAM: torque off, gains at default, goal = present position."""
        environment = {item: self.get(item) for item in _ENVIRONMENT_ITEMS}
        self.image[EEPROM_END_ADDRESS:] = bytes(256 - EEPROM_END_ADDRESS)
        for item, value in environment.items():
            self.set(item, value)
        self.set(XControlTable.POSITION_P_GAIN, self._defaults.position_p_gain)
        self.set(XControlTable.GOAL_CURRENT, goal_current)
        self.set(XControlTable.GOAL_POSITION, self.present_position)
        self.refresh_present()

    def reset_gains(self) -> None:
        self.set(XControlTable.POSITION_P_GAIN, self._defaults.position_p_gain)
        self.set(XControlTable.POSITION_I_GAIN, 0)
        self.set(XControlTable.POSITION_D_GAIN, 0)

    def refresh_present(self) -> None:
        """Copies the plant state into the ``PRESENT_*`` registers."""
        self.set(XControlTable.PRESENT_CURRENT, self.present_current)
        control_item: ControlItem = XControlTable.PRESENT_VELOCITY.value
        word = encode_value(self.present_velocity, control_item.dtype)
        self.image[control_item.address : control_item.address + 4] = word.to_bytes(4, "little")
        control_item = XControlTable.PRESENT_POSITION.value
        word = encode_value(self.present_position, control_item.dtype)
        self.image[control_item.address : control_item.address + 4] = word.to_bytes(4, "little")
        moving = abs(self.joint.velocity_counts_per_s) >= VELOCITY_UNIT_COUNTS_PER_S
        self.set(XControlTable.MOVING, int(moving))

    def snapshot(self) -> Dict[str, int]:
        """``{item_name: value}`` for every control-table item, for diffing."""
        self.refresh_present()
        return {item.name: self.get(item) for item in XControlTable}

    # -- named views --------------------------------------------------------

    @property
    def operating_mode(self) -> int:
        return self.get(XControlTable.OPERATING_MODE)

    @property
    def torque_enable(self) -> int:
        return self.get(XControlTable.TORQUE_ENABLE)

    @property
    def goal_current(self) -> int:
        return self.get(XControlTable.GOAL_CURRENT)

    @property
    def goal_position(self) -> int:
        return self.get(XControlTable.GOAL_POSITION)

    @property
    def current_limit(self) -> int:
        return self.get(XControlTable.CURRENT_LIMIT)

    @property
    def drive_mode(self) -> int:
        return self.get(XControlTable.DRIVE_MODE)

    @property
    def direction(self) -> int:
        """``-1`` when ``DRIVE_MODE`` bit 0 (Reverse Mode) is set, else ``1``."""
        return -1 if self.drive_mode & 0x01 else 1

    @property
    def bus_watchdog(self) -> int:
        return self.get(XControlTable.BUS_WATCHDOG)

    @property
    def profile_velocity(self) -> int:
        return self.get(XControlTable.PROFILE_VELOCITY)

    @property
    def profile_acceleration(self) -> int:
        return self.get(XControlTable.PROFILE_ACCELERATION)

    @property
    def hardware_error_status(self) -> int:
        return self.get(XControlTable.HARDWARE_ERROR_STATUS)

    @property
    def present_position(self) -> int:
        return int(round(self.joint.position_counts))

    @property
    def present_velocity(self) -> int:
        return int(round(self.joint.velocity_counts_per_s / VELOCITY_UNIT_COUNTS_PER_S))

    @property
    def present_current(self) -> int:
        """Toy: the commanded current while torque is on in current mode, else 0."""
        if self.torque_enable and self.operating_mode == OperatingMode.CURRENT:
            return self.goal_current
        return 0


@dataclass
class _SimPortHandler:
    """Stand-in for the SDK port handler: just a name and an open flag."""

    port_name: str
    is_open: bool = False


class SimulatedDynamixelBus:
    """A stand-in for :class:`robopy.motor.dynamixel_bus.DynamixelBus`.

    Attributes:
        motors: ``{name: DynamixelMotor}``, as on the real bus.
        clock: Object with ``monotonic_ns()`` and ``advance(seconds)`` (or
            ``sleep``); every timestamp and timeout uses it.
        silent: Names of motors that do not answer any read.
        instruction_log: Every host write, in order, as ``(item_name, {name: value})``
            with the values as sent (rejected writes are logged too).
        rejected_writes: ``(item_name, motor_name, value, reason)`` for every
            write the firmware would have ignored.
        transaction_count: Number of bus transactions (reads and writes).
        read_duration_s / write_duration_s: Simulated time consumed by a
            successful transaction (default 0, so timing assertions stay exact).
        mode_change_goal_current / torque_on_resets_goal_current /
        power_on_goal_current: Per-instance copies of the module policies.
    """

    def __init__(
        self,
        motors: Mapping[str, DynamixelMotor],
        *,
        joints: Mapping[str, SimulatedJoint] | None = None,
        port: str = "sim://dynamixel",
        clock: Any | None = None,
        auto_step: bool = True,
        mode_change_goal_current: GoalCurrentPolicy = MODE_CHANGE_GOAL_CURRENT,
        torque_on_resets_goal_current: bool = TORQUE_ON_RESETS_GOAL_CURRENT,
        power_on_goal_current: GoalCurrentPolicy = POWER_ON_GOAL_CURRENT,
    ) -> None:
        """Creates a simulated bus.

        Args:
            motors: ``{name: DynamixelMotor}``, exactly as the real bus takes.
            joints: Per-motor plant; missing motors get a default
                :class:`SimulatedJoint` at 2048 counts.
            port: Cosmetic port name (``set_port`` changes it).
            clock: Shared clock; defaults to a fresh :class:`SimClock`.
            auto_step: Integrate the plant by the simulated time elapsed since
                the last transaction on every transaction and on
                :meth:`advance`.  ``False`` moves the plant only through
                :meth:`step`.
            mode_change_goal_current: See :data:`MODE_CHANGE_GOAL_CURRENT`.
            torque_on_resets_goal_current: See :data:`TORQUE_ON_RESETS_GOAL_CURRENT`.
            power_on_goal_current: See :data:`POWER_ON_GOAL_CURRENT`.
        """
        self.motors: Dict[str, DynamixelMotor] = dict(motors)
        self.port_handler = _SimPortHandler(port)
        self.clock: Any = SimClock() if clock is None else clock
        self.auto_step = auto_step
        self.mode_change_goal_current: GoalCurrentPolicy = mode_change_goal_current
        self.torque_on_resets_goal_current = torque_on_resets_goal_current
        self.power_on_goal_current: GoalCurrentPolicy = power_on_goal_current
        self.calibration: Dict[str, Tuple[int, bool]] = {}
        self.silent: set[str] = set()
        self.instruction_log: List[Tuple[str, Dict[str, int]]] = []
        self.rejected_writes: List[Tuple[str, str, int, str]] = []
        self.transaction_count = 0
        self.read_duration_s = 0.0
        self.write_duration_s = 0.0
        self._lock = threading.RLock()
        self._registers: Dict[str, SimulatedMotorRegisters] = {
            name: SimulatedMotorRegisters(motor, (joints or {}).get(name) or SimulatedJoint())
            for name, motor in self.motors.items()
        }
        self._coupled_gravity: CoupledGravity | None = None
        self._plant_ns = self.clock.monotonic_ns()

    # -- test controls ------------------------------------------------------

    def registers(self, motor_name: str) -> SimulatedMotorRegisters:
        """Register file and plant of ``motor_name``."""
        if motor_name not in self._registers:
            raise ValueError(f"Unknown motor '{motor_name}' on {self.port_handler.port_name}.")
        return self._registers[motor_name]

    def joint(self, motor_name: str) -> SimulatedJoint:
        """The plant behind ``motor_name``."""
        return self.registers(motor_name).joint

    def set_fault(self, motor_name: str, hardware_error_status: int) -> None:
        """Latches ``HARDWARE_ERROR_STATUS`` on ``motor_name``.

        Bits that are also set in ``SHUTDOWN`` switch torque off, and the motor
        refuses ``TORQUE_ENABLE=1`` until :meth:`power_cycle` (or a fault of 0).
        """
        registers = self.registers(motor_name)
        registers.set(XControlTable.HARDWARE_ERROR_STATUS, hardware_error_status)
        if hardware_error_status & registers.get(XControlTable.SHUTDOWN):
            registers.set(XControlTable.TORQUE_ENABLE, 0)

    def power_cycle(self) -> None:
        """Re-initialises RAM on every motor and keeps EEPROM.

        Torque goes off, ``GOAL_CURRENT`` follows ``power_on_goal_current``,
        hardware errors clear, and the multi-turn position collapses to the
        absolute 12-bit encoder reading (``0..4095``).
        """
        with self._lock:
            for registers in self._registers.values():
                registers.joint.position_counts %= 4096.0
                registers.joint.velocity_counts_per_s = 0.0
                goal_current = (
                    registers.current_limit if self.power_on_goal_current == "limit" else 0
                )
                registers.reset_ram(goal_current=goal_current)
            self._plant_ns = self.clock.monotonic_ns()

    def advance(self, seconds: float) -> None:
        """Moves the clock forward (and the plant, when ``auto_step`` is on)."""
        self._advance_clock(seconds)
        self._catch_up()

    def set_coupled_gravity(self, fn: CoupledGravity | None) -> None:
        """Replaces the per-joint gravity by a function of every joint's position.

        ``fn({name: position_counts})`` returns ``{name: holding current}`` in
        the count frame (the convention of :attr:`SimulatedJoint.gravity_ma`)
        for the joints whose gravity depends on the whole arm pose; joints it
        does not return keep their own ``gravity_ma``.  It is evaluated once
        per :meth:`step` at the positions the step starts from.  ``None``
        restores the per-joint gravity everywhere.
        """
        with self._lock:
            self._coupled_gravity = fn

    def step(self, dt: float) -> None:
        """Integrates every joint by ``dt`` seconds under its current command."""
        if dt <= 0.0:
            return
        with self._lock:
            coupled: Mapping[str, float] = {}
            if self._coupled_gravity is not None:
                coupled = self._coupled_gravity(
                    {name: r.joint.position_counts for name, r in self._registers.items()}
                )
                unknown = [name for name in coupled if name not in self._registers]
                if unknown:
                    raise ValueError(f"coupled gravity names unknown motor(s): {unknown}")
            for name, registers in self._registers.items():
                gravity = coupled.get(name)
                self._integrate(registers, dt, None if gravity is None else float(gravity))
            self._plant_ns = self.clock.monotonic_ns()

    # -- bus surface: lifecycle ----------------------------------------------

    def open(self, baudrate: int = 1_000_000) -> None:
        """Marks the simulated port open."""
        self.port_handler.is_open = True
        logger.info(f"Opened simulated port {self.port_handler.port_name} (Baudrate: {baudrate})")

    def close(self) -> None:
        self.port_handler.is_open = False
        logger.info(f"Closed simulated port {self.port_handler.port_name}.")

    def __enter__(self) -> "SimulatedDynamixelBus":
        self.open()
        return self

    def __exit__(
        self,
        exc_type: Type[BaseException] | None,
        exc_val: BaseException | None,
        exc_tb: TracebackType | None,
    ) -> None:
        self.close()

    def set_port(self, port: str) -> None:
        """Changes the port name; only allowed while the bus is closed."""
        if self.port_handler.is_open:
            raise RuntimeError(
                f"Cannot change the port while {self.port_handler.port_name} is open."
            )
        self.port_handler = _SimPortHandler(port)

    def set_calibration(self, calibration_data: Dict[str, Tuple[int, bool]]) -> None:
        """Stored for parity; the simulated bus works in raw counts only."""
        self.calibration = calibration_data

    def __len__(self) -> int:
        return len(self.motors)

    def __repr__(self) -> str:
        motor_list = ", ".join(self.motors)
        return f"SimulatedDynamixelBus(port={self.port_handler.port_name}, motors=[{motor_list}])"

    # -- bus surface: legacy API --------------------------------------------

    def sync_write(self, item: Enum, values: Mapping[str, int | float]) -> None:
        """Writes one item to several motors; unknown names are skipped as on the real bus.

        Firmware rules apply per motor (EEPROM under torque, Data Range) and a
        refused value is recorded in ``rejected_writes``.  A value that does not
        fit the item's wire type raises ``ValueError`` here, where the real bus
        would transmit a truncated word.
        """
        control_item = self._control_item(item)
        names = [name for name in self.motors if name in values]
        with self._lock:
            self._begin_transaction()
            requested = {name: int(values[name]) for name in names}
            self.instruction_log.append((item.name, requested))
            for name, value in requested.items():
                self._write_item(name, item, control_item, value)
            self._advance_clock(self.write_duration_s)

    def sync_read(self, item: Enum, motor_names: List[str]) -> Dict[str, Any]:
        """Reads one item from several motors with the legacy retry-until-dead contract."""
        self._control_item(item)
        names = [name for name in motor_names if name in self.motors]
        with self._lock:
            self._begin_transaction()
            if not names:
                raise DynamixelCommError(
                    f"Failed to sync read {item.name}.", dxl.COMM_NOT_AVAILABLE
                )
            if any(name in self.silent for name in names):
                self._advance_clock(NUM_READ_RETRY * LEGACY_READ_TIMEOUT_S)
                raise DynamixelCommError(f"Failed to sync read {item.name}.", dxl.COMM_RX_TIMEOUT)
            self._advance_clock(self.read_duration_s)
            return {name: self._read_item(name, item) for name in names}

    def read(self, item: Enum, motor_name: str) -> Any:
        return self.sync_read(item, [motor_name]).get(motor_name)

    def write(self, item: Enum, motor_name: str, value: int | float) -> None:
        self.sync_write(item, {motor_name: value})

    def torque_disabled(self, specific_motor_names: List[str] | None = None) -> None:
        names = list(self.motors) if specific_motor_names is None else specific_motor_names
        self.sync_write(XControlTable.TORQUE_ENABLE, {name: 0 for name in names})

    def torque_enabled(self, specific_motor_names: List[str] | None = None) -> None:
        names = list(self.motors) if specific_motor_names is None else specific_motor_names
        self.sync_write(XControlTable.TORQUE_ENABLE, {name: 1 for name in names})

    # -- bus surface: deadline-bounded API ----------------------------------

    def read_state_block(
        self,
        motor_names: Sequence[str],
        *,
        timeout_s: float,
        attempts: int = 1,
    ) -> Tuple[Dict[str, MotorStateReading], int, int]:
        """Reads current, velocity and position of many motors in one transaction.

        A silent motor makes every attempt time out: the clock advances by
        ``timeout_s`` per attempt and :class:`DynamixelTimeoutError` is raised
        with the real bus's message.
        """
        names = self._select_names(motor_names)
        if timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}.")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}.")
        with self._lock:
            start_ns = self.clock.monotonic_ns()
            self._begin_transaction()
            self._raise_if_silent(names, timeout_s, attempts, "State block read")
            self._advance_clock(self.read_duration_s)
            readings = {}
            for name in names:
                registers = self._registers[name]
                registers.refresh_present()
                readings[name] = MotorStateReading(
                    position=registers.present_position,
                    velocity=registers.present_velocity,
                    current_raw=registers.present_current,
                )
            end_ns = self.clock.monotonic_ns()
        return readings, start_ns, end_ns

    def write_goal_current_raw(self, values: Mapping[str, int]) -> None:
        """Writes signed raw ``GOAL_CURRENT`` counts in one transaction, no read-back.

        Values beyond ``CURRENT_LIMIT`` are dropped by the motor (Data Range),
        not clipped; values outside INT16 raise ``ValueError`` before anything
        is "sent".
        """
        unknown = [name for name in values if name not in self.motors]
        if unknown:
            raise ValueError(f"Unknown motor(s) on {self.port_handler.port_name}: {unknown}")
        names = [name for name in self.motors if name in values]
        if not names:
            raise ValueError("write_goal_current_raw needs at least one motor.")
        requested: Dict[str, int] = {}
        for name in names:
            try:
                encode_value(int(values[name]), XControlTable.GOAL_CURRENT.value.dtype)
            except ValueError as exc:
                raise ValueError(f"GOAL_CURRENT for '{name}': {exc}") from exc
            requested[name] = int(values[name])
        control_item: ControlItem = XControlTable.GOAL_CURRENT.value
        with self._lock:
            self._begin_transaction()
            self.instruction_log.append((XControlTable.GOAL_CURRENT.name, requested))
            for name, value in requested.items():
                self._write_item(name, XControlTable.GOAL_CURRENT, control_item, value)
            self._advance_clock(self.write_duration_s)

    def write_with_readback(
        self,
        item: Enum,
        values: Mapping[str, int | float],
        *,
        tolerance: int = 0,
        attempts: int = 5,
        settle_s: float = 0.005,
    ) -> None:
        """Writes and confirms each motor accepted it, exactly like the real bus.

        Each attempt is one ``sync_write`` of the still-unconfirmed motors, a
        ``settle_s`` pause, then one single-motor read-back per motor (a silent
        motor costs ``READBACK_TIMEOUT_S`` and reads as ``None``).

        Raises:
            TypeError: If ``item`` is not a control-table member.
            ValueError: For an unknown motor or fewer than one attempt.
            DynamixelCommError: When motors still mismatch or did not answer
                after ``attempts`` rounds.
        """
        self._control_item(item)
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}.")
        names = self._select_names(list(values))
        pending: Dict[str, int] = {name: int(values[name]) for name in names}
        mismatched: Dict[str, Tuple[int, int | None]] = {}
        for _ in range(attempts):
            self.sync_write(item, dict(pending))
            self._advance_clock(settle_s)
            mismatched = {}
            for name, wanted in pending.items():
                got = self._read_single(name, item)
                if got is None or abs(got - wanted) > tolerance:
                    mismatched[name] = (wanted, got)
            if not mismatched:
                return
            pending = {name: pending[name] for name in mismatched}
        silent = [name for name, (_, got) in mismatched.items() if got is None]
        raise DynamixelCommError(
            f"{item.name} read-back on {self.port_handler.port_name} still differs after "
            f"{attempts} attempt(s): {mismatched} (wanted, read; None = no response).",
            dxl.COMM_RX_TIMEOUT if silent else dxl.COMM_NOT_AVAILABLE,
        )

    def read_diagnostics(
        self,
        motor_names: Sequence[str],
        *,
        timeout_s: float = 0.05,
    ) -> Dict[str, DiagnosticReading]:
        """Hardware error status, input voltage and temperature (two reads)."""
        names = self._select_names(motor_names)
        with self._lock:
            self._begin_transaction()
            self._raise_if_silent(names, timeout_s, 1, "Hardware error status read")
            self._advance_clock(self.read_duration_s)
            self._begin_transaction()
            self._advance_clock(self.read_duration_s)
            return {
                name: DiagnosticReading(
                    hardware_error_status=self._read_item(
                        name, XControlTable.HARDWARE_ERROR_STATUS
                    ),
                    voltage_v=self._read_item(name, XControlTable.PRESENT_INPUT_VOLTAGE) / 10.0,
                    temperature_c=self._read_item(name, XControlTable.PRESENT_TEMPERATURE),
                )
                for name in names
            }

    def read_model_numbers(self, motor_names: Sequence[str]) -> Dict[str, int]:
        """``MODEL_NUMBER`` of each motor that answers, read one motor at a time."""
        found: Dict[str, int] = {}
        for name in self._select_names(motor_names):
            model_number = self._read_single(name, XControlTable.MODEL_NUMBER)
            if model_number is not None:
                found[name] = model_number
        return found

    def verify_models(self, motor_names: Sequence[str] | None = None) -> None:
        """Checks every motor reports the model its configuration declares.

        Raises:
            ConnectionError: Listing every silent or mismatched motor, in the
                real bus's wording.
        """
        names = list(self.motors) if motor_names is None else list(motor_names)
        self._select_names(names)
        found = self.read_model_numbers(names)
        problems: List[str] = []
        for name in names:
            motor = self.motors[name]
            if name not in found:
                problems.append(f"{name} (ID {motor.id}): no response")
            elif found[name] != motor.model_number:
                problems.append(
                    f"{name} (ID {motor.id}): declared {motor.model_name} "
                    f"(model {motor.model_number}) but the motor reports {found[name]}"
                )
        if problems:
            raise ConnectionError(
                f"Motor model check failed on {self.port_handler.port_name}: " + "; ".join(problems)
            )
        logger.info("Verified %d motor model(s) on %s.", len(names), self.port_handler.port_name)

    # -- internals: bookkeeping ---------------------------------------------

    @staticmethod
    def _control_item(item: Enum) -> ControlItem:
        if not isinstance(item.value, ControlItem):
            raise TypeError("Item must be an Enum member with a ControlItem value.")
        return item.value

    def _select_names(self, motor_names: Sequence[str]) -> List[str]:
        unknown = [name for name in motor_names if name not in self.motors]
        if unknown:
            raise ValueError(f"Unknown motor(s) on {self.port_handler.port_name}: {unknown}")
        if not motor_names:
            raise ValueError("At least one motor name is required.")
        return list(motor_names)

    def _begin_transaction(self) -> None:
        self.transaction_count += 1
        self._catch_up()

    def _advance_clock(self, seconds: float) -> None:
        if seconds <= 0.0:
            return
        advance = getattr(self.clock, "advance", None)
        if advance is not None:
            advance(seconds)
        else:
            self.clock.sleep(seconds)

    def _catch_up(self) -> None:
        """Integrates the plant up to the current simulated time (``auto_step``)."""
        now_ns = self.clock.monotonic_ns()
        if not self.auto_step:
            self._plant_ns = now_ns
            return
        dt = min((now_ns - self._plant_ns) / 1e9, _MAX_CATCH_UP_S)
        self._plant_ns = now_ns
        if dt > 0.0:
            self.step(dt)

    def _raise_if_silent(
        self, names: Sequence[str], timeout_s: float, attempts: int, what: str
    ) -> None:
        silent = [name for name in names if name in self.silent]
        if not silent:
            return
        for _ in range(attempts):
            self._advance_clock(timeout_s)
        raise DynamixelTimeoutError(
            f"{what} on {self.port_handler.port_name}: no complete response within "
            f"{attempts} attempt(s) of {timeout_s * 1e3:.1f} ms; no data from {silent}.",
            dxl.COMM_RX_TIMEOUT,
        )

    def _read_single(self, name: str, item: Enum) -> int | None:
        """One-motor read with the fixed read-back timeout; ``None`` if silent."""
        with self._lock:
            self._begin_transaction()
            if name in self.silent:
                self._advance_clock(READBACK_TIMEOUT_S)
                return None
            self._advance_clock(self.read_duration_s)
            return self._read_item(name, item)

    def _read_item(self, name: str, item: Enum) -> int:
        registers = self._registers[name]
        registers.refresh_present()
        return registers.get(self._as_x_item(item))

    @staticmethod
    def _as_x_item(item: Enum) -> XControlTable:
        if isinstance(item, XControlTable):
            return item
        raise ValueError(f"The simulated bus only knows the X-series table, not {item!r}.")

    # -- internals: firmware rules ------------------------------------------

    def _write_item(self, name: str, item: Enum, control_item: ControlItem, value: int) -> None:
        registers = self._registers[name]
        x_item = self._as_x_item(item)
        reason = self._rejection_reason(registers, x_item, control_item, value)
        if reason is not None:
            self.rejected_writes.append((item.name, name, value, reason))
            logger.debug("Simulated %s ignored %s=%d: %s", name, item.name, value, reason)
            return
        if x_item is XControlTable.OPERATING_MODE:
            self._change_mode(registers, value)
        elif x_item is XControlTable.TORQUE_ENABLE:
            self._set_torque(registers, value)
        else:
            registers.set(x_item, value)

    @staticmethod
    def _rejection_reason(
        registers: SimulatedMotorRegisters,
        item: XControlTable,
        control_item: ControlItem,
        value: int,
    ) -> str | None:
        if control_item.access == "R":
            return "read-only register"
        encode_value(value, control_item.dtype)  # ValueError: does not fit the wire format
        if control_item.address < EEPROM_END_ADDRESS and registers.torque_enable:
            return "EEPROM write while TORQUE_ENABLE is 1"
        if item is XControlTable.OPERATING_MODE:
            if value not in tuple(OperatingMode):
                return "unsupported operating mode"
        elif item is XControlTable.TORQUE_ENABLE:
            if value not in (0, 1):
                return "TORQUE_ENABLE must be 0 or 1"
            if value == 1 and registers.hardware_error_status & registers.get(
                XControlTable.SHUTDOWN
            ):
                return "hardware error latched; reboot required"
        elif item is XControlTable.GOAL_CURRENT:
            if abs(value) > registers.current_limit:
                return f"GOAL_CURRENT beyond CURRENT_LIMIT {registers.current_limit}"
        elif item is XControlTable.CURRENT_LIMIT:
            if not 0 <= value <= registers.current_limit_max:
                return f"CURRENT_LIMIT beyond {registers.current_limit_max}"
        elif item is XControlTable.GOAL_POSITION:
            mode = registers.operating_mode
            if mode == OperatingMode.POSITION:
                low = registers.get(XControlTable.MIN_POSITION_LIMIT)
                high = registers.get(XControlTable.MAX_POSITION_LIMIT)
                if not low <= value <= high:
                    return f"GOAL_POSITION outside position limits {low}..{high}"
            elif abs(value) > MAX_MULTI_TURN_COUNTS:
                return "GOAL_POSITION outside the multi-turn range"
        elif item is XControlTable.BUS_WATCHDOG:
            if not 0 <= value <= 127:
                return "BUS_WATCHDOG must be 0..127"
        return None

    def _change_mode(self, registers: SimulatedMotorRegisters, value: int) -> None:
        if value == registers.operating_mode:
            return
        registers.set(XControlTable.OPERATING_MODE, value)
        if value == OperatingMode.POSITION:
            registers.joint.position_counts %= 4096.0
        registers.set(XControlTable.GOAL_POSITION, registers.present_position)
        registers.set(XControlTable.PROFILE_ACCELERATION, 0)
        registers.set(XControlTable.PROFILE_VELOCITY, 0)
        registers.reset_gains()
        if self.mode_change_goal_current == "limit":
            registers.set(XControlTable.GOAL_CURRENT, registers.current_limit)
        elif self.mode_change_goal_current == "zero":
            registers.set(XControlTable.GOAL_CURRENT, 0)

    def _set_torque(self, registers: SimulatedMotorRegisters, value: int) -> None:
        if value == 1 and not registers.torque_enable and self.torque_on_resets_goal_current:
            registers.set(XControlTable.GOAL_CURRENT, registers.current_limit)
        registers.set(XControlTable.TORQUE_ENABLE, value)

    # -- internals: plant ----------------------------------------------------

    def _integrate(
        self, registers: SimulatedMotorRegisters, dt: float, gravity_ma: float | None = None
    ) -> None:
        """One ``step`` of one joint; ``gravity_ma`` overrides the joint's own gravity."""
        joint = registers.joint
        torque_on = registers.torque_enable == 1
        mode = registers.operating_mode
        if torque_on and mode in _POSITION_MODES:
            self._track_goal_position(registers, dt)
            return
        command_ma = 0.0
        if torque_on and mode == OperatingMode.CURRENT:
            command_ma = registers.direction * registers.goal_current * registers.current_unit_ma
        remaining = dt
        while remaining > 1e-12:
            h = min(_SUBSTEP_S, remaining)
            remaining -= h
            position = joint.position_counts
            velocity = joint.velocity_counts_per_s
            gravity = joint.gravity_at(position) if gravity_ma is None else gravity_ma
            net_ma = command_ma - gravity + joint.contact_at(position)
            if abs(velocity) > 1e-9:
                net_ma -= math.copysign(joint.coulomb_ma, velocity)
            elif abs(net_ma) <= joint.coulomb_ma:
                net_ma = 0.0
            acceleration = joint.counts_per_s2_per_ma * net_ma - joint.damping_per_s * velocity
            velocity += acceleration * h
            joint.velocity_counts_per_s = velocity
            joint.position_counts = position + velocity * h

    @staticmethod
    def _track_goal_position(registers: SimulatedMotorRegisters, dt: float) -> None:
        joint = registers.joint
        error = registers.goal_position - joint.position_counts
        tau = joint.position_time_constant_s
        delta = error if tau <= 0.0 else error * (1.0 - math.exp(-dt / tau))
        profile_velocity = registers.profile_velocity
        if profile_velocity > 0:
            max_step = profile_velocity * VELOCITY_UNIT_COUNTS_PER_S * dt
            delta = max(-max_step, min(max_step, delta))
        joint.velocity_counts_per_s = delta / dt
        joint.position_counts += delta
