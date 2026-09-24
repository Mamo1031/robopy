# robopy/motors.py

"""
DynamixelBus for managing multiple motors and DynamixelMotor for holding
individual motor information. This design promotes modularity and easy
integration into larger robotic systems.
"""

import logging
import time
from dataclasses import dataclass
from enum import Enum
from types import TracebackType
from typing import Any, Callable, Dict, List, Mapping, Sequence, Tuple, Type

import dynamixel_sdk as dxl
import numpy as np
import serial
from numpy.typing import NDArray

from .dynamixel_control_table import (
    STATE_BLOCK_NUM_BYTES,
    STATE_BLOCK_START_ADDRESS,
    ControlItem,
    Dtype,
    XControlTable,
    cast_value,
    encode_value,
    get_model_definition,
)

logger = logging.getLogger(__name__)

# Constants from the original script
BAUDRATE = 1_000_000
PROTOCOL_VERSION = 2.0
NUM_READ_RETRY = 10
NUM_WRITE_RETRY = 2  # 10から2に削減 (パフォーマンス向上のため)

#: Packet timeout of the single-motor read-back that follows a configuration
#: write (``write_with_readback``) and of the per-motor model-number reads.
READBACK_TIMEOUT_S = 0.040
#: pyserial write timeout applied by ``open()``.  A write that cannot complete
#: within this time raises instead of stalling the control thread (spec D32).
SERIAL_WRITE_TIMEOUT_S = 0.02

#: SDK results that mean "the motors did not answer in time"; they are retried
#: and finally reported as :class:`DynamixelTimeoutError`.  Every other failure
#: (``COMM_TX_*``, ``COMM_PORT_BUSY``, ``COMM_NOT_AVAILABLE``) is a host-side
#: problem and raises :class:`DynamixelCommError` at once.
_RX_RESULTS = frozenset(
    {dxl.COMM_RX_FAIL, dxl.COMM_RX_WAITING, dxl.COMM_RX_TIMEOUT, dxl.COMM_RX_CORRUPT}
)

# ``PRESENT_INPUT_VOLTAGE`` (144..145) and ``PRESENT_TEMPERATURE`` (146) are
# adjacent, so ``read_diagnostics`` fetches both in one 3-byte block.
_ENVIRONMENT_BLOCK_START_ADDRESS: int = XControlTable.PRESENT_INPUT_VOLTAGE.value.address
_ENVIRONMENT_BLOCK_NUM_BYTES: int = 3


@dataclass(frozen=True)
class MotorStateReading:
    """One motor's state decoded from a single 10-byte block read.

    Attributes:
        position: Raw signed ``PRESENT_POSITION`` count (multi-turn values
            are returned unwrapped).
        velocity: Raw signed ``PRESENT_VELOCITY`` count (0.229 rpm per count).
        current_raw: Raw signed ``PRESENT_CURRENT`` count in the model's own
            unit (see ``CURRENT_UNIT_MA``), motor sign convention.
    """

    position: int
    velocity: int
    current_raw: int


@dataclass(frozen=True)
class DiagnosticReading:
    """Slow-rate health registers of one motor.

    Attributes:
        hardware_error_status: Raw ``HARDWARE_ERROR_STATUS`` bit field; 0 is healthy.
        voltage_v: ``PRESENT_INPUT_VOLTAGE`` converted from 0.1 V units.
        temperature_c: ``PRESENT_TEMPERATURE`` in degrees Celsius.
    """

    hardware_error_status: int
    voltage_v: float
    temperature_c: int


def decode_state_block(data_bytes: "bytes | bytearray | Sequence[int]") -> MotorStateReading:
    """Decodes the 10-byte ``PRESENT_CURRENT..PRESENT_POSITION`` block.

    Args:
        data_bytes: Exactly ``STATE_BLOCK_NUM_BYTES`` bytes, little-endian, as
            received from address ``STATE_BLOCK_START_ADDRESS``.

    Returns:
        The signed current, velocity and position.

    Raises:
        ValueError: If the block does not have exactly 10 bytes.
    """
    block = bytes(data_bytes)
    if len(block) != STATE_BLOCK_NUM_BYTES:
        raise ValueError(f"State block must be {STATE_BLOCK_NUM_BYTES} bytes, got {len(block)}.")
    return MotorStateReading(
        position=int.from_bytes(block[6:10], "little", signed=True),
        velocity=int.from_bytes(block[2:6], "little", signed=True),
        current_raw=int.from_bytes(block[0:2], "little", signed=True),
    )


class DynamixelCommError(ConnectionError):
    """Exception representing a Dynamixel communication error."""

    def __init__(self, message: str, dxl_comm_result_code: int) -> None:
        packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        dxl_comm_result = packet_handler.getTxRxResult(dxl_comm_result_code)
        super().__init__(f"{message}\n[CommResult: {dxl_comm_result}]")


class DynamixelTimeoutError(DynamixelCommError, TimeoutError):
    """The motors did not answer within the caller's packet timeout.

    Raised by the deadline-bounded read path (``read_state_block`` and
    friends), and by every transmit (legacy ``sync_read``/``sync_write``
    included) when the serial write itself stalled past
    ``SERIAL_WRITE_TIMEOUT_S``.  It is both a :class:`DynamixelCommError` (so
    existing ``ConnectionError`` handlers still catch it) and a
    :class:`TimeoutError`.
    """


class DynamixelMotor:
    """Class that holds the definition and state of an individual motor."""

    def __init__(self, motor_id: int, motor_name: str, model_name: str) -> None:
        self.id = motor_id
        self.motor_name = motor_name
        self.model_name = model_name

        definition = get_model_definition(model_name)
        self.control_table: type[Enum] = definition["control_table"]
        self.model_number: int = definition["model_number"]
        self.resolution: int = definition["resolution"]


class DynamixelBus:
    """
    Manages communication with multiple Dynamixel motors on a single bus.
    Handles synchronized reading and writing, calibration, and error handling.
    """

    def __init__(
        self,
        port: str,
        motors: Dict[str, DynamixelMotor],
        need_calibration: bool = True,
    ) -> None:
        self.port_handler = dxl.PortHandler(port)
        self.packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        self.motors = motors
        # Calibration data: {motor_name: (homing_offset, inverted)}
        self.calibration: Dict[str, Tuple[int, bool]] = {}
        # GroupSyncRead/GroupSyncWrite handles reused by the deadline-bounded
        # path, keyed by (address, num_bytes, motor IDs).  The legacy
        # sync_read/sync_write below deliberately keep building fresh groups.
        self._read_groups: Dict[Tuple[int, int, Tuple[int, ...]], Any] = {}
        self._write_groups: Dict[Tuple[int, int, Tuple[int, ...]], Any] = {}

    def open(self, baudrate: int = BAUDRATE) -> None:
        """Opens the communication port."""
        if not self.port_handler.openPort():
            raise ConnectionError(f"Failed to open port {self.port_handler.port_name}.")
        if not self.port_handler.setBaudRate(baudrate):
            raise ConnectionError(f"Failed to set baudrate to {baudrate}.")
        self._set_serial_write_timeout()
        logger.info(f"Opened port {self.port_handler.port_name} (Baudrate: {baudrate})")

    def close(self) -> None:
        """Closes the communication port."""
        self.port_handler.closePort()
        logger.info(f"Closed port {self.port_handler.port_name}.")

    def __enter__(self) -> "DynamixelBus":
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
        """Changes the serial port; only allowed while the bus is closed.

        Args:
            port: Device path such as ``/dev/ttyUSB1``.

        Raises:
            RuntimeError: If the port is currently open.
        """
        if self._is_open():
            raise RuntimeError(
                f"Cannot change the port while {self.port_handler.port_name} is open."
            )
        self.port_handler = dxl.PortHandler(port)
        # The cached groups hold a reference to the old port handler.
        self._read_groups.clear()
        self._write_groups.clear()

    def _is_open(self) -> bool:
        return bool(getattr(self.port_handler, "is_open", False))

    def _set_serial_write_timeout(self) -> None:
        """Makes a blocked serial write raise instead of stalling the caller."""
        ser = getattr(self.port_handler, "ser", None)
        if ser is None:
            return
        try:
            ser.write_timeout = SERIAL_WRITE_TIMEOUT_S
        except AttributeError:
            logger.debug("Serial object of %s has no write_timeout.", self.port_handler.port_name)

    def set_calibration(self, calibration_data: Dict[str, Tuple[int, bool]]) -> None:
        """Sets the calibration data for the motors."""
        self.calibration = calibration_data
        logger.info("Calibration data set.")

    def _split_into_byte_chunks(self, value: int, length: int) -> List[int]:
        """Converts an integer value into a list of bytes for transmission."""
        if length == 1:
            return [value]
        if length == 2:
            return [dxl.DXL_LOBYTE(value), dxl.DXL_HIBYTE(value)]
        if length == 4:
            return [
                dxl.DXL_LOBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_LOWORD(value)),
                dxl.DXL_LOBYTE(dxl.DXL_HIWORD(value)),
                dxl.DXL_HIBYTE(dxl.DXL_HIWORD(value)),
            ]
        raise ValueError(f"Unsupported byte length: {length}")

    def sync_write(self, item: Enum, values: Dict[str, int | float]) -> None:
        """
        Writes values to a specific control table item for multiple motors simultaneously.

        Example:
            bus.sync_write(XControlTable.TORQUE_ENABLE, {"motor1": 1, "motor2": 1})
        """
        if not isinstance(item.value, ControlItem):
            raise TypeError("Item must be an Enum member with a ControlItem value.")

        control_item: ControlItem = item.value
        group_sync_write = dxl.GroupSyncWrite(
            self.port_handler, self.packet_handler, control_item.address, control_item.num_bytes
        )

        processed_values = np.array([values[name] for name in self.motors if name in values])
        motor_names_to_write = [name for name in self.motors if name in values]

        # Apply inverse calibration if needed (e.g., degrees to steps)
        if control_item.calibration_required and self.calibration:
            processed_values = self._revert_calibration(processed_values, motor_names_to_write)

        # Add parameters to the sync write group
        for i, name in enumerate(motor_names_to_write):
            motor = self.motors[name]
            # Ensure the motor uses the same control table
            if motor.control_table != item.__class__:
                logger.warning(f"Skipping {name} due to mismatched control table.")
                continue

            data = self._split_into_byte_chunks(int(processed_values[i]), control_item.num_bytes)
            if not group_sync_write.addParam(motor.id, data):
                logger.error(f"Failed to add parameter for {name} (ID-{motor.id}).")

        # Transmit the packet with retries
        comm_result = dxl.COMM_NOT_AVAILABLE  # initialize before loop
        for _ in range(NUM_WRITE_RETRY):
            comm_result = self._tx(group_sync_write, f"{item.name} sync write")
            if comm_result == dxl.COMM_SUCCESS:
                return

        raise DynamixelCommError(f"Failed to sync write {item.name}.", comm_result)

    def sync_read(self, item: Enum, motor_names: List[str]) -> Dict[str, Any]:
        """
        Reads values from a specific control table item for multiple motors simultaneously.

        Example:
            positions = bus.sync_read(XControlTable.PRESENT_POSITION, ["motor1", "motor2"])
        """
        if not isinstance(item.value, ControlItem):
            raise TypeError("Item must be an Enum member with a ControlItem value.")

        control_item: ControlItem = item.value
        group_sync_read = dxl.GroupSyncRead(
            self.port_handler, self.packet_handler, control_item.address, control_item.num_bytes
        )

        # Add parameters to the sync read group
        motors_to_read: List[DynamixelMotor] = []
        for name in motor_names:
            if name not in self.motors:
                continue
            motor = self.motors[name]
            if motor.control_table == item.__class__:
                group_sync_read.addParam(motor.id)
                motors_to_read.append(motor)

        # Transmit the packet with retries
        comm_result = dxl.COMM_NOT_AVAILABLE  # ループ前に初期化
        for _ in range(NUM_READ_RETRY):
            comm_result = self._transmit(group_sync_read.txRxPacket, f"{item.name} sync read")
            if comm_result == dxl.COMM_SUCCESS:
                break
        else:
            # ループがbreakされずに終了した場合（一度も成功しなかった場合）
            raise DynamixelCommError(f"Failed to sync read {item.name}.", comm_result)

        # Process received data
        raw_values = []
        results: Dict[str, Any] = {}
        for motor in motors_to_read:
            if group_sync_read.isAvailable(motor.id, control_item.address, control_item.num_bytes):
                raw_value = group_sync_read.getData(
                    motor.id, control_item.address, control_item.num_bytes
                )
                results[motor.motor_name] = cast_value(raw_value, control_item.dtype)
                raw_values.append(results[motor.motor_name])

        if control_item.calibration_required and self.calibration:
            calibrated_values = self._apply_calibration(
                np.array(raw_values), [m.motor_name for m in motors_to_read]
            )
            for i, motor in enumerate(motors_to_read):
                results[motor.motor_name] = calibrated_values[i]

        return results

    def _apply_calibration(
        self,
        values: NDArray[np.int32],
        motor_names: List[str],
    ) -> NDArray[np.float32]:
        """Converts raw motor steps (int32) to calibrated degrees (float32)."""
        values = values.astype(np.int32)
        for i, name in enumerate(motor_names):
            if name not in self.calibration:
                continue
            homing_offset, inverted = self.calibration[name]
            if inverted:
                values[i] *= -1
            values[i] += homing_offset

        # Convert from steps to degrees
        float_values = values.astype(np.float32)
        for i, name in enumerate(motor_names):
            resolution = self.motors[name].resolution
            float_values[i] = float_values[i] / (resolution / 2) * 180

        return float_values

    def _revert_calibration(
        self,
        values: NDArray[np.float32],
        motor_names: List[str],
    ) -> NDArray[np.int32]:
        """Converts calibrated degrees (float32) back to raw motor steps (int32)."""
        # Convert from degrees to steps
        step_values = values.astype(np.float32)
        for i, name in enumerate(motor_names):
            resolution = self.motors[name].resolution
            step_values[i] = step_values[i] / 180 * (resolution / 2)

        int_values = np.round(step_values).astype(np.int32)

        for i, name in enumerate(motor_names):
            if name not in self.calibration:
                continue
            homing_offset, inverted = self.calibration[name]
            int_values[i] -= homing_offset
            if inverted:
                int_values[i] *= -1

        return int_values

    def read(self, item: Enum, motor_name: str) -> Any:
        """Reads a value from a specific control table item for a single motor.

        Args:
            item: The control table item enum.
            motor_name: Name of the motor to read from.

        Returns:
            The value read from the motor.
        """
        result = self.sync_read(item, [motor_name])
        return result.get(motor_name)

    def write(self, item: Enum, motor_name: str, value: int | float) -> None:
        """Writes a value to a specific control table item for a single motor.

        Args:
            item: The control table item enum.
            motor_name: Name of the motor to write to.
            value: Value to write.
        """
        self.sync_write(item, {motor_name: value})

    def torque_disabled(self, specific_motor_names: List[str] | None = None) -> None:
        """torque_disabled for multiple motors.

        Args:
            specific_motor_names (List[str] | None, optional): List of motor names to
            disable torque. If None, disables torque for all motors. Defaults to None.
        """
        motor_names: List[str]

        if specific_motor_names is None:
            motor_names = list(self.motors.keys())
        else:
            motor_names = specific_motor_names
        torque_off_values: Dict[str, int | float] = {name: 0 for name in motor_names}

        self.sync_write(XControlTable.TORQUE_ENABLE, torque_off_values)

    def torque_enabled(self, specific_motor_names: List[str] | None = None) -> None:
        """torque_enabled for multiple motors.

        Args:
            specific_motor_names (List[str] | None, optional): List of motor names to
            enable torque. If None, enables torque for all motors. Defaults to None.
        """
        motor_names: List[str]
        if specific_motor_names is None:
            motor_names = list(self.motors.keys())
        else:
            motor_names = specific_motor_names
        torque_on_values: Dict[str, int | float] = {name: 1 for name in motor_names}
        self.sync_write(XControlTable.TORQUE_ENABLE, torque_on_values)

    # ------------------------------------------------------------------
    # Deadline-bounded transactions (Rakuda bilateral / current-control path)
    #
    # Everything below is additive.  The legacy ``sync_read``/``sync_write``
    # above keep their retry behaviour (they only share the ``_transmit``
    # guard, so a stalled serial write cannot wedge the port handler); the
    # control loop never calls them because a ten-retry read (about 37 ms
    # per attempt) blows any cycle budget.  Here one attempt is ``txPacket
    # -> setPacketTimeoutMillis -> rxPacket``, so the SDK's busy-poll can
    # never run longer than the caller's ``timeout_s`` (spec D32).
    # ------------------------------------------------------------------

    def _select_motors(self, motor_names: Sequence[str]) -> List[DynamixelMotor]:
        """Resolves names to motors, in the given order."""
        unknown = [name for name in motor_names if name not in self.motors]
        if unknown:
            raise ValueError(f"Unknown motor(s) on {self.port_handler.port_name}: {unknown}")
        if not motor_names:
            raise ValueError("At least one motor name is required.")
        return [self.motors[name] for name in motor_names]

    def _get_read_group(self, address: int, num_bytes: int, motor_ids: Sequence[int]) -> Any:
        """Returns a cached ``GroupSyncRead`` registered for exactly ``motor_ids``."""
        key = (address, num_bytes, tuple(motor_ids))
        group = self._read_groups.get(key)
        if group is None:
            group = dxl.GroupSyncRead(self.port_handler, self.packet_handler, address, num_bytes)
            for motor_id in key[2]:
                if not group.addParam(motor_id):
                    raise ValueError(f"Duplicate motor ID {motor_id} in a read at {address}.")
            self._read_groups[key] = group
        return group

    def _get_write_group(self, address: int, num_bytes: int, motor_ids: Sequence[int]) -> Any:
        """Returns a cached, cleared ``GroupSyncWrite`` for ``motor_ids``."""
        key = (address, num_bytes, tuple(motor_ids))
        group = self._write_groups.get(key)
        if group is None:
            group = dxl.GroupSyncWrite(self.port_handler, self.packet_handler, address, num_bytes)
            self._write_groups[key] = group
        group.clearParam()
        return group

    def _transmit(self, transmit: Callable[[], int], what: str) -> int:
        """Runs one SDK transmit; a serial write stalled past the write timeout is a timeout.

        The SDK's ``txPacket`` sets ``port.is_using`` before the serial write
        and clears it only on its own return paths or in ``rxPacket``.  A
        ``SerialTimeoutException`` escapes between the two, and a stuck flag
        would make every later transmit on this handler (including the hold
        sequence that must follow a fault) return ``COMM_PORT_BUSY``.  The bus
        is single-threaded (spec D41), so a set flag at entry is always stale
        and is cleared as well.

        Args:
            transmit: ``group.txPacket`` or ``group.txRxPacket``.
            what: Short description used in the error message.

        Returns:
            The SDK communication result.

        Raises:
            DynamixelTimeoutError: When the serial write stalled.
        """
        self.port_handler.is_using = False
        try:
            return transmit()
        except serial.SerialTimeoutException as exc:
            self.port_handler.is_using = False
            raise DynamixelTimeoutError(
                f"{what} on {self.port_handler.port_name}: serial write stalled for more than "
                f"{SERIAL_WRITE_TIMEOUT_S * 1e3:.0f} ms.",
                dxl.COMM_TX_FAIL,
            ) from exc

    def _tx(self, group: Any, what: str) -> int:
        """Transmits a group packet through :meth:`_transmit`."""
        return self._transmit(group.txPacket, what)

    def _discard_pending_input(self) -> None:
        """Drops status packets still queued from an earlier, timed-out read.

        The SDK only flushes the output buffer before a transmit; late answers
        of a previous transaction stay in the input buffer, and ``readRx``
        accepts a queued packet whose ID matches as if it were fresh.
        """
        ser = getattr(self.port_handler, "ser", None)
        if ser is None:
            return
        try:
            ser.reset_input_buffer()
        except AttributeError:
            logger.debug(
                "Serial object of %s has no reset_input_buffer.", self.port_handler.port_name
            )

    def _read_block(
        self,
        address: int,
        num_bytes: int,
        motors: Sequence[DynamixelMotor],
        *,
        timeout_s: float,
        attempts: int,
        what: str,
    ) -> Dict[int, bytes]:
        """Runs a deadline-bounded SyncRead and returns each motor's raw bytes.

        Args:
            address: Start address of the block.
            num_bytes: Block length.
            motors: Motors to read; every one must answer for an attempt to count.
            timeout_s: Hard upper bound on the receive wait of one attempt.
            attempts: Number of attempts before giving up.
            what: Short description used in error messages.

        Returns:
            ``{motor_id: bytes}`` with exactly ``num_bytes`` per motor.

        Raises:
            DynamixelCommError: On a transmit-side failure (``COMM_TX_*``,
                ``COMM_PORT_BUSY``); this is never retried.
            DynamixelTimeoutError: When every attempt ended in a ``COMM_RX_*``
                result or with a motor missing from (or truncated in) the response.
        """
        if timeout_s <= 0.0:
            raise ValueError(f"timeout_s must be positive, got {timeout_s}.")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}.")
        port_name = self.port_handler.port_name
        group = self._get_read_group(address, num_bytes, [m.id for m in motors])
        timeout_ms = timeout_s * 1e3

        last_result = dxl.COMM_RX_TIMEOUT
        missing: List[str] = []
        for _ in range(attempts):
            self._discard_pending_input()
            result = self._tx(group, what)
            if result != dxl.COMM_SUCCESS:
                raise DynamixelCommError(f"{what} on {port_name}: transmit failed.", result)
            # syncReadTx() just armed the SDK's own (length-based) timeout;
            # override it with the caller's budget before the busy-poll starts.
            self.port_handler.setPacketTimeoutMillis(timeout_ms)
            result = group.rxPacket()
            if result in _RX_RESULTS:
                last_result = result
                continue
            if result != dxl.COMM_SUCCESS:
                raise DynamixelCommError(f"{what} on {port_name}: receive failed.", result)

            blocks: Dict[int, bytes] = {}
            missing = []
            for motor in motors:
                # ``data_dict`` is the SDK's per-ID receive buffer (the list
                # ``getData`` slices).  Taking the whole block at once keeps
                # the decoding in one place instead of one getData per field.
                # ``isAvailable`` does not check the length: a CRC-valid but
                # short status packet (an error status without parameters)
                # yields fewer bytes and must count as no data.
                block = bytes(group.data_dict[motor.id])
                if group.isAvailable(motor.id, address, num_bytes) and len(block) == num_bytes:
                    blocks[motor.id] = block
                else:
                    missing.append(motor.motor_name)
            if not missing:
                return blocks
            last_result = dxl.COMM_RX_TIMEOUT

        detail = f"; no data from {missing}" if missing else ""
        raise DynamixelTimeoutError(
            f"{what} on {port_name}: no complete response within {attempts} attempt(s) "
            f"of {timeout_s * 1e3:.1f} ms{detail}.",
            last_result,
        )

    def _read_single(self, control_item: ControlItem, motor: DynamixelMotor) -> int | None:
        """Reads one item from one motor; ``None`` if it did not answer in time."""
        try:
            blocks = self._read_block(
                control_item.address,
                control_item.num_bytes,
                [motor],
                timeout_s=READBACK_TIMEOUT_S,
                attempts=1,
                what=f"Read of address {control_item.address} from '{motor.motor_name}'",
            )
        except DynamixelTimeoutError:
            return None
        return cast_value(int.from_bytes(blocks[motor.id], "little"), control_item.dtype)

    def read_state_block(
        self,
        motor_names: Sequence[str],
        *,
        timeout_s: float,
        attempts: int = 1,
    ) -> Tuple[Dict[str, MotorStateReading], int, int]:
        """Reads current, velocity and position of many motors in one transaction.

        One 10-byte SyncRead from ``PRESENT_CURRENT`` (126) replaces three
        per-item reads.  SyncRead reduces packets; it does not make the motors
        sample simultaneously, so the returned timestamps bound how far apart
        the samples in this snapshot can be.

        Args:
            motor_names: Motors to read, in any order.  Every motor must answer
                for an attempt to succeed (the SDK stops at the first silent ID).
            timeout_s: Hard upper bound on the receive wait of one attempt.
            attempts: Attempts before giving up; attempts run back to back.

        Returns:
            ``(readings, start_ns, end_ns)``: readings keyed by motor name and the
            ``time.monotonic_ns()`` stamps taken just before the first transmit
            and just after the successful receive.

        Raises:
            ValueError: For unknown motor names, an empty list, a non-positive
                timeout or fewer than one attempt.
            DynamixelCommError: On a transmit-side failure (not retried).
            DynamixelTimeoutError: When all ``attempts`` ended without a complete
                response.  It is also a ``ConnectionError`` and a ``TimeoutError``.
        """
        motors = self._select_motors(motor_names)
        start_ns = time.monotonic_ns()
        blocks = self._read_block(
            STATE_BLOCK_START_ADDRESS,
            STATE_BLOCK_NUM_BYTES,
            motors,
            timeout_s=timeout_s,
            attempts=attempts,
            what="State block read",
        )
        end_ns = time.monotonic_ns()
        readings = {motor.motor_name: decode_state_block(blocks[motor.id]) for motor in motors}
        return readings, start_ns, end_ns

    def write_goal_current_raw(self, values: Mapping[str, int]) -> None:
        """Writes signed raw ``GOAL_CURRENT`` counts to several motors at once.

        Exactly one SyncWrite packet is transmitted; there is no retry and no
        read-back, so the call never blocks beyond the serial write timeout.

        Args:
            values: ``{motor_name: raw_count}``.  Counts are in each motor's own
                unit and sign convention; callers clamp to ``CURRENT_LIMIT``.

        Raises:
            ValueError: For an unknown motor, an empty mapping or a value outside
                the ``INT16`` wire range.
            DynamixelCommError: If the packet could not be transmitted.
        """
        item: ControlItem = XControlTable.GOAL_CURRENT.value
        unknown = [name for name in values if name not in self.motors]
        if unknown:
            raise ValueError(f"Unknown motor(s) on {self.port_handler.port_name}: {unknown}")
        names = [name for name in self.motors if name in values]
        if not names:
            raise ValueError("write_goal_current_raw needs at least one motor.")

        encoded: Dict[str, List[int]] = {}
        for name in names:
            try:
                word = encode_value(int(values[name]), Dtype.INT16)
            except ValueError as exc:
                raise ValueError(f"GOAL_CURRENT for '{name}': {exc}") from exc
            encoded[name] = self._split_into_byte_chunks(word, item.num_bytes)

        group = self._get_write_group(
            item.address, item.num_bytes, [self.motors[name].id for name in names]
        )
        for name in names:
            if not group.addParam(self.motors[name].id, encoded[name]):
                raise ValueError(f"Duplicate motor ID for '{name}' in GOAL_CURRENT write.")
        result = self._tx(group, "GOAL_CURRENT write")
        if result != dxl.COMM_SUCCESS:
            raise DynamixelCommError(
                f"GOAL_CURRENT write on {self.port_handler.port_name} failed.", result
            )

    def write_with_readback(
        self,
        item: Enum,
        values: Mapping[str, int | float],
        *,
        tolerance: int = 0,
        attempts: int = 5,
        settle_s: float = 0.005,
    ) -> None:
        """Writes a control-table item and confirms each motor accepted it.

        A successful SyncWrite only says the packet left the host.  EEPROM
        items are silently ignored while torque is on, and a mode change can
        reset dependent registers, so configuration writes go through here.
        Each attempt is: ``sync_write`` the still-unconfirmed motors, sleep
        ``settle_s``, then read every one of them back individually with a
        ``READBACK_TIMEOUT_S`` packet timeout.  Values are compared as raw
        register counts; bus calibration is not applied.

        Args:
            item: Control-table item to write.
            values: ``{motor_name: value}``.
            tolerance: Allowed absolute difference between written and read-back value.
            attempts: Write/read-back rounds before giving up.
            settle_s: Pause between the write and its read-back.

        Raises:
            TypeError: If ``item`` is not a control-table member.
            ValueError: For an unknown motor or fewer than one attempt.
            DynamixelCommError: When motors still mismatch (or did not answer)
                after ``attempts`` rounds; the message lists them with the
                wanted and read values (``None`` = no response).
        """
        if not isinstance(item.value, ControlItem):
            raise TypeError("Item must be an Enum member with a ControlItem value.")
        if attempts < 1:
            raise ValueError(f"attempts must be at least 1, got {attempts}.")
        control_item: ControlItem = item.value
        motors = {motor.motor_name: motor for motor in self._select_motors(list(values))}

        pending: Dict[str, int] = {name: int(value) for name, value in values.items()}
        mismatched: Dict[str, Tuple[int, int | None]] = {}
        for _ in range(attempts):
            to_write: Dict[str, int | float] = dict(pending)
            self.sync_write(item, to_write)
            time.sleep(settle_s)
            mismatched = {}
            for name, wanted in pending.items():
                got = self._read_single(control_item, motors[name])
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
        """Reads hardware error status, input voltage and temperature.

        Two deadline-bounded SyncReads (``HARDWARE_ERROR_STATUS`` and the
        adjacent voltage/temperature pair), meant for a slow health schedule
        rather than the per-cycle state block.

        Args:
            motor_names: Motors to query.
            timeout_s: Hard upper bound on the receive wait of each read.

        Raises:
            ValueError: For unknown motor names or an empty list.
            DynamixelCommError: On a transmit-side failure.
            DynamixelTimeoutError: If a motor did not answer either read.
        """
        motors = self._select_motors(motor_names)
        hw_item: ControlItem = XControlTable.HARDWARE_ERROR_STATUS.value
        errors = self._read_block(
            hw_item.address,
            hw_item.num_bytes,
            motors,
            timeout_s=timeout_s,
            attempts=1,
            what="Hardware error status read",
        )
        environment = self._read_block(
            _ENVIRONMENT_BLOCK_START_ADDRESS,
            _ENVIRONMENT_BLOCK_NUM_BYTES,
            motors,
            timeout_s=timeout_s,
            attempts=1,
            what="Voltage/temperature read",
        )
        return {
            motor.motor_name: DiagnosticReading(
                hardware_error_status=errors[motor.id][0],
                voltage_v=int.from_bytes(environment[motor.id][0:2], "little") / 10.0,
                temperature_c=environment[motor.id][2],
            )
            for motor in motors
        }

    def read_model_numbers(self, motor_names: Sequence[str]) -> Dict[str, int]:
        """Reads ``MODEL_NUMBER`` from each motor individually.

        Per-motor reads (rather than one SyncRead, which stops at the first
        silent ID) so that every unreachable motor can be reported.

        Args:
            motor_names: Motors to query.

        Returns:
            ``{motor_name: model_number}`` for the motors that answered; a motor
            that did not answer within ``READBACK_TIMEOUT_S`` is omitted.

        Raises:
            ValueError: For unknown motor names or an empty list.
            DynamixelCommError: On a transmit-side failure.
        """
        item: ControlItem = XControlTable.MODEL_NUMBER.value
        found: Dict[str, int] = {}
        for motor in self._select_motors(motor_names):
            model_number = self._read_single(item, motor)
            if model_number is not None:
                found[motor.motor_name] = model_number
        return found

    def verify_models(self, motor_names: Sequence[str] | None = None) -> None:
        """Checks that every motor on the bus is the model the configuration declares.

        Args:
            motor_names: Motors to check; ``None`` checks all of them.

        Raises:
            ConnectionError: Listing every motor that did not answer or whose
                ``MODEL_NUMBER`` differs from its declared model.
        """
        names = list(self.motors) if motor_names is None else list(motor_names)
        motors = self._select_motors(names)
        found = self.read_model_numbers(names)
        problems: List[str] = []
        for motor in motors:
            if motor.motor_name not in found:
                problems.append(f"{motor.motor_name} (ID {motor.id}): no response")
            elif found[motor.motor_name] != motor.model_number:
                problems.append(
                    f"{motor.motor_name} (ID {motor.id}): declared {motor.model_name} "
                    f"(model {motor.model_number}) but the motor reports "
                    f"{found[motor.motor_name]}"
                )
        if problems:
            raise ConnectionError(
                f"Motor model check failed on {self.port_handler.port_name}: " + "; ".join(problems)
            )
        logger.info("Verified %d motor model(s) on %s.", len(motors), self.port_handler.port_name)

    def __repr__(self) -> str:
        motor_list = ", ".join(self.motors.keys())
        return f"DynamixelBus(port={self.port_handler.port_name}, motors=[{motor_list}])"

    def __len__(self) -> int:
        return len(self.motors)
