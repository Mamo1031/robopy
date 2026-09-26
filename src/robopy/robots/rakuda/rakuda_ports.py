"""Rakuda serial-port detection and register maintenance.

Two Rakuda buses hang off the host: the mini-rakuda leader (XC330-T288 arms
plus one XM430-W350 torso) and the follower (XM540-W270 shoulders, XM430-W350
elsewhere).  Which ``/dev/ttyUSB*`` node each one gets is up to udev, so
:func:`detect_rakuda_ports` broadcast-pings every candidate and classifies each
bus from the ``{id: model_number}`` pattern that the arm classes declare.

The module doubles as the bring-up CLI (``robopy-rakuda-ports`` or
``python -m robopy.robots.rakuda.rakuda_ports``) with the subcommands ``scan``,
``show``, ``bench``, ``set-return-delay``, ``write-eeprom`` and ``release``.
Every subcommand that writes to a motor asks ``y/N`` unless ``--yes`` is given.
EEPROM writes are torque-off only ("Mode A"): they are refused, with nothing
written, when any target motor has torque on.

Run it with the venv's SDK, not the ROS one on ``PYTHONPATH``::

    env -u PYTHONPATH uv run --frozen python -m robopy.robots.rakuda.rakuda_ports scan
"""

from __future__ import annotations

import argparse
import glob
import logging
import os
import sys
import time
from dataclasses import dataclass, field
from typing import Any, Callable, Dict, List, Literal, Mapping, Sequence, Tuple

import numpy as np

from robopy.config.robot_config import RAKUDA_JOINT_NAMES, RakudaConfig
from robopy.config.robot_config.rakuda_config import (
    PORT_AUTO,
    RAKUDA_CURRENT_CAPABLE_JOINTS,
    RAKUDA_GRIPPER_JOINT_NAMES,
)
from robopy.motor.dynamixel_bus import (
    _RX_RESULTS,
    BAUDRATE,
    PROTOCOL_VERSION,
    DynamixelBus,
    DynamixelCommError,
    DynamixelMotor,
    DynamixelTimeoutError,
)
from robopy.motor.dynamixel_control_table import (
    CURRENT_LIMIT_MAX_RAW,
    CURRENT_UNIT_MA,
    STATE_BLOCK_NUM_BYTES,
    STATE_BLOCK_START_ADDRESS,
    OperatingMode,
    XControlTable,
)

from .rakuda_follower import RakudaFollower
from .rakuda_leader import RakudaLeader

logger = logging.getLogger(__name__)

Side = Literal["leader", "follower"]
SIDES: Tuple[Side, ...] = ("leader", "follower")

#: Device-node patterns scanned when no candidates are given.
CANDIDATE_PATTERNS: Tuple[str, ...] = ("/dev/ttyUSB*", "/dev/ttyACM*")
#: udev's stable per-adapter symlinks, used to deduplicate and label candidates.
SERIAL_BY_ID_DIR = "/dev/serial/by-id"
#: sysfs root that holds the FTDI ``latency_timer`` attribute of each USB serial port.
USB_SERIAL_SYSFS_DIR = "/sys/bus/usb-serial/devices"

#: A bus is a Rakuda side when at least this many of its 17 expected IDs answer.
MIN_MATCHING_MOTORS = 12

#: EEPROM items this module may write, and on which motors.
#: ``OPERATING_MODE``, ``HOMING_OFFSET`` and ``DRIVE_MODE`` are deliberately absent.
EEPROM_WRITE_WHITELIST: Dict[XControlTable, frozenset[str]] = {
    XControlTable.RETURN_DELAY_TIME: frozenset(RAKUDA_JOINT_NAMES),
    XControlTable.CURRENT_LIMIT: frozenset(RAKUDA_GRIPPER_JOINT_NAMES),
    XControlTable.MAX_POSITION_LIMIT: frozenset(RAKUDA_GRIPPER_JOINT_NAMES),
    XControlTable.MIN_POSITION_LIMIT: frozenset(RAKUDA_GRIPPER_JOINT_NAMES),
}

#: Process exit code of a refused EEPROM write (torque was on).
EXIT_REFUSED = 2

_DRIVE_MODE_BIT_NAMES: Dict[int, str] = {
    0: "reverse",
    1: "reserved1",
    2: "time_profile",
    3: "torque_on_by_goal",
}


# --- scanning and classification -------------------------------------------


@dataclass(frozen=True)
class ScannedMotor:
    """One motor's answer to a broadcast ping."""

    model_number: int
    firmware: int


@dataclass(frozen=True)
class PortScan:
    """What one candidate port looked like.

    Attributes:
        port: Device node that was scanned.
        motors: Answering motors keyed by ID (empty when the scan failed).
        side: Classification, or ``None`` if the bus is not a Rakuda side.
        error: Why the port could not be scanned, or ``None``.
        by_id: The port's ``/dev/serial/by-id`` symlink, if udev made one.
        latency_timer: FTDI latency timer in ms, or ``None`` when not readable.
    """

    port: str
    motors: Dict[int, ScannedMotor]
    side: Side | None
    error: str | None = None
    by_id: str | None = None
    latency_timer: int | None = None

    def describe(self) -> str:
        """One-line summary for logs and error messages."""
        if self.error is not None:
            return f"error: {self.error}"
        return f"{self.side or 'unclassified'} ({len(self.motors)} motor(s) answered)"


@dataclass(frozen=True)
class RakudaPorts:
    """Result of :func:`detect_rakuda_ports`.

    Attributes:
        leader: Device node of the leader bus.
        follower: Device node of the follower bus.
        by_id: ``{device: /dev/serial/by-id/... symlink}`` for every scanned
            candidate that has one.
        latency_timer: ``{device: ms}`` for every scanned candidate.
    """

    leader: str
    follower: str
    by_id: Dict[str, str] = field(default_factory=dict)
    latency_timer: Dict[str, int | None] = field(default_factory=dict)

    def port(self, side: Side) -> str:
        """Returns the device node of ``side``."""
        return self.leader if side == "leader" else self.follower


def rakuda_motor_table(side: Side) -> Dict[str, DynamixelMotor]:
    """Returns the motor table (name -> motor) that the ``side`` arm class declares.

    The table comes from the arm's own ``_create_motors`` so that detection and
    ``connect()`` can never disagree.  Constructing the arm creates a
    ``PortHandler`` for the placeholder port ``auto`` but opens nothing.
    """
    cfg = RakudaConfig(leader_port=PORT_AUTO, follower_port=PORT_AUTO)
    arm = RakudaLeader(cfg) if side == "leader" else RakudaFollower(cfg)
    return dict(arm.motors.motors)


def expected_bus_models(side: Side) -> Dict[int, int]:
    """Returns ``{motor_id: model_number}`` that a ``side`` bus must show."""
    return {motor.id: motor.model_number for motor in rakuda_motor_table(side).values()}


def scan_port(port: str, baudrate: int = BAUDRATE) -> Dict[int, ScannedMotor]:
    """Broadcast-pings ``port`` and returns every motor that answered.

    Only ``PING`` is transmitted, so nothing on the bus changes.  The SDK is
    imported here rather than at module level so the function can be replaced
    in tests and so the SDK used is the one :func:`check_sdk_location` reports.

    Args:
        port: Device node such as ``/dev/ttyUSB1``.
        baudrate: Bus baud rate; both Rakuda buses run at 1 Mbaud.

    Returns:
        ``{motor_id: ScannedMotor}`` sorted by ID.

    Raises:
        OSError: If the port cannot be opened or configured (``serial.SerialException``
            is an ``OSError``).
    """
    import dynamixel_sdk as dxl

    port_handler = dxl.PortHandler(port)
    if not port_handler.openPort():
        raise OSError(f"Could not open {port}.")
    try:
        if not port_handler.setBaudRate(baudrate):
            raise OSError(f"Could not set {port} to {baudrate} baud.")
        packet_handler = dxl.PacketHandler(PROTOCOL_VERSION)
        found, result = packet_handler.broadcastPing(port_handler)
    finally:
        port_handler.closePort()
    if result not in (dxl.COMM_SUCCESS, dxl.COMM_RX_TIMEOUT):
        logger.warning(
            "Broadcast ping on %s ended with %s; keeping the %d answer(s) that parsed.",
            port,
            packet_handler.getTxRxResult(result),
            len(found),
        )
    return {
        int(motor_id): ScannedMotor(model_number=int(model), firmware=int(firmware))
        for motor_id, (model, firmware) in sorted(found.items())
    }


def classify_rakuda_bus(scan: Mapping[int, ScannedMotor]) -> Side | None:
    """Decides whether a scan is the leader bus, the follower bus or neither.

    A side matches when at least ``MIN_MATCHING_MOTORS`` of its 17 expected IDs
    answered and every answering expected ID reports the expected model number.
    IDs outside the table are ignored.  A scan matching both sides (impossible
    with the real motor models) is reported as ``None``.
    """
    matches: List[Side] = []
    for side in SIDES:
        expected = expected_bus_models(side)
        answering = [motor_id for motor_id in expected if motor_id in scan]
        if len(answering) < MIN_MATCHING_MOTORS:
            continue
        if all(scan[motor_id].model_number == expected[motor_id] for motor_id in answering):
            matches.append(side)
    return matches[0] if len(matches) == 1 else None


# --- host-side port information ---------------------------------------------


def by_id_links() -> Dict[str, str]:
    """Returns ``{resolved device node: /dev/serial/by-id symlink}``."""
    links: Dict[str, str] = {}
    for entry in sorted(glob.glob(os.path.join(SERIAL_BY_ID_DIR, "*"))):
        links[os.path.realpath(entry)] = entry
    return links


def read_latency_timer(port: str) -> int | None:
    """Reads the FTDI ``latency_timer`` (ms) of ``port`` from sysfs.

    Returns:
        The value, or ``None`` when the attribute does not exist (non-FTDI
        adapter, no sysfs) or cannot be parsed.
    """
    tty = os.path.basename(os.path.realpath(port))
    path = os.path.join(USB_SERIAL_SYSFS_DIR, tty, "latency_timer")
    try:
        with open(path, encoding="ascii") as handle:
            return int(handle.read().strip())
    except (OSError, ValueError):
        return None


def _dedupe_ports(candidates: Sequence[str]) -> List[str]:
    """Drops candidates that resolve to the same device node, keeping the first spelling."""
    seen: Dict[str, str] = {}
    for candidate in candidates:
        seen.setdefault(os.path.realpath(candidate), candidate)
    return list(seen.values())


def list_candidate_ports() -> List[str]:
    """Returns the device nodes matching ``CANDIDATE_PATTERNS``, deduplicated and sorted."""
    devices: List[str] = []
    for pattern in CANDIDATE_PATTERNS:
        devices.extend(sorted(glob.glob(pattern)))
    return sorted(_dedupe_ports(devices))


def scan_candidates(candidates: Sequence[str] | None = None) -> List[PortScan]:
    """Scans and classifies every candidate port; a failed scan becomes an error entry."""
    ports = list_candidate_ports() if candidates is None else _dedupe_ports(candidates)
    links = by_id_links()
    scans: List[PortScan] = []
    for port in ports:
        by_id = links.get(os.path.realpath(port))
        latency = read_latency_timer(port)
        try:
            motors = scan_port(port)
        except OSError as exc:
            logger.warning("Could not scan %s: %s", port, exc)
            scans.append(
                PortScan(port, {}, None, error=str(exc), by_id=by_id, latency_timer=latency)
            )
            continue
        side = classify_rakuda_bus(motors)
        logger.info("%s: %s", port, PortScan(port, motors, side).describe())
        scans.append(PortScan(port, motors, side, by_id=by_id, latency_timer=latency))
    return scans


def select_rakuda_ports(scans: Sequence[PortScan]) -> RakudaPorts:
    """Picks the leader and follower out of scan results.

    Raises:
        OSError: Unless exactly one leader and exactly one follower were found;
            the message lists every candidate with its result.
    """
    found: Dict[Side, List[str]] = {"leader": [], "follower": []}
    for scan in scans:
        if scan.side is not None:
            found[scan.side].append(scan.port)
    if len(found["leader"]) == 1 and len(found["follower"]) == 1:
        return RakudaPorts(
            leader=found["leader"][0],
            follower=found["follower"][0],
            by_id={scan.port: scan.by_id for scan in scans if scan.by_id is not None},
            latency_timer={scan.port: scan.latency_timer for scan in scans},
        )
    if not scans:
        detail = "no candidate ports (looked for " + ", ".join(CANDIDATE_PATTERNS) + ")"
    else:
        detail = "; ".join(f"{scan.port}: {scan.describe()}" for scan in scans)
    raise OSError(
        "Could not identify the Rakuda ports: need exactly one leader and one follower, "
        f"found {len(found['leader'])} leader(s) and {len(found['follower'])} follower(s). "
        f"Candidates: {detail}."
    )


def detect_rakuda_ports(candidates: Sequence[str] | None = None) -> RakudaPorts:
    """Finds the leader and follower buses by scanning the serial ports.

    Args:
        candidates: Device nodes to scan; ``None`` scans ``CANDIDATE_PATTERNS``.
            Duplicates (including ``/dev/serial/by-id`` spellings) are dropped.

    Raises:
        OSError: Unless exactly one leader and one follower answered.
    """
    return select_rakuda_ports(scan_candidates(candidates))


def resolve_port(value: str, side: Side) -> str:
    """Returns ``value`` unless it is ``PORT_AUTO``, in which case the bus is detected."""
    if value != PORT_AUTO:
        return value
    ports = detect_rakuda_ports()
    port = ports.port(side)
    logger.info(
        "Resolved %s port 'auto' to %s (%s).", side, port, ports.by_id.get(port, "no by-id")
    )
    return port


def resolve_ports(leader: str, follower: str) -> Tuple[str, str]:
    """Resolves both ports with at most one scan."""
    if leader != PORT_AUTO and follower != PORT_AUTO:
        return leader, follower
    ports = detect_rakuda_ports()
    return (
        ports.leader if leader == PORT_AUTO else leader,
        ports.follower if follower == PORT_AUTO else follower,
    )


def check_sdk_location() -> str:
    """Logs where ``dynamixel_sdk`` was imported from; warns if outside this venv.

    A ROS ``PYTHONPATH`` can shadow the venv's SDK with an older build for
    another Python version.
    """
    import dynamixel_sdk

    path = os.path.realpath(dynamixel_sdk.__file__)
    prefix = os.path.realpath(sys.prefix)
    if path.startswith(prefix + os.sep):
        logger.info("dynamixel_sdk: %s", path)
    else:
        logger.warning(
            "dynamixel_sdk is loaded from %s, outside this interpreter's environment %s. "
            "Run with `env -u PYTHONPATH uv run --frozen ...`.",
            path,
            prefix,
        )
    return path


# --- register dump ----------------------------------------------------------

REGISTER_COLUMNS: Tuple[str, ...] = (
    "name",
    "id",
    "model",
    "firmware",
    "return_delay_time",
    "drive_mode",
    "drive_mode_bits",
    "operating_mode",
    "homing_offset",
    "current_limit",
    "goal_current",
    "temperature_limit",
    "min_position_limit",
    "max_position_limit",
    "torque_enable",
    "present_position",
    "voltage_v",
    "temperature_c",
    "hardware_error_status",
)

_REGISTER_ITEMS: Dict[str, XControlTable] = {
    "firmware": XControlTable.FIRMWARE_VERSION,
    "return_delay_time": XControlTable.RETURN_DELAY_TIME,
    "drive_mode": XControlTable.DRIVE_MODE,
    "operating_mode": XControlTable.OPERATING_MODE,
    "homing_offset": XControlTable.HOMING_OFFSET,
    "current_limit": XControlTable.CURRENT_LIMIT,
    "goal_current": XControlTable.GOAL_CURRENT,
    "temperature_limit": XControlTable.TEMPERATURE_LIMIT,
    "min_position_limit": XControlTable.MIN_POSITION_LIMIT,
    "max_position_limit": XControlTable.MAX_POSITION_LIMIT,
    "torque_enable": XControlTable.TORQUE_ENABLE,
    "present_position": XControlTable.PRESENT_POSITION,
    "voltage_v": XControlTable.PRESENT_INPUT_VOLTAGE,
    "temperature_c": XControlTable.PRESENT_TEMPERATURE,
    "hardware_error_status": XControlTable.HARDWARE_ERROR_STATUS,
}


def decode_drive_mode(value: int) -> str:
    """Names the set bits of ``DRIVE_MODE`` (``reverse|time_profile``; ``-`` when none).

    Bits: 0 reverse, 1 reserved, 2 time-based profile, 3 torque-on-by-goal-update;
    4..7 are reserved and shown as ``bitN``.
    """
    names = [_DRIVE_MODE_BIT_NAMES.get(bit, f"bit{bit}") for bit in range(8) if (value >> bit) & 1]
    return "|".join(names) if names else "-"


def dump_registers(bus: DynamixelBus, motor_names: Sequence[str]) -> List[Dict[str, Any]]:
    """Reads the bring-up register set of each motor (reads only).

    Args:
        bus: An open bus.
        motor_names: Motors to read; each must be on the bus and answering.

    Returns:
        One dict per motor, in ``motor_names`` order, with ``REGISTER_COLUMNS``
        as keys.  Values are raw register counts except ``voltage_v`` (volts),
        ``drive_mode_bits`` (see :func:`decode_drive_mode`) and ``model``
        (declared model name).
    """
    names = list(motor_names)
    rows: List[Dict[str, Any]] = [
        {"name": name, "id": bus.motors[name].id, "model": bus.motors[name].model_name}
        for name in names
    ]
    for column, item in _REGISTER_ITEMS.items():
        values = bus.sync_read(item, names)
        for row in rows:
            value = values.get(row["name"])
            if column == "voltage_v" and value is not None:
                value = int(value) / 10.0
            row[column] = value
    for row in rows:
        drive_mode = row.get("drive_mode")
        row["drive_mode_bits"] = None if drive_mode is None else decode_drive_mode(int(drive_mode))
    return [{column: row[column] for column in REGISTER_COLUMNS} for row in rows]


def format_register_table(rows: Sequence[Mapping[str, Any]]) -> List[str]:
    """Renders :func:`dump_registers` rows as aligned text lines."""
    headers = {
        "name": "name",
        "id": "id",
        "model": "model",
        "firmware": "fw",
        "return_delay_time": "rdt",
        "drive_mode": "drv",
        "drive_mode_bits": "drv_bits",
        "operating_mode": "mode",
        "homing_offset": "homing",
        "current_limit": "cur_lim",
        "goal_current": "goal_cur",
        "temperature_limit": "temp_lim",
        "min_position_limit": "min_pos",
        "max_position_limit": "max_pos",
        "torque_enable": "trq",
        "present_position": "pos",
        "voltage_v": "volt",
        "temperature_c": "temp",
        "hardware_error_status": "hw_err",
    }
    table = [[headers[column] for column in REGISTER_COLUMNS]]
    for row in rows:
        table.append(["-" if row.get(c) is None else str(row[c]) for c in REGISTER_COLUMNS])
    widths = [max(len(line[i]) for line in table) for i in range(len(REGISTER_COLUMNS))]
    return [
        "  ".join(cell.ljust(widths[i]) for i, cell in enumerate(line)).rstrip() for line in table
    ]


# --- bench ------------------------------------------------------------------


def _legacy_state_block_read(bus: DynamixelBus, motor_names: Sequence[str]) -> None:
    """One ``txRxPacket()`` SyncRead: the SDK's own length-based packet timeout applies.

    Failures are classified like ``DynamixelBus._read_block`` so that both
    bench modes agree: a ``COMM_RX_*`` result is a :class:`DynamixelTimeoutError`
    (counted by the bench), anything else is a :class:`DynamixelCommError`.
    """
    import dynamixel_sdk as dxl

    group = dxl.GroupSyncRead(
        bus.port_handler, bus.packet_handler, STATE_BLOCK_START_ADDRESS, STATE_BLOCK_NUM_BYTES
    )
    for name in motor_names:
        group.addParam(bus.motors[name].id)
    result = group.txRxPacket()
    if result in _RX_RESULTS:
        raise DynamixelTimeoutError("Legacy state block read timed out.", result)
    if result != dxl.COMM_SUCCESS:
        raise DynamixelCommError("Legacy state block read failed.", result)


def bench_state_block(
    bus: DynamixelBus,
    motor_names: Sequence[str],
    n: int = 500,
    timeout_s: float = 0.020,
    silent_id: int | None = None,
    legacy: bool = False,
) -> Dict[str, Any]:
    """Times ``n`` state-block reads and derives the recommended read timeout.

    Args:
        bus: An open bus.
        motor_names: Motors to include in every read.
        n: Number of reads.
        timeout_s: Per-attempt receive bound passed to ``read_state_block``.
        silent_id: An ID that is *not* on the bus.  It is added to the read so
            that every attempt has one silent motor, reproducing a lost motor
            without touching the wiring; each sample should then equal
            ``timeout_s``.  The phantom motor is removed again afterwards.
        legacy: Use ``txRxPacket()`` (the SDK's own ~37.6 ms timeout for 17
            motors) instead of the deadline-bounded path.

    Returns:
        Statistics in milliseconds (``p50_ms``, ``p95_ms``, ``p99_ms``,
        ``max_ms``, ``min_ms``, ``mean_ms``), the counts ``ok``/``timeouts``,
        the raw ``samples_ms`` and ``recommended_read_timeout_s`` =
        ``max(1.5 * p99, p99 + 3 ms)`` in seconds.

    Raises:
        ValueError: For ``n < 1``, no motors, or a ``silent_id`` already on the bus.
    """
    if n < 1:
        raise ValueError(f"n must be at least 1, got {n}.")
    names = list(motor_names)
    if not names:
        raise ValueError("bench_state_block needs at least one motor.")
    phantom: str | None = None
    if silent_id is not None:
        if any(motor.id == silent_id for motor in bus.motors.values()):
            raise ValueError(f"ID {silent_id} is a real motor on this bus; pick an unused ID.")
        phantom = f"_silent_{silent_id}"
        bus.motors[phantom] = DynamixelMotor(silent_id, phantom, bus.motors[names[0]].model_name)
        names.append(phantom)

    samples: List[float] = []
    timeouts = 0
    try:
        for _ in range(n):
            start_ns = time.monotonic_ns()
            try:
                if legacy:
                    _legacy_state_block_read(bus, names)
                else:
                    bus.read_state_block(names, timeout_s=timeout_s, attempts=1)
            except DynamixelTimeoutError:
                timeouts += 1
            samples.append((time.monotonic_ns() - start_ns) / 1e6)
    finally:
        if phantom is not None:
            del bus.motors[phantom]

    array = np.asarray(samples, dtype=float)
    p50, p95, p99 = (float(v) for v in np.percentile(array, [50, 95, 99]))
    return {
        "n": n,
        "motors": len(names),
        "timeout_ms": timeout_s * 1e3,
        "silent_id": silent_id,
        "legacy": legacy,
        "ok": n - timeouts,
        "timeouts": timeouts,
        "p50_ms": p50,
        "p95_ms": p95,
        "p99_ms": p99,
        "max_ms": float(array.max()),
        "min_ms": float(array.min()),
        "mean_ms": float(array.mean()),
        "samples_ms": samples,
        "recommended_read_timeout_s": max(1.5 * p99, p99 + 3.0) / 1e3,
    }


def format_bench_report(report: Mapping[str, Any]) -> List[str]:
    """Renders :func:`bench_state_block` output as text lines."""
    mode = "legacy txRxPacket" if report["legacy"] else "deadline-bounded"
    silent = f", silent ID {report['silent_id']}" if report["silent_id"] is not None else ""
    lines = [
        f"state block read: {report['motors']} motor(s), n={report['n']}, "
        f"timeout {report['timeout_ms']:.1f} ms, {mode}{silent}",
        f"  ok {report['ok']}  timeouts {report['timeouts']}",
        f"  p50 {report['p50_ms']:.2f}  p95 {report['p95_ms']:.2f}  p99 {report['p99_ms']:.2f}  "
        f"max {report['max_ms']:.2f} ms (min {report['min_ms']:.2f}, mean {report['mean_ms']:.2f})",
        f"  recommended read_timeout_s = max(1.5*p99, p99+3 ms) = "
        f"{report['recommended_read_timeout_s']:.4f} s",
    ]
    if report["silent_id"] is not None and not report["legacy"]:
        lines.append(
            f"  one attempt with a silent motor should take about {report['timeout_ms']:.1f} ms"
        )
    return lines


# --- EEPROM writes (Mode A) --------------------------------------------------


@dataclass(frozen=True)
class EepromWriteReport:
    """Outcome of :func:`write_eeprom_safely`.

    Attributes:
        mode: ``"already_off"`` when torque was off on every target (writes
            went through, or nothing needed writing); ``"refused"`` when any
            target had torque on (nothing was written); ``"cancelled"`` when
            ``confirm`` declined (nothing was written).
        item: Name of the register.
        previous: Value read from each target before the write.
        written: ``{name: value}`` actually written and read back.
        skipped_same_value: Targets that already held the requested value.
        torque_on: Targets with ``TORQUE_ENABLE == 1`` (only when refused).
    """

    mode: Literal["already_off", "refused", "cancelled"]
    item: str
    previous: Dict[str, int]
    written: Dict[str, int]
    skipped_same_value: Dict[str, int]
    torque_on: Tuple[str, ...] = ()


def _check_eeprom_value(item: XControlTable, motor: DynamixelMotor, value: int) -> None:
    """Rejects values the firmware would refuse (or that make no sense for the motor)."""
    if item is XControlTable.RETURN_DELAY_TIME:
        low, high = 0, 254
    elif item is XControlTable.CURRENT_LIMIT:
        low, high = 0, CURRENT_LIMIT_MAX_RAW.get(motor.model_name, 0xFFFF)
    else:  # position limits
        low, high = 0, motor.resolution - 1
    if not low <= value <= high:
        raise ValueError(
            f"{item.name}={value} is outside {low}..{high} for '{motor.motor_name}' "
            f"({motor.model_name})."
        )


def write_eeprom_safely(
    bus: DynamixelBus,
    item: XControlTable,
    values: Mapping[str, int],
    *,
    confirm: Callable[[str], bool] | None = None,
) -> EepromWriteReport:
    """Writes a whitelisted EEPROM register, only while torque is already off (Mode A).

    The X series silently ignores EEPROM writes while ``TORQUE_ENABLE`` is 1,
    and switching torque off on an arm joint drops the arm.  So this never
    touches ``TORQUE_ENABLE``: if any target has torque on, nothing is written
    and the report says ``refused``.  Otherwise the motors whose current value
    differs are written with read-back verification.

    Args:
        bus: An open bus.
        item: A key of ``EEPROM_WRITE_WHITELIST``.
        values: ``{motor_name: value}``; every motor must be allowed for ``item``.
        confirm: Called once with a prompt before anything is written; return
            ``False`` to cancel.  ``None`` skips the prompt.

    Raises:
        ValueError: For an item outside the whitelist, a motor not allowed for
            it, an unknown motor, an empty mapping, or a value out of range.
        ConnectionError: If a target did not answer the pre-write reads, or
            the read-back after writing disagreed.
    """
    allowed = EEPROM_WRITE_WHITELIST.get(item)
    if allowed is None:
        whitelist = ", ".join(entry.name for entry in EEPROM_WRITE_WHITELIST)
        raise ValueError(f"{item.name} is not on the EEPROM write whitelist ({whitelist}).")
    names = list(values)
    if not names:
        raise ValueError("write_eeprom_safely needs at least one motor.")
    unknown = [name for name in names if name not in bus.motors]
    if unknown:
        raise ValueError(f"Unknown motor(s) on {bus.port_handler.port_name}: {unknown}")
    forbidden = [name for name in names if name not in allowed]
    if forbidden:
        raise ValueError(
            f"{item.name} may only be written on {sorted(allowed)}; refused for {forbidden}."
        )
    wanted = {name: int(values[name]) for name in names}
    for name, value in wanted.items():
        _check_eeprom_value(item, bus.motors[name], value)

    torque = bus.sync_read(XControlTable.TORQUE_ENABLE, names)
    current = bus.sync_read(item, names)
    silent = [name for name in names if name not in torque or name not in current]
    if silent:
        raise ConnectionError(f"No reading from {silent}; not writing {item.name}.")
    previous = {name: int(current[name]) for name in names}

    torque_on = tuple(name for name in names if int(torque[name]) != 0)
    if torque_on:
        logger.error(
            "%s: torque is on for %s; nothing written. Power-cycle the arm in a hanging pose "
            "and re-run before connecting (Mode A only).",
            item.name,
            list(torque_on),
        )
        return EepromWriteReport(
            "refused", item.name, previous, {}, dict(previous), torque_on=torque_on
        )

    pending = {name: value for name, value in wanted.items() if previous[name] != value}
    skipped = {name: previous[name] for name in names if name not in pending}
    if not pending:
        logger.info("%s already set on %s; nothing to write.", item.name, names)
        return EepromWriteReport("already_off", item.name, previous, {}, skipped)

    if confirm is not None:
        changes = ", ".join(f"{name}: {previous[name]} -> {new}" for name, new in pending.items())
        prompt = (
            f"Torque is off on all {len(names)} target motor(s). Write {item.name} "
            f"({changes}) on {bus.port_handler.port_name}? [y/N] "
        )
        if not confirm(prompt):
            logger.info("%s write cancelled; nothing written.", item.name)
            return EepromWriteReport("cancelled", item.name, previous, {}, skipped)

    bus.write_with_readback(item, dict(pending))
    logger.info("%s written on %s: %s", item.name, bus.port_handler.port_name, pending)
    return EepromWriteReport("already_off", item.name, previous, pending, skipped)


# --- release ----------------------------------------------------------------


@dataclass(frozen=True)
class ReleaseReport:
    """Outcome of :func:`release_arm`.

    Attributes:
        restored_position_mode: Current-capable joints that were not in
            position mode and were switched back to it first.
        goal_positions: ``GOAL_POSITION`` written to those joints (their
            position as read after the mode change).
        released: Every motor whose torque was switched off.
    """

    restored_position_mode: Tuple[str, ...]
    goal_positions: Dict[str, int]
    released: Tuple[str, ...]


def release_arm(bus: DynamixelBus, side: Side) -> ReleaseReport:
    """Switches torque off on every motor of the bus, restoring position mode first.

    For each current-capable joint (``RAKUDA_CURRENT_CAPABLE_JOINTS``) whose
    ``OPERATING_MODE`` is not position control: torque off, ``OPERATING_MODE=3``
    with read-back, then ``GOAL_POSITION`` = position read *after* the mode
    change (a current-mode reading can be multi-turn, outside the position-mode
    goal range).  Then torque is switched off on every motor.  The library does
    not ask for confirmation; the CLI does.

    Args:
        bus: An open bus.
        side: Which arm this is, for log messages.

    Raises:
        ConnectionError: If a joint did not answer while its mode was being
            restored; torque is still switched off on the whole bus.
    """
    names = list(bus.motors)
    current_capable = [name for name in RAKUDA_CURRENT_CAPABLE_JOINTS if name in bus.motors]
    modes = bus.sync_read(XControlTable.OPERATING_MODE, current_capable)
    stale = [name for name in current_capable if modes.get(name) != OperatingMode.POSITION]
    goals: Dict[str, int] = {}
    try:
        if stale:
            logger.warning(
                "%s bus: %s not in position mode (%s); restoring mode 3 before release.",
                side,
                stale,
                {name: modes.get(name) for name in stale},
            )
            bus.torque_disabled(stale)
            bus.write_with_readback(
                XControlTable.OPERATING_MODE, {name: int(OperatingMode.POSITION) for name in stale}
            )
            present = bus.sync_read(XControlTable.PRESENT_POSITION, stale)
            missing = [name for name in stale if name not in present]
            if missing:
                raise ConnectionError(f"No position from {missing} after restoring mode 3.")
            goals = {name: int(present[name]) for name in stale}
            bus.write_with_readback(XControlTable.GOAL_POSITION, dict(goals))
    finally:
        bus.torque_disabled()
        logger.info("%s bus: torque off on %d motor(s).", side, len(names))
    return ReleaseReport(tuple(stale), goals, tuple(names))


# --- CLI --------------------------------------------------------------------


def open_bus(port: str, side: Side) -> DynamixelBus:
    """Builds the ``side`` bus on ``port`` and opens it."""
    bus = DynamixelBus(port=port, motors=rakuda_motor_table(side))
    bus.open()
    return bus


def _ask(prompt: str) -> bool:
    """``y/N`` question on stdin."""
    try:
        answer = input(prompt)
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _classify_port(port: str) -> Side:
    side = classify_rakuda_bus(scan_port(port))
    if side is None:
        raise OSError(f"{port} does not look like a Rakuda bus; pass --side explicitly.")
    logger.info("%s classified as the %s bus.", port, side)
    return side


def _side_of(args: argparse.Namespace) -> Side:
    side: Side | None = args.side
    return side if side is not None else _classify_port(args.port)


def _model_mismatches(bus: DynamixelBus, found: Mapping[str, int]) -> List[str]:
    """Describes every answering motor whose ``MODEL_NUMBER`` differs from the declared table."""
    return [
        f"{name} (ID {bus.motors[name].id}) reports model {model_number}, declared "
        f"{bus.motors[name].model_name} ({bus.motors[name].model_number})"
        for name, model_number in found.items()
        if model_number != bus.motors[name].model_number
    ]


def _cmd_scan(args: argparse.Namespace) -> int:
    scans = scan_candidates(args.candidates or None)
    if not scans:
        print("No candidate ports found (" + ", ".join(CANDIDATE_PATTERNS) + ").")
        return 1
    for scan in scans:
        print(f"{scan.port}  by-id: {scan.by_id or '-'}  latency_timer: {scan.latency_timer}")
        if scan.latency_timer is not None and scan.latency_timer != 1:
            print(f"  WARNING: latency_timer is {scan.latency_timer} ms; set it to 1.")
        print(f"  {scan.describe()}")
        groups: Dict[Tuple[int, int], List[int]] = {}
        for motor_id, motor in scan.motors.items():
            groups.setdefault((motor.model_number, motor.firmware), []).append(motor_id)
        for (model, firmware), ids in sorted(groups.items()):
            print(f"  model {model} fw {firmware}: IDs {sorted(ids)}")
    try:
        ports = select_rakuda_ports(scans)
    except OSError as exc:
        print(str(exc))
        return 1
    print(f"leader: {ports.leader}\nfollower: {ports.follower}")
    print("YAML: leader: {port: auto} / follower: {port: auto} resolves to the same result.")
    return 0


def _cmd_show(args: argparse.Namespace) -> int:
    side = _side_of(args)
    bus = open_bus(args.port, side)
    try:
        names = list(bus.motors)
        found = bus.read_model_numbers(names)
        missing = [name for name in names if name not in found]
        if missing:
            logger.warning("No response from %s; they are left out of the table.", missing)
        for mismatch in _model_mismatches(bus, found):
            logger.warning("%s.", mismatch)
        rows = dump_registers(bus, [name for name in names if name in found])
    finally:
        bus.close()
    print(f"{side} bus on {args.port} ({len(rows)} of {len(names)} motors):")
    for line in format_register_table(rows):
        print(line)
    units = ", ".join(f"{model} {unit} mA/LSB" for model, unit in CURRENT_UNIT_MA.items())
    print(f"current units: {units}; drv bits: {decode_drive_mode(0x0F)}")
    return 0


def _cmd_bench(args: argparse.Namespace) -> int:
    side = _side_of(args)
    bus = open_bus(args.port, side)
    try:
        report = bench_state_block(
            bus,
            list(bus.motors),
            n=args.n,
            timeout_s=args.timeout_ms / 1e3,
            silent_id=args.silent_id,
            legacy=args.legacy,
        )
    finally:
        bus.close()
    print(f"{side} bus on {args.port}:")
    for line in format_bench_report(report):
        print(line)
    return 0


def _print_eeprom_report(report: EepromWriteReport) -> int:
    if report.mode == "refused":
        print(
            f"{report.item}: REFUSED, torque is on for {list(report.torque_on)}; nothing written. "
            "Power-cycle the arm in a hanging pose and re-run before connecting."
        )
        return EXIT_REFUSED
    if report.mode == "cancelled":
        print(f"{report.item}: cancelled; nothing written.")
        return 1
    if report.written:
        print(f"{report.item}: written and read back {report.written}")
    if report.skipped_same_value:
        print(f"{report.item}: already set on {report.skipped_same_value}")
    return 0


def _run_eeprom_write(
    args: argparse.Namespace, item: XControlTable, targets: Callable[[DynamixelBus], List[str]]
) -> int:
    side = _side_of(args)
    bus = open_bus(args.port, side)
    try:
        names = targets(bus)
        bus.verify_models(names)
        report = write_eeprom_safely(
            bus, item, {name: args.value for name in names}, confirm=None if args.yes else _ask
        )
    finally:
        bus.close()
    return _print_eeprom_report(report)


def _cmd_set_return_delay(args: argparse.Namespace) -> int:
    return _run_eeprom_write(args, XControlTable.RETURN_DELAY_TIME, lambda bus: list(bus.motors))


def _cmd_write_eeprom(args: argparse.Namespace) -> int:
    item = XControlTable[args.item]
    motors = [name.strip() for name in args.motors.split(",") if name.strip()]
    if not motors:
        raise ValueError("--motors needs at least one motor name.")
    return _run_eeprom_write(args, item, lambda bus: motors)


def _cmd_release(args: argparse.Namespace) -> int:
    side: Side = args.side
    bus = open_bus(args.port, side)
    try:
        # Leader and follower share the same ID layout, so --side alone cannot
        # tell the buses apart: check the model numbers before anything is
        # written, or the wrong (unsupported) arm would be dropped.  A silent
        # motor must not block a release, so only answering motors are compared.
        names = list(bus.motors)
        found = bus.read_model_numbers(names)
        silent = [name for name in names if name not in found]
        if silent:
            logger.warning("No response from %s; torque is still switched off on them.", silent)
        mismatches = _model_mismatches(bus, found)
        if mismatches:
            print(f"{args.port} does not carry the {side} arm; nothing written:")
            for mismatch in mismatches:
                print(f"  {mismatch}")
            print("Run `scan` and pass the --port/--side pair it reports.")
            return 1
        if not args.yes and not _ask(
            f"Switch torque OFF on every motor of the {side} bus at {args.port} "
            f"({len(found)} of {len(names)} motors answered with the {side} model numbers)? "
            "The arm will drop if it is not supported. [y/N] "
        ):
            print("cancelled.")
            return 1
        report = release_arm(bus, side)
    finally:
        bus.close()
    if report.restored_position_mode:
        print(
            f"restored position mode on {list(report.restored_position_mode)} "
            f"(goal positions {report.goal_positions})"
        )
    print(f"torque off on {len(report.released)} motor(s) of the {side} bus.")
    return 0


def build_parser() -> argparse.ArgumentParser:
    """Builds the ``robopy-rakuda-ports`` argument parser."""
    parser = argparse.ArgumentParser(
        prog="robopy-rakuda-ports",
        description=(
            "Rakuda serial-port detection and register maintenance. "
            "Run as `env -u PYTHONPATH uv run --frozen python -m "
            "robopy.robots.rakuda.rakuda_ports <subcommand>`."
        ),
    )
    parser.add_argument("-v", "--verbose", action="store_true", help="debug logging")
    subparsers = parser.add_subparsers(dest="command", required=True)

    scan = subparsers.add_parser("scan", help="ping every serial port and classify the buses")
    scan.add_argument("candidates", nargs="*", help="device nodes to scan (default: all)")
    scan.set_defaults(func=_cmd_scan)

    def add_port_and_side(sub: argparse.ArgumentParser, side_required: bool = False) -> None:
        sub.add_argument("--port", required=True, help="device node, e.g. /dev/ttyUSB1")
        sub.add_argument(
            "--side",
            choices=SIDES,
            required=side_required,
            default=None,
            help="which arm is on the port" + ("" if side_required else " (default: scan)"),
        )

    show = subparsers.add_parser("show", help="print the bring-up register table (reads only)")
    add_port_and_side(show)
    show.set_defaults(func=_cmd_show)

    bench = subparsers.add_parser("bench", help="time state-block reads")
    add_port_and_side(bench)
    bench.add_argument("-n", type=int, default=500, help="number of reads (default 500)")
    bench.add_argument(
        "--timeout-ms", type=float, default=20.0, help="per-attempt bound (default 20)"
    )
    bench.add_argument(
        "--silent-id", type=int, default=None, help="unused ID to add as a silent motor"
    )
    bench.add_argument("--legacy", action="store_true", help="use txRxPacket() (SDK timeout)")
    bench.set_defaults(func=_cmd_bench)

    rdt = subparsers.add_parser(
        "set-return-delay", help="write RETURN_DELAY_TIME on every motor (Mode A)"
    )
    add_port_and_side(rdt)
    rdt.add_argument("--value", type=int, required=True, help="0..254 (x2 us)")
    rdt.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    rdt.set_defaults(func=_cmd_set_return_delay)

    eeprom = subparsers.add_parser("write-eeprom", help="write a whitelisted EEPROM item (Mode A)")
    add_port_and_side(eeprom)
    eeprom.add_argument(
        "--item", required=True, choices=[item.name for item in EEPROM_WRITE_WHITELIST]
    )
    eeprom.add_argument("--motors", required=True, help="comma-separated motor names")
    eeprom.add_argument("--value", type=int, required=True)
    eeprom.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    eeprom.set_defaults(func=_cmd_write_eeprom)

    release = subparsers.add_parser(
        "release", help="restore position mode where needed, then torque off the whole bus"
    )
    add_port_and_side(release, side_required=True)
    release.add_argument("--yes", action="store_true", help="do not ask for confirmation")
    release.set_defaults(func=_cmd_release)
    return parser


def main(argv: Sequence[str] | None = None) -> int:
    """CLI entry point; returns the process exit code."""
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    check_sdk_location()
    try:
        return int(args.func(args))
    except KeyboardInterrupt:
        print("interrupted.")
        return 130
    except (OSError, ValueError) as exc:
        logger.error("%s", exc, exc_info=args.verbose)
        return 1


if __name__ == "__main__":
    sys.exit(main())
