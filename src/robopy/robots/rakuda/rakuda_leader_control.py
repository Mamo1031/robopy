"""Leader-side bilateral control: the current loop, follower I/O and the hold.

This module is the home of the leader current-control stack: the exception
hierarchy, the loop state machine vocabulary (:class:`LoopState`,
:class:`LoopFault`), :func:`hold_joints` (the "stop = hold in place" sequence
that the loop, the connection code and the CLI all share),
:class:`FollowerPositionIO` (the position copy to the follower, run from the
control thread) and :class:`LeaderCurrentLoop` (preflight, configure,
the 50 Hz cycle, the latched faults and the control thread life cycle).
The control law itself lives in ``rakuda_control_laws``; the loop only
calls its ``reset``/``compute``.

Nothing in this module opens a port: every function takes a bus object
(:class:`robopy.motor.dynamixel_bus.DynamixelBus` or
:class:`robopy.motor.sim_dynamixel_bus.SimulatedDynamixelBus`) that already is.
"""

from __future__ import annotations

import logging
import threading
import time
from collections import deque
from dataclasses import dataclass, replace
from enum import Enum
from typing import (
    Any,
    Callable,
    Deque,
    Dict,
    List,
    Literal,
    Mapping,
    Protocol,
    Sequence,
    Tuple,
)

import numpy as np
from numpy.typing import NDArray

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_HEAD_JOINT_NAMES,
    RakudaArmObs,
    RakudaArmState,
    RakudaBilateralParams,
)
from robopy.motor.dynamixel_bus import (
    DiagnosticReading,
    DynamixelCommError,
    DynamixelMotor,
    DynamixelTimeoutError,
    MotorStateReading,
)
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, OperatingMode, XControlTable

from .rakuda_control_laws import ControlLaw, LawOutput, SignCheckOvershoot

__all__ = [
    "HOLD_BUDGET_S",
    "HOLD_BLOCK_READ_TIMEOUT_S",
    "HOLD_MODE_SETTLE_S",
    "BilateralError",
    "BilateralNotReady",
    "ConfigureError",
    "ControlBus",
    "FollowerLost",
    "FollowerPositionIO",
    "HoldFailed",
    "HoldParams",
    "HoldReport",
    "HoldSource",
    "LeaderCurrentLoop",
    "LoopFault",
    "LoopState",
    "LoopStopped",
    "PairSnapshot",
    "PositionSnapshot",
    "hold_joints",
    "pair_snapshot_to_obs",
    "wrap_delta_counts",
]

logger = logging.getLogger(__name__)

#: Upper bound on one hold sequence.  Every read inside the torque-off window
#: is deadline-bounded (``HOLD_BLOCK_READ_TIMEOUT_S`` per silent joint, at
#: most twice per joint), so twelve silent joints cost the window about 1.0 s;
#: the read-back after torque-on (steps 7-8, legacy ``sync_read``) adds at most
#: two 10-retry timeouts of about 0.37 s when a joint held at the snapshot does
#: not answer.  Worst case about 2.0 s; the typical case is 30-60 ms.
HOLD_BUDGET_S: float = 3.0
#: Receive budget of one bounded read inside the hold: the grouped state
#: block read and the one-joint reads that follow a failure.
HOLD_BLOCK_READ_TIMEOUT_S: float = 0.04
#: Pause between the ``OPERATING_MODE`` write and its read-back.
HOLD_MODE_SETTLE_S: float = 0.005

#: Counts per revolution of the X-series absolute encoder.
_COUNTS_PER_TURN = 4096


# --- exceptions and loop vocabulary ---------------------------------------------


class LoopState(Enum):
    """Life cycle of ``LeaderCurrentLoop``.

    ``IDLE -> CONFIGURED -> RUNNING -> HELD | FAULT``; ``RELEASED`` after ``release()``.
    """

    IDLE = "idle"
    CONFIGURED = "configured"
    RUNNING = "running"
    HELD = "held"
    FAULT = "fault"
    RELEASED = "released"


@dataclass(frozen=True)
class LoopFault:
    """Why the loop stopped, latched until the next start.

    Attributes:
        reason: Short machine-readable cause (``"overheat"``, ``"snapshot_stale"``, ...).
        detail: Human-readable detail for logs and ``metadata.json``.
        t_ns: ``time.monotonic_ns()`` when the fault was raised.
        hold_window_ms: Torque-off window of the hold that followed, or ``None``
            when no hold ran (or it did not get as far as torque-on).
        state_after: The loop state once the hold finished.
    """

    reason: str
    detail: str
    t_ns: int
    hold_window_ms: float | None
    state_after: LoopState


class BilateralError(RuntimeError):
    """Base class of every error the bilateral stack raises on purpose."""


class LoopStopped(BilateralError):
    """The loop is not RUNNING, so the requested operation cannot proceed.

    Attributes:
        state: The loop state at the time of the call.
        fault: The latched fault, or ``None`` when the operator stopped the loop.
    """

    def __init__(self, state: LoopState, fault: LoopFault | None = None) -> None:
        self.state = state
        self.fault = fault
        if fault is not None:
            why = f"{fault.reason}: {fault.detail}"
        elif state is LoopState.HELD:
            why = "stopped by operator"
        else:
            why = "not running"
        super().__init__(f"LeaderCurrentLoop is {state.name} ({why})")


class BilateralNotReady(BilateralError):
    """``wait_pair()`` timed out before a fresh leader/follower pair was published."""


class FollowerLost(BilateralError):
    """The follower bus stopped answering; the leader keeps gravity compensation."""


class ConfigureError(BilateralError):
    """``configure()`` could not bring the joints into current mode and rolled back."""


class HoldFailed(BilateralError):
    """A bus call inside :func:`hold_joints` raised; some joints may be torque-off."""


# --- inputs --------------------------------------------------------------------


class ControlBus(Protocol):
    """The subset of the Dynamixel bus API the leader control stack uses.

    Both :class:`~robopy.motor.dynamixel_bus.DynamixelBus` and
    :class:`~robopy.motor.sim_dynamixel_bus.SimulatedDynamixelBus` satisfy it.
    """

    @property
    def motors(self) -> Mapping[str, DynamixelMotor]: ...

    def torque_disabled(self, specific_motor_names: List[str] | None = None) -> None: ...

    def torque_enabled(self, specific_motor_names: List[str] | None = None) -> None: ...

    def sync_write(self, item: Enum, values: Dict[str, int | float]) -> None: ...

    def sync_read(self, item: Enum, motor_names: List[str]) -> Dict[str, Any]: ...

    def read_state_block(
        self,
        motor_names: Sequence[str],
        *,
        timeout_s: float,
        attempts: int = 1,
    ) -> Tuple[Dict[str, MotorStateReading], int, int]: ...

    def write_goal_current_raw(self, values: Mapping[str, int]) -> None: ...

    def write_with_readback(
        self,
        item: Enum,
        values: Mapping[str, int | float],
        *,
        tolerance: int = 0,
        attempts: int = 5,
        settle_s: float = 0.005,
    ) -> None: ...

    def read_diagnostics(
        self,
        motor_names: Sequence[str],
        *,
        timeout_s: float = 0.05,
    ) -> Dict[str, DiagnosticReading]: ...

    def read_model_numbers(self, motor_names: Sequence[str]) -> Dict[str, int]: ...

    def verify_models(self, motor_names: Sequence[str] | None = None) -> None: ...


class PositionSnapshot(Protocol):
    """The last good reading a hold may fall back on (:func:`hold_joints` step 4c).

    ``RakudaArmState`` satisfies it; any object with these three read-only
    attributes does.  ``position`` is in ``names`` order and may hold
    multi-turn values (current mode); the hold reduces them to one turn.
    """

    @property
    def names(self) -> Sequence[str]: ...

    @property
    def position(self) -> Sequence[int] | NDArray[np.integer[Any]]: ...

    @property
    def t_end_ns(self) -> int: ...


@dataclass(frozen=True)
class HoldParams:
    """Tunables of :func:`hold_joints`; see ``RakudaBilateralParams.hold_*``.

    Attributes:
        read_retries: State-block read attempts (step 4a) before falling back
            to one read per joint.
        max_snapshot_age_s: A snapshot older than this (measured at the start
            of the hold) is not used as a goal; the joint stays torque-off.
        max_jump_counts: A fresh reading further than this from a usable
            snapshot (shortest way around the turn) is re-read once, then
            distrusted.  A snapshot older than ``max_snapshot_age_s`` is no
            reference: the bound is sized for that age.
        profile_velocity: ``PROFILE_VELOCITY`` written after the mode change
            (which zeroes it), so an overshoot returns in a trapezoid.
        readback_attempts: ``write_with_readback`` rounds for ``OPERATING_MODE``
            on the joints that answer; a silent joint is never retried inside
            the torque-off window.
    """

    read_retries: int = 3
    max_snapshot_age_s: float = 0.06
    max_jump_counts: int = 200
    profile_velocity: int = 40
    readback_attempts: int = 5

    def __post_init__(self) -> None:
        if self.read_retries < 1:
            raise ValueError(f"read_retries must be >= 1, got {self.read_retries}")
        if self.max_snapshot_age_s < 0.0:
            raise ValueError(f"max_snapshot_age_s must be >= 0, got {self.max_snapshot_age_s}")
        if self.max_jump_counts < 0:
            raise ValueError(f"max_jump_counts must be >= 0, got {self.max_jump_counts}")
        if not 0 <= self.profile_velocity <= 32767:
            raise ValueError(
                f"profile_velocity must be within 0..32767, got {self.profile_velocity}"
            )
        if self.readback_attempts < 1:
            raise ValueError(f"readback_attempts must be >= 1, got {self.readback_attempts}")

    @classmethod
    def from_bilateral_params(cls, params: Any) -> "HoldParams":
        """Builds the hold tunables from a ``RakudaBilateralParams``-like object.

        Only the attribute names are relied on (``hold_read_retries``,
        ``effective_hold_max_snapshot_age_s``, ``hold_max_jump_counts``,
        ``hold_profile_velocity``), so this module does not import the config.
        """
        return cls(
            read_retries=int(getattr(params, "hold_read_retries")),
            max_snapshot_age_s=float(getattr(params, "effective_hold_max_snapshot_age_s")),
            max_jump_counts=int(getattr(params, "hold_max_jump_counts")),
            profile_velocity=int(getattr(params, "hold_profile_velocity")),
        )


# --- report --------------------------------------------------------------------


HoldSource = Literal["block_read", "single_read", "snapshot", "none"]


@dataclass(frozen=True)
class HoldReport:
    """What one :func:`hold_joints` call did.

    Attributes:
        source: Where each joint's goal came from: ``"block_read"`` (step 4a),
            ``"single_read"`` (step 4b or the jump-check re-read),
            ``"snapshot"`` (step 4c) or ``"none"`` (left torque-off).
        goal_counts: ``GOAL_POSITION`` written per held joint.
        none: Joints left torque-off, in call order.
        sag_counts: ``goal - snapshot`` (shortest way around) for joints that
            had both; how far the joint sagged in the torque-off window.
        window_ms: Torque-off window, from the start of the call to the end of
            the ``TORQUE_ENABLE=1`` write (or to the decision that nothing can
            be held).
        goal_rewritten_at_torque_on: The firmware changed ``GOAL_POSITION``
            when torque came on, and the second write restored it (H3).
        verified: Every joint is held: none were left off, and the read-back
            of ``TORQUE_ENABLE``/``GOAL_POSITION`` matched.
        notes: Human-readable anomalies, for logs and ``control_report()``.
    """

    source: Dict[str, HoldSource]
    goal_counts: Dict[str, int]
    none: Tuple[str, ...]
    sag_counts: Dict[str, int]
    window_ms: float
    goal_rewritten_at_torque_on: bool
    verified: bool
    notes: Tuple[str, ...]


# --- helpers -------------------------------------------------------------------


def wrap_delta_counts(a: int, b: int) -> int:
    """Shortest signed difference ``a - b`` on the 4096-count encoder circle (-2048..2047)."""
    return (a - b + _COUNTS_PER_TURN // 2) % _COUNTS_PER_TURN - _COUNTS_PER_TURN // 2


def _first_line(exc: BaseException) -> str:
    """The first line of an exception message (``DynamixelCommError`` adds a second)."""
    text = str(exc)
    return text.splitlines()[0] if text else type(exc).__name__


def _read_position(bus: ControlBus, name: str) -> int | None:
    """One deadline-bounded position read of one joint; ``None`` when it did not answer.

    This is the only per-joint read inside the torque-off window: it costs at
    most ``HOLD_BLOCK_READ_TIMEOUT_S``, where the legacy ``sync_read`` would
    retry a silent joint ten times (about 0.37 s).
    """
    try:
        readings, _, _ = bus.read_state_block(
            [name], timeout_s=HOLD_BLOCK_READ_TIMEOUT_S, attempts=1
        )
    except DynamixelCommError:
        return None
    return readings[name].position


def _read_items(
    bus: ControlBus, item: XControlTable, names: Sequence[str]
) -> Dict[str, int] | None:
    """Grouped ``sync_read``; ``None`` when the transaction failed."""
    try:
        values = bus.sync_read(item, list(names))
    except DynamixelCommError:
        return None
    return {name: int(value) for name, value in values.items()}


def _snapshot_goals(snapshot: PositionSnapshot | None, names: Sequence[str]) -> Dict[str, int]:
    """``{joint: position mod 4096}`` for the joints the snapshot covers."""
    if snapshot is None:
        return {}
    wanted = set(names)
    return {
        str(name): int(position) % _COUNTS_PER_TURN
        for name, position in zip(snapshot.names, snapshot.position)
        if name in wanted
    }


def _confirm_position_mode(
    bus: ControlBus, names: Sequence[str], attempts: int, log: logging.Logger
) -> bool:
    """``OPERATING_MODE=3`` with read-back for ``names``; False when it did not confirm.

    Each round reads every joint back with the bus's fixed read-back timeout,
    so a silent joint costs about 40 ms per round.
    """
    try:
        bus.write_with_readback(
            XControlTable.OPERATING_MODE,
            {name: OperatingMode.POSITION for name in names},
            attempts=attempts,
            settle_s=HOLD_MODE_SETTLE_S,
        )
    except DynamixelCommError as exc:
        log.warning("hold: OPERATING_MODE=3 read-back failed (%s)", _first_line(exc))
        return False
    return True


def _enter_position_mode(
    bus: ControlBus,
    names: Sequence[str],
    params: HoldParams,
    notes: List[str],
    log: logging.Logger,
) -> Tuple[List[str], List[str]]:
    """Step 2: ``OPERATING_MODE=3`` with read-back; returns ``(confirmed, silent)``.

    The fast path is one grouped ``write_with_readback`` round.  When it fails
    the exception does not say which joint failed or why, and every read from
    here on must stay deadline-bounded (one dead joint must not stall the
    other eleven torque-off).  So each joint gets one bounded
    position read to tell "does not answer" from "answers": a silent joint
    stays in play, because the write was sent and a fresh snapshot may still
    hold it (step 4c), and is not read again inside the window; the joints
    that answer are confirmed together with ``readback_attempts`` rounds and,
    only if that fails too, one by one.  A joint that answers but does not
    confirm mode 3 is dropped: torque-enabling it in mode 0 would resume
    driving the last ``GOAL_CURRENT``.
    """
    if _confirm_position_mode(bus, names, 1, log):
        return list(names), []
    silent: List[str] = []
    responsive: List[str] = []
    for name in names:
        if _read_position(bus, name) is None:
            silent.append(name)
            notes.append(f"{name}: OPERATING_MODE=3 not confirmed (no answer)")
        else:
            responsive.append(name)
    if not responsive or _confirm_position_mode(bus, responsive, params.readback_attempts, log):
        return responsive, silent
    confirmed: List[str] = []
    for name in responsive:
        if _confirm_position_mode(bus, [name], 1, log):
            confirmed.append(name)
        else:
            notes.append(f"{name}: OPERATING_MODE=3 not confirmed by read-back; torque left off")
            log.warning("hold: %s did not confirm mode 3; it will not be torque-enabled", name)
    return confirmed, silent


def _read_positions(
    bus: ControlBus,
    names: Sequence[str],
    params: HoldParams,
    log: logging.Logger,
) -> Tuple[Dict[str, int], HoldSource]:
    """Steps 4a/4b: one state block read (retried), then one bounded read per joint."""
    if not names:
        return {}, "none"
    for attempt in range(1, params.read_retries + 1):
        try:
            readings, _, _ = bus.read_state_block(
                names, timeout_s=HOLD_BLOCK_READ_TIMEOUT_S, attempts=1
            )
        except DynamixelCommError as exc:
            log.warning(
                "hold: state block read %d/%d failed (%s)",
                attempt,
                params.read_retries,
                _first_line(exc),
            )
            continue
        return {name: readings[name].position for name in names}, "block_read"
    positions: Dict[str, int] = {}
    for name in names:
        position = _read_position(bus, name)
        if position is not None:
            positions[name] = position
    return positions, "single_read"


def _verify_hold(
    bus: ControlBus, names: Sequence[str], goals: Mapping[str, int]
) -> List[str] | None:
    """Step 8 read-back: ``TORQUE_ENABLE`` all 1 and ``GOAL_POSITION`` as written.

    Returns the mismatches (empty = verified), or ``None`` when the read-back
    itself failed and there is nothing to compare.
    """
    torque = _read_items(bus, XControlTable.TORQUE_ENABLE, names)
    goal_read = None if torque is None else _read_items(bus, XControlTable.GOAL_POSITION, names)
    if torque is None or goal_read is None:
        return None
    problems: List[str] = []
    for name in names:
        if torque.get(name) != 1:
            problems.append(f"{name}: TORQUE_ENABLE={torque.get(name)}")
        if goal_read.get(name) != goals[name]:
            problems.append(f"{name}: GOAL_POSITION={goal_read.get(name)} != {goals[name]}")
    return problems


# --- the hold sequence ---------------------------------------------------------


def hold_joints(
    bus: ControlBus,
    joints: Sequence[str],
    params: HoldParams,
    snapshot: PositionSnapshot | None = None,
    *,
    clock: Callable[[], int] = time.monotonic_ns,
    logger: logging.Logger | None = None,
) -> HoldReport:
    """Holds ``joints`` in place: position mode, goal = where they are, torque on.

    The sequence keeps three hold principles: (H1) a joint is only
    torque-enabled after its ``GOAL_POSITION`` was written in this call, (H2)
    the goal is preferably a reading taken *after* the switch to mode 3 (one
    turn), with the snapshot as a bounded-age fallback, and (H3)
    ``GOAL_POSITION`` is written once before and once after torque-on and
    read back.  Steps, in order:

    0. ``q_ref`` from the snapshot (mod 4096); start of the torque-off window.
    1. ``TORQUE_ENABLE=0``.
    2. ``OPERATING_MODE=3`` with read-back; a joint that answers but does not
       confirm mode 3 is left off ("none"); a joint that does not answer
       stays in play for step 4c.
    3. ``PROFILE_VELOCITY=params.profile_velocity`` (the mode change zeroed it).
    4. Goal per joint: (4a) a state block read of the confirmed joints (up to
       ``read_retries``), else (4b) one bounded read per joint, else (4c) the
       snapshot if not older than ``max_snapshot_age_s``; with such a snapshot,
       a reading more than ``max_jump_counts`` from ``q_ref`` is re-read once
       and then distrusted (an older snapshot is no reference, so it neither
       holds nor vetoes).  Every read inside the window is deadline-bounded
       (``HOLD_BLOCK_READ_TIMEOUT_S``): a silent joint costs at most two
       timeouts, not the legacy ten retries.
    5. ``GOAL_POSITION`` for the joints with a goal.
    6. ``TORQUE_ENABLE=1`` for them (end of the window).
    7. ``GOAL_POSITION`` again, after one read to spot a firmware rewrite.
    8. Read back ``TORQUE_ENABLE`` and ``GOAL_POSITION``; on a mismatch redo
       step 7 once.
    9./10. Build the report.  Joints without any reading are left torque-off
       and reported at CRITICAL; the call never raises for them.

    The function is stateless and may be called again on already held joints
    (they are re-held at their present position).  Callers that need
    exactly-once semantics serialise around it.

    Args:
        bus: An open bus; every joint must be one of ``bus.motors``.
        joints: Joints to hold, in the order the report lists them.
        params: Tunables; see :class:`HoldParams`.
        snapshot: The last good reading of the bus (``RakudaArmState`` or any
            :class:`PositionSnapshot`), or ``None``.
        clock: ``time.monotonic_ns``-compatible clock (tests pass the sim clock).
        logger: Logger for the sequence; defaults to this module's.

    Returns:
        A :class:`HoldReport`; ``verified`` is True only when every joint is held.

    Raises:
        ValueError: If ``joints`` is empty or names a motor the bus lacks.
        HoldFailed: If a bus call raised an exception the sequence does not
            handle (a transmit failure, a programming error, ...).  Joints not
            yet torque-enabled stay off; nothing is retried here.
    """
    log = logger if logger is not None else logging.getLogger(__name__)
    names: Tuple[str, ...] = tuple(dict.fromkeys(joints))
    if not names:
        raise ValueError("hold_joints needs at least one joint.")
    unknown = [name for name in names if name not in bus.motors]
    if unknown:
        raise ValueError(f"Unknown joint(s) for hold: {unknown}")

    # 0. Reference positions and the start of the torque-off window.
    q_ref = _snapshot_goals(snapshot, names)
    t_hold0 = clock()
    snapshot_age_s = None if snapshot is None else (t_hold0 - int(snapshot.t_end_ns)) / 1e9
    snapshot_usable = snapshot_age_s is not None and snapshot_age_s <= params.max_snapshot_age_s

    source: Dict[str, HoldSource] = {}
    goals: Dict[str, int] = {}
    notes: List[str] = []
    rewritten = False
    step = "TORQUE_ENABLE=0"
    try:
        # 1. Torque off: the window opens.
        bus.torque_disabled(list(names))

        # 2. Position mode, confirmed per joint; silent joints stay in play for 4c.
        step = "OPERATING_MODE=3"
        confirmed, silent = _enter_position_mode(bus, names, params, notes, log)
        in_play = [name for name in names if name in confirmed or name in silent]
        for name in names:
            if name not in in_play:
                source[name] = "none"

        # 3. The mode change reset PROFILE_VELOCITY to 0 (= as fast as possible).
        step = "PROFILE_VELOCITY"
        if in_play:
            bus.sync_write(
                XControlTable.PROFILE_VELOCITY,
                {name: params.profile_velocity for name in in_play},
            )

        # 4. A goal for every joint that can be read, validated against a usable q_ref.
        # A joint silent in step 2 already failed one bounded read after the mode
        # write; it is not read again inside the window (snapshot or "none").
        step = "goal acquisition"
        fresh, fresh_source = _read_positions(bus, confirmed, params, log)
        for name in in_play:
            if name in fresh:
                goal = fresh[name] % _COUNTS_PER_TURN
                how: HoldSource = fresh_source
                ref = q_ref.get(name) if snapshot_usable else None
                if ref is not None and abs(wrap_delta_counts(goal, ref)) > params.max_jump_counts:
                    again = _read_position(bus, name)
                    if again is None or abs(wrap_delta_counts(again % _COUNTS_PER_TURN, ref)) > (
                        params.max_jump_counts
                    ):
                        log.critical(
                            "hold: %s reads %d, snapshot says %d (> %d counts apart) and the "
                            "re-read gives %s; torque left OFF",
                            name,
                            goal,
                            ref,
                            params.max_jump_counts,
                            again,
                        )
                        notes.append(f"{name}: reading {goal} disagrees with snapshot {ref}")
                        source[name] = "none"
                        continue
                    goal = again % _COUNTS_PER_TURN
                    how = "single_read"
                goals[name] = goal
                source[name] = how
            elif name in q_ref and snapshot_usable:
                goals[name] = q_ref[name]
                source[name] = "snapshot"
            else:
                source[name] = "none"
        if snapshot_age_s is not None and not snapshot_usable:
            notes.append(
                f"snapshot too old ({snapshot_age_s * 1e3:.0f} ms > "
                f"{params.max_snapshot_age_s * 1e3:.0f} ms)"
            )
        held = [name for name in names if source[name] != "none"]
        if any(source[name] == "snapshot" for name in held):
            log.warning(
                "hold: no fresh reading for %s; holding at the snapshot (%.0f ms old)",
                [name for name in held if source[name] == "snapshot"],
                (snapshot_age_s or 0.0) * 1e3,
            )

        # 5./6. Goal first (H1), then torque on: the window closes.
        if held:
            step = "GOAL_POSITION"
            bus.sync_write(XControlTable.GOAL_POSITION, {name: goals[name] for name in held})
            step = "TORQUE_ENABLE=1"
            bus.torque_enabled(held)
        t_torque_on = clock()

        # 7./8. Second goal write and read-back (H3).
        verified_regs = bool(held)
        if held:
            step = "GOAL_POSITION read-back"
            goal_read = _read_items(bus, XControlTable.GOAL_POSITION, held)
            if goal_read is None:
                notes.append("GOAL_POSITION unreadable right after torque-on")
            else:
                changed = {n: goal_read.get(n) for n in held if goal_read.get(n) != goals[n]}
                if changed:
                    rewritten = True
                    log.info("hold: firmware rewrote GOAL_POSITION at torque-on: %s", changed)
            step = "GOAL_POSITION (second write)"
            bus.sync_write(XControlTable.GOAL_POSITION, {name: goals[name] for name in held})
            step = "verification"
            problems = _verify_hold(bus, held, goals)
            if problems:
                log.warning("hold: read-back mismatch %s; rewriting GOAL_POSITION once", problems)
                bus.sync_write(XControlTable.GOAL_POSITION, {name: goals[name] for name in held})
                problems = _verify_hold(bus, held, goals)
            if problems is None:
                verified_regs = False
                notes.append("TORQUE_ENABLE/GOAL_POSITION read-back failed; hold not verified")
                log.critical("hold: read-back failed; the hold could not be verified")
            elif problems:
                verified_regs = False
                notes.extend(problems)
                log.critical("hold: not verified after the second write: %s", problems)
    except Exception as exc:
        log.critical("hold: aborted at %s: %s", step, _first_line(exc))
        raise HoldFailed(f"hold aborted at {step}: {_first_line(exc)}") from exc

    # 9./10. Report.
    none = tuple(name for name in names if source[name] == "none")
    window_ms = (t_torque_on - t_hold0) / 1e6
    sag = {name: wrap_delta_counts(goals[name], q_ref[name]) for name in held if name in q_ref}
    if none:
        log.critical(
            "cannot hold %s: no position reading; torque left OFF, use the power switch",
            ", ".join(none),
        )
    log.info(
        "hold: %d/%d joint(s) held in %.1f ms (sources %s, sag %s, verified %s)",
        len(held),
        len(names),
        window_ms,
        {name: source[name] for name in names},
        sag,
        verified_regs and not none,
    )
    return HoldReport(
        source=source,
        goal_counts=goals,
        none=none,
        sag_counts=sag,
        window_ms=window_ms,
        goal_rewritten_at_torque_on=rewritten,
        verified=verified_regs and not none,
        notes=tuple(notes),
    )


# --- loop vocabulary -----------------------------------------------------------

#: State-block reads of the preflight bus-budget measurement.
_PREFLIGHT_READS = 50
#: Attempts of the one-off state reads in ``preflight()``/``configure()`` (not the cycle).
_SETUP_READ_ATTEMPTS = 3
#: ``diagnostics_stale`` fault: no successful ``read_diagnostics`` for this long.
_DIAGNOSTICS_STALE_S = 3.0
#: Consecutive cycles longer than ``overrun_warn_factor * T`` before one warning.
_OVERRUN_WARN_STREAK = 20
#: Smallest ``CURRENT_LIMIT`` (in mA) a current-controlled joint may have.
_MIN_CURRENT_LIMIT_MA = 100.0
#: ``DRIVE_MODE`` bits: bit 0 reverse (allowed, recorded), bit 2
#: time-based profile (rejected), bit 3 torque-on-by-goal-update (warning),
#: bits 1 and 4-7 reserved (rejected).
_DRIVE_MODE_REVERSE = 0x01
_DRIVE_MODE_TIME_PROFILE = 0x04
_DRIVE_MODE_TORQUE_ON_BY_GOAL = 0x08
_DRIVE_MODE_RESERVED = 0x02 | 0xF0
#: Registers shown in the preflight table, besides the diagnostics.
_PREFLIGHT_ITEMS: Tuple[XControlTable, ...] = (
    XControlTable.OPERATING_MODE,
    XControlTable.DRIVE_MODE,
    XControlTable.CURRENT_LIMIT,
    XControlTable.RETURN_DELAY_TIME,
    XControlTable.TEMPERATURE_LIMIT,
    XControlTable.SHUTDOWN,
    XControlTable.TORQUE_ENABLE,
    XControlTable.GOAL_CURRENT,
)


@dataclass(frozen=True)
class PairSnapshot:
    """A leader state and the follower state read right after it.

    ``leader.position`` is exactly what :class:`FollowerPositionIO` wrote to
    the follower as ``GOAL_POSITION`` in that cycle; ``seq`` is the loop's
    ``pair_seq`` at publication.
    """

    leader: RakudaArmState
    follower: RakudaArmState
    seq: int


def pair_snapshot_to_obs(snapshot: PairSnapshot) -> RakudaArmObs:
    """One recording frame from a published pair.

    The two states already carry currents in mA with the motor's raw sign,
    so no unit is applied here: positions and velocities are widened to
    float32 and ``leader_t_ns``/``follower_t_ns`` take the two ``t_end_ns``
    stamps for :meth:`RakudaArmObs.stamped`.  ``obs.leader`` is what the
    follower received as ``GOAL_POSITION`` in that cycle.
    """
    return RakudaArmObs.from_states(snapshot.leader, snapshot.follower)


class _Trip(Exception):
    """Raised inside a cycle to latch a loop fault (never leaves the loop)."""

    def __init__(self, reason: str, detail: str) -> None:
        super().__init__(f"{reason}: {detail}")
        self.reason = reason
        self.detail = detail


class _Percentiles:
    """A bounded sample window with a p50/p95/p99/max summary."""

    def __init__(self, maxlen: int = 4096) -> None:
        self._samples: Deque[float] = deque(maxlen=maxlen)
        self.count = 0
        self.max = 0.0

    def record(self, value: float) -> None:
        self._samples.append(value)
        self.count += 1
        if value > self.max:
            self.max = value

    def summary(self) -> Dict[str, float | int]:
        if not self._samples:
            return {"p50": 0.0, "p95": 0.0, "p99": 0.0, "max": 0.0, "count": 0}
        p50, p95, p99 = np.percentile(np.fromiter(self._samples, dtype=np.float64), [50, 95, 99])
        return {
            "p50": float(p50),
            "p95": float(p95),
            "p99": float(p99),
            "max": self.max,
            "count": self.count,
        }


def _main_thread_alive() -> bool:
    return threading.main_thread().is_alive()


def bus_port_name(bus: object) -> str:
    """The device node of ``bus`` (``port_handler.port_name``), or ``"<port>"`` if unknown."""
    port = getattr(getattr(bus, "port_handler", None), "port_name", None)
    return str(port) if port else "<port>"


def ports_command(subcommand: str, port: str, side: str) -> str:
    """A copy-pasteable ``robopy-rakuda-ports`` command line for recovery hints."""
    return f"`robopy-rakuda-ports {subcommand} --port {port} --side {side}`"


def _current_units(
    bus: ControlBus, override: Mapping[str, float] | None = None
) -> Dict[str, float]:
    """mA per raw current count for every motor of ``bus``, from its model (or ``override``).

    Raises:
        ValueError: If a motor's model has no entry in ``CURRENT_UNIT_MA`` and
            ``override`` does not name it.
    """
    units: Dict[str, float] = {}
    for name, motor in bus.motors.items():
        if override is not None and name in override:
            units[name] = float(override[name])
            continue
        unit = CURRENT_UNIT_MA.get(motor.model_name)
        if unit is None:
            raise ValueError(f"No current unit known for '{name}' ({motor.model_name}).")
        units[name] = unit
    return units


def _make_state(
    names: Sequence[str],
    readings: Mapping[str, MotorStateReading],
    start_ns: int,
    end_ns: int,
    seq: int,
    units: Mapping[str, float],
) -> RakudaArmState:
    """Packs one ``read_state_block`` result into a :class:`RakudaArmState` (currents in mA)."""
    return RakudaArmState(
        names=tuple(names),
        position=np.array([readings[name].position for name in names], dtype=np.int32),
        velocity=np.array([readings[name].velocity for name in names], dtype=np.int32),
        current_ma=np.array(
            [readings[name].current_raw * units[name] for name in names], dtype=np.float32
        ),
        t_start_ns=start_ns,
        t_end_ns=end_ns,
        seq=seq,
    )


def _fault_dict(fault: LoopFault) -> Dict[str, Any]:
    return {
        "reason": fault.reason,
        "detail": fault.detail,
        "t_ns": fault.t_ns,
        "hold_window_ms": fault.hold_window_ms,
        "state_after": fault.state_after.name,
    }


# --- follower I/O --------------------------------------------------------------


class FollowerPositionIO:
    """The follower half of one control cycle, run by the control thread.

    :meth:`step` copies the leader positions to the follower by name, exactly
    like today's ``teleoperate_step`` (every motor that is torque-enabled,
    head and grippers included), in one ``sync_write(GOAL_POSITION)``, then
    reads the whole follower in one deadline-bounded ``read_state_block``.

    Args:
        bus: The follower bus; its ``motors`` are the 17 names.
        enabled_names: Follower motors that are torque-enabled; only these
            receive a goal (``resolve_torque_policy(cfg).follower``).
        read_timeout_s: Hard upper bound of the one read attempt.
        head_and_grippers_from_leader: When False the head and gripper motors
            are left out of the copy (for a caller that drives them itself).

    Attributes:
        failures: Failed steps in total.
        consecutive_failures: Failed steps since the last success; the loop
            declares ``follower_lost`` when it reaches ``follower_lost_cycles``.
        steps: Successful steps.
    """

    def __init__(
        self,
        bus: ControlBus,
        enabled_names: Sequence[str],
        read_timeout_s: float,
        *,
        head_and_grippers_from_leader: bool = True,
    ) -> None:
        if read_timeout_s <= 0.0:
            raise ValueError(f"read_timeout_s must be positive, got {read_timeout_s}")
        self.bus = bus
        self.names: Tuple[str, ...] = tuple(bus.motors)
        excluded: Tuple[str, ...] = ()
        if not head_and_grippers_from_leader:
            excluded = RAKUDA_HEAD_JOINT_NAMES + RAKUDA_GRIPPER_JOINT_NAMES
        wanted = set(enabled_names)
        self.copied: Tuple[str, ...] = tuple(
            name for name in self.names if name in wanted and name not in excluded
        )
        self.read_timeout_s = float(read_timeout_s)
        self.failures = 0
        self.consecutive_failures = 0
        self.steps = 0
        self._units = _current_units(bus)
        self._copied_set = frozenset(self.copied)
        self._seq = 0
        self._latest: RakudaArmState | None = None
        self._lock = threading.Lock()

    def step(self, q_L: RakudaArmState) -> RakudaArmState:
        """Writes the leader positions as follower goals, then reads the follower.

        Args:
            q_L: The leader state of this cycle; positions are copied by name
                for every motor in ``copied`` that ``q_L.names`` contains,
                reduced to one turn (``% 4096``): a leader joint in current
                mode can report a multi-turn position, which the follower's
                position mode would reject (Data Range error, silently
                dropped by SyncWrite).

        Returns:
            The follower state read after the write.

        Raises:
            DynamixelCommError: When the write or the read failed (the
                counters are updated first); ``DynamixelTimeoutError`` when
                the read did not complete within ``read_timeout_s``.
        """
        goals: Dict[str, int | float] = {
            str(name): int(position) % _COUNTS_PER_TURN
            for name, position in zip(q_L.names, q_L.position)
            if name in self._copied_set
        }
        try:
            if goals:
                self.bus.sync_write(XControlTable.GOAL_POSITION, goals)
            readings, start_ns, end_ns = self.bus.read_state_block(
                self.names, timeout_s=self.read_timeout_s, attempts=1
            )
        except DynamixelCommError:
            self.failures += 1
            self.consecutive_failures += 1
            raise
        self.consecutive_failures = 0
        self.steps += 1
        self._seq += 1
        state = _make_state(self.names, readings, start_ns, end_ns, self._seq, self._units)
        with self._lock:
            self._latest = state
        return state

    def latest(self) -> RakudaArmState | None:
        """The last follower state read, or ``None`` before the first success."""
        with self._lock:
            return self._latest


# --- the leader current loop ---------------------------------------------------


def _initial_current_ma(law: ControlLaw, leader: RakudaArmState) -> NDArray[np.float64]:
    """The torque-on current of ``configure()``: the law's gravity term.

    ``BilateralLaw.gravity_term()`` is what ``configure()`` writes first; the
    identification laws (``PoseHoldLaw``, ``SignCheckLaw``) return 0 mA, so
    their first cycle runs inside the loop where the per-cycle checks are active.
    """
    current = np.asarray(law.gravity_term(leader), dtype=np.float64)
    if current.shape != (len(law.joints),) or not np.all(np.isfinite(current)):
        raise ValueError(
            f"gravity term must be finite with shape ({len(law.joints)},), got {current!r}"
        )
    return current


class LeaderCurrentLoop:
    """The leader current-control loop.

    Life cycle: :meth:`preflight` (read-only checks) -> :meth:`configure`
    (the current joints ``J`` into current mode, torque on at the law's
    initial current) -> :meth:`start` (daemon thread) or :meth:`run_once`
    per cycle -> :meth:`stop` (hold in place, ``HELD``) or a fault (hold,
    ``FAULT``) -> :meth:`release` (torque off).  While it runs, the loop is
    the only user of the leader bus and, through ``follower``, of the
    follower bus.  Every fault latches, holds the joints
    exactly once and ends the thread; the arm is never released by the loop.

    Args:
        bus: The leader bus (open); its ``motors`` are the 17 names.
        joints: The current-controlled joints ``J`` (``params.current_joints``).
        params: Loop parameters.
        law: The control law (``rakuda_control_laws.ControlLaw``) for ``joints``;
            its optional ``joint_range_counts`` mapping enables the range checks.
        units_ma: mA per raw current count of every joint in ``joints``.
        current_sign: ``+1``/``-1`` per joint, applied to the law's joint-convention
            current right before it is written.
        follower: The follower I/O run every ``follower_divider`` cycles, or ``None``.
        clock: ``time.monotonic_ns``-compatible clock (tests pass a simulated one).
        sleep: ``time.sleep``-compatible sleep used between cycles and in preflight.
        main_alive: Checked at the top of every cycle (``main_thread_exited``).
        gripper_hold: ``GOAL_POSITION`` written to the leader grippers at the end
            of :meth:`configure` (``LEADER_GRIP_HOLD_POSITION``), or ``None``.
    """

    def __init__(
        self,
        bus: ControlBus,
        joints: Sequence[str],
        params: RakudaBilateralParams,
        law: ControlLaw,
        units_ma: Mapping[str, float],
        current_sign: Mapping[str, int],
        follower: FollowerPositionIO | None = None,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
        main_alive: Callable[[], bool] = _main_thread_alive,
        gripper_hold: Mapping[str, int] | None = None,
    ) -> None:
        names: Tuple[str, ...] = tuple(dict.fromkeys(joints))
        if not names:
            raise ValueError("LeaderCurrentLoop needs at least one joint.")
        unknown = [name for name in names if name not in bus.motors]
        if unknown:
            raise ValueError(f"Unknown joint(s) for the leader loop: {unknown}")
        law_joints = getattr(law, "joints", None)
        if law_joints is not None and tuple(law_joints) != names:
            raise ValueError(f"law.joints {tuple(law_joints)} != loop joints {names}")
        for name in names:
            if name not in units_ma or not float(units_ma[name]) > 0.0:
                raise ValueError(f"units_ma must give a positive mA/count for '{name}'")
            if current_sign.get(name) not in (1, -1):
                raise ValueError(f"current_sign['{name}'] must be +1 or -1")
        if gripper_hold is not None:
            bad = [name for name in gripper_hold if name not in bus.motors or name in names]
            if bad:
                raise ValueError(f"gripper_hold names must be non-loop motors of the bus: {bad}")

        self._bus = bus
        self._joints = names
        self._params = params
        self._law = law
        self._follower = follower
        self._clock = clock
        self._sleep = sleep
        self._main_alive = main_alive
        self._gripper_hold: Dict[str, int] = (
            {} if gripper_hold is None else {k: int(v) for k, v in gripper_hold.items()}
        )
        self._names: Tuple[str, ...] = tuple(bus.motors)
        self._j_index = np.array([self._names.index(name) for name in names], dtype=np.intp)
        self._units = _current_units(bus, units_ma)
        self._sign: Dict[str, int] = {name: int(current_sign[name]) for name in names}
        self._period_s = 1.0 / params.control_hz
        self._period_ns = int(round(1e9 / params.control_hz))
        ranges = getattr(law, "joint_range_counts", None)
        self._ranges: Dict[str, Tuple[int, int]] = (
            {}
            if ranges is None
            else {n: (int(ranges[n][0]), int(ranges[n][1])) for n in names if n in ranges}
        )

        self._lock = threading.RLock()
        self._cond = threading.Condition(self._lock)
        self._hold_lock = threading.Lock()
        self._law_lock = threading.Lock()
        self._stop_event = threading.Event()
        self._thread: threading.Thread | None = None
        self._hold_thread: threading.Thread | None = None

        self._state = LoopState.IDLE
        self._fault: LoopFault | None = None
        self._faults: List[LoopFault] = []
        self._preflight: Dict[str, Any] = {}
        self._temperature_limit: Dict[str, int] = {}
        self._current_limit_raw: Dict[str, int] = {}
        self._i_max_ma: Dict[str, float] = {}
        self._hold_report: HoldReport | None = None
        self._latest: RakudaArmState | None = None
        self._latest_follower: RakudaArmState | None = None
        self._pair: PairSnapshot | None = None
        self._seq = 0
        self._reset_session()

    def _reset_session(self) -> None:
        """Clears everything one ``configure()`` -> hold session accumulates."""
        self._fault = None
        self._faults = []
        self._hold_done = False
        self._hold_verified = False
        self._hold_report = None
        self._engaged = False
        self._follower_lost = False
        self._pair = None
        self._pair_seq = 0
        self._pair_consumed = 0
        self._pairs = 0
        self._last_valid_ns: int | None = None
        self._start_ns: int | None = None
        self._next_deadline_ns: int | None = None
        self._prev_cycle_ns: int | None = None
        self._cycles = 0
        self._ok_cycles = 0
        self._read_fail_count = 0
        self._overruns = 0
        self._consecutive_overruns = 0
        self._skipped = 0
        self._period_ms = _Percentiles()
        self._read_ms = _Percentiles()
        self._write_ms = _Percentiles()
        self._follower_step_ms = _Percentiles()
        self._follower_age_ms = _Percentiles()
        self._write_fail_since_ns: int | None = None
        self._runaway_since_ns: int | None = None
        self._saturation_since_ns: int | None = None
        self._last_diag_ns: int | None = None
        self._last_diag_attempt_ns: int | None = None
        self._prev_gate: float | None = None
        self._last_gate = 0.0
        self._gate_drops = 0
        self._gate_low_cycles = 0
        self._log_last_ns: Dict[str, int] = {}
        self._last_output: LawOutput | None = None
        self._goal_current_rewritten_at_torque_on = False
        self._stop_event.clear()

    # -- properties -----------------------------------------------------------

    @property
    def joints(self) -> Tuple[str, ...]:
        return self._joints

    @property
    def port_name(self) -> str:
        """The leader bus device node (``"<port>"`` if the bus does not expose it)."""
        return bus_port_name(self._bus)

    def recovery_hint(self) -> str:
        """The bring-up commands that inspect, then release, the leader bus by hand."""
        port = self.port_name
        return (
            f"{ports_command('show', port, 'leader')}, "
            f"then {ports_command('release', port, 'leader')}"
        )

    @property
    def state(self) -> LoopState:
        return self._state

    @property
    def fault(self) -> LoopFault | None:
        """The fault that stopped the loop (the first latched one), or ``None``."""
        return self._fault

    @property
    def faults(self) -> Tuple[LoopFault, ...]:
        """Every fault latched since ``configure()``, in order (``hold_failed`` included)."""
        return tuple(self._faults)

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def follower_lost(self) -> bool:
        return self._follower_lost

    @property
    def pair_seq(self) -> int:
        return self._pair_seq

    @property
    def hold_verified(self) -> bool:
        return self._hold_verified

    @property
    def hold_report(self) -> HoldReport | None:
        return self._hold_report

    @property
    def thread_alive(self) -> bool:
        return self._thread is not None and self._thread.is_alive()

    def _hold_in_progress(self) -> bool:
        """True while a hold runs on the bus (control thread or ``stop()``'s helper)."""
        helper = self._hold_thread
        if helper is not None and helper.is_alive():
            return True
        if not self._hold_lock.acquire(blocking=False):
            return True
        self._hold_lock.release()
        return False

    @property
    def running(self) -> bool:
        return self._state is LoopState.RUNNING

    @property
    def next_deadline_ns(self) -> int | None:
        """Start of the next cycle on ``clock``; ``None`` before the first cycle."""
        return self._next_deadline_ns

    @property
    def current_sign(self) -> Dict[str, int]:
        return dict(self._sign)

    # -- preflight ------------------------------------------------------------

    def preflight(self) -> Dict[str, Any]:
        """Read-only checks of the leader before ``configure()``.

        In order: ``verify_models()``; a register table of all 17 motors
        (logged at INFO); a joint of ``J`` left in current mode with torque on
        is held first (``stale_current_mode_recovered``); the per-register
        rejections and warnings; a state read with the range check when the
        law knows ``joint_range_counts``; and the bus-budget measurement
        (``_PREFLIGHT_READS`` state reads at the control period).  The
        result is also kept for ``control_report()["preflight"]``.

        Returns:
            ``registers`` (per motor), ``drive_mode`` (per joint), ``warnings``,
            ``notes``, ``read_ms`` (p50/p95/p99/max/count/timeouts),
            ``range_checked`` and ``stale_current_mode_recovered``.

        Raises:
            RuntimeError: If the loop is CONFIGURED or RUNNING.
            ConnectionError: Listing every rejected condition (registers unchanged
                except for the stale-mode hold).
            DynamixelCommError: When a read itself failed.
        """
        if self._state in (LoopState.CONFIGURED, LoopState.RUNNING):
            raise RuntimeError(f"preflight() is not allowed while the loop is {self._state.name}")
        bus, names, joints, params = self._bus, self._names, self._joints, self._params
        problems: List[str] = []
        warnings: List[str] = []
        notes: List[str] = []

        bus.verify_models(names)
        table: Dict[str, Dict[str, Any]] = {name: {} for name in names}
        for item in _PREFLIGHT_ITEMS:
            values = bus.sync_read(item, list(names))
            for name in names:
                table[name][item.name.lower()] = int(values[name])
        for name, reading in bus.read_diagnostics(names).items():
            table[name]["hardware_error_status"] = reading.hardware_error_status
            table[name]["voltage_v"] = reading.voltage_v
            table[name]["temperature_c"] = reading.temperature_c
        self._temperature_limit = {name: int(table[name]["temperature_limit"]) for name in names}
        for name in names:
            logger.info("preflight: %s %s", name, table[name])

        stale = [
            name
            for name in joints
            if table[name]["operating_mode"] == OperatingMode.CURRENT
            and table[name]["torque_enable"] == 1
        ]
        if stale:
            logger.info(
                "preflight: %s left in current mode with torque on (GOAL_CURRENT %s); holding "
                "before configure (stale_current_mode_recovered)",
                stale,
                {name: table[name]["goal_current"] for name in stale},
            )
            report = hold_joints(
                bus,
                stale,
                HoldParams.from_bilateral_params(params),
                None,
                clock=self._clock,
                logger=logger,
            )
            if not report.verified:
                problems.append(f"stale current mode on {stale} could not be held: {report.notes}")
            for name in stale:
                table[name]["operating_mode"] = int(OperatingMode.POSITION)
                table[name]["torque_enable"] = 1 if name not in report.none else 0

        health = params.leader_health
        for name in names:
            row = table[name]
            if row["hardware_error_status"] != 0:
                problems.append(
                    f"{name}: HARDWARE_ERROR_STATUS=0x{row['hardware_error_status']:02x}"
                )
            if not health.min_voltage_v <= row["voltage_v"] <= health.max_voltage_v:
                problems.append(
                    f"{name}: {row['voltage_v']:.1f} V outside "
                    f"[{health.min_voltage_v}, {health.max_voltage_v}] V"
                )
            if row["temperature_c"] >= health.warn_temperature_c:
                problems.append(
                    f"{name}: {row['temperature_c']} C >= warn {health.warn_temperature_c:.0f} C"
                )
            if row["return_delay_time"] != 0:
                warnings.append(
                    f"{name}: RETURN_DELAY_TIME={row['return_delay_time']} (0 expected)"
                )
        drive_mode: Dict[str, int] = {}
        for name in joints:
            row = table[name]
            if health.max_temperature_c >= row["temperature_limit"]:
                problems.append(
                    f"{name}: TEMPERATURE_LIMIT {row['temperature_limit']} C <= "
                    f"leader_health.max_temperature_c {health.max_temperature_c:.0f} C "
                    "(the motor would shut down before the loop holds)"
                )
            limit_ma = row["current_limit"] * self._units[name]
            if limit_ma < _MIN_CURRENT_LIMIT_MA:
                problems.append(
                    f"{name}: CURRENT_LIMIT {row['current_limit']} = {limit_ma:.0f} mA "
                    f"< {_MIN_CURRENT_LIMIT_MA:.0f} mA"
                )
            if row["torque_enable"] == 1 and row["operating_mode"] not in (
                OperatingMode.CURRENT,
                OperatingMode.POSITION,
            ):
                problems.append(
                    f"{name}: torque on in OPERATING_MODE {row['operating_mode']} (not 0 or 3)"
                )
            mode = row["drive_mode"]
            drive_mode[name] = mode
            if mode & _DRIVE_MODE_TIME_PROFILE:
                problems.append(f"{name}: DRIVE_MODE bit2 (time-based profile) is set")
            if mode & _DRIVE_MODE_RESERVED:
                problems.append(f"{name}: DRIVE_MODE has reserved bits set (0x{mode:02x})")
            if mode & _DRIVE_MODE_TORQUE_ON_BY_GOAL:
                warnings.append(f"{name}: DRIVE_MODE bit3 (torque on by goal update) is set")
            if mode & _DRIVE_MODE_REVERSE:
                notes.append(f"{name}: DRIVE_MODE bit0 (reverse) is set; covered by current_sign")

        readings, _, _ = bus.read_state_block(
            names, timeout_s=params.read_timeout_s, attempts=_SETUP_READ_ATTEMPTS
        )
        if self._ranges:
            for name, (low, high) in self._ranges.items():
                position = readings[name].position
                if (
                    not low - params.hard_margin_counts
                    <= position
                    <= (high + params.hard_margin_counts)
                ):
                    problems.append(
                        f"{name}: position {position} outside range [{low}, {high}] "
                        f"± {params.hard_margin_counts}"
                    )
        else:
            notes.append("joint ranges unknown to the law; the range check was skipped")

        budget = self._measure_read_budget()
        period_ms = self._period_s * 1e3
        if budget["timeouts"]:
            problems.append(
                f"{budget['timeouts']}/{_PREFLIGHT_READS} state reads timed out at "
                f"{params.read_timeout_s * 1e3:.1f} ms"
            )
        if budget["p95"] > params.read_timeout_s * 1e3 / 1.5:
            problems.append(
                f"state read p95 {budget['p95']:.2f} ms > read_timeout_s/1.5 = "
                f"{params.read_timeout_s * 1e3 / 1.5:.2f} ms; re-run "
                f"{ports_command('bench', self.port_name, 'leader')}"
            )
        if budget["p95"] + 2.0 > period_ms:
            warnings.append(
                f"state read p95 {budget['p95']:.2f} ms + 2 ms exceeds the period "
                f"{period_ms:.1f} ms"
            )

        for text in warnings:
            logger.warning("preflight: %s", text)
        result: Dict[str, Any] = {
            "t_ns": self._clock(),
            "registers": table,
            "drive_mode": drive_mode,
            "warnings": warnings,
            "notes": notes,
            "problems": problems,
            "read_ms": budget,
            "range_checked": bool(self._ranges),
            "stale_current_mode_recovered": stale,
        }
        self._preflight = result
        if problems:
            raise ConnectionError("preflight rejected the leader: " + "; ".join(problems))
        return result

    def _measure_read_budget(self) -> Dict[str, Any]:
        """``_PREFLIGHT_READS`` deadline-bounded state reads at the control period."""
        samples = _Percentiles()
        timeouts = 0
        next_ns = self._clock()
        for _ in range(_PREFLIGHT_READS):
            t0 = self._clock()
            try:
                self._bus.read_state_block(
                    self._names, timeout_s=self._params.read_timeout_s, attempts=1
                )
            except DynamixelTimeoutError:
                timeouts += 1
            samples.record((self._clock() - t0) / 1e6)
            next_ns += self._period_ns
            remaining = (next_ns - self._clock()) / 1e9
            if remaining > 0.0:
                self._sleep(remaining)
        return {**samples.summary(), "timeouts": timeouts}

    # -- configure ------------------------------------------------------------

    def configure(self) -> None:
        """Brings ``J`` into current mode, torque on at the law's gravity term.

        Steps: torque off -> ``OPERATING_MODE=0`` (read back) -> log
        ``GOAL_CURRENT`` -> ``CURRENT_LIMIT`` where it differs -> ``BUS_WATCHDOG=0``
        (read back) -> state read, ``law.reset``, initial ``GOAL_CURRENT`` (the
        law's ``gravity_term()``; the law's first
        ``compute()`` happens in the loop) ->
        read back (must match) -> torque on, ``TORQUE_ENABLE`` read back ->
        ``GOAL_CURRENT`` read again, rewritten once on a difference -> gripper
        hold -> ``CONFIGURED``.  Any failure before torque-on rolls back to
        position mode with torque off (``IDLE``); a ``GOAL_CURRENT`` that still
        differs after the rewrite (torque already on) is held instead
        (``FAULT``, ``configure_failed``).

        Raises:
            RuntimeError: If the loop is CONFIGURED or RUNNING, or a hold is in progress.
            ConfigureError: On any failure; the rollback (or hold) has run.
        """
        if self._state in (LoopState.CONFIGURED, LoopState.RUNNING):
            raise RuntimeError(f"configure() is not allowed while the loop is {self._state.name}")
        if self._hold_in_progress():
            raise RuntimeError("configure() while a hold is in progress on the bus")
        bus, joints, params = self._bus, list(self._joints), self._params
        self._reset_session()
        positions: Dict[str, int] | None = None
        completed = False
        held_instead = False
        t0 = self._clock()
        logger.info("configure: entering current mode on %s", joints)
        try:
            bus.torque_disabled(joints)
            bus.write_with_readback(
                XControlTable.OPERATING_MODE, {name: OperatingMode.CURRENT for name in joints}
            )
            logger.info(
                "configure: GOAL_CURRENT after the mode change: %s",
                _read_items(bus, XControlTable.GOAL_CURRENT, joints),
            )
            limits = {
                n: int(v) for n, v in bus.sync_read(XControlTable.CURRENT_LIMIT, joints).items()
            }
            wanted = {n: int(round(params.current_limit_ma / self._units[n])) for n in joints}
            differing: Dict[str, int | float] = {
                n: wanted[n] for n in joints if limits.get(n) != wanted[n]
            }
            if differing:
                logger.info(
                    "configure: CURRENT_LIMIT %s -> %s",
                    {n: limits.get(n) for n in differing},
                    differing,
                )
                bus.write_with_readback(XControlTable.CURRENT_LIMIT, differing)
            self._current_limit_raw = dict(wanted)
            self._i_max_ma = {
                n: min(params.current_max_ma, wanted[n] * self._units[n]) for n in joints
            }
            bus.write_with_readback(XControlTable.BUS_WATCHDOG, {name: 0 for name in joints})

            state = self._read_leader(attempts=_SETUP_READ_ATTEMPTS)
            positions = {
                name: int(state.position[index]) % _COUNTS_PER_TURN
                for name, index in zip(self._joints, self._j_index)
            }
            with self._law_lock:
                self._law.reset(state)
                initial = _initial_current_ma(self._law, state)
            raw, _ = self._to_raw(initial)
            bus.write_goal_current_raw(raw)

            readback = bus.sync_read(XControlTable.GOAL_CURRENT, joints)
            mismatch = {n: (raw[n], readback.get(n)) for n in joints if readback.get(n) != raw[n]}
            if mismatch:
                raise ConfigureError(
                    f"GOAL_CURRENT read-back differs before torque-on (wanted, read): {mismatch}"
                )
            bus.torque_enabled(joints)
            window_ms = (self._clock() - t0) / 1e6
            torque = bus.sync_read(XControlTable.TORQUE_ENABLE, joints)
            off = [n for n in joints if torque.get(n) != 1]
            if off:
                raise ConfigureError(f"TORQUE_ENABLE read-back: {off} did not switch on")
            again = bus.sync_read(XControlTable.GOAL_CURRENT, joints)
            changed = {n: again.get(n) for n in joints if again.get(n) != raw[n]}
            if changed:
                self._goal_current_rewritten_at_torque_on = True
                logger.warning(
                    "configure: GOAL_CURRENT changed at torque-on (%s); rewriting once",
                    changed,
                )
                bus.write_goal_current_raw({n: raw[n] for n in changed})
                again = bus.sync_read(XControlTable.GOAL_CURRENT, joints)
                still = {n: (raw[n], again.get(n)) for n in joints if again.get(n) != raw[n]}
                if still:
                    held_instead = True
                    self._latch(
                        "configure_failed",
                        f"GOAL_CURRENT still differs after the rewrite (wanted, read): {still}",
                        self._clock(),
                    )
                    self._finish()
                    raise ConfigureError(f"GOAL_CURRENT could not be set; joints held: {still}")
            if self._gripper_hold:
                bus.sync_write(XControlTable.GOAL_POSITION, dict(self._gripper_hold))
            self._last_diag_ns = self._last_diag_attempt_ns = self._clock()
            self._set_state(LoopState.CONFIGURED)
            completed = True
            logger.info(
                "configure: %d joint(s) in current mode, torque on at %s (torque-off window "
                "%.1f ms)",
                len(joints),
                {n: f"{float(i):.0f} mA" for n, i in zip(self._joints, initial)},
                window_ms,
            )
        except ConfigureError:
            raise
        except Exception as exc:
            raise ConfigureError(f"configure failed: {_first_line(exc)}") from exc
        finally:
            if not completed and not held_instead:
                self._rollback(positions)

    def _rollback(self, positions: Mapping[str, int] | None) -> None:
        """Undoes a failed ``configure()``: torque off, position mode, goal = last read (H1)."""
        joints = list(self._joints)
        logger.critical("configure: rolling back %s to position mode; torque stays OFF", joints)
        try:
            self._bus.torque_disabled(joints)
        except Exception as exc:
            logger.critical("configure rollback: TORQUE_ENABLE=0 failed (%s)", _first_line(exc))
        if not _confirm_position_mode(self._bus, joints, HoldParams().readback_attempts, logger):
            logger.critical("configure rollback: OPERATING_MODE=3 not confirmed on every joint")
        if positions:
            try:
                self._bus.sync_write(XControlTable.GOAL_POSITION, dict(positions))
            except Exception as exc:
                logger.critical(
                    "configure rollback: GOAL_POSITION write failed (%s)", _first_line(exc)
                )
        self._set_state(LoopState.IDLE)

    # -- running --------------------------------------------------------------

    def start(self) -> None:
        """Starts the daemon control thread.

        Raises:
            RuntimeError: If the loop is already running or is not CONFIGURED.
        """
        if self._state is LoopState.RUNNING or self.thread_alive:
            raise RuntimeError("LeaderCurrentLoop is already running")
        if self._state is not LoopState.CONFIGURED:
            raise RuntimeError(f"start() needs a CONFIGURED loop, it is {self._state.name}")
        self._stop_event.clear()
        self._start_ns = self._clock()
        self._set_state(LoopState.RUNNING)
        self._thread = threading.Thread(target=self._run, name="rakuda-leader-loop", daemon=True)
        self._thread.start()

    def run_once(self, now_ns: int) -> None:
        """Runs one control cycle starting at ``now_ns`` on ``clock``.

        The first call after ``configure()`` marks the loop RUNNING (as
        ``start()`` does before launching the thread), so tests can drive the
        loop without a thread.  A fault detected in the cycle is latched, the
        joints are held (exactly once) and the loop ends up HELD-like FAULT;
        the method itself does not raise for it.  A ``SignCheckOvershoot``
        from the law is the fault ``sign_check_overshoot``; any
        other exception is ``cycle_failed``.  The scheduling bookkeeping
        (``next_deadline_ns``, overruns, skipped cycles) is updated at the end.

        Raises:
            LoopStopped: If the loop is neither CONFIGURED nor RUNNING, or
                ``stop()`` has been called (its hold may still be running).
        """
        if self._stop_event.is_set():
            raise LoopStopped(self._state, self._fault)
        if self._state is LoopState.CONFIGURED:
            self._start_ns = now_ns
            self._set_state(LoopState.RUNNING)
        elif self._state is not LoopState.RUNNING:
            raise LoopStopped(self._state, self._fault)
        if self._next_deadline_ns is None:
            self._next_deadline_ns = now_ns
        self._cycles += 1
        if self._prev_cycle_ns is not None:
            self._period_ms.record((now_ns - self._prev_cycle_ns) / 1e6)
        self._prev_cycle_ns = now_ns
        try:
            self._cycle(now_ns)
        except _Trip as trip:
            self._trip(trip.reason, trip.detail)
        except SignCheckOvershoot as exc:
            self._trip(
                exc.reason,
                f"{exc.joint} moved {exc.displacement_counts:+d} counts "
                f"(abort at ±{exc.abort_counts})",
            )
        except Exception as exc:
            logger.exception("cycle failed")
            self._trip("cycle_failed", f"{type(exc).__name__}: {_first_line(exc)}")
        finally:
            end_ns = self._clock()
            duration_ns = end_ns - now_ns
            if duration_ns > self._period_ns:
                self._overruns += 1
            if duration_ns > self._params.overrun_warn_factor * self._period_ns:
                self._consecutive_overruns += 1
                if self._consecutive_overruns == _OVERRUN_WARN_STREAK:
                    logger.warning(
                        "%d consecutive cycles longer than %.0f x %.1f ms (last %.1f ms)",
                        _OVERRUN_WARN_STREAK,
                        self._params.overrun_warn_factor,
                        self._period_s * 1e3,
                        duration_ns / 1e6,
                    )
            else:
                self._consecutive_overruns = 0
            missed = max(0, (end_ns - self._next_deadline_ns) // self._period_ns)
            self._skipped += missed
            self._next_deadline_ns += (missed + 1) * self._period_ns

    def _cycle(self, now_ns: int) -> None:
        """One control cycle (main-thread check, then steps 0-5); raises :class:`_Trip`."""
        if not self._main_alive():
            raise _Trip("main_thread_exited", "the main thread is no longer alive")
        bus, params = self._bus, self._params

        # 0. Read (one attempt, hard deadline).  A failed read writes nothing:
        # the motors keep the previous GOAL_CURRENT (gravity compensation).
        t_read = self._clock()
        try:
            readings, start_ns, end_ns = bus.read_state_block(
                self._names, timeout_s=params.read_timeout_s, attempts=1
            )
        except DynamixelCommError as exc:
            self._read_fail_count += 1
            now = self._clock()
            self._log_limited(
                logging.WARNING,
                "read",
                now,
                "leader read failed (%d so far): %s",
                self._read_fail_count,
                _first_line(exc),
            )
            stale_ns = now - (self._last_valid_ns if self._last_valid_ns is not None else 0)
            if self._last_valid_ns is None or stale_ns > params.snapshot_stale_s * 1e9:
                raise _Trip(
                    "snapshot_stale",
                    f"no valid leader reading for {stale_ns / 1e9:.3f} s "
                    f"(> {params.snapshot_stale_s} s)",
                ) from exc
            return
        self._read_ms.record((self._clock() - t_read) / 1e6)
        state = self._leader_state(readings, start_ns, end_ns)
        self._publish_leader(state)
        self._ok_cycles += 1
        q = state.position[self._j_index]
        v = state.velocity[self._j_index]
        for k, name in enumerate(self._joints):
            span = self._ranges.get(name)
            if span is not None and not (
                span[0] - params.hard_margin_counts <= q[k] <= span[1] + params.hard_margin_counts
            ):
                raise _Trip(
                    "joint_out_of_range",
                    f"{name} at {int(q[k])} counts, range {span} ± {params.hard_margin_counts}",
                )

        # 1-3. The law (gravity, barrier, feedback; the follower is hidden once lost).
        follower = None if self._follower_lost else self._latest_follower
        age_s = None if follower is None else max(0.0, (end_ns - follower.t_end_ns) / 1e9)
        with self._law_lock:
            output = self._law.compute(state, follower, age_s, self._period_s)
        self._record_gate(float(output.gate), age_s, now_ns)
        current = np.asarray(output.current_ma, dtype=np.float64)
        if current.shape != (len(self._joints),) or not np.all(np.isfinite(current)):
            raise ValueError(
                f"law output must be finite with shape ({len(self._joints)},), got {current!r}"
            )

        # 4. Shape and write (J only).
        raw, saturated = self._to_raw(current)
        t_write = self._clock()
        try:
            bus.write_goal_current_raw(raw)
        except DynamixelCommError as exc:
            now = self._clock()
            if self._write_fail_since_ns is None:
                self._write_fail_since_ns = now
            self._log_limited(
                logging.WARNING, "write", now, "GOAL_CURRENT write failed: %s", _first_line(exc)
            )
            if now - self._write_fail_since_ns >= params.write_fault_s * 1e9:
                raise _Trip(
                    "write_failed",
                    "GOAL_CURRENT writes failing for "
                    f"{(now - self._write_fail_since_ns) / 1e9:.3f} s",
                ) from exc
        else:
            self._write_fail_since_ns = None
            self._write_ms.record((self._clock() - t_write) / 1e6)
        self._last_output = output

        # 4b. Follower I/O every follower_divider-th cycle.
        if self._follower is not None and self._cycles % params.follower_divider == 0:
            self._follower_io(state)

        # 5. Real-time fault timers and the slow diagnostics.
        now = self._clock()
        self._saturation_since_ns = self._since(saturated, self._saturation_since_ns, now)
        if self._saturation_since_ns is not None and (
            now - self._saturation_since_ns >= params.saturation_fault_s * 1e9
        ):
            raise _Trip(
                "current_saturated",
                f"|GOAL_CURRENT| at the clamp for {params.saturation_fault_s} s",
            )
        runaway = bool(np.any(np.abs(v) > params.runaway_velocity_counts))
        self._runaway_since_ns = self._since(runaway, self._runaway_since_ns, now)
        if self._runaway_since_ns is not None and (
            now - self._runaway_since_ns >= params.runaway_s * 1e9
        ):
            fast = {
                n: int(v[k])
                for k, n in enumerate(self._joints)
                if abs(v[k]) > params.runaway_velocity_counts
            }
            raise _Trip(
                "runaway",
                f"|velocity| > {params.runaway_velocity_counts} for {params.runaway_s} s: {fast}",
            )
        if self._last_diag_attempt_ns is None:
            self._last_diag_ns = self._last_diag_attempt_ns = now
        elif now - self._last_diag_attempt_ns >= params.diagnostics_period_s * 1e9:
            self._diagnostics(now)

    @staticmethod
    def _since(active: bool, since_ns: int | None, now_ns: int) -> int | None:
        """Start of the current run of ``active`` samples, or ``None``."""
        if not active:
            return None
        return now_ns if since_ns is None else since_ns

    def _leader_state(
        self, readings: Mapping[str, MotorStateReading], start_ns: int, end_ns: int
    ) -> RakudaArmState:
        missing = [name for name in self._names if name not in readings]
        if missing:
            raise _Trip("invalid_reading", f"no data for {missing}")
        self._seq += 1
        state = _make_state(self._names, readings, start_ns, end_ns, self._seq, self._units)
        if not np.all(np.isfinite(state.current_ma)):
            raise _Trip("invalid_reading", "non-finite current in the leader reading")
        return state

    def _read_leader(self, *, attempts: int) -> RakudaArmState:
        """One-off leader read outside the cycle (``configure``); publishes the snapshot."""
        readings, start_ns, end_ns = self._bus.read_state_block(
            self._names, timeout_s=self._params.read_timeout_s, attempts=attempts
        )
        state = self._leader_state(readings, start_ns, end_ns)
        self._publish_leader(state)
        return state

    def _publish_leader(self, state: RakudaArmState) -> None:
        with self._cond:
            self._latest = state
            self._last_valid_ns = state.t_end_ns
            self._cond.notify_all()

    def _to_raw(self, current_ma: NDArray[np.float64]) -> Tuple[Dict[str, int], bool]:
        """Joint-convention mA -> ``{joint: raw int16}`` in motor convention.

        Clamps to ``±min(current_max_ma, CURRENT_LIMIT * unit)`` first; the
        second value says whether any joint sits at that clamp (saturation).
        """
        raw: Dict[str, int] = {}
        saturated = False
        for k, name in enumerate(self._joints):
            i_max = self._i_max_ma[name]
            i = float(np.clip(current_ma[k], -i_max, i_max))
            if abs(i) >= i_max:
                saturated = True
            value = int(round(self._sign[name] * i / self._units[name]))
            limit = self._current_limit_raw[name]
            raw[name] = max(-limit, min(limit, value))
        return raw, saturated

    def _record_gate(self, gate: float, age_s: float | None, now_ns: int) -> None:
        """Records the gate: gate crossings and the follower age histogram."""
        self._last_gate = gate
        if age_s is not None:
            self._follower_age_ms.record(age_s * 1e3)
        if gate < 0.5:
            self._gate_low_cycles += 1
        if self._prev_gate is not None:
            if self._prev_gate >= 0.5 > gate:
                self._gate_drops += 1
                self._log_limited(
                    logging.INFO,
                    "gate_drop",
                    now_ns,
                    "feedback gate dropped below 0.5 (%.2f)",
                    gate,
                )
            elif self._prev_gate < 0.5 <= gate:
                self._log_limited(
                    logging.INFO, "gate_rise", now_ns, "feedback gate back above 0.5 (%.2f)", gate
                )
        self._prev_gate = gate

    def _follower_io(self, state: RakudaArmState) -> None:
        """Step 4b: follower write + read, pair publication, ``follower_lost`` bookkeeping."""
        follower = self._follower
        assert follower is not None
        t0 = self._clock()
        try:
            follower_state = follower.step(state)
        except DynamixelCommError as exc:
            now = self._clock()
            self._log_limited(
                logging.WARNING,
                "follower",
                now,
                "follower I/O failed (%d consecutive): %s",
                follower.consecutive_failures,
                _first_line(exc),
            )
            if (
                not self._follower_lost
                and follower.consecutive_failures >= self._params.follower_lost_cycles
            ):
                self._follower_lost = True
                logger.error(
                    "follower lost after %d consecutive failures; the leader keeps gravity "
                    "compensation, feedback is gated off (stop_bilateral() then "
                    "start_bilateral() to recover)",
                    follower.consecutive_failures,
                )
                with self._cond:
                    self._cond.notify_all()
            return
        self._follower_step_ms.record((self._clock() - t0) / 1e6)
        with self._cond:
            self._latest_follower = follower_state
            self._pair_seq += 1
            self._pairs += 1
            self._pair = PairSnapshot(state, follower_state, self._pair_seq)
            self._cond.notify_all()

    def _diagnostics(self, now_ns: int) -> None:
        """The 1 Hz health read of ``J`` (faults hardware_error / overheat / voltage).

        A failed read is retried at the same 1 Hz schedule (not every cycle);
        ``diagnostics_stale`` counts from the last successful read.
        """
        params = self._params
        self._last_diag_attempt_ns = now_ns
        try:
            readings = self._bus.read_diagnostics(
                list(self._joints), timeout_s=params.read_timeout_s
            )
        except DynamixelCommError as exc:
            self._log_limited(
                logging.WARNING, "diag", now_ns, "diagnostics read failed: %s", _first_line(exc)
            )
            assert self._last_diag_ns is not None
            if now_ns - self._last_diag_ns >= _DIAGNOSTICS_STALE_S * 1e9:
                raise _Trip(
                    "diagnostics_stale", f"no diagnostics for {_DIAGNOSTICS_STALE_S:.0f} s"
                ) from exc
            return
        self._last_diag_ns = now_ns
        health = params.leader_health
        for name in self._joints:
            reading = readings[name]
            if reading.hardware_error_status != 0:
                raise _Trip(
                    "hardware_error",
                    f"{name}: HARDWARE_ERROR_STATUS=0x{reading.hardware_error_status:02x}",
                )
            # The configured hold threshold (66 C) sits just under the motor's
            # TEMPERATURE_LIMIT (70 C); preflight() rejects a configuration where it
            # does not, so no ``TEMPERATURE_LIMIT - 10`` clamp (that would hold at 60 C
            # on the warm idle leader).
            overheat_c = health.max_temperature_c
            if reading.temperature_c >= overheat_c:
                raise _Trip(
                    "overheat", f"{name} at {reading.temperature_c} C (>= {overheat_c:.0f} C)"
                )
            if reading.temperature_c >= health.warn_temperature_c:
                logger.warning(
                    "%s at %d C (warn %.0f C, hold at %.0f C)",
                    name,
                    reading.temperature_c,
                    health.warn_temperature_c,
                    overheat_c,
                )
            if not health.min_voltage_v <= reading.voltage_v <= health.max_voltage_v:
                raise _Trip(
                    "voltage",
                    f"{name} at {reading.voltage_v:.1f} V, outside "
                    f"[{health.min_voltage_v}, {health.max_voltage_v}] V",
                )

    def _run(self) -> None:
        """Body of the control thread."""
        try:
            while not self._stop_event.is_set():
                try:
                    self.run_once(self._clock())
                except LoopStopped:
                    break
                if self._stop_event.is_set():
                    break
                assert self._next_deadline_ns is not None
                remaining = (self._next_deadline_ns - self._clock()) / 1e9
                if remaining > 0.0:
                    self._sleep(remaining)
        finally:
            self._finish()

    # -- faults and the hold ----------------------------------------------------

    def _latch(self, reason: str, detail: str, t_ns: int) -> None:
        fault = LoopFault(reason, detail, t_ns, None, self._state)
        self._faults.append(fault)
        if self._fault is None:
            self._fault = fault
        level = logging.CRITICAL if reason in ("hold_failed", "stop_timeout") else logging.ERROR
        logger.log(level, "fault %s: %s", reason, detail)

    def _trip(self, reason: str, detail: str) -> None:
        """A fault inside a cycle: latch, stop, hold."""
        self._latch(reason, detail, self._clock())
        self._stop_event.set()
        self._finish()

    def _finish(self) -> None:
        """Holds the joints exactly once when the loop ends.

        Called from the control thread's ``finally``, from ``run_once`` on a
        fault, and from ``stop()``'s helper thread when the loop ended
        without holding (or the hold raised, which ``stop()`` retries once per call).
        """
        with self._hold_lock:
            if self._hold_done:
                return
            if (
                self._fault is None
                and not self._stop_event.is_set()
                and self._state is LoopState.RUNNING
            ):
                self._latch("thread_exit", "the loop thread exited while RUNNING", self._clock())
            self._stop_event.set()
            self._hold_done = self._hold()

    def _hold(self) -> bool:
        """Runs :func:`hold_joints` on ``J``; False when it raised (``hold_failed``)."""
        try:
            report = hold_joints(
                self._bus,
                self._joints,
                HoldParams.from_bilateral_params(self._params),
                self.latest(),
                clock=self._clock,
                logger=logger,
            )
        except BaseException as exc:
            self._latch("hold_failed", _first_line(exc), self._clock())
            self._hold_verified = False
            self._set_state(LoopState.FAULT)
            return False
        self._hold_report = report
        self._hold_verified = report.verified
        state = LoopState.FAULT if self._fault is not None else LoopState.HELD
        self._faults = [
            replace(f, hold_window_ms=report.window_ms, state_after=state) for f in self._faults
        ]
        self._fault = self._faults[0] if self._faults else None
        self._set_state(state)
        return True

    def stop(self, timeout_s: float | None = None) -> bool:
        """Stops the loop and holds the joints in place.

        Sets the stop event and joins the control thread; if the thread did
        not hold (it died, never ran, or its hold raised) a non-daemon helper
        thread runs the hold within ``HOLD_BUDGET_S``.  Idempotent: a
        second call touches no bus.

        Args:
            timeout_s: Join budget; default ``5/control_hz + HOLD_BUDGET_S``.

        Returns:
            True when every joint is verified held (or there was nothing to
            hold: IDLE/RELEASED); False after ``stop_timeout`` or an unverified hold.
        """
        if timeout_s is None:
            timeout_s = 5.0 / self._params.control_hz + HOLD_BUDGET_S
        self._stop_event.set()
        with self._cond:
            self._cond.notify_all()
        thread = self._thread
        if thread is not None and thread.is_alive():
            if thread is threading.current_thread():
                raise RuntimeError("stop() must not be called from the control thread")
            thread.join(timeout_s)
            if thread.is_alive():
                self._latch(
                    "stop_timeout",
                    f"the control thread did not exit within {timeout_s:.2f} s; do not touch the "
                    f"port from another thread ({self.recovery_hint()})",
                    self._clock(),
                )
                return False
        if self._state in (LoopState.IDLE, LoopState.RELEASED):
            return True
        if not self._hold_done:
            helper = threading.Thread(target=self._finish, name="rakuda-leader-hold", daemon=False)
            self._hold_thread = helper
            helper.start()
            helper.join(HOLD_BUDGET_S)
            if helper.is_alive():
                self._latch(
                    "stop_timeout",
                    f"the hold did not finish within {HOLD_BUDGET_S:.1f} s",
                    self._clock(),
                )
                return False
        return self._hold_verified

    def release(self) -> None:
        """Torque off on ``J``; only from HELD, FAULT or CONFIGURED.

        A CONFIGURED loop (current mode) is also put back into position mode
        so the arm is left in the connect-time "ok" state.

        Raises:
            RuntimeError: In any other state, while the thread is alive, or
                while a hold is still running on the bus (``stop_timeout``).
        """
        if self._state not in (LoopState.HELD, LoopState.FAULT, LoopState.CONFIGURED):
            raise RuntimeError(
                f"release() is only allowed in HELD/FAULT/CONFIGURED, not {self._state.name}"
            )
        if self.thread_alive:
            raise RuntimeError("release() while the control thread is alive; call stop() first")
        if self._hold_in_progress():
            raise RuntimeError("release() while a hold is in progress on the bus; wait for it")
        joints = list(self._joints)
        self._bus.torque_disabled(joints)
        if self._state is LoopState.CONFIGURED:
            _confirm_position_mode(self._bus, joints, HoldParams().readback_attempts, logger)
        self._set_state(LoopState.RELEASED)
        logger.warning("released: %s torque OFF", joints)

    # -- snapshots --------------------------------------------------------------

    def latest(self) -> RakudaArmState | None:
        """The last leader reading (non-None from ``configure()`` on)."""
        with self._lock:
            return self._latest

    def latest_follower(self) -> RakudaArmState | None:
        """The last follower reading published by the loop, or ``None``."""
        with self._lock:
            return self._latest_follower

    def wait_first_snapshot(self, timeout_s: float) -> bool:
        """Waits until the running loop has completed one successful read."""
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while self._ok_cycles == 0 and self._state is LoopState.RUNNING:
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    return False
                self._cond.wait(remaining)
            return self._ok_cycles > 0

    def wait_pair(self, timeout_s: float) -> PairSnapshot:
        """Waits for the next leader/follower pair not yet handed out.

        Raises:
            LoopStopped: If the loop is not RUNNING (also when it stops during the wait).
            FollowerLost: If the follower was declared lost.
            BilateralNotReady: If no new pair was published within ``timeout_s``.
            RuntimeError: If the loop has no follower I/O.
        """
        if self._follower is None:
            raise RuntimeError("wait_pair() needs a loop with follower I/O")
        deadline = time.monotonic() + timeout_s
        with self._cond:
            while True:
                if self._state is not LoopState.RUNNING:
                    raise LoopStopped(self._state, self._fault)
                if self._follower_lost:
                    raise FollowerLost("the follower bus stopped answering")
                if self._pair is not None and self._pair.seq > self._pair_consumed:
                    self._pair_consumed = self._pair.seq
                    return self._pair
                remaining = deadline - time.monotonic()
                if remaining <= 0.0:
                    raise BilateralNotReady(f"no new leader/follower pair within {timeout_s:.3f} s")
                self._cond.wait(remaining)

    def re_engage(self) -> None:
        """Restarts the feedback ramp and gate (``t_engaged=0, gate=0``).

        Calls the law's ``engage()`` so the rate limiter and filter keep their
        state and the commanded current stays continuous (a barrier holding a
        joint is not cut).
        """
        if self.latest() is None:
            raise BilateralNotReady("re_engage() needs a leader snapshot; configure() first")
        with self._law_lock:
            self._law.engage()
        self._engaged = True
        self._prev_gate = None

    # -- reports ----------------------------------------------------------------

    def timing_report(self) -> Dict[str, Any]:
        """The ``measured`` block of :meth:`control_report`."""
        return {
            "cycles": self._cycles,
            "ok_cycles": self._ok_cycles,
            "period_ms": self._period_ms.summary(),
            "read_ms": self._read_ms.summary(),
            "write_ms": self._write_ms.summary(),
            "overruns": self._overruns,
            "skipped_cycles": self._skipped,
            "read_fail_count": self._read_fail_count,
            "hold_window_ms": None if self._hold_report is None else self._hold_report.window_ms,
        }

    def control_report(self) -> Dict[str, Any]:
        """The ``control`` block of ``metadata.json``, as plain dicts."""
        report = self._hold_report
        now_ns = self._clock()
        elapsed_s = 0.0 if self._start_ns is None else max(0.0, (now_ns - self._start_ns) / 1e9)
        return {
            "mode": "leader_current",
            "control_hz": self._params.control_hz,
            "state": self._state.value,
            "joints": list(self._joints),
            "current_sign": dict(self._sign),
            "current_sign_convention": "motor_raw",
            "measured": self.timing_report(),
            "preflight": dict(self._preflight),
            "configure": {
                "goal_current_rewritten_at_torque_on": self._goal_current_rewritten_at_torque_on,
                "current_limit_raw": dict(self._current_limit_raw),
            },
            "hold": None
            if report is None
            else {
                "source": dict(report.source),
                "sag_counts": dict(report.sag_counts),
                "window_ms": report.window_ms,
                "goal_rewritten_at_torque_on": report.goal_rewritten_at_torque_on,
                "none": list(report.none),
                "verified": report.verified,
                "notes": list(report.notes),
            },
            "faults": [_fault_dict(fault) for fault in self._faults],
            "follower_io": {
                "attached": self._follower is not None,
                "divider": self._params.follower_divider,
                "hz_effective": self._pairs / elapsed_s if elapsed_s > 0.0 else 0.0,
                "age_ms": self._follower_age_ms.summary(),
                "step_ms": self._follower_step_ms.summary(),
                "failures": 0 if self._follower is None else self._follower.failures,
                "lost": self._follower_lost,
            },
            "feedback_gate": {
                "drops": self._gate_drops,
                "low_cycles": self._gate_low_cycles,
                "gate": self._last_gate,
                "engaged": self._engaged,
            },
        }

    # -- helpers ----------------------------------------------------------------

    def _set_state(self, state: LoopState) -> None:
        with self._cond:
            self._state = state
            self._cond.notify_all()

    def _log_limited(self, level: int, key: str, now_ns: int, msg: str, *args: Any) -> None:
        """Logs ``msg`` at most once per second per ``key`` (on ``clock``)."""
        last = self._log_last_ns.get(key)
        if last is not None and now_ns - last < 1_000_000_000:
            return
        self._log_last_ns[key] = now_ns
        logger.log(level, msg, *args)
