"""Base class of the Rakuda arms: bus ownership and the connect-time sequence.

``connect()`` follows a fixed order:

1. open the port and ``verify_models()``;
2. read ``OPERATING_MODE``/``TORQUE_ENABLE`` of all motors and ``GOAL_CURRENT``
   of the current-capable joints, classify them and bring every
   current-capable joint back to position mode without opening a new
   torque-off window (``_restore_position_mode`` / ``_hold_now``);
3. write the grippers' EEPROM only where it differs from the code;
4. apply the side's torque policy (``_apply_torque_policy``), which
   the subclasses implement.

Nothing here starts the bilateral loop.
"""

import logging
from abc import abstractmethod
from dataclasses import dataclass
from typing import Callable, Collection, Dict, List, Literal, Sequence, Tuple

from robopy.config.robot_config import RakudaConfig
from robopy.config.robot_config.rakuda_config import (
    RAKUDA_CURRENT_CAPABLE_JOINTS,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_HEAD_JOINT_NAMES,
)
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelCommError, DynamixelMotor
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, OperatingMode, XControlTable
from robopy.robots.common.arm import Arm

from .rakuda_leader_control import HoldFailed, HoldParams, hold_joints

logger = logging.getLogger(__name__)

#: ``bus_factory(port, motors) -> bus``; the default is :class:`DynamixelBus` itself.
#: Tests inject a ``SimulatedDynamixelBus`` (a documented drop-in) through it.
BusFactory = Callable[[str, Dict[str, DynamixelMotor]], DynamixelBus]

#: Connect-time state of one current-capable joint:
#: ``ok`` (mode 3, torque 0), ``held`` (3/1, a previous session's hold),
#: ``stale_mode`` (0/0), ``stale_mode_torque_on`` (0/1) or ``unexpected``
#: (any other mode).
ModeState = Literal["ok", "held", "stale_mode", "stale_mode_torque_on", "unexpected"]

_RELEASE_HINT = "run `robopy-rakuda-ports release --port {port} --side {side}`"

#: Modes in which ``TORQUE_ENABLE=1`` makes the motor hold a position rather than
#: drive ``GOAL_CURRENT``/``GOAL_VELOCITY``; the only ones a gripper may be
#: torque-enabled in (no path writes torque-on in current mode).
_POSITION_FAMILY_MODES = frozenset(
    {OperatingMode.POSITION, OperatingMode.EXTENDED_POSITION, OperatingMode.CURRENT_BASED_POSITION}
)


@dataclass
class ConnectState:
    """Registers read at connect time, kept up to date by the steps that write them.

    Attributes:
        mode: ``OPERATING_MODE`` per motor.
        torque: ``TORQUE_ENABLE`` per motor.
        goal_current: Raw ``GOAL_CURRENT`` of the current-capable joints.
        classification: :data:`ModeState` of the current-capable joints.
    """

    mode: Dict[str, int]
    torque: Dict[str, int]
    goal_current: Dict[str, int]
    classification: Dict[str, ModeState]

    @property
    def torque_on(self) -> frozenset[str]:
        """Motors whose ``TORQUE_ENABLE`` is 1 as last known."""
        return frozenset(name for name, value in self.torque.items() if value == 1)


class RakudaArm(Arm):
    """Base class for Rakuda robotic arms (leader and follower)."""

    #: ``"leader"`` or ``"follower"``, for messages and the CLI hint.
    SIDE: str
    #: ``CURRENT_LIMIT`` (EEPROM) the grippers must have; written only when it differs.
    GRIP_CURRENT_LIMIT: int
    #: ``GOAL_CURRENT`` (RAM) written to the grippers on every connect.
    GRIP_GOAL_CURRENT: int

    def __init__(self, cfg: RakudaConfig, port: str, bus_factory: BusFactory | None = None):
        """Builds the bus for ``port`` without opening it.

        Args:
            cfg: Robot configuration (torque lists, ``bilateral``, ...).
            port: Serial device, or ``PORT_AUTO`` to be resolved by ``set_port()``.
            bus_factory: ``(port, motors) -> bus``; defaults to :class:`DynamixelBus`.
        """
        self.config = cfg
        self._port = port
        self._is_connected = False
        factory: BusFactory = DynamixelBus if bus_factory is None else bus_factory
        self._motors = factory(port, self._create_motors())

    @abstractmethod
    def _create_motors(self) -> dict[str, DynamixelMotor]:
        """Create motor configuration specific to each arm type."""

    @abstractmethod
    def _apply_torque_policy(self, state: ConnectState) -> None:
        """Last step of ``connect()``: the side's ``TORQUE_ENABLE`` handling."""

    # ------------------------------------------------------------------
    # Lifecycle
    # ------------------------------------------------------------------

    def set_port(self, port: str) -> None:
        """Points the (closed) bus at ``port``; used to apply a resolved ``auto`` port."""
        self._motors.set_port(port)
        self._port = port

    def connect(self) -> None:
        if self._is_connected:
            logger.info(f"Already connected to the {self.__class__.__name__}.")
            return
        opened = False
        try:
            self._motors.open()
            opened = True
            self._motors.verify_models()
            state = self._classify_mode_state()
            self._recover_mode_state(state)
            for name in self._write_gripper_eeprom():
                state.torque[name] = 0
            self._apply_torque_policy(state)
            self._is_connected = True
            print(f"Connected to the {self.__class__.__name__}.")
        except Exception as e:
            logger.error(f"Failed to connect to the {self.__class__.__name__}: {e}")
            if opened:
                self._close_quietly()
            raise ConnectionError(f"Failed to connect to the {self.__class__.__name__}: {e}") from e

    def disconnect(self, *, torque_off: bool = True) -> None:
        """Closes the port, switching every motor off first unless ``torque_off`` is False.

        ``torque_off=False`` is the ``hold_on_disconnect`` path: the
        motors keep holding their last goal after the port is closed.
        """
        if not self._is_connected:
            logger.info(f"Not connected to the {self.__class__.__name__}.")
            return
        if torque_off:
            try:
                self._motors.torque_disabled()
            except Exception as e:
                logger.warning(f"Failed to disable torque during disconnect: {e}")
        else:
            logger.info(f"{self.SIDE}: torque left on at disconnect (hold_on_disconnect).")
        try:
            self._motors.close()
            self._is_connected = False
            logger.info(f"Disconnected from the {self.__class__.__name__}.")
        except Exception as e:
            logger.error(f"Failed to disconnect from the {self.__class__.__name__}: {e}")
            raise ConnectionError(f"Failed to disconnect from the {self.__class__.__name__}: {e}")

    def _close_quietly(self) -> None:
        try:
            self._motors.close()
        except Exception as e:
            logger.warning(f"Failed to close {self._port} after a connect error: {e}")

    # ------------------------------------------------------------------
    # Connect-time register handling
    # ------------------------------------------------------------------

    def _current_capable_joints(self) -> List[str]:
        return [name for name in RAKUDA_CURRENT_CAPABLE_JOINTS if name in self._motors.motors]

    def _classify_mode_state(self) -> ConnectState:
        """Reads mode/torque of all motors and classifies the current-capable joints.

        Grippers not in mode 5 and head joints not in mode 3 are only warned
        about; the grippers are fixed by :meth:`_write_gripper_eeprom`, the
        head is never written.

        Raises:
            ConnectionError: If a motor did not answer one of the three reads.
        """
        names = self.motor_names
        capable = self._current_capable_joints()
        bus = self._motors
        mode = {n: int(v) for n, v in bus.sync_read(XControlTable.OPERATING_MODE, names).items()}
        torque = {n: int(v) for n, v in bus.sync_read(XControlTable.TORQUE_ENABLE, names).items()}
        goal_current = {
            n: int(v) for n, v in bus.sync_read(XControlTable.GOAL_CURRENT, capable).items()
        }
        missing = sorted(
            {n for n in names if n not in mode or n not in torque}
            | {n for n in capable if n not in goal_current}
        )
        if missing:
            raise ConnectionError(
                f"No OPERATING_MODE/TORQUE_ENABLE/GOAL_CURRENT reading from {missing}."
            )

        classification: Dict[str, ModeState] = {}
        for name in capable:
            if mode[name] == OperatingMode.POSITION:
                classification[name] = "held" if torque[name] else "ok"
            elif mode[name] == OperatingMode.CURRENT:
                classification[name] = "stale_mode_torque_on" if torque[name] else "stale_mode"
            else:
                classification[name] = "unexpected"
        state = ConnectState(mode, torque, goal_current, classification)
        logger.info(
            "%s: register state at connect\n%s", self.SIDE, "\n".join(self._state_table(state))
        )

        wrong_grippers = {
            n: mode[n]
            for n in RAKUDA_GRIPPER_JOINT_NAMES
            if n in mode and mode[n] != OperatingMode.CURRENT_BASED_POSITION
        }
        if wrong_grippers:
            logger.warning(
                "%s: gripper OPERATING_MODE %s is not %d (current-based position); "
                "it is corrected in the gripper EEPROM step.",
                self.SIDE,
                wrong_grippers,
                int(OperatingMode.CURRENT_BASED_POSITION),
            )
        wrong_head = {
            n: mode[n]
            for n in RAKUDA_HEAD_JOINT_NAMES
            if n in mode and mode[n] != OperatingMode.POSITION
        }
        if wrong_head:
            logger.warning(
                "%s: head OPERATING_MODE %s is not %d (position); left as is.",
                self.SIDE,
                wrong_head,
                int(OperatingMode.POSITION),
            )
        return state

    def _state_table(self, state: ConnectState) -> List[str]:
        rows = [f"{'motor':<16} {'id':>3} {'mode':>4} {'torque':>6} {'goal_current':>12}  state"]
        for name, motor in self._motors.motors.items():
            goal = state.goal_current.get(name)
            rows.append(
                f"{name:<16} {motor.id:>3} {state.mode[name]:>4} {state.torque[name]:>6} "
                f"{'-' if goal is None else goal:>12}  {state.classification.get(name, '-')}"
            )
        return rows

    def _recover_mode_state(self, state: ConnectState) -> None:
        """Brings every current-capable joint back to position mode.

        Torque-off joints go through :meth:`_restore_position_mode`, torque-on
        joints through :meth:`_hold_now`; afterwards mode and torque are
        re-read and the invariant "every current-capable joint is in mode 3"
        is enforced.

        Raises:
            ConnectionError: If a joint is still not in position mode, or the
                hold left a joint torque-off for want of a position reading.
        """
        restore = [
            n
            for n, c in state.classification.items()
            if c == "stale_mode" or (c == "unexpected" and not state.torque[n])
        ]
        hold = [
            n
            for n, c in state.classification.items()
            if c == "stale_mode_torque_on" or (c == "unexpected" and state.torque[n])
        ]
        left_off: Tuple[str, ...] = ()
        if restore:
            self._restore_position_mode(restore, state)
        if hold:
            left_off = self._hold_now(hold, state)
        touched = restore + hold
        if not touched:
            return

        bus = self._motors
        mode = bus.sync_read(XControlTable.OPERATING_MODE, touched)
        torque = bus.sync_read(XControlTable.TORQUE_ENABLE, touched)
        state.mode.update({n: int(v) for n, v in mode.items()})
        state.torque.update({n: int(v) for n, v in torque.items()})
        not_position = {n: mode.get(n) for n in touched if mode.get(n) != OperatingMode.POSITION}
        if not_position:
            raise ConnectionError(
                f"{list(not_position)} could not be returned to position mode "
                f"(OPERATING_MODE {not_position}); use the power switch."
            )
        if left_off:
            # No GOAL_POSITION was written after the mode change: the
            # torque policy must not enable these; refuse instead.
            raise ConnectionError(
                f"cannot hold {list(left_off)}: no position reading; torque left OFF, "
                "use the power switch."
            )
        for name in touched:
            state.classification[name] = "held" if state.torque.get(name) == 1 else "ok"

    def _restore_position_mode(self, names: Sequence[str], state: ConnectState) -> None:
        """Mode 3 for torque-off joints found in another mode (``stale_mode``); torque stays off.

        The goal is the position read *after* the mode change (one turn), so a
        later torque-on holds the joint where it is.
        """
        bus = self._motors
        logger.warning(
            "%s: %s found in OPERATING_MODE %s with torque off; restoring position mode "
            "(torque stays off).",
            self.SIDE,
            list(names),
            {n: state.mode[n] for n in names},
        )
        bus.torque_disabled(list(names))  # idempotent: they read 0
        bus.write_with_readback(
            XControlTable.OPERATING_MODE, {n: int(OperatingMode.POSITION) for n in names}
        )
        present = bus.sync_read(XControlTable.PRESENT_POSITION, list(names))
        missing = [n for n in names if n not in present]
        if missing:
            raise ConnectionError(f"No PRESENT_POSITION from {missing} after restoring mode 3.")
        bus.sync_write(XControlTable.GOAL_POSITION, {n: int(present[n]) for n in names})

    def _hold_now(self, names: Sequence[str], state: ConnectState) -> Tuple[str, ...]:
        """Holds joints a previous session left in current mode with torque on.

        Runs :func:`hold_joints` on the calling thread; the joints are then
        ``held`` (mode 3, torque 1) like a normal previous hold.

        Returns:
            The joints the hold left torque-off (``HoldReport.none``); the
            caller refuses the connect for them.

        Raises:
            HoldFailed: Propagated from :func:`hold_joints`.
        """
        bilateral = self.config.bilateral
        params = HoldParams() if bilateral is None else HoldParams.from_bilateral_params(bilateral)
        goal_ma = ", ".join(self._describe_goal_current(n, state.goal_current[n]) for n in names)
        try:
            report = hold_joints(self._motors, names, params, snapshot=None, logger=logger)
        except HoldFailed:
            logger.critical(
                "%s: previous session left %s in current mode with torque on (GOAL_CURRENT %s); "
                "the hold failed, support the arm and use the power switch.",
                self.SIDE,
                list(names),
                goal_ma,
            )
            raise
        logger.critical(
            "%s: previous session left %s in current mode with torque on (GOAL_CURRENT %s); "
            "held now (window %.1f ms, verified=%s%s).",
            self.SIDE,
            list(names),
            goal_ma,
            report.window_ms,
            report.verified,
            f", torque left off on {list(report.none)}" if report.none else "",
        )
        return report.none

    def _describe_goal_current(self, name: str, raw: int) -> str:
        unit = CURRENT_UNIT_MA.get(self._motors.motors[name].model_name)
        return f"{name}={raw} raw" if unit is None else f"{name}={raw * unit:.0f} mA"

    def _write_gripper_eeprom(self, names: Sequence[str] = RAKUDA_GRIPPER_JOINT_NAMES) -> List[str]:
        """Gripper ``OPERATING_MODE``/``CURRENT_LIMIT`` where they differ, then ``GOAL_CURRENT``.

        Only grippers that need a write and read torque 1 are switched off
        first; a ``CURRENT_LIMIT`` read-back mismatch is a warning. An
        ``OPERATING_MODE`` write is re-read afterwards: a gripper still outside
        the position family (3/4/5) would drive ``GOAL_CURRENT`` once torque
        comes on, so it refuses the connect. ``GOAL_CURRENT`` (RAM)
        is written on every connect.

        Args:
            names: Grippers to handle; must be a subset of
                ``RAKUDA_GRIPPER_JOINT_NAMES`` (asserted).

        Returns:
            The grippers whose torque was switched off for the write.

        Raises:
            ConnectionError: If a gripper is not in a position-family mode
                after the ``OPERATING_MODE`` write.
        """
        targets = list(names)
        assert set(targets) <= set(RAKUDA_GRIPPER_JOINT_NAMES), (
            f"gripper EEPROM writes are limited to {RAKUDA_GRIPPER_JOINT_NAMES}, got {targets}"
        )
        bus = self._motors
        torque = bus.sync_read(XControlTable.TORQUE_ENABLE, targets)
        current: Dict[XControlTable, Dict[str, int]] = {
            XControlTable.OPERATING_MODE: bus.sync_read(XControlTable.OPERATING_MODE, targets),
            XControlTable.CURRENT_LIMIT: bus.sync_read(XControlTable.CURRENT_LIMIT, targets),
        }
        wanted = {
            XControlTable.OPERATING_MODE: int(OperatingMode.CURRENT_BASED_POSITION),
            XControlTable.CURRENT_LIMIT: self.GRIP_CURRENT_LIMIT,
        }
        pending: Dict[XControlTable, Dict[str, int]] = {}
        for item, want in wanted.items():
            differing = [n for n in targets if current[item].get(n) != want]
            if differing:
                logger.warning(
                    "%s: gripper %s reads %s, code wants %d; writing EEPROM.",
                    self.SIDE,
                    item.name,
                    {n: current[item].get(n) for n in differing},
                    want,
                )
                pending[item] = {n: want for n in differing}

        switched_off: List[str] = []
        if pending:
            to_write = set().union(*pending.values())
            switched_off = [n for n in targets if n in to_write and torque.get(n) == 1]
            if switched_off:
                bus.torque_disabled(switched_off)
            for item, values in pending.items():
                try:
                    bus.write_with_readback(item, values)
                except DynamixelCommError as e:
                    logger.warning(
                        "%s: gripper %s write not confirmed: %s", self.SIDE, item.name, e
                    )
            if XControlTable.OPERATING_MODE in pending:
                self._require_position_family_mode(list(pending[XControlTable.OPERATING_MODE]))
        bus.sync_write(XControlTable.GOAL_CURRENT, {n: self.GRIP_GOAL_CURRENT for n in targets})
        return switched_off

    def _require_position_family_mode(self, names: Sequence[str]) -> None:
        """Re-reads ``OPERATING_MODE`` and refuses grippers that may not be torque-enabled.

        Raises:
            ConnectionError: If a gripper reads a mode outside 3/4/5 or does
                not answer.
        """
        mode = self._motors.sync_read(XControlTable.OPERATING_MODE, list(names))
        wrong = {n: mode.get(n) for n in names if mode.get(n) not in _POSITION_FAMILY_MODES}
        if wrong:
            raise ConnectionError(
                f"gripper {list(wrong)} still in OPERATING_MODE {wrong} after the EEPROM write; "
                "torque left OFF; inspect it with `robopy-rakuda-ports show --port "
                f"{self._port} --side {self.SIDE}`, power-cycle the arm and connect again."
            )

    def _switch_torque(
        self,
        state: ConnectState,
        want: Collection[str],
        *,
        untouched: Collection[str] = (),
    ) -> None:
        """Diff-based ``TORQUE_ENABLE``: off for on-and-unwanted, on for wanted-and-off.

        A motor that is already on and wanted is never switched off; motors in
        ``untouched`` are not written at all.
        """
        on = state.torque_on
        off = [n for n in self.motor_names if n in on and n not in want and n not in untouched]
        turn_on = [n for n in self.motor_names if n in want and n not in on and n not in untouched]
        if off:
            self._motors.torque_disabled(off)
            state.torque.update({n: 0 for n in off})
        if turn_on:
            self._motors.torque_enabled(turn_on)
            state.torque.update({n: 1 for n in turn_on})
        logger.info(
            "%s: torque switched off %s, on %s; already on and kept %s.",
            self.SIDE,
            off,
            turn_on,
            [n for n in self.motor_names if n in on and n in want],
        )

    def _held_by_previous_session_error(self, names: Sequence[str]) -> ConnectionError:
        hint = _RELEASE_HINT.format(port=self._port, side=self.SIDE)
        return ConnectionError(
            f"previous session left {list(names)} torque-enabled (held); "
            f"{hint} or power-cycle before connecting"
        )

    # ------------------------------------------------------------------
    # Properties
    # ------------------------------------------------------------------

    @property
    def port(self) -> str:
        return self._port

    @property
    def motors(self) -> DynamixelBus:
        return self._motors

    @property
    def is_connected(self) -> bool:
        return self._is_connected

    @property
    def motor_names(self) -> list[str]:
        return list(self._motors.motors.keys())

    @property
    def motor_models(self) -> list[str]:
        return [motor.model_name for motor in self._motors.motors.values()]
