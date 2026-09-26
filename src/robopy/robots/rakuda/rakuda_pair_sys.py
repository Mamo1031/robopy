import atexit
import importlib
import logging
import math
import os
import pickle
import threading
import time
from dataclasses import dataclass, replace
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray
from rich import print

from robopy.config.dotrobopy import apply_rakuda_dotconfig
from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    PORT_AUTO,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_MOTOR_MAPPING,
    RakudaArmObs,
    RakudaArmState,
    RakudaBilateralParams,
    RakudaConfig,
    RakudaTorquePolicy,
    resolve_torque_policy,
)
from robopy.motor.dynamixel_bus import DynamixelBus
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, XControlTable

from ..common.robot import Robot
from .rakuda_arm import BusFactory
from .rakuda_control_laws import BilateralLaw, GravityModel
from .rakuda_follower import RakudaFollower
from .rakuda_leader import RakudaLeader
from .rakuda_leader_control import (
    BilateralNotReady,
    FollowerLost,
    FollowerPositionIO,
    LeaderCurrentLoop,
    LoopState,
    LoopStopped,
    pair_snapshot_to_obs,
    ports_command,
    wrap_delta_counts,
)

logger = logging.getLogger(__name__)

#: One state-block read of the conventional path: a generous bound per
#: attempt (measured p99 is about 5 ms) and one retry.
STATE_READ_TIMEOUT_S = 0.02
STATE_READ_ATTEMPTS = 2

#: Rate of the linear ``GOAL_POSITION`` interpolation of :meth:`RakudaPairSys.ramp_follower_to`.
RAMP_HZ = 50.0
#: How long ``start_bilateral()`` waits for the first reading of the new control thread.
FIRST_SNAPSHOT_TIMEOUT_S = 0.5

#: Counts per revolution of the X-series absolute encoder.
_COUNTS_PER_TURN = 4096
#: The module that provides ``load_bilateral_setup``, imported lazily.
_GRAVITY_MODULE = "robopy.robots.rakuda.rakuda_gravity"

_NOT_CONNECTED = "RakudaPairSys is not connected. Call connect() first."
_LOOP_RUNNING = "bilateral loop is running; call stop_bilateral() first"


@dataclass(frozen=True)
class BilateralSetup:
    """What ``start_bilateral()`` needs from the identification file.

    The gravity identification module (``rakuda_gravity``) produces one from
    ``.robopy/rakuda/leader_gravity.json``; tests build one directly.

    Attributes:
        current_sign: ``joint -> +1/-1`` (``sign-check``); every current joint required.
        joint_range_counts: ``joint -> (lo, hi)`` (``range``); every current joint required.
        drive_mode: ``joint -> DRIVE_MODE`` recorded at ``sign-check``; compared with
            the leader bus when the loop starts.
        gravity: The fitted model, or ``None`` for an uncompensated loop
            (needs ``params.allow_uncompensated``).
        gravity_validated: ``False`` needs ``params.allow_unvalidated_gravity``
            (only checked when there is a model).
        gravity_peak_ma: ``joint -> peak`` of the fitted model, for the gravity clamp.
        source: File path or description, for logs and ``metadata.json``.
        gravity_joints: Joints the model actually compensates, or ``None`` when it
            covers every joint.  A current joint outside it (other than
            ``torso_yaw``, whose axis is vertical) would run with no gravity
            term, which needs ``params.allow_uncompensated``.
    """

    current_sign: Mapping[str, int]
    joint_range_counts: Mapping[str, Tuple[int, int]]
    drive_mode: Mapping[str, int]
    gravity: GravityModel | None
    gravity_validated: bool
    gravity_peak_ma: Mapping[str, float] | None = None
    source: str = ""
    gravity_joints: frozenset[str] | None = None


def load_bilateral_setup(path: str | os.PathLike[str]) -> BilateralSetup:
    """Loads the identification file through ``rakuda_gravity.load_bilateral_setup``.

    The gravity module is imported lazily so this module does not depend on
    it at import time.

    Raises:
        FileNotFoundError: No identification file at ``path``.
        ValueError: A malformed file (the message names the key).
        RuntimeError: When the gravity identification module cannot be imported.
        TypeError: When the module returned something other than a :class:`BilateralSetup`.
    """
    try:
        module = importlib.import_module(_GRAVITY_MODULE)
    except ModuleNotFoundError as exc:
        if exc.name != _GRAVITY_MODULE:
            raise
        raise RuntimeError(
            "gravity identification module not available yet; pass setup= explicitly"
        ) from exc
    setup = module.load_bilateral_setup(path)
    if not isinstance(setup, BilateralSetup):
        raise TypeError(
            f"{_GRAVITY_MODULE}.load_bilateral_setup must return a BilateralSetup, "
            f"got {type(setup).__name__}"
        )
    return setup


#: Current-capable joints whose gravity term is zero by construction: the torso
#: turns about the vertical axis, so no gravity model needs to cover it.
_GRAVITY_FREE_JOINTS = frozenset({"torso_yaw"})


def _check_bilateral_setup(setup: BilateralSetup, params: RakudaBilateralParams) -> None:
    """The two gravity gates plus the completeness of the per-joint tables.

    Raises:
        ValueError: Naming the joints without ``current_sign`` /
            ``joint_range_counts`` / ``drive_mode``, an invalid sign, a missing
            gravity model without ``allow_uncompensated`` or an unvalidated
            one without ``allow_unvalidated_gravity``.
    """
    where = setup.source or "the bilateral setup"
    joints = params.current_joints
    for table_name in ("current_sign", "joint_range_counts", "drive_mode"):
        table = getattr(setup, table_name)
        missing = [name for name in joints if name not in table]
        if missing:
            raise ValueError(f"{where}: no {table_name} for current joint(s) {missing}")
    bad_sign = {
        name: setup.current_sign[name] for name in joints if setup.current_sign[name] not in (1, -1)
    }
    if bad_sign:
        raise ValueError(f"{where}: current_sign must be +1 or -1, got {bad_sign}")
    if setup.gravity is None:
        if not params.allow_uncompensated:
            raise ValueError(
                f"no gravity model in {where}; set bilateral.allow_uncompensated=true to run "
                "without compensation"
            )
        return
    if setup.gravity_joints is not None and not params.allow_uncompensated:
        uncovered = [
            name
            for name in joints
            if name not in setup.gravity_joints and name not in _GRAVITY_FREE_JOINTS
        ]
        if uncovered:
            raise ValueError(
                f"the gravity model in {where} does not cover current joint(s) {uncovered}; "
                "identify that arm, narrow bilateral.current_joints, or set "
                "bilateral.allow_uncompensated=true"
            )
    if not setup.gravity_validated and not params.allow_unvalidated_gravity:
        raise ValueError(
            f"the gravity model in {where} is not validated; run `verify` or set "
            "bilateral.allow_unvalidated_gravity=true"
        )


def _alignment_error_counts(
    leader: RakudaArmState, follower: RakudaArmState, joints: Sequence[str]
) -> Dict[str, int]:
    """``q_F - q_L`` per joint, shortest way round the encoder circle."""
    leader_index = {name: i for i, name in enumerate(leader.names)}
    follower_index = {name: i for i, name in enumerate(follower.names)}
    return {
        name: wrap_delta_counts(
            int(follower.position[follower_index[name]]),
            int(leader.position[leader_index[name]]),
        )
        for name in joints
    }


def _worst_alignment(errors: Mapping[str, int]) -> Tuple[int, str]:
    """``(max |error|, "joint=error, ...")`` for messages."""
    worst = max(abs(error) for error in errors.values())
    detail = ", ".join(f"{name}={error:+d}" for name, error in errors.items() if error != 0)
    return worst, detail or "aligned"


def _filter_action_by_enabled_joints(
    action: Dict[str, float],
    enabled_joints: set[str],
) -> Dict[str, float]:
    return {name: value for name, value in action.items() if name in enabled_joints}


def _current_unit_ma(bus: DynamixelBus) -> NDArray[np.float64]:
    """``CURRENT_UNIT_MA`` of every motor on ``bus``, in bus order.

    Kept in float64 so that ``raw * unit`` is rounded to float32 only once.

    Raises:
        ValueError: For a motor model without a known current unit.
    """
    units = []
    for name, motor in bus.motors.items():
        if motor.model_name not in CURRENT_UNIT_MA:
            raise ValueError(f"No current unit is known for {name} ({motor.model_name})")
        units.append(CURRENT_UNIT_MA[motor.model_name])
    return np.asarray(units, dtype=np.float64)


class _ArmStateReader:
    """Reads one bus as a :class:`RakudaArmState`; one per arm.

    Holds what the conversion needs: the motor order, the mA-per-count of each
    motor's model and the running ``seq`` of the state.
    """

    def __init__(self, bus: DynamixelBus) -> None:
        self._bus = bus
        self._names: Tuple[str, ...] = tuple(bus.motors)
        self._unit_ma = _current_unit_ma(bus)
        self._seq = 0

    def read(self) -> RakudaArmState:
        """One ``read_state_block`` of every motor; raises what the bus raises."""
        readings, start_ns, end_ns = self._bus.read_state_block(
            self._names, timeout_s=STATE_READ_TIMEOUT_S, attempts=STATE_READ_ATTEMPTS
        )
        count = len(self._names)
        position = np.fromiter((readings[n].position for n in self._names), np.int32, count)
        velocity = np.fromiter((readings[n].velocity for n in self._names), np.int32, count)
        current_raw = np.fromiter((readings[n].current_raw for n in self._names), np.int32, count)
        self._seq += 1
        return RakudaArmState(
            names=self._names,
            position=position,
            velocity=velocity,
            current_ma=(current_raw * self._unit_ma).astype(np.float32),
            t_start_ns=start_ns,
            t_end_ns=end_ns,
            seq=self._seq,
        )


class RakudaPairSys(Robot):
    """The Rakuda leader/follower pair: conventional position teleoperation and, when
    ``cfg.bilateral`` is set, the leader current loop.

    Bus ownership: while the bilateral loop is RUNNING its control thread is
    the only user of both buses; the observation methods then return
    the loop's snapshots and every method that would write or read a bus
    itself raises ``RuntimeError``.  In every other state (conventional mode,
    no loop yet, HELD/FAULT/RELEASED) the calling thread owns the buses,
    the follower under ``_follower_lock``.
    """

    def __init__(
        self,
        cfg: RakudaConfig,
        bus_factory: BusFactory | None = None,
        *,
        clock: Callable[[], int] = time.monotonic_ns,
        sleep: Callable[[float], None] = time.sleep,
    ) -> None:
        """Builds both arms without opening a port.

        Args:
            cfg: Robot configuration; ``.robopy/rakuda/config.yaml`` overrides are applied.
            bus_factory: ``(port, motors) -> bus`` handed to both arms; defaults
                to the real :class:`DynamixelBus` (tests inject simulated buses).
            clock: ``time.monotonic_ns``-compatible clock of the alignment ramp and
                the leader loop (tests pass the simulated buses' clock).
            sleep: ``time.sleep``-compatible sleep used with ``clock``.
        """
        cfg = apply_rakuda_dotconfig(cfg)
        self.config = cfg
        self._clock = clock
        self._sleep = sleep
        # The torque defaults are resolved once here; a follower list
        # that does not cover the bilateral joints is a ValueError at construction.
        self._torque_policy = resolve_torque_policy(cfg)
        self._leader = RakudaLeader(cfg, bus_factory)
        self._follower = RakudaFollower(cfg, bus_factory)
        self._is_connected = False
        self._motor_mapping = (
            RAKUDA_MOTOR_MAPPING  # key: leader motor name, value: follower motor name
        )
        self._leader_motor_names = list(self._leader.motors.motors.keys())
        self._follower_motor_names = list(self._follower.motors.motors.keys())

        # Cache torque-enabled joints for safe write filtering.
        self._leader_torque_enabled: set[str] = set(self._torque_policy.leader)
        self._follower_torque_enabled: set[str] = set(self._torque_policy.follower)

        # Conventional-path state reads. The follower lock keeps
        # a step's follower write and read together.
        self._leader_state_reader = _ArmStateReader(self._leader.motors)
        self._follower_state_reader = _ArmStateReader(self._follower.motors)
        self._follower_lock = threading.RLock()

        # The bilateral loop: None until start_bilateral() succeeds; a
        # stopped loop is kept for control_report() and release().
        self._leader_loop: LeaderCurrentLoop | None = None
        self._bilateral_setup: BilateralSetup | None = None
        self._atexit_registered = False
        # The first direct bus read after a loop stopped is warned once.
        self._direct_read_warned = False

    def connect(self) -> None:
        """Connect to both leader and follower arms."""
        if self.is_connected:
            logger.info("Successfully connected to both leader and follower arms.")
            return

        try:
            self._resolve_auto_ports()
            self._leader.connect()
            self._follower.connect()
            logger.info("Successfully connected to both leader and follower arms.")
            print("[cyan]Successfully connected to both leader and follower arms.[/cyan]")
            self._is_connected = True
        except (OSError, IOError, PermissionError) as e:
            logger.error(f"Failed to connect to arms: {e}")
            raise ConnectionError(f"Failed to connect to arms: {e}")
        except (pickle.PickleError, EOFError) as e:
            logger.error(f"Calibration data corrupted: {e}")
            raise ConnectionError(f"Calibration data error: {e}")

    def _resolve_auto_ports(self) -> None:
        """Replaces ``PORT_AUTO`` by the detected devices before any port is opened.

        The resolved ports are stored back into ``self.config`` so that
        ``RakudaPairSys.config`` names the devices actually opened.
        """
        leader_port, follower_port = self.config.leader_port, self.config.follower_port
        if PORT_AUTO not in (leader_port, follower_port):
            return
        # Imported here: rakuda_ports imports the arm classes of this package.
        from .rakuda_ports import resolve_ports

        leader_port, follower_port = resolve_ports(leader_port, follower_port)
        if leader_port != self._leader.port:
            self._leader.set_port(leader_port)
        if follower_port != self._follower.port:
            self._follower.set_port(follower_port)
        self.config = replace(self.config, leader_port=leader_port, follower_port=follower_port)

    def disconnect(self) -> None:
        """Disconnect from both leader and follower arms.

        A running bilateral loop is stopped first (both arms held, never
        raises) and the ``atexit`` hook removed.  Torque is then switched off
        unless ``effective_hold_on_disconnect`` is set, in which case both
        arms keep holding their last goal.
        """
        self.stop_bilateral()
        self._unregister_atexit()
        torque_off = not self.config.effective_hold_on_disconnect
        self.leader.disconnect(torque_off=torque_off)
        self.follower.disconnect(torque_off=torque_off)

    def _require_connected(self) -> None:
        if not self._is_connected:
            raise ConnectionError(_NOT_CONNECTED)

    def _bilateral_params(self) -> RakudaBilateralParams:
        params = self.config.bilateral
        if params is None:
            raise RuntimeError(
                "bilateral mode is off: pass RakudaConfig(bilateral=RakudaBilateralParams(...))"
            )
        return params

    def _running_loop(self) -> LeaderCurrentLoop | None:
        """The loop while it is RUNNING (it owns both buses), else ``None``."""
        loop = self._leader_loop
        return loop if loop is not None and loop.running else None

    def _refuse_while_running(self) -> None:
        """No bus I/O from the calling thread while the loop runs."""
        if self._running_loop() is not None:
            raise RuntimeError(_LOOP_RUNNING)

    @staticmethod
    def _snapshot(state: RakudaArmState | None, side: str) -> RakudaArmState:
        if state is None:
            raise ValueError(f"the bilateral loop has not published a {side} reading yet")
        return state

    def _warn_direct_read_once(self) -> None:
        """One warning per stopped loop when the buses are read directly."""
        loop = self._leader_loop
        if loop is None or self._direct_read_warned:
            return
        self._direct_read_warned = True
        logger.warning(
            "bilateral loop is %s: reading the buses directly (the leader is in position mode)",
            loop.state.name,
        )

    def _read_arms(self) -> Tuple[RakudaArmState, RakudaArmState]:
        """Both arms from the calling thread, leader first (no loop RUNNING)."""
        leader_state = self._leader_state_reader.read()
        with self._follower_lock:
            follower_state = self._follower_state_reader.read()
        return leader_state, follower_state

    def _to_follower_goal(self, leader_positions: Dict[str, float]) -> Dict[str, float]:
        """Maps leader joint positions onto follower goal positions."""
        follower_goal_positions: Dict[str, float] = {}
        for leader_name, position in leader_positions.items():
            follower_name = self._motor_mapping.get(leader_name)
            if follower_name:
                follower_goal_positions[follower_name] = position
        return follower_goal_positions

    def get_leader_state(self) -> RakudaArmState:
        """Position, velocity and current of every leader motor.

        One ``read_state_block`` on the calling thread; while the bilateral
        loop runs, its latest leader snapshot instead (no bus I/O).

        Raises:
            ValueError: If the running loop has no leader reading yet.
        """
        self._require_connected()
        loop = self._running_loop()
        if loop is not None:
            return self._snapshot(loop.latest(), "leader")
        self._warn_direct_read_once()
        return self._leader_state_reader.read()

    def get_follower_state(self) -> RakudaArmState:
        """Position, velocity and current of every follower motor.

        One ``read_state_block`` under ``_follower_lock``; while the bilateral
        loop runs, its latest follower snapshot instead (no bus I/O).

        Raises:
            ValueError: If the running loop has no follower reading yet.
        """
        self._require_connected()
        loop = self._running_loop()
        if loop is not None:
            return self._snapshot(loop.latest_follower(), "follower")
        self._warn_direct_read_once()
        with self._follower_lock:
            return self._follower_state_reader.read()

    def get_observation(self) -> RakudaArmObs:
        """Reads both arms (leader first) without commanding anything.

        While the bilateral loop runs this is built from its two latest
        snapshots without touching a bus.
        """
        self._require_connected()
        leader_state = self.get_leader_state()
        follower_state = self.get_follower_state()
        return RakudaArmObs.from_states(leader_state, follower_state)

    def teleoperate(self, max_seconds: float | None = None) -> None:
        """Leader controls follower; with ``max_seconds`` set, returns after that long.

        Conventional mode is unchanged: position copy as fast as the bus
        allows, Ctrl-C switches the follower off and returns.  Bilateral mode:
        ``ensure_bilateral_running()`` first, then one
        :meth:`teleoperate_step` per published pair; the loop stays RUNNING
        on return.  ``LoopStopped``/``FollowerLost`` are logged and re-raised;
        Ctrl-C holds both arms (:meth:`stop_bilateral`) and is re-raised.
        """
        self._require_connected()
        if self.config.bilateral is not None:
            self._teleoperate_bilateral(max_seconds)
            return

        logger.info("Starting teleoperation. Leader will control follower.")
        start_time = time.time()
        try:
            while True:
                # Get current positions from leader arm
                leader_positions = self.get_leader_action()
                # Map leader positions to follower positions
                follower_positions = self._to_follower_goal(leader_positions)

                # Send positions to follower arm
                try:
                    self.send_follower_action(follower_positions)
                    logger.info(f"Sent follower action: {follower_positions}")
                except Exception:
                    logger.exception("Failed to send follower action; continuing loop.")

                # Check for max_seconds
                if max_seconds is not None and (time.time() - start_time) >= max_seconds:
                    logger.info("Reached max_seconds; exiting teleoperate.")
                    break

            # TODO: add a better way to stop smoothly, eg. set a home position
        except KeyboardInterrupt:
            self.follower.motors.torque_disabled()
            logger.info("Teleoperation stopped by user.")
        except Exception:
            logger.exception("Error during teleoperation.")
            raise

    def _teleoperate_bilateral(self, max_seconds: float | None) -> None:
        """The bilateral body of :meth:`teleoperate`."""
        self.ensure_bilateral_running()
        logger.info("Starting bilateral teleoperation (leader current loop).")
        start_ns = self._clock()
        try:
            while True:
                try:
                    self.teleoperate_step()
                except BilateralNotReady as exc:
                    logger.warning("%s", exc)
                if max_seconds is not None and self._clock() - start_ns >= max_seconds * 1e9:
                    logger.info("Reached max_seconds; exiting teleoperate (loop keeps running).")
                    return
        except (LoopStopped, FollowerLost) as exc:
            logger.error("Bilateral teleoperation stopped: %s", exc)
            raise
        except KeyboardInterrupt:
            self.stop_bilateral()
            logger.info("Teleoperation stopped by user; both arms are held in place.")
            raise

    def teleoperate_step(self) -> RakudaArmObs:
        """One teleoperation cycle with the full observation.

        Conventional mode: four bus transactions in the order of the
        position-only implementation: leader read -> follower
        ``GOAL_POSITION`` -> leader gripper hold (``LEADER_GRIP_HOLD_POSITION``)
        -> follower read.  A failed write is logged and the cycle continues.

        Once a bilateral loop exists it is the next pair the loop published:
        no bus I/O here, each pair handed out once.

        Returns:
            Positions, velocities and currents of both arms plus the read
            stamps (``leader_t_ns``/``follower_t_ns``); ``obs.leader`` is what
            was sent to the follower.

        Raises:
            LoopStopped: If the loop is HELD/FAULT/RELEASED (bilateral only).
            FollowerLost: If the loop declared the follower lost.
            BilateralNotReady: If no new pair arrived within ``4 * D / control_hz``.
        """
        self._require_connected()
        loop = self._leader_loop
        if loop is not None:
            params = self._bilateral_params()
            timeout_s = 4.0 * params.follower_divider / params.control_hz
            return pair_snapshot_to_obs(loop.wait_pair(timeout_s))

        leader_state = self.get_leader_state()
        leader_positions = dict(zip(leader_state.names, leader_state.position.tolist()))
        follower_goal_positions = self._to_follower_goal(leader_positions)
        with self._follower_lock:
            try:
                self.send_follower_action(follower_goal_positions)
                self.send_leader_action(
                    {name: LEADER_GRIP_HOLD_POSITION for name in RAKUDA_GRIPPER_JOINT_NAMES}
                )
            except Exception:
                logger.exception("Failed to send follower action; continuing.")

            follower_state = self.get_follower_state()
        return RakudaArmObs.from_states(leader_state, follower_state)

    def control_step(self) -> Dict[str, float]:
        """
        LeRobot-style high-frequency control step.

        Performs minimal necessary operations for teleoperation:
        1. Read leader positions
        2. Map to follower
        3. Send to follower

        This is FAST (~10-16ms) and suitable for 60Hz control loops.

        Returns:
            Dict[str, float]: Leader positions that were sent to follower
        """
        self._require_connected()
        self._refuse_while_running()

        # Read leader positions
        leader_positions = self.get_leader_action()

        # Map leader positions to follower goal positions
        follower_goal_positions = self._to_follower_goal(leader_positions)

        # Send to follower
        self.send_follower_action(follower_goal_positions)

        return leader_positions

    def get_observation_with_leader(self, leader_positions: Dict[str, float]) -> RakudaArmObs:
        """
        Get observation using pre-read leader positions.

        This is useful when you already have leader positions from control_step()
        and want to avoid redundant communication.

        Args:
            leader_positions: Pre-read leader positions

        Returns:
            RakudaArmObs: Current arm observation
        """
        self._require_connected()
        self._refuse_while_running()

        # Read follower positions
        follower_positions = self.get_follower_action()

        # Convert to arrays
        leader_obs = np.array(list(leader_positions.values()), dtype=np.float32)
        follower_obs = np.array(list(follower_positions.values()), dtype=np.float32)

        return RakudaArmObs(leader=leader_obs, follower=follower_obs)

    def get_leader_action(self) -> Dict[str, float]:
        """Get the current action (positions) from the leader arm."""
        self._require_connected()
        self._refuse_while_running()

        leader_motor_names = list(self._leader.motors.motors.keys())
        leader_positions = self._leader.motors.sync_read(
            XControlTable.PRESENT_POSITION, leader_motor_names
        )
        return leader_positions

    def send_leader_action(self, action: Dict[str, float]) -> None:
        """Send action to the leader arm only (refused while the bilateral loop runs)."""
        self._require_connected()
        self._refuse_while_running()
        filtered = _filter_action_by_enabled_joints(action, self._leader_torque_enabled)
        if not filtered:
            return
        self._leader.motors.sync_write(XControlTable.GOAL_POSITION, filtered)

    def get_follower_action(self) -> Dict[str, float]:
        """Get the current action (positions) from the follower arm."""
        self._require_connected()
        self._refuse_while_running()

        follower_motor_names = self._follower_motor_names

        follower_positions = self._follower.motors.sync_read(
            XControlTable.PRESENT_POSITION, follower_motor_names
        )
        return follower_positions

    def send_follower_action(self, action: Dict[str, float]) -> None:
        """Send action to the follower arm only (refused while the bilateral loop runs)."""
        self._require_connected()
        self._refuse_while_running()
        filtered = _filter_action_by_enabled_joints(action, self._follower_torque_enabled)
        if not filtered:
            return

        self._follower.motors.sync_write(XControlTable.GOAL_POSITION, filtered)

    # ------------------------------------------------------------------
    # Bilateral loop
    # ------------------------------------------------------------------

    def start_bilateral(self, setup: BilateralSetup | None = None) -> LeaderCurrentLoop:
        """Starts the leader current loop.

        In order: the current joints must still be in ``torque_policy.follower``;
        the SDK location and the FTDI ``latency_timer`` of both ports are
        checked (warnings only); ``setup`` is loaded from
        ``params.gravity_model_path`` when not given and gated
        (``_check_bilateral_setup``); the leader's ``DRIVE_MODE`` must match
        the sign-check; both arms are read
        and a misalignment above ``max_alignment_counts`` is refused; the
        follower is ramped onto the leader (:meth:`ramp_follower_to`,
        ``align_s``); then ``preflight()`` -> ``configure()`` -> ``start()`` ->
        first leader reading and first leader/follower pair -> ``re_engage()``.
        The ``atexit`` hold is registered once per instance.  Allowed
        again after HELD/FAULT/RELEASED; the stopped loop is replaced.

        Anything raised after ``configure()`` (Ctrl-C or ``SystemExit`` included)
        stops and holds the new loop before propagating; nothing is stored, so
        the pair system stays NOT_STARTED.

        Returns:
            The running :class:`LeaderCurrentLoop`.

        Raises:
            RuntimeError: Conventional configuration, a loop already RUNNING, a
                current joint outside the follower torque policy, the arms
                misaligned, or no first reading within ``FIRST_SNAPSHOT_TIMEOUT_S``.
            ValueError: A gate of ``_check_bilateral_setup`` or a ``DRIVE_MODE``
                change since the sign-check.
            ConnectionError: Propagated from ``preflight()``.
            ConfigureError: Propagated from ``configure()`` (rolled back).
            BilateralNotReady: No leader/follower pair within ``FIRST_SNAPSHOT_TIMEOUT_S``.
            LoopStopped: The loop faulted before its first pair.
            FollowerLost: The follower stopped answering before its first pair.
        """
        params = self._bilateral_params()
        self._require_connected()
        if self._running_loop() is not None:
            raise RuntimeError("already running")
        joints = params.current_joints
        missing = sorted(set(joints) - self._follower_torque_enabled)
        if missing:
            raise RuntimeError(
                f"bilateral current_joints must be torque-enabled on the follower; "
                f"missing: {missing}"
            )
        self._log_environment()

        if setup is None:
            setup = load_bilateral_setup(params.gravity_model_path)
        _check_bilateral_setup(setup, params)
        self._check_drive_mode(setup, joints)
        loop = self._build_loop(setup, params)

        leader_state, follower_state = self._read_arms()
        errors = _alignment_error_counts(leader_state, follower_state, joints)
        worst, detail = _worst_alignment(errors)
        if worst > params.max_alignment_counts:
            raise RuntimeError(
                f"leader/follower misaligned by {worst} counts (> max_alignment_counts "
                f"{params.max_alignment_counts}): {detail}; move the arms closer and retry"
            )
        leader_positions = dict(zip(leader_state.names, leader_state.position.tolist()))
        with self._follower_lock:
            self.ramp_follower_to(self._to_follower_goal(leader_positions), params.align_s)
            errors = _alignment_error_counts(*self._read_arms(), joints)
        worst, detail = _worst_alignment(errors)
        if worst > 2 * params.feedback_deadband_counts:
            logger.warning(
                "follower still %d counts from the leader after the %.1f s alignment ramp: %s",
                worst,
                params.align_s,
                detail,
            )

        loop.preflight()
        deadline = time.monotonic() + FIRST_SNAPSHOT_TIMEOUT_S
        try:
            loop.configure()
            loop.start()
            if not loop.wait_first_snapshot(FIRST_SNAPSHOT_TIMEOUT_S):
                raise RuntimeError(
                    f"the leader loop produced no reading within {FIRST_SNAPSHOT_TIMEOUT_S} s; "
                    f"stopped and held (faults: {[f.reason for f in loop.faults]})"
                )
            # RUNNING must imply both snapshots exist, so the first
            # pair is awaited (and consumed) here; this also raises when the loop
            # faulted right after its first reading.
            loop.wait_pair(max(deadline - time.monotonic(), 0.0))
            loop.re_engage()
        except BaseException:
            # From configure() on the leader is in current mode, possibly under a
            # thread nothing else references yet: hold before propagating.
            self._stop_loop(loop)
            raise
        self._register_atexit()
        self._leader_loop = loop
        self._direct_read_warned = False
        self._bilateral_setup = setup
        logger.info(
            "bilateral loop running on %s (%s, gravity %s)",
            list(joints),
            setup.source or "in-memory setup",
            "uncompensated"
            if setup.gravity is None
            else ("validated" if setup.gravity_validated else "NOT validated"),
        )
        return loop

    def _build_loop(
        self, setup: BilateralSetup, params: RakudaBilateralParams
    ) -> LeaderCurrentLoop:
        """The follower I/O, the law and the loop; validates without touching a bus."""
        joints = params.current_joints
        follower_io = FollowerPositionIO(
            self._follower.motors,
            tuple(self._torque_policy.follower),
            params.follower_read_timeout_s,
        )
        law = BilateralLaw(
            joints,
            params,
            setup.gravity,
            setup.joint_range_counts,
            gravity_peak_ma=setup.gravity_peak_ma,
        )
        leader_motors = self._leader.motors.motors
        units_ma = {name: CURRENT_UNIT_MA[leader_motors[name].model_name] for name in joints}
        return LeaderCurrentLoop(
            self._leader.motors,
            joints,
            params,
            law,
            units_ma,
            {name: int(setup.current_sign[name]) for name in joints},
            follower_io,
            clock=self._clock,
            sleep=self._sleep,
            gripper_hold={name: LEADER_GRIP_HOLD_POSITION for name in RAKUDA_GRIPPER_JOINT_NAMES},
        )

    def _check_drive_mode(self, setup: BilateralSetup, joints: Sequence[str]) -> None:
        """``DRIVE_MODE`` of the current joints must be what the sign-check saw."""
        read = self._leader.motors.sync_read(XControlTable.DRIVE_MODE, list(joints))
        changed = {
            name: (setup.drive_mode[name], None if name not in read else int(read[name]))
            for name in joints
            if name not in read or int(read[name]) != setup.drive_mode[name]
        }
        if changed:
            raise ValueError(
                f"DRIVE_MODE changed since sign-check (recorded, read): {changed}; "
                "re-run `sign-check`"
            )

    def _log_environment(self) -> None:
        """The SDK's location and the FTDI latency of both ports."""
        # Imported here: rakuda_ports imports the arm classes of this package.
        from .rakuda_ports import check_sdk_location, read_latency_timer

        check_sdk_location()
        for side, port in (("leader", self._leader.port), ("follower", self._follower.port)):
            latency_ms = read_latency_timer(port)
            if latency_ms is not None and latency_ms != 1:
                logger.warning(
                    "%s port %s: FTDI latency_timer is %d ms (1 expected); bus reads will "
                    "be slower than the bench values",
                    side,
                    port,
                    latency_ms,
                )

    def ramp_follower_to(self, goal: Mapping[str, float], duration_s: float) -> None:
        """Moves the torque-enabled follower motors to ``goal`` with a linear
        ``GOAL_POSITION`` ramp at ``RAMP_HZ``.

        The ramp starts from the follower's present position, goes the shortest
        way round the encoder circle and, like the running loop, writes goals
        reduced to one turn; the last write is exactly ``goal % 4096``.  The
        follower bus is borrowed under ``_follower_lock`` for the whole ramp.

        Args:
            goal: Follower motor name -> target position in counts; motors
                outside ``torque_policy.follower`` are ignored.
            duration_s: Ramp duration; ``<= 0`` writes the goal once.

        Raises:
            RuntimeError: While the bilateral loop runs (it owns the follower bus).
        """
        self._require_connected()
        self._refuse_while_running()
        targets = {
            name: int(round(value))
            for name, value in goal.items()
            if name in self._follower_torque_enabled
        }
        if not targets:
            return
        steps = max(1, math.ceil(duration_s * RAMP_HZ))
        period_ns = round(1e9 / RAMP_HZ)
        with self._follower_lock:
            state = self._follower_state_reader.read()
            start = {
                name: int(position)
                for name, position in zip(state.names, state.position.tolist())
                if name in targets
            }
            delta = {name: wrap_delta_counts(targets[name], start[name]) for name in start}
            deadline_ns = self._clock()
            for k in range(1, steps + 1):
                alpha = k / steps
                values: Dict[str, int | float] = {
                    name: int(round(start[name] + alpha * delta[name])) % _COUNTS_PER_TURN
                    for name in start
                }
                self._follower.motors.sync_write(XControlTable.GOAL_POSITION, values)
                deadline_ns += period_ns
                remaining_s = (deadline_ns - self._clock()) / 1e9
                if remaining_s > 0.0:
                    self._sleep(remaining_s)

    def stop_bilateral(self) -> bool:
        """Stops the loop and holds the leader in place; never raises.

        Returns:
            True when there is no loop or every joint is verified held; False
            (with a CRITICAL log of the loop's faults) otherwise.  The loop
            object is kept for :meth:`control_report` and :meth:`release`.
        """
        loop = self._leader_loop
        if loop is None:
            return True
        return self._stop_loop(loop)

    @staticmethod
    def _stop_loop(loop: LeaderCurrentLoop) -> bool:
        """``loop.stop()`` that never raises; an unverified hold is logged CRITICAL."""
        try:
            held = loop.stop()
        except Exception:
            logger.critical(
                "stop_bilateral: stop() raised; the leader may not be held", exc_info=True
            )
            return False
        if not held:
            logger.critical(
                "stop_bilateral: the hold is not verified (state %s, faults %s); support the "
                "leader and check %s",
                loop.state.name,
                [f"{f.reason}: {f.detail}" for f in loop.faults],
                ports_command("show", loop.port_name, "leader"),
            )
        return held

    def ensure_bilateral_running(self) -> None:
        """Entry-point hook: start, re-align or refuse.

        Conventional mode: no-op.  No loop yet: :meth:`start_bilateral`.
        RUNNING: :meth:`_realign_follower`.  HELD/FAULT/RELEASED: raises.

        Raises:
            LoopStopped: With the loop's state and latched fault.
        """
        if self.config.bilateral is None:
            return
        loop = self._leader_loop
        if loop is None:
            self.start_bilateral()
        elif loop.running:
            self._realign_follower(loop)
        else:
            raise LoopStopped(loop.state, loop.fault)

    def _realign_follower(self, loop: LeaderCurrentLoop) -> None:
        """Re-entry of ``ensure_bilateral_running()`` while RUNNING.

        The running loop owns the follower bus and copies the leader
        positions every cycle, so no ramp runs here: a gap above
        ``2 * feedback_deadband_counts`` is only logged.  The feedback ramp
        and gate are restarted (``re_engage()``).  ``loop`` is the caller's
        RUNNING loop; a fault landing in between is harmless here and surfaces
        as ``LoopStopped`` from the next ``wait_pair()``.
        """
        params = self._bilateral_params()
        leader, follower = loop.latest(), loop.latest_follower()
        if leader is not None and follower is not None:
            worst, detail = _worst_alignment(
                _alignment_error_counts(leader, follower, params.current_joints)
            )
            if worst > 2 * params.feedback_deadband_counts:
                logger.warning(
                    "follower is %d counts from the leader (%s); the running loop keeps "
                    "copying positions, no alignment ramp",
                    worst,
                    detail,
                )
        loop.re_engage()

    def release(self) -> None:
        """Torque off: the leader's current joints after a hold, then
        every follower motor.  Nothing asks for confirmation here (the CLI does).

        Conventional mode releases the follower only.  Idempotent for the loop.

        Raises:
            RuntimeError: If the control thread is still alive after
                ``stop_bilateral()`` (``stop_timeout``): nothing is released.
        """
        self._require_connected()
        loop = self._leader_loop
        if not self.stop_bilateral() and loop is not None and loop.thread_alive:
            raise RuntimeError(
                "release() refused: the control thread is still alive; do not touch the ports "
                f"({loop.recovery_hint()})"
            )
        if loop is not None and loop.state in (
            LoopState.HELD,
            LoopState.FAULT,
            LoopState.CONFIGURED,
        ):
            loop.release()
        with self._follower_lock:
            self._follower.motors.torque_disabled()
        logger.warning("released: follower torque OFF")

    @property
    def bilateral_active(self) -> bool:
        """True while the leader current loop is RUNNING (it owns both buses)."""
        return self._running_loop() is not None

    def control_report(self) -> Dict[str, Any] | None:
        """The loop's ``control`` block of ``metadata.json``, or ``None``.

        ``None`` until a loop has been started (conventional mode included);
        ``RakudaRobot.control_report()`` merges it with the position-teleop
        report.  Adds ``setup`` (``source``, ``gravity_validated``,
        ``uncompensated``) to ``LeaderCurrentLoop.control_report()``.
        """
        loop, setup = self._leader_loop, self._bilateral_setup
        if loop is None or setup is None:
            return None
        report = loop.control_report()
        report["setup"] = {
            "source": setup.source,
            "gravity_validated": setup.gravity_validated,
            "uncompensated": setup.gravity is None,
        }
        return report

    def _atexit_hold(self) -> None:
        """Interpreter-exit hold; a no-op once the loop is held."""
        try:
            self.stop_bilateral()
        except Exception:
            logger.critical("atexit: stop_bilateral() failed", exc_info=True)

    def _register_atexit(self) -> None:
        if not self._atexit_registered:
            atexit.register(self._atexit_hold)
            self._atexit_registered = True

    def _unregister_atexit(self) -> None:
        if self._atexit_registered:
            atexit.unregister(self._atexit_hold)
            self._atexit_registered = False

    @property
    def is_connected(self) -> bool:
        """Check if both arms are connected."""
        return self._is_connected

    @property
    def torque_policy(self) -> RakudaTorquePolicy:
        """The resolved torque policy of this configuration."""
        return self._torque_policy

    @property
    def port(self) -> str:
        """Get the ports of both arms."""
        return f"Leader: {self.leader.port}, Follower: {self.follower.port}"

    @property
    def motors(self) -> dict[str, DynamixelBus]:
        """Get the motor buses of both arms."""
        return {"leader": self.leader.motors, "follower": self.follower.motors}

    @property
    def leader(self) -> RakudaLeader:
        return self._leader

    @property
    def follower(self) -> RakudaFollower:
        return self._follower
