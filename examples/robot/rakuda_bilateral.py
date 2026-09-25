"""Manual driving of the Rakuda bilateral loop.

Runs the leader current loop (gravity compensation + joint-limit barrier +
optional follower feedback) from the command line, holds both arms in place
on Ctrl-C / SIGTERM / SIGHUP / any fault, prints the timing and hold summary
and optionally releases the arms at the end.  Stopping never drops the arm:
without ``--release`` the joints stay torque-on in position mode after the
program exits; ``robopy-rakuda-ports release --port <port> --side leader``
(or ``--side follower``) switches them off later.

Typical bring-up sequence (docs/robots/rakuda_bilateral_bringup.md):

* ``robopy-rakuda-gravity range``, then ``sign-check``, then ``identify`` /
  ``fit`` / ``verify`` first, on the hardware: the current sign of every joint
  must be known before any current is commanded.
* ``--no-follower --uncompensated --setup-json .robopy/rakuda/leader_gravity.json``
  (step 7): leader only, gravity term zero, barrier and hold checks.
* ``--setup-json ...`` (step 11): both arms, feedback gains 0.
* ``--feedback-kp 0.2`` ... (step 12): the force-feedback term.

``--sim`` runs the whole program on two simulated buses (constant gravity on
the leader arm joints) so the flow can be exercised without hardware.

Every signal handler here converts SIGTERM/SIGHUP into ``SystemExit``; the
library never touches signals, so a custom script must install the same
handlers to get the ``finally: stop`` behaviour.
"""

from __future__ import annotations

import argparse
import logging
import signal
import sys
import time
from dataclasses import replace
from types import FrameType
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple, cast

import numpy as np
from numpy.typing import NDArray

from robopy.config.dotrobopy import apply_rakuda_dotconfig
from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    PORT_AUTO,
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaBilateralParams,
    RakudaConfig,
)
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelMotor
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, XControlTable
from robopy.motor.sim_dynamixel_bus import SimulatedDynamixelBus, SimulatedJoint
from robopy.robots.rakuda.rakuda_arm import BusFactory
from robopy.robots.rakuda.rakuda_control_laws import BilateralLaw, GravityModel
from robopy.robots.rakuda.rakuda_leader import RakudaLeader
from robopy.robots.rakuda.rakuda_leader_control import LeaderCurrentLoop
from robopy.robots.rakuda.rakuda_pair_sys import (
    BilateralSetup,
    RakudaPairSys,
    load_bilateral_setup,
)
from robopy.robots.rakuda.rakuda_ports import check_sdk_location, resolve_port

logger = logging.getLogger(__name__)

SIM_LEADER_PORT = "sim://leader"
SIM_FOLLOWER_PORT = "sim://follower"
#: Constant holding current of every simulated leader arm joint (a toy plant).
SIM_GRAVITY_MA = 30.0
#: Simulated joints start at 2048 counts (``SimulatedJoint`` default).
SIM_HOME_COUNTS = 2048
#: A stiff "rest" a little below home so a torque-off joint does not fall away
#: while ``connect()`` and ``preflight()`` run (about 1 s under gravity).
SIM_REST_COUNTS = SIM_HOME_COUNTS - 20
#: Safe range of every simulated arm joint (``--range`` when running ``--sim``).
SIM_RANGE_COUNTS = (SIM_HOME_COUNTS - 600, SIM_HOME_COUNTS + 600)

EXIT_OK = 0
EXIT_FAULT = 1
EXIT_USAGE = 2
EXIT_INTERRUPT = 130


# --- signals -------------------------------------------------------------------


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """Turns a termination signal into ``SystemExit(128 + signum)``."""
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    """SIGTERM/SIGHUP raise ``SystemExit`` so the ``finally: stop`` blocks run; SIGINT stays."""
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _exit_on_signal)


# --- simulation --------------------------------------------------------------


class _WallClock:
    """The real monotonic clock in the ``SimClock`` shape the simulated bus expects."""

    monotonic_ns = staticmethod(time.monotonic_ns)
    sleep = staticmethod(time.sleep)


class ConstantGravity:
    """``GravityModel`` of ``--sim``: the plant's constant holding current on ``joints``."""

    def __init__(self, joints: Sequence[str], gravity_ma: float) -> None:
        self._index = [RAKUDA_JOINT_NAMES.index(name) for name in joints]
        self._gravity_ma = float(gravity_ma)

    def predict_ma(self, q_counts: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
        out = np.zeros(len(RAKUDA_JOINT_NAMES), dtype=np.float64)
        out[self._index] = self._gravity_ma
        return out


class SimBusFactory:
    """``bus_factory`` of ``--sim``: one simulated bus per port on the real clock.

    The leader (recognised by its XC330 arm motors) gets ``gravity_ma`` on
    every arm joint and a rest just below home; the follower is a plain
    position-mode plant.  ``RETURN_DELAY_TIME`` is 0 as on the real arms after
    ``robopy-rakuda-ports set-return-delay``.  Created buses are kept in :attr:`buses` by port.
    """

    def __init__(self, gravity_ma: float = SIM_GRAVITY_MA) -> None:
        self.gravity_ma = gravity_ma
        self.buses: Dict[str, SimulatedDynamixelBus] = {}

    def __call__(self, port: str, motors: Dict[str, DynamixelMotor]) -> DynamixelBus:
        is_leader = motors[RAKUDA_ARM_JOINT_NAMES[0]].model_name.startswith("xc330")
        joints: Dict[str, SimulatedJoint] = {}
        if is_leader:
            for name in RAKUDA_ARM_JOINT_NAMES:
                joint = SimulatedJoint()
                joint.gravity_ma = self.gravity_ma
                joint.contact_lower_counts = SIM_REST_COUNTS
                joints[name] = joint
        bus = SimulatedDynamixelBus(
            motors, joints=joints, port=port, clock=_WallClock(), auto_step=True
        )
        for name in motors:
            bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
        self.buses[port] = bus
        return cast(DynamixelBus, bus)


def build_sim_bus_factory() -> SimBusFactory:
    """The factory ``--sim`` hands to the arms."""
    return SimBusFactory()


# --- arguments -----------------------------------------------------------------


#: The ``control_hz`` range ``RakudaBilateralParams`` accepts.
MIN_CONTROL_HZ = 20.0
MAX_CONTROL_HZ = 250.0


def _control_hz(text: str) -> float:
    """``--control-hz`` value, rejected by argparse outside the accepted range."""
    try:
        hz = float(text)
    except ValueError:
        raise argparse.ArgumentTypeError(f"expected a number, got {text!r}") from None
    if not MIN_CONTROL_HZ <= hz <= MAX_CONTROL_HZ:
        raise argparse.ArgumentTypeError(
            f"must be within [{MIN_CONTROL_HZ:g}, {MAX_CONTROL_HZ:g}] Hz, got {text}"
        )
    return hz


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(
        description=(
            "Drive the Rakuda bilateral loop by hand "
            "(docs/robots/rakuda_bilateral_bringup.md, steps 7/11/12)."
        ),
        epilog=(
            "Run `robopy-rakuda-gravity range` and then `sign-check` on the hardware before the "
            "first current-mode run: --sign/--sign-all must match what it measured. "
            "Without --release the arms "
            "keep holding after exit; `robopy-rakuda-ports release --port <port> --side "
            "leader|follower` switches them off."
        ),
    )
    parser.add_argument("--leader-port", default=PORT_AUTO)
    parser.add_argument("--follower-port", default=PORT_AUTO)
    parser.add_argument(
        "--no-follower",
        action="store_true",
        help="leader only: no follower I/O, no position feedback (bring-up step 7)",
    )
    parser.add_argument(
        "--uncompensated",
        action="store_true",
        help="zero gravity term (sets allow_uncompensated); needs the sign and range of every "
        "current joint from --setup-json or --sign/--sign-all + --range",
    )
    parser.add_argument(
        "--setup-json",
        metavar="PATH",
        help="identification file with current_sign, joint_range_counts and the gravity model",
    )
    parser.add_argument(
        "--sign",
        action="append",
        default=[],
        metavar="JOINT=+1|-1",
        help="current sign of one joint (repeatable); overrides --sign-all and --setup-json",
    )
    parser.add_argument(
        "--sign-all", choices=("+1", "-1"), help="one current sign for every joint (dry run)"
    )
    parser.add_argument(
        "--range",
        action="append",
        default=[],
        metavar="JOINT=LO,HI",
        help="safe range of one joint in counts (repeatable); overrides --setup-json",
    )
    parser.add_argument("--feedback-kp", type=float, metavar="MA_PER_COUNT")
    parser.add_argument("--feedback-kd", type=float, metavar="MA_PER_VCOUNT")
    parser.add_argument(
        "--control-hz",
        type=_control_hz,
        metavar="HZ",
        help=f"loop rate within [{MIN_CONTROL_HZ:g}, {MAX_CONTROL_HZ:g}] (default 50)",
    )
    parser.add_argument(
        "--seconds", type=float, metavar="N", help="run for N seconds, then stop (default: Ctrl-C)"
    )
    parser.add_argument(
        "--sim", action="store_true", help="two simulated buses instead of hardware"
    )
    parser.add_argument(
        "--release",
        action="store_true",
        help="ask 'release now? [y/N]' after the stop (default: keep holding)",
    )
    return parser


def _parse_joint_values(items: Sequence[str], what: str) -> Dict[str, str]:
    out: Dict[str, str] = {}
    for item in items:
        joint, sep, value = item.partition("=")
        if not sep or not joint or not value:
            raise ValueError(f"{what} expects JOINT=VALUE, got {item!r}")
        if joint not in RAKUDA_JOINT_NAMES:
            raise ValueError(f"{what}: unknown joint {joint!r}")
        out[joint] = value
    return out


def parse_signs(items: Sequence[str]) -> Dict[str, int]:
    """``["r_arm_sh_roll=+1", ...]`` -> ``{"r_arm_sh_roll": 1, ...}``."""
    signs: Dict[str, int] = {}
    for joint, value in _parse_joint_values(items, "--sign").items():
        if value not in ("+1", "1", "-1"):
            raise ValueError(f"--sign {joint}: expected +1 or -1, got {value!r}")
        signs[joint] = -1 if value == "-1" else 1
    return signs


def parse_ranges(items: Sequence[str]) -> Dict[str, Tuple[int, int]]:
    """``["r_arm_sh_roll=1500,2600", ...]`` -> ``{"r_arm_sh_roll": (1500, 2600), ...}``."""
    ranges: Dict[str, Tuple[int, int]] = {}
    for joint, value in _parse_joint_values(items, "--range").items():
        lo_text, sep, hi_text = value.partition(",")
        try:
            lo, hi = int(lo_text), int(hi_text)
        except ValueError:
            sep = ""
        if not sep or lo >= hi:
            raise ValueError(f"--range {joint}: expected LO,HI with LO < HI, got {value!r}")
        ranges[joint] = (lo, hi)
    return ranges


def build_params(args: argparse.Namespace) -> RakudaBilateralParams:
    """``RakudaBilateralParams`` with only the given command-line overrides."""
    overrides: Dict[str, Any] = {}
    if args.feedback_kp is not None:
        overrides["feedback_kp_ma_per_count"] = args.feedback_kp
    if args.feedback_kd is not None:
        overrides["feedback_kd_ma_per_vcount"] = args.feedback_kd
    if args.control_hz is not None:
        overrides["control_hz"] = args.control_hz
        # runaway_s must span two periods; the default does so only from 50 Hz up.
        # The default read timeouts (8 ms = 2/250 Hz) fit every accepted rate.
        overrides["runaway_s"] = max(RakudaBilateralParams.runaway_s, 2 * (1.0 / args.control_hz))
    if args.uncompensated:
        overrides["allow_uncompensated"] = True
    return RakudaBilateralParams(**overrides)


def build_setup(args: argparse.Namespace, joints: Sequence[str]) -> BilateralSetup:
    """The ``BilateralSetup`` of this run: ``--setup-json`` and/or the manual flags.

    Manual ``--sign``/``--range`` entries override the file; ``--sign-all``
    fills the joints no ``--sign`` names; ``--sim`` fills the rest with the
    simulated plant.  ``--uncompensated`` drops the gravity model.

    Raises:
        ValueError: A joint of ``joints`` without a sign or a range, a
            malformed flag or a malformed ``--setup-json`` file.
        OSError: The ``--setup-json`` file cannot be read (``FileNotFoundError``
            when it does not exist).
        RuntimeError: The identification module cannot be imported.
    """
    signs: Dict[str, int] = {}
    ranges: Dict[str, Tuple[int, int]] = {}
    drive_mode: Dict[str, int] = {}
    gravity: GravityModel | None = None
    gravity_validated = False
    gravity_peak_ma: Mapping[str, float] | None = None
    sources = []
    if args.setup_json is not None:
        loaded = load_bilateral_setup(args.setup_json)
        signs.update(loaded.current_sign)
        ranges.update(loaded.joint_range_counts)
        drive_mode.update(loaded.drive_mode)
        gravity = loaded.gravity
        gravity_validated = loaded.gravity_validated
        gravity_peak_ma = loaded.gravity_peak_ma
        sources.append(loaded.source or str(args.setup_json))
    if args.sim:
        signs.update({name: 1 for name in joints if name not in signs})
        ranges.update({name: SIM_RANGE_COUNTS for name in joints if name not in ranges})
        if gravity is None:
            gravity = ConstantGravity(joints, SIM_GRAVITY_MA)
            gravity_validated = True
            gravity_peak_ma = None
        sources.append("simulated plant")
    if args.sign_all is not None:
        signs.update({name: int(args.sign_all) for name in joints})
        sources.append(f"--sign-all {args.sign_all}")
    manual_signs = parse_signs(args.sign)
    manual_ranges = parse_ranges(args.range)
    signs.update(manual_signs)
    ranges.update(manual_ranges)
    if manual_signs or manual_ranges:
        sources.append("command line")
    if args.uncompensated:
        gravity = None
        gravity_peak_ma = None

    missing_sign = [name for name in joints if name not in signs]
    missing_range = [name for name in joints if name not in ranges]
    if missing_sign or missing_range:
        raise ValueError(
            "every current joint needs a sign and a range (from --setup-json, --sign/--sign-all "
            f"and --range): missing sign {missing_sign}, missing range {missing_range}"
        )
    return BilateralSetup(
        current_sign={name: signs[name] for name in joints},
        joint_range_counts={name: ranges[name] for name in joints},
        drive_mode=drive_mode,
        gravity=gravity,
        gravity_validated=gravity_validated,
        gravity_peak_ma=gravity_peak_ma,
        source=", ".join(sources),
    )


def build_config(args: argparse.Namespace, params: RakudaBilateralParams) -> RakudaConfig:
    leader_port = SIM_LEADER_PORT if args.sim else args.leader_port
    follower_port = SIM_FOLLOWER_PORT if args.sim else args.follower_port
    return RakudaConfig(leader_port=leader_port, follower_port=follower_port, bilateral=params)


# --- reporting -----------------------------------------------------------------


def _ms(block: Mapping[str, Any] | None) -> str:
    if not block:
        return "n/a"
    keys = ("p50", "p95", "p99", "max")
    return " ".join(f"{key} {block[key]:.2f}" for key in keys if block.get(key) is not None)


def print_preflight(preflight: Mapping[str, Any]) -> None:
    """The preflight table of bring-up step 7: drive modes, warnings, read budget."""
    print("preflight:")
    print(f"  drive_mode: {dict(preflight.get('drive_mode', {}))}")
    print(f"  read_ms:    {_ms(preflight.get('read_ms'))}")
    for warning in preflight.get("warnings", ()):
        print(f"  warning: {warning}")
    for note in preflight.get("notes", ()):
        print(f"  note: {note}")


def print_summary(report: Mapping[str, Any]) -> None:
    """Timing and hold summary of a ``control_report()`` (``measured``/``hold``)."""
    measured = report.get("measured") or {}
    hold = report.get("hold")
    print(f"loop: state {report.get('state')}, control_hz {report.get('control_hz')}")
    print(
        f"  cycles {measured.get('cycles')} (ok {measured.get('ok_cycles')}), "
        f"overruns {measured.get('overruns')}, skipped {measured.get('skipped_cycles')}, "
        f"read failures {measured.get('read_fail_count')}"
    )
    print(f"  period_ms: {_ms(measured.get('period_ms'))}")
    print(f"  read_ms:   {_ms(measured.get('read_ms'))}")
    print(f"  write_ms:  {_ms(measured.get('write_ms'))}")
    follower_io = report.get("follower_io") or {}
    if follower_io.get("attached"):
        print(
            f"  follower: {follower_io.get('hz_effective', 0.0):.1f} Hz, "
            f"age_ms {_ms(follower_io.get('age_ms'))}, failures {follower_io.get('failures')}"
        )
    if hold is None:
        print("  hold: none")
    else:
        window = hold.get("window_ms")
        window_text = "n/a" if window is None else f"{window:.1f} ms"
        print(
            f"  hold: verified {hold.get('verified')}, window {window_text}, "
            f"sag {hold.get('sag_counts')}, none {hold.get('none')}"
        )
    for fault in report.get("faults") or ():
        print(f"  fault: {fault.get('reason')}: {fault.get('detail')}")


def _confirm_release() -> bool:
    try:
        answer = input("release now? [y/N] ")
    except EOFError:
        return False
    return answer.strip().lower() in ("y", "yes")


def _wait(seconds: float | None, running: Callable[[], bool]) -> None:
    """Sleeps until ``seconds`` elapsed (``None``: forever) or the loop stopped by itself."""
    deadline = None if seconds is None else time.monotonic() + seconds
    while running():
        remaining = 0.2 if deadline is None else deadline - time.monotonic()
        if remaining <= 0.0:
            return
        time.sleep(min(0.2, remaining))


def _exit_code(report: Mapping[str, Any], held: bool) -> int:
    if report.get("faults") or not held:
        return EXIT_FAULT
    return EXIT_OK


# --- the two runs --------------------------------------------------------------


def _check_setup(setup: BilateralSetup, params: RakudaBilateralParams) -> None:
    """The gravity gates for the leader-only run.

    The pair run has the same gates inside ``start_bilateral()``.
    """
    if setup.gravity is None and not params.allow_uncompensated:
        raise ValueError(
            "no gravity model: pass --uncompensated (allow_uncompensated) for a run without one"
        )
    if setup.gravity is not None and not setup.gravity_validated:
        if not params.allow_unvalidated_gravity:
            raise ValueError(
                "the gravity model is not validated (robopy-rakuda-gravity verify --arm ...); "
                "set bilateral.allow_unvalidated_gravity to use it anyway"
            )
    if setup.drive_mode:
        logger.info("DRIVE_MODE at sign-check: %s", dict(setup.drive_mode))


def _complete_drive_mode(
    setup: BilateralSetup, bus: DynamixelBus, joints: Sequence[str]
) -> BilateralSetup:
    """Fills ``drive_mode`` of the joints a sign-check file did not cover from the bus.

    A sign given by hand (``--sign``/``--sign-all``) is the operator's claim
    for the wiring as it is now, so the ``DRIVE_MODE`` read now is what that
    claim was recorded against.
    """
    missing = [name for name in joints if name not in setup.drive_mode]
    if not missing:
        return setup
    read = bus.sync_read(XControlTable.DRIVE_MODE, missing)
    logger.info("DRIVE_MODE of %s taken from the leader bus (no sign-check file)", missing)
    return replace(
        setup, drive_mode={**setup.drive_mode, **{name: int(read[name]) for name in missing}}
    )


def _check_drive_mode(setup: BilateralSetup, preflight: Mapping[str, Any]) -> None:
    """A joint whose DRIVE_MODE changed since the sign-check may have flipped its sign."""
    found = preflight.get("drive_mode", {})
    changed = {
        name: (mode, found[name])
        for name, mode in setup.drive_mode.items()
        if name in found and found[name] != mode
    }
    if changed:
        raise ConnectionError(
            f"DRIVE_MODE differs from the sign-check (joint: recorded, now): {changed}; "
            "run sign-check again"
        )


def run_leader_only(
    config: RakudaConfig,
    setup: BilateralSetup,
    *,
    seconds: float | None,
    release: bool,
    bus_factory: BusFactory | None,
) -> int:
    """Leader arm alone: ``RakudaLeader`` + ``LeaderCurrentLoop`` without follower I/O."""
    config = apply_rakuda_dotconfig(config)
    params = config.bilateral
    assert params is not None
    _check_setup(setup, params)
    if config.leader_port == PORT_AUTO:
        config = replace(config, leader_port=resolve_port(PORT_AUTO, "leader"))
    if bus_factory is None:
        check_sdk_location()

    joints = params.current_joints
    leader = RakudaLeader(config, bus_factory)
    leader.connect()
    try:
        bus = leader.motors
        setup = _complete_drive_mode(setup, bus, joints)
        law = BilateralLaw(
            joints,
            params,
            setup.gravity,
            setup.joint_range_counts,
            gravity_peak_ma=setup.gravity_peak_ma,
        )
        loop = LeaderCurrentLoop(
            bus,
            joints,
            params,
            law,
            {name: CURRENT_UNIT_MA[bus.motors[name].model_name] for name in joints},
            setup.current_sign,
            None,
            gripper_hold={name: LEADER_GRIP_HOLD_POSITION for name in RAKUDA_GRIPPER_JOINT_NAMES},
        )
        preflight = loop.preflight()
        print_preflight(preflight)
        _check_drive_mode(setup, preflight)
        loop.configure()
        try:
            loop.start()
            logger.info("leader loop running (setup: %s); Ctrl-C holds in place", setup.source)
            _wait(seconds, lambda: loop.running)
        finally:
            held = loop.stop()
            if not held:
                logger.critical("stop() could not verify the hold: %s", loop.faults)
            report = loop.control_report()
            print_summary(report)
            if release and _confirm_release():
                loop.release()
        return _exit_code(report, held)
    finally:
        leader.disconnect(torque_off=False)


def run_pair(
    config: RakudaConfig,
    setup: BilateralSetup,
    *,
    seconds: float | None,
    release: bool,
    bus_factory: BusFactory | None,
) -> int:
    """Both arms through ``RakudaPairSys.start_bilateral()``."""
    pair = RakudaPairSys(config, bus_factory=bus_factory)
    pair.connect()
    try:
        params = pair.config.bilateral
        assert params is not None
        setup = _complete_drive_mode(setup, pair.leader.motors, params.current_joints)
        try:
            loop = pair.start_bilateral(setup=setup)
            print_preflight(loop.control_report().get("preflight") or {})
            logger.info("bilateral loop running (setup: %s); Ctrl-C holds both arms", setup.source)
            _wait(seconds, lambda: pair.bilateral_active)
        finally:
            held = pair.stop_bilateral()
            report = pair.control_report() or {}
            print_summary(report)
            if release and _confirm_release():
                pair.release()
        return _exit_code(report, held)
    finally:
        pair.disconnect()


def main(argv: Sequence[str] | None = None) -> int:
    """Entry point; returns the exit code (0 ok, 1 fault/unverified hold, 130 Ctrl-C)."""
    install_signal_handlers()
    logging.basicConfig(
        level=logging.INFO, stream=sys.stdout, format="%(asctime)s %(levelname)s %(message)s"
    )
    parser = build_parser()
    args = parser.parse_args(argv)
    params = build_params(args)
    try:
        setup = build_setup(args, params.current_joints)
    except (ValueError, RuntimeError, OSError) as exc:
        parser.error(str(exc))
    config = build_config(args, params)
    bus_factory: BusFactory | None = build_sim_bus_factory() if args.sim else None
    run = run_leader_only if args.no_follower else run_pair
    try:
        return run(
            config, setup, seconds=args.seconds, release=args.release, bus_factory=bus_factory
        )
    except KeyboardInterrupt:
        logger.info("stopped by operator (Ctrl-C); the arms are holding")
        return EXIT_INTERRUPT


if __name__ == "__main__":
    sys.exit(main())
