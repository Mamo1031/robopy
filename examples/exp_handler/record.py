"""Record one Rakuda episode with ``RakudaExpHandler`` (conventional or ``--bilateral``).

``--bilateral`` enables the leader current loop: the handler starts it
before the first frame and every stop, fault, Ctrl-C, SIGTERM or SIGHUP
leaves both arms holding in place.  Without the
flag the recording is the conventional position teleoperation, unchanged.
"""

import argparse
import signal
from logging import INFO, getLogger
from types import FrameType
from typing import Sequence

from robopy.config import RakudaConfig
from robopy.config.robot_config.rakuda_config import RakudaBilateralParams
from robopy.utils import MetaDataConfig, RakudaExpHandler

logger = getLogger(__name__)
logger.setLevel(INFO)


def exp_handler_import():
    from robopy.utils.exp_interface.exp_handler import ExpHandler
    from robopy.utils.exp_interface.rakuda_exp_handler import RakudaExpHandler

    assert ExpHandler is not None
    assert RakudaExpHandler is not None


def rakuda_exp_send():
    handler = RakudaExpHandler(
        rakuda_config=RakudaConfig(
            leader_port="/dev/ttyUSB1",
            follower_port="/dev/ttyUSB0",
        ),
        fps=10,
        metadata_config=MetaDataConfig(
            task_name="test_task",
            description="This is a test task",
            date="2024-06-01",
        ),
    )

    try:
        action = handler.record(max_frames=200).arms.leader
        handler.send(max_frame=200, fps=10, leader_action=action)
    finally:
        handler.close()


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """Turns a termination signal into ``SystemExit(128 + signum)``."""
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    """SIGTERM/SIGHUP raise ``SystemExit`` so ``finally: handler.close()`` runs; SIGINT stays."""
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _exit_on_signal)


def build_parser() -> argparse.ArgumentParser:
    parser = argparse.ArgumentParser(description="Record one Rakuda episode.")
    parser.add_argument("--leader-port", default="/dev/ttyUSB1")
    parser.add_argument("--follower-port", default="/dev/ttyUSB0")
    parser.add_argument(
        "--bilateral",
        action="store_true",
        help="leader current control (gravity compensation + force feedback); the arms keep "
        "holding after the program ends",
    )
    parser.add_argument("--max-frames", type=int, default=100)
    parser.add_argument("--save-path", default="test_01")
    return parser


def build_config(args: argparse.Namespace) -> RakudaConfig:
    """The robot configuration of this run; ``--bilateral`` adds the default loop parameters."""
    return RakudaConfig(
        leader_port=args.leader_port,
        follower_port=args.follower_port,
        # sensors=RakudaSensorParams(
        #    tactile=[
        #        TactileParams(serial_num="D20542", name="left"),
        #        TactileParams(serial_num="D20537", name="right"),
        #    ],
        # ),
        bilateral=RakudaBilateralParams() if args.bilateral else None,
        hold_on_disconnect=None,
    )


def main(argv: Sequence[str] | None = None) -> None:
    args = build_parser().parse_args(argv)
    if args.bilateral:
        install_signal_handlers()
    handler = RakudaExpHandler(
        rakuda_config=build_config(args),
        metadata_config=MetaDataConfig(
            task_name="test_task",
            description="This is a test task of no tactile sensors",
            date="2024-06-01",
        ),
        fps=10,
    )
    try:
        handler.record_save(max_frames=args.max_frames, save_path=args.save_path)
    finally:
        handler.close()


if __name__ == "__main__":
    main()
