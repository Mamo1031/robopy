"""The two bilateral examples: argument parsing and the ``--sim`` runs.

``examples/robot/rakuda_bilateral.py`` is exercised end to end on its own
simulated buses (``--sim``) by calling ``main([...])`` in-process; every run
must end with the current joints held (position mode, torque on).
``examples/exp_handler/record.py`` is checked for the configuration and the
signal handlers its ``--bilateral`` flag builds, with the handler mocked.
"""

from __future__ import annotations

import importlib.util
import signal
import sys
from pathlib import Path
from types import ModuleType
from typing import Any, Dict, Iterator, List, Sequence

import numpy as np
import pytest

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaBilateralParams,
    RakudaConfig,
)
from robopy.motor.dynamixel_control_table import OperatingMode
from robopy.motor.sim_dynamixel_bus import SimulatedDynamixelBus
from robopy.robots.rakuda.rakuda_pair_sys import BilateralSetup

EXAMPLES = Path(__file__).resolve().parents[2] / "examples"
ARM = RAKUDA_ARM_JOINT_NAMES
SIGNALS = (signal.SIGTERM, signal.SIGHUP)


def _load_example(name: str, relative: str) -> ModuleType:
    """Imports an example script (not a package) under ``name``."""
    spec = importlib.util.spec_from_file_location(name, EXAMPLES / relative)
    assert spec is not None and spec.loader is not None
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


@pytest.fixture(scope="module")
def bilateral_mod() -> ModuleType:
    return _load_example("example_rakuda_bilateral", "robot/rakuda_bilateral.py")


@pytest.fixture(scope="module")
def record_mod() -> ModuleType:
    return _load_example("example_record", "exp_handler/record.py")


@pytest.fixture
def restore_signals() -> Iterator[None]:
    """Puts the process's SIGTERM/SIGHUP handlers back after a ``main()`` call."""
    saved = {signum: signal.getsignal(signum) for signum in SIGNALS}
    yield
    for signum, handler in saved.items():
        signal.signal(signum, handler)  # type: ignore[arg-type]


@pytest.fixture
def sim_buses(
    bilateral_mod: ModuleType, monkeypatch: pytest.MonkeyPatch, tmp_path: Path
) -> Dict[str, SimulatedDynamixelBus]:
    """Captures the buses a ``--sim`` run creates; the dotconfig YAML lands in ``tmp_path``."""
    monkeypatch.chdir(tmp_path)
    factory = bilateral_mod.SimBusFactory()
    monkeypatch.setattr(bilateral_mod, "build_sim_bus_factory", lambda: factory)
    return factory.buses


def assert_held(bus: SimulatedDynamixelBus, names: Sequence[str]) -> None:
    """Every joint in position mode, torque on, goal = present."""
    for name in names:
        regs = bus.registers(name)
        assert regs.operating_mode == OperatingMode.POSITION, name
        assert regs.torque_enable == 1, name
        assert regs.goal_position == regs.present_position, name


# --- rakuda_bilateral.py: arguments ------------------------------------------------


class TestBilateralParser:
    def test_defaults(self, bilateral_mod: ModuleType) -> None:
        args = bilateral_mod.build_parser().parse_args([])
        assert (args.leader_port, args.follower_port) == ("auto", "auto")
        assert args.no_follower is False and args.uncompensated is False
        assert args.setup_json is None and args.sign == [] and args.range == []
        assert args.sign_all is None
        assert args.feedback_kp is None and args.feedback_kd is None
        assert args.control_hz is None and args.seconds is None
        assert args.sim is False and args.release is False

    def test_params_carry_only_the_given_overrides(self, bilateral_mod: ModuleType) -> None:
        parser = bilateral_mod.build_parser()
        assert bilateral_mod.build_params(parser.parse_args([])) == RakudaBilateralParams()
        params = bilateral_mod.build_params(
            parser.parse_args(
                ["--feedback-kp", "0.2", "--feedback-kd", "0.1", "--control-hz", "100"]
            )
        )
        assert params == RakudaBilateralParams(
            feedback_kp_ma_per_count=0.2, feedback_kd_ma_per_vcount=0.1, control_hz=100.0
        )
        assert bilateral_mod.build_params(parser.parse_args(["--uncompensated"])) == (
            RakudaBilateralParams(allow_uncompensated=True)
        )

    @pytest.mark.parametrize(
        ("hz", "runaway_s"), [("20", 0.1), ("25", 0.08), ("50", 0.04), ("250", 0.04)]
    )
    def test_every_accepted_control_hz_builds(
        self, bilateral_mod: ModuleType, hz: str, runaway_s: float
    ) -> None:
        args = bilateral_mod.build_parser().parse_args(["--control-hz", hz])
        params = bilateral_mod.build_params(args)
        assert params.control_hz == float(hz)
        assert params.runaway_s == pytest.approx(runaway_s)
        assert params.read_timeout_s == params.follower_read_timeout_s == 0.008

    @pytest.mark.parametrize("hz", ["19.9", "251", "nan", "fast"])
    def test_control_hz_outside_the_range_is_an_argparse_error(
        self, bilateral_mod: ModuleType, hz: str, capsys: pytest.CaptureFixture[str]
    ) -> None:
        with pytest.raises(SystemExit) as info:
            bilateral_mod.build_parser().parse_args(["--control-hz", hz])
        assert info.value.code == 2
        assert "--control-hz" in capsys.readouterr().err

    def test_sign_and_range_flags(self, bilateral_mod: ModuleType) -> None:
        assert bilateral_mod.parse_signs(["r_arm_sh_roll=+1", "l_arm_el_yaw=-1"]) == {
            "r_arm_sh_roll": 1,
            "l_arm_el_yaw": -1,
        }
        assert bilateral_mod.parse_ranges(["r_arm_sh_roll=1500,2600"]) == {
            "r_arm_sh_roll": (1500, 2600)
        }
        for bad in (["r_arm_sh_roll"], ["nope=+1"], ["r_arm_sh_roll=2"]):
            with pytest.raises(ValueError):
                bilateral_mod.parse_signs(bad)
        for bad in (["r_arm_sh_roll=2600,1500"], ["r_arm_sh_roll=1500"], ["r_arm_sh_roll=a,b"]):
            with pytest.raises(ValueError):
                bilateral_mod.parse_ranges(bad)

    def test_manual_setup_needs_every_joint(self, bilateral_mod: ModuleType) -> None:
        parser = bilateral_mod.build_parser()
        with pytest.raises(ValueError, match="missing sign"):
            bilateral_mod.build_setup(parser.parse_args(["--uncompensated"]), ARM)
        ranges = [f"--range={name}=1000,3000" for name in ARM]
        args = parser.parse_args(["--uncompensated", "--sign-all", "-1", *ranges])
        setup = bilateral_mod.build_setup(args, ARM)
        assert setup.gravity is None and setup.gravity_validated is False
        assert setup.current_sign == {name: -1 for name in ARM}
        assert setup.joint_range_counts == {name: (1000, 3000) for name in ARM}
        assert setup.drive_mode == {}

    def test_setup_json_is_loaded_and_overridden_by_flags(
        self, bilateral_mod: ModuleType, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        class Gravity:
            def predict_ma(self, q_counts: Any) -> Any:
                return np.zeros(len(RAKUDA_JOINT_NAMES))

        gravity = Gravity()
        loaded = BilateralSetup(
            current_sign={name: 1 for name in ARM},
            joint_range_counts={name: (1000, 3000) for name in ARM},
            drive_mode={name: 0 for name in ARM},
            gravity=gravity,
            gravity_validated=True,
            gravity_peak_ma={name: 100.0 for name in ARM},
            source="file.json",
        )
        paths: List[str] = []

        def fake_load(path: str) -> Any:
            paths.append(path)
            return loaded

        monkeypatch.setattr(bilateral_mod, "load_bilateral_setup", fake_load)
        parser = bilateral_mod.build_parser()
        setup = bilateral_mod.build_setup(
            parser.parse_args(["--setup-json", "f.json", "--sign", "r_arm_sh_roll=-1"]), ARM
        )
        assert paths == ["f.json"]
        assert setup.gravity is gravity and setup.gravity_validated is True
        assert setup.current_sign["r_arm_sh_roll"] == -1
        assert setup.current_sign["l_arm_sh_roll"] == 1
        assert setup.drive_mode == loaded.drive_mode
        assert setup.gravity_peak_ma == loaded.gravity_peak_ma

        setup = bilateral_mod.build_setup(
            parser.parse_args(["--setup-json", "f.json", "--uncompensated"]), ARM
        )
        assert setup.gravity is None and setup.gravity_peak_ma is None

    def test_setup_error_is_a_usage_error(
        self,
        bilateral_mod: ModuleType,
        restore_signals: None,
        capsys: pytest.CaptureFixture[str],
        tmp_path: Path,
    ) -> None:
        # The real loader (rakuda_gravity): a missing and a malformed identification file.
        with pytest.raises(SystemExit) as info:
            bilateral_mod.main(["--setup-json", str(tmp_path / "missing.json")])
        assert info.value.code == bilateral_mod.EXIT_USAGE
        assert "No such file" in capsys.readouterr().err
        (tmp_path / "bad.json").write_text('{"schema_version": 99}')
        with pytest.raises(SystemExit) as info:
            bilateral_mod.main(["--setup-json", str(tmp_path / "bad.json")])
        assert info.value.code == bilateral_mod.EXIT_USAGE
        assert "unknown schema_version 99" in capsys.readouterr().err
        with pytest.raises(SystemExit) as info:
            bilateral_mod.main(["--uncompensated"])
        assert info.value.code == bilateral_mod.EXIT_USAGE
        assert "missing sign" in capsys.readouterr().err

    def test_signal_handlers_convert_to_system_exit(
        self, bilateral_mod: ModuleType, restore_signals: None
    ) -> None:
        bilateral_mod.install_signal_handlers()
        for signum in SIGNALS:
            handler = signal.getsignal(signum)
            assert callable(handler)
            with pytest.raises(SystemExit) as info:
                handler(signum, None)
            assert info.value.code == 128 + signum
        assert signal.getsignal(signal.SIGINT) is signal.default_int_handler


# --- rakuda_bilateral.py: --sim runs --------------------------------------------


class TestSimRuns:
    def test_leader_only_run_ends_held(
        self,
        bilateral_mod: ModuleType,
        sim_buses: Dict[str, SimulatedDynamixelBus],
        restore_signals: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = bilateral_mod.main(["--sim", "--no-follower", "--seconds", "1"])
        assert code == bilateral_mod.EXIT_OK
        assert list(sim_buses) == [bilateral_mod.SIM_LEADER_PORT]
        bus = sim_buses[bilateral_mod.SIM_LEADER_PORT]
        assert_held(bus, ARM)
        for name in RAKUDA_GRIPPER_JOINT_NAMES:
            assert bus.registers(name).torque_enable == 1
        # Compensated: the arm stayed within a few counts of where it rested.
        for name in ARM:
            assert abs(bus.joint(name).position_counts - bilateral_mod.SIM_HOME_COUNTS) < 60
        out = capsys.readouterr().out
        assert "preflight:" in out
        assert "loop: state held" in out
        assert "hold: verified True" in out

    def test_leader_only_uncompensated_run_ends_held(
        self,
        bilateral_mod: ModuleType,
        sim_buses: Dict[str, SimulatedDynamixelBus],
        restore_signals: None,
    ) -> None:
        code = bilateral_mod.main(
            ["--sim", "--no-follower", "--uncompensated", "--sign-all", "+1", "--seconds", "0.5"]
        )
        assert code == bilateral_mod.EXIT_OK
        assert_held(sim_buses[bilateral_mod.SIM_LEADER_PORT], ARM)

    @pytest.mark.parametrize("answer, torque", [("y", 0), ("n", 1)])
    def test_release_asks_then_releases(
        self,
        bilateral_mod: ModuleType,
        sim_buses: Dict[str, SimulatedDynamixelBus],
        restore_signals: None,
        monkeypatch: pytest.MonkeyPatch,
        answer: str,
        torque: int,
    ) -> None:
        prompts: List[str] = []

        def fake_input(prompt: str) -> str:
            prompts.append(prompt)
            return answer

        monkeypatch.setattr("builtins.input", fake_input)
        code = bilateral_mod.main(["--sim", "--no-follower", "--seconds", "0.5", "--release"])
        assert code == bilateral_mod.EXIT_OK
        assert prompts == ["release now? [y/N] "]
        bus = sim_buses[bilateral_mod.SIM_LEADER_PORT]
        for name in ARM:
            assert bus.registers(name).torque_enable == torque
            assert bus.registers(name).operating_mode == OperatingMode.POSITION

    def test_no_release_flag_never_asks(
        self,
        bilateral_mod: ModuleType,
        sim_buses: Dict[str, SimulatedDynamixelBus],
        restore_signals: None,
        monkeypatch: pytest.MonkeyPatch,
    ) -> None:
        def fail(prompt: str) -> str:
            raise AssertionError("must not prompt")

        monkeypatch.setattr("builtins.input", fail)
        assert bilateral_mod.main(["--sim", "--no-follower", "--seconds", "0.5"]) == 0

    def test_pair_run_ends_held_on_both_buses(
        self,
        bilateral_mod: ModuleType,
        sim_buses: Dict[str, SimulatedDynamixelBus],
        restore_signals: None,
        capsys: pytest.CaptureFixture[str],
    ) -> None:
        code = bilateral_mod.main(["--sim", "--seconds", "1"])
        assert code == bilateral_mod.EXIT_OK
        assert set(sim_buses) == {bilateral_mod.SIM_LEADER_PORT, bilateral_mod.SIM_FOLLOWER_PORT}
        assert_held(sim_buses[bilateral_mod.SIM_LEADER_PORT], ARM)
        follower = sim_buses[bilateral_mod.SIM_FOLLOWER_PORT]
        for name in RAKUDA_JOINT_NAMES:
            assert follower.registers(name).torque_enable == 1
        out = capsys.readouterr().out
        assert "loop: state held" in out


# --- record.py ------------------------------------------------------------------


class FakeHandler:
    """Stands in for ``RakudaExpHandler``: records the constructor and the calls."""

    instances: List["FakeHandler"] = []
    fail_record = False

    def __init__(self, rakuda_config: RakudaConfig, metadata_config: Any, fps: int) -> None:
        self.config = rakuda_config
        self.fps = fps
        self.calls: List[Any] = []
        FakeHandler.instances.append(self)

    def record_save(self, max_frames: int, save_path: str) -> None:
        self.calls.append(("record_save", max_frames, save_path))
        if FakeHandler.fail_record:
            raise RuntimeError("boom")

    def close(self) -> None:
        self.calls.append(("close",))


@pytest.fixture
def fake_handler(record_mod: ModuleType, monkeypatch: pytest.MonkeyPatch) -> type[FakeHandler]:
    monkeypatch.setattr(record_mod, "RakudaExpHandler", FakeHandler)
    FakeHandler.instances = []
    FakeHandler.fail_record = False
    return FakeHandler


class TestRecord:
    def test_parser_defaults(self, record_mod: ModuleType) -> None:
        args = record_mod.build_parser().parse_args([])
        assert (args.leader_port, args.follower_port) == ("/dev/ttyUSB1", "/dev/ttyUSB0")
        assert args.bilateral is False
        assert (args.max_frames, args.save_path) == (100, "test_01")

    def test_conventional_run_is_unchanged(
        self, record_mod: ModuleType, fake_handler: type[FakeHandler], restore_signals: None
    ) -> None:
        record_mod.main([])
        (handler,) = fake_handler.instances
        assert handler.config.bilateral is None
        assert handler.config.effective_hold_on_disconnect is False
        assert handler.config.leader_port == "/dev/ttyUSB1"
        assert handler.fps == 10
        assert handler.calls == [("record_save", 100, "test_01"), ("close",)]
        assert signal.getsignal(signal.SIGTERM) is not record_mod._exit_on_signal

    def test_bilateral_builds_the_config_and_installs_the_handlers(
        self, record_mod: ModuleType, fake_handler: type[FakeHandler], restore_signals: None
    ) -> None:
        record_mod.main(["--bilateral", "--max-frames", "7", "--save-path", "ep"])
        (handler,) = fake_handler.instances
        assert handler.config.bilateral == RakudaBilateralParams()
        assert handler.config.hold_on_disconnect is None
        assert handler.config.effective_hold_on_disconnect is True
        assert handler.calls == [("record_save", 7, "ep"), ("close",)]
        for signum in SIGNALS:
            with pytest.raises(SystemExit) as info:
                record_mod._exit_on_signal(signum, None)
            assert info.value.code == 128 + signum
            assert signal.getsignal(signum) is record_mod._exit_on_signal

    def test_close_runs_when_record_save_raises(
        self, record_mod: ModuleType, fake_handler: type[FakeHandler], restore_signals: None
    ) -> None:
        fake_handler.fail_record = True
        with pytest.raises(RuntimeError, match="boom"):
            record_mod.main(["--bilateral"])
        (handler,) = fake_handler.instances
        assert handler.calls[-1] == ("close",)
