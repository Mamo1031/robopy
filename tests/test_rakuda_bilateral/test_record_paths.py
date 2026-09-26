"""The three ``RakudaRobot`` record paths on a mocked robot system.

``RakudaRobot`` is built with ``object.__new__`` and a ``MagicMock`` robot
system, the way ``tests/test_rakuda_send_action.py`` does, so no port, camera
or worker thread of the real robot is involved.  ``time.monotonic_ns`` of the
robot module is replaced by a counter that advances 1 ms per call, which makes
the ``t0`` / read-stamp / frame-stamp ordering exact; the other clocks stay real
because ``record_parallel``/``record_with_fixed_leader`` run a worker thread.
"""

from __future__ import annotations

import json
import threading
import time
from collections import OrderedDict
from pathlib import Path
from types import SimpleNamespace
from typing import Any, Dict, List
from unittest.mock import MagicMock

import numpy as np
import pytest

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_JOINT_NAMES,
    RakudaArmObs,
    RakudaArmState,
    RakudaBilateralParams,
    RakudaConfig,
    RakudaObs,
    RakudaSensorObs,
)
from robopy.config.sensor_config import Sensors
from robopy.motor.dynamixel_bus import DynamixelTimeoutError
from robopy.robots.rakuda import rakuda_robot
from robopy.robots.rakuda.rakuda_robot import RakudaRobot
from robopy.utils.exp_interface.meta_data_config import MetaDataConfig
from robopy.utils.exp_interface.rakuda_exp_handler import RakudaExpHandler
from robopy.utils.worker.rakuda_save_worker import RakudaSaveWorker

N_JOINTS = len(RAKUDA_JOINT_NAMES)
MS = 1_000_000


class StepClock:
    """``monotonic_ns`` that advances 1 ms per call (thread-safe); ``monotonic`` only reads it."""

    def __init__(self, start_ns: int = 5_000_000_000) -> None:
        self.start_ns = start_ns
        self.calls = 0
        self._lock = threading.Lock()

    def monotonic_ns(self) -> int:
        with self._lock:
            self.calls += 1
            return self.start_ns + self.calls * MS

    def monotonic(self) -> float:
        with self._lock:
            return (self.start_ns + self.calls * MS) / 1e9


class FakePairSys:
    """A ``teleoperate_step``/``get_follower_state`` source with read stamps from ``clock``.

    ``fail_at`` raises ``error`` on that (1-based) call.
    """

    def __init__(
        self, clock: StepClock, *, fail_at: int | None = None, error: BaseException | None = None
    ) -> None:
        self.clock = clock
        self.fail_at = fail_at
        self.error = error or RuntimeError("bus died")
        self.calls = 0
        self.stamps: List[int] = []

    def _next(self) -> int:
        self.calls += 1
        if self.fail_at is not None and self.calls == self.fail_at:
            raise self.error
        stamp = self.clock.monotonic_ns()
        self.stamps.append(stamp)
        return stamp

    def teleoperate_step(self) -> RakudaArmObs:
        leader_t_ns = self._next()
        follower_t_ns = self.clock.monotonic_ns()
        seq = float(self.calls)
        return RakudaArmObs(
            leader=np.full(N_JOINTS, seq, np.float32),
            follower=np.full(N_JOINTS, seq + 0.5, np.float32),
            leader_velocity=np.zeros(N_JOINTS, np.float32),
            follower_velocity=np.zeros(N_JOINTS, np.float32),
            leader_current=np.full(N_JOINTS, -seq, np.float32),
            follower_current=np.full(N_JOINTS, seq, np.float32),
            leader_t_ns=leader_t_ns,
            follower_t_ns=follower_t_ns,
        )

    def get_follower_state(self) -> RakudaArmState:
        t_end_ns = self._next()
        return RakudaArmState(
            names=RAKUDA_JOINT_NAMES,
            position=np.full(N_JOINTS, self.calls, np.int32),
            velocity=np.full(N_JOINTS, -self.calls, np.int32),
            current_ma=np.full(N_JOINTS, 2.69 * self.calls, np.float32),
            t_start_ns=t_end_ns - MS,
            t_end_ns=t_end_ns,
            seq=self.calls,
        )


def make_robot(fake: FakePairSys) -> tuple[RakudaRobot, MagicMock]:
    robot = object.__new__(RakudaRobot)
    pair_sys = MagicMock()
    pair_sys.is_connected = True
    pair_sys.teleoperate_step.side_effect = fake.teleoperate_step
    pair_sys.get_follower_state.side_effect = fake.get_follower_state
    pair_sys.leader.motors.motors = OrderedDict((name, object()) for name in RAKUDA_JOINT_NAMES)
    pair_sys.leader.motor_names = list(RAKUDA_JOINT_NAMES)
    pair_sys.leader.motor_models = ["xc330-t288"] * N_JOINTS
    pair_sys.follower.motor_models = ["xm430-w350"] * N_JOINTS
    pair_sys.leader.port = "/dev/ttyUSB1"
    pair_sys.follower.port = "/dev/ttyUSB0"
    # A conventional pair system: no bilateral loop.
    pair_sys.bilateral_active = False
    pair_sys.control_report.return_value = None
    robot._pair_sys = pair_sys
    robot._sensors = Sensors(cameras=[], tactile=[], audio=[])
    robot._last_record_t0_ns = None
    robot._last_record_stats = {}
    robot._last_record_summary = {}
    return robot, pair_sys


@pytest.fixture
def clock(monkeypatch: pytest.MonkeyPatch) -> StepClock:
    """Installs the stepping ``monotonic_ns`` into the robot module's ``time``."""
    clock = StepClock()
    fake_time = SimpleNamespace(
        monotonic_ns=clock.monotonic_ns,
        monotonic=clock.monotonic,
        perf_counter=time.perf_counter,
        sleep=time.sleep,
        time=time.time,
    )
    monkeypatch.setattr(rakuda_robot, "time", fake_time)
    return clock


def assert_stamped_after_t0(obs: RakudaObs, robot: RakudaRobot, clock: StepClock) -> None:
    """``t0`` was taken once, before every read; frames are stamped at commit."""
    arms = obs.arms
    summary = robot._last_record_summary
    frames = arms.leader.shape[0]
    assert robot._last_record_t0_ns == summary["t0_monotonic_ns"]
    assert summary["t0_monotonic_ns"] == clock.start_ns + MS  # the first monotonic_ns call
    assert summary["frames"] == frames
    assert isinstance(summary["t0_unix_s"], float)
    assert arms.frame_time_s is not None and arms.follower_time_s is not None
    assert arms.frame_time_s.shape == (frames,) and arms.frame_time_s.dtype == np.float32
    assert np.all(arms.frame_time_s > 0)
    assert np.all(np.diff(arms.frame_time_s) >= 0)
    assert np.all(arms.follower_time_s > 0)
    assert np.all(arms.follower_time_s < arms.frame_time_s)
    if arms.leader_time_s is not None:
        assert np.all(arms.leader_time_s > 0)
        assert np.all(arms.leader_time_s < arms.follower_time_s)
    assert arms.leader_t_ns is None and arms.follower_t_ns is None


# --- record() ------------------------------------------------------------------


class TestRecord:
    def test_stamps_frames_and_completes(self, clock: StepClock) -> None:
        fake = FakePairSys(clock)
        robot, _ = make_robot(fake)

        obs = robot.record(max_frame=3, fps=1000)

        assert obs.arms.leader.shape == (3, N_JOINTS)
        assert obs.arms.leader_current is not None
        assert_stamped_after_t0(obs, robot, clock)
        summary = robot._last_record_summary
        assert summary["terminated_by"] == "max_frame"
        assert summary["frames_requested"] == 3
        assert summary["worker_error"] is None
        assert summary["teleop_hz_effective"] > 0
        assert robot._last_record_stats == {
            "frames": 3,
            "queue_empty_waits": 0,
            "over_budget_frames": 0,
            "duplicate_snapshots": 0,
        }

    def test_keyboard_interrupt_returns_partial(self, clock: StepClock) -> None:
        fake = FakePairSys(clock, fail_at=3, error=KeyboardInterrupt())
        robot, _ = make_robot(fake)

        obs = robot.record(max_frame=10, fps=1000)

        assert obs.arms.leader.shape[0] == 2
        assert robot._last_record_summary["terminated_by"] == "keyboard_interrupt"
        assert robot._last_record_summary["frames"] == 2
        assert robot._last_record_summary["frames_requested"] == 10
        assert_stamped_after_t0(obs, robot, clock)

    def test_keyboard_interrupt_with_zero_frames_raises(self, clock: StepClock) -> None:
        fake = FakePairSys(clock, fail_at=1, error=KeyboardInterrupt())
        robot, _ = make_robot(fake)

        with pytest.raises(RuntimeError, match="0 frames"):
            robot.record(max_frame=10, fps=1000)

        assert robot._last_record_summary["terminated_by"] == "keyboard_interrupt"
        assert robot._last_record_summary["frames"] == 0

    def test_other_exceptions_propagate(self, clock: StepClock) -> None:
        fake = FakePairSys(clock, fail_at=2, error=ConnectionError("gone"))
        robot, _ = make_robot(fake)

        with pytest.raises(ConnectionError, match="gone"):
            robot.record(max_frame=10, fps=1000)


# --- record_parallel() ---------------------------------------------------------


class TestRecordParallel:
    def test_stamps_frames_and_completes(self, clock: StepClock) -> None:
        fake = FakePairSys(clock)
        robot, _ = make_robot(fake)

        obs = robot.record_parallel(max_frame=3, fps=100, teleop_hz=200)

        assert obs.arms.leader.shape == (3, N_JOINTS)
        assert_stamped_after_t0(obs, robot, clock)
        summary = robot._last_record_summary
        assert summary["terminated_by"] == "max_frame"
        assert summary["frames_requested"] == 3
        assert summary["worker_error"] is None
        stats = robot._last_record_stats
        assert stats["frames"] == 3 and stats["over_budget_frames"] == 0
        assert set(stats) == {
            "frames",
            "queue_empty_waits",
            "over_budget_frames",
            "duplicate_snapshots",
        }

    def test_worker_error_returns_partial_and_stops(
        self, clock: StepClock, caplog: pytest.LogCaptureFixture
    ) -> None:
        """A worker exception ends the recording with the frames collected so far."""
        fake = FakePairSys(clock, fail_at=3)
        robot, _ = make_robot(fake)

        started = time.monotonic()
        obs = robot.record_parallel(max_frame=100, fps=100, teleop_hz=200)

        assert time.monotonic() - started < 2.0  # the sampler did not wait for 100 frames
        frames = obs.arms.leader.shape[0]
        assert 1 <= frames <= 2
        summary = robot._last_record_summary
        assert summary["terminated_by"] == "teleop_stopped"
        assert summary["frames"] == frames
        assert "bus died" in summary["worker_error"]
        assert any("bus died" in record.getMessage() for record in caplog.records)
        assert_stamped_after_t0(obs, robot, clock)

    def test_worker_error_before_any_frame_raises(self, clock: StepClock) -> None:
        fake = FakePairSys(clock, fail_at=1)
        robot, _ = make_robot(fake)

        with pytest.raises(RuntimeError, match="teleop_stopped.*0 frames.*bus died"):
            robot.record_parallel(max_frame=100, fps=100, teleop_hz=200)

        assert robot._last_record_summary["frames"] == 0
        assert robot._last_record_summary["terminated_by"] == "teleop_stopped"

    def test_keyboard_interrupt_in_main_thread_returns_partial(
        self, clock: StepClock, monkeypatch: pytest.MonkeyPatch
    ) -> None:
        fake = FakePairSys(clock)
        robot, _ = make_robot(fake)
        main_sleeps = 0

        def sleep(seconds: float) -> None:
            nonlocal main_sleeps
            if threading.current_thread() is threading.main_thread():
                main_sleeps += 1
                if main_sleeps == 2:
                    raise KeyboardInterrupt
            time.sleep(seconds)

        monkeypatch.setattr(rakuda_robot.time, "sleep", sleep)

        obs = robot.record_parallel(max_frame=100, fps=100, teleop_hz=200)

        assert obs.arms.leader.shape[0] == 2
        assert robot._last_record_summary["terminated_by"] == "keyboard_interrupt"
        assert_stamped_after_t0(obs, robot, clock)

    def test_duplicate_snapshots_counted_by_follower_time(self, clock: StepClock) -> None:
        """The same follower snapshot committed twice counts once as a duplicate."""
        fake = FakePairSys(clock)
        robot, pair_sys = make_robot(fake)
        constant = fake.teleoperate_step()
        pair_sys.teleoperate_step.side_effect = None
        pair_sys.teleoperate_step.return_value = constant

        obs = robot.record_parallel(max_frame=3, fps=100, teleop_hz=200)

        assert obs.arms.leader.shape[0] == 3
        assert robot._last_record_stats["duplicate_snapshots"] == 2


# --- record_with_fixed_leader() ------------------------------------------------


class TestRecordWithFixedLeader:
    def test_leader_is_the_action_and_has_no_read_stamp(self, clock: StepClock) -> None:
        fake = FakePairSys(clock)
        robot, pair_sys = make_robot(fake)
        leader_action = np.arange(3 * N_JOINTS, dtype=np.float32).reshape(3, N_JOINTS)

        obs = robot.record_with_fixed_leader(
            max_frame=3, leader_action=leader_action, fps=100, teleop_hz=200
        )

        arms = obs.arms
        np.testing.assert_array_equal(arms.leader, leader_action)
        assert arms.leader_time_s is None
        assert arms.leader_velocity is None and arms.leader_current is None
        assert arms.follower.shape == (3, N_JOINTS)
        assert arms.follower_velocity is not None and arms.follower_velocity.shape == (3, N_JOINTS)
        assert arms.follower_current is not None
        assert_stamped_after_t0(obs, robot, clock)
        assert robot._last_record_summary["terminated_by"] == "max_frame"
        assert pair_sys.send_follower_action.call_count >= 1
        pair_sys.get_observation.assert_not_called()

    def test_worker_error_stops_the_recording(self, clock: StepClock) -> None:
        fake = FakePairSys(clock)
        robot, pair_sys = make_robot(fake)
        pair_sys.send_follower_action.side_effect = ConnectionError("follower gone")
        leader_action = np.zeros((100, N_JOINTS), dtype=np.float32)

        with pytest.raises(RuntimeError, match="teleop_stopped.*0 frames"):
            robot.record_with_fixed_leader(
                max_frame=100, leader_action=leader_action, fps=100, teleop_hz=200
            )

        assert "follower gone" in robot._last_record_summary["worker_error"]

    def test_follower_read_failure_stops_the_recording(self, clock: StepClock) -> None:
        """A follower read that keeps failing ends the recording, it does not starve it."""
        fake = FakePairSys(clock)
        robot, pair_sys = make_robot(fake)
        successes = 3

        def get_follower_state() -> RakudaArmState:
            if fake.calls >= successes:
                raise DynamixelTimeoutError("follower read timed out", -3001)
            return fake.get_follower_state()

        pair_sys.get_follower_state.side_effect = get_follower_state
        leader_action = np.zeros((100, N_JOINTS), dtype=np.float32)

        started = time.monotonic()
        obs = robot.record_with_fixed_leader(
            max_frame=100, leader_action=leader_action, fps=100, teleop_hz=200
        )

        assert time.monotonic() - started < 2.0  # the sampler did not wait for 100 frames
        frames = obs.arms.follower.shape[0]
        assert 1 <= frames <= successes
        summary = robot._last_record_summary
        assert summary["terminated_by"] == "teleop_stopped"
        assert summary["frames"] == frames
        assert "follower read timed out" in summary["worker_error"]


# --- config / control_report() / metadata ----------------------------------------


def test_config_is_the_pair_sys_config(clock: StepClock) -> None:
    """The ports RakudaPairSys resolved are what ``RakudaRobot.config`` reports."""
    robot, pair_sys = make_robot(FakePairSys(clock))
    pair_sys.config = RakudaConfig(leader_port="/dev/ttyUSB1", follower_port="/dev/ttyUSB0")

    assert robot.config is pair_sys.config


class TestControlReport:
    def test_conventional_report(self, clock: StepClock) -> None:
        fake = FakePairSys(clock)
        robot, _ = make_robot(fake)
        robot.record(max_frame=2, fps=1000)

        report = robot.control_report()

        assert set(report) == {
            "mode",
            "current_sign",
            "record",
            "faults",
            "teleop_hz_effective",
            "sampler",
            "motors",
            "dynamixel_sdk",
            "ports",
        }
        assert report["mode"] == "position_teleop"
        assert report["current_sign"] == {} and report["faults"] == []
        assert report["record"] == robot._last_record_summary
        assert report["record"]["terminated_by"] == "max_frame"
        assert report["teleop_hz_effective"] == report["record"]["teleop_hz_effective"]
        assert report["sampler"] == robot._last_record_stats
        assert report["motors"]["names"] == list(RAKUDA_JOINT_NAMES)
        assert report["motors"]["leader_models"] == ["xc330-t288"] * N_JOINTS
        assert report["ports"] == {"leader": "/dev/ttyUSB1", "follower": "/dev/ttyUSB0"}
        assert isinstance(report["dynamixel_sdk"]["version"], str)
        assert isinstance(report["dynamixel_sdk"]["path"], str)
        # The report is JSON-ready as is.
        json.dumps(report)

    def test_report_before_any_record_is_empty(self, clock: StepClock) -> None:
        robot, _ = make_robot(FakePairSys(clock))

        report = robot.control_report()

        assert report["record"] == {} and report["sampler"] == {}
        assert report["teleop_hz_effective"] is None


def make_handler(tmp_path: Path, config: RakudaConfig, report: Dict[str, Any]) -> RakudaExpHandler:
    handler = object.__new__(RakudaExpHandler)
    handler.metadata_config = MetaDataConfig(task_name="t", description="d", date="2026-09-24")
    robot = MagicMock()
    robot.config = config
    robot.control_report.return_value = dict(report)
    handler._robot = robot
    return handler


class TestSaveMetadata:
    REPORT = {"mode": "position_teleop", "current_sign": {}, "record": {"frames": 1}, "faults": []}

    def test_adds_control_and_nested_bilateral_config(
        self, tmp_path: Path, caplog: pytest.LogCaptureFixture
    ) -> None:
        config = RakudaConfig(
            leader_port="/dev/ttyUSB1",
            follower_port="/dev/ttyUSB0",
            bilateral=RakudaBilateralParams(control_hz=100.0),
        )
        handler = make_handler(tmp_path, config, self.REPORT)

        with caplog.at_level("WARNING", logger="robopy.utils.exp_interface.rakuda_exp_handler"):
            handler.save_metadata(str(tmp_path), {"arms": {"leader": [1, 17]}})

        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert metadata["task_details"]["task_name"] == "t"
        assert metadata["data_shape"] == {"arms": {"leader": [1, 17]}}
        assert metadata["robot_config"]["bilateral"]["control_hz"] == 100.0
        assert metadata["robot_config"]["bilateral"]["leader_health"]["max_temperature_c"] == 66.0
        assert metadata["control"]["mode"] == "position_teleop"
        assert metadata["control"]["record"] == {"frames": 1}
        # bilateral configured but the conventional path recorded: flagged + warned.
        assert metadata["control"]["config_mismatch"] is True
        assert any(
            record.levelname == "WARNING" and "config_mismatch" in record.getMessage()
            for record in caplog.records
        )

    def test_no_mismatch_flag_in_conventional_config(self, tmp_path: Path) -> None:
        config = RakudaConfig(leader_port="/dev/ttyUSB1", follower_port="/dev/ttyUSB0")
        handler = make_handler(tmp_path, config, self.REPORT)

        handler.save_metadata(str(tmp_path))

        metadata = json.loads((tmp_path / "metadata.json").read_text())
        assert "config_mismatch" not in metadata["control"]
        assert metadata["robot_config"]["bilateral"] is None
        assert "data_shape" not in metadata


# --- save worker end to end ----------------------------------------------------


def test_save_all_obs_writes_v2_file_with_report(tmp_path: Path) -> None:
    """``save_all_obs`` asks ``control_report`` and writes the v2 ``arm`` group."""
    frames = [
        RakudaArmObs(
            leader=np.full(N_JOINTS, i, np.float32),
            follower=np.full(N_JOINTS, i, np.float32),
            leader_velocity=np.zeros(N_JOINTS, np.float32),
            follower_velocity=np.zeros(N_JOINTS, np.float32),
            leader_current=np.zeros(N_JOINTS, np.float32),
            follower_current=np.zeros(N_JOINTS, np.float32),
            leader_t_ns=10 * MS * (i + 1),
            follower_t_ns=10 * MS * (i + 1) + MS,
        ).stamped(t0_ns=0, frame_t_ns=10 * MS * (i + 1) + 2 * MS)
        for i in range(2)
    ]
    obs = RakudaObs(
        arms=RakudaArmObs.stack(frames),
        sensors=RakudaSensorObs(
            cameras={"main": np.zeros((2, 3, 4, 4), np.float32)}, tactile={}, audio={}
        ),
    )
    calls: List[int] = []

    def report() -> Dict[str, Any]:
        calls.append(1)
        return {
            "mode": "position_teleop",
            "current_sign": {},
            "record": {"t0_monotonic_ns": 0, "terminated_by": "max_frame", "frames_requested": 2},
            "motors": {"names": list(RAKUDA_JOINT_NAMES)},
        }

    config = RakudaConfig(leader_port="/dev/ttyUSB1", follower_port="/dev/ttyUSB0")
    worker = RakudaSaveWorker(config, worker_num=1, fps=20, control_report=report)
    try:
        worker.save_all_obs(obs, str(tmp_path / "run"), save_gif=False)
        worker.wait_all_saved()
    finally:
        worker.shutdown()

    assert calls == [1]
    import h5py

    with h5py.File(str(tmp_path / "run" / "rakuda_observations.h5"), "r") as f:
        arm = f["arm"]
        assert isinstance(arm, h5py.Group)
        assert set(arm.keys()) == set(RakudaArmObs.ARRAY_FIELDS)
        assert arm.attrs["schema_version"] == 2
        assert arm.attrs["terminated_by"] == "max_frame"
        assert arm.attrs["record_fps"] == 20
        camera = f["camera/main"]
        assert isinstance(camera, h5py.Dataset) and camera.shape == (2, 3, 4, 4)
    assert (tmp_path / "run" / "arm_obs.jpg").exists()
