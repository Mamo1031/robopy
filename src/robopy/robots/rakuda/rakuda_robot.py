import importlib.metadata
import os
import queue
import threading
import time
from collections import defaultdict
from concurrent.futures import Future, ThreadPoolExecutor
from logging import getLogger
from typing import Any, Dict, List, Literal

import dynamixel_sdk
import numpy as np
from numpy.typing import NDArray
from rich.console import Console
from rich.table import Table

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_MOTOR_MAPPING,
    RakudaArmObs,
    RakudaArmState,
    RakudaConfig,
    RakudaObs,
    RakudaSensorConfigs,
    RakudaSensorObs,
)
from robopy.config.sensor_config import RealsenseCameraConfig, Sensors
from robopy.config.sensor_config.params_config import AudioParams, TactileParams
from robopy.sensors.audio import AudioSensor
from robopy.sensors.tactile import DigitSensor
from robopy.sensors.visual import RealsenseCamera

from ..common.composed import ComposedRobot
from .rakuda_arm import BusFactory
from .rakuda_leader_control import BilateralNotReady, FollowerLost, LoopStopped
from .rakuda_pair_sys import RakudaPairSys

logger = getLogger(__name__)

#: Why a recording ended. ``loop_fault`` is reported by the bilateral loop.
TerminatedBy = Literal["max_frame", "keyboard_interrupt", "teleop_stopped", "loop_fault"]


def _stop_reason(error: BaseException | None) -> TerminatedBy:
    """Why a recording ended before ``max_frame``.

    ``loop_fault`` when the bilateral loop latched a fault (the ``LoopStopped``
    it raised carries it), ``teleop_stopped`` for every other stop.
    """
    if isinstance(error, LoopStopped) and error.fault is not None:
        return "loop_fault"
    return "teleop_stopped"


def _dynamixel_sdk_info() -> Dict[str, str | None]:
    """Installed ``dynamixel_sdk`` version and location, for ``control_report()``."""
    try:
        version: str | None = importlib.metadata.version("dynamixel_sdk")
    except importlib.metadata.PackageNotFoundError:
        version = None
    path = getattr(dynamixel_sdk, "__file__", None)
    return {"version": version, "path": None if path is None else os.path.dirname(path)}


class _Sampler:
    """Per-recording counters of the frame sampler.

    ``duplicate_snapshots`` counts committed frames whose ``follower_time_s``
    equals the previous frame's, i.e. the follower snapshot did not change.
    """

    def __init__(self) -> None:
        self.queue_empty_waits = 0
        self.over_budget_frames = 0
        self.duplicate_snapshots = 0
        self._last_follower_time_s: float | None = None

    def commit(self, obs: RakudaArmObs) -> None:
        """Accounts one committed (stamped) frame."""
        if obs.follower_time_s is None:
            return
        follower_time_s = float(obs.follower_time_s)
        if follower_time_s == self._last_follower_time_s:
            self.duplicate_snapshots += 1
        self._last_follower_time_s = follower_time_s

    def as_dict(self, frames: int) -> Dict[str, int]:
        return {
            "frames": frames,
            "queue_empty_waits": self.queue_empty_waits,
            "over_budget_frames": self.over_budget_frames,
            "duplicate_snapshots": self.duplicate_snapshots,
        }


class RakudaRobot(ComposedRobot[RakudaPairSys, Sensors, RakudaObs]):
    def __init__(self, cfg: RakudaConfig, bus_factory: BusFactory | None = None):
        """Builds the arm pair and the sensors; no serial port is opened here.

        Args:
            cfg: Robot configuration.  ``RakudaPairSys`` applies the
                ``.robopy/rakuda/config.yaml`` overrides and owns the config
                from here on (see the ``config`` property).
            bus_factory: ``(port, motors) -> bus`` handed to both arms;
                defaults to the real ``DynamixelBus`` (tests inject simulated buses).
        """
        self._pair_sys = RakudaPairSys(cfg, bus_factory)
        self._sensor_configs: RakudaSensorConfigs = self._init_config()
        self._sensors: Sensors = self._init_sensors()
        # Facts of the last recording for control_report().
        self._last_record_t0_ns: int | None = None
        self._last_record_stats: Dict[str, int] = {}
        self._last_record_summary: Dict[str, Any] = {}

    def connect(self) -> None:
        try:
            self._pair_sys.connect()
        except Exception as e:
            self._pair_sys.disconnect()
            raise e

    def disconnect(self) -> None:
        """Disconnects the arms, then the sensors.

        ``RakudaPairSys.disconnect()`` holds both arms first when the bilateral
        loop is running (``stop_bilateral()``, never raises).
        """
        self._pair_sys.disconnect()

        for cam in self._sensors.cameras or []:
            cam.disconnect()

        for tac in self._sensors.tactile or []:
            tac.disconnect()

        for audio in self._sensors.audio or []:
            audio.disconnect()

    def teleoperation(self, max_seconds: float | None = None) -> None:
        """Leader drives follower until ``max_seconds`` (if positive) elapse or Ctrl-C.

        ``RakudaPairSys.teleoperate()`` in both modes.  Conventional mode is
        unchanged (Ctrl-C is swallowed there and the follower released).  In
        bilateral mode the pair system starts the loop if needed, watches it,
        and re-raises ``KeyboardInterrupt`` after holding both arms as well as
        ``LoopStopped``/``FollowerLost``; nothing is swallowed here, so
        ``record_save`` ends in one go.
        """
        if not self.is_connected:
            raise ConnectionError("RakudaRobot is not connected. Call connect() first.")

        if max_seconds is not None and max_seconds > 0:
            self._pair_sys.teleoperate(max_seconds=max_seconds)
        else:
            self._pair_sys.teleoperate()

    def record(self, max_frame: int, fps: int = 5) -> RakudaObs:
        if not self.is_connected:
            self.connect()

        if max_frame <= 0:
            raise ValueError("max_frame must be greater than 0.")

        # Bilateral: start the loop, or re-align the follower, before the
        # recording clock starts; a failure propagates as is.
        self._pair_sys.ensure_bilateral_running()

        arm_frames: List[RakudaArmObs] = []
        camera_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)
        tactile_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)
        audio_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)

        get_obs_interval = 1.0 / fps
        frame_count = 0
        teleop_steps = 0
        sampler = _Sampler()
        terminated_by: TerminatedBy = "max_frame"
        loop_error: Exception | None = None
        # The time origin of every *_time_s in this recording.
        t0_ns = time.monotonic_ns()
        t0_unix_s = time.time()
        self._last_record_t0_ns = t0_ns
        interval_start = time.monotonic()

        try:
            while frame_count < max_frame:
                try:
                    temp_arm_obs = self.robot_system.teleoperate_step()
                except BilateralNotReady as e:
                    # A late pair is not a stop: the loop is still RUNNING.
                    logger.warning("No fresh leader/follower pair: %s", e)
                    continue
                teleop_steps += 1

                if time.monotonic() - interval_start < get_obs_interval:
                    continue

                arm_obs = temp_arm_obs.stamped(t0_ns=t0_ns, frame_t_ns=time.monotonic_ns())
                arm_frames.append(arm_obs)
                sampler.commit(arm_obs)

                sensor_data = self.sensors_observation()
                camera_data = sensor_data.cameras
                tactile_data = sensor_data.tactile
                audio_data = sensor_data.audio

                for cam_name, cam_frame in camera_data.items():
                    camera_obs[cam_name].append(cam_frame)

                for tac_name, tac_frame in tactile_data.items():
                    tactile_obs[tac_name].append(tac_frame)

                for audio_name, audio_frame in audio_data.items():
                    audio_obs[audio_name].append(audio_frame)

                frame_count += 1
                interval_start = time.monotonic()

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user.")
            terminated_by = "keyboard_interrupt"
        except (LoopStopped, FollowerLost) as e:
            # The bilateral loop ended the episode (it holds or keeps running as
            # it is); the frames collected so far are returned.
            logger.warning("The bilateral loop ended the recording: %s", e)
            terminated_by = _stop_reason(e)
            loop_error = e
        except Exception as e:
            logger.error(f"An error occurred during recording: {e}")
            raise e

        self._finish_record(
            terminated_by=terminated_by,
            frames=frame_count,
            frames_requested=max_frame,
            t0_ns=t0_ns,
            t0_unix_s=t0_unix_s,
            teleop_steps=teleop_steps,
            worker_error=loop_error,
            sampler=sampler,
        )
        arms = RakudaArmObs.stack(arm_frames)
        # process camera observations
        camera_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for cam_name, frames in camera_obs.items():
            if frames:
                camera_obs_np[cam_name] = np.array(frames)
            else:
                camera_obs_np[cam_name] = None

        # process tactile observations
        tactile_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for tac_name, frames in tactile_obs.items():
            if frames:
                tactile_obs_np[tac_name] = np.array(frames)
            else:
                tactile_obs_np[tac_name] = None

        # process audio observations
        audio_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for audio_name, frames in audio_obs.items():
            if frames:
                audio_obs_np[audio_name] = np.array(frames)
            else:
                audio_obs_np[audio_name] = None

        sensors_obs = RakudaSensorObs(
            cameras=camera_obs_np, tactile=tactile_obs_np, audio=audio_obs_np
        )
        return RakudaObs(arms=arms, sensors=sensors_obs)

    def record_parallel(
        self,
        max_frame: int,
        fps: int = 20,
        teleop_hz: int = 25,
        max_processing_time_ms: float = 40,
    ) -> RakudaObs:
        """
        Fast recording: runs teleoperate_step at teleop_hz, records the latest
        arm_obs every frame at fps and reads the sensors in parallel.
        """
        if not self.is_connected:
            self.connect()

        if max_frame <= 0:
            raise ValueError("max_frame must be greater than 0.")

        # Bilateral: start the loop, or re-align the follower, before any
        # thread exists; a failure propagates as is.
        self._pair_sys.ensure_bilateral_running()

        # Queue that carries arm_obs at the teleoperation rate
        arm_obs_queue: queue.Queue[RakudaArmObs] = queue.Queue(maxsize=teleop_hz * 2)
        stop_event = threading.Event()
        worker_error: Exception | None = None
        teleop_steps = 0

        def teleop_worker() -> None:
            nonlocal worker_error, teleop_steps
            interval = 1.0 / teleop_hz
            try:
                while not stop_event.is_set():
                    start_time = time.perf_counter()
                    try:
                        obs = self.robot_system.teleoperate_step()
                    except BilateralNotReady as e:
                        # A late pair is not a stop: the loop is still RUNNING.
                        logger.warning("No fresh leader/follower pair: %s", e)
                        continue
                    teleop_steps += 1

                    try:
                        arm_obs_queue.put(obs, timeout=interval)
                    except queue.Full:
                        pass
                    elapsed = time.perf_counter() - start_time
                    sleep_time = max(0, interval - elapsed)
                    time.sleep(sleep_time)
            except (LoopStopped, FollowerLost) as e:
                # The bilateral loop ended the episode; the sampler returns the
                # frames collected so far.
                worker_error = e
                logger.warning("The bilateral loop stopped the recording: %s", e)
            except Exception as e:
                # A dead worker must end the recording, not starve it.
                worker_error = e
                logger.exception("Teleoperation worker stopped.")
            finally:
                stop_event.set()

        # The time origin of every *_time_s in this recording.
        t0_ns = time.monotonic_ns()
        t0_unix_s = time.time()
        self._last_record_t0_ns = t0_ns
        teleop_thread = threading.Thread(target=teleop_worker, daemon=True)
        teleop_thread.start()

        arm_frames: List[RakudaArmObs] = []
        camera_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)
        tactile_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)
        audio_obs: Dict[str, List[NDArray[np.float32] | None]] = defaultdict(list)

        get_obs_interval = 1.0 / fps
        max_processing_time = max_processing_time_ms / 1000.0
        frame_count = 0
        total_processing_time = 0.0
        sampler = _Sampler()
        terminated_by: TerminatedBy = "max_frame"

        logger.info(f"Starting parallel recording: {max_frame} frames at {fps}Hz")
        logger.info(
            f"""Target interval: {get_obs_interval * 1000:.1f}ms,
            Max processing time: {max_processing_time_ms}ms"""
        )

        try:
            while frame_count < max_frame and not stop_event.is_set():
                frame_start_time = time.perf_counter()

                # Take the latest arm_obs (wait while the buffer is empty)
                try:
                    while True:
                        arm_obs = arm_obs_queue.get(timeout=get_obs_interval)
                        while not arm_obs_queue.empty():
                            arm_obs = arm_obs_queue.get_nowait()
                        break
                except queue.Empty:
                    sampler.queue_empty_waits += 1
                    logger.warning("No arm_obs available in time.")
                    continue

                # Read the sensors in parallel
                try:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        # Camera futures
                        camera_futures: Dict[str, Future[NDArray[np.float32] | None]] = {}
                        if self._sensors.cameras:
                            for cam in self._sensors.cameras:
                                if cam.is_connected:
                                    camera_futures[cam.name] = executor.submit(
                                        cam.async_read, timeout_ms=5
                                    )

                        # Tactile futures
                        tactile_futures: Dict[str, Future[NDArray[np.float32] | None]] = {}
                        if self._sensors.tactile:
                            for tac in self._sensors.tactile:
                                if tac.is_connected:
                                    tactile_futures[tac.name] = executor.submit(
                                        tac.async_read, timeout_ms=5
                                    )

                        # Audio futures
                        audio_futures: Dict[str, Future[NDArray[np.float32] | None]] = {}
                        if self._sensors.audio:
                            for audio in self._sensors.audio:
                                if audio.is_connected:
                                    audio_futures[audio.name] = executor.submit(
                                        audio.async_read, timeout_ms=5
                                    )

                        timeout = max_processing_time * 0.5

                        camera_data: Dict[str, NDArray[np.float32] | None] = {}
                        for cam_name, future in camera_futures.items():
                            try:
                                camera_data[cam_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Camera {cam_name} failed in frame {frame_count}: {e}"
                                )
                                camera_data[cam_name] = None

                        tactile_data: Dict[str, NDArray[np.float32] | None] = {}
                        for tac_name, future in tactile_futures.items():
                            try:
                                tactile_data[tac_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Tactile {tac_name} failed in frame {frame_count}: {e}"
                                )
                                tactile_data[tac_name] = None

                        audio_data: Dict[str, NDArray[np.float32] | None] = {}
                        for audio_name, future in audio_futures.items():
                            try:
                                audio_data[audio_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Audio {audio_name} failed in frame {frame_count}: {e}"
                                )
                                audio_data[audio_name] = None

                    # Record
                    stamped = arm_obs.stamped(t0_ns=t0_ns, frame_t_ns=time.monotonic_ns())
                    arm_frames.append(stamped)
                    sampler.commit(stamped)

                    for cam_name, cam_frame in camera_data.items():
                        camera_obs[cam_name].append(cam_frame)

                    for tac_name, tac_frame in tactile_data.items():
                        tactile_obs[tac_name].append(tac_frame)

                    for audio_name, audio_frame in audio_data.items():
                        audio_obs[audio_name].append(audio_frame)

                    frame_count += 1
                    logger.info("Recording progress: %s/%s frames", frame_count, max_frame)

                    # Timing
                    processing_time = time.perf_counter() - frame_start_time
                    total_processing_time += processing_time

                    if processing_time > max_processing_time:
                        sampler.over_budget_frames += 1
                        logger.warning(
                            f"""Frame {frame_count} took {processing_time * 1000:.1f}ms
                            (>{max_processing_time_ms}ms), skipping"""
                        )
                        continue

                    elapsed = time.perf_counter() - frame_start_time
                    sleep_time = max(0, get_obs_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"Unexpected error in frame {frame_count}: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user.")
            terminated_by = "keyboard_interrupt"
        except Exception as e:
            logger.error(f"An error occurred during parallel recording: {e}")
            raise e
        finally:
            stop_event.set()
            teleop_thread.join(timeout=1.0)

        if terminated_by == "max_frame" and frame_count < max_frame:
            terminated_by = _stop_reason(worker_error)
        avg_processing_time = total_processing_time / max(1, frame_count) * 1000
        logger.info(
            f"Recording completed: {frame_count} frames, {sampler.over_budget_frames} over budget"
        )
        logger.info(f"Average processing time: {avg_processing_time:.1f}ms")

        self._finish_record(
            terminated_by=terminated_by,
            frames=frame_count,
            frames_requested=max_frame,
            t0_ns=t0_ns,
            t0_unix_s=t0_unix_s,
            teleop_steps=teleop_steps,
            worker_error=worker_error,
            sampler=sampler,
        )
        arms = RakudaArmObs.stack(arm_frames)

        camera_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for cam_name, frames in camera_obs.items():
            if frames and all(frame is not None for frame in frames):
                camera_obs_np[cam_name] = np.array(frames)
            else:
                camera_obs_np[cam_name] = None

        tactile_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for tac_name, frames in tactile_obs.items():
            if frames and all(frame is not None for frame in frames):
                tactile_obs_np[tac_name] = np.array(frames).transpose(0, 3, 1, 2)
            else:
                tactile_obs_np[tac_name] = None

        audio_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for audio_name, frames in audio_obs.items():
            if frames and all(frame is not None for frame in frames):
                audio_obs_np[audio_name] = np.array(frames)
            else:
                audio_obs_np[audio_name] = None

        sensors_obs = RakudaSensorObs(
            cameras=camera_obs_np, tactile=tactile_obs_np, audio=audio_obs_np
        )
        return RakudaObs(arms=arms, sensors=sensors_obs)

    def record_with_fixed_leader(
        self,
        max_frame: int,
        leader_action: NDArray[np.float32],
        fps: int = 20,
        teleop_hz: int = 100,
        max_processing_time_ms: float = 40,
    ) -> RakudaObs:
        """
        Replays the given leader sequence and collects the follower and other observations.

        Raises:
            RuntimeError: While the bilateral loop is running.
        """
        if not self.is_connected:
            self.connect()
        self._reject_while_bilateral_active("record_with_fixed_leader()")

        if max_frame <= 0:
            raise ValueError("max_frame must be greater than 0.")

        if len(leader_action) != max_frame:
            raise ValueError("Length of leader_action must match max_frame.")

        # Queue that carries follower states at the teleoperation rate
        follower_queue: queue.Queue[RakudaArmState] = queue.Queue(maxsize=teleop_hz * 2)
        stop_event = threading.Event()
        worker_error: Exception | None = None
        teleop_steps = 0

        def control_worker() -> None:
            nonlocal worker_error, teleop_steps
            interval = 1.0 / teleop_hz
            start_time = time.perf_counter()  # interpolation only

            try:
                while not stop_event.is_set():
                    loop_start = time.perf_counter()
                    t = loop_start - start_time

                    # Linear interpolation; past max_frame the last frame is used
                    idx_float = t * fps
                    if idx_float >= max_frame - 1:
                        action = leader_action[-1]
                    else:
                        idx0 = int(np.floor(idx_float))
                        idx1 = min(idx0 + 1, max_frame - 1)
                        alpha = idx_float - idx0
                        action = (1 - alpha) * leader_action[idx0] + alpha * leader_action[idx1]

                    # Send to the follower
                    self.send_frame_action(action)
                    teleop_steps += 1

                    # Read the follower state. A failed read propagates instead of
                    # being swallowed: a silently empty queue would starve the sampler.
                    follower_state = self._pair_sys.get_follower_state()
                    try:
                        follower_queue.put(follower_state, timeout=interval)
                    except queue.Full:
                        pass

                    elapsed = time.perf_counter() - loop_start
                    sleep_time = max(0, interval - elapsed)
                    time.sleep(sleep_time)
            except Exception as e:
                # A dead worker must end the recording, not starve it.
                worker_error = e
                logger.exception("Control worker stopped.")
            finally:
                stop_event.set()

        # The time origin of every *_time_s in this recording.
        t0_ns = time.monotonic_ns()
        t0_unix_s = time.time()
        self._last_record_t0_ns = t0_ns
        control_thread = threading.Thread(target=control_worker, daemon=True)
        control_thread.start()

        arm_frames: List[RakudaArmObs] = []
        camera_obs: Dict[str, List] = defaultdict(list)
        tactile_obs: Dict[str, List] = defaultdict(list)
        audio_obs: Dict[str, List] = defaultdict(list)

        get_obs_interval = 1.0 / fps
        max_processing_time = max_processing_time_ms / 1000.0
        frame_count = 0
        total_processing_time = 0.0
        sampler = _Sampler()
        terminated_by: TerminatedBy = "max_frame"

        logger.info(f"Starting fixed leader recording: {max_frame} frames at {fps}Hz")

        try:
            while frame_count < max_frame and not stop_event.is_set():
                frame_start_time = time.perf_counter()

                # Take the latest follower state
                try:
                    while True:
                        current_follower = follower_queue.get(timeout=get_obs_interval)
                        while not follower_queue.empty():
                            current_follower = follower_queue.get_nowait()
                        break
                except queue.Empty:
                    sampler.queue_empty_waits += 1
                    logger.warning("No follower_obs available in time.")
                    continue

                # Read the sensors in parallel
                try:
                    with ThreadPoolExecutor(max_workers=4) as executor:
                        # Camera futures
                        camera_futures = {}
                        if self._sensors.cameras:
                            for cam in self._sensors.cameras:
                                if cam.is_connected:
                                    camera_futures[cam.name] = executor.submit(
                                        cam.async_read, timeout_ms=5
                                    )

                        # Tactile futures
                        tactile_futures = {}
                        if self._sensors.tactile:
                            for tac in self._sensors.tactile:
                                if tac.is_connected:
                                    tactile_futures[tac.name] = executor.submit(
                                        tac.async_read, timeout_ms=5
                                    )

                        # Audio futures
                        audio_futures = {}
                        if self._sensors.audio:
                            for audio in self._sensors.audio:
                                if audio.is_connected:
                                    audio_futures[audio.name] = executor.submit(
                                        audio.async_read, timeout_ms=5
                                    )

                        timeout = max_processing_time * 0.5

                        camera_data: Dict[str, NDArray | None] = {}
                        for cam_name, future in camera_futures.items():
                            try:
                                camera_data[cam_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Camera {cam_name} failed in frame {frame_count}: {e}"
                                )
                                camera_data[cam_name] = None

                        tactile_data: Dict[str, NDArray | None] = {}
                        for tac_name, future in tactile_futures.items():
                            try:
                                tactile_data[tac_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Tactile {tac_name} failed in frame {frame_count}: {e}"
                                )
                                tactile_data[tac_name] = None

                        audio_data: Dict[str, NDArray | None] = {}
                        for audio_name, future in audio_futures.items():
                            try:
                                audio_data[audio_name] = future.result(timeout=timeout / 2)
                            except Exception as e:
                                logger.warning(
                                    f"Audio {audio_name} failed in frame {frame_count}: {e}"
                                )
                                audio_data[audio_name] = None

                    # Record
                    # The leader is the given action (no read stamp -> leader_time_s None)
                    stamped = RakudaArmObs(
                        leader=leader_action[frame_count],
                        follower=current_follower.position.astype(np.float32),
                        follower_velocity=current_follower.velocity.astype(np.float32),
                        follower_current=current_follower.current_ma,
                        follower_t_ns=current_follower.t_end_ns,
                    ).stamped(t0_ns=t0_ns, frame_t_ns=time.monotonic_ns())
                    arm_frames.append(stamped)
                    sampler.commit(stamped)

                    for cam_name, cam_frame in camera_data.items():
                        camera_obs[cam_name].append(cam_frame)

                    for tac_name, tac_frame in tactile_data.items():
                        tactile_obs[tac_name].append(tac_frame)

                    for audio_name, audio_frame in audio_data.items():
                        audio_obs[audio_name].append(audio_frame)

                    frame_count += 1

                    # Timing
                    processing_time = time.perf_counter() - frame_start_time
                    total_processing_time += processing_time

                    if processing_time > max_processing_time:
                        sampler.over_budget_frames += 1
                        logger.warning(
                            f"""Frame {frame_count} took {processing_time * 1000:.1f}ms
                            (>{max_processing_time_ms}ms), skipping"""
                        )
                        continue

                    elapsed = time.perf_counter() - frame_start_time
                    sleep_time = max(0, get_obs_interval - elapsed)
                    if sleep_time > 0:
                        time.sleep(sleep_time)

                except Exception as e:
                    logger.error(f"Unexpected error in frame {frame_count}: {e}")
                    continue

        except KeyboardInterrupt:
            logger.info("Recording interrupted by user.")
            terminated_by = "keyboard_interrupt"
        except Exception as e:
            logger.error(f"An error occurred during parallel recording: {e}")
            raise e
        finally:
            stop_event.set()
            control_thread.join(timeout=1.0)

        if terminated_by == "max_frame" and frame_count < max_frame:
            terminated_by = "teleop_stopped"
        avg_processing_time = total_processing_time / max(1, frame_count) * 1000
        logger.info(
            f"Recording completed: {frame_count} frames, {sampler.over_budget_frames} over budget"
        )
        logger.info(f"Average processing time: {avg_processing_time:.1f}ms")

        self._finish_record(
            terminated_by=terminated_by,
            frames=frame_count,
            frames_requested=max_frame,
            t0_ns=t0_ns,
            t0_unix_s=t0_unix_s,
            teleop_steps=teleop_steps,
            worker_error=worker_error,
            sampler=sampler,
        )
        arms = RakudaArmObs.stack(arm_frames)

        camera_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for cam_name, frames in camera_obs.items():
            if frames and all(frame is not None for frame in frames):
                camera_obs_np[cam_name] = np.array(frames)
            else:
                camera_obs_np[cam_name] = None

        tactile_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for tac_name, frames in tactile_obs.items():
            if frames and all(frame is not None for frame in frames):
                tactile_obs_np[tac_name] = np.array(frames).transpose(0, 3, 1, 2)
            else:
                tactile_obs_np[tac_name] = None

        audio_obs_np: Dict[str, NDArray[np.float32] | None] = {}
        for audio_name, frames in audio_obs.items():
            if frames and all(frame is not None for frame in frames):
                audio_obs_np[audio_name] = np.array(frames)
            else:
                audio_obs_np[audio_name] = None

        sensors_obs = RakudaSensorObs(
            cameras=camera_obs_np, tactile=tactile_obs_np, audio=audio_obs_np
        )
        return RakudaObs(arms=arms, sensors=sensors_obs)

    def _finish_record(
        self,
        *,
        terminated_by: TerminatedBy,
        frames: int,
        frames_requested: int,
        t0_ns: int,
        t0_unix_s: float,
        teleop_steps: int,
        worker_error: Exception | None,
        sampler: _Sampler,
    ) -> None:
        """Stores the summary of the recording that just ended.

        ``worker_error`` is the exception that stopped the teleoperation (the
        worker's, or the loop's in the serial :meth:`record`), if any.

        Raises:
            RuntimeError: When no frame was collected; a partial recording is
                returned by the caller otherwise.
        """
        elapsed_s = (time.monotonic_ns() - t0_ns) / 1e9
        self._last_record_stats = sampler.as_dict(frames)
        self._last_record_summary = {
            "t0_monotonic_ns": t0_ns,
            "t0_unix_s": t0_unix_s,
            "duration_s": elapsed_s,
            "frames": frames,
            "frames_requested": frames_requested,
            "terminated_by": terminated_by,
            "teleop_hz_effective": teleop_steps / elapsed_s if elapsed_s > 0 else 0.0,
            "worker_error": None if worker_error is None else repr(worker_error),
        }
        if worker_error is not None:
            logger.warning("The teleoperation worker stopped the recording: %r", worker_error)
        if frames == 0:
            detail = "" if worker_error is None else f": {worker_error!r}"
            raise RuntimeError(f"Recording ended ({terminated_by}) with 0 frames{detail}")

    def control_report(self) -> Dict[str, Any]:
        """Control-side facts for ``metadata.json`` and the HDF5 ``arm`` attributes.

        The bilateral loop's report (``RakudaPairSys.control_report()``) when
        the loop exists, otherwise the conventional position-teleoperation
        report; both carry ``record``/``sampler`` of the last recording (empty
        until one has ended) and the robot-level facts.  Every fault gets
        ``t_s``, its time relative to the last recording's ``t0``, or None
        outside that recording.
        """
        record = dict(self._last_record_summary)
        pair = self._pair_sys
        loop_report = pair.control_report()
        report: Dict[str, Any] = (
            {"mode": "position_teleop", "current_sign": {}}
            if loop_report is None
            else dict(loop_report)
        )
        report["record"] = record
        report["faults"] = [
            {**fault, "t_s": self._record_relative_s(fault.get("t_ns"))}
            for fault in report.get("faults", [])
        ]
        report["teleop_hz_effective"] = record.get("teleop_hz_effective")
        report["sampler"] = dict(self._last_record_stats)
        report["motors"] = {
            "names": list(pair.leader.motor_names),
            "leader_models": list(pair.leader.motor_models),
            "follower_models": list(pair.follower.motor_models),
        }
        report["dynamixel_sdk"] = _dynamixel_sdk_info()
        report["ports"] = {"leader": pair.leader.port, "follower": pair.follower.port}
        return report

    def _record_relative_s(self, t_ns: int | None) -> float | None:
        """Seconds of ``t_ns`` since the last recording's ``t0``.

        None when no recording has ended yet or ``t_ns`` lies outside the
        recording (before its ``t0`` or after its end).
        """
        record = self._last_record_summary
        t0_ns = record.get("t0_monotonic_ns")
        duration_s = record.get("duration_s")
        if t_ns is None or t0_ns is None or duration_s is None:
            return None
        t_s = (t_ns - t0_ns) / 1e9
        if t_s < 0.0 or t_s > duration_s:
            return None
        return t_s

    def _reject_while_bilateral_active(self, operation: str) -> None:
        """Refuses a follower-bus operation while the control thread owns both buses."""
        if self._pair_sys.bilateral_active:
            raise RuntimeError(
                f"{operation}: bilateral loop is running; call stop_bilateral() first"
            )

    def get_observation(self) -> RakudaObs:
        """get_observation get the current observation from the robot system and sensors."""
        if not self.is_connected:
            raise ConnectionError("RakudaRobot is not connected. Call connect() first.")

        arm_obs = self.get_arm_observation()
        sensor_obs = self.sensors_observation()
        return RakudaObs(arms=arm_obs, sensors=sensor_obs)

    def get_arm_observation(self) -> RakudaArmObs:
        """The current observation of both arms.

        A direct read of both buses, or the latest snapshots of the control
        thread while the bilateral loop is running (``RakudaPairSys`` decides).
        """
        if not self.is_connected:
            raise ConnectionError("RakudaRobot is not connected. Call connect() first.")

        return self._pair_sys.get_observation()

    def sensors_observation(self) -> RakudaSensorObs:
        """Get the current observation from the sensors."""
        if not self.is_connected:
            raise ConnectionError("RakudaRobot is not connected. Call connect() first.")

        if self._sensors is None:
            raise RuntimeError("Sensors are not initialized.")

        # Get camera data with reduced timeout for better performance
        camera_data: Dict[str, NDArray[np.float32] | None] = {}
        if self._sensors.cameras is not None:
            for cam in self._sensors.cameras:
                if cam.is_connected:
                    # Reduced timeout from 10ms to 5ms for 30Hz performance
                    camera_data[cam.name] = cam.async_read(timeout_ms=16)
                else:
                    logger.warning(f"Camera {cam.name} is not connected.")
                    camera_data[cam.name] = None
        else:
            logger.warning("No cameras are initialized in sensors.")
            camera_data = {}

        # Get tactile data using async_read when available
        tactile_data: Dict[str, NDArray[np.float32] | None] = {}
        if self._sensors.tactile is not None:
            for tac in self._sensors.tactile:
                if tac.is_connected:
                    # Use async_read if available for better performance
                    tac_data = tac.async_read(timeout_ms=50)
                    if tac_data is not None and tac_data.ndim == 3 and tac_data.shape[2] == 3:
                        tac_data = tac_data.transpose(2, 0, 1)  # HWC to CHW
                    tactile_data[tac.name] = tac_data
                else:
                    tactile_data[tac.name] = None
        else:
            logger.warning("No tactile sensors are initialized in sensors.")
            tactile_data = {}

        # Get audio data using async_read when available
        audio_data: Dict[str, NDArray[np.float32] | None] = {}
        if self._sensors.audio is not None:
            for audio in self._sensors.audio:
                if audio.is_connected:
                    # Use async_read if available for better performance
                    audio_frame = audio.async_read(timeout_ms=50)
                    if audio_frame is not None and audio_frame.ndim == 2:
                        audio_frame = audio_frame.transpose(1, 0)  # CHW to HWC
                    audio_data[audio.name] = audio_frame
                else:
                    audio_data[audio.name] = None
        else:
            logger.warning("No audio sensors are initialized in sensors.")
            audio_data = {}

        return RakudaSensorObs(cameras=camera_data, tactile=tactile_data, audio=audio_data)

    def send(
        self,
        max_frame: int,
        fps: int,
        leader_action: NDArray[np.float32],
        teleop_hz: int = 100,
    ) -> None:
        """send leader action sequence to Rakuda robot with interpolation.

        Args:
            max_frame (int): max frame to send
            fps (int): frame per second of leader_action
            leader_action (NDArray[np.float32]): action array of shape (max_frame, 17)
            teleop_hz (int, optional): The frequency to teleoperate
            leader-follower. Defaults to 100.

        Raises:
            ConnectionError: RakudaRobot is not connected. Call connect() first.
            RuntimeError: While the bilateral loop is running.
            ValueError: max_frame must be greater than 0.
            ValueError: Length of leader_action must match max_frame.
            ValueError: leader_action must be of shape (max_frame, 17).
            ValueError: Leader action length does not match number of leader motors.
        """
        if not self.is_connected:
            raise ConnectionError("RakudaRobot is not connected. Call connect() first.")
        self._reject_while_bilateral_active("send()")

        if max_frame <= 0:
            raise ValueError("max_frame must be greater than 0.")

        if len(leader_action) != max_frame:
            raise ValueError("Length of leader_action must match max_frame.")

        if leader_action.ndim != 2 or leader_action.shape[1] != 17:
            raise ValueError("leader_action must be of shape (max_frame, 17).")

        ramp_sent_count = self._send_initial_follower_ramp(
            leader_action[0],
            fps=fps,
            teleop_hz=teleop_hz,
        )

        interval = 1.0 / teleop_hz
        total_time = (max_frame - 1) / fps
        start_time = time.perf_counter()
        sent_count = 0

        try:
            while True:
                now = time.perf_counter()
                t = now - start_time
                if t > total_time:
                    break

                # Frame index for the current time
                idx_float = t * fps
                idx0 = int(np.floor(idx_float))
                idx1 = min(idx0 + 1, max_frame - 1)
                alpha = idx_float - idx0

                # Linear interpolation
                action = (1 - alpha) * leader_action[idx0] + alpha * leader_action[idx1]
                self.send_frame_action(action)
                sent_count += 1

                # busy wait
                next_time = start_time + sent_count * interval
                while time.perf_counter() < next_time:
                    pass

            # Send the last frame once more to be sure
            self.send_frame_action(leader_action[-1])

            total_sent_count = ramp_sent_count + sent_count
            elapsed = time.perf_counter() - start_time + (ramp_sent_count / teleop_hz)
            table = Table(title="Send Action Summary")
            table.add_column("Metric", style="cyan", no_wrap=True)
            table.add_column("Value", style="magenta")
            table.add_row("Original Frames (fps)", f"{max_frame} ({fps}Hz)")
            table.add_row(
                "Sent Frames (after interpolation)",
                f"{total_sent_count} ({total_sent_count / elapsed:.2f}Hz)",
            )
            table.add_row("Total Time (s)", f"{elapsed:.2f}")
            console = Console()
            console.print(table)
        except KeyboardInterrupt:
            logger.info("Send interrupted by user.")
            self._pair_sys.disconnect()
        except Exception as e:
            logger.error(f"An error occurred during send: {e}")
            raise e

    def send_frame_action(self, leader_action: NDArray[np.float32]) -> None:
        """Send one leader-ordered action to the follower.

        While the bilateral loop is running the pair system refuses the write
        with ``RuntimeError``.
        """
        self._pair_sys.send_follower_action(self._leader_action_to_follower_action(leader_action))

    def get_follower_frame_action(self) -> NDArray[np.float32]:
        """Return the current follower positions in follower motor order."""
        follower_positions = self._pair_sys.get_follower_action()
        follower_motor_names = list(self._pair_sys.follower.motors.motors.keys())
        return np.asarray(
            [follower_positions[name] for name in follower_motor_names],
            dtype=np.float32,
        )

    def send_follower_frame_action(self, follower_action: NDArray[np.float32]) -> None:
        """Send one action expressed directly in follower motor order.

        While the bilateral loop is running the pair system refuses the write
        with ``RuntimeError``.
        """
        action = np.asarray(follower_action, dtype=np.float32)
        follower_motor_names = list(self._pair_sys.follower.motors.motors.keys())
        if action.ndim != 1 or action.shape[0] != len(follower_motor_names):
            raise ValueError(
                f"follower_action must be a 1D array with {len(follower_motor_names)} elements."
            )
        if not np.all(np.isfinite(action)):
            raise ValueError("follower_action must contain only finite values.")

        self._pair_sys.send_follower_action(
            {name: float(action[i]) for i, name in enumerate(follower_motor_names)}
        )

    def _leader_action_to_follower_action(
        self, leader_action: NDArray[np.float32]
    ) -> Dict[str, float]:
        follower_action: Dict[str, float] = {}

        leader_motor_names = list(self._pair_sys.leader.motors.motors.keys())
        if len(leader_action) != len(leader_motor_names):
            raise ValueError(
                f"Leader action length {len(leader_action)} does not match "
                f"number of leader motors {len(leader_motor_names)}"
            )

        for i, motor_name in enumerate(leader_motor_names):
            follower_name = RAKUDA_MOTOR_MAPPING.get(motor_name)
            if follower_name is not None:
                follower_action[follower_name] = float(leader_action[i])

        return follower_action

    def _send_initial_follower_ramp(
        self,
        first_leader_action: NDArray[np.float32],
        fps: int,
        teleop_hz: int,
    ) -> int:
        current = self._pair_sys.get_follower_action()
        target = self._leader_action_to_follower_action(first_leader_action)
        steps = max(1, int(round(teleop_hz / fps)))
        interval = 1.0 / teleop_hz

        for step in range(1, steps + 1):
            alpha = step / steps
            action = {
                name: (1 - alpha) * float(current.get(name, value)) + alpha * value
                for name, value in target.items()
            }
            self._pair_sys.send_follower_action(action)
            time.sleep(interval)

        return steps

    def _init_config(self) -> RakudaSensorConfigs:
        """Initialize sensor configurations based on the provided robot configuration."""

        # if sensors config is provided in RakudaConfig, use it
        if self.config.sensors is not None:
            camera_params = self.config.sensors.cameras
            camera_configs: List[RealsenseCameraConfig] = []

            for cam_param in camera_params:
                came_cfg = RealsenseCameraConfig()
                came_cfg.name = cam_param.name
                came_cfg.width = cam_param.width
                came_cfg.height = cam_param.height
                came_cfg.fps = cam_param.fps
                came_cfg.index = cam_param.index  # Add index attribute

                camera_configs.append(came_cfg)

            tactile_configs: List[TactileParams] = []
            tactile_params = self.config.sensors.tactile
            for tac_param in tactile_params:
                tactile_configs.append(tac_param)

            audio_configs: List[AudioParams] = []
            audio_params = self.config.sensors.audio
            for audio_param in audio_params:
                audio_configs.append(audio_param)
        else:
            camera_configs = [RealsenseCameraConfig()]
            tactile_configs = []
            audio_configs = []

        sensor_configs = RakudaSensorConfigs(
            cameras=camera_configs, tactile=tactile_configs, audio=audio_configs
        )
        return sensor_configs

    def _init_sensors(self) -> Sensors:
        if self._sensor_configs is None:
            raise RuntimeError("Failed to initialize sensor configurations.")

        cameras: List[RealsenseCamera] = []
        for cam_cfg in self._sensor_configs.cameras:
            cam = RealsenseCamera(cam_cfg)
            cam.connect()
            cameras.append(cam)

        tactiles: List[DigitSensor] = []
        audios: List[AudioSensor] = []
        if self._sensor_configs.tactile is not None:
            for tac_cfg in self._sensor_configs.tactile:
                digit = DigitSensor(tac_cfg)
                digit.connect()
                tactiles.append(digit)

        if self._sensor_configs.audio is not None:
            for audio_cfg in self._sensor_configs.audio:
                audio = AudioSensor(audio_cfg)
                audio.connect()
                audios.append(audio)

        sensors = Sensors(cameras=cameras, tactile=tactiles, audio=audios)
        self._sensors = sensors

        table = Table(title="Initialized Sensors")
        table.add_column("Type", style="cyan", no_wrap=True)
        table.add_column("Name", style="magenta")
        table.add_column("Details", style="green")
        for cam in cameras:
            table.add_row("Camera", cam.name, repr(cam))
        for tac in tactiles:
            table.add_row("Tactile", tac.name, repr(tac))
        for audio in audios:
            table.add_row("Audio", audio.name, repr(audio))
        console = Console()
        console.print(table)

        return sensors

    @property
    def config(self) -> RakudaConfig:
        """The robot configuration as the pair system holds it.

        After ``connect()`` this names the ports actually opened: ``PORT_AUTO``
        is resolved by ``RakudaPairSys``, and ``metadata.json`` records the
        resolved ports through this property.
        """
        return self._pair_sys.config

    @property
    def sensor_configs(self) -> RakudaSensorConfigs:
        if self._sensor_configs is None:
            raise RuntimeError("Failed to initialize sensor configurations.")
        return self._sensor_configs

    @property
    def is_connected(self) -> bool:
        return self._pair_sys.is_connected

    @property
    def sensors(self) -> Sensors:
        if self._sensors is None:
            raise RuntimeError("Failed to initialize sensors.")
        return self._sensors

    @property
    def robot_system(self) -> RakudaPairSys:
        return self._pair_sys

    def __del__(self) -> None:
        try:
            self.disconnect()
        except Exception:
            pass
