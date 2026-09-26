"""Leader gravity identification CLI: range, sign-check, identify, fit, verify.

Every step that commands a current runs through :class:`LeaderCurrentLoop`
(preflight, configure, the per-cycle fault table, the hold on stop), so the
safety of the bilateral stack applies unchanged: a fault or Ctrl-C holds the
arm in position mode, and nothing here releases an arm without the operator
saying so.  The subcommands, in bring-up order:

* ``range``: both arms torque off; the reference pose (1 s average) and the
  measured min/max of every arm joint.
* ``sign-check``: the sign of ``GOAL_CURRENT`` per arm joint with
  displacement-terminated pulses, grouped by ``DRIVE_MODE``.
* ``identify --arm``: the operator places the arm, :class:`ManualPoseLaw`
  holds it from both sides and records the settled currents; the dataset is
  saved after every pose and fitted once there are enough poses.
* ``fit --arm``: (re)fits a saved dataset.
* ``verify --arm``: gravity compensation alone (no force feedback, no
  follower); the arm must not drift at ten operator-chosen poses.
* ``show``: prints the identification file.

The model itself (``PoseSample``, ``fit_arm``, the file format) lives in
``rakuda_gravity``, which must not import this module; so the CLI is this
module (``python -m robopy.robots.rakuda.rakuda_gravity_cli``) or the
``robopy-rakuda-gravity`` console script.  Run with the venv's SDK::

    env -u PYTHONPATH uv run --frozen robopy-rakuda-gravity range

Commands take a :class:`CliEnv` so tests can inject a scripted console, a
simulated bus and a runner that steps the loop on a simulated clock.
"""

from __future__ import annotations

import argparse
import hashlib
import logging
import math
import os
import signal
import sys
import threading
import time
from dataclasses import asdict, dataclass, field, replace
from datetime import datetime, timezone
from enum import Enum
from types import FrameType
from typing import Any, Callable, Dict, List, Literal, Mapping, Protocol, Sequence, Tuple, cast

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robopy.config.dotrobopy import apply_rakuda_dotconfig
from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    PORT_AUTO,
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_GRIPPER_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaArmState,
    RakudaBilateralParams,
    RakudaConfig,
)
from robopy.motor.dynamixel_bus import DiagnosticReading, DynamixelCommError
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, XControlTable

from .rakuda_arm import BusFactory
from .rakuda_control_laws import (
    BilateralLaw,
    CurrentShaper,
    IdentifyParams,
    LawOutput,
    SignCheckLaw,
    SignCheckParams,
    VelocityFilter,
    _check_dt,
    _joint_indices,
    _validate_state,
)
from .rakuda_gravity import (
    ARM_SIDES,
    MIN_FIT_SAMPLES,
    ArmGravityFit,
    GravityFile,
    LeaderGravityModel,
    PoseSample,
    arm_fit_from_dict,
    arm_fit_to_dict,
    arm_joints,
    fit_arm,
    load_dataset,
    load_gravity_file,
    save_dataset,
    save_gravity_file,
)
from .rakuda_leader import RakudaLeader
from .rakuda_leader_control import (
    BilateralError,
    ControlBus,
    LeaderCurrentLoop,
    LoopState,
    ports_command,
)
from .rakuda_pair_sys import STATE_READ_ATTEMPTS, STATE_READ_TIMEOUT_S
from .rakuda_ports import check_sdk_location, resolve_port

__all__ = [
    "DEFAULT_GRAVITY_FILE",
    "CliEnv",
    "CliError",
    "Console",
    "LoopRunner",
    "ManualPoseLaw",
    "ManualPoseParams",
    "ManualPoseStatus",
    "PoseMeasurement",
    "PoseState",
    "TerminalConsole",
    "ThreadedRunner",
    "build_parser",
    "check_sign_groups",
    "cmd_fit",
    "cmd_identify",
    "cmd_range",
    "cmd_show",
    "cmd_sign_check",
    "cmd_verify",
    "coverage",
    "decide_sign",
    "install_signal_handlers",
    "main",
    "provisional_seed_ma",
    "recommend_gravity_scale_step",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

#: Where the identification file lives by default (the bilateral loop reads it there).
DEFAULT_GRAVITY_FILE: str = RakudaBilateralParams.gravity_model_path

EXIT_OK = 0
#: A check failed, the loop faulted, or the operator stopped before the end.
EXIT_FAILED = 1
#: Bad arguments or a missing prerequisite (``range`` / ``sign-check`` not run).
EXIT_USAGE = 2
EXIT_INTERRUPT = 130

#: Reference pose average (``range``).
REFERENCE_AVERAGE_S = 1.0
#: Sampling rate of the ``range`` min/max tracking and the reference average.
RANGE_SAMPLE_HZ = 20.0
#: A reference average whose samples spread more than this is reported (arm moved).
REFERENCE_STILL_COUNTS = 20

#: ``sign-check``: joints whose position moved more than this over the check are reported.
SIGN_CHECK_RETURN_WARN_COUNTS = 40

#: ``identify``: every this-th pose is a validation pose.
VALIDATION_EVERY = 5
#: ``identify``: poses needed before the provisional model seeds the integrator.
PROVISIONAL_MIN_POSES = 12
#: ``identify``: nearest poses averaged by the provisional model.
PROVISIONAL_NEIGHBOURS = 4
#: ``identify``: suggest a pause when a motor of the arm reaches this temperature.
PAUSE_TEMPERATURE_C = 60.0
#: ``identify``: bins per joint of the coverage display.
COVERAGE_BINS = 10
#: ``identify``: a pose that has not finished after this long is abandoned.
POSE_TIMEOUT_S = 60.0

#: ``verify``: defaults and the pass limit.
VERIFY_POSES = 10
VERIFY_SECONDS = 5.0
VERIFY_DRIFT_LIMIT_COUNTS = 34
#: ``verify``: size of the recommended ``gravity_scale`` change.
GRAVITY_SCALE_STEP = 0.05
#: ``verify``: joint-pose pairs below these are ignored by the scale recommendation.
RECOMMEND_MIN_GRAVITY_MA = 20.0
RECOMMEND_MIN_DRIFT_COUNTS = 8.0

#: What happens at the end of a command that leaves joints held.
EndAction = Literal["ask", "keep", "release"]

_EPS = 1e-9


# --- the manual-pose law ----------------------------------------------------------


@dataclass(frozen=True)
class ManualPoseParams:
    """Timing and tolerances of :class:`ManualPoseLaw`.

    Attributes:
        delta_counts: Offset ``delta`` of the two approach points ``p ± delta``.
        goal_rate_counts_per_s: Ramp rate of the goal between ``p`` and ``p ± delta``.
        capture_int_window_counts: Integrator window while capturing (wider than
            ``IdentifyParams.id_int_error_counts`` so the hold recovers from the
            sag right after the operator lets go).
        settle_error_counts: Settled when every ``|q_goal - q|`` is at most this ...
        settle_velocity_vcounts: ... and every ``|v|`` at most this ...
        settle_s: ... for this long without interruption.
        settle_timeout_s: A settle phase longer than this rejects the pose.
        measure_s: Averaging window of the commanded current.
        saturation_s: A hold current at ``±id_current_max_ma`` this long rejects the pose.
        measure_std_flag_ma: A current std above this in a window flags the pose.
    """

    delta_counts: float = 57.0
    goal_rate_counts_per_s: float = 150.0
    capture_int_window_counts: float = 150.0
    settle_error_counts: float = 8.0
    settle_velocity_vcounts: float = 2.0
    settle_s: float = 0.5
    settle_timeout_s: float = 6.0
    measure_s: float = 0.5
    saturation_s: float = 0.5
    measure_std_flag_ma: float = 8.0

    def __post_init__(self) -> None:
        for f in (
            "delta_counts",
            "goal_rate_counts_per_s",
            "capture_int_window_counts",
            "settle_error_counts",
            "settle_velocity_vcounts",
            "settle_s",
            "settle_timeout_s",
            "measure_s",
            "saturation_s",
            "measure_std_flag_ma",
        ):
            value = getattr(self, f)
            if isinstance(value, bool) or not isinstance(value, (int, float)):
                raise ValueError(f"ManualPoseParams.{f} must be a number, got {value!r}")
            if not math.isfinite(value) or value <= 0:
                raise ValueError(f"ManualPoseParams.{f} must be > 0, got {value!r}")
        if self.settle_timeout_s <= self.settle_s:
            raise ValueError("ManualPoseParams.settle_timeout_s must exceed settle_s")


class PoseState(Enum):
    """States of :class:`ManualPoseLaw`."""

    FREE = "free"
    CAPTURE = "capture"
    UP = "up"
    RETURN_PLUS = "return_plus"
    SETTLE_PLUS = "settle_plus"
    MEASURE_PLUS = "measure_plus"
    DOWN = "down"
    RETURN_MINUS = "return_minus"
    SETTLE_MINUS = "settle_minus"
    MEASURE_MINUS = "measure_minus"
    DONE = "done"
    REJECTED = "rejected"


#: The measurement sequence from ``capture()`` to ``DONE``.
_SEQUENCE: Tuple[PoseState, ...] = (
    PoseState.CAPTURE,
    PoseState.UP,
    PoseState.RETURN_PLUS,
    PoseState.SETTLE_PLUS,
    PoseState.MEASURE_PLUS,
    PoseState.DOWN,
    PoseState.RETURN_MINUS,
    PoseState.SETTLE_MINUS,
    PoseState.MEASURE_MINUS,
    PoseState.DONE,
)
_NEXT: Dict[PoseState, PoseState] = dict(zip(_SEQUENCE[:-1], _SEQUENCE[1:]))
_MEASURING = frozenset(_SEQUENCE[:-1])
_SETTLING = frozenset({PoseState.CAPTURE, PoseState.SETTLE_PLUS, PoseState.SETTLE_MINUS})
_RAMPING = frozenset({PoseState.UP, PoseState.RETURN_PLUS, PoseState.DOWN, PoseState.RETURN_MINUS})


@dataclass(frozen=True)
class PoseMeasurement:
    """The result of one completed pose, in ``ManualPoseLaw.joints`` order (joint convention).

    Attributes:
        q_counts: Mean position over both averaging windows (the held pose).
        i_plus_ma: Mean commanded current at ``p`` after approaching from ``p + delta``.
        i_minus_ma: The same after approaching from ``p - delta``.
        std_plus_ma: Standard deviation of the commanded current in the ``+`` window.
        std_minus_ma: The same for the ``-`` window.
        flags: ``"noisy_plus"``/``"noisy_minus"`` (std above the flag limit) and
            ``"moved_plus"``/``"moved_minus"`` (left the settle band while averaging).
    """

    q_counts: FloatArray
    i_plus_ma: FloatArray
    i_minus_ma: FloatArray
    std_plus_ma: FloatArray
    std_minus_ma: FloatArray
    flags: Tuple[str, ...]


@dataclass(frozen=True)
class ManualPoseStatus:
    """A consistent snapshot of :class:`ManualPoseLaw` for the CLI (thread-safe copy).

    Attributes:
        state: The current state.
        progress: ``0..1`` through the measurement sequence (0 in ``FREE``).
        state_elapsed_s: Time spent in ``state`` (sum of ``dt``).
        error_counts: ``q_goal - q`` of the last cycle (zeros in ``FREE``).
        current_ma: The last commanded current.
        measurement: The result once ``DONE``, else ``None``.
        reason: Why the pose was rejected (``"settle_timeout"``, ``"saturated"``, or
            ``"near_range_end: <joints>"``), else ``None``.
    """

    state: PoseState
    progress: float
    state_elapsed_s: float
    error_counts: FloatArray
    current_ma: FloatArray
    measurement: PoseMeasurement | None
    reason: str | None


class _Window:
    """Running sums of one averaging window."""

    def __init__(self, n: int) -> None:
        self.count = 0
        self.current = np.zeros(n)
        self.current_sq = np.zeros(n)
        self.position = np.zeros(n)
        self.moved = False

    def add(self, current: FloatArray, position: FloatArray, moved: bool) -> None:
        self.count += 1
        self.current += current
        self.current_sq += current * current
        self.position += position
        self.moved = self.moved or moved

    def mean(self) -> Tuple[FloatArray, FloatArray, FloatArray]:
        """``(mean current, current std, mean position)``."""
        n = max(self.count, 1)
        mean = self.current / n
        variance = np.maximum(self.current_sq / n - mean * mean, 0.0)
        return mean, np.sqrt(variance), self.position / n


class ManualPoseLaw:
    """Holds an operator-placed pose from both sides and records the currents.

    State machine (every timer is the sum of the loop's ``dt``; no wall clock)::

        FREE --capture()--> CAPTURE --settle--> UP --> RETURN_PLUS --> SETTLE_PLUS
          --> MEASURE_PLUS --> DOWN --> RETURN_MINUS --> SETTLE_MINUS --> MEASURE_MINUS --> DONE

    * ``FREE``: the output is rate-limited toward 0 mA (the operator carries the arm).
    * ``CAPTURE``: the goal ``p`` is the position at the first cycle after
      :meth:`capture`; PD + integrator as :class:`PoseHoldLaw`, but with the
      integrator window widened to ``capture_int_window_counts`` and the
      integrator seeded with ``seed_ma`` (the provisional model).
    * ``UP``/``DOWN`` ramp the goal to ``p ± delta`` and ``RETURN_*`` back to ``p``
      at ``goal_rate_counts_per_s``, so ``p`` is reached once from above and
      once from below; static friction then biases the two settled currents
      in opposite directions and their mean is the gravity current.
    * Settled: every ``|e| <= settle_error_counts`` and ``|v| <= settle_velocity_vcounts``
      for ``settle_s``; ``MEASURE_*`` averages the commanded current for ``measure_s``.
    * ``REJECTED``: a settle phase exceeded ``settle_timeout_s``, the hold
      current sat at ``±id_current_max_ma`` for ``saturation_s``, or (at once,
      on the first cycle) ``p ± delta`` of some joint leaves its recorded
      range, since the approach would drive it past the operator's safe end.
      Like ``DONE`` it keeps holding at ``p`` until :meth:`release_to_free`.

    :meth:`capture`, :meth:`release_to_free` and :meth:`status` may be called
    from any thread while the control thread runs :meth:`compute`.

    Args:
        joints: The arm's joints, in ``RAKUDA_JOINT_NAMES`` order.
        params: Hold gains, clamp and rate limit (``IdentifyParams``).
        joint_range_counts: ``{joint: (lo, hi)}`` from ``range``; exposed as
            :attr:`joint_range_counts` so the loop's ``joint_out_of_range``
            fault applies.
        manual: Timing and tolerances.

    Raises:
        ValueError: Unknown or unordered joints, or a missing/invalid range.
    """

    joints: Tuple[str, ...]

    def __init__(
        self,
        joints: Sequence[str],
        params: IdentifyParams,
        joint_range_counts: Mapping[str, Tuple[int, int]],
        manual: ManualPoseParams | None = None,
    ) -> None:
        self.joints, self._idx = _joint_indices(joints)
        self._params = params
        self._manual = ManualPoseParams() if manual is None else manual
        ranges: Dict[str, Tuple[int, int]] = {}
        for name in self.joints:
            if name not in joint_range_counts:
                raise ValueError(f"joint_range_counts has no entry for {name}")
            lo, hi = (int(v) for v in joint_range_counts[name])
            if lo >= hi:
                raise ValueError(f"joint_range_counts[{name!r}] must have lo < hi, got {lo, hi}")
            ranges[name] = (lo, hi)
        self.joint_range_counts: Dict[str, Tuple[int, int]] = ranges
        self._lo = np.array([ranges[name][0] for name in self.joints], dtype=np.float64)
        self._hi = np.array([ranges[name][1] for name in self.joints], dtype=np.float64)
        n = len(self.joints)
        self._zeros = np.zeros(n, dtype=np.float64)
        self._filter = VelocityFilter(params.velocity_filter_hz)
        self._shaper = CurrentShaper(params.id_current_rate_ma_per_s, params.id_current_max_ma)
        self._lock = threading.Lock()
        self._ready = False
        self._state = PoseState.FREE
        self._pose: FloatArray | None = None
        self._goal: FloatArray | None = None
        self._ramp_target: FloatArray | None = None
        self._ramp_done = False
        self._i_int = np.zeros(n, dtype=np.float64)
        self._error = np.zeros(n, dtype=np.float64)
        self._t_state = 0.0
        self._t_settled = 0.0
        self._t_saturated = 0.0
        self._window: _Window | None = None
        self._plus: Tuple[FloatArray, FloatArray, FloatArray, bool] | None = None
        self._measurement: PoseMeasurement | None = None
        self._reason: str | None = None
        self._rejected_progress = 0.0

    # -- commands (any thread) --------------------------------------------------

    @property
    def state(self) -> PoseState:
        with self._lock:
            return self._state

    def capture(self, seed_ma: ArrayLike | None = None) -> None:
        """Starts measuring the pose the arm is in at the next cycle.

        Args:
            seed_ma: Initial integrator value per joint (``(J,)`` mA, joint
                convention), typically the provisional model's prediction;
                clipped to ``±id_current_max_ma``.  ``None`` starts from 0.

        Raises:
            RuntimeError: Before the loop's ``configure()`` (``reset``) or while
                a measurement is running.
            ValueError: ``seed_ma`` of the wrong shape or not finite.
        """
        seed = self._zeros.copy()
        if seed_ma is not None:
            seed = np.asarray(seed_ma, dtype=np.float64)
            if seed.shape != self._zeros.shape or not bool(np.all(np.isfinite(seed))):
                raise ValueError(f"seed_ma must be finite with shape {self._zeros.shape}")
        limit = self._params.id_current_max_ma
        with self._lock:
            if not self._ready:
                raise RuntimeError("ManualPoseLaw.reset() must run before capture()")
            if self._state in _MEASURING:
                raise RuntimeError(f"a measurement is running ({self._state.value})")
            self._clear()
            self._i_int = np.clip(seed, -limit, limit).astype(np.float64)
            self._enter(PoseState.CAPTURE)

    def release_to_free(self) -> None:
        """Back to ``FREE`` from any state: the output ramps to 0 (support the arm first)."""
        with self._lock:
            self._clear()
            self._enter(PoseState.FREE)

    def status(self) -> ManualPoseStatus:
        """A consistent copy of the state for display and for building the ``PoseSample``."""
        with self._lock:
            current = self._shaper.current_ma
            if self._state is PoseState.REJECTED:
                progress = self._rejected_progress
            else:
                progress = self._progress(self._state)
            return ManualPoseStatus(
                state=self._state,
                progress=progress,
                state_elapsed_s=self._t_state,
                error_counts=self._error.copy(),
                current_ma=self._zeros.copy() if current is None else current,
                measurement=self._measurement,
                reason=self._reason,
            )

    # -- ControlLaw ---------------------------------------------------------------

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        """``FREE`` at 0 mA (``configure()`` torques the arm on limp)."""
        _validate_state(leader, "leader")
        with self._lock:
            self._filter.reset(leader.velocity[self._idx])
            self._shaper.reset(self._zeros)
            self._clear()
            self._enter(PoseState.FREE)
            self._ready = True

    def engage(self) -> None:
        """No-op: there is no follower feedback to ramp."""

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """Zero: the operator supports the arm when the loop switches it to current mode."""
        _validate_state(leader, "leader")
        return self._zeros.copy()

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """One cycle; see the class docstring.  ``follower`` is ignored.

        Raises:
            RuntimeError: :meth:`reset` has not run.
            ValueError: Invalid snapshot or ``dt``.
        """
        dt = _check_dt(dt)
        _validate_state(leader, "leader")
        with self._lock:
            if not self._ready:
                raise RuntimeError("ManualPoseLaw.reset() must run before compute()")
            q = leader.position[self._idx].astype(np.float64)
            v = leader.velocity[self._idx].astype(np.float64)
            v_hat = self._filter.update(v, dt)
            self._t_state += dt
            if self._state is PoseState.FREE:
                self._error = self._zeros.copy()
                current = self._shaper.shape(self._zeros, dt)
                return self._output(current, self._zeros)
            if self._goal is None:
                # First cycle after capture(): the pose is where the arm is now.
                self._pose = q.copy()
                self._goal = q.copy()
                delta = self._manual.delta_counts
                near = (q + delta > self._hi) | (q - delta < self._lo)
                if bool(np.any(near)):
                    names = [name for name, bad in zip(self.joints, near) if bad]
                    self._reject("near_range_end: " + ", ".join(names))
            self._advance_goal(dt)
            e = self._goal - q
            i_pid = self._pid(e, v_hat, dt)
            current = self._shaper.shape(i_pid, dt)
            self._error = e
            self._update(q, v, e, current, dt)
            return self._output(current, i_pid)

    # -- internals (called with the lock held) ------------------------------------

    def _output(self, current: FloatArray, feedback: FloatArray) -> LawOutput:
        return LawOutput(
            current_ma=current,
            gravity_ma=self._zeros.copy(),
            barrier_ma=self._zeros.copy(),
            feedback_ma=np.asarray(feedback, dtype=np.float64).copy(),
            gate=1.0,
            ramp=1.0,
        )

    def _clear(self) -> None:
        self._pose = None
        self._goal = None
        self._ramp_target = None
        self._i_int = self._zeros.copy()
        self._t_saturated = 0.0
        self._window = None
        self._plus = None
        self._measurement = None
        self._reason = None

    def _progress(self, state: PoseState) -> float:
        if state not in _SEQUENCE:
            return 0.0
        return _SEQUENCE.index(state) / (len(_SEQUENCE) - 1)

    def _enter(self, state: PoseState) -> None:
        self._state = state
        self._t_state = 0.0
        self._t_settled = 0.0
        pose = self._pose
        delta = self._manual.delta_counts
        if pose is None or (state not in _RAMPING and state is not PoseState.REJECTED):
            self._ramp_target = None
        elif state is PoseState.UP:
            self._ramp_target = pose + delta
        elif state is PoseState.DOWN:
            self._ramp_target = pose - delta
        else:
            self._ramp_target = pose.copy()
        self._ramp_done = self._ramp_target is None
        if state in (PoseState.MEASURE_PLUS, PoseState.MEASURE_MINUS):
            self._window = _Window(len(self.joints))

    def _reject(self, reason: str) -> None:
        self._rejected_progress = self._progress(self._state)
        self._reason = reason
        self._enter(PoseState.REJECTED)
        logger.info("pose rejected: %s", reason)

    def _advance_goal(self, dt: float) -> None:
        if self._ramp_target is None or self._goal is None:
            return
        step = self._manual.goal_rate_counts_per_s * dt
        diff = self._ramp_target - self._goal
        close = np.abs(diff) <= step
        self._goal = np.where(close, self._ramp_target, self._goal + np.sign(diff) * step)
        self._ramp_done = bool(np.all(close))

    def _pid(self, e: FloatArray, v_hat: FloatArray, dt: float) -> FloatArray:
        p = self._params
        window = (
            self._manual.capture_int_window_counts
            if self._state is PoseState.CAPTURE
            else float(p.id_int_error_counts)
        )
        abs_e = np.abs(e)
        integrate = (abs_e > p.id_int_deadband_counts) & (abs_e <= window) & ~self._shaper.saturated
        self._i_int = np.clip(
            self._i_int + np.where(integrate, p.Ki_s * e * dt, 0.0),
            -p.id_current_max_ma,
            p.id_current_max_ma,
        ).astype(np.float64)
        return (self._i_int + p.Kp_s * e - p.Kd_s * v_hat).astype(np.float64)

    def _update(
        self, q: FloatArray, v: FloatArray, e: FloatArray, current: FloatArray, dt: float
    ) -> None:
        m = self._manual
        state = self._state
        if state in _MEASURING:
            if bool(np.any(self._shaper.saturated)):
                self._t_saturated += dt
                if self._t_saturated >= m.saturation_s - _EPS:
                    self._reject("saturated")
                    return
            else:
                self._t_saturated = 0.0
        if state in _SETTLING:
            settled = bool(
                np.all(np.abs(e) <= m.settle_error_counts)
                and np.all(np.abs(v) <= m.settle_velocity_vcounts)
            )
            self._t_settled = self._t_settled + dt if settled else 0.0
            if self._t_settled >= m.settle_s - _EPS:
                self._enter(_NEXT[state])
            elif self._t_state >= m.settle_timeout_s - _EPS:
                self._reject("settle_timeout")
        elif state in _RAMPING:
            if self._ramp_done:
                self._enter(_NEXT[state])
        elif state in (PoseState.MEASURE_PLUS, PoseState.MEASURE_MINUS):
            assert self._window is not None
            moved = bool(np.any(np.abs(e) > m.settle_error_counts))
            self._window.add(current, q, moved)
            if self._t_state >= m.measure_s - _EPS:
                self._finish_window(state)

    def _finish_window(self, state: PoseState) -> None:
        assert self._window is not None
        mean, std, position = self._window.mean()
        moved = self._window.moved
        if state is PoseState.MEASURE_PLUS:
            self._plus = (mean, std, position, moved)
            self._enter(PoseState.DOWN)
            return
        assert self._plus is not None
        plus_mean, plus_std, plus_position, plus_moved = self._plus
        flags: List[str] = []
        limit = self._manual.measure_std_flag_ma
        if bool(np.any(plus_std > limit)):
            flags.append("noisy_plus")
        if bool(np.any(std > limit)):
            flags.append("noisy_minus")
        if plus_moved:
            flags.append("moved_plus")
        if moved:
            flags.append("moved_minus")
        self._measurement = PoseMeasurement(
            q_counts=(plus_position + position) / 2.0,
            i_plus_ma=plus_mean,
            i_minus_ma=mean,
            std_plus_ma=plus_std,
            std_minus_ma=std,
            flags=tuple(flags),
        )
        self._enter(PoseState.DONE)


# --- the sign-check pulse sequencer ---------------------------------------------------


class _PulseLaw:
    """``SignCheckLaw`` pulses on one joint, one at a time, 0 mA in between.

    The loop is configured once per joint; :meth:`start_pulse` (from the CLI
    thread) arms the next pulse, which starts from the position of the next
    cycle.  A stopped pulse keeps checking ``pulse_abort_counts`` until the
    next one starts, so an overshoot after the cut still faults the loop.
    """

    joints: Tuple[str, ...]

    def __init__(self, joint: str, params: SignCheckParams, joint_range: Tuple[int, int]) -> None:
        self.joints, self._idx = _joint_indices((joint,))
        self.joint_range_counts: Dict[str, Tuple[int, int]] = {
            joint: (int(joint_range[0]), int(joint_range[1]))
        }
        self._params = params
        self._lock = threading.Lock()
        self._pending: float | None = None
        self._pulse: SignCheckLaw | None = None
        self._last_out = 0.0
        self._level_ma: float | None = None
        # PRESENT_CURRENT sign agreements of the current level, keyed by the
        # direction of the commanded current (+1 / -1).
        self._signs: Dict[int, List[bool]] = {1: [], -1: []}

    def start_pulse(self, level_ma: float) -> None:
        """Arms a pulse of ``level_ma`` (signed) for the next cycle."""
        with self._lock:
            self._pending = float(level_ma)
            self._pulse = None
            if self._level_ma != abs(float(level_ma)):
                self._level_ma = abs(float(level_ma))
                self._signs = {1: [], -1: []}

    def pulse_finished(self) -> bool:
        """The armed pulse has run and ended (its output is 0)."""
        with self._lock:
            return self._pending is None and (self._pulse is None or self._pulse.stopped)

    def displacement(self) -> int:
        """``q - q_start`` of the current pulse at the last cycle (0 before any pulse)."""
        with self._lock:
            return 0 if self._pulse is None else self._pulse.displacement

    def present_current_matches(self) -> Tuple[bool | None, bool | None]:
        """``(plus, minus)``: did ``PRESENT_CURRENT`` have the commanded sign in each pulse?

        Covers both pulses of the current level; ``None`` for a pulse whose
        commanded and present currents were never both non-zero.
        """
        with self._lock:
            plus, minus = (self._signs[k] for k in (1, -1))
            return (all(plus) if plus else None, all(minus) if minus else None)

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        _validate_state(leader, "leader")
        with self._lock:
            self._pending = None
            self._pulse = None
            self._last_out = 0.0

    def engage(self) -> None:
        """No-op."""

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """Zero: the joint hangs; torque comes on at 0 mA."""
        _validate_state(leader, "leader")
        return np.zeros(1, dtype=np.float64)

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """Runs the armed pulse (``SignCheckOvershoot`` propagates to the loop)."""
        _validate_state(leader, "leader")
        with self._lock:
            present = float(leader.current_ma[self._idx[0]])
            if self._last_out != 0.0 and present != 0.0:
                direction = 1 if self._last_out > 0 else -1
                self._signs[direction].append((present > 0) == (self._last_out > 0))
            if self._pending is not None:
                self._pulse = SignCheckLaw(self.joints[0], self._pending, self._params)
                self._pulse.reset(leader)
                self._pending = None
            if self._pulse is None:
                zeros = np.zeros(1, dtype=np.float64)
                self._last_out = 0.0
                return LawOutput(zeros, zeros.copy(), zeros.copy(), zeros.copy(), 1.0, 1.0)
            output = self._pulse.compute(leader, follower, follower_age_s, dt)
            self._last_out = float(output.current_ma[0])
            return output


# --- console and runner ----------------------------------------------------------------


class Console(Protocol):
    """The operator's terminal, injectable for tests."""

    def ask(self, prompt: str) -> str:
        """Shows ``prompt`` and returns the typed line (``"q"`` at end of input)."""
        ...

    def say(self, text: str) -> None:
        """Prints ``text``."""
        ...

    def ask_while(self, prompt: str, tick: Callable[[], None], period_s: float) -> str:
        """Like :meth:`ask`, calling ``tick()`` about every ``period_s`` until answered."""
        ...


class TerminalConsole:
    """:class:`Console` on stdin/stdout; ``ask_while`` ticks from a helper thread."""

    def ask(self, prompt: str) -> str:
        try:
            return input(prompt)
        except EOFError:
            return "q"

    def say(self, text: str) -> None:
        print(text, flush=True)

    def ask_while(self, prompt: str, tick: Callable[[], None], period_s: float) -> str:
        """The main thread waits in ``input()`` (so Ctrl-C lands there); a helper ticks."""
        stop = threading.Event()
        errors: List[BaseException] = []

        def worker() -> None:
            while not stop.is_set():
                try:
                    tick()
                except BaseException as exc:
                    errors.append(exc)
                    return
                stop.wait(period_s)

        thread = threading.Thread(target=worker, name="rakuda-gravity-sampler", daemon=True)
        thread.start()
        try:
            answer = self.ask(prompt)
        finally:
            stop.set()
            thread.join()
        if errors:
            raise errors[0]
        return answer


class LoopRunner(Protocol):
    """Drives a configured :class:`LeaderCurrentLoop` (a thread, or cycles on a sim clock)."""

    def start(self, loop: LeaderCurrentLoop) -> None:
        """Starts running the configured loop."""
        ...

    def run_until(
        self,
        loop: LeaderCurrentLoop,
        done: Callable[[], bool],
        timeout_s: float | None,
        *,
        tick: Callable[[], None] | None = None,
        tick_s: float = 0.25,
    ) -> bool:
        """Lets the loop run until ``done()``; False on timeout or when the loop stopped."""
        ...

    def stop(self, loop: LeaderCurrentLoop) -> bool:
        """Stops the loop and holds (``LeaderCurrentLoop.stop``)."""
        ...


class ThreadedRunner:
    """:class:`LoopRunner` of the real CLI: the loop's own control thread, polled here."""

    def __init__(self, poll_s: float = 0.02) -> None:
        self.poll_s = poll_s

    def start(self, loop: LeaderCurrentLoop) -> None:
        loop.start()

    def run_until(
        self,
        loop: LeaderCurrentLoop,
        done: Callable[[], bool],
        timeout_s: float | None,
        *,
        tick: Callable[[], None] | None = None,
        tick_s: float = 0.25,
    ) -> bool:
        t0 = time.monotonic()
        next_tick = t0
        while True:
            if done():
                return True
            if not loop.running:
                return False
            now = time.monotonic()
            if timeout_s is not None and now - t0 >= timeout_s:
                return False
            if tick is not None and now >= next_tick:
                tick()
                next_tick = now + tick_s
            time.sleep(self.poll_s)

    def stop(self, loop: LeaderCurrentLoop) -> bool:
        return loop.stop()


# --- environment ------------------------------------------------------------------


@dataclass
class CliEnv:
    """What a command needs besides its arguments; tests inject the simulated pieces.

    Attributes:
        console: The operator's terminal.
        bus_factory: ``RakudaLeader`` bus factory (``None``: the real ``DynamixelBus``).
        runner: How loops are driven.
        clock: ``time.monotonic_ns``-compatible clock shared with the loops.
        sleep: ``time.sleep``-compatible sleep (the ``range`` sampling, the loops' preflight).
        params: Loop parameters (``.robopy/rakuda/config.yaml`` in the real CLI);
            each command replaces ``current_joints`` (and ``verify`` the feedback gains).
    """

    console: Console
    bus_factory: BusFactory | None = None
    runner: LoopRunner = field(default_factory=ThreadedRunner)
    clock: Callable[[], int] = time.monotonic_ns
    sleep: Callable[[float], None] = time.sleep
    params: RakudaBilateralParams = field(default_factory=RakudaBilateralParams)


class CliError(Exception):
    """A command cannot proceed; ``main`` prints the message and exits with ``exit_code``."""

    def __init__(self, message: str, exit_code: int = EXIT_USAGE) -> None:
        super().__init__(message)
        self.exit_code = exit_code


class _DiagnosticsTap:
    """The leader bus as the loop sees it, keeping the temperatures its reads return.

    While the loop runs it is the only user of the bus, so the
    temperature recorded with each pose comes from the loop's own 1 Hz
    ``read_diagnostics`` (and the preflight's), never from a second reader.
    Every other attribute is the bus's own.
    """

    def __init__(self, bus: ControlBus) -> None:
        self._bus = bus
        self._lock = threading.Lock()
        self._temperature: Dict[str, float] = {}

    def __getattr__(self, name: str) -> Any:
        return getattr(self._bus, name)

    def read_diagnostics(
        self, motor_names: Sequence[str], *, timeout_s: float = 0.05
    ) -> Dict[str, DiagnosticReading]:
        readings = self._bus.read_diagnostics(motor_names, timeout_s=timeout_s)
        with self._lock:
            self._temperature.update({n: float(r.temperature_c) for n, r in readings.items()})
        return readings

    def temperatures(self, names: Sequence[str]) -> FloatArray:
        """Last temperature per joint (NaN when never read)."""
        with self._lock:
            return np.array([self._temperature.get(n, math.nan) for n in names], dtype=np.float64)


# --- shared helpers -----------------------------------------------------------------


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def _is(answer: str, letter: str) -> bool:
    return answer.strip().lower() == letter


def _load_file(path: str) -> GravityFile:
    """The identification file, or ``CliError`` telling the operator to run ``range``."""
    try:
        return load_gravity_file(path)
    except FileNotFoundError as exc:
        raise CliError(f"no identification file at {path}; run `range` first") from exc


def _load_or_new(path: str) -> GravityFile:
    try:
        return load_gravity_file(path)
    except FileNotFoundError:
        return GravityFile()


def _require(data: GravityFile, joints: Sequence[str], path: str, *tables: str) -> None:
    """Every joint must have an entry in every named table of the file."""
    hints = {
        "reference_counts": "range",
        "joint_range_counts": "range",
        "current_sign": "sign-check",
        "drive_mode": "sign-check",
    }
    for table in tables:
        missing = [name for name in joints if name not in getattr(data, table)]
        if missing:
            raise CliError(f"{path} has no {table} for {missing}; run `{hints[table]}` first")


def _connect_leader(env: CliEnv, port: str) -> RakudaLeader:
    """Opens the leader with every arm joint left as found at connect.

    ``current_joints`` is all twelve arm joints so that ``connect()`` switches
    none of them off; each command then decides explicitly (after asking)
    which arm to release and which joints the loop drives.
    """
    if port == PORT_AUTO:
        port = resolve_port(PORT_AUTO, "leader")
    params = replace(env.params, current_joints=RAKUDA_ARM_JOINT_NAMES, allow_uncompensated=True)
    config = RakudaConfig(leader_port=port, follower_port="unused", bilateral=params)
    leader = RakudaLeader(config, env.bus_factory)
    leader.connect()
    return leader


def _make_loop(
    env: CliEnv,
    bus: ControlBus,
    joints: Sequence[str],
    law: Any,
    current_sign: Mapping[str, int],
    params: RakudaBilateralParams | None = None,
) -> LeaderCurrentLoop:
    """A leader-only loop on ``joints`` with the env's clock (grippers held as in teleop)."""
    params = replace(env.params if params is None else params, current_joints=tuple(joints))
    units = {name: CURRENT_UNIT_MA[bus.motors[name].model_name] for name in joints}
    return LeaderCurrentLoop(
        bus,
        joints,
        params,
        law,
        units,
        {name: int(current_sign[name]) for name in joints},
        None,
        clock=env.clock,
        sleep=env.sleep,
        gripper_hold={name: LEADER_GRIP_HOLD_POSITION for name in RAKUDA_GRIPPER_JOINT_NAMES},
    )


def _check_drive_mode(
    recorded: Mapping[str, int], preflight: Mapping[str, Any], joints: Sequence[str]
) -> None:
    """``DRIVE_MODE`` must be what the sign-check saw (a change may flip the sign)."""
    found = preflight.get("drive_mode", {})
    changed = {
        name: (recorded.get(name), found.get(name))
        for name in joints
        if recorded.get(name) != found.get(name)
    }
    if changed:
        raise CliError(
            f"DRIVE_MODE changed since sign-check (recorded, now): {changed}; re-run `sign-check`",
            EXIT_FAILED,
        )


def _release_other_arm(env: CliEnv, bus: ControlBus, side: str) -> bool:
    """Switches the other arm off (after asking) so it hangs; False when the operator cancels."""
    other = "left" if side == "right" else "right"
    joints = list(arm_joints(other))
    torque = bus.sync_read(XControlTable.TORQUE_ENABLE, joints)
    on = [name for name in joints if int(torque.get(name, 0)) == 1]
    if not on:
        return True
    answer = env.console.ask(
        f"The {other} arm is holding ({len(on)} joints); it will be switched OFF. "
        "Support it or let it hang, then press Enter (q: cancel): "
    )
    if _is(answer, "q"):
        return False
    bus.torque_disabled(on)
    return True


def _pause(
    env: CliEnv,
    loop: LeaderCurrentLoop,
    seconds: float,
    tick: Callable[[], None] | None = None,
    tick_s: float = 0.25,
) -> bool:
    """Lets the loop run for ``seconds`` on ``env.clock``; False if it stopped meanwhile."""
    deadline_ns = env.clock() + int(round(seconds * 1e9))
    return env.runner.run_until(
        loop, lambda: env.clock() >= deadline_ns, None, tick=tick, tick_s=tick_s
    )


def _stop_loop(env: CliEnv, loop: LeaderCurrentLoop) -> bool:
    """Stops (holds) the loop and reports a fault or an unverified hold.

    The stop cannot be interrupted: a Ctrl-C or a converted SIGTERM/SIGHUP
    arriving while it waits for the hold only repeats the (idempotent) stop,
    and the first such interrupt is re-raised once the hold has finished.
    Otherwise the caller's ``disconnect`` could close the port while the
    control thread is between the torque-off and torque-on of ``hold_joints``.
    """
    interrupt: BaseException | None = None
    while True:
        try:
            held = env.runner.stop(loop)
            break
        except (KeyboardInterrupt, SystemExit) as exc:
            if interrupt is None:
                interrupt = exc
                logger.warning("interrupted while holding the arm; finishing the hold first")
    fault = loop.fault
    if fault is not None:
        env.console.say(f"loop fault: {fault.reason}: {fault.detail} (joints held)")
    if not held and loop.state not in (LoopState.IDLE, LoopState.RELEASED):
        env.console.say(
            "CRITICAL: the hold could not be verified; support the arm and check "
            f"{ports_command('show', loop.port_name, 'leader')}"
        )
    if interrupt is not None:
        raise interrupt
    return held


def _end_session(env: CliEnv, loop: LeaderCurrentLoop, end: EndAction) -> None:
    """After the stop: keep holding, or release the loop's joints.

    ``"release"`` still asks the operator to support the arm at that moment
    (``q`` keeps holding): the arm is held in its last, often raised, pose.
    """
    if loop.state not in (LoopState.HELD, LoopState.FAULT):
        return
    if end == "keep":
        release = False
    elif end == "release":
        answer = env.console.ask(
            "Support the arm and press Enter to release it (q: keep holding): "
        )
        release = not _is(answer, "q")
    else:
        answer = env.console.ask("Enter: keep holding / r: release the arm (support it first): ")
        release = _is(answer, "r")
    if release:
        loop.release()
        env.console.say(f"released {list(loop.joints)} (torque off)")
    else:
        env.console.say(
            "the arm keeps holding; "
            f"{ports_command('release', loop.port_name, 'leader')} releases it"
        )


def _fmt(value: Any) -> str:
    """Compact text of a report value."""
    if isinstance(value, float):
        return f"{value:.3g}"
    if isinstance(value, Mapping):
        return ", ".join(f"{k}={_fmt(v)}" for k, v in value.items())
    if isinstance(value, (list, tuple)):
        return "[" + ", ".join(_fmt(v) for v in value) + "]"
    return str(value)


def _short(name: str) -> str:
    """``r_arm_sh_pitch1`` -> ``sh_pitch1`` for tables of one arm."""
    return name.split("_arm_", 1)[-1]


def _read_positions(bus: ControlBus, names: Sequence[str]) -> Dict[str, int]:
    readings, _, _ = bus.read_state_block(
        names, timeout_s=STATE_READ_TIMEOUT_S, attempts=STATE_READ_ATTEMPTS
    )
    return {name: int(readings[name].position) for name in names}


# --- pure helpers ---------------------------------------------------------------------


def decide_sign(delta_plus: int, delta_minus: int, threshold: float) -> int | None:
    """The sign verdict: ``Δ = Δq⁺ - Δq⁻``; ``+1`` / ``-1`` beyond ``±threshold``, else ``None``."""
    delta = delta_plus - delta_minus
    if delta >= threshold:
        return 1
    if delta <= -threshold:
        return -1
    return None


def check_sign_groups(
    signs: Mapping[str, int], drive_mode: Mapping[str, int]
) -> Tuple[Dict[str, List[str]], List[str], bool | None]:
    """Groups the joints by ``DRIVE_MODE`` bit 0 and checks each group agrees.

    Returns:
        ``(groups, problems, reverse_flips_current)``: ``groups`` maps
        ``"normal"``/``"reverse"`` to joints; ``problems`` names every group
        whose signs differ (nothing may be saved then); the flag says whether
        the reverse group's sign is opposite to the normal group's (``None``
        when a group is empty or inconsistent).
    """
    groups: Dict[str, List[str]] = {"normal": [], "reverse": []}
    for name in signs:
        groups["reverse" if drive_mode[name] & 0x01 else "normal"].append(name)
    problems: List[str] = []
    group_sign: Dict[str, int] = {}
    for group, names in groups.items():
        values = {signs[name] for name in names}
        if len(values) > 1:
            detail = ", ".join(f"{name}={signs[name]:+d}" for name in names)
            problems.append(f"{group} DRIVE_MODE group disagrees: {detail}")
        elif values:
            group_sign[group] = values.pop()
    flips = None
    if len(group_sign) == 2:
        flips = group_sign["normal"] != group_sign["reverse"]
    return groups, problems, flips


def coverage(
    positions: Sequence[ArrayLike],
    joints: Sequence[str],
    ranges: Mapping[str, Tuple[int, int]],
    bins: int = COVERAGE_BINS,
) -> Dict[str, NDArray[np.bool_]]:
    """Which of ``bins`` equal slices of each joint's range the poses visited."""
    visited = {name: np.zeros(bins, dtype=np.bool_) for name in joints}
    for q in positions:
        q_arr = np.asarray(q, dtype=np.float64)
        for k, name in enumerate(joints):
            lo, hi = ranges[name]
            index = int(np.floor((q_arr[k] - lo) / (hi - lo) * bins))
            visited[name][min(max(index, 0), bins - 1)] = True
    return visited


def provisional_seed_ma(
    samples: Sequence[PoseSample], q_counts: ArrayLike, neighbours: int = PROVISIONAL_NEIGHBOURS
) -> FloatArray | None:
    """The provisional model: inverse-distance mean of the nearest poses' gravity currents.

    Only seeds the ``CAPTURE`` integrator; ``None`` below
    ``PROVISIONAL_MIN_POSES``.  ``fit_arm`` needs ``MIN_FIT_SAMPLES`` poses,
    more than this seed has to work with.
    """
    if len(samples) < PROVISIONAL_MIN_POSES:
        return None
    q = np.asarray(q_counts, dtype=np.float64)
    positions = np.stack([s.q_counts for s in samples])
    gravity = np.stack([s.i_gravity_ma for s in samples])
    distance = np.linalg.norm(positions - q, axis=1)
    nearest = np.argsort(distance)[:neighbours]
    weight = 1.0 / (distance[nearest] + 1.0)
    return np.asarray(weight @ gravity[nearest] / weight.sum(), dtype=np.float64)


def recommend_gravity_scale_step(
    drift_counts: Sequence[ArrayLike],
    gravity_ma: Sequence[ArrayLike],
    *,
    step: float = GRAVITY_SCALE_STEP,
    min_gravity_ma: float = RECOMMEND_MIN_GRAVITY_MA,
    min_drift_counts: float = RECOMMEND_MIN_DRIFT_COUNTS,
) -> float:
    """``+step`` when most loaded joints sank along gravity, ``-step`` when most rose, else 0.

    A positive holding current means gravity pulls toward smaller counts, so a
    joint sank when its drift and its gravity current have opposite signs.
    Joint-pose pairs with little gravity or little drift are not counted.
    """
    sinks = rises = 0
    for drift, gravity in zip(drift_counts, gravity_ma):
        for d, g in zip(np.asarray(drift, dtype=np.float64), np.asarray(gravity, dtype=np.float64)):
            if abs(g) < min_gravity_ma or abs(d) < min_drift_counts:
                continue
            if d * g < 0:
                sinks += 1
            else:
                rises += 1
    if sinks > rises:
        return step
    if rises > sinks:
        return -step
    return 0.0


# --- range -------------------------------------------------------------------------


class _RangeTracker:
    """Min/max of the arm joints sampled while the operator moves them."""

    def __init__(
        self,
        bus: ControlBus,
        names: Sequence[str],
        start: Mapping[str, int],
        console: Console,
        clock: Callable[[], int],
    ) -> None:
        self._bus = bus
        self._names = list(names)
        self.lo: Dict[str, int] = {name: int(start[name]) for name in names}
        self.hi: Dict[str, int] = dict(self.lo)
        self._console = console
        self._clock = clock
        self._last_display_ns: int | None = None
        self.failures = 0

    def tick(self) -> None:
        try:
            positions = _read_positions(self._bus, self._names)
        except DynamixelCommError as exc:
            self.failures += 1
            logger.warning("range: read failed (%s)", exc)
            return
        for name, position in positions.items():
            self.lo[name] = min(self.lo[name], position)
            self.hi[name] = max(self.hi[name], position)
        now = self._clock()
        if self._last_display_ns is None or now - self._last_display_ns >= 1_000_000_000:
            self._last_display_ns = now
            self._console.say(self.line())

    def line(self) -> str:
        """One display line per arm: ``joint lo..hi``."""
        rows = []
        for side in ARM_SIDES:
            cells = [
                f"{_short(name)} {self.lo[name]}..{self.hi[name]}"
                for name in arm_joints(side)
                if name in self.lo
            ]
            rows.append(f"  {side:<5} " + "  ".join(cells))
        return "\n".join(rows)


def _motor_table(bus: ControlBus) -> List[Dict[str, Any]]:
    """``name/id/model/unit_ma/drive_mode`` of every leader motor, as stored in the gravity file."""
    names = list(bus.motors)
    drive = bus.sync_read(XControlTable.DRIVE_MODE, names)
    return [
        {
            "name": name,
            "id": motor.id,
            "model": motor.model_name,
            "unit_ma": CURRENT_UNIT_MA[motor.model_name],
            "drive_mode": int(drive[name]),
        }
        for name, motor in bus.motors.items()
    ]


def cmd_range(env: CliEnv, *, port: str = PORT_AUTO, path: str = DEFAULT_GRAVITY_FILE) -> int:
    """``range``: reference pose and joint ranges with both arms torque off.

    Both arms are switched off after the operator confirms; the reference is
    the 1 s average of the reference pose (arm joints and ``torso_yaw``,
    informational); the ranges are the raw min/max seen at 20 Hz while the
    operator moves every joint to both safe ends (no shrinking: the barrier
    acts ``limit_margin_counts`` inside).  The file is created or updated
    (other sections kept); a joint that moved no more than twice the margin
    gets no new range and the command exits with 1.  The arms stay released
    at the end.
    """
    console = env.console
    arm = list(RAKUDA_ARM_JOINT_NAMES)
    leader = _connect_leader(env, port)
    try:
        bus = leader.motors
        answer = console.ask(
            "range: both arms will be switched OFF (torque off). Support them or let them "
            "hang, then press Enter (q: cancel): "
        )
        if _is(answer, "q"):
            console.say("cancelled; nothing written")
            return EXIT_FAILED
        bus.torque_disabled(arm)

        answer = console.ask(
            "Put both arms in the reference pose (straight down, elbows extended, wrists "
            "neutral, palms toward the body), hold them still and press Enter (q: cancel): "
        )
        if _is(answer, "q"):
            console.say("cancelled; nothing written")
            return EXIT_FAILED
        names = arm + ["torso_yaw"]
        samples: List[List[int]] = []
        count = max(1, int(round(REFERENCE_AVERAGE_S * RANGE_SAMPLE_HZ)))
        for k in range(count):
            positions = _read_positions(bus, names)
            samples.append([positions[name] for name in names])
            if k + 1 < count:
                env.sleep(1.0 / RANGE_SAMPLE_HZ)
        table = np.asarray(samples, dtype=np.float64)
        reference = {name: int(round(v)) for name, v in zip(names, table.mean(axis=0))}
        spread = {n: int(s) for n, s in zip(names, np.ptp(table, axis=0))}
        moved = {n: s for n, s in spread.items() if s > REFERENCE_STILL_COUNTS}
        if moved:
            console.say(f"warning: the arms moved during the reference average: {moved}")
        console.say("reference: " + ", ".join(f"{n}={reference[n]}" for n in names))

        tracker = _RangeTracker(bus, arm, reference, console, env.clock)
        console.ask_while(
            "Move every arm joint to both of its safe ends (the display shows min..max), "
            "then press Enter: ",
            tracker.tick,
            1.0 / RANGE_SAMPLE_HZ,
        )
        measured = {name: (tracker.lo[name], tracker.hi[name]) for name in arm}
        for name in arm:
            lo, hi = measured[name]
            console.say(f"  {name:<16} ref {reference[name]:>5}  range {lo:>5}..{hi:<5}")
        # The loop refuses a range no wider than twice the barrier margin, so such a
        # joint (not moved to both ends) is not written; an earlier range is kept.
        margin = env.params.limit_margin_counts
        narrow = [name for name, (lo, hi) in measured.items() if hi - lo <= 2 * margin]
        ranges = {name: span for name, span in measured.items() if name not in narrow}

        data = _load_or_new(path)
        data.motors = _motor_table(bus)
        data.reference_counts = {**data.reference_counts, **reference}
        data.joint_range_counts = {**data.joint_range_counts, **ranges}
        save_gravity_file(path, data)
        if data.arms:
            console.say(
                "note: the fitted arms keep their own reference; re-identify them if the "
                "arm hardware changed"
            )
        console.say(f"wrote reference_counts and joint_range_counts to {path}")
        if narrow:
            kept = [name for name in narrow if name in data.joint_range_counts]
            console.say(
                f"warning: {narrow} moved {2 * margin} counts or less (2 x limit_margin_counts); "
                f"their range was not written (earlier range kept for {kept}); move them to "
                "both safe ends and re-run range"
            )
            return EXIT_FAILED
        return EXIT_OK
    finally:
        leader.disconnect(torque_off=False)


# --- sign-check ----------------------------------------------------------------------


@dataclass
class _JointVerdict:
    joint: str
    status: Literal["ok", "skipped", "undetermined", "fault"]
    drive_mode: int | None = None
    sign: int | None = None
    level_ma: float | None = None
    deltas: List[Tuple[float, int, int]] = field(default_factory=list)
    present_matches: Tuple[bool | None, bool | None] = (None, None)
    detail: str = ""


def _run_pulse(
    env: CliEnv, loop: LeaderCurrentLoop, law: _PulseLaw, level: float, sc: SignCheckParams
) -> int | None:
    """One pulse, then ``settle_s``; the displacement, or ``None`` when the loop stopped."""
    law.start_pulse(level)
    env.runner.run_until(loop, law.pulse_finished, sc.pulse_max_s + 1.0)
    if not loop.running or not _pause(env, loop, sc.settle_s):
        return None
    return law.displacement()


def _sign_check_joint(
    env: CliEnv,
    bus: ControlBus,
    joint: str,
    joint_range: Tuple[int, int],
    sc: SignCheckParams,
) -> Tuple[_JointVerdict, LeaderCurrentLoop | None]:
    """Sign check, one joint: ``J={joint}``, ascending levels, ``+``/``-`` pulses, hold, release."""
    law = _PulseLaw(joint, sc, joint_range)
    loop = _make_loop(env, bus, (joint,), law, {joint: 1})
    preflight = loop.preflight()
    verdict = _JointVerdict(joint, "undetermined", drive_mode=int(preflight["drive_mode"][joint]))
    lo, hi = joint_range
    q_start = _read_positions(bus, [joint])[joint]
    if not lo + sc.range_guard_counts <= q_start <= hi - sc.range_guard_counts:
        verdict.status = "skipped"
        verdict.detail = (
            f"at {q_start}, within {sc.range_guard_counts} counts of its range {joint_range}"
        )
        return verdict, None
    loop.configure()
    try:
        env.runner.start(loop)
        for level in sc.current_levels_ma:
            plus = _run_pulse(env, loop, law, level, sc)
            if plus is None or not _pause(env, loop, sc.between_pulses_s):
                break
            minus = _run_pulse(env, loop, law, -level, sc)
            if minus is None:
                break
            verdict.deltas.append((level, plus, minus))
            env.console.say(f"    {level:5.0f} mA: dq+ {plus:+4d}  dq- {minus:+4d}")
            if max(abs(plus), abs(minus)) >= sc.move_threshold_counts:
                verdict.level_ma = level
                verdict.sign = decide_sign(plus, minus, sc.move_threshold_counts)
                verdict.present_matches = law.present_current_matches()
                break
            if not _pause(env, loop, sc.between_pulses_s):
                break
    finally:
        _stop_loop(env, loop)
    if loop.fault is not None:
        verdict.status = "fault"
        verdict.detail = f"{loop.fault.reason}: {loop.fault.detail}"
        return verdict, loop
    q_end = _read_positions(bus, [joint])[joint]
    if abs(q_end - q_start) > SIGN_CHECK_RETURN_WARN_COUNTS:
        env.console.say(f"    warning: {joint} ended {q_end - q_start:+d} counts from its start")
    loop.release()
    if verdict.sign is None:
        verdict.status = "undetermined"
        verdict.detail = (
            "no movement at any level"
            if verdict.level_ma is None
            else f"dq+ - dq- within ±{sc.move_threshold_counts} counts"
        )
    else:
        verdict.status = "ok"
    return verdict, None


def cmd_sign_check(
    env: CliEnv,
    *,
    port: str = PORT_AUTO,
    path: str = DEFAULT_GRAVITY_FILE,
    levels_ma: Sequence[float] | None = None,
    end: EndAction = "ask",
) -> int:
    """``sign-check``: the ``GOAL_CURRENT`` sign of every arm joint.

    Both arms hang torque off.  Per joint: ``J={joint}``, ``current_sign=+1``,
    :class:`SignCheckLaw` pulses in ascending levels (``+`` then ``-``),
    verdict from ``Δq⁺ - Δq⁻``, hold, release.  A joint too close to its
    range end is skipped.  An undetermined joint, a fault (the joint stays
    held) or a ``DRIVE_MODE`` group whose signs disagree stops without saving;
    otherwise the verdicts are merged into ``current_sign``, ``drive_mode``
    and ``sign_check``: a joint not measured in this run keeps its earlier
    sign while its ``DRIVE_MODE`` is still the recorded one, and the group
    check covers the merged table.  Joints left without a sign are listed.
    """
    console = env.console
    data = _load_file(path)
    arm = list(RAKUDA_ARM_JOINT_NAMES)
    _require(data, arm, path, "joint_range_counts")
    sc = SignCheckParams()
    if levels_ma is not None:
        try:
            sc = replace(sc, current_levels_ma=tuple(float(v) for v in levels_ma))
        except ValueError as exc:
            raise CliError(f"--levels: {exc}") from exc
    leader = _connect_leader(env, port)
    try:
        bus = leader.motors
        answer = console.ask(
            "sign-check: both arms will be switched OFF and must hang freely (gravity ~0). "
            "Support them while they are released, then press Enter (q: cancel): "
        )
        if _is(answer, "q"):
            console.say("cancelled; nothing written")
            return EXIT_FAILED
        bus.torque_disabled(arm)
        drive_now = {
            name: int(v) for name, v in bus.sync_read(XControlTable.DRIVE_MODE, arm).items()
        }
        verdicts: Dict[str, _JointVerdict] = {}
        stopped: str | None = None
        for k, joint in enumerate(arm, start=1):
            answer = console.ask(
                f"[{k}/{len(arm)}] {joint}: make sure it can swing ±5 deg freely, then press "
                "Enter (s: skip, q: stop): "
            )
            if _is(answer, "q"):
                stopped = "stopped by the operator"
                break
            if _is(answer, "s"):
                verdicts[joint] = _JointVerdict(joint, "skipped", detail="skipped by the operator")
                continue
            verdict, held_loop = _sign_check_joint(
                env, bus, joint, data.joint_range_counts[joint], sc
            )
            verdicts[joint] = verdict
            sign_text = "?" if verdict.sign is None else f"{verdict.sign:+d}"
            console.say(f"  {joint}: {verdict.status} sign {sign_text} {verdict.detail}".rstrip())
            if held_loop is not None:
                _end_session(env, held_loop, end)
                stopped = f"{joint}: {verdict.detail}"
                break
            if verdict.status == "undetermined":
                stopped = f"{joint} undetermined ({verdict.detail})"
                break

        _say_sign_table(console, verdicts)
        if stopped is not None:
            console.say(f"sign-check stopped ({stopped}); nothing written")
            return EXIT_FAILED
        ok = {j: v for j, v in verdicts.items() if v.status == "ok"}
        data = _load_file(path)
        # An earlier sign stays valid only while DRIVE_MODE is still the one it was
        # measured with (a change may flip the current direction).
        earlier = [j for j in arm if j not in ok and j in data.current_sign]
        kept = [j for j in earlier if data.drive_mode.get(j) == drive_now[j]]
        dropped = [j for j in earlier if j not in kept]
        signs: Dict[str, int] = {}
        modes: Dict[str, int] = {}
        for j in arm:
            if j in ok:
                signs[j], modes[j] = int(cast(int, ok[j].sign)), int(cast(int, ok[j].drive_mode))
            elif j in kept:
                signs[j], modes[j] = int(data.current_sign[j]), int(data.drive_mode[j])
        groups, problems, flips = check_sign_groups(signs, modes)
        if problems:
            for problem in problems:
                console.say(f"  {problem}")
            console.say("the signs within a DRIVE_MODE group disagree; nothing written")
            return EXIT_FAILED
        if flips is True:
            logger.info("DRIVE_MODE bit0 (Reverse Mode) also reverses the current direction")
            console.say("Reverse Mode also reverses the current (expected)")
        elif flips is False:
            logger.warning("the reverse and normal DRIVE_MODE groups share one current sign")
            console.say("warning: the reverse and normal DRIVE_MODE groups share one current sign")
        unexpected = [j for j in groups["normal"] if signs[j] != 1]
        if unexpected:
            console.say(f"warning: {unexpected} have DRIVE_MODE bit0=0 but sign -1 (check wiring)")
        if kept:
            console.say(f"kept the earlier signs of {kept} (not measured in this run)")
        if dropped:
            console.say(
                f"warning: DRIVE_MODE of {dropped} changed since their earlier sign-check; "
                "their signs were dropped"
            )
        missing = [j for j in arm if j not in signs]
        if missing:
            console.say(
                f"warning: no sign for {missing}; run `robopy-rakuda-gravity sign-check` again "
                "(s skips the joints already checked) before using them"
            )

        def merged(key: str, new: Mapping[str, Any]) -> Dict[str, Any]:
            """Per-joint ``sign_check`` details: this run's, else the kept joints' earlier ones."""
            previous = data.sign_check.get(key)
            previous = previous if isinstance(previous, Mapping) else {}
            return {
                j: new[j] if j in new else previous[j] for j in signs if j in new or j in previous
            }

        present = merged(
            "present_current_matches", {j: list(v.present_matches) for j, v in ok.items()}
        )
        observed = [m for pair in present.values() for m in pair if m is not None]
        mismatched = [j for j, pair in present.items() if False in pair]
        if mismatched:
            console.say(
                f"warning: PRESENT_CURRENT of {mismatched} had the opposite sign of the "
                "commanded current during a pulse (information only)"
            )
        data.current_sign = signs
        data.drive_mode = modes
        data.sign_check = {
            "levels_ma": merged("levels_ma", {j: v.level_ma for j, v in ok.items()}),
            "delta_counts": merged(
                "delta_counts", {j: [[lv, p, m] for lv, p, m in v.deltas] for j, v in ok.items()}
            ),
            "present_current_matches": present,
            "drive_mode_groups": {g: list(names) for g, names in groups.items()},
            "reverse_flips_current": flips,
            "present_current_sign_matches_goal": all(observed) if observed else None,
            "skipped": [j for j in arm if j not in ok],
            "kept": kept,
            "missing": missing,
            "created_at": _now_iso(),
        }
        save_gravity_file(path, data)
        console.say(
            f"wrote current_sign / drive_mode / sign_check to {path}: {len(ok)} joints measured, "
            f"{len(kept)} kept, {len(missing)} without a sign"
        )
        return EXIT_OK
    finally:
        leader.disconnect(torque_off=False)


def _say_sign_table(console: Console, verdicts: Mapping[str, _JointVerdict]) -> None:
    console.say(f"  {'joint':<16} {'status':<12} {'drive':>5} {'sign':>4} {'level':>6}  dq+/dq-")
    for joint, v in verdicts.items():
        last = v.deltas[-1] if v.deltas else None
        console.say(
            f"  {joint:<16} {v.status:<12} {'-' if v.drive_mode is None else v.drive_mode:>5} "
            f"{'-' if v.sign is None else f'{v.sign:+d}':>4} "
            f"{'-' if v.level_ma is None else f'{v.level_ma:.0f}':>6}  "
            f"{'-' if last is None else f'{last[1]:+d}/{last[2]:+d}'}"
        )


# --- identify / fit -------------------------------------------------------------------


def _dataset_path(path: str, side: str) -> str:
    return os.path.join(os.path.dirname(path) or ".", f"leader_gravity_dataset_{side}.npz")


def _file_sha256(path: str) -> str:
    with open(path, "rb") as handle:
        return hashlib.sha256(handle.read()).hexdigest()


def _say_fit(console: Console, fit: ArmGravityFit) -> None:
    console.say(f"{fit.side} arm: model {fit.model_type}, accepted {fit.accepted}")
    for key, value in fit.fit_report.items():
        if key == "models" and isinstance(value, Mapping):
            for model, report in value.items():
                rms = report.get("eval_rms_ma") if isinstance(report, Mapping) else None
                console.say(f"  {model}: eval RMS {_fmt(rms)} mA")
                per_joint = report.get("per_joint", {}) if isinstance(report, Mapping) else {}
                for joint, row in per_joint.items():
                    console.say(f"    {joint:<16} {_fmt(row)}")
        elif not isinstance(value, Mapping):
            console.say(f"  {key}: {_fmt(value)}")
    console.say(f"  peak_ma: {_fmt(fit.peak_ma)}")
    console.say(f"  friction_ma: {_fmt(fit.friction_ma)}")


def _fit_and_store(
    env: CliEnv,
    path: str,
    side: str,
    samples: Sequence[PoseSample],
    dataset_sha256: str,
    ridge: float,
) -> ArmGravityFit:
    """``fit_arm`` on the dataset and the arm's entry in the file (``validated`` false)."""
    data = _load_file(path)
    joints = arm_joints(side)
    _require(data, joints, path, "reference_counts")
    q_ref = {name: data.reference_counts[name] for name in joints}
    fit = fit_arm(samples, side, q_ref, ridge=ridge)
    _say_fit(env.console, fit)
    temperature = np.stack([s.temperature_c for s in samples])
    mean_c: Dict[str, float | None] = {}
    for k, name in enumerate(joints):
        column = temperature[:, k]
        column = column[np.isfinite(column)]
        mean_c[name] = float(column.mean()) if column.size else None
    entry = {
        **arm_fit_to_dict(fit),
        "dataset_sha256": dataset_sha256,
        "validated": False,
        "validated_at": None,
        "temperature": mean_c,
    }
    data.arms = {**data.arms, side: entry}
    save_gravity_file(path, data)
    env.console.say(f"stored the {side} arm model in {path} (validated: false; run `verify`)")
    if not fit.accepted:
        env.console.say(
            "warning: the fit is not accepted; the loop needs allow_unvalidated_gravity to use it"
        )
    return fit


def cmd_fit(
    env: CliEnv,
    *,
    path: str = DEFAULT_GRAVITY_FILE,
    arm: str,
    dataset_path: str | None = None,
    ridge: float = 1e-2,
) -> int:
    """``fit --arm``: fits the saved dataset and stores the arm (``validated`` false)."""
    dataset = _dataset_path(path, arm) if dataset_path is None else dataset_path
    try:
        samples, header = load_dataset(dataset)
    except FileNotFoundError as exc:
        raise CliError(f"no dataset at {dataset}; run `identify --arm {arm}` first") from exc
    if header.get("side") not in (None, arm):
        raise CliError(f"{dataset} holds the {header.get('side')} arm, not the {arm} arm")
    if len(samples) < MIN_FIT_SAMPLES:
        raise CliError(
            f"{dataset} has {len(samples)} poses; at least {MIN_FIT_SAMPLES} are needed",
            EXIT_FAILED,
        )
    fit = _fit_and_store(env, path, arm, samples, _file_sha256(dataset), ridge)
    return EXIT_OK if fit.accepted else EXIT_FAILED


def _say_coverage(
    console: Console,
    samples: Sequence[PoseSample],
    joints: Sequence[str],
    ranges: Mapping[str, Tuple[int, int]],
) -> None:
    visited = coverage([s.q_counts for s in samples], joints, ranges)
    for name in joints:
        bins = visited[name]
        missing = [k for k, seen in enumerate(bins) if not seen]
        marks = "".join("#" if seen else "." for seen in bins)
        console.say(f"  {_short(name):<10} [{marks}]" + (f" missing {missing}" if missing else ""))


def _say_measurement(console: Console, joints: Sequence[str], m: PoseMeasurement) -> None:
    gravity = (m.i_plus_ma + m.i_minus_ma) / 2.0
    friction = np.abs(m.i_plus_ma - m.i_minus_ma) / 2.0
    console.say(
        "  I_g  " + " ".join(f"{_short(n)}={g:+.0f}" for n, g in zip(joints, gravity)) + " mA"
    )
    console.say(
        "  f_s  " + " ".join(f"{_short(n)}={f:.0f}" for n, f in zip(joints, friction)) + " mA"
    )
    if m.flags:
        console.say(f"  flags: {list(m.flags)}")


def _open_dataset(
    console: Console, dataset: str, side: str
) -> Tuple[List[PoseSample], Dict[str, Any]]:
    """The poses to continue from (after asking), or an empty list."""
    if not os.path.exists(dataset):
        return [], {}
    samples, header = load_dataset(dataset)
    if header.get("side") not in (None, side):
        raise CliError(f"{dataset} holds the {header.get('side')} arm, not the {side} arm")
    answer = console.ask(
        f"{dataset} has {len(samples)} poses. Enter: continue it / n: start a new dataset "
        "(the old file is kept under a timestamped name): "
    )
    if _is(answer, "n"):
        archive = _archive_path(dataset)
        os.replace(dataset, archive)
        console.say(f"moved the old dataset to {archive}")
        return [], {}
    return list(samples), dict(header)


def _archive_path(dataset: str) -> str:
    """A name next to ``dataset`` that no earlier archive uses (``<name>.<UTC time>[-n].npz``)."""
    root, ext = os.path.splitext(dataset)
    stem = f"{root}.{datetime.now(timezone.utc).strftime('%Y%m%dT%H%M%SZ')}"
    candidate, n = stem + ext, 1
    while os.path.exists(candidate):
        candidate, n = f"{stem}-{n}{ext}", n + 1
    return candidate


def cmd_identify(
    env: CliEnv,
    *,
    port: str = PORT_AUTO,
    path: str = DEFAULT_GRAVITY_FILE,
    arm: str,
    dataset_path: str | None = None,
    ridge: float = 1e-2,
    end: EndAction = "ask",
    manual: ManualPoseParams | None = None,
) -> int:
    """``identify --arm``: operator-placed poses held by :class:`ManualPoseLaw`.

    Needs ``range`` and ``sign-check`` for the arm; the other arm is switched
    off (it hangs).  Per pose: the operator places and supports the arm and
    presses Enter (capture; from ``PROVISIONAL_MIN_POSES`` poses on the
    integrator starts from the provisional model), lets go, the law holds
    from both sides and reports the currents, the pose is appended to the
    dataset (saved after every pose, every ``VALIDATION_EVERY``-th pose is a
    validation pose), then Enter releases the arm to ``FREE`` (``q``
    finishes).  At ``PAUSE_TEMPERATURE_C`` the CLI suggests a pause.  At the
    end the loop holds; with at least ``MIN_FIT_SAMPLES`` poses the dataset
    is fitted and stored.
    """
    console = env.console
    side = arm
    joints = arm_joints(side)
    data = _load_file(path)
    _require(
        data, joints, path, "reference_counts", "joint_range_counts", "current_sign", "drive_mode"
    )
    ranges = {name: data.joint_range_counts[name] for name in joints}
    dataset = _dataset_path(path, side) if dataset_path is None else dataset_path
    samples, header = _open_dataset(console, dataset, side)
    header = {
        **header,
        "created_at": header.get("created_at", _now_iso()),
        "reference_counts": {name: data.reference_counts[name] for name in joints},
        "joint_range_counts": {name: list(ranges[name]) for name in joints},
        "current_sign": {name: data.current_sign[name] for name in joints},
        "drive_mode": {name: data.drive_mode[name] for name in joints},
        "identify_params": asdict(IdentifyParams()),
        "manual_pose_params": asdict(ManualPoseParams() if manual is None else manual),
    }
    dataset_sha = ""
    leader = _connect_leader(env, port)
    try:
        bus = leader.motors
        if not _release_other_arm(env, bus, side):
            console.say("cancelled")
            return EXIT_FAILED
        tap = _DiagnosticsTap(bus)
        law = ManualPoseLaw(joints, IdentifyParams(), ranges, manual)
        loop = _make_loop(env, cast(ControlBus, tap), joints, law, data.current_sign)
        answer = console.ask(
            f"Support the {side} arm with your hand: at Enter its joints switch to current "
            "mode at 0 mA (limp) (q: cancel): "
        )
        if _is(answer, "q"):
            console.say("cancelled")
            return EXIT_FAILED
        preflight = loop.preflight()
        _check_drive_mode(data.drive_mode, preflight, joints)
        loop.configure()
        rejected = 0
        try:
            env.runner.start(loop)
            while loop.running:
                console.say(
                    f"{len(samples)} poses (target 40-60, fit needs {MIN_FIT_SAMPLES}); coverage:"
                )
                _say_coverage(console, samples, joints, ranges)
                hottest = float(np.nanmax(tap.temperatures(joints), initial=-math.inf))
                if hottest >= PAUSE_TEMPERATURE_C:
                    answer = console.ask(
                        f"The arm's motors reach {hottest:.0f} C: let them cool (the arm is limp, "
                        "support or rest it). Enter: continue / q: finish: "
                    )
                    if _is(answer, "q"):
                        break
                number = len(samples) + 1
                answer = console.ask(
                    f"Pose {number}: support the arm in a new pose and press Enter while "
                    "supporting it (q: finish): "
                )
                if _is(answer, "q") or not loop.running:
                    break
                latest = loop.latest()
                seed = None
                if latest is not None:
                    q_arm = latest.position[[RAKUDA_JOINT_NAMES.index(n) for n in joints]]
                    seed = provisional_seed_ma(samples, q_arm)
                law.capture(seed)
                console.say("  let go gently; measuring ...")
                status = _measure_pose(env, loop, law)
                if status is None:
                    break
                if status.state is PoseState.DONE and status.measurement is not None:
                    m = status.measurement
                    sample = PoseSample(
                        q_counts=m.q_counts,
                        i_plus_ma=m.i_plus_ma,
                        i_minus_ma=m.i_minus_ma,
                        temperature_c=tap.temperatures(joints),
                        flags=m.flags,
                        is_val=number % VALIDATION_EVERY == 0,
                    )
                    samples.append(sample)
                    header["updated_at"] = _now_iso()
                    dataset_sha = save_dataset(dataset, side, samples, header)
                    console.say(f"  pose {number} saved{' (validation)' if sample.is_val else ''}")
                    _say_measurement(console, joints, m)
                else:
                    rejected += 1
                    reason = status.reason or "timeout"
                    console.say(f"  pose rejected: {reason}")
                    if reason.startswith("near_range_end"):
                        delta = (ManualPoseParams() if manual is None else manual).delta_counts
                        console.say(
                            f"  those joints must be at least {delta:.0f} counts inside their "
                            "recorded range (the pose is approached from both sides); move "
                            "them inward"
                        )
                answer = console.ask(
                    "Support the arm and press Enter to release it (q: finish, the arm keeps "
                    "holding): "
                )
                if _is(answer, "q") or not loop.running:
                    break
                law.release_to_free()
        finally:
            _stop_loop(env, loop)
        console.say(f"{len(samples)} poses in {dataset} ({rejected} rejected this session)")
        failed = loop.fault is not None
        if len(samples) >= MIN_FIT_SAMPLES:
            sha = dataset_sha or _file_sha256(dataset)
            try:
                fit = _fit_and_store(env, path, side, samples, sha, ridge)
                failed = failed or not fit.accepted
            except Exception as exc:  # the dataset is saved; the arm must still be ended below
                logger.error("fit failed", exc_info=True)
                console.say(
                    f"fit failed: {exc}; the dataset is saved, retry with `fit --arm {side}`"
                )
                failed = True
        else:
            console.say(
                f"not fitted: {MIN_FIT_SAMPLES} poses are needed; run identify again to add more"
            )
        _end_session(env, loop, end)
        return EXIT_FAILED if failed else EXIT_OK
    finally:
        leader.disconnect(torque_off=False)


def _measure_pose(
    env: CliEnv, loop: LeaderCurrentLoop, law: ManualPoseLaw
) -> ManualPoseStatus | None:
    """Runs the loop until the pose is DONE or REJECTED; ``None`` when the loop stopped."""
    shown: List[PoseState] = []

    def show() -> None:
        status = law.status()
        if not shown or shown[-1] is not status.state:
            shown.append(status.state)
            env.console.say(f"    {status.state.value:<14} {status.progress:4.0%}")

    finished = env.runner.run_until(
        loop,
        lambda: law.state in (PoseState.DONE, PoseState.REJECTED),
        POSE_TIMEOUT_S,
        tick=show,
        tick_s=0.25,
    )
    if not loop.running:
        return None
    status = law.status()
    if not finished:
        return replace(status, state=PoseState.REJECTED, reason="timeout")
    return status


# --- verify ---------------------------------------------------------------------------


def cmd_verify(
    env: CliEnv,
    *,
    port: str = PORT_AUTO,
    path: str = DEFAULT_GRAVITY_FILE,
    arm: str,
    poses: int = VERIFY_POSES,
    seconds: float = VERIFY_SECONDS,
    end: EndAction = "ask",
) -> int:
    """``verify --arm``: gravity compensation alone must hold the arm still.

    ``BilateralLaw`` with the arm's model (``gravity_scale`` from the params,
    feedback gains 0, barrier on), no follower.  At each of ``poses`` poses
    the operator places the arm, lets go and presses Enter; the positions are
    recorded for ``seconds``.  The arm passes when every joint stays within
    ``VERIFY_DRIFT_LIMIT_COUNTS`` of where it was let go at every pose; then
    ``validated`` is set.  A run shorter than the default ``VERIFY_POSES`` x
    ``VERIFY_SECONDS`` is recorded but never validates.  The sink/rise tendency
    gives the recommended ``gravity_scale`` change.

    Raises:
        CliError: ``poses < 1`` or ``seconds`` not a positive number, or a
            missing prerequisite.
    """
    console = env.console
    side = arm
    if isinstance(poses, bool) or not isinstance(poses, int) or poses < 1:
        raise CliError(f"--poses must be at least 1, got {poses!r}")
    if not math.isfinite(seconds) or seconds <= 0:
        raise CliError(f"--seconds must be a positive number, got {seconds!r}")
    full_length = poses >= VERIFY_POSES and seconds >= VERIFY_SECONDS
    joints = arm_joints(side)
    data = _load_file(path)
    _require(data, joints, path, "joint_range_counts", "current_sign", "drive_mode")
    if side not in data.arms:
        raise CliError(f"{path} has no {side} arm model; run `identify --arm {side}` first")
    fit = arm_fit_from_dict(data.arms[side])
    if not fit.accepted:
        console.say("note: this fit was not accepted by `fit`; verifying it anyway")
    model = LeaderGravityModel({side: fit})
    params = replace(
        env.params,
        current_joints=joints,
        feedback_kp_ma_per_count=0.0,
        feedback_kd_ma_per_vcount=0.0,
        allow_uncompensated=False,
    )
    ranges = {name: data.joint_range_counts[name] for name in joints}
    law = BilateralLaw(
        joints, params, model, ranges, gravity_peak_ma={n: model.peak_ma[n] for n in joints}
    )
    index = [RAKUDA_JOINT_NAMES.index(name) for name in joints]
    leader = _connect_leader(env, port)
    try:
        bus = leader.motors
        if not _release_other_arm(env, bus, side):
            console.say("cancelled")
            return EXIT_FAILED
        loop = _make_loop(env, bus, joints, law, data.current_sign, params)
        answer = console.ask(
            f"Support the {side} arm: at Enter gravity compensation takes it over (q: cancel): "
        )
        if _is(answer, "q"):
            console.say("cancelled")
            return EXIT_FAILED
        preflight = loop.preflight()
        _check_drive_mode(data.drive_mode, preflight, joints)
        loop.configure()
        drifts: List[FloatArray] = []
        worst: List[FloatArray] = []
        gravity: List[FloatArray] = []
        try:
            env.runner.start(loop)
            for k in range(1, poses + 1):
                answer = console.ask(
                    f"Verify pose {k}/{poses}: place the arm, let go, then press Enter (q: stop): "
                )
                if _is(answer, "q") or not loop.running:
                    break
                start = loop.latest()
                if start is None:
                    break
                q0 = start.position[index].astype(np.float64)
                gravity.append(model.predict_ma(start.position)[index])
                deviation = np.zeros(len(joints))

                def sample() -> None:
                    state = loop.latest()
                    if state is not None:
                        moved = np.abs(state.position[index].astype(np.float64) - q0)
                        np.maximum(deviation, moved, out=deviation)

                _pause(env, loop, seconds, tick=sample, tick_s=0.05)
                if not loop.running:
                    break
                sample()
                end_state = loop.latest()
                assert end_state is not None
                drift = end_state.position[index].astype(np.float64) - q0
                drifts.append(drift)
                worst.append(deviation.copy())
                verdict = "ok" if bool(np.all(deviation < VERIFY_DRIFT_LIMIT_COUNTS)) else "DRIFT"
                console.say(
                    f"  pose {k}: {verdict}  "
                    + " ".join(f"{_short(n)}={d:+.0f}" for n, d in zip(joints, drift))
                    + f"  (max |dq| {deviation.max():.0f} counts)"
                )
        finally:
            _stop_loop(env, loop)
        completed = len(drifts) == poses and loop.fault is None
        drift_ok = completed and all(bool(np.all(w < VERIFY_DRIFT_LIMIT_COUNTS)) for w in worst)
        passed = drift_ok and full_length
        step = recommend_gravity_scale_step(drifts, gravity)
        scale = env.params.gravity_scale
        if step:
            direction = "sank" if step > 0 else "rose"
            target = (
                f"{scale + step:.2f}"
                if not isinstance(scale, Mapping)
                else f"each entry {step:+.2f}"
            )
            console.say(
                f"most loaded joints {direction}: recommended gravity_scale {_fmt(scale)} -> "
                f"{target}"
            )
        else:
            console.say(f"gravity_scale {_fmt(scale)}: no clear sink/rise tendency")
        data = _load_file(path)
        entry = dict(data.arms[side])
        entry["validated"] = passed
        entry["validated_at"] = _now_iso() if passed else None
        entry["verify"] = {
            "passed": passed,
            "drift_ok": drift_ok,
            "too_short": not full_length,
            "at": _now_iso(),
            "poses": len(drifts),
            "seconds": seconds,
            "max_abs_drift_counts": {
                name: float(max((w[k] for w in worst), default=0.0))
                for k, name in enumerate(joints)
            },
            "gravity_scale": scale,
            "gravity_scale_step": step,
        }
        data.arms = {**data.arms, side: entry}
        save_gravity_file(path, data)
        if passed:
            console.say(f"verify passed: the {side} arm is validated")
        elif not completed:
            console.say(f"verify incomplete ({len(drifts)}/{poses} poses): not validated")
        elif drift_ok:
            console.say(
                f"verify too short to validate ({poses} poses x {seconds:g} s; validation needs "
                f"{VERIFY_POSES} x {VERIFY_SECONDS:g} s): no drift, but not validated"
            )
        else:
            console.say(
                f"verify failed: drift of {VERIFY_DRIFT_LIMIT_COUNTS} counts or more; not validated"
            )
        _end_session(env, loop, end)
        return EXIT_OK if passed else EXIT_FAILED
    finally:
        leader.disconnect(torque_off=False)


# --- show -----------------------------------------------------------------------------


def cmd_show(env: CliEnv, *, path: str = DEFAULT_GRAVITY_FILE) -> int:
    """``show``: a summary of the identification file."""
    console = env.console
    data = _load_file(path)
    console.say(f"{path}: schema_version {data.schema_version}, {len(data.motors)} motors")
    console.say(f"  {'joint':<16} {'ref':>5} {'range':>11} {'sign':>4} {'drive':>5}")
    for name in RAKUDA_ARM_JOINT_NAMES:
        span = data.joint_range_counts.get(name)
        sign = data.current_sign.get(name)
        console.say(
            f"  {name:<16} {data.reference_counts.get(name, '-')!s:>5} "
            f"{'-' if span is None else f'{span[0]}..{span[1]}':>11} "
            f"{'-' if sign is None else f'{sign:+d}':>4} "
            f"{data.drive_mode.get(name, '-')!s:>5}"
        )
    if "torso_yaw" in data.reference_counts:
        console.say(f"  torso_yaw reference {data.reference_counts['torso_yaw']} (informational)")
    if data.sign_check:
        sc = data.sign_check
        console.say(
            f"sign_check: {sc.get('created_at')}, groups {_fmt(sc.get('drive_mode_groups', {}))}, "
            f"reverse_flips_current {sc.get('reverse_flips_current')}"
        )
    for side in ARM_SIDES:
        entry = data.arms.get(side)
        if entry is None:
            console.say(f"{side} arm: not identified")
            continue
        report = entry.get("fit_report", {})
        console.say(
            f"{side} arm: {entry.get('model_type')}, accepted {entry.get('accepted')}, "
            f"validated {entry.get('validated')} ({entry.get('validated_at') or '-'}), "
            f"{report.get('n_samples', '?')} poses, dataset "
            f"{str(entry.get('dataset_sha256') or '-')[:12]}"
        )
        console.say(f"  peak_ma: {_fmt(entry.get('peak_ma', {}))}")
    return EXIT_OK


# --- entry point ----------------------------------------------------------------------


def _exit_on_signal(signum: int, frame: FrameType | None) -> None:
    """Turns a termination signal into ``SystemExit(128 + signum)``."""
    raise SystemExit(128 + signum)


def install_signal_handlers() -> None:
    """SIGTERM/SIGHUP raise ``SystemExit`` so every ``finally`` holds the arm; SIGINT stays."""
    for signum in (signal.SIGTERM, signal.SIGHUP):
        signal.signal(signum, _exit_on_signal)


def _arm_side(value: str) -> str:
    aliases = {"r": "right", "right": "right", "l": "left", "left": "left"}
    side = aliases.get(value.strip().lower())
    if side is None:
        raise argparse.ArgumentTypeError(f"--arm must be right/left (or R/L), got {value!r}")
    return side


def _levels(value: str) -> Tuple[float, ...]:
    try:
        levels = tuple(float(v) for v in value.split(",") if v.strip())
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"--levels expects mA values like 40,60,80: {exc}")
    if not levels:
        raise argparse.ArgumentTypeError("--levels needs at least one value")
    return levels


def _positive_int(value: str) -> int:
    try:
        number = int(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected an integer, got {value!r}") from exc
    if number < 1:
        raise argparse.ArgumentTypeError(f"must be at least 1, got {number}")
    return number


def _positive_float(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(number) or number <= 0:
        raise argparse.ArgumentTypeError(f"must be a positive number, got {value!r}")
    return number


def _ridge(value: str) -> float:
    try:
        number = float(value)
    except ValueError as exc:
        raise argparse.ArgumentTypeError(f"expected a number, got {value!r}") from exc
    if not math.isfinite(number) or number < 0:
        raise argparse.ArgumentTypeError(f"--ridge must be finite and >= 0, got {value!r}")
    return number


def build_parser() -> argparse.ArgumentParser:
    """The ``robopy-rakuda-gravity`` argument parser."""
    common = argparse.ArgumentParser(add_help=False)
    common.add_argument(
        "--port", default=PORT_AUTO, help="leader port (default: auto = scan, or config.yaml)"
    )
    common.add_argument(
        "--file",
        default=DEFAULT_GRAVITY_FILE,
        help=f"identification file (default {DEFAULT_GRAVITY_FILE})",
    )
    common.add_argument("-v", "--verbose", action="store_true", help="debug logging")

    ending = argparse.ArgumentParser(add_help=False)
    group = ending.add_mutually_exclusive_group()
    group.add_argument(
        "--release",
        action="store_true",
        help="release the arm at the end (still asks once to support it)",
    )
    group.add_argument("--keep", action="store_true", help="keep holding at the end without asking")

    arm = argparse.ArgumentParser(add_help=False)
    arm.add_argument("--arm", type=_arm_side, required=True, help="right|left (or R|L)")

    parser = argparse.ArgumentParser(
        prog="robopy-rakuda-gravity",
        description=(
            "Leader gravity identification (range, sign-check, identify, fit, verify, show). "
            "Run as `env -u PYTHONPATH uv run --frozen robopy-rakuda-gravity <command>`."
        ),
    )
    sub = parser.add_subparsers(dest="command", required=True)
    sub.add_parser("range", parents=[common], help="reference pose and joint ranges (torque off)")
    sign = sub.add_parser(
        "sign-check", parents=[common, ending], help="GOAL_CURRENT sign of every arm joint"
    )
    sign.add_argument(
        "--levels", type=_levels, default=None, help="pulse levels in mA (default 40,60,80,120,160)"
    )
    identify = sub.add_parser(
        "identify", parents=[common, arm, ending], help="record operator-placed poses of one arm"
    )
    identify.add_argument("--dataset", default=None, help="dataset .npz (default next to --file)")
    identify.add_argument("--ridge", type=_ridge, default=1e-2, help="ridge of the trig fallback")
    fit = sub.add_parser("fit", parents=[common, arm], help="fit a saved dataset")
    fit.add_argument("--dataset", default=None, help="dataset .npz (default next to --file)")
    fit.add_argument("--ridge", type=_ridge, default=1e-2, help="ridge of the trig fallback")
    verify = sub.add_parser(
        "verify", parents=[common, arm, ending], help="drift test with gravity compensation only"
    )
    verify.add_argument(
        "--poses",
        type=_positive_int,
        default=VERIFY_POSES,
        help=f"poses to test (default {VERIFY_POSES}; fewer never validates)",
    )
    verify.add_argument(
        "--seconds",
        type=_positive_float,
        default=VERIFY_SECONDS,
        help=f"drift test per pose (default {VERIFY_SECONDS:g}; shorter never validates)",
    )
    sub.add_parser("show", parents=[common], help="print the identification file")
    return parser


def _end_action(args: argparse.Namespace) -> EndAction:
    if getattr(args, "release", False):
        return "release"
    if getattr(args, "keep", False):
        return "keep"
    return "ask"


def _default_env(args: argparse.Namespace) -> Tuple[CliEnv, str]:
    """The real environment: terminal, hardware bus, config.yaml params and port."""
    config = apply_rakuda_dotconfig(
        RakudaConfig(
            leader_port=args.port, follower_port=PORT_AUTO, bilateral=RakudaBilateralParams()
        )
    )
    assert config.bilateral is not None
    port = args.port if args.port != PORT_AUTO else config.leader_port
    check_sdk_location()
    return CliEnv(console=TerminalConsole(), params=config.bilateral), port


def _dispatch(env: CliEnv, args: argparse.Namespace, port: str) -> int:
    command, path = args.command, args.file
    if command == "range":
        return cmd_range(env, port=port, path=path)
    if command == "sign-check":
        return cmd_sign_check(
            env, port=port, path=path, levels_ma=args.levels, end=_end_action(args)
        )
    if command == "identify":
        return cmd_identify(
            env,
            port=port,
            path=path,
            arm=args.arm,
            dataset_path=args.dataset,
            ridge=args.ridge,
            end=_end_action(args),
        )
    if command == "fit":
        return cmd_fit(env, path=path, arm=args.arm, dataset_path=args.dataset, ridge=args.ridge)
    if command == "verify":
        return cmd_verify(
            env,
            port=port,
            path=path,
            arm=args.arm,
            poses=args.poses,
            seconds=args.seconds,
            end=_end_action(args),
        )
    return cmd_show(env, path=path)


def main(argv: Sequence[str] | None = None, *, env: CliEnv | None = None) -> int:
    """Entry point; returns the exit code.

    0: done; 1: a check failed, the loop faulted or the operator stopped early;
    2: bad arguments or a missing prerequisite; 130: Ctrl-C (the arm holds).
    SIGTERM/SIGHUP exit with ``128 + signum`` after the ``finally`` blocks held
    the arm.

    Args:
        argv: Command line without the program name (default ``sys.argv[1:]``).
        env: Injected environment (tests); ``None`` builds the real one.
    """
    install_signal_handlers()
    args = build_parser().parse_args(argv)
    logging.basicConfig(
        level=logging.DEBUG if args.verbose else logging.INFO,
        format="%(levelname)s %(name)s: %(message)s",
    )
    try:
        if env is None:
            if args.command in ("fit", "show"):
                env, port = CliEnv(console=TerminalConsole()), args.port
            else:
                env, port = _default_env(args)
        else:
            port = args.port
        return _dispatch(env, args, port)
    except CliError as exc:
        (env.console.say if env is not None else print)(str(exc))
        return exc.exit_code
    except KeyboardInterrupt:
        print("interrupted; any arm under control is holding", file=sys.stderr)
        return EXIT_INTERRUPT
    except (BilateralError, ConnectionError, OSError, ValueError) as exc:
        logger.error("%s", exc, exc_info=args.verbose)
        return EXIT_FAILED


if __name__ == "__main__":
    sys.exit(main())
