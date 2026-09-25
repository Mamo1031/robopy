"""Control laws of the Rakuda leader current loop.

Pure numpy: nothing here touches a bus, a port or a wall clock.  Every law is
driven by the ``dt`` the loop measured between two cycles and works in the
*joint convention* (a positive current pushes the joint toward increasing
encoder counts); the loop applies ``current_sign`` and the motor's
mA/LSB unit before writing ``GOAL_CURRENT``.

* :class:`BilateralLaw`: gravity compensation + one-sided joint-limit barrier
  + gated/ramped position-error feedback, rate-limited and clamped.
* :class:`PoseHoldLaw`: PD + conditional integrator that holds a pose during
  gravity identification (no pose planning, no collision detection).
* :class:`SignCheckLaw`: one displacement-terminated current pulse on a single
  joint, used to measure the sign of ``GOAL_CURRENT``.
* :class:`VelocityFilter` and :class:`CurrentShaper` are the two building
  blocks the laws share.

Units: positions in encoder counts, velocities in raw velocity counts
(0.229 rpm/LSB), currents in mA, times in seconds.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Protocol, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_JOINT_NAMES,
    RakudaArmState,
    RakudaBilateralParams,
)

__all__ = [
    "BilateralLaw",
    "ControlLaw",
    "CurrentShaper",
    "GravityModel",
    "IdentifyParams",
    "LawOutput",
    "PoseHoldLaw",
    "SignCheckLaw",
    "SignCheckOvershoot",
    "SignCheckParams",
    "VelocityFilter",
]

FloatArray = NDArray[np.float64]


# --- interface -----------------------------------------------------------------


@dataclass(frozen=True)
class LawOutput:
    """One cycle's result, every array in ``ControlLaw.joints`` order (joint convention).

    Attributes:
        current_ma: The current to command: already rate-limited and clamped.
        gravity_ma: Gravity-compensation term (never ramped).
        barrier_ma: One-sided joint-limit spring-damper term.
        feedback_ma: Position-error feedback term after gate and ramp.
        gate: Follower-freshness gate in ``0..1`` (first-order).
        ramp: Feedback ramp in ``0..1`` since the last engage.
    """

    current_ma: FloatArray
    gravity_ma: FloatArray
    barrier_ma: FloatArray
    feedback_ma: FloatArray
    gate: float
    ramp: float


class GravityModel(Protocol):
    """A fitted leader gravity model."""

    def predict_ma(self, q_counts: NDArray[np.integer[Any]]) -> FloatArray:
        """Gravity current for all 17 joints in ``RAKUDA_JOINT_NAMES`` order.

        Args:
            q_counts: ``(17,)`` present positions in encoder counts.

        Returns:
            ``(17,)`` mA in the joint convention; zero for joints without a model.
        """
        ...


class ControlLaw(Protocol):
    """What ``LeaderCurrentLoop`` needs from a law."""

    joints: Tuple[str, ...]

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        """(Re)initialises filters, ramp and the rate limiter from a fresh leader state."""
        ...

    def engage(self) -> None:
        """Restarts the feedback ramp and gate; the rate limiter keeps its state."""
        ...

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """The ``(J,)`` current ``configure()`` writes at torque-on."""
        ...

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """Computes the currents for one cycle; ``dt`` is the time since the previous one."""
        ...


class SignCheckOvershoot(RuntimeError):
    """A sign-check pulse moved the joint past ``pulse_abort_counts`` (fault).

    Attributes:
        joint: The pulsed joint.
        displacement_counts: Signed displacement from the pulse start.
        abort_counts: The limit that was crossed.
    """

    reason = "sign_check_overshoot"

    def __init__(self, joint: str, displacement_counts: int, abort_counts: int) -> None:
        self.joint = joint
        self.displacement_counts = displacement_counts
        self.abort_counts = abort_counts
        super().__init__(
            f"{self.reason}: {joint} moved {displacement_counts:+d} counts "
            f"(abort at ±{abort_counts})"
        )


# --- helpers -------------------------------------------------------------------


def _require_positive(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value <= 0:
        raise ValueError(f"{name} must be > 0, got {value!r}")
    return float(value)


def _require_non_negative(name: str, value: float) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    if not math.isfinite(value) or value < 0:
        raise ValueError(f"{name} must be >= 0, got {value!r}")
    return float(value)


def _joint_indices(joints: Sequence[str]) -> Tuple[Tuple[str, ...], NDArray[np.intp]]:
    """Validates ``joints`` and returns them with their indices into ``RAKUDA_JOINT_NAMES``.

    The order must be the canonical one: the loop pairs the law's output with
    its own per-joint tables (units, signs) by position, so a reordered tuple
    is rejected instead of silently sorted.
    """
    names = tuple(joints)
    if not names:
        raise ValueError("joints must not be empty")
    if len(set(names)) != len(names):
        raise ValueError(f"joints contains duplicates: {list(names)}")
    unknown = [name for name in names if name not in RAKUDA_JOINT_NAMES]
    if unknown:
        raise ValueError(f"unknown joint name(s): {unknown}")
    indices = np.array([RAKUDA_JOINT_NAMES.index(name) for name in names], dtype=np.intp)
    if len(names) > 1 and not bool(np.all(np.diff(indices) > 0)):
        raise ValueError(f"joints must be in RAKUDA_JOINT_NAMES order, got {list(names)}")
    return names, indices


def _validate_state(state: RakudaArmState, what: str) -> None:
    """Rejects a snapshot the laws cannot use (``ValueError`` -> fault)."""
    if tuple(state.names) != RAKUDA_JOINT_NAMES:
        raise ValueError(
            f"{what} state must carry the {len(RAKUDA_JOINT_NAMES)} Rakuda joints in "
            f"RAKUDA_JOINT_NAMES order, got {len(state.names)} names"
        )
    for attr in ("position", "velocity"):
        if not bool(np.all(np.isfinite(getattr(state, attr)))):
            raise ValueError(f"{what} state.{attr} contains non-finite values")


def _as_finite_float(name: str, value: Any) -> float:
    """``value`` as a float (Python or numpy scalar); bools, strings and non-finite are errors."""
    if isinstance(value, (bool, str, bytes)):
        raise ValueError(f"{name} must be a number, got {value!r}")
    try:
        result = float(value)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"{name} must be a number, got {value!r}") from exc
    if not math.isfinite(result):
        raise ValueError(f"{name} must be finite, got {value!r}")
    return result


def _check_dt(dt: float) -> float:
    value = _as_finite_float("dt", dt)
    if value < 0:
        raise ValueError(f"dt must be >= 0, got {dt!r}")
    return value


def _per_joint(
    values: Mapping[str, float] | ArrayLike, joints: Tuple[str, ...], what: str
) -> FloatArray:
    """``values`` as a finite ``(len(joints),)`` float array, from a mapping or a sequence."""
    if isinstance(values, Mapping):
        missing = [name for name in joints if name not in values]
        if missing:
            raise ValueError(f"{what} has no entry for joint(s) {missing}")
        array = np.array([float(values[name]) for name in joints], dtype=np.float64)
    else:
        array = np.asarray(values, dtype=np.float64)
        if array.shape != (len(joints),):
            raise ValueError(f"{what} must have shape {(len(joints),)}, got {array.shape}")
    if not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{what} contains non-finite values")
    return array


# --- building blocks -----------------------------------------------------------


class VelocityFilter:
    """First-order low-pass filter of a velocity vector.

    ``alpha = dt / (tau + dt)`` with ``tau = 1 / (2 pi cutoff_hz)``.  A cutoff of
    ``None`` passes the input through.  The first sample after construction (or
    :meth:`reset`) initialises the state, so a step at start-up is not smeared.
    """

    def __init__(self, cutoff_hz: float | None) -> None:
        self._tau: float | None = None
        if cutoff_hz is not None:
            self._tau = 1.0 / (2.0 * math.pi * _require_positive("cutoff_hz", cutoff_hz))
        self._value: FloatArray | None = None

    @property
    def value(self) -> FloatArray | None:
        """The filtered velocity, or ``None`` before the first sample."""
        return None if self._value is None else self._value.copy()

    def reset(self, velocity: ArrayLike | None = None) -> None:
        """Sets the state to ``velocity`` (``None``: the next sample initialises it)."""
        self._value = None if velocity is None else np.asarray(velocity, dtype=np.float64).copy()

    def update(self, velocity: ArrayLike, dt: float) -> FloatArray:
        """Feeds one sample and returns the filtered velocity."""
        sample = np.asarray(velocity, dtype=np.float64)
        if self._tau is None or self._value is None:
            self._value = sample.copy()
        else:
            alpha = dt / (self._tau + dt)
            self._value = self._value + alpha * (sample - self._value)
        return self._value.copy()


class CurrentShaper:
    """Rate limit, then clamp (order fixed).

    ``i_rl = i_prev + clip(target - i_prev, ±rate·dt)``, ``i_cmd = clip(i_rl, ±max)``;
    ``i_prev`` is the *clamped* output, so a saturated command backs off at the
    rate limit from the clamp, not from the unclamped value.
    """

    def __init__(self, rate_ma_per_s: float, max_ma: float) -> None:
        self.rate_ma_per_s = _require_positive("rate_ma_per_s", rate_ma_per_s)
        self.max_ma = _require_positive("max_ma", max_ma)
        self._prev: FloatArray | None = None

    @property
    def current_ma(self) -> FloatArray | None:
        """The last output (``i_prev``), or ``None`` before :meth:`reset`."""
        return None if self._prev is None else self._prev.copy()

    @property
    def saturated(self) -> NDArray[np.bool_]:
        """Per joint: the last output sits at ``±max_ma``.  All False before :meth:`reset`."""
        if self._prev is None:
            return np.zeros(0, dtype=np.bool_)
        return np.abs(self._prev) >= self.max_ma

    def reset(self, current_ma: ArrayLike) -> None:
        """Sets ``i_prev`` (clamped to ``±max_ma``)."""
        start = np.asarray(current_ma, dtype=np.float64)
        if not bool(np.all(np.isfinite(start))):
            raise ValueError("CurrentShaper.reset: current_ma contains non-finite values")
        self._prev = np.clip(start, -self.max_ma, self.max_ma).astype(np.float64)

    def shape(self, target_ma: ArrayLike, dt: float) -> FloatArray:
        """Moves toward ``target_ma`` by at most ``rate·dt`` and clamps; returns the output."""
        if self._prev is None:
            raise RuntimeError("CurrentShaper.reset() must run before shape()")
        target = np.asarray(target_ma, dtype=np.float64)
        step = self.rate_ma_per_s * dt
        limited = self._prev + np.clip(target - self._prev, -step, step)
        self._prev = np.clip(limited, -self.max_ma, self.max_ma).astype(np.float64)
        return self._prev.copy()


# --- bilateral law -------------------------------------------------------------


class BilateralLaw:
    """Gravity + joint-limit barrier + gated position feedback.

    Per cycle, on the ``J`` joints::

        v_hat = LPF(v_L)                                   # velocity_filter_hz
        i_g   = clip(k_g * G(q_L), ±gravity_clamp_factor * peak)   # no ramp
        i_lim = one-sided spring-damper outside range ± limit_margin_counts, ±limit_max_ma
        e     = q_F - q_L;  e_db = sign(e) * max(0, |e| - deadband)
        g    += alpha * (g* - g);  g* = 1 iff engaged and follower fresh
        r     = min(1, t_engaged / ramp_s)                 # 0 at the first cycle after engage
        i_fb  = g * r * clip(Kp_fb * e_db - Kd_fb * v_hat, ±feedback_max_ma)
        i_cmd = CurrentShaper(i_g + i_lim + i_fb)          # rate limit, then ±current_max_ma

    ``e < 0`` when the follower lags, so the feedback pulls the leader back toward
    the follower.  ``i_prev`` starts at the gravity term so the arm does not sink
    on engage; ``gravity_term()`` returns that same value for ``configure()``.

    Args:
        joints: The current-controlled joints, in ``RAKUDA_JOINT_NAMES`` order.
        params: Loop parameters.
        gravity: Fitted gravity model, or ``None`` for a zero gravity term.  The
            law itself does not gate that: ``start_bilateral()`` refuses to run
            without a model unless ``params.allow_uncompensated``.
        joint_range_counts: ``{joint: (lo, hi)}`` safe range from ``range``;
            the barrier engages ``limit_margin_counts`` inside it.  Kept on the
            law as the public ``joint_range_counts`` (``joints`` only), which
            ``LeaderCurrentLoop`` reads for its ``hard_margin_counts`` checks.
        engaged_clock: Optional monotonic clock in seconds.  When given, the
            ramp measures real time since the first cycle after engage;
            otherwise it accumulates ``dt``.
        gravity_peak_ma: ``{joint: peak}`` of the fitted model; the
            gravity term is clipped to ``±gravity_clamp_factor * peak``.  When
            ``None`` the clip is ``±current_max_ma``.

    ``params.gravity_scale`` is one factor for every joint or a ``{joint: factor}``
    mapping; a joint absent from the mapping uses ``1.0`` (the model as fitted).

    Raises:
        ValueError: On a joint outside ``RAKUDA_JOINT_NAMES``, a wrong order, a
            missing range or peak entry, a range narrower than twice the
            margin, or an out-of-range parameter.
    """

    joints: Tuple[str, ...]

    def __init__(
        self,
        joints: Sequence[str],
        params: RakudaBilateralParams,
        gravity: GravityModel | None,
        joint_range_counts: Mapping[str, Tuple[int, int]],
        *,
        engaged_clock: Callable[[], float] | None = None,
        gravity_peak_ma: Mapping[str, float] | None = None,
    ) -> None:
        self.joints, self._idx = _joint_indices(joints)
        self._params = params
        self._gravity = gravity
        self._clock = engaged_clock
        n = len(self.joints)

        if not 0.0 < params.feedback_gate_alpha <= 1.0:
            raise ValueError(
                f"feedback_gate_alpha must be within (0, 1], got {params.feedback_gate_alpha}"
            )
        _require_non_negative("ramp_s", params.ramp_s)
        _require_non_negative("follower_stale_s", params.follower_stale_s)
        _require_non_negative("limit_margin_counts", params.limit_margin_counts)
        _require_non_negative("limit_max_ma", params.limit_max_ma)
        _require_non_negative("feedback_deadband_counts", params.feedback_deadband_counts)
        _require_non_negative("feedback_max_ma", params.feedback_max_ma)
        _require_positive("gravity_clamp_factor", params.gravity_clamp_factor)

        scale = params.gravity_scale
        if isinstance(scale, Mapping):
            self._k_g = np.array([float(scale.get(name, 1.0)) for name in self.joints])
        else:
            self._k_g = np.full(n, float(scale), dtype=np.float64)
        if gravity_peak_ma is None:
            self._gravity_bound = np.full(n, float(params.current_max_ma), dtype=np.float64)
        else:
            peak = _per_joint(gravity_peak_ma, self.joints, "gravity_peak_ma")
            if bool(np.any(peak < 0)):
                raise ValueError("gravity_peak_ma must be non-negative")
            self._gravity_bound = params.gravity_clamp_factor * peak

        missing = [name for name in self.joints if name not in joint_range_counts]
        if missing:
            raise ValueError(f"joint_range_counts has no entry for joint(s) {missing}")
        lo = np.array([float(joint_range_counts[name][0]) for name in self.joints])
        hi = np.array([float(joint_range_counts[name][1]) for name in self.joints])
        self._q_lo = lo + params.limit_margin_counts
        self._q_hi = hi - params.limit_margin_counts
        too_narrow = [name for name, a, b in zip(self.joints, self._q_lo, self._q_hi) if a >= b]
        if too_narrow:
            raise ValueError(
                f"joint_range_counts narrower than 2 * limit_margin_counts "
                f"({2 * params.limit_margin_counts}) for {too_narrow}"
            )
        self.joint_range_counts: Dict[str, Tuple[int, int]] = {
            name: (int(lo[k]), int(hi[k])) for k, name in enumerate(self.joints)
        }

        self._filter = VelocityFilter(params.velocity_filter_hz)
        self._shaper = CurrentShaper(params.current_rate_ma_per_s, params.current_max_ma)
        self._gate = 0.0
        self._engaged = False
        self._t_engaged = 0.0
        self._t0: float | None = None
        self._ready = False

    # -- state ------------------------------------------------------------------

    @property
    def engaged(self) -> bool:
        return self._engaged

    @property
    def gate(self) -> float:
        return self._gate

    @property
    def current_ma(self) -> FloatArray | None:
        """The last commanded current (``i_prev``), or ``None`` before :meth:`reset`."""
        return self._shaper.current_ma

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        """Re-initialises filter, gate, ramp and ``i_prev`` (to the gravity term).

        Args:
            leader: A fresh leader snapshot.
            engaged: ``True`` also calls :meth:`engage`.
        """
        _validate_state(leader, "leader")
        self._filter.reset(leader.velocity[self._idx])
        self._shaper.reset(self._gravity_term(leader.position))
        self._gate = 0.0
        self._engaged = False
        self._t_engaged = 0.0
        self._t0 = None
        self._ready = True
        if engaged:
            self.engage()

    def engage(self) -> None:
        """Starts (or restarts) the feedback ramp and gate from zero (``re_engage()``).

        Unlike :meth:`reset` this keeps the filter and ``i_prev``, so the
        commanded current stays continuous.
        """
        self._engaged = True
        self._gate = 0.0
        self._t_engaged = 0.0
        self._t0 = None

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """``clip(k_g * G(q_L), ±bound)`` on ``joints``: what ``configure()`` writes first."""
        _validate_state(leader, "leader")
        return self._gravity_term(leader.position)

    # -- one cycle --------------------------------------------------------------

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """Runs the law of the class docstring for one cycle.

        Args:
            leader: The leader snapshot of this cycle.
            follower: The latest follower snapshot, possibly stale, or ``None``.
            follower_age_s: Age of ``follower`` at this cycle (``None`` when unknown).
            dt: Seconds since the previous ``compute``.

        Raises:
            ValueError: Invalid snapshot (wrong joints, non-finite) or ``dt``.
            RuntimeError: :meth:`reset` has not run.
        """
        if not self._ready:
            raise RuntimeError("BilateralLaw.reset() must run before compute()")
        dt = _check_dt(dt)
        _validate_state(leader, "leader")
        if follower is not None:
            _validate_state(follower, "follower")
        p = self._params

        q = leader.position[self._idx].astype(np.float64)
        v_hat = self._filter.update(leader.velocity[self._idx], dt)
        i_g = self._gravity_term(leader.position)
        i_lim = self._barrier(q, v_hat)

        fresh = (
            follower is not None
            and follower_age_s is not None
            and follower_age_s <= p.follower_stale_s
        )
        target = 1.0 if (self._engaged and fresh) else 0.0
        self._gate += p.feedback_gate_alpha * (target - self._gate)
        ramp = self._advance_ramp(dt)

        if follower is None:
            e = np.zeros(len(self.joints), dtype=np.float64)
        else:
            e = (follower.position[self._idx] - leader.position[self._idx]).astype(np.float64)
        e_db = np.sign(e) * np.maximum(0.0, np.abs(e) - p.feedback_deadband_counts)
        raw_fb = p.feedback_kp_ma_per_count * e_db - p.feedback_kd_ma_per_vcount * v_hat
        i_fb = self._gate * ramp * np.clip(raw_fb, -p.feedback_max_ma, p.feedback_max_ma)

        i_cmd = self._shaper.shape(i_g + i_lim + i_fb, dt)
        return LawOutput(
            current_ma=i_cmd,
            gravity_ma=i_g,
            barrier_ma=i_lim,
            feedback_ma=i_fb.astype(np.float64),
            gate=self._gate,
            ramp=ramp,
        )

    # -- terms ------------------------------------------------------------------

    def _gravity_term(self, q_all: NDArray[np.integer[Any]]) -> FloatArray:
        if self._gravity is None:
            return np.zeros(len(self.joints), dtype=np.float64)
        prediction = np.asarray(self._gravity.predict_ma(q_all), dtype=np.float64)
        if prediction.shape != (len(RAKUDA_JOINT_NAMES),):
            raise ValueError(
                f"GravityModel.predict_ma must return shape {(len(RAKUDA_JOINT_NAMES),)}, "
                f"got {prediction.shape}"
            )
        if not bool(np.all(np.isfinite(prediction))):
            raise ValueError("GravityModel.predict_ma returned non-finite values")
        scaled = self._k_g * prediction[self._idx]
        return np.clip(scaled, -self._gravity_bound, self._gravity_bound).astype(np.float64)

    def _barrier(self, q: FloatArray, v_hat: FloatArray) -> FloatArray:
        p = self._params
        above = q - self._q_hi
        below = q - self._q_lo
        excess = np.where(above > 0.0, above, np.where(below < 0.0, below, 0.0))
        active = excess != 0.0
        spring_damper = -p.limit_kp_ma_per_count * excess - p.limit_kd_ma_per_vcount * v_hat
        i_lim = np.where(active, spring_damper, 0.0)
        return np.clip(i_lim, -p.limit_max_ma, p.limit_max_ma).astype(np.float64)

    def _advance_ramp(self, dt: float) -> float:
        if not self._engaged:
            return 0.0
        if self._clock is not None:
            now = float(self._clock())
            if self._t0 is None:
                self._t0 = now
            elapsed = now - self._t0
        else:
            elapsed = self._t_engaged
            self._t_engaged += dt
        ramp_s = self._params.ramp_s
        if ramp_s <= 0.0:
            return 1.0
        return min(1.0, max(0.0, elapsed / ramp_s))


# --- identification laws -------------------------------------------------------


@dataclass(frozen=True)
class IdentifyParams:
    """Tunables of :class:`PoseHoldLaw`.

    Only what the manual-pose hold consumes lives here; the knobs of the
    dropped unattended identification (goal ramp, collision detection, pose
    planning) were removed with it.

    Attributes:
        Kp_s: Hold stiffness [mA/count].
        Kd_s: Hold damping [mA/vcount].
        Ki_s: Conditional integral gain [mA/(count·s)].
        id_current_max_ma: Clamp of the hold current (``CurrentShaper``).
        id_current_rate_ma_per_s: Rate limit of the hold current.
        id_int_error_counts: The integrator runs only while ``|e|`` is at most this.
        id_int_deadband_counts: ... and not when ``|e|`` is at most this.
        velocity_filter_hz: Low-pass cutoff of the velocity term (``None``: raw).
    """

    id_int_error_counts: int = 40
    id_int_deadband_counts: int = 3
    id_current_max_ma: float = 300.0
    id_current_rate_ma_per_s: float = 1500.0
    Kp_s: float = 1.5
    Kd_s: float = 0.8
    Ki_s: float = 3.0
    velocity_filter_hz: float | None = 20.0

    def __post_init__(self) -> None:
        for name in ("id_int_error_counts", "id_current_max_ma", "id_current_rate_ma_per_s"):
            _require_positive(f"IdentifyParams.{name}", getattr(self, name))
        for name in ("Kp_s", "Kd_s", "Ki_s", "id_int_deadband_counts"):
            _require_non_negative(f"IdentifyParams.{name}", getattr(self, name))
        if self.velocity_filter_hz is not None:
            _require_positive("IdentifyParams.velocity_filter_hz", self.velocity_filter_hz)
        if self.id_int_deadband_counts >= self.id_int_error_counts:
            raise ValueError(
                "IdentifyParams.id_int_deadband_counts must be below id_int_error_counts "
                f"({self.id_int_deadband_counts} >= {self.id_int_error_counts})"
            )
        integral_window = self.Ki_s * self.id_int_error_counts
        if integral_window > 0.5 * self.id_current_max_ma:
            raise ValueError(
                "IdentifyParams: Ki_s * id_int_error_counts must not exceed "
                f"0.5 * id_current_max_ma ({integral_window} > {0.5 * self.id_current_max_ma})"
            )


class PoseHoldLaw:
    """PD + conditional integrator holding ``joints`` at a goal pose.

    ``e = q_goal - q``, ``i = i_int + Kp_s * e - Kd_s * v_hat``, then
    :class:`CurrentShaper` (``id_current_rate_ma_per_s``, ``±id_current_max_ma``).
    The integrator advances by ``Ki_s * e * dt`` only while
    ``id_int_deadband_counts < |e| <= id_int_error_counts`` **and** the previous
    output was not saturated; it is clamped to ``±id_current_max_ma``.

    :meth:`reset` targets the leader's present position (a hold in place) and
    starts from 0 mA; call :meth:`set_goal` afterwards to move.  ``follower``
    and ``follower_age_s`` are ignored; ``gate`` and ``ramp`` report 1.
    """

    joints: Tuple[str, ...]

    def __init__(self, joints: Sequence[str], params: IdentifyParams) -> None:
        self.joints, self._idx = _joint_indices(joints)
        self._params = params
        self._filter = VelocityFilter(params.velocity_filter_hz)
        self._shaper = CurrentShaper(params.id_current_rate_ma_per_s, params.id_current_max_ma)
        n = len(self.joints)
        self._goal: FloatArray | None = None
        self._i_int = np.zeros(n, dtype=np.float64)
        self._error = np.zeros(n, dtype=np.float64)

    @property
    def goal_counts(self) -> FloatArray | None:
        return None if self._goal is None else self._goal.copy()

    @property
    def integrator_ma(self) -> FloatArray:
        return self._i_int.copy()

    @property
    def error_counts(self) -> FloatArray:
        """``q_goal - q`` of the last cycle."""
        return self._error.copy()

    @property
    def saturated(self) -> NDArray[np.bool_]:
        """Per joint: the last output sits at ``±id_current_max_ma``."""
        return self._shaper.saturated

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        """Holds in place from 0 mA: goal := present position, integrator := 0."""
        _validate_state(leader, "leader")
        self._goal = leader.position[self._idx].astype(np.float64)
        self._filter.reset(leader.velocity[self._idx])
        self._shaper.reset(np.zeros(len(self.joints)))
        self._i_int = np.zeros(len(self.joints), dtype=np.float64)
        self._error = np.zeros(len(self.joints), dtype=np.float64)

    def engage(self) -> None:
        """No-op: the hold has no follower feedback to ramp."""

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """Zero: the hold starts from 0 mA so its first cycle runs under the loop's checks."""
        _validate_state(leader, "leader")
        return np.zeros(len(self.joints), dtype=np.float64)

    def set_goal(self, q_goal: Mapping[str, float] | ArrayLike) -> None:
        """Sets the goal pose in counts: a ``{joint: counts}`` mapping or a ``(J,)`` array.

        Raises:
            RuntimeError: :meth:`reset` has not run (it would overwrite the goal).
            ValueError: Missing joint, wrong shape or non-finite value.
        """
        if self._goal is None:
            raise RuntimeError("PoseHoldLaw.reset() must run before set_goal()")
        self._goal = _per_joint(q_goal, self.joints, "q_goal")

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """One cycle of the hold; see the class docstring."""
        if self._goal is None:
            raise RuntimeError("PoseHoldLaw.reset() must run before compute()")
        dt = _check_dt(dt)
        _validate_state(leader, "leader")
        p = self._params

        q = leader.position[self._idx].astype(np.float64)
        v_hat = self._filter.update(leader.velocity[self._idx], dt)
        e = self._goal - q
        abs_e = np.abs(e)
        integrate = (
            (abs_e > p.id_int_deadband_counts)
            & (abs_e <= p.id_int_error_counts)
            & ~self._shaper.saturated
        )
        self._i_int = np.clip(
            self._i_int + np.where(integrate, p.Ki_s * e * dt, 0.0),
            -p.id_current_max_ma,
            p.id_current_max_ma,
        ).astype(np.float64)
        i_pid = (self._i_int + p.Kp_s * e - p.Kd_s * v_hat).astype(np.float64)
        i_cmd = self._shaper.shape(i_pid, dt)
        self._error = e
        zeros = np.zeros(len(self.joints), dtype=np.float64)
        return LawOutput(
            current_ma=i_cmd,
            gravity_ma=zeros,
            barrier_ma=zeros.copy(),
            feedback_ma=i_pid,
            gate=1.0,
            ramp=1.0,
        )


@dataclass(frozen=True)
class SignCheckParams:
    """Tunables of ``sign-check`` and :class:`SignCheckLaw`.

    Attributes:
        current_levels_ma: Pulse amplitudes tried in ascending order.
        pulse_max_s: A pulse ends after this long even if nothing moved.
        pulse_stop_counts: A pulse ends (current 0 at once) at this displacement.
        pulse_stop_velocity_counts: ... or at this speed.
        pulse_abort_counts: Beyond this displacement the law raises
            :class:`SignCheckOvershoot` (fault ``sign_check_overshoot``).
        settle_s: Wait after a pulse before measuring the displacement (CLI).
        move_threshold_counts: Minimum displacement that counts as a verdict (CLI).
        range_guard_counts: A joint closer than this to its range limit is skipped (CLI).
        rate_ma_per_s: Ramp of the pulse current.
        between_pulses_s: Pause between the + and - pulse (CLI).
    """

    current_levels_ma: Tuple[float, ...] = (40.0, 60.0, 80.0, 120.0, 160.0)
    pulse_max_s: float = 0.3
    pulse_stop_counts: int = 20
    pulse_stop_velocity_counts: int = 60
    pulse_abort_counts: int = 80
    settle_s: float = 0.5
    move_threshold_counts: int = 10
    range_guard_counts: int = 80
    rate_ma_per_s: float = 3000.0
    between_pulses_s: float = 1.0

    def __post_init__(self) -> None:
        levels = tuple(float(level) for level in self.current_levels_ma)
        object.__setattr__(self, "current_levels_ma", levels)
        if not levels:
            raise ValueError("SignCheckParams.current_levels_ma must not be empty")
        for level in levels:
            _require_positive("SignCheckParams.current_levels_ma[*]", level)
        if any(b <= a for a, b in zip(levels, levels[1:])):
            raise ValueError(
                f"SignCheckParams.current_levels_ma must be strictly increasing, got {levels}"
            )
        for name in (
            "pulse_max_s",
            "pulse_stop_counts",
            "pulse_stop_velocity_counts",
            "pulse_abort_counts",
            "move_threshold_counts",
            "rate_ma_per_s",
        ):
            _require_positive(f"SignCheckParams.{name}", getattr(self, name))
        for name in ("settle_s", "range_guard_counts", "between_pulses_s"):
            _require_non_negative(f"SignCheckParams.{name}", getattr(self, name))
        if self.pulse_abort_counts <= self.pulse_stop_counts:
            raise ValueError(
                "SignCheckParams.pulse_abort_counts must exceed pulse_stop_counts "
                f"({self.pulse_abort_counts} <= {self.pulse_stop_counts})"
            )
        if self.move_threshold_counts > self.pulse_stop_counts:
            raise ValueError(
                "SignCheckParams.move_threshold_counts must not exceed pulse_stop_counts "
                f"({self.move_threshold_counts} > {self.pulse_stop_counts})"
            )


class SignCheckLaw:
    """One current pulse on one joint, cut by displacement, speed or time.

    From :meth:`reset` the output ramps at ``rate_ma_per_s`` toward ``level_ma``
    (signed: pass a negative level for the ``-`` pulse) and drops to exactly 0
    as soon as ``|q - q_start| >= pulse_stop_counts``, ``|v| >= pulse_stop_velocity_counts``
    or ``pulse_max_s`` has elapsed.  ``|q - q_start| >= pulse_abort_counts``
    (checked every cycle, also after the stop) raises :class:`SignCheckOvershoot`
    so the loop faults and holds.  No gravity, no barrier (the sign is unknown).
    """

    joints: Tuple[str, ...]

    def __init__(self, joint: str, level_ma: float, params: SignCheckParams) -> None:
        self.joints, self._idx = _joint_indices((joint,))
        self.level_ma = _as_finite_float("level_ma", level_ma)
        if self.level_ma == 0:
            raise ValueError("level_ma must be non-zero")
        self._params = params
        self._shaper = CurrentShaper(params.rate_ma_per_s, abs(self.level_ma))
        self._q_start: int | None = None
        self._displacement = 0
        self._elapsed = 0.0
        self._stopped = False
        self._stop_reason: str | None = None
        self._abort = False

    @property
    def joint(self) -> str:
        return self.joints[0]

    @property
    def stopped(self) -> bool:
        """The pulse has ended (output 0)."""
        return self._stopped

    @property
    def stop_reason(self) -> str | None:
        """``"displacement"``, ``"velocity"``, ``"timeout"``, ``"abort"`` or ``None``."""
        return self._stop_reason

    @property
    def abort(self) -> bool:
        """The joint crossed ``pulse_abort_counts``."""
        return self._abort

    @property
    def displacement(self) -> int:
        """Signed ``q - q_start`` in counts at the last cycle."""
        return self._displacement

    @property
    def elapsed_s(self) -> float:
        """``dt`` accumulated since the first cycle after :meth:`reset`."""
        return self._elapsed

    def engage(self) -> None:
        """No-op: a pulse has nothing to ramp."""

    def gravity_term(self, leader: RakudaArmState) -> FloatArray:
        """Zero: the pulse ramps from 0 mA inside the loop (the sign is unknown)."""
        _validate_state(leader, "leader")
        return np.zeros(1, dtype=np.float64)

    def reset(self, leader: RakudaArmState, *, engaged: bool = False) -> None:
        """Starts a new pulse from the present position."""
        _validate_state(leader, "leader")
        self._q_start = int(leader.position[self._idx[0]])
        self._shaper.reset(np.zeros(1))
        self._displacement = 0
        self._elapsed = 0.0
        self._stopped = False
        self._stop_reason = None
        self._abort = False

    def compute(
        self,
        leader: RakudaArmState,
        follower: RakudaArmState | None,
        follower_age_s: float | None,
        dt: float,
    ) -> LawOutput:
        """One cycle of the pulse; see the class docstring.

        Raises:
            SignCheckOvershoot: The displacement reached ``pulse_abort_counts``.
        """
        if self._q_start is None:
            raise RuntimeError("SignCheckLaw.reset() must run before compute()")
        dt = _check_dt(dt)
        _validate_state(leader, "leader")
        p = self._params

        self._displacement = int(leader.position[self._idx[0]]) - self._q_start
        velocity = int(leader.velocity[self._idx[0]])
        if abs(self._displacement) >= p.pulse_abort_counts:
            self._stop("abort")
            self._abort = True
            raise SignCheckOvershoot(self.joint, self._displacement, p.pulse_abort_counts)
        if not self._stopped:
            if abs(self._displacement) >= p.pulse_stop_counts:
                self._stop("displacement")
            elif abs(velocity) >= p.pulse_stop_velocity_counts:
                self._stop("velocity")
            elif self._elapsed >= p.pulse_max_s:
                self._stop("timeout")
        self._elapsed += dt

        target = 0.0 if self._stopped else self.level_ma
        i_cmd = self._shaper.shape(np.array([target]), dt)
        zeros = np.zeros(1, dtype=np.float64)
        return LawOutput(
            current_ma=i_cmd,
            gravity_ma=zeros,
            barrier_ma=zeros.copy(),
            feedback_ma=i_cmd.copy(),
            gate=1.0,
            ramp=1.0,
        )

    def _stop(self, reason: str) -> None:
        self._stopped = True
        self._stop_reason = reason
        self._shaper.reset(np.zeros(1))
