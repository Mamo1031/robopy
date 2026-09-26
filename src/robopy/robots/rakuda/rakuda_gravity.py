"""Leader gravity model of the Rakuda arms: model, fitting and storage.

Pure numpy apart from the axis refinement of :func:`fit_arm` (scipy, imported
lazily).  Nothing here touches a bus; the identification procedure itself
(``range`` / ``sign-check`` / ``identify`` / ``verify``) lives in
``rakuda_gravity_cli``; ``python -m robopy.robots.rakuda.rakuda_gravity`` runs it.

**Model** (``poe_first_moment``, per arm).  Body frame B: x forward, y left,
z up; gravity ``g = (0, 0, -1)`` (its magnitude is absorbed by the moments).
With ``theta_j = (q_j - q_ref_j) * 2*pi/4096`` and the space-frame product of
exponentials ``R_k = exp([a_1] theta_1) ... exp([a_k] theta_k)`` (``a_j`` the
unit axis of joint ``j`` at the reference pose), the potential is
``V = -g . sum_k R_k L_k`` (``L_k`` the lumped first moment of link ``k`` and
everything distal to it) and the holding current in the joint convention is::

    i_j = dV/dtheta_j = -(g x w_j) . sum_{k>=j} R_k L_k,   w_j = R_{j-1} a_j

which is linear in the 18 moments.  Only 12 combinations are identifiable
(the component of ``L_k`` along ``a_k`` is indistinguishable from moving it
to the previous link), so ``L`` is the minimum-norm least-squares solution.
Gravity cannot see a rotation of the whole arm about the vertical, nor a
reflection through a vertical plane, so the axes are only determined up to
those; :func:`fit_arm` fixes that gauge (the first horizontal axis keeps the
azimuth and sign of its idealised start).

**Fallback** (``trig_gradient``): a trigonometric potential with every joint
and every joint pair (``1 + 12 + 60 = 73`` terms) whose analytic gradient is
fitted to all six joints at once (ridge).

**Storage**: ``.robopy/rakuda/leader_gravity.json`` (:class:`GravityFile`,
``schema_version`` 1) and one ``.npz`` dataset per arm (:func:`save_dataset`).

Units: positions in encoder counts, currents in mA (joint convention: a
positive current pushes the joint toward increasing counts), angles in rad.
"""

from __future__ import annotations

import contextlib
import functools
import hashlib
import io
import json
import logging
import math
import numbers
import os
import shutil
import tempfile
from dataclasses import dataclass, field
from itertools import product
from typing import TYPE_CHECKING, Any, Callable, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import ArrayLike, NDArray

from robopy.config.robot_config.rakuda_config import RAKUDA_ARM_JOINT_NAMES, RAKUDA_JOINT_NAMES

if TYPE_CHECKING:
    from .rakuda_pair_sys import BilateralSetup

__all__ = [
    "ARM_SIDES",
    "MIN_FIT_SAMPLES",
    "MODEL_POE",
    "MODEL_TRIG",
    "SCHEMA_VERSION",
    "ArmGravityFit",
    "GravityFile",
    "LeaderGravityModel",
    "PoseSample",
    "arm_fit_from_dict",
    "arm_fit_to_dict",
    "arm_joints",
    "fit_arm",
    "load_bilateral_setup",
    "load_dataset",
    "load_gravity_file",
    "save_dataset",
    "save_gravity_file",
]

logger = logging.getLogger(__name__)

FloatArray = NDArray[np.float64]

ARM_SIDES: Tuple[str, ...] = ("right", "left")
_SIDE_PREFIX = {"right": "r_arm_", "left": "l_arm_"}

SCHEMA_VERSION = 1
MODEL_POE = "poe_first_moment"
MODEL_TRIG = "trig_gradient"
#: Fewer poses than this are never fitted.
MIN_FIT_SAMPLES = 25

_N_JOINTS = 6
_N_MOMENTS = 3 * _N_JOINTS
_N_TRIG = 1 + 2 * _N_JOINTS + 4 * (_N_JOINTS * (_N_JOINTS - 1) // 2)
_RAD_PER_COUNT = 2.0 * math.pi / 4096.0
#: Relative singular-value cut-off of every PoE least-squares solve.
_RCOND = 1e-6
#: Each of the two tilt angles of a refined axis stays within this of its start.
_REFINE_BOUND_RAD = math.pi / 4.0
#: Default number of best screened axis candidates refined nonlinearly.  Kept
#: well above 8: at idealised axes the screen cannot rank the signs of the
#: distal joints, so the true start can sit well below the top 8.
_REFINE_TOP_K = 32
#: Candidates are screened this many at a time (bounds the temporary arrays).
_SCREEN_CHUNK = 128

# Acceptance, evaluated per joint on the validation poses.
_ACCEPT_RMS_FLOOR_MA = 5.0
_ACCEPT_RMS_RANGE_FRACTION = 0.10
_ACCEPT_MAX_FLOOR_MA = 10.0
_ACCEPT_MAX_FRICTION_FACTOR = 0.8

# Signed axis codes: 0..2 = +x, +y, +z; 3..5 = -x, -y, -z.
_AXIS_LABELS = ("+x", "+y", "+z", "-x", "-y", "-z")
_SIGNED_UNIT = np.concatenate((np.eye(3), -np.eye(3)))
#: Joint type of an axis in B at the reference pose: roll (x), pitch (y), yaw (z).
_AXIS_LETTERS = "RPY"
#: Preference of the canonical representative: +y first, then positive signs.
_CANONICAL_RANK = (1, 0, 2, 4, 5, 3)
#: The arrays each model stores in its ``arms.<side>`` entry, with their shapes.
_MODEL_ARRAYS: Dict[str, Dict[str, Tuple[int, ...]]] = {
    MODEL_POE: {"axes": (_N_JOINTS, 3), "L": (_N_MOMENTS,)},
    MODEL_TRIG: {"trig_coef": (_N_TRIG,)},
}
#: ``ArmGravityFit`` array field -> its key in the JSON entry and in ``_MODEL_ARRAYS``.
_ARRAY_FIELDS = {"axes": "axes", "first_moments": "L", "trig_coef": "trig_coef"}
#: ``(15, 2)`` joint pairs ``j < k`` of the trigonometric potential.
_TRIG_PAIRS = np.array([(j, k) for j in range(_N_JOINTS) for k in range(j + 1, _N_JOINTS)])
#: ``[j, k] = 1`` where link ``k`` is distal to (or is) joint ``j``.
_DISTAL = np.triu(np.ones((_N_JOINTS, _N_JOINTS)))


def arm_joints(side: str) -> Tuple[str, ...]:
    """The six joints of one arm in chain order (shoulder to wrist).

    Raises:
        ValueError: ``side`` is not ``"right"`` or ``"left"``.
    """
    if side not in _SIDE_PREFIX:
        raise ValueError(f"arm side must be one of {ARM_SIDES}, got {side!r}")
    prefix = _SIDE_PREFIX[side]
    return tuple(name for name in RAKUDA_ARM_JOINT_NAMES if name.startswith(prefix))


# --- samples -------------------------------------------------------------------


def _vector(value: ArrayLike, what: str, *, finite: bool = True) -> FloatArray:
    """``value`` as a read-only ``(6,)`` float array."""
    array = np.array(value, dtype=np.float64)
    if array.shape != (_N_JOINTS,):
        raise ValueError(f"{what} must have shape ({_N_JOINTS},), got {array.shape}")
    if finite and not bool(np.all(np.isfinite(array))):
        raise ValueError(f"{what} contains non-finite values")
    array.flags.writeable = False
    return array


@dataclass(frozen=True, eq=False)
class PoseSample:
    """One identified pose of one arm (joint convention).

    Attributes:
        q_counts: ``(6,)`` the held pose ``p`` in counts.
        i_plus_ma: ``(6,)`` settled commanded current at ``p`` approached from ``p + delta``.
        i_minus_ma: ``(6,)`` the same approached from ``p - delta``.
        temperature_c: ``(6,)`` motor temperatures during the measurement.
        flags: Free-form markers of the capture (for example a noisy average).
        is_val: ``True`` for a validation pose (never used to fit).
    """

    q_counts: FloatArray
    i_plus_ma: FloatArray
    i_minus_ma: FloatArray
    temperature_c: FloatArray
    flags: Tuple[str, ...] = ()
    is_val: bool = False

    def __post_init__(self) -> None:
        for name in ("q_counts", "i_plus_ma", "i_minus_ma"):
            object.__setattr__(self, name, _vector(getattr(self, name), f"PoseSample.{name}"))
        temperature = _vector(self.temperature_c, "PoseSample.temperature_c", finite=False)
        object.__setattr__(self, "temperature_c", temperature)
        object.__setattr__(self, "flags", tuple(str(flag) for flag in self.flags))
        object.__setattr__(self, "is_val", bool(self.is_val))

    @property
    def i_gravity_ma(self) -> FloatArray:
        """``(i_plus + i_minus) / 2``: the gravity current (friction cancels)."""
        return np.asarray((self.i_plus_ma + self.i_minus_ma) / 2.0, dtype=np.float64)

    @property
    def friction_ma(self) -> FloatArray:
        """``|i_plus - i_minus| / 2``: the static friction seen at this pose."""
        return np.asarray(np.abs(self.i_plus_ma - self.i_minus_ma) / 2.0, dtype=np.float64)


# --- model math ----------------------------------------------------------------


def _skew(v: FloatArray) -> FloatArray:
    """``(..., 3) -> (..., 3, 3)`` cross-product matrices."""
    x, y, z = v[..., 0], v[..., 1], v[..., 2]
    zero = np.zeros_like(x)
    rows = (
        np.stack((zero, -z, y), axis=-1),
        np.stack((z, zero, -x), axis=-1),
        np.stack((-y, x, zero), axis=-1),
    )
    return np.stack(rows, axis=-2)


def _chain(theta: FloatArray, axes: FloatArray) -> Tuple[FloatArray, FloatArray]:
    """Cumulative PoE rotations and gravity lever vectors for many axis sets at once.

    Args:
        theta: ``(N, 6)`` joint angles from the reference pose (rad).
        axes: ``(C, 6, 3)`` unit axes at the reference pose.

    Returns:
        ``R`` of shape ``(C, N, 6, 3, 3)`` with ``R[..., k, :, :] = R_k`` and
        ``m`` of shape ``(C, N, 6, 3)`` with ``m_j = e_z x w_j = -(g x w_j)``, so
        that ``i_j = m_j . sum_{k>=j} R_k L_k``.
    """
    k = _skew(axes)[:, None]
    sin = np.sin(theta)[None, :, :, None, None]
    cos = np.cos(theta)[None, :, :, None, None]
    # Rodrigues: exp([a] t) = I + sin(t) [a] + (1 - cos(t)) [a]^2.
    step = np.eye(3) + sin * k + (1.0 - cos) * (k @ k)
    rot = np.empty_like(step)
    rot[:, :, 0] = step[:, :, 0]
    for j in range(1, _N_JOINTS):
        rot[:, :, j] = rot[:, :, j - 1] @ step[:, :, j]
    w = np.empty(rot.shape[:3] + (3,))
    w[:, :, 0] = axes[:, None, 0]
    w[:, :, 1:] = np.einsum("cnjab,cjb->cnja", rot[:, :, :-1], axes[:, 1:])
    lever = np.stack((-w[..., 1], w[..., 0], np.zeros(w.shape[:-1])), axis=-1)
    return rot, lever


def _poe_regressor(theta: FloatArray, axes: FloatArray) -> FloatArray:
    """``Phi`` of shape ``(C, N, 6, 18)`` with ``i = Phi @ L`` (``L`` flattened link-major)."""
    rot, lever = _chain(theta, axes)
    phi = np.einsum("cnja,cnkab->cnjkb", lever, rot) * _DISTAL[:, :, None]
    return phi.reshape(phi.shape[:3] + (_N_MOMENTS,))


def _poe_currents(theta: FloatArray, axes: FloatArray, moments: FloatArray) -> FloatArray:
    """``(N, 6)`` holding currents of one PoE model (``axes`` and ``moments`` ``(6, 3)``)."""
    rot, lever = _chain(theta, axes[None])
    moment_b = np.einsum("nkab,kb->nka", rot[0], moments)
    distal_sum = np.cumsum(moment_b[:, ::-1], axis=1)[:, ::-1]
    return np.asarray(np.einsum("nja,nja->nj", lever[0], distal_sum), dtype=np.float64)


def _trig_gradient_features(theta: FloatArray) -> FloatArray:
    """``(N, 6, 73)``: the gradient of every term of the trigonometric potential.

    Term order: the constant; ``cos t_j, sin t_j`` per joint; then per joint
    pair ``j < k``: ``c_j c_k, c_j s_k, s_j c_k, s_j s_k``.
    """
    cos, sin = np.cos(theta), np.sin(theta)
    grad = np.zeros((theta.shape[0], _N_JOINTS, _N_TRIG))
    joints = np.arange(_N_JOINTS)
    grad[:, joints, 1 + 2 * joints] = -sin
    grad[:, joints, 2 + 2 * joints] = cos
    j, k = _TRIG_PAIRS[:, :1], _TRIG_PAIRS[:, 1:]
    cols = 1 + 2 * _N_JOINTS + 4 * np.arange(len(_TRIG_PAIRS))[:, None] + np.arange(4)
    cj, sj, ck, sk = cos[:, j], sin[:, j], cos[:, k], sin[:, k]
    grad[:, j, cols] = np.concatenate((-sj * ck, -sj * sk, cj * ck, cj * sk), axis=-1)
    grad[:, k, cols] = np.concatenate((-cj * sk, cj * ck, -sj * sk, sj * ck), axis=-1)
    return grad


# --- fitted arm ----------------------------------------------------------------


@dataclass(frozen=True, eq=False)
class ArmGravityFit:
    """The gravity model of one arm.

    Attributes:
        side: ``"right"`` or ``"left"``.
        joints: :func:`arm_joints` of ``side``.
        model_type: :data:`MODEL_POE` or :data:`MODEL_TRIG`.
        q_ref_counts: ``(6,)`` reference pose (``theta = 0``) in counts.
        axes: ``(6, 3)`` unit axes at the reference pose (PoE only, else ``None``).
        first_moments: ``(18,)`` link-major first moments in mA (PoE only, else ``None``).
        trig_coef: ``(73,)`` potential coefficients (trig only, else ``None``).
        peak_ma: ``joint -> max |prediction|`` over the dataset (gravity clamp).
        friction_ma: ``joint -> median static friction`` of the dataset.
        fit_report: JSON-ready diagnostics of :func:`fit_arm`.
        accepted: The chosen model met the acceptance limits.
    """

    side: str
    joints: Tuple[str, ...]
    model_type: str
    q_ref_counts: FloatArray
    axes: FloatArray | None
    first_moments: FloatArray | None
    trig_coef: FloatArray | None
    peak_ma: Dict[str, float]
    friction_ma: Dict[str, float]
    fit_report: Dict[str, Any] = field(default_factory=dict)
    accepted: bool = False

    def __post_init__(self) -> None:
        expected = arm_joints(self.side)
        if tuple(self.joints) != expected:
            raise ValueError(f"joints must be {list(expected)} for side {self.side!r}")
        object.__setattr__(self, "joints", expected)
        object.__setattr__(self, "q_ref_counts", _vector(self.q_ref_counts, "q_ref_counts"))
        wanted = _MODEL_ARRAYS.get(self.model_type)
        if wanted is None:
            raise ValueError(f"model_type must be {MODEL_POE!r} or {MODEL_TRIG!r}")
        for name, key in _ARRAY_FIELDS.items():
            value = getattr(self, name)
            if key not in wanted:
                if value is not None:
                    raise ValueError(f"{name} must be None for model_type {self.model_type!r}")
                continue
            if value is None:
                raise ValueError(f"{name} is required for model_type {self.model_type!r}")
            array = np.array(value, dtype=np.float64)
            if array.shape != wanted[key] or not bool(np.all(np.isfinite(array))):
                raise ValueError(f"{name} must be finite with shape {wanted[key]}")
            array.flags.writeable = False
            object.__setattr__(self, name, array)
        if self.axes is not None and not np.allclose(np.linalg.norm(self.axes, axis=1), 1.0):
            raise ValueError("axes must be unit vectors")
        for name in ("peak_ma", "friction_ma"):
            table = getattr(self, name)
            missing = [joint for joint in expected if joint not in table]
            if missing:
                raise ValueError(f"{name} has no entry for joint(s) {missing}")
            try:
                values = {joint: float(table[joint]) for joint in expected}
            except (TypeError, ValueError) as exc:
                raise ValueError(f"{name} must map every joint to a number") from exc
            if not all(math.isfinite(v) and v >= 0.0 for v in values.values()):
                raise ValueError(f"{name} must be finite and non-negative")
            object.__setattr__(self, name, values)
        object.__setattr__(self, "accepted", bool(self.accepted))

    def predict_arm_ma(self, q_arm_counts: ArrayLike) -> FloatArray:
        """Gravity currents of this arm.

        Args:
            q_arm_counts: ``(6,)`` or ``(N, 6)`` positions in :attr:`joints` order (counts).

        Returns:
            mA in the joint convention, same shape as the input.
        """
        q = np.asarray(q_arm_counts, dtype=np.float64)
        if q.shape[-1:] != (_N_JOINTS,) or q.ndim not in (1, 2):
            raise ValueError(f"q_arm_counts must have shape (6,) or (N, 6), got {q.shape}")
        theta = (np.atleast_2d(q) - self.q_ref_counts) * _RAD_PER_COUNT
        if self.model_type == MODEL_POE:
            assert self.axes is not None and self.first_moments is not None
            out = _poe_currents(theta, self.axes, self.first_moments.reshape(_N_JOINTS, 3))
        else:
            assert self.trig_coef is not None
            out = _trig_gradient_features(theta) @ self.trig_coef
        return out[0] if q.ndim == 1 else out


class LeaderGravityModel:
    """Both arms' fits behind the ``GravityModel`` protocol of the control laws.

    Attributes:
        arms: ``side -> fit`` of the identified arms.
        peak_ma: ``joint -> peak`` for all 17 joints; ``0`` where there is no model
            (so the gravity clamp keeps those joints at zero).
    """

    def __init__(self, arms: Mapping[str, ArmGravityFit]) -> None:
        self.arms: Dict[str, ArmGravityFit] = {}
        self._index: Dict[str, NDArray[np.intp]] = {}
        self.peak_ma: Dict[str, float] = {name: 0.0 for name in RAKUDA_JOINT_NAMES}
        for side, fit in arms.items():
            if fit.side != side:
                raise ValueError(f"arms[{side!r}] holds the fit of side {fit.side!r}")
            self.arms[side] = fit
            self._index[side] = np.array([RAKUDA_JOINT_NAMES.index(j) for j in fit.joints])
            self.peak_ma.update(fit.peak_ma)

    def predict_ma(self, q_counts: ArrayLike) -> FloatArray:
        """``(17,)`` positions in ``RAKUDA_JOINT_NAMES`` order -> ``(17,)`` mA.

        Joints of arms without a fit and joints outside the arms get 0.
        """
        n = len(RAKUDA_JOINT_NAMES)
        q = np.asarray(q_counts, dtype=np.float64)
        if q.shape != (n,):
            raise ValueError(f"q_counts must have shape ({n},), got {q.shape}")
        out = np.zeros(n, dtype=np.float64)
        for side, fit in self.arms.items():
            idx = self._index[side]
            out[idx] = fit.predict_arm_ma(q[idx])
        return out


# --- fitting -------------------------------------------------------------------


@functools.lru_cache(maxsize=1)
def _symmetry_images() -> NDArray[np.intp]:
    """``(8, 6)``: the axis code each gravity symmetry maps each axis code to.

    Rotating B by a multiple of 90 deg about z, or reflecting it through a
    vertical plane (an axial vector ``a`` maps to ``-M a``), leaves every
    prediction unchanged.
    """
    turn = np.array([[0.0, -1.0, 0.0], [1.0, 0.0, 0.0], [0.0, 0.0, 1.0]])
    mirror = np.diag([1.0, -1.0, 1.0])
    ops: List[FloatArray] = []
    for quarter in range(4):
        rotation = np.linalg.matrix_power(turn, quarter)
        ops.extend((rotation, -(rotation @ mirror)))
    return np.array([np.argmax(_SIGNED_UNIT @ (op @ _SIGNED_UNIT.T), axis=0) for op in ops])


def _canonical(codes: NDArray[np.intp]) -> NDArray[np.intp]:
    """The representative of each axis set ``(..., 6)`` among its 8 gravity twins.

    The first horizontal axis becomes ``+y`` and the next non-``y`` axis positive.
    """
    images = _symmetry_images()[:, codes]
    key = (np.array(_CANONICAL_RANK)[images] * 6 ** np.arange(_N_JOINTS - 1, -1, -1)).sum(-1)
    pick = np.argmin(key, axis=0)
    return np.asarray(np.take_along_axis(images, pick[None, ..., None], axis=0)[0])


@functools.lru_cache(maxsize=1)
def _canonical_candidates() -> NDArray[np.intp]:
    """``(768, 6)`` signed axis codes: one per gravity-equivalence class.

    The candidates are every axis-aligned set whose consecutive axes are
    orthogonal (``3 * 2**5`` patterns x ``2**6`` signs = 6144), 8 of which
    always predict the same currents.
    """
    letters = np.array(
        [p for p in product(range(3), repeat=_N_JOINTS) if all(a != b for a, b in zip(p, p[1:]))]
    )
    signs = np.array(list(product((0, 3), repeat=_N_JOINTS)))
    codes = (letters[:, None, :] + signs[None, :, :]).reshape(-1, _N_JOINTS)
    return np.unique(_canonical(codes), axis=0)


def _code_axes(codes: NDArray[np.intp]) -> FloatArray:
    """Signed axis codes ``(..., 6)`` -> unit axes ``(..., 6, 3)``."""
    return np.asarray(_SIGNED_UNIT[codes], dtype=np.float64)


def _labels(codes: NDArray[np.intp]) -> List[str]:
    return [_AXIS_LABELS[int(c)] for c in codes]


def _pattern(codes: NDArray[np.intp]) -> str:
    return "-".join(_AXIS_LETTERS[int(c) % 3] for c in codes)


def _screen_rms(theta: FloatArray, target: FloatArray, axes: FloatArray) -> FloatArray:
    """Training RMS of the linear (min-norm) fit of ``L`` for each axis set ``(C, 6, 3)``."""
    out = np.empty(axes.shape[0])
    for start in range(0, axes.shape[0], _SCREEN_CHUNK):
        chunk = axes[start : start + _SCREEN_CHUNK]
        a = _poe_regressor(theta, chunk).reshape(chunk.shape[0], -1, _N_MOMENTS)
        u, s, _ = np.linalg.svd(a, full_matrices=False)
        keep = s > _RCOND * s[:, :1]
        coord = np.einsum("cmr,m->cr", u, target) * keep
        fitted = np.einsum("cmr,cr->cm", u, coord)
        out[start : start + chunk.shape[0]] = np.sqrt(np.mean((target - fitted) ** 2, axis=1))
    return out


def _solve_moments(
    theta: FloatArray, target: FloatArray, axes: FloatArray
) -> Tuple[FloatArray, FloatArray]:
    """Min-norm ``L`` ``(18,)`` for fixed ``axes`` and the regressor's singular values."""
    a = _poe_regressor(theta, axes[None])[0].reshape(-1, _N_MOMENTS)
    moments, _, _, singular = np.linalg.lstsq(a, target, rcond=_RCOND)
    return np.asarray(moments, dtype=np.float64), np.asarray(singular, dtype=np.float64)


def _tilted_axes(start: FloatArray, angles: FloatArray) -> FloatArray:
    """Axes tilted from ``start`` ``(6, 3)`` by two angles each (``angles`` ``(6, 2)``).

    ``a = cos(u) cos(v) a0 + sin(u) cos(v) e2 + sin(v) e1`` with ``(a0, e1, e2)``
    orthonormal: smooth around the start and unit by construction.
    """
    helper = np.eye(3)[np.argmin(np.abs(start), axis=1)]
    e1 = np.cross(start, helper)
    e1 /= np.linalg.norm(e1, axis=1, keepdims=True)
    e2 = np.cross(start, e1)
    u, v = angles[:, :1], angles[:, 1:]
    return np.asarray(
        np.cos(u) * np.cos(v) * start + np.sin(u) * np.cos(v) * e2 + np.sin(v) * e1,
        dtype=np.float64,
    )


def _refine_axes(
    theta: FloatArray, target: FloatArray, start: FloatArray
) -> Tuple[FloatArray, NDArray[np.bool_]]:
    """Variable projection: 11 tilt angles by bounded least squares, ``L`` solved inside.

    A common rotation of every axis about the vertical is invisible to gravity,
    so it is removed from the parametrisation: the azimuth tilt of the first
    horizontal start axis is held at zero (that axis may only tilt up or down).
    The remaining 11 angles have a full-rank Jacobian, so each stays within
    :data:`_REFINE_BOUND_RAD` of the start in the sense documented there.

    Returns:
        The refined ``(6, 3)`` axes and, per joint, whether one of its tilt
        angles ended on the bound (the optimum lies beyond the start's reach).
    """
    # scipy is not a direct dependency; it is always installed through librosa.
    from scipy.optimize import least_squares

    # The first horizontal start exists: consecutive start axes are orthogonal.
    # For an axis-aligned horizontal start, angle ``u`` of _tilted_axes is its azimuth.
    gauge = int(np.flatnonzero(np.abs(start[:, 2]) < 0.5)[0])
    free = np.ones((_N_JOINTS, 2), dtype=bool)
    free[gauge, 0] = False

    def angles(x: FloatArray) -> FloatArray:
        full = np.zeros((_N_JOINTS, 2))
        full[free] = x
        return full

    def residual(x: FloatArray) -> FloatArray:
        axes = _tilted_axes(start, angles(x))
        a = _poe_regressor(theta, axes[None])[0].reshape(-1, _N_MOMENTS)
        moments = np.linalg.lstsq(a, target, rcond=_RCOND)[0]
        return np.asarray(a @ moments - target, dtype=np.float64)

    result = least_squares(
        residual,
        np.zeros(int(free.sum())),
        bounds=(-_REFINE_BOUND_RAD, _REFINE_BOUND_RAD),
        method="trf",
    )
    on_bound = np.zeros((_N_JOINTS, 2), dtype=bool)
    on_bound[free] = np.asarray(result.active_mask) != 0
    return _tilted_axes(start, angles(np.asarray(result.x))), np.any(on_bound, axis=1)


def _rms(error: FloatArray) -> float:
    return float(np.sqrt(np.mean(error**2))) if error.size else float("nan")


def _finite_or_none(value: float) -> float | None:
    return float(value) if math.isfinite(value) else None


@dataclass(frozen=True)
class _Model:
    """One fitted candidate model of :func:`fit_arm` (training data only)."""

    model_type: str
    predict: Callable[[FloatArray], FloatArray]
    axes: FloatArray | None
    first_moments: FloatArray | None
    trig_coef: FloatArray | None
    report: Dict[str, Any]


@dataclass(frozen=True)
class _Refined:
    """One refined PoE candidate (moments fitted on the training poses)."""

    metric: float
    at_bound: bool
    structure: Tuple[int, ...]
    axes: FloatArray
    moments: FloatArray
    singular: FloatArray
    summary: Dict[str, Any]


def _refine_candidate(
    theta: FloatArray,
    y: FloatArray,
    theta_val: FloatArray,
    y_val: FloatArray,
    start_codes: NDArray[np.intp],
    screen_rms: float,
) -> _Refined:
    """Refines one screened start; ``metric`` is the validation (else training) RMS.

    The refined axes are labelled by their nearest axis-aligned set (the
    structure they ended at, which may differ from the start) and compared
    across candidates through its canonical form.  ``at_bound`` marks the
    joints whose tilt ended on the refinement bound: their label is not a
    clean match and the candidate is only a boundary optimum.
    """
    target = y.reshape(-1)
    axes, on_bound = _refine_axes(theta, target, _code_axes(start_codes))
    moments, singular = _solve_moments(theta, target, axes)
    l6 = moments.reshape(_N_JOINTS, 3)
    train_rms = _rms(_poe_currents(theta, axes, l6) - y)
    has_val = theta_val.shape[0] > 0
    val_rms = _rms(_poe_currents(theta_val, axes, l6) - y_val) if has_val else math.nan
    nearest = np.argmax(axes @ _SIGNED_UNIT.T, axis=1)
    cosine = np.clip(np.sum(axes * _SIGNED_UNIT[nearest], axis=1), -1.0, 1.0)
    summary = {
        "pattern": _pattern(nearest),
        "ideal_axes": _labels(nearest),
        "axis_tilt_deg": [round(float(t), 3) for t in np.degrees(np.arccos(cosine))],
        "at_bound": [bool(b) for b in on_bound],
        "start_axes": _labels(start_codes),
        "screen_rms_ma": screen_rms,
        "train_rms_ma": train_rms,
        "val_rms_ma": _finite_or_none(val_rms),
    }
    return _Refined(
        metric=val_rms if has_val else train_rms,
        at_bound=bool(np.any(on_bound)),
        structure=tuple(int(c) for c in _canonical(nearest)),
        axes=axes,
        moments=moments,
        singular=singular,
        summary=summary,
    )


def _fit_poe(
    theta: FloatArray,
    y: FloatArray,
    theta_val: FloatArray,
    y_val: FloatArray,
    refine_top_k: int,
) -> _Model:
    """Axis screening over the canonical candidates, refinement, selection.

    Every one of the ``refine_top_k`` best screened candidates is refined, sign
    variants of one pattern included.  Candidates that ended on the tilt bound
    rank after every interior one, then by ``metric``.
    """
    codes = _canonical_candidates()
    screen = _screen_rms(theta, y.reshape(-1), _code_axes(codes))
    starts = np.argsort(screen, kind="stable")[:refine_top_k]
    refined = sorted(
        (
            _refine_candidate(theta, y, theta_val, y_val, codes[idx], float(screen[idx]))
            for idx in starts
        ),
        key=lambda candidate: (candidate.at_bound, candidate.metric),
    )
    best = refined[0]
    if best.at_bound:
        logger.warning(
            "every refined PoE candidate ended on the tilt bound; the chosen axes %s are "
            "a boundary optimum",
            best.summary["ideal_axes"],
        )
    # Starts that converged to the best structure are not competitors.  A margin
    # below 1 means a boundary optimum fitted better than the interior winner.
    rivals = sorted(c.metric for c in refined[1:] if c.structure != best.structure)
    margin = rivals[0] / best.metric if rivals and best.metric > 0.0 else None
    kept = best.singular[best.singular > _RCOND * best.singular[0]]
    report = {
        **best.summary,
        "axes": best.axes.tolist(),
        "rank": int(kept.size),
        "cond": float(kept[0] / kept[-1]),
        "pattern_margin": margin,
        "n_screened": int(codes.shape[0]),
        "candidates": [c.summary for c in refined],
    }
    axes, l6 = best.axes, best.moments.reshape(_N_JOINTS, 3)
    return _Model(
        MODEL_POE, lambda th: _poe_currents(th, axes, l6), axes, best.moments, None, report
    )


def _fit_trig(
    theta: FloatArray, y: FloatArray, theta_val: FloatArray, y_val: FloatArray, ridge: float
) -> _Model:
    """Ridge fit of the trigonometric potential's gradient to all joints at once."""
    grad = _trig_gradient_features(theta).reshape(-1, _N_TRIG)
    a = np.vstack((grad, math.sqrt(ridge) * np.eye(_N_TRIG)))
    b = np.concatenate((y.reshape(-1), np.zeros(_N_TRIG)))
    coef = np.asarray(np.linalg.lstsq(a, b, rcond=None)[0], dtype=np.float64)

    def predict(th: FloatArray) -> FloatArray:
        return np.asarray(_trig_gradient_features(th) @ coef, dtype=np.float64)

    val_rms = _rms(predict(theta_val) - y_val) if theta_val.shape[0] else math.nan
    report = {
        "ridge": float(ridge),
        "train_rms_ma": _rms(predict(theta) - y),
        "val_rms_ma": _finite_or_none(val_rms),
    }
    return _Model(MODEL_TRIG, predict, None, None, coef, report)


def fit_arm(
    samples: Sequence[PoseSample],
    side: str,
    q_ref_counts: Mapping[str, int],
    *,
    ridge: float = 1e-2,
    refine_top_k: int = _REFINE_TOP_K,
) -> ArmGravityFit:
    """Fits both models to one arm's dataset and keeps the better one.

    Both models are fitted on the training poses (``is_val`` false).  The PoE
    axes are screened over every canonical axis-aligned candidate, the
    ``refine_top_k`` best of them are refined (11 bounded tilt angles with the
    yaw gauge fixed, moments solved linearly inside) and the refined candidate
    with the lowest validation RMS wins, one that ended inside the tilt bound
    before one that ended on it.  Each model is then judged
    per joint on the validation poses (training poses when there are none):
    RMS <= ``max(5 mA, 10 % of the joint's I_g range)`` and max |error| <=
    ``max(10 mA, 0.8 * median friction)``.  An accepted model is preferred,
    then the lower validation RMS; ``accepted`` is false only when both fail.

    Args:
        samples: The identified poses (at least :data:`MIN_FIT_SAMPLES`).
        side: ``"right"`` or ``"left"``.
        q_ref_counts: ``joint -> reference counts``; must contain the arm's joints.
        ridge: Ridge weight of the trigonometric fallback.
        refine_top_k: Number of best screened axis candidates refined nonlinearly.

    Returns:
        The chosen model with ``peak_ma`` (max |prediction| over every sample),
        ``friction_ma`` (median ``f_s``) and a JSON-ready ``fit_report``.

    Raises:
        ValueError: Too few samples, no training sample, a missing reference
            joint or an invalid parameter.
    """
    joints = arm_joints(side)
    if len(samples) < MIN_FIT_SAMPLES:
        raise ValueError(f"{len(samples)} poses; at least {MIN_FIT_SAMPLES} are needed to fit")
    if refine_top_k < 1:
        raise ValueError(f"refine_top_k must be >= 1, got {refine_top_k}")
    if not (ridge >= 0.0 and math.isfinite(ridge)):
        raise ValueError(f"ridge must be finite and >= 0, got {ridge}")
    missing = [name for name in joints if name not in q_ref_counts]
    if missing:
        raise ValueError(f"q_ref_counts has no entry for joint(s) {missing}")
    q_ref = np.array([float(q_ref_counts[name]) for name in joints])

    theta = (np.stack([s.q_counts for s in samples]) - q_ref) * _RAD_PER_COUNT
    y = np.stack([s.i_gravity_ma for s in samples])
    friction = np.stack([s.friction_ma for s in samples])
    is_val = np.array([s.is_val for s in samples], dtype=bool)
    train = ~is_val
    if not bool(np.any(train)):
        raise ValueError("every pose is a validation pose; nothing to fit")
    evaluate = is_val if bool(np.any(is_val)) else train

    models = (
        _fit_poe(theta[train], y[train], theta[is_val], y[is_val], refine_top_k),
        _fit_trig(theta[train], y[train], theta[is_val], y[is_val], ridge),
    )

    span = np.ptp(y, axis=0)
    friction_median = np.median(friction, axis=0)
    rms_limit = np.maximum(_ACCEPT_RMS_FLOOR_MA, _ACCEPT_RMS_RANGE_FRACTION * span)
    max_limit = np.maximum(_ACCEPT_MAX_FLOOR_MA, _ACCEPT_MAX_FRICTION_FACTOR * friction_median)
    judged = []
    for model in models:
        error = model.predict(theta[evaluate]) - y[evaluate]
        rms = np.sqrt(np.mean(error**2, axis=0))
        worst = np.max(np.abs(error), axis=0)
        ok = (rms <= rms_limit) & (worst <= max_limit)
        model.report.update(
            eval_rms_ma=_rms(error),
            accepted=bool(np.all(ok)),
            per_joint={
                name: {
                    "rms_ma": float(rms[j]),
                    "max_abs_ma": float(worst[j]),
                    "ok": bool(ok[j]),
                }
                for j, name in enumerate(joints)
            },
        )
        judged.append((not bool(np.all(ok)), _rms(error), model))
    judged.sort(key=lambda item: (item[0], item[1]))
    rejected, _, best = judged[0]

    prediction = best.predict(theta)
    peak = np.max(np.abs(prediction), axis=0)
    flag_counts: Dict[str, int] = {}
    for sample in samples:
        for flag in sample.flags:
            flag_counts[flag] = flag_counts.get(flag, 0) + 1
    report = {
        "chosen": best.model_type,
        "accepted": not rejected,
        "n_samples": len(samples),
        "n_train": int(np.count_nonzero(train)),
        "n_val": int(np.count_nonzero(is_val)),
        "evaluated_on": "val" if bool(np.any(is_val)) else "train",
        "limits": {
            name: {
                "range_ma": float(span[j]),
                "friction_median_ma": float(friction_median[j]),
                "rms_limit_ma": float(rms_limit[j]),
                "max_limit_ma": float(max_limit[j]),
            }
            for j, name in enumerate(joints)
        },
        "models": {model.model_type: model.report for model in models},
        "flags": flag_counts,
    }
    logger.info(
        "%s arm: %s chosen (%s), eval RMS %.1f mA (poe %.1f, trig %.1f)",
        side,
        best.model_type,
        "accepted" if not rejected else "NOT accepted",
        best.report["eval_rms_ma"],
        models[0].report["eval_rms_ma"],
        models[1].report["eval_rms_ma"],
    )
    return ArmGravityFit(
        side=side,
        joints=joints,
        model_type=best.model_type,
        q_ref_counts=q_ref,
        axes=best.axes,
        first_moments=best.first_moments,
        trig_coef=best.trig_coef,
        peak_ma={name: float(peak[j]) for j, name in enumerate(joints)},
        friction_ma={name: float(friction_median[j]) for j, name in enumerate(joints)},
        fit_report=report,
        accepted=not rejected,
    )


# --- storage -------------------------------------------------------------------


def _json_default(value: Any) -> Any:
    """``json.dumps`` hook for numpy scalars and arrays."""
    if isinstance(value, np.generic):
        return value.item()
    if isinstance(value, np.ndarray):
        return value.tolist()
    raise TypeError(f"{type(value).__name__} is not JSON serialisable")


def _atomic_write(path: str | os.PathLike[str], payload: bytes) -> None:
    """Writes ``payload`` through a temporary file and ``os.replace``; keeps ``<path>.bak``."""
    target = os.fspath(path)
    directory = os.path.dirname(os.path.abspath(target))
    os.makedirs(directory, exist_ok=True)
    fd, tmp = tempfile.mkstemp(prefix=os.path.basename(target) + ".", suffix=".tmp", dir=directory)
    try:
        with os.fdopen(fd, "wb") as handle:
            handle.write(payload)
            handle.flush()
            os.fsync(handle.fileno())
        if os.path.exists(target):
            shutil.copy2(target, target + ".bak")
        os.replace(tmp, target)
    except BaseException:
        with contextlib.suppress(OSError):
            os.unlink(tmp)
        raise


def save_dataset(
    path: str | os.PathLike[str],
    side: str,
    samples: Sequence[PoseSample],
    header: Mapping[str, Any],
) -> str:
    """Writes one arm's poses as ``.npz`` (atomic, previous file kept as ``.bak``).

    Arrays: ``q_counts``, ``i_plus_ma``, ``i_minus_ma``, ``temperature_c``
    ``(N, 6)``, ``is_val`` ``(N,)``; ``flags_json`` and ``header_json`` are JSON
    strings (no pickled objects).  ``side`` and ``joints`` are added to the header.

    Returns:
        The sha256 hex digest of the written file.
    """
    joints = arm_joints(side)

    def stack(name: str) -> FloatArray:
        return np.array([getattr(s, name) for s in samples], dtype=np.float64).reshape(-1, 6)

    full_header = {**dict(header), "side": side, "joints": list(joints)}
    buffer = io.BytesIO()
    np.savez(
        buffer,
        q_counts=stack("q_counts"),
        i_plus_ma=stack("i_plus_ma"),
        i_minus_ma=stack("i_minus_ma"),
        temperature_c=stack("temperature_c"),
        is_val=np.array([s.is_val for s in samples], dtype=bool),
        flags_json=np.array(json.dumps([list(s.flags) for s in samples])),
        header_json=np.array(json.dumps(full_header, default=_json_default)),
    )
    payload = buffer.getvalue()
    _atomic_write(path, payload)
    return hashlib.sha256(payload).hexdigest()


def load_dataset(path: str | os.PathLike[str]) -> Tuple[List[PoseSample], Dict[str, Any]]:
    """Reads a dataset written by :func:`save_dataset`.

    Returns:
        The poses and the header.

    Raises:
        FileNotFoundError: No such file.
        ValueError: A missing array or inconsistent shapes (naming the key).
    """
    with np.load(os.fspath(path), allow_pickle=False) as data:
        arrays = {}
        for key in ("q_counts", "i_plus_ma", "i_minus_ma", "temperature_c", "is_val"):
            if key not in data:
                raise ValueError(f"{path}: dataset has no {key!r}")
            arrays[key] = np.asarray(data[key])
        texts = {}
        for key in ("flags_json", "header_json"):
            if key not in data:
                raise ValueError(f"{path}: dataset has no {key!r}")
            texts[key] = json.loads(str(data[key][()]))
    n = arrays["is_val"].shape[0]
    for key in ("q_counts", "i_plus_ma", "i_minus_ma", "temperature_c"):
        if arrays[key].shape != (n, _N_JOINTS):
            raise ValueError(f"{path}: {key} must have shape ({n}, 6), got {arrays[key].shape}")
    flags = texts["flags_json"]
    if not isinstance(flags, list) or len(flags) != n:
        raise ValueError(f"{path}: flags_json must list {n} entries")
    header = texts["header_json"]
    if not isinstance(header, dict):
        raise ValueError(f"{path}: header_json must be a JSON object")
    samples = [
        PoseSample(
            q_counts=arrays["q_counts"][i],
            i_plus_ma=arrays["i_plus_ma"][i],
            i_minus_ma=arrays["i_minus_ma"][i],
            temperature_c=arrays["temperature_c"][i],
            flags=tuple(flags[i]),
            is_val=bool(arrays["is_val"][i]),
        )
        for i in range(n)
    ]
    return samples, header


def arm_fit_to_dict(fit: ArmGravityFit) -> Dict[str, Any]:
    """The JSON form of one arm's fit (``arms.<side>`` of the gravity file)."""

    def as_list(array: FloatArray | None) -> Any:
        return None if array is None else array.tolist()

    return {
        "side": fit.side,
        "joints": list(fit.joints),
        "model_type": fit.model_type,
        "q_ref": fit.q_ref_counts.tolist(),
        "axes": as_list(fit.axes),
        "L": as_list(fit.first_moments),
        "trig_coef": as_list(fit.trig_coef),
        "peak_ma": dict(fit.peak_ma),
        "friction_ma": dict(fit.friction_ma),
        "fit_report": json.loads(json.dumps(fit.fit_report, default=_json_default)),
        "accepted": fit.accepted,
    }


def arm_fit_from_dict(d: Mapping[str, Any]) -> ArmGravityFit:
    """Inverse of :func:`arm_fit_to_dict`; extra keys (``validated`` ...) are ignored.

    Raises:
        ValueError: Naming the missing or malformed key.
    """
    if not isinstance(d, Mapping):
        raise ValueError(f"expected an object, got {type(d).__name__}")

    def need(key: str, kind: type | Tuple[type, ...], expected: str) -> Any:
        if key not in d:
            raise ValueError(f"{key}: missing")
        if not isinstance(d[key], kind):
            raise ValueError(f"{key}: expected {expected}, got {d[key]!r}")
        return d[key]

    def array(key: str, shape: Tuple[int, ...]) -> FloatArray:
        try:
            value = np.array(d[key], dtype=np.float64)
        except (TypeError, ValueError) as exc:
            raise ValueError(f"{key}: expected numbers") from exc
        if value.shape != shape or not bool(np.all(np.isfinite(value))):
            raise ValueError(f"{key}: expected finite numbers of shape {shape}")
        return value

    model_type = need("model_type", str, "a string")
    shapes = _MODEL_ARRAYS.get(model_type)
    if shapes is None:
        raise ValueError(f"model_type: expected {MODEL_POE!r} or {MODEL_TRIG!r}")
    need("q_ref", list, "a list")
    arrays: Dict[str, FloatArray | None] = {}
    for key in ("axes", "L", "trig_coef"):
        if key in shapes:
            need(key, list, "a list")
            arrays[key] = array(key, shapes[key])
        elif d.get(key) is not None:
            raise ValueError(f"{key}: must be null for model_type {model_type!r}")
        else:
            arrays[key] = None
    report = d.get("fit_report", {})
    if not isinstance(report, Mapping):
        raise ValueError("fit_report: expected an object")
    return ArmGravityFit(
        side=need("side", str, "a string"),
        joints=tuple(need("joints", list, "a list")),
        model_type=model_type,
        q_ref_counts=array("q_ref", (_N_JOINTS,)),
        axes=arrays["axes"],
        first_moments=arrays["L"],
        trig_coef=arrays["trig_coef"],
        peak_ma=dict(need("peak_ma", Mapping, "an object")),
        friction_ma=dict(need("friction_ma", Mapping, "an object")),
        fit_report=dict(report),
        accepted=need("accepted", bool, "true or false"),
    )


@dataclass
class GravityFile:
    """``.robopy/rakuda/leader_gravity.json`` (``schema_version`` 1).

    ``range`` writes ``reference_counts`` / ``joint_range_counts``,
    ``sign-check`` adds ``current_sign`` / ``drive_mode`` / ``sign_check`` and
    ``identify`` / ``fit`` / ``verify`` fill ``arms``.  Every per-joint table is
    keyed by joint name.  An ``arms.<side>`` entry is :func:`arm_fit_to_dict`
    plus ``dataset_sha256``, ``validated`` (bool), ``validated_at`` and
    ``temperature``; of those only ``validated`` is read here.
    """

    schema_version: int = SCHEMA_VERSION
    motors: List[Dict[str, Any]] = field(default_factory=list)
    reference_counts: Dict[str, int] = field(default_factory=dict)
    joint_range_counts: Dict[str, Tuple[int, int]] = field(default_factory=dict)
    current_sign: Dict[str, int] = field(default_factory=dict)
    drive_mode: Dict[str, int] = field(default_factory=dict)
    sign_check: Dict[str, Any] = field(default_factory=dict)
    arms: Dict[str, Dict[str, Any]] = field(default_factory=dict)


_FILE_KEYS = (
    "schema_version",
    "motors",
    "reference_counts",
    "joint_range_counts",
    "current_sign",
    "drive_mode",
    "sign_check",
    "arms",
)


def _as_count(value: Any) -> int | None:
    if isinstance(value, bool) or not isinstance(value, numbers.Integral):
        return None
    return int(value)


def _as_range(value: Any) -> Tuple[int, int] | None:
    if not isinstance(value, (list, tuple)) or len(value) != 2:
        return None
    lo, hi = _as_count(value[0]), _as_count(value[1])
    if lo is None or hi is None or lo >= hi:
        return None
    return lo, hi


def _as_sign(value: Any) -> int | None:
    count = _as_count(value)
    return count if count in (1, -1) else None


def _as_drive_mode(value: Any) -> int | None:
    count = _as_count(value)
    return count if count is not None and count >= 0 else None


def _joint_table(
    raw: Mapping[str, Any], key: str, convert: Callable[[Any], Any], expected: str, where: str
) -> Dict[str, Any]:
    table = raw.get(key, {})
    if not isinstance(table, Mapping):
        raise ValueError(f"{where}: {key}: expected an object keyed by joint name")
    out = {}
    for name, value in table.items():
        if name not in RAKUDA_JOINT_NAMES:
            raise ValueError(f"{where}: {key}.{name}: unknown joint")
        converted = convert(value)
        if converted is None:
            raise ValueError(f"{where}: {key}.{name}: expected {expected}, got {value!r}")
        out[name] = converted
    return out


def _parse_gravity_file(raw: Any, where: str) -> GravityFile:
    """Validates the decoded JSON and builds a :class:`GravityFile`."""
    if not isinstance(raw, Mapping):
        raise ValueError(f"{where}: expected a JSON object")
    version = raw.get("schema_version")
    if isinstance(version, bool) or version != SCHEMA_VERSION:
        raise ValueError(f"{where}: unknown schema_version {version!r} (expected {SCHEMA_VERSION})")
    unknown = sorted(set(raw) - set(_FILE_KEYS))
    if unknown:
        logger.warning("%s: ignoring unknown key(s) %s", where, unknown)
    motors = raw.get("motors", [])
    if not isinstance(motors, list) or not all(isinstance(m, Mapping) for m in motors):
        raise ValueError(f"{where}: motors: expected a list of objects")
    sign_check = raw.get("sign_check", {})
    if not isinstance(sign_check, Mapping):
        raise ValueError(f"{where}: sign_check: expected an object")
    arms_raw = raw.get("arms", {})
    if not isinstance(arms_raw, Mapping):
        raise ValueError(f"{where}: arms: expected an object keyed by side")
    arms: Dict[str, Dict[str, Any]] = {}
    for side, entry in arms_raw.items():
        if side not in ARM_SIDES:
            raise ValueError(f"{where}: arms.{side}: unknown side (expected one of {ARM_SIDES})")
        try:
            fit = arm_fit_from_dict(entry)
        except ValueError as exc:
            raise ValueError(f"{where}: arms.{side}.{exc}") from exc
        if fit.side != side:
            raise ValueError(f"{where}: arms.{side}.side: {fit.side!r} does not match")
        if not isinstance(entry.get("validated", False), bool):
            raise ValueError(f"{where}: arms.{side}.validated: expected true or false")
        arms[side] = dict(entry)
    return GravityFile(
        schema_version=SCHEMA_VERSION,
        motors=[dict(m) for m in motors],
        reference_counts=_joint_table(raw, "reference_counts", _as_count, "an integer", where),
        joint_range_counts=_joint_table(
            raw, "joint_range_counts", _as_range, "[lo, hi] integers with lo < hi", where
        ),
        current_sign=_joint_table(raw, "current_sign", _as_sign, "+1 or -1", where),
        drive_mode=_joint_table(raw, "drive_mode", _as_drive_mode, "an integer >= 0", where),
        sign_check=dict(sign_check),
        arms=arms,
    )


def load_gravity_file(path: str | os.PathLike[str]) -> GravityFile:
    """Reads and validates the gravity file.

    Raises:
        FileNotFoundError: No such file.
        ValueError: Invalid JSON, an unknown ``schema_version`` or a malformed
            entry (the message names the key).
    """
    with open(os.fspath(path), encoding="utf-8") as handle:
        text = handle.read()
    try:
        raw = json.loads(text)
    except json.JSONDecodeError as exc:
        raise ValueError(f"{path}: invalid JSON: {exc}") from exc
    return _parse_gravity_file(raw, str(path))


def save_gravity_file(path: str | os.PathLike[str], data: GravityFile) -> None:
    """Validates ``data`` and writes it atomically; the previous file is kept as ``.bak``.

    Raises:
        ValueError: ``data`` would not load back (the message names the key).
    """
    payload = {
        "schema_version": data.schema_version,
        "motors": data.motors,
        "reference_counts": data.reference_counts,
        "joint_range_counts": {name: list(span) for name, span in data.joint_range_counts.items()},
        "current_sign": data.current_sign,
        "drive_mode": data.drive_mode,
        "sign_check": data.sign_check,
        "arms": data.arms,
    }
    text = json.dumps(payload, indent=2, allow_nan=False, default=_json_default)
    _parse_gravity_file(json.loads(text), str(path))
    _atomic_write(path, (text + "\n").encode("utf-8"))


def load_bilateral_setup(path: str | os.PathLike[str]) -> BilateralSetup:
    """What ``RakudaPairSys.start_bilateral()`` needs from the gravity file.

    ``gravity`` holds every identified arm (``None`` when there is none) and
    ``gravity_joints`` names the joints those arms cover, so a loop that also
    drives an unidentified arm is refused; ``gravity_validated`` needs both
    arms identified, accepted and validated (a fit that is not accepted counts
    as unvalidated).

    Raises:
        FileNotFoundError: No such file.
        ValueError: See :func:`load_gravity_file`.
    """
    # Imported here: rakuda_pair_sys imports this module lazily, and loading it
    # pulls in the whole robot stack, which the model itself does not need.
    from .rakuda_pair_sys import BilateralSetup

    data = load_gravity_file(path)
    fits = {side: arm_fit_from_dict(entry) for side, entry in data.arms.items()}
    model = LeaderGravityModel(fits) if fits else None
    validated = set(fits) == set(ARM_SIDES) and all(
        fit.accepted and data.arms[side].get("validated", False) is True
        for side, fit in fits.items()
    )
    return BilateralSetup(
        current_sign=dict(data.current_sign),
        joint_range_counts=dict(data.joint_range_counts),
        drive_mode=dict(data.drive_mode),
        gravity=model,
        gravity_validated=validated,
        gravity_peak_ma=None if model is None else dict(model.peak_ma),
        source=str(path),
        gravity_joints=frozenset(j for fit in fits.values() for j in fit.joints),
    )


if __name__ == "__main__":
    # `python -m robopy.robots.rakuda.rakuda_gravity <cmd>` runs the CLI.
    # Imported only here: rakuda_gravity_cli imports this module.
    import sys

    from .rakuda_gravity_cli import main

    sys.exit(main())
