"""Leader gravity model: PoE math, fitting, storage and ``load_bilateral_setup``.

Hardware-free.  The reference potential and the rotations of the synthetic
arms are computed independently of the module (``scipy`` ``expm`` /
``Rotation``) so the analytic regressor is checked against a different code path.
"""

from __future__ import annotations

import dataclasses
import hashlib
import json
import os
import subprocess
import sys
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Dict, List, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray
from scipy.linalg import expm
from scipy.spatial.transform import Rotation

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaArmState,
    RakudaBilateralParams,
)
from robopy.robots.rakuda import rakuda_gravity as rg
from robopy.robots.rakuda import rakuda_pair_sys
from robopy.robots.rakuda.rakuda_control_laws import BilateralLaw, GravityModel

FloatArray = NDArray[np.float64]

RAD_PER_COUNT = 2.0 * np.pi / 4096.0
AXIS = {
    "+x": (1.0, 0.0, 0.0),
    "+y": (0.0, 1.0, 0.0),
    "+z": (0.0, 0.0, 1.0),
    "-x": (-1.0, 0.0, 0.0),
    "-y": (0.0, -1.0, 0.0),
    "-z": (0.0, 0.0, -1.0),
}
RIGHT = rg.arm_joints("right")
LEFT = rg.arm_joints("left")


# --- independent reference implementation ----------------------------------------


def skew(a: FloatArray) -> FloatArray:
    return np.array([[0.0, -a[2], a[1]], [a[2], 0.0, -a[0]], [-a[1], a[0], 0.0]])


def poe_potential(theta: FloatArray, axes: FloatArray, moments: FloatArray) -> float:
    """``V = -g . sum_k R_k L_k`` with ``g = (0, 0, -1)``, rotations from ``expm``."""
    rot = np.eye(3)
    value = 0.0
    for j in range(6):
        rot = rot @ expm(skew(axes[j]) * theta[j])
        value += float((rot @ moments[j])[2])
    return value


def central_gradient(f: Callable[[FloatArray], float], theta: FloatArray) -> FloatArray:
    h = 1e-6
    step = np.eye(theta.size) * h
    return np.array([(f(theta + step[j]) - f(theta - step[j])) / (2 * h) for j in range(6)])


def axes_of(labels: Sequence[str]) -> FloatArray:
    return np.array([AXIS[label] for label in labels])


def yaw_gauge(axes: FloatArray, start: FloatArray) -> FloatArray:
    """Rotates ``axes`` about z so the first horizontal start keeps its azimuth (as the fit)."""
    j = next(i for i in range(6) if abs(start[i, 2]) < 0.5)
    angle = np.arctan2(start[j, 1], start[j, 0]) - np.arctan2(axes[j, 1], axes[j, 0])
    return np.asarray(Rotation.from_rotvec([0.0, 0.0, angle]).apply(axes))


def mirror_y(vectors: FloatArray, *, axial: bool) -> FloatArray:
    """Reflection through the x-z plane; an axial vector also flips (``a -> -M a``)."""
    reflected = vectors * np.array([1.0, -1.0, 1.0])
    return -reflected if axial else reflected


# --- synthetic data ------------------------------------------------------------


@dataclass
class Case:
    """A synthetic dataset and, for PoE data, the model that generated it."""

    samples: List[rg.PoseSample]
    q_ref: FloatArray
    q: FloatArray
    i_gravity: FloatArray
    axes: FloatArray | None = None


def make_samples(
    q: FloatArray, i_gravity: FloatArray, rng: np.random.Generator
) -> List[rg.PoseSample]:
    """3 % multiplicative noise on each approach, 15-25 mA friction, every 5th pose validates."""
    n = q.shape[0]
    friction = rng.uniform(15.0, 25.0, (n, 6))
    i_plus = (i_gravity + friction) * (1.0 + 0.03 * rng.normal(size=(n, 6)))
    i_minus = (i_gravity - friction) * (1.0 + 0.03 * rng.normal(size=(n, 6)))
    return [
        rg.PoseSample(q[i], i_plus[i], i_minus[i], np.full(6, 40.0), is_val=i % 5 == 4)
        for i in range(n)
    ]


def poe_case(start_labels: Sequence[str], seed: int, n: int = 60) -> Case:
    """Axes = ``start_labels`` tilted by 10-20 deg; every joint peaks at 50-200 mA."""
    rng = np.random.default_rng(seed)
    axes = []
    for a in axes_of(start_labels):
        tilt_axis = np.cross(a, rng.normal(size=3))
        tilt_axis /= np.linalg.norm(tilt_axis)
        axes.append(Rotation.from_rotvec(tilt_axis * np.radians(rng.uniform(10, 20))).apply(a))
    true_axes = np.array(axes)
    q_ref = rng.integers(1500, 2600, 6).astype(np.float64)
    q = q_ref + rng.uniform(-1024.0, 1024.0, (n, 6))  # +-90 deg around the reference
    theta = (q - q_ref) * RAD_PER_COUNT
    moments = rng.normal(size=(6, 3))

    def currents(m: FloatArray) -> FloatArray:
        return np.array(
            [central_gradient(lambda t: poe_potential(t, true_axes, m), th) for th in theta]
        )

    # Give the wrist a moment comparable to the shoulder, then scale to a 200 mA peak.
    moments[5] *= 150.0 / np.max(np.abs(currents(moments)[:, 5]))
    i_gravity = currents(moments)
    i_gravity *= 200.0 / np.max(np.abs(i_gravity))
    peaks = np.max(np.abs(i_gravity), axis=0)
    assert np.all((peaks >= 50.0) & (peaks <= 200.0 + 1e-9)), peaks
    return Case(make_samples(q, i_gravity, rng), q_ref, q, i_gravity, true_axes)


def trig_case(seed: int, n: int = 60) -> Case:
    """Gradient of a random smooth pairwise potential (not a serial chain)."""
    rng = np.random.default_rng(seed)
    single = rng.normal(size=(6, 2))
    pair = rng.normal(size=(6, 6)) * 0.7
    phase = rng.uniform(-np.pi, np.pi, (6, 6))

    def potential(t: FloatArray) -> float:
        value = float(np.sum(single[:, 0] * np.cos(t) + single[:, 1] * np.sin(t)))
        for j in range(6):
            for k in range(j + 1, 6):
                value += float(pair[j, k] * np.cos(t[j] - t[k] + phase[j, k]))
        return value

    q_ref = np.full(6, 2048.0)
    q = q_ref + rng.uniform(-1024.0, 1024.0, (n, 6))
    theta = (q - q_ref) * RAD_PER_COUNT
    i_gravity = np.array([central_gradient(potential, th) for th in theta])
    i_gravity *= 150.0 / np.max(np.abs(i_gravity))
    return Case(make_samples(q, i_gravity, rng), q_ref, q, i_gravity)


def reference(q_ref: FloatArray, joints: Tuple[str, ...] = RIGHT) -> Dict[str, int]:
    return {name: int(value) for name, value in zip(joints, q_ref)}


def fit_case(case: Case) -> rg.ArmGravityFit:
    # q_ref is integral here, so passing it as int counts is exact.
    return rg.fit_arm(case.samples, "right", reference(case.q_ref))


# --- hand-made fits --------------------------------------------------------------


def make_fit(
    side: str, model_type: str = rg.MODEL_POE, *, accepted: bool = True
) -> rg.ArmGravityFit:
    rng = np.random.default_rng(7 if side == "right" else 8)
    joints = rg.arm_joints(side)
    poe = model_type == rg.MODEL_POE
    return rg.ArmGravityFit(
        side=side,
        joints=joints,
        model_type=model_type,
        q_ref_counts=np.full(6, 2048.0),
        axes=axes_of(["+y", "+x", "+z", "+y", "+z", "+y"]) if poe else None,
        first_moments=rng.normal(size=18) * 40.0 if poe else None,
        trig_coef=None if poe else rng.normal(size=73) * 10.0,
        peak_ma={name: 100.0 + i for i, name in enumerate(joints)},
        friction_ma={name: 20.0 for name in joints},
        fit_report={"chosen": model_type, "pattern_margin": None},
        accepted=accepted,
    )


def arm_entry(fit: rg.ArmGravityFit, *, validated: bool) -> Dict[str, Any]:
    return {
        **rg.arm_fit_to_dict(fit),
        "dataset_sha256": "0" * 64,
        "validated": validated,
        "validated_at": "2026-09-25T10:00:00" if validated else None,
        "temperature": {name: 45.0 for name in fit.joints},
    }


def make_file(arms: Dict[str, Dict[str, Any]] | None = None) -> rg.GravityFile:
    return rg.GravityFile(
        motors=[
            {"name": name, "id": i + 1, "model": "xc330-t288", "unit_ma": 1.0, "drive_mode": 0}
            for i, name in enumerate(RAKUDA_JOINT_NAMES)
        ],
        reference_counts={name: 2048 for name in RAKUDA_ARM_JOINT_NAMES},
        joint_range_counts={name: (1000, 3000) for name in RAKUDA_ARM_JOINT_NAMES},
        current_sign={name: 1 for name in RAKUDA_ARM_JOINT_NAMES},
        drive_mode={name: 0 for name in RAKUDA_ARM_JOINT_NAMES},
        sign_check={"levels_ma": {"r_arm_sh_pitch1": 40}, "created_at": "2026-09-25T09:00:00"},
        arms=arms or {},
    )


def q17(rng: np.random.Generator) -> NDArray[np.int32]:
    return rng.integers(1200, 2900, len(RAKUDA_JOINT_NAMES)).astype(np.int32)


# --- basics ----------------------------------------------------------------------


def test_arm_joints_are_the_chain_order_of_each_arm() -> None:
    suffixes = ("sh_pitch1", "sh_roll", "sh_pitch2", "el_yaw", "wr_roll", "wr_yaw")
    assert RIGHT == tuple(f"r_arm_{s}" for s in suffixes)
    assert LEFT == tuple(f"l_arm_{s}" for s in suffixes)
    assert set(RIGHT) | set(LEFT) == set(RAKUDA_ARM_JOINT_NAMES)
    assert rg.ARM_SIDES == ("right", "left")
    with pytest.raises(ValueError, match="arm side"):
        rg.arm_joints("R")


def test_pose_sample_gravity_and_friction() -> None:
    def a(*values: float) -> FloatArray:
        return np.array(values, dtype=np.float64)

    ones, zeros = np.ones(6), np.zeros(6)
    sample = rg.PoseSample(ones, a(30, -10, 5, 0, 0, 0), a(10, -30, 5, 0, 0, 0), zeros + 40)
    np.testing.assert_allclose(sample.i_gravity_ma, [20, -20, 5, 0, 0, 0])
    np.testing.assert_allclose(sample.friction_ma, [10, 10, 0, 0, 0, 0])
    assert sample.flags == () and sample.is_val is False
    with pytest.raises(ValueError, match="q_counts"):
        rg.PoseSample(np.ones(5), zeros, zeros, zeros)
    with pytest.raises(ValueError, match="i_plus_ma"):
        rg.PoseSample(ones, zeros + np.nan, zeros, zeros)


# --- model math ------------------------------------------------------------------


def test_poe_regressor_matches_finite_difference_of_the_potential() -> None:
    rng = np.random.default_rng(0)
    axes = rng.normal(size=(6, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    moments = rng.normal(size=(6, 3)) * 50.0
    theta = rng.uniform(-2.0, 2.0, (5, 6))
    phi = rg._poe_regressor(theta, axes[None])[0]
    assert phi.shape == (5, 6, 18)
    fit = rg.ArmGravityFit(
        side="left",
        joints=LEFT,
        model_type=rg.MODEL_POE,
        q_ref_counts=np.full(6, 2000.0),
        axes=axes,
        first_moments=moments.reshape(-1),
        trig_coef=None,
        peak_ma={name: 1.0 for name in LEFT},
        friction_ma={name: 1.0 for name in LEFT},
    )
    predicted = fit.predict_arm_ma(2000.0 + theta / RAD_PER_COUNT)
    for n in range(theta.shape[0]):
        expected = central_gradient(lambda t: poe_potential(t, axes, moments), theta[n])
        np.testing.assert_allclose(phi[n] @ moments.reshape(-1), expected, atol=1e-6)
        np.testing.assert_allclose(predicted[n], expected, atol=1e-6)
    # Rank 12 of 18: each L_k's component along a_k merges into the previous link.
    many = rg._poe_regressor(rng.uniform(-1.5, 1.5, (40, 6)), axes[None])[0].reshape(-1, 18)
    assert np.linalg.matrix_rank(many, tol=1e-9 * np.linalg.norm(many, 2)) == 12


def test_trig_gradient_matches_finite_difference_of_the_potential() -> None:
    rng = np.random.default_rng(1)
    coef = rng.normal(size=73)

    def potential(t: FloatArray) -> float:
        c, s = np.cos(t), np.sin(t)
        terms = [1.0]
        for j in range(6):
            terms += [c[j], s[j]]
        for j in range(6):
            for k in range(j + 1, 6):
                terms += [c[j] * c[k], c[j] * s[k], s[j] * c[k], s[j] * s[k]]
        return float(np.dot(coef, terms))

    theta = rng.uniform(-3.0, 3.0, (4, 6))
    grad = rg._trig_gradient_features(theta)
    assert grad.shape == (4, 6, 73)
    np.testing.assert_array_equal(grad[:, :, 0], 0.0)
    for n in range(theta.shape[0]):
        np.testing.assert_allclose(grad[n] @ coef, central_gradient(potential, theta[n]), atol=1e-6)


def test_gravity_cannot_see_yaw_rotations_or_vertical_mirrors() -> None:
    """Why the axis candidates are reduced to 768 canonical sets."""
    rng = np.random.default_rng(2)
    axes = rng.normal(size=(6, 3))
    axes /= np.linalg.norm(axes, axis=1, keepdims=True)
    moments = rng.normal(size=(6, 3)) * 30.0
    theta = rng.uniform(-1.5, 1.5, (10, 6))
    base = rg._poe_currents(theta, axes, moments)
    yaw = Rotation.from_rotvec([0.0, 0.0, 0.7])
    np.testing.assert_allclose(
        rg._poe_currents(theta, yaw.apply(axes), yaw.apply(moments)), base, atol=1e-9
    )
    np.testing.assert_allclose(
        rg._poe_currents(theta, mirror_y(axes, axial=True), mirror_y(moments, axial=False)),
        base,
        atol=1e-9,
    )
    candidates = rg._canonical_candidates()
    assert candidates.shape == (768, 6)
    assert len({tuple(c) for c in candidates}) == 768
    # Canonical: the first horizontal axis is +y (code 1).
    first_horizontal = [next(c for c in row if c % 3 != 2) for row in candidates]
    assert set(first_horizontal) == {1}


# --- fitting ---------------------------------------------------------------------


@pytest.mark.parametrize(
    ("labels", "mirrored"),
    [
        (("+y", "+x", "+z", "+y", "+z", "+y"), False),
        # Not canonical: the fit returns its mirror twin (same predictions).
        (("+y", "-x", "+y", "-z", "+y", "+z"), True),
    ],
)
def test_fit_recovers_a_poe_arm(labels: Tuple[str, ...], mirrored: bool) -> None:
    case = poe_case(labels, seed=11 if not mirrored else 12)
    fit = fit_case(case)

    assert fit.model_type == rg.MODEL_POE
    assert fit.accepted is True
    report = fit.fit_report
    assert report["chosen"] == rg.MODEL_POE and report["accepted"] is True
    assert report["evaluated_on"] == "val" and report["n_val"] == 12
    poe = report["models"][rg.MODEL_POE]
    assert poe["rank"] == 12
    assert poe["pattern"] == "-".join("RPY"["xyz".index(label[1])] for label in labels)
    val = [s for s in case.samples if s.is_val]
    error = fit.predict_arm_ma(np.stack([s.q_counts for s in val])) - np.stack(
        [s.i_gravity_ma for s in val]
    )
    span = np.ptp(case.i_gravity, axis=0)
    assert np.all(np.sqrt(np.mean(error**2, axis=0)) < 0.05 * span)

    assert case.axes is not None and fit.axes is not None
    truth = mirror_y(case.axes, axial=True) if mirrored else case.axes
    truth = yaw_gauge(truth, axes_of(poe["start_axes"]))
    cosine = np.abs(np.sum(truth * fit.axes, axis=1))
    assert np.all(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) < 5.0)
    # The fit is JSON-ready as a whole.
    json.dumps(rg.arm_fit_to_dict(fit), allow_nan=False)


def hanging_case(seed: int, n: int = 30) -> Case:
    """A hanging arm (moments mostly along -z), axes tilted 20-30 deg, 5 mA noise."""
    rng = np.random.default_rng(seed)
    axes = []
    for a in axes_of(("+y", "+x", "+z", "+y", "+z", "+y")):
        tilt_axis = np.cross(a, rng.normal(size=3))
        tilt_axis /= np.linalg.norm(tilt_axis)
        axes.append(Rotation.from_rotvec(tilt_axis * np.radians(rng.uniform(20, 30))).apply(a))
    true_axes = np.array(axes)
    moments = np.zeros((6, 3))
    moments[:, 2] = -np.array([1.0, 1.0, 0.8, 0.6, 0.4, 0.3]) * np.array(
        [0.05, 0.35, 0.35, 0.3, 0.15, 0.1]
    )
    moments[:, :2] = rng.normal(scale=0.03, size=(6, 2))
    q_ref = rng.integers(1500, 2600, 6).astype(np.float64)
    q = q_ref + rng.uniform(-1024.0, 1024.0, (n, 6))
    theta = (q - q_ref) * RAD_PER_COUNT
    i_gravity = np.array(
        [central_gradient(lambda t: poe_potential(t, true_axes, moments), th) for th in theta]
    )
    i_gravity *= 200.0 / np.max(np.abs(i_gravity))
    friction = rng.uniform(15.0, 25.0, (n, 6))
    i_plus = i_gravity + friction + rng.normal(scale=5.0, size=(n, 6))
    i_minus = i_gravity - friction + rng.normal(scale=5.0, size=(n, 6))
    samples = [
        rg.PoseSample(q[i], i_plus[i], i_minus[i], np.full(6, 40.0), is_val=i % 5 == 4)
        for i in range(n)
    ]
    return Case(samples, q_ref, q, i_gravity, true_axes)


def test_fit_reaches_the_true_structure_when_the_screen_ranks_it_low() -> None:
    """At idealised axes the screen cannot rank the distal signs; refining only the best
    sign combination per pattern once ended on a boundary optimum here (val RMS 5.1 mA,
    rejected) instead of the true structure (3.4 mA)."""
    case = hanging_case(seed=16)
    fit = fit_case(case)
    poe = fit.fit_report["models"][rg.MODEL_POE]

    train = np.array([not s.is_val for s in case.samples])
    theta = (case.q - case.q_ref) * RAD_PER_COUNT
    y = np.stack([s.i_gravity_ma for s in case.samples])
    truth = rg._canonical(np.array([1, 0, 2, 1, 2, 1]))
    from_truth = rg._refine_candidate(theta[train], y[train], theta[~train], y[~train], truth, 0.0)

    assert not from_truth.at_bound
    assert poe["val_rms_ma"] <= 1.02 * from_truth.metric
    assert poe["at_bound"] == [False] * 6
    assert len(poe["candidates"]) == 32
    assert fit.model_type == rg.MODEL_POE and fit.accepted is True


def test_refinement_fixes_the_yaw_gauge_inside_the_bounds() -> None:
    case = poe_case(("+y", "+x", "+z", "+y", "+z", "+y"), seed=11)
    theta = (case.q - case.q_ref) * RAD_PER_COUNT
    target = np.stack([s.i_gravity_ma for s in case.samples]).reshape(-1)
    start = axes_of(("+y", "+x", "+z", "+y", "+z", "+y"))
    axes, on_bound = rg._refine_axes(theta, target, start)

    assert on_bound.shape == (6,) and not on_bound.any()
    # The first horizontal start (+y) only tilts vertically: its azimuth is kept.
    assert abs(axes[0, 0]) < 1e-12 and axes[0, 1] > 0.0
    # Both tilt angles within 45 deg of the start: at most 60 deg away in total.
    cosine = np.sum(axes * start, axis=1)
    assert np.all(np.degrees(np.arccos(np.clip(cosine, -1.0, 1.0))) <= 60.0 + 1e-9)


def test_module_runs_the_cli() -> None:
    """``python -m robopy.robots.rakuda.rakuda_gravity`` is the CLI."""
    env = {k: v for k, v in os.environ.items() if k != "PYTHONPATH"}
    done = subprocess.run(
        [sys.executable, "-m", "robopy.robots.rakuda.rakuda_gravity", "--help"],
        capture_output=True,
        text=True,
        env=env,
        timeout=120,
        check=False,
    )
    assert done.returncode == 0, done.stderr
    assert "sign-check" in done.stdout and "verify" in done.stdout


def test_peak_and_friction_come_from_the_dataset() -> None:
    case = poe_case(("+y", "+x", "+y", "+z", "+y", "+z"), seed=21)
    fit = fit_case(case)
    peak = np.max(np.abs(fit.predict_arm_ma(case.q)), axis=0)
    friction = np.median(np.stack([s.friction_ma for s in case.samples]), axis=0)
    assert fit.peak_ma == pytest.approx(dict(zip(RIGHT, peak)))
    assert fit.friction_ma == pytest.approx(dict(zip(RIGHT, friction)))


def test_trig_fallback_wins_on_non_poe_data() -> None:
    case = trig_case(seed=5)
    fit = fit_case(case)
    models = fit.fit_report["models"]
    assert fit.model_type == rg.MODEL_TRIG
    assert fit.axes is None and fit.first_moments is None and fit.trig_coef is not None
    assert models[rg.MODEL_TRIG]["eval_rms_ma"] < models[rg.MODEL_POE]["eval_rms_ma"]
    assert fit.accepted is True


def test_a_validation_outlier_rejects_both_models() -> None:
    """One validation pose 30 mA off on one joint: only that joint's max-error limit fails."""
    case = poe_case(("+y", "+x", "+z", "+y", "+z", "+y"), seed=11)
    bad_pose, bad_joint = 4, 3  # the first validation pose, el_yaw
    assert case.samples[bad_pose].is_val
    shift = np.zeros(6)
    shift[bad_joint] = 30.0  # moves I_g, keeps f_s
    s = case.samples[bad_pose]
    case.samples[bad_pose] = rg.PoseSample(
        s.q_counts, s.i_plus_ma + shift, s.i_minus_ma + shift, s.temperature_c, is_val=True
    )
    fit = fit_case(case)

    assert fit.accepted is False
    report = fit.fit_report
    assert report["accepted"] is False
    # Both rejected: the lower evaluation RMS is kept.
    assert report["chosen"] == fit.model_type == rg.MODEL_POE
    models = report["models"]
    assert models[rg.MODEL_POE]["accepted"] is False and models[rg.MODEL_TRIG]["accepted"] is False
    assert models[rg.MODEL_POE]["eval_rms_ma"] < models[rg.MODEL_TRIG]["eval_rms_ma"]

    y = np.stack([s.i_gravity_ma for s in case.samples])
    friction = np.median(np.stack([s.friction_ma for s in case.samples]), axis=0)
    for j, name in enumerate(RIGHT):
        limits = report["limits"][name]
        assert limits["rms_limit_ma"] == pytest.approx(max(5.0, 0.1 * np.ptp(y[:, j])))
        assert limits["max_limit_ma"] == pytest.approx(max(10.0, 0.8 * friction[j]))
        joint = models[rg.MODEL_POE]["per_joint"][name]
        assert joint["ok"] is (j != bad_joint)
        assert joint["rms_ma"] <= limits["rms_limit_ma"]  # RMS alone would pass
        assert (joint["max_abs_ma"] > limits["max_limit_ma"]) is (j == bad_joint)
    assert models[rg.MODEL_TRIG]["per_joint"][RIGHT[bad_joint]]["ok"] is False


def test_fit_refuses_too_few_or_unusable_samples() -> None:
    case = trig_case(seed=6, n=25)
    with pytest.raises(ValueError, match="at least 25"):
        rg.fit_arm(case.samples[:24], "right", reference(case.q_ref))
    all_val = [
        rg.PoseSample(s.q_counts, s.i_plus_ma, s.i_minus_ma, s.temperature_c, is_val=True)
        for s in case.samples
    ]
    with pytest.raises(ValueError, match="validation"):
        rg.fit_arm(all_val, "right", reference(case.q_ref))
    with pytest.raises(ValueError, match="r_arm_wr_yaw"):
        rg.fit_arm(case.samples, "right", reference(case.q_ref, RIGHT[:5]))


# --- prediction ------------------------------------------------------------------


@pytest.mark.parametrize("model_type", [rg.MODEL_POE, rg.MODEL_TRIG])
def test_predict_arm_shapes(model_type: str) -> None:
    fit = make_fit("right", model_type)
    rng = np.random.default_rng(3)
    q = rng.uniform(1500, 2600, (4, 6))
    batch = fit.predict_arm_ma(q)
    assert batch.shape == (4, 6)
    single = fit.predict_arm_ma(q[2])
    assert single.shape == (6,)
    np.testing.assert_allclose(single, batch[2])
    with pytest.raises(ValueError, match="shape"):
        fit.predict_arm_ma(np.zeros(7))


def test_leader_model_places_each_arm_and_zeros_the_rest() -> None:
    right, left = make_fit("right"), make_fit("left", rg.MODEL_TRIG)
    rng = np.random.default_rng(4)
    q = q17(rng)
    idx_r = [RAKUDA_JOINT_NAMES.index(name) for name in RIGHT]
    idx_l = [RAKUDA_JOINT_NAMES.index(name) for name in LEFT]

    one: GravityModel = rg.LeaderGravityModel({"right": right})
    out = one.predict_ma(q)
    assert out.shape == (17,)
    np.testing.assert_allclose(out[idx_r], right.predict_arm_ma(q[idx_r]))
    others = [i for i in range(17) if i not in idx_r]
    np.testing.assert_array_equal(out[others], 0.0)

    both = rg.LeaderGravityModel({"right": right, "left": left})
    out = both.predict_ma(q)
    np.testing.assert_allclose(out[idx_l], left.predict_arm_ma(q[idx_l]))
    np.testing.assert_allclose(out[idx_r], right.predict_arm_ma(q[idx_r]))
    np.testing.assert_array_equal(out[[RAKUDA_JOINT_NAMES.index("torso_yaw")]], 0.0)
    assert set(both.peak_ma) == set(RAKUDA_JOINT_NAMES)
    assert both.peak_ma["r_arm_sh_roll"] == right.peak_ma["r_arm_sh_roll"]
    assert both.peak_ma["head_yaw"] == 0.0 and both.peak_ma["torso_yaw"] == 0.0
    assert rg.LeaderGravityModel({}).predict_ma(q).tolist() == [0.0] * 17

    with pytest.raises(ValueError, match="side"):
        rg.LeaderGravityModel({"left": right})
    with pytest.raises(ValueError, match="shape"):
        both.predict_ma(q[:12])


# --- storage ---------------------------------------------------------------------


def test_dataset_round_trip_with_sha256_and_backup(tmp_path: Path) -> None:
    rng = np.random.default_rng(9)
    samples = [
        rg.PoseSample(
            rng.uniform(1500, 2600, 6),
            rng.normal(size=6) * 50,
            rng.normal(size=6) * 50,
            rng.uniform(30, 50, 6),
            flags=("noisy",) if i == 1 else (),
            is_val=i % 5 == 4,
        )
        for i in range(7)
    ]
    path = tmp_path / "leader_gravity_dataset_right.npz"
    digest = rg.save_dataset(path, "right", samples, {"operator": "test", "q_ref": [2048] * 6})
    assert digest == hashlib.sha256(path.read_bytes()).hexdigest()

    loaded, header = rg.load_dataset(path)
    assert header["side"] == "right" and header["joints"] == list(RIGHT)
    assert header["operator"] == "test"
    assert len(loaded) == len(samples)
    for a, b in zip(loaded, samples):
        for attr in ("q_counts", "i_plus_ma", "i_minus_ma", "temperature_c"):
            np.testing.assert_array_equal(getattr(a, attr), getattr(b, attr))
        assert a.flags == b.flags and a.is_val == b.is_val

    first = path.read_bytes()
    rg.save_dataset(path, "right", samples[:3], {})
    assert (tmp_path / "leader_gravity_dataset_right.npz.bak").read_bytes() == first
    assert len(rg.load_dataset(path)[0]) == 3


def test_gravity_file_round_trip(tmp_path: Path) -> None:
    right, left = make_fit("right"), make_fit("left", rg.MODEL_TRIG, accepted=False)
    data = make_file(
        {"right": arm_entry(right, validated=True), "left": arm_entry(left, validated=False)}
    )
    path = tmp_path / "rakuda" / "leader_gravity.json"
    rg.save_gravity_file(path, data)
    assert rg.load_gravity_file(path) == data
    assert json.loads(path.read_text())["schema_version"] == 1

    restored = rg.arm_fit_from_dict(rg.load_gravity_file(path).arms["right"])
    assert restored.model_type == right.model_type and restored.accepted is True
    q = np.random.default_rng(5).uniform(1500, 2600, (3, 6))
    np.testing.assert_allclose(restored.predict_arm_ma(q), right.predict_arm_ma(q))
    restored_left = rg.arm_fit_from_dict(rg.arm_fit_to_dict(left))
    np.testing.assert_allclose(restored_left.predict_arm_ma(q), left.predict_arm_ma(q))
    assert restored_left.peak_ma == left.peak_ma and restored_left.accepted is False


def test_partial_file_before_identification(tmp_path: Path) -> None:
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, rg.GravityFile(reference_counts={"r_arm_sh_roll": 2000}))
    loaded = rg.load_gravity_file(path)
    assert loaded.reference_counts == {"r_arm_sh_roll": 2000}
    assert loaded.arms == {} and loaded.current_sign == {} and loaded.drive_mode == {}


def test_save_is_atomic_and_keeps_a_backup(tmp_path: Path) -> None:
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, make_file())
    first = path.read_text()
    second = make_file({"right": arm_entry(make_fit("right"), validated=False)})
    rg.save_gravity_file(path, second)
    assert (tmp_path / "leader_gravity.json.bak").read_text() == first
    assert rg.load_gravity_file(path) == second
    # Invalid data is refused before anything is written.
    bad = make_file()
    bad.current_sign["r_arm_sh_roll"] = 0
    with pytest.raises(ValueError, match=r"current_sign\.r_arm_sh_roll"):
        rg.save_gravity_file(path, bad)
    assert rg.load_gravity_file(path) == second
    assert sorted(os.listdir(tmp_path)) == ["leader_gravity.json", "leader_gravity.json.bak"]


def test_missing_file_and_unknown_schema(tmp_path: Path) -> None:
    with pytest.raises(FileNotFoundError):
        rg.load_gravity_file(tmp_path / "none.json")
    path = tmp_path / "leader_gravity.json"
    path.write_text(json.dumps({"schema_version": 2}))
    with pytest.raises(ValueError, match="schema_version"):
        rg.load_gravity_file(path)
    path.write_text("{not json")
    with pytest.raises(ValueError, match="invalid JSON"):
        rg.load_gravity_file(path)


def _break(raw: Dict[str, Any], dotted: str, value: Any) -> None:
    *parents, last = dotted.split(".")
    node = raw
    for key in parents:
        node = node[key]
    node[last] = value


@pytest.mark.parametrize(
    ("dotted", "value", "match"),
    [
        ("current_sign.r_arm_sh_roll", 2, r"current_sign\.r_arm_sh_roll"),
        ("joint_range_counts.l_arm_wr_yaw", [3000, 1000], r"joint_range_counts\.l_arm_wr_yaw"),
        ("reference_counts.r_arm_elbow", 2048, r"reference_counts\.r_arm_elbow"),
        ("drive_mode.r_arm_sh_roll", -1, r"drive_mode\.r_arm_sh_roll"),
        ("motors", {"name": "x"}, "motors"),
        ("arms.middle", {}, r"arms\.middle"),
        ("arms.right.axes", [[1.0, 0.0, 0.0]], r"arms\.right\.axes"),
        ("arms.right.L", None, r"arms\.right\.L"),
        ("arms.right.model_type", "spline", r"arms\.right\.model_type"),
        ("arms.right.validated", "yes", r"arms\.right\.validated"),
        ("arms.right.peak_ma", {"r_arm_sh_roll": 1.0}, r"arms\.right\.peak_ma"),
    ],
)
def test_malformed_entries_name_the_key(
    tmp_path: Path, dotted: str, value: Any, match: str
) -> None:
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, make_file({"right": arm_entry(make_fit("right"), validated=True)}))
    raw = json.loads(path.read_text())
    _break(raw, dotted, value)
    path.write_text(json.dumps(raw))
    with pytest.raises(ValueError, match=match):
        rg.load_gravity_file(path)


# --- load_bilateral_setup ----------------------------------------------------------


def _gate(setup: rakuda_pair_sys.BilateralSetup, **params: Any) -> None:
    rakuda_pair_sys._check_bilateral_setup(setup, RakudaBilateralParams(**params))


def test_setup_without_arms_is_uncompensated(tmp_path: Path) -> None:
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, make_file())
    setup = rg.load_bilateral_setup(path)
    assert isinstance(setup, rakuda_pair_sys.BilateralSetup)
    assert setup.gravity is None and setup.gravity_validated is False
    assert setup.gravity_peak_ma is None and setup.source == str(path)
    assert setup.current_sign == {name: 1 for name in RAKUDA_ARM_JOINT_NAMES}
    assert setup.joint_range_counts["l_arm_sh_roll"] == (1000, 3000)
    assert setup.drive_mode["r_arm_wr_yaw"] == 0
    with pytest.raises(ValueError, match="allow_uncompensated"):
        _gate(setup)
    _gate(setup, allow_uncompensated=True)


def test_setup_with_one_arm_is_not_validated(tmp_path: Path) -> None:
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, make_file({"right": arm_entry(make_fit("right"), validated=True)}))
    setup = rg.load_bilateral_setup(path)
    assert isinstance(setup.gravity, rg.LeaderGravityModel)
    assert set(setup.gravity.arms) == {"right"}
    assert setup.gravity_validated is False
    assert setup.gravity_peak_ma is not None and setup.gravity_peak_ma["l_arm_sh_roll"] == 0.0
    right_arm = tuple(name for name in RAKUDA_ARM_JOINT_NAMES if name.startswith("r_arm_"))
    assert setup.gravity_joints == frozenset(right_arm)
    with pytest.raises(ValueError, match="allow_unvalidated_gravity"):
        _gate(setup, current_joints=right_arm)
    _gate(setup, current_joints=right_arm, allow_unvalidated_gravity=True)


def test_setup_refuses_current_joints_the_model_does_not_cover(tmp_path: Path) -> None:
    # A right-arm model must not let the unidentified left arm run in current
    # mode with a zero gravity term behind allow_unvalidated_gravity alone.
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(path, make_file({"right": arm_entry(make_fit("right"), validated=True)}))
    setup = rg.load_bilateral_setup(path)
    left_arm = [name for name in RAKUDA_ARM_JOINT_NAMES if name.startswith("l_arm_")]
    with pytest.raises(ValueError, match="does not cover") as excinfo:
        _gate(setup, allow_unvalidated_gravity=True)
    for name in left_arm:
        assert name in str(excinfo.value)
    # Running the left arm uncompensated is an explicit choice.
    _gate(setup, allow_unvalidated_gravity=True, allow_uncompensated=True)
    # torso_yaw turns about the vertical axis and needs no model.
    with_torso = dataclasses.replace(
        setup,
        current_sign={**setup.current_sign, "torso_yaw": 1},
        joint_range_counts={**setup.joint_range_counts, "torso_yaw": (1000, 3000)},
        drive_mode={**setup.drive_mode, "torso_yaw": 0},
    )
    right_and_torso = ("torso_yaw",) + tuple(
        name for name in RAKUDA_ARM_JOINT_NAMES if name.startswith("r_arm_")
    )
    _gate(with_torso, current_joints=right_and_torso, allow_unvalidated_gravity=True)


def test_setup_with_both_arms_validated_passes_the_gate(tmp_path: Path) -> None:
    right, left = make_fit("right"), make_fit("left", rg.MODEL_TRIG)
    path = tmp_path / "leader_gravity.json"
    rg.save_gravity_file(
        path,
        make_file(
            {"right": arm_entry(right, validated=True), "left": arm_entry(left, validated=True)}
        ),
    )
    # Through the pair system's lazy import, as start_bilateral() does.
    setup = rakuda_pair_sys.load_bilateral_setup(path)
    assert setup.gravity_validated is True
    assert setup.gravity_joints == frozenset(RAKUDA_ARM_JOINT_NAMES)
    _gate(setup)

    params = RakudaBilateralParams()
    law = BilateralLaw(
        params.current_joints,
        params,
        setup.gravity,
        setup.joint_range_counts,
        gravity_peak_ma=setup.gravity_peak_ma,
    )
    rng = np.random.default_rng(6)
    q = q17(rng)
    state = RakudaArmState(
        names=RAKUDA_JOINT_NAMES,
        position=q,
        velocity=np.zeros(17, dtype=np.int32),
        current_ma=np.zeros(17, dtype=np.float32),
        t_start_ns=0,
        t_end_ns=0,
        seq=0,
    )
    assert setup.gravity is not None and setup.gravity_peak_ma is not None
    idx = [RAKUDA_JOINT_NAMES.index(name) for name in params.current_joints]
    bound = params.gravity_clamp_factor * np.array(
        [setup.gravity_peak_ma[name] for name in params.current_joints]
    )
    assert isinstance(params.gravity_scale, float)
    expected = np.clip(params.gravity_scale * setup.gravity.predict_ma(q)[idx], -bound, bound)
    np.testing.assert_allclose(law.gravity_term(state), expected)


def test_setup_is_not_validated_when_a_fit_was_not_accepted(tmp_path: Path) -> None:
    path = tmp_path / "leader_gravity.json"
    arms = {
        "right": arm_entry(make_fit("right"), validated=True),
        "left": arm_entry(make_fit("left", accepted=False), validated=True),
    }
    rg.save_gravity_file(path, make_file(arms))
    assert rg.load_bilateral_setup(path).gravity_validated is False
