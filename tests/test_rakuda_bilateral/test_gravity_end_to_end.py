"""End to end: identify the right leader arm, then run bilateral with the fit.

The whole gravity bring-up runs on one simulated leader and one
simulated follower sharing a clock, with the same gravity file throughout:
``range`` -> ``sign-check`` (right arm; the left arm is skipped) ->
``identify --arm right`` (30 poses) -> ``fit`` -> ``verify``, and then
``RakudaPairSys.start_bilateral()`` loads that file through
``load_bilateral_setup`` and holds the arm with the fitted model.

The right arm's plant feels the gravity of a synthetic PoE arm whose axes are
an axis-aligned pattern tilted by 10-15 deg (so the fit has to refine them),
with 10 mA of static friction.  The operator's hands are the
``test_gravity_cli`` harness: while they support the arm it feels no gravity
(``range`` and ``sign-check`` model the premise that the arm hangs at a
gravity-free pose that way).  The CLI loops run cycle by cycle on the shared
clock (``SteppedRunner``); the bilateral loop runs on its real control thread.

``gravity_scale`` is 1.0 throughout: the default 0.9 under-compensates a
150 mA joint by more than its friction, and raising it is exactly what
``verify`` recommends on such an arm.
"""

from __future__ import annotations

import time
from dataclasses import replace
from pathlib import Path
from typing import Any, Callable, Dict, Iterator, List, Mapping, Tuple, cast

import numpy as np
import pytest

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_ARM_JOINT_NAMES,
    RakudaBilateralParams,
    RakudaConfig,
)
from robopy.motor.dynamixel_bus import DynamixelBus, DynamixelMotor
from robopy.motor.dynamixel_control_table import OperatingMode
from robopy.motor.sim_dynamixel_bus import SimulatedDynamixelBus
from robopy.robots.rakuda import rakuda_gravity_cli as cli
from robopy.robots.rakuda import rakuda_pair_sys
from robopy.robots.rakuda.rakuda_gravity import (
    MODEL_POE,
    ArmGravityFit,
    LeaderGravityModel,
    arm_fit_from_dict,
    load_dataset,
    load_gravity_file,
)
from robopy.robots.rakuda.rakuda_leader_control import LoopState
from robopy.robots.rakuda.rakuda_pair_sys import RakudaPairSys

from .conftest import make_follower_bus
from .test_gravity_cli import (
    HOME,
    LEFT,
    RIGHT,
    Operator,
    ScriptedConsole,
    SimLeader,
    SteppedRunner,
    identify_script,
    is_held,
    is_released,
    make_env,
    place,
    random_poses,
    verify_script,
)
from .test_pair_sys_bilateral import YieldingSimClock

LEADER_PORT = "sim://leader"
FOLLOWER_PORT = "sim://follower"
COULOMB_MA = 10.0
#: The ranges ``range`` measures: every arm joint moved to ``HOME ± RANGE_HALF``.
RANGE_HALF = 900
IDENTIFY_POSES = 30
DRIFT_LIMIT_COUNTS = cli.VERIFY_DRIFT_LIMIT_COUNTS
BILATERAL_S = 2.0
#: Where the follower's right arm starts relative to the leader (the alignment ramp closes it).
FOLLOWER_OFFSET = 40

#: The truth: the P-R-Y-P-Y-P pattern of ``test_gravity_cli``, each axis tilted by
#: ``TILT_DEG`` toward the azimuth ``TILT_AZIMUTH_DEG`` around it.
IDEAL_AXES = np.array(
    [(0, 1, 0), (1, 0, 0), (0, 0, 1), (0, 1, 0), (0, 0, 1), (0, 1, 0)], dtype=np.float64
)
TILT_DEG = (12.0, 10.0, 15.0, 13.0, 11.0, 14.0)
TILT_AZIMUTH_DEG = (30.0, 110.0, 200.0, 290.0, 60.0, 160.0)
#: Link-major first moments (mA): a hanging arm whose wrist and hand reach forward, so
#: every joint (yaw and roll ones included) peaks at 50-150 mA over the identify poses.
MOMENTS = np.array(
    [(0, 0, -15), (0, 0, -10), (0, 5, -10), (5, 0, -15), (55, 0, -5), (40, 0, -80)],
    dtype=np.float64,
)


def tilted_axes() -> np.ndarray:
    """``IDEAL_AXES`` tilted by ``TILT_DEG`` toward ``TILT_AZIMUTH_DEG`` (unit vectors)."""
    axes = []
    for ideal, tilt, azimuth in zip(IDEAL_AXES, np.radians(TILT_DEG), np.radians(TILT_AZIMUTH_DEG)):
        u = np.cross(ideal, np.eye(3)[int(np.argmin(np.abs(ideal)))])
        u /= np.linalg.norm(u)
        side = np.cos(azimuth) * u + np.sin(azimuth) * np.cross(ideal, u)
        axes.append(np.cos(tilt) * ideal + np.sin(tilt) * side)
    return np.array(axes)


TRUTH = ArmGravityFit(
    side="right",
    joints=RIGHT,
    model_type=MODEL_POE,
    q_ref_counts=np.full(len(RIGHT), float(HOME)),
    axes=tilted_axes(),
    first_moments=MOMENTS.ravel(),
    trig_coef=None,
    peak_ma={name: 150.0 for name in RIGHT},
    friction_ma={name: COULOMB_MA for name in RIGHT},
    fit_report={"note": "synthetic truth"},
    accepted=True,
)


class WatchedOperator(Operator):
    """``Operator`` that also records the right arm's largest excursion from ``origin``.

    The plant evaluates the gravity once per integration step, so the excursion
    covers every step of the loop, whatever the thread that stepped it.
    """

    def __init__(self, fit: ArmGravityFit) -> None:
        super().__init__(fit)
        self.origin: np.ndarray | None = None
        self.excursion = np.zeros(len(RIGHT))

    def watch(self, origin: np.ndarray) -> None:
        self.excursion = np.zeros(len(RIGHT))
        self.origin = origin.copy()

    def gravity(self, positions: Mapping[str, float]) -> Dict[str, float]:
        if self.origin is not None:
            q = np.array([positions[name] for name in RIGHT])
            np.maximum(self.excursion, np.abs(q - self.origin), out=self.excursion)
        return super().gravity(positions)


class PersistentSimLeader(SimLeader):
    """``SimLeader`` that builds its bus once: every command reconnects to the same arm."""

    def make(self, port: str = LEADER_PORT) -> SimulatedDynamixelBus:
        return super().make(port) if self.bus is None else self.bus


def right_positions(bus: SimulatedDynamixelBus) -> np.ndarray:
    return np.array([bus.joint(name).position_counts for name in RIGHT])


def wait_until(condition: Callable[[], bool], timeout_s: float = 10.0) -> None:
    """Polls ``condition`` on the wall clock (the loop thread runs on the sim clock)."""
    deadline = time.monotonic() + timeout_s
    while not condition():
        assert time.monotonic() < deadline, "condition not met in time"
        time.sleep(0.005)


@pytest.fixture
def pairs(monkeypatch: pytest.MonkeyPatch) -> Iterator[List[RakudaPairSys]]:
    """Every pair a test builds; loops are stopped and atexit hooks removed at teardown."""
    # Keep the user's .robopy/rakuda/config.yaml out of the configuration.
    monkeypatch.setattr(rakuda_pair_sys, "apply_rakuda_dotconfig", lambda cfg: cfg)
    made: List[RakudaPairSys] = []
    yield made
    for pair in made:
        RakudaPairSys.stop_bilateral(pair)  # never leave a control thread behind
        RakudaPairSys._unregister_atexit(pair)


def range_script(factory: SimLeader) -> ScriptedConsole:
    """Reference pose at ``HOME``, then every arm joint to both ends and back."""

    def bus() -> SimulatedDynamixelBus:
        assert factory.bus is not None
        return factory.bus

    def reference_pose() -> str:
        place(bus(), {name: HOME for name in (*RAKUDA_ARM_JOINT_NAMES, "torso_yaw")})
        return ""

    def move_every_joint(tick: Callable[[], None]) -> str:
        tick()
        for name in RAKUDA_ARM_JOINT_NAMES:
            for position in (HOME - RANGE_HALF, HOME + RANGE_HALF, HOME):
                place(bus(), {name: position})
                tick()
        return ""

    return ScriptedConsole(
        ("switched OFF", ""),
        ("reference pose", reference_pose),
        ("safe ends", move_every_joint),
    )


def sign_check_script() -> ScriptedConsole:
    """Every right-arm joint is checked; the left arm is skipped."""
    steps: List[Tuple[str, Any]] = [("must hang freely", "")]
    steps += [
        (f"{name}: make sure", "" if name in RIGHT else "s") for name in RAKUDA_ARM_JOINT_NAMES
    ]
    return ScriptedConsole(*steps)


def test_identify_then_bilateral_with_the_fitted_model(
    tmp_path: Path, pairs: List[RakudaPairSys]
) -> None:
    path = tmp_path / "leader_gravity.json"
    clock = YieldingSimClock()
    operator = WatchedOperator(TRUTH)
    factory = PersistentSimLeader(clock, operator, coulomb_ma=COULOMB_MA)
    leader = factory.make()
    runner = SteppedRunner(clock)
    params = RakudaBilateralParams(gravity_scale=1.0)

    def env(console: ScriptedConsole) -> cli.CliEnv:
        return make_env(console, factory, params=params, runner=runner)

    identify_poses = random_poses(IDENTIFY_POSES, seed=1)
    truth_on_poses = TRUTH.predict_arm_ma(np.array([list(p.values()) for p in identify_poses]))
    peaks = np.max(np.abs(truth_on_poses), axis=0)
    assert np.all((peaks >= 50.0) & (peaks <= 150.0)), peaks

    # -- range: reference pose and ranges, both arms released ------------------------
    console = range_script(factory)
    assert cli.cmd_range(env(console), port=LEADER_PORT, path=str(path)) == 0, console.text
    console.assert_done()
    data = load_gravity_file(path)
    assert all(data.reference_counts[name] == HOME for name in RAKUDA_ARM_JOINT_NAMES)
    assert data.joint_range_counts == {
        name: (HOME - RANGE_HALF, HOME + RANGE_HALF) for name in RAKUDA_ARM_JOINT_NAMES
    }
    assert is_released(leader, RAKUDA_ARM_JOINT_NAMES)

    # -- sign-check: the right arm, the left one skipped -----------------------------
    console = sign_check_script()
    assert cli.cmd_sign_check(env(console), port=LEADER_PORT, path=str(path)) == 0, console.text
    console.assert_done()
    data = load_gravity_file(path)
    assert data.current_sign == {name: 1 for name in RIGHT}
    assert data.drive_mode == {name: 0 for name in RIGHT}
    assert data.sign_check["skipped"] == list(LEFT)
    assert is_released(leader, RAKUDA_ARM_JOINT_NAMES)

    # -- identify: 30 operator-placed poses, fitted at the end, the arm kept held ------
    console = identify_script(factory, operator, identify_poses)
    code = cli.cmd_identify(env(console), port=LEADER_PORT, path=str(path), arm="right")
    assert code == 0, console.text
    console.assert_done()
    samples, header = load_dataset(tmp_path / "leader_gravity_dataset_right.npz")
    assert len(samples) == IDENTIFY_POSES and header["side"] == "right"
    assert sum(s.is_val for s in samples) == IDENTIFY_POSES // cli.VALIDATION_EVERY
    measured = np.stack([s.i_gravity_ma for s in samples])
    truth = TRUTH.predict_arm_ma(np.stack([s.q_counts for s in samples]))
    assert float(np.max(np.abs(measured - truth))) <= COULOMB_MA
    assert is_held(leader, RIGHT) and is_released(leader, LEFT)

    # -- fit: the PoE model is accepted and predicts the truth ------------------------
    console = ScriptedConsole()
    assert cli.cmd_fit(env(console), path=str(path), arm="right") == 0, console.text
    entry = load_gravity_file(path).arms["right"]
    assert entry["accepted"] is True and entry["validated"] is False
    fitted = arm_fit_from_dict(entry)
    assert fitted.model_type == MODEL_POE
    check = np.array([list(p.values()) for p in random_poses(50, seed=99, spread=350.0)])
    error = fitted.predict_arm_ma(check) - TRUTH.predict_arm_ma(check)
    assert float(np.max(np.abs(error))) <= COULOMB_MA

    # -- verify: gravity compensation alone holds the arm; validated ------------------
    verify_poses = random_poses(cli.VERIFY_POSES, seed=11, spread=350.0)
    console = verify_script(factory, operator, runner, verify_poses)
    assert cli.cmd_verify(env(console), port=LEADER_PORT, path=str(path), arm="right") == 0, (
        console.text
    )
    console.assert_done()
    entry = load_gravity_file(path).arms["right"]
    assert entry["validated"] is True and entry["validated_at"]
    assert max(entry["verify"]["max_abs_drift_counts"].values()) < DRIFT_LIMIT_COUNTS
    assert is_held(leader, RIGHT)

    # -- the setup the pair loads: one identified arm is not "validated" --
    bilateral = replace(
        params, current_joints=RIGHT, gravity_model_path=str(path), allow_unvalidated_gravity=True
    )
    setup = rakuda_pair_sys.load_bilateral_setup(path)
    assert isinstance(setup.gravity, LeaderGravityModel) and set(setup.gravity.arms) == {"right"}
    assert setup.gravity_validated is False and setup.source == str(path)
    with pytest.raises(ValueError, match="allow_unvalidated_gravity=true"):
        rakuda_pair_sys._check_bilateral_setup(
            setup, replace(bilateral, allow_unvalidated_gravity=False)
        )

    # -- bilateral: the pair starts from the file and the fitted model holds the arm ---
    q0 = right_positions(leader)
    truth_at_q0 = TRUTH.predict_arm_ma(q0)
    assert int(np.count_nonzero(np.abs(truth_at_q0) > 2 * COULOMB_MA)) >= 3  # not friction
    follower = make_follower_bus(clock=clock, auto_step=True, port=FOLLOWER_PORT)
    for name, position in zip(RIGHT, q0):
        follower.joint(name).position_counts = position + FOLLOWER_OFFSET

    def bus_factory(port: str, motors: Dict[str, DynamixelMotor]) -> DynamixelBus:
        return cast(DynamixelBus, follower) if port == FOLLOWER_PORT else factory(port, motors)

    config = RakudaConfig(leader_port=LEADER_PORT, follower_port=FOLLOWER_PORT, bilateral=bilateral)
    pair = RakudaPairSys(config, bus_factory, clock=clock.monotonic_ns, sleep=clock.sleep)
    pairs.append(pair)
    pair.connect()
    operator.watch(q0)
    loop = pair.start_bilateral()
    assert loop.running and loop.fault is None
    report = pair.control_report()
    assert report is not None and report["setup"] == {
        "source": str(path),
        "gravity_validated": False,
        "uncompensated": False,
    }
    t_end = clock.now_ns + int(BILATERAL_S * 1e9)
    wait_until(lambda: clock.now_ns >= t_end or not loop.running)
    assert loop.running and loop.fault is None
    goal_current = np.array([leader.registers(name).goal_current for name in RIGHT], float)
    assert np.all(np.abs(goal_current - truth_at_q0) <= COULOMB_MA)
    for name in RIGHT:
        assert abs(follower.joint(name).position_counts - leader.joint(name).position_counts) <= 5

    assert pair.stop_bilateral() is True
    assert loop.state is LoopState.HELD and loop.hold_verified
    assert is_held(leader, RIGHT)
    assert all(leader.registers(name).operating_mode == OperatingMode.POSITION for name in RIGHT)
    assert float(np.max(operator.excursion)) < DRIFT_LIMIT_COUNTS, operator.excursion
    assert float(np.max(np.abs(right_positions(leader) - q0))) < DRIFT_LIMIT_COUNTS
