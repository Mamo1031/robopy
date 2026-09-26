"""``LeaderCurrentLoop`` driving the real laws on simulated buses.

``test_leader_loop.py`` runs the loop against a stub law and
``test_control_laws.py`` runs the laws on hand-built snapshots; here
``BilateralLaw`` and ``SignCheckLaw`` run *inside* the loop on
``SimulatedDynamixelBus`` plants with gravity, so the joint convention, the
unit/sign tables, the range exposure, the pair-to-observation conversion and
the fault mapping are checked end to end.  No port, no thread: ``run_once``
is driven on a shared ``SimClock`` with ``auto_step`` on, so the plant moves
by exactly the simulated time between transactions.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Dict, Mapping, Sequence, Tuple

import numpy as np
import pytest
from numpy.typing import NDArray

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    RakudaArmObs,
    RakudaBilateralParams,
)
from robopy.motor.dynamixel_control_table import CURRENT_UNIT_MA, OperatingMode, XControlTable
from robopy.motor.sim_dynamixel_bus import SimClock, SimulatedDynamixelBus
from robopy.robots.rakuda.rakuda_control_laws import (
    BilateralLaw,
    ControlLaw,
    IdentifyParams,
    PoseHoldLaw,
    SignCheckLaw,
    SignCheckParams,
)
from robopy.robots.rakuda.rakuda_leader_control import (
    FollowerPositionIO,
    LeaderCurrentLoop,
    LoopState,
    LoopStopped,
    pair_snapshot_to_obs,
)

from .conftest import make_follower_bus, make_leader_bus

J3: Tuple[str, ...] = RAKUDA_ARM_JOINT_NAMES[:3]  # r_arm_sh_pitch1, r_arm_sh_roll, r_arm_sh_pitch2
HEAVY = "r_arm_sh_roll"  # the joint the tests load with gravity
HOME = 2048  # every simulated joint starts here
#: Safe range of ``J3``: the lower barrier (``lo + limit_margin_counts``) sits
#: 93 counts below HOME so an uncompensated joint reaches it within a second.
RANGE: Dict[str, Tuple[int, int]] = {name: (HOME - 150, HOME + 1024) for name in J3}
GRAVITY_MA = 30.0
N = len(RAKUDA_JOINT_NAMES)


class SimGravity:
    """A ``GravityModel`` that knows the simulated plant's holding current of ``joints``.

    The simulator's ``gravity_ma`` is the current in the count frame that
    balances gravity, which is exactly the joint-convention term the law
    expects (positive pushes toward increasing counts); other joints get 0.
    """

    def __init__(self, bus: SimulatedDynamixelBus, joints: Sequence[str]) -> None:
        self.bus = bus
        self.joints = tuple(joints)

    def predict_ma(self, q_counts: NDArray[np.integer[Any]]) -> NDArray[np.float64]:
        out = np.zeros(N, dtype=np.float64)
        for name in self.joints:
            index = RAKUDA_JOINT_NAMES.index(name)
            out[index] = self.bus.joint(name).gravity_at(float(q_counts[index]))
        return out


@dataclass
class LawHarness:
    """A leader loop over simulated buses (plants stepping with the clock) and a real law."""

    clock: SimClock
    bus: SimulatedDynamixelBus
    law: ControlLaw
    loop: LeaderCurrentLoop
    follower_bus: SimulatedDynamixelBus | None

    def run(self, cycles: int = 1) -> None:
        """Runs ``cycles`` cycles, moving the clock to each next deadline like the scheduler."""
        for _ in range(cycles):
            self.loop.run_once(self.clock.monotonic_ns())
            deadline = self.loop.next_deadline_ns
            if deadline is not None and deadline > self.clock.now_ns:
                self.clock.now_ns = deadline

    def position(self, name: str) -> float:
        return self.bus.joint(name).position_counts

    def goal_currents(self) -> list[Dict[str, int]]:
        item = XControlTable.GOAL_CURRENT.name
        return [values for name, values in self.bus.instruction_log if name == item]


def make_params(**overrides: Any) -> RakudaBilateralParams:
    kwargs: Dict[str, Any] = dict(current_joints=J3)
    kwargs.update(overrides)
    return RakudaBilateralParams(**kwargs)


def make_law_loop(
    law_factory: Callable[[SimulatedDynamixelBus], ControlLaw],
    *,
    params: RakudaBilateralParams,
    joints: Sequence[str] = J3,
    gravity_ma: Mapping[str, float] | None = None,
    with_follower: bool = False,
) -> LawHarness:
    """A ``LeaderCurrentLoop`` on a simulated leader (and optionally follower) with a real law.

    ``gravity_ma`` loads the named plants before the first transaction, so the
    torque-off joints have not moved yet when ``configure()`` runs.
    """
    clock = SimClock()
    bus = make_leader_bus(clock=clock, auto_step=True)
    for name in bus.motors:
        bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
    for name, value in (gravity_ma or {}).items():
        bus.joint(name).gravity_ma = value
    law = law_factory(bus)
    units = {name: CURRENT_UNIT_MA[bus.motors[name].model_name] for name in joints}
    sign = {name: 1 for name in joints}
    follower_bus: SimulatedDynamixelBus | None = None
    follower: FollowerPositionIO | None = None
    if with_follower:
        follower_bus = make_follower_bus(clock=clock, auto_step=True)
        for name in follower_bus.motors:
            follower_bus.registers(name).set(XControlTable.RETURN_DELAY_TIME, 0)
        follower_bus.torque_enabled()
        follower_bus.instruction_log.clear()
        follower = FollowerPositionIO(
            follower_bus, RAKUDA_JOINT_NAMES, params.follower_read_timeout_s
        )
    loop = LeaderCurrentLoop(
        bus,
        joints,
        params,
        law,
        units,
        sign,
        follower,
        clock=clock.monotonic_ns,
        sleep=clock.sleep,
        main_alive=lambda: True,
    )
    return LawHarness(clock, bus, law, loop, follower_bus)


def bilateral(
    params: RakudaBilateralParams, *, compensated: bool
) -> Callable[[SimulatedDynamixelBus], ControlLaw]:
    def factory(bus: SimulatedDynamixelBus) -> ControlLaw:
        gravity = SimGravity(bus, J3) if compensated else None
        return BilateralLaw(J3, params, gravity, RANGE)

    return factory


def assert_held_in_place(h: LawHarness, names: Sequence[str]) -> None:
    """Every joint in position mode, torque on, goal = present, and the plant not moving."""
    for name in names:
        regs = h.bus.registers(name)
        assert regs.operating_mode == OperatingMode.POSITION and regs.torque_enable == 1
        assert regs.goal_position == regs.present_position
    before = {name: h.position(name) for name in names}
    h.clock.advance(1.0)
    h.bus.sync_read(XControlTable.PRESENT_POSITION, list(names))  # lets the plant catch up
    for name in names:
        assert h.position(name) == pytest.approx(before[name], abs=1.0)


# --- protocol ------------------------------------------------------------------


def test_every_law_satisfies_the_control_law_protocol() -> None:
    """Static check (mypy) that the three laws are ``ControlLaw`` for the loop."""
    params = make_params()
    laws: list[ControlLaw] = [
        BilateralLaw(J3, params, None, RANGE),
        PoseHoldLaw(J3, IdentifyParams()),
        SignCheckLaw(HEAVY, 40.0, SignCheckParams()),
    ]
    assert [law.joints for law in laws] == [J3, J3, (HEAVY,)]


def test_bilateral_law_exposes_the_ranges_the_loop_checks() -> None:
    law = BilateralLaw(J3, make_params(), None, {**RANGE, "l_arm_grip": (0, 4095)})
    assert law.joint_range_counts == RANGE  # only the law's joints, as plain ints
    h = make_law_loop(bilateral(make_params(), compensated=False), params=make_params())
    assert h.loop.preflight()["range_checked"] is True
    h.bus.registers(HEAVY).set(XControlTable.PRESENT_POSITION, RANGE[HEAVY][1] + 101)
    with pytest.raises(ConnectionError, match="outside range"):
        h.loop.preflight()


# --- gravity compensation ------------------------------------------------------


class TestGravity:
    def test_compensated_leader_holds_still_then_stop_holds(self) -> None:
        params = make_params(gravity_scale=1.0)
        h = make_law_loop(
            bilateral(params, compensated=True), params=params, gravity_ma={HEAVY: GRAVITY_MA}
        )
        h.loop.configure()
        # configure() wrote the gravity term first: 30 mA is 30 raw on the XC330.
        assert h.goal_currents()[0] == {name: (30 if name == HEAVY else 0) for name in J3}
        assert h.bus.registers(HEAVY).goal_current == 30

        # The joint sags a few counts during configure()'s torque-off window
        # (15 ms of settle sleeps under gravity, then it coasts); from torque-on
        # the compensated current holds it: settled within 25 cycles, no drift.
        h.run(25)
        settled = {name: h.position(name) for name in J3}
        h.run(25)
        assert h.loop.state is LoopState.RUNNING and h.loop.fault is None
        for name in J3:
            assert h.position(name) == pytest.approx(settled[name], abs=0.5)
            assert h.position(name) == pytest.approx(HOME, abs=10.0)
        assert HOME - 10 < h.position(HEAVY) < HOME  # sagged, never lifted
        latest = h.loop.latest()
        assert latest is not None
        assert latest.current_ma[RAKUDA_JOINT_NAMES.index(HEAVY)] == pytest.approx(30.0)
        assert h.loop.control_report()["measured"]["ok_cycles"] == 50

        assert h.loop.stop() is True
        assert h.loop.state is LoopState.HELD and h.loop.hold_verified
        assert_held_in_place(h, J3)
        with pytest.raises(LoopStopped, match="stopped by operator"):
            h.loop.run_once(h.clock.monotonic_ns())

    def test_uncompensated_leader_sinks_until_the_barrier_holds_it(self) -> None:
        params = make_params(allow_uncompensated=True)
        h = make_law_loop(
            bilateral(params, compensated=False), params=params, gravity_ma={HEAVY: GRAVITY_MA}
        )
        h.loop.configure()
        assert h.goal_currents()[0] == {name: 0 for name in J3}

        h.run(100)
        assert h.loop.state is LoopState.RUNNING and h.loop.fault is None
        low, _ = RANGE[HEAVY]
        barrier_on = low + params.limit_margin_counts
        # Fell past the barrier onset and was caught inside the hard margin ...
        assert low - params.hard_margin_counts < h.position(HEAVY) < barrier_on
        # ... where the spring balances gravity (2 mA/count against 30 mA).
        assert h.position(HEAVY) == pytest.approx(barrier_on - GRAVITY_MA / 2.0, abs=3.0)
        assert abs(h.bus.joint(HEAVY).velocity_counts_per_s) < 5.0
        assert h.goal_currents()[-1][HEAVY] == pytest.approx(30, abs=3)
        for name in J3:
            if name != HEAVY:
                assert h.position(name) == pytest.approx(HOME, abs=1.0)

        assert h.loop.stop() is True
        assert_held_in_place(h, J3)

    def test_re_engage_keeps_the_barrier_current_continuous(self) -> None:
        # re_engage() is t_engaged=0, gate=0 only.  A joint resting on its
        # limit barrier must not have the barrier current cut back to the gravity
        # term (a reset would restart the rate limiter from 0 mA here).
        params = make_params(allow_uncompensated=True, current_rate_ma_per_s=500.0)
        h = make_law_loop(
            bilateral(params, compensated=False), params=params, gravity_ma={HEAVY: GRAVITY_MA}
        )
        h.loop.configure()
        h.run(150)
        before = h.goal_currents()[-1][HEAVY]
        assert before == pytest.approx(30, abs=3)  # the barrier balances gravity
        h.loop.re_engage()
        h.run(1)
        after = h.goal_currents()[-1][HEAVY]
        assert abs(after - before) <= 2
        assert h.loop.control_report()["feedback_gate"]["engaged"] is True


# --- follower pairs and feedback ----------------------------------------------


class TestFollower:
    def test_pair_snapshot_becomes_a_recording_frame(self) -> None:
        params = make_params(gravity_scale=1.0)
        h = make_law_loop(
            bilateral(params, compensated=True),
            params=params,
            gravity_ma={HEAVY: GRAVITY_MA},
            with_follower=True,
        )
        assert h.follower_bus is not None
        h.loop.configure()
        h.run(3)
        pair = h.loop.wait_pair(0.1)
        assert pair.seq == 3

        obs = pair_snapshot_to_obs(pair)
        assert isinstance(obs, RakudaArmObs)
        for field in RakudaArmObs.ARRAY_FIELDS[:6]:
            array = getattr(obs, field)
            assert array is not None and array.dtype == np.float32 and array.shape == (N,)
        assert obs.leader_time_s is None and obs.frame_time_s is None
        assert obs.leader_t_ns == pair.leader.t_end_ns
        assert obs.follower_t_ns == pair.follower.t_end_ns
        assert obs.leader_t_ns is not None and obs.follower_t_ns is not None
        assert obs.leader_t_ns <= obs.follower_t_ns
        # obs.leader is exactly the GOAL_POSITION the follower received in that cycle.
        goals = [v for item, v in h.follower_bus.instruction_log if item == "GOAL_POSITION"]
        assert goals[-1] == dict(zip(RAKUDA_JOINT_NAMES, obs.leader.astype(int).tolist()))
        # Currents come through in mA, motor sign: the sim mirrors GOAL_CURRENT.
        assert obs.leader_current is not None
        assert obs.leader_current[RAKUDA_JOINT_NAMES.index(HEAVY)] == pytest.approx(30.0)

        t0_ns = h.clock.start_ns
        frame = obs.stamped(t0_ns=t0_ns, frame_t_ns=h.clock.monotonic_ns())
        assert frame.leader_time_s is not None and frame.follower_time_s is not None
        assert frame.frame_time_s is not None
        assert 0.0 <= frame.leader_time_s <= frame.follower_time_s <= frame.frame_time_s
        stacked = RakudaArmObs.stack([frame, frame])
        assert stacked.leader.shape == (2, N) and stacked.frame_time_s is not None
        assert stacked.frame_time_s.shape == (2,)

    def test_feedback_pulls_the_leader_toward_a_lagging_follower(self) -> None:
        params = make_params(gravity_scale=1.0, feedback_kp_ma_per_count=1.0)
        h = make_law_loop(
            bilateral(params, compensated=True),
            params=params,
            gravity_ma={HEAVY: GRAVITY_MA},
            with_follower=True,
        )
        assert h.follower_bus is not None
        # The follower's HEAVY joint is stuck 100 counts below the leader.
        stuck = h.follower_bus.joint(HEAVY)
        stuck.position_counts = HOME - 100.0
        stuck.position_time_constant_s = 1e6
        h.loop.configure()
        h.run(25)  # settles after the configure() sag; not engaged, so no pull yet
        before = h.position(HEAVY)
        assert before == pytest.approx(HOME, abs=10.0)
        assert h.loop.control_report()["feedback_gate"]["gate"] == 0.0

        h.loop.re_engage()
        h.run(100)  # gate rises in a few cycles, the ramp completes after 1 s
        assert h.loop.state is LoopState.RUNNING and h.loop.fault is None
        assert h.loop.control_report()["feedback_gate"]["gate"] == pytest.approx(1.0, abs=1e-6)
        # Pulled toward the follower (e = q_F - q_L < 0 -> current below the gravity
        # term) until the error is inside the deadband, where the feedback is 0 again.
        assert min(values[HEAVY] for values in h.goal_currents()) < 30
        assert h.position(HEAVY) < before - 40
        assert h.position(HEAVY) == pytest.approx(
            HOME - 100, abs=params.feedback_deadband_counts + 5
        )
        assert h.goal_currents()[-1][HEAVY] == 30
        # The other joints see no error (the follower tracks them) and stay put.
        for name in J3:
            if name != HEAVY:
                assert h.position(name) == pytest.approx(HOME, abs=1.0)


# --- sign check ------------------------------------------------------------------


class TestSignCheck:
    def test_pulse_runs_in_the_loop_and_stops_on_displacement(self) -> None:
        params = make_params(current_joints=(HEAVY,))
        law = SignCheckLaw(HEAVY, 40.0, SignCheckParams())
        h = make_law_loop(lambda bus: law, params=params, joints=(HEAVY,))
        h.loop.configure()
        # configure() writes 0 mA (no gravity term); the pulse starts in the loop,
        # where the overshoot abort is checked every cycle:
        # 3000 mA/s * 20 ms reaches the 40 mA level on the first cycle.
        assert h.goal_currents() == [{HEAVY: 0}]
        h.run(1)
        assert h.goal_currents()[-1] == {HEAVY: 40}
        h.run(31)
        assert law.stopped and law.stop_reason == "displacement"
        assert abs(law.displacement) >= SignCheckParams().pulse_stop_counts
        assert h.goal_currents()[-1] == {HEAVY: 0}
        assert h.loop.state is LoopState.RUNNING and h.position(HEAVY) > HOME

    def test_overshoot_is_the_sign_check_overshoot_fault_and_holds(self) -> None:
        params = make_params(current_joints=(HEAVY,))
        law = SignCheckLaw(HEAVY, 40.0, SignCheckParams())
        h = make_law_loop(lambda bus: law, params=params, joints=(HEAVY,))
        h.loop.configure()
        h.run(1)
        h.bus.registers(HEAVY).set(XControlTable.PRESENT_POSITION, HOME + 100)
        h.run(1)
        fault = h.loop.fault
        assert fault is not None and fault.reason == "sign_check_overshoot"
        # +100 plus the count the plant moved under the pulse before the read.
        assert re.fullmatch(rf"{HEAVY} moved \+10[0-9] counts \(abort at ±80\)", fault.detail)
        assert fault.state_after is LoopState.FAULT and fault.hold_window_ms is not None
        assert law.abort and law.stop_reason == "abort"
        assert h.loop.state is LoopState.FAULT and h.loop.hold_verified
        assert_held_in_place(h, (HEAVY,))
        with pytest.raises(LoopStopped, match="sign_check_overshoot"):
            h.loop.run_once(h.clock.monotonic_ns())
