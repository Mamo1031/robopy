import math
from dataclasses import dataclass, field, fields, replace
from typing import ClassVar, Dict, List, Mapping, Sequence, Tuple

import numpy as np
from numpy.typing import NDArray

from robopy.config.sensor_config.params_config import AudioParams, CameraParams, TactileParams
from robopy.config.sensor_config.visual_config.camera_config import RealsenseCameraConfig


@dataclass
class RakudaConfig:
    """Configuration class for Rakuda robot."""

    leader_port: str
    follower_port: str
    sensors: "RakudaSensorParams | None" = field(default=None)
    slow_mode: bool = False
    # Torque enable policy (joint names). If None, defaults preserve current behavior.
    # - leader_torque_enabled: default is grippers only
    # - follower_torque_enabled: default is all joints
    leader_torque_enabled: List[str] | None = None
    follower_torque_enabled: List[str] | None = None
    # Bilateral (leader current control) parameters. None -> conventional position
    # teleoperation; the loop is enabled only by passing this in code.
    bilateral: "RakudaBilateralParams | None" = None
    # Keep both arms torque-on at their current position on disconnect. None -> True
    # in bilateral mode, False in conventional mode.
    hold_on_disconnect: bool | None = None

    @property
    def effective_hold_on_disconnect(self) -> bool:
        """``hold_on_disconnect`` with the mode-dependent default resolved."""
        if self.hold_on_disconnect is not None:
            return self.hold_on_disconnect
        return self.bilateral is not None


@dataclass
class RakudaSensorParams:
    cameras: List[CameraParams] = field(default_factory=list)
    tactile: List[TactileParams] = field(default_factory=list)
    audio: List[AudioParams] = field(default_factory=list)


@dataclass
class RakudaSensorConfigs:
    cameras: List[RealsenseCameraConfig]
    tactile: List[TactileParams]
    audio: List[AudioParams]


def _seconds_since(t_ns: int | None, t0_ns: int) -> NDArray[np.float32] | None:
    """``(t_ns - t0_ns) / 1e9`` as a 0-d float32 array, or None when ``t_ns`` is None."""
    if t_ns is None:
        return None
    return np.asarray((t_ns - t0_ns) / 1e9, dtype=np.float32)


@dataclass
class RakudaArmObs:
    """Observation of both arms: one frame, or a whole recording after :meth:`stack`.

    Positions are encoder counts, velocities raw velocity counts (0.229 rpm per
    count, joint convention) and currents mA in the motor's own sign
    convention (``PRESENT_CURRENT`` raw sign).  Per frame the
    arrays are ``(17,)`` and the times 0-d; after :meth:`stack` they are
    ``(N, 17)`` and ``(N,)``.  ``*_time_s`` are seconds since the
    recording's ``t0``; consumers may only assume that
    ``frame_time_s`` is non-negative and non-decreasing and that
    ``leader_time_s``/``follower_time_s <= frame_time_s`` (the first
    ``leader_time_s`` can be slightly negative).

    Only ``leader`` and ``follower`` are required, so the pre-v2 keyword and
    positional constructions keep working.  ``leader_t_ns``/``follower_t_ns``
    are the raw ``time.monotonic_ns`` stamps of the two bus reads; they only
    feed :meth:`stamped` and are never written to HDF5.
    """

    #: Fields that hold arrays, in HDF5 dataset order.
    ARRAY_FIELDS: ClassVar[Tuple[str, ...]] = (
        "leader",
        "follower",
        "leader_velocity",
        "follower_velocity",
        "leader_current",
        "follower_current",
        "leader_time_s",
        "follower_time_s",
        "frame_time_s",
    )

    leader: NDArray[np.float32]
    follower: NDArray[np.float32]
    leader_velocity: NDArray[np.float32] | None = None
    follower_velocity: NDArray[np.float32] | None = None
    leader_current: NDArray[np.float32] | None = None
    follower_current: NDArray[np.float32] | None = None
    leader_time_s: NDArray[np.float32] | None = None
    follower_time_s: NDArray[np.float32] | None = None
    frame_time_s: NDArray[np.float32] | None = None
    leader_t_ns: int | None = None
    follower_t_ns: int | None = None

    @classmethod
    def from_states(cls, leader: "RakudaArmState", follower: "RakudaArmState") -> "RakudaArmObs":
        """One frame from a leader and a follower state read.

        Positions and velocities are widened to float32; ``current_ma`` is
        taken as is; the ``t_end_ns`` stamps become ``*_t_ns``.
        """
        return cls(
            leader=leader.position.astype(np.float32),
            follower=follower.position.astype(np.float32),
            leader_velocity=leader.velocity.astype(np.float32),
            follower_velocity=follower.velocity.astype(np.float32),
            leader_current=leader.current_ma,
            follower_current=follower.current_ma,
            leader_t_ns=leader.t_end_ns,
            follower_t_ns=follower.t_end_ns,
        )

    def stamped(self, *, t0_ns: int, frame_t_ns: int) -> "RakudaArmObs":
        """A copy with the three ``*_time_s`` fields set relative to ``t0_ns``.

        ``self`` is left untouched.  A read stamp that is None keeps its
        ``*_time_s`` None (the leader of ``record_with_fixed_leader``).

        Args:
            t0_ns: ``time.monotonic_ns()`` taken once at the start of the recording.
            frame_t_ns: ``time.monotonic_ns()`` at the moment the frame is committed.
        """
        return replace(
            self,
            leader_time_s=_seconds_since(self.leader_t_ns, t0_ns),
            follower_time_s=_seconds_since(self.follower_t_ns, t0_ns),
            frame_time_s=_seconds_since(frame_t_ns, t0_ns),
        )

    @staticmethod
    def stack(frames: Sequence["RakudaArmObs"]) -> "RakudaArmObs":
        """Stacks per-frame observations along a new first axis (``np.stack``).

        A field that is None in any frame is None in the result; the
        ``*_t_ns`` stamps are always None.  The frames are not modified.

        Raises:
            ValueError: If ``frames`` is empty.
        """
        if not frames:
            raise ValueError("RakudaArmObs.stack needs at least one frame")

        def column(name: str) -> NDArray[np.float32] | None:
            values = [getattr(frame, name) for frame in frames]
            if any(value is None for value in values):
                return None
            return np.stack(values)

        return RakudaArmObs(
            leader=np.stack([frame.leader for frame in frames]),
            follower=np.stack([frame.follower for frame in frames]),
            leader_velocity=column("leader_velocity"),
            follower_velocity=column("follower_velocity"),
            leader_current=column("leader_current"),
            follower_current=column("follower_current"),
            leader_time_s=column("leader_time_s"),
            follower_time_s=column("follower_time_s"),
            frame_time_s=column("frame_time_s"),
        )


@dataclass
class RakudaSensorObs:
    cameras: Dict[str, NDArray[np.float32] | None]
    tactile: Dict[str, NDArray[np.float32] | None]
    audio: Dict[str, NDArray[np.float32] | None]


@dataclass
class RakudaObs:
    """
    Overall observation structure for Rakuda robot.
    arms: Observations from the robot arms (leader and follower).
    sensors: Observations from the sensors (cameras, tactile and audio).
    """

    arms: RakudaArmObs
    sensors: RakudaSensorObs | None


RAKUDA_MOTOR_MAPPING: Dict[str, str] = {
    "torso_yaw": "torso_yaw",
    "head_yaw": "head_yaw",
    "head_pitch": "head_pitch",
    "r_arm_sh_pitch1": "r_arm_sh_pitch1",
    "r_arm_sh_roll": "r_arm_sh_roll",
    "r_arm_sh_pitch2": "r_arm_sh_pitch2",
    "r_arm_el_yaw": "r_arm_el_yaw",
    "r_arm_wr_roll": "r_arm_wr_roll",
    "r_arm_wr_yaw": "r_arm_wr_yaw",
    "r_arm_grip": "r_arm_grip",
    "l_arm_sh_pitch1": "l_arm_sh_pitch1",
    "l_arm_sh_roll": "l_arm_sh_roll",
    "l_arm_sh_pitch2": "l_arm_sh_pitch2",
    "l_arm_el_yaw": "l_arm_el_yaw",
    "l_arm_wr_roll": "l_arm_wr_roll",
    "l_arm_wr_yaw": "l_arm_wr_yaw",
    "l_arm_grip": "l_arm_grip",
}


# Canonical Rakuda joint names (used for validation and config templates).
RAKUDA_JOINT_NAMES: Tuple[str, ...] = tuple(RAKUDA_MOTOR_MAPPING.keys())

#: Port value meaning "find the bus by scanning".
PORT_AUTO = "auto"

RAKUDA_GRIPPER_JOINT_NAMES: Tuple[str, ...] = ("l_arm_grip", "r_arm_grip")
RAKUDA_HEAD_JOINT_NAMES: Tuple[str, ...] = ("head_yaw", "head_pitch")
#: The 12 arm joints (both arms, shoulder to wrist), in ``RAKUDA_JOINT_NAMES`` order.
RAKUDA_ARM_JOINT_NAMES: Tuple[str, ...] = tuple(
    name
    for name in RAKUDA_JOINT_NAMES
    if name != "torso_yaw"
    and name not in RAKUDA_GRIPPER_JOINT_NAMES
    and name not in RAKUDA_HEAD_JOINT_NAMES
)
#: Joints the bilateral loop may drive in current mode: the 12 arm joints and ``torso_yaw``.
RAKUDA_CURRENT_CAPABLE_JOINTS: Tuple[str, ...] = tuple(
    name for name in RAKUDA_JOINT_NAMES if name == "torso_yaw" or name in RAKUDA_ARM_JOINT_NAMES
)


#: GOAL_POSITION sent to the leader grippers while they hold.
LEADER_GRIP_HOLD_POSITION: int = 2400


def _check_joint_names(names: Sequence[str] | None, *, field_name: str) -> None:
    """Raises ``ValueError`` when ``names`` contains a name outside ``RAKUDA_JOINT_NAMES``."""
    if names is None:
        return
    unknown = sorted(set(names) - set(RAKUDA_JOINT_NAMES))
    if unknown:
        raise ValueError(
            f"Unknown joint name(s) in {field_name}: {unknown}. "
            f"Allowed: {', '.join(RAKUDA_JOINT_NAMES)}"
        )


def _require_number(field_name: str, value: object) -> float:
    """Returns ``value`` as a float, rejecting bools, non-numbers, NaN and infinities."""
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        raise ValueError(f"{field_name} must be a number, got {value!r}")
    if not math.isfinite(value):
        raise ValueError(f"{field_name} must be finite, got {value!r}")
    return float(value)


def _require_int(field_name: str, value: object) -> int:
    if isinstance(value, bool) or not isinstance(value, int):
        raise ValueError(f"{field_name} must be an integer, got {value!r}")
    return value


@dataclass(frozen=True)
class BusHealthThresholds:
    """Temperature / input-voltage limits for one bus.

    ``warn_temperature_c`` logs a warning, ``max_temperature_c`` is a fault; the
    voltage window is a fault on either side.
    """

    warn_temperature_c: float
    max_temperature_c: float
    min_voltage_v: float
    max_voltage_v: float

    def __post_init__(self) -> None:
        for name in ("warn_temperature_c", "max_temperature_c", "min_voltage_v", "max_voltage_v"):
            _require_number(f"BusHealthThresholds.{name}", getattr(self, name))
        if self.warn_temperature_c > self.max_temperature_c:
            raise ValueError(
                "BusHealthThresholds.warn_temperature_c must not exceed max_temperature_c "
                f"({self.warn_temperature_c} > {self.max_temperature_c})"
            )
        if self.min_voltage_v >= self.max_voltage_v:
            raise ValueError(
                "BusHealthThresholds.min_voltage_v must be below max_voltage_v "
                f"({self.min_voltage_v} >= {self.max_voltage_v})"
            )


#: Leader (XC330-T288, 12 V): idle motors self-heat to 40-60 °C, so warn above that
#: and hold at 66, a few degrees under the motors' own TEMPERATURE_LIMIT of 70 at
#: which they cut torque by themselves.
LEADER_HEALTH_DEFAULT = BusHealthThresholds(
    warn_temperature_c=62.0, max_temperature_c=66.0, min_voltage_v=9.0, max_voltage_v=13.5
)
#: Follower (XM540 / XM430, 12 V). Configurable; the bilateral loop does not check it.
FOLLOWER_HEALTH_DEFAULT = BusHealthThresholds(
    warn_temperature_c=60.0, max_temperature_c=70.0, min_voltage_v=10.0, max_voltage_v=15.0
)

#: The four gains of ``RakudaBilateralParams`` that must be non-negative.
_BILATERAL_GAIN_FIELDS: Tuple[str, ...] = (
    "limit_kp_ma_per_count",
    "limit_kd_ma_per_vcount",
    "feedback_kp_ma_per_count",
    "feedback_kd_ma_per_vcount",
)


@dataclass(frozen=True)
class RakudaBilateralParams:
    """Parameters of the leader current-control loop.

    Units: positions in encoder counts, velocities in velocity counts
    (0.229 rpm/LSB), currents in mA, times in seconds.
    Every rule in ``__post_init__`` raises ``ValueError``.
    """

    #: Joints driven in current mode; subset of ``RAKUDA_CURRENT_CAPABLE_JOINTS``.
    #: ``torso_yaw`` is opt-in.
    current_joints: Tuple[str, ...] = RAKUDA_ARM_JOINT_NAMES
    #: Loop rate, divider and read timeouts are bench-measured defaults (RETURN_DELAY_TIME=0:
    #: read_state_block(17) p99 ~5.1 ms; read timeout = max(1.5 * p99, p99 + 3 ms) -> 8 ms).
    control_hz: float = 50.0
    #: Follower I/O runs every ``follower_divider``-th cycle.
    follower_divider: int = 1
    #: Hard upper bound of one leader read attempt.
    read_timeout_s: float = 0.008
    follower_read_timeout_s: float = 0.008
    follower_lost_cycles: int = 3
    snapshot_stale_s: float = 0.10
    write_fault_s: float = 0.05
    diagnostics_period_s: float = 1.0
    velocity_filter_hz: float | None = 20.0
    #: Gravity-compensation scale: one float, or one per joint.
    gravity_scale: float | Dict[str, float] = 0.9
    gravity_clamp_factor: float = 1.3
    allow_uncompensated: bool = False
    allow_unvalidated_gravity: bool = False
    limit_kp_ma_per_count: float = 2.0
    limit_kd_ma_per_vcount: float = 1.0
    limit_margin_counts: int = 57
    limit_max_ma: float = 300.0
    hard_margin_counts: int = 100
    feedback_kp_ma_per_count: float = 0.0
    feedback_kd_ma_per_vcount: float = 0.0
    feedback_deadband_counts: int = 10
    feedback_max_ma: float = 150.0
    feedback_gate_alpha: float = 0.35
    follower_stale_s: float = 0.2
    ramp_s: float = 1.0
    #: CURRENT_LIMIT register value written once (XC330 default 910).
    current_limit_ma: float = 500.0
    #: Software clamp of the commanded current.
    current_max_ma: float = 400.0
    current_rate_ma_per_s: float = 3000.0
    runaway_velocity_counts: int = 280
    runaway_s: float = 0.04
    saturation_fault_s: float = 2.0
    overrun_warn_factor: float = 3.0
    max_alignment_counts: int = 300
    align_s: float = 2.0
    hold_read_retries: int = 3
    #: Oldest snapshot ``_hold()`` may fall back to; None -> ``3 / control_hz`` capped at 0.1.
    hold_max_snapshot_age_s: float | None = None
    hold_max_jump_counts: int = 200
    hold_profile_velocity: int = 40
    leader_health: BusHealthThresholds = LEADER_HEALTH_DEFAULT
    follower_health: BusHealthThresholds = FOLLOWER_HEALTH_DEFAULT
    #: Relative to the current working directory.
    gravity_model_path: str = ".robopy/rakuda/leader_gravity.json"

    def __post_init__(self) -> None:
        self._check_field_types()
        self._check_current_joints()
        self._check_gravity_scale()

        hz = self.control_hz
        if not 20 <= hz <= 250:
            raise ValueError(f"control_hz must be within [20, 250], got {hz}")
        period = 1.0 / hz
        if not 1 <= self.follower_divider <= 10:
            raise ValueError(
                f"follower_divider must be within [1, 10], got {self.follower_divider}"
            )
        if hz / self.follower_divider < 10:
            raise ValueError(
                "control_hz / follower_divider must be >= 10 Hz, got "
                f"{hz} / {self.follower_divider} = {hz / self.follower_divider:.2f}"
            )
        for name in ("read_timeout_s", "follower_read_timeout_s"):
            timeout = getattr(self, name)
            if not 0.004 <= timeout <= 2 * period:
                raise ValueError(
                    f"{name} must be within [0.004, 2/control_hz = {2 * period:.4f}] s, "
                    f"got {timeout}"
                )
        if self.snapshot_stale_s < 2 * self.read_timeout_s:
            raise ValueError(
                f"snapshot_stale_s must be >= 2 * read_timeout_s = {2 * self.read_timeout_s}, "
                f"got {self.snapshot_stale_s}"
            )
        if self.runaway_s < 2 * period:
            raise ValueError(
                f"runaway_s must be >= 2/control_hz = {2 * period:.4f} s, got {self.runaway_s}"
            )
        if self.current_max_ma > self.current_limit_ma:
            raise ValueError(
                f"current_max_ma ({self.current_max_ma}) must not exceed "
                f"current_limit_ma ({self.current_limit_ma})"
            )
        if self.hold_max_snapshot_age_s is not None:
            age = _require_number("hold_max_snapshot_age_s", self.hold_max_snapshot_age_s)
            if not 0 <= age <= 0.1:
                raise ValueError(f"hold_max_snapshot_age_s must be within [0, 0.1] s, got {age}")
        if self.hold_profile_velocity <= 0:
            raise ValueError(f"hold_profile_velocity must be > 0, got {self.hold_profile_velocity}")
        if self.velocity_filter_hz is not None:
            cutoff = _require_number("velocity_filter_hz", self.velocity_filter_hz)
            if cutoff <= 0:
                raise ValueError(f"velocity_filter_hz must be > 0 or None, got {cutoff}")
        for name in _BILATERAL_GAIN_FIELDS:
            if getattr(self, name) < 0:
                raise ValueError(f"{name} must be non-negative, got {getattr(self, name)}")

    def _check_field_types(self) -> None:
        """Rejects YAML scalars of the wrong kind (a bool is not a number here)."""
        for f in fields(self):
            value = getattr(self, f.name)
            if f.type is bool:
                if not isinstance(value, bool):
                    raise ValueError(f"{f.name} must be true or false, got {value!r}")
            elif f.type is int:
                _require_int(f.name, value)
            elif f.type is float:
                _require_number(f.name, value)
            elif f.type is BusHealthThresholds and not isinstance(value, BusHealthThresholds):
                raise ValueError(f"{f.name} must be a BusHealthThresholds, got {value!r}")
        if not isinstance(self.gravity_model_path, str):
            raise ValueError(f"gravity_model_path must be a str, got {self.gravity_model_path!r}")

    def _check_current_joints(self) -> None:
        if isinstance(self.current_joints, str) or not isinstance(self.current_joints, Sequence):
            raise ValueError(
                f"current_joints must be a sequence of joint names, got {self.current_joints!r}"
            )
        joints = tuple(self.current_joints)
        object.__setattr__(self, "current_joints", joints)
        if not joints:
            raise ValueError("current_joints must not be empty")
        if len(set(joints)) != len(joints):
            raise ValueError(f"current_joints contains duplicates: {list(joints)}")
        unsupported = [name for name in joints if name not in RAKUDA_CURRENT_CAPABLE_JOINTS]
        if unsupported:
            raise ValueError(
                f"current_joints must be a subset of RAKUDA_CURRENT_CAPABLE_JOINTS; "
                f"not allowed: {unsupported}. Allowed: {', '.join(RAKUDA_CURRENT_CAPABLE_JOINTS)}"
            )

    def _check_gravity_scale(self) -> None:
        scale = self.gravity_scale
        if isinstance(scale, Mapping):
            unknown = [name for name in scale if name not in RAKUDA_CURRENT_CAPABLE_JOINTS]
            if unknown:
                raise ValueError(
                    f"gravity_scale has keys that are not current-capable joints: {unknown}"
                )
            for name, value in scale.items():
                if _require_number(f"gravity_scale[{name!r}]", value) <= 0:
                    raise ValueError(f"gravity_scale[{name!r}] must be > 0, got {value}")
            return
        if _require_number("gravity_scale", scale) <= 0:
            raise ValueError(f"gravity_scale must be > 0, got {scale}")

    @property
    def effective_hold_max_snapshot_age_s(self) -> float:
        """``hold_max_snapshot_age_s``, or ``3 / control_hz`` capped at 0.1 s when None."""
        if self.hold_max_snapshot_age_s is not None:
            return self.hold_max_snapshot_age_s
        return min(3.0 / self.control_hz, 0.1)


@dataclass(frozen=True)
class RakudaArmState:
    """One synchronous read of every motor on a bus.

    The arrays are in ``names`` order. ``current_ma`` keeps the motor's raw
    sign. ``t_start_ns`` / ``t_end_ns`` bracket the bus transaction
    on ``time.monotonic_ns``; ``seq`` increases by one per read.
    """

    names: Tuple[str, ...]
    position: NDArray[np.int32]
    velocity: NDArray[np.int32]
    current_ma: NDArray[np.float32]
    t_start_ns: int
    t_end_ns: int
    seq: int

    def __post_init__(self) -> None:
        object.__setattr__(self, "names", tuple(self.names))
        expected = (len(self.names),)
        for attr in ("position", "velocity", "current_ma"):
            shape = getattr(getattr(self, attr), "shape", None)
            if shape != expected:
                raise ValueError(f"RakudaArmState.{attr} must have shape {expected}, got {shape}")


@dataclass(frozen=True)
class RakudaTorquePolicy:
    """Which joints each arm torque-enables at connect time.

    Attributes:
        leader: Leader joints held in position mode by ``RakudaLeader.connect()``;
            always contains the grippers. In bilateral mode the current-controlled
            joints are excluded because the loop owns their TORQUE_ENABLE.
        follower: Follower joints to torque-enable.
        dropped_from_leader: The bilateral joints that were removed from
            ``leader`` (sorted); empty in conventional mode. Callers log them once.
    """

    leader: frozenset[str]
    follower: frozenset[str]
    dropped_from_leader: Tuple[str, ...]


def resolve_torque_policy(cfg: RakudaConfig) -> RakudaTorquePolicy:
    """Resolves the torque defaults of ``cfg`` and checks them against bilateral mode.

    Rules:

    1. ``leader = (cfg.leader_torque_enabled or ()) ∪ grippers`` (grippers always).
    2. ``follower`` = all joints when ``cfg.follower_torque_enabled`` is None,
       else exactly that list.
    3. Bilateral only: every ``current_joints`` entry must be in ``follower``
       (``ValueError`` naming the missing joints), and ``current_joints`` are
       removed from ``leader`` and reported as ``dropped_from_leader``.
    4. Conventional mode drops nothing.

    Raises:
        ValueError: On an unknown joint name or a follower list that does not
            cover the bilateral joints.
    """
    _check_joint_names(cfg.leader_torque_enabled, field_name="leader_torque_enabled")
    _check_joint_names(cfg.follower_torque_enabled, field_name="follower_torque_enabled")

    leader = set(cfg.leader_torque_enabled or ()) | set(RAKUDA_GRIPPER_JOINT_NAMES)
    follower = (
        set(RAKUDA_JOINT_NAMES)
        if cfg.follower_torque_enabled is None
        else set(cfg.follower_torque_enabled)
    )
    dropped: Tuple[str, ...] = ()
    if cfg.bilateral is not None:
        current = set(cfg.bilateral.current_joints)
        missing = sorted(current - follower)
        if missing:
            raise ValueError(
                "bilateral current_joints must be torque-enabled on the follower; "
                f"missing: {missing}"
            )
        dropped = tuple(sorted(leader & current))
        leader -= current
    return RakudaTorquePolicy(
        leader=frozenset(leader), follower=frozenset(follower), dropped_from_leader=dropped
    )


@dataclass
class RAKUDA_CONTROLTABLE_VALUES:
    GRIP_OPEN_POSITION: int = 2500  # Open position for gripper
    GRIP_PID: Tuple[int, int, int] = (128, 32, 64)  # PID values for gripper control
    GRIP_PID_SLOW: Tuple[int, int, int] = (
        512,
        64,
        1024,
    )  # PID values for gripper control in slow mode
    FOLLOWER_GRIP_GOAL_CURRENT: int = 128  # mA, goal current for follower gripper
    FOLLOWER_GRIP_CURRENT_LIMIT: int = 128  # mA, current limit for follower gripper

    LEADER_GRIP_GOAL_CURRENT: int = 30  # mA, goal current for leader gripper
    LEADER_GRIP_CURRENT_LIMIT: int = 30  # mA, current limit for leader gripper

    GRIP_MAX_POSITION: int = 2600  # Maximum position for gripper
    CURRENT_BASED_OPERATING_MODE: int = (
        5  # Operating mode for gripper motors (Current-based position control)
    )
    POSITION_CONTROL_MODE: int = 3  # Operating mode for non-gripper motors (Position control)
