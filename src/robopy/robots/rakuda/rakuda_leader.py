import logging

from robopy.config.robot_config import RAKUDA_CONTROLTABLE_VALUES, RakudaConfig
from robopy.config.robot_config.rakuda_config import (
    LEADER_GRIP_HOLD_POSITION,
    RAKUDA_GRIPPER_JOINT_NAMES,
    resolve_torque_policy,
)
from robopy.motor.dynamixel_bus import DynamixelMotor
from robopy.motor.dynamixel_control_table import XControlTable

from .rakuda_arm import BusFactory, ConnectState, RakudaArm

logger = logging.getLogger(__name__)


class RakudaLeader(RakudaArm):
    """Class representing the leader arm of the Rakuda robotic system."""

    SIDE = "leader"
    GRIP_CURRENT_LIMIT = RAKUDA_CONTROLTABLE_VALUES.LEADER_GRIP_CURRENT_LIMIT
    GRIP_GOAL_CURRENT = RAKUDA_CONTROLTABLE_VALUES.LEADER_GRIP_GOAL_CURRENT

    def __init__(self, cfg: RakudaConfig, bus_factory: BusFactory | None = None):
        super().__init__(cfg, cfg.leader_port, bus_factory)

    def _create_motors(self) -> dict[str, DynamixelMotor]:
        """Create motor configuration for the leader arm using xc330-t288 motors."""
        return {
            # head
            "torso_yaw": DynamixelMotor(27, "torso_yaw", "xm430-w350"),  # different model
            "head_yaw": DynamixelMotor(28, "head_yaw", "xc330-t288"),
            "head_pitch": DynamixelMotor(29, "head_pitch", "xc330-t288"),
            # right
            "r_arm_sh_pitch1": DynamixelMotor(1, "r_arm_sh_pitch1", "xc330-t288"),
            "r_arm_sh_roll": DynamixelMotor(3, "r_arm_sh_roll", "xc330-t288"),
            "r_arm_sh_pitch2": DynamixelMotor(5, "r_arm_sh_pitch2", "xc330-t288"),
            "r_arm_el_yaw": DynamixelMotor(7, "r_arm_el_yaw", "xc330-t288"),
            "r_arm_wr_roll": DynamixelMotor(9, "r_arm_wr_roll", "xc330-t288"),
            "r_arm_wr_yaw": DynamixelMotor(11, "r_arm_wr_yaw", "xc330-t288"),
            "r_arm_grip": DynamixelMotor(31, "r_arm_grip", "xc330-t288"),
            # left
            "l_arm_sh_pitch1": DynamixelMotor(2, "l_arm_sh_pitch1", "xc330-t288"),
            "l_arm_sh_roll": DynamixelMotor(4, "l_arm_sh_roll", "xc330-t288"),
            "l_arm_sh_pitch2": DynamixelMotor(6, "l_arm_sh_pitch2", "xc330-t288"),
            "l_arm_el_yaw": DynamixelMotor(8, "l_arm_el_yaw", "xc330-t288"),
            "l_arm_wr_roll": DynamixelMotor(10, "l_arm_wr_roll", "xc330-t288"),
            "l_arm_wr_yaw": DynamixelMotor(12, "l_arm_wr_yaw", "xc330-t288"),
            "l_arm_grip": DynamixelMotor(30, "l_arm_grip", "xc330-t288"),
        }

    def _apply_torque_policy(self, state: ConnectState) -> None:
        """Leader torque policy.

        Conventional mode: a motor that is on but not wanted is a previous
        session's hold and connecting is refused (``ConnectionError``), so the
        arm is never dropped silently. Bilateral mode: the current-controlled
        joints are only logged (the loop owns their ``TORQUE_ENABLE``) and a
        ``leader_torque_enabled`` entry among them is warned about once.
        Grippers are always held at ``LEADER_GRIP_HOLD_POSITION``.
        """
        policy = resolve_torque_policy(self.config)
        want = policy.leader
        bilateral = self.config.bilateral
        if bilateral is None:
            held = [n for n in self.motor_names if n in state.torque_on and n not in want]
            if held:
                raise self._held_by_previous_session_error(held)
            current_joints: tuple[str, ...] = ()
        else:
            current_joints = bilateral.current_joints
            logger.info(
                "leader: bilateral joints left as found (mode/torque): %s",
                {n: (state.mode[n], state.torque[n]) for n in current_joints},
            )
            if policy.dropped_from_leader:
                logger.warning(
                    "leader_torque_enabled names the bilateral joints %s; ignored, the current "
                    "loop owns their TORQUE_ENABLE.",
                    list(policy.dropped_from_leader),
                )
        self._switch_torque(state, want, untouched=current_joints)

        # Fix leader initial gripper pose.
        self._motors.sync_write(
            XControlTable.GOAL_POSITION,
            {name: LEADER_GRIP_HOLD_POSITION for name in RAKUDA_GRIPPER_JOINT_NAMES},
        )
