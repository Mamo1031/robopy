import logging

from robopy.config.robot_config import RAKUDA_CONTROLTABLE_VALUES, RakudaConfig
from robopy.config.robot_config.rakuda_config import resolve_torque_policy
from robopy.motor.dynamixel_bus import DynamixelMotor

from .rakuda_arm import BusFactory, ConnectState, RakudaArm

logger = logging.getLogger(__name__)


class RakudaFollower(RakudaArm):
    """Class representing the follower arm of the Rakuda robotic system."""

    SIDE = "follower"
    GRIP_CURRENT_LIMIT = RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_CURRENT_LIMIT
    GRIP_GOAL_CURRENT = RAKUDA_CONTROLTABLE_VALUES.FOLLOWER_GRIP_GOAL_CURRENT

    def __init__(self, cfg: RakudaConfig, bus_factory: BusFactory | None = None):
        super().__init__(cfg, cfg.follower_port, bus_factory)

    def _create_motors(self) -> dict[str, DynamixelMotor]:
        """Create motor configuration for the follower arm using xm430-w350 motors."""
        return {
            # head
            "torso_yaw": DynamixelMotor(27, "torso_yaw", "xm540-w270"),
            "head_yaw": DynamixelMotor(28, "head_yaw", "xm430-w350"),
            "head_pitch": DynamixelMotor(29, "head_pitch", "xm430-w350"),
            # right
            "r_arm_sh_pitch1": DynamixelMotor(1, "r_arm_sh_pitch1", "xm540-w270"),
            "r_arm_sh_roll": DynamixelMotor(3, "r_arm_sh_roll", "xm540-w270"),
            "r_arm_sh_pitch2": DynamixelMotor(5, "r_arm_sh_pitch2", "xm430-w350"),
            "r_arm_el_yaw": DynamixelMotor(7, "r_arm_el_yaw", "xm430-w350"),
            "r_arm_wr_roll": DynamixelMotor(9, "r_arm_wr_roll", "xm430-w350"),
            "r_arm_wr_yaw": DynamixelMotor(11, "r_arm_wr_yaw", "xm430-w350"),
            "r_arm_grip": DynamixelMotor(31, "r_arm_grip", "xm430-w350"),
            # left
            "l_arm_sh_pitch1": DynamixelMotor(2, "l_arm_sh_pitch1", "xm540-w270"),
            "l_arm_sh_roll": DynamixelMotor(4, "l_arm_sh_roll", "xm540-w270"),
            "l_arm_sh_pitch2": DynamixelMotor(6, "l_arm_sh_pitch2", "xm430-w350"),
            "l_arm_el_yaw": DynamixelMotor(8, "l_arm_el_yaw", "xm430-w350"),
            "l_arm_wr_roll": DynamixelMotor(10, "l_arm_wr_roll", "xm430-w350"),
            "l_arm_wr_yaw": DynamixelMotor(12, "l_arm_wr_yaw", "xm430-w350"),
            "l_arm_grip": DynamixelMotor(30, "l_arm_grip", "xm430-w350"),
        }

    def _apply_torque_policy(self, state: ConnectState) -> None:
        """Follower torque policy (spec D15/D35), the same in both modes.

        Motors that are on but not wanted are switched off, wanted motors that
        are off are switched on; a wanted motor that is already on is never
        touched (the gripper EEPROM step ran before this).
        """
        self._switch_torque(state, resolve_torque_policy(self.config).follower)
