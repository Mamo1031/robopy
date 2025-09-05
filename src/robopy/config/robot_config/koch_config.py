from dataclasses import dataclass, field
from typing import Dict

from numpy.typing import ArrayLike

from robopy.config.visual_config.camera_config import RealsenseCameraConfig, WebCameraConfig


@dataclass
class KochSensorConfig:
    """Sensor configuration for Koch robot."""

    cameras: Dict[str, RealsenseCameraConfig | WebCameraConfig] = field(default_factory=dict)
    # 将来的に他のセンサも追加可能
    # tactile: Optional[TactileConfig] = None
    # force_torque: Optional[FTConfig] = None


@dataclass
class KochConfig:
    """Configuration class for Koch robot."""

    leader_port: str
    follower_port: str
    calibration_path: str
    sensors: KochSensorConfig = field(default_factory=KochSensorConfig)

    # Backward compatibility
    @property
    def camera(self) -> Dict[str, RealsenseCameraConfig | WebCameraConfig]:
        """Backward compatibility property for camera access."""
        return self.sensors.cameras

    @camera.setter
    def camera(self, value: Dict[str, RealsenseCameraConfig | WebCameraConfig]) -> None:
        """Backward compatibility setter for camera."""
        self.sensors.cameras = value


@dataclass
class KochSensorRetuns:
    """Sensor returns class for Koch robot."""

    CAMERA: ArrayLike | None
    TACTILE: ArrayLike | None


@dataclass
class KochObservation:
    """Observation class for Koch robot."""

    leader: ArrayLike
    follower: ArrayLike
    sensors: KochSensorRetuns
    sensors: KochSensorRetuns
    sensors: KochSensorRetuns
    sensors: KochSensorRetuns
    sensors: KochSensorRetuns
