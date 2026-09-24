"""``RakudaExpHandler._init_config`` keeps every field of the config (spec D34, T20)."""

from __future__ import annotations

from dataclasses import fields

from robopy.config.robot_config.rakuda_config import (
    RAKUDA_JOINT_NAMES,
    RakudaBilateralParams,
    RakudaConfig,
    RakudaSensorParams,
)
from robopy.config.sensor_config.params_config import CameraParams, TactileParams
from robopy.utils.exp_interface.rakuda_exp_handler import RakudaExpHandler

DEFAULT_CAMERA = CameraParams(name="main", width=640, height=480, fps=30)


def init_config(cfg: RakudaConfig) -> RakudaConfig:
    """Calls the method on a bare instance: no robot, no port, no sensors."""
    handler = object.__new__(RakudaExpHandler)
    return handler._init_config(cfg)


def full_config(**kwargs: object) -> RakudaConfig:
    return RakudaConfig(
        leader_port="auto",
        follower_port="/dev/ttyUSB0",
        slow_mode=True,
        leader_torque_enabled=["head_yaw"],
        follower_torque_enabled=list(RAKUDA_JOINT_NAMES),
        bilateral=RakudaBilateralParams(gravity_scale=0.8),
        hold_on_disconnect=False,
        **kwargs,  # type: ignore[arg-type]
    )


def test_default_camera_is_added_and_every_other_field_kept() -> None:
    cfg = full_config()

    out = init_config(cfg)

    assert out is not cfg
    assert out.sensors is not None
    assert out.sensors.cameras == [DEFAULT_CAMERA]
    assert out.sensors.tactile == []
    assert out.sensors.audio == []
    for f in fields(RakudaConfig):
        if f.name != "sensors":
            assert getattr(out, f.name) == getattr(cfg, f.name), f.name
    assert out.bilateral is cfg.bilateral
    assert out.effective_hold_on_disconnect is False


def test_input_is_not_mutated() -> None:
    cfg = full_config()

    init_config(cfg)

    assert cfg.sensors is None
    assert cfg.slow_mode is True


def test_config_with_cameras_is_returned_as_is() -> None:
    sensors = RakudaSensorParams(cameras=[CameraParams("side", 320, 240, 15)])
    cfg = full_config(sensors=sensors)

    out = init_config(cfg)

    assert out is cfg
    assert out.sensors is sensors
    assert sensors.cameras == [CameraParams("side", 320, 240, 15)]


def test_sensors_without_cameras_get_the_default_camera_without_mutation() -> None:
    tactile = [TactileParams(serial_num="D20542", name="left")]
    sensors = RakudaSensorParams(tactile=tactile)
    cfg = full_config(sensors=sensors)

    out = init_config(cfg)

    assert out is not cfg
    assert out.sensors is not None and out.sensors is not sensors
    assert out.sensors.cameras == [DEFAULT_CAMERA]
    assert out.sensors.tactile == tactile
    assert out.bilateral is cfg.bilateral
    assert sensors.cameras == []  # the caller's sensors are untouched
