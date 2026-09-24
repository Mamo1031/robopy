"""Tests for the new ``.robopy/rakuda/config.yaml`` sections (spec §7.3, D10, D17, D18)."""

from __future__ import annotations

import logging
from dataclasses import replace
from pathlib import Path
from typing import Any

import pytest

from robopy.config.dotrobopy import (
    apply_rakuda_dotconfig,
    get_rakuda_yaml_path,
    update_rakuda_yaml_torque_enabled,
)
from robopy.config.robot_config.rakuda_config import (
    LEADER_HEALTH_DEFAULT,
    PORT_AUTO,
    RAKUDA_ARM_JOINT_NAMES,
    RAKUDA_JOINT_NAMES,
    BusHealthThresholds,
    RakudaBilateralParams,
    RakudaConfig,
)

LEADER = "/dev/ttyUSB1"
FOLLOWER = "/dev/ttyUSB0"


@pytest.fixture
def yaml_path(tmp_path: Path, monkeypatch: pytest.MonkeyPatch) -> Path:
    """Points the dotconfig at ``tmp_path`` (like tests/test_rakuda_torque_config.py)."""
    monkeypatch.chdir(tmp_path)
    return get_rakuda_yaml_path()


def write_yaml(yaml_path: Path, text: str) -> None:
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    yaml_path.write_text(text.lstrip("\n"), encoding="utf-8")


def conventional() -> RakudaConfig:
    return RakudaConfig(leader_port=LEADER, follower_port=FOLLOWER)


def bilateral(**kwargs: Any) -> RakudaConfig:
    return RakudaConfig(
        leader_port=LEADER, follower_port=FOLLOWER, bilateral=RakudaBilateralParams(**kwargs)
    )


# --- default template ---------------------------------------------------------


def test_default_template_round_trips_to_no_overrides(yaml_path: Path) -> None:
    cfg = bilateral()
    out = apply_rakuda_dotconfig(cfg)
    assert yaml_path.exists()
    assert out is cfg
    assert out.leader_port == LEADER
    assert out.follower_port == FOLLOWER
    assert out.leader_torque_enabled is None
    assert out.follower_torque_enabled is None
    assert out.hold_on_disconnect is None
    assert out.bilateral == RakudaBilateralParams()

    text = yaml_path.read_text(encoding="utf-8")
    for key in ("port:", "safety:", "hold_on_disconnect:", "bilateral:", "control_hz:"):
        assert key in text
    for line in text.splitlines():
        # Everything new is commented out, so the file is non-invasive (D18).
        if not line.startswith("#") and line.strip():
            assert line in {"leader:", "follower:", "  torque_enabled: null"} or line.startswith(
                "  #"
            ), line


def test_default_template_examples_are_valid_when_uncommented(yaml_path: Path) -> None:
    apply_rakuda_dotconfig(conventional())
    text = yaml_path.read_text(encoding="utf-8")
    head, marker, tail = text.partition("# safety:")
    assert marker
    uncommented = "\n".join(
        line[2:] if line.startswith("# ") else line for line in (marker + tail).splitlines()
    )
    yaml_path.write_text(head + uncommented + "\n", encoding="utf-8")

    out = apply_rakuda_dotconfig(bilateral(control_hz=100, hold_read_retries=5))
    # The examples carry the dataclass defaults, so they override the code values
    # back to the defaults; keys the template does not list keep the code value.
    assert out.bilateral is not None
    assert out.bilateral.control_hz == RakudaBilateralParams().control_hz
    assert out.bilateral.hold_read_retries == 5
    assert out.bilateral.leader_health == LEADER_HEALTH_DEFAULT
    assert out.hold_on_disconnect is None


def test_default_template_survives_update_helper(yaml_path: Path) -> None:
    update_rakuda_yaml_torque_enabled(leader=["head_yaw"])
    out = apply_rakuda_dotconfig(bilateral())
    assert out.leader_torque_enabled == ["head_yaw"]
    assert out.follower_torque_enabled is None
    assert (out.leader_port, out.follower_port) == (LEADER, FOLLOWER)
    assert out.bilateral == RakudaBilateralParams()


# --- ports and safety -----------------------------------------------------------


def test_ports_override_only_when_non_null(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
leader:
  port: auto
follower:
  port: null
""",
    )
    out = apply_rakuda_dotconfig(conventional())
    assert out.leader_port == PORT_AUTO
    assert out.follower_port == FOLLOWER


def test_ports_accept_paths_and_case_insensitive_auto(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
leader:
  port: " /dev/serial/by-id/usb-FTDI_leader "
follower:
  port: AUTO
""",
    )
    out = apply_rakuda_dotconfig(conventional())
    assert out.leader_port == "/dev/serial/by-id/usb-FTDI_leader"
    assert out.follower_port == PORT_AUTO


@pytest.mark.parametrize("value", ["1", '""', "[]", "true"])
def test_port_rejects_non_string_values(yaml_path: Path, value: str) -> None:
    write_yaml(yaml_path, f"follower:\n  port: {value}\n")
    with pytest.raises(ValueError, match="follower.port must be a device path, 'auto', or null"):
        apply_rakuda_dotconfig(conventional())


@pytest.mark.parametrize(("value", "expected"), [("true", True), ("false", False)])
def test_safety_hold_on_disconnect(yaml_path: Path, value: str, expected: bool) -> None:
    write_yaml(yaml_path, f"safety:\n  hold_on_disconnect: {value}\n")
    out = apply_rakuda_dotconfig(conventional())
    assert out.hold_on_disconnect is expected
    assert out.effective_hold_on_disconnect is expected


def test_safety_null_keeps_code_value(yaml_path: Path) -> None:
    write_yaml(yaml_path, "safety:\n  hold_on_disconnect: null\n")
    cfg = replace(conventional(), hold_on_disconnect=True)
    assert apply_rakuda_dotconfig(cfg) is cfg


@pytest.mark.parametrize("value", ["yes please", "1", "[]"])
def test_safety_rejects_non_bool(yaml_path: Path, value: str) -> None:
    write_yaml(yaml_path, f"safety:\n  hold_on_disconnect: {value}\n")
    with pytest.raises(ValueError, match="safety.hold_on_disconnect must be true, false, or null"):
        apply_rakuda_dotconfig(conventional())


# --- unknown keys -----------------------------------------------------------------


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("leader:\n  prot: auto\n", r"Unknown key\(s\) in leader: \['prot'\]"),
        ("follower:\n  torque: all\n", r"Unknown key\(s\) in follower: \['torque'\]"),
        ("safety:\n  hold: true\n", r"Unknown key\(s\) in safety: \['hold'\]"),
        ("bilateral:\n  controll_hz: 50\n", r"Unknown key\(s\) in bilateral: \['controll_hz'\]"),
        ("bilateral:\n  enabled: true\n", r"Unknown key\(s\) in bilateral: \['enabled'\]"),
        (
            "bilateral:\n  gil_switch_interval_s: 0.001\n",
            r"Unknown key\(s\) in bilateral: \['gil_switch_interval_s'\]",
        ),
        (
            "bilateral:\n  leader_health:\n    temperature_c: 60\n",
            r"Unknown key\(s\) in bilateral.leader_health: \['temperature_c'\]",
        ),
        (
            "bilateral:\n  follower_health:\n    max_voltage_v: 15\n    foo: 1\n",
            r"Unknown key\(s\) in bilateral.follower_health: \['foo'\]",
        ),
    ],
)
def test_unknown_keys_raise_naming_the_key(yaml_path: Path, text: str, match: str) -> None:
    write_yaml(yaml_path, text)
    with pytest.raises(ValueError, match=match):
        apply_rakuda_dotconfig(bilateral())


def test_bilateral_unknown_key_raises_even_in_conventional_mode(yaml_path: Path) -> None:
    write_yaml(yaml_path, "bilateral:\n  nope: 1\n")
    with pytest.raises(ValueError, match=r"Unknown key\(s\) in bilateral: \['nope'\]"):
        apply_rakuda_dotconfig(conventional())


def test_unknown_top_level_key_warns_but_is_ignored(
    yaml_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_yaml(yaml_path, "extras:\n  foo: 1\n")
    caplog.set_level(logging.WARNING, logger="robopy.config.dotrobopy")
    cfg = conventional()
    assert apply_rakuda_dotconfig(cfg) is cfg
    assert any("extras" in record.getMessage() for record in caplog.records)


@pytest.mark.parametrize("section", ["leader", "follower", "safety", "bilateral"])
def test_section_must_be_a_mapping(yaml_path: Path, section: str) -> None:
    write_yaml(yaml_path, f"{section}: [1, 2]\n")
    with pytest.raises(ValueError, match=f"Expected a mapping in {section}"):
        apply_rakuda_dotconfig(bilateral())


# --- bilateral section ---------------------------------------------------------


def test_bilateral_ignored_with_info_when_cfg_bilateral_is_none(
    yaml_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_yaml(yaml_path, "bilateral:\n  control_hz: 25\n")
    caplog.set_level(logging.INFO, logger="robopy.config.dotrobopy")
    cfg = conventional()
    out = apply_rakuda_dotconfig(cfg)
    assert out is cfg
    assert out.bilateral is None
    infos = [r for r in caplog.records if r.levelno == logging.INFO]
    assert len(infos) == 1
    assert "bilateral" in infos[0].getMessage()
    assert "control_hz" in infos[0].getMessage()


def test_bilateral_empty_section_logs_nothing(
    yaml_path: Path, caplog: pytest.LogCaptureFixture
) -> None:
    write_yaml(yaml_path, "bilateral:\n  control_hz: null\n")
    caplog.set_level(logging.INFO, logger="robopy.config.dotrobopy")
    cfg = conventional()
    assert apply_rakuda_dotconfig(cfg) is cfg
    assert not caplog.records


def test_bilateral_every_key_is_applied(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
bilateral:
  current_joints: [r_arm_sh_pitch1, r_arm_sh_roll, torso_yaw]
  control_hz: 100
  follower_divider: 4
  read_timeout_s: 0.010
  follower_read_timeout_s: 0.012
  follower_lost_cycles: 5
  snapshot_stale_s: 0.05
  write_fault_s: 0.02
  diagnostics_period_s: 2.0
  velocity_filter_hz: 15
  gravity_scale: 0.8
  gravity_clamp_factor: 1.1
  allow_uncompensated: true
  allow_unvalidated_gravity: true
  limit_kp_ma_per_count: 1.5
  limit_kd_ma_per_vcount: 0.5
  limit_margin_counts: 60
  limit_max_ma: 250
  hard_margin_counts: 90
  feedback_kp_ma_per_count: 0.2
  feedback_kd_ma_per_vcount: 0.1
  feedback_deadband_counts: 12
  feedback_max_ma: 120
  feedback_gate_alpha: 0.5
  follower_stale_s: 0.3
  ramp_s: 2.0
  current_limit_ma: 600
  current_max_ma: 450
  current_rate_ma_per_s: 2000
  runaway_velocity_counts: 300
  runaway_s: 0.05
  saturation_fault_s: 1.5
  overrun_warn_factor: 2.5
  max_alignment_counts: 250
  align_s: 3.0
  hold_read_retries: 4
  hold_max_snapshot_age_s: 0.02
  hold_max_jump_counts: 150
  hold_profile_velocity: 30
  leader_health:
    {warn_temperature_c: 60, max_temperature_c: 66, min_voltage_v: 10, max_voltage_v: 13}
  follower_health:
    {warn_temperature_c: 58, max_temperature_c: 69, min_voltage_v: 10.5, max_voltage_v: 14}
  gravity_model_path: models/gravity.json
""",
    )
    out = apply_rakuda_dotconfig(bilateral())
    expected = RakudaBilateralParams(
        current_joints=("r_arm_sh_pitch1", "r_arm_sh_roll", "torso_yaw"),
        control_hz=100,
        follower_divider=4,
        read_timeout_s=0.010,
        follower_read_timeout_s=0.012,
        follower_lost_cycles=5,
        snapshot_stale_s=0.05,
        write_fault_s=0.02,
        diagnostics_period_s=2.0,
        velocity_filter_hz=15,
        gravity_scale=0.8,
        gravity_clamp_factor=1.1,
        allow_uncompensated=True,
        allow_unvalidated_gravity=True,
        limit_kp_ma_per_count=1.5,
        limit_kd_ma_per_vcount=0.5,
        limit_margin_counts=60,
        limit_max_ma=250,
        hard_margin_counts=90,
        feedback_kp_ma_per_count=0.2,
        feedback_kd_ma_per_vcount=0.1,
        feedback_deadband_counts=12,
        feedback_max_ma=120,
        feedback_gate_alpha=0.5,
        follower_stale_s=0.3,
        ramp_s=2.0,
        current_limit_ma=600,
        current_max_ma=450,
        current_rate_ma_per_s=2000,
        runaway_velocity_counts=300,
        runaway_s=0.05,
        saturation_fault_s=1.5,
        overrun_warn_factor=2.5,
        max_alignment_counts=250,
        align_s=3.0,
        hold_read_retries=4,
        hold_max_snapshot_age_s=0.02,
        hold_max_jump_counts=150,
        hold_profile_velocity=30,
        leader_health=BusHealthThresholds(60, 66, 10, 13),
        follower_health=BusHealthThresholds(58, 69, 10.5, 14),
        gravity_model_path="models/gravity.json",
    )
    assert out.bilateral == expected
    assert isinstance(out.bilateral.current_joints, tuple)
    assert out.hold_on_disconnect is None
    assert out.effective_hold_on_disconnect is True


def test_bilateral_nested_health_merges_onto_code_values(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
bilateral:
  leader_health:
    max_temperature_c: 66
    min_voltage_v: null
""",
    )
    cfg = bilateral(leader_health=BusHealthThresholds(50.0, 60.0, 11.0, 12.5))
    out = apply_rakuda_dotconfig(cfg)
    assert out.bilateral is not None and cfg.bilateral is not None
    assert out.bilateral.leader_health == BusHealthThresholds(50.0, 66, 11.0, 12.5)
    assert out.bilateral.follower_health == cfg.bilateral.follower_health


def test_bilateral_gravity_scale_dict(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
bilateral:
  gravity_scale:
    r_arm_sh_pitch1: 0.95
    l_arm_sh_pitch1: null
    torso_yaw: 1.0
""",
    )
    out = apply_rakuda_dotconfig(bilateral())
    assert out.bilateral is not None
    assert out.bilateral.gravity_scale == {"r_arm_sh_pitch1": 0.95, "torso_yaw": 1.0}


def test_bilateral_null_values_keep_code_values(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
bilateral:
  control_hz: null
  follower_divider: 1
  gravity_scale: null
""",
    )
    cfg = bilateral(control_hz=100, gravity_scale=0.7)
    out = apply_rakuda_dotconfig(cfg)
    assert out.bilateral is not None
    assert out.bilateral.control_hz == 100
    assert out.bilateral.gravity_scale == 0.7
    assert out.bilateral.follower_divider == 1
    assert out.bilateral.current_joints == RAKUDA_ARM_JOINT_NAMES


@pytest.mark.parametrize(
    ("text", "match"),
    [
        ("bilateral:\n  control_hz: 5\n", "control_hz"),
        ("bilateral:\n  control_hz: fast\n", "control_hz must be a number"),
        ("bilateral:\n  current_joints: [head_yaw]\n", "not allowed"),
        ("bilateral:\n  current_joints: []\n", "must not be empty"),
        ("bilateral:\n  current_max_ma: 900\n", "current_max_ma"),
        ("bilateral:\n  hold_profile_velocity: 0\n", "hold_profile_velocity"),
        ("bilateral:\n  gravity_scale: {r_arm_el_yaw: -1}\n", "gravity_scale"),
        ("bilateral:\n  leader_health: {max_temperature_c: 50}\n", "warn_temperature_c"),
        ("bilateral:\n  leader_health: 68\n", "Expected a mapping in bilateral.leader_health"),
        ("bilateral:\n  allow_uncompensated: yes please\n", "allow_uncompensated"),
    ],
)
def test_bilateral_values_are_validated_when_applied(
    yaml_path: Path, text: str, match: str
) -> None:
    write_yaml(yaml_path, text)
    with pytest.raises(ValueError, match=match):
        apply_rakuda_dotconfig(bilateral())


def test_all_sections_together(yaml_path: Path) -> None:
    write_yaml(
        yaml_path,
        """
leader:
  port: auto
  torque_enabled: all
follower:
  port: /dev/ttyUSB7
  torque_enabled: null
safety:
  hold_on_disconnect: false
bilateral:
  control_hz: 25
  follower_divider: 1
  runaway_s: 0.08
""",
    )
    out = apply_rakuda_dotconfig(bilateral())
    assert out.leader_port == PORT_AUTO
    assert out.follower_port == "/dev/ttyUSB7"
    assert out.leader_torque_enabled == list(RAKUDA_JOINT_NAMES)
    assert out.follower_torque_enabled is None
    assert out.hold_on_disconnect is False
    assert out.effective_hold_on_disconnect is False
    assert out.bilateral is not None
    assert (out.bilateral.control_hz, out.bilateral.follower_divider) == (25, 1)
    assert out.bilateral.effective_hold_max_snapshot_age_s == pytest.approx(0.1)
