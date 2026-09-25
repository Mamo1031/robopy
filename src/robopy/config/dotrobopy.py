"""User overrides for Rakuda from ``.robopy/rakuda/config.yaml``.

Every value that is ``null`` (or absent) leaves the ``RakudaConfig`` built in
code untouched, so the generated default file changes nothing. A non-null YAML
value overrides the value in code, including command-line flags that a script
has put into the config; the rule is YAML (non-null) > code > dataclass
defaults. When the YAML replaces a value the caller set explicitly, one
WARNING per key names both values.
"""

from __future__ import annotations

import logging
from dataclasses import fields, replace
from functools import reduce
from pathlib import Path
from typing import TYPE_CHECKING, Any, Collection, Mapping

import yaml

if TYPE_CHECKING:
    from robopy.config.robot_config.rakuda_config import RakudaBilateralParams, RakudaConfig

logger = logging.getLogger(__name__)

_UNSET: Any = object()

_TOP_LEVEL_KEYS: frozenset[str] = frozenset({"leader", "follower", "safety", "bilateral"})
_SIDE_KEYS: frozenset[str] = frozenset({"port", "torque_enabled"})
_SAFETY_KEYS: frozenset[str] = frozenset({"hold_on_disconnect"})
_NESTED_BILATERAL_KEYS: tuple[str, ...] = ("leader_health", "follower_health")


def _as_dict(value: Any, *, name: str = "YAML") -> dict[str, Any]:
    if value is None:
        return {}
    if not isinstance(value, dict):
        raise ValueError(f"Expected a mapping in {name}, got {type(value).__name__}.")
    return value


def _reject_unknown_keys(
    section: Mapping[str, Any], *, allowed: Collection[str], section_name: str
) -> None:
    unknown = sorted(str(key) for key in section if key not in allowed)
    if unknown:
        raise ValueError(f"Unknown key(s) in {section_name}: {unknown}. Allowed: {sorted(allowed)}")


def _as_str_list_or_none(value: Any, *, field_name: str) -> list[str] | None:
    if value is None:
        return None
    if isinstance(value, list):
        if not all(isinstance(x, str) for x in value):
            raise ValueError(f"{field_name} must be a list[str] or null.")
        return list(value)
    raise ValueError(f"{field_name} must be a list[str] or null.")


def _parse_torque_enabled_yaml(
    value: Any,
    *,
    field_name: str,
    all_joint_names: tuple[str, ...],
) -> list[str] | None:
    """Parse YAML torque_enabled into list[str] | None.

    Accepts:
    - null -> None (use default behavior)
    - list[str] -> explicit list (empty list allowed)
    - str keywords:
        - 'all' -> all_joint_names
        - 'default'/'null' -> None
        - 'none'/'off' -> []
    """

    if value is None:
        return None
    if isinstance(value, list):
        if not all(isinstance(x, str) for x in value):
            raise ValueError(f"{field_name} must be a list[str], a keyword, or null.")
        return list(value)
    if isinstance(value, str):
        v = value.strip().lower()
        if v in {"default", "null"}:
            return None
        if v in {"none", "off"}:
            return []
        if v == "all":
            return list(all_joint_names)
        raise ValueError(f"{field_name} must be a list[str], one of (all/default/none), or null.")
    raise ValueError(f"{field_name} must be a list[str], a keyword, or null.")


def _parse_port_yaml(value: Any, *, field_name: str) -> str | None:
    """Parse YAML port into a device path, ``PORT_AUTO`` or None (null)."""

    from robopy.config.robot_config.rakuda_config import PORT_AUTO

    if value is None:
        return None
    if isinstance(value, str) and value.strip():
        port = value.strip()
        return PORT_AUTO if port.lower() == PORT_AUTO else port
    raise ValueError(f"{field_name} must be a device path, '{PORT_AUTO}', or null.")


def _parse_bool_yaml(value: Any, *, field_name: str) -> bool | None:
    if value is None or isinstance(value, bool):
        return value
    raise ValueError(f"{field_name} must be true, false, or null.")


def _non_null_overrides(
    section: Mapping[str, Any], *, allowed: Collection[str], section_name: str
) -> dict[str, Any]:
    _reject_unknown_keys(section, allowed=allowed, section_name=section_name)
    return {key: value for key, value in section.items() if value is not None}


def _parse_bilateral_yaml(section: Mapping[str, Any]) -> dict[str, Any]:
    """Collect the non-null ``bilateral:`` overrides, validating keys (not values).

    Nested ``leader_health`` / ``follower_health`` mappings come back as dicts
    of their own non-null keys; a ``gravity_scale`` mapping drops null entries.
    Values are validated by ``RakudaBilateralParams.__post_init__`` when applied.
    """

    from robopy.config.robot_config.rakuda_config import (
        BusHealthThresholds,
        RakudaBilateralParams,
    )

    overrides = _non_null_overrides(
        section,
        allowed=[f.name for f in fields(RakudaBilateralParams)],
        section_name="bilateral",
    )
    health_keys = [f.name for f in fields(BusHealthThresholds)]
    for key in _NESTED_BILATERAL_KEYS:
        if key in overrides:
            overrides[key] = _non_null_overrides(
                _as_dict(overrides[key], name=f"bilateral.{key}"),
                allowed=health_keys,
                section_name=f"bilateral.{key}",
            )
    if isinstance(overrides.get("gravity_scale"), dict):
        overrides["gravity_scale"] = {
            joint: value for joint, value in overrides["gravity_scale"].items() if value is not None
        }
    return overrides


def _apply_bilateral_overrides(
    base: RakudaBilateralParams, overrides: Mapping[str, Any]
) -> RakudaBilateralParams:
    """``dataclasses.replace(base, **overrides)`` with the nested thresholds merged."""

    kwargs = dict(overrides)
    for key in _NESTED_BILATERAL_KEYS:
        if key in kwargs:
            kwargs[key] = replace(getattr(base, key), **kwargs[key])
    return replace(base, **kwargs)


def _warn_override(yaml_path: Path, key: str, code_value: Any, yaml_value: Any) -> None:
    logger.warning(
        "config.yaml overrides %s: code %r -> yaml %r (edit or remove it in %s to use the "
        "code value)",
        key,
        code_value,
        yaml_value,
        yaml_path,
    )


def _warn_bilateral_overrides(
    yaml_path: Path,
    base: RakudaBilateralParams,
    applied: RakudaBilateralParams,
    overrides: Mapping[str, Any],
) -> None:
    """Warns for every overridden ``bilateral`` key whose code value is not the default.

    A code value equal to the ``RakudaBilateralParams()`` default is taken as
    "not set by the caller", so overriding it is silent.
    """

    from robopy.config.robot_config.rakuda_config import RakudaBilateralParams

    defaults = RakudaBilateralParams()
    for key, value in overrides.items():
        paths = [(key, sub) for sub in value] if key in _NESTED_BILATERAL_KEYS else [(key,)]
        for path in paths:
            default, code, new = (reduce(getattr, path, obj) for obj in (defaults, base, applied))
            if code != default and new != code:
                _warn_override(yaml_path, "bilateral." + ".".join(path), code, new)


def get_dotrobopy_dir(base_dir: Path | None = None) -> Path:
    """Return the base `.robopy` directory.

    Default: current working directory.
    """

    return (base_dir or Path.cwd()) / ".robopy"


def get_rakuda_dotdir(base_dir: Path | None = None) -> Path:
    """Return the Rakuda config directory: `.robopy/rakuda`."""

    return get_dotrobopy_dir(base_dir) / "rakuda"


def get_rakuda_yaml_path(base_dir: Path | None = None) -> Path:
    """Return the Rakuda YAML config path: `.robopy/rakuda/config.yaml`."""

    return get_rakuda_dotdir(base_dir) / "config.yaml"


def ensure_rakuda_dotfiles(base_dir: Path | None = None) -> Path:
    """Ensure `.robopy/rakuda` exists and return its path."""

    dotdir = get_rakuda_dotdir(base_dir)
    dotdir.mkdir(parents=True, exist_ok=True)
    return dotdir


def ensure_rakuda_yaml_exists(base_dir: Path | None = None) -> Path:
    """Ensure `.robopy/rakuda/config.yaml` exists and return its path.

    This is the file-level entry point: callers should prefer this over checking
    directory existence.
    """

    from robopy.config.robot_config.rakuda_config import RAKUDA_JOINT_NAMES

    yaml_path = get_rakuda_yaml_path(base_dir)
    yaml_path.parent.mkdir(parents=True, exist_ok=True)
    ensure_default_rakuda_yaml(yaml_path, joint_names=RAKUDA_JOINT_NAMES)
    return yaml_path


def _yaml_scalar(value: Any) -> str:
    return yaml.safe_dump(value, default_flow_style=True).strip().removesuffix("\n...")


def _bilateral_template_lines() -> list[str]:
    """The commented-out ``bilateral:`` example block, built from the dataclass defaults."""

    from robopy.config.robot_config.rakuda_config import RakudaBilateralParams

    defaults = RakudaBilateralParams()
    example_keys = (
        "control_hz",
        "follower_divider",
        "read_timeout_s",
        "follower_read_timeout_s",
        "gravity_scale",
        "feedback_kp_ma_per_count",
        "feedback_kd_ma_per_vcount",
        "limit_kp_ma_per_count",
        "limit_kd_ma_per_vcount",
        "current_max_ma",
    )
    lines = ["# bilateral:"]
    for key in example_keys:
        lines.append(f"#   {key}: {_yaml_scalar(getattr(defaults, key))}")
    lines.append("#   # gravity_scale: {r_arm_sh_pitch1: 0.95, l_arm_sh_pitch1: 0.95}")
    health = defaults.leader_health
    lines.append(
        "#   leader_health: {"
        f"warn_temperature_c: {_yaml_scalar(health.warn_temperature_c)}, "
        f"max_temperature_c: {_yaml_scalar(health.max_temperature_c)}, "
        f"max_voltage_v: {_yaml_scalar(health.max_voltage_v)}"
        "}"
    )
    return lines


def ensure_default_rakuda_yaml(path: Path, *, joint_names: tuple[str, ...]) -> None:
    """Create a default Rakuda YAML if it does not exist.

    The default file should be non-invasive: it must not change runtime behavior
    unless the user edits it.
    """

    if path.exists():
        return

    joint_list_comment = "\n".join([f"  # - {name}" for name in joint_names])

    content = "\n".join(
        [
            "# robopy user config (Rakuda)",
            "#",
            "# This file is created automatically. Edit it to customize Rakuda behavior.",
            "# A null (or commented-out) value keeps whatever the code passed in.",
            "#",
            "# Semantics:",
            "# - leader.port / follower.port: serial device path, or 'auto' to detect the bus",
            "#   by scanning (null -> the port given in code)",
            "# - leader.torque_enabled: joints to torque ON (null -> default: grippers only)",
            "# - follower.torque_enabled: joints to torque ON (null -> default: all joints)",
            "# - safety.hold_on_disconnect: keep both arms torque-on in place on disconnect",
            "#   (null -> true in bilateral mode, false in conventional mode)",
            "# - bilateral: gain / timing overrides for the leader current-control loop.",
            "#   Ignored (INFO) unless RakudaConfig(bilateral=RakudaBilateralParams(...)) is",
            "#   passed in code; there is no 'enabled' key. When bilateral is active,",
            "#   follower.torque_enabled must include every bilateral.current_joints entry,",
            "#   and those joints are dropped from leader.torque_enabled (the loop owns them).",
            "#",
            "# Available joint names:",
            joint_list_comment,
            "",
            "leader:",
            "  torque_enabled: null",
            "  # port: auto",
            "  # torque_enabled:",
            "  #   - l_arm_grip",
            "  #   - r_arm_grip",
            "",
            "follower:",
            "  torque_enabled: null",
            "  # port: /dev/ttyUSB0",
            "  # torque_enabled:",
            "  #   - torso_yaw",
            "",
            "# safety:",
            "#   hold_on_disconnect: null",
            "",
            *_bilateral_template_lines(),
            "",
        ]
    )

    path.write_text(content, encoding="utf-8")


def load_yaml(path: Path) -> dict[str, Any]:
    """Load YAML file safely. Returns {} for empty files."""

    if not path.exists():
        return {}
    text = path.read_text(encoding="utf-8")
    if not text.strip():
        return {}
    loaded = yaml.safe_load(text)
    return _as_dict(loaded)


def validate_joint_names(
    names: list[str] | None,
    *,
    allowed: set[str],
    field_name: str,
) -> None:
    if names is None:
        return
    unknown = sorted(set(names) - allowed)
    if unknown:
        allowed_preview = ", ".join(sorted(allowed))
        raise ValueError(
            f"Unknown joint name(s) in {field_name}: {unknown}. Allowed: {allowed_preview}"
        )


def apply_rakuda_dotconfig(
    cfg: "RakudaConfig",
    *,
    base_dir: Path | None = None,
) -> "RakudaConfig":
    """Apply `.robopy/rakuda/config.yaml` overrides to a RakudaConfig.

    This also ensures the directory and default YAML exist. Only non-null YAML
    values override ``cfg``; the ``bilateral:`` section is applied with
    ``dataclasses.replace`` onto ``cfg.bilateral`` and ignored (INFO) when
    ``cfg.bilateral`` is None. Unknown keys inside the known sections raise
    ``ValueError``.

    One WARNING is logged per key where the YAML replaces a value the caller
    set explicitly: a port that differs (ports are always explicit), a
    ``torque_enabled`` list or ``hold_on_disconnect`` that is not None in
    ``cfg`` and differs, and a ``bilateral`` key whose ``cfg`` value differs
    from the ``RakudaBilateralParams()`` default and from the YAML value.
    """

    # Local import to avoid circular dependency in robopy.config package.
    from robopy.config.robot_config.rakuda_config import RAKUDA_JOINT_NAMES, RakudaConfig

    if not isinstance(cfg, RakudaConfig):
        raise TypeError("apply_rakuda_dotconfig expects a RakudaConfig")

    yaml_path = ensure_rakuda_yaml_exists(base_dir)

    data = load_yaml(yaml_path)

    unknown_sections = sorted(str(key) for key in data if key not in _TOP_LEVEL_KEYS)
    if unknown_sections:
        logger.warning("Ignoring unknown top-level key(s) in %s: %s", yaml_path, unknown_sections)

    leader = _as_dict(data.get("leader"), name="leader")
    follower = _as_dict(data.get("follower"), name="follower")
    safety = _as_dict(data.get("safety"), name="safety")
    _reject_unknown_keys(leader, allowed=_SIDE_KEYS, section_name="leader")
    _reject_unknown_keys(follower, allowed=_SIDE_KEYS, section_name="follower")
    _reject_unknown_keys(safety, allowed=_SAFETY_KEYS, section_name="safety")
    bilateral_overrides = _parse_bilateral_yaml(_as_dict(data.get("bilateral"), name="bilateral"))

    leader_torque_enabled = _parse_torque_enabled_yaml(
        leader.get("torque_enabled"),
        field_name="leader.torque_enabled",
        all_joint_names=RAKUDA_JOINT_NAMES,
    )
    follower_torque_enabled = _parse_torque_enabled_yaml(
        follower.get("torque_enabled"),
        field_name="follower.torque_enabled",
        all_joint_names=RAKUDA_JOINT_NAMES,
    )

    allowed = set(RAKUDA_JOINT_NAMES)
    validate_joint_names(leader_torque_enabled, allowed=allowed, field_name="leader.torque_enabled")
    validate_joint_names(
        follower_torque_enabled, allowed=allowed, field_name="follower.torque_enabled"
    )

    # Only override if YAML explicitly provides a non-null value.
    updates: dict[str, Any] = {}
    leader_port = _parse_port_yaml(leader.get("port"), field_name="leader.port")
    if leader_port is not None:
        updates["leader_port"] = leader_port
        if leader_port != cfg.leader_port:
            _warn_override(yaml_path, "leader.port", cfg.leader_port, leader_port)
    follower_port = _parse_port_yaml(follower.get("port"), field_name="follower.port")
    if follower_port is not None:
        updates["follower_port"] = follower_port
        if follower_port != cfg.follower_port:
            _warn_override(yaml_path, "follower.port", cfg.follower_port, follower_port)
    for side, yaml_joints, code_joints in (
        ("leader", leader_torque_enabled, cfg.leader_torque_enabled),
        ("follower", follower_torque_enabled, cfg.follower_torque_enabled),
    ):
        if yaml_joints is None:
            continue
        updates[f"{side}_torque_enabled"] = yaml_joints
        if code_joints is not None and set(code_joints) != set(yaml_joints):
            _warn_override(yaml_path, f"{side}.torque_enabled", code_joints, yaml_joints)
    hold_on_disconnect = _parse_bool_yaml(
        safety.get("hold_on_disconnect"), field_name="safety.hold_on_disconnect"
    )
    if hold_on_disconnect is not None:
        updates["hold_on_disconnect"] = hold_on_disconnect
        if cfg.hold_on_disconnect is not None and cfg.hold_on_disconnect != hold_on_disconnect:
            _warn_override(
                yaml_path, "safety.hold_on_disconnect", cfg.hold_on_disconnect, hold_on_disconnect
            )
    if bilateral_overrides:
        if cfg.bilateral is None:
            logger.info(
                "Ignoring the bilateral: section of %s (%s): RakudaConfig.bilateral is None, "
                "so the conventional position-teleoperation path is used.",
                yaml_path,
                sorted(bilateral_overrides),
            )
        else:
            applied = _apply_bilateral_overrides(cfg.bilateral, bilateral_overrides)
            _warn_bilateral_overrides(yaml_path, cfg.bilateral, applied, bilateral_overrides)
            updates["bilateral"] = applied

    if not updates:
        return cfg

    return replace(cfg, **updates)


def update_rakuda_yaml_torque_enabled(
    *,
    leader: list[str] | None | Any = _UNSET,
    follower: list[str] | None | Any = _UNSET,
    base_dir: Path | None = None,
) -> Path:
    """Update `.robopy/rakuda/config.yaml` torque settings.

    - leader / follower:
        - _UNSET: do not modify the field
        - None: write YAML null (meaning: use default behavior)
        - list[str]: explicit joints to torque ON (empty list means torque OFF for all)
    """

    from robopy.config.robot_config.rakuda_config import RAKUDA_JOINT_NAMES

    yaml_path = ensure_rakuda_yaml_exists(base_dir)

    data = load_yaml(yaml_path)
    if not isinstance(data, dict):
        data = {}

    allowed = set(RAKUDA_JOINT_NAMES)
    if leader is not _UNSET:
        validate_joint_names(leader, allowed=allowed, field_name="leader.torque_enabled")
    if follower is not _UNSET:
        validate_joint_names(follower, allowed=allowed, field_name="follower.torque_enabled")

    leader_dict = _as_dict(data.get("leader"), name="leader")
    follower_dict = _as_dict(data.get("follower"), name="follower")

    if leader is not _UNSET:
        leader_dict["torque_enabled"] = leader
    if follower is not _UNSET:
        follower_dict["torque_enabled"] = follower

    data["leader"] = leader_dict
    data["follower"] = follower_dict

    yaml_path.write_text(
        yaml.safe_dump(data, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return yaml_path
