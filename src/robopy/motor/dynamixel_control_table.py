# robopy/control_table.py

"""
Control table definitions for Dynamixel motors.

This module provides structured access to the control table of various Dynamixel
motor series using Enums and dataclasses, enhancing type safety and code clarity.
"""

from dataclasses import dataclass, field
from enum import Enum, IntEnum
from typing import Dict, Literal, TypedDict


class Dtype(Enum):
    """Data types for control table items."""

    UINT8 = "UINT8"
    UINT16 = "UINT16"
    UINT32 = "UINT32"
    INT8 = "INT8"
    INT16 = "INT16"
    INT32 = "INT32"


@dataclass
class ControlItem:
    """
    Represents an item in the Dynamixel control table.

    Attributes:
        address: The memory address of the item.
        num_bytes: The size of the data in bytes (1, 2, or 4).
        dtype: The data type of the item.
        access: Access mode ("R" for read, "R/W" for read/write).
        calibration_required: Flag indicating if this item requires calibration
        (e.g., converting steps to degrees).
    """

    address: int
    num_bytes: Literal[1, 2, 4]
    dtype: Dtype
    access: Literal["R", "R/W"]
    calibration_required: bool = field(default=False, kw_only=True)


class XControlTable(Enum):
    """Control table for Dynamixel X-Series motors."""

    MODEL_NUMBER = ControlItem(0, 2, Dtype.UINT16, "R")
    ID = ControlItem(7, 1, Dtype.UINT8, "R/W")
    BAUD_RATE = ControlItem(8, 1, Dtype.UINT8, "R/W")
    DRIVE_MODE = ControlItem(10, 1, Dtype.UINT8, "R/W")
    OPERATING_MODE = ControlItem(11, 1, Dtype.UINT8, "R/W")
    HOMING_OFFSET = ControlItem(20, 4, Dtype.INT32, "R/W")
    TORQUE_ENABLE = ControlItem(64, 1, Dtype.UINT8, "R/W")
    LED = ControlItem(65, 1, Dtype.UINT8, "R/W")
    GOAL_CURRENT = ControlItem(102, 2, Dtype.INT16, "R/W")
    GOAL_VELOCITY = ControlItem(104, 4, Dtype.INT32, "R/W")
    GOAL_POSITION = ControlItem(116, 4, Dtype.INT32, "R/W", calibration_required=True)
    PRESENT_CURRENT = ControlItem(126, 2, Dtype.INT16, "R")
    PRESENT_VELOCITY = ControlItem(128, 4, Dtype.INT32, "R")
    PRESENT_POSITION = ControlItem(132, 4, Dtype.INT32, "R", calibration_required=True)
    PRESENT_INPUT_VOLTAGE = ControlItem(144, 2, Dtype.UINT16, "R")
    PRESENT_TEMPERATURE = ControlItem(146, 1, Dtype.UINT8, "R")
    POSITION_P_GAIN = ControlItem(84, 2, Dtype.UINT16, "R/W")
    POSITION_I_GAIN = ControlItem(82, 2, Dtype.UINT16, "R/W")
    POSITION_D_GAIN = ControlItem(80, 2, Dtype.UINT16, "R/W")
    CURRENT_LIMIT = ControlItem(38, 2, Dtype.UINT16, "R/W")
    # --- Registers used by the Rakuda bilateral / current-control stack ---
    FIRMWARE_VERSION = ControlItem(6, 1, Dtype.UINT8, "R")
    RETURN_DELAY_TIME = ControlItem(9, 1, Dtype.UINT8, "R/W")
    TEMPERATURE_LIMIT = ControlItem(31, 1, Dtype.UINT8, "R/W")
    MAX_POSITION_LIMIT = ControlItem(48, 4, Dtype.UINT32, "R/W")
    MIN_POSITION_LIMIT = ControlItem(52, 4, Dtype.UINT32, "R/W")
    SHUTDOWN = ControlItem(63, 1, Dtype.UINT8, "R/W")
    HARDWARE_ERROR_STATUS = ControlItem(70, 1, Dtype.UINT8, "R")
    # Signed: -1 is latched when the watchdog expires; 0 disables it and
    # 1..127 arms a timeout of 20 ms per count.
    BUS_WATCHDOG = ControlItem(98, 1, Dtype.INT8, "R/W")
    PROFILE_ACCELERATION = ControlItem(108, 4, Dtype.UINT32, "R/W")
    PROFILE_VELOCITY = ControlItem(112, 4, Dtype.UINT32, "R/W")
    MOVING = ControlItem(122, 1, Dtype.UINT8, "R")


class OperatingMode(IntEnum):
    """Values of the X-series ``OPERATING_MODE`` register (address 11, EEPROM)."""

    CURRENT = 0
    VELOCITY = 1
    POSITION = 3
    EXTENDED_POSITION = 4
    CURRENT_BASED_POSITION = 5
    PWM = 16


#: Addresses below this value are EEPROM on the X series; they persist across a
#: power cycle and are rejected (silently) while ``TORQUE_ENABLE`` is 1.
EEPROM_END_ADDRESS: int = 64

#: The contiguous ``PRESENT_CURRENT`` (int16 @126), ``PRESENT_VELOCITY``
#: (int32 @128) and ``PRESENT_POSITION`` (int32 @132) registers, read as one
#: 10-byte SyncRead by ``DynamixelBus.read_state_block``.
STATE_BLOCK_START_ADDRESS: int = 126
STATE_BLOCK_NUM_BYTES: int = 10

#: Milliamps per raw ``GOAL_CURRENT``/``PRESENT_CURRENT`` count, per model.
#: The XC330 measures current on the supply input, the XM series on the motor
#: winding; the unit is what the e-Manual states for each.
CURRENT_UNIT_MA: Dict[str, float] = {
    "xc330-t288": 1.0,
    "xm430-w350": 2.69,
    "xm540-w270": 2.69,
}

#: Largest value the firmware accepts for ``CURRENT_LIMIT`` (address 38).
CURRENT_LIMIT_MAX_RAW: Dict[str, int] = {
    "xc330-t288": 910,
    "xm430-w350": 1193,
    "xm540-w270": 2047,
}

#: Nominal torque constant in N·m/A, for display and logging only.  These are
#: datasheet ratios (stall torque over stall current), not a validated torque
#: model; control and identification work in milliamps (spec D8).
TORQUE_CONSTANT_NM_PER_A: Dict[str, float] = {
    "xc330-t288": 1.15,
    "xm430-w350": 1.78,
    "xm540-w270": 2.41,
}


# --- Model Specific Definitions ---


class ModelDefinition(TypedDict):
    """Typed dictionary for a motor model's definition."""

    model_number: int
    control_table: type[Enum]  # e.g., XControlTable
    resolution: int


# A dictionary mapping model names to their detailed definitions.
# This makes it easy to add support for new motor models in the future.
MODEL_DEFINITIONS: Dict[str, ModelDefinition] = {
    "xl330-m077": {"model_number": 1190, "control_table": XControlTable, "resolution": 4096},
    "xl330-m288": {"model_number": 1200, "control_table": XControlTable, "resolution": 4096},
    "xc330-t288": {"model_number": 1220, "control_table": XControlTable, "resolution": 4096},
    "xl430-w250": {"model_number": 1060, "control_table": XControlTable, "resolution": 4096},
    "xm430-w350": {"model_number": 1020, "control_table": XControlTable, "resolution": 4096},
    "xm540-w270": {"model_number": 1120, "control_table": XControlTable, "resolution": 4096},
    "xc430-w150": {"model_number": 1070, "control_table": XControlTable, "resolution": 4096},
}

# --- Utility Functions ---


def get_model_definition(model_name: str) -> ModelDefinition:
    """
    Retrieves the definition for a given motor model name.

    Raises:
        ValueError: If the model name is not defined.
    """
    if model_name not in MODEL_DEFINITIONS:
        raise ValueError(f"Model '{model_name}' is not defined.")
    return MODEL_DEFINITIONS[model_name]


def cast_value(value: int, dtype: Dtype) -> int:
    """
    Casts a raw integer value from the motor to the correct signed/unsigned type.
    Handles two's complement for signed integers.
    """
    if dtype == Dtype.INT8:
        return value - 0x100 if value & 0x80 else value
    if dtype == Dtype.INT16:
        # If the highest bit (sign bit) is 1, it's a negative number.
        return value - 0x10000 if value & 0x8000 else value
    if dtype == Dtype.INT32:
        return value - 0x100000000 if value & 0x80000000 else value
    # For unsigned types, no conversion is needed.
    return value


_SIGNED_RANGES: Dict[Dtype, tuple[int, int, int]] = {
    Dtype.INT8: (-0x80, 0x7F, 0xFF),
    Dtype.INT16: (-0x8000, 0x7FFF, 0xFFFF),
    Dtype.INT32: (-0x80000000, 0x7FFFFFFF, 0xFFFFFFFF),
}
_UNSIGNED_LIMITS: Dict[Dtype, int] = {
    Dtype.UINT8: 0xFF,
    Dtype.UINT16: 0xFFFF,
    Dtype.UINT32: 0xFFFFFFFF,
}


def encode_value(value: int, dtype: Dtype) -> int:
    """Encodes a Python int into the unsigned word the wire format expects.

    This is the inverse of :func:`cast_value`: negative values of a signed
    ``dtype`` are two's-complement wrapped (``-5`` -> ``0xFFFB`` for INT16).

    Raises:
        ValueError: If ``value`` does not fit the range of ``dtype``.
    """
    if dtype in _SIGNED_RANGES:
        low, high, mask = _SIGNED_RANGES[dtype]
        if not low <= value <= high:
            raise ValueError(f"{value} does not fit in {dtype.value}.")
        return value & mask
    limit = _UNSIGNED_LIMITS[dtype]
    if not 0 <= value <= limit:
        raise ValueError(f"{value} does not fit in {dtype.value}.")
    return value
