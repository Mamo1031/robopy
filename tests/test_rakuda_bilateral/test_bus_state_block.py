"""Deadline-bounded bus transactions and the extended control table."""

from __future__ import annotations

import dynamixel_sdk as dxl
import pytest
import serial

from robopy.motor.dynamixel_bus import (
    READBACK_TIMEOUT_S,
    SERIAL_WRITE_TIMEOUT_S,
    DiagnosticReading,
    DynamixelBus,
    DynamixelCommError,
    DynamixelTimeoutError,
    MotorStateReading,
    decode_state_block,
)
from robopy.motor.dynamixel_control_table import (
    CURRENT_LIMIT_MAX_RAW,
    CURRENT_UNIT_MA,
    EEPROM_END_ADDRESS,
    STATE_BLOCK_NUM_BYTES,
    STATE_BLOCK_START_ADDRESS,
    TORQUE_CONSTANT_NM_PER_A,
    Dtype,
    OperatingMode,
    XControlTable,
    cast_value,
    encode_value,
)

from .conftest import FakeClock, FakeSdk, make_bus

GOAL_CURRENT_ADDRESS = XControlTable.GOAL_CURRENT.value.address


# --- control table ----------------------------------------------------------


class TestControlTable:
    @pytest.mark.parametrize(
        ("item", "address", "num_bytes", "dtype"),
        [
            (XControlTable.FIRMWARE_VERSION, 6, 1, Dtype.UINT8),
            (XControlTable.RETURN_DELAY_TIME, 9, 1, Dtype.UINT8),
            (XControlTable.DRIVE_MODE, 10, 1, Dtype.UINT8),
            (XControlTable.TEMPERATURE_LIMIT, 31, 1, Dtype.UINT8),
            (XControlTable.MAX_POSITION_LIMIT, 48, 4, Dtype.UINT32),
            (XControlTable.MIN_POSITION_LIMIT, 52, 4, Dtype.UINT32),
            (XControlTable.SHUTDOWN, 63, 1, Dtype.UINT8),
            (XControlTable.HARDWARE_ERROR_STATUS, 70, 1, Dtype.UINT8),
            (XControlTable.BUS_WATCHDOG, 98, 1, Dtype.INT8),
            (XControlTable.PROFILE_ACCELERATION, 108, 4, Dtype.UINT32),
            (XControlTable.PROFILE_VELOCITY, 112, 4, Dtype.UINT32),
            (XControlTable.MOVING, 122, 1, Dtype.UINT8),
            (XControlTable.PRESENT_INPUT_VOLTAGE, 144, 2, Dtype.UINT16),
            (XControlTable.PRESENT_TEMPERATURE, 146, 1, Dtype.UINT8),
        ],
    )
    def test_register_layout(
        self, item: XControlTable, address: int, num_bytes: int, dtype: Dtype
    ) -> None:
        assert (item.value.address, item.value.num_bytes, item.value.dtype) == (
            address,
            num_bytes,
            dtype,
        )

    def test_constants(self) -> None:
        assert STATE_BLOCK_START_ADDRESS == XControlTable.PRESENT_CURRENT.value.address == 126
        assert STATE_BLOCK_NUM_BYTES == 10
        assert EEPROM_END_ADDRESS == XControlTable.TORQUE_ENABLE.value.address == 64
        assert [m.value for m in OperatingMode] == [0, 1, 3, 4, 5, 16]
        assert OperatingMode.CURRENT_BASED_POSITION == 5
        assert CURRENT_UNIT_MA == {"xc330-t288": 1.0, "xm430-w350": 2.69, "xm540-w270": 2.69}
        assert CURRENT_LIMIT_MAX_RAW == {"xc330-t288": 910, "xm430-w350": 1193, "xm540-w270": 2047}
        assert set(TORQUE_CONSTANT_NM_PER_A) == set(CURRENT_UNIT_MA)

    @pytest.mark.parametrize(
        ("dtype", "value"),
        [
            (Dtype.INT8, -1),
            (Dtype.INT8, -128),
            (Dtype.INT16, -5),
            (Dtype.INT16, -32768),
            (Dtype.INT32, -1),
            (Dtype.UINT16, 65535),
        ],
    )
    def test_encode_decode_round_trip(self, dtype: Dtype, value: int) -> None:
        assert cast_value(encode_value(value, dtype), dtype) == value

    def test_encode_rejects_out_of_range(self) -> None:
        with pytest.raises(ValueError, match="INT16"):
            encode_value(40000, Dtype.INT16)
        with pytest.raises(ValueError, match="INT8"):
            encode_value(128, Dtype.INT8)
        with pytest.raises(ValueError, match="UINT8"):
            encode_value(-1, Dtype.UINT8)

    def test_watchdog_minus_one_decodes_as_int8(self) -> None:
        assert cast_value(0xFF, Dtype.INT8) == -1


# --- decoding -------------------------------------------------------------------


class TestDecodeStateBlock:
    def test_signed_fields(self) -> None:
        block = (
            bytes([0xFB, 0xFF])
            + (-2).to_bytes(4, "little", signed=True)
            + bytes((-1000).to_bytes(4, "little", signed=True))
        )
        assert decode_state_block(block) == MotorStateReading(
            position=-1000, velocity=-2, current_raw=-5
        )

    def test_accepts_int_sequence_and_positive_values(self) -> None:
        block = [0x10, 0x00, 0x05, 0, 0, 0, 0x00, 0x10, 0, 0]
        assert decode_state_block(block) == MotorStateReading(
            position=4096, velocity=5, current_raw=16
        )

    def test_rejects_wrong_length(self) -> None:
        with pytest.raises(ValueError, match="10 bytes"):
            decode_state_block(bytes(9))

    def test_reading_is_frozen_and_has_no_valid_field(self) -> None:
        reading = MotorStateReading(position=1, velocity=2, current_raw=3)
        assert not hasattr(reading, "valid")
        with pytest.raises(AttributeError):
            reading.position = 5  # type: ignore[misc]


# --- read_state_block -------------------------------------------------------


class TestReadStateBlock:
    def test_call_order_and_values(self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock) -> None:
        sdk.set_state(1, position=2048, velocity=-3, current=-5)
        sdk.set_state(2, position=-100, velocity=7, current=910)
        sdk.read_duration_s = 0.014

        readings, start_ns, end_ns = bus.read_state_block(["a", "b"], timeout_s=0.012)

        assert readings == {
            "a": MotorStateReading(position=2048, velocity=-3, current_raw=-5),
            "b": MotorStateReading(position=-100, velocity=7, current_raw=910),
        }
        assert sdk.method_calls() == ["txPacket", "setPacketTimeoutMillis", "rxPacket"]
        assert sdk.packet_timeouts_ms() == [12.0]
        assert end_ns - start_ns == 14_000_000
        assert end_ns == clock.now_ns

    def test_timeout_is_a_hard_bound_of_one_attempt(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock
    ) -> None:
        sdk.silent_ids.add(2)

        with pytest.raises(DynamixelTimeoutError):
            bus.read_state_block(["a", "b"], timeout_s=0.012)

        assert clock.elapsed_ns == 12_000_000
        assert sdk.method_calls() == ["txPacket", "setPacketTimeoutMillis", "rxPacket"]

    def test_timeout_error_is_also_a_connection_error_and_timeout_error(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.silent_ids.add(1)
        with pytest.raises(ConnectionError):
            bus.read_state_block(["a"], timeout_s=0.012)
        with pytest.raises(TimeoutError):
            bus.read_state_block(["a"], timeout_s=0.012)
        with pytest.raises(DynamixelCommError):
            bus.read_state_block(["a"], timeout_s=0.012)

    def test_attempts_run_back_to_back(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock
    ) -> None:
        sdk.silent_ids.add(1)

        with pytest.raises(DynamixelTimeoutError, match="3 attempt"):
            bus.read_state_block(["a", "b", "c"], timeout_s=0.012, attempts=3)

        assert clock.elapsed_ns == 36_000_000
        assert sdk.method_calls() == ["txPacket", "setPacketTimeoutMillis", "rxPacket"] * 3

    def test_second_attempt_can_succeed(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock
    ) -> None:
        sdk.set_state(1, position=10, velocity=0, current=0)
        sdk.rx_results = [dxl.COMM_RX_CORRUPT]

        readings, _, _ = bus.read_state_block(["a"], timeout_s=0.012, attempts=2)

        assert readings["a"].position == 10
        assert clock.elapsed_ns == 12_000_000
        assert sdk.method_calls().count("rxPacket") == 2

    @pytest.mark.parametrize("code", [dxl.COMM_TX_FAIL, dxl.COMM_TX_ERROR, dxl.COMM_PORT_BUSY])
    def test_transmit_failure_raises_comm_error_without_retry(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock, code: int
    ) -> None:
        sdk.tx_results = [code]

        with pytest.raises(DynamixelCommError) as info:
            bus.read_state_block(["a"], timeout_s=0.012, attempts=3)

        assert not isinstance(info.value, DynamixelTimeoutError)
        assert clock.elapsed_ns == 0
        assert sdk.method_calls() == ["txPacket"]

    def test_stalled_serial_write_is_a_timeout_error(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="stalled"):
            bus.read_state_block(["a"], timeout_s=0.012, attempts=3)
        assert sdk.method_calls() == ["txPacket"]

    def test_missing_motor_counts_as_failed_attempt(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.unavailable_ids.add(2)

        with pytest.raises(DynamixelTimeoutError, match=r"no data from \['b'\]"):
            bus.read_state_block(["a", "b"], timeout_s=0.012, attempts=2)

        assert sdk.method_calls().count("rxPacket") == 2

    def test_truncated_block_counts_as_missing_not_value_error(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        # A CRC-valid but short status packet passes isAvailable in the SDK.
        sdk.short_ids.add(2)

        with pytest.raises(DynamixelTimeoutError, match=r"no data from \['b'\]"):
            bus.read_state_block(["a", "b"], timeout_s=0.012, attempts=2)

        assert sdk.method_calls().count("rxPacket") == 2

    def test_bus_recovers_after_a_stalled_serial_write(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        # The SDK leaves port.is_using set when the write raises; the next
        # transmit would otherwise be COMM_PORT_BUSY forever.
        sdk.set_state(1, position=42, velocity=0, current=0)
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="stalled"):
            bus.read_state_block(["a"], timeout_s=0.012)
        assert bus.port_handler.is_using is False

        readings, _, _ = bus.read_state_block(["a"], timeout_s=0.012)

        assert readings["a"].position == 42

    def test_stale_is_using_flag_is_cleared_before_transmit(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        bus.port_handler.is_using = True
        bus.read_state_block(["a"], timeout_s=0.012)
        bus.port_handler.is_using = True
        bus.write_goal_current_raw({"a": 1})
        assert sdk.get_register(1, XControlTable.GOAL_CURRENT) == 1

    def test_pending_input_is_discarded_before_each_transmit(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        bus.open()
        ser = sdk.ports[-1].ser
        assert ser is not None
        sdk.set_state(1, position=1, velocity=0, current=0)
        sdk.set_state(2, position=100, velocity=0, current=0)
        # ID 1 does not answer: the SDK gives up there and ID 2's answer stays
        # queued in the input buffer.
        sdk.silent_ids.add(1)
        with pytest.raises(DynamixelTimeoutError):
            bus.read_state_block(["a", "b"], timeout_s=0.012)
        assert 2 in ser.input_buffer

        sdk.silent_ids.clear()
        sdk.set_state(2, position=200, velocity=0, current=0)
        readings, _, _ = bus.read_state_block(["a", "b"], timeout_s=0.012)

        assert readings["b"].position == 200  # fresh, not the queued 100
        assert ser.input_buffer == {}
        assert (
            sdk.method_calls()
            == ["reset_input_buffer", "txPacket", "setPacketTimeoutMillis", "rxPacket"] * 2
        )

    def test_read_without_serial_object_skips_input_flush(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        # Not opened (ser is None) and a port whose serial object has no
        # reset_input_buffer must both still read.
        bus.read_state_block(["a"], timeout_s=0.012)
        assert "reset_input_buffer" not in sdk.method_calls()
        bus.open()
        bus.port_handler.ser = object()  # type: ignore[assignment]
        bus.read_state_block(["a"], timeout_s=0.012)

    def test_unknown_or_empty_motor_list_is_rejected(self, bus: DynamixelBus) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            bus.read_state_block(["a", "zzz"], timeout_s=0.012)
        with pytest.raises(ValueError):
            bus.read_state_block([], timeout_s=0.012)

    def test_invalid_timeout_or_attempts_is_rejected(self, bus: DynamixelBus) -> None:
        with pytest.raises(ValueError, match="timeout_s"):
            bus.read_state_block(["a"], timeout_s=0.0)
        with pytest.raises(ValueError, match="attempts"):
            bus.read_state_block(["a"], timeout_s=0.012, attempts=0)

    def test_group_handle_is_cached_per_motor_set(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.read_state_block(["a", "b"], timeout_s=0.012)
        bus.read_state_block(["a", "b"], timeout_s=0.012)
        assert len(sdk.read_groups) == 1
        bus.read_state_block(["b", "a"], timeout_s=0.012)
        assert len(sdk.read_groups) == 2
        assert list(sdk.read_groups[0].data_dict) == [1, 2]


# --- write_goal_current_raw ---------------------------------------------------


class TestWriteGoalCurrentRaw:
    def test_encodes_int16_and_transmits_once(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.write_goal_current_raw({"a": -100})

        assert sdk.writes == [(GOAL_CURRENT_ADDRESS, {1: [0x9C, 0xFF]})]
        assert sdk.method_calls() == ["txPacket"]
        assert sdk.get_register(1, XControlTable.GOAL_CURRENT) == 0xFF9C

    def test_several_motors_in_declaration_order(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.write_goal_current_raw({"c": 300, "a": 1})
        assert sdk.writes == [(GOAL_CURRENT_ADDRESS, {1: [1, 0], 3: [0x2C, 0x01]})]

    @pytest.mark.parametrize("value", [32768, -32769, 40000])
    def test_out_of_int16_range_is_rejected_before_transmit(
        self, sdk: FakeSdk, bus: DynamixelBus, value: int
    ) -> None:
        with pytest.raises(ValueError, match="'a'"):
            bus.write_goal_current_raw({"a": value})
        assert sdk.writes == []

    def test_unknown_motor_or_empty_mapping_is_rejected(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            bus.write_goal_current_raw({"zzz": 0})
        with pytest.raises(ValueError):
            bus.write_goal_current_raw({})
        assert sdk.writes == []

    def test_transmit_failure_raises_comm_error(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.write_results = [dxl.COMM_TX_FAIL]
        with pytest.raises(DynamixelCommError):
            bus.write_goal_current_raw({"a": 5})
        assert sdk.method_calls() == ["txPacket"]

    def test_stalled_serial_write_is_a_timeout_error(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="stalled") as info:
            bus.write_goal_current_raw({"a": 5})
        assert isinstance(info.value.__cause__, serial.SerialTimeoutException)
        assert sdk.writes == []

    def test_write_group_is_reused(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.write_goal_current_raw({"a": 1})
        bus.write_goal_current_raw({"a": 2})
        assert len(sdk.write_groups) == 1
        assert sdk.get_register(1, XControlTable.GOAL_CURRENT) == 2


# --- write_with_readback ------------------------------------------------------


class TestWriteWithReadback:
    def test_success_on_first_attempt(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock
    ) -> None:
        bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 3, "b": 3})

        assert sdk.get_register(1, XControlTable.OPERATING_MODE) == 3
        assert sdk.get_register(2, XControlTable.OPERATING_MODE) == 3
        assert len(sdk.writes) == 1
        assert clock.elapsed_ns == 5_000_000  # one settle_s, read-backs take no fake time
        # Read-backs are single-motor reads with the fixed 40 ms packet timeout.
        assert sdk.packet_timeouts_ms() == [READBACK_TIMEOUT_S * 1e3] * 2

    def test_success_on_second_attempt_rewrites_only_the_mismatch(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.set_register(1, XControlTable.OPERATING_MODE, 3)
        sdk.rejected_writes = {1: 1}

        bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 0, "b": 0})

        assert [set(ids) for _, ids in sdk.writes] == [{1, 2}, {1}]
        assert sdk.get_register(1, XControlTable.OPERATING_MODE) == 0

    def test_mismatch_after_attempts_raises_comm_error(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.rejected_writes = {1: 99}
        sdk.set_register(1, XControlTable.OPERATING_MODE, 3)

        with pytest.raises(DynamixelCommError, match=r"'a': \(0, 3\)"):
            bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 0}, attempts=3)

        assert len(sdk.writes) == 3

    def test_silent_motor_is_reported_as_no_response(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.silent_ids.add(2)

        with pytest.raises(DynamixelCommError, match=r"'b': \(1, None\)"):
            bus.write_with_readback(XControlTable.TORQUE_ENABLE, {"a": 1, "b": 1}, attempts=2)

        assert len(sdk.writes) == 2

    def test_truncated_readback_never_confirms_a_zero(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        # An empty status payload must not decode to 0 and "confirm" a wanted 0.
        sdk.short_ids.add(1)
        sdk.set_register(1, XControlTable.OPERATING_MODE, 3)

        with pytest.raises(DynamixelCommError, match=r"'a': \(0, None\)"):
            bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 0}, attempts=1)

    def test_readback_succeeds_after_a_stalled_write(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        # A stall on the current-write path must not poison the hold sequence
        # (legacy sync_write + read-back on the same port handler).
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="stalled"):
            bus.write_goal_current_raw({"a": 5})

        bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 3}, attempts=1)
        bus.torque_disabled(["a"])

        assert sdk.get_register(1, XControlTable.OPERATING_MODE) == 3
        assert sdk.get_register(1, XControlTable.TORQUE_ENABLE) == 0

    def test_tolerance_accepts_small_difference(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.rejected_writes = {1: 99}
        sdk.set_register(1, XControlTable.GOAL_POSITION, 2050)

        bus.write_with_readback(XControlTable.GOAL_POSITION, {"a": 2048}, tolerance=2)
        with pytest.raises(DynamixelCommError):
            bus.write_with_readback(
                XControlTable.GOAL_POSITION, {"a": 2048}, tolerance=1, attempts=1
            )

    def test_signed_readback_compares_as_signed(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.write_with_readback(XControlTable.GOAL_CURRENT, {"a": -50})
        assert cast_value(sdk.get_register(1, XControlTable.GOAL_CURRENT), Dtype.INT16) == -50

    def test_invalid_arguments(self, bus: DynamixelBus) -> None:
        with pytest.raises(ValueError, match="Unknown"):
            bus.write_with_readback(XControlTable.OPERATING_MODE, {"zzz": 3})
        with pytest.raises(ValueError, match="attempts"):
            bus.write_with_readback(XControlTable.OPERATING_MODE, {"a": 3}, attempts=0)


# --- diagnostics / identity -------------------------------------------------


class TestDiagnosticsAndIdentity:
    def test_read_diagnostics(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.set_register(1, XControlTable.HARDWARE_ERROR_STATUS, 0x04)
        sdk.set_register(1, XControlTable.PRESENT_INPUT_VOLTAGE, 118)
        sdk.set_register(1, XControlTable.PRESENT_TEMPERATURE, 45)
        sdk.set_register(3, XControlTable.PRESENT_INPUT_VOLTAGE, 120)

        result = bus.read_diagnostics(["a", "c"], timeout_s=0.02)

        assert result == {
            "a": DiagnosticReading(hardware_error_status=4, voltage_v=11.8, temperature_c=45),
            "c": DiagnosticReading(hardware_error_status=0, voltage_v=12.0, temperature_c=0),
        }
        assert sdk.packet_timeouts_ms() == [20.0, 20.0]

    def test_read_diagnostics_timeout(
        self, sdk: FakeSdk, bus: DynamixelBus, clock: FakeClock
    ) -> None:
        sdk.silent_ids.add(3)
        with pytest.raises(DynamixelTimeoutError):
            bus.read_diagnostics(["a", "c"])
        assert clock.elapsed_ns == 50_000_000

    def test_read_model_numbers_omits_silent_motors(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.silent_ids.add(2)
        assert bus.read_model_numbers(["a", "b", "c"]) == {"a": 1220, "c": 1020}
        assert sdk.packet_timeouts_ms() == [40.0, 40.0, 40.0]

    def test_read_model_numbers_omits_truncated_answers(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.short_ids.add(2)
        assert bus.read_model_numbers(["a", "b", "c"]) == {"a": 1220, "c": 1020}

    def test_read_diagnostics_truncated_answer_is_a_timeout(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.short_ids.add(3)
        with pytest.raises(DynamixelTimeoutError, match=r"no data from \['c'\]"):
            bus.read_diagnostics(["a", "c"])

    def test_verify_models_passes_when_bus_matches_config(self, bus: DynamixelBus) -> None:
        bus.verify_models()
        bus.verify_models(["a"])

    def test_verify_models_names_mismatched_and_unreachable_motors(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.set_register(1, XControlTable.MODEL_NUMBER, 1020)  # declared xc330-t288 (1220)
        sdk.silent_ids.add(2)

        with pytest.raises(ConnectionError) as info:
            bus.verify_models()

        message = str(info.value)
        assert "a (ID 1)" in message and "xc330-t288" in message and "1020" in message
        assert "b (ID 2): no response" in message
        assert "c (ID 3)" not in message


# --- port lifecycle -----------------------------------------------------------


class TestPortLifecycle:
    def test_open_sets_serial_write_timeout(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        bus.open()
        port = sdk.ports[-1]
        assert port.is_open and port.ser is not None
        assert port.ser.write_timeout == SERIAL_WRITE_TIMEOUT_S == 0.02
        bus.close()
        assert not port.is_open

    def test_open_tolerates_port_without_serial_object(self, sdk: FakeSdk) -> None:
        sdk.ports_have_serial = False
        bus = make_bus()
        bus.open()
        assert sdk.ports[-1].ser is None

    def test_open_failure_raises_connection_error(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.open_fails = True
        with pytest.raises(ConnectionError, match="Failed to open port"):
            bus.open()

    def test_set_port_while_closed_replaces_handler_and_caches(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        bus.read_state_block(["a"], timeout_s=0.012)
        bus.write_goal_current_raw({"a": 0})

        bus.set_port("/dev/fake1")

        assert bus.port_handler.port_name == "/dev/fake1"
        assert "/dev/fake1" in repr(bus)
        bus.read_state_block(["a"], timeout_s=0.012)
        bus.write_goal_current_raw({"a": 0})
        assert len(sdk.read_groups) == 2 and len(sdk.write_groups) == 2
        assert sdk.read_groups[-1].port is bus.port_handler

    def test_set_port_while_open_raises(self, bus: DynamixelBus) -> None:
        bus.open()
        with pytest.raises(RuntimeError, match="open"):
            bus.set_port("/dev/fake1")
        assert bus.port_handler.port_name == "/dev/fake0"


# --- legacy API regression --------------------------------------------------


class TestLegacyApiUnchanged:
    def test_sync_read_uses_txrxpacket_and_a_fresh_group(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.set_register(1, XControlTable.PRESENT_POSITION, 2048)
        sdk.set_register(2, XControlTable.PRESENT_POSITION, -1)

        result = bus.sync_read(XControlTable.PRESENT_POSITION, ["a", "b"])

        assert result == {"a": 2048, "b": -1}
        assert sdk.method_calls() == ["txRxPacket"]
        bus.sync_read(XControlTable.PRESENT_POSITION, ["a", "b"])
        assert len(sdk.read_groups) == 2  # not cached, as before

    def test_sync_read_retries_ten_times_then_raises(self, sdk: FakeSdk, bus: DynamixelBus) -> None:
        sdk.silent_ids.add(1)
        with pytest.raises(DynamixelCommError) as info:
            bus.sync_read(XControlTable.PRESENT_POSITION, ["a"])
        assert not isinstance(info.value, DynamixelTimeoutError)
        assert sdk.method_calls() == ["txRxPacket"] * 10

    def test_sync_write_uses_txpacket_and_a_fresh_group(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        bus.sync_write(XControlTable.TORQUE_ENABLE, {"a": 1, "c": 1})
        bus.torque_disabled(["a"])
        bus.write(XControlTable.GOAL_POSITION, "b", 100)

        assert sdk.method_calls() == ["txPacket"] * 3
        assert len(sdk.write_groups) == 3
        assert sdk.get_register(1, XControlTable.TORQUE_ENABLE) == 0
        assert sdk.get_register(3, XControlTable.TORQUE_ENABLE) == 1
        assert sdk.get_register(2, XControlTable.GOAL_POSITION) == 100
        assert bus.read(XControlTable.GOAL_POSITION, "b") == 100

    def test_sync_write_stall_is_a_timeout_and_the_next_write_succeeds(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        # The hold sequence runs on the legacy path; a stall inside it must not
        # leave port.is_using set, or the retry hold would see COMM_PORT_BUSY.
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="TORQUE_ENABLE sync write.*stalled"):
            bus.torque_disabled(["a"])
        assert bus.port_handler.is_using is False
        assert sdk.method_calls() == ["txPacket"]  # a stall is not retried

        bus.torque_disabled(["a"])

        assert sdk.get_register(1, XControlTable.TORQUE_ENABLE) == 0
        assert sdk.method_calls() == ["txPacket"] * 2

    def test_sync_read_stall_is_a_timeout_and_the_next_read_succeeds(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        sdk.set_register(1, XControlTable.PRESENT_POSITION, 2048)
        sdk.stall_writes = 1
        with pytest.raises(DynamixelTimeoutError, match="PRESENT_POSITION sync read.*stalled"):
            bus.sync_read(XControlTable.PRESENT_POSITION, ["a"])
        assert bus.port_handler.is_using is False
        assert sdk.method_calls() == ["txRxPacket"]

        assert bus.sync_read(XControlTable.PRESENT_POSITION, ["a"]) == {"a": 2048}

    def test_legacy_paths_clear_a_stale_is_using_flag(
        self, sdk: FakeSdk, bus: DynamixelBus
    ) -> None:
        bus.port_handler.is_using = True
        bus.torque_disabled(["a"])
        bus.port_handler.is_using = True
        bus.sync_read(XControlTable.PRESENT_POSITION, ["a"])
        assert sdk.method_calls() == ["txPacket", "txRxPacket"]  # no COMM_PORT_BUSY retry
