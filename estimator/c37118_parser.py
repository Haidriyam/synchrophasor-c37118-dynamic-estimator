"""
IEEE C37.118.2-2011 Synchrophasor Telemetry Protocol Dissector.
Parses streaming binary data frames (SYNC 0xAA01) and validates timing invariants.
"""
from dataclasses import dataclass
import struct
from typing import List, Tuple


@dataclass(frozen=True)
class PMUDataRecord:
    station_id: int
    soc_timestamp: int
    fraction_of_second: int
    time_synchronized: bool
    data_valid: bool
    phasors: List[Tuple[float, float]]  # (Magnitude, Phase Angle in radians)
    frequency_hz: float
    rocof: float  # Rate of change of frequency (Hz/s)


class C37118FrameParser:
    DATA_SYNC = 0xAA01
    MAX_FRAME_SIZE = 65535

    @staticmethod
    def unpack_data_frame(
        payload: bytes, num_phasors: int = 1, is_floating: bool = True
    ) -> PMUDataRecord:
        """
        Unpack a single IEEE C37.118.2 data frame with strict length and boundary checks.
        Expects Big-Endian byte order.
        """
        if len(payload) < 14:
            raise ValueError(f"Packet undersized: {len(payload)} bytes (minimum 14 required).")

        sync, frame_size, station_id = struct.unpack(">HHH", payload[:6])
        if sync != C37118FrameParser.DATA_SYNC:
            raise ValueError(f"Invalid frame sync header: {hex(sync)} (expected 0xAA01).")

        if frame_size != len(payload):
            raise ValueError(
                f"Frame size mismatch: header declares {frame_size} bytes, got {len(payload)}."
            )

        soc, fracsec_raw = struct.unpack(">II", payload[6:14])
        # Top 8 bits of FRACSEC store time quality flags, lower 24 bits are fraction
        fracsec = fracsec_raw & 0x00FFFFFF

        # Parse STAT word
        offset = 14
        if len(payload) < offset + 2:
            raise ValueError("Premature EOF reading STAT word.")
        stat = struct.unpack(">H", payload[offset:offset + 2])[0]
        offset += 2

        # STAT bits: Bit 15 = Data Valid (0 = valid, 1 = invalid), Bit 13 = Clock Sync
        data_valid = not bool(stat & 0x8000)
        time_sync = not bool(stat & 0x2000)

        phasors: List[Tuple[float, float]] = []
        if is_floating:
            # 8 bytes per phasor (2x float32: Real, Imag or Mag, Angle)
            expected_p_bytes = num_phasors * 8
            if len(payload) < offset + expected_p_bytes + 8:
                raise ValueError("Payload insufficient for declared floating-point channels.")

            for _ in range(num_phasors):
                val1, val2 = struct.unpack(">ff", payload[offset:offset + 8])
                phasors.append((val1, val2))
                offset += 8

            freq, rocof = struct.unpack(">ff", payload[offset:offset + 8])
            offset += 8
        else:
            # 4 bytes per phasor (2x int16)
            expected_p_bytes = num_phasors * 4
            if len(payload) < offset + expected_p_bytes + 4:
                raise ValueError("Payload insufficient for declared fixed-point channels.")

            for _ in range(num_phasors):
                val1, val2 = struct.unpack(">hh", payload[offset:offset + 4])
                phasors.append((float(val1), float(val2)))
                offset += 4

            f_raw, r_raw = struct.unpack(">hh", payload[offset:offset + 4])
            freq = 50.0 + (f_raw / 1000.0)
            rocof = r_raw / 100.0
            offset += 4

        return PMUDataRecord(
            station_id=station_id,
            soc_timestamp=soc,
            fraction_of_second=fracsec,
            time_synchronized=time_sync,
            data_valid=data_valid,
            phasors=phasors,
            frequency_hz=freq,
            rocof=rocof,
        )

    @staticmethod
    def pack_test_frame(
        station_id: int,
        soc: int,
        fracsec: int,
        voltage_mag: float,
        voltage_phase_rad: float,
        freq_hz: float,
        rocof: float = 0.0,
        sync_valid: bool = True,
    ) -> bytes:
        """Helper to construct standard C37.118 data frames for integration tests."""
        sync = C37118FrameParser.DATA_SYNC
        frame_size = 32  # 14 (header) + 2 (stat) + 8 (1 phasor float) + 8 (freq/rocof float)
        stat = 0x0000 if sync_valid else 0x2000

        header = struct.pack(">HHHIIH", sync, frame_size, station_id, soc, fracsec, stat)
        phasor_bytes = struct.pack(">ff", voltage_mag, voltage_phase_rad)
        analog_bytes = struct.pack(">ff", freq_hz, rocof)

        return header + phasor_bytes + analog_bytes