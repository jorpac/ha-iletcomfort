"""sn8-gated model decode profiles.

Different C3 heat-pump *models* pack their status/sensor frames differently, but
the Dollin cloud exposes no device-class field: both an air-to-water (ATW) unit
and an air-to-air unit report ``applianceType="0xC3"`` and ``modelNumber="0"``.
The only differentiator is ``sn8`` — an 8-char model-code serial prefix carried
in the appliance metadata (cached on the coordinator as
``appliance_meta["sn8"]``).

Model-specific decoding is therefore gated on an ``sn8 -> profile`` lookup table.
Any UNKNOWN or missing sn8 falls back to the STANDARD decode unchanged, so
existing/working models cannot be corrupted by a new profile.

Profiles
--------
STANDARD
    Default. Byte-for-byte identical to ``decode_its_status`` /
    ``decode_its_sensors``. ``apply_profile_to_*`` return the input untouched.

ATW (sn8 ``171H120F``, Italtherm air-to-water, issue #22)
    A 25-byte status frame with a different field layout. Decoded by
    ``decode_atw_status``; the values are surfaced through the *existing*
    ITSStatus/ITSSensors fields so no entity wiring changes are needed (see the
    field map below). Some units return a known subtype-0x02 placeholder
    template rather than live sensor telemetry; that exact response is
    suppressed so entities remain unavailable rather than reporting fabricated
    values (issue #38). A hardware-validated 27-byte Galmet status variant uses
    direct degrees at byte[6] for its Zone-1 target; it is gated by the frame's
    structural signature because the sn8 is shared with the 25-byte layout. On
    that variant flag bit 0 is Zone-1 power, while compressor state/frequency
    are unavailable because active and stopped captures are otherwise identical.

AQUAPURA (sn8 ``171000AU``, AQS Energie AQUAPURA split HPWH, issue #12)
    Standard status/sensor decode, except the tank temperature must read
    ``status.box_bottom_temp`` (status byte[17], offset-decoded) and is surfaced
    on ``th_temp`` (the "DHW Tank Temperature" sensor) instead of
    ``sensors.twin_temp`` (which is null-filled to 0 on this model).

Known, unconfirmed model codes
------------------------------
``17100007`` is the KJRH-120L2 listed in the vendor's C3 model catalogue. No
hardware report or frame capture exists yet, so it deliberately has no decode
profile and resolves to STANDARD.

For both ATW and AQUAPURA the tank temperature is routed to ``th_temp`` and the
"Water Inlet Temperature" sensor (``twin_temp``) is left honest (no real inlet
reading). Climate ``current_temperature`` is profile-aware: the confirmed ATW
placeholder has no Zone-1 current reading, while other ATW captures retain the
validated tank behavior; AQUAPURA returns ``th_temp`` and STANDARD returns
``twin_temp``.
"""

from __future__ import annotations

import dataclasses
from enum import Enum

from .api import MODE_HEAT, MODE_OFF, ITSSensors, ITSStatus

# sn8 model codes (8-char serial prefixes) → profile.
ATW_SN8 = "171H120F"
AQUAPURA_SN8 = "171000AU"
KJRH120L_SN8 = "17100003"
KJRH120L2_SN8 = "17100007"  # Known catalogue entry; no validated profile yet.
KJR86T3_SN8 = "17186T3A"


class ModelProfile(Enum):
    """Decode profile selected from a device's sn8 model code."""

    STANDARD = "standard"
    ATW = "atw"
    AQUAPURA = "aquapura"
    KJRH120L = "kjrh120l"
    KJR86T3 = "kjr86t3"


_SN8_PROFILES: dict[str, ModelProfile] = {
    ATW_SN8: ModelProfile.ATW,
    AQUAPURA_SN8: ModelProfile.AQUAPURA,
    KJRH120L_SN8: ModelProfile.KJRH120L,
    KJR86T3_SN8: ModelProfile.KJR86T3,
}


def resolve_profile(sn8: str | None) -> ModelProfile:
    """Map an sn8 model code to a decode profile.

    Unknown or missing sn8 → STANDARD (the safe, unchanged default).
    """
    if not sn8:
        return ModelProfile.STANDARD
    return _SN8_PROFILES.get(sn8, ModelProfile.STANDARD)


def build_query_command(profile: ModelProfile, subtype: int) -> str:
    """Return the cloud query command for a status/sensors subtype.

    KJRH-120L's cloud rejects the standard long C3 query frame (1214); its
    app uses a short ``ffff<ss><ss><ss>ff`` form. STANDARD/other profiles use
    the normal build_c3_query frame, unchanged.
    """
    if profile is ModelProfile.KJRH120L:
        return "ffff%02x%02x%02xff" % (subtype, subtype, subtype)
    # Lazy import: api.py imports model_profiles, so importing build_c3_query at
    # module level would risk a circular import.
    from .api import build_c3_query

    return build_c3_query(subtype)


# ---------------------------------------------------------------------------
# ATW profile (issue #22) — validated against five real frames from yoavaviram
# ---------------------------------------------------------------------------
#
# status raw_body, 0-indexed ([0] = 0x01 subtype byte):
#   byte[1]  flags: on the 25-byte Italtherm layout bit0 (0x01) tracks
#            space-heat demand and 0x04 tracks DHW activity. On the 27-byte
#            layout (Galmet #38, NØRDIS #132) bit0 is confirmed Zone-1 power
#            (ON in both active and stopped Heating states; clear when Zone-1
#            is OFF).
#   byte[8]  DHW setpoint in °C — DIRECT value (not +35-offset encoded).
#   byte[9]  Zone-1 setpoint × 2 (0.5° resolution) → zone1 = byte[9] / 2.
#   byte[22] DHW tank current temp in °C — DIRECT value.
#   byte[24] constant 0x80 across all captured states → a flags/MSB byte,
#            NOT an error code. The app shows no fault, so error_code = 0.
#
# The STANDARD decoder misreads this 25-byte frame (set_temperature=3,
# error_code=128, spurious comp_running, DHW tank=0, mode=Unknown(7), …);
# decode_atw_status fixes those by surfacing the confirmed values through the
# existing ITSStatus fields the entities already read:
#   - set_temperature  ← byte[8]   (no DHW-setpoint entity today; kept for SET
#                                    echo / future use)
#   - t5s_def          ← byte[9]/2 (climate target_temperature reads t5s_def)
#   - box_bottom_temp  ← byte[22]  (routed to th_temp / "DHW Tank Temperature";
#                                    see apply_profile_to_sensors. Climate
#                                    current_temperature is profile-aware and
#                                    reads th_temp for ATW.)
#
# CONSERVATIVE / ASSUMED (await hardware validation by the reporter):
#   - comp_running is forced False: none of the five frames carries a confirmed
#     compressor-running signal, and there is no comp_frq field in a 25-byte
#     frame, so we do not derive "running" from byte[14] (which the STANDARD
#     path misread as a comp flag).
#   - The 25-byte Italtherm HVAC mode/action semantics remain at the dataclass
#     defaults (mode 0 / "Off"); its bit0 is demand, not persistent power. The
#     27-byte layout is separately hardware-validated as Heat/Off power.

ATW_DHW_SETPOINT_INDEX = 8
ATW_ZONE1_SETPOINT_X2_INDEX = 9
ATW_DHW_TANK_TEMP_INDEX = 22
ATW_FLAGS_INDEX = 1
ATW_SPACE_HEAT_DEMAND_BIT = 0x01
ATW_GALMET_ZONE1_SETPOINT_INDEX = 6


def _is_27_byte_atw_status(body: bytearray | bytes) -> bool:
    """Return whether ``body`` has the hardware-validated 27-byte ATW layout.

    Galmet (#38) and NØRDIS (#132) units share this layout but differ at
    byte[2] and the [24:27] tail, so neither is part of the signature. Every
    captured 25-byte 171H120F frame has 0x90/0x50/0x68 at byte[3] instead.
    """
    return len(body) == 27 and body[0] == 0x01 and body[3:6] == b"\xa0\x03\x03"


def decode_atw_status(body: bytearray | bytes) -> ITSStatus:
    """Decode an ATW (sn8 171H120F) status frame into an ITSStatus.

    Routes the confirmed ATW byte values into the existing ITSStatus fields the
    entities consume. See the module-level field map for the full layout and the
    conservative choices made for unvalidated fields.
    """
    status = ITSStatus()
    status.raw_body = bytes(body)
    body_len = len(body)

    # Defensive: a frame too short to carry the confirmed fields decodes to an
    # all-defaults ITSStatus (the caller's truncated-frame guard handles the
    # query path; this keeps direct decode calls safe).
    if body_len <= ATW_DHW_TANK_TEMP_INDEX:
        return status

    flags = body[ATW_FLAGS_INDEX]
    status.status_flags_raw = flags
    # The bit's exact semantics depend on the frame layout; see module notes.
    status.pump_system = bool(flags & ATW_SPACE_HEAT_DEMAND_BIT)
    is_27_byte = _is_27_byte_atw_status(body)
    if is_27_byte:
        # Three controlled hardware states establish bit 0 as Zone-1 power for
        # this layout: set for both active and idle Heating, clear when Off.
        status.mode = MODE_HEAT if status.pump_system else MODE_OFF
        status.mode_name = "Heat" if status.pump_system else "Off"

    # DHW setpoint — direct °C value.
    status.set_temperature = body[ATW_DHW_SETPOINT_INDEX]
    # Zone-1 / climate target setpoint. The original 25-byte Italtherm layout
    # stores a half-degree value at byte[9]. The hardware-validated 27-byte
    # variant stores direct degrees at byte[6] despite sharing the sn8.
    if is_27_byte:
        status.t5s_def = float(body[ATW_GALMET_ZONE1_SETPOINT_INDEX])
    else:
        status.t5s_def = body[ATW_ZONE1_SETPOINT_X2_INDEX] / 2
    # DHW tank current temp — direct °C value. Surfaced via box_bottom_temp,
    # which apply_profile_to_sensors routes to th_temp ("DHW Tank Temperature").
    status.box_bottom_temp = float(body[ATW_DHW_TANK_TEMP_INDEX])

    # byte[24] is a flags/MSB byte, not a fault → no error. The Galmet active
    # and stopped captures are otherwise identical, so compressor state and
    # frequency are unavailable rather than false zeroes. Preserve the older
    # conservative False/0 behavior for the 25-byte Italtherm layout.
    status.error_code = 0
    if is_27_byte:
        status.comp_running = None
        status.comp_frq = None
    else:
        status.comp_running = False

    return status


# ---------------------------------------------------------------------------
# KJRH-120L profile (issues #21 / #5) — clean decode from real diagnostics
# ---------------------------------------------------------------------------
#
# The KJRH-120L (sn8 17100003) connects via the short ffff query form
# (build_query_command above), but its 95-byte status frame does NOT match the
# STANDARD C3 layout. The STANDARD decoder misreads it and emits garbage — from
# the reporter's real frame (device Off, app DHW setpoint 60 °C):
#   set_temperature=66, t5s_def=-35, error_code=1 (fake fault),
#   comp_running=True (it's Off), total_kwh=1340, comp_frq=6425, every temp -35.
#
# Confirmed status fields across multiple captured states:
#   body[10] = power state (0x00 = Off, non-zero = On). Found by diffing a real
#              OFF frame vs a real ON frame (issue #35): body[10] is the only
#              state byte that flips with power. body[2] stays 0x00 in BOTH the
#              off and on captures, so it is NOT the power/mode byte.
#   body[15] = DHW setpoint in °C — DIRECT value (0x3c=60, 0x41=65).
# The dual Zone-1 + DHW variant is identified by body[8:10] == 01 01. On that
# variant body[12] is the Zone-1 setpoint and body[15] remains DHW. The climate
# shows Zone-1 read-only, while a separate Number controls DHW through the
# confirmed field-0x07 command. A tested field-0x08 Zone-1 write forced 23 °C
# regardless of the requested value, so it must not be used.
#
# On the pure-DHW variant body[15] remains the climate setpoint. error_code is
# 0 (no fault). comp_running is left False:
# power-on is not a compressor-running signal and no compressor byte is
# confirmed. Everything else in the frame is unmapped garbage for this model and
# is left at the dataclass defaults (None/0).
#
# Water tank temp and outdoor temp are NOT decodable for this model — they do
# not appear in the status frame and the sensors frame is static/cached (three
# different water readings and two outdoor readings never appear in any byte).
# We deliberately do not guess a mapping; apply_profile_to_sensors nulls every
# temperature field so HA shows them unavailable instead of a wrong constant.

# body[10] carries the power state (0 = Off, non-zero = On). When On we report
# mode 1 ("Heat"): climate.py's _QUERY_MODE_TO_HVAC maps query-mode 1 →
# HVACMode.HEAT, a non-off state, so the card shows the unit ON and the Off
# button becomes usable (HA no longer thinks it is already off).
KJRH120L_POWER_INDEX = 10
KJRH120L_MODE_OFF = 0
KJRH120L_MODE_ON = 1
KJRH120L_DHW_SETPOINT_INDEX = 15
KJRH120L_ZONE1_CAP_INDEX_A = 8
KJRH120L_ZONE1_CAP_INDEX_B = 9
KJRH120L_ZONE1_SETPOINT_INDEX = 12
KJRH120L_DHW_TEMP_MIN = 35
KJRH120L_DHW_TEMP_MAX = 60
_KJRH120L_MODES = {0: "Off", 1: "Heat", 2: "Cool", 3: "Auto", 4: "Water Pump"}

# Temperature fields suppressed for the KJRH-120L (not exposed by its API).
_KJRH120L_SUPPRESSED_TEMPS = {
    "t3_temp": None,
    "t4_temp": None,
    "t2_temp": None,
    "t2b_temp": None,
    "twin_temp": None,
    "twout_temp": None,
    "t1_temp": None,
}

# Galmet Prima 8GT (issue #38) returned this exact 75-byte subtype-0x02
# template while the official app displayed live Zone-1 water and outdoor
# values. It has no telemetry: after the subtype, it consists solely of 50
# zeroes and twelve ``03,1e`` pairs. This is deliberately stricter than an
# sn8-only rule because 171H120F covers more than one hardware variant.
_ATW_PLACEHOLDER_SENSORS_BODY = (
    b"\x02\x00"
    + b"\x03\x1e\x00\x00\x00\x00" * 6
    + b"\x00"
    + b"\x03\x1e\x00\x00\x00\x00" * 6
)
_ATW_PLACEHOLDER_SENSOR_FIELDS = {
    "tf_temp": None,
    "tp_temp": None,
    "t3_temp": None,
    "t4_temp": None,
    "t2_temp": None,
    "t2b_temp": None,
    "twin_temp": None,
    "twout_temp": None,
    "t1_temp": None,
    "odu_current": None,
    "odu_voltage": None,
}


# The same template appears in other shapes: 38 bytes with constant 0x28 and no
# separator (Rotenso #124, Kaisai #123), and Galmet's 75-byte shape whose group
# payloads are not all zero (NØRDIS #132; they read like hh,00,hh,00 time slots,
# while the app's live water and outdoor values appear nowhere in the frame).
# Match the structure: subtype 0x02, a 0x00, then only ``03,<c>,x,x,x,x``
# groups sharing one ``c``, with at most a single 0x00 between groups.
_ATW_TEMPLATE_GROUP_LEN = 6
_ATW_TEMPLATE_MIN_GROUPS = 6


def _atw_sensors_match_template_shape(raw_body: bytes) -> bool:
    if len(raw_body) < 4 or raw_body[:2] != b"\x02\x00":
        return False
    constant = raw_body[3]
    groups = 0
    after_separator = False
    i = 2
    while i < len(raw_body):
        if raw_body[i] == 0x00 and groups and not after_separator:
            after_separator = True
            i += 1
            continue
        group = raw_body[i : i + _ATW_TEMPLATE_GROUP_LEN]
        if (
            len(group) != _ATW_TEMPLATE_GROUP_LEN
            or group[0] != 0x03
            or group[1] != constant
        ):
            return False
        groups += 1
        after_separator = False
        i += _ATW_TEMPLATE_GROUP_LEN
    return groups >= _ATW_TEMPLATE_MIN_GROUPS and not after_separator


def atw_sensors_are_placeholder(raw_body: bytes) -> bool:
    """Return whether an ATW sensor frame is a known non-telemetry template."""
    return raw_body == _ATW_PLACEHOLDER_SENSORS_BODY or _atw_sensors_match_template_shape(
        bytes(raw_body)
    )


def kjrh120l_has_zone1(body: bytearray | bytes) -> bool:
    """Return whether a KJRH-120L frame carries the validated Zone-1 variant."""
    return (
        len(body) > KJRH120L_ZONE1_CAP_INDEX_B
        and body[KJRH120L_ZONE1_CAP_INDEX_A] == 0x01
        and body[KJRH120L_ZONE1_CAP_INDEX_B] == 0x01
    )


def decode_kjrh120l_status(body: bytearray | bytes) -> ITSStatus:
    """Decode a KJRH-120L (sn8 17100003) status frame into a clean ITSStatus.

    Surfaces only the confirmed fields (power from body[10], DHW setpoint from
    body[15]) and suppresses the garbage the STANDARD decoder produces. See the
    module-level notes for the full rationale.
    """
    status = ITSStatus()
    status.raw_body = bytes(body)
    body_len = len(body)

    # No fault is confirmed for every captured state. comp_running stays False:
    # power-on is not a compressor-running signal (no compressor byte confirmed).
    status.error_code = 0
    status.comp_running = False

    # Power state from body[10]: 0 → Off, non-zero → On. When On we report mode 1
    # ("Heat") so the climate entity resolves a non-off hvac_mode (HVACMode.HEAT)
    # — making the card read ON and the Off button usable. Defaults to Off when
    # the frame is too short to carry the power byte.
    if body_len > KJRH120L_POWER_INDEX:
        if body[KJRH120L_POWER_INDEX] == 0:
            status.mode = KJRH120L_MODE_OFF
        else:
            status.mode = KJRH120L_MODE_ON
        status.mode_name = _KJRH120L_MODES.get(
            status.mode, f"Unknown({status.mode})"
        )

    # The dual variant exposes Zone-1 at body[12] and DHW at body[15]. The strict
    # body[8]/body[9] capability gate is false on the pure-DHW unit, preserving
    # its existing single-setpoint behavior (issue #5 hardware captures).
    if body_len > KJRH120L_DHW_SETPOINT_INDEX:
        dhw_setpoint = body[KJRH120L_DHW_SETPOINT_INDEX]
        if kjrh120l_has_zone1(body):
            zone1_setpoint = body[KJRH120L_ZONE1_SETPOINT_INDEX]
            status.t5s_def = float(zone1_setpoint)
            status.set_temperature = zone1_setpoint
            status.kjrh120l_dhw_setpoint = float(dhw_setpoint)
        else:
            status.t5s_def = float(dhw_setpoint)
            status.set_temperature = dhw_setpoint

    return status


# ---------------------------------------------------------------------------
# KJRH-120L WRITE / control commands (issue #35)
# ---------------------------------------------------------------------------
#
# The KJRH-120L's cloud rejects the standard 62-byte C3 SET frame (build_c3_set)
# for the same reason it rejected the standard long query: this controller only
# accepts the short literal command strings the official app sends. These were
# captured from the app's proxy traffic (code:0 responses, issues #35 / #21).
#
# General write shape: ``00 <field> 01 <value> ff`` — no checksum byte (the
# cloud frames the command). Confirmed fields:
#   field 0x07 = DHW setpoint  → ``0007 01 <T °C as 1-byte hex> ff``
#                                (T=49 → 00070131ff, T=60 → 0007013cff)
#   field 0x02 = DHW power     → ON ``00020101ff`` / OFF ``00020100ff``
#
# DHW setpoint range for this HPWH (captured 49/60/65 °C) sits above the air-side
# HEAT range, so the climate entity uses a profile-specific min/max.
KJRH120L_DHW_ON = "00020101ff"
KJRH120L_DHW_OFF = "00020100ff"
# Setpoint range the official app allows for this HPWH (issue #35). Captured
# setpoints (49/60/65 °C) sit above the air-side HEAT range, so the climate
# entity uses this profile-specific min/max.
KJRH120L_TEMP_MIN = 20
KJRH120L_TEMP_MAX = 70


def build_kjrh120l_set_temperature(temp: int) -> str:
    """Return the KJRH-120L short DHW-setpoint write command for ``temp`` °C.

    Shape ``0007 01 <temp as 2-hex-digit byte> ff`` — e.g. 60 → ``0007013cff``,
    49 → ``00070131ff`` (confirmed captures). No checksum; the cloud frames it.
    """
    return "000701%02xff" % temp


# ---------------------------------------------------------------------------
# KJR-86T3 table profile (sn8 17186T3A, AQS Energie AQUAPURA variant, #126)
# ---------------------------------------------------------------------------
#
# This wired controller ignores every standard C3 query (the cloud echoes the
# command back). It answers "table" queries instead: message type 0x03 with a
# 16-byte body ``01 <table u16 BE>`` zero-padded; the same body without the
# padding is echoed too. The official app reads tables 0/100/400/500/600/610/
# 900/1000 via ``luaQuery({"u16_table": N})``.
#
# Reply body (frame[10:-1]): [0]=00, [1..2]=table, [5]=series (0x83), [11]=
# period, then table data. u16 values are big-endian signed; temperatures are
# x10. 0x7fff/0xffff mark an unused field. Offsets below were matched against
# the app's own decoded luaQuery JSON captured at the same moment.
KJR86T3_TABLE_LIMITS = 100
KJR86T3_TABLE_SETTINGS = 500
KJR86T3_TABLE_ICONS = 900
KJR86T3_TABLE_PARAMS = 1000
KJR86T3_QUERY_BODY_LEN = 16

KJR86T3_POWER_INDEX = 13  # table 500, u8s_power_on
KJR86T3_MODE_INDEX = 14  # table 500, u8s_mode (0x40 = Eco)
KJR86T3_SETPOINT_INDEX = 15  # table 500, u16s_t5s (x10)
KJR86T3_SILENCE_INDEX = 17  # table 500, u8s_silence_active
KJR86T3_SMART_GRID_INDEX = 20  # table 500, u8s_smart_grid_active ("SG ready")
KJR86T3_TANK_TEMP_INDEX = 30  # table 900, u16_icons_temp (x10)
KJR86T3_TP_INDEX = 43  # table 1000, u16_parm_tp (x10)
KJR86T3_TH_INDEX = 45  # table 1000, u16_parm_th (x10)
KJR86T3_T3_INDEX = 47  # table 1000, u16_parm_t3 (x10)
KJR86T3_T4_INDEX = 49  # table 1000, u16_parm_t4 (x10)
KJR86T3_VOLTAGE_INDEX = 31  # table 1000, u16_parm_voltage
KJR86T3_ENERGY_INDEX = 85  # table 1000, u32_parm_ap (x100, kWh)
# DHW setpoint range reported by table 100 (u16_t5s_min / u16_t5s_max).
KJR86T3_TEMP_MIN = 38
KJR86T3_TEMP_MAX = 70

_KJR86T3_INVALID_U16 = (0x7FFF, 0xFFFF)

# Fields this controller does not expose; None keeps their entities
# unavailable instead of showing a misleading 0.
_KJR86T3_UNEXPOSED_STATUS = {
    "comp_running": None,
    "comp_frq": None,
    "comp_total_run_hours": None,
    "fan_total_run_hours": None,
    "pressure_h": None,
    "pressure_l": None,
}


def build_kjr86t3_table_query(table: int) -> str:
    """Return the read-only table query frame for ``table``."""
    body = [0x01, (table >> 8) & 0xFF, table & 0xFF]
    body += [0x00] * (KJR86T3_QUERY_BODY_LEN - len(body))
    frame = [0xAA, 0, 0xC3, 0, 0, 0, 0, 0, 0, 0x03, *body]
    frame[1] = len(frame)
    frame.append((~sum(frame[1:]) + 1) & 0xFF)
    return bytes(frame).hex()


# Writes (validated on hardware for table 500: setpoint and silence):
# message type 0x02, body ``00 <table u16 BE> ff ff 83 ff 00`` + data, where a
# field read at reply body[k] is written at body[k - 4] (the write header has
# no ``ff ff ff <period>`` block) and 0xff leaves a byte unchanged. The body
# keeps the reply's length. The device answers with the updated table; an
# echo of the command means it was rejected.
KJR86T3_WRITE_SHIFT = 4
KJR86T3_WRITE_HEADER = (0xFF, 0xFF, 0x83, 0xFF, 0x00)
KJR86T3_TABLE_SETTINGS_LEN = 36
# Byte 13 (power) is not applied through this table; power stays app-only
# until its write is found.
KJR86T3_WRITABLE_SETTINGS = frozenset(
    {KJR86T3_SETPOINT_INDEX, KJR86T3_SETPOINT_INDEX + 1, KJR86T3_SILENCE_INDEX}
)


def build_kjr86t3_write(
    table: int, fields: dict[int, int], body_len: int = KJR86T3_TABLE_SETTINGS_LEN,
) -> str:
    """Return a sparse write frame setting ``fields`` (reply index -> byte)."""
    body = [0x00, (table >> 8) & 0xFF, table & 0xFF, *KJR86T3_WRITE_HEADER]
    body += [0xFF] * (body_len - len(body))
    for index, value in fields.items():
        pos = index - KJR86T3_WRITE_SHIFT
        if not len(KJR86T3_WRITE_HEADER) + 3 <= pos < body_len:
            raise ValueError(f"table {table} index {index} is not writable")
        body[pos] = value & 0xFF
    frame = [0xAA, 0, 0xC3, 0, 0, 0, 0, 0, 0, 0x02, *body]
    frame[1] = len(frame)
    frame.append((~sum(frame[1:]) + 1) & 0xFF)
    return bytes(frame).hex()


def build_kjr86t3_settings_write(
    *, setpoint: int | None = None, silence: bool | None = None,
) -> str:
    """Return the table-500 write for a DHW setpoint and/or silent mode."""
    fields: dict[int, int] = {}
    if setpoint is not None:
        if not KJR86T3_TEMP_MIN <= setpoint <= KJR86T3_TEMP_MAX:
            raise ValueError(
                f"Temperature {setpoint}C out of range "
                f"(allowed: {KJR86T3_TEMP_MIN}-{KJR86T3_TEMP_MAX}C)"
            )
        raw = setpoint * 10
        fields[KJR86T3_SETPOINT_INDEX] = raw >> 8
        fields[KJR86T3_SETPOINT_INDEX + 1] = raw & 0xFF
    if silence is not None:
        fields[KJR86T3_SILENCE_INDEX] = 1 if silence else 0
    if not fields:
        raise ValueError("nothing to write")
    assert set(fields) <= KJR86T3_WRITABLE_SETTINGS
    return build_kjr86t3_write(KJR86T3_TABLE_SETTINGS, fields)


def kjr86t3_table_matches(body: bytearray | bytes, table: int) -> bool:
    """Return whether ``body`` is a real reply for ``table`` (not an echo).

    An echoed request starts with 0x01; a device reply starts with 0x00.
    """
    return (
        len(body) > 12
        and body[0] == 0x00
        and body[1] == (table >> 8) & 0xFF
        and body[2] == table & 0xFF
    )


def _kjr86t3_u16(body: bytearray | bytes, index: int) -> int | None:
    if len(body) < index + 2:
        return None
    raw = (body[index] << 8) | body[index + 1]
    if raw in _KJR86T3_INVALID_U16:
        return None
    return raw - 0x10000 if raw & 0x8000 else raw


def _kjr86t3_temp(body: bytearray | bytes, index: int) -> float | None:
    raw = _kjr86t3_u16(body, index)
    return None if raw is None else raw / 10


def decode_kjr86t3_status(
    settings: bytearray | bytes, params: bytearray | bytes,
) -> ITSStatus:
    """Decode KJR-86T3 tables 500 (settings) and 1000 (parameters)."""
    status = dataclasses.replace(ITSStatus(), **_KJR86T3_UNEXPOSED_STATUS)
    status.raw_body = bytes(settings)
    status.error_code = 0

    # Query mode 1 ("Heat") maps to HVACMode.HEAT in climate.py, so a powered
    # DHW unit reads ON; the controller's own mode byte (0x40) is not a C3 mode.
    if len(settings) > KJR86T3_POWER_INDEX:
        power_on = settings[KJR86T3_POWER_INDEX] != 0
        status.mode = 1 if power_on else 0
        status.mode_name = "Heat" if power_on else "Off"

    setpoint = _kjr86t3_temp(settings, KJR86T3_SETPOINT_INDEX)
    if setpoint is not None:
        status.t5s_def = setpoint
        status.set_temperature = int(setpoint)

    if len(settings) > KJR86T3_SMART_GRID_INDEX:
        status.kjr86t3_mode = settings[KJR86T3_MODE_INDEX]
        status.kjr86t3_silence = settings[KJR86T3_SILENCE_INDEX] == 1
        status.mute_valid = status.kjr86t3_silence
        status.kjr86t3_smart_grid = settings[KJR86T3_SMART_GRID_INDEX] == 1

    if len(params) >= KJR86T3_ENERGY_INDEX + 4:
        raw = int.from_bytes(params[KJR86T3_ENERGY_INDEX:KJR86T3_ENERGY_INDEX + 4], "big")
        if raw != 0xFFFFFFFF:
            status.total_kwh = raw / 100

    return status


def decode_kjr86t3_sensors(
    params: bytearray | bytes, icons: bytearray | bytes,
) -> ITSSensors:
    """Decode KJR-86T3 tables 1000 (parameters) and 900 (display icons).

    The DHW tank temperature (the value the app shows) is routed to
    ``th_temp``, the field the "DHW Tank Temperature" sensor reads. The
    controller's own Th probe is not exposed, so that slot is not reused.
    """
    sensors = ITSSensors()
    sensors.raw_body = bytes(params)
    sensors.th_temp = _kjr86t3_temp(icons, KJR86T3_TANK_TEMP_INDEX)
    sensors.tp_temp = _kjr86t3_temp(params, KJR86T3_TP_INDEX)
    sensors.t3_temp = _kjr86t3_temp(params, KJR86T3_T3_INDEX)
    sensors.t4_temp = _kjr86t3_temp(params, KJR86T3_T4_INDEX)
    sensors.odu_voltage = _kjr86t3_u16(params, KJR86T3_VOLTAGE_INDEX)
    sensors.tf_temp = None
    sensors.odu_current = None
    return sensors


def apply_profile_to_status(profile: ModelProfile, status: ITSStatus) -> ITSStatus:
    """Return the profile-canonical ITSStatus for a decoded status object.

    STANDARD and AQUAPURA return ``status`` untouched (AQUAPURA's only override
    is on the sensors side). ATW re-decodes the raw frame via the ATW layout;
    KJRH120L re-decodes it via the KJRH-120L layout (clean setpoint, no garbage).
    """
    if profile is ModelProfile.ATW:
        return decode_atw_status(status.raw_body)
    if profile is ModelProfile.KJRH120L:
        return decode_kjrh120l_status(status.raw_body)
    return status


def apply_profile_to_sensors(
    profile: ModelProfile,
    sensors: ITSSensors,
    status: ITSStatus,
) -> ITSSensors:
    """Return the profile-canonical ITSSensors for the decoded sensors object.

    STANDARD returns ``sensors`` untouched. ATW and AQUAPURA both route a tank
    temperature into ``th_temp`` — the field the "DHW Tank Temperature" sensor
    reads — so that entity shows the meaningful value without entity-wiring
    changes. ``twin_temp`` (the "Water Inlet Temperature" sensor) is deliberately
    left untouched: these units expose no real inlet reading, so it stays honest
    (None/0) rather than being mislabeled with the tank temperature.

    Climate ``current_temperature`` is made profile-aware separately (it returns
    ``th_temp`` for ATW/AQUAPURA) so the climate card still shows the tank temp.

    - ATW: ``status.box_bottom_temp`` carries the DHW tank temp (byte[22]).
    - AQUAPURA: ``status.box_bottom_temp`` (status byte[17], offset-decoded) is
      the water tank temperature the app shows; the STANDARD ``twin_temp`` source
      is null-filled to 0 on this model (issue #12).
    - KJRH120L: water/outdoor temps are NOT exposed by this model's API (the
      sensors frame is static/cached), so every temperature field is nulled — HA
      shows them unavailable rather than the misleading constant -35 the STANDARD
      decode would produce (issues #21 / #5).
    """
    if profile is ModelProfile.KJRH120L:
        return dataclasses.replace(sensors, **_KJRH120L_SUPPRESSED_TEMPS)
    if profile in (ModelProfile.ATW, ModelProfile.AQUAPURA):
        if profile is ModelProfile.ATW and atw_sensors_are_placeholder(
            sensors.raw_body,
        ):
            sensors = dataclasses.replace(sensors, **_ATW_PLACEHOLDER_SENSOR_FIELDS)
        if status is not None and status.box_bottom_temp is not None:
            return dataclasses.replace(sensors, th_temp=status.box_bottom_temp)
    return sensors


__all__ = [
    "AQUAPURA_SN8",
    "ATW_SN8",
    "KJR86T3_SN8",
    "KJR86T3_TEMP_MAX",
    "KJR86T3_TEMP_MIN",
    "KJRH120L_DHW_OFF",
    "KJRH120L_DHW_ON",
    "KJRH120L_DHW_TEMP_MAX",
    "KJRH120L_DHW_TEMP_MIN",
    "KJRH120L_SN8",
    "KJRH120L_TEMP_MAX",
    "KJRH120L_TEMP_MIN",
    "ModelProfile",
    "apply_profile_to_sensors",
    "apply_profile_to_status",
    "atw_sensors_are_placeholder",
    "build_kjr86t3_table_query",
    "build_kjrh120l_set_temperature",
    "build_query_command",
    "decode_atw_status",
    "decode_kjr86t3_sensors",
    "decode_kjr86t3_status",
    "decode_kjrh120l_status",
    "kjr86t3_table_matches",
    "kjrh120l_has_zone1",
    "resolve_profile",
]
