"""SPEC.md §22 / §22.2 — the canonical telemetry codec.

Cross-platform JSON differences are a recurring source of silent corruption, and exports that
parse differently on two platforms cannot be pooled (failure mode 47). Every rule §22.2 states
is enforced here in **both** directions, because a decoder that quietly accepts ``NaN`` makes
a strict encoder pointless:

* **Casing.** Enum values ``UPPER_SNAKE_CASE``; property keys ``lowerCamelCase``; event-type
  identifiers ``lower_snake_case``. No exceptions, including fixtures.
* **Nonfinite literals forbidden.** JSON has no ``NaN``/``Infinity``. Unavailable → ``null``
  plus a sibling status field. Encoders fail rather than emit nonstandard literals; decoders
  reject them.
* **Locale independence.** ``.`` decimal separator, no digit grouping, no localized number or
  date formatting, regardless of device locale.
* **Precision.** Doubles serialized with shortest round-trip. Never through a ``Float``.
* **Timestamps.** Wall clock is RFC 3339 UTC with explicit ``Z``; monotonic is integer
  nanoseconds with a named clock domain. Never mixed in one field.
* **Units in names.** Numeric field names end with their unit unless dimensionless and
  documented as such — :data:`DOCUMENTED_DIMENSIONLESS_FIELDS` is that documentation.

The three canonical monotonic timestamps have three meanings and are never interchangeable.
**Freshness is always computed from mapped source time, never arrival** (failure mode 11), so
:class:`TelemetryEnvelope` exposes the source/arrival distinction rather than one "timestamp".
"""

from __future__ import annotations

import json
import math
import re
from dataclasses import dataclass
from typing import Any

__all__ = [
    "DOCUMENTED_DIMENSIONLESS_FIELDS",
    "EVENT_TYPES",
    "TelemetryCodecError",
    "TelemetryEnvelope",
    "decode_event",
    "encode_event",
    "numeric_fields_missing_unit_suffix",
]


class TelemetryCodecError(ValueError):
    """A document that violates a §22.2 encoding rule, in either direction."""


#: §22 event types, ``lower_snake_case``, a separate namespace from enum values.
EVENT_TYPES: frozenset[str] = frozenset(
    {
        "session_start",
        "session_end",
        "app_lifecycle",
        "clock_mapping",
        "ground_truth",
        "fixture_state",
        "location_sample",
        "location_authorization",
        "location_provider_state",
        "magnetometer_calibrated",
        "magnetometer_uncalibrated",
        "accelerometer",
        "gravity",
        "gyroscope",
        "rotation_vector",
        "device_motion",
        "os_heading",
        "fused_orientation",
        "capability_resolution",
        "wmm_output",
        "reference_resolution",
        "engine_output",
        "state_transition",
        "precision_lock",
        "sensor_health",
        "calibration_request",
        "calibration_prompt",
        "calibration_result",
        "target_heading_request",
        "target_guidance",
        "deviation_profile_lookup",
        "deviation_correction",
        "certification_lookup",
        "display_frame_marker",
        "thermal_state",
        "battery_state",
        "charging_state",
        "power_mode",
        "space_weather_advisory",
        "orientation_change",
        "sensor_discontinuity",
        "dropped_sample_summary",
        "user_action",
    }
)

#: §22 ``sourceClock`` domain identifiers.
SOURCE_CLOCKS: frozenset[str] = frozenset(
    {"ELAPSED_REALTIME", "CORE_MOTION_BOOT_TIME", "PROVIDER_DATE", "FIXTURE_CLOCK"}
)

#: §22.2 unit suffixes. A numeric field name ends with one of these unless it is documented
#: dimensionless below.
UNIT_SUFFIXES: tuple[str, ...] = (
    "Deg",
    "Ms",
    "Ns",
    "Us",
    "MicroTesla",
    "NanoTesla",
    "Hz",
    "Km",
    "M",
    "G",
)

#: The documented dimensionless numeric fields. §22.2 permits a numeric name without a unit
#: suffix only when it is "dimensionless and documented as such"; this set *is* that
#: documentation, so adding a field here is a deliberate, reviewable act.
DOCUMENTED_DIMENSIONLESS_FIELDS: frozenset[str] = frozenset(
    {
        "eventId",  # counter
        "sequence",  # counter
        "kp",  # NOAA planetary K-index, a dimensionless coarse advisory scale
        "uncertaintyCoverageTarget",  # a probability in [0, 1]
        "circularResultantLength",  # R in [0, 1]
        "relativeMagnitudeResidual",  # a ratio
        "effectiveHeadingSampleCount",  # count
        "periodicSupportSampleCount",  # count
        "sectorCount",  # count
        "repetitions",  # count
        "truthCoverageFactor",  # a dimensionless coverage factor k
    }
)

_LOWER_CAMEL_CASE = re.compile(r"^[a-z][A-Za-z0-9]*$")
_LOWER_SNAKE_CASE = re.compile(r"^[a-z][a-z0-9_]*$")
#: RFC 3339 UTC with an explicit ``Z``; no offset form, no local time, no space separator.
_RFC3339_UTC = re.compile(r"^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$")


@dataclass(frozen=True)
class TelemetryEnvelope:
    """§22's common envelope. Typed so a timestamp cannot land in the wrong field."""

    schema_version: str
    session_id: str
    event_id: int
    event_type: str
    platform: str
    app_version: str
    app_build: str
    engine_version: str
    config_version: str
    config_hash: str
    device_anonymous_id: str
    hardware_runtime_identity: str
    sensor_runtime_identity: str
    os_build_identity: str
    wall_time_utc: str
    record_monotonic_time_ns: int
    source_monotonic_time_ns: int
    arrival_monotonic_time_ns: int
    source_clock: str
    clock_mapping_id: str
    sequence: int

    def __post_init__(self) -> None:
        if self.event_type not in EVENT_TYPES:
            raise TelemetryCodecError(
                f"unknown eventType {self.event_type!r}; §22 fixes the event-type namespace"
            )
        if not _LOWER_SNAKE_CASE.match(self.event_type):
            raise TelemetryCodecError(f"eventType {self.event_type!r} is not lower_snake_case")
        if self.source_clock not in SOURCE_CLOCKS:
            raise TelemetryCodecError(
                f"unknown sourceClock {self.source_clock!r}; freshness cannot be computed "
                "from a timestamp whose clock domain is unidentified (failure mode 10)"
            )
        if not _RFC3339_UTC.match(self.wall_time_utc):
            raise TelemetryCodecError(
                f"wallTimeUtc {self.wall_time_utc!r} is not RFC 3339 UTC with an explicit Z"
            )
        for name in (
            "record_monotonic_time_ns",
            "source_monotonic_time_ns",
            "arrival_monotonic_time_ns",
            "event_id",
            "sequence",
        ):
            value = getattr(self, name)
            if not isinstance(value, int) or isinstance(value, bool):
                raise TelemetryCodecError(
                    f"{name} must be an integer; §22.2 keeps monotonic time in integer "
                    "nanoseconds and never mixes it with a wall-clock string"
                )

    def to_document(self) -> dict[str, Any]:
        return {
            "schemaVersion": self.schema_version,
            "sessionId": self.session_id,
            "eventId": self.event_id,
            "eventType": self.event_type,
            "platform": self.platform,
            "appVersion": self.app_version,
            "appBuild": self.app_build,
            "engineVersion": self.engine_version,
            "configVersion": self.config_version,
            "configHash": self.config_hash,
            "deviceAnonymousId": self.device_anonymous_id,
            "hardwareRuntimeIdentity": self.hardware_runtime_identity,
            "sensorRuntimeIdentity": self.sensor_runtime_identity,
            "osBuildIdentity": self.os_build_identity,
            "wallTimeUtc": self.wall_time_utc,
            "recordMonotonicTimeNs": self.record_monotonic_time_ns,
            "sourceMonotonicTimeNs": self.source_monotonic_time_ns,
            "arrivalMonotonicTimeNs": self.arrival_monotonic_time_ns,
            "sourceClock": self.source_clock,
            "clockMappingId": self.clock_mapping_id,
            "sequence": self.sequence,
        }

    @staticmethod
    def from_document(document: dict[str, Any]) -> "TelemetryEnvelope":
        return TelemetryEnvelope(
            schema_version=document["schemaVersion"],
            session_id=document["sessionId"],
            event_id=document["eventId"],
            event_type=document["eventType"],
            platform=document["platform"],
            app_version=document["appVersion"],
            app_build=document["appBuild"],
            engine_version=document["engineVersion"],
            config_version=document["configVersion"],
            config_hash=document["configHash"],
            device_anonymous_id=document["deviceAnonymousId"],
            hardware_runtime_identity=document["hardwareRuntimeIdentity"],
            sensor_runtime_identity=document["sensorRuntimeIdentity"],
            os_build_identity=document["osBuildIdentity"],
            wall_time_utc=document["wallTimeUtc"],
            record_monotonic_time_ns=document["recordMonotonicTimeNs"],
            source_monotonic_time_ns=document["sourceMonotonicTimeNs"],
            arrival_monotonic_time_ns=document["arrivalMonotonicTimeNs"],
            source_clock=document["sourceClock"],
            clock_mapping_id=document["clockMappingId"],
            sequence=document["sequence"],
        )


def _walk(value: Any, path: str, visit_key, visit_number) -> None:
    if isinstance(value, dict):
        for key, child in value.items():
            visit_key(key, path)
            _walk(child, f"{path}.{key}" if path else key, visit_key, visit_number)
    elif isinstance(value, list):
        for index, child in enumerate(value):
            _walk(child, f"{path}[{index}]", visit_key, visit_number)
    elif isinstance(value, bool):
        return
    elif isinstance(value, (int, float)):
        visit_number(value, path)


def assert_encoding_rules(document: dict[str, Any]) -> None:
    """Every §22.2 structural rule, applied to a whole event document."""

    def visit_key(key: str, path: str) -> None:
        if not _LOWER_CAMEL_CASE.match(key):
            raise TelemetryCodecError(
                f"property key {key!r} at {path or '<root>'} is not lowerCamelCase (§22.2)"
            )

    def visit_number(value: float, path: str) -> None:
        if isinstance(value, float) and not math.isfinite(value):
            raise TelemetryCodecError(
                f"nonfinite number at {path}: JSON has no NaN/Infinity. An unavailable value "
                "is null plus a sibling status field, never a nonstandard literal (§22.2)."
            )

    _walk(document, "", visit_key, visit_number)


def numeric_fields_missing_unit_suffix(document: dict[str, Any]) -> tuple[str, ...]:
    """§22.2's units-in-names rule, made executable.

    Returns the dotted paths of numeric fields whose names carry neither a unit suffix nor a
    place in :data:`DOCUMENTED_DIMENSIONLESS_FIELDS`.
    """
    offenders: list[str] = []

    def visit_key(key: str, path: str) -> None:  # noqa: ARG001 - required by _walk
        return

    def visit_number(value: float, path: str) -> None:  # noqa: ARG001
        name = path.split(".")[-1]
        if "[" in name:  # a numeric array element inherits its array's name
            name = name.split("[")[0]
        if name in DOCUMENTED_DIMENSIONLESS_FIELDS:
            return
        if not any(name.endswith(suffix) for suffix in UNIT_SUFFIXES):
            offenders.append(path)

    _walk(document, "", visit_key, visit_number)
    return tuple(offenders)


def encode_event(envelope: TelemetryEnvelope, payload: dict[str, Any]) -> str:
    """Encode one JSONL line.

    ``allow_nan=False`` makes the encoder **fail** rather than emit ``NaN``/``Infinity``;
    ``ensure_ascii=False`` keeps the ruleset's glyphs as themselves; ``separators`` removes
    incidental whitespace so a line hash is stable. Python's float repr is already the
    shortest round-trip form §22.2 requires, and ``json`` never applies locale formatting —
    the locale test proves that rather than assuming it.
    """
    document = envelope.to_document()
    document["payload"] = payload
    assert_encoding_rules(document)
    return json.dumps(
        document, allow_nan=False, ensure_ascii=False, separators=(",", ":"), sort_keys=False
    )


def _reject_constant(token: str) -> Any:
    raise TelemetryCodecError(
        f"decoder rejected the nonstandard JSON literal {token!r}; §22.2 forbids "
        "NaN/Infinity in either direction"
    )


def decode_event(line: str) -> tuple[TelemetryEnvelope, dict[str, Any]]:
    """Decode one JSONL line, rejecting anything the encoder would have refused to write."""
    document = json.loads(line, parse_constant=_reject_constant)
    if not isinstance(document, dict):
        raise TelemetryCodecError("a telemetry event must be a JSON object")
    assert_encoding_rules(document)
    payload = document.get("payload")
    if not isinstance(payload, dict):
        raise TelemetryCodecError("every event carries a typed object payload (§22)")
    return TelemetryEnvelope.from_document(document), payload
