"""SPEC.md §22 / §22.2 — the canonical telemetry codec.

Failure mode 47 names the specific defects: Swift/Kotlin ``NaN`` divergence, a ``Double``
serialized through a ``Float``, locale decimal separators, and differing default estimators.
Exports that parse differently on two platforms cannot be pooled, so every rule is checked in
both directions and the locale rule is checked by *actually switching locale*, not by trusting
that the JSON encoder is locale-independent.
"""

from __future__ import annotations

import json
import locale
import math

import pytest

from fscompass_analysis import artifacts, fixtures, telemetry


@pytest.fixture(scope="module")
def codec_fixture():
    return fixtures.load(fixtures.TELEMETRY_CODEC)


@pytest.fixture(scope="module")
def envelope(example_event):
    return telemetry.TelemetryEnvelope.from_document(example_event)


def test_the_shipped_example_round_trips_byte_for_byte(example_event, envelope):
    line = telemetry.encode_event(envelope, example_event["payload"])
    decoded_envelope, decoded_payload = telemetry.decode_event(line)
    assert decoded_envelope == envelope
    assert decoded_payload == example_event["payload"]
    # Re-encoding the decoded form is stable, so a line hash is reproducible (§29.7, §37.2).
    assert telemetry.encode_event(decoded_envelope, decoded_payload) == line


def test_nonfinite_values_are_refused_by_the_encoder(envelope):
    """§22.2: "Encoders MUST fail rather than emit nonstandard literals"."""
    for value in (float("nan"), float("inf"), float("-inf")):
        with pytest.raises(telemetry.TelemetryCodecError):
            telemetry.encode_event(envelope, {"trueHeadingDeg": value})


def test_nonfinite_literals_are_refused_by_the_decoder(codec_fixture):
    """§22.2: "decoders MUST reject them". A permissive decoder makes a strict encoder
    pointless, because the corrupt file still enters the analysis."""
    for case in codec_fixture["rejectedDocuments"]:
        with pytest.raises(telemetry.TelemetryCodecError):
            telemetry.decode_event(case["line"])


def test_python_json_would_accept_nan_by_default():
    """The discrimination check: the rejection is deliberate, not inherited."""
    assert math.isnan(json.loads('{"x": NaN}')["x"])
    with pytest.raises(telemetry.TelemetryCodecError):
        telemetry.decode_event('{"schemaVersion":"1.0.0","payload":{"trueHeadingDeg":NaN}}')


def test_unavailable_is_null_plus_a_sibling_status_field(envelope):
    """§22.2's prescribed alternative to a nonfinite literal, exercised end to end."""
    line = telemetry.encode_event(
        envelope,
        {
            "providerErrorTermDeg": None,
            "providerErrorSource": "NONE",
            "displayQualityGrade": None,
            "boundCalibrationState": "CANDIDATE",
        },
    )
    _, payload = telemetry.decode_event(line)
    assert payload["providerErrorTermDeg"] is None
    assert payload["providerErrorSource"] == "NONE"


def test_property_keys_must_be_lower_camel_case(envelope):
    for key in ("schema_version", "SchemaVersion", "TRUE_HEADING", "_leading"):
        with pytest.raises(telemetry.TelemetryCodecError):
            telemetry.encode_event(envelope, {key: 1.0})


def test_nested_keys_are_checked_too(envelope):
    with pytest.raises(telemetry.TelemetryCodecError):
        telemetry.encode_event(envelope, {"spaceWeather": {"observation_time_utc": "x"}})


def test_event_types_are_lower_snake_case_and_from_the_fixed_namespace(
    codec_fixture, example_event
):
    """§22: event-type identifiers are a separate namespace from enum values."""
    assert example_event["eventType"] in telemetry.EVENT_TYPES
    for unknown in codec_fixture["unknownEventTypes"]:
        document = dict(example_event)
        document["eventType"] = unknown
        with pytest.raises(telemetry.TelemetryCodecError):
            telemetry.TelemetryEnvelope.from_document(document)


def test_enum_values_are_upper_snake_case(example_event):
    """§6/§22.2: stable ``UPPER_SNAKE_CASE`` everywhere, including fixtures."""
    payload = example_event["payload"]
    for field in (
        "providerId",
        "providerErrorSource",
        "resolvedReference",
        "referenceResolutionMethod",
        "magneticState",
        "measurementState",
        "trustAction",
        "gradeLimitedBy",
        "boundCalibrationState",
        "uncertaintyCoverageEvidenceState",
        "measurementMode",
        "placementMethod",
        "altitudeReference",
        "chargingState",
    ):
        value = payload[field]
        assert value == value.upper(), field
        assert " " not in value and "-" not in value, field


def test_the_export_path_runs_under_a_comma_decimal_locale(codec_fixture, envelope, example_event):
    """§22.2: "A test MUST run the export path under a comma-decimal locale".

    If no such locale is installed the test says so rather than passing silently — a skipped
    locale check reported as a pass is exactly the kind of quiet gap §33.1 is meant to close.
    """
    available = None
    original = locale.setlocale(locale.LC_ALL)
    try:
        candidates = list(codec_fixture["commaDecimalLocales"])
        candidates += [name.replace("UTF-8", "utf8") for name in codec_fixture["commaDecimalLocales"]]
        for candidate in candidates:
            try:
                locale.setlocale(locale.LC_ALL, candidate)
                available = candidate
                break
            except locale.Error:
                continue
        if available is None:
            pytest.skip(
                "no comma-decimal locale installed in this environment; §22.2 requires this "
                "check to run in CI on a host that has one"
            )
        assert locale.localeconv()["decimal_point"] == ","
        line = telemetry.encode_event(envelope, example_event["payload"])
        # The serialized numbers still use "." — a comma would either corrupt the value or
        # split the JSON object, and both parse differently on the other platform.
        assert '"declinationDeg":8.29' in line
        assert '"observedOrientationRateHz":48.7' in line
        assert '"declinationDeg":8,29' not in line
        _, payload = telemetry.decode_event(line)
        assert payload["declinationDeg"] == pytest.approx(8.29)
        assert payload["observedOrientationRateHz"] == pytest.approx(48.7)
        # And a locale-formatted number is not silently accepted on the way back in.
        with pytest.raises(Exception):
            telemetry.decode_event('{"schemaVersion":"1,0","payload":{"aDeg":8,29}}')
    finally:
        locale.setlocale(locale.LC_ALL, original)


def test_doubles_round_trip_at_full_precision(codec_fixture, envelope):
    """§22.2: shortest round-trip, and never through a ``Float``.

    ``359.9999998999999`` is in the set on purpose: a float32 round trip collapses it to
    ``360.0``, which the §9 normalization then maps to ``0.0`` — a full-circle error from a
    serialization choice.
    """
    values = codec_fixture["roundTripDoubles"]
    line = telemetry.encode_event(
        envelope, {f"value{index}Deg": value for index, value in enumerate(values)}
    )
    _, payload = telemetry.decode_event(line)
    for index, value in enumerate(values):
        assert payload[f"value{index}Deg"] == value, value

    import struct

    through_float32 = struct.unpack("f", struct.pack("f", 359.9999998999999))[0]
    assert through_float32 != 359.9999998999999
    assert payload["value6Deg"] == 359.9999998999999


def test_numeric_field_names_carry_their_unit(example_event):
    """§22.2's units-in-names rule, applied to the shipped example."""
    offenders = telemetry.numeric_fields_missing_unit_suffix(example_event)
    assert offenders == (), (
        "numeric fields without a unit suffix and not in "
        f"DOCUMENTED_DIMENSIONLESS_FIELDS: {offenders}"
    )


def test_the_units_rule_is_not_vacuous(example_event):
    """A rule that fires on nothing is not a rule."""
    document = dict(example_event)
    document["payload"] = dict(example_event["payload"])
    document["payload"]["magneticDeclination"] = 8.29
    assert "payload.magneticDeclination" in telemetry.numeric_fields_missing_unit_suffix(document)


def test_the_three_monotonic_timestamps_stay_distinct(example_event, envelope):
    """§22: three canonical monotonic timestamps, three meanings, never interchangeable."""
    assert envelope.source_monotonic_time_ns != envelope.arrival_monotonic_time_ns
    assert envelope.arrival_monotonic_time_ns != envelope.record_monotonic_time_ns
    assert envelope.source_monotonic_time_ns < envelope.arrival_monotonic_time_ns
    assert envelope.arrival_monotonic_time_ns < envelope.record_monotonic_time_ns


def test_wall_clock_is_rfc3339_utc_with_an_explicit_z(example_event):
    """§22.2: never mixed with monotonic nanoseconds in one field."""
    assert example_event["wallTimeUtc"].endswith("Z")
    for bad in ("2026-08-29T12:34:56.123456+02:00", "2026-08-29 12:34:56Z", "2026-08-29T12:34:56"):
        document = dict(example_event)
        document["wallTimeUtc"] = bad
        with pytest.raises(telemetry.TelemetryCodecError):
            telemetry.TelemetryEnvelope.from_document(document)


def test_monotonic_fields_must_be_integers(example_event):
    document = dict(example_event)
    document["sourceMonotonicTimeNs"] = 1234567889000.5
    with pytest.raises(telemetry.TelemetryCodecError):
        telemetry.TelemetryEnvelope.from_document(document)


def test_an_unidentified_source_clock_is_rejected(example_event):
    """Failure mode 10: freshness cannot be computed from a timestamp whose clock domain is
    unidentified."""
    document = dict(example_event)
    document["sourceClock"] = "SYSTEM_CLOCK"
    with pytest.raises(telemetry.TelemetryCodecError):
        telemetry.TelemetryEnvelope.from_document(document)


def test_the_envelope_carries_the_config_and_ruleset_provenance(example_event):
    """§37.2: ``config_version`` and ``config_hash`` MUST match the telemetry envelope for
    every event in a run, and a saved record without its ruleset is uninterpretable (fm 42)."""
    assert example_event["configVersion"] == "precision-v1-candidate-1"
    assert example_event["configHash"] == artifacts.sha256_of(artifacts.PRECISION_PROFILE)
    payload = example_event["payload"]
    assert payload["fengShuiRuleSetVersion"] == "fengshui-v1"
    assert payload["fengShuiRuleSetHash"] == artifacts.sha256_of(artifacts.FENG_SHUI_RULES)


def test_the_unvendored_wmm_hashes_are_recorded_as_not_vendored(example_event):
    """D-2: the example carries the literal ``NOT_VENDORED`` rather than a plausible digest."""
    payload = example_event["payload"]
    assert payload["declinationCoefficientSha256"] == "NOT_VENDORED"
    assert payload["declinationErrorModelSha256"] == "NOT_VENDORED"
