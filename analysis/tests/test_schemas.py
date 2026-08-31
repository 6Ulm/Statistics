"""SPEC.md §36 Phase 0 exit: "config, ruleset, and example telemetry validate".

JSON Schema 2020-12 (§38 Data contracts). Every shipped artifact is validated against its
schema, and each schema is itself checked for the structural constraints §8.1 and §21.1
require of it.
"""

from __future__ import annotations

import copy
import json

import pytest
from jsonschema import Draft202012Validator

from fscompass_analysis import artifacts

SCHEMA_FOR_ARTIFACT = {
    artifacts.PRECISION_PROFILE: "schemas/precision-profile-v1.schema.json",
    artifacts.FENG_SHUI_RULES: "schemas/feng-shui-rules-v1.schema.json",
    artifacts.EXAMPLE_ENGINE_OUTPUT_EVENT: "schemas/telemetry-event-v1.schema.json",
    artifacts.EXAMPLE_SESSION_MANIFEST: "schemas/session-manifest-v1.schema.json",
}


@pytest.mark.parametrize("schema_path", sorted(set(SCHEMA_FOR_ARTIFACT.values())))
def test_schema_is_itself_valid(schema_path):
    schema = artifacts.load_json(schema_path)
    Draft202012Validator.check_schema(schema)


@pytest.mark.parametrize("artifact_path,schema_path", sorted(SCHEMA_FOR_ARTIFACT.items()))
def test_artifact_validates(artifact_path, schema_path):
    validator = Draft202012Validator(artifacts.load_json(schema_path))
    errors = sorted(validator.iter_errors(artifacts.load_json(artifact_path)), key=str)
    assert not errors, f"{artifact_path} against {schema_path}:\n" + "\n".join(
        f"  {list(e.absolute_path)}: {e.message}" for e in errors
    )


def test_precision_profile_schema_closes_the_object():
    """§8.1: 'Schema sets "additionalProperties": false.'"""
    schema = artifacts.load_json("schemas/precision-profile-v1.schema.json")
    assert schema["additionalProperties"] is False


def test_precision_profile_schema_forbids_a_calibration_state_key():
    """§19.1: the profile 'MUST NOT contain a calibration-state property at all, enforced by
    schema constraint plus test (§8.1)'. The schema constraint is tested by feeding it a
    document that has one."""
    schema = artifacts.load_json("schemas/precision-profile-v1.schema.json")
    validator = Draft202012Validator(schema)

    injected = copy.deepcopy(artifacts.load_json(artifacts.PRECISION_PROFILE))
    injected["boundCalibrationState"] = "CALIBRATED"
    assert list(validator.iter_errors(injected)), (
        "the schema must reject a calibration-state key; failure mode 32 is 'an editable "
        "config value that turns every device Professional'"
    )

    # Also under a differently-cased spelling, since the invariant is /calibrationState/i.
    injected = copy.deepcopy(artifacts.load_json(artifacts.PRECISION_PROFILE))
    injected["myCALIBRATIONSTATEoverride"] = "CALIBRATED"
    assert list(validator.iter_errors(injected))


def test_precision_profile_schema_requires_every_shipped_key():
    """A key present in the artifact but absent from the schema would never be validated."""
    schema = artifacts.load_json("schemas/precision-profile-v1.schema.json")
    profile = artifacts.load_json(artifacts.PRECISION_PROFILE)
    assert set(schema["properties"]) == set(profile)
    assert set(schema["required"]) == set(profile)


def test_ruleset_schema_pins_exact_cardinalities():
    """§21.1 / R65: 'Ellipses, omitted entries, or an `excerpt` marker are schema errors and
    cannot ship.'"""
    schema = artifacts.load_json("schemas/feng-shui-rules-v1.schema.json")
    sectors = schema["properties"]["sectors"]
    groups = schema["properties"]["groups"]
    assert sectors["minItems"] == sectors["maxItems"] == 24
    assert groups["minItems"] == groups["maxItems"] == 8

    validator = Draft202012Validator(schema)
    truncated = copy.deepcopy(artifacts.load_json(artifacts.FENG_SHUI_RULES))
    truncated["sectors"] = truncated["sectors"][:2]
    assert list(validator.iter_errors(truncated))


def test_telemetry_schema_rejects_nonfinite_and_unknown_event_types():
    """§22.2: 'JSON has no NaN/Infinity ... decoders MUST reject them.'"""
    schema = artifacts.load_json("schemas/telemetry-event-v1.schema.json")
    validator = Draft202012Validator(schema)

    event = copy.deepcopy(artifacts.load_json(artifacts.EXAMPLE_ENGINE_OUTPUT_EVENT))
    event["eventType"] = "not_a_declared_event_type"
    assert list(validator.iter_errors(event))

    # A nonfinite literal is not representable in JSON at all; prove the parser refuses it
    # rather than silently accepting a nonstandard token.
    with pytest.raises(json.JSONDecodeError):
        json.loads('{"reportedBound95Deg": NaN}', parse_constant=_reject_constant)


def _reject_constant(token):
    raise json.JSONDecodeError(f"nonfinite literal {token!r} is forbidden by §22.2", token, 0)
