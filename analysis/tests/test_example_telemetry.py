"""SPEC.md R62 and the §35 checklist line on executable examples.

"Every executable/display example passes production bound composition and reachability
checks; no uncertified flat-freehand fixture locks under the shipped 4° + 3° minimum."

The §22.1 example is deliberately a good-looking measurement — fresh confident provider,
clean field, verified reference, level device — that is still not a Precision Lock.
"""

from __future__ import annotations

import pytest

from fscompass_analysis import artifacts
from fscompass_analysis.grade_reachability import (
    CertificationState,
    MagneticState,
    PlacementMethod,
    QualityGrade,
    compute,
    quality_grade_for_reported_bound,
)


@pytest.fixture()
def payload(example_event):
    return example_event["payload"]


def test_bounds_compose_exactly(payload):
    """§19: reportedBound95Deg = min(180, instrument + placement)."""
    instrument = payload["instrumentBound95Deg"]
    placement = payload["placementBound95Deg"]
    assert payload["reportedBound95Deg"] == pytest.approx(min(180.0, instrument + placement))


def test_example_matches_the_shipped_uncertified_minimum(payload, profile):
    assert payload["measurementMode"] == "FLAT_TOP_EDGE"
    assert payload["placementMethod"] == "FREEHAND"
    assert payload["magneticState"] == "CLEAN"
    assert payload["boundCalibrationState"] == "CANDIDATE"

    computed = compute(
        PlacementMethod.FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile
    )
    assert payload["reportedBound95Deg"] == pytest.approx(computed.minimum_reported_bound_95_deg)
    assert payload["instrumentBound95Deg"] == pytest.approx(profile["unknownDeviceFloor95Deg"])
    assert payload["placementBound95Deg"] == pytest.approx(profile["flatFreehandPlacementBound95Deg"])


def test_example_degrades_and_is_not_a_lock(payload, profile):
    assert payload["measurementState"] == "DEGRADED"
    assert payload["trustAction"] == "SHOW_DEGRADED_RESULT"
    assert payload["provisionalQualityGrade"] == "LOW_CONFIDENCE"
    assert payload["displayQualityGrade"] is None, (
        "§19.1: a CANDIDATE consumer result MUST NOT show a standalone certified grade"
    )
    assert payload["reportedBound95Deg"] > profile["usableBound95MaxDeg"], (
        "§18.5: PRECISION_LOCKED requires reportedBound95Deg <= usableBound95MaxDeg"
    )
    assert (
        quality_grade_for_reported_bound(payload["reportedBound95Deg"], profile)
        is QualityGrade.LOW_CONFIDENCE
    )


def test_candidate_coverage_evidence_invariant(payload):
    """§19.1: CALIBRATED <=> EMPIRICALLY_CALIBRATED; CANDIDATE => {TARGET_ONLY, UNDEFINED}."""
    calibration = payload["boundCalibrationState"]
    evidence = payload["uncertaintyCoverageEvidenceState"]
    if calibration == "CALIBRATED":
        assert evidence == "EMPIRICALLY_CALIBRATED"
    else:
        assert calibration == "CANDIDATE"
        assert evidence in {"TARGET_ONLY", "UNDEFINED"}


def test_deviation_correction_is_none(payload):
    """§19.3: production default is NONE, correction exactly 0.0, corrected == uncorrected."""
    assert payload["deviationCorrectionState"] == "NONE"
    assert payload["deviationCorrectionDeg"] == 0.0
    assert payload["deviationCorrectionProfileHash"] == "NONE"
    assert payload["trueHeadingDeg"] == payload["uncorrectedTrueHeadingDeg"]


def test_straddle_is_reported(payload):
    """§21.3: a USABLE-or-wider bound straddles often enough that straddle is the primary
    layout; a 7.0° bound at 189.00° crosses the 187.5° boundary."""
    assert payload["boundaryStraddled"] is True
    assert payload["possibleFengShuiSectors"] == ["wu", "ding"]
    assert payload["primaryFengShuiSector"] in payload["possibleFengShuiSectors"]


def test_envelope_hashes_match_the_shipped_artifacts(example_event, payload):
    """§22 / §24: the config and ruleset hashes in a record must resolve to real artifacts."""
    assert example_event["configHash"] == artifacts.sha256_of(artifacts.PRECISION_PROFILE)
    assert payload["fengShuiRuleSetHash"] == artifacts.sha256_of(artifacts.FENG_SHUI_RULES)


def test_wmm_hashes_are_declared_not_vendored(payload):
    """The NOAA artifacts are not vendored yet (Phase 1). §5 forbids interchanging missing
    with zero or with a plausible value, so the example says NOT_VENDORED rather than
    carrying an invented hash."""
    assert payload["declinationCoefficientSha256"] == "NOT_VENDORED"
    assert payload["declinationErrorModelSha256"] == "NOT_VENDORED"


def test_session_manifest_config_identity_matches_the_envelope(example_event):
    """§37.2: 'config_version and config_hash MUST match the telemetry envelope for every
    event in the run. One identifier for the acceptance configuration.'"""
    manifest = artifacts.load_json(artifacts.EXAMPLE_SESSION_MANIFEST)
    assert manifest["configVersion"] == example_event["configVersion"]
    assert manifest["configHash"] == example_event["configHash"]
