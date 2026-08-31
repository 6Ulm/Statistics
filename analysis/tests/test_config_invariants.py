"""SPEC.md §8.1 "Enforced invariants" — analysis-runtime half of the Phase 0 gate."""

from __future__ import annotations

import copy

import pytest

from fscompass_analysis import config_invariants as ci


def test_profile_identifies_itself(profile):
    assert profile["schemaVersion"] == "1.0.0"
    assert profile["configVersion"] == "precision-v1-candidate-1"


def test_all_invariants_hold(profile):
    violations = ci.check(profile)
    assert not violations, "SPEC.md §8.1 invariant violations:\n" + "\n".join(
        f"  {v}" for v in violations
    )


def test_no_calibration_state_key_anywhere(profile):
    offending = [n for n in ci.iter_property_names(profile) if ci.CALIBRATION_STATE_KEY.search(n)]
    assert not offending, (
        "boundCalibrationState is derived from a §24 certification lookup (§19.1); a "
        f"configurable calibration state is failure mode 32. Offending: {offending}"
    )


def test_calibration_state_detection_is_not_vacuous():
    # A passing assertion over an absent key proves nothing unless the detector fires on a
    # document that does contain one, including at nesting depth.
    injected = {"a": 1, "nested": {"boundCalibrationState": "CALIBRATED"}}
    offending = [n for n in ci.iter_property_names(injected) if ci.CALIBRATION_STATE_KEY.search(n)]
    assert offending == ["boundCalibrationState"]


@pytest.mark.parametrize(
    "invariant_id,mutations",
    [
        (
            "INV-02-REFERENCE-SEPARATION-ORDERING",
            {"referenceSeparationMarginDeg": 99.0},
        ),
        ("INV-03-GRADE-THRESHOLD-ORDERING", {"highBound95MaxDeg": 1.0}),
        ("INV-04-FREEHAND-CANNOT-REACH-PROFESSIONAL", {"flatFreehandPlacementBound95Deg": 0.5}),
        ("INV-05-DECLINATION-ENVELOPE-ORDERING", {"declinationEnvelopeUsableMaxDeg": 0.01}),
        (
            "INV-06-SUSPECT-BELOW-DISTURBED-inclination",
            {"inclinationResidualSuspectDeg": 99.0},
        ),
        ("INV-07-PERIODIC-SUPPORT-SAMPLES-ACHIEVABLE", {"minPeriodicSupportSamples": 500}),
        ("INV-08-ORIENTATION-AGE-ORDERING", {"orientationMaxAgeMs": 500}),
        ("INV-09-LOCATION-FRESHNESS-ORDERING", {"freshLocationAtStartMaxAgeMs": 999999999}),
        ("INV-10-SPACE-WEATHER-ORDERING", {"spaceWeatherRejectKpMin": 1.0}),
        ("INV-01-NO-CALIBRATION-STATE-KEY", {"boundCalibrationState": "CALIBRATED"}),
    ],
)
def test_each_invariant_detects_its_own_violation(profile, invariant_id, mutations):
    """Every invariant must discriminate.

    An invariant that cannot fail is not a gate. The shipped file is never edited to make a
    test pass (§37 rule 12); these mutations are applied to an in-memory copy.
    """
    broken = copy.deepcopy(profile)
    broken.update(mutations)
    ids = [v.invariant_id for v in ci.check(broken)]
    assert invariant_id in ids, f"expected {invariant_id} in {ids}"
