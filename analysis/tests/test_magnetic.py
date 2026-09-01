"""SPEC.md §16 — magnetic interference detection and the §11 reference precheck."""

from __future__ import annotations

import math

import pytest

from fscompass_analysis import fixtures, magnetic
from fscompass_analysis.enums import MagneticState, ReferenceMagneticPrecheckState


@pytest.fixture(scope="module")
def magnetic_fixture():
    return fixtures.load(fixtures.MAGNETIC_CLASSIFICATION)


@pytest.fixture(scope="module")
def thresholds(profile):
    return magnetic.MagneticThresholds.from_profile(profile)


def _features(case) -> magnetic.MagneticFeatures:
    return magnetic.MagneticFeatures(
        relative_magnitude_residual=case["relativeMagnitudeResidual"],
        inclination_residual_deg=case["inclinationResidualDeg"],
        stationary_field_mad_micro_tesla=case["stationaryFieldMadMicroTesla"],
        pipeline_agreement_deg=case["pipelineAgreementDeg"],
        sensor_saturated=case.get("sensorSaturated", False),
        os_calibration_invalid=case.get("osCalibrationInvalid", False),
        any_value_nonfinite=case.get("anyValueNonFinite", False),
    )


# --------------------------------------------------------------------------------------
# R60 — the mandatory minus sign
# --------------------------------------------------------------------------------------
def test_inclination_matches_the_frozen_fixture(magnetic_fixture):
    for case in magnetic_fixture["inclinationCases"]:
        observed = magnetic.measured_inclination_positive_down_deg(
            case["upMicroTesla"], case["magnitudeMicroTesla"]
        )
        assert observed == pytest.approx(
            case["expectedMeasuredInclinationPositiveDownDeg"], abs=1e-9
        ), case["id"]


def test_northern_hemisphere_field_gives_positive_down_inclination():
    """R60: canonical ENU ``Bup`` is positive **upward**; WMM ``I`` is positive **downward**.

    In the northern hemisphere the field points into the ground, so ``Bup`` is negative and
    the positive-down inclination must come out positive. Dropping the minus sign reverses
    the observed sign and can reject a clean northern field as disturbed.
    """
    observed = magnetic.measured_inclination_positive_down_deg(-43.9, 48.7)
    assert observed > 0.0
    without_the_minus_sign = math.degrees(math.asin(-43.9 / 48.7))
    assert without_the_minus_sign < 0.0
    assert observed == pytest.approx(-without_the_minus_sign, abs=1e-12)


def test_missing_minus_sign_would_reject_a_clean_field(thresholds):
    """The consequence, made concrete: a clean northern field at ``I = 64.4°``."""
    expected_wmm_inclination = 64.4
    correct = magnetic.measured_inclination_positive_down_deg(-43.9, 48.7)
    correct_residual = magnetic.inclination_residual_deg(correct, expected_wmm_inclination)
    assert abs(correct_residual) < thresholds.inclination_residual_suspect_deg

    sign_flipped = math.degrees(math.asin(-43.9 / 48.7))
    flipped_residual = sign_flipped - expected_wmm_inclination
    assert abs(flipped_residual) >= thresholds.inclination_residual_disturbed_deg


def test_inclination_residual_is_linear_not_circular():
    """§16: "Inclination cannot wrap; a circular difference there is a category error".

    A circular difference would rescale a residual near the poles; the linear one keeps the
    full magnitude visible.
    """
    assert magnetic.inclination_residual_deg(80.0, -80.0) == 160.0
    assert magnetic.inclination_residual_deg(-80.0, 80.0) == -160.0


def test_inclination_input_range_is_asserted():
    with pytest.raises(magnetic.MagneticFeatureError):
        magnetic.inclination_residual_deg(120.0, 0.0)
    with pytest.raises(magnetic.MagneticFeatureError):
        magnetic.inclination_residual_deg(0.0, float("nan"))


def test_inclination_clamps_before_asin():
    """Failure mode 6: a component can exceed the magnitude by a rounding bit."""
    assert magnetic.measured_inclination_positive_down_deg(
        -48.700000000000003, 48.7
    ) == pytest.approx(90.0, abs=1e-9)


def test_zero_or_negative_magnitude_is_rejected():
    with pytest.raises(magnetic.MagneticFeatureError):
        magnetic.measured_inclination_positive_down_deg(0.0, 0.0)
    with pytest.raises(magnetic.MagneticFeatureError):
        magnetic.relative_magnitude_residual(48.0, 0.0)


# --------------------------------------------------------------------------------------
# §16 classifier
# --------------------------------------------------------------------------------------
def test_classifier_matches_the_frozen_fixture(magnetic_fixture, thresholds):
    for case in magnetic_fixture["classifierCases"]:
        features = _features(case)
        assert (
            magnetic.classify_magnetic_state(features, thresholds).value
            == case["expectedMagneticState"]
        ), case["id"]
        assert (
            magnetic.reference_magnetic_precheck_state(features, thresholds).value
            == case["expectedPrecheckState"]
        ), case["id"]


def test_magnitude_alone_cannot_declare_a_field_clean(thresholds):
    """Failure mode 23: a disturbance that rotates the field vector with normal magnitude.

    This is the case producing a *confident wrong bearing* rather than an obviously broken
    one, and a magnitude-only detector misses it entirely.
    """
    rotated = magnetic.MagneticFeatures(
        relative_magnitude_residual=0.01,  # well inside the clean magnitude band
        inclination_residual_deg=14.0,
        stationary_field_mad_micro_tesla=0.4,
        pipeline_agreement_deg=1.0,
    )
    assert magnetic.classify_magnetic_state(rotated, thresholds) is MagneticState.DISTURBED


def test_absent_features_resolve_unknown_never_clean(thresholds):
    """§16: absent evidence is not zero. Both absence paths are covered.

    ``stationaryFieldMadMicroTesla`` is absent while the device is moving;
    ``pipelineAgreementDeg`` is absent with fewer than two valid active-axis pipelines.
    """
    moving = magnetic.MagneticFeatures(0.01, 0.5, None, 0.5)
    single_pipeline = magnetic.MagneticFeatures(0.01, 0.5, 0.4, None)
    for features in (moving, single_pipeline):
        assert magnetic.classify_magnetic_state(features, thresholds) is MagneticState.UNKNOWN


def test_disturbed_wins_over_absent_evidence(thresholds):
    """A present feature above its disturbed threshold still disturbs, whatever is missing."""
    features = magnetic.MagneticFeatures(0.6, None, None, None)
    assert magnetic.classify_magnetic_state(features, thresholds) is MagneticState.DISTURBED


def test_invalid_input_precedes_every_other_branch(thresholds):
    for kwargs in (
        {"sensor_saturated": True},
        {"os_calibration_invalid": True},
        {"any_value_nonfinite": True},
    ):
        features = magnetic.MagneticFeatures(0.01, 0.5, 0.4, 0.5, **kwargs)
        assert magnetic.classify_magnetic_state(features, thresholds) is MagneticState.INVALID


def test_thresholds_are_inclusive_at_the_boundary(profile, thresholds):
    """§16's pseudocode uses ``>=``, so a feature exactly at its threshold trips it."""
    at_suspect = magnetic.MagneticFeatures(
        profile["magneticMagnitudeResidualSuspectFraction"], 0.5, 0.4, 0.5
    )
    assert magnetic.classify_magnetic_state(at_suspect, thresholds) is MagneticState.SUSPECT
    at_disturbed = magnetic.MagneticFeatures(
        profile["magneticMagnitudeResidualDisturbedFraction"], 0.5, 0.4, 0.5
    )
    assert magnetic.classify_magnetic_state(at_disturbed, thresholds) is MagneticState.DISTURBED


# --------------------------------------------------------------------------------------
# R59 — the precheck stays acyclic
# --------------------------------------------------------------------------------------
def test_precheck_ignores_pipeline_agreement_entirely(thresholds):
    """R59 is Critical: reference resolution must not require a reference-resolved pipeline.

    Varying ``pipelineAgreementDeg`` across its whole range must not move the precheck, which
    is what keeps the dependency order ``precheck → §11 resolution → pipeline agreement →
    final MagneticState → lock`` acyclic.
    """
    outcomes = {
        magnetic.reference_magnetic_precheck_state(
            magnetic.MagneticFeatures(0.01, 0.5, 0.4, pipeline), thresholds
        )
        for pipeline in (None, 0.0, 0.5, 5.0, 50.0, 180.0)
    }
    assert outcomes == {ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE}


def test_precheck_is_unknown_when_its_own_evidence_is_absent(thresholds):
    features = magnetic.MagneticFeatures(0.01, None, 0.4, 0.5)
    assert (
        magnetic.reference_magnetic_precheck_state(features, thresholds)
        is ReferenceMagneticPrecheckState.UNKNOWN
    )


def test_precheck_and_final_state_are_recorded_separately(thresholds):
    """§16: they are different fields with different jobs, and can legitimately differ."""
    features = magnetic.MagneticFeatures(0.01, 0.5, 0.4, 12.0)
    assert (
        magnetic.reference_magnetic_precheck_state(features, thresholds)
        is ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE
    )
    assert magnetic.classify_magnetic_state(features, thresholds) is MagneticState.DISTURBED
