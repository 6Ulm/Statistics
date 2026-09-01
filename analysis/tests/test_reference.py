"""SPEC.md §11 — north-reference resolution, and the double-correction signature §30.5 hunts."""

from __future__ import annotations

import pytest

from fscompass_analysis import fixtures, reference
from fscompass_analysis.circular import absolute_circular_difference_deg, normalize360
from fscompass_analysis.enums import (
    GeomagneticModelId,
    MeasurementMode,
    ReferenceAxis,
    ReferenceMagneticPrecheckState,
    ReferenceResolutionMethod,
    ResolvedReference,
)


@pytest.fixture(scope="module")
def reference_fixture():
    return fixtures.load(fixtures.REFERENCE_RESOLUTION)


@pytest.fixture(scope="module")
def thresholds(profile):
    return reference.ReferenceResolutionThresholds.from_profile(profile)


def _hypotheses(case, eligible: bool = True) -> reference.GoogleReferenceHypotheses:
    return reference.GoogleReferenceHypotheses(
        measurement_mode=MeasurementMode(case["measurementMode"]),
        g_axis_deg=case["gAxisDeg"],
        m_axis_deg=case["mAxisDeg"],
        declination_deg=case["declinationDeg"],
        precheck_state=ReferenceMagneticPrecheckState(case["precheckState"]),
        geomagnetic_model_id=GeomagneticModelId.WMM2025,
        source_window_start_monotonic_ns=1_000,
        source_window_end_monotonic_ns=3_000,
        evidence_is_eligible=eligible,
    )


def test_resolution_matches_the_frozen_fixture(reference_fixture, thresholds):
    for case in reference_fixture["cases"]:
        result = reference.resolve_google_reference(_hypotheses(case), thresholds)
        assert result.resolved_reference.value == case["expectedResolvedReference"], case["id"]
        assert result.correction_deg == pytest.approx(
            case["expectedCorrectionDeg"], abs=1e-12
        ), case["id"]
        assert result.reference_ambiguity_bound_95_deg == pytest.approx(
            case["expectedReferenceAmbiguityBound95Deg"], abs=1e-12
        ), case["id"]
        if case["expectedCanonicalTrueHeadingDeg"] is None:
            assert result.canonical_true_heading_deg is None, case["id"]
        else:
            assert result.canonical_true_heading_deg == pytest.approx(
                case["expectedCanonicalTrueHeadingDeg"], abs=1e-9
            ), case["id"]
        if case.get("expectedResidualsAbsent"):
            assert result.reference_hypothesis_residual_true_deg is None
            assert result.reference_hypothesis_residual_magnetic_deg is None


def test_correction_is_exactly_zero_or_plus_declination(reference_fixture, thresholds):
    """§11: the **single** Google magnetic→true correction site.

    Failure mode 21 is a Critical failure whose signature is exactly ``2 x declination`` — a
    plausible bearing, roughly 16° off at a site with ``|d| = 8°``. Making the value a
    constructor invariant means no code path can produce anything else.
    """
    for case in reference_fixture["cases"]:
        result = reference.resolve_google_reference(_hypotheses(case), thresholds)
        assert result.correction_deg in (0.0, result.declination_deg), case["id"]


def test_the_double_correction_signature_is_detectable(thresholds):
    """The §30.5 test, in deterministic form: what a second application would look like."""
    declination = 8.29
    case = {
        "measurementMode": "FLAT_TOP_EDGE",
        "gAxisDeg": 180.71,
        "mAxisDeg": 180.71,
        "declinationDeg": declination,
        "precheckState": "CLEAN_FOR_REFERENCE",
    }
    result = reference.resolve_google_reference(_hypotheses(case), thresholds)
    assert result.resolved_reference is ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC
    correct = result.canonical_true_heading_deg
    doubly_corrected = normalize360(correct + declination)
    assert absolute_circular_difference_deg(doubly_corrected, correct) == pytest.approx(
        declination, abs=1e-9
    )
    # The signature §30.5 looks for is 2*d from the magnetic bearing, not d.
    assert absolute_circular_difference_deg(
        doubly_corrected, case["gAxisDeg"]
    ) == pytest.approx(2.0 * declination, abs=1e-9)


def test_ineligible_evidence_refuses_to_resolve(thresholds):
    """R59: the engine does not manufacture a reference to compute the evidence for one."""
    case = {
        "measurementMode": "FLAT_TOP_EDGE",
        "gAxisDeg": 189.0,
        "mAxisDeg": 180.71,
        "declinationDeg": 8.29,
        "precheckState": "CLEAN_FOR_REFERENCE",
    }
    result = reference.resolve_google_reference(_hypotheses(case, eligible=False), thresholds)
    assert result.resolved_reference is ResolvedReference.UNVERIFIED
    assert result.reference_resolution_method is ReferenceResolutionMethod.NOT_RESOLVED
    assert result.canonical_true_heading_deg is None


def test_unknown_precheck_also_refuses(thresholds):
    case = {
        "measurementMode": "FLAT_TOP_EDGE",
        "gAxisDeg": 189.0,
        "mAxisDeg": 180.71,
        "declinationDeg": 8.29,
        "precheckState": "UNKNOWN",
    }
    result = reference.resolve_google_reference(_hypotheses(case), thresholds)
    assert result.resolved_reference is ResolvedReference.UNVERIFIED


def test_there_is_no_declination_dead_band(profile, thresholds):
    """§8.1/§11: since ``rMag - rTrue <= abs(d)``, a separation margin above the ambiguity
    allowance would create a band that always resolves ``UNVERIFIED`` with no visible cause.

    Swept across the whole small-declination range, with the provider actually emitting true
    north, every declination resolves to *something* rather than silently failing.
    """
    assert profile["referenceSeparationMarginDeg"] <= profile["smallDeclinationAmbiguityMaxDeg"]
    step = profile["smallDeclinationAmbiguityMaxDeg"] / 20.0
    for index in range(41):
        declination = index * step
        case = {
            "measurementMode": "FLAT_TOP_EDGE",
            "gAxisDeg": 100.0,
            "mAxisDeg": normalize360(100.0 - declination),
            "declinationDeg": declination,
            "precheckState": "CLEAN_FOR_REFERENCE",
        }
        result = reference.resolve_google_reference(_hypotheses(case), thresholds)
        assert result.resolved_reference is not ResolvedReference.UNVERIFIED, declination
        assert result.canonical_true_heading_deg == pytest.approx(100.0, abs=1e-9)


def test_the_ambiguity_term_never_exceeds_the_declination(thresholds):
    """§11: hypothesis separation cannot exceed ``abs(d)``, so neither can the term."""
    for declination in (-2.0, -1.5, 0.0, 0.5, 1.5, 2.0):
        case = {
            "measurementMode": "WALL_FLUSH_BACK",
            "gAxisDeg": 100.0,
            "mAxisDeg": 100.0,
            "declinationDeg": declination,
            "precheckState": "CLEAN_FOR_REFERENCE",
        }
        result = reference.resolve_google_reference(_hypotheses(case), thresholds)
        assert result.reference_ambiguity_bound_95_deg <= abs(declination) + 1e-12


def test_the_result_is_bound_to_its_mode_and_axis(thresholds):
    """§11: "A flat result is not reusable for a wall pose or vice versa"."""
    flat = reference.resolve_google_reference(
        _hypotheses(
            {
                "measurementMode": "FLAT_TOP_EDGE",
                "gAxisDeg": 189.0,
                "mAxisDeg": 180.71,
                "declinationDeg": 8.29,
                "precheckState": "CLEAN_FOR_REFERENCE",
            }
        ),
        thresholds,
    )
    wall = reference.resolve_google_reference(
        _hypotheses(
            {
                "measurementMode": "WALL_FLUSH_BACK",
                "gAxisDeg": 189.0,
                "mAxisDeg": 180.71,
                "declinationDeg": 8.29,
                "precheckState": "CLEAN_FOR_REFERENCE",
            }
        ),
        thresholds,
    )
    assert flat.reference_axis is ReferenceAxis.PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION
    assert wall.reference_axis is ReferenceAxis.OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION
    assert flat.measurement_mode is not wall.measurement_mode


def test_the_resolver_never_writes_a_reported_bound(thresholds):
    """§11: "the resolver never writes or overwrites ``reportedBound95Deg``"."""
    result = reference.resolve_google_reference(
        _hypotheses(
            {
                "measurementMode": "FLAT_TOP_EDGE",
                "gAxisDeg": 100.0,
                "mAxisDeg": 100.0,
                "declinationDeg": 1.5,
                "precheckState": "CLEAN_FOR_REFERENCE",
            }
        ),
        thresholds,
    )
    assert not hasattr(result, "reported_bound_95_deg")
    assert not hasattr(result, "instrument_bound_95_deg")


# --------------------------------------------------------------------------------------
# R51 — the explicit non-Google contracts, rather than the Google resolver with fake inputs
# --------------------------------------------------------------------------------------
def test_apple_flat_uses_the_explicit_provider_contract():
    result = reference.apple_provider_contract_reference_resolution(
        MeasurementMode.FLAT_TOP_EDGE, 123.4, 8.29, GeomagneticModelId.WMM2025, 0, 1
    )
    assert result.reference_resolution_method is ReferenceResolutionMethod.PROVIDER_CONTRACT_EXPLICIT
    assert result.resolved_reference is ResolvedReference.TRUE_VERIFIED
    assert result.correction_deg == 0.0
    assert result.reference_ambiguity_bound_95_deg == 0.0
    assert result.canonical_true_heading_deg == pytest.approx(123.4)


def test_apple_wall_requires_the_frame_to_be_actually_active():
    """§12: the requested frame is an intention; the observed frame is the fact."""
    active = reference.apple_attitude_frame_reference_resolution(
        MeasurementMode.WALL_FLUSH_BACK, 200.0, 8.29, GeomagneticModelId.WMM2025, 0, 1, True
    )
    assert active.reference_resolution_method is ReferenceResolutionMethod.ATTITUDE_FRAME_EXPLICIT
    inactive = reference.apple_attitude_frame_reference_resolution(
        MeasurementMode.WALL_FLUSH_BACK, 200.0, 8.29, GeomagneticModelId.WMM2025, 0, 1, False
    )
    assert inactive.resolved_reference is ResolvedReference.UNVERIFIED
    assert inactive.canonical_true_heading_deg is None


def test_and_rv_applies_declination_once_and_never_claims_true_verified():
    """§30.4: no ``TRUE_VERIFIED`` without an independent reference check, and §11's
    ambiguity rule does not apply to this path."""
    result = reference.and_rv_reference_resolution(
        MeasurementMode.WALL_FLUSH_BACK, 355.0, 8.29, GeomagneticModelId.WMM2025, 0, 1
    )
    assert result.resolved_reference is ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC
    assert result.reference_resolution_method is ReferenceResolutionMethod.APP_APPLIED_DECLINATION
    assert result.correction_deg == 8.29
    assert result.reference_ambiguity_bound_95_deg == 0.0
    assert result.canonical_true_heading_deg == pytest.approx(3.29, abs=1e-9)


def test_a_correction_that_is_neither_zero_nor_declination_is_rejected():
    """The invariant is enforced by construction, not only by the resolver's own branches."""
    with pytest.raises(ValueError):
        reference.ReferenceResolutionResult(
            measurement_mode=MeasurementMode.FLAT_TOP_EDGE,
            reference_axis=ReferenceAxis.PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION,
            resolved_reference=ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
            reference_resolution_method=ReferenceResolutionMethod.APP_APPLIED_DECLINATION,
            declination_deg=8.29,
            correction_deg=16.58,  # the 2*d signature
            reference_ambiguity_bound_95_deg=0.0,
            geomagnetic_model_id=GeomagneticModelId.WMM2025,
            source_window_start_monotonic_ns=0,
            source_window_end_monotonic_ns=1,
        )
