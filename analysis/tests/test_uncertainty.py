"""SPEC.md §19 — uncertainty composition, both bounds, and ``gradeLimitedBy``."""

from __future__ import annotations

import pytest

from fscompass_analysis import fixtures, uncertainty
from fscompass_analysis.enums import GradeLimitingFactor, MagneticState, RejectionReason
from fscompass_analysis.grade_reachability import quality_grade_for_reported_bound


@pytest.fixture(scope="module")
def composition_fixture():
    return fixtures.load(fixtures.UNCERTAINTY_COMPOSITION)


def _terms(case) -> uncertainty.UncertaintyTerms:
    return uncertainty.UncertaintyTerms(
        provider_reported_bound_term_deg=case["providerReportedBoundTermDeg"],
        sample_bound_95_deg=case["sampleBound95Deg"],
        device_floor_95_deg=case["deviceFloor95Deg"],
        placement_bound_95_deg=case["placementBound95Deg"],
        declination_model_bound_95_deg=case.get("declinationModelBound95Deg"),
        location_time_sensitivity_bound_95_deg=case.get(
            "locationTimeSensitivityBound95Deg", 0.0
        ),
        reference_ambiguity_bound_95_deg=case.get("referenceAmbiguityBound95Deg", 0.0),
        interference_bound_95_deg=case.get("interferenceBound95Deg", 0.0),
        deviation_correction_residual_bound_95_deg=case.get(
            "deviationCorrectionResidualBound95Deg", 0.0
        ),
    )


def test_composition_matches_the_frozen_fixture(composition_fixture, profile):
    for case in composition_fixture["cases"]:
        composed = uncertainty.compose_bounds(_terms(case))
        assert composed.base_heading_bound_95_deg == pytest.approx(
            case["expectedBaseHeadingBound95Deg"], abs=1e-12
        ), case["id"]
        assert composed.instrument_bound_95_deg == pytest.approx(
            case["expectedInstrumentBound95Deg"], abs=1e-12
        ), case["id"]
        assert composed.reported_bound_95_deg == pytest.approx(
            case["expectedReportedBound95Deg"], abs=1e-12
        ), case["id"]
        assert composed.grade_limited_by.value == case["expectedGradeLimitedBy"], case["id"]

        # The lock/degraded distinction §18.5 draws on the **total** bound.
        lock_ceiling = profile["usableBound95MaxDeg"]
        expected_state = case["expectedMeasurementState"]
        locked = composed.reported_bound_95_deg <= lock_ceiling
        assert locked == (expected_state == "PRECISION_LOCKED"), case["id"]


def test_the_shipped_example_degrades_rather_than_locking(composition_fixture, example_event):
    """R62: the §22.1 example is an executable fixture, not decoration.

    Fresh confident provider, clean field, verified reference, level device — and still
    ``4.0 + 3.0 = 7.0°``, ``DEGRADED``/``LOW_CONFIDENCE``/``CANDIDATE``. A candidate state
    does not waive the unknown-device floor.
    """
    case = next(
        entry
        for entry in composition_fixture["cases"]
        if entry["id"] == "example-uncertified-flat-freehand-degrades"
    )
    composed = uncertainty.compose_bounds(_terms(case))
    assert composed.reported_bound_95_deg == 7.0
    payload = example_event["payload"]
    assert payload["instrumentBound95Deg"] == composed.instrument_bound_95_deg
    assert payload["placementBound95Deg"] == case["placementBound95Deg"]
    assert payload["reportedBound95Deg"] == composed.reported_bound_95_deg
    assert payload["gradeLimitedBy"] == composed.grade_limited_by.value
    assert payload["measurementState"] == "DEGRADED"
    assert payload["boundCalibrationState"] == "CANDIDATE"
    assert payload["displayQualityGrade"] is None


def test_an_absent_provider_term_is_not_zero_evidence(profile):
    """§19/failure mode 28: absence is absence.

    A ``0.0`` provider term would be indistinguishable in a ``max`` from a real, excellent
    measurement — and on the wall paths, which expose no documented degree error at all, it
    would silently claim evidence that does not exist (R61).
    """
    absent = uncertainty.UncertaintyTerms(
        provider_reported_bound_term_deg=None,
        sample_bound_95_deg=0.4,
        device_floor_95_deg=4.0,
        placement_bound_95_deg=3.0,
    )
    zero = uncertainty.UncertaintyTerms(
        provider_reported_bound_term_deg=0.0,
        sample_bound_95_deg=0.4,
        device_floor_95_deg=4.0,
        placement_bound_95_deg=3.0,
    )
    # Numerically the max is the same here; what differs is that the absent case never names
    # PROVIDER_ERROR as the limiting factor, because there is no provider evidence to blame.
    assert uncertainty.compose_bounds(absent).instrument_bound_95_deg == 4.0
    assert uncertainty.compose_bounds(zero).instrument_bound_95_deg == 4.0
    assert (
        uncertainty.compose_bounds(absent).grade_limited_by is GradeLimitingFactor.DEVICE_FLOOR
    )


def test_adding_a_provider_term_never_lowers_the_base(profile):
    """R63: the base is a ``max``, so the RV term can only leave it or raise it.

    Property-checked across the range rather than at one point, because "it happened not to
    lower it in my example" is not the claim §35 requires.
    """
    floor = profile["unknownDeviceFloor95Deg"]
    baseline = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(None, 0.5, floor, placement_bound_95_deg=3.0)
    )
    previous = baseline.base_heading_bound_95_deg
    for term_tenths in range(0, 200, 5):
        term = term_tenths / 10.0
        composed = uncertainty.compose_bounds(
            uncertainty.UncertaintyTerms(term, 0.5, floor, placement_bound_95_deg=3.0)
        )
        assert composed.base_heading_bound_95_deg >= baseline.base_heading_bound_95_deg
        assert composed.base_heading_bound_95_deg >= previous
        previous = composed.base_heading_bound_95_deg


def test_only_certification_can_lower_the_floor(profile):
    """R63's second half: reachability improves through a certified floor, a smaller
    validated placement bound, or a changed lock ceiling — never through a provider term."""
    unknown_floor = profile["unknownDeviceFloor95Deg"]
    lock_ceiling = profile["usableBound95MaxDeg"]
    placement = profile["flatFreehandPlacementBound95Deg"]
    uncertified = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(0.1, 0.1, unknown_floor, placement_bound_95_deg=placement)
    )
    assert uncertified.reported_bound_95_deg > lock_ceiling
    certified = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(0.1, 0.1, 1.2, placement_bound_95_deg=0.5)
    )
    assert certified.reported_bound_95_deg <= lock_ceiling


def test_base_terms_take_a_max_and_the_rest_add():
    """§19: the asymmetry is deliberate — same quantity vs different, additive sources."""
    composed = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(
            provider_reported_bound_term_deg=1.0,
            sample_bound_95_deg=2.0,
            device_floor_95_deg=1.5,
            declination_model_bound_95_deg=0.4,
            location_time_sensitivity_bound_95_deg=0.05,
            reference_ambiguity_bound_95_deg=1.5,
            interference_bound_95_deg=0.0,
            placement_bound_95_deg=0.5,
        )
    )
    assert composed.base_heading_bound_95_deg == 2.0  # max(1.0, 2.0, 1.5)
    assert composed.instrument_bound_95_deg == pytest.approx(2.0 + 0.4 + 0.05 + 1.5)
    assert composed.reported_bound_95_deg == pytest.approx(composed.instrument_bound_95_deg + 0.5)


def test_both_bounds_are_capped_at_180():
    composed = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(
            provider_reported_bound_term_deg=170.0,
            sample_bound_95_deg=1.0,
            device_floor_95_deg=1.0,
            reference_ambiguity_bound_95_deg=100.0,
            placement_bound_95_deg=50.0,
        )
    )
    assert composed.instrument_bound_95_deg == uncertainty.MAX_BOUND_DEG
    assert composed.reported_bound_95_deg == uncertainty.MAX_BOUND_DEG


def test_placement_is_never_zero():
    """§18.5: "Placement uncertainty | finite bound from method | ... **never zero**".

    §20: an implementation reaching Professional freehand has dropped or falsified the
    placement term — a certification failure, not a feature.
    """
    with pytest.raises(ValueError) as raised:
        uncertainty.UncertaintyTerms(1.0, 0.5, 4.0, placement_bound_95_deg=0.0)
    assert "never zero" in str(raised.value)


def test_reported_bound_is_always_at_least_the_instrument_bound():
    for placement in (0.1, 0.5, 3.0, 5.0):
        composed = uncertainty.compose_bounds(
            uncertainty.UncertaintyTerms(None, 0.5, 1.0, placement_bound_95_deg=placement)
        )
        assert composed.reported_bound_95_deg >= composed.instrument_bound_95_deg


# --------------------------------------------------------------------------------------
# §19 interference term
# --------------------------------------------------------------------------------------
def test_interference_term_by_magnetic_state(profile):
    suspect_bound = profile["suspectInterferenceBound95Deg"]
    assert uncertainty.interference_bound_95_deg(MagneticState.CLEAN, suspect_bound) == 0.0
    assert (
        uncertainty.interference_bound_95_deg(MagneticState.SUSPECT, suspect_bound)
        == suspect_bound
    )
    for state, reason in (
        (MagneticState.DISTURBED, RejectionReason.MAGNETIC_FIELD_DISTURBED),
        (MagneticState.INVALID, RejectionReason.MAGNETIC_CALIBRATION_INVALID),
        (MagneticState.UNKNOWN, RejectionReason.MAGNETIC_FIELD_UNKNOWN),
    ):
        with pytest.raises(uncertainty.InterferenceRejection) as raised:
            uncertainty.interference_bound_95_deg(state, suspect_bound)
        assert raised.value.reason is reason


def test_suspect_prevents_a_freehand_lock_outright(profile):
    """§8.1.1 row 3, reproduced arithmetically rather than asserted in prose.

    ``SUSPECT`` does not merely cap the grade freehand: the 3.0° term alone exceeds the 2.0°
    flat-freehand instrument budget, so no sensor quality can recover a lock.
    """
    budget = profile["usableBound95MaxDeg"] - profile["flatFreehandPlacementBound95Deg"]
    assert profile["suspectInterferenceBound95Deg"] > budget
    best_possible = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(
            provider_reported_bound_term_deg=0.0,
            sample_bound_95_deg=0.0,
            device_floor_95_deg=0.0,  # a hypothetically perfect certified device
            interference_bound_95_deg=profile["suspectInterferenceBound95Deg"],
            placement_bound_95_deg=profile["flatFreehandPlacementBound95Deg"],
        )
    )
    assert best_possible.reported_bound_95_deg > profile["usableBound95MaxDeg"]


# --------------------------------------------------------------------------------------
# §19 gradeLimitedBy
# --------------------------------------------------------------------------------------
def test_policy_ceilings_take_the_fixed_precedence():
    """§19: ``CERTIFICATION_CEILING -> SPACE_WEATHER -> CHARGING_STATE``, before any term."""
    terms = uncertainty.UncertaintyTerms(None, 0.5, 4.0, placement_bound_95_deg=3.0)
    assert (
        uncertainty.compose_bounds(
            terms,
            frozenset(
                {
                    GradeLimitingFactor.CHARGING_STATE,
                    GradeLimitingFactor.SPACE_WEATHER,
                    GradeLimitingFactor.CERTIFICATION_CEILING,
                }
            ),
        ).grade_limited_by
        is GradeLimitingFactor.CERTIFICATION_CEILING
    )
    assert (
        uncertainty.compose_bounds(
            terms,
            frozenset(
                {GradeLimitingFactor.CHARGING_STATE, GradeLimitingFactor.SPACE_WEATHER}
            ),
        ).grade_limited_by
        is GradeLimitingFactor.SPACE_WEATHER
    )
    assert (
        uncertainty.compose_bounds(
            terms, frozenset({GradeLimitingFactor.CHARGING_STATE})
        ).grade_limited_by
        is GradeLimitingFactor.CHARGING_STATE
    )


def test_charging_state_is_in_the_enum():
    """R57: the value was emitted as ``gradeLimitedBy`` while missing from the enum."""
    assert GradeLimitingFactor.CHARGING_STATE.value == "CHARGING_STATE"


def test_placement_is_named_when_it_dominates():
    """§21.5/§20: naming the placement term tells the user what would actually help."""
    composed = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(None, 0.2, 0.5, placement_bound_95_deg=3.0)
    )
    assert composed.grade_limited_by is GradeLimitingFactor.PLACEMENT_UNCERTAINTY


def test_exact_ties_resolve_by_stable_enum_order():
    """Two runtimes must not disagree about which of two equal terms is named."""
    composed = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(
            provider_reported_bound_term_deg=None,
            sample_bound_95_deg=2.0,
            device_floor_95_deg=2.0,
            placement_bound_95_deg=2.0,
        )
    )
    order = list(GradeLimitingFactor)
    assert order.index(composed.grade_limited_by) == min(
        order.index(GradeLimitingFactor.PLACEMENT_UNCERTAINTY),
        order.index(GradeLimitingFactor.SAMPLE_DISPERSION),
        order.index(GradeLimitingFactor.DEVICE_FLOOR),
    )


def test_grades_come_from_the_reported_bound_not_the_instrument_bound(profile):
    """§20/failure mode 30: grading on ``instrumentBound95Deg`` would advertise precision the
    practitioner cannot physically realize."""
    composed = uncertainty.compose_bounds(
        uncertainty.UncertaintyTerms(None, 0.2, 1.5, placement_bound_95_deg=3.0)
    )
    assert composed.instrument_bound_95_deg == 1.5
    assert composed.reported_bound_95_deg == 4.5
    on_reported = quality_grade_for_reported_bound(composed.reported_bound_95_deg, profile)
    on_instrument = quality_grade_for_reported_bound(composed.instrument_bound_95_deg, profile)
    assert on_reported.value == "USABLE" or on_reported.name == "USABLE"
    assert on_instrument.name == "PROFESSIONAL"
    assert on_reported is not on_instrument
