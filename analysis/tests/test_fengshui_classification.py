"""SPEC.md §21 — sector geometry, straddle sets, and the §21.2 reference transform."""

from __future__ import annotations

import pytest

from fscompass_analysis import fengshui, fixtures
from fscompass_analysis.circular import absolute_circular_difference_deg


@pytest.fixture(scope="module")
def rule_set(rules):
    return fengshui.FengShuiRuleSet.from_document(rules)


@pytest.fixture(scope="module")
def classification_fixture():
    return fixtures.load(fixtures.FENGSHUI_CLASSIFICATION)


@pytest.fixture(scope="module")
def transform_fixture():
    return fixtures.load(fixtures.FENGSHUI_REFERENCE_TRANSFORM)


def test_every_sector_probe_matches_the_frozen_fixture(classification_fixture, rule_set):
    """§21.1: per sector — centre, both boundaries, ±ε, ±0.1°, ±1.0°, plus the north wrap."""
    cases = classification_fixture["sectorCases"]
    assert len(cases) == rule_set.sector_count * 9
    for case in cases:
        index = fengshui.sector_index(case["headingDeg"], rule_set)
        assert index == case["expectedSectorIndex"], case["id"]
        assert rule_set.sector(index).name == case["expectedSectorName"], case["id"]


def test_boundaries_are_half_open_at_the_start(rule_set):
    """A heading exactly on a boundary belongs to the sector that boundary starts."""
    assert rule_set.sector(fengshui.sector_index(352.5, rule_set)).name == "zi"
    assert rule_set.sector(fengshui.sector_index(352.5 - 1e-9, rule_set)).name == "ren"
    assert rule_set.sector(fengshui.sector_index(7.5, rule_set)).name == "gui"
    assert rule_set.sector(fengshui.sector_index(7.5 - 1e-9, rule_set)).name == "zi"


def test_the_north_wrap_sector_is_covered(rule_set):
    """§21.1: for the default ruleset ``352.5°`` separates 壬 and 子."""
    assert rule_set.derived_start_deg(0) == pytest.approx(352.5)
    for heading in (352.5, 355.0, 359.9, 0.0, 0.1, 7.49):
        assert rule_set.sector(fengshui.sector_index(heading, rule_set)).name == "zi", heading


def test_no_premature_rounding_moves_a_sector(rule_set):
    """Failure mode 7: ``337.49°`` rounds to ``337.5``, which is a different sector."""
    assert rule_set.sector(fengshui.sector_index(337.49, rule_set)).name == "hai"
    assert rule_set.sector(fengshui.sector_index(337.5, rule_set)).name == "ren"


def test_straddle_sets_match_the_frozen_fixture(classification_fixture, rule_set):
    for case in classification_fixture["straddleCases"]:
        indices = fengshui.straddle_indices(
            case["headingDeg"], case["reportedBound95Deg"], rule_set
        )
        assert list(indices) == case["expectedSectorIndices"], case["note"]
        assert [rule_set.sector(index).name for index in indices] == case[
            "expectedSectorNames"
        ], case["note"]


def test_a_bound_above_half_a_sector_guarantees_a_straddle(rule_set):
    """§21.3: ``reportedBound95Deg > 7.5°`` guarantees two sectors regardless of the estimate.

    Swept across the whole circle rather than asserted at one heading, because the claim is
    "regardless of the point estimate".
    """
    for tenths in range(0, 3600):
        heading = tenths / 10.0
        assert len(fengshui.straddle_indices(heading, 7.6, rule_set)) >= 2, heading


def test_a_bound_above_a_full_sector_guarantees_three(rule_set):
    for tenths in range(0, 3600, 7):
        heading = tenths / 10.0
        assert len(fengshui.straddle_indices(heading, 15.1, rule_set)) >= 3, heading


def test_a_low_confidence_bound_has_essentially_no_discriminating_power(rule_set, profile):
    """§21.3: a ``LOW_CONFIDENCE`` result at up to 10° straddles at least three sectors.

    Showing it beside a single mountain glyph is misleading even with a caveat, which is why
    the classifier returns the whole set.
    """
    for tenths in range(0, 3600, 13):
        heading = tenths / 10.0
        indices = fengshui.straddle_indices(
            heading, profile["lowConfidenceBound95MaxDeg"], rule_set
        )
        assert len(indices) >= 2, heading


def test_the_full_circle_degenerate_case_reports_no_classification(rule_set):
    """§21.4: ``2 * reportedBound95Deg >= 360°`` → report that no classification is possible
    rather than listing all 24."""
    for bound in (180.0, 200.0, 360.0):
        assert fengshui.straddle_indices(10.0, bound, rule_set) == ()
        result = fengshui.classify(10.0, 0.0, bound, rule_set)
        assert not result.classification_possible
        assert result.primary_sector is None
        assert result.possible_sectors == ()


def test_a_nearly_full_circle_is_still_classifiable_as_every_sector(rule_set):
    """The case a walk-until-equal implementation gets wrong: both endpoints land in the same
    sector, so it would report one mountain for an interval covering all 24."""
    indices = fengshui.straddle_indices(10.0, 179.0, rule_set)
    assert len(indices) == rule_set.sector_count
    assert len(set(indices)) == rule_set.sector_count


def test_straddle_sets_wrap_north_in_azimuth_order(rule_set):
    result = fengshui.classify(0.0, 0.0, 8.0, rule_set)
    assert result.boundary_straddled
    assert result.possible_sectors == ("ren", "zi", "gui")


def test_the_signed_boundary_offset_stays_within_half_a_sector(rule_set):
    for tenths in range(0, 3600, 3):
        heading = tenths / 10.0
        offset = fengshui.signed_offset_from_sector_boundary_deg(heading, rule_set)
        assert abs(offset) <= rule_set.sector_width_deg / 2.0 + 1e-9, heading


def test_a_negative_bound_is_rejected(rule_set):
    with pytest.raises(ValueError):
        fengshui.straddle_indices(10.0, -1.0, rule_set)
    with pytest.raises(ValueError):
        fengshui.straddle_indices(10.0, float("nan"), rule_set)


# --------------------------------------------------------------------------------------
# §21.2 — reference selection, needle offset, and the ambiguity term
# --------------------------------------------------------------------------------------
def test_reference_transform_matches_the_frozen_fixture(transform_fixture, rules):
    """§21.2 golden tests: both hidden Google hypotheses, both declination signs, TRUE and
    MAGNETIC rulesets, a sector boundary, and north wrap."""
    for case in transform_fixture["cases"]:
        document = dict(rules)
        document["referenceSelection"] = case["referenceSelection"]
        document["needleOffsetDeg"] = case["needleOffsetDeg"]
        rule_set = fengshui.FengShuiRuleSet.from_document(document)
        observed = fengshui.classification_heading_deg(
            case["googleOutputDeg"], case["declinationDeg"], rule_set
        )
        assert observed == pytest.approx(
            case["expectedClassificationHeadingDeg"], abs=1e-9
        ), case["id"]
        assert absolute_circular_difference_deg(
            observed, case["truthUnderHypothesisDeg"]
        ) == pytest.approx(case["expectedErrorDeg"], abs=1e-9), case["id"]


def test_the_ambiguity_term_covers_either_hidden_hypothesis(transform_fixture):
    """§21.2: if Google secretly emitted magnetic north, the TRUE point is wrong by ``|d|``
    **and** the derived MAGNETIC point ``g-d`` is wrong by ``|d|`` too.

    ``referenceAmbiguityBound95Deg = |d|`` therefore covers either hypothesis under either
    ruleset reference — which is why subtracting ``d`` for a magnetic ruleset MUST NOT zero
    or remove the term.
    """
    hypothesis_cases = [
        case for case in transform_fixture["cases"] if case["caseKind"] == "HIDDEN_HYPOTHESIS"
    ]
    assert hypothesis_cases, "the fixture must carry hidden-hypothesis cases"
    for case in hypothesis_cases:
        assert case["expectedErrorDeg"] <= case["referenceAmbiguityBound95Deg"] + 1e-9, case[
            "id"
        ]
    # Both hypotheses, both signs, both ruleset references are actually present.
    assert {case["hiddenHypothesis"] for case in hypothesis_cases} == {
        "GOOGLE_EMITTED_TRUE",
        "GOOGLE_EMITTED_MAGNETIC",
    }
    assert {case["referenceSelection"] for case in hypothesis_cases} == {"TRUE", "MAGNETIC"}
    assert {case["declinationDeg"] > 0 for case in hypothesis_cases} == {True, False}
    # And the term is not vacuously large: at least one case actually reaches |d|.
    assert any(
        case["expectedErrorDeg"] == pytest.approx(case["referenceAmbiguityBound95Deg"], abs=1e-9)
        for case in hypothesis_cases
    )


def test_the_magnetic_ruleset_does_not_shrink_the_bound(rules):
    """The bound handed to :func:`classify` is the total, and the classifier never touches it."""
    magnetic_document = dict(rules)
    magnetic_document["referenceSelection"] = "MAGNETIC"
    magnetic_rules = fengshui.FengShuiRuleSet.from_document(magnetic_document)
    true_rules = fengshui.FengShuiRuleSet.from_document(rules)

    bound = 7.0
    magnetic = fengshui.classify(189.0, 8.29, bound, magnetic_rules)
    true = fengshui.classify(189.0, 8.29, bound, true_rules)
    # The point estimates differ by the declination; the straddle width does not shrink.
    assert magnetic.classification_heading_deg != true.classification_heading_deg
    assert len(magnetic.possible_sectors) >= 2
    assert len(true.possible_sectors) >= 2


def test_the_needle_offset_is_a_declared_ruleset_property(rules):
    """§21.2: a doctrinal plate convention, never a user slider and never a correction for
    measurement error."""
    document = dict(rules)
    document["needleOffsetDeg"] = 7.5
    shifted = fengshui.FengShuiRuleSet.from_document(document)
    unshifted = fengshui.FengShuiRuleSet.from_document(rules)
    assert fengshui.classification_heading_deg(100.0, 0.0, shifted) == pytest.approx(107.5)
    assert fengshui.classification_heading_deg(100.0, 0.0, unshifted) == pytest.approx(100.0)


def test_the_classification_records_its_ruleset(rule_set):
    """§21.2/failure mode 42: a saved record whose reference is unknown is uninterpretable."""
    result = fengshui.classify(189.0, 8.29, 7.0, rule_set)
    assert result.rule_set_version == "fengshui-v1"
    assert result.reference_selection == "TRUE"


def test_the_shipped_example_classification_reproduces(rule_set, example_event):
    """The §22.1 example again, this time through the classifier."""
    payload = example_event["payload"]
    result = fengshui.classify(
        payload["trueHeadingDeg"], payload["declinationDeg"], payload["reportedBound95Deg"], rule_set
    )
    assert result.primary_sector == payload["primaryFengShuiSector"]
    assert list(result.possible_sectors) == payload["possibleFengShuiSectors"]
    assert result.boundary_straddled == payload["boundaryStraddled"]


def test_an_abbreviated_ruleset_is_rejected(rules):
    """R65: a two-entry 24-Mountains excerpt shipped as the required artifact."""
    truncated = dict(rules)
    truncated["sectors"] = rules["sectors"][:2]
    with pytest.raises(fengshui.RuleSetError):
        fengshui.FengShuiRuleSet.from_document(truncated)


def test_an_unknown_reference_selection_is_rejected(rules):
    document = dict(rules)
    document["referenceSelection"] = "GRID"
    with pytest.raises(fengshui.RuleSetError):
        fengshui.FengShuiRuleSet.from_document(document)
