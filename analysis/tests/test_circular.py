"""SPEC.md §9 / §9.1 / §15 — circular utilities and pinned estimators, analysis runtime.

Two kinds of check run here and both are needed. The **fixture** checks prove this runtime
agrees with the frozen ``fixtures-v1`` contract the Kotlin and Swift runtimes read. The
**spec-literal** checks assert the values SPEC.md states in prose, so a regenerated fixture
carrying a wrong expectation cannot make a broken implementation pass.
"""

from __future__ import annotations

import math

import pytest

from fscompass_analysis import circular, fixtures


@pytest.fixture(scope="module")
def circular_fixture():
    return fixtures.load(fixtures.CIRCULAR_MATH)


@pytest.fixture(scope="module")
def estimator_fixture():
    return fixtures.load(fixtures.ESTIMATORS)


@pytest.fixture(scope="module")
def aggregate_fixture():
    return fixtures.load(fixtures.CIRCULAR_AGGREGATE)


# --------------------------------------------------------------------------------------
# normalize360 (§9)
# --------------------------------------------------------------------------------------
def test_normalize360_matches_the_frozen_fixture(circular_fixture):
    tolerance = circular_fixture["workingAngleToleranceDeg"]
    for case in circular_fixture["normalize360"]:
        observed = circular.normalize360(case["inputDeg"])
        assert observed == pytest.approx(case["expectedDeg"], abs=tolerance), case


@pytest.mark.parametrize(
    ("value", "expected"),
    [(-360.0, 0.0), (-0.0, 0.0), (360.0, 0.0), (0.0, 0.0), (-1.0, 359.0)],
)
def test_normalize360_spec_literal_cases(value, expected):
    """§9 names these cases explicitly: ``-360``, ``-0.0``, ``359.9999999``, ``360.0``."""
    assert circular.normalize360(value) == expected


def test_normalize360_359_9999999_keeps_the_documented_residual():
    """See ``docs/IMPLEMENTATION_NOTES.md`` F-1.

    The mandated ``((x % 360) + 360) % 360`` is not bit-exact for a value already in range;
    the residual is ~1e-10, three orders inside the declared 1e-6 tolerance. The value is
    pinned so a change that *enlarged* it surfaces here rather than as a bearing.
    """
    observed = circular.normalize360(359.9999999)
    assert observed == pytest.approx(359.9999999, abs=1e-9)
    assert abs(observed - 359.9999999) < 1e-9


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_normalize360_rejects_nonfinite(value):
    with pytest.raises(circular.AngleError):
        circular.normalize360(value)


# --------------------------------------------------------------------------------------
# shortestSignedDifferenceDeg (§3, §9) — the single normative contract
# --------------------------------------------------------------------------------------
def test_signed_difference_matches_the_frozen_fixture(circular_fixture):
    tolerance = circular_fixture["workingAngleToleranceDeg"]
    for case in circular_fixture["shortestSignedDifferenceDeg"]:
        observed = circular.shortest_signed_difference_deg(case["aDeg"], case["bDeg"])
        assert observed == pytest.approx(case["expectedDeg"], abs=tolerance), case


@pytest.mark.parametrize(
    ("a", "b"), [(0.0, 180.0), (180.0, 0.0), (90.0, 270.0), (270.0, 90.0), (120.0, 300.0), (300.0, 120.0)]
)
def test_antipode_is_plus_180_in_both_orderings(a, b):
    """§3/§35: ``+180``, never ``-180``, for **both orderings** of at least two distinct pairs.

    A test that only checks ``deltaDeg(180, 0)`` passes on a broken implementation, because
    raw ``atan2`` returns ``-180.0`` only for the ordering whose ``sin`` lands on a tiny
    negative.
    """
    assert circular.shortest_signed_difference_deg(a, b) == 180.0


def test_raw_atan2_would_fail_this_contract():
    """The discrimination check: the prohibited formula is shown to break the contract.

    §33.1 permits documentation and tests to *quote* a prohibited formula as text. Computing
    it here proves the antipode normalization in the real implementation is load-bearing
    rather than decorative.
    """
    prohibited = math.degrees(
        math.atan2(math.sin(math.radians(0.0 - 180.0)), math.cos(math.radians(0.0 - 180.0)))
    )
    assert prohibited == -180.0
    assert circular.shortest_signed_difference_deg(0.0, 180.0) == 180.0


def test_signed_difference_range_is_half_open_at_minus_180():
    for a in range(0, 360):
        for b in (0, 37, 180, 359):
            delta = circular.shortest_signed_difference_deg(float(a), float(b))
            assert -180.0 < delta <= 180.0


def test_signed_difference_is_antisymmetric_away_from_the_antipode():
    for a, b in [(10.0, 350.0), (45.0, 44.0), (0.0, 90.0), (200.0, 15.0)]:
        forward = circular.shortest_signed_difference_deg(a, b)
        backward = circular.shortest_signed_difference_deg(b, a)
        assert forward == pytest.approx(-backward, abs=1e-12)


@pytest.mark.parametrize("value", [float("nan"), float("inf"), float("-inf")])
def test_signed_difference_rejects_nonfinite(value):
    with pytest.raises(circular.AngleError):
        circular.shortest_signed_difference_deg(value, 0.0)
    with pytest.raises(circular.AngleError):
        circular.shortest_signed_difference_deg(0.0, value)


# --------------------------------------------------------------------------------------
# The exact delegating wrappers (§9, R68)
# --------------------------------------------------------------------------------------
def test_target_delta_matches_the_frozen_fixture(circular_fixture):
    tolerance = circular_fixture["workingAngleToleranceDeg"]
    for case in circular_fixture["shortestTargetDeltaDeg"]:
        observed = circular.shortest_target_delta_deg(case["currentDeg"], case["targetDeg"])
        assert observed == pytest.approx(case["expectedDeg"], abs=tolerance), case


def test_target_delta_is_exactly_the_specified_delegation():
    """``shortestTargetDeltaDeg(cur, target) = shortestSignedDifferenceDeg(target, cur)``."""
    for current, target in [(359.0, 1.0), (1.0, 359.0), (90.0, 275.0), (0.0, 180.0)]:
        assert circular.shortest_target_delta_deg(
            current, target
        ) == circular.shortest_signed_difference_deg(target, current)


def test_target_delta_sign_convention_across_north_wrap():
    """§18.2: positive = clockwise/right. From ``359°`` to ``1°`` is a short right turn."""
    assert circular.shortest_target_delta_deg(359.0, 1.0) > 0.0
    assert circular.shortest_target_delta_deg(1.0, 359.0) < 0.0
    assert abs(circular.shortest_target_delta_deg(359.0, 1.0)) == pytest.approx(2.0, abs=1e-9)


def test_absolute_difference_matches_the_frozen_fixture(circular_fixture):
    tolerance = circular_fixture["workingAngleToleranceDeg"]
    for case in circular_fixture["absoluteCircularDifferenceDeg"]:
        observed = circular.absolute_circular_difference_deg(case["aDeg"], case["bDeg"])
        assert observed == pytest.approx(case["expectedDeg"], abs=tolerance), case


def test_absolute_difference_is_exactly_the_specified_delegation():
    for a, b in [(10.0, 350.0), (0.0, 180.0), (180.0, 0.0), (300.0, 120.0)]:
        assert circular.absolute_circular_difference_deg(a, b) == abs(
            circular.shortest_signed_difference_deg(a, b)
        )
        assert 0.0 <= circular.absolute_circular_difference_deg(a, b) <= 180.0


# --------------------------------------------------------------------------------------
# §9.1 pinned estimators
# --------------------------------------------------------------------------------------
def test_quantile_and_median_match_the_frozen_fixture(estimator_fixture):
    for case in estimator_fixture["quantile"]:
        assert circular.quantile(case["values"], case["probability"]) == case["expected"], case
    for case in estimator_fixture["median"]:
        assert circular.median(case["values"]) == case["expected"], case


def test_quantile_is_nearest_rank_not_an_interpolating_estimator():
    """§9.1: common library estimators disagree by a full sample position at these sizes.

    With ``n = 20`` the nearest-rank P95 is ``x[18]``. A linear-interpolation estimator would
    return ``19.05``; a device-computed P95 and a report-computed P95 of the same window must
    agree *exactly*, so the estimator is pinned rather than inherited.
    """
    values = [float(index) for index in range(1, 21)]
    assert circular.quantile(values, 0.95) == 19.0
    assert circular.quantile(values, 1.0) == 20.0
    assert circular.quantile(values, 0.0) == 1.0


def test_median_even_and_odd():
    assert circular.median([3.0, 1.0, 2.0]) == 2.0
    assert circular.median([1.0, 2.0, 3.0, 4.0]) == 2.5


def test_estimators_are_undefined_on_empty_input():
    """§9.1: a typed UNDEFINED, never indexing element zero."""
    with pytest.raises(circular.UndefinedResult):
        circular.quantile([], 0.95)
    with pytest.raises(circular.UndefinedResult):
        circular.median([])


def test_estimators_reject_nonfinite_before_sorting():
    with pytest.raises(circular.AngleError):
        circular.quantile([1.0, float("nan"), 3.0], 0.5)
    with pytest.raises(circular.AngleError):
        circular.median([1.0, float("inf")])


def test_quantile_rejects_a_probability_outside_the_unit_interval():
    with pytest.raises(ValueError):
        circular.quantile([1.0], 1.5)


# --------------------------------------------------------------------------------------
# §15 circular aggregation
# --------------------------------------------------------------------------------------
def test_circular_aggregate_matches_the_frozen_fixture(aggregate_fixture):
    for window in aggregate_fixture["windows"]:
        aggregate = circular.circular_aggregate(window["samples"])
        assert aggregate.is_defined == window["meanIsDefined"], window["id"]
        assert aggregate.resultant_length == pytest.approx(
            window["expectedResultantLength"], abs=1e-12
        ), window["id"]
        if window["meanIsDefined"]:
            assert aggregate.mean_deg == pytest.approx(
                window["expectedMeanDeg"], abs=1e-9
            ), window["id"]
            assert circular.circular_residual_quantile_deg(
                window["samples"], 0.95
            ) == pytest.approx(window["expectedResidualP95Deg"], abs=1e-9), window["id"]


def test_circular_mean_crosses_north_without_linear_averaging():
    """Failure mode 1: linear averaging across north turns ``359`` and ``1`` into ``180``."""
    mean = circular.circular_mean_deg([359.0, 1.0])
    assert mean == pytest.approx(0.0, abs=1e-9) or mean == pytest.approx(360.0, abs=1e-9)
    assert circular.circular_mean_deg([359.0, 359.5, 0.0, 0.5, 1.0]) == pytest.approx(
        0.0, abs=1e-9
    )


def test_a_weak_resultant_is_an_explicit_failure_not_a_north_reading(profile):
    """§15 decision 3 / failure mode 6, in the shape it actually takes in floating point.

    ``atan2(0, 0)`` returning zero is the textbook case, but an antipodal window does not
    reach it: ``sin(0) + sin(180°)`` cancels to ``1.2e-16``, not to ``0``, so the mean is
    numerically *defined* and comes back as a confident ``90°`` that means nothing. Only the
    configured ``minCircularResultantLength`` gate catches that, which is exactly why §15
    states the rule on ``R`` rather than on the mean.
    """
    minimum = profile["minCircularResultantLength"]
    for samples in ([0.0, 180.0], [0.0, 90.0, 180.0, 270.0]):
        aggregate = circular.circular_aggregate(samples)
        assert aggregate.resultant_length == pytest.approx(0.0, abs=1e-12)
        assert circular.circular_mean_is_undefined(aggregate, minimum), samples
    tight = circular.circular_aggregate([84.7, 85.3, 84.9, 85.4, 85.0])
    assert not circular.circular_mean_is_undefined(tight, minimum)


def test_circular_mean_is_undefined_on_an_empty_window(profile):
    aggregate = circular.circular_aggregate([])
    assert not aggregate.is_defined
    assert circular.circular_mean_is_undefined(aggregate, profile["minCircularResultantLength"])
    with pytest.raises(circular.UndefinedResult):
        circular.circular_mean_deg([])


def test_resultant_gate_matches_the_frozen_fixture(aggregate_fixture):
    minimum = aggregate_fixture["minCircularResultantLength"]
    for window in aggregate_fixture["windows"]:
        aggregate = circular.circular_aggregate(window["samples"])
        assert (
            circular.circular_mean_is_undefined(aggregate, minimum)
            == window["expectedCircularMeanUndefinedUnderGate"]
        ), window["id"]


def test_resultant_length_is_bounded_and_maximal_for_identical_samples():
    assert circular.circular_resultant_length([42.0] * 10) == pytest.approx(1.0, abs=1e-12)
    assert 0.0 <= circular.circular_resultant_length([0.0, 5.0, 355.0]) <= 1.0


def test_empty_window_has_no_mean():
    aggregate = circular.circular_aggregate([])
    assert aggregate.count == 0
    assert not aggregate.is_defined


def test_no_trimming_inside_the_window():
    """§15 decision 2: once accepted, a sample counts.

    Trimming would let the window discard exactly the evidence that it is unreliable, and
    would make the dispersion gate and the dispersion-derived bound disagree about which
    samples exist.
    """
    steady = [85.0] * 19
    with_outlier = steady + [95.0]
    assert circular.circular_residual_quantile_deg(
        with_outlier, 0.95
    ) > circular.circular_residual_quantile_deg(steady, 0.95)


# --------------------------------------------------------------------------------------
# §19.2 boundFromSigma
# --------------------------------------------------------------------------------------
def test_bound_from_sigma_is_the_single_conversion(profile):
    factor = profile["declinationSigmaToBound95Factor"]
    assert factor == 1.96
    assert circular.bound_from_sigma(0.36, factor) == pytest.approx(0.7056, abs=1e-12)
    assert circular.bound_from_sigma(0.0, factor) == 0.0


def test_bound_from_sigma_rejects_invalid_inputs(profile):
    factor = profile["declinationSigmaToBound95Factor"]
    with pytest.raises(circular.AngleError):
        circular.bound_from_sigma(-0.1, factor)
    with pytest.raises(ValueError):
        circular.bound_from_sigma(0.36, 0.0)
    with pytest.raises(circular.AngleError):
        circular.bound_from_sigma(float("nan"), factor)
