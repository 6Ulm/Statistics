"""SPEC.md §8.1.1 — the required build-time grade-reachability analysis, analysis runtime."""

from __future__ import annotations

import copy

import pytest

from fscompass_analysis.grade_reachability import (
    CertificationState,
    MagneticState,
    PlacementMethod,
    QualityGrade,
    compute,
    instrument_budget_deg,
    interference_bound_95_deg,
    placement_bound_95_deg,
    quality_grade_for_reported_bound,
    verify,
)


def test_every_claimed_grade_is_reachable(claims, profile):
    findings = verify(claims, profile)
    assert not findings, (
        "SPEC.md §8.1.1 grade-reachability findings. A failing gate is a finding, not an "
        "obstacle (§37 rule 12): fix the claim or the evidence, never the fixture.\n"
        + "\n".join(f"  {f}" for f in findings)
    )


def test_flat_freehand_uncertified_cannot_lock(profile):
    """§8.1.1 table row 1."""
    budget = instrument_budget_deg(PlacementMethod.FREEHAND, profile)
    assert budget == pytest.approx(2.0)
    assert profile["unknownDeviceFloor95Deg"] > budget

    r = compute(PlacementMethod.FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile)
    assert r.minimum_reported_bound_95_deg == pytest.approx(7.0)
    assert r.lock_reachable is False
    assert r.max_reachable_grade is QualityGrade.LOW_CONFIDENCE


def test_wall_freehand_can_never_lock(profile):
    """§8.1.1 table row 2: 'WALL_FLUSH_FREEHAND can never lock'."""
    assert instrument_budget_deg(PlacementMethod.WALL_FLUSH_FREEHAND, profile) == pytest.approx(0.0)

    assert not compute(
        PlacementMethod.WALL_FLUSH_FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile
    ).lock_reachable

    certified = compute(
        PlacementMethod.WALL_FLUSH_FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile
    )
    assert certified.required_device_floor_at_most_deg == pytest.approx(0.0)
    assert certified.lock_reachable is False


def test_suspect_prevents_freehand_lock_outright(profile):
    """§8.1.1 table row 3: SUSPECT prevents locking, it does not merely cap the grade."""
    for method in (PlacementMethod.FREEHAND, PlacementMethod.WALL_FLUSH_FREEHAND):
        assert profile["suspectInterferenceBound95Deg"] > instrument_budget_deg(method, profile)
        for certification in CertificationState:
            r = compute(method, certification, MagneticState.SUSPECT, profile)
            assert r.lock_reachable is False, r.explanation


def test_certified_flat_freehand_required_floor(profile):
    """§8.1.1 certification bootstrap: the floor is swept, not assumed."""
    r = compute(PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile)
    assert r.required_device_floor_at_most_deg == pytest.approx(2.0)
    assert r.lock_reachable is True

    assert compute(
        PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile,
        certified_device_floor_95_deg=2.0,
    ).lock_reachable
    assert not compute(
        PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile,
        certified_device_floor_95_deg=2.0001,
    ).lock_reachable


@pytest.mark.parametrize(
    "method", [PlacementMethod.NONMAGNETIC_ALIGNMENT_JIG, PlacementMethod.SURVEY_FIXTURE]
)
def test_unmeasured_placement_yields_no_grade(profile, method):
    """§29.5 / §35: no grade above USABLE without a measured method, and no default bound."""
    assert placement_bound_95_deg(method, profile) is None
    r = compute(method, CertificationState.CERTIFIED, MagneticState.CLEAN, profile)
    assert r.max_reachable_grade is QualityGrade.NOT_SUPPORTED
    assert r.lock_reachable is False


@pytest.mark.parametrize(
    "state", [MagneticState.DISTURBED, MagneticState.INVALID, MagneticState.UNKNOWN]
)
def test_rejecting_magnetic_states_produce_no_measurement(profile, state):
    assert interference_bound_95_deg(state, profile) is None
    r = compute(PlacementMethod.FREEHAND, CertificationState.CERTIFIED, state, profile)
    assert r.minimum_reported_bound_95_deg is None
    assert r.max_reachable_grade is QualityGrade.INVALID


@pytest.mark.parametrize(
    "bound,grade",
    [
        (0.0, QualityGrade.PROFESSIONAL),
        (2.0, QualityGrade.PROFESSIONAL),
        (2.0000001, QualityGrade.HIGH),
        (3.0, QualityGrade.HIGH),
        (3.0000001, QualityGrade.USABLE),
        (5.0, QualityGrade.USABLE),
        (5.0000001, QualityGrade.LOW_CONFIDENCE),
        (10.0, QualityGrade.LOW_CONFIDENCE),
        (10.0000001, QualityGrade.INVALID),
        (180.0, QualityGrade.INVALID),
    ],
)
def test_grade_function_is_total_and_ordered(profile, bound, grade):
    """§20: explicit half-open intervals so the grade function is total."""
    assert quality_grade_for_reported_bound(bound, profile) is grade


def test_verifier_detects_a_forbidden_claim(claims, profile):
    """An in-memory overreach - the shipped claims file is never edited to pass a test."""
    overreaching = copy.deepcopy(claims)
    overreaching["combinations"] = [
        c | {"claimedMaxGrade": "USABLE", "claimedLockReachable": True}
        for c in claims["combinations"]
        if c["id"] == "flat-freehand-uncertified-clean"
    ]
    findings = verify(overreaching, profile)
    assert any("arithmetically forbidden" in f.problem for f in findings), findings
    assert any("Precision Lock is reachable" in f.problem for f in findings), findings


def test_claims_cover_every_supported_combination(claims, profile):
    """§8.1.1: the analysis must cover EVERY combination the product claims to support.

    A claims file that silently omits a combination would let an unchecked grade ship, so
    the freehand placement methods must be enumerated across every certification state and
    every magnetic state.
    """
    covered = {
        (c["placementMethod"], c["certificationState"], c["magneticState"])
        for c in claims["combinations"]
    }
    for method in ("FREEHAND", "WALL_FLUSH_FREEHAND"):
        for certification in ("UNCERTIFIED", "CERTIFIED"):
            for magnetic in ("CLEAN", "SUSPECT", "DISTURBED", "UNKNOWN", "INVALID"):
                assert (method, certification, magnetic) in covered, (
                    f"§8.1.1 requires a claim for ({method}, {certification}, {magnetic})"
                )
    # Unmeasured placement methods are covered by a single ANY/ANY row each.
    for method in ("NONMAGNETIC_ALIGNMENT_JIG", "SURVEY_FIXTURE"):
        assert (method, "ANY", "ANY") in covered
