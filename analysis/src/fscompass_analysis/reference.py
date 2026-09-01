"""SPEC.md §11 — north-reference resolution for the Google FOP path.

Google FOP exposes the same ambiguous contract through both its scalar heading and its
attitude frame: true north when declination is available, magnetic north otherwise, with no
per-sample flag. This module resolves that ambiguity **without replacing Google's fusion**
and without using a geometrically ill-conditioned axis: both hypotheses are formed from the
*same physical reference axis* of the *active* measurement mode.

Two rules here prevent Critical failures:

* ``correctionDeg`` is the single Google magnetic→true correction site and is exactly ``0.0``
  or ``+declinationDeg``. Double application yields a plausible but catastrophic
  ``2 x declination`` error (failure mode 21), which §30.5 hunts for by name.
* The resolver never writes or overwrites ``reportedBound95Deg``. The ambiguity branch emits
  one uncertainty term, which flows into §19 composition.

iOS does not use this test — valid ``CLHeading.trueHeading`` is ``PROVIDER_CONTRACT_EXPLICIT``
and an active ``.xTrueNorthZVertical`` frame is ``ATTITUDE_FRAME_EXPLICIT`` — and ``AND-RV``
owns the conversion itself (``APP_APPLIED_DECLINATION``). Those contracts are represented by
the explicit constructors at the bottom of this module rather than by running the hypothesis
test with fabricated inputs (R51: N/A is not failure, and is not fabricated evidence).
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .circular import absolute_circular_difference_deg, normalize360
from .enums import (
    GeomagneticModelId,
    MeasurementMode,
    ReferenceAxis,
    ReferenceMagneticPrecheckState,
    ReferenceResolutionMethod,
    ResolvedReference,
    reference_axis_for_mode,
)

__all__ = [
    "GoogleReferenceHypotheses",
    "ReferenceResolutionResult",
    "ReferenceResolutionThresholds",
    "and_rv_reference_resolution",
    "apple_attitude_frame_reference_resolution",
    "apple_provider_contract_reference_resolution",
    "resolve_google_reference",
]


@dataclass(frozen=True)
class ReferenceResolutionThresholds:
    """The §8 keys the resolver reads.

    §8.1 requires ``referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg``: since
    ``rMag - rTrue <= abs(d)``, a margin above the ambiguity allowance would create a
    declination dead band that always resolves ``UNVERIFIED`` with no visible cause.
    """

    provider_cross_check_max_deg: float
    reference_separation_margin_deg: float
    small_declination_ambiguity_max_deg: float

    @staticmethod
    def from_profile(profile: dict) -> "ReferenceResolutionThresholds":
        return ReferenceResolutionThresholds(
            provider_cross_check_max_deg=profile["providerCrossCheckMaxDeg"],
            reference_separation_margin_deg=profile["referenceSeparationMarginDeg"],
            small_declination_ambiguity_max_deg=profile["smallDeclinationAmbiguityMaxDeg"],
        )


@dataclass(frozen=True)
class GoogleReferenceHypotheses:
    """Everything §11 needs for one active-mode stable window.

    ``g_axis_deg`` is the aggregated Google bearing **of the active mode's reference axis**;
    ``m_axis_deg`` is the synchronized diagnostic magnetic-north bearing of that *same*
    physical axis, derived through a platform magnetic orientation path — never raw
    magnetometer X/Y, and never an axis whose horizontal projection is singular.
    """

    measurement_mode: MeasurementMode
    g_axis_deg: float
    m_axis_deg: float
    declination_deg: float
    precheck_state: ReferenceMagneticPrecheckState
    geomagnetic_model_id: GeomagneticModelId
    source_window_start_monotonic_ns: int
    source_window_end_monotonic_ns: int
    #: Fresh location/model evidence, valid synchronized source timestamps and a valid
    #: diagnostic magnetic orientation. The test is ineligible without all of them.
    evidence_is_eligible: bool = True

    def __post_init__(self) -> None:
        for name in ("g_axis_deg", "m_axis_deg", "declination_deg"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise ValueError(f"GoogleReferenceHypotheses.{name} must be finite, got {value!r}")
        if self.source_window_end_monotonic_ns < self.source_window_start_monotonic_ns:
            raise ValueError("the source window must not end before it starts")


@dataclass(frozen=True)
class ReferenceResolutionResult:
    """§5 ``ReferenceResolutionResult``, bound to its mode, axis and source window.

    "A flat result is not reusable for a wall pose or vice versa." The mode and axis are
    fields precisely so a caller cannot transfer one.
    """

    measurement_mode: MeasurementMode
    reference_axis: ReferenceAxis
    resolved_reference: ResolvedReference
    reference_resolution_method: ReferenceResolutionMethod
    declination_deg: float
    correction_deg: float
    reference_ambiguity_bound_95_deg: float
    geomagnetic_model_id: GeomagneticModelId
    source_window_start_monotonic_ns: int
    source_window_end_monotonic_ns: int
    reference_hypothesis_residual_true_deg: float | None = None
    reference_hypothesis_residual_magnetic_deg: float | None = None
    #: The canonical true-north bearing of the active axis, or ``None`` when unresolved.
    canonical_true_heading_deg: float | None = None

    def __post_init__(self) -> None:
        if self.correction_deg not in (0.0,) and self.correction_deg != self.declination_deg:
            raise ValueError(
                "§11: correctionDeg is exactly 0.0 or +declinationDeg — the single Google "
                f"magnetic→true correction site. Got {self.correction_deg!r} with "
                f"declination {self.declination_deg!r} (failure mode 21)."
            )
        if self.reference_ambiguity_bound_95_deg < 0.0:
            raise ValueError("referenceAmbiguityBound95Deg must be non-negative")

    @property
    def is_true_referenced(self) -> bool:
        return self.resolved_reference in (
            ResolvedReference.TRUE_VERIFIED,
            ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
            ResolvedReference.TRUE_WITH_AMBIGUITY_BOUND,
        )


def _unresolved(
    hypotheses: GoogleReferenceHypotheses,
    residual_true_deg: float | None,
    residual_magnetic_deg: float | None,
) -> ReferenceResolutionResult:
    return ReferenceResolutionResult(
        measurement_mode=hypotheses.measurement_mode,
        reference_axis=reference_axis_for_mode(hypotheses.measurement_mode),
        resolved_reference=ResolvedReference.UNVERIFIED,
        reference_resolution_method=ReferenceResolutionMethod.NOT_RESOLVED,
        declination_deg=hypotheses.declination_deg,
        correction_deg=0.0,
        reference_ambiguity_bound_95_deg=0.0,
        geomagnetic_model_id=hypotheses.geomagnetic_model_id,
        source_window_start_monotonic_ns=hypotheses.source_window_start_monotonic_ns,
        source_window_end_monotonic_ns=hypotheses.source_window_end_monotonic_ns,
        reference_hypothesis_residual_true_deg=residual_true_deg,
        reference_hypothesis_residual_magnetic_deg=residual_magnetic_deg,
        canonical_true_heading_deg=None,
    )


def resolve_google_reference(
    hypotheses: GoogleReferenceHypotheses,
    thresholds: ReferenceResolutionThresholds,
) -> ReferenceResolutionResult:
    """§11's hypothesis test, per active measurement mode and stable window.

    Eligibility first: without fresh evidence, or with a precheck that is not
    ``CLEAN_FOR_REFERENCE``, the result is ``UNVERIFIED`` and the residuals are ``null`` —
    the engine does not manufacture a Google pipeline reference in order to compute the
    evidence that would have been needed to resolve it (R59).
    """
    if (
        not hypotheses.evidence_is_eligible
        or hypotheses.precheck_state is not ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE
    ):
        return _unresolved(hypotheses, None, None)

    g_axis = normalize360(hypotheses.g_axis_deg)
    m_axis = normalize360(hypotheses.m_axis_deg)
    declination = hypotheses.declination_deg
    t_axis = normalize360(m_axis + declination)

    residual_true = absolute_circular_difference_deg(g_axis, t_axis)
    residual_magnetic = absolute_circular_difference_deg(g_axis, m_axis)

    common = {
        "measurement_mode": hypotheses.measurement_mode,
        "reference_axis": reference_axis_for_mode(hypotheses.measurement_mode),
        "reference_resolution_method": (
            ReferenceResolutionMethod.FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK
        ),
        "declination_deg": declination,
        "geomagnetic_model_id": hypotheses.geomagnetic_model_id,
        "source_window_start_monotonic_ns": hypotheses.source_window_start_monotonic_ns,
        "source_window_end_monotonic_ns": hypotheses.source_window_end_monotonic_ns,
        "reference_hypothesis_residual_true_deg": residual_true,
        "reference_hypothesis_residual_magnetic_deg": residual_magnetic,
    }

    if (
        residual_true <= thresholds.provider_cross_check_max_deg
        and (residual_magnetic - residual_true) >= thresholds.reference_separation_margin_deg
    ):
        # Google was already emitting true north: use it exactly, correct nothing.
        return ReferenceResolutionResult(
            resolved_reference=ResolvedReference.TRUE_VERIFIED,
            correction_deg=0.0,
            reference_ambiguity_bound_95_deg=0.0,
            canonical_true_heading_deg=g_axis,
            **common,
        )

    if (
        residual_magnetic <= thresholds.provider_cross_check_max_deg
        and (residual_true - residual_magnetic) >= thresholds.reference_separation_margin_deg
    ):
        # Google was emitting magnetic north: apply declination exactly once, here.
        return ReferenceResolutionResult(
            resolved_reference=ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
            correction_deg=declination,
            reference_ambiguity_bound_95_deg=0.0,
            canonical_true_heading_deg=normalize360(g_axis + declination),
            **common,
        )

    if abs(declination) <= thresholds.small_declination_ambiguity_max_deg:
        # The hypotheses are inseparable because |d| is small; carry |d| as an explicit term
        # rather than picking a branch. §21.2 keeps this term after the reference transform.
        return ReferenceResolutionResult(
            resolved_reference=ResolvedReference.TRUE_WITH_AMBIGUITY_BOUND,
            correction_deg=0.0,
            reference_ambiguity_bound_95_deg=abs(declination),
            canonical_true_heading_deg=g_axis,
            **common,
        )

    return _unresolved(hypotheses, residual_true, residual_magnetic)


def apple_provider_contract_reference_resolution(
    mode: MeasurementMode,
    true_heading_deg: float,
    declination_deg: float,
    geomagnetic_model_id: GeomagneticModelId,
    source_window_start_monotonic_ns: int,
    source_window_end_monotonic_ns: int,
) -> ReferenceResolutionResult:
    """iOS flat: valid ``CLHeading.trueHeading`` is explicit (``PROVIDER_CONTRACT_EXPLICIT``).

    No hypothesis test, no ambiguity term, no correction — Apple owns the conversion. Running
    the Google resolver here would fabricate evidence (R51).
    """
    return ReferenceResolutionResult(
        measurement_mode=mode,
        reference_axis=reference_axis_for_mode(mode),
        resolved_reference=ResolvedReference.TRUE_VERIFIED,
        reference_resolution_method=ReferenceResolutionMethod.PROVIDER_CONTRACT_EXPLICIT,
        declination_deg=declination_deg,
        correction_deg=0.0,
        reference_ambiguity_bound_95_deg=0.0,
        geomagnetic_model_id=geomagnetic_model_id,
        source_window_start_monotonic_ns=source_window_start_monotonic_ns,
        source_window_end_monotonic_ns=source_window_end_monotonic_ns,
        canonical_true_heading_deg=normalize360(true_heading_deg),
    )


def apple_attitude_frame_reference_resolution(
    mode: MeasurementMode,
    projected_true_heading_deg: float,
    declination_deg: float,
    geomagnetic_model_id: GeomagneticModelId,
    source_window_start_monotonic_ns: int,
    source_window_end_monotonic_ns: int,
    frame_is_active: bool,
) -> ReferenceResolutionResult:
    """iOS wall: ``.xTrueNorthZVertical`` is explicit **when that frame is actually active**.

    The requested frame is an intention; the observed ``attitudeReferenceFrame`` is the fact
    (§12). If it is not active the reference is ``UNVERIFIED``.
    """
    if not frame_is_active:
        return ReferenceResolutionResult(
            measurement_mode=mode,
            reference_axis=reference_axis_for_mode(mode),
            resolved_reference=ResolvedReference.UNVERIFIED,
            reference_resolution_method=ReferenceResolutionMethod.NOT_RESOLVED,
            declination_deg=declination_deg,
            correction_deg=0.0,
            reference_ambiguity_bound_95_deg=0.0,
            geomagnetic_model_id=geomagnetic_model_id,
            source_window_start_monotonic_ns=source_window_start_monotonic_ns,
            source_window_end_monotonic_ns=source_window_end_monotonic_ns,
            canonical_true_heading_deg=None,
        )
    return ReferenceResolutionResult(
        measurement_mode=mode,
        reference_axis=reference_axis_for_mode(mode),
        resolved_reference=ResolvedReference.TRUE_VERIFIED,
        reference_resolution_method=ReferenceResolutionMethod.ATTITUDE_FRAME_EXPLICIT,
        declination_deg=declination_deg,
        correction_deg=0.0,
        reference_ambiguity_bound_95_deg=0.0,
        geomagnetic_model_id=geomagnetic_model_id,
        source_window_start_monotonic_ns=source_window_start_monotonic_ns,
        source_window_end_monotonic_ns=source_window_end_monotonic_ns,
        canonical_true_heading_deg=normalize360(projected_true_heading_deg),
    )


def and_rv_reference_resolution(
    mode: MeasurementMode,
    magnetic_axis_heading_deg: float,
    declination_deg: float,
    geomagnetic_model_id: GeomagneticModelId,
    source_window_start_monotonic_ns: int,
    source_window_end_monotonic_ns: int,
) -> ReferenceResolutionResult:
    """``AND-RV``: the app owns magnetic→true conversion, known by construction.

    §30.4: ``resolvedReference`` MUST be ``TRUE_CORRECTED_FROM_MAGNETIC`` with
    ``APP_APPLIED_DECLINATION``; there is no ``TRUE_VERIFIED`` here without an independent
    reference check, and §11's ambiguity rule does not apply. Declination is applied exactly
    once, here.
    """
    return ReferenceResolutionResult(
        measurement_mode=mode,
        reference_axis=reference_axis_for_mode(mode),
        resolved_reference=ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
        reference_resolution_method=ReferenceResolutionMethod.APP_APPLIED_DECLINATION,
        declination_deg=declination_deg,
        correction_deg=declination_deg,
        reference_ambiguity_bound_95_deg=0.0,
        geomagnetic_model_id=geomagnetic_model_id,
        source_window_start_monotonic_ns=source_window_start_monotonic_ns,
        source_window_end_monotonic_ns=source_window_end_monotonic_ns,
        canonical_true_heading_deg=normalize360(magnetic_axis_heading_deg + declination_deg),
    )
