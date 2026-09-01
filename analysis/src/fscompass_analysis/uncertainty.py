"""SPEC.md §19 — uncertainty composition producing **both** bounds.

``instrumentBound95Deg`` says how well the pipeline knows where the *device axis* points.
``reportedBound95Deg`` says how well the app knows where the *building plane* points; it is
what the practitioner sees, what drives classification, and what determines the grade.
**Never display ``instrumentBound95Deg`` as the measurement uncertainty** — it omits the
largest term (failure mode 30).

The asymmetry in the formula is deliberate and is a modelling choice, not a derivation: the
three base terms combine with ``max`` because they estimate the *same* quantity; the rest add
because they are different, additive error sources. §19.1 is why the result carries
``CANDIDATE`` until held-out coverage exists for the exact certification key — summing several
nominally-95% terms does not yield a 95% interval, and which way it errs cannot be reasoned
out, only measured.

A missing provider error is **absent, never 0° evidence** (§19, failure mode 28). Absence is
modelled as ``None`` throughout; the ``max`` is taken over present values only.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .enums import GradeLimitingFactor, MagneticState, RejectionReason

__all__ = [
    "BoundComposition",
    "InterferenceRejection",
    "UncertaintyTerms",
    "compose_bounds",
    "interference_bound_95_deg",
]

#: §19: both bounds are capped at 180°, beyond which "north is somewhere" carries no
#: information anyway.
MAX_BOUND_DEG = 180.0


class InterferenceRejection(Exception):
    """§19: ``UNKNOWN`` / ``DISTURBED`` / ``INVALID`` are a rejection, not a wider bound.

    Widening a bound to absorb an unclassifiable field would convert a refusal into a
    confidently-labelled measurement, which is the behaviour §1 exists to prevent.
    """

    def __init__(self, reason: RejectionReason) -> None:
        super().__init__(f"magnetic state rejects the measurement: {reason.value}")
        self.reason = reason


def interference_bound_95_deg(
    magnetic_state: MagneticState, suspect_interference_bound_95_deg: float
) -> float:
    """§19: ``0`` when ``CLEAN``, the configured penalty when ``SUSPECT``, else rejection.

    §8.1.1 row 3: with the candidate constants the ``SUSPECT`` term alone exceeds the freehand
    instrument budget, so ``SUSPECT`` prevents a freehand lock outright rather than merely
    capping the grade. That consequence is arithmetic, not a special case in this function.
    """
    if magnetic_state is MagneticState.CLEAN:
        return 0.0
    if magnetic_state is MagneticState.SUSPECT:
        return suspect_interference_bound_95_deg
    if magnetic_state is MagneticState.DISTURBED:
        raise InterferenceRejection(RejectionReason.MAGNETIC_FIELD_DISTURBED)
    if magnetic_state is MagneticState.INVALID:
        raise InterferenceRejection(RejectionReason.MAGNETIC_CALIBRATION_INVALID)
    raise InterferenceRejection(RejectionReason.MAGNETIC_FIELD_UNKNOWN)


@dataclass(frozen=True)
class UncertaintyTerms:
    """Every §19 input term. ``None`` is **absent**; ``0.0`` is a measured zero."""

    #: Present only for provider/mode paths exposing a documented degree error. iOS wall and
    #: Google FOP wall expose none for their outward-normal projection, and FOP's display-top
    #: scalar error MUST NOT enter a wall bound (R61).
    provider_reported_bound_term_deg: float | None
    #: P95 of residuals over **all** accepted samples about the circular mean (§15, §19).
    #: A dispersion floor, not an error estimate: it detects an unsteady hold and can never
    #: detect a steady wrong answer.
    sample_bound_95_deg: float
    #: The certified floor for the exact §24 key, else ``unknownDeviceFloor95Deg``.
    device_floor_95_deg: float
    #: ``0`` when the deviation-correction state is ``NONE`` (§19.3, the v1 default).
    deviation_correction_residual_bound_95_deg: float = 0.0
    #: ``boundFromSigma(model.declinationSigma1Deg)``, only when the app performs or may need
    #: the magnetic→true conversion. Absent where the provider owns it.
    declination_model_bound_95_deg: float | None = None
    #: Worst declination change over accepted position/altitude/time uncertainty.
    location_time_sensitivity_bound_95_deg: float = 0.0
    #: From the §11 ``ReferenceResolutionResult``; ``0`` when verified.
    reference_ambiguity_bound_95_deg: float = 0.0
    #: From §19's interference rule.
    interference_bound_95_deg: float = 0.0
    #: The configured freehand bound for the mode, or a repeatability-tested method bound.
    #: §18.5: "finite bound from method ... **never zero**".
    placement_bound_95_deg: float = 0.0

    def __post_init__(self) -> None:
        for name in (
            "sample_bound_95_deg",
            "device_floor_95_deg",
            "deviation_correction_residual_bound_95_deg",
            "location_time_sensitivity_bound_95_deg",
            "reference_ambiguity_bound_95_deg",
            "interference_bound_95_deg",
            "placement_bound_95_deg",
        ):
            value = getattr(self, name)
            if not math.isfinite(value) or value < 0.0:
                raise ValueError(f"{name} must be a finite, non-negative bound, got {value!r}")
        for name in ("provider_reported_bound_term_deg", "declination_model_bound_95_deg"):
            value = getattr(self, name)
            if value is not None and (not math.isfinite(value) or value < 0.0):
                raise ValueError(
                    f"{name} is either absent (None) or a finite non-negative bound, got {value!r}"
                )
        if self.placement_bound_95_deg <= 0.0:
            raise ValueError(
                "§18.5: placement uncertainty is a finite bound from the method and is never "
                "zero. A zero placement term is how a build reaches Professional freehand "
                "(§20), which is a certification failure, not a feature."
            )


@dataclass(frozen=True)
class BoundComposition:
    """The composed result: both bounds, the base term, and what limited the grade."""

    base_heading_bound_95_deg: float
    instrument_bound_95_deg: float
    reported_bound_95_deg: float
    grade_limited_by: GradeLimitingFactor


#: §19: the fixed precedence for a **non-numeric policy ceiling** that lowers the grade.
POLICY_CEILING_PRECEDENCE: tuple[GradeLimitingFactor, ...] = (
    GradeLimitingFactor.CERTIFICATION_CEILING,
    GradeLimitingFactor.SPACE_WEATHER,
    GradeLimitingFactor.CHARGING_STATE,
)


def compose_bounds(
    terms: UncertaintyTerms,
    active_policy_ceilings: frozenset[GradeLimitingFactor] = frozenset(),
) -> BoundComposition:
    """§19 composition.

    ``active_policy_ceilings`` names non-numeric ceilings currently lowering the grade — a
    certification ceiling, ``PROFESSIONAL_SUPPRESSED`` space weather, active wireless charging.
    They do not change either bound; they only take precedence in ``gradeLimitedBy``, because
    a numeric term the user could act on is useless advice when a policy is the real ceiling.
    """
    base_candidates: list[tuple[float, GradeLimitingFactor]] = [
        (terms.sample_bound_95_deg, GradeLimitingFactor.SAMPLE_DISPERSION),
        (terms.device_floor_95_deg, GradeLimitingFactor.DEVICE_FLOOR),
    ]
    if terms.provider_reported_bound_term_deg is not None:
        base_candidates.append(
            (terms.provider_reported_bound_term_deg, GradeLimitingFactor.PROVIDER_ERROR)
        )
    base = max(value for value, _ in base_candidates)

    declination_term = terms.declination_model_bound_95_deg or 0.0
    instrument = min(
        MAX_BOUND_DEG,
        base
        + declination_term
        + terms.location_time_sensitivity_bound_95_deg
        + terms.reference_ambiguity_bound_95_deg
        + terms.deviation_correction_residual_bound_95_deg
        + terms.interference_bound_95_deg,
    )
    reported = min(MAX_BOUND_DEG, instrument + terms.placement_bound_95_deg)

    return BoundComposition(
        base_heading_bound_95_deg=base,
        instrument_bound_95_deg=instrument,
        reported_bound_95_deg=reported,
        grade_limited_by=_grade_limited_by(terms, base_candidates, base, active_policy_ceilings),
    )


def _grade_limited_by(
    terms: UncertaintyTerms,
    base_candidates: list[tuple[float, GradeLimitingFactor]],
    base: float,
    active_policy_ceilings: frozenset[GradeLimitingFactor],
) -> GradeLimitingFactor:
    for ceiling in POLICY_CEILING_PRECEDENCE:
        if ceiling in active_policy_ceilings:
            return ceiling

    # Only the base term that actually won the `max` contributes to the sum, so only it can
    # be the limiting one among the three.
    contributing: list[tuple[float, GradeLimitingFactor]] = [
        (value, factor) for value, factor in base_candidates if value == base
    ]
    contributing.append((terms.placement_bound_95_deg, GradeLimitingFactor.PLACEMENT_UNCERTAINTY))
    if terms.declination_model_bound_95_deg is not None:
        contributing.append(
            (terms.declination_model_bound_95_deg, GradeLimitingFactor.DECLINATION_MODEL)
        )
    contributing.extend(
        [
            (
                terms.location_time_sensitivity_bound_95_deg,
                GradeLimitingFactor.LOCATION_TIME_SENSITIVITY,
            ),
            (terms.reference_ambiguity_bound_95_deg, GradeLimitingFactor.REFERENCE_AMBIGUITY),
            (
                terms.deviation_correction_residual_bound_95_deg,
                GradeLimitingFactor.DEVIATION_PROFILE_RESIDUAL,
            ),
            (terms.interference_bound_95_deg, GradeLimitingFactor.INTERFERENCE_PENALTY),
        ]
    )

    largest = max(value for value, _ in contributing)
    if largest <= 0.0:
        return GradeLimitingFactor.NONE
    # Exact ties resolve by stable enum order, so two runtimes cannot disagree about which
    # of two equal terms is named.
    order = list(GradeLimitingFactor)
    return min(
        (factor for value, factor in contributing if value == largest),
        key=order.index,
    )
