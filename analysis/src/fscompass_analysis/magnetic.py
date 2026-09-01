"""SPEC.md §16 — magnetic interference detection, and the §11 reference precheck.

Two rules dominate this module and both have a named failure mode behind them:

* **The detector MUST NOT use magnitude alone** (failure mode 23). A disturbance can rotate
  the field vector with little magnitude change, and that is precisely the case producing a
  confident wrong bearing. Magnitude, inclination and stationary variability are fused, plus
  independent-pipeline disagreement.
* **Absent evidence is not passing evidence.** A feature that could not be measured — the
  stationary MAD while the device is moving, ``pipelineAgreementDeg`` with fewer than two
  valid active-axis pipelines — makes the classifier resolve ``UNKNOWN``, never ``CLEAN``.

The precheck and the final state are separate fields by construction (R59): the precheck
reads no pipeline- or reference-dependent feature, so the dependency order
``evidence → precheck → §11 resolution → pipeline agreement → final MagneticState → lock``
stays acyclic.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .enums import MagneticState, ReferenceMagneticPrecheckState

__all__ = [
    "MagneticFeatures",
    "MagneticThresholds",
    "classify_magnetic_state",
    "inclination_residual_deg",
    "measured_inclination_positive_down_deg",
    "reference_magnetic_precheck_state",
    "relative_magnitude_residual",
]


class MagneticFeatureError(ValueError):
    """An invalid magnetic feature input."""


def measured_inclination_positive_down_deg(up_micro_tesla: float, magnitude_micro_tesla: float) -> float:
    """§16: ``degrees(asin(clamp(-Bup / M, -1, 1)))`` — input is canonical REFERENCE_ENU.

    **The minus sign is mandatory** (R60, failure mode 23's sibling): canonical REFERENCE_ENU
    has ``Bup`` positive *upward*, while WMM inclination ``I`` and WMM vertical component
    ``Z`` are positive *downward*. Comparing ``asin(Bup/M)`` directly with WMM ``I`` reverses
    the observed sign and can reject a clean northern-hemisphere field as disturbed.

    The clamp is not decorative either: ``asin`` of ``1 + 1e-16`` is a domain error, and a
    measured component can exceed the magnitude by a rounding bit (failure mode 6).
    """
    if not math.isfinite(up_micro_tesla) or not math.isfinite(magnitude_micro_tesla):
        raise MagneticFeatureError("inclination requires finite field components")
    if magnitude_micro_tesla <= 0.0:
        raise MagneticFeatureError(
            f"field magnitude must be positive, got {magnitude_micro_tesla!r}"
        )
    ratio = -up_micro_tesla / magnitude_micro_tesla
    return math.degrees(math.asin(max(-1.0, min(1.0, ratio))))


def inclination_residual_deg(
    measured_positive_down_deg: float,
    expected_positive_down_deg: float,
) -> float:
    """§16: a **linear** difference in ``[-90, 90]``, never a circular one.

    "Inclination cannot wrap; a circular difference there is a category error that silently
    rescales the residual near the poles." Both operands are positive-down, so this is the
    one place in the codebase where an angular difference is deliberately *not* routed
    through ``shortestSignedDifferenceDeg``.
    """
    for name, value in (
        ("measured", measured_positive_down_deg),
        ("expected", expected_positive_down_deg),
    ):
        if not math.isfinite(value):
            raise MagneticFeatureError(f"{name} inclination must be finite, got {value!r}")
        if not -90.0 <= value <= 90.0:
            raise MagneticFeatureError(
                f"{name} inclination must lie in [-90, 90] degrees positive-down, got {value!r}"
            )
    return measured_positive_down_deg - expected_positive_down_deg


def relative_magnitude_residual(
    measured_micro_tesla: float, expected_micro_tesla: float
) -> float:
    """§16: ``abs(M - expectedM) / expectedM``."""
    if not math.isfinite(measured_micro_tesla) or not math.isfinite(expected_micro_tesla):
        raise MagneticFeatureError("magnitude residual requires finite inputs")
    if expected_micro_tesla <= 0.0:
        raise MagneticFeatureError(
            f"expected field magnitude must be positive, got {expected_micro_tesla!r}"
        )
    return abs(measured_micro_tesla - expected_micro_tesla) / expected_micro_tesla


@dataclass(frozen=True)
class MagneticThresholds:
    """The §8 candidate gates the classifier reads, versioned jointly with the model.

    §10.1: changing ``geomagneticModelId`` silently re-tunes these gates, so a model change
    invalidates threshold calibration and requires re-running §30.3.
    """

    magnitude_residual_suspect_fraction: float
    magnitude_residual_disturbed_fraction: float
    inclination_residual_suspect_deg: float
    inclination_residual_disturbed_deg: float
    stationary_field_mad_suspect_micro_tesla: float
    stationary_field_mad_disturbed_micro_tesla: float
    pipeline_disagreement_suspect_deg: float
    pipeline_disagreement_disturbed_deg: float

    @staticmethod
    def from_profile(profile: dict) -> "MagneticThresholds":
        return MagneticThresholds(
            magnitude_residual_suspect_fraction=profile["magneticMagnitudeResidualSuspectFraction"],
            magnitude_residual_disturbed_fraction=profile[
                "magneticMagnitudeResidualDisturbedFraction"
            ],
            inclination_residual_suspect_deg=profile["inclinationResidualSuspectDeg"],
            inclination_residual_disturbed_deg=profile["inclinationResidualDisturbedDeg"],
            stationary_field_mad_suspect_micro_tesla=profile[
                "stationaryFieldMadSuspectMicroTesla"
            ],
            stationary_field_mad_disturbed_micro_tesla=profile[
                "stationaryFieldMadDisturbedMicroTesla"
            ],
            pipeline_disagreement_suspect_deg=profile["pipelineDisagreementSuspectDeg"],
            pipeline_disagreement_disturbed_deg=profile["pipelineDisagreementDisturbedDeg"],
        )


@dataclass(frozen=True)
class MagneticFeatures:
    """The §16 feature set. ``None`` means **absent**, which is never zero (§5).

    ``stationary_field_mad_micro_tesla`` is absent whenever the motion gates do not indicate
    stationary; ``pipeline_agreement_deg`` is absent whenever fewer than two valid
    independent **active-axis** pipelines exist (§15.1).
    """

    relative_magnitude_residual: float | None
    inclination_residual_deg: float | None
    stationary_field_mad_micro_tesla: float | None
    pipeline_agreement_deg: float | None
    any_value_nonfinite: bool = False
    sensor_saturated: bool = False
    os_calibration_invalid: bool = False

    @property
    def is_invalid(self) -> bool:
        return self.any_value_nonfinite or self.sensor_saturated or self.os_calibration_invalid

    @property
    def non_pipeline_features_present(self) -> bool:
        return None not in (
            self.relative_magnitude_residual,
            self.inclination_residual_deg,
            self.stationary_field_mad_micro_tesla,
        )

    @property
    def all_required_features_present(self) -> bool:
        return self.non_pipeline_features_present and self.pipeline_agreement_deg is not None


def _at_or_above(value: float | None, threshold: float) -> bool:
    """An absent feature cannot exceed a threshold — and cannot clear one either.

    Clearing is decided separately by the presence check, so this helper never converts
    absence into evidence of a clean field.
    """
    return value is not None and value >= threshold


def classify_magnetic_state(
    features: MagneticFeatures, thresholds: MagneticThresholds
) -> MagneticState:
    """§16's classifier, in the specified order, run **after** any §11 Google resolution."""
    if features.is_invalid:
        return MagneticState.INVALID

    if (
        _at_or_above(
            features.relative_magnitude_residual, thresholds.magnitude_residual_disturbed_fraction
        )
        or _at_or_above(
            None if features.inclination_residual_deg is None else abs(features.inclination_residual_deg),
            thresholds.inclination_residual_disturbed_deg,
        )
        or _at_or_above(
            features.stationary_field_mad_micro_tesla,
            thresholds.stationary_field_mad_disturbed_micro_tesla,
        )
        or _at_or_above(
            features.pipeline_agreement_deg, thresholds.pipeline_disagreement_disturbed_deg
        )
    ):
        return MagneticState.DISTURBED

    if (
        _at_or_above(
            features.relative_magnitude_residual, thresholds.magnitude_residual_suspect_fraction
        )
        or _at_or_above(
            None if features.inclination_residual_deg is None else abs(features.inclination_residual_deg),
            thresholds.inclination_residual_suspect_deg,
        )
        or _at_or_above(
            features.stationary_field_mad_micro_tesla,
            thresholds.stationary_field_mad_suspect_micro_tesla,
        )
        or _at_or_above(
            features.pipeline_agreement_deg, thresholds.pipeline_disagreement_suspect_deg
        )
    ):
        return MagneticState.SUSPECT

    if features.all_required_features_present:
        return MagneticState.CLEAN

    # An unmeasured feature is not a passing feature.
    return MagneticState.UNKNOWN


def reference_magnetic_precheck_state(
    features: MagneticFeatures, thresholds: MagneticThresholds
) -> ReferenceMagneticPrecheckState:
    """§11/§16 precheck — the narrow eligibility gate for the §11 hypothesis test.

    It reads **only** the three non-pipeline features. It MUST NOT read
    ``pipelineAgreementDeg``, ``resolvedReference``, or any feature whose construction needs
    a reference-resolved Google heading: that circularity is R59, a Critical failure in which
    reference resolution requires the final magnetic state while the final state requires a
    reference-resolved pipeline.

    This is never a substitute for final classification; a Google lock still requires the
    post-resolution final ``MagneticState``.
    """
    if features.is_invalid:
        return ReferenceMagneticPrecheckState.NOT_CLEAN_FOR_REFERENCE
    if not features.non_pipeline_features_present:
        return ReferenceMagneticPrecheckState.UNKNOWN

    inclination = features.inclination_residual_deg
    assert inclination is not None  # guarded by non_pipeline_features_present
    if (
        _at_or_above(
            features.relative_magnitude_residual, thresholds.magnitude_residual_suspect_fraction
        )
        or _at_or_above(abs(inclination), thresholds.inclination_residual_suspect_deg)
        or _at_or_above(
            features.stationary_field_mad_micro_tesla,
            thresholds.stationary_field_mad_suspect_micro_tesla,
        )
    ):
        return ReferenceMagneticPrecheckState.NOT_CLEAN_FOR_REFERENCE
    return ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE
