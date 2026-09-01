"""SPEC.md §19.3 — deviation correction. The v1 production state is fixed to ``NONE``.

Default production state is ``NONE``, ``deviationCorrectionDeg = 0.0``,
``trueHeadingDeg = uncorrectedTrueHeadingDeg``. This module exists so the *types* are in
place — the certification key needs ``deviationCorrectionProfileHash`` and the bound needs
``deviationCorrectionResidualBound95Deg`` — while the lookup returns ``NONE`` by construction.

Two rules are enforced here rather than documented:

* A ``UNIT``-scope profile never produces ``CALIBRATED`` output. v1's certification database
  intentionally does not bind to physical-unit identity, so a per-unit correction cannot be
  matched by a runtime lookup.
* A correction is applied **exactly once**, after reference resolution and before lock
  aggregation. There is one application site, mirroring §11's single ``correctionDeg`` site.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .circular import normalize360
from .enums import (
    DeviationCorrectionScope,
    DeviationCorrectionState,
    DeviationStructureClass,
    MeasurementMode,
    PlacementMethod,
    ProviderId,
)

__all__ = [
    "DeviationCorrectionOutcome",
    "DeviationCorrectionProfileMetadata",
    "NO_DEVIATION_CORRECTION",
    "apply_deviation_correction",
    "lookup_deviation_profile",
]

#: §24: "literal NONE when correction is disabled". The sentinel is a string, not ``null``,
#: because it is a key component and a missing component must not silently match.
NONE_PROFILE_HASH = "NONE"


@dataclass(frozen=True)
class DeviationCorrectionProfileMetadata:
    """§5 ``DeviationCorrectionProfileMetadata``.

    Every scope-defining field is required. §19.3: a profile's scope is explicit — unit or
    model class, provider path, mode, placement, OS/provider range, model/config hashes.
    """

    profile_id: str
    profile_hash: str
    scope: DeviationCorrectionScope
    structure_class: DeviationStructureClass
    correction_method_id: str
    measurement_mode: MeasurementMode
    placement_method: PlacementMethod
    provider_id: ProviderId
    covered_provider_runtime_identities: tuple[str, ...]
    covered_os_build_identities: tuple[str, ...]
    geomagnetic_model_id: str
    geomagnetic_coefficient_hash: str
    precision_config_hash: str
    held_out_residual_bound_95_deg: float
    training_evidence_id: str
    held_out_evidence_id: str

    def __post_init__(self) -> None:
        if not math.isfinite(self.held_out_residual_bound_95_deg) or (
            self.held_out_residual_bound_95_deg < 0.0
        ):
            raise ValueError("heldOutResidualBound95Deg must be a finite, non-negative bound")
        if self.profile_hash == NONE_PROFILE_HASH:
            raise ValueError("a real profile may not use the NONE sentinel as its hash")

    @property
    def may_produce_calibrated_output(self) -> bool:
        """§19.3/§30.6: only a ``MODEL_CLASS`` profile can appear in a ``CALIBRATED`` record."""
        return self.scope is DeviationCorrectionScope.MODEL_CLASS


@dataclass(frozen=True)
class DeviationCorrectionOutcome:
    """The result of the single application site."""

    state: DeviationCorrectionState
    correction_deg: float
    uncorrected_true_heading_deg: float
    true_heading_deg: float
    profile_id: str | None
    profile_hash: str
    residual_bound_95_deg: float


#: The v1 production outcome shape: no profile, no correction, no residual term.
NO_DEVIATION_CORRECTION = DeviationCorrectionState.NONE


def lookup_deviation_profile(*_live_context: object) -> None:
    """§7 ``DeviationCorrectionProvider.lookup`` — v1 always returns no profile.

    This is not a placeholder: §19.3 fixes the production default at ``NONE`` and §30.6 gates
    any promotion on held-out evidence that does not exist. Returning ``None`` is the correct
    behaviour, and the signature accepts the live context so a Phase 5 profile can be added
    without changing call sites.
    """
    return None


def apply_deviation_correction(
    uncorrected_true_heading_deg: float,
    profile_correction_deg: float | None = None,
    profile: DeviationCorrectionProfileMetadata | None = None,
) -> DeviationCorrectionOutcome:
    """§19.3: apply **exactly once**, after reference resolution, before lock aggregation.

    With no certified profile the correction is ``0.0`` and the corrected heading is the
    uncorrected one — identical numbers with different names, kept as separate fields so the
    raw uncorrected heading is always retained beside the correction.
    """
    uncorrected = normalize360(uncorrected_true_heading_deg)
    if profile is None or profile_correction_deg is None:
        return DeviationCorrectionOutcome(
            state=DeviationCorrectionState.NONE,
            correction_deg=0.0,
            uncorrected_true_heading_deg=uncorrected,
            true_heading_deg=uncorrected,
            profile_id=None,
            profile_hash=NONE_PROFILE_HASH,
            residual_bound_95_deg=0.0,
        )
    if not profile.may_produce_calibrated_output:
        raise ValueError(
            f"profile {profile.profile_id} has {profile.scope.value} scope; §19.3 keeps UNIT "
            "profiles experimental and forbids them from producing CALIBRATED output"
        )
    if not math.isfinite(profile_correction_deg):
        raise ValueError("a deviation correction must be a finite number of degrees")
    return DeviationCorrectionOutcome(
        state=DeviationCorrectionState.CERTIFIED_PROFILE,
        correction_deg=profile_correction_deg,
        uncorrected_true_heading_deg=uncorrected,
        true_heading_deg=normalize360(uncorrected + profile_correction_deg),
        profile_id=profile.profile_id,
        profile_hash=profile.profile_hash,
        residual_bound_95_deg=profile.held_out_residual_bound_95_deg,
    )
