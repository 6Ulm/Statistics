"""SPEC.md §24 — the certification key, the database, and ``miss → CANDIDATE``.

§24 is *authoritative* for this schema: two platform agents inventing two lookup schemas is a
realistic failure mode, so every lookup elsewhere refers here rather than restating the key.

Three properties are structural rather than procedural:

* **A record exists only for ``CALIBRATED``.** Absence already means ``CANDIDATE``; storing
  both invites writing a ``CANDIDATE`` record and editing its state field.
* **Every key field is derivable in the production process** from a public runtime value or an
  app-bundled artifact hash. Lab-only facts — sales-region SKU, unit serial, repair history,
  operator name — belong in the evidence inventory, never in the runtime key (R66). A value
  that is genuinely not runtime-observable uses the explicit
  :data:`NOT_RUNTIME_OBSERVABLE` sentinel with pooled worst-case evidence.
* **Lookup is exact on every component.** ``osBuildIdentity``, ``providerRuntimeIdentity`` and
  ``locationProviderRuntimeIdentity`` are exact observed identities, never semantic or
  open-ended ranges that silently admit a future release.

§37 rule 12: an agent MUST NOT add records to make tests pass. The shipped database is empty,
so every lookup misses and every measurement is ``CANDIDATE`` with ``unknownDeviceFloor95Deg``
— which, with the shipped constants, means no freehand grade is arithmetically reachable at
all (§8.1.1).
"""

from __future__ import annotations

import math
from dataclasses import asdict, dataclass

from .enums import (
    BoundCalibrationState,
    GeomagneticModelId,
    LocationProviderId,
    MeasurementMode,
    PlacementMethod,
    ProviderErrorSource,
    ProviderId,
    QualityGrade,
    UncertaintyCoverageEvidenceState,
)

__all__ = [
    "CERTIFICATION_SCHEMA_VERSION",
    "CertificationDatabase",
    "CertificationKey",
    "CertificationLookupOutcome",
    "CertificationRecord",
    "NOT_RUNTIME_OBSERVABLE",
    "assert_calibration_invariants",
]

#: §24: "prevents a newer client from reinterpreting an old tuple".
CERTIFICATION_SCHEMA_VERSION = "certification-v1"

#: §24: the explicit sentinel for a component the runtime genuinely cannot observe. It is a
#: value, not a missing field, so evidence gathered under it is pooled worst-case evidence
#: rather than an invented key field (R66).
NOT_RUNTIME_OBSERVABLE = "NOT_RUNTIME_OBSERVABLE"

#: §24: for OS-owned providers without a separate public version, the provider runtime
#: identity is derived from the OS build rather than left ``UNKNOWN`` or filled with a
#: marketing version.
OS_BUILD_PROVIDER_IDENTITY_PREFIX = "OS_BUILD:"


class CertificationKeyError(ValueError):
    """A key component is missing, empty, or an open-ended range."""


#: Substrings that betray a semantic or open-ended version range. §24 requires exact observed
#: identities; a range "silently admits a future release" that was never measured.
_OPEN_ENDED_MARKERS = ("+", "*", "..", ">=", "<=", "latest", "any", "unknown")


def _require_exact(name: str, value: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise CertificationKeyError(f"{name} is required and must be a non-empty exact identity")
    if value == NOT_RUNTIME_OBSERVABLE:
        return value
    lowered = value.lower()
    for marker in _OPEN_ENDED_MARKERS:
        if marker in lowered:
            raise CertificationKeyError(
                f"{name}={value!r} looks like an open-ended or semantic range. §24 requires an "
                "exact observed identity; evidence covering several exact builds generates "
                "several records pointing to the same report (R66)."
            )
    return value


@dataclass(frozen=True)
class CertificationKey:
    """§24 ``CertificationKey`` — the exact tuple a measurement context must match."""

    certification_schema_version: str
    hardware_runtime_identity: str
    sensor_runtime_identity: str
    os_build_identity: str
    provider_id: ProviderId
    provider_runtime_identity: str
    provider_error_source: ProviderErrorSource
    location_provider_id: LocationProviderId
    location_provider_runtime_identity: str
    measurement_mode: MeasurementMode
    placement_method: PlacementMethod
    placement_profile_hash: str
    geomagnetic_model_id: GeomagneticModelId
    geomagnetic_coefficient_hash: str
    geomagnetic_error_model_hash: str
    deviation_correction_profile_hash: str
    engine_decision_logic_hash: str
    precision_config_hash: str

    def __post_init__(self) -> None:
        for name in (
            "certification_schema_version",
            "hardware_runtime_identity",
            "sensor_runtime_identity",
            "os_build_identity",
            "provider_runtime_identity",
            "location_provider_runtime_identity",
            "placement_profile_hash",
            "geomagnetic_coefficient_hash",
            "geomagnetic_error_model_hash",
            "deviation_correction_profile_hash",
            "engine_decision_logic_hash",
            "precision_config_hash",
        ):
            _require_exact(name, getattr(self, name))

    def as_lookup_tuple(self) -> tuple:
        """A hashable, order-stable tuple; lookup is exact on every component."""
        values = asdict(self)
        return tuple(
            value.value if hasattr(value, "value") else value for _, value in sorted(values.items())
        )


@dataclass(frozen=True)
class CertificationRecord:
    """§24 ``CertificationRecord``. Exists **only** for ``CALIBRATED``."""

    key: CertificationKey
    device_floor_95_deg: float
    supported_quality_grade: QualityGrade
    earned_under_engine_version: str
    evidence_report_id: str
    certification_date: str
    bound_calibration_state: BoundCalibrationState = BoundCalibrationState.CALIBRATED

    def __post_init__(self) -> None:
        if self.bound_calibration_state is not BoundCalibrationState.CALIBRATED:
            raise ValueError(
                "§24: a record exists only for CALIBRATED. Absence already means CANDIDATE; "
                "storing both invites writing a CANDIDATE record and editing its state field."
            )
        if not math.isfinite(self.device_floor_95_deg) or self.device_floor_95_deg <= 0.0:
            raise ValueError("deviceFloor95Deg must be a finite, positive bound")
        if not self.evidence_report_id.strip():
            raise ValueError(
                "evidenceReportId MUST resolve to archived raw telemetry; an empty one makes "
                "the record unauditable (§24)"
            )


@dataclass(frozen=True)
class CertificationLookupOutcome:
    """What the engine consumes: a state, a floor, and a ceiling.

    §19.1: ``boundCalibrationState`` is derived at runtime from this lookup, never read from
    config. There is no invalidation step to forget — changing model, config, provider path,
    mode or placement changes the key and therefore misses.
    """

    bound_calibration_state: BoundCalibrationState
    uncertainty_coverage_evidence_state: UncertaintyCoverageEvidenceState
    device_floor_95_deg: float
    supported_quality_grade: QualityGrade
    record: CertificationRecord | None

    def __post_init__(self) -> None:
        assert_calibration_invariants(
            self.bound_calibration_state, self.uncertainty_coverage_evidence_state
        )


def assert_calibration_invariants(
    bound_calibration_state: BoundCalibrationState,
    coverage_evidence_state: UncertaintyCoverageEvidenceState,
) -> None:
    """§19.1's two invariants, asserted on every emitted result.

    ``CALIBRATED  <=> uncertaintyCoverageEvidenceState == EMPIRICALLY_CALIBRATED``
    ``CANDIDATE    => uncertaintyCoverageEvidenceState in {TARGET_ONLY, UNDEFINED}``

    The two fields are near-redundant by design — one is the gate, the other the claim — and
    the redundancy is safe only while the invariant holds, because drift lets a ``95%`` label
    appear on a ``CANDIDATE`` measurement (failure mode 31).
    """
    calibrated = bound_calibration_state is BoundCalibrationState.CALIBRATED
    empirical = coverage_evidence_state is UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED
    if calibrated != empirical:
        raise ValueError(
            "§19.1 invariant violated: boundCalibrationState="
            f"{bound_calibration_state.value} with uncertaintyCoverageEvidenceState="
            f"{coverage_evidence_state.value}"
        )
    if not calibrated and coverage_evidence_state not in (
        UncertaintyCoverageEvidenceState.TARGET_ONLY,
        UncertaintyCoverageEvidenceState.UNDEFINED,
    ):
        raise ValueError(
            "§19.1 invariant violated: CANDIDATE requires TARGET_ONLY or UNDEFINED coverage "
            f"evidence, got {coverage_evidence_state.value}"
        )


class CertificationDatabase:
    """§7 ``CertificationDatabase``. Generated from benchmark evidence, versioned with the app.

    The shipped instance is empty. §24 and §37 rule 12 forbid adding a record to make a test
    pass; :meth:`with_records` exists so a *test* can build its own in-memory database to
    exercise the hit path, and it never touches a shipped artifact.
    """

    def __init__(self, records: tuple[CertificationRecord, ...] = ()) -> None:
        self._records = {record.key.as_lookup_tuple(): record for record in records}

    @staticmethod
    def shipped() -> "CertificationDatabase":
        """The database that actually ships in v1: empty, because no evidence exists."""
        return CertificationDatabase(())

    @staticmethod
    def with_records(records: tuple[CertificationRecord, ...]) -> "CertificationDatabase":
        return CertificationDatabase(records)

    def __len__(self) -> int:
        return len(self._records)

    def lookup(
        self, key: CertificationKey, unknown_device_floor_95_deg: float
    ) -> CertificationLookupOutcome:
        """Exact lookup. A miss on **any** component yields ``CANDIDATE``.

        §24: a miss returns nothing, so the engine uses ``CANDIDATE``,
        ``unknownDeviceFloor95Deg``, and a provisional ceiling no higher than ``USABLE`` —
        an upper limit, not a promise.
        """
        record = self._records.get(key.as_lookup_tuple())
        if record is None:
            return CertificationLookupOutcome(
                bound_calibration_state=BoundCalibrationState.CANDIDATE,
                uncertainty_coverage_evidence_state=UncertaintyCoverageEvidenceState.TARGET_ONLY,
                device_floor_95_deg=unknown_device_floor_95_deg,
                supported_quality_grade=QualityGrade.USABLE,
                record=None,
            )
        return CertificationLookupOutcome(
            bound_calibration_state=BoundCalibrationState.CALIBRATED,
            uncertainty_coverage_evidence_state=(
                UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED
            ),
            device_floor_95_deg=record.device_floor_95_deg,
            supported_quality_grade=record.supported_quality_grade,
            record=record,
        )
