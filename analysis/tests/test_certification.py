"""SPEC.md §24 / §19.1 — certification-key construction, ``miss → CANDIDATE``, invariants."""

from __future__ import annotations

import dataclasses

import pytest

from fscompass_analysis import certification, deviation, fixtures
from fscompass_analysis.enums import (
    BoundCalibrationState,
    DeviationCorrectionScope,
    DeviationCorrectionState,
    DeviationStructureClass,
    GeomagneticModelId,
    LocationProviderId,
    MeasurementMode,
    PlacementMethod,
    ProviderErrorSource,
    ProviderId,
    QualityGrade,
    UncertaintyCoverageEvidenceState,
)


@pytest.fixture(scope="module")
def key_fixture():
    return fixtures.load(fixtures.CERTIFICATION_KEY)


def _key(document) -> certification.CertificationKey:
    return certification.CertificationKey(
        certification_schema_version=document["certificationSchemaVersion"],
        hardware_runtime_identity=document["hardwareRuntimeIdentity"],
        sensor_runtime_identity=document["sensorRuntimeIdentity"],
        os_build_identity=document["osBuildIdentity"],
        provider_id=ProviderId(document["providerId"]),
        provider_runtime_identity=document["providerRuntimeIdentity"],
        provider_error_source=ProviderErrorSource(document["providerErrorSource"]),
        location_provider_id=LocationProviderId(document["locationProviderId"]),
        location_provider_runtime_identity=document["locationProviderRuntimeIdentity"],
        measurement_mode=MeasurementMode(document["measurementMode"]),
        placement_method=PlacementMethod(document["placementMethod"]),
        placement_profile_hash=document["placementProfileHash"],
        geomagnetic_model_id=GeomagneticModelId(document["geomagneticModelId"]),
        geomagnetic_coefficient_hash=document["geomagneticCoefficientHash"],
        geomagnetic_error_model_hash=document["geomagneticErrorModelHash"],
        deviation_correction_profile_hash=document["deviationCorrectionProfileHash"],
        engine_decision_logic_hash=document["engineDecisionLogicHash"],
        precision_config_hash=document["precisionConfigHash"],
    )


def test_the_shipped_database_is_empty(key_fixture):
    """§37 rule 12: an agent MUST NOT add records to make tests pass."""
    assert key_fixture["shippedDatabaseRecordCount"] == 0
    assert len(certification.CertificationDatabase.shipped()) == 0


def test_a_miss_yields_candidate_and_the_unknown_floor(key_fixture, profile):
    """§24: a miss returns nothing, so the engine uses ``CANDIDATE``,
    ``unknownDeviceFloor95Deg``, and a provisional ceiling no higher than ``USABLE``."""
    outcome = certification.CertificationDatabase.shipped().lookup(
        _key(key_fixture["completeKey"]), profile["unknownDeviceFloor95Deg"]
    )
    expected = key_fixture["expectedLookupOnShippedDatabase"]
    assert outcome.bound_calibration_state.value == expected["boundCalibrationState"]
    assert (
        outcome.uncertainty_coverage_evidence_state.value
        == expected["uncertaintyCoverageEvidenceState"]
    )
    assert outcome.supported_quality_grade.name == expected["supportedQualityGrade"]
    assert outcome.device_floor_95_deg == profile["unknownDeviceFloor95Deg"]
    assert outcome.record is None


def test_every_named_component_actually_differentiates_the_key(key_fixture, profile):
    """§24: "Lookup is exact on every component."

    Each component listed in the fixture is perturbed in turn; a database holding the
    unperturbed record must miss every time. A component that did not differentiate would let
    a certification silently survive a change §24 says invalidates it (R54, R66).
    """
    complete = _key(key_fixture["completeKey"])
    record = certification.CertificationRecord(
        key=complete,
        device_floor_95_deg=1.2,
        supported_quality_grade=QualityGrade.USABLE,
        earned_under_engine_version="heading-3.2.0",
        evidence_report_id="report-for-this-test-only",
        certification_date="2026-01-01",
    )
    database = certification.CertificationDatabase.with_records((record,))

    # Sanity: the exact key hits.
    hit = database.lookup(complete, profile["unknownDeviceFloor95Deg"])
    assert hit.bound_calibration_state is BoundCalibrationState.CALIBRATED
    assert hit.device_floor_95_deg == 1.2

    field_for_component = {
        "providerErrorSource": ("provider_error_source", ProviderErrorSource.GOOGLE_ORDINARY),
        "locationProviderRuntimeIdentity": (
            "location_provider_runtime_identity",
            "GMS:some-other-exact-version",
        ),
        "placementProfileHash": ("placement_profile_hash", "sha256:0123456789abcdef"),
        "geomagneticErrorModelHash": ("geomagnetic_error_model_hash", "sha256:fedcba9876543210"),
        "engineDecisionLogicHash": ("engine_decision_logic_hash", "sha256:1111111111111111"),
        "precisionConfigHash": ("precision_config_hash", "sha256:2222222222222222"),
        "measurementMode": ("measurement_mode", MeasurementMode.WALL_FLUSH_BACK),
    }
    for component in key_fixture["keyComponentsThatMustDifferentiate"]:
        field, replacement = field_for_component[component]
        perturbed = dataclasses.replace(complete, **{field: replacement})
        outcome = database.lookup(perturbed, profile["unknownDeviceFloor95Deg"])
        assert outcome.bound_calibration_state is BoundCalibrationState.CANDIDATE, component
        assert outcome.record is None, component
        assert outcome.device_floor_95_deg == profile["unknownDeviceFloor95Deg"], component


def test_open_ended_identities_are_rejected(key_fixture):
    """§24/R66: exact observed identities only — never a range that admits a future release."""
    for open_ended in key_fixture["rejectedOpenEndedIdentities"]:
        document = dict(key_fixture["completeKey"])
        document["osBuildIdentity"] = open_ended
        with pytest.raises(certification.CertificationKeyError):
            _key(document)


def test_the_not_runtime_observable_sentinel_is_accepted(key_fixture):
    """§24: an unobservable distinction is handled by an explicit sentinel with pooled
    worst-case evidence, not by an invented key field."""
    document = dict(key_fixture["completeKey"])
    document["sensorRuntimeIdentity"] = key_fixture["notRuntimeObservableSentinel"]
    key = _key(document)
    assert key.sensor_runtime_identity == certification.NOT_RUNTIME_OBSERVABLE


def test_an_empty_component_is_rejected(key_fixture):
    document = dict(key_fixture["completeKey"])
    document["engineDecisionLogicHash"] = "  "
    with pytest.raises(certification.CertificationKeyError):
        _key(document)


def test_a_record_exists_only_for_calibrated(key_fixture):
    """§24: storing a ``CANDIDATE`` record invites editing its state field."""
    with pytest.raises(ValueError):
        certification.CertificationRecord(
            key=_key(key_fixture["completeKey"]),
            device_floor_95_deg=1.2,
            supported_quality_grade=QualityGrade.USABLE,
            earned_under_engine_version="heading-3.2.0",
            evidence_report_id="report",
            certification_date="2026-01-01",
            bound_calibration_state=BoundCalibrationState.CANDIDATE,
        )


def test_a_record_without_resolvable_evidence_is_rejected(key_fixture):
    """§24: ``evidenceReportId`` MUST resolve to archived raw telemetry."""
    with pytest.raises(ValueError):
        certification.CertificationRecord(
            key=_key(key_fixture["completeKey"]),
            device_floor_95_deg=1.2,
            supported_quality_grade=QualityGrade.USABLE,
            earned_under_engine_version="heading-3.2.0",
            evidence_report_id="",
            certification_date="2026-01-01",
        )


# --------------------------------------------------------------------------------------
# §19.1 invariants
# --------------------------------------------------------------------------------------
def test_the_calibration_invariants_hold_both_ways():
    """``CALIBRATED <=> EMPIRICALLY_CALIBRATED``; ``CANDIDATE => {TARGET_ONLY, UNDEFINED}``.

    The two fields are near-redundant by design — one is the gate, the other the claim — and
    the redundancy is safe only while the invariant holds, because drift lets a ``95%`` label
    appear on a ``CANDIDATE`` measurement (failure mode 31).
    """
    certification.assert_calibration_invariants(
        BoundCalibrationState.CALIBRATED,
        UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
    )
    certification.assert_calibration_invariants(
        BoundCalibrationState.CANDIDATE, UncertaintyCoverageEvidenceState.TARGET_ONLY
    )
    certification.assert_calibration_invariants(
        BoundCalibrationState.CANDIDATE, UncertaintyCoverageEvidenceState.UNDEFINED
    )
    for state, evidence in (
        (BoundCalibrationState.CALIBRATED, UncertaintyCoverageEvidenceState.TARGET_ONLY),
        (BoundCalibrationState.CALIBRATED, UncertaintyCoverageEvidenceState.UNDEFINED),
        (
            BoundCalibrationState.CANDIDATE,
            UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
        ),
    ):
        with pytest.raises(ValueError):
            certification.assert_calibration_invariants(state, evidence)


def test_the_invariants_are_enforced_on_every_lookup_outcome(key_fixture, profile):
    """Not only in a helper a caller might forget: the outcome type asserts on construction."""
    with pytest.raises(ValueError):
        certification.CertificationLookupOutcome(
            bound_calibration_state=BoundCalibrationState.CANDIDATE,
            uncertainty_coverage_evidence_state=(
                UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED
            ),
            device_floor_95_deg=profile["unknownDeviceFloor95Deg"],
            supported_quality_grade=QualityGrade.USABLE,
            record=None,
        )


def test_boundcalibrationstate_is_never_read_from_config(profile):
    """§19.1/§8.1/failure mode 32: it is derived from a §24 lookup, never configured.

    §8.1's INV-01 already forbids the key; this asserts the *positive* half — the value the
    engine uses comes from a lookup outcome.
    """
    assert not any("calibrationstate" in key.lower() for key in profile)
    outcome = certification.CertificationDatabase.shipped().lookup(
        _key(fixtures.load(fixtures.CERTIFICATION_KEY)["completeKey"]),
        profile["unknownDeviceFloor95Deg"],
    )
    assert outcome.bound_calibration_state is BoundCalibrationState.CANDIDATE


# --------------------------------------------------------------------------------------
# §19.3 deviation correction, whose hash is a key component
# --------------------------------------------------------------------------------------
def test_the_production_deviation_state_is_none():
    """§19.3: default production state is ``NONE``, correction ``0.0``, true = uncorrected."""
    assert deviation.lookup_deviation_profile("any", "live", "context") is None
    outcome = deviation.apply_deviation_correction(189.0)
    assert outcome.state is DeviationCorrectionState.NONE
    assert outcome.correction_deg == 0.0
    assert outcome.true_heading_deg == outcome.uncorrected_true_heading_deg == 189.0
    assert outcome.profile_hash == deviation.NONE_PROFILE_HASH
    assert outcome.residual_bound_95_deg == 0.0


def test_the_none_sentinel_is_a_literal_not_a_null(key_fixture):
    """§24: "literal NONE when correction is disabled" — a missing component must not match."""
    assert key_fixture["completeKey"]["deviationCorrectionProfileHash"] == "NONE"
    assert deviation.NONE_PROFILE_HASH == "NONE"


def _profile_with_scope(scope: DeviationCorrectionScope):
    return deviation.DeviationCorrectionProfileMetadata(
        profile_id="test-profile",
        profile_hash="sha256:abc",
        scope=scope,
        structure_class=DeviationStructureClass.MODEL_CLASS_STABLE,
        correction_method_id="circular-harmonic-v1",
        measurement_mode=MeasurementMode.FLAT_TOP_EDGE,
        placement_method=PlacementMethod.NONMAGNETIC_ALIGNMENT_JIG,
        provider_id=ProviderId.GOOGLE_FOP,
        covered_provider_runtime_identities=("GMS:exact",),
        covered_os_build_identities=("exact-build",),
        geomagnetic_model_id="WMM2025",
        geomagnetic_coefficient_hash="sha256:def",
        precision_config_hash="sha256:ghi",
        held_out_residual_bound_95_deg=0.4,
        training_evidence_id="train",
        held_out_evidence_id="heldout",
    )


def test_a_unit_scope_profile_can_never_produce_calibrated_output():
    """§19.3/§30.6: v1's certification database intentionally does not bind to physical-unit
    identity, so a per-unit correction cannot be matched by a runtime lookup."""
    unit = _profile_with_scope(DeviationCorrectionScope.UNIT)
    assert not unit.may_produce_calibrated_output
    with pytest.raises(ValueError):
        deviation.apply_deviation_correction(189.0, 0.3, unit)


def test_a_model_class_profile_applies_exactly_once():
    """§19.3: applied once, after reference resolution and before lock aggregation."""
    model_class = _profile_with_scope(DeviationCorrectionScope.MODEL_CLASS)
    outcome = deviation.apply_deviation_correction(355.0, 8.0, model_class)
    assert outcome.state is DeviationCorrectionState.CERTIFIED_PROFILE
    assert outcome.uncorrected_true_heading_deg == 355.0
    assert outcome.true_heading_deg == pytest.approx(3.0, abs=1e-9)
    assert outcome.residual_bound_95_deg == 0.4


def test_a_real_profile_may_not_use_the_none_sentinel_as_its_hash():
    with pytest.raises(ValueError):
        dataclasses.replace(
            _profile_with_scope(DeviationCorrectionScope.MODEL_CLASS), profile_hash="NONE"
        )
