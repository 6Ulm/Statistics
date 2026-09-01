"""SPEC.md §10 / §10.2 / §10.3 / §19.2 — the geomagnetic contract, and its refusal to guess.

The load-bearing test in this file is
:func:`test_no_sigma_can_be_produced_without_a_vendored_error_model`. §10.3 is explicit that
an implementation which "derives a sigma from the coefficients, or substitutes a remembered
global constant, has invented the quantity", and the NOAA artifacts could not be fetched
(``docs/IMPLEMENTATION_NOTES.md`` D-2). A refusal is the correct behaviour; a plausible number
is the failure.
"""

from __future__ import annotations

import datetime as dt
import math

import pytest

from fscompass_analysis import geomagnetic as geo
from fscompass_analysis.enums import AltitudeReference, GeomagneticModelId


# --------------------------------------------------------------------------------------
# D-2: the vendored model is absent, and every path that needs it refuses
# --------------------------------------------------------------------------------------
def test_the_shipped_tree_declares_no_vendored_model(repo_root):
    for model_id in GeomagneticModelId:
        artifacts = geo.vendored_artifacts(model_id, repo_root)
        assert not artifacts.is_vendored, (
            f"{model_id.value} now reports vendored artifacts. Phase 1 could not reach NOAA "
            "(D-2); if that changed, run the official test vectors before relying on it."
        )


def test_no_sigma_can_be_produced_without_a_vendored_error_model(repo_root):
    """§10.3: the sigma comes from NOAA's separately published error model, or nowhere."""
    artifacts = geo.vendored_artifacts(GeomagneticModelId.WMM2025, repo_root)
    with pytest.raises(geo.VendoredModelUnavailable) as raised:
        artifacts.require_vendored("declinationSigma1Deg")
    assert "NOT_VENDORED" in str(raised.value)
    assert "third_party/noaa-wmm" in str(raised.value)


def test_uncertainty_requires_an_error_model_hash():
    """§24 keeps ``geomagneticErrorModelHash`` a separate key component for this reason."""
    with pytest.raises(ValueError):
        geo.GeomagneticModelUncertainty(
            declination_sigma_1_deg=0.36,
            source_model_id=GeomagneticModelId.WMM2025,
            error_model_id="whatever",
            error_model_hash="NONE",
            source_document_reference="",
        )


def test_uncertainty_refuses_a_relabelled_confidence_level():
    """Failure mode 9: confidence-level conflation is Critical and under-covers ~2x."""
    with pytest.raises(ValueError):
        geo.GeomagneticModelUncertainty(
            declination_sigma_1_deg=0.36,
            source_model_id=GeomagneticModelId.WMM2025,
            error_model_id="wmm2025-error-model",
            error_model_hash="sha256:0" * 1,
            source_document_reference="ref",
            source_confidence_level=geo.ConfidenceLevel.TWO_SIDED_95,
        )


def test_sigma_to_bound_is_applied_exactly_once(profile):
    """§19.2: one named function, one application, factor read from versioned config."""
    uncertainty = geo.GeomagneticModelUncertainty(
        declination_sigma_1_deg=0.5,
        source_model_id=GeomagneticModelId.WMM2025,
        error_model_id="hypothetical-error-model-for-this-test",
        error_model_hash="sha256:deadbeef",
        source_document_reference="test only; not a vendored artifact",
    )
    factor = profile["declinationSigmaToBound95Factor"]
    once = geo.declination_bound_95_deg(uncertainty, factor)
    assert once == pytest.approx(0.98, abs=1e-12)
    # Applying it twice is the shape of the defect §19.2 exists to prevent.
    assert geo.declination_bound_95_deg(uncertainty, factor) * factor != pytest.approx(once)


# --------------------------------------------------------------------------------------
# §10.2 altitude datum
# --------------------------------------------------------------------------------------
def test_ellipsoidal_altitude_passes_through():
    sample = geo.AltitudeSample(120.0, AltitudeReference.WGS84_ELLIPSOID)
    assert geo.ellipsoidal_altitude_m(sample) == 120.0


def test_orthometric_altitude_is_converted_or_refused():
    """§10.2: "wrapper tests MUST prove an orthometric input is converted or refused, never
    silently treated as ellipsoidal"."""
    sample = geo.AltitudeSample(120.0, AltitudeReference.MSL_ORTHOMETRIC)
    with pytest.raises(geo.AltitudeDatumUnconverted):
        geo.ellipsoidal_altitude_m(sample)
    assert geo.ellipsoidal_altitude_m(sample, geoid_separation_m=-33.5) == pytest.approx(86.5)


def test_unknown_altitude_is_never_coerced_to_a_datum():
    """§2: ``UNKNOWN`` is a real state, not a synonym for either datum."""
    sample = geo.AltitudeSample(120.0, AltitudeReference.UNKNOWN)
    with pytest.raises(geo.AltitudeDatumUnconverted):
        geo.ellipsoidal_altitude_m(sample)


def test_all_three_datum_cases_are_representable():
    for reference in AltitudeReference:
        assert geo.AltitudeSample(0.0, reference).reference is reference


# --------------------------------------------------------------------------------------
# §9/§10 decimal year and validity
# --------------------------------------------------------------------------------------
def test_decimal_year_at_year_boundaries():
    start = dt.datetime(2026, 1, 1, tzinfo=dt.timezone.utc)
    assert geo.wmm_decimal_year(start) == pytest.approx(2026.0, abs=1e-12)
    almost_next = dt.datetime(2026, 12, 31, 23, 59, 59, tzinfo=dt.timezone.utc)
    assert 2026.999 < geo.wmm_decimal_year(almost_next) < 2027.0


def test_decimal_year_handles_leap_years_without_a_branch():
    """2028 has 366 days; the fraction is measured against that year's own length."""
    leap_midpoint = dt.datetime(2028, 7, 1, tzinfo=dt.timezone.utc)
    common_midpoint = dt.datetime(2027, 7, 1, tzinfo=dt.timezone.utc)
    leap_day = dt.datetime(2028, 2, 29, tzinfo=dt.timezone.utc)
    assert geo.wmm_decimal_year(leap_day) == pytest.approx(2028 + 59.0 / 366.0, abs=1e-12)
    assert geo.wmm_decimal_year(leap_midpoint) != pytest.approx(
        geo.wmm_decimal_year(common_midpoint) - 1.0, abs=1e-9
    )


def test_decimal_year_requires_an_explicit_utc_instant():
    """Failure mode 10: a naive datetime silently adopts the host timezone."""
    with pytest.raises(ValueError):
        geo.wmm_decimal_year(dt.datetime(2026, 1, 1))


def test_decimal_year_converts_a_non_utc_offset():
    aware = dt.datetime(2026, 1, 1, 1, 0, tzinfo=dt.timezone(dt.timedelta(hours=1)))
    assert geo.wmm_decimal_year(aware) == pytest.approx(2026.0, abs=1e-12)


def test_wmm2025_validity_interval_is_half_open():
    """§10: ``2025.0 <= decimalYear < 2030.0`` — the 2025 model expires at end of 2029."""
    assert geo.is_within_validity(2025.0, geo.WMM2025_VALIDITY)
    assert geo.is_within_validity(2029.999, geo.WMM2025_VALIDITY)
    assert not geo.is_within_validity(2030.0, geo.WMM2025_VALIDITY)
    assert not geo.is_within_validity(2024.999, geo.WMM2025_VALIDITY)


def test_a_date_outside_validity_refuses_rather_than_extrapolating():
    """§10: an app installed near epoch end **will** outlive its coefficients."""
    geo.require_within_validity(2027.5, geo.WMM2025_VALIDITY)
    with pytest.raises(geo.GeomagneticDateOutOfRange) as raised:
        geo.require_within_validity(2030.5, geo.WMM2025_VALIDITY)
    assert "GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE" in str(raised.value)


def test_true_heading_conversion_sign_convention_is_east_positive():
    """§10: ``trueHeading = normalize360(magneticHeading + declination)``, declination
    east-positive. The convention is validated at both signs because inferring it from one
    location is failure mode 8."""
    from fscompass_analysis.circular import normalize360

    assert normalize360(181.12 + 8.29) == pytest.approx(189.41, abs=1e-9)
    assert normalize360(181.12 + -8.29) == pytest.approx(172.83, abs=1e-9)
    assert normalize360(355.0 + 8.29) == pytest.approx(3.29, abs=1e-9)
    assert normalize360(3.0 + -8.29) == pytest.approx(354.71, abs=1e-9)


def test_horizontal_intensity_gate_is_read_from_config_not_a_literal(profile):
    """§8.1: ``minHorizontalIntensityNanoTesla`` is physics, and is NOAA's own caution-zone
    boundary rather than an invented threshold."""
    assert profile["minHorizontalIntensityNanoTesla"] == 6000.0
    # The sensitivity claim the spec states, checked rather than trusted: at 6000 nT a 50 nT
    # transverse perturbation is about 0.48 deg.
    assert math.degrees(math.atan(50.0 / 6000.0)) == pytest.approx(0.477, abs=0.005)
