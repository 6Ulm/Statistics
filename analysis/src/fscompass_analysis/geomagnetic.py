"""SPEC.md §10, §10.2, §10.3, §19.2 — geomagnetic model contract and uncertainty.

What this module **is**: the typed contract around the vendored NOAA model — altitude datum
handling, decimal-year conversion, validity enforcement, the
:class:`GeomagneticModelUncertainty` shape, and ``boundFromSigma`` as the single conversion
from one sigma to a 95% bound.

What this module is **not**: a geomagnetic model. §10 requires the *same official NOAA C
sources* compiled on both platforms and forbids separate Swift and Kotlin ports of the
spherical-harmonic core; §10.3 states that an implementation which "derives a sigma from the
coefficients, or substitutes a remembered global constant, has invented the quantity". The
NOAA artifacts are not vendored (``docs/IMPLEMENTATION_NOTES.md`` D-2), so every path that
would need a coefficient, an error-model formula or a hash raises
:class:`VendoredModelUnavailable` naming the missing artifact. A refusal is the honest state;
a plausible number is not (§5: missing is never zero, §37 "do not hide incomplete work
behind placeholders").
"""

from __future__ import annotations

import datetime as _datetime
import math
from dataclasses import dataclass
from enum import Enum
from pathlib import Path

from .circular import bound_from_sigma
from .enums import AltitudeReference, GeomagneticModelId

__all__ = [
    "AltitudeSample",
    "ConfidenceLevel",
    "GeomagneticModelUncertainty",
    "VendoredModelUnavailable",
    "VendoredWmmArtifacts",
    "WMM2025_VALIDITY",
    "declination_bound_95_deg",
    "ellipsoidal_altitude_m",
    "is_within_validity",
    "wmm_decimal_year",
]


class VendoredModelUnavailable(RuntimeError):
    """The vendored NOAA artifact this operation needs does not exist.

    Raised instead of returning a remembered constant. The message names the exact artifact
    and the rule that forbids substituting for it.
    """


class GeomagneticDateOutOfRange(ValueError):
    """§10: ``GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE``.

    An app installed near epoch end **will** outlive its coefficients on some devices; the
    model refuses and the app surfaces "model expired, update the app" rather than
    extrapolating.
    """


class AltitudeDatumUnconverted(ValueError):
    """§10.2: an orthometric or unknown altitude reached the model without conversion.

    "wrapper tests MUST prove an orthometric input is converted or refused, never silently
    treated as ellipsoidal". Geoid separation exceeds 100 m in places, so this is a silent
    systematic cross-platform divergence — fixed by typing, not by measuring its effect.
    """


@dataclass(frozen=True)
class AltitudeSample:
    """§5 ``AltitudeSample``. The datum is always explicit; ``UNKNOWN`` is a real state."""

    value_m: float
    reference: AltitudeReference
    vertical_accuracy_m: float | None = None

    def __post_init__(self) -> None:
        if not math.isfinite(self.value_m):
            raise ValueError(f"AltitudeSample.value_m must be finite, got {self.value_m!r}")
        if self.vertical_accuracy_m is not None and not math.isfinite(self.vertical_accuracy_m):
            raise ValueError("AltitudeSample.vertical_accuracy_m must be finite when present")


def ellipsoidal_altitude_m(
    altitude: AltitudeSample,
    geoid_separation_m: float | None = None,
) -> float:
    """Return WGS84 ellipsoidal height, or refuse (§10.2).

    Canonical model input is ellipsoidal height. An ``MSL_ORTHOMETRIC`` sample is accepted
    only with an explicitly supplied, documented geoid separation; ``UNKNOWN`` is never
    coerced, because ``UNKNOWN`` "is not a synonym for either datum".
    """
    if altitude.reference is AltitudeReference.WGS84_ELLIPSOID:
        return altitude.value_m
    if altitude.reference is AltitudeReference.MSL_ORTHOMETRIC:
        if geoid_separation_m is None:
            raise AltitudeDatumUnconverted(
                "MSL_ORTHOMETRIC altitude requires a documented geoid separation before it "
                "can enter the model; §10.2 forbids treating it as ellipsoidal"
            )
        if not math.isfinite(geoid_separation_m):
            raise ValueError("geoid_separation_m must be finite")
        return altitude.value_m + geoid_separation_m
    raise AltitudeDatumUnconverted(
        "altitude reference is UNKNOWN; it downgrades quality and is not a datum. §10.2"
    )


def wmm_decimal_year(instant: _datetime.datetime) -> float:
    """§9/§10 ``wmmDecimalYear`` — UTC instant to decimal year, leap years included.

    ``year + elapsed / length_of_that_year``, where the year length is measured from the two
    UTC year boundaries so 2028 gets 366 days without a leap-year branch to get wrong.
    """
    if instant.tzinfo is None:
        raise ValueError(
            "wmmDecimalYear requires a timezone-aware UTC instant; a naive datetime "
            "silently adopts the host timezone (failure mode 10)"
        )
    utc = instant.astimezone(_datetime.timezone.utc)
    start = _datetime.datetime(utc.year, 1, 1, tzinfo=_datetime.timezone.utc)
    end = _datetime.datetime(utc.year + 1, 1, 1, tzinfo=_datetime.timezone.utc)
    return utc.year + (utc - start).total_seconds() / (end - start).total_seconds()


@dataclass(frozen=True)
class ValidityInterval:
    """``[start, end)`` in decimal years, per model (§10)."""

    start_decimal_year: float
    end_decimal_year: float

    def contains(self, decimal_year: float) -> bool:
        return self.start_decimal_year <= decimal_year < self.end_decimal_year


#: §10: "WMM2025's v1 epoch interval is 2025.0 <= decimalYear < 2030.0". This interval is
#: stated in SPEC.md itself, so it is a specification constant rather than a value read out
#: of an unvendored artifact. The **coefficients** it applies to are still absent (D-2), so
#: nothing here can evaluate a field.
WMM2025_VALIDITY = ValidityInterval(2025.0, 2030.0)


def is_within_validity(decimal_year: float, validity: ValidityInterval) -> bool:
    return validity.contains(decimal_year)


def require_within_validity(decimal_year: float, validity: ValidityInterval) -> float:
    if not validity.contains(decimal_year):
        raise GeomagneticDateOutOfRange(
            f"decimal year {decimal_year} is outside "
            f"[{validity.start_decimal_year}, {validity.end_decimal_year}); "
            "emit GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE and prompt to update the app"
        )
    return decimal_year


class ConfidenceLevel(Enum):
    """The confidence semantics of an error quantity.

    Failure mode 9 — confidence-level conflation — is Critical: adding a one-sigma number to
    a sum of nominal 95% terms under-covers by roughly a factor of two. Every error term
    therefore carries its level explicitly and conversion happens in exactly one function.
    """

    ONE_STANDARD_DEVIATION = "ONE_STANDARD_DEVIATION"
    TWO_SIDED_95 = "TWO_SIDED_95"
    SIXTY_EIGHT_PERCENT = "SIXTY_EIGHT_PERCENT"


@dataclass(frozen=True)
class GeomagneticModelUncertainty:
    """§7/§10.3 ``GeomagneticModelUncertainty``.

    ``declination_sigma_1_deg`` comes from NOAA's separately published error model for that
    exact coefficient set — never from evaluating the coefficients, never from a remembered
    global constant. ``error_model_hash`` is therefore mandatory: §24 makes it a separate
    component of the certification key because the error model changes reported uncertainty
    even when coefficient evaluation is unchanged.
    """

    declination_sigma_1_deg: float
    source_model_id: GeomagneticModelId
    error_model_id: str
    error_model_hash: str
    source_document_reference: str
    source_confidence_level: ConfidenceLevel = ConfidenceLevel.ONE_STANDARD_DEVIATION

    def __post_init__(self) -> None:
        if self.source_confidence_level is not ConfidenceLevel.ONE_STANDARD_DEVIATION:
            raise ValueError(
                "§10.3 fixes sourceConfidenceLevel to ONE_STANDARD_DEVIATION for the NOAA "
                "declination error model; a different level needs its own conversion, not a "
                "relabelled field"
            )
        if not math.isfinite(self.declination_sigma_1_deg) or self.declination_sigma_1_deg < 0.0:
            raise ValueError("declinationSigma1Deg must be a finite, non-negative number of degrees")
        if not self.error_model_hash or self.error_model_hash == "NONE":
            raise ValueError(
                "a declination sigma without an error-model hash is an invented quantity "
                "(§10.3, §24)"
            )


def declination_bound_95_deg(
    uncertainty: GeomagneticModelUncertainty,
    sigma_to_bound_95_factor: float,
) -> float:
    """§19.2: the **only** path from a published sigma to this project's 95% bound.

    Applied exactly once, and recorded as an assumption (§35). The factor is versioned
    configuration, passed in rather than written here.
    """
    return bound_from_sigma(uncertainty.declination_sigma_1_deg, sigma_to_bound_95_factor)


@dataclass(frozen=True)
class VendoredWmmArtifacts:
    """The vendored artifact set for one model, or the declared absence of it.

    §10 requires the C source, the coefficient file, the separately published error model,
    the licence, the retrieval URL/date and a SHA-256 per file, all under
    ``third_party/noaa-wmm/``. Phase 1 could not fetch them; this type makes the absence a
    typed, testable state rather than a comment.
    """

    model_id: GeomagneticModelId
    coefficient_hash: str | None
    error_model_hash: str | None
    source_hash: str | None

    NOT_VENDORED: str = "NOT_VENDORED"

    @property
    def is_vendored(self) -> bool:
        return None not in (self.coefficient_hash, self.error_model_hash, self.source_hash)

    def require_vendored(self, operation: str) -> "VendoredWmmArtifacts":
        if not self.is_vendored:
            raise VendoredModelUnavailable(
                f"{operation} needs the vendored NOAA {self.model_id.value} artifacts "
                "(C source, coefficients, error model, hashes) under third_party/noaa-wmm/. "
                "They are NOT_VENDORED — see third_party/noaa-wmm/UPSTREAM.md and "
                "docs/IMPLEMENTATION_NOTES.md D-2. §10.3 forbids deriving a sigma from the "
                "coefficients or substituting a remembered constant, and §2.3 forbids a "
                "home-grown replacement for the official reference implementation."
            )
        return self


def vendored_artifacts(model_id: GeomagneticModelId, repository_root: Path) -> VendoredWmmArtifacts:
    """Read the vendored state for ``model_id`` from ``third_party/noaa-wmm/sha256.txt``.

    Absence is reported, never guessed. The manifest is the single source of truth for what
    has actually been vendored; ``scripts/verify-artifacts.sh`` independently fails the build
    if the files and the manifest ever stop agreeing.
    """
    manifest = repository_root / "third_party" / "noaa-wmm" / "sha256.txt"
    entries: dict[str, str] = {}
    if manifest.is_file():
        for line in manifest.read_text(encoding="utf-8").splitlines():
            stripped = line.strip()
            if not stripped or stripped.startswith("#"):
                continue
            digest, _, path = stripped.partition("  ")
            if digest and path:
                entries[path.strip()] = digest

    token = model_id.value.lower()

    def find(directory: str) -> str | None:
        for path, digest in entries.items():
            if path.startswith(f"{directory}/") and token in path.lower():
                return f"sha256:{digest}"
        return None

    return VendoredWmmArtifacts(
        model_id=model_id,
        coefficient_hash=find("coefficients"),
        error_model_hash=find("error-model"),
        source_hash=next(
            (f"sha256:{digest}" for path, digest in entries.items() if path.startswith("src/")),
            None,
        ),
    )
