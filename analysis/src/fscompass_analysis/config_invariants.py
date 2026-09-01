"""SPEC.md §8.1 "Enforced invariants" — the third runtime.

§36 makes these part of Phase 0 and §33.1 runs them on every commit. Invariant identifiers
match the Kotlin and Swift implementations so a failure reads the same everywhere.
"""

from __future__ import annotations

import re
from dataclasses import dataclass
from typing import Any, Callable, Iterator

CALIBRATION_STATE_KEY = re.compile("calibrationState", re.IGNORECASE)


@dataclass(frozen=True)
class InvariantViolation:
    invariant_id: str
    requirement: str
    prevents: str
    detail: str

    def __str__(self) -> str:
        return (
            f"[{self.invariant_id}] {self.requirement} -- observed: {self.detail} "
            f"-- prevents: {self.prevents}"
        )


def iter_property_names(value: Any) -> Iterator[str]:
    """Every property name at any nesting depth.

    §8.1's first invariant is "no key matching ``/calibrationState/i`` exists **anywhere**
    in the profile", so a nested object may not smuggle one in either.
    """
    if isinstance(value, dict):
        for key, nested in value.items():
            yield key
            yield from iter_property_names(nested)
    elif isinstance(value, list):
        for element in value:
            yield from iter_property_names(element)


def check(profile: dict[str, Any]) -> list[InvariantViolation]:
    violations: list[InvariantViolation] = []

    def require(
        invariant_id: str,
        holds: bool,
        requirement: str,
        prevents: str,
        detail: Callable[[], str],
    ) -> None:
        if not holds:
            violations.append(InvariantViolation(invariant_id, requirement, prevents, detail()))

    offending = [n for n in iter_property_names(profile) if CALIBRATION_STATE_KEY.search(n)]
    require(
        "INV-01-NO-CALIBRATION-STATE-KEY",
        not offending,
        "No key matching /calibrationState/i exists anywhere in the profile",
        "boundCalibrationState is derived from a §24 certification lookup (§19.1). One "
        "editable value that turns every device Professional is the shortcut an agent under "
        "pressure takes.",
        lambda: f"offending keys: {offending}",
    )

    require(
        "INV-02-REFERENCE-SEPARATION-ORDERING",
        profile["referenceSeparationMarginDeg"] <= profile["smallDeclinationAmbiguityMaxDeg"],
        "referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg",
        "Since rMag - rTrue <= abs(d), a margin above the ambiguity allowance creates a "
        "declination dead band that always resolves UNVERIFIED with no visible cause (§11).",
        lambda: f"{profile['referenceSeparationMarginDeg']} vs "
        f"{profile['smallDeclinationAmbiguityMaxDeg']}",
    )

    require(
        "INV-03-GRADE-THRESHOLD-ORDERING",
        profile["professionalBound95MaxDeg"]
        < profile["highBound95MaxDeg"]
        < profile["usableBound95MaxDeg"]
        < profile["lowConfidenceBound95MaxDeg"],
        "professionalBound95MaxDeg < highBound95MaxDeg < usableBound95MaxDeg "
        "< lowConfidenceBound95MaxDeg",
        "Grade function must be total and ordered.",
        lambda: ", ".join(
            str(profile[k])
            for k in (
                "professionalBound95MaxDeg",
                "highBound95MaxDeg",
                "usableBound95MaxDeg",
                "lowConfidenceBound95MaxDeg",
            )
        ),
    )

    require(
        "INV-04-FREEHAND-CANNOT-REACH-PROFESSIONAL",
        profile["professionalBound95MaxDeg"] < profile["flatFreehandPlacementBound95Deg"],
        "professionalBound95MaxDeg < flatFreehandPlacementBound95Deg",
        "Encodes in config that freehand cannot reach the top grade (§20). A future edit "
        "breaking this trips the intended alarm.",
        lambda: f"{profile['professionalBound95MaxDeg']} vs "
        f"{profile['flatFreehandPlacementBound95Deg']}",
    )

    require(
        "INV-05-DECLINATION-ENVELOPE-ORDERING",
        profile["declinationEnvelopeProfessionalMaxDeg"]
        <= profile["declinationEnvelopeUsableMaxDeg"],
        "declinationEnvelopeProfessionalMaxDeg <= declinationEnvelopeUsableMaxDeg",
        "Ordered gates.",
        lambda: f"{profile['declinationEnvelopeProfessionalMaxDeg']} vs "
        f"{profile['declinationEnvelopeUsableMaxDeg']}",
    )

    suspect_disturbed_pairs = (
        (
            "magnitude",
            "magneticMagnitudeResidualSuspectFraction",
            "magneticMagnitudeResidualDisturbedFraction",
        ),
        ("inclination", "inclinationResidualSuspectDeg", "inclinationResidualDisturbedDeg"),
        (
            "stationaryMad",
            "stationaryFieldMadSuspectMicroTesla",
            "stationaryFieldMadDisturbedMicroTesla",
        ),
        ("pipeline", "pipelineDisagreementSuspectDeg", "pipelineDisagreementDisturbedDeg"),
    )
    for name, suspect_key, disturbed_key in suspect_disturbed_pairs:
        require(
            f"INV-06-SUSPECT-BELOW-DISTURBED-{name}",
            profile[suspect_key] < profile[disturbed_key],
            f"suspect < disturbed for the {name} pair",
            "A suspect threshold above disturbed makes SUSPECT unreachable.",
            lambda s=suspect_key, d=disturbed_key: f"suspect={profile[s]}, disturbed={profile[d]}",
        )

    # §8.1: periodic support streams request 50 Hz and the gate tolerates a 50% callback
    # shortfall. This invariant does NOT apply to event-driven CLHeading; flat iOS has a
    # separate in-window heading-anchor count (§12, R52).
    achievable = (
        profile["stableWindowMinMs"] * (profile["periodicOrientationRequestedHz"] / 2.0) / 1000.0
    )
    require(
        "INV-07-PERIODIC-SUPPORT-SAMPLES-ACHIEVABLE",
        achievable >= profile["minPeriodicSupportSamples"],
        "stableWindowMinMs * (periodicOrientationRequestedHz / 2) / 1000 "
        ">= minPeriodicSupportSamples",
        "Periodic support streams request 50 Hz; the candidate gate tolerates a 50% callback "
        "shortfall. Does not apply to event-driven CLHeading.",
        lambda: f"achievable={achievable}, required={profile['minPeriodicSupportSamples']}",
    )

    require(
        "INV-08-ORIENTATION-AGE-ORDERING",
        profile["orientationMaxAgeMs"] < profile["orientationInvalidAfterMs"],
        "orientationMaxAgeMs < orientationInvalidAfterMs",
        "Drop and invalidate are different thresholds.",
        lambda: f"{profile['orientationMaxAgeMs']} vs {profile['orientationInvalidAfterMs']}",
    )

    require(
        "INV-09-LOCATION-FRESHNESS-ORDERING",
        profile["freshLocationAtStartMaxAgeMs"]
        <= profile["locationAtLockMaxAgeMs"]
        <= profile["usableLocationMaxAgeMs"],
        "freshLocationAtStartMaxAgeMs <= locationAtLockMaxAgeMs <= usableLocationMaxAgeMs",
        "Ordered freshness tiers.",
        lambda: ", ".join(
            str(profile[k])
            for k in (
                "freshLocationAtStartMaxAgeMs",
                "locationAtLockMaxAgeMs",
                "usableLocationMaxAgeMs",
            )
        ),
    )

    require(
        "INV-10-SPACE-WEATHER-ORDERING",
        profile["spaceWeatherAdvisoryKpMin"]
        <= profile["spaceWeatherProfessionalSuppressKpMin"]
        < profile["spaceWeatherRejectKpMin"],
        "spaceWeatherAdvisoryKpMin <= spaceWeatherProfessionalSuppressKpMin "
        "< spaceWeatherRejectKpMin",
        "Ordered advisory/suppression/refusal tiers.",
        lambda: ", ".join(
            str(profile[k])
            for k in (
                "spaceWeatherAdvisoryKpMin",
                "spaceWeatherProfessionalSuppressKpMin",
                "spaceWeatherRejectKpMin",
            )
        ),
    )

    # §36 Phase 1: "config parser/validator including periodic-vs-event-driven sampling
    # invariants". INV-07 above bounds the *periodic* stream against the requested rate; this
    # one guards the other half, which R52 records as a Critical failure: requiring iOS flat
    # to deliver 50 CLHeading events in 2 s, or applying the periodic freshness rule to a
    # stationary CLHeading value, rejects a perfectly good measurement. The anchor minimum is
    # therefore a small positive count that is *independent* of periodicOrientationRequestedHz
    # — if it were ever derived from the rate, the event-driven path would inherit the
    # periodic contract by arithmetic instead of by decision.
    anchor_minimum = profile["clHeadingMinSamplesPerStableWindow"]
    rate_derived_minimum = (
        profile["stableWindowMinMs"] * profile["periodicOrientationRequestedHz"] / 1000.0
    )
    require(
        "INV-11-EVENT-DRIVEN-ANCHOR-MINIMUM",
        1 <= anchor_minimum < rate_derived_minimum,
        "1 <= clHeadingMinSamplesPerStableWindow < "
        "stableWindowMinMs * periodicOrientationRequestedHz / 1000",
        "CLHeading is event-driven, not a guaranteed 50 Hz stream (§12, R52). At least one "
        "valid anchor must fall inside the stable window, and the anchor count must stay "
        "strictly below what the periodic stream would produce, so the event-driven path "
        "never silently inherits the periodic sampling contract.",
        lambda: (
            f"clHeadingMinSamplesPerStableWindow={anchor_minimum}, "
            f"rate-derived periodic count={rate_derived_minimum}"
        ),
    )

    return violations
