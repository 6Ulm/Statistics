package com.fengshuicompass.headingcore.config

import kotlinx.serialization.json.JsonObject

/**
 * A single §8.1 invariant that did not hold.
 *
 * @param invariantId stable identifier, shared with the Swift and Python implementations
 *   so a failure reads the same in all three runtimes.
 * @param requirement the invariant, quoted from SPEC.md §8.1.
 * @param prevents what the invariant prevents — §8.1 pairs every invariant with a
 *   specific silent failure, and a violation report that omits it invites "just relax
 *   the check".
 * @param detail the observed values.
 */
public data class InvariantViolation(
    val invariantId: String,
    val requirement: String,
    val prevents: String,
    val detail: String,
) {
    override fun toString(): String =
        "[$invariantId] $requirement -- observed: $detail -- prevents: $prevents"
}

/**
 * SPEC.md §8.1 "Enforced invariants": a build-time check of every row of that table.
 *
 * §36 makes these part of Phase 0 and §33.1 runs them on every commit. They are
 * intentionally implemented before any core logic: each one prevents a specific silent
 * failure that is invisible from reading the gate table.
 */
public object ConfigurationInvariants {

    /** The literal regex from §8.1's first row. */
    public val CALIBRATION_STATE_KEY: Regex = Regex("calibrationState", RegexOption.IGNORE_CASE)

    /**
     * @param profile the typed §8 profile.
     * @param rawTree the same document untyped, so the key scan sees nested objects too.
     * @return every violation found. An empty list means the configuration is admissible;
     *   the caller decides how to fail.
     */
    public fun check(profile: PrecisionProfile, rawTree: JsonObject): List<InvariantViolation> {
        val violations = mutableListOf<InvariantViolation>()

        fun require(
            id: String,
            holds: Boolean,
            requirement: String,
            prevents: String,
            detail: () -> String,
        ) {
            if (!holds) violations += InvariantViolation(id, requirement, prevents, detail())
        }

        val offendingKeys = collectPropertyNames(rawTree)
            .filter { CALIBRATION_STATE_KEY.containsMatchIn(it) }
        require(
            id = "INV-01-NO-CALIBRATION-STATE-KEY",
            holds = offendingKeys.isEmpty(),
            requirement = "No key matching /calibrationState/i exists anywhere in the profile",
            prevents = "boundCalibrationState is derived from a §24 certification lookup (§19.1). " +
                "One editable value that turns every device Professional is the shortcut an agent " +
                "under pressure takes.",
        ) { "offending keys: $offendingKeys" }

        require(
            id = "INV-02-REFERENCE-SEPARATION-ORDERING",
            holds = profile.referenceSeparationMarginDeg <= profile.smallDeclinationAmbiguityMaxDeg,
            requirement = "referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg",
            prevents = "Since rMag - rTrue <= abs(d), a margin above the ambiguity allowance creates " +
                "a declination dead band that always resolves UNVERIFIED with no visible cause (§11).",
        ) {
            "referenceSeparationMarginDeg=${profile.referenceSeparationMarginDeg}, " +
                "smallDeclinationAmbiguityMaxDeg=${profile.smallDeclinationAmbiguityMaxDeg}"
        }

        require(
            id = "INV-03-GRADE-THRESHOLD-ORDERING",
            holds = profile.professionalBound95MaxDeg < profile.highBound95MaxDeg &&
                profile.highBound95MaxDeg < profile.usableBound95MaxDeg &&
                profile.usableBound95MaxDeg < profile.lowConfidenceBound95MaxDeg,
            requirement = "professionalBound95MaxDeg < highBound95MaxDeg < usableBound95MaxDeg " +
                "< lowConfidenceBound95MaxDeg",
            prevents = "Grade function must be total and ordered.",
        ) {
            "professional=${profile.professionalBound95MaxDeg}, high=${profile.highBound95MaxDeg}, " +
                "usable=${profile.usableBound95MaxDeg}, lowConfidence=${profile.lowConfidenceBound95MaxDeg}"
        }

        require(
            id = "INV-04-FREEHAND-CANNOT-REACH-PROFESSIONAL",
            holds = profile.professionalBound95MaxDeg < profile.flatFreehandPlacementBound95Deg,
            requirement = "professionalBound95MaxDeg < flatFreehandPlacementBound95Deg",
            prevents = "Encodes in config that freehand cannot reach the top grade (§20). A future " +
                "edit breaking this trips the intended alarm.",
        ) {
            "professionalBound95MaxDeg=${profile.professionalBound95MaxDeg}, " +
                "flatFreehandPlacementBound95Deg=${profile.flatFreehandPlacementBound95Deg}"
        }

        require(
            id = "INV-05-DECLINATION-ENVELOPE-ORDERING",
            holds = profile.declinationEnvelopeProfessionalMaxDeg <= profile.declinationEnvelopeUsableMaxDeg,
            requirement = "declinationEnvelopeProfessionalMaxDeg <= declinationEnvelopeUsableMaxDeg",
            prevents = "Ordered gates.",
        ) {
            "professional=${profile.declinationEnvelopeProfessionalMaxDeg}, " +
                "usable=${profile.declinationEnvelopeUsableMaxDeg}"
        }

        // §8.1: "suspect < disturbed for magnitude, inclination, stationary-MAD, pipeline pairs".
        val suspectDisturbedPairs = listOf(
            Triple("magnitude", profile.magneticMagnitudeResidualSuspectFraction, profile.magneticMagnitudeResidualDisturbedFraction),
            Triple("inclination", profile.inclinationResidualSuspectDeg, profile.inclinationResidualDisturbedDeg),
            Triple("stationaryMad", profile.stationaryFieldMadSuspectMicroTesla, profile.stationaryFieldMadDisturbedMicroTesla),
            Triple("pipeline", profile.pipelineDisagreementSuspectDeg, profile.pipelineDisagreementDisturbedDeg),
        )
        suspectDisturbedPairs.forEach { (name, suspect, disturbed) ->
            require(
                id = "INV-06-SUSPECT-BELOW-DISTURBED-$name",
                holds = suspect < disturbed,
                requirement = "suspect < disturbed for the $name pair",
                prevents = "A suspect threshold above disturbed makes SUSPECT unreachable.",
            ) { "suspect=$suspect, disturbed=$disturbed" }
        }

        // §8.1: periodic support streams request 50 Hz and the gate tolerates a 50%
        // callback shortfall. This invariant does NOT apply to event-driven CLHeading;
        // flat iOS has its own in-window heading-anchor count (§12, R52).
        val achievableSupportSamples =
            profile.stableWindowMinMs * (profile.periodicOrientationRequestedHz / 2.0) / 1000.0
        require(
            id = "INV-07-PERIODIC-SUPPORT-SAMPLES-ACHIEVABLE",
            holds = achievableSupportSamples >= profile.minPeriodicSupportSamples,
            requirement = "stableWindowMinMs * (periodicOrientationRequestedHz / 2) / 1000 " +
                ">= minPeriodicSupportSamples",
            prevents = "Periodic support streams request 50 Hz; the candidate gate tolerates a 50% " +
                "callback shortfall. Does not apply to event-driven CLHeading.",
        ) {
            "achievable=$achievableSupportSamples, required=${profile.minPeriodicSupportSamples}"
        }

        require(
            id = "INV-08-ORIENTATION-AGE-ORDERING",
            holds = profile.orientationMaxAgeMs < profile.orientationInvalidAfterMs,
            requirement = "orientationMaxAgeMs < orientationInvalidAfterMs",
            prevents = "Drop and invalidate are different thresholds.",
        ) {
            "orientationMaxAgeMs=${profile.orientationMaxAgeMs}, " +
                "orientationInvalidAfterMs=${profile.orientationInvalidAfterMs}"
        }

        require(
            id = "INV-09-LOCATION-FRESHNESS-ORDERING",
            holds = profile.freshLocationAtStartMaxAgeMs <= profile.locationAtLockMaxAgeMs &&
                profile.locationAtLockMaxAgeMs <= profile.usableLocationMaxAgeMs,
            requirement = "freshLocationAtStartMaxAgeMs <= locationAtLockMaxAgeMs " +
                "<= usableLocationMaxAgeMs",
            prevents = "Ordered freshness tiers.",
        ) {
            "start=${profile.freshLocationAtStartMaxAgeMs}, atLock=${profile.locationAtLockMaxAgeMs}, " +
                "usable=${profile.usableLocationMaxAgeMs}"
        }

        require(
            id = "INV-10-SPACE-WEATHER-ORDERING",
            holds = profile.spaceWeatherAdvisoryKpMin <= profile.spaceWeatherProfessionalSuppressKpMin &&
                profile.spaceWeatherProfessionalSuppressKpMin < profile.spaceWeatherRejectKpMin,
            requirement = "spaceWeatherAdvisoryKpMin <= spaceWeatherProfessionalSuppressKpMin " +
                "< spaceWeatherRejectKpMin",
            prevents = "Ordered advisory/suppression/refusal tiers.",
        ) {
            "advisory=${profile.spaceWeatherAdvisoryKpMin}, " +
                "suppress=${profile.spaceWeatherProfessionalSuppressKpMin}, " +
                "reject=${profile.spaceWeatherRejectKpMin}"
        }

        // §36 Phase 1: "config parser/validator including periodic-vs-event-driven sampling
        // invariants". INV-07 above bounds the *periodic* stream against the requested rate;
        // this one guards the other half, which R52 records as a Critical failure: requiring
        // iOS flat to deliver 50 CLHeading events in 2 s, or applying the periodic freshness
        // rule to a stationary CLHeading value, rejects a perfectly good measurement. The
        // anchor minimum is therefore a small positive count that is *independent* of
        // periodicOrientationRequestedHz — if it were ever derived from the rate, the
        // event-driven path would inherit the periodic contract by arithmetic instead of by
        // decision.
        val rateDerivedPeriodicCount =
            profile.stableWindowMinMs * profile.periodicOrientationRequestedHz / 1000.0
        require(
            id = "INV-11-EVENT-DRIVEN-ANCHOR-MINIMUM",
            holds = profile.clHeadingMinSamplesPerStableWindow >= 1 &&
                profile.clHeadingMinSamplesPerStableWindow < rateDerivedPeriodicCount,
            requirement = "1 <= clHeadingMinSamplesPerStableWindow < " +
                "stableWindowMinMs * periodicOrientationRequestedHz / 1000",
            prevents = "CLHeading is event-driven, not a guaranteed 50 Hz stream (§12, R52). At " +
                "least one valid anchor must fall inside the stable window, and the anchor " +
                "count must stay strictly below what the periodic stream would produce, so the " +
                "event-driven path never silently inherits the periodic sampling contract.",
        ) {
            "clHeadingMinSamplesPerStableWindow=${profile.clHeadingMinSamplesPerStableWindow}, " +
                "rate-derived periodic count=$rateDerivedPeriodicCount"
        }

        return violations
    }
}
