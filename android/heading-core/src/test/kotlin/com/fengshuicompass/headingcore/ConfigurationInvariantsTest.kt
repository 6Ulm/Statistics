package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.ConfigurationInvariants
import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.config.collectPropertyNames
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/**
 * SPEC.md §8.1 "Enforced invariants" — the build-time test that §8.1 requires for each
 * row, and §36 places in Phase 0 before any core logic.
 *
 * Every test names the silent failure the invariant prevents, so a future reader who
 * wants to relax one sees the cost first.
 */
class ConfigurationInvariantsTest {

    private val profileFile = SharedArtifacts.precisionProfileFile
    private val profile: PrecisionProfile = PrecisionProfile.load(profileFile)
    private val rawTree: JsonObject = PrecisionProfile.loadRawTree(profileFile)

    @Test
    @DisplayName("§8: the shipped profile decodes strictly - an unknown key is a build failure")
    fun profileDecodesStrictly() {
        assertEquals("1.0.0", profile.schemaVersion)
        assertEquals("precision-v1-candidate-1", profile.configVersion)
    }

    @Test
    @DisplayName("§8.1: every invariant holds for the shipped configuration")
    fun allInvariantsHold() {
        val violations = ConfigurationInvariants.check(profile, rawTree)
        assertTrue(violations.isEmpty()) {
            "SPEC.md §8.1 invariant violations in ${profileFile.name}:\n" +
                violations.joinToString("\n") { "  $it" }
        }
    }

    @Test
    @DisplayName("§8.1 INV-01: no key matching /calibrationState/i exists anywhere in the profile")
    fun noCalibrationStateKeyAnywhere() {
        val offending = collectPropertyNames(rawTree)
            .filter { ConfigurationInvariants.CALIBRATION_STATE_KEY.containsMatchIn(it) }
        assertTrue(offending.isEmpty()) {
            "boundCalibrationState is derived from a §24 certification lookup (§19.1). " +
                "A configurable calibration state is failure mode 32. Offending keys: $offending"
        }
    }

    @Test
    @DisplayName("§8.1 INV-01: the check actually detects an injected calibration-state key")
    fun calibrationStateKeyDetectionIsNotVacuous() {
        // A passing assertion over an absent key proves nothing unless the detector fires
        // on a document that does contain one, including at nesting depth.
        val injected = Json.parseToJsonElement(
            """{ "a": 1, "nested": { "boundCalibrationState": "CALIBRATED" } }"""
        ) as JsonObject
        val offending = collectPropertyNames(injected)
            .filter { ConfigurationInvariants.CALIBRATION_STATE_KEY.containsMatchIn(it) }
        assertEquals(listOf("boundCalibrationState"), offending)
    }

    @Test
    @DisplayName("§8.1 INV-02: referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg")
    fun referenceSeparationOrdering() {
        assertTrue(profile.referenceSeparationMarginDeg <= profile.smallDeclinationAmbiguityMaxDeg) {
            "A margin above the ambiguity allowance creates a declination dead band that always " +
                "resolves UNVERIFIED with no visible cause (§11)."
        }
    }

    @Test
    @DisplayName("§8.1 INV-03: grade thresholds are strictly ordered so the grade function is total")
    fun gradeThresholdOrdering() {
        assertTrue(profile.professionalBound95MaxDeg < profile.highBound95MaxDeg)
        assertTrue(profile.highBound95MaxDeg < profile.usableBound95MaxDeg)
        assertTrue(profile.usableBound95MaxDeg < profile.lowConfidenceBound95MaxDeg)
    }

    @Test
    @DisplayName("§8.1 INV-04: professionalBound95MaxDeg < flatFreehandPlacementBound95Deg")
    fun freehandCannotReachProfessional() {
        assertTrue(profile.professionalBound95MaxDeg < profile.flatFreehandPlacementBound95Deg) {
            "§20: an implementation reaching Professional freehand has dropped or falsified the " +
                "placement term - a certification failure, not a feature."
        }
    }

    @Test
    @DisplayName("§8.1 INV-05: declination envelope gates are ordered")
    fun declinationEnvelopeOrdering() {
        assertTrue(
            profile.declinationEnvelopeProfessionalMaxDeg <= profile.declinationEnvelopeUsableMaxDeg
        )
    }

    @Test
    @DisplayName("§8.1 INV-06: suspect < disturbed for magnitude, inclination, stationary-MAD, pipeline")
    fun suspectBelowDisturbed() {
        assertTrue(
            profile.magneticMagnitudeResidualSuspectFraction <
                profile.magneticMagnitudeResidualDisturbedFraction
        ) { "A suspect threshold above disturbed makes SUSPECT unreachable." }
        assertTrue(profile.inclinationResidualSuspectDeg < profile.inclinationResidualDisturbedDeg)
        assertTrue(
            profile.stationaryFieldMadSuspectMicroTesla < profile.stationaryFieldMadDisturbedMicroTesla
        )
        assertTrue(profile.pipelineDisagreementSuspectDeg < profile.pipelineDisagreementDisturbedDeg)
    }

    @Test
    @DisplayName("§8.1 INV-07: the periodic support-sample gate is achievable at half the requested rate")
    fun periodicSupportSamplesAchievable() {
        val achievable = profile.stableWindowMinMs * (profile.periodicOrientationRequestedHz / 2.0) / 1000.0
        assertTrue(achievable >= profile.minPeriodicSupportSamples) {
            "The candidate gate tolerates a 50% callback shortfall: achievable=$achievable, " +
                "required=${profile.minPeriodicSupportSamples}. This invariant does not apply to " +
                "event-driven CLHeading, which has its own in-window anchor count (§12, R52)."
        }
    }

    @Test
    @DisplayName("§8.1 INV-08: orientationMaxAgeMs < orientationInvalidAfterMs")
    fun orientationAgeOrdering() {
        assertTrue(profile.orientationMaxAgeMs < profile.orientationInvalidAfterMs) {
            "Drop and invalidate are different thresholds."
        }
    }

    @Test
    @DisplayName("§8.1 INV-09: location freshness tiers are ordered")
    fun locationFreshnessOrdering() {
        assertTrue(profile.freshLocationAtStartMaxAgeMs <= profile.locationAtLockMaxAgeMs)
        assertTrue(profile.locationAtLockMaxAgeMs <= profile.usableLocationMaxAgeMs)
    }

    @Test
    @DisplayName("§8.1 INV-10: space-weather advisory/suppression/refusal tiers are ordered")
    fun spaceWeatherOrdering() {
        assertTrue(profile.spaceWeatherAdvisoryKpMin <= profile.spaceWeatherProfessionalSuppressKpMin)
        assertTrue(profile.spaceWeatherProfessionalSuppressKpMin < profile.spaceWeatherRejectKpMin)
    }

    @Test
    @DisplayName("§8.1 INV-11: the event-driven anchor minimum is independent of the periodic rate")
    fun eventDrivenAnchorMinimumIsIndependentOfTheRate() {
        // R52: requiring iOS flat to deliver 50 CLHeading events in 2 s, or applying the
        // periodic freshness rule to a stationary CLHeading value, rejects a perfectly good
        // measurement. At least one anchor must fall in the window, and the anchor count must
        // stay strictly below what the periodic stream would produce.
        val rateDerived = profile.stableWindowMinMs * profile.periodicOrientationRequestedHz / 1000.0
        assertTrue(profile.clHeadingMinSamplesPerStableWindow >= 1)
        assertTrue(profile.clHeadingMinSamplesPerStableWindow < rateDerived) {
            "clHeadingMinSamplesPerStableWindow=${profile.clHeadingMinSamplesPerStableWindow} " +
                "must stay below the rate-derived periodic count $rateDerived, or the " +
                "event-driven path inherits the periodic contract by arithmetic."
        }
    }

    @Test
    @DisplayName("§8.1: the invariant checker reports a violation when one is introduced")
    fun invariantCheckerIsNotVacuous() {
        // Mutating a copy proves the checker discriminates. The shipped file is untouched;
        // §37 rule 12 forbids editing an artifact to make a test pass, and this test would
        // still fail if the checker were a no-op.
        val broken = profile.copy(
            referenceSeparationMarginDeg = profile.smallDeclinationAmbiguityMaxDeg + 1.0,
            orientationMaxAgeMs = profile.orientationInvalidAfterMs,
            // The R52 defect, expressed as config: an anchor count derived from the periodic
            // rate rather than decided independently.
            clHeadingMinSamplesPerStableWindow =
                (profile.stableWindowMinMs * profile.periodicOrientationRequestedHz / 1000.0).toInt(),
        )
        val violations = ConfigurationInvariants.check(broken, rawTree)
        val ids = violations.map { it.invariantId }
        assertTrue("INV-02-REFERENCE-SEPARATION-ORDERING" in ids) { "got $ids" }
        assertTrue("INV-08-ORIENTATION-AGE-ORDERING" in ids) { "got $ids" }
        assertTrue("INV-11-EVENT-DRIVEN-ANCHOR-MINIMUM" in ids) { "got $ids" }
    }
}
