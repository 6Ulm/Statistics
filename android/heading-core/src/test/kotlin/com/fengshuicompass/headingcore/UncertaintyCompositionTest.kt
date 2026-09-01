package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.grade.QualityGrade
import com.fengshuicompass.headingcore.grade.qualityGradeForReportedBound
import com.fengshuicompass.headingcore.model.GradeLimitingFactor
import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.RejectionReason
import com.fengshuicompass.headingcore.uncertainty.InterferenceRejectionException
import com.fengshuicompass.headingcore.uncertainty.UncertaintyComposition
import com.fengshuicompass.headingcore.uncertainty.UncertaintyTerms
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotSame
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/** SPEC.md §19 — uncertainty composition, both bounds, and `gradeLimitedBy`. */
class UncertaintyCompositionTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.uncertaintyCompositionFixture.readText()
        ).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private fun JsonObject.optionalDouble(key: String): Double? {
        val element = this[key] ?: return null
        return if (element is JsonNull) null else element.jsonPrimitive.double
    }

    private fun terms(case: JsonObject) = UncertaintyTerms(
        providerReportedBoundTermDeg = case.optionalDouble("providerReportedBoundTermDeg"),
        sampleBound95Deg = case["sampleBound95Deg"]!!.jsonPrimitive.double,
        deviceFloor95Deg = case["deviceFloor95Deg"]!!.jsonPrimitive.double,
        placementBound95Deg = case["placementBound95Deg"]!!.jsonPrimitive.double,
        declinationModelBound95Deg = case.optionalDouble("declinationModelBound95Deg"),
        locationTimeSensitivityBound95Deg =
            case.optionalDouble("locationTimeSensitivityBound95Deg") ?: 0.0,
        referenceAmbiguityBound95Deg = case.optionalDouble("referenceAmbiguityBound95Deg") ?: 0.0,
        interferenceBound95Deg = case.optionalDouble("interferenceBound95Deg") ?: 0.0,
        deviationCorrectionResidualBound95Deg =
            case.optionalDouble("deviationCorrectionResidualBound95Deg") ?: 0.0,
    )

    @Test
    @DisplayName("§19: composition matches the frozen fixture")
    fun compositionMatchesTheFrozenFixture() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val composed = UncertaintyComposition.composeBounds(terms(case))
            assertEquals(
                case["expectedBaseHeadingBound95Deg"]!!.jsonPrimitive.double,
                composed.baseHeadingBound95Deg,
                1e-12,
                id,
            )
            assertEquals(
                case["expectedInstrumentBound95Deg"]!!.jsonPrimitive.double,
                composed.instrumentBound95Deg,
                1e-12,
                id,
            )
            assertEquals(
                case["expectedReportedBound95Deg"]!!.jsonPrimitive.double,
                composed.reportedBound95Deg,
                1e-12,
                id,
            )
            assertEquals(
                case["expectedGradeLimitedBy"]!!.jsonPrimitive.content,
                composed.gradeLimitedBy.wire,
                id,
            )
            // The lock/degraded distinction §18.5 draws on the **total** bound.
            val locked = composed.reportedBound95Deg <= profile.usableBound95MaxDeg
            assertEquals(
                case["expectedMeasurementState"]!!.jsonPrimitive.content == "PRECISION_LOCKED",
                locked,
                id,
            )
        }
    }

    @Test
    @DisplayName("§19/failure mode 28: an absent provider term is not zero evidence")
    fun anAbsentProviderTermIsNotZeroEvidence() {
        // On the wall paths, which expose no documented degree error at all, a 0.0 would
        // silently claim evidence that does not exist (R61).
        val absent = UncertaintyTerms(null, 0.4, 4.0, placementBound95Deg = 3.0)
        assertEquals(4.0, UncertaintyComposition.composeBounds(absent).instrumentBound95Deg)
        assertSame(
            GradeLimitingFactor.DEVICE_FLOOR,
            UncertaintyComposition.composeBounds(absent).gradeLimitedBy,
        )
    }

    @Test
    @DisplayName("R63: adding a provider term never lowers the base")
    fun addingAProviderTermNeverLowersTheBase() {
        // Property-checked across the range rather than at one point, because "it happened not
        // to lower it in my example" is not the claim §35 requires.
        val floor = profile.unknownDeviceFloor95Deg
        val baseline = UncertaintyComposition.composeBounds(
            UncertaintyTerms(null, 0.5, floor, placementBound95Deg = 3.0)
        )
        var previous = baseline.baseHeadingBound95Deg
        for (tenths in 0 until 200 step 5) {
            val term = tenths / 10.0
            val composed = UncertaintyComposition.composeBounds(
                UncertaintyTerms(term, 0.5, floor, placementBound95Deg = 3.0)
            )
            assertTrue(composed.baseHeadingBound95Deg >= baseline.baseHeadingBound95Deg)
            assertTrue(composed.baseHeadingBound95Deg >= previous)
            previous = composed.baseHeadingBound95Deg
        }
    }

    @Test
    @DisplayName("R63/§8.1.1: only certification can lower the floor into lock range")
    fun onlyCertificationCanLowerTheFloor() {
        val uncertified = UncertaintyComposition.composeBounds(
            UncertaintyTerms(
                0.1,
                0.1,
                profile.unknownDeviceFloor95Deg,
                placementBound95Deg = profile.flatFreehandPlacementBound95Deg,
            )
        )
        assertTrue(uncertified.reportedBound95Deg > profile.usableBound95MaxDeg)
        val certified = UncertaintyComposition.composeBounds(
            UncertaintyTerms(0.1, 0.1, 1.2, placementBound95Deg = 0.5)
        )
        assertTrue(certified.reportedBound95Deg <= profile.usableBound95MaxDeg)
    }

    @Test
    @DisplayName("§19: the base terms take a max and the rest add")
    fun baseTermsTakeAMaxAndTheRestAdd() {
        val composed = UncertaintyComposition.composeBounds(
            UncertaintyTerms(
                providerReportedBoundTermDeg = 1.0,
                sampleBound95Deg = 2.0,
                deviceFloor95Deg = 1.5,
                placementBound95Deg = 0.5,
                declinationModelBound95Deg = 0.4,
                locationTimeSensitivityBound95Deg = 0.05,
                referenceAmbiguityBound95Deg = 1.5,
                interferenceBound95Deg = 0.0,
            )
        )
        assertEquals(2.0, composed.baseHeadingBound95Deg) // max(1.0, 2.0, 1.5)
        assertEquals(2.0 + 0.4 + 0.05 + 1.5, composed.instrumentBound95Deg, 1e-12)
        assertEquals(composed.instrumentBound95Deg + 0.5, composed.reportedBound95Deg, 1e-12)
    }

    @Test
    @DisplayName("§19: both bounds are capped at 180")
    fun bothBoundsAreCappedAt180() {
        val composed = UncertaintyComposition.composeBounds(
            UncertaintyTerms(
                providerReportedBoundTermDeg = 170.0,
                sampleBound95Deg = 1.0,
                deviceFloor95Deg = 1.0,
                placementBound95Deg = 50.0,
                referenceAmbiguityBound95Deg = 100.0,
            )
        )
        assertEquals(UncertaintyComposition.MAX_BOUND_DEG, composed.instrumentBound95Deg)
        assertEquals(UncertaintyComposition.MAX_BOUND_DEG, composed.reportedBound95Deg)
    }

    @Test
    @DisplayName("§18.5/§20: the placement term is never zero")
    fun placementIsNeverZero() {
        val failure = assertThrows<IllegalArgumentException> {
            UncertaintyTerms(1.0, 0.5, 4.0, placementBound95Deg = 0.0)
        }
        assertTrue(failure.message!!.contains("never"))
    }

    @Test
    @DisplayName("§19: the reported bound is always at least the instrument bound")
    fun reportedBoundIsAlwaysAtLeastTheInstrumentBound() {
        listOf(0.1, 0.5, 3.0, 5.0).forEach { placement ->
            val composed = UncertaintyComposition.composeBounds(
                UncertaintyTerms(null, 0.5, 1.0, placementBound95Deg = placement)
            )
            assertTrue(composed.reportedBound95Deg >= composed.instrumentBound95Deg)
        }
    }

    // ---------------------------------------------------------------------------------
    // §19 interference term
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§19: the interference term by magnetic state")
    fun interferenceTermByMagneticState() {
        val suspectBound = profile.suspectInterferenceBound95Deg
        assertEquals(
            0.0,
            UncertaintyComposition.interferenceBound95Deg(MagneticState.CLEAN, suspectBound),
        )
        assertEquals(
            suspectBound,
            UncertaintyComposition.interferenceBound95Deg(MagneticState.SUSPECT, suspectBound),
        )
        mapOf(
            MagneticState.DISTURBED to RejectionReason.MAGNETIC_FIELD_DISTURBED,
            MagneticState.INVALID to RejectionReason.MAGNETIC_CALIBRATION_INVALID,
            MagneticState.UNKNOWN to RejectionReason.MAGNETIC_FIELD_UNKNOWN,
        ).forEach { (state, reason) ->
            val failure = assertThrows<InterferenceRejectionException> {
                UncertaintyComposition.interferenceBound95Deg(state, suspectBound)
            }
            assertSame(reason, failure.reason)
        }
    }

    @Test
    @DisplayName("§8.1.1 row 3: SUSPECT prevents a freehand lock outright")
    fun suspectPreventsAFreehandLockOutright() {
        // Not merely capping the grade: the 3.0 deg term alone exceeds the 2.0 deg
        // flat-freehand instrument budget, so no sensor quality can recover a lock.
        val budget = profile.usableBound95MaxDeg - profile.flatFreehandPlacementBound95Deg
        assertTrue(profile.suspectInterferenceBound95Deg > budget)
        val bestPossible = UncertaintyComposition.composeBounds(
            UncertaintyTerms(
                providerReportedBoundTermDeg = 0.0,
                sampleBound95Deg = 0.0,
                deviceFloor95Deg = 0.0, // a hypothetically perfect certified device
                placementBound95Deg = profile.flatFreehandPlacementBound95Deg,
                interferenceBound95Deg = profile.suspectInterferenceBound95Deg,
            )
        )
        assertTrue(bestPossible.reportedBound95Deg > profile.usableBound95MaxDeg)
    }

    // ---------------------------------------------------------------------------------
    // §19 gradeLimitedBy
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§19: policy ceilings take the fixed precedence")
    fun policyCeilingsTakeTheFixedPrecedence() {
        val base = UncertaintyTerms(null, 0.5, 4.0, placementBound95Deg = 3.0)
        assertSame(
            GradeLimitingFactor.CERTIFICATION_CEILING,
            UncertaintyComposition.composeBounds(
                base,
                setOf(
                    GradeLimitingFactor.CHARGING_STATE,
                    GradeLimitingFactor.SPACE_WEATHER,
                    GradeLimitingFactor.CERTIFICATION_CEILING,
                ),
            ).gradeLimitedBy,
        )
        assertSame(
            GradeLimitingFactor.SPACE_WEATHER,
            UncertaintyComposition.composeBounds(
                base,
                setOf(GradeLimitingFactor.CHARGING_STATE, GradeLimitingFactor.SPACE_WEATHER),
            ).gradeLimitedBy,
        )
        assertSame(
            GradeLimitingFactor.CHARGING_STATE,
            UncertaintyComposition.composeBounds(
                base,
                setOf(GradeLimitingFactor.CHARGING_STATE),
            ).gradeLimitedBy,
        )
    }

    @Test
    @DisplayName("R57: CHARGING_STATE is in the enum")
    fun chargingStateIsInTheEnum() {
        assertEquals("CHARGING_STATE", GradeLimitingFactor.CHARGING_STATE.wire)
    }

    @Test
    @DisplayName("§21.5/§20: placement is named when it dominates")
    fun placementIsNamedWhenItDominates() {
        assertSame(
            GradeLimitingFactor.PLACEMENT_UNCERTAINTY,
            UncertaintyComposition.composeBounds(
                UncertaintyTerms(null, 0.2, 0.5, placementBound95Deg = 3.0)
            ).gradeLimitedBy,
        )
    }

    @Test
    @DisplayName("§19: exact ties resolve by stable enum order")
    fun exactTiesResolveByStableEnumOrder() {
        // Two runtimes must not disagree about which of two equal terms is named.
        val composed = UncertaintyComposition.composeBounds(
            UncertaintyTerms(null, 2.0, 2.0, placementBound95Deg = 2.0)
        )
        assertEquals(
            minOf(
                GradeLimitingFactor.PLACEMENT_UNCERTAINTY.ordinal,
                GradeLimitingFactor.SAMPLE_DISPERSION.ordinal,
                GradeLimitingFactor.DEVICE_FLOOR.ordinal,
            ),
            composed.gradeLimitedBy.ordinal,
        )
    }

    @Test
    @DisplayName("§20/failure mode 30: grades come from the reported bound")
    fun gradesComeFromTheReportedBound() {
        val composed = UncertaintyComposition.composeBounds(
            UncertaintyTerms(null, 0.2, 1.5, placementBound95Deg = 3.0)
        )
        assertEquals(1.5, composed.instrumentBound95Deg)
        assertEquals(4.5, composed.reportedBound95Deg)
        val onReported = qualityGradeForReportedBound(composed.reportedBound95Deg, profile)
        val onInstrument = qualityGradeForReportedBound(composed.instrumentBound95Deg, profile)
        assertSame(QualityGrade.USABLE, onReported)
        assertSame(QualityGrade.PROFESSIONAL, onInstrument)
        assertNotSame(onReported, onInstrument)
    }
}
