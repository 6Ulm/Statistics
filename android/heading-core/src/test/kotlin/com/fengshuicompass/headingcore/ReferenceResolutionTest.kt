package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.ReferenceAxis
import com.fengshuicompass.headingcore.model.ReferenceMagneticPrecheckState
import com.fengshuicompass.headingcore.model.ReferenceResolutionMethod
import com.fengshuicompass.headingcore.model.ResolvedReference
import com.fengshuicompass.headingcore.reference.GoogleReferenceHypotheses
import com.fengshuicompass.headingcore.reference.ReferenceResolution
import com.fengshuicompass.headingcore.reference.ReferenceResolutionResult
import com.fengshuicompass.headingcore.reference.ReferenceResolutionThresholds
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs

/** SPEC.md §11 — north-reference resolution, and the double-correction signature §30.5 hunts. */
class ReferenceResolutionTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.referenceResolutionFixture.readText()
        ).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private val thresholds: ReferenceResolutionThresholds
        get() = ReferenceResolutionThresholds.fromProfile(profile)

    private fun hypotheses(case: JsonObject, eligible: Boolean = true) = GoogleReferenceHypotheses(
        measurementMode = MeasurementMode.valueOf(case["measurementMode"]!!.jsonPrimitive.content),
        gAxisDeg = case["gAxisDeg"]!!.jsonPrimitive.double,
        mAxisDeg = case["mAxisDeg"]!!.jsonPrimitive.double,
        declinationDeg = case["declinationDeg"]!!.jsonPrimitive.double,
        precheckState = ReferenceMagneticPrecheckState.valueOf(
            case["precheckState"]!!.jsonPrimitive.content
        ),
        geomagneticModelId = GeomagneticModelId.WMM2025,
        sourceWindowStartMonotonicNs = 1_000L,
        sourceWindowEndMonotonicNs = 3_000L,
        evidenceIsEligible = eligible,
    )

    private fun caseOf(vararg pairs: Pair<String, String>): JsonObject =
        json.parseToJsonElement(
            pairs.joinToString(",", "{", "}") { (key, value) -> "\"$key\":$value" }
        ).jsonObject

    @Test
    @DisplayName("§11: resolution matches the frozen fixture")
    fun resolutionMatchesTheFrozenFixture() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val result = ReferenceResolution.resolveGoogleReference(hypotheses(case), thresholds)
            assertEquals(
                case["expectedResolvedReference"]!!.jsonPrimitive.content,
                result.resolvedReference.wire,
                id,
            )
            assertEquals(
                case["expectedCorrectionDeg"]!!.jsonPrimitive.double,
                result.correctionDeg,
                1e-12,
                id,
            )
            assertEquals(
                case["expectedReferenceAmbiguityBound95Deg"]!!.jsonPrimitive.double,
                result.referenceAmbiguityBound95Deg,
                1e-12,
                id,
            )
            val expectedHeading = case["expectedCanonicalTrueHeadingDeg"]!!
            if (expectedHeading is JsonNull) {
                assertNull(result.canonicalTrueHeadingDeg, id)
            } else {
                assertEquals(
                    expectedHeading.jsonPrimitive.double,
                    result.canonicalTrueHeadingDeg!!,
                    1e-9,
                    id,
                )
            }
            if (case["expectedResidualsAbsent"]?.jsonPrimitive?.boolean == true) {
                assertNull(result.referenceHypothesisResidualTrueDeg, id)
                assertNull(result.referenceHypothesisResidualMagneticDeg, id)
            }
        }
    }

    @Test
    @DisplayName("§11/failure mode 21: correctionDeg is exactly 0.0 or +declinationDeg")
    fun correctionIsExactlyZeroOrPlusDeclination() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val result = ReferenceResolution.resolveGoogleReference(hypotheses(case), thresholds)
            assertTrue(
                result.correctionDeg == 0.0 || result.correctionDeg == result.declinationDeg,
                case["id"]!!.jsonPrimitive.content,
            )
        }
    }

    @Test
    @DisplayName("§30.5: the 2 x declination double-correction signature is detectable")
    fun theDoubleCorrectionSignatureIsDetectable() {
        val declination = 8.29
        val case = caseOf(
            "measurementMode" to "\"FLAT_TOP_EDGE\"",
            "gAxisDeg" to "180.71",
            "mAxisDeg" to "180.71",
            "declinationDeg" to "$declination",
            "precheckState" to "\"CLEAN_FOR_REFERENCE\"",
        )
        val result = ReferenceResolution.resolveGoogleReference(hypotheses(case), thresholds)
        assertEquals(ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC, result.resolvedReference)
        val correct = result.canonicalTrueHeadingDeg!!
        val doublyCorrected = CircularMath.normalize360(correct + declination)
        // The signature §30.5 looks for is 2*d from the magnetic bearing, not d.
        assertEquals(
            2.0 * declination,
            CircularMath.absoluteCircularDifferenceDeg(doublyCorrected, 180.71),
            1e-9,
        )
    }

    @Test
    @DisplayName("R59: ineligible evidence refuses to resolve")
    fun ineligibleEvidenceRefusesToResolve() {
        val case = caseOf(
            "measurementMode" to "\"FLAT_TOP_EDGE\"",
            "gAxisDeg" to "189.0",
            "mAxisDeg" to "180.71",
            "declinationDeg" to "8.29",
            "precheckState" to "\"CLEAN_FOR_REFERENCE\"",
        )
        val result = ReferenceResolution.resolveGoogleReference(
            hypotheses(case, eligible = false),
            thresholds,
        )
        assertEquals(ResolvedReference.UNVERIFIED, result.resolvedReference)
        assertEquals(ReferenceResolutionMethod.NOT_RESOLVED, result.referenceResolutionMethod)
        assertNull(result.canonicalTrueHeadingDeg)
    }

    @Test
    @DisplayName("§8.1/§11: there is no declination dead band")
    fun thereIsNoDeclinationDeadBand() {
        // Since rMag - rTrue <= abs(d), a separation margin above the ambiguity allowance
        // would create a band that always resolves UNVERIFIED with no visible cause.
        assertTrue(
            profile.referenceSeparationMarginDeg <= profile.smallDeclinationAmbiguityMaxDeg
        )
        val step = profile.smallDeclinationAmbiguityMaxDeg / 20.0
        for (index in 0..40) {
            val declination = index * step
            val case = caseOf(
                "measurementMode" to "\"FLAT_TOP_EDGE\"",
                "gAxisDeg" to "100.0",
                "mAxisDeg" to "${CircularMath.normalize360(100.0 - declination)}",
                "declinationDeg" to "$declination",
                "precheckState" to "\"CLEAN_FOR_REFERENCE\"",
            )
            val result = ReferenceResolution.resolveGoogleReference(hypotheses(case), thresholds)
            assertNotEquals(
                ResolvedReference.UNVERIFIED,
                result.resolvedReference,
                "declination=$declination",
            )
            assertEquals(100.0, result.canonicalTrueHeadingDeg!!, 1e-9)
        }
    }

    @Test
    @DisplayName("§11: the ambiguity term never exceeds the declination")
    fun theAmbiguityTermNeverExceedsTheDeclination() {
        listOf(-2.0, -1.5, 0.0, 0.5, 1.5, 2.0).forEach { declination ->
            val case = caseOf(
                "measurementMode" to "\"WALL_FLUSH_BACK\"",
                "gAxisDeg" to "100.0",
                "mAxisDeg" to "100.0",
                "declinationDeg" to "$declination",
                "precheckState" to "\"CLEAN_FOR_REFERENCE\"",
            )
            val result = ReferenceResolution.resolveGoogleReference(hypotheses(case), thresholds)
            assertTrue(result.referenceAmbiguityBound95Deg <= abs(declination) + 1e-12)
        }
    }

    @Test
    @DisplayName("§11: the result is bound to its mode and axis")
    fun theResultIsBoundToItsModeAndAxis() {
        fun resolve(mode: String) = ReferenceResolution.resolveGoogleReference(
            hypotheses(
                caseOf(
                    "measurementMode" to "\"$mode\"",
                    "gAxisDeg" to "189.0",
                    "mAxisDeg" to "180.71",
                    "declinationDeg" to "8.29",
                    "precheckState" to "\"CLEAN_FOR_REFERENCE\"",
                )
            ),
            thresholds,
        )
        assertEquals(
            ReferenceAxis.PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION,
            resolve("FLAT_TOP_EDGE").referenceAxis,
        )
        assertEquals(
            ReferenceAxis.OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION,
            resolve("WALL_FLUSH_BACK").referenceAxis,
        )
    }

    // ---------------------------------------------------------------------------------
    // R51 — the explicit non-Google contracts
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("R51: iOS flat uses the explicit provider contract")
    fun appleFlatUsesTheExplicitProviderContract() {
        val result = ReferenceResolution.appleProviderContractReferenceResolution(
            MeasurementMode.FLAT_TOP_EDGE, 123.4, 8.29, GeomagneticModelId.WMM2025, 0L, 1L,
        )
        assertEquals(
            ReferenceResolutionMethod.PROVIDER_CONTRACT_EXPLICIT,
            result.referenceResolutionMethod,
        )
        assertEquals(ResolvedReference.TRUE_VERIFIED, result.resolvedReference)
        assertEquals(0.0, result.correctionDeg)
        assertEquals(0.0, result.referenceAmbiguityBound95Deg)
        assertEquals(123.4, result.canonicalTrueHeadingDeg!!, 1e-9)
    }

    @Test
    @DisplayName("§12: iOS wall requires the frame to be actually active")
    fun appleWallRequiresTheFrameToBeActive() {
        val active = ReferenceResolution.appleAttitudeFrameReferenceResolution(
            MeasurementMode.WALL_FLUSH_BACK, 200.0, 8.29, GeomagneticModelId.WMM2025, 0L, 1L, true,
        )
        assertEquals(
            ReferenceResolutionMethod.ATTITUDE_FRAME_EXPLICIT,
            active.referenceResolutionMethod,
        )
        val inactive = ReferenceResolution.appleAttitudeFrameReferenceResolution(
            MeasurementMode.WALL_FLUSH_BACK, 200.0, 8.29, GeomagneticModelId.WMM2025, 0L, 1L, false,
        )
        assertEquals(ResolvedReference.UNVERIFIED, inactive.resolvedReference)
        assertNull(inactive.canonicalTrueHeadingDeg)
    }

    @Test
    @DisplayName("§30.4: AND-RV applies declination once and never claims TRUE_VERIFIED")
    fun andRvAppliesDeclinationOnce() {
        val result = ReferenceResolution.andRvReferenceResolution(
            MeasurementMode.WALL_FLUSH_BACK, 355.0, 8.29, GeomagneticModelId.WMM2025, 0L, 1L,
        )
        assertEquals(ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC, result.resolvedReference)
        assertEquals(
            ReferenceResolutionMethod.APP_APPLIED_DECLINATION,
            result.referenceResolutionMethod,
        )
        assertEquals(8.29, result.correctionDeg)
        assertEquals(0.0, result.referenceAmbiguityBound95Deg)
        assertEquals(3.29, result.canonicalTrueHeadingDeg!!, 1e-9)
    }

    @Test
    @DisplayName("failure mode 21: a correction that is neither 0 nor +d cannot be constructed")
    fun anInvalidCorrectionIsRejectedByConstruction() {
        assertThrows<IllegalArgumentException> {
            ReferenceResolutionResult(
                measurementMode = MeasurementMode.FLAT_TOP_EDGE,
                referenceAxis = ReferenceAxis.PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION,
                resolvedReference = ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
                referenceResolutionMethod = ReferenceResolutionMethod.APP_APPLIED_DECLINATION,
                declinationDeg = 8.29,
                correctionDeg = 16.58, // the 2*d signature
                referenceAmbiguityBound95Deg = 0.0,
                geomagneticModelId = GeomagneticModelId.WMM2025,
                sourceWindowStartMonotonicNs = 0L,
                sourceWindowEndMonotonicNs = 1L,
            )
        }
    }
}
