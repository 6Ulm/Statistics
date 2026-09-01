package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.magnetic.MagneticClassification
import com.fengshuicompass.headingcore.magnetic.MagneticFeatures
import com.fengshuicompass.headingcore.magnetic.MagneticThresholds
import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.ReferenceMagneticPrecheckState
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs
import kotlin.math.asin

/** SPEC.md §16 — magnetic interference detection and the §11 reference precheck. */
class MagneticClassificationTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.magneticClassificationFixture.readText()
        ).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private val thresholds: MagneticThresholds get() = MagneticThresholds.fromProfile(profile)

    private fun JsonObject.optionalDouble(key: String): Double? {
        val element = this[key] ?: return null
        return if (element is JsonNull) null else element.jsonPrimitive.double
    }

    private fun JsonObject.flag(key: String): Boolean =
        this[key]?.jsonPrimitive?.boolean ?: false

    private fun features(case: JsonObject) = MagneticFeatures(
        relativeMagnitudeResidual = case.optionalDouble("relativeMagnitudeResidual"),
        inclinationResidualDeg = case.optionalDouble("inclinationResidualDeg"),
        stationaryFieldMadMicroTesla = case.optionalDouble("stationaryFieldMadMicroTesla"),
        pipelineAgreementDeg = case.optionalDouble("pipelineAgreementDeg"),
        anyValueNonFinite = case.flag("anyValueNonFinite"),
        sensorSaturated = case.flag("sensorSaturated"),
        osCalibrationInvalid = case.flag("osCalibrationInvalid"),
    )

    // ---------------------------------------------------------------------------------
    // R60 — the mandatory minus sign
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("R60: the inclination conversion matches the frozen fixture")
    fun inclinationMatchesTheFrozenFixture() {
        fixture["inclinationCases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            assertEquals(
                case["expectedMeasuredInclinationPositiveDownDeg"]!!.jsonPrimitive.double,
                MagneticClassification.measuredInclinationPositiveDownDeg(
                    case["upMicroTesla"]!!.jsonPrimitive.double,
                    case["magnitudeMicroTesla"]!!.jsonPrimitive.double,
                ),
                1e-9,
                case["id"]!!.jsonPrimitive.content,
            )
        }
    }

    @Test
    @DisplayName("R60: a northern-hemisphere field gives a positive-down inclination")
    fun northernHemisphereFieldGivesPositiveDownInclination() {
        // Canonical ENU Bup is positive *upward*; WMM I is positive *downward*. In the
        // northern hemisphere the field points into the ground, so Bup is negative and the
        // positive-down inclination must come out positive.
        val observed = MagneticClassification.measuredInclinationPositiveDownDeg(-43.9, 48.7)
        assertTrue(observed > 0.0)
        val withoutTheMinusSign = Math.toDegrees(asin(-43.9 / 48.7))
        assertTrue(withoutTheMinusSign < 0.0)
        assertEquals(-withoutTheMinusSign, observed, 1e-12)
    }

    @Test
    @DisplayName("R60: a missing minus sign would reject a clean field")
    fun missingMinusSignWouldRejectACleanField() {
        val expectedWmmInclination = 64.4
        val correct = MagneticClassification.measuredInclinationPositiveDownDeg(-43.9, 48.7)
        val correctResidual =
            MagneticClassification.inclinationResidualDeg(correct, expectedWmmInclination)
        assertTrue(abs(correctResidual) < thresholds.inclinationResidualSuspectDeg)

        val signFlipped = Math.toDegrees(asin(-43.9 / 48.7))
        val flippedResidual = signFlipped - expectedWmmInclination
        assertTrue(abs(flippedResidual) >= thresholds.inclinationResidualDisturbedDeg)
    }

    @Test
    @DisplayName("§16: the inclination residual is linear, never circular")
    fun inclinationResidualIsLinear() {
        // A circular difference would rescale a residual near the poles; the linear one keeps
        // the full magnitude visible.
        assertEquals(160.0, MagneticClassification.inclinationResidualDeg(80.0, -80.0))
        assertEquals(-160.0, MagneticClassification.inclinationResidualDeg(-80.0, 80.0))
    }

    @Test
    @DisplayName("§16: the inclination input range is asserted")
    fun inclinationInputRangeIsAsserted() {
        assertThrows<IllegalArgumentException> {
            MagneticClassification.inclinationResidualDeg(120.0, 0.0)
        }
        assertThrows<IllegalArgumentException> {
            MagneticClassification.inclinationResidualDeg(0.0, Double.NaN)
        }
    }

    @Test
    @DisplayName("failure mode 6: the ratio is clamped before asin")
    fun inclinationClampsBeforeAsin() {
        assertEquals(
            90.0,
            MagneticClassification.measuredInclinationPositiveDownDeg(-48.700000000000003, 48.7),
            1e-9,
        )
    }

    @Test
    @DisplayName("§16: a zero or negative magnitude is rejected")
    fun zeroOrNegativeMagnitudeIsRejected() {
        assertThrows<IllegalArgumentException> {
            MagneticClassification.measuredInclinationPositiveDownDeg(0.0, 0.0)
        }
        assertThrows<IllegalArgumentException> {
            MagneticClassification.relativeMagnitudeResidual(48.0, 0.0)
        }
    }

    // ---------------------------------------------------------------------------------
    // §16 classifier
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§16: the classifier and precheck match the frozen fixture")
    fun classifierMatchesTheFrozenFixture() {
        fixture["classifierCases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val observed = features(case)
            assertEquals(
                case["expectedMagneticState"]!!.jsonPrimitive.content,
                MagneticClassification.classifyMagneticState(observed, thresholds).wire,
                id,
            )
            assertEquals(
                case["expectedPrecheckState"]!!.jsonPrimitive.content,
                MagneticClassification.referenceMagneticPrecheckState(observed, thresholds).wire,
                id,
            )
        }
    }

    @Test
    @DisplayName("failure mode 23: magnitude alone cannot declare a field clean")
    fun magnitudeAloneCannotDeclareAFieldClean() {
        // A disturbance that rotates the field vector with normal magnitude is the case
        // producing a *confident wrong bearing* rather than an obviously broken one.
        val rotated = MagneticFeatures(
            relativeMagnitudeResidual = 0.01, // well inside the clean magnitude band
            inclinationResidualDeg = 14.0,
            stationaryFieldMadMicroTesla = 0.4,
            pipelineAgreementDeg = 1.0,
        )
        assertEquals(
            MagneticState.DISTURBED,
            MagneticClassification.classifyMagneticState(rotated, thresholds),
        )
    }

    @Test
    @DisplayName("§16: absent features resolve UNKNOWN, never CLEAN")
    fun absentFeaturesResolveUnknown() {
        // stationaryFieldMadMicroTesla is absent while the device is moving;
        // pipelineAgreementDeg is absent with fewer than two valid active-axis pipelines.
        listOf(
            MagneticFeatures(0.01, 0.5, null, 0.5),
            MagneticFeatures(0.01, 0.5, 0.4, null),
        ).forEach {
            assertEquals(
                MagneticState.UNKNOWN,
                MagneticClassification.classifyMagneticState(it, thresholds),
            )
        }
    }

    @Test
    @DisplayName("§16: a present disturbed feature wins over absent evidence")
    fun disturbedWinsOverAbsentEvidence() {
        assertEquals(
            MagneticState.DISTURBED,
            MagneticClassification.classifyMagneticState(
                MagneticFeatures(0.6, null, null, null),
                thresholds,
            ),
        )
    }

    @Test
    @DisplayName("§16: invalid input precedes every other branch")
    fun invalidInputPrecedesEveryOtherBranch() {
        listOf(
            MagneticFeatures(0.01, 0.5, 0.4, 0.5, sensorSaturated = true),
            MagneticFeatures(0.01, 0.5, 0.4, 0.5, osCalibrationInvalid = true),
            MagneticFeatures(0.01, 0.5, 0.4, 0.5, anyValueNonFinite = true),
        ).forEach {
            assertEquals(
                MagneticState.INVALID,
                MagneticClassification.classifyMagneticState(it, thresholds),
            )
        }
    }

    @Test
    @DisplayName("§16: thresholds are inclusive at the boundary")
    fun thresholdsAreInclusiveAtTheBoundary() {
        assertEquals(
            MagneticState.SUSPECT,
            MagneticClassification.classifyMagneticState(
                MagneticFeatures(profile.magneticMagnitudeResidualSuspectFraction, 0.5, 0.4, 0.5),
                thresholds,
            ),
        )
        assertEquals(
            MagneticState.DISTURBED,
            MagneticClassification.classifyMagneticState(
                MagneticFeatures(profile.magneticMagnitudeResidualDisturbedFraction, 0.5, 0.4, 0.5),
                thresholds,
            ),
        )
    }

    // ---------------------------------------------------------------------------------
    // R59 — the precheck stays acyclic
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("R59: the precheck ignores pipeline agreement entirely")
    fun precheckIgnoresPipelineAgreementEntirely() {
        // Varying pipelineAgreementDeg across its whole range must not move the precheck,
        // which is what keeps the dependency order acyclic.
        val outcomes = listOf(null, 0.0, 0.5, 5.0, 50.0, 180.0).map { pipeline ->
            MagneticClassification.referenceMagneticPrecheckState(
                MagneticFeatures(0.01, 0.5, 0.4, pipeline),
                thresholds,
            )
        }.toSet()
        assertEquals(setOf(ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE), outcomes)
    }

    @Test
    @DisplayName("R59: the precheck is UNKNOWN when its own evidence is absent")
    fun precheckIsUnknownWhenItsOwnEvidenceIsAbsent() {
        assertEquals(
            ReferenceMagneticPrecheckState.UNKNOWN,
            MagneticClassification.referenceMagneticPrecheckState(
                MagneticFeatures(0.01, null, 0.4, 0.5),
                thresholds,
            ),
        )
    }

    @Test
    @DisplayName("§16: the precheck and the final state are recorded separately")
    fun precheckAndFinalStateAreRecordedSeparately() {
        val observed = MagneticFeatures(0.01, 0.5, 0.4, 12.0)
        assertEquals(
            ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE,
            MagneticClassification.referenceMagneticPrecheckState(observed, thresholds),
        )
        assertEquals(
            MagneticState.DISTURBED,
            MagneticClassification.classifyMagneticState(observed, thresholds),
        )
    }
}
