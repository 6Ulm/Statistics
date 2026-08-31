package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.grade.CertificationState
import com.fengshuicompass.headingcore.grade.GradeReachability
import com.fengshuicompass.headingcore.grade.MagneticState
import com.fengshuicompass.headingcore.grade.PlacementMethod
import com.fengshuicompass.headingcore.grade.QualityGrade
import com.fengshuicompass.headingcore.grade.qualityGradeForReportedBound
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/**
 * SPEC.md R62 and the §35 checklist line "Every executable/display example passes
 * production bound composition and reachability checks; no uncertified flat-freehand
 * fixture locks under the shipped 4° + 3° minimum."
 *
 * The §22.1 example is deliberately a good-looking measurement — fresh confident provider,
 * clean field, verified reference, level device — that is still not a Precision Lock. That
 * makes it an executable fixture, not decoration: if someone raises a grade by dropping the
 * placement term or flooring a bound, this test fails.
 */
class ExampleTelemetryReachabilityTest {

    private val profile: PrecisionProfile = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)
    private val json = Json { ignoreUnknownKeys = true }

    private val event: JsonObject =
        json.parseToJsonElement(SharedArtifacts.exampleEngineOutputEventFile.readText()).jsonObject

    private val payload: JsonObject get() = event.getValue("payload").jsonObject

    private fun num(key: String): Double = payload.getValue(key).jsonPrimitive.double
    private fun str(key: String): String = payload.getValue(key).jsonPrimitive.content

    @Test
    @DisplayName("§19: the example's bounds compose exactly - instrument + placement = reported")
    fun boundsCompose() {
        val instrument = num("instrumentBound95Deg")
        val placement = num("placementBound95Deg")
        val reported = num("reportedBound95Deg")
        assertEquals(instrument + placement, reported, 1e-9)
        assertEquals(minOf(180.0, instrument + placement), reported, 1e-9)
    }

    @Test
    @DisplayName("R62: the example matches the shipped 4.0 + 3.0 uncertified flat-freehand minimum")
    fun exampleMatchesShippedMinimum() {
        assertEquals("FLAT_TOP_EDGE", str("measurementMode"))
        assertEquals("FREEHAND", str("placementMethod"))
        assertEquals("CLEAN", str("magneticState"))
        assertEquals("CANDIDATE", str("boundCalibrationState"))

        val computed = GradeReachability.compute(
            PlacementMethod.FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile
        )
        assertEquals(computed.minimumReportedBound95Deg!!, num("reportedBound95Deg"), 1e-9)
        assertEquals(profile.unknownDeviceFloor95Deg, num("instrumentBound95Deg"), 1e-9)
        assertEquals(profile.flatFreehandPlacementBound95Deg, num("placementBound95Deg"), 1e-9)
    }

    @Test
    @DisplayName("R62: the example degrades - it is not a Precision Lock and shows no certified grade")
    fun exampleDegradesAndIsNotALock() {
        assertEquals("DEGRADED", str("measurementState"))
        assertEquals("SHOW_DEGRADED_RESULT", str("trustAction"))
        assertEquals("LOW_CONFIDENCE", str("provisionalQualityGrade"))
        assertTrue(payload.getValue("displayQualityGrade").jsonPrimitive.content == "null") {
            "§19.1: a CANDIDATE consumer result MUST NOT show a standalone certified grade."
        }
        assertFalse(num("reportedBound95Deg") <= profile.usableBound95MaxDeg) {
            "§18.5: PRECISION_LOCKED requires reportedBound95Deg <= usableBound95MaxDeg."
        }
        assertEquals(
            QualityGrade.LOW_CONFIDENCE,
            qualityGradeForReportedBound(num("reportedBound95Deg"), profile)
        )
    }

    @Test
    @DisplayName("§19.1: the CANDIDATE / coverage-evidence invariant holds on the example")
    fun candidateCoverageInvariant() {
        val calibration = str("boundCalibrationState")
        val evidence = str("uncertaintyCoverageEvidenceState")
        // CALIBRATED <=> EMPIRICALLY_CALIBRATED; CANDIDATE => {TARGET_ONLY, UNDEFINED}
        if (calibration == "CALIBRATED") {
            assertEquals("EMPIRICALLY_CALIBRATED", evidence)
        } else {
            assertEquals("CANDIDATE", calibration)
            assertTrue(evidence in setOf("TARGET_ONLY", "UNDEFINED")) { "got $evidence" }
        }
    }

    @Test
    @DisplayName("§21.3/§21.4: a 7.0 deg bound exceeds half a 15 deg sector, so it must straddle")
    fun wideBoundMustStraddle() {
        assertTrue(num("reportedBound95Deg") > 7.5 / 2) // sanity on the sector geometry below
        assertTrue(payload.getValue("boundaryStraddled").jsonPrimitive.boolean) {
            "§21.3: reportedBound95Deg > 7.5° guarantees a two-sector straddle regardless of the " +
                "point estimate; this example is at 7.0° and straddles on its point estimate."
        }
        val sectors = payload.getValue("possibleFengShuiSectors")
        assertTrue(sectors.toString().contains("wu") && sectors.toString().contains("ding"))
    }

    @Test
    @DisplayName("§19.3: production deviation-correction state is NONE and the correction is exactly 0")
    fun deviationCorrectionIsNone() {
        assertEquals("NONE", str("deviationCorrectionState"))
        assertEquals(0.0, num("deviationCorrectionDeg"), 0.0)
        assertEquals(num("uncorrectedTrueHeadingDeg"), num("trueHeadingDeg"), 0.0)
        assertEquals("NONE", str("deviationCorrectionProfileHash"))
    }

    @Test
    @DisplayName("§22: the envelope's configHash matches the shipped configuration file")
    fun envelopeConfigHashMatchesShippedConfig() {
        val expected = "sha256:" + java.security.MessageDigest.getInstance("SHA-256")
            .digest(SharedArtifacts.precisionProfileFile.readBytes())
            .joinToString("") { "%02x".format(it) }
        assertEquals(expected, event.getValue("configHash").jsonPrimitive.content)
        assertEquals(profile.configVersion, event.getValue("configVersion").jsonPrimitive.content)
    }
}
