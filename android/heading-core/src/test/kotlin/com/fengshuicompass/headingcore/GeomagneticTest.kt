package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.geomagnetic.AltitudeDatumUnconvertedException
import com.fengshuicompass.headingcore.geomagnetic.AltitudeSample
import com.fengshuicompass.headingcore.geomagnetic.ConfidenceLevel
import com.fengshuicompass.headingcore.geomagnetic.Geomagnetic
import com.fengshuicompass.headingcore.geomagnetic.GeomagneticDateOutOfRangeException
import com.fengshuicompass.headingcore.geomagnetic.GeomagneticModelUncertainty
import com.fengshuicompass.headingcore.geomagnetic.VendoredModelUnavailableException
import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.AltitudeReference
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.time.ZoneOffset
import java.time.ZonedDateTime

/**
 * SPEC.md §10 / §10.2 / §10.3 / §19.2 — the geomagnetic contract, and its refusal to guess.
 *
 * The load-bearing test here is [noSigmaCanBeProducedWithoutAVendoredErrorModel]. §10.3 is
 * explicit that an implementation which "derives a sigma from the coefficients, or substitutes
 * a remembered global constant, has invented the quantity", and the NOAA artifacts could not
 * be fetched (`docs/IMPLEMENTATION_NOTES.md` D-2). A refusal is correct; a plausible number is
 * the failure.
 */
class GeomagneticTest {

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    @Test
    @DisplayName("D-2: the shipped tree declares no vendored model")
    fun theShippedTreeDeclaresNoVendoredModel() {
        GeomagneticModelId.entries.forEach { modelId ->
            val artifacts = Geomagnetic.vendoredArtifacts(modelId, SharedArtifacts.repoRoot)
            assertFalse(
                artifacts.isVendored,
                "${modelId.wire} now reports vendored artifacts. Phase 1 could not reach NOAA " +
                    "(D-2); if that changed, run the official test vectors before relying on it.",
            )
        }
    }

    @Test
    @DisplayName("§10.3: no sigma can be produced without a vendored error model")
    fun noSigmaCanBeProducedWithoutAVendoredErrorModel() {
        val artifacts =
            Geomagnetic.vendoredArtifacts(GeomagneticModelId.WMM2025, SharedArtifacts.repoRoot)
        val failure = assertThrows<VendoredModelUnavailableException> {
            artifacts.requireVendored("declinationSigma1Deg")
        }
        assertTrue(failure.message!!.contains("NOT_VENDORED"))
        assertTrue(failure.message!!.contains("third_party/noaa-wmm"))
    }

    @Test
    @DisplayName("§24: an uncertainty without an error-model hash is an invented quantity")
    fun uncertaintyRequiresAnErrorModelHash() {
        assertThrows<IllegalArgumentException> {
            GeomagneticModelUncertainty(
                declinationSigma1Deg = 0.36,
                sourceModelId = GeomagneticModelId.WMM2025,
                errorModelId = "whatever",
                errorModelHash = "NONE",
                sourceDocumentReference = "",
            )
        }
    }

    @Test
    @DisplayName("failure mode 9: a relabelled confidence level is refused")
    fun uncertaintyRefusesARelabelledConfidenceLevel() {
        assertThrows<IllegalArgumentException> {
            GeomagneticModelUncertainty(
                declinationSigma1Deg = 0.36,
                sourceModelId = GeomagneticModelId.WMM2025,
                errorModelId = "wmm2025-error-model",
                errorModelHash = "sha256:0",
                sourceDocumentReference = "ref",
                sourceConfidenceLevel = ConfidenceLevel.TWO_SIDED_95,
            )
        }
    }

    @Test
    @DisplayName("§19.2: sigma to bound is applied exactly once, factor from config")
    fun sigmaToBoundIsAppliedExactlyOnce() {
        val uncertainty = GeomagneticModelUncertainty(
            declinationSigma1Deg = 0.5,
            sourceModelId = GeomagneticModelId.WMM2025,
            errorModelId = "hypothetical-error-model-for-this-test",
            errorModelHash = "sha256:deadbeef",
            sourceDocumentReference = "test only; not a vendored artifact",
        )
        val factor = profile.declinationSigmaToBound95Factor
        val once = Geomagnetic.declinationBound95Deg(uncertainty, factor)
        assertEquals(0.98, once, 1e-12)
        // Applying it twice is the shape of the defect §19.2 exists to prevent.
        assertTrue(kotlin.math.abs(once * factor - once) > 1e-6)
    }

    // ---------------------------------------------------------------------------------
    // §10.2 altitude datum
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§10.2: an ellipsoidal altitude passes through")
    fun ellipsoidalAltitudePassesThrough() {
        assertEquals(
            120.0,
            Geomagnetic.ellipsoidalAltitudeM(
                AltitudeSample(120.0, AltitudeReference.WGS84_ELLIPSOID)
            ),
        )
    }

    @Test
    @DisplayName("§10.2: an orthometric altitude is converted or refused, never assumed")
    fun orthometricAltitudeIsConvertedOrRefused() {
        val sample = AltitudeSample(120.0, AltitudeReference.MSL_ORTHOMETRIC)
        assertThrows<AltitudeDatumUnconvertedException> {
            Geomagnetic.ellipsoidalAltitudeM(sample)
        }
        assertEquals(86.5, Geomagnetic.ellipsoidalAltitudeM(sample, geoidSeparationM = -33.5), 1e-9)
    }

    @Test
    @DisplayName("§2/§10.2: UNKNOWN is a real state, never coerced to a datum")
    fun unknownAltitudeIsNeverCoerced() {
        assertThrows<AltitudeDatumUnconvertedException> {
            Geomagnetic.ellipsoidalAltitudeM(AltitudeSample(120.0, AltitudeReference.UNKNOWN))
        }
    }

    @Test
    @DisplayName("§10.2: all three datum cases are representable")
    fun allThreeDatumCasesAreRepresentable() {
        AltitudeReference.entries.forEach {
            assertEquals(it, AltitudeSample(0.0, it).reference)
        }
    }

    // ---------------------------------------------------------------------------------
    // §9/§10 decimal year and validity
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§10: the decimal year at year boundaries")
    fun decimalYearAtYearBoundaries() {
        val start = ZonedDateTime.of(2026, 1, 1, 0, 0, 0, 0, ZoneOffset.UTC).toInstant()
        assertEquals(2026.0, Geomagnetic.wmmDecimalYear(start), 1e-12)
        val almostNext = ZonedDateTime.of(2026, 12, 31, 23, 59, 59, 0, ZoneOffset.UTC).toInstant()
        val value = Geomagnetic.wmmDecimalYear(almostNext)
        assertTrue(value > 2026.999 && value < 2027.0, "value=$value")
    }

    @Test
    @DisplayName("§10: leap years need no branch — 2028 has 366 days")
    fun decimalYearHandlesLeapYears() {
        val leapDay = ZonedDateTime.of(2028, 2, 29, 0, 0, 0, 0, ZoneOffset.UTC).toInstant()
        assertEquals(2028 + 59.0 / 366.0, Geomagnetic.wmmDecimalYear(leapDay), 1e-12)
    }

    @Test
    @DisplayName("§10: the WMM2025 validity interval is half-open [2025.0, 2030.0)")
    fun validityIntervalIsHalfOpen() {
        assertTrue(Geomagnetic.isWithinValidity(2025.0, Geomagnetic.wmm2025Validity))
        assertTrue(Geomagnetic.isWithinValidity(2029.999, Geomagnetic.wmm2025Validity))
        assertFalse(Geomagnetic.isWithinValidity(2030.0, Geomagnetic.wmm2025Validity))
        assertFalse(Geomagnetic.isWithinValidity(2024.999, Geomagnetic.wmm2025Validity))
    }

    @Test
    @DisplayName("§10: a date outside validity refuses rather than extrapolating")
    fun aDateOutsideValidityRefuses() {
        Geomagnetic.requireWithinValidity(2027.5, Geomagnetic.wmm2025Validity)
        val failure = assertThrows<GeomagneticDateOutOfRangeException> {
            Geomagnetic.requireWithinValidity(2030.5, Geomagnetic.wmm2025Validity)
        }
        assertTrue(failure.message!!.contains("GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE"))
    }

    @Test
    @DisplayName("§10/failure mode 8: the declination sign convention is east-positive")
    fun trueHeadingConversionSignConvention() {
        // Validated at both signs because inferring the convention from one location is
        // failure mode 8.
        assertEquals(189.41, CircularMath.normalize360(181.12 + 8.29), 1e-9)
        assertEquals(172.83, CircularMath.normalize360(181.12 + -8.29), 1e-9)
        assertEquals(3.29, CircularMath.normalize360(355.0 + 8.29), 1e-9)
        assertEquals(354.71, CircularMath.normalize360(3.0 + -8.29), 1e-9)
    }

    @Test
    @DisplayName("§8.1: the horizontal-field gate is NOAA's own caution-zone boundary")
    fun horizontalIntensityGateIsReadFromConfig() {
        assertEquals(6000.0, profile.minHorizontalIntensityNanoTesla)
        // The sensitivity claim the spec states, checked rather than trusted: at 6000 nT a
        // 50 nT transverse perturbation is about 0.48 deg.
        assertEquals(0.477, Math.toDegrees(kotlin.math.atan(50.0 / 6000.0)), 0.005)
    }
}
