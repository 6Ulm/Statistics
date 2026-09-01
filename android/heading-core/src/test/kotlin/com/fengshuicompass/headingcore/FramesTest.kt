package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.frames.AxisSelector
import com.fengshuicompass.headingcore.frames.CanonicalAttitude
import com.fengshuicompass.headingcore.frames.ConventionVerification
import com.fengshuicompass.headingcore.frames.EnuVector
import com.fengshuicompass.headingcore.frames.Frames
import com.fengshuicompass.headingcore.frames.NativeAttitudeConvention
import com.fengshuicompass.headingcore.frames.NativeAttitudeFrame
import com.fengshuicompass.headingcore.frames.NativeConventions
import com.fengshuicompass.headingcore.frames.Quaternion
import com.fengshuicompass.headingcore.frames.SingularProjectionException
import com.fengshuicompass.headingcore.frames.TransformDirection
import com.fengshuicompass.headingcore.frames.Vector3
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.ProviderReferenceContract
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs
import kotlin.math.cos
import kotlin.math.sin

/** SPEC.md §3 / §11.1 / §14 — frames, quaternion order and mode axis projection. */
class FramesTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(SharedArtifacts.attitudeGoldenFixture.readText()).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private fun attitude(components: List<Double>) = CanonicalAttitude(
        quaternionDeviceToReferenceEnuXYZW =
            Quaternion(components[0], components[1], components[2], components[3]),
        referenceContract = ProviderReferenceContract.TRUE,
        nativeFrame = NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
    )

    private fun gateFor(mode: MeasurementMode): Double =
        when (requireNotNull(Frames.modeAxisElevationGateKey[mode])) {
            "flatModePitchAbsMaxDeg" -> profile.flatModePitchAbsMaxDeg
            "wallNormalElevationAbsMaxDeg" -> profile.wallNormalElevationAbsMaxDeg
            else -> error("unmapped gate key")
        }

    @Test
    @DisplayName("§3: the fixture declares the conventions the types encode")
    fun fixtureDeclaresTheConventions() {
        assertEquals("XYZW", fixture["quaternionComponentOrder"]!!.jsonPrimitive.content)
        assertEquals("DEVICE_TO_REFERENCE", fixture["transformDirection"]!!.jsonPrimitive.content)
        assertEquals("REFERENCE_ENU", fixture["canonicalFrame"]!!.jsonPrimitive.content)
        assertEquals(
            listOf(0.0, 1.0, 0.0),
            fixture["modeReferenceVectors"]!!.jsonObject["FLAT_TOP_EDGE"]!!.jsonArray
                .map { it.jsonPrimitive.double },
        )
        assertEquals(
            listOf(0.0, 0.0, 1.0),
            fixture["modeReferenceVectors"]!!.jsonObject["WALL_FLUSH_BACK"]!!.jsonArray
                .map { it.jsonPrimitive.double },
        )
    }

    @Test
    @DisplayName("§11.1/§14: physical N/E/S/W golden poses project to the expected bearing")
    fun goldenPosesProjectToTheExpectedBearing() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val mode = MeasurementMode.valueOf(case["measurementMode"]!!.jsonPrimitive.content)
            val canonical = attitude(
                case["quaternionDeviceToReferenceEnuXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            )
            val gate = gateFor(mode)
            if (case["expectIllConditioned"]?.jsonPrimitive?.boolean == true) {
                assertThrows<SingularProjectionException>({ id }) {
                    Frames.modeAxisBearingOrReject(canonical, mode, gate)
                }
                return@forEach
            }
            val bearing = Frames.modeAxisBearingOrReject(canonical, mode, gate)
            assertEquals(
                case["expectedHeadingDeg"]!!.jsonPrimitive.double,
                bearing.headingDeg,
                1e-9,
                id,
            )
            assertEquals(
                case["expectedElevationDeg"]!!.jsonPrimitive.double,
                bearing.elevationDeg,
                1e-9,
                id,
            )
        }
    }

    @Test
    @DisplayName("§14: only an exactly vertical axis is caught by the zero check")
    fun onlyAnExactlyVerticalAxisIsCaughtByTheZeroCheck() {
        // wall-axis-vertical-singular projects to an exact (0, 0) and the bearing function
        // itself refuses. flat-axis-vertical-ill-conditioned projects to 2.2e-16 — the zero
        // check passes it and atan2 hands back a confident, arbitrary bearing. Only the
        // configured elevation gate rejects the second.
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            if (case["expectIllConditioned"]?.jsonPrimitive?.boolean != true) return@forEach
            val mode = MeasurementMode.valueOf(case["measurementMode"]!!.jsonPrimitive.content)
            val canonical = attitude(
                case["quaternionDeviceToReferenceEnuXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            )
            val enu = Frames.deviceVectorToReferenceEnu(
                canonical,
                requireNotNull(Frames.modeReferenceVectors[mode]),
            )
            if (case["expectExactlySingular"]!!.jsonPrimitive.boolean) {
                assertEquals(0.0, enu.horizontalNorm)
                assertThrows<SingularProjectionException> { Frames.enuBearingDeg(enu) }
            } else {
                assertTrue(enu.horizontalNorm > 0.0 && enu.horizontalNorm < 1e-12)
                // The naive check would let this through with a plausible-looking bearing.
                assertTrue(Frames.enuBearingDeg(enu) in 0.0..360.0)
            }
            assertEquals(90.0, abs(Frames.enuElevationDeg(enu)), 1e-6)
        }
    }

    @Test
    @DisplayName("§11.1/§16.1: the two extraction routes agree (~0 transformAgreementDeg)")
    fun theTwoExtractionRoutesAgree() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            if (case["expectIllConditioned"]?.jsonPrimitive?.boolean == true) return@forEach
            val canonical = attitude(
                case["quaternionDeviceToReferenceEnuXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            )
            val mode = MeasurementMode.valueOf(case["measurementMode"]!!.jsonPrimitive.content)
            assertEquals(
                0.0,
                Frames.transformAgreementDeg(canonical, mode),
                1e-9,
                case["id"]!!.jsonPrimitive.content,
            )
        }
    }

    @Test
    @DisplayName("R50/§11.1: wall mode does not fall through to the top edge")
    fun wallModeDoesNotFallThroughToTheTopEdge() {
        val wallCase = fixture["cases"]!!.jsonArray
            .map { it.jsonObject }
            .first { it["id"]!!.jsonPrimitive.content == "wall-outward-normal-east" }
        val canonical = attitude(
            wallCase["quaternionDeviceToReferenceEnuXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
        )
        assertEquals(
            90.0,
            Frames.modeAxisBearingOrReject(
                canonical,
                MeasurementMode.WALL_FLUSH_BACK,
                profile.wallNormalElevationAbsMaxDeg,
            ).headingDeg,
            1e-9,
        )
        assertThrows<SingularProjectionException> {
            Frames.modeAxisBearingOrReject(
                canonical,
                MeasurementMode.FLAT_TOP_EDGE,
                profile.flatModePitchAbsMaxDeg,
            )
        }
    }

    @Test
    @DisplayName("R49: the Core Motion native conversion matches the declared convention")
    fun coreMotionNativeConversionMatchesTheDeclaredConvention() {
        // Phase 1 pins the conversion *given* the declared convention. The convention itself
        // is DECLARED_UNVERIFIED and Phase 2 must confirm it against the pinned SDK with
        // physical poses — this test does not and cannot establish it.
        fixture["nativeFrameConversion"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            assertEquals("DECLARED_UNVERIFIED", case["verification"]!!.jsonPrimitive.content)
            val convention = requireNotNull(
                NativeConventions.byFrame[
                    NativeAttitudeFrame.valueOf(case["nativeFrame"]!!.jsonPrimitive.content),
                ]
            )
            assertSame(ConventionVerification.DECLARED_UNVERIFIED, convention.verification)
            val nativeComponents =
                case["nativeQuaternionXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            val canonical = Frames.canonicalizeNativeAttitude(
                Quaternion(
                    nativeComponents[0],
                    nativeComponents[1],
                    nativeComponents[2],
                    nativeComponents[3],
                ),
                convention,
            )
            val expected =
                case["expectedCanonicalQuaternionXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            // A quaternion and its negation are the same rotation, so compare the rotation.
            listOf(
                Vector3(1.0, 0.0, 0.0),
                Vector3(0.0, 1.0, 0.0),
                Vector3(0.0, 0.0, 1.0),
            ).forEach { deviceVector ->
                val got = Frames.rotateVectorByQuaternion(
                    canonical.quaternionDeviceToReferenceEnuXYZW,
                    deviceVector,
                )
                val want = Frames.rotateVectorByQuaternion(
                    Quaternion(expected[0], expected[1], expected[2], expected[3]),
                    deviceVector,
                )
                assertEquals(want.east, got.east, 1e-12)
                assertEquals(want.north, got.north, 1e-12)
                assertEquals(want.up, got.up, 1e-12)
            }
            assertEquals(
                case["expectedFlatHeadingDeg"]!!.jsonPrimitive.double,
                Frames.modeReferenceVectorHeadingDeg(
                    canonical,
                    MeasurementMode.FLAT_TOP_EDGE,
                ).headingDeg,
                1e-9,
            )
            assertSame(convention.frame, canonical.nativeFrame)
            assertSame(convention.referenceContract, canonical.referenceContract)
        }
    }

    @Test
    @DisplayName("§11.1: every declared native convention is a proper rotation")
    fun everyDeclaredConventionIsAProperRotation() {
        NativeConventions.byFrame.values.forEach {
            assertEquals(1.0, it.permutationDeterminant(), 1e-12, it.frame.wire)
        }
    }

    @Test
    @DisplayName("§11.1: a reflecting axis map is rejected")
    fun aReflectingAxisMapIsRejected() {
        assertThrows<IllegalArgumentException> {
            NativeAttitudeConvention(
                frame = NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
                east = AxisSelector(0, 1.0),
                north = AxisSelector(1, 1.0),
                up = AxisSelector(2, -1.0), // flips handedness
                direction = TransformDirection.DEVICE_TO_REFERENCE,
                referenceContract = ProviderReferenceContract.TRUE,
                verification = ConventionVerification.DECLARED_UNVERIFIED,
            )
        }
    }

    @Test
    @DisplayName("failure mode 5: the transform direction is converted, not assumed")
    fun transformDirectionIsConvertedNotAssumed() {
        val deviceToReference = Quaternion(0.0, 0.0, -0.7071067811865475, 0.7071067811865476)
        val referenceToDevice = NativeAttitudeConvention(
            frame = NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
            east = AxisSelector(0, 1.0),
            north = AxisSelector(1, 1.0),
            up = AxisSelector(2, 1.0),
            direction = TransformDirection.REFERENCE_TO_DEVICE,
            referenceContract = ProviderReferenceContract.TRUE,
            verification = ConventionVerification.DECLARED_UNVERIFIED,
        )
        val canonical =
            Frames.canonicalizeNativeAttitude(deviceToReference.conjugate(), referenceToDevice)
        assertEquals(
            90.0,
            Frames.modeReferenceVectorHeadingDeg(canonical, MeasurementMode.FLAT_TOP_EDGE)
                .headingDeg,
            1e-9,
        )
    }

    @Test
    @DisplayName("§11.1: the quaternion/matrix round trip preserves the rotation")
    fun quaternionAndMatrixRoundTrip() {
        fixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val components =
                case["quaternionDeviceToReferenceEnuXyzw"]!!.jsonArray.map { it.jsonPrimitive.double }
            val quaternion =
                Quaternion(components[0], components[1], components[2], components[3])
            val recovered = Frames.quaternionFromRotationMatrix(
                Frames.rotationMatrixFromQuaternion(quaternion)
            )
            listOf(
                Vector3(1.0, 0.0, 0.0),
                Vector3(0.0, 1.0, 0.0),
                Vector3(0.0, 0.0, 1.0),
            ).forEach { deviceVector ->
                val original = Frames.rotateVectorByQuaternion(quaternion, deviceVector)
                val again = Frames.rotateVectorByQuaternion(recovered, deviceVector)
                assertEquals(original.east, again.east, 1e-12)
                assertEquals(original.north, again.north, 1e-12)
                assertEquals(original.up, again.up, 1e-12)
            }
        }
    }

    @Test
    @DisplayName("§11.1: a canonical attitude requires a unit quaternion")
    fun canonicalAttitudeRequiresAUnitQuaternion() {
        assertThrows<IllegalArgumentException> {
            CanonicalAttitude(
                quaternionDeviceToReferenceEnuXYZW = Quaternion(0.0, 0.0, 0.0, 0.5),
                referenceContract = ProviderReferenceContract.TRUE,
                nativeFrame = NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
            )
        }
    }

    @Test
    @DisplayName("§5: nonfinite frame components are rejected")
    fun nonFiniteComponentsAreRejected() {
        assertThrows<IllegalArgumentException> { Quaternion(Double.NaN, 0.0, 0.0, 1.0) }
        assertThrows<IllegalArgumentException> { Vector3(0.0, Double.POSITIVE_INFINITY, 0.0) }
        assertThrows<IllegalArgumentException> { EnuVector(0.0, 0.0, Double.NaN) }
    }

    @Test
    @DisplayName("§14: the bearing is atan2(east, north), not atan2(north, east)")
    fun bearingUsesAtan2EastNorth() {
        // Swapping the arguments mirrors every bearing about north, which looks entirely
        // plausible on a dial.
        assertEquals(90.0, Frames.enuBearingDeg(EnuVector(1.0, 0.0, 0.0)), 1e-12)
        assertEquals(0.0, Frames.enuBearingDeg(EnuVector(0.0, 1.0, 0.0)), 1e-12)
        assertEquals(270.0, Frames.enuBearingDeg(EnuVector(-1.0, 0.0, 0.0)), 1e-12)
        assertEquals(45.0, Frames.enuBearingDeg(EnuVector(1.0, 1.0, 0.0)), 1e-12)
    }

    @Test
    @DisplayName("§14: elevation is the conditioning measure")
    fun elevationIsTheConditioningMeasure() {
        assertEquals(0.0, Frames.enuElevationDeg(EnuVector(0.0, 1.0, 0.0)), 1e-12)
        assertEquals(90.0, Frames.enuElevationDeg(EnuVector(0.0, 0.0, 1.0)), 1e-12)
        val fiveDegrees = EnuVector(
            0.0,
            cos(Math.toRadians(5.0)),
            sin(Math.toRadians(5.0)),
        )
        assertEquals(5.0, Frames.enuElevationDeg(fiveDegrees), 1e-9)
    }

    @Test
    @DisplayName("§14/failure mode 18: sitting is the labelled derived opposite of facing")
    fun sittingIsTheLabelledDerivedOpposite() {
        assertEquals(270.0, Frames.sittingFromFacingDeg(90.0), 1e-12)
        assertEquals(170.0, Frames.sittingFromFacingDeg(350.0), 1e-12)
        assertEquals(180.0, Frames.sittingFromFacingDeg(0.0), 1e-12)
    }
}
