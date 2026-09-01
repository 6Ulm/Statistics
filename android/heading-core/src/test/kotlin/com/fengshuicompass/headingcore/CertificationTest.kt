package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.certification.Certification
import com.fengshuicompass.headingcore.certification.CertificationDatabase
import com.fengshuicompass.headingcore.certification.CertificationKey
import com.fengshuicompass.headingcore.certification.CertificationKeyException
import com.fengshuicompass.headingcore.certification.CertificationLookupOutcome
import com.fengshuicompass.headingcore.certification.CertificationRecord
import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.deviation.DeviationCorrection
import com.fengshuicompass.headingcore.deviation.DeviationCorrectionProfileMetadata
import com.fengshuicompass.headingcore.grade.QualityGrade
import com.fengshuicompass.headingcore.model.BoundCalibrationState
import com.fengshuicompass.headingcore.model.DeviationCorrectionScope
import com.fengshuicompass.headingcore.model.DeviationCorrectionState
import com.fengshuicompass.headingcore.model.DeviationStructureClass
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import com.fengshuicompass.headingcore.model.LocationProviderId
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.PlacementMethod
import com.fengshuicompass.headingcore.model.ProviderErrorSource
import com.fengshuicompass.headingcore.model.ProviderId
import com.fengshuicompass.headingcore.model.UncertaintyCoverageEvidenceState
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows

/** SPEC.md §24 / §19.1 / §19.3 — the key, `miss -> CANDIDATE`, and the invariants. */
class CertificationTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.certificationKeyFixture.readText()
        ).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private fun key(document: JsonObject, overrides: Map<String, String> = emptyMap()) =
        CertificationKey(
            certificationSchemaVersion = text(document, overrides, "certificationSchemaVersion"),
            hardwareRuntimeIdentity = text(document, overrides, "hardwareRuntimeIdentity"),
            sensorRuntimeIdentity = text(document, overrides, "sensorRuntimeIdentity"),
            osBuildIdentity = text(document, overrides, "osBuildIdentity"),
            providerId = ProviderId.valueOf(text(document, overrides, "providerId")),
            providerRuntimeIdentity = text(document, overrides, "providerRuntimeIdentity"),
            providerErrorSource =
                ProviderErrorSource.valueOf(text(document, overrides, "providerErrorSource")),
            locationProviderId =
                LocationProviderId.valueOf(text(document, overrides, "locationProviderId")),
            locationProviderRuntimeIdentity =
                text(document, overrides, "locationProviderRuntimeIdentity"),
            measurementMode =
                MeasurementMode.valueOf(text(document, overrides, "measurementMode")),
            placementMethod =
                PlacementMethod.valueOf(text(document, overrides, "placementMethod")),
            placementProfileHash = text(document, overrides, "placementProfileHash"),
            geomagneticModelId =
                GeomagneticModelId.valueOf(text(document, overrides, "geomagneticModelId")),
            geomagneticCoefficientHash = text(document, overrides, "geomagneticCoefficientHash"),
            geomagneticErrorModelHash = text(document, overrides, "geomagneticErrorModelHash"),
            deviationCorrectionProfileHash =
                text(document, overrides, "deviationCorrectionProfileHash"),
            engineDecisionLogicHash = text(document, overrides, "engineDecisionLogicHash"),
            precisionConfigHash = text(document, overrides, "precisionConfigHash"),
        )

    private fun text(document: JsonObject, overrides: Map<String, String>, name: String): String =
        overrides[name] ?: document[name]!!.jsonPrimitive.content

    @Test
    @DisplayName("§37 rule 12: the shipped database is empty")
    fun theShippedDatabaseIsEmpty() {
        assertEquals(0, fixture["shippedDatabaseRecordCount"]!!.jsonPrimitive.content.toInt())
        assertEquals(0, CertificationDatabase.shipped().size)
    }

    @Test
    @DisplayName("§24: a miss yields CANDIDATE and the unknown floor")
    fun aMissYieldsCandidateAndTheUnknownFloor() {
        val outcome = CertificationDatabase.shipped()
            .lookup(key(fixture["completeKey"]!!.jsonObject), profile.unknownDeviceFloor95Deg)
        val expected = fixture["expectedLookupOnShippedDatabase"]!!.jsonObject
        assertEquals(
            expected["boundCalibrationState"]!!.jsonPrimitive.content,
            outcome.boundCalibrationState.wire,
        )
        assertEquals(
            expected["uncertaintyCoverageEvidenceState"]!!.jsonPrimitive.content,
            outcome.uncertaintyCoverageEvidenceState.wire,
        )
        assertEquals(
            expected["supportedQualityGrade"]!!.jsonPrimitive.content,
            outcome.supportedQualityGrade.name,
        )
        assertEquals(profile.unknownDeviceFloor95Deg, outcome.deviceFloor95Deg)
        assertNull(outcome.record)
    }

    @Test
    @DisplayName("§24/R54/R66: every named component actually differentiates the key")
    fun everyNamedComponentDifferentiatesTheKey() {
        // A component that did not differentiate would let a certification silently survive a
        // change §24 says invalidates it.
        val complete = key(fixture["completeKey"]!!.jsonObject)
        val database = CertificationDatabase.withRecords(
            listOf(
                CertificationRecord(
                    key = complete,
                    deviceFloor95Deg = 1.2,
                    supportedQualityGrade = QualityGrade.USABLE,
                    earnedUnderEngineVersion = "heading-3.2.0",
                    evidenceReportId = "report-for-this-test-only",
                    certificationDate = "2026-01-01",
                )
            )
        )
        val hit = database.lookup(complete, profile.unknownDeviceFloor95Deg)
        assertSame(BoundCalibrationState.CALIBRATED, hit.boundCalibrationState)
        assertEquals(1.2, hit.deviceFloor95Deg)

        val replacements = mapOf(
            "providerErrorSource" to "GOOGLE_ORDINARY",
            "locationProviderRuntimeIdentity" to "GMS:some-other-exact-version",
            "placementProfileHash" to "sha256:0123456789abcdef",
            "geomagneticErrorModelHash" to "sha256:fedcba9876543210",
            "engineDecisionLogicHash" to "sha256:1111111111111111",
            "precisionConfigHash" to "sha256:2222222222222222",
            "measurementMode" to "WALL_FLUSH_BACK",
        )
        fixture["keyComponentsThatMustDifferentiate"]!!.jsonArray.forEach { entry ->
            val component = entry.jsonPrimitive.content
            val perturbed = key(
                fixture["completeKey"]!!.jsonObject,
                mapOf(component to requireNotNull(replacements[component])),
            )
            val outcome = database.lookup(perturbed, profile.unknownDeviceFloor95Deg)
            assertSame(BoundCalibrationState.CANDIDATE, outcome.boundCalibrationState, component)
            assertNull(outcome.record, component)
            assertEquals(profile.unknownDeviceFloor95Deg, outcome.deviceFloor95Deg, component)
        }
    }

    @Test
    @DisplayName("§24/R66: open-ended identities are rejected")
    fun openEndedIdentitiesAreRejected() {
        fixture["rejectedOpenEndedIdentities"]!!.jsonArray.forEach { entry ->
            assertThrows<CertificationKeyException>({ entry.jsonPrimitive.content }) {
                key(
                    fixture["completeKey"]!!.jsonObject,
                    mapOf("osBuildIdentity" to entry.jsonPrimitive.content),
                )
            }
        }
    }

    @Test
    @DisplayName("§24: the NOT_RUNTIME_OBSERVABLE sentinel is accepted")
    fun theNotRuntimeObservableSentinelIsAccepted() {
        val sentinel = fixture["notRuntimeObservableSentinel"]!!.jsonPrimitive.content
        val built = key(
            fixture["completeKey"]!!.jsonObject,
            mapOf("sensorRuntimeIdentity" to sentinel),
        )
        assertEquals(Certification.NOT_RUNTIME_OBSERVABLE, built.sensorRuntimeIdentity)
    }

    @Test
    @DisplayName("§24: an empty component is rejected")
    fun anEmptyComponentIsRejected() {
        assertThrows<CertificationKeyException> {
            key(fixture["completeKey"]!!.jsonObject, mapOf("engineDecisionLogicHash" to "  "))
        }
    }

    @Test
    @DisplayName("§24: a record exists only for CALIBRATED")
    fun aRecordExistsOnlyForCalibrated() {
        assertThrows<IllegalArgumentException> {
            CertificationRecord(
                key = key(fixture["completeKey"]!!.jsonObject),
                deviceFloor95Deg = 1.2,
                supportedQualityGrade = QualityGrade.USABLE,
                earnedUnderEngineVersion = "heading-3.2.0",
                evidenceReportId = "report",
                certificationDate = "2026-01-01",
                boundCalibrationState = BoundCalibrationState.CANDIDATE,
            )
        }
    }

    @Test
    @DisplayName("§24: a record without resolvable evidence is rejected")
    fun aRecordWithoutResolvableEvidenceIsRejected() {
        assertThrows<IllegalArgumentException> {
            CertificationRecord(
                key = key(fixture["completeKey"]!!.jsonObject),
                deviceFloor95Deg = 1.2,
                supportedQualityGrade = QualityGrade.USABLE,
                earnedUnderEngineVersion = "heading-3.2.0",
                evidenceReportId = "",
                certificationDate = "2026-01-01",
            )
        }
    }

    // ---------------------------------------------------------------------------------
    // §19.1 invariants
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§19.1: the calibration invariants hold both ways")
    fun theCalibrationInvariantsHoldBothWays() {
        Certification.assertCalibrationInvariants(
            BoundCalibrationState.CALIBRATED,
            UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
        )
        Certification.assertCalibrationInvariants(
            BoundCalibrationState.CANDIDATE,
            UncertaintyCoverageEvidenceState.TARGET_ONLY,
        )
        Certification.assertCalibrationInvariants(
            BoundCalibrationState.CANDIDATE,
            UncertaintyCoverageEvidenceState.UNDEFINED,
        )
        listOf(
            BoundCalibrationState.CALIBRATED to UncertaintyCoverageEvidenceState.TARGET_ONLY,
            BoundCalibrationState.CALIBRATED to UncertaintyCoverageEvidenceState.UNDEFINED,
            BoundCalibrationState.CANDIDATE to
                UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
        ).forEach { (state, evidence) ->
            assertThrows<IllegalArgumentException> {
                Certification.assertCalibrationInvariants(state, evidence)
            }
        }
    }

    @Test
    @DisplayName("§19.1: the invariants are enforced on every lookup outcome")
    fun theInvariantsAreEnforcedOnEveryLookupOutcome() {
        assertThrows<IllegalArgumentException> {
            CertificationLookupOutcome(
                boundCalibrationState = BoundCalibrationState.CANDIDATE,
                uncertaintyCoverageEvidenceState =
                    UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
                deviceFloor95Deg = profile.unknownDeviceFloor95Deg,
                supportedQualityGrade = QualityGrade.USABLE,
                record = null,
            )
        }
    }

    // ---------------------------------------------------------------------------------
    // §19.3 deviation correction, whose hash is a key component
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§19.3: the production deviation state is NONE")
    fun theProductionDeviationStateIsNone() {
        assertNull(DeviationCorrection.lookupDeviationProfile("any", "live", "context"))
        val outcome = DeviationCorrection.applyDeviationCorrection(189.0)
        assertSame(DeviationCorrectionState.NONE, outcome.state)
        assertEquals(0.0, outcome.correctionDeg)
        assertEquals(189.0, outcome.trueHeadingDeg)
        assertEquals(189.0, outcome.uncorrectedTrueHeadingDeg)
        assertEquals(DeviationCorrection.NONE_PROFILE_HASH, outcome.profileHash)
        assertEquals(0.0, outcome.residualBound95Deg)
    }

    @Test
    @DisplayName("§24: the NONE sentinel is a literal, not a null")
    fun theNoneSentinelIsALiteral() {
        assertEquals(
            "NONE",
            fixture["completeKey"]!!.jsonObject["deviationCorrectionProfileHash"]!!
                .jsonPrimitive.content,
        )
        assertEquals("NONE", DeviationCorrection.NONE_PROFILE_HASH)
    }

    private fun profileWithScope(scope: DeviationCorrectionScope) =
        DeviationCorrectionProfileMetadata(
            profileId = "test-profile",
            profileHash = "sha256:abc",
            scope = scope,
            structureClass = DeviationStructureClass.MODEL_CLASS_STABLE,
            correctionMethodId = "circular-harmonic-v1",
            measurementMode = MeasurementMode.FLAT_TOP_EDGE,
            placementMethod = PlacementMethod.NONMAGNETIC_ALIGNMENT_JIG,
            providerId = ProviderId.GOOGLE_FOP,
            coveredProviderRuntimeIdentities = listOf("GMS:exact"),
            coveredOsBuildIdentities = listOf("exact-build"),
            geomagneticModelId = "WMM2025",
            geomagneticCoefficientHash = "sha256:def",
            precisionConfigHash = "sha256:ghi",
            heldOutResidualBound95Deg = 0.4,
            trainingEvidenceId = "train",
            heldOutEvidenceId = "heldout",
        )

    @Test
    @DisplayName("§19.3/§30.6: a UNIT-scope profile can never produce CALIBRATED output")
    fun unitScopeProfileCanNeverProduceCalibratedOutput() {
        val unit = profileWithScope(DeviationCorrectionScope.UNIT)
        assertFalse(unit.mayProduceCalibratedOutput)
        assertThrows<IllegalArgumentException> {
            DeviationCorrection.applyDeviationCorrection(189.0, 0.3, unit)
        }
    }

    @Test
    @DisplayName("§19.3: a model-class profile applies exactly once")
    fun aModelClassProfileAppliesExactlyOnce() {
        val outcome = DeviationCorrection.applyDeviationCorrection(
            355.0,
            8.0,
            profileWithScope(DeviationCorrectionScope.MODEL_CLASS),
        )
        assertSame(DeviationCorrectionState.CERTIFIED_PROFILE, outcome.state)
        assertEquals(355.0, outcome.uncorrectedTrueHeadingDeg)
        assertEquals(3.0, outcome.trueHeadingDeg, 1e-9)
        assertEquals(0.4, outcome.residualBound95Deg)
    }

    @Test
    @DisplayName("§24: a real profile may not use the NONE sentinel as its hash")
    fun aRealProfileMayNotUseTheNoneSentinel() {
        assertThrows<IllegalArgumentException> {
            profileWithScope(DeviationCorrectionScope.MODEL_CLASS).copy(profileHash = "NONE")
        }
    }

    @Test
    @DisplayName("§19.1/failure mode 32: boundCalibrationState is never read from config")
    fun boundCalibrationStateIsNeverReadFromConfig() {
        val raw = PrecisionProfile.loadRawTree(SharedArtifacts.precisionProfileFile)
        assertTrue(raw.keys.none { it.lowercase().contains("calibrationstate") })
        val outcome = CertificationDatabase.shipped()
            .lookup(key(fixture["completeKey"]!!.jsonObject), profile.unknownDeviceFloor95Deg)
        assertSame(BoundCalibrationState.CANDIDATE, outcome.boundCalibrationState)
    }
}
