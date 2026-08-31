package com.fengshuicompass.headingcore.config

import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonArray
import java.io.File

/**
 * SPEC.md §8 `config/precision-profile-v1.json`, typed.
 *
 * Every gate in the engine reads a named key from this type; §18.5 forbids comparing a
 * gate against a numeric literal. The decoder is strict — an unknown key is an error,
 * mirroring the schema's `"additionalProperties": false` — so a key added to the file
 * without a corresponding field fails the build instead of being ignored.
 *
 * There is deliberately **no** calibration-state property, and §8.1 asserts that no key
 * matching `/calibrationState/i` exists anywhere in the document: `boundCalibrationState`
 * is derived at runtime from a §24 certification lookup (§19.1, failure mode 32).
 */
@Serializable
public data class PrecisionProfile(
    val schemaVersion: String,
    val configVersion: String,

    val orientationMaxAgeMs: Long,
    val orientationInvalidAfterMs: Long,
    val freshLocationAtStartMaxAgeMs: Long,
    val locationAtLockMaxAgeMs: Long,
    val usableLocationMaxAgeMs: Long,
    val locationJumpRequiresFreshFixKm: Double,
    val declinationEnvelopeProfessionalMaxDeg: Double,
    val declinationEnvelopeUsableMaxDeg: Double,

    val stableWindowMinMs: Long,
    val acquisitionTimeoutMs: Long,
    val periodicOrientationRequestedHz: Double,
    val minPeriodicSupportSamples: Int,
    val clHeadingMinSamplesPerStableWindow: Int,
    val minCircularResultantLength: Double,
    val angularSpeedP95MaxDegPerSec: Double,
    val linearAccelerationP95MaxG: Double,
    val circularResidualP95MaxDeg: Double,

    val flatModePitchAbsMaxDeg: Double,
    val flatModeRollAbsMaxDeg: Double,
    val flatFreehandPlacementBound95Deg: Double,
    val wallNormalElevationAbsMaxDeg: Double,
    val wallTopAxisFromVerticalMaxDeg: Double,
    val wallFreehandPlacementBound95Deg: Double,

    val targetNearZoneDeg: Double,
    val targetCenteringToleranceDeg: Double,

    val providerCrossCheckMaxDeg: Double,
    val referenceSeparationMarginDeg: Double,
    val smallDeclinationAmbiguityMaxDeg: Double,
    val transformAgreementMaxDeg: Double,

    val magneticMagnitudeResidualSuspectFraction: Double,
    val magneticMagnitudeResidualDisturbedFraction: Double,
    val inclinationResidualSuspectDeg: Double,
    val inclinationResidualDisturbedDeg: Double,
    val stationaryFieldMadSuspectMicroTesla: Double,
    val stationaryFieldMadDisturbedMicroTesla: Double,
    val pipelineDisagreementSuspectDeg: Double,
    val pipelineDisagreementDisturbedDeg: Double,
    val suspectInterferenceBound95Deg: Double,
    val recoveryCleanWindowMs: Long,

    val minHorizontalIntensityNanoTesla: Double,

    val unknownDeviceFloor95Deg: Double,
    val professionalBound95MaxDeg: Double,
    val highBound95MaxDeg: Double,
    val usableBound95MaxDeg: Double,
    val lowConfidenceBound95MaxDeg: Double,

    val spaceWeatherAdvisoryKpMin: Double,
    val spaceWeatherProfessionalSuppressKpMin: Double,
    val spaceWeatherRejectKpMin: Double,
    val spaceWeatherCacheMaxAgeMs: Long,

    val thermalRestrictionBlocksLock: Boolean,
    val wirelessChargingBlocksGradeAboveUsable: Boolean,

    val precisionScreenOrientation: String,
    val requireBoundaryStraddleReporting: Boolean,
    val geomagneticModelId: String,
    val canonicalAltitudeReference: String,
    val declinationSigmaToBound95Factor: Double,
) {
    public companion object {
        /** Strict: rejects unknown keys, exactly as the schema rejects extra properties. */
        private val strictJson = Json {
            ignoreUnknownKeys = false
            isLenient = false
            allowSpecialFloatingPointValues = false
        }

        public fun load(file: File): PrecisionProfile =
            strictJson.decodeFromString(serializer(), file.readText())

        /** The same document as an untyped tree, for the §8.1 whole-document key scan. */
        public fun loadRawTree(file: File): JsonObject =
            strictJson.parseToJsonElement(file.readText()) as JsonObject
    }
}

/**
 * Collects every property name appearing anywhere in [element], at any nesting depth.
 * §8.1's first invariant is "no key matching `/calibrationState/i` exists **anywhere** in
 * the profile", so a nested object may not smuggle one in either.
 */
public fun collectPropertyNames(element: JsonElement, into: MutableList<String> = mutableListOf()): List<String> {
    when (element) {
        is JsonObject -> element.forEach { (name, value) ->
            into += name
            collectPropertyNames(value, into)
        }
        is JsonArray -> element.forEach { collectPropertyNames(it, into) }
        else -> Unit
    }
    return into
}
