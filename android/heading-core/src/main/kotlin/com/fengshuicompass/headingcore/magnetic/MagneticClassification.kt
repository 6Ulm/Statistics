package com.fengshuicompass.headingcore.magnetic

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.ReferenceMagneticPrecheckState
import kotlin.math.abs
import kotlin.math.asin

/**
 * SPEC.md §16 — magnetic interference detection, and the §11 reference precheck.
 *
 * Two rules dominate this file and both have a named failure mode behind them:
 *
 * * **The detector MUST NOT use magnitude alone** (failure mode 23). A disturbance can rotate
 *   the field vector with little magnitude change, and that is precisely the case producing a
 *   confident wrong bearing. Magnitude, inclination and stationary variability are fused, plus
 *   independent-pipeline disagreement.
 * * **Absent evidence is not passing evidence.** A feature that could not be measured — the
 *   stationary MAD while the device is moving, `pipelineAgreementDeg` with fewer than two
 *   valid active-axis pipelines — makes the classifier resolve `UNKNOWN`, never `CLEAN`.
 *
 * The precheck and the final state are separate by construction (R59): the precheck reads no
 * pipeline- or reference-dependent feature, so the dependency order
 * `evidence -> precheck -> §11 resolution -> pipeline agreement -> final MagneticState -> lock`
 * stays acyclic.
 */
public object MagneticClassification {

    /**
     * §16: `degrees(asin(clamp(-Bup / M, -1, 1)))` — input is canonical REFERENCE_ENU.
     *
     * **The minus sign is mandatory** (R60): canonical REFERENCE_ENU has `Bup` positive
     * *upward*, while WMM inclination `I` and WMM vertical component `Z` are positive
     * *downward*. Comparing `asin(Bup/M)` directly with WMM `I` reverses the observed sign and
     * can reject a clean northern-hemisphere field as disturbed.
     *
     * The clamp is not decorative either: `asin` of `1 + 1e-16` is a domain error, and a
     * measured component can exceed the magnitude by a rounding bit (failure mode 6).
     */
    public fun measuredInclinationPositiveDownDeg(
        upMicroTesla: Double,
        magnitudeMicroTesla: Double,
    ): Double {
        require(upMicroTesla.isFinite() && magnitudeMicroTesla.isFinite()) {
            "inclination requires finite field components"
        }
        require(magnitudeMicroTesla > 0.0) {
            "field magnitude must be positive, got $magnitudeMicroTesla"
        }
        return Math.toDegrees(asin((-upMicroTesla / magnitudeMicroTesla).coerceIn(-1.0, 1.0)))
    }

    /**
     * §16: a **linear** difference in `[-90, 90]`, never a circular one.
     *
     * "Inclination cannot wrap; a circular difference there is a category error that silently
     * rescales the residual near the poles." Both operands are positive-down, so this is the
     * one place in the codebase where an angular difference is deliberately *not* routed
     * through `shortestSignedDifferenceDeg`.
     */
    public fun inclinationResidualDeg(
        measuredPositiveDownDeg: Double,
        expectedPositiveDownDeg: Double,
    ): Double {
        listOf(
            "measured" to measuredPositiveDownDeg,
            "expected" to expectedPositiveDownDeg,
        ).forEach { (name, value) ->
            require(value.isFinite()) { "$name inclination must be finite, got $value" }
            require(value in -90.0..90.0) {
                "$name inclination must lie in [-90, 90] degrees positive-down, got $value"
            }
        }
        return measuredPositiveDownDeg - expectedPositiveDownDeg
    }

    /** §16: `abs(M - expectedM) / expectedM`. */
    public fun relativeMagnitudeResidual(
        measuredMicroTesla: Double,
        expectedMicroTesla: Double,
    ): Double {
        require(measuredMicroTesla.isFinite() && expectedMicroTesla.isFinite()) {
            "magnitude residual requires finite inputs"
        }
        require(expectedMicroTesla > 0.0) {
            "expected field magnitude must be positive, got $expectedMicroTesla"
        }
        return abs(measuredMicroTesla - expectedMicroTesla) / expectedMicroTesla
    }

    /** §16's classifier, in the specified order, run **after** any §11 Google resolution. */
    public fun classifyMagneticState(
        features: MagneticFeatures,
        thresholds: MagneticThresholds,
    ): MagneticState {
        if (features.isInvalid) return MagneticState.INVALID

        if (atOrAbove(features.relativeMagnitudeResidual, thresholds.magnitudeResidualDisturbedFraction) ||
            atOrAbove(features.inclinationResidualDeg?.let(::abs), thresholds.inclinationResidualDisturbedDeg) ||
            atOrAbove(features.stationaryFieldMadMicroTesla, thresholds.stationaryFieldMadDisturbedMicroTesla) ||
            atOrAbove(features.pipelineAgreementDeg, thresholds.pipelineDisagreementDisturbedDeg)
        ) {
            return MagneticState.DISTURBED
        }

        if (atOrAbove(features.relativeMagnitudeResidual, thresholds.magnitudeResidualSuspectFraction) ||
            atOrAbove(features.inclinationResidualDeg?.let(::abs), thresholds.inclinationResidualSuspectDeg) ||
            atOrAbove(features.stationaryFieldMadMicroTesla, thresholds.stationaryFieldMadSuspectMicroTesla) ||
            atOrAbove(features.pipelineAgreementDeg, thresholds.pipelineDisagreementSuspectDeg)
        ) {
            return MagneticState.SUSPECT
        }

        // An unmeasured feature is not a passing feature.
        return if (features.allRequiredFeaturesPresent) MagneticState.CLEAN else MagneticState.UNKNOWN
    }

    /**
     * §11/§16 precheck — the narrow eligibility gate for the §11 hypothesis test.
     *
     * It reads **only** the three non-pipeline features. It MUST NOT read
     * `pipelineAgreementDeg`, `resolvedReference`, or any feature whose construction needs a
     * reference-resolved Google heading: that circularity is R59, a Critical failure in which
     * reference resolution requires the final magnetic state while the final state requires a
     * reference-resolved pipeline.
     *
     * This is never a substitute for final classification; a Google lock still requires the
     * post-resolution final `MagneticState`.
     */
    public fun referenceMagneticPrecheckState(
        features: MagneticFeatures,
        thresholds: MagneticThresholds,
    ): ReferenceMagneticPrecheckState {
        if (features.isInvalid) return ReferenceMagneticPrecheckState.NOT_CLEAN_FOR_REFERENCE
        if (!features.nonPipelineFeaturesPresent) return ReferenceMagneticPrecheckState.UNKNOWN

        return if (
            atOrAbove(features.relativeMagnitudeResidual, thresholds.magnitudeResidualSuspectFraction) ||
            atOrAbove(features.inclinationResidualDeg?.let(::abs), thresholds.inclinationResidualSuspectDeg) ||
            atOrAbove(features.stationaryFieldMadMicroTesla, thresholds.stationaryFieldMadSuspectMicroTesla)
        ) {
            ReferenceMagneticPrecheckState.NOT_CLEAN_FOR_REFERENCE
        } else {
            ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE
        }
    }

    /**
     * An absent feature cannot exceed a threshold — and cannot clear one either. Clearing is
     * decided separately by the presence check, so this never converts absence into evidence
     * of a clean field.
     */
    private fun atOrAbove(value: Double?, threshold: Double): Boolean =
        value != null && value >= threshold
}

/**
 * The §8 candidate gates the classifier reads, versioned jointly with the model.
 *
 * §10.1: changing `geomagneticModelId` silently re-tunes these gates, so a model change
 * invalidates threshold calibration and requires re-running §30.3.
 */
public data class MagneticThresholds(
    val magnitudeResidualSuspectFraction: Double,
    val magnitudeResidualDisturbedFraction: Double,
    val inclinationResidualSuspectDeg: Double,
    val inclinationResidualDisturbedDeg: Double,
    val stationaryFieldMadSuspectMicroTesla: Double,
    val stationaryFieldMadDisturbedMicroTesla: Double,
    val pipelineDisagreementSuspectDeg: Double,
    val pipelineDisagreementDisturbedDeg: Double,
) {
    public companion object {
        public fun fromProfile(profile: PrecisionProfile): MagneticThresholds = MagneticThresholds(
            magnitudeResidualSuspectFraction = profile.magneticMagnitudeResidualSuspectFraction,
            magnitudeResidualDisturbedFraction = profile.magneticMagnitudeResidualDisturbedFraction,
            inclinationResidualSuspectDeg = profile.inclinationResidualSuspectDeg,
            inclinationResidualDisturbedDeg = profile.inclinationResidualDisturbedDeg,
            stationaryFieldMadSuspectMicroTesla = profile.stationaryFieldMadSuspectMicroTesla,
            stationaryFieldMadDisturbedMicroTesla = profile.stationaryFieldMadDisturbedMicroTesla,
            pipelineDisagreementSuspectDeg = profile.pipelineDisagreementSuspectDeg,
            pipelineDisagreementDisturbedDeg = profile.pipelineDisagreementDisturbedDeg,
        )
    }
}

/**
 * The §16 feature set. `null` means **absent**, which is never zero (§5).
 *
 * [stationaryFieldMadMicroTesla] is absent whenever the motion gates do not indicate
 * stationary; [pipelineAgreementDeg] is absent whenever fewer than two valid independent
 * **active-axis** pipelines exist (§15.1).
 */
public data class MagneticFeatures(
    val relativeMagnitudeResidual: Double?,
    val inclinationResidualDeg: Double?,
    val stationaryFieldMadMicroTesla: Double?,
    val pipelineAgreementDeg: Double?,
    val anyValueNonFinite: Boolean = false,
    val sensorSaturated: Boolean = false,
    val osCalibrationInvalid: Boolean = false,
) {
    val isInvalid: Boolean get() = anyValueNonFinite || sensorSaturated || osCalibrationInvalid

    val nonPipelineFeaturesPresent: Boolean
        get() = relativeMagnitudeResidual != null &&
            inclinationResidualDeg != null &&
            stationaryFieldMadMicroTesla != null

    val allRequiredFeaturesPresent: Boolean
        get() = nonPipelineFeaturesPresent && pipelineAgreementDeg != null
}
