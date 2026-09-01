package com.fengshuicompass.headingcore.reference

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.ReferenceAxis
import com.fengshuicompass.headingcore.model.ReferenceMagneticPrecheckState
import com.fengshuicompass.headingcore.model.ReferenceResolutionMethod
import com.fengshuicompass.headingcore.model.ResolvedReference
import com.fengshuicompass.headingcore.model.referenceAxisForMode
import kotlin.math.abs

/**
 * SPEC.md §11 — north-reference resolution for the Google FOP path.
 *
 * Google FOP exposes the same ambiguous contract through both its scalar heading and its
 * attitude frame: true north when declination is available, magnetic north otherwise, with no
 * per-sample flag. This resolves that ambiguity **without replacing Google's fusion** and
 * without using a geometrically ill-conditioned axis: both hypotheses are formed from the
 * *same physical reference axis* of the *active* measurement mode.
 *
 * Two rules here prevent Critical failures:
 *
 * * `correctionDeg` is the single Google magnetic->true correction site and is exactly `0.0`
 *   or `+declinationDeg`. Double application yields a plausible but catastrophic
 *   `2 x declination` error (failure mode 21), which §30.5 hunts for by name.
 * * The resolver never writes or overwrites `reportedBound95Deg`. The ambiguity branch emits
 *   one uncertainty term, which flows into §19 composition.
 *
 * iOS does not use this test, and `AND-RV` owns the conversion itself. Those contracts are
 * represented by the explicit constructors at the bottom rather than by running the hypothesis
 * test with fabricated inputs (R51: N/A is not failure, and is not fabricated evidence).
 */
public object ReferenceResolution {

    /** §11's hypothesis test, per active measurement mode and stable window. */
    public fun resolveGoogleReference(
        hypotheses: GoogleReferenceHypotheses,
        thresholds: ReferenceResolutionThresholds,
    ): ReferenceResolutionResult {
        // Eligibility first: without fresh evidence, or with a precheck that is not
        // CLEAN_FOR_REFERENCE, the result is UNVERIFIED and the residuals are null — the
        // engine does not manufacture a Google pipeline reference in order to compute the
        // evidence that would have been needed to resolve it (R59).
        if (!hypotheses.evidenceIsEligible ||
            hypotheses.precheckState != ReferenceMagneticPrecheckState.CLEAN_FOR_REFERENCE
        ) {
            return unresolved(hypotheses, null, null)
        }

        val gAxis = CircularMath.normalize360(hypotheses.gAxisDeg)
        val mAxis = CircularMath.normalize360(hypotheses.mAxisDeg)
        val declination = hypotheses.declinationDeg
        val tAxis = CircularMath.normalize360(mAxis + declination)

        val residualTrue = CircularMath.absoluteCircularDifferenceDeg(gAxis, tAxis)
        val residualMagnetic = CircularMath.absoluteCircularDifferenceDeg(gAxis, mAxis)

        fun result(
            resolved: ResolvedReference,
            correctionDeg: Double,
            ambiguityDeg: Double,
            canonicalTrueHeadingDeg: Double?,
        ) = ReferenceResolutionResult(
            measurementMode = hypotheses.measurementMode,
            referenceAxis = referenceAxisForMode(hypotheses.measurementMode),
            resolvedReference = resolved,
            referenceResolutionMethod =
                ReferenceResolutionMethod.FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK,
            declinationDeg = declination,
            correctionDeg = correctionDeg,
            referenceAmbiguityBound95Deg = ambiguityDeg,
            geomagneticModelId = hypotheses.geomagneticModelId,
            sourceWindowStartMonotonicNs = hypotheses.sourceWindowStartMonotonicNs,
            sourceWindowEndMonotonicNs = hypotheses.sourceWindowEndMonotonicNs,
            referenceHypothesisResidualTrueDeg = residualTrue,
            referenceHypothesisResidualMagneticDeg = residualMagnetic,
            canonicalTrueHeadingDeg = canonicalTrueHeadingDeg,
        )

        // Google was already emitting true north: use it exactly, correct nothing.
        if (residualTrue <= thresholds.providerCrossCheckMaxDeg &&
            (residualMagnetic - residualTrue) >= thresholds.referenceSeparationMarginDeg
        ) {
            return result(ResolvedReference.TRUE_VERIFIED, 0.0, 0.0, gAxis)
        }

        // Google was emitting magnetic north: apply declination exactly once, here.
        if (residualMagnetic <= thresholds.providerCrossCheckMaxDeg &&
            (residualTrue - residualMagnetic) >= thresholds.referenceSeparationMarginDeg
        ) {
            return result(
                ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
                declination,
                0.0,
                CircularMath.normalize360(gAxis + declination),
            )
        }

        // The hypotheses are inseparable because |d| is small; carry |d| as an explicit term
        // rather than picking a branch. §21.2 keeps this term after the reference transform.
        if (abs(declination) <= thresholds.smallDeclinationAmbiguityMaxDeg) {
            return result(
                ResolvedReference.TRUE_WITH_AMBIGUITY_BOUND,
                0.0,
                abs(declination),
                gAxis,
            )
        }

        return unresolved(hypotheses, residualTrue, residualMagnetic)
    }

    /**
     * iOS flat: valid `CLHeading.trueHeading` is explicit (`PROVIDER_CONTRACT_EXPLICIT`).
     *
     * No hypothesis test, no ambiguity term, no correction — Apple owns the conversion.
     * Running the Google resolver here would fabricate evidence (R51).
     */
    public fun appleProviderContractReferenceResolution(
        mode: MeasurementMode,
        trueHeadingDeg: Double,
        declinationDeg: Double,
        geomagneticModelId: GeomagneticModelId,
        sourceWindowStartMonotonicNs: Long,
        sourceWindowEndMonotonicNs: Long,
    ): ReferenceResolutionResult = ReferenceResolutionResult(
        measurementMode = mode,
        referenceAxis = referenceAxisForMode(mode),
        resolvedReference = ResolvedReference.TRUE_VERIFIED,
        referenceResolutionMethod = ReferenceResolutionMethod.PROVIDER_CONTRACT_EXPLICIT,
        declinationDeg = declinationDeg,
        correctionDeg = 0.0,
        referenceAmbiguityBound95Deg = 0.0,
        geomagneticModelId = geomagneticModelId,
        sourceWindowStartMonotonicNs = sourceWindowStartMonotonicNs,
        sourceWindowEndMonotonicNs = sourceWindowEndMonotonicNs,
        canonicalTrueHeadingDeg = CircularMath.normalize360(trueHeadingDeg),
    )

    /**
     * iOS wall: `.xTrueNorthZVertical` is explicit **when that frame is actually active**.
     * §12: the requested frame is an intention; the observed `attitudeReferenceFrame` is the
     * fact. If it is not active the reference is `UNVERIFIED`.
     */
    public fun appleAttitudeFrameReferenceResolution(
        mode: MeasurementMode,
        projectedTrueHeadingDeg: Double,
        declinationDeg: Double,
        geomagneticModelId: GeomagneticModelId,
        sourceWindowStartMonotonicNs: Long,
        sourceWindowEndMonotonicNs: Long,
        frameIsActive: Boolean,
    ): ReferenceResolutionResult = ReferenceResolutionResult(
        measurementMode = mode,
        referenceAxis = referenceAxisForMode(mode),
        resolvedReference =
            if (frameIsActive) ResolvedReference.TRUE_VERIFIED else ResolvedReference.UNVERIFIED,
        referenceResolutionMethod =
            if (frameIsActive) {
                ReferenceResolutionMethod.ATTITUDE_FRAME_EXPLICIT
            } else {
                ReferenceResolutionMethod.NOT_RESOLVED
            },
        declinationDeg = declinationDeg,
        correctionDeg = 0.0,
        referenceAmbiguityBound95Deg = 0.0,
        geomagneticModelId = geomagneticModelId,
        sourceWindowStartMonotonicNs = sourceWindowStartMonotonicNs,
        sourceWindowEndMonotonicNs = sourceWindowEndMonotonicNs,
        canonicalTrueHeadingDeg =
            if (frameIsActive) CircularMath.normalize360(projectedTrueHeadingDeg) else null,
    )

    /**
     * `AND-RV`: the app owns magnetic->true conversion, known by construction.
     *
     * §30.4: `resolvedReference` MUST be `TRUE_CORRECTED_FROM_MAGNETIC` with
     * `APP_APPLIED_DECLINATION`; there is no `TRUE_VERIFIED` here without an independent
     * reference check, and §11's ambiguity rule does not apply. Declination is applied exactly
     * once, here.
     */
    public fun andRvReferenceResolution(
        mode: MeasurementMode,
        magneticAxisHeadingDeg: Double,
        declinationDeg: Double,
        geomagneticModelId: GeomagneticModelId,
        sourceWindowStartMonotonicNs: Long,
        sourceWindowEndMonotonicNs: Long,
    ): ReferenceResolutionResult = ReferenceResolutionResult(
        measurementMode = mode,
        referenceAxis = referenceAxisForMode(mode),
        resolvedReference = ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
        referenceResolutionMethod = ReferenceResolutionMethod.APP_APPLIED_DECLINATION,
        declinationDeg = declinationDeg,
        correctionDeg = declinationDeg,
        referenceAmbiguityBound95Deg = 0.0,
        geomagneticModelId = geomagneticModelId,
        sourceWindowStartMonotonicNs = sourceWindowStartMonotonicNs,
        sourceWindowEndMonotonicNs = sourceWindowEndMonotonicNs,
        canonicalTrueHeadingDeg =
            CircularMath.normalize360(magneticAxisHeadingDeg + declinationDeg),
    )

    private fun unresolved(
        hypotheses: GoogleReferenceHypotheses,
        residualTrueDeg: Double?,
        residualMagneticDeg: Double?,
    ) = ReferenceResolutionResult(
        measurementMode = hypotheses.measurementMode,
        referenceAxis = referenceAxisForMode(hypotheses.measurementMode),
        resolvedReference = ResolvedReference.UNVERIFIED,
        referenceResolutionMethod = ReferenceResolutionMethod.NOT_RESOLVED,
        declinationDeg = hypotheses.declinationDeg,
        correctionDeg = 0.0,
        referenceAmbiguityBound95Deg = 0.0,
        geomagneticModelId = hypotheses.geomagneticModelId,
        sourceWindowStartMonotonicNs = hypotheses.sourceWindowStartMonotonicNs,
        sourceWindowEndMonotonicNs = hypotheses.sourceWindowEndMonotonicNs,
        referenceHypothesisResidualTrueDeg = residualTrueDeg,
        referenceHypothesisResidualMagneticDeg = residualMagneticDeg,
        canonicalTrueHeadingDeg = null,
    )
}

/**
 * The §8 keys the resolver reads.
 *
 * §8.1 requires `referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg`: since
 * `rMag - rTrue <= abs(d)`, a margin above the ambiguity allowance would create a declination
 * dead band that always resolves `UNVERIFIED` with no visible cause.
 */
public data class ReferenceResolutionThresholds(
    val providerCrossCheckMaxDeg: Double,
    val referenceSeparationMarginDeg: Double,
    val smallDeclinationAmbiguityMaxDeg: Double,
) {
    public companion object {
        public fun fromProfile(profile: PrecisionProfile): ReferenceResolutionThresholds =
            ReferenceResolutionThresholds(
                providerCrossCheckMaxDeg = profile.providerCrossCheckMaxDeg,
                referenceSeparationMarginDeg = profile.referenceSeparationMarginDeg,
                smallDeclinationAmbiguityMaxDeg = profile.smallDeclinationAmbiguityMaxDeg,
            )
    }
}

/**
 * Everything §11 needs for one active-mode stable window.
 *
 * [gAxisDeg] is the aggregated Google bearing **of the active mode's reference axis**;
 * [mAxisDeg] is the synchronized diagnostic magnetic-north bearing of that *same* physical
 * axis, derived through a platform magnetic orientation path — never raw magnetometer X/Y, and
 * never an axis whose horizontal projection is singular.
 */
public data class GoogleReferenceHypotheses(
    val measurementMode: MeasurementMode,
    val gAxisDeg: Double,
    val mAxisDeg: Double,
    val declinationDeg: Double,
    val precheckState: ReferenceMagneticPrecheckState,
    val geomagneticModelId: GeomagneticModelId,
    val sourceWindowStartMonotonicNs: Long,
    val sourceWindowEndMonotonicNs: Long,
    /**
     * Fresh location/model evidence, valid synchronized source timestamps and a valid
     * diagnostic magnetic orientation. The test is ineligible without all of them.
     */
    val evidenceIsEligible: Boolean = true,
) {
    init {
        require(gAxisDeg.isFinite() && mAxisDeg.isFinite() && declinationDeg.isFinite()) {
            "GoogleReferenceHypotheses angles must be finite"
        }
        require(sourceWindowEndMonotonicNs >= sourceWindowStartMonotonicNs) {
            "the source window must not end before it starts"
        }
    }
}

/**
 * §5 `ReferenceResolutionResult`, bound to its mode, axis and source window.
 *
 * "A flat result is not reusable for a wall pose or vice versa." The mode and axis are fields
 * precisely so a caller cannot transfer one.
 */
public data class ReferenceResolutionResult(
    val measurementMode: MeasurementMode,
    val referenceAxis: ReferenceAxis,
    val resolvedReference: ResolvedReference,
    val referenceResolutionMethod: ReferenceResolutionMethod,
    val declinationDeg: Double,
    val correctionDeg: Double,
    val referenceAmbiguityBound95Deg: Double,
    val geomagneticModelId: GeomagneticModelId,
    val sourceWindowStartMonotonicNs: Long,
    val sourceWindowEndMonotonicNs: Long,
    val referenceHypothesisResidualTrueDeg: Double? = null,
    val referenceHypothesisResidualMagneticDeg: Double? = null,
    /** The canonical true-north bearing of the active axis, or `null` when unresolved. */
    val canonicalTrueHeadingDeg: Double? = null,
) {
    init {
        require(correctionDeg == 0.0 || correctionDeg == declinationDeg) {
            "§11: correctionDeg is exactly 0.0 or +declinationDeg — the single Google " +
                "magnetic->true correction site. Got $correctionDeg with declination " +
                "$declinationDeg (failure mode 21)."
        }
        require(referenceAmbiguityBound95Deg >= 0.0) {
            "referenceAmbiguityBound95Deg must be non-negative"
        }
    }

    val isTrueReferenced: Boolean
        get() = resolvedReference in setOf(
            ResolvedReference.TRUE_VERIFIED,
            ResolvedReference.TRUE_CORRECTED_FROM_MAGNETIC,
            ResolvedReference.TRUE_WITH_AMBIGUITY_BOUND,
        )
}
