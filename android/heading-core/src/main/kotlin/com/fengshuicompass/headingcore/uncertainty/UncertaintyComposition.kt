package com.fengshuicompass.headingcore.uncertainty

import com.fengshuicompass.headingcore.model.GradeLimitingFactor
import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.RejectionReason
import kotlin.math.min

/**
 * SPEC.md §19 — uncertainty composition producing **both** bounds.
 *
 * `instrumentBound95Deg` says how well the pipeline knows where the *device axis* points.
 * `reportedBound95Deg` says how well the app knows where the *building plane* points; it is
 * what the practitioner sees, what drives classification, and what determines the grade.
 * **Never display `instrumentBound95Deg` as the measurement uncertainty** — it omits the
 * largest term (failure mode 30).
 *
 * The asymmetry in the formula is deliberate and is a modelling choice, not a derivation: the
 * three base terms combine with `max` because they estimate the *same* quantity; the rest add
 * because they are different, additive error sources. §19.1 is why the result carries
 * `CANDIDATE` until held-out coverage exists for the exact certification key.
 *
 * A missing provider error is **absent, never 0 deg evidence** (§19, failure mode 28).
 * Absence is modelled as `null` throughout; the `max` is taken over present values only.
 */
public object UncertaintyComposition {

    /** §19: both bounds are capped at 180 deg. */
    public const val MAX_BOUND_DEG: Double = 180.0

    /** §19: the fixed precedence for a **non-numeric policy ceiling** that lowers the grade. */
    public val policyCeilingPrecedence: List<GradeLimitingFactor> = listOf(
        GradeLimitingFactor.CERTIFICATION_CEILING,
        GradeLimitingFactor.SPACE_WEATHER,
        GradeLimitingFactor.CHARGING_STATE,
    )

    /**
     * §19: `0` when `CLEAN`, the configured penalty when `SUSPECT`, else rejection.
     *
     * §8.1.1 row 3: with the candidate constants the `SUSPECT` term alone exceeds the freehand
     * instrument budget, so `SUSPECT` prevents a freehand lock outright rather than merely
     * capping the grade. That consequence is arithmetic, not a special case here.
     */
    public fun interferenceBound95Deg(
        magneticState: MagneticState,
        suspectInterferenceBound95Deg: Double,
    ): Double = when (magneticState) {
        MagneticState.CLEAN -> 0.0
        MagneticState.SUSPECT -> suspectInterferenceBound95Deg
        MagneticState.DISTURBED ->
            throw InterferenceRejectionException(RejectionReason.MAGNETIC_FIELD_DISTURBED)
        MagneticState.INVALID ->
            throw InterferenceRejectionException(RejectionReason.MAGNETIC_CALIBRATION_INVALID)
        MagneticState.UNKNOWN ->
            throw InterferenceRejectionException(RejectionReason.MAGNETIC_FIELD_UNKNOWN)
    }

    /**
     * §19 composition.
     *
     * [activePolicyCeilings] names non-numeric ceilings currently lowering the grade — a
     * certification ceiling, `PROFESSIONAL_SUPPRESSED` space weather, active wireless
     * charging. They do not change either bound; they only take precedence in
     * `gradeLimitedBy`, because a numeric term the user could act on is useless advice when a
     * policy is the real ceiling.
     */
    public fun composeBounds(
        terms: UncertaintyTerms,
        activePolicyCeilings: Set<GradeLimitingFactor> = emptySet(),
    ): BoundComposition {
        val baseCandidates = buildList {
            add(terms.sampleBound95Deg to GradeLimitingFactor.SAMPLE_DISPERSION)
            add(terms.deviceFloor95Deg to GradeLimitingFactor.DEVICE_FLOOR)
            terms.providerReportedBoundTermDeg?.let {
                add(it to GradeLimitingFactor.PROVIDER_ERROR)
            }
        }
        val base = baseCandidates.maxOf { it.first }
        val declinationTerm = terms.declinationModelBound95Deg ?: 0.0

        val instrument = min(
            MAX_BOUND_DEG,
            base +
                declinationTerm +
                terms.locationTimeSensitivityBound95Deg +
                terms.referenceAmbiguityBound95Deg +
                terms.deviationCorrectionResidualBound95Deg +
                terms.interferenceBound95Deg,
        )
        val reported = min(MAX_BOUND_DEG, instrument + terms.placementBound95Deg)

        return BoundComposition(
            baseHeadingBound95Deg = base,
            instrumentBound95Deg = instrument,
            reportedBound95Deg = reported,
            gradeLimitedBy = gradeLimitedBy(terms, baseCandidates, base, activePolicyCeilings),
        )
    }

    private fun gradeLimitedBy(
        terms: UncertaintyTerms,
        baseCandidates: List<Pair<Double, GradeLimitingFactor>>,
        base: Double,
        activePolicyCeilings: Set<GradeLimitingFactor>,
    ): GradeLimitingFactor {
        policyCeilingPrecedence.firstOrNull { it in activePolicyCeilings }?.let { return it }

        // Only the base term that actually won the `max` contributes to the sum, so only it
        // can be the limiting one among the three.
        val contributing = buildList {
            addAll(baseCandidates.filter { it.first == base })
            add(terms.placementBound95Deg to GradeLimitingFactor.PLACEMENT_UNCERTAINTY)
            terms.declinationModelBound95Deg?.let {
                add(it to GradeLimitingFactor.DECLINATION_MODEL)
            }
            add(
                terms.locationTimeSensitivityBound95Deg to
                    GradeLimitingFactor.LOCATION_TIME_SENSITIVITY
            )
            add(terms.referenceAmbiguityBound95Deg to GradeLimitingFactor.REFERENCE_AMBIGUITY)
            add(
                terms.deviationCorrectionResidualBound95Deg to
                    GradeLimitingFactor.DEVIATION_PROFILE_RESIDUAL
            )
            add(terms.interferenceBound95Deg to GradeLimitingFactor.INTERFERENCE_PENALTY)
        }

        val largest = contributing.maxOf { it.first }
        if (largest <= 0.0) return GradeLimitingFactor.NONE
        // Exact ties resolve by stable enum order, so two runtimes cannot disagree about which
        // of two equal terms is named.
        return contributing.filter { it.first == largest }.minByOrNull { it.second.ordinal }!!.second
    }
}

/** Every §19 input term. `null` is **absent**; `0.0` is a measured zero. */
public data class UncertaintyTerms(
    /**
     * Present only for provider/mode paths exposing a documented degree error. iOS wall and
     * Google FOP wall expose none for their outward-normal projection, and FOP's display-top
     * scalar error MUST NOT enter a wall bound (R61).
     */
    val providerReportedBoundTermDeg: Double?,
    /**
     * P95 of residuals over **all** accepted samples about the circular mean (§15, §19). A
     * dispersion floor, not an error estimate: it detects an unsteady hold and can never
     * detect a steady wrong answer.
     */
    val sampleBound95Deg: Double,
    /** The certified floor for the exact §24 key, else `unknownDeviceFloor95Deg`. */
    val deviceFloor95Deg: Double,
    /** The configured freehand bound for the mode, or a repeatability-tested method bound. */
    val placementBound95Deg: Double,
    /** `0` when the deviation-correction state is `NONE` (§19.3, the v1 default). */
    val deviationCorrectionResidualBound95Deg: Double = 0.0,
    /**
     * `boundFromSigma(model.declinationSigma1Deg)`, only when the app performs or may need the
     * magnetic->true conversion. Absent where the provider owns it.
     */
    val declinationModelBound95Deg: Double? = null,
    /** Worst declination change over accepted position/altitude/time uncertainty. */
    val locationTimeSensitivityBound95Deg: Double = 0.0,
    /** From the §11 `ReferenceResolutionResult`; `0` when verified. */
    val referenceAmbiguityBound95Deg: Double = 0.0,
    /** From §19's interference rule. */
    val interferenceBound95Deg: Double = 0.0,
) {
    init {
        mapOf(
            "sampleBound95Deg" to sampleBound95Deg,
            "deviceFloor95Deg" to deviceFloor95Deg,
            "deviationCorrectionResidualBound95Deg" to deviationCorrectionResidualBound95Deg,
            "locationTimeSensitivityBound95Deg" to locationTimeSensitivityBound95Deg,
            "referenceAmbiguityBound95Deg" to referenceAmbiguityBound95Deg,
            "interferenceBound95Deg" to interferenceBound95Deg,
            "placementBound95Deg" to placementBound95Deg,
        ).forEach { (name, value) ->
            require(value.isFinite() && value >= 0.0) {
                "$name must be a finite, non-negative bound, got $value"
            }
        }
        mapOf(
            "providerReportedBoundTermDeg" to providerReportedBoundTermDeg,
            "declinationModelBound95Deg" to declinationModelBound95Deg,
        ).forEach { (name, value) ->
            require(value == null || (value.isFinite() && value >= 0.0)) {
                "$name is either absent (null) or a finite non-negative bound, got $value"
            }
        }
        require(placementBound95Deg > 0.0) {
            "§18.5: placement uncertainty is a finite bound from the method and is never " +
                "zero. A zero placement term is how a build reaches Professional freehand " +
                "(§20), which is a certification failure, not a feature."
        }
    }
}

/** The composed result: both bounds, the base term, and what limited the grade. */
public data class BoundComposition(
    val baseHeadingBound95Deg: Double,
    val instrumentBound95Deg: Double,
    val reportedBound95Deg: Double,
    val gradeLimitedBy: GradeLimitingFactor,
)

/**
 * §19: `UNKNOWN` / `DISTURBED` / `INVALID` are a rejection, not a wider bound.
 *
 * Widening a bound to absorb an unclassifiable field would convert a refusal into a
 * confidently-labelled measurement, which is the behaviour §1 exists to prevent.
 */
public class InterferenceRejectionException(public val reason: RejectionReason) :
    IllegalStateException("magnetic state rejects the measurement: ${reason.wire}")
