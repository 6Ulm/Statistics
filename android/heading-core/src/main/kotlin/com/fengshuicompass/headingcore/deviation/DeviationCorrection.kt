package com.fengshuicompass.headingcore.deviation

import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.DeviationCorrectionScope
import com.fengshuicompass.headingcore.model.DeviationCorrectionState
import com.fengshuicompass.headingcore.model.DeviationStructureClass
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.PlacementMethod
import com.fengshuicompass.headingcore.model.ProviderId

/**
 * SPEC.md §19.3 — deviation correction. The v1 production state is fixed to `NONE`.
 *
 * Default production state is `NONE`, `deviationCorrectionDeg = 0.0`,
 * `trueHeadingDeg = uncorrectedTrueHeadingDeg`. This file exists so the *types* are in place —
 * the certification key needs `deviationCorrectionProfileHash` and the bound needs
 * `deviationCorrectionResidualBound95Deg` — while the lookup returns `NONE` by construction.
 *
 * Two rules are enforced here rather than documented:
 *
 * * A `UNIT`-scope profile never produces `CALIBRATED` output. v1's certification database
 *   intentionally does not bind to physical-unit identity, so a per-unit correction cannot be
 *   matched by a runtime lookup.
 * * A correction is applied **exactly once**, after reference resolution and before lock
 *   aggregation. There is one application site, mirroring §11's single `correctionDeg` site.
 */
public object DeviationCorrection {

    /**
     * §24: "literal NONE when correction is disabled". The sentinel is a string, not `null`,
     * because it is a key component and a missing component must not silently match.
     */
    public const val NONE_PROFILE_HASH: String = "NONE"

    /**
     * §7 `DeviationCorrectionProvider.lookup` — v1 always returns no profile.
     *
     * This is not a placeholder: §19.3 fixes the production default at `NONE` and §30.6 gates
     * any promotion on held-out evidence that does not exist. Returning `null` is the correct
     * behaviour, and the signature accepts the live context so a Phase 5 profile can be added
     * without changing call sites.
     */
    @Suppress("UNUSED_PARAMETER")
    public fun lookupDeviationProfile(
        vararg liveContext: Any?,
    ): DeviationCorrectionProfileMetadata? = null

    /**
     * §19.3: apply **exactly once**, after reference resolution, before lock aggregation.
     *
     * With no certified profile the correction is `0.0` and the corrected heading is the
     * uncorrected one — identical numbers with different names, kept as separate fields so the
     * raw uncorrected heading is always retained beside the correction.
     */
    public fun applyDeviationCorrection(
        uncorrectedTrueHeadingDeg: Double,
        profileCorrectionDeg: Double? = null,
        profile: DeviationCorrectionProfileMetadata? = null,
    ): DeviationCorrectionOutcome {
        val uncorrected = CircularMath.normalize360(uncorrectedTrueHeadingDeg)
        if (profile == null || profileCorrectionDeg == null) {
            return DeviationCorrectionOutcome(
                state = DeviationCorrectionState.NONE,
                correctionDeg = 0.0,
                uncorrectedTrueHeadingDeg = uncorrected,
                trueHeadingDeg = uncorrected,
                profileId = null,
                profileHash = NONE_PROFILE_HASH,
                residualBound95Deg = 0.0,
            )
        }
        require(profile.mayProduceCalibratedOutput) {
            "profile ${profile.profileId} has ${profile.scope.wire} scope; §19.3 keeps UNIT " +
                "profiles experimental and forbids them from producing CALIBRATED output"
        }
        require(profileCorrectionDeg.isFinite()) {
            "a deviation correction must be a finite number of degrees"
        }
        return DeviationCorrectionOutcome(
            state = DeviationCorrectionState.CERTIFIED_PROFILE,
            correctionDeg = profileCorrectionDeg,
            uncorrectedTrueHeadingDeg = uncorrected,
            trueHeadingDeg = CircularMath.normalize360(uncorrected + profileCorrectionDeg),
            profileId = profile.profileId,
            profileHash = profile.profileHash,
            residualBound95Deg = profile.heldOutResidualBound95Deg,
        )
    }
}

/**
 * §5 `DeviationCorrectionProfileMetadata`.
 *
 * Every scope-defining field is required. §19.3: a profile's scope is explicit — unit or model
 * class, provider path, mode, placement, OS/provider range, model/config hashes.
 */
public data class DeviationCorrectionProfileMetadata(
    val profileId: String,
    val profileHash: String,
    val scope: DeviationCorrectionScope,
    val structureClass: DeviationStructureClass,
    val correctionMethodId: String,
    val measurementMode: MeasurementMode,
    val placementMethod: PlacementMethod,
    val providerId: ProviderId,
    val coveredProviderRuntimeIdentities: List<String>,
    val coveredOsBuildIdentities: List<String>,
    val geomagneticModelId: String,
    val geomagneticCoefficientHash: String,
    val precisionConfigHash: String,
    val heldOutResidualBound95Deg: Double,
    val trainingEvidenceId: String,
    val heldOutEvidenceId: String,
) {
    init {
        require(heldOutResidualBound95Deg.isFinite() && heldOutResidualBound95Deg >= 0.0) {
            "heldOutResidualBound95Deg must be a finite, non-negative bound"
        }
        require(profileHash != DeviationCorrection.NONE_PROFILE_HASH) {
            "a real profile may not use the NONE sentinel as its hash"
        }
    }

    /** §19.3/§30.6: only a `MODEL_CLASS` profile can appear in a `CALIBRATED` record. */
    val mayProduceCalibratedOutput: Boolean
        get() = scope == DeviationCorrectionScope.MODEL_CLASS
}

/** The result of the single application site. */
public data class DeviationCorrectionOutcome(
    val state: DeviationCorrectionState,
    val correctionDeg: Double,
    val uncorrectedTrueHeadingDeg: Double,
    val trueHeadingDeg: Double,
    val profileId: String?,
    val profileHash: String,
    val residualBound95Deg: Double,
)
