package com.fengshuicompass.headingcore.grade

import com.fengshuicompass.headingcore.config.PrecisionProfile
import kotlinx.serialization.SerialName
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * SPEC.md §8.1.1 — the required build-time grade-reachability analysis.
 *
 * `reportedBound95Deg = instrumentBound95Deg + placementBound95Deg` and the lock ceiling
 * is `usableBound95MaxDeg`, so each placement method has a fixed **instrument budget** of
 * `usableBound95MaxDeg - placementBound95Deg`. Any single uncertainty term larger than
 * that budget makes a Precision Lock arithmetically impossible for that combination, no
 * matter how good the sensor is. §8.1.1 requires a build-time test that computes this for
 * every combination the product claims to support and fails when the spec text claims a
 * grade the constants forbid — "the specific defect class that survives coverage review:
 * two internally consistent sections, contradicted only by arithmetic."
 *
 * The analysis computes the **infimum** of `reportedBound95Deg` for a combination. Every
 * §19 term that can legitimately be zero is taken at zero, so the computed grade is the
 * best case: if a claimed grade is unreachable even here, it is unreachable everywhere.
 */
public object GradeReachability {

    /** Claim vocabulary from `testdata/grade-reachability-claims-v1.json`. */
    public enum class ClaimedGrade {
        PROFESSIONAL, HIGH, USABLE, LOW_CONFIDENCE, INVALID,

        /**
         * The product claims no grade for this combination — used where the placement
         * method has no measured bound in the shipped profile (§29.5, §35).
         */
        NOT_SUPPORTED,
    }

    /** Whether the shipped profile carries a placement bound for the method. */
    public enum class PlacementBoundStatus { CONFIGURED, UNMEASURED }

    @Serializable
    public data class ClaimsDocument(
        val schemaVersion: String,
        val claimsVersion: String,
        val appliesToConfigVersion: String,
        val purpose: String,
        val gradeVocabulary: List<String>,
        val notes: List<String>,
        val combinations: List<Claim>,
    )

    @Serializable
    public data class Claim(
        val id: String,
        val measurementMode: String,
        val placementMethod: String,
        val placementBoundStatus: String,
        val certificationState: String,
        val magneticState: String,
        val claimedMaxGrade: String,
        val claimedLockReachable: Boolean,
        @SerialName("requiresDeviceFloorAtMostDeg")
        val requiresDeviceFloorAtMostDeg: Double? = null,
        val specBasis: String,
    )

    /** One computed reachability result, independent of what was claimed. */
    public data class Reachability(
        /** null when the combination rejects outright and produces no measurement. */
        val minimumReportedBound95Deg: Double?,
        val maxReachableGrade: ClaimedGrade,
        val lockReachable: Boolean,
        /**
         * For a CERTIFIED combination: the largest `deviceFloor95Deg` that still permits a
         * Precision Lock. `<= 0.0` means no real device can lock, because a device floor is
         * strictly positive. null when the notion does not apply.
         */
        val requiredDeviceFloorAtMostDeg: Double?,
        val explanation: String,
    )

    /** A claim contradicted by the arithmetic. §37 rule 12: this is a finding, not an obstacle. */
    public data class ReachabilityFinding(
        val claimId: String,
        val problem: String,
        val claimed: String,
        val computed: String,
    ) {
        override fun toString(): String =
            "[$claimId] $problem -- claimed: $claimed -- computed from the shipped constants: $computed"
    }

    private val json = Json { ignoreUnknownKeys = false; isLenient = false }

    public fun loadClaims(file: File): ClaimsDocument =
        json.decodeFromString(ClaimsDocument.serializer(), file.readText())

    /**
     * The placement term for a method, or null when the shipped profile carries no
     * measured bound for it. §18.5: "Placement uncertainty: finite bound from method.
     * Default to freehand bound; **never zero**."
     */
    public fun placementBound95Deg(method: PlacementMethod, profile: PrecisionProfile): Double? =
        when (method) {
            PlacementMethod.FREEHAND -> profile.flatFreehandPlacementBound95Deg
            PlacementMethod.WALL_FLUSH_FREEHAND -> profile.wallFreehandPlacementBound95Deg
            // §29.5 makes these Phase 5 outputs. The shipped profile has no key for them,
            // and inventing one is exactly the edit §8.1.1 forbids.
            PlacementMethod.NONMAGNETIC_ALIGNMENT_JIG, PlacementMethod.SURVEY_FIXTURE -> null
        }

    /**
     * The §19 interference term, or null when the magnetic state rejects outright.
     * §19: 0 when CLEAN, `suspectInterferenceBound95Deg` when SUSPECT, rejection when
     * UNKNOWN / DISTURBED / INVALID.
     */
    public fun interferenceBound95Deg(state: MagneticState, profile: PrecisionProfile): Double? =
        when (state) {
            MagneticState.CLEAN -> 0.0
            MagneticState.SUSPECT -> profile.suspectInterferenceBound95Deg
            MagneticState.DISTURBED, MagneticState.INVALID, MagneticState.UNKNOWN -> null
        }

    /**
     * The instrument budget for a placement method: `usableBound95MaxDeg - placementBound95Deg`.
     * Null when the placement bound is unmeasured.
     */
    public fun instrumentBudgetDeg(method: PlacementMethod, profile: PrecisionProfile): Double? =
        placementBound95Deg(method, profile)?.let { profile.usableBound95MaxDeg - it }

    /**
     * Computes reachability for one combination.
     *
     * @param certifiedDeviceFloor95Deg the floor a CERTIFIED record would supply. Left null
     *   for the CERTIFIED case so the analysis reports the *required* floor instead of
     *   assuming one — §8.1.1's certification bootstrap makes `deviceFloor95Deg` an output
     *   of the benchmark, not an input to it.
     */
    public fun compute(
        placementMethod: PlacementMethod,
        certificationState: CertificationState,
        magneticState: MagneticState,
        profile: PrecisionProfile,
        certifiedDeviceFloor95Deg: Double? = null,
    ): Reachability {
        val placement = placementBound95Deg(placementMethod, profile)
            ?: return Reachability(
                minimumReportedBound95Deg = null,
                maxReachableGrade = ClaimedGrade.NOT_SUPPORTED,
                lockReachable = false,
                requiredDeviceFloorAtMostDeg = null,
                explanation = "$placementMethod has no measured placement bound in " +
                    "${profile.configVersion}; §29.5 makes it a benchmark output and §18.5 " +
                    "forbids defaulting it to zero, so no grade is computable.",
            )

        val interference = interferenceBound95Deg(magneticState, profile)
            ?: return Reachability(
                minimumReportedBound95Deg = null,
                maxReachableGrade = ClaimedGrade.INVALID,
                lockReachable = false,
                requiredDeviceFloorAtMostDeg = null,
                explanation = "MagneticState $magneticState rejects outright in v1 (§16, §18.5); " +
                    "no measurement is produced, so no grade exists.",
            )

        // §19: baseHeadingBound95Deg = max(present provider term, sampleBound95Deg,
        // deviceFloor95Deg). Provider term and sample dispersion can be arbitrarily small
        // in the best case, so the infimum of the base term is the device floor itself.
        // Every remaining additive §19 term (declination model, location-time sensitivity,
        // reference ambiguity, deviation-correction residual) can legitimately be zero, so
        // the infimum takes them at zero.
        val budget = profile.usableBound95MaxDeg - placement
        val requiredFloor = budget - interference

        return when (certificationState) {
            CertificationState.UNCERTIFIED -> {
                val floor = profile.unknownDeviceFloor95Deg
                val minReported = minOf(180.0, floor + interference + placement)
                val grade = ClaimedGrade.valueOf(
                    qualityGradeForReportedBound(minReported, profile).name
                )
                Reachability(
                    minimumReportedBound95Deg = minReported,
                    maxReachableGrade = grade,
                    lockReachable = minReported <= profile.usableBound95MaxDeg,
                    requiredDeviceFloorAtMostDeg = requiredFloor,
                    explanation = "unknownDeviceFloor95Deg=$floor + interference=$interference " +
                        "+ placement=$placement = $minReported; instrument budget for " +
                        "$placementMethod is $budget.",
                )
            }

            CertificationState.CERTIFIED -> {
                if (certifiedDeviceFloor95Deg != null) {
                    val minReported = minOf(180.0, certifiedDeviceFloor95Deg + interference + placement)
                    val grade = ClaimedGrade.valueOf(
                        qualityGradeForReportedBound(minReported, profile).name
                    )
                    Reachability(
                        minimumReportedBound95Deg = minReported,
                        maxReachableGrade = grade,
                        lockReachable = minReported <= profile.usableBound95MaxDeg,
                        requiredDeviceFloorAtMostDeg = requiredFloor,
                        explanation = "certified floor=$certifiedDeviceFloor95Deg + " +
                            "interference=$interference + placement=$placement = $minReported.",
                    )
                } else {
                    // A device floor is a strictly positive quantity, so a required floor of
                    // zero or less means no certification can make this combination lock.
                    val lockPossible = requiredFloor > 0.0
                    val grade = if (lockPossible) {
                        ClaimedGrade.USABLE
                    } else {
                        // Best case a certified floor can approach is just above zero.
                        val minReported = minOf(180.0, interference + placement)
                        ClaimedGrade.valueOf(qualityGradeForReportedBound(minReported, profile).name)
                    }
                    Reachability(
                        minimumReportedBound95Deg = if (lockPossible) null else minOf(180.0, interference + placement),
                        maxReachableGrade = grade,
                        lockReachable = lockPossible,
                        requiredDeviceFloorAtMostDeg = requiredFloor,
                        explanation = "instrument budget for $placementMethod is $budget; after the " +
                            "$magneticState interference term $interference a certified " +
                            "deviceFloor95Deg must be <= $requiredFloor to lock" +
                            if (lockPossible) "." else ", which no real device floor can satisfy.",
                    )
                }
            }
        }
    }

    /**
     * Checks every claim in [claims] against the arithmetic implied by [profile].
     *
     * @return the findings. Empty means every claimed grade is arithmetically reachable.
     */
    public fun verify(claims: ClaimsDocument, profile: PrecisionProfile): List<ReachabilityFinding> {
        val findings = mutableListOf<ReachabilityFinding>()

        if (claims.appliesToConfigVersion != profile.configVersion) {
            findings += ReachabilityFinding(
                claimId = claims.claimsVersion,
                problem = "The claims document targets a different configuration version, so its " +
                    "rows were never checked against these constants.",
                claimed = "appliesToConfigVersion=${claims.appliesToConfigVersion}",
                computed = "configVersion=${profile.configVersion}",
            )
        }

        for (claim in claims.combinations) {
            val claimedGrade = ClaimedGrade.valueOf(claim.claimedMaxGrade)
            val declaredStatus = PlacementBoundStatus.valueOf(claim.placementBoundStatus)
            val method = PlacementMethod.valueOf(claim.placementMethod)

            val actualStatus =
                if (placementBound95Deg(method, profile) == null) PlacementBoundStatus.UNMEASURED
                else PlacementBoundStatus.CONFIGURED
            if (declaredStatus != actualStatus) {
                findings += ReachabilityFinding(
                    claim.id,
                    "The claim's placement-bound status disagrees with the shipped profile.",
                    "placementBoundStatus=$declaredStatus",
                    "profile ${profile.configVersion} has $actualStatus for $method",
                )
                continue
            }

            if (actualStatus == PlacementBoundStatus.UNMEASURED) {
                if (claimedGrade != ClaimedGrade.NOT_SUPPORTED || claim.claimedLockReachable) {
                    findings += ReachabilityFinding(
                        claim.id,
                        "A placement method with no measured bound may claim no grade and no lock " +
                            "(§29.5; §35 'no grade above USABLE without a measured method').",
                        "claimedMaxGrade=$claimedGrade, claimedLockReachable=${claim.claimedLockReachable}",
                        "placement bound is UNMEASURED for $method",
                    )
                }
                continue
            }

            val certification = CertificationState.valueOf(claim.certificationState)
            val magnetic = MagneticState.valueOf(claim.magneticState)
            val computed = compute(method, certification, magnetic, profile)

            if (claimedGrade.isStrongerThan(computed.maxReachableGrade)) {
                findings += ReachabilityFinding(
                    claim.id,
                    "The claimed maximum grade is arithmetically forbidden by the shipped constants.",
                    "claimedMaxGrade=$claimedGrade (${claim.specBasis})",
                    "maxReachableGrade=${computed.maxReachableGrade}; ${computed.explanation}",
                )
            }

            if (claim.claimedLockReachable != computed.lockReachable) {
                findings += ReachabilityFinding(
                    claim.id,
                    "The claim disagrees with the arithmetic about whether a Precision Lock is " +
                        "reachable at all.",
                    "claimedLockReachable=${claim.claimedLockReachable}",
                    "lockReachable=${computed.lockReachable}; ${computed.explanation}",
                )
            }

            val declaredRequiredFloor = claim.requiresDeviceFloorAtMostDeg
            if (declaredRequiredFloor != null &&
                computed.requiredDeviceFloorAtMostDeg != null &&
                !nearlyEqual(declaredRequiredFloor, computed.requiredDeviceFloorAtMostDeg)
            ) {
                findings += ReachabilityFinding(
                    claim.id,
                    "The device floor the claim says is required does not match the instrument " +
                        "budget the constants leave.",
                    "requiresDeviceFloorAtMostDeg=$declaredRequiredFloor",
                    "required floor=${computed.requiredDeviceFloorAtMostDeg}; ${computed.explanation}",
                )
            }

            if (certification == CertificationState.CERTIFIED &&
                claim.claimedLockReachable &&
                declaredRequiredFloor == null
            ) {
                findings += ReachabilityFinding(
                    claim.id,
                    "A CERTIFIED lock claim must state the device floor it depends on; " +
                        "§8.1.1 makes deviceFloor95Deg an output of the benchmark, not an assumption.",
                    "requiresDeviceFloorAtMostDeg absent",
                    "required floor=${computed.requiredDeviceFloorAtMostDeg}",
                )
            }
        }
        return findings
    }

    private fun nearlyEqual(a: Double, b: Double): Boolean = kotlin.math.abs(a - b) <= 1e-9
}

private fun GradeReachability.ClaimedGrade.isStrongerThan(other: GradeReachability.ClaimedGrade): Boolean =
    ordinal < other.ordinal
