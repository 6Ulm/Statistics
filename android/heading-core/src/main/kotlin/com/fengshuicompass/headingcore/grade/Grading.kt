package com.fengshuicompass.headingcore.grade

import com.fengshuicompass.headingcore.config.PrecisionProfile

/**
 * SPEC.md §6 vocabulary re-exported into this package.
 *
 * Phase 0 declared `PlacementMethod` and `MagneticState` here because §8.1.1's reachability
 * analysis was the only consumer. Phase 1 gives §6 a single home in
 * `com.fengshuicompass.headingcore.model`, and §6 allows exactly one vocabulary — so these are
 * aliases to that home rather than second declarations. A second enum with the same case names
 * is how two parts of one runtime end up disagreeing about a wire value.
 */
public typealias PlacementMethod = com.fengshuicompass.headingcore.model.PlacementMethod

/** See [PlacementMethod]: an alias to the single §6 declaration, not a second one. */
public typealias MagneticState = com.fengshuicompass.headingcore.model.MagneticState

/**
 * SPEC.md §6 `QualityGrade`, plus the two sentinel values the §8.1.1 reachability
 * analysis needs to express a claim about a combination.
 *
 * Only the enum cases named in §6 may appear in telemetry or on a wire; [NOT_SUPPORTED]
 * is an analysis-side claim vocabulary value from
 * `testdata/grade-reachability-claims-v1.json` and is never an emitted grade.
 */
public enum class QualityGrade {
    PROFESSIONAL,
    HIGH,
    USABLE,
    LOW_CONFIDENCE,
    INVALID,
    ;

    /** Higher ordinal means a weaker grade, so a smaller ordinal is a stronger claim. */
    public fun isStrongerThan(other: QualityGrade): Boolean = ordinal < other.ordinal
}

/**
 * Whether a §24 `CertificationRecord` exists for the exact certification key.
 *
 * A record exists only for `CALIBRATED` (§24); a miss on any key component means
 * [UNCERTIFIED], and the engine then uses `unknownDeviceFloor95Deg`.
 */
public enum class CertificationState {
    UNCERTIFIED,
    CERTIFIED,
}

/**
 * SPEC.md §20: grades come from `reportedBound95Deg`, on explicit half-open intervals so
 * the function is total. Grading on `instrumentBound95Deg` would advertise precision the
 * practitioner cannot physically realize (failure mode 30).
 *
 * The "Also required" column of §20 (clean field, fresh location, certified device,
 * repeatability-tested alignment method, ...) is enforced by the §18.5 lock gates in
 * Phase 3; this function answers only the bound-to-grade half, which is what the §8.1.1
 * arithmetic-reachability analysis needs.
 */
public fun qualityGradeForReportedBound(
    reportedBound95Deg: Double,
    profile: PrecisionProfile,
): QualityGrade {
    require(reportedBound95Deg.isFinite() && reportedBound95Deg >= 0.0) {
        "reportedBound95Deg must be a finite non-negative angle, got $reportedBound95Deg"
    }
    return when {
        reportedBound95Deg <= profile.professionalBound95MaxDeg -> QualityGrade.PROFESSIONAL
        reportedBound95Deg <= profile.highBound95MaxDeg -> QualityGrade.HIGH
        reportedBound95Deg <= profile.usableBound95MaxDeg -> QualityGrade.USABLE
        reportedBound95Deg <= profile.lowConfidenceBound95MaxDeg -> QualityGrade.LOW_CONFIDENCE
        else -> QualityGrade.INVALID
    }
}
