package com.fengshuicompass.fengshuicore

import com.fengshuicompass.headingcore.math.CircularMath
import kotlin.math.floor
import kotlin.math.min

/**
 * SPEC.md §21 — the Feng Shui direction engine: geometry, classification, straddle.
 *
 * The engine consumes a full-precision canonical heading and its bound. It never rounds before
 * classification (failure mode 7: `337.49` moves a sector), and boundaries are derived from
 * the ruleset rather than hand-typed (R65).
 *
 * §21.3 sets the honest expectation this classifier is built around: a sector is `15` deg
 * wide, so a `reportedBound95Deg` above `7.5` deg *guarantees* a two-sector straddle
 * regardless of the point estimate. **Straddles are the common case**, which is why
 * [FengShuiClassification] returns the full set rather than a primary with a footnote.
 */
public object FengShuiClassifier {

    /**
     * §21.1's derived index — the only place a boundary is computed.
     *
     * `floor(normalize360(h - firstSectorCenterDeg + sectorWidthDeg/2) / sectorWidthDeg) mod
     * sectorCount`. For the default ruleset this puts boundaries at `7.5 + 15k`, so `352.5`
     * separates 壬 and 子. Half-open `[start, end)`: a heading exactly on a boundary belongs
     * to the sector that boundary starts.
     */
    public fun sectorIndex(headingDeg: Double, ruleSet: FengShuiRuleSet): Int {
        val offset = CircularMath.normalize360(
            headingDeg - ruleSet.firstSectorCenterDeg + ruleSet.sectorWidthDeg / 2.0
        )
        return Math.floorMod(
            floor(offset / ruleSet.sectorWidthDeg).toInt(),
            ruleSet.sectorCount,
        )
    }

    /**
     * How far past its own start boundary [headingDeg] sits, in `[0, sectorWidthDeg)`.
     *
     * Derived from the *same* wrapped quantity [sectorIndex] uses, so the index and the offset
     * cannot disagree by a rounding bit at a boundary.
     */
    public fun offsetWithinSectorDeg(headingDeg: Double, ruleSet: FengShuiRuleSet): Double {
        val wrapped = CircularMath.normalize360(
            headingDeg - ruleSet.firstSectorCenterDeg + ruleSet.sectorWidthDeg / 2.0
        )
        return wrapped - floor(wrapped / ruleSet.sectorWidthDeg) * ruleSet.sectorWidthDeg
    }

    /**
     * §21.2's final, explicit, recorded reference step.
     *
     * `TRUE` uses the canonical true heading; `MAGNETIC` derives `trueHeading - declination`
     * from that *same* canonical measurement rather than substituting an unvalidated magnetic
     * path. `needleOffsetDeg` expresses doctrinal plate conventions and is a declared property
     * of a named ruleset — never a user slider, never a correction for measurement error.
     */
    public fun classificationHeadingDeg(
        trueHeadingDeg: Double,
        declinationDeg: Double,
        ruleSet: FengShuiRuleSet,
    ): Double {
        val base = when (ruleSet.referenceSelection) {
            "TRUE" -> trueHeadingDeg
            "MAGNETIC" -> CircularMath.normalize360(trueHeadingDeg - declinationDeg)
            else -> throw IllegalArgumentException(
                "unknown referenceSelection ${ruleSet.referenceSelection}"
            )
        }
        return CircularMath.normalize360(base + ruleSet.needleOffsetDeg)
    }

    /**
     * §21.4: every ruleset sector intersecting the circular interval, in azimuth order.
     *
     * The interval is the **closed** `[h - bound, h + bound]` while sectors are half-open, so
     * an interval endpoint landing exactly on a boundary includes the sector that boundary
     * starts. That asymmetry is deliberately conservative: naming one fewer mountain is a
     * false-precision failure, naming one more is not.
     *
     * The count comes from the **arc length**, not from walking forward until the end index is
     * met. At a bound approaching `180` deg the interval wraps almost the whole circle and both
     * endpoints land in the *same* sector; a walk-until-equal would then report one sector for
     * an interval covering all 24 — a single-mountain claim from a measurement that
     * discriminates nothing, which is the exact false-precision failure §21.3 warns about.
     */
    public fun straddleIndices(
        classificationHeadingDeg: Double,
        reportedBound95Deg: Double,
        ruleSet: FengShuiRuleSet,
    ): List<Int> {
        require(reportedBound95Deg.isFinite() && reportedBound95Deg >= 0.0) {
            "reportedBound95Deg must be a finite, non-negative bound, got $reportedBound95Deg"
        }
        // §21.4: report that no classification is possible rather than listing all 24.
        if (2.0 * reportedBound95Deg >= 360.0) return emptyList()

        val low = classificationHeadingDeg - reportedBound95Deg
        val startIndex = sectorIndex(low, ruleSet)
        val offset = offsetWithinSectorDeg(low, ruleSet)
        val spanned = min(
            ruleSet.sectorCount,
            1 + floor((offset + 2.0 * reportedBound95Deg) / ruleSet.sectorWidthDeg).toInt(),
        )
        return (0 until spanned).map { Math.floorMod(startIndex + it, ruleSet.sectorCount) }
    }

    /**
     * Signed circular difference from the **nearest** sector boundary to the heading.
     *
     * Positive means the heading lies clockwise of that boundary. Magnitude never exceeds
     * `sectorWidthDeg / 2`. Reported so a practitioner can see how close to a boundary a
     * result sits even when the bound happens not to straddle.
     */
    public fun signedOffsetFromSectorBoundaryDeg(
        classificationHeadingDeg: Double,
        ruleSet: FengShuiRuleSet,
    ): Double {
        val index = sectorIndex(classificationHeadingDeg, ruleSet)
        val candidates = listOf(
            ruleSet.derivedSectorStartDeg(index),
            ruleSet.derivedSectorStartDeg(Math.floorMod(index + 1, ruleSet.sectorCount)),
        )
        val nearest = candidates.minByOrNull {
            CircularMath.absoluteCircularDifferenceDeg(classificationHeadingDeg, it)
        }!!
        return CircularMath.shortestSignedDifferenceDeg(classificationHeadingDeg, nearest)
    }

    /**
     * §21: classify the whole circular bound interval, never the point estimate alone.
     *
     * [reportedBound95Deg] is the **total** bound — instrument plus placement, including any
     * `referenceAmbiguityBound95Deg`. §21.2: subtracting declination for a magnetic ruleset
     * MUST NOT zero or remove the ambiguity term, because if the provider secretly emitted the
     * other reference the derived magnetic point is wrong by `|d|` too. This function
     * therefore never touches the bound it is handed.
     */
    public fun classify(
        trueHeadingDeg: Double,
        declinationDeg: Double,
        reportedBound95Deg: Double,
        ruleSet: FengShuiRuleSet,
    ): FengShuiClassification {
        val heading = classificationHeadingDeg(trueHeadingDeg, declinationDeg, ruleSet)
        val indices = straddleIndices(heading, reportedBound95Deg, ruleSet)
        val offset = signedOffsetFromSectorBoundaryDeg(heading, ruleSet)
        if (indices.isEmpty()) {
            return FengShuiClassification(
                ruleSetVersion = ruleSet.ruleSetVersion,
                referenceSelection = ruleSet.referenceSelection,
                classificationHeadingDeg = heading,
                primarySector = null,
                possibleSectors = emptyList(),
                possibleSectorIndices = emptyList(),
                boundaryStraddled = true,
                signedOffsetFromSectorBoundaryDeg = offset,
                classificationPossible = false,
            )
        }
        val primaryIndex = sectorIndex(heading, ruleSet)
        return FengShuiClassification(
            ruleSetVersion = ruleSet.ruleSetVersion,
            referenceSelection = ruleSet.referenceSelection,
            classificationHeadingDeg = heading,
            primarySector = ruleSet.sectors[primaryIndex].name,
            possibleSectors = indices.map { ruleSet.sectors[it].name },
            possibleSectorIndices = indices,
            boundaryStraddled = indices.size > 1,
            signedOffsetFromSectorBoundaryDeg = offset,
            classificationPossible = true,
        )
    }
}

/** §5.1's `classification:` block for one measurement. */
public data class FengShuiClassification(
    val ruleSetVersion: String,
    val referenceSelection: String,
    val classificationHeadingDeg: Double,
    val primarySector: String?,
    val possibleSectors: List<String>,
    val possibleSectorIndices: List<Int>,
    val boundaryStraddled: Boolean,
    val signedOffsetFromSectorBoundaryDeg: Double,
    val classificationPossible: Boolean,
)
