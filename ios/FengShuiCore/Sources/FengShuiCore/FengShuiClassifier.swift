import Foundation
import HeadingCore

/// SPEC.md §21 — the Feng Shui direction engine: geometry, classification, straddle.
///
/// The engine consumes a full-precision canonical heading and its bound. It never rounds before
/// classification (failure mode 7: `337.49°` moves a sector), and boundaries are derived from
/// the ruleset rather than hand-typed (R65).
///
/// §21.3 sets the honest expectation this classifier is built around: a sector is `15°` wide, so
/// a `reportedBound95Deg` above `7.5°` *guarantees* a two-sector straddle regardless of the
/// point estimate. **Straddles are the common case**, which is why `FengShuiClassification`
/// returns the full set rather than a primary with a footnote.
///
/// > Warning: this file has never been compiled — see `docs/IMPLEMENTATION_NOTES.md` D-3.
public enum FengShuiClassifier {

    public enum ClassifierError: Error, CustomStringConvertible {
        case invalidBound(Double)
        case unknownReferenceSelection(String)

        public var description: String {
            switch self {
            case .invalidBound(let value):
                return "reportedBound95Deg must be a finite, non-negative bound, got \(value)"
            case .unknownReferenceSelection(let value):
                return "unknown referenceSelection \(value)"
            }
        }
    }

    /// §21.1's derived index — the only place a boundary is computed.
    ///
    /// `floor(normalize360(h - firstSectorCenterDeg + sectorWidthDeg/2) / sectorWidthDeg) mod
    /// sectorCount`. For the default ruleset this puts boundaries at `7.5° + 15k`, so `352.5°`
    /// separates 壬 and 子. Half-open `[start, end)`: a heading exactly on a boundary belongs to
    /// the sector that boundary starts.
    public static func sectorIndex(
        _ headingDeg: Double,
        _ ruleSet: FengShuiRuleSet
    ) throws -> Int {
        let offset = try CircularMath.normalize360(
            headingDeg - ruleSet.firstSectorCenterDeg + ruleSet.sectorWidthDeg / 2.0
        )
        let raw = Int((offset / ruleSet.sectorWidthDeg).rounded(.down))
        return ((raw % ruleSet.sectorCount) + ruleSet.sectorCount) % ruleSet.sectorCount
    }

    /// How far past its own start boundary `headingDeg` sits, in `[0, sectorWidthDeg)`.
    ///
    /// Derived from the *same* wrapped quantity `sectorIndex` uses, so the index and the offset
    /// cannot disagree by a rounding bit at a boundary.
    public static func offsetWithinSectorDeg(
        _ headingDeg: Double,
        _ ruleSet: FengShuiRuleSet
    ) throws -> Double {
        let wrapped = try CircularMath.normalize360(
            headingDeg - ruleSet.firstSectorCenterDeg + ruleSet.sectorWidthDeg / 2.0
        )
        return wrapped - (wrapped / ruleSet.sectorWidthDeg).rounded(.down) * ruleSet.sectorWidthDeg
    }

    /// §21.2's final, explicit, recorded reference step.
    ///
    /// `TRUE` uses the canonical true heading; `MAGNETIC` derives `trueHeading - declination`
    /// from that *same* canonical measurement rather than substituting an unvalidated magnetic
    /// path. `needleOffsetDeg` expresses doctrinal plate conventions and is a declared property
    /// of a named ruleset — never a user slider, never a correction for measurement error.
    public static func classificationHeadingDeg(
        trueHeadingDeg: Double,
        declinationDeg: Double,
        ruleSet: FengShuiRuleSet
    ) throws -> Double {
        let base: Double
        switch ruleSet.referenceSelection {
        case "TRUE":
            base = trueHeadingDeg
        case "MAGNETIC":
            base = try CircularMath.normalize360(trueHeadingDeg - declinationDeg)
        default:
            throw ClassifierError.unknownReferenceSelection(ruleSet.referenceSelection)
        }
        return try CircularMath.normalize360(base + ruleSet.needleOffsetDeg)
    }

    /// §21.4: every ruleset sector intersecting the circular interval, in azimuth order.
    ///
    /// The interval is the **closed** `[h - bound, h + bound]` while sectors are half-open, so an
    /// interval endpoint landing exactly on a boundary includes the sector that boundary starts.
    /// That asymmetry is deliberately conservative: naming one fewer mountain is a
    /// false-precision failure, naming one more is not.
    ///
    /// The count comes from the **arc length**, not from walking forward until the end index is
    /// met. At a bound approaching `180°` the interval wraps almost the whole circle and both
    /// endpoints land in the *same* sector; a walk-until-equal would then report one sector for
    /// an interval covering all 24 — a single-mountain claim from a measurement that
    /// discriminates nothing.
    public static func straddleIndices(
        classificationHeadingDeg: Double,
        reportedBound95Deg: Double,
        ruleSet: FengShuiRuleSet
    ) throws -> [Int] {
        guard reportedBound95Deg.isFinite, reportedBound95Deg >= 0.0 else {
            throw ClassifierError.invalidBound(reportedBound95Deg)
        }
        // §21.4: report that no classification is possible rather than listing all 24.
        if 2.0 * reportedBound95Deg >= 360.0 { return [] }

        let low = classificationHeadingDeg - reportedBound95Deg
        let startIndex = try sectorIndex(low, ruleSet)
        let offset = try offsetWithinSectorDeg(low, ruleSet)
        let spanned = min(
            ruleSet.sectorCount,
            1 + Int(((offset + 2.0 * reportedBound95Deg) / ruleSet.sectorWidthDeg).rounded(.down))
        )
        return (0..<spanned).map { (startIndex + $0) % ruleSet.sectorCount }
    }

    /// Signed circular difference from the **nearest** sector boundary to the heading.
    ///
    /// Positive means the heading lies clockwise of that boundary. Magnitude never exceeds
    /// `sectorWidthDeg / 2`.
    public static func signedOffsetFromSectorBoundaryDeg(
        classificationHeadingDeg: Double,
        ruleSet: FengShuiRuleSet
    ) throws -> Double {
        let index = try sectorIndex(classificationHeadingDeg, ruleSet)
        let candidates = [
            try ruleSet.derivedSectorStartDeg(index),
            try ruleSet.derivedSectorStartDeg((index + 1) % ruleSet.sectorCount),
        ]
        var nearest = candidates[0]
        var smallest = try CircularMath.absoluteCircularDifferenceDeg(
            classificationHeadingDeg, candidates[0]
        )
        for candidate in candidates.dropFirst() {
            let distance = try CircularMath.absoluteCircularDifferenceDeg(
                classificationHeadingDeg, candidate
            )
            if distance < smallest {
                smallest = distance
                nearest = candidate
            }
        }
        return try CircularMath.shortestSignedDifferenceDeg(classificationHeadingDeg, nearest)
    }

    /// §21: classify the whole circular bound interval, never the point estimate alone.
    ///
    /// `reportedBound95Deg` is the **total** bound — instrument plus placement, including any
    /// `referenceAmbiguityBound95Deg`. §21.2: subtracting declination for a magnetic ruleset MUST
    /// NOT zero or remove the ambiguity term, because if the provider secretly emitted the other
    /// reference the derived magnetic point is wrong by `|d|` too. This function therefore never
    /// touches the bound it is handed.
    public static func classify(
        trueHeadingDeg: Double,
        declinationDeg: Double,
        reportedBound95Deg: Double,
        ruleSet: FengShuiRuleSet
    ) throws -> FengShuiClassification {
        let heading = try classificationHeadingDeg(
            trueHeadingDeg: trueHeadingDeg,
            declinationDeg: declinationDeg,
            ruleSet: ruleSet
        )
        let indices = try straddleIndices(
            classificationHeadingDeg: heading,
            reportedBound95Deg: reportedBound95Deg,
            ruleSet: ruleSet
        )
        let offset = try signedOffsetFromSectorBoundaryDeg(
            classificationHeadingDeg: heading, ruleSet: ruleSet
        )
        if indices.isEmpty {
            return FengShuiClassification(
                ruleSetVersion: ruleSet.ruleSetVersion,
                referenceSelection: ruleSet.referenceSelection,
                classificationHeadingDeg: heading,
                primarySector: nil,
                possibleSectors: [],
                possibleSectorIndices: [],
                boundaryStraddled: true,
                signedOffsetFromSectorBoundaryDeg: offset,
                classificationPossible: false
            )
        }
        let primaryIndex = try sectorIndex(heading, ruleSet)
        return FengShuiClassification(
            ruleSetVersion: ruleSet.ruleSetVersion,
            referenceSelection: ruleSet.referenceSelection,
            classificationHeadingDeg: heading,
            primarySector: ruleSet.sectors[primaryIndex].name,
            possibleSectors: indices.map { ruleSet.sectors[$0].name },
            possibleSectorIndices: indices,
            boundaryStraddled: indices.count > 1,
            signedOffsetFromSectorBoundaryDeg: offset,
            classificationPossible: true
        )
    }
}

/// §5.1's `classification:` block for one measurement.
public struct FengShuiClassification: Equatable, Sendable {
    public let ruleSetVersion: String
    public let referenceSelection: String
    public let classificationHeadingDeg: Double
    public let primarySector: String?
    public let possibleSectors: [String]
    public let possibleSectorIndices: [Int]
    public let boundaryStraddled: Bool
    public let signedOffsetFromSectorBoundaryDeg: Double
    public let classificationPossible: Bool
}
