import Foundation

/// SPEC.md §9 deterministic utilities.
///
/// Phase 0 defines only `normalize360`, which the §21.1 derived-boundary check needs. The
/// remaining §9 utilities — `shortestSignedDifferenceDeg` and its exact delegating
/// wrappers, the circular mean/resultant, and the §9.1 pinned quantile/median — are Phase 1
/// work and are deliberately absent rather than stubbed, so no caller can bind to a
/// placeholder definition (§37).
///
/// This enum is the single allowlisted home for circular math in the iOS runtime
/// (§33.1, R67/R68). No other file may restate these formulas.
public enum CircularMath {

    public enum AngleError: Error, CustomStringConvertible {
        case nonFinite(Double)
        public var description: String {
            switch self {
            case .nonFinite(let value):
                return "normalize360 requires a finite angle, got \(value)"
            }
        }
    }

    /// SPEC.md §9: `((x % 360) + 360) % 360` with a finite check; exactly `360.0` maps to
    /// `0.0`. Written with the explicit double-modulo because the language remainder
    /// operator differs for negative operands (failure mode 2).
    ///
    /// Note: this mandated form is not bit-exact for values already in `[0, 360)` — the
    /// round trip through `+360` and `% 360` loses low bits, about `1e-10`, three orders of
    /// magnitude inside the `1e-6°` cross-runtime tolerance. Kotlin, Swift and Python
    /// perform the same IEEE-754 operations in the same order, so parity is unaffected.
    public static func normalize360(_ deg: Double) throws -> Double {
        guard deg.isFinite else { throw AngleError.nonFinite(deg) }
        let wrapped = deg.truncatingRemainder(dividingBy: 360.0)
            .adding360AndWrap()
        return wrapped == 360.0 ? 0.0 : wrapped + 0.0
    }
}

private extension Double {
    func adding360AndWrap() -> Double {
        (self + 360.0).truncatingRemainder(dividingBy: 360.0)
    }
}
