import Foundation

/// SPEC.md §9.1 — the pinned quantile and median.
///
/// Common library estimators disagree by a full sample position at these window sizes, so the
/// estimator is pinned rather than inherited: both platforms MUST be bit-identical on
/// `testdata/angles/`, and `analysis/` MUST use the same definition so a device-computed P95
/// and a report-computed P95 of the same window agree exactly (failure mode 47).
///
/// > Warning: this file has never been compiled — see `docs/IMPLEMENTATION_NOTES.md` D-3.
public enum Estimators {

    /// SPEC.md §9.1 **nearest-rank**:
    /// `quantile(x, q) = x[min(n-1, max(0, ceil(q*n) - 1))]` over `x` sorted ascending.
    public static func quantile(_ values: [Double], _ q: Double) throws -> Double {
        guard (0.0...1.0).contains(q) else {
            throw CircularMath.AngleError.probabilityOutOfRange(q)
        }
        let ordered = try sortedFinite(values, "quantile")
        let n = ordered.count
        let index = min(n - 1, max(0, Int((q * Double(n)).rounded(.up)) - 1))
        return ordered[index]
    }

    /// SPEC.md §9.1: odd `n` → the middle element; even `n` → the mean of the two middles.
    public static func median(_ values: [Double]) throws -> Double {
        let ordered = try sortedFinite(values, "median")
        let n = ordered.count
        if n % 2 == 1 { return ordered[(n - 1) / 2] }
        return (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    }

    private static func sortedFinite(_ values: [Double], _ what: String) throws -> [Double] {
        guard !values.isEmpty else { throw CircularMath.UndefinedResult.emptyWindow(what) }
        // Nonfinite members are rejected *before* sorting (§9.1) — NaN would otherwise corrupt
        // the ordering silently rather than failing.
        for value in values where !value.isFinite {
            throw CircularMath.AngleError.nonFinite("\(what) sample", value)
        }
        return values.sorted()
    }
}
