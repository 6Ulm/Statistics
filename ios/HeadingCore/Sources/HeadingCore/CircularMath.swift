import Foundation

/// SPEC.md §9 deterministic utilities.
///
/// This enum is the **single allowlisted home** for the signed circular difference in the iOS
/// runtime (§33.1, R67/R68). `shortestSignedDifferenceDeg` below is the one implementation;
/// `shortestTargetDeltaDeg` and `absoluteCircularDifferenceDeg` are exact delegating wrappers
/// containing no angle arithmetic of their own. No other file in this runtime may restate the
/// formula, and the analysis-runtime audit enforces that across all three runtimes.
///
/// Two `atan2` call sites live here and are the ones §33.1 allowlists: the signed-difference
/// implementation, and the circular mean/resultant (a bearing projection, a different quantity).
///
/// > Warning: this file has never been compiled — see `docs/IMPLEMENTATION_NOTES.md` D-3.
public enum CircularMath {

    public enum AngleError: Error, CustomStringConvertible {
        case nonFinite(String, Double)
        case negativeSigma(Double)
        case invalidFactor(Double)
        case probabilityOutOfRange(Double)

        public var description: String {
            switch self {
            case .nonFinite(let what, let value):
                return "\(what) requires a finite value, got \(value)"
            case .negativeSigma(let value):
                return "boundFromSigma requires a non-negative sigma, got \(value)"
            case .invalidFactor(let value):
                return "declinationSigmaToBound95Factor must be positive, got \(value)"
            case .probabilityOutOfRange(let value):
                return "quantile probability must lie in [0, 1], got \(value)"
            }
        }
    }

    /// SPEC.md §9.1's typed `UNDEFINED` outcome.
    ///
    /// `quantile` and `median` on empty input "MUST return a typed `UNDEFINED`/validation
    /// failure and MUST NOT index element zero".
    public enum UndefinedResult: Error, CustomStringConvertible {
        case emptyWindow(String)
        case zeroResultant

        public var description: String {
            switch self {
            case .emptyWindow(let what):
                return "\(what) is UNDEFINED on empty input; §9.1 forbids indexing element zero"
            case .zeroResultant:
                return "circularMeanDeg is UNDEFINED: zero resultant (CIRCULAR_MEAN_UNDEFINED)"
            }
        }
    }

    private static func requireFinite(_ value: Double, _ what: String) throws -> Double {
        guard value.isFinite else { throw AngleError.nonFinite(what, value) }
        return value
    }

    /// SPEC.md §9: `((x % 360) + 360) % 360` with a finite check; exactly `360.0` maps to `0.0`.
    ///
    /// Written with the explicit double-modulo because the language remainder operator differs
    /// for negative operands (failure mode 2). Note that this mandated form is not bit-exact
    /// for values already in `[0, 360)` — the round trip loses about `1e-10`, three orders
    /// inside the declared `1e-6` cross-runtime tolerance. Kotlin, Swift and Python perform the
    /// same IEEE-754 operations in the same order, so parity is unaffected (F-1).
    public static func normalize360(_ deg: Double) throws -> Double {
        let value = try requireFinite(deg, "normalize360")
        let wrapped = (value.truncatingRemainder(dividingBy: 360.0) + 360.0)
            .truncatingRemainder(dividingBy: 360.0)
        return wrapped == 360.0 ? 0.0 : wrapped + 0.0
    }

    /// SPEC.md §9/§3: `a - b` as the shortest signed circular difference in `(-180, 180]`.
    ///
    /// The `atan2` convention, **and** the mandatory antipode normalization: raw
    /// `atan2(sin(a-b), cos(a-b))` returns `-180.0` whenever `sin` evaluates to a tiny negative
    /// rather than exactly zero, which is what happens for ordinary inputs such as `a=0, b=180`
    /// and `a=90, b=270`. An exact `-180.0` is therefore mapped to `+180.0` before returning.
    ///
    /// This is the one normative contract in the spec and the only implementation in this
    /// runtime.
    public static func shortestSignedDifferenceDeg(_ a: Double, _ b: Double) throws -> Double {
        let left = try requireFinite(a, "shortestSignedDifferenceDeg")
        let right = try requireFinite(b, "shortestSignedDifferenceDeg")
        let radians = (left - right) * Double.pi / 180.0
        let delta = atan2(sin(radians), cos(radians)) * 180.0 / Double.pi
        return delta == -180.0 ? 180.0 : delta
    }

    /// SPEC.md §9: `shortestSignedDifferenceDeg(target, current)`; positive = clockwise.
    /// A thin delegating wrapper with the exact §9 definition (R68).
    public static func shortestTargetDeltaDeg(
        current currentDeg: Double,
        target targetDeg: Double
    ) throws -> Double {
        try shortestSignedDifferenceDeg(targetDeg, currentDeg)
    }

    /// SPEC.md §9: `abs(shortestSignedDifferenceDeg(a, b))`, range `[0, 180]`. A thin
    /// delegating wrapper (R68).
    public static func absoluteCircularDifferenceDeg(_ a: Double, _ b: Double) throws -> Double {
        abs(try shortestSignedDifferenceDeg(a, b))
    }

    /// SPEC.md §15 circular mean and resultant length with **uniform weights**.
    ///
    /// `w_i = 1` for every accepted sample. Weighting by provider error, recency or dispersion
    /// is a plausible and untested improvement; §15 makes it a named benchmark variant, not a
    /// quiet implementation choice. There is no trimming here either — rejection happens at the
    /// per-sample gate, before entry.
    public static func circularAggregate(_ samples: [Double]) throws -> CircularAggregate {
        for sample in samples { _ = try requireFinite(sample, "circularAggregate sample") }
        guard !samples.isEmpty else {
            return CircularAggregate(meanDeg: nil, resultantLength: 0.0, count: 0)
        }
        let count = Double(samples.count)
        let cosine = samples.reduce(0.0) { $0 + cos($1 * Double.pi / 180.0) } / count
        let sine = samples.reduce(0.0) { $0 + sin($1 * Double.pi / 180.0) } / count
        // A resultant may exceed 1 only by floating-point rounding; §6 declares the range [0,1].
        let resultant = min(1.0, max(0.0, (cosine * cosine + sine * sine).squareRoot()))
        if cosine == 0.0 && sine == 0.0 {
            return CircularAggregate(
                meanDeg: nil, resultantLength: resultant, count: samples.count
            )
        }
        return CircularAggregate(
            meanDeg: try normalize360(atan2(sine, cosine) * 180.0 / Double.pi),
            resultantLength: resultant,
            count: samples.count
        )
    }

    /// SPEC.md §9 `circularMeanDeg`; throws for `UNDEFINED`.
    public static func circularMeanDeg(_ samples: [Double]) throws -> Double {
        let aggregate = try circularAggregate(samples)
        guard let mean = aggregate.meanDeg else {
            throw aggregate.count == 0
                ? UndefinedResult.emptyWindow("circularMeanDeg")
                : UndefinedResult.zeroResultant
        }
        return mean
    }

    /// SPEC.md §9 `circularResultantLength` → `[0, 1]`.
    public static func circularResultantLength(_ samples: [Double]) throws -> Double {
        try circularAggregate(samples).resultantLength
    }

    /// SPEC.md §15 decision 3: "A weak resultant is an explicit failure."
    ///
    /// The exactly-degenerate `atan2(0, 0)` case is only half the problem, and the smaller
    /// half: for an antipodal pair the two sines cancel to `6.1e-17` rather than to zero, so
    /// the mean is *numerically* defined and comes back as a confident, completely arbitrary
    /// bearing. Only the configured `minCircularResultantLength` gate catches that, which is
    /// why this decision reads a config key and never an epsilon invented here.
    public static func circularMeanIsUndefined(
        _ aggregate: CircularAggregate,
        minCircularResultantLength: Double
    ) -> Bool {
        aggregate.meanDeg == nil || aggregate.resultantLength < minCircularResultantLength
    }

    /// Absolute residuals about an accepted circular mean, in sample order (§19).
    public static func circularResidualsDeg(
        _ samples: [Double],
        meanDeg: Double
    ) throws -> [Double] {
        try samples.map { try absoluteCircularDifferenceDeg($0, meanDeg) }
    }

    /// SPEC.md §9/§9.1: the linear estimator applied to absolute residuals about the mean.
    ///
    /// §15 fixes the sample set: **all** accepted samples, no trimming, so the dispersion gate
    /// and the dispersion-derived bound cannot disagree about which samples exist.
    public static func circularResidualQuantileDeg(
        _ samples: [Double],
        _ q: Double
    ) throws -> Double {
        let mean = try circularMeanDeg(samples)
        return try Estimators.quantile(try circularResidualsDeg(samples, meanDeg: mean), q)
    }

    /// SPEC.md §19.2: the single named conversion from one sigma to a 95% bound.
    ///
    /// The candidate factor `1.96` is the Gaussian two-sided 95% multiplier — a **modelling
    /// assumption**, not a property of the model, which is why it lives in versioned
    /// configuration and is passed in here rather than written as a literal.
    public static func boundFromSigma(
        _ sigma1Deg: Double,
        sigmaToBound95Factor: Double
    ) throws -> Double {
        let sigma = try requireFinite(sigma1Deg, "boundFromSigma sigma")
        let factor = try requireFinite(sigmaToBound95Factor, "boundFromSigma factor")
        guard sigma >= 0.0 else { throw AngleError.negativeSigma(sigma) }
        guard factor > 0.0 else { throw AngleError.invalidFactor(factor) }
        return factor * sigma
    }
}

/// The result of aggregating a window of angles under SPEC.md §15.
///
/// `meanDeg` is `nil` exactly when the mean is `UNDEFINED`: an empty window, or a resultant of
/// exactly zero, where `atan2(0, 0)` returns `0.0` on every platform and would disguise a
/// bimodal set as a north-facing measurement (failure mode 6).
public struct CircularAggregate: Equatable, Sendable {
    public let meanDeg: Double?
    public let resultantLength: Double
    public let count: Int

    public var isDefined: Bool { meanDeg != nil }

    public init(meanDeg: Double?, resultantLength: Double, count: Int) {
        self.meanDeg = meanDeg
        self.resultantLength = resultantLength
        self.count = count
    }
}
