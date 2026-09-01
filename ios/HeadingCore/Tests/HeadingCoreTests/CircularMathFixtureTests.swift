import XCTest
@testable import HeadingCore

/// SPEC.md §9 / §9.1 / §15 — the iOS runtime read against the frozen `fixtures-v1` contract.
///
/// §36 Phase 1's exit criterion is that identical fixtures agree within `1e-6°` across the
/// Android core, the iOS core and `analysis/`. These tests are that check for iOS: the same
/// files the other two runtimes read, compared with the tolerance the fixture itself declares.
///
/// > Warning: this file has never been compiled or executed — see
/// > `docs/IMPLEMENTATION_NOTES.md` D-3. On a macOS host, run `cd ios && swift test`.
final class CircularMathFixtureTests: XCTestCase {

    private func fixture(_ url: URL) throws -> [String: Any] {
        let data = try Data(contentsOf: url)
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "fixture", code: 1)
        }
        return object
    }

    private func circularFixture() throws -> [String: Any] {
        try fixture(SharedArtifacts.circularMathFixtureURL())
    }

    private func workingTolerance() throws -> Double {
        try XCTUnwrap(circularFixture()["workingAngleToleranceDeg"] as? Double)
    }

    // MARK: - normalize360 (§9)

    func testNormalize360MatchesTheFrozenFixture() throws {
        let tolerance = try workingTolerance()
        let cases = try XCTUnwrap(circularFixture()["normalize360"] as? [[String: Any]])
        for entry in cases {
            let input = try XCTUnwrap(entry["inputDeg"] as? Double)
            let expected = try XCTUnwrap(entry["expectedDeg"] as? Double)
            XCTAssertEqual(try CircularMath.normalize360(input), expected,
                           accuracy: tolerance, "input=\(input)")
        }
    }

    // MARK: - shortestSignedDifferenceDeg (§3, §9)

    func testSignedDifferenceMatchesTheFrozenFixture() throws {
        let tolerance = try workingTolerance()
        let cases = try XCTUnwrap(
            circularFixture()["shortestSignedDifferenceDeg"] as? [[String: Any]]
        )
        for entry in cases {
            let a = try XCTUnwrap(entry["aDeg"] as? Double)
            let b = try XCTUnwrap(entry["bDeg"] as? Double)
            let expected = try XCTUnwrap(entry["expectedDeg"] as? Double)
            XCTAssertEqual(try CircularMath.shortestSignedDifferenceDeg(a, b), expected,
                           accuracy: tolerance, "a=\(a) b=\(b)")
        }
    }

    /// §3/§35: `+180`, never `-180`, for **both orderings** of at least two distinct pairs.
    ///
    /// A test that only checks `deltaDeg(180, 0)` passes on a broken implementation, because
    /// raw `atan2` returns `-180.0` only for the ordering whose `sin` lands on a tiny negative.
    func testAntipodeIsPlus180InBothOrderings() throws {
        let pairs: [(Double, Double)] = [
            (0.0, 180.0), (180.0, 0.0),
            (90.0, 270.0), (270.0, 90.0),
            (120.0, 300.0), (300.0, 120.0),
        ]
        for (a, b) in pairs {
            XCTAssertEqual(try CircularMath.shortestSignedDifferenceDeg(a, b), 180.0,
                           "a=\(a) b=\(b)")
        }
    }

    /// §33.1 permits documentation and tests to *quote* a prohibited formula as text.
    /// Computing it here proves the antipode normalization is load-bearing, not decorative.
    func testRawAtan2WouldFailThisContract() throws {
        let radians = (0.0 - 180.0) * Double.pi / 180.0
        let prohibited = atan2(sin(radians), cos(radians)) * 180.0 / Double.pi
        XCTAssertEqual(prohibited, -180.0)
        XCTAssertEqual(try CircularMath.shortestSignedDifferenceDeg(0.0, 180.0), 180.0)
    }

    func testSignedDifferenceRangeIsHalfOpenAtMinus180() throws {
        for a in stride(from: 0.0, to: 360.0, by: 1.0) {
            for b in [0.0, 37.0, 180.0, 359.0] {
                let delta = try CircularMath.shortestSignedDifferenceDeg(a, b)
                XCTAssertTrue(delta > -180.0 && delta <= 180.0, "a=\(a) b=\(b) delta=\(delta)")
            }
        }
    }

    // MARK: - the exact delegating wrappers (§9, R68)

    func testTargetDeltaIsExactlyTheSpecifiedDelegation() throws {
        let tolerance = try workingTolerance()
        let cases = try XCTUnwrap(
            circularFixture()["shortestTargetDeltaDeg"] as? [[String: Any]]
        )
        for entry in cases {
            let current = try XCTUnwrap(entry["currentDeg"] as? Double)
            let target = try XCTUnwrap(entry["targetDeg"] as? Double)
            let expected = try XCTUnwrap(entry["expectedDeg"] as? Double)
            XCTAssertEqual(
                try CircularMath.shortestTargetDeltaDeg(current: current, target: target),
                expected, accuracy: tolerance, "current=\(current) target=\(target)"
            )
            XCTAssertEqual(
                try CircularMath.shortestTargetDeltaDeg(current: current, target: target),
                try CircularMath.shortestSignedDifferenceDeg(target, current)
            )
        }
    }

    /// §18.2: positive = clockwise/right. From `359°` to `1°` is a short right turn.
    func testTargetDeltaSignConventionAcrossNorthWrap() throws {
        XCTAssertGreaterThan(
            try CircularMath.shortestTargetDeltaDeg(current: 359.0, target: 1.0), 0.0
        )
        XCTAssertLessThan(
            try CircularMath.shortestTargetDeltaDeg(current: 1.0, target: 359.0), 0.0
        )
    }

    func testAbsoluteDifferenceIsExactlyTheSpecifiedDelegation() throws {
        let tolerance = try workingTolerance()
        let cases = try XCTUnwrap(
            circularFixture()["absoluteCircularDifferenceDeg"] as? [[String: Any]]
        )
        for entry in cases {
            let a = try XCTUnwrap(entry["aDeg"] as? Double)
            let b = try XCTUnwrap(entry["bDeg"] as? Double)
            let expected = try XCTUnwrap(entry["expectedDeg"] as? Double)
            let observed = try CircularMath.absoluteCircularDifferenceDeg(a, b)
            XCTAssertEqual(observed, expected, accuracy: tolerance, "a=\(a) b=\(b)")
            XCTAssertTrue(observed >= 0.0 && observed <= 180.0)
        }
    }

    // MARK: - §9.1 pinned estimators

    func testEstimatorsMatchTheFrozenFixture() throws {
        let document = try fixture(SharedArtifacts.estimatorsFixtureURL())
        for entry in try XCTUnwrap(document["quantile"] as? [[String: Any]]) {
            let values = try XCTUnwrap(entry["values"] as? [Double])
            let probability = try XCTUnwrap(entry["probability"] as? Double)
            let expected = try XCTUnwrap(entry["expected"] as? Double)
            XCTAssertEqual(try Estimators.quantile(values, probability), expected,
                           "q=\(probability)")
        }
        for entry in try XCTUnwrap(document["median"] as? [[String: Any]]) {
            let values = try XCTUnwrap(entry["values"] as? [Double])
            let expected = try XCTUnwrap(entry["expected"] as? Double)
            XCTAssertEqual(try Estimators.median(values), expected)
        }
    }

    /// §9.1: with `n = 20` the nearest-rank P95 is `x[18]`. A linear-interpolation estimator
    /// would return `19.05`; a device-computed P95 and a report-computed P95 must agree exactly.
    func testQuantileIsNearestRank() throws {
        let values = (1...20).map(Double.init)
        XCTAssertEqual(try Estimators.quantile(values, 0.95), 19.0)
        XCTAssertEqual(try Estimators.quantile(values, 1.0), 20.0)
        XCTAssertEqual(try Estimators.quantile(values, 0.0), 1.0)
    }

    func testEstimatorsAreUndefinedOnEmptyInput() {
        XCTAssertThrowsError(try Estimators.quantile([], 0.95))
        XCTAssertThrowsError(try Estimators.median([]))
    }

    func testEstimatorsRejectNonFiniteBeforeSorting() {
        XCTAssertThrowsError(try Estimators.quantile([1.0, Double.nan, 3.0], 0.5))
        XCTAssertThrowsError(try Estimators.median([1.0, Double.infinity]))
    }

    // MARK: - §15 circular aggregation

    func testCircularAggregateMatchesTheFrozenFixture() throws {
        let document = try fixture(SharedArtifacts.circularAggregateFixtureURL())
        let minimumResultant = try XCTUnwrap(
            document["minCircularResultantLength"] as? Double
        )
        for window in try XCTUnwrap(document["windows"] as? [[String: Any]]) {
            let id = try XCTUnwrap(window["id"] as? String)
            let samples = try XCTUnwrap(window["samples"] as? [Double])
            let aggregate = try CircularMath.circularAggregate(samples)

            XCTAssertEqual(aggregate.isDefined,
                           try XCTUnwrap(window["meanIsDefined"] as? Bool), id)
            XCTAssertEqual(aggregate.resultantLength,
                           try XCTUnwrap(window["expectedResultantLength"] as? Double),
                           accuracy: 1e-12, id)
            XCTAssertEqual(
                CircularMath.circularMeanIsUndefined(
                    aggregate, minCircularResultantLength: minimumResultant
                ),
                try XCTUnwrap(window["expectedCircularMeanUndefinedUnderGate"] as? Bool),
                id
            )
            if aggregate.isDefined {
                XCTAssertEqual(try XCTUnwrap(aggregate.meanDeg),
                               try XCTUnwrap(window["expectedMeanDeg"] as? Double),
                               accuracy: 1e-9, id)
                XCTAssertEqual(
                    try CircularMath.circularResidualQuantileDeg(samples, 0.95),
                    try XCTUnwrap(window["expectedResidualP95Deg"] as? Double),
                    accuracy: 1e-9, id
                )
            }
        }
    }

    /// §15 decision 3 / failure mode 6, in the shape it actually takes in floating point.
    ///
    /// `atan2(0, 0)` returning zero is the textbook case, but an antipodal window does not
    /// reach it: `sin(0) + sin(180°)` cancels to `1.2e-16`, so the mean is numerically
    /// *defined* and comes back as a confident `90°` that means nothing. Only the configured
    /// `minCircularResultantLength` gate catches that.
    func testAWeakResultantIsAnExplicitFailure() throws {
        let minimum = 0.995
        for samples in [[0.0, 180.0], [0.0, 90.0, 180.0, 270.0]] {
            let aggregate = try CircularMath.circularAggregate(samples)
            XCTAssertEqual(aggregate.resultantLength, 0.0, accuracy: 1e-12)
            XCTAssertTrue(
                CircularMath.circularMeanIsUndefined(
                    aggregate, minCircularResultantLength: minimum
                ),
                "\(samples)"
            )
        }
        let tight = try CircularMath.circularAggregate([84.7, 85.3, 84.9, 85.4, 85.0])
        XCTAssertFalse(
            CircularMath.circularMeanIsUndefined(tight, minCircularResultantLength: minimum)
        )
    }

    /// Failure mode 1: linear averaging across north turns `359` and `1` into `180`.
    func testCircularMeanCrossesNorth() throws {
        XCTAssertEqual(try CircularMath.circularMeanDeg([359.0, 359.5, 0.0, 0.5, 1.0]), 0.0,
                       accuracy: 1e-9)
    }

    /// §15 decision 2: once accepted, a sample counts. Trimming would let the window discard
    /// exactly the evidence that it is unreliable.
    func testNoTrimmingInsideTheWindow() throws {
        let steady = Array(repeating: 85.0, count: 19)
        let withOutlier = steady + [95.0]
        XCTAssertGreaterThan(
            try CircularMath.circularResidualQuantileDeg(withOutlier, 0.95),
            try CircularMath.circularResidualQuantileDeg(steady, 0.95)
        )
    }

    // MARK: - §19.2 boundFromSigma

    func testBoundFromSigmaIsTheSingleConversion() throws {
        XCTAssertEqual(try CircularMath.boundFromSigma(0.36, sigmaToBound95Factor: 1.96),
                       0.7056, accuracy: 1e-12)
        XCTAssertEqual(try CircularMath.boundFromSigma(0.0, sigmaToBound95Factor: 1.96), 0.0)
        XCTAssertThrowsError(try CircularMath.boundFromSigma(-0.1, sigmaToBound95Factor: 1.96))
        XCTAssertThrowsError(try CircularMath.boundFromSigma(0.36, sigmaToBound95Factor: 0.0))
        XCTAssertThrowsError(
            try CircularMath.boundFromSigma(Double.nan, sigmaToBound95Factor: 1.96)
        )
    }
}
