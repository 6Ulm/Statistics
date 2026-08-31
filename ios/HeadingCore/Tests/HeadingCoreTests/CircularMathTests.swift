import XCTest
@testable import HeadingCore

/// SPEC.md §9: `normalize360` MUST be `((x % 360) + 360) % 360` with a finite check.
/// Test `-360`, `-0.0`, `359.9999999`, `360.0`. Mirrors the Kotlin cases exactly.
final class CircularMathTests: XCTestCase {

    /// §33.1 / §36 Phase 1 declare the cross-runtime angle tolerance as 1e-6°.
    private let angleToleranceDeg = 1e-6

    func testNamedCases() throws {
        XCTAssertEqual(try CircularMath.normalize360(-360.0), 0.0, accuracy: 0.0)
        XCTAssertEqual(try CircularMath.normalize360(-0.0), 0.0, accuracy: 0.0)
        XCTAssertEqual(try CircularMath.normalize360(359.9999999), 359.9999999,
                       accuracy: angleToleranceDeg)
        XCTAssertEqual(try CircularMath.normalize360(360.0), 0.0, accuracy: 0.0)
    }

    func testNoNegativeZeroEscapes() throws {
        // -0.0 == 0.0 compares true, so assert the bit pattern: a negative zero escaping into
        // a bearing is the kind of value that reappears as a sign flip downstream.
        XCTAssertEqual(try CircularMath.normalize360(-0.0).bitPattern, (0.0).bitPattern)
        XCTAssertEqual(try CircularMath.normalize360(360.0).bitPattern, (0.0).bitPattern)
        XCTAssertEqual(try CircularMath.normalize360(-720.0).bitPattern, (0.0).bitPattern)
    }

    func testNegativeInputsWrap() throws {
        XCTAssertEqual(try CircularMath.normalize360(-1.0), 359.0, accuracy: 1e-12)
        XCTAssertEqual(try CircularMath.normalize360(-719.0), 1.0, accuracy: 1e-12)
        XCTAssertEqual(try CircularMath.normalize360(-180.0), 180.0, accuracy: 1e-12)
    }

    func testInRangeValuesAreNotBitExact() throws {
        // The mandated double-modulo form loses low bits for values already in [0,360). The
        // residual is ~1e-10, three orders of magnitude inside the 1e-6 cross-runtime
        // tolerance, and every runtime performs the same IEEE-754 operations in the same
        // order, so parity is unaffected. Pinned here so a regression would surface as a
        // test failure rather than as a bearing.
        for value in [359.9999999, 0.1, 123.456789, 359.5, 1e-8] {
            let residual = abs(try CircularMath.normalize360(value) - value)
            XCTAssertEqual(residual, 0.0, accuracy: 1e-9, "normalize360(\(value)) drifted by \(residual)")
        }
        XCTAssertEqual(try CircularMath.normalize360(359.9999999), 359.9999998999999, accuracy: 0.0)
    }

    func testNonfiniteRejected() {
        XCTAssertThrowsError(try CircularMath.normalize360(.nan))
        XCTAssertThrowsError(try CircularMath.normalize360(.infinity))
        XCTAssertThrowsError(try CircularMath.normalize360(-.infinity))
    }
}
