import XCTest
import HeadingCore
@testable import FengShuiCore

/// SPEC.md §21.1 required schema/derived-boundary test, plus R65's completeness rule.
/// Mirrors `android/fengshui-core/.../FengShuiRuleSetGeometryTest.kt`.
final class FengShuiRuleSetGeometryTests: XCTestCase {

    private var ruleSet: FengShuiRuleSet!

    override func setUpWithError() throws {
        ruleSet = try FengShuiRuleSet.load(contentsOf: SharedArtifacts.fengShuiRuleSetURL())
    }

    func testRuleSetIsComplete() {
        XCTAssertEqual(ruleSet.ruleSetVersion, "fengshui-v1")
        XCTAssertEqual(ruleSet.sectorCount, 24)
        XCTAssertEqual(ruleSet.sectors.count, 24)
        XCTAssertEqual(ruleSet.groups.count, 8)
    }

    func testGeometryIsConsistent() {
        let violations = FengShuiRuleSetGeometry.check(ruleSet)
        XCTAssertTrue(violations.isEmpty,
                      "SPEC.md §21.1 ruleset violations:\n"
                        + violations.map { "  \($0)" }.joined(separator: "\n"))
    }

    func testBoundariesAreAtSevenPointFivePlusFifteenK() throws {
        let starts = try (0..<24).map { try ruleSet.derivedSectorStartDeg($0) }.sorted()
        let expected = (0..<24).map { (7.5 + 15.0 * Double($0)).truncatingRemainder(dividingBy: 360.0) }
            .sorted()
        for (actual, want) in zip(starts, expected) {
            XCTAssertEqual(actual, want, accuracy: 1e-9)
        }
        // The north-wrap boundary: 352.5 separates 壬 (ren, index 23) from 子 (zi, index 0).
        XCTAssertEqual(try ruleSet.derivedSectorStartDeg(0), 352.5, accuracy: 1e-9)
        XCTAssertEqual(ruleSet.sectors.first { $0.index == 23 }?.name, "ren")
        XCTAssertEqual(ruleSet.sectors.first { $0.index == 0 }?.name, "zi")
    }
}
