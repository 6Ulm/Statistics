import XCTest
@testable import HeadingCore

/// SPEC.md §8.1.1 — the required build-time grade-reachability analysis, iOS half.
/// Mirrors `android/heading-core/.../GradeReachabilityTest.kt` case for case, against the
/// same repository-root artifacts.
final class GradeReachabilityTests: XCTestCase {

    private var profile: PrecisionProfile!
    private var claims: GradeReachability.ClaimsDocument!

    override func setUpWithError() throws {
        profile = try PrecisionProfile.load(contentsOf: SharedArtifacts.precisionProfileURL())
        claims = try GradeReachability.loadClaims(contentsOf: SharedArtifacts.gradeReachabilityClaimsURL())
    }

    func testEveryClaimedGradeIsReachable() {
        let findings = GradeReachability.verify(claims: claims, profile: profile)
        XCTAssertTrue(findings.isEmpty,
                      "SPEC.md §8.1.1 grade-reachability findings. A failing gate is a finding, not "
                        + "an obstacle (§37 rule 12): fix the claim or the evidence, never the fixture.\n"
                        + findings.map { "  \($0)" }.joined(separator: "\n"))
    }

    func testFlatFreehandUncertifiedCannotLock() throws {
        let budget = try XCTUnwrap(GradeReachability.instrumentBudgetDeg(.freehand, profile: profile))
        XCTAssertEqual(budget, 2.0, accuracy: 1e-12)
        XCTAssertGreaterThan(profile.unknownDeviceFloor95Deg, budget,
                             "That is what makes a Precision Lock impossible on any uncertified "
                                + "device in the ordinary user gesture.")

        let r = GradeReachability.compute(placementMethod: .freehand,
                                          certificationState: .uncertified,
                                          magneticState: .clean, profile: profile)
        XCTAssertEqual(try XCTUnwrap(r.minimumReportedBound95Deg), 7.0, accuracy: 1e-12)
        XCTAssertFalse(r.lockReachable)
        XCTAssertEqual(r.maxReachableGrade, .lowConfidence)
    }

    func testWallFreehandCanNeverLock() throws {
        let budget = try XCTUnwrap(
            GradeReachability.instrumentBudgetDeg(.wallFlushFreehand, profile: profile))
        XCTAssertEqual(budget, 0.0, accuracy: 1e-12)

        XCTAssertFalse(GradeReachability.compute(placementMethod: .wallFlushFreehand,
                                                 certificationState: .uncertified,
                                                 magneticState: .clean, profile: profile).lockReachable)

        // Certification cannot rescue it: the required floor is zero and a device floor is
        // strictly positive.
        let certified = GradeReachability.compute(placementMethod: .wallFlushFreehand,
                                                  certificationState: .certified,
                                                  magneticState: .clean, profile: profile)
        XCTAssertEqual(try XCTUnwrap(certified.requiredDeviceFloorAtMostDeg), 0.0, accuracy: 1e-12)
        XCTAssertFalse(certified.lockReachable)
    }

    func testSuspectPreventsFreehandLockOutright() throws {
        let flatBudget = try XCTUnwrap(GradeReachability.instrumentBudgetDeg(.freehand, profile: profile))
        let wallBudget = try XCTUnwrap(
            GradeReachability.instrumentBudgetDeg(.wallFlushFreehand, profile: profile))
        XCTAssertGreaterThan(profile.suspectInterferenceBound95Deg, flatBudget)
        XCTAssertGreaterThan(profile.suspectInterferenceBound95Deg, wallBudget)

        for method in [PlacementMethod.freehand, .wallFlushFreehand] {
            for certification in CertificationState.allCases {
                let r = GradeReachability.compute(placementMethod: method,
                                                  certificationState: certification,
                                                  magneticState: .suspect, profile: profile)
                XCTAssertFalse(r.lockReachable,
                               "\(method.rawValue)/\(certification.rawValue) under SUSPECT must not "
                                + "be lock-reachable: \(r.explanation)")
            }
        }
    }

    func testCertifiedFlatFreehandRequiredFloor() throws {
        let r = GradeReachability.compute(placementMethod: .freehand, certificationState: .certified,
                                          magneticState: .clean, profile: profile)
        XCTAssertEqual(try XCTUnwrap(r.requiredDeviceFloorAtMostDeg), 2.0, accuracy: 1e-12)
        XCTAssertTrue(r.lockReachable)

        // Sweeping the floor as an explicit parameter, as §8.1.1 requires of Phase 5.
        XCTAssertTrue(GradeReachability.compute(placementMethod: .freehand,
                                                certificationState: .certified,
                                                magneticState: .clean, profile: profile,
                                                certifiedDeviceFloor95Deg: 2.0).lockReachable)
        XCTAssertFalse(GradeReachability.compute(placementMethod: .freehand,
                                                 certificationState: .certified,
                                                 magneticState: .clean, profile: profile,
                                                 certifiedDeviceFloor95Deg: 2.0001).lockReachable)
    }

    func testUnmeasuredPlacementYieldsNoGrade() {
        for method in [PlacementMethod.nonmagneticAlignmentJig, .surveyFixture] {
            XCTAssertNil(GradeReachability.placementBound95Deg(method, profile: profile))
            let r = GradeReachability.compute(placementMethod: method, certificationState: .certified,
                                              magneticState: .clean, profile: profile)
            XCTAssertEqual(r.maxReachableGrade, .notSupported)
            XCTAssertFalse(r.lockReachable)
        }
    }

    func testRejectingMagneticStatesProduceNoMeasurement() {
        for state in [MagneticState.disturbed, .invalid, .unknown] {
            XCTAssertNil(GradeReachability.interferenceBound95Deg(state, profile: profile))
            let r = GradeReachability.compute(placementMethod: .freehand,
                                              certificationState: .certified,
                                              magneticState: state, profile: profile)
            XCTAssertNil(r.minimumReportedBound95Deg)
            XCTAssertEqual(r.maxReachableGrade, .invalid)
        }
    }

    func testGradeFunctionIsTotalAndOrdered() {
        XCTAssertEqual(qualityGradeForReportedBound(0.0, profile: profile), .professional)
        XCTAssertEqual(qualityGradeForReportedBound(2.0, profile: profile), .professional)
        XCTAssertEqual(qualityGradeForReportedBound(2.0000001, profile: profile), .high)
        XCTAssertEqual(qualityGradeForReportedBound(3.0, profile: profile), .high)
        XCTAssertEqual(qualityGradeForReportedBound(3.0000001, profile: profile), .usable)
        XCTAssertEqual(qualityGradeForReportedBound(5.0, profile: profile), .usable)
        XCTAssertEqual(qualityGradeForReportedBound(5.0000001, profile: profile), .lowConfidence)
        XCTAssertEqual(qualityGradeForReportedBound(10.0, profile: profile), .lowConfidence)
        XCTAssertEqual(qualityGradeForReportedBound(10.0000001, profile: profile), .invalid)
        XCTAssertEqual(qualityGradeForReportedBound(180.0, profile: profile), .invalid)
    }
}
