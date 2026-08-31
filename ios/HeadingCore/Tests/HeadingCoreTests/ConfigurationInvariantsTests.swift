import XCTest
@testable import HeadingCore

/// SPEC.md §8.1 "Enforced invariants" — the iOS half of the Phase 0 gate. The invariant
/// identifiers match the Kotlin and Python implementations so a failure reads the same in
/// all three runtimes (§37.1: platforms stay aligned through shared fixtures, not copied
/// assumptions).
final class ConfigurationInvariantsTests: XCTestCase {

    private var profile: PrecisionProfile!
    private var rawTree: [String: Any]!

    override func setUpWithError() throws {
        let url = try SharedArtifacts.precisionProfileURL()
        profile = try PrecisionProfile.load(contentsOf: url)
        rawTree = try PrecisionProfile.loadRawTree(contentsOf: url)
    }

    func testProfileDecodesAndIdentifiesItself() {
        XCTAssertEqual(profile.schemaVersion, "1.0.0")
        XCTAssertEqual(profile.configVersion, "precision-v1-candidate-1")
    }

    func testAllInvariantsHold() {
        let violations = ConfigurationInvariants.check(profile: profile, rawTree: rawTree)
        XCTAssertTrue(violations.isEmpty,
                      "SPEC.md §8.1 invariant violations:\n"
                        + violations.map { "  \($0)" }.joined(separator: "\n"))
    }

    func testNoCalibrationStateKeyAnywhere() {
        var names: [String] = []
        collectPropertyNames(rawTree as Any, into: &names)
        let offending = names.filter(ConfigurationInvariants.containsCalibrationStateKey)
        XCTAssertTrue(offending.isEmpty,
                      "boundCalibrationState is derived from a §24 certification lookup (§19.1); "
                        + "a configurable calibration state is failure mode 32. Offending: \(offending)")
    }

    func testCalibrationStateDetectionIsNotVacuous() {
        // A passing assertion over an absent key proves nothing unless the detector fires on
        // a document that does contain one, including at nesting depth.
        let injected: [String: Any] = ["a": 1, "nested": ["boundCalibrationState": "CALIBRATED"]]
        var names: [String] = []
        collectPropertyNames(injected as Any, into: &names)
        XCTAssertEqual(names.filter(ConfigurationInvariants.containsCalibrationStateKey),
                       ["boundCalibrationState"])
    }

    func testReferenceSeparationOrdering() {
        XCTAssertLessThanOrEqual(profile.referenceSeparationMarginDeg,
                                 profile.smallDeclinationAmbiguityMaxDeg,
                                 "A margin above the ambiguity allowance creates a declination dead "
                                    + "band that always resolves UNVERIFIED with no visible cause (§11).")
    }

    func testGradeThresholdOrdering() {
        XCTAssertLessThan(profile.professionalBound95MaxDeg, profile.highBound95MaxDeg)
        XCTAssertLessThan(profile.highBound95MaxDeg, profile.usableBound95MaxDeg)
        XCTAssertLessThan(profile.usableBound95MaxDeg, profile.lowConfidenceBound95MaxDeg)
    }

    func testFreehandCannotReachProfessional() {
        XCTAssertLessThan(profile.professionalBound95MaxDeg, profile.flatFreehandPlacementBound95Deg,
                          "§20: an implementation reaching Professional freehand has dropped or "
                            + "falsified the placement term - a certification failure, not a feature.")
    }

    func testDeclinationEnvelopeOrdering() {
        XCTAssertLessThanOrEqual(profile.declinationEnvelopeProfessionalMaxDeg,
                                 profile.declinationEnvelopeUsableMaxDeg)
    }

    func testSuspectBelowDisturbed() {
        XCTAssertLessThan(profile.magneticMagnitudeResidualSuspectFraction,
                          profile.magneticMagnitudeResidualDisturbedFraction)
        XCTAssertLessThan(profile.inclinationResidualSuspectDeg, profile.inclinationResidualDisturbedDeg)
        XCTAssertLessThan(profile.stationaryFieldMadSuspectMicroTesla,
                          profile.stationaryFieldMadDisturbedMicroTesla)
        XCTAssertLessThan(profile.pipelineDisagreementSuspectDeg, profile.pipelineDisagreementDisturbedDeg)
    }

    func testPeriodicSupportSamplesAchievable() {
        let achievable = Double(profile.stableWindowMinMs)
            * (profile.periodicOrientationRequestedHz / 2.0) / 1000.0
        XCTAssertGreaterThanOrEqual(
            achievable, Double(profile.minPeriodicSupportSamples),
            "The candidate gate tolerates a 50% callback shortfall. This invariant does not apply "
                + "to event-driven CLHeading, which has its own in-window anchor count (§12, R52).")
    }

    func testOrientationAgeOrdering() {
        XCTAssertLessThan(profile.orientationMaxAgeMs, profile.orientationInvalidAfterMs,
                          "Drop and invalidate are different thresholds.")
    }

    func testLocationFreshnessOrdering() {
        XCTAssertLessThanOrEqual(profile.freshLocationAtStartMaxAgeMs, profile.locationAtLockMaxAgeMs)
        XCTAssertLessThanOrEqual(profile.locationAtLockMaxAgeMs, profile.usableLocationMaxAgeMs)
    }

    func testSpaceWeatherOrdering() {
        XCTAssertLessThanOrEqual(profile.spaceWeatherAdvisoryKpMin,
                                 profile.spaceWeatherProfessionalSuppressKpMin)
        XCTAssertLessThan(profile.spaceWeatherProfessionalSuppressKpMin, profile.spaceWeatherRejectKpMin)
    }

    func testInvariantCheckerIsNotVacuous() throws {
        // Feed the checker a document with two broken invariants. The shipped file is never
        // edited to make a test pass (§37 rule 12); this proves the checker discriminates.
        var broken = rawTree!
        broken["referenceSeparationMarginDeg"] = profile.smallDeclinationAmbiguityMaxDeg + 1.0
        broken["orientationMaxAgeMs"] = profile.orientationInvalidAfterMs
        let data = try JSONSerialization.data(withJSONObject: broken)
        let brokenProfile = try JSONDecoder().decode(PrecisionProfile.self, from: data)

        let ids = ConfigurationInvariants.check(profile: brokenProfile, rawTree: broken)
            .map(\.invariantId)
        XCTAssertTrue(ids.contains("INV-02-REFERENCE-SEPARATION-ORDERING"), "got \(ids)")
        XCTAssertTrue(ids.contains("INV-08-ORIENTATION-AGE-ORDERING"), "got \(ids)")
    }
}
