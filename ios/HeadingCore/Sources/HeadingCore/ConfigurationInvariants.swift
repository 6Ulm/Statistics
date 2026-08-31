import Foundation

/// A single §8.1 invariant that did not hold. `invariantId` is shared with the Kotlin and
/// Python implementations so a failure reads the same in all three runtimes.
public struct InvariantViolation: Equatable, CustomStringConvertible {
    public let invariantId: String
    public let requirement: String
    public let prevents: String
    public let detail: String

    public var description: String {
        "[\(invariantId)] \(requirement) -- observed: \(detail) -- prevents: \(prevents)"
    }
}

/// SPEC.md §8.1 "Enforced invariants": a build-time check of every row of that table.
///
/// §36 makes these part of Phase 0 and §33.1 runs them on every commit. They are
/// intentionally implemented before any core logic: each one prevents a specific silent
/// failure that is invisible from reading the gate table.
public enum ConfigurationInvariants {

    /// The literal regex from §8.1's first row.
    public static let calibrationStateKeyPattern = "calibrationState"

    public static func containsCalibrationStateKey(_ name: String) -> Bool {
        name.range(of: calibrationStateKeyPattern, options: .caseInsensitive) != nil
    }

    public static func check(profile: PrecisionProfile, rawTree: [String: Any]) -> [InvariantViolation] {
        var violations: [InvariantViolation] = []

        func require(
            _ id: String,
            _ holds: Bool,
            requirement: String,
            prevents: String,
            detail: @autoclosure () -> String
        ) {
            if !holds {
                violations.append(
                    InvariantViolation(invariantId: id, requirement: requirement,
                                       prevents: prevents, detail: detail())
                )
            }
        }

        var names: [String] = []
        collectPropertyNames(rawTree, into: &names)
        let offendingKeys = names.filter(containsCalibrationStateKey)
        require(
            "INV-01-NO-CALIBRATION-STATE-KEY", offendingKeys.isEmpty,
            requirement: "No key matching /calibrationState/i exists anywhere in the profile",
            prevents: "boundCalibrationState is derived from a §24 certification lookup (§19.1). "
                + "One editable value that turns every device Professional is the shortcut an agent "
                + "under pressure takes.",
            detail: "offending keys: \(offendingKeys)"
        )

        require(
            "INV-02-REFERENCE-SEPARATION-ORDERING",
            profile.referenceSeparationMarginDeg <= profile.smallDeclinationAmbiguityMaxDeg,
            requirement: "referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg",
            prevents: "Since rMag - rTrue <= abs(d), a margin above the ambiguity allowance creates "
                + "a declination dead band that always resolves UNVERIFIED with no visible cause (§11).",
            detail: "\(profile.referenceSeparationMarginDeg) vs \(profile.smallDeclinationAmbiguityMaxDeg)"
        )

        require(
            "INV-03-GRADE-THRESHOLD-ORDERING",
            profile.professionalBound95MaxDeg < profile.highBound95MaxDeg
                && profile.highBound95MaxDeg < profile.usableBound95MaxDeg
                && profile.usableBound95MaxDeg < profile.lowConfidenceBound95MaxDeg,
            requirement: "professionalBound95MaxDeg < highBound95MaxDeg < usableBound95MaxDeg "
                + "< lowConfidenceBound95MaxDeg",
            prevents: "Grade function must be total and ordered.",
            detail: "\(profile.professionalBound95MaxDeg), \(profile.highBound95MaxDeg), "
                + "\(profile.usableBound95MaxDeg), \(profile.lowConfidenceBound95MaxDeg)"
        )

        require(
            "INV-04-FREEHAND-CANNOT-REACH-PROFESSIONAL",
            profile.professionalBound95MaxDeg < profile.flatFreehandPlacementBound95Deg,
            requirement: "professionalBound95MaxDeg < flatFreehandPlacementBound95Deg",
            prevents: "Encodes in config that freehand cannot reach the top grade (§20). A future "
                + "edit breaking this trips the intended alarm.",
            detail: "\(profile.professionalBound95MaxDeg) vs \(profile.flatFreehandPlacementBound95Deg)"
        )

        require(
            "INV-05-DECLINATION-ENVELOPE-ORDERING",
            profile.declinationEnvelopeProfessionalMaxDeg <= profile.declinationEnvelopeUsableMaxDeg,
            requirement: "declinationEnvelopeProfessionalMaxDeg <= declinationEnvelopeUsableMaxDeg",
            prevents: "Ordered gates.",
            detail: "\(profile.declinationEnvelopeProfessionalMaxDeg) vs \(profile.declinationEnvelopeUsableMaxDeg)"
        )

        let suspectDisturbedPairs: [(String, Double, Double)] = [
            ("magnitude", profile.magneticMagnitudeResidualSuspectFraction,
             profile.magneticMagnitudeResidualDisturbedFraction),
            ("inclination", profile.inclinationResidualSuspectDeg, profile.inclinationResidualDisturbedDeg),
            ("stationaryMad", profile.stationaryFieldMadSuspectMicroTesla,
             profile.stationaryFieldMadDisturbedMicroTesla),
            ("pipeline", profile.pipelineDisagreementSuspectDeg, profile.pipelineDisagreementDisturbedDeg),
        ]
        for (name, suspect, disturbed) in suspectDisturbedPairs {
            require(
                "INV-06-SUSPECT-BELOW-DISTURBED-\(name)", suspect < disturbed,
                requirement: "suspect < disturbed for the \(name) pair",
                prevents: "A suspect threshold above disturbed makes SUSPECT unreachable.",
                detail: "suspect=\(suspect), disturbed=\(disturbed)"
            )
        }

        // §8.1: periodic support streams request 50 Hz and the gate tolerates a 50% callback
        // shortfall. This invariant does NOT apply to event-driven CLHeading; flat iOS has its
        // own in-window heading-anchor count (§12, R52).
        let achievableSupportSamples =
            Double(profile.stableWindowMinMs) * (profile.periodicOrientationRequestedHz / 2.0) / 1000.0
        require(
            "INV-07-PERIODIC-SUPPORT-SAMPLES-ACHIEVABLE",
            achievableSupportSamples >= Double(profile.minPeriodicSupportSamples),
            requirement: "stableWindowMinMs * (periodicOrientationRequestedHz / 2) / 1000 "
                + ">= minPeriodicSupportSamples",
            prevents: "Periodic support streams request 50 Hz; the candidate gate tolerates a 50% "
                + "callback shortfall. Does not apply to event-driven CLHeading.",
            detail: "achievable=\(achievableSupportSamples), required=\(profile.minPeriodicSupportSamples)"
        )

        require(
            "INV-08-ORIENTATION-AGE-ORDERING",
            profile.orientationMaxAgeMs < profile.orientationInvalidAfterMs,
            requirement: "orientationMaxAgeMs < orientationInvalidAfterMs",
            prevents: "Drop and invalidate are different thresholds.",
            detail: "\(profile.orientationMaxAgeMs) vs \(profile.orientationInvalidAfterMs)"
        )

        require(
            "INV-09-LOCATION-FRESHNESS-ORDERING",
            profile.freshLocationAtStartMaxAgeMs <= profile.locationAtLockMaxAgeMs
                && profile.locationAtLockMaxAgeMs <= profile.usableLocationMaxAgeMs,
            requirement: "freshLocationAtStartMaxAgeMs <= locationAtLockMaxAgeMs <= usableLocationMaxAgeMs",
            prevents: "Ordered freshness tiers.",
            detail: "\(profile.freshLocationAtStartMaxAgeMs), \(profile.locationAtLockMaxAgeMs), "
                + "\(profile.usableLocationMaxAgeMs)"
        )

        require(
            "INV-10-SPACE-WEATHER-ORDERING",
            profile.spaceWeatherAdvisoryKpMin <= profile.spaceWeatherProfessionalSuppressKpMin
                && profile.spaceWeatherProfessionalSuppressKpMin < profile.spaceWeatherRejectKpMin,
            requirement: "spaceWeatherAdvisoryKpMin <= spaceWeatherProfessionalSuppressKpMin "
                + "< spaceWeatherRejectKpMin",
            prevents: "Ordered advisory/suppression/refusal tiers.",
            detail: "\(profile.spaceWeatherAdvisoryKpMin), \(profile.spaceWeatherProfessionalSuppressKpMin), "
                + "\(profile.spaceWeatherRejectKpMin)"
        )

        return violations
    }
}
