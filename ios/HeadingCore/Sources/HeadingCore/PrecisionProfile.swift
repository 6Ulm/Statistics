import Foundation

/// SPEC.md §8 `config/precision-profile-v1.json`, typed.
///
/// Every gate in the engine reads a named key from this type; §18.5 forbids comparing a
/// gate against a numeric literal. There is deliberately **no** calibration-state property:
/// `boundCalibrationState` is derived at runtime from a §24 certification lookup
/// (§19.1, failure mode 32), and §8.1 asserts no key matching `/calibrationState/i` exists
/// anywhere in the document.
public struct PrecisionProfile: Decodable, Equatable {
    public let schemaVersion: String
    public let configVersion: String

    public let orientationMaxAgeMs: Int64
    public let orientationInvalidAfterMs: Int64
    public let freshLocationAtStartMaxAgeMs: Int64
    public let locationAtLockMaxAgeMs: Int64
    public let usableLocationMaxAgeMs: Int64
    public let locationJumpRequiresFreshFixKm: Double
    public let declinationEnvelopeProfessionalMaxDeg: Double
    public let declinationEnvelopeUsableMaxDeg: Double

    public let stableWindowMinMs: Int64
    public let acquisitionTimeoutMs: Int64
    public let periodicOrientationRequestedHz: Double
    public let minPeriodicSupportSamples: Int
    public let clHeadingMinSamplesPerStableWindow: Int
    public let minCircularResultantLength: Double
    public let angularSpeedP95MaxDegPerSec: Double
    public let linearAccelerationP95MaxG: Double
    public let circularResidualP95MaxDeg: Double

    public let flatModePitchAbsMaxDeg: Double
    public let flatModeRollAbsMaxDeg: Double
    public let flatFreehandPlacementBound95Deg: Double
    public let wallNormalElevationAbsMaxDeg: Double
    public let wallTopAxisFromVerticalMaxDeg: Double
    public let wallFreehandPlacementBound95Deg: Double

    public let targetNearZoneDeg: Double
    public let targetCenteringToleranceDeg: Double

    public let providerCrossCheckMaxDeg: Double
    public let referenceSeparationMarginDeg: Double
    public let smallDeclinationAmbiguityMaxDeg: Double
    public let transformAgreementMaxDeg: Double

    public let magneticMagnitudeResidualSuspectFraction: Double
    public let magneticMagnitudeResidualDisturbedFraction: Double
    public let inclinationResidualSuspectDeg: Double
    public let inclinationResidualDisturbedDeg: Double
    public let stationaryFieldMadSuspectMicroTesla: Double
    public let stationaryFieldMadDisturbedMicroTesla: Double
    public let pipelineDisagreementSuspectDeg: Double
    public let pipelineDisagreementDisturbedDeg: Double
    public let suspectInterferenceBound95Deg: Double
    public let recoveryCleanWindowMs: Int64

    public let minHorizontalIntensityNanoTesla: Double

    public let unknownDeviceFloor95Deg: Double
    public let professionalBound95MaxDeg: Double
    public let highBound95MaxDeg: Double
    public let usableBound95MaxDeg: Double
    public let lowConfidenceBound95MaxDeg: Double

    public let spaceWeatherAdvisoryKpMin: Double
    public let spaceWeatherProfessionalSuppressKpMin: Double
    public let spaceWeatherRejectKpMin: Double
    public let spaceWeatherCacheMaxAgeMs: Int64

    public let thermalRestrictionBlocksLock: Bool
    public let wirelessChargingBlocksGradeAboveUsable: Bool

    public let precisionScreenOrientation: String
    public let requireBoundaryStraddleReporting: Bool
    public let geomagneticModelId: String
    public let canonicalAltitudeReference: String
    public let declinationSigmaToBound95Factor: Double

    public static func load(contentsOf url: URL) throws -> PrecisionProfile {
        try JSONDecoder().decode(PrecisionProfile.self, from: Data(contentsOf: url))
    }

    /// The same document as an untyped tree, for the §8.1 whole-document key scan.
    public static func loadRawTree(contentsOf url: URL) throws -> [String: Any] {
        let object = try JSONSerialization.jsonObject(with: Data(contentsOf: url))
        guard let dictionary = object as? [String: Any] else {
            throw SharedArtifacts.ArtifactError.missingArtifact(url.lastPathComponent)
        }
        return dictionary
    }
}

/// Collects every property name appearing anywhere in `value`, at any nesting depth.
/// §8.1's first invariant is "no key matching `/calibrationState/i` exists **anywhere** in
/// the profile", so a nested object may not smuggle one in either.
public func collectPropertyNames(_ value: Any, into names: inout [String]) {
    if let dictionary = value as? [String: Any] {
        for (key, nested) in dictionary {
            names.append(key)
            collectPropertyNames(nested, into: &names)
        }
    } else if let array = value as? [Any] {
        for element in array { collectPropertyNames(element, into: &names) }
    }
}
