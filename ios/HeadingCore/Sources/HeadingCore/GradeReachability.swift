import Foundation

/// SPEC.md §6 `QualityGrade`, plus the `notSupported` claim-vocabulary value the §8.1.1
/// analysis needs. Only the §6 cases may appear in telemetry or on a wire.
public enum QualityGrade: String, CaseIterable, Comparable {
    case professional = "PROFESSIONAL"
    case high = "HIGH"
    case usable = "USABLE"
    case lowConfidence = "LOW_CONFIDENCE"
    case invalid = "INVALID"
    case notSupported = "NOT_SUPPORTED"

    /// Strength order: a smaller index is a stronger claim.
    private var strengthIndex: Int {
        switch self {
        case .professional: return 0
        case .high: return 1
        case .usable: return 2
        case .lowConfidence: return 3
        case .invalid: return 4
        case .notSupported: return 5
        }
    }

    public static func < (lhs: QualityGrade, rhs: QualityGrade) -> Bool {
        lhs.strengthIndex < rhs.strengthIndex
    }

    public func isStrongerThan(_ other: QualityGrade) -> Bool { self < other }
}

/// SPEC.md §6 `PlacementMethod`.
public enum PlacementMethod: String, CaseIterable {
    case freehand = "FREEHAND"
    case wallFlushFreehand = "WALL_FLUSH_FREEHAND"
    case nonmagneticAlignmentJig = "NONMAGNETIC_ALIGNMENT_JIG"
    case surveyFixture = "SURVEY_FIXTURE"
}

/// SPEC.md §6 `MagneticState`.
public enum MagneticState: String, CaseIterable {
    case clean = "CLEAN"
    case suspect = "SUSPECT"
    case disturbed = "DISTURBED"
    case invalid = "INVALID"
    case unknown = "UNKNOWN"
}

/// Whether a §24 `CertificationRecord` exists for the exact certification key. A record
/// exists only for `CALIBRATED`; a miss on any component means `uncertified`.
public enum CertificationState: String, CaseIterable {
    case uncertified = "UNCERTIFIED"
    case certified = "CERTIFIED"
}

/// SPEC.md §20: grades come from `reportedBound95Deg`, on explicit half-open intervals so
/// the function is total. Grading on `instrumentBound95Deg` would advertise precision the
/// practitioner cannot physically realize (failure mode 30).
public func qualityGradeForReportedBound(_ reportedBound95Deg: Double,
                                         profile: PrecisionProfile) -> QualityGrade {
    precondition(reportedBound95Deg.isFinite && reportedBound95Deg >= 0,
                 "reportedBound95Deg must be a finite non-negative angle, got \(reportedBound95Deg)")
    if reportedBound95Deg <= profile.professionalBound95MaxDeg { return .professional }
    if reportedBound95Deg <= profile.highBound95MaxDeg { return .high }
    if reportedBound95Deg <= profile.usableBound95MaxDeg { return .usable }
    if reportedBound95Deg <= profile.lowConfidenceBound95MaxDeg { return .lowConfidence }
    return .invalid
}

/// SPEC.md §8.1.1 — the required build-time grade-reachability analysis.
///
/// Each placement method has a fixed instrument budget of
/// `usableBound95MaxDeg - placementBound95Deg`. Any single uncertainty term larger than
/// that budget makes a Precision Lock arithmetically impossible, no matter how good the
/// sensor is. The analysis computes the infimum of `reportedBound95Deg`: every §19 term
/// that can legitimately be zero is taken at zero, so if a claimed grade is unreachable
/// even in the best case it is unreachable everywhere.
public enum GradeReachability {

    public enum PlacementBoundStatus: String { case configured = "CONFIGURED", unmeasured = "UNMEASURED" }

    public struct ClaimsDocument: Decodable {
        public let schemaVersion: String
        public let claimsVersion: String
        public let appliesToConfigVersion: String
        public let purpose: String
        public let gradeVocabulary: [String]
        public let notes: [String]
        public let combinations: [Claim]
    }

    public struct Claim: Decodable {
        public let id: String
        public let measurementMode: String
        public let placementMethod: String
        public let placementBoundStatus: String
        public let certificationState: String
        public let magneticState: String
        public let claimedMaxGrade: String
        public let claimedLockReachable: Bool
        public let requiresDeviceFloorAtMostDeg: Double?
        public let specBasis: String
    }

    public struct Reachability {
        public let minimumReportedBound95Deg: Double?
        public let maxReachableGrade: QualityGrade
        public let lockReachable: Bool
        public let requiredDeviceFloorAtMostDeg: Double?
        public let explanation: String
    }

    /// A claim contradicted by the arithmetic. §37 rule 12: a finding, not an obstacle.
    public struct Finding: CustomStringConvertible {
        public let claimId: String
        public let problem: String
        public let claimed: String
        public let computed: String
        public var description: String {
            "[\(claimId)] \(problem) -- claimed: \(claimed) -- computed from the shipped constants: \(computed)"
        }
    }

    public static func loadClaims(contentsOf url: URL) throws -> ClaimsDocument {
        try JSONDecoder().decode(ClaimsDocument.self, from: Data(contentsOf: url))
    }

    /// The placement term, or nil when the shipped profile carries no measured bound.
    /// §18.5: "Placement uncertainty: finite bound from method ... **never zero**."
    public static func placementBound95Deg(_ method: PlacementMethod,
                                           profile: PrecisionProfile) -> Double? {
        switch method {
        case .freehand: return profile.flatFreehandPlacementBound95Deg
        case .wallFlushFreehand: return profile.wallFreehandPlacementBound95Deg
        // §29.5 makes these Phase 5 outputs. Inventing a value is the edit §8.1.1 forbids.
        case .nonmagneticAlignmentJig, .surveyFixture: return nil
        }
    }

    /// §19 interference term, or nil when the magnetic state rejects outright.
    public static func interferenceBound95Deg(_ state: MagneticState,
                                              profile: PrecisionProfile) -> Double? {
        switch state {
        case .clean: return 0.0
        case .suspect: return profile.suspectInterferenceBound95Deg
        case .disturbed, .invalid, .unknown: return nil
        }
    }

    public static func instrumentBudgetDeg(_ method: PlacementMethod,
                                           profile: PrecisionProfile) -> Double? {
        placementBound95Deg(method, profile: profile).map { profile.usableBound95MaxDeg - $0 }
    }

    public static func compute(placementMethod: PlacementMethod,
                               certificationState: CertificationState,
                               magneticState: MagneticState,
                               profile: PrecisionProfile,
                               certifiedDeviceFloor95Deg: Double? = nil) -> Reachability {
        guard let placement = placementBound95Deg(placementMethod, profile: profile) else {
            return Reachability(
                minimumReportedBound95Deg: nil,
                maxReachableGrade: .notSupported,
                lockReachable: false,
                requiredDeviceFloorAtMostDeg: nil,
                explanation: "\(placementMethod.rawValue) has no measured placement bound in "
                    + "\(profile.configVersion); §29.5 makes it a benchmark output and §18.5 forbids "
                    + "defaulting it to zero, so no grade is computable."
            )
        }
        guard let interference = interferenceBound95Deg(magneticState, profile: profile) else {
            return Reachability(
                minimumReportedBound95Deg: nil,
                maxReachableGrade: .invalid,
                lockReachable: false,
                requiredDeviceFloorAtMostDeg: nil,
                explanation: "MagneticState \(magneticState.rawValue) rejects outright in v1 "
                    + "(§16, §18.5); no measurement is produced, so no grade exists."
            )
        }

        let budget = profile.usableBound95MaxDeg - placement
        let requiredFloor = budget - interference

        switch certificationState {
        case .uncertified:
            let floor = profile.unknownDeviceFloor95Deg
            let minReported = min(180.0, floor + interference + placement)
            return Reachability(
                minimumReportedBound95Deg: minReported,
                maxReachableGrade: qualityGradeForReportedBound(minReported, profile: profile),
                lockReachable: minReported <= profile.usableBound95MaxDeg,
                requiredDeviceFloorAtMostDeg: requiredFloor,
                explanation: "unknownDeviceFloor95Deg=\(floor) + interference=\(interference) "
                    + "+ placement=\(placement) = \(minReported); instrument budget for "
                    + "\(placementMethod.rawValue) is \(budget)."
            )

        case .certified:
            if let floor = certifiedDeviceFloor95Deg {
                let minReported = min(180.0, floor + interference + placement)
                return Reachability(
                    minimumReportedBound95Deg: minReported,
                    maxReachableGrade: qualityGradeForReportedBound(minReported, profile: profile),
                    lockReachable: minReported <= profile.usableBound95MaxDeg,
                    requiredDeviceFloorAtMostDeg: requiredFloor,
                    explanation: "certified floor=\(floor) + interference=\(interference) "
                        + "+ placement=\(placement) = \(minReported)."
                )
            }
            // A device floor is strictly positive, so a required floor of zero or less means
            // no certification can make this combination lock.
            let lockPossible = requiredFloor > 0.0
            let bestCase = min(180.0, interference + placement)
            return Reachability(
                minimumReportedBound95Deg: lockPossible ? nil : bestCase,
                maxReachableGrade: lockPossible
                    ? .usable
                    : qualityGradeForReportedBound(bestCase, profile: profile),
                lockReachable: lockPossible,
                requiredDeviceFloorAtMostDeg: requiredFloor,
                explanation: "instrument budget for \(placementMethod.rawValue) is \(budget); after "
                    + "the \(magneticState.rawValue) interference term \(interference) a certified "
                    + "deviceFloor95Deg must be <= \(requiredFloor) to lock"
                    + (lockPossible ? "." : ", which no real device floor can satisfy.")
            )
        }
    }

    public static func verify(claims: ClaimsDocument, profile: PrecisionProfile) -> [Finding] {
        var findings: [Finding] = []

        if claims.appliesToConfigVersion != profile.configVersion {
            findings.append(Finding(
                claimId: claims.claimsVersion,
                problem: "The claims document targets a different configuration version, so its rows "
                    + "were never checked against these constants.",
                claimed: "appliesToConfigVersion=\(claims.appliesToConfigVersion)",
                computed: "configVersion=\(profile.configVersion)"))
        }

        for claim in claims.combinations {
            guard let claimedGrade = QualityGrade(rawValue: claim.claimedMaxGrade),
                  let declaredStatus = PlacementBoundStatus(rawValue: claim.placementBoundStatus),
                  let method = PlacementMethod(rawValue: claim.placementMethod) else {
                findings.append(Finding(claimId: claim.id,
                                        problem: "Unrecognized claim vocabulary.",
                                        claimed: "\(claim.claimedMaxGrade) / \(claim.placementBoundStatus) "
                                            + "/ \(claim.placementMethod)",
                                        computed: "not a declared enum value"))
                continue
            }

            let actualStatus: PlacementBoundStatus =
                placementBound95Deg(method, profile: profile) == nil ? .unmeasured : .configured
            if declaredStatus != actualStatus {
                findings.append(Finding(
                    claimId: claim.id,
                    problem: "The claim's placement-bound status disagrees with the shipped profile.",
                    claimed: "placementBoundStatus=\(declaredStatus.rawValue)",
                    computed: "profile \(profile.configVersion) has \(actualStatus.rawValue) for "
                        + method.rawValue))
                continue
            }

            if actualStatus == .unmeasured {
                if claimedGrade != .notSupported || claim.claimedLockReachable {
                    findings.append(Finding(
                        claimId: claim.id,
                        problem: "A placement method with no measured bound may claim no grade and no "
                            + "lock (§29.5; §35 'no grade above USABLE without a measured method').",
                        claimed: "claimedMaxGrade=\(claimedGrade.rawValue), "
                            + "claimedLockReachable=\(claim.claimedLockReachable)",
                        computed: "placement bound is UNMEASURED for \(method.rawValue)"))
                }
                continue
            }

            guard let certification = CertificationState(rawValue: claim.certificationState),
                  let magnetic = MagneticState(rawValue: claim.magneticState) else {
                findings.append(Finding(claimId: claim.id,
                                        problem: "Unrecognized certification or magnetic state.",
                                        claimed: "\(claim.certificationState) / \(claim.magneticState)",
                                        computed: "not a declared enum value"))
                continue
            }

            let computed = compute(placementMethod: method, certificationState: certification,
                                   magneticState: magnetic, profile: profile)

            if claimedGrade.isStrongerThan(computed.maxReachableGrade) {
                findings.append(Finding(
                    claimId: claim.id,
                    problem: "The claimed maximum grade is arithmetically forbidden by the shipped constants.",
                    claimed: "claimedMaxGrade=\(claimedGrade.rawValue) (\(claim.specBasis))",
                    computed: "maxReachableGrade=\(computed.maxReachableGrade.rawValue); \(computed.explanation)"))
            }

            if claim.claimedLockReachable != computed.lockReachable {
                findings.append(Finding(
                    claimId: claim.id,
                    problem: "The claim disagrees with the arithmetic about whether a Precision Lock is "
                        + "reachable at all.",
                    claimed: "claimedLockReachable=\(claim.claimedLockReachable)",
                    computed: "lockReachable=\(computed.lockReachable); \(computed.explanation)"))
            }

            if let declaredFloor = claim.requiresDeviceFloorAtMostDeg,
               let computedFloor = computed.requiredDeviceFloorAtMostDeg,
               abs(declaredFloor - computedFloor) > 1e-9 {
                findings.append(Finding(
                    claimId: claim.id,
                    problem: "The device floor the claim says is required does not match the instrument "
                        + "budget the constants leave.",
                    claimed: "requiresDeviceFloorAtMostDeg=\(declaredFloor)",
                    computed: "required floor=\(computedFloor); \(computed.explanation)"))
            }

            if certification == .certified && claim.claimedLockReachable
                && claim.requiresDeviceFloorAtMostDeg == nil {
                findings.append(Finding(
                    claimId: claim.id,
                    problem: "A CERTIFIED lock claim must state the device floor it depends on; §8.1.1 "
                        + "makes deviceFloor95Deg an output of the benchmark, not an assumption.",
                    claimed: "requiresDeviceFloorAtMostDeg absent",
                    computed: "required floor=\(String(describing: computed.requiredDeviceFloorAtMostDeg))"))
            }
        }
        return findings
    }
}
