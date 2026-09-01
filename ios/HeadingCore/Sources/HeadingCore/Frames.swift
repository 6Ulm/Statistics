import Foundation

/// SPEC.md §3, §11.1, §14 — frames, quaternion order, and mode axis projection.
///
/// Canonical orientation basis is **REFERENCE_ENU**: `+Z` up, `+Y` toward the north reference
/// named by `providerReferenceContract`, and `+X = +Y × +Z`. With contract `TRUE` this is
/// geographic ENU; with `MAGNETIC` it is magnetic-east/magnetic-north/up. A magnetic basis is
/// never relabelled geographic ENU merely because its axis order is east/north/up — which is
/// why `CanonicalAttitude` carries the contract beside the quaternion.
///
/// Quaternion types name component order **and** transform direction: the canonical field is
/// `attitudeQuaternionDeviceToReferenceEnuXYZW`. A bare 4-element array never travels beyond a
/// provider adapter, so `Quaternion` is the only shape core code accepts.
///
/// `atan2` appears here twice, both allowlisted by §33.1 as bearing projections: the horizontal
/// bearing of a reference axis, and the matrix→quaternion recovery. Neither is a signed
/// circular difference; that has exactly one implementation, in `CircularMath`.
///
/// > Warning: this file has never been compiled — see `docs/IMPLEMENTATION_NOTES.md` D-3.
public enum Frames {

    public enum FrameError: Error, CustomStringConvertible {
        case nonFiniteComponent(String)
        case zeroLengthVector(String)
        case notAUnitQuaternion(Double)
        case reflectingAxisMap(String)
        case duplicateAxisSelection(String)

        public var description: String {
            switch self {
            case .nonFiniteComponent(let what):
                return "\(what) components must be finite"
            case .zeroLengthVector(let what):
                return "cannot \(what) a zero-length vector"
            case .notAUnitQuaternion(let norm):
                return "canonical attitude requires a unit quaternion; got norm \(norm). "
                    + "Normalize at the adapter boundary, not in core math."
            case .reflectingAxisMap(let frame):
                return "\(frame): the native→ENU axis map is a reflection (determinant -1), "
                    + "which would mirror every bearing"
            case .duplicateAxisSelection(let frame):
                return "\(frame): east/north/up must select three distinct native axes"
            }
        }
    }

    /// The reference axis has no usable horizontal projection, so its bearing does not exist.
    ///
    /// §14: "Reject if the horizontal projection is ill-conditioned". The *practical*
    /// conditioning gate is the mode's configured pose limit, evaluated against
    /// `AxisBearing.elevationDeg`; no new numeric constant is introduced.
    public struct SingularProjection: Error, CustomStringConvertible {
        public let reason: String
        public var description: String { reason }
    }

    /// §3: the portrait top edge is device `+y`; the outward screen normal is device `+z`.
    public static let modeReferenceVectors: [MeasurementMode: Vector3] = [
        .flatTopEdge: Vector3(x: 0.0, y: 1.0, z: 0.0),
        .wallFlushBack: Vector3(x: 0.0, y: 0.0, z: 1.0),
    ]

    /// §14/§18.5: the configured pose limit that bounds each mode's reference-axis elevation.
    ///
    /// Flat mode's top edge tilts with pitch, so `flatModePitchAbsMaxDeg` bounds it; wall mode
    /// states its own limit directly. No new constant is introduced — the conditioning gate is
    /// the pose gate, read from configuration by name (§18.5 forbids a numeric literal here).
    public static let modeAxisElevationGateKey: [MeasurementMode: String] = [
        .flatTopEdge: "flatModePitchAbsMaxDeg",
        .wallFlushBack: "wallNormalElevationAbsMaxDeg",
    ]

    /// Convert a provider-native attitude to canonical device → REFERENCE_ENU (R49).
    ///
    /// Both the transform **direction** and the axis **convention** are converted: a
    /// `referenceToDevice` quaternion is conjugated first, then the native reference axes are
    /// remapped to `(east, north, up)`. The remap is composed as a rotation, so the result is a
    /// single canonical quaternion rather than a matrix the caller might apply in the wrong
    /// order.
    public static func canonicalizeNativeAttitude(
        _ nativeQuaternion: Quaternion,
        convention: NativeAttitudeConvention
    ) throws -> CanonicalAttitude {
        let unit = try nativeQuaternion.normalized()
        let deviceToNative = convention.direction == .deviceToReference ? unit : unit.conjugate()
        let permutation = try quaternionFromRotationMatrix(convention.permutationMatrix())
        return try CanonicalAttitude(
            quaternionDeviceToReferenceEnuXYZW: permutation.multiplied(by: deviceToNative)
                .normalized(),
            referenceContract: convention.referenceContract,
            nativeFrame: convention.frame
        )
    }

    /// Direct quaternion-vector rotation: `v' = v + 2w(u × v) + 2(u × (u × v))`.
    ///
    /// One of the two independent extraction routes §11.1 requires for
    /// `transformAgreementDeg`; `rotateVectorByMatrix` is the other.
    public static func rotateVectorByQuaternion(
        _ quaternion: Quaternion,
        _ vector: Vector3
    ) throws -> EnuVector {
        let u = try quaternion.normalized()
        let cx = u.y * vector.z - u.z * vector.y
        let cy = u.z * vector.x - u.x * vector.z
        let cz = u.x * vector.y - u.y * vector.x

        let ccx = u.y * cz - u.z * cy
        let ccy = u.z * cx - u.x * cz
        let ccz = u.x * cy - u.y * cx

        return try EnuVector(
            east: vector.x + 2.0 * u.w * cx + 2.0 * ccx,
            north: vector.y + 2.0 * u.w * cy + 2.0 * ccy,
            up: vector.z + 2.0 * u.w * cz + 2.0 * ccz
        )
    }

    /// The 3×3 rotation matrix for the same transform direction as `quaternion`.
    public static func rotationMatrixFromQuaternion(
        _ quaternion: Quaternion
    ) throws -> [[Double]] {
        let u = try quaternion.normalized()
        return [
            [
                1.0 - 2.0 * (u.y * u.y + u.z * u.z),
                2.0 * (u.x * u.y - u.z * u.w),
                2.0 * (u.x * u.z + u.y * u.w),
            ],
            [
                2.0 * (u.x * u.y + u.z * u.w),
                1.0 - 2.0 * (u.x * u.x + u.z * u.z),
                2.0 * (u.y * u.z - u.x * u.w),
            ],
            [
                2.0 * (u.x * u.z - u.y * u.w),
                2.0 * (u.y * u.z + u.x * u.w),
                1.0 - 2.0 * (u.x * u.x + u.y * u.y),
            ],
        ]
    }

    /// The rotation-matrix extraction route (§11.1's second, independent implementation).
    public static func rotateVectorByMatrix(
        _ matrix: [[Double]],
        _ vector: Vector3
    ) throws -> EnuVector {
        let components = [vector.x, vector.y, vector.z]
        func row(_ index: Int) -> Double {
            (0..<3).reduce(0.0) { $0 + matrix[index][$1] * components[$1] }
        }
        return try EnuVector(east: row(0), north: row(1), up: row(2))
    }

    /// Shepperd's method: pick the largest divisor so no branch divides by ~0.
    ///
    /// Used to compose the native→ENU axis permutation into the canonical quaternion, and to
    /// close the matrix↔quaternion round trip the golden vectors check.
    public static func quaternionFromRotationMatrix(_ m: [[Double]]) throws -> Quaternion {
        let trace = m[0][0] + m[1][1] + m[2][2]
        let candidate: Quaternion
        if trace > 0.0 {
            let s = (trace + 1.0).squareRoot() * 2.0
            candidate = try Quaternion(
                x: (m[2][1] - m[1][2]) / s,
                y: (m[0][2] - m[2][0]) / s,
                z: (m[1][0] - m[0][1]) / s,
                w: 0.25 * s
            )
        } else if m[0][0] > m[1][1] && m[0][0] > m[2][2] {
            let s = (1.0 + m[0][0] - m[1][1] - m[2][2]).squareRoot() * 2.0
            candidate = try Quaternion(
                x: 0.25 * s,
                y: (m[0][1] + m[1][0]) / s,
                z: (m[0][2] + m[2][0]) / s,
                w: (m[2][1] - m[1][2]) / s
            )
        } else if m[1][1] > m[2][2] {
            let s = (1.0 + m[1][1] - m[0][0] - m[2][2]).squareRoot() * 2.0
            candidate = try Quaternion(
                x: (m[0][1] + m[1][0]) / s,
                y: 0.25 * s,
                z: (m[1][2] + m[2][1]) / s,
                w: (m[0][2] - m[2][0]) / s
            )
        } else {
            let s = (1.0 + m[2][2] - m[0][0] - m[1][1]).squareRoot() * 2.0
            candidate = try Quaternion(
                x: (m[0][2] + m[2][0]) / s,
                y: (m[1][2] + m[2][1]) / s,
                z: 0.25 * s,
                w: (m[1][0] - m[0][1]) / s
            )
        }
        return try candidate.normalized()
    }

    /// §9 `deviceVectorToReferenceEnu`. Accepts only a canonical attitude (R49).
    public static func deviceVectorToReferenceEnu(
        _ attitude: CanonicalAttitude,
        _ deviceVector: Vector3
    ) throws -> EnuVector {
        try rotateVectorByQuaternion(
            attitude.quaternionDeviceToReferenceEnuXYZW, deviceVector
        )
    }

    /// §14: `normalize360(degrees(atan2(east, north)))`.
    ///
    /// Throws `SingularProjection` on an exactly-zero horizontal projection rather than
    /// returning the `atan2(0, 0)` zero that looks like north (failure mode 6).
    public static func enuBearingDeg(_ vector: EnuVector) throws -> Double {
        guard vector.horizontalNorm != 0.0 else {
            throw SingularProjection(
                reason: "the axis is exactly vertical in REFERENCE_ENU; its horizontal bearing "
                    + "does not exist"
            )
        }
        return try CircularMath.normalize360(
            atan2(vector.east, vector.north) * 180.0 / Double.pi
        )
    }

    /// Elevation above the horizontal plane, positive up, in `[-90, 90]`.
    ///
    /// The engine compares this against the mode's configured pose gate; it is the conditioning
    /// measure the spec's "ill-conditioned" wording refers to.
    public static func enuElevationDeg(_ vector: EnuVector) throws -> Double {
        let norm = (vector.east * vector.east + vector.north * vector.north
            + vector.up * vector.up).squareRoot()
        guard norm != 0.0 else { throw FrameError.zeroLengthVector("take the elevation of") }
        return asin(min(1.0, max(-1.0, vector.up / norm))) * 180.0 / Double.pi
    }

    /// §9 `modeReferenceVectorHeadingDeg` — the active mode's axis, never another axis.
    ///
    /// §11.1: if the active reference-axis projection is singular or ill-conditioned, reject
    /// the pose; never resolve the north reference on a convenient different axis and transfer
    /// that label.
    public static func modeReferenceVectorHeadingDeg(
        _ attitude: CanonicalAttitude,
        mode: MeasurementMode
    ) throws -> AxisBearing {
        let enu = try deviceVectorToReferenceEnu(attitude, modeReferenceVectors[mode]!)
        return AxisBearing(
            headingDeg: try enuBearingDeg(enu),
            elevationDeg: try enuElevationDeg(enu),
            horizontalNorm: enu.horizontalNorm
        )
    }

    /// §14: reject an ill-conditioned reference-axis projection rather than bearing it.
    ///
    /// An exactly-vertical axis is the textbook singularity, but floating point rarely delivers
    /// it: rotating device `+y` into a wall pose leaves a horizontal projection of `2.2e-16`,
    /// from which `atan2` returns a confident, arbitrary `180°`. The conditioning decision is
    /// therefore made on the axis **elevation** against the mode's configured pose limit, and
    /// the exact-zero check remains only as the degenerate special case.
    public static func modeAxisBearingOrReject(
        _ attitude: CanonicalAttitude,
        mode: MeasurementMode,
        maxAxisElevationAbsDeg: Double
    ) throws -> AxisBearing {
        let enu = try deviceVectorToReferenceEnu(attitude, modeReferenceVectors[mode]!)
        let elevation = try enuElevationDeg(enu)
        guard abs(elevation) <= maxAxisElevationAbsDeg else {
            throw SingularProjection(
                reason: "\(mode.wire) reference axis is \(elevation)° from horizontal, beyond "
                    + "the configured \(maxAxisElevationAbsDeg)° limit; its horizontal bearing "
                    + "is ill-conditioned and MUST NOT be resolved on a different axis "
                    + "(§11.1, §14)"
            )
        }
        return AxisBearing(
            headingDeg: try enuBearingDeg(enu),
            elevationDeg: elevation,
            horizontalNorm: enu.horizontalNorm
        )
    }

    /// §11.1/§16.1 `transformAgreementDeg` for two same-observation extraction routes.
    ///
    /// A large value is a code fault — frame transform, quaternion order, axis selection or
    /// remapping — and MUST NOT contribute to `MagneticState`: telling a user to move away from
    /// metal because the wall-mode quaternion has a swapped axis is a failure that survives a
    /// long time in the field.
    public static func transformAgreementDeg(
        _ attitude: CanonicalAttitude,
        mode: MeasurementMode
    ) throws -> Double {
        let deviceVector = modeReferenceVectors[mode]!
        let quaternion = attitude.quaternionDeviceToReferenceEnuXYZW
        let direct = try enuBearingDeg(rotateVectorByQuaternion(quaternion, deviceVector))
        let viaMatrix = try enuBearingDeg(
            rotateVectorByMatrix(rotationMatrixFromQuaternion(quaternion), deviceVector)
        )
        return try CircularMath.absoluteCircularDifferenceDeg(direct, viaMatrix)
    }

    /// §14: sitting (坐) is exactly `normalize360(facing + 180)`, computed only on request.
    ///
    /// Kept as a named function so the derived opposite is always labelled as such. Reporting
    /// the wrong one is a 180° error that looks entirely plausible on a dial (failure mode 18).
    public static func sittingFromFacingDeg(_ facingDeg: Double) throws -> Double {
        try CircularMath.normalize360(facingDeg + 180.0)
    }
}

/// A vector in the **device** frame: `+x` right, `+y` toward the portrait top edge, `+z` out of
/// the screen.
public struct Vector3: Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let z: Double

    public init(x: Double, y: Double, z: Double) {
        self.x = x
        self.y = y
        self.z = z
    }
}

/// A vector in project REFERENCE_ENU. Named components, never a positional triple.
public struct EnuVector: Equatable, Sendable {
    public let east: Double
    public let north: Double
    public let up: Double

    public var horizontalNorm: Double { (east * east + north * north).squareRoot() }

    public init(east: Double, north: Double, up: Double) throws {
        guard east.isFinite, north.isFinite, up.isFinite else {
            throw Frames.FrameError.nonFiniteComponent("EnuVector")
        }
        self.east = east
        self.north = north
        self.up = up
    }
}

/// A unit quaternion in explicit `(x, y, z, w)` component order.
///
/// Failure mode 5 is the `wxyz`/`xyzw` swap, non-normalized input, active/passive inversion and
/// multiplication-order error. The type names the order; the transform direction is named by
/// whatever field or parameter holds it, never inferred.
public struct Quaternion: Equatable, Sendable {
    public let x: Double
    public let y: Double
    public let z: Double
    public let w: Double

    public var norm: Double { (x * x + y * y + z * z + w * w).squareRoot() }

    public init(x: Double, y: Double, z: Double, w: Double) throws {
        guard x.isFinite, y.isFinite, z.isFinite, w.isFinite else {
            throw Frames.FrameError.nonFiniteComponent("Quaternion")
        }
        self.x = x
        self.y = y
        self.z = z
        self.w = w
    }

    public func normalized() throws -> Quaternion {
        let n = norm
        guard n != 0.0 else { throw Frames.FrameError.zeroLengthVector("normalize") }
        return try Quaternion(x: x / n, y: y / n, z: z / n, w: w / n)
    }

    /// The inverse rotation for a unit quaternion — i.e. the opposite transform direction.
    public func conjugate() -> Quaternion {
        // Safe: negation of finite components stays finite.
        try! Quaternion(x: -x, y: -y, z: -z, w: w)
    }

    /// Hamilton product `self ⊗ other`. Composition order is explicit so it cannot be reversed
    /// by accident: applying the result to a vector applies `other` first, then `self`.
    public func multiplied(by other: Quaternion) throws -> Quaternion {
        try Quaternion(
            x: w * other.x + x * other.w + y * other.z - z * other.y,
            y: w * other.y - x * other.z + y * other.w + z * other.x,
            z: w * other.z + x * other.y - y * other.x + z * other.w,
            w: w * other.w - x * other.x - y * other.y - z * other.z
        )
    }

    public static func identity() -> Quaternion {
        try! Quaternion(x: 0.0, y: 0.0, z: 0.0, w: 1.0)
    }
}

/// Which way a provider's attitude quaternion transforms.
///
/// Naming this is not pedantry: a transposed attitude is failure mode 5 and produces a plausible
/// bearing, never a crash.
public enum TransformDirection: String, Sendable {
    case deviceToReference = "DEVICE_TO_REFERENCE"
    case referenceToDevice = "REFERENCE_TO_DEVICE"
}

/// Provider-native earth-axis conventions, retained as provenance (§5, §11.1, R49).
public enum NativeAttitudeFrame: String, Sendable, CaseIterable {
    case googleFopEnu = "GOOGLE_FOP_ENU"
    case androidRotationVectorEnu = "ANDROID_ROTATION_VECTOR_ENU"
    case coreMotionXTrueNorthZVertical = "CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL"
    case replayReferenceEnu = "REPLAY_REFERENCE_ENU"

    public var wire: String { rawValue }
}

/// Whether a declared native convention has been checked against the pinned SDK.
///
/// §37 rule 4 requires verifying the **installed** API signature and behaviour rather than
/// assuming a documentation page is unchanged, and §11.1 forbids inferring permutation,
/// transpose or signs from yaw intuition. Phase 1 has no SDK, so conventions it cannot check are
/// marked `declaredUnverified` and the Phase 2 adapter must confirm them with physical
/// N/E/S/W/up poses before the sample enters production.
public enum ConventionVerification: String, Sendable {
    case declaredUnverified = "DECLARED_UNVERIFIED"
    case verifiedAgainstPinnedSdk = "VERIFIED_AGAINST_PINNED_SDK"
}

/// One canonical axis expressed as a signed native axis: `sign * native[index]`.
public struct AxisSelector: Equatable, Sendable {
    public let index: Int
    public let sign: Double

    public init(_ index: Int, _ sign: Double) {
        precondition((0...2).contains(index), "AxisSelector.index must be 0, 1 or 2")
        precondition(sign == 1.0 || sign == -1.0, "AxisSelector.sign must be +1.0 or -1.0")
        self.index = index
        self.sign = sign
    }
}

/// A provider's declared attitude convention: axis permutation plus transform direction.
///
/// The permutation must be a proper rotation (determinant `+1`); a reflection would silently
/// mirror every bearing.
public struct NativeAttitudeConvention: Sendable {
    public let frame: NativeAttitudeFrame
    public let east: AxisSelector
    public let north: AxisSelector
    public let up: AxisSelector
    public let direction: TransformDirection
    public let referenceContract: ProviderReferenceContract
    public let verification: ConventionVerification
    public let note: String

    public init(
        frame: NativeAttitudeFrame,
        east: AxisSelector,
        north: AxisSelector,
        up: AxisSelector,
        direction: TransformDirection,
        referenceContract: ProviderReferenceContract,
        verification: ConventionVerification,
        note: String = ""
    ) throws {
        self.frame = frame
        self.east = east
        self.north = north
        self.up = up
        self.direction = direction
        self.referenceContract = referenceContract
        self.verification = verification
        self.note = note

        guard Set([east.index, north.index, up.index]).count == 3 else {
            throw Frames.FrameError.duplicateAxisSelection(frame.wire)
        }
        guard abs(permutationDeterminant() - 1.0) <= 1e-12 else {
            throw Frames.FrameError.reflectingAxisMap(frame.wire)
        }
    }

    /// Rows `(east, north, up)` over native columns `(0, 1, 2)`.
    public func permutationMatrix() -> [[Double]] {
        [east, north, up].map { selector in
            (0..<3).map { column in column == selector.index ? selector.sign : 0.0 }
        }
    }

    public func permutationDeterminant() -> Double {
        let m = permutationMatrix()
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/// §5 `ProviderAttitudeSample` attitude: always device → project REFERENCE_ENU.
///
/// `nativeFrame` is provenance only. `referenceContract` says whether the horizontal axes are
/// true- or magnetic-referenced; the axes alone never carry that claim.
public struct CanonicalAttitude: Sendable {
    public let quaternionDeviceToReferenceEnuXYZW: Quaternion
    public let referenceContract: ProviderReferenceContract
    public let nativeFrame: NativeAttitudeFrame

    public init(
        quaternionDeviceToReferenceEnuXYZW: Quaternion,
        referenceContract: ProviderReferenceContract,
        nativeFrame: NativeAttitudeFrame
    ) throws {
        let norm = quaternionDeviceToReferenceEnuXYZW.norm
        guard abs(norm - 1.0) <= 1e-9 else { throw Frames.FrameError.notAUnitQuaternion(norm) }
        self.quaternionDeviceToReferenceEnuXYZW = quaternionDeviceToReferenceEnuXYZW
        self.referenceContract = referenceContract
        self.nativeFrame = nativeFrame
    }
}

/// A reference axis projected into REFERENCE_ENU: its bearing and its conditioning.
public struct AxisBearing: Equatable, Sendable {
    public let headingDeg: Double
    public let elevationDeg: Double
    public let horizontalNorm: Double
}

/// The declared native conventions, keyed by frame.
public enum NativeConventions {

    /// §13: `getAttitude()` is `[qx, qy, qz, qw]` mapping device → ENU. The axis order already
    /// matches project REFERENCE_ENU, so the permutation is the identity. The true-vs-magnetic
    /// ambiguity lives in the contract, not in the axes (§11).
    public static let googleFop = try! NativeAttitudeConvention(
        frame: .googleFopEnu,
        east: AxisSelector(0, 1.0),
        north: AxisSelector(1, 1.0),
        up: AxisSelector(2, 1.0),
        direction: .deviceToReference,
        referenceContract: .trueIfDeclinationAvailableElseMagnetic,
        verification: .declaredUnverified,
        note: "§13. Confirm against the pinned Play services build in Phase 2."
    )

    /// §11.1: Android `TYPE_ROTATION_VECTOR` normalizes to the same axis order with an explicit
    /// `MAGNETIC` contract; the app applies WMM declination exactly once, later.
    public static let androidRotationVector = try! NativeAttitudeConvention(
        frame: .androidRotationVectorEnu,
        east: AxisSelector(0, 1.0),
        north: AxisSelector(1, 1.0),
        up: AxisSelector(2, 1.0),
        direction: .deviceToReference,
        referenceContract: .magnetic,
        verification: .declaredUnverified,
        note: "§13: obtained through SensorManager.getRotationMatrixFromVector."
    )

    /// Core Motion `.xTrueNorthZVertical`: native `+X` true north, `+Z` vertical, hence a
    /// right-handed native `+Y` pointing west and `east = -native_y`. **Declared, not
    /// verified:** §11.1 requires the adapter to prove both the axis convention and the
    /// transform direction with N/E/S/W/up golden vectors against the pinned SDK, which Phase 1
    /// cannot run.
    public static let coreMotionTrueNorth = try! NativeAttitudeConvention(
        frame: .coreMotionXTrueNorthZVertical,
        east: AxisSelector(1, -1.0),
        north: AxisSelector(0, 1.0),
        up: AxisSelector(2, 1.0),
        direction: .deviceToReference,
        referenceContract: .trueReference,
        verification: .declaredUnverified,
        note: "R49: both the axis permutation AND the transform direction MUST be confirmed "
            + "against the pinned Core Motion SDK with physical N/E/S/W/up poses in Phase 2."
    )

    /// Replay fixtures are authored directly in project REFERENCE_ENU.
    public static let replay = try! NativeAttitudeConvention(
        frame: .replayReferenceEnu,
        east: AxisSelector(0, 1.0),
        north: AxisSelector(1, 1.0),
        up: AxisSelector(2, 1.0),
        direction: .deviceToReference,
        referenceContract: .trueReference,
        verification: .verifiedAgainstPinnedSdk,
        note: "Fixture data is authored in canonical REFERENCE_ENU by definition."
    )

    public static let byFrame: [NativeAttitudeFrame: NativeAttitudeConvention] = [
        .googleFopEnu: googleFop,
        .androidRotationVectorEnu: androidRotationVector,
        .coreMotionXTrueNorthZVertical: coreMotionTrueNorth,
        .replayReferenceEnu: replay,
    ]
}
