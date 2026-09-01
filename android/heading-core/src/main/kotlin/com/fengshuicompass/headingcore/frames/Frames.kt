package com.fengshuicompass.headingcore.frames

import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.ProviderReferenceContract
import kotlin.math.abs
import kotlin.math.asin
import kotlin.math.atan2
import kotlin.math.hypot
import kotlin.math.sqrt

/**
 * SPEC.md §3, §11.1, §14 — frames, quaternion order, and mode axis projection.
 *
 * Canonical orientation basis is **REFERENCE_ENU**: `+Z` up, `+Y` toward the north reference
 * named by `providerReferenceContract`, and `+X = +Y x +Z`. With contract `TRUE` this is
 * geographic ENU; with `MAGNETIC` it is magnetic-east/magnetic-north/up. A magnetic basis is
 * never relabelled geographic ENU merely because its axis order is east/north/up — which is
 * why [CanonicalAttitude] carries the contract beside the quaternion.
 *
 * Quaternion types name component order **and** transform direction: the canonical field is
 * `attitudeQuaternionDeviceToReferenceEnuXYZW`. A bare 4-element array never travels beyond a
 * provider adapter, so [Quaternion] is the only shape core code accepts.
 *
 * `atan2` appears here twice, both allowlisted by §33.1 as bearing projections: the horizontal
 * bearing of a reference axis, and the matrix->quaternion recovery. Neither is a signed
 * circular difference; that has exactly one implementation, in [CircularMath].
 */
public object Frames {

    /** §3: the portrait top edge is device `+y`; the outward screen normal is device `+z`. */
    public val modeReferenceVectors: Map<MeasurementMode, Vector3> = mapOf(
        MeasurementMode.FLAT_TOP_EDGE to Vector3(0.0, 1.0, 0.0),
        MeasurementMode.WALL_FLUSH_BACK to Vector3(0.0, 0.0, 1.0),
    )

    /**
     * §14/§18.5: the configured pose limit that bounds each mode's reference-axis elevation.
     *
     * Flat mode's top edge tilts with pitch, so `flatModePitchAbsMaxDeg` bounds it; wall mode
     * states its own limit directly. No new constant is introduced — the conditioning gate is
     * the pose gate, read from configuration by name (§18.5 forbids a numeric literal here).
     */
    public val modeAxisElevationGateKey: Map<MeasurementMode, String> = mapOf(
        MeasurementMode.FLAT_TOP_EDGE to "flatModePitchAbsMaxDeg",
        MeasurementMode.WALL_FLUSH_BACK to "wallNormalElevationAbsMaxDeg",
    )

    /**
     * Convert a provider-native attitude to canonical device -> REFERENCE_ENU (R49).
     *
     * Both the transform **direction** and the axis **convention** are converted: a
     * `REFERENCE_TO_DEVICE` quaternion is conjugated first, then the native reference axes
     * are remapped to `(east, north, up)`. The remap is composed as a rotation, so the result
     * is a single canonical quaternion rather than a matrix the caller might apply in the
     * wrong order.
     */
    public fun canonicalizeNativeAttitude(
        nativeQuaternion: Quaternion,
        convention: NativeAttitudeConvention,
    ): CanonicalAttitude {
        val unit = nativeQuaternion.normalized()
        val deviceToNative =
            if (convention.direction == TransformDirection.DEVICE_TO_REFERENCE) {
                unit
            } else {
                unit.conjugate()
            }
        val permutation = quaternionFromRotationMatrix(convention.permutationMatrix())
        return CanonicalAttitude(
            quaternionDeviceToReferenceEnuXYZW =
                permutation.multipliedBy(deviceToNative).normalized(),
            referenceContract = convention.referenceContract,
            nativeFrame = convention.frame,
        )
    }

    /**
     * Direct quaternion-vector rotation: `v' = v + 2w(u x v) + 2(u x (u x v))`.
     *
     * One of the two independent extraction routes §11.1 requires for
     * `transformAgreementDeg`; [rotateVectorByMatrix] is the other.
     */
    public fun rotateVectorByQuaternion(quaternion: Quaternion, vector: Vector3): EnuVector {
        val unit = quaternion.normalized()
        val cx = unit.y * vector.z - unit.z * vector.y
        val cy = unit.z * vector.x - unit.x * vector.z
        val cz = unit.x * vector.y - unit.y * vector.x

        val ccx = unit.y * cz - unit.z * cy
        val ccy = unit.z * cx - unit.x * cz
        val ccz = unit.x * cy - unit.y * cx

        return EnuVector(
            east = vector.x + 2.0 * unit.w * cx + 2.0 * ccx,
            north = vector.y + 2.0 * unit.w * cy + 2.0 * ccy,
            up = vector.z + 2.0 * unit.w * cz + 2.0 * ccz,
        )
    }

    /** The 3x3 rotation matrix for the same transform direction as [quaternion]. */
    public fun rotationMatrixFromQuaternion(quaternion: Quaternion): List<List<Double>> {
        val u = quaternion.normalized()
        return listOf(
            listOf(
                1.0 - 2.0 * (u.y * u.y + u.z * u.z),
                2.0 * (u.x * u.y - u.z * u.w),
                2.0 * (u.x * u.z + u.y * u.w),
            ),
            listOf(
                2.0 * (u.x * u.y + u.z * u.w),
                1.0 - 2.0 * (u.x * u.x + u.z * u.z),
                2.0 * (u.y * u.z - u.x * u.w),
            ),
            listOf(
                2.0 * (u.x * u.z - u.y * u.w),
                2.0 * (u.y * u.z + u.x * u.w),
                1.0 - 2.0 * (u.x * u.x + u.y * u.y),
            ),
        )
    }

    /** The rotation-matrix extraction route (§11.1's second, independent implementation). */
    public fun rotateVectorByMatrix(matrix: List<List<Double>>, vector: Vector3): EnuVector {
        val components = listOf(vector.x, vector.y, vector.z)
        fun row(index: Int): Double = (0..2).sumOf { matrix[index][it] * components[it] }
        return EnuVector(east = row(0), north = row(1), up = row(2))
    }

    /**
     * Shepperd's method: pick the largest divisor so no branch divides by ~0.
     *
     * Used to compose the native->ENU axis permutation into the canonical quaternion, and to
     * close the matrix<->quaternion round trip the golden vectors check.
     */
    public fun quaternionFromRotationMatrix(m: List<List<Double>>): Quaternion {
        val trace = m[0][0] + m[1][1] + m[2][2]
        return when {
            trace > 0.0 -> {
                val s = sqrt(trace + 1.0) * 2.0
                Quaternion(
                    x = (m[2][1] - m[1][2]) / s,
                    y = (m[0][2] - m[2][0]) / s,
                    z = (m[1][0] - m[0][1]) / s,
                    w = 0.25 * s,
                )
            }
            m[0][0] > m[1][1] && m[0][0] > m[2][2] -> {
                val s = sqrt(1.0 + m[0][0] - m[1][1] - m[2][2]) * 2.0
                Quaternion(
                    x = 0.25 * s,
                    y = (m[0][1] + m[1][0]) / s,
                    z = (m[0][2] + m[2][0]) / s,
                    w = (m[2][1] - m[1][2]) / s,
                )
            }
            m[1][1] > m[2][2] -> {
                val s = sqrt(1.0 + m[1][1] - m[0][0] - m[2][2]) * 2.0
                Quaternion(
                    x = (m[0][1] + m[1][0]) / s,
                    y = 0.25 * s,
                    z = (m[1][2] + m[2][1]) / s,
                    w = (m[0][2] - m[2][0]) / s,
                )
            }
            else -> {
                val s = sqrt(1.0 + m[2][2] - m[0][0] - m[1][1]) * 2.0
                Quaternion(
                    x = (m[0][2] + m[2][0]) / s,
                    y = (m[1][2] + m[2][1]) / s,
                    z = 0.25 * s,
                    w = (m[1][0] - m[0][1]) / s,
                )
            }
        }.normalized()
    }

    /** §9 `deviceVectorToReferenceEnu`. Accepts only a canonical attitude (R49). */
    public fun deviceVectorToReferenceEnu(
        attitude: CanonicalAttitude,
        deviceVector: Vector3,
    ): EnuVector = rotateVectorByQuaternion(attitude.quaternionDeviceToReferenceEnuXYZW, deviceVector)

    /**
     * §14: `normalize360(degrees(atan2(east, north)))`.
     *
     * Throws [SingularProjectionException] on an exactly-zero horizontal projection rather
     * than returning the `atan2(0, 0)` zero that looks like north (failure mode 6).
     */
    public fun enuBearingDeg(vector: EnuVector): Double {
        if (vector.horizontalNorm == 0.0) {
            throw SingularProjectionException(
                "the axis is exactly vertical in REFERENCE_ENU; its horizontal bearing does not exist"
            )
        }
        return CircularMath.normalize360(Math.toDegrees(atan2(vector.east, vector.north)))
    }

    /**
     * Elevation above the horizontal plane, positive up, in `[-90, 90]`.
     *
     * The engine compares this against the mode's configured pose gate; it is the
     * conditioning measure the spec's "ill-conditioned" wording refers to.
     */
    public fun enuElevationDeg(vector: EnuVector): Double {
        val norm = sqrt(
            vector.east * vector.east + vector.north * vector.north + vector.up * vector.up
        )
        require(norm != 0.0) { "cannot take the elevation of a zero-length vector" }
        return Math.toDegrees(asin((vector.up / norm).coerceIn(-1.0, 1.0)))
    }

    /**
     * §9 `modeReferenceVectorHeadingDeg` — the active mode's axis, never another axis.
     *
     * §11.1: if the active reference-axis projection is singular or ill-conditioned, reject
     * the pose; never resolve the north reference on a convenient different axis and transfer
     * that label. In wall mode the top edge is close to vertical and its bearing is
     * ill-conditioned, which is why the mode selects the axis rather than the caller.
     */
    public fun modeReferenceVectorHeadingDeg(
        attitude: CanonicalAttitude,
        mode: MeasurementMode,
    ): AxisBearing {
        val enu = deviceVectorToReferenceEnu(attitude, requireNotNull(modeReferenceVectors[mode]))
        return AxisBearing(
            headingDeg = enuBearingDeg(enu),
            elevationDeg = enuElevationDeg(enu),
            horizontalNorm = enu.horizontalNorm,
        )
    }

    /**
     * §14: reject an ill-conditioned reference-axis projection rather than bearing it.
     *
     * An exactly-vertical axis is the textbook singularity, but floating point rarely
     * delivers it: rotating device `+y` into a wall pose leaves a horizontal projection of
     * `2.2e-16`, from which `atan2` returns a confident, arbitrary `180`. The conditioning
     * decision is therefore made on the axis **elevation** against the mode's configured pose
     * limit, and the exact-zero check remains only as the degenerate special case.
     */
    public fun modeAxisBearingOrReject(
        attitude: CanonicalAttitude,
        mode: MeasurementMode,
        maxAxisElevationAbsDeg: Double,
    ): AxisBearing {
        val enu = deviceVectorToReferenceEnu(attitude, requireNotNull(modeReferenceVectors[mode]))
        val elevation = enuElevationDeg(enu)
        if (abs(elevation) > maxAxisElevationAbsDeg) {
            throw SingularProjectionException(
                "${mode.wire} reference axis is $elevation deg from horizontal, beyond the " +
                    "configured $maxAxisElevationAbsDeg deg limit; its horizontal bearing is " +
                    "ill-conditioned and MUST NOT be resolved on a different axis (§11.1, §14)"
            )
        }
        return AxisBearing(
            headingDeg = enuBearingDeg(enu),
            elevationDeg = elevation,
            horizontalNorm = enu.horizontalNorm,
        )
    }

    /**
     * §11.1/§16.1 `transformAgreementDeg` for two same-observation extraction routes.
     *
     * Direct quaternion-vector rotation versus the rotation-matrix route, over the **same**
     * canonical attitude and the **same** physical axis. A large value is a code fault —
     * frame transform, quaternion order, axis selection or remapping — and MUST NOT
     * contribute to `MagneticState`: telling a user to move away from metal because the
     * wall-mode quaternion has a swapped axis is a failure that survives a long time in the
     * field.
     */
    public fun transformAgreementDeg(
        attitude: CanonicalAttitude,
        mode: MeasurementMode,
    ): Double {
        val deviceVector = requireNotNull(modeReferenceVectors[mode])
        val quaternion = attitude.quaternionDeviceToReferenceEnuXYZW
        val direct = enuBearingDeg(rotateVectorByQuaternion(quaternion, deviceVector))
        val viaMatrix = enuBearingDeg(
            rotateVectorByMatrix(rotationMatrixFromQuaternion(quaternion), deviceVector)
        )
        return CircularMath.absoluteCircularDifferenceDeg(direct, viaMatrix)
    }

    /**
     * §14: sitting (坐) is exactly `normalize360(facing + 180)`, computed only on request.
     *
     * Kept as a named function so the derived opposite is always labelled as such. Reporting
     * the wrong one is a 180 deg error that looks entirely plausible on a dial (failure mode 18).
     */
    public fun sittingFromFacingDeg(facingDeg: Double): Double =
        CircularMath.normalize360(facingDeg + 180.0)
}

/**
 * A vector in the **device** frame: `+x` right, `+y` toward the portrait top edge, `+z` out of
 * the screen.
 */
public data class Vector3(val x: Double, val y: Double, val z: Double) {
    init {
        require(x.isFinite() && y.isFinite() && z.isFinite()) {
            "Vector3 components must be finite, got ($x, $y, $z)"
        }
    }
}

/** A vector in project REFERENCE_ENU. Named components, never a positional triple. */
public data class EnuVector(val east: Double, val north: Double, val up: Double) {
    init {
        require(east.isFinite() && north.isFinite() && up.isFinite()) {
            "EnuVector components must be finite, got ($east, $north, $up)"
        }
    }

    val horizontalNorm: Double get() = hypot(east, north)
}

/**
 * A unit quaternion in explicit `(x, y, z, w)` component order.
 *
 * Failure mode 5 is the `wxyz`/`xyzw` swap, non-normalized input, active/passive inversion
 * and multiplication-order error. The type names the order; the transform direction is named
 * by whatever field or parameter holds it, never inferred.
 */
public data class Quaternion(val x: Double, val y: Double, val z: Double, val w: Double) {
    init {
        require(x.isFinite() && y.isFinite() && z.isFinite() && w.isFinite()) {
            "Quaternion components must be finite, got ($x, $y, $z, $w)"
        }
    }

    val norm: Double get() = sqrt(x * x + y * y + z * z + w * w)

    public fun normalized(): Quaternion {
        val n = norm
        require(n != 0.0) { "cannot normalize a zero-norm quaternion" }
        return Quaternion(x / n, y / n, z / n, w / n)
    }

    /** The inverse rotation for a unit quaternion — i.e. the opposite transform direction. */
    public fun conjugate(): Quaternion = Quaternion(-x, -y, -z, w)

    /**
     * Hamilton product `this ⊗ other`. Composition order is explicit so it cannot be reversed
     * by accident: applying the result to a vector applies [other] first, then `this`.
     */
    public fun multipliedBy(other: Quaternion): Quaternion = Quaternion(
        x = w * other.x + x * other.w + y * other.z - z * other.y,
        y = w * other.y - x * other.z + y * other.w + z * other.x,
        z = w * other.z + x * other.y - y * other.x + z * other.w,
        w = w * other.w - x * other.x - y * other.y - z * other.z,
    )

    public companion object {
        public fun identity(): Quaternion = Quaternion(0.0, 0.0, 0.0, 1.0)
    }
}

/**
 * Which way a provider's attitude quaternion transforms.
 *
 * Naming this is not pedantry: a transposed attitude is failure mode 5 and produces a
 * plausible bearing, never a crash.
 */
public enum class TransformDirection {
    DEVICE_TO_REFERENCE,
    REFERENCE_TO_DEVICE,
}

/**
 * Provider-native earth-axis conventions, retained as provenance (§5, §11.1, R49).
 *
 * The canonical sample carries a project REFERENCE_ENU quaternion; the native frame stays in
 * telemetry for replay. Nothing in core code may consume a provider-native quaternion
 * directly.
 */
public enum class NativeAttitudeFrame(public val wire: String) {
    GOOGLE_FOP_ENU("GOOGLE_FOP_ENU"),
    ANDROID_ROTATION_VECTOR_ENU("ANDROID_ROTATION_VECTOR_ENU"),
    CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL("CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL"),
    REPLAY_REFERENCE_ENU("REPLAY_REFERENCE_ENU"),
}

/**
 * Whether a declared native convention has been checked against the pinned SDK.
 *
 * §37 rule 4 requires verifying the **installed** API signature and behaviour rather than
 * assuming a documentation page is unchanged, and §11.1 forbids inferring permutation,
 * transpose or signs from yaw intuition. Phase 1 has no SDK, so conventions it cannot check
 * are marked [DECLARED_UNVERIFIED] and the Phase 2 adapter must confirm them with physical
 * N/E/S/W/up poses before the sample enters production.
 */
public enum class ConventionVerification {
    DECLARED_UNVERIFIED,
    VERIFIED_AGAINST_PINNED_SDK,
}

/** One canonical axis expressed as a signed native axis: `sign * native[index]`. */
public data class AxisSelector(val index: Int, val sign: Double) {
    init {
        require(index in 0..2) { "AxisSelector.index must be 0, 1 or 2, got $index" }
        require(sign == 1.0 || sign == -1.0) {
            "AxisSelector.sign must be +1.0 or -1.0, got $sign"
        }
    }
}

/**
 * A provider's declared attitude convention: axis permutation plus transform direction.
 *
 * [east]/[north]/[up] say which signed native reference axis carries each canonical axis. The
 * permutation must be a proper rotation (determinant `+1`); a reflection would silently
 * mirror every bearing.
 */
public data class NativeAttitudeConvention(
    val frame: NativeAttitudeFrame,
    val east: AxisSelector,
    val north: AxisSelector,
    val up: AxisSelector,
    val direction: TransformDirection,
    val referenceContract: ProviderReferenceContract,
    val verification: ConventionVerification,
    val note: String = "",
) {
    init {
        require(setOf(east.index, north.index, up.index).size == 3) {
            "${frame.wire}: east/north/up must select three distinct native axes"
        }
        require(abs(permutationDeterminant() - 1.0) <= 1e-12) {
            "${frame.wire}: the native->ENU axis map is a reflection (determinant -1), which " +
                "would mirror every bearing"
        }
    }

    /** Rows `(east, north, up)` over native columns `(0, 1, 2)`. */
    public fun permutationMatrix(): List<List<Double>> {
        val rows = listOf(east, north, up)
        return rows.map { selector ->
            (0..2).map { column -> if (column == selector.index) selector.sign else 0.0 }
        }
    }

    public fun permutationDeterminant(): Double {
        val m = permutationMatrix()
        return m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1]) -
            m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0]) +
            m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
    }
}

/**
 * §5 `ProviderAttitudeSample` attitude: always device -> project REFERENCE_ENU.
 *
 * [nativeFrame] is provenance only. [referenceContract] says whether the horizontal axes are
 * true- or magnetic-referenced; the axes alone never carry that claim.
 */
public data class CanonicalAttitude(
    val quaternionDeviceToReferenceEnuXYZW: Quaternion,
    val referenceContract: ProviderReferenceContract,
    val nativeFrame: NativeAttitudeFrame,
) {
    init {
        require(abs(quaternionDeviceToReferenceEnuXYZW.norm - 1.0) <= 1e-9) {
            "canonical attitude requires a unit quaternion; got norm " +
                "${quaternionDeviceToReferenceEnuXYZW.norm}. Normalize at the adapter " +
                "boundary, not in core math."
        }
    }
}

/** A reference axis projected into REFERENCE_ENU: its bearing and its conditioning. */
public data class AxisBearing(
    val headingDeg: Double,
    val elevationDeg: Double,
    val horizontalNorm: Double,
)

/**
 * The reference axis has no usable horizontal projection, so its bearing does not exist.
 *
 * §14: "Reject if the horizontal projection is ill-conditioned". The *practical* conditioning
 * gate is the mode's configured pose limit, evaluated against [AxisBearing.elevationDeg]; no
 * new numeric constant is introduced.
 */
public class SingularProjectionException(message: String) : IllegalStateException(message)

/** The declared native conventions, keyed by frame. */
public object NativeConventions {

    /**
     * §13: `getAttitude()` is `[qx, qy, qz, qw]` mapping device -> ENU. The axis order already
     * matches project REFERENCE_ENU, so the permutation is the identity. The
     * true-vs-magnetic ambiguity lives in the contract, not in the axes (§11).
     */
    public val googleFop: NativeAttitudeConvention = NativeAttitudeConvention(
        frame = NativeAttitudeFrame.GOOGLE_FOP_ENU,
        east = AxisSelector(0, 1.0),
        north = AxisSelector(1, 1.0),
        up = AxisSelector(2, 1.0),
        direction = TransformDirection.DEVICE_TO_REFERENCE,
        referenceContract = ProviderReferenceContract.TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC,
        verification = ConventionVerification.DECLARED_UNVERIFIED,
        note = "§13. Confirm against the pinned Play services build in Phase 2 before any " +
            "sample reaches the engine.",
    )

    /**
     * §11.1: Android `TYPE_ROTATION_VECTOR` normalizes to the same axis order with an explicit
     * `MAGNETIC` contract; the app applies WMM declination exactly once, later.
     */
    public val androidRotationVector: NativeAttitudeConvention = NativeAttitudeConvention(
        frame = NativeAttitudeFrame.ANDROID_ROTATION_VECTOR_ENU,
        east = AxisSelector(0, 1.0),
        north = AxisSelector(1, 1.0),
        up = AxisSelector(2, 1.0),
        direction = TransformDirection.DEVICE_TO_REFERENCE,
        referenceContract = ProviderReferenceContract.MAGNETIC,
        verification = ConventionVerification.DECLARED_UNVERIFIED,
        note = "§13: obtained through SensorManager.getRotationMatrixFromVector. Confirm the " +
            "installed signature and the resulting axis order in Phase 2.",
    )

    /**
     * Core Motion `.xTrueNorthZVertical`: native `+X` true north, `+Z` vertical, hence a
     * right-handed native `+Y` pointing west and `east = -native_y`. **Declared, not
     * verified:** §11.1 requires the adapter to prove both the axis convention and the
     * transform direction with N/E/S/W/up golden vectors against the pinned SDK, which Phase 1
     * cannot run.
     */
    public val coreMotionTrueNorth: NativeAttitudeConvention = NativeAttitudeConvention(
        frame = NativeAttitudeFrame.CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL,
        east = AxisSelector(1, -1.0),
        north = AxisSelector(0, 1.0),
        up = AxisSelector(2, 1.0),
        direction = TransformDirection.DEVICE_TO_REFERENCE,
        referenceContract = ProviderReferenceContract.TRUE,
        verification = ConventionVerification.DECLARED_UNVERIFIED,
        note = "R49: both the axis permutation AND the transform direction MUST be confirmed " +
            "against the pinned Core Motion SDK with physical N/E/S/W/up poses in Phase 2.",
    )

    /** Replay fixtures are authored directly in project REFERENCE_ENU. */
    public val replay: NativeAttitudeConvention = NativeAttitudeConvention(
        frame = NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
        east = AxisSelector(0, 1.0),
        north = AxisSelector(1, 1.0),
        up = AxisSelector(2, 1.0),
        direction = TransformDirection.DEVICE_TO_REFERENCE,
        referenceContract = ProviderReferenceContract.TRUE,
        verification = ConventionVerification.VERIFIED_AGAINST_PINNED_SDK,
        note = "Fixture data is authored in canonical REFERENCE_ENU by definition.",
    )

    public val byFrame: Map<NativeAttitudeFrame, NativeAttitudeConvention> =
        listOf(googleFop, androidRotationVector, coreMotionTrueNorth, replay)
            .associateBy { it.frame }
}
