"""SPEC.md §3, §11.1, §14 — frames, quaternion order, and mode axis projection.

Canonical orientation basis is **REFERENCE_ENU**: ``+Z`` up, ``+Y`` toward the north reference
named by ``providerReferenceContract``, and ``+X = +Y x +Z`` (east relative to that reference).
With contract ``TRUE`` this is geographic ENU; with ``MAGNETIC`` it is
magnetic-east/magnetic-north/up. A magnetic basis is never relabelled geographic ENU merely
because its axis order is east/north/up — which is why :class:`CanonicalAttitude` carries the
contract beside the quaternion.

Quaternion types name component order **and** transform direction: the canonical field is
``attitudeQuaternionDeviceToReferenceEnuXYZW`` — components ``(x, y, z, w)``, transform
device → project REFERENCE_ENU. A bare 4-element array never travels beyond a provider
adapter, so :class:`Quaternion` is the only shape core code accepts.

``atan2`` appears here twice, both allowlisted by §33.1 as bearing projections: the horizontal
bearing of a reference axis, and the matrix→quaternion recovery. Neither is a signed circular
difference; that has exactly one implementation, in :mod:`fscompass_analysis.circular`.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from enum import Enum

from .circular import normalize360
from .enums import MeasurementMode, ProviderReferenceContract


class FrameError(ValueError):
    """An invalid frame, quaternion or vector input."""


class SingularProjection(ValueError):
    """The reference axis has no horizontal projection, so its bearing does not exist.

    §14: "Reject if the horizontal projection is ill-conditioned". This exception marks the
    exactly-degenerate case, where ``atan2(0, 0)`` would otherwise return ``0.0`` and present
    a singular pose as a north-facing measurement (failure mode 6). The *practical*
    conditioning gate is the mode's configured pose limit — ``flatModePitchAbsMaxDeg`` /
    ``flatModeRollAbsMaxDeg`` for flat, ``wallNormalElevationAbsMaxDeg`` /
    ``wallTopAxisFromVerticalMaxDeg`` for wall — evaluated by the engine against
    :attr:`AxisBearing.elevation_deg`. No new numeric constant is introduced here.
    """


@dataclass(frozen=True)
class Vector3:
    """A vector in the **device** frame: ``+x`` right, ``+y`` toward the portrait top edge,
    ``+z`` out of the screen."""

    x: float
    y: float
    z: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise FrameError(f"Vector3.{name} must be finite, got {value!r}")

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z)

    def normalized(self) -> "Vector3":
        norm = self.norm
        if norm == 0.0:
            raise FrameError("cannot normalize a zero-length vector")
        return Vector3(self.x / norm, self.y / norm, self.z / norm)


@dataclass(frozen=True)
class EnuVector:
    """A vector in project REFERENCE_ENU. Named components, never a positional triple."""

    east: float
    north: float
    up: float

    def __post_init__(self) -> None:
        for name in ("east", "north", "up"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise FrameError(f"EnuVector.{name} must be finite, got {value!r}")

    @property
    def horizontal_norm(self) -> float:
        return math.hypot(self.east, self.north)


@dataclass(frozen=True)
class Quaternion:
    """A unit quaternion in explicit ``(x, y, z, w)`` component order.

    Failure mode 5 is the ``wxyz``/``xyzw`` swap, non-normalized input, active/passive
    inversion and multiplication-order error. The type names the order; the transform
    direction is named by whatever field or parameter holds it, never inferred.
    """

    x: float
    y: float
    z: float
    w: float

    def __post_init__(self) -> None:
        for name in ("x", "y", "z", "w"):
            value = getattr(self, name)
            if not math.isfinite(value):
                raise FrameError(f"Quaternion.{name} must be finite, got {value!r}")

    @property
    def norm(self) -> float:
        return math.sqrt(self.x * self.x + self.y * self.y + self.z * self.z + self.w * self.w)

    def normalized(self) -> "Quaternion":
        norm = self.norm
        if norm == 0.0:
            raise FrameError("cannot normalize a zero-norm quaternion")
        return Quaternion(self.x / norm, self.y / norm, self.z / norm, self.w / norm)

    def conjugate(self) -> "Quaternion":
        """The inverse rotation for a unit quaternion — i.e. the opposite transform direction."""
        return Quaternion(-self.x, -self.y, -self.z, self.w)

    def multiplied_by(self, other: "Quaternion") -> "Quaternion":
        """Hamilton product ``self ⊗ other``.

        Composition order is explicit so it cannot be reversed by accident: applying the
        result to a vector applies ``other`` first, then ``self``.
        """
        return Quaternion(
            x=self.w * other.x + self.x * other.w + self.y * other.z - self.z * other.y,
            y=self.w * other.y - self.x * other.z + self.y * other.w + self.z * other.x,
            z=self.w * other.z + self.x * other.y - self.y * other.x + self.z * other.w,
            w=self.w * other.w - self.x * other.x - self.y * other.y - self.z * other.z,
        )

    @staticmethod
    def identity() -> "Quaternion":
        return Quaternion(0.0, 0.0, 0.0, 1.0)


class TransformDirection(Enum):
    """Which way a provider's attitude quaternion transforms.

    Naming this is not pedantry: a transposed attitude is failure mode 5 and produces a
    plausible bearing, never a crash.
    """

    DEVICE_TO_REFERENCE = "DEVICE_TO_REFERENCE"
    REFERENCE_TO_DEVICE = "REFERENCE_TO_DEVICE"


class NativeAttitudeFrame(Enum):
    """Provider-native earth-axis conventions, retained as provenance (§5, §11.1, R49).

    The canonical sample carries a project REFERENCE_ENU quaternion; the native frame stays
    in telemetry for replay. Nothing in core code may consume a provider-native quaternion
    directly.
    """

    GOOGLE_FOP_ENU = "GOOGLE_FOP_ENU"
    ANDROID_ROTATION_VECTOR_ENU = "ANDROID_ROTATION_VECTOR_ENU"
    CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL = "CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL"
    REPLAY_REFERENCE_ENU = "REPLAY_REFERENCE_ENU"


class ConventionVerification(Enum):
    """Whether a declared native convention has been checked against the pinned SDK.

    §37 rule 4 requires verifying the **installed** API signature and behaviour rather than
    assuming a documentation page is unchanged, and §11.1 forbids inferring permutation,
    transpose or signs from yaw intuition. Phase 1 has no SDK, so conventions it cannot check
    are marked :attr:`DECLARED_UNVERIFIED` and the Phase 2 adapter must confirm them with
    physical N/E/S/W/up poses before the sample enters production.
    """

    DECLARED_UNVERIFIED = "DECLARED_UNVERIFIED"
    VERIFIED_AGAINST_PINNED_SDK = "VERIFIED_AGAINST_PINNED_SDK"


@dataclass(frozen=True)
class AxisSelector:
    """One canonical axis expressed as a signed native axis: ``sign * native[index]``."""

    index: int
    sign: float

    def __post_init__(self) -> None:
        if self.index not in (0, 1, 2):
            raise FrameError(f"AxisSelector.index must be 0, 1 or 2, got {self.index!r}")
        if self.sign not in (1.0, -1.0):
            raise FrameError(f"AxisSelector.sign must be +1.0 or -1.0, got {self.sign!r}")


@dataclass(frozen=True)
class NativeAttitudeConvention:
    """A provider's declared attitude convention: axis permutation plus transform direction.

    ``east``/``north``/``up`` say which signed native reference axis carries each canonical
    axis. The permutation must be a proper rotation (determinant ``+1``); a reflection would
    silently mirror every bearing.
    """

    frame: NativeAttitudeFrame
    east: AxisSelector
    north: AxisSelector
    up: AxisSelector
    direction: TransformDirection
    reference_contract: ProviderReferenceContract
    verification: ConventionVerification
    note: str = ""

    def __post_init__(self) -> None:
        indices = {self.east.index, self.north.index, self.up.index}
        if len(indices) != 3:
            raise FrameError(
                f"{self.frame.value}: east/north/up must select three distinct native axes"
            )
        if abs(self.permutation_determinant() - 1.0) > 1e-12:
            raise FrameError(
                f"{self.frame.value}: the native→ENU axis map is a reflection "
                "(determinant -1), which would mirror every bearing"
            )

    def permutation_matrix(self) -> list[list[float]]:
        """Rows ``(east, north, up)`` over native columns ``(0, 1, 2)``."""
        matrix = [[0.0, 0.0, 0.0] for _ in range(3)]
        for row, selector in enumerate((self.east, self.north, self.up)):
            matrix[row][selector.index] = selector.sign
        return matrix

    def permutation_determinant(self) -> float:
        m = self.permutation_matrix()
        return (
            m[0][0] * (m[1][1] * m[2][2] - m[1][2] * m[2][1])
            - m[0][1] * (m[1][0] * m[2][2] - m[1][2] * m[2][0])
            + m[0][2] * (m[1][0] * m[2][1] - m[1][1] * m[2][0])
        )


#: Google FOP documents device → ENU in ``[qx, qy, qz, qw]``; the axis order already matches
#: project REFERENCE_ENU, so the permutation is the identity. The true-vs-magnetic ambiguity
#: lives in the contract, not in the axes (§11).
GOOGLE_FOP_CONVENTION = NativeAttitudeConvention(
    frame=NativeAttitudeFrame.GOOGLE_FOP_ENU,
    east=AxisSelector(0, 1.0),
    north=AxisSelector(1, 1.0),
    up=AxisSelector(2, 1.0),
    direction=TransformDirection.DEVICE_TO_REFERENCE,
    reference_contract=ProviderReferenceContract.TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC,
    verification=ConventionVerification.DECLARED_UNVERIFIED,
    note="§13: getAttitude() is [qx, qy, qz, qw] mapping device → ENU. Confirm against the "
    "pinned Play services build in Phase 2 before any sample reaches the engine.",
)

#: Android ``TYPE_ROTATION_VECTOR`` normalizes to the same axis order with an explicit
#: ``MAGNETIC`` contract (§11.1); the app applies WMM declination exactly once, later.
ANDROID_ROTATION_VECTOR_CONVENTION = NativeAttitudeConvention(
    frame=NativeAttitudeFrame.ANDROID_ROTATION_VECTOR_ENU,
    east=AxisSelector(0, 1.0),
    north=AxisSelector(1, 1.0),
    up=AxisSelector(2, 1.0),
    direction=TransformDirection.DEVICE_TO_REFERENCE,
    reference_contract=ProviderReferenceContract.MAGNETIC,
    verification=ConventionVerification.DECLARED_UNVERIFIED,
    note="§13: obtained through SensorManager.getRotationMatrixFromVector. Confirm the "
    "installed signature and the resulting axis order in Phase 2.",
)

#: Core Motion ``.xTrueNorthZVertical``: native ``+X`` true north, ``+Z`` vertical (up), so a
#: right-handed native ``+Y`` points west and ``east = -native_y``. **Declared, not verified:**
#: §11.1 requires the adapter to prove both the axis convention and the transform direction
#: with N/E/S/W/up golden vectors against the pinned SDK, which Phase 1 cannot run.
CORE_MOTION_TRUE_NORTH_CONVENTION = NativeAttitudeConvention(
    frame=NativeAttitudeFrame.CORE_MOTION_X_TRUE_NORTH_Z_VERTICAL,
    east=AxisSelector(1, -1.0),
    north=AxisSelector(0, 1.0),
    up=AxisSelector(2, 1.0),
    direction=TransformDirection.DEVICE_TO_REFERENCE,
    reference_contract=ProviderReferenceContract.TRUE,
    verification=ConventionVerification.DECLARED_UNVERIFIED,
    note="R49: both the axis permutation AND the transform direction MUST be confirmed "
    "against the pinned Core Motion SDK with physical N/E/S/W/up poses in Phase 2. Phase 1 "
    "pins the conversion given this declared convention; it does not assert the convention.",
)

#: Replay fixtures are authored directly in project REFERENCE_ENU, so the conversion is the
#: identity and is verified by construction.
REPLAY_CONVENTION = NativeAttitudeConvention(
    frame=NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
    east=AxisSelector(0, 1.0),
    north=AxisSelector(1, 1.0),
    up=AxisSelector(2, 1.0),
    direction=TransformDirection.DEVICE_TO_REFERENCE,
    reference_contract=ProviderReferenceContract.TRUE,
    verification=ConventionVerification.VERIFIED_AGAINST_PINNED_SDK,
    note="Fixture data is authored in canonical REFERENCE_ENU by definition.",
)

NATIVE_CONVENTIONS: dict[NativeAttitudeFrame, NativeAttitudeConvention] = {
    convention.frame: convention
    for convention in (
        GOOGLE_FOP_CONVENTION,
        ANDROID_ROTATION_VECTOR_CONVENTION,
        CORE_MOTION_TRUE_NORTH_CONVENTION,
        REPLAY_CONVENTION,
    )
}


@dataclass(frozen=True)
class CanonicalAttitude:
    """§5 ``ProviderAttitudeSample`` attitude: always device → project REFERENCE_ENU.

    ``native_frame`` is provenance only. ``reference_contract`` says whether the horizontal
    axes are true- or magnetic-referenced; the axes alone never carry that claim.
    """

    quaternion_device_to_reference_enu_xyzw: Quaternion
    reference_contract: ProviderReferenceContract
    native_frame: NativeAttitudeFrame

    def __post_init__(self) -> None:
        norm = self.quaternion_device_to_reference_enu_xyzw.norm
        if abs(norm - 1.0) > 1e-9:
            raise FrameError(
                "canonical attitude requires a unit quaternion; got norm "
                f"{norm!r}. Normalize at the adapter boundary, not in core math."
            )


def canonicalize_native_attitude(
    native_quaternion: Quaternion,
    convention: NativeAttitudeConvention,
) -> CanonicalAttitude:
    """Convert a provider-native attitude to canonical device → REFERENCE_ENU (R49).

    Both the transform **direction** and the axis **convention** are converted:

    1. a ``REFERENCE_TO_DEVICE`` quaternion is conjugated to device → native reference;
    2. the native reference axes are remapped to ``(east, north, up)``.

    The remap is composed as a rotation, so the result is a single canonical quaternion
    rather than a matrix the caller might apply in the wrong order.
    """
    unit = native_quaternion.normalized()
    device_to_native = (
        unit if convention.direction is TransformDirection.DEVICE_TO_REFERENCE else unit.conjugate()
    )
    permutation = quaternion_from_rotation_matrix(convention.permutation_matrix())
    canonical = permutation.multiplied_by(device_to_native).normalized()
    return CanonicalAttitude(
        quaternion_device_to_reference_enu_xyzw=canonical,
        reference_contract=convention.reference_contract,
        native_frame=convention.frame,
    )


def rotate_vector_by_quaternion(quaternion: Quaternion, vector: Vector3) -> EnuVector:
    """Direct quaternion-vector rotation: ``v' = v + 2w(u x v) + 2(u x (u x v))``.

    One of the two independent extraction routes §11.1 requires for
    ``transformAgreementDeg``; :func:`rotate_vector_by_matrix` is the other.
    """
    unit = quaternion.normalized()
    ux, uy, uz, w = unit.x, unit.y, unit.z, unit.w
    vx, vy, vz = vector.x, vector.y, vector.z

    cx = uy * vz - uz * vy
    cy = uz * vx - ux * vz
    cz = ux * vy - uy * vx

    ccx = uy * cz - uz * cy
    ccy = uz * cx - ux * cz
    ccz = ux * cy - uy * cx

    return EnuVector(
        east=vx + 2.0 * w * cx + 2.0 * ccx,
        north=vy + 2.0 * w * cy + 2.0 * ccy,
        up=vz + 2.0 * w * cz + 2.0 * ccz,
    )


def rotation_matrix_from_quaternion(quaternion: Quaternion) -> list[list[float]]:
    """The 3x3 rotation matrix for the same transform direction as ``quaternion``."""
    unit = quaternion.normalized()
    x, y, z, w = unit.x, unit.y, unit.z, unit.w
    return [
        [1.0 - 2.0 * (y * y + z * z), 2.0 * (x * y - z * w), 2.0 * (x * z + y * w)],
        [2.0 * (x * y + z * w), 1.0 - 2.0 * (x * x + z * z), 2.0 * (y * z - x * w)],
        [2.0 * (x * z - y * w), 2.0 * (y * z + x * w), 1.0 - 2.0 * (x * x + y * y)],
    ]


def rotate_vector_by_matrix(matrix: list[list[float]], vector: Vector3) -> EnuVector:
    """The rotation-matrix extraction route (§11.1's second, independent implementation)."""
    components = (vector.x, vector.y, vector.z)
    east, north, up = (
        sum(matrix[row][column] * components[column] for column in range(3)) for row in range(3)
    )
    return EnuVector(east=east, north=north, up=up)


def quaternion_from_rotation_matrix(matrix: list[list[float]]) -> Quaternion:
    """Shepperd's method: pick the largest divisor so no branch divides by ~0.

    Used to compose the native→ENU axis permutation into the canonical quaternion, and to
    close the matrix↔quaternion round trip the golden vectors check.
    """
    trace = matrix[0][0] + matrix[1][1] + matrix[2][2]
    if trace > 0.0:
        scale = math.sqrt(trace + 1.0) * 2.0
        return Quaternion(
            x=(matrix[2][1] - matrix[1][2]) / scale,
            y=(matrix[0][2] - matrix[2][0]) / scale,
            z=(matrix[1][0] - matrix[0][1]) / scale,
            w=0.25 * scale,
        ).normalized()
    if matrix[0][0] > matrix[1][1] and matrix[0][0] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[0][0] - matrix[1][1] - matrix[2][2]) * 2.0
        return Quaternion(
            x=0.25 * scale,
            y=(matrix[0][1] + matrix[1][0]) / scale,
            z=(matrix[0][2] + matrix[2][0]) / scale,
            w=(matrix[2][1] - matrix[1][2]) / scale,
        ).normalized()
    if matrix[1][1] > matrix[2][2]:
        scale = math.sqrt(1.0 + matrix[1][1] - matrix[0][0] - matrix[2][2]) * 2.0
        return Quaternion(
            x=(matrix[0][1] + matrix[1][0]) / scale,
            y=0.25 * scale,
            z=(matrix[1][2] + matrix[2][1]) / scale,
            w=(matrix[0][2] - matrix[2][0]) / scale,
        ).normalized()
    scale = math.sqrt(1.0 + matrix[2][2] - matrix[0][0] - matrix[1][1]) * 2.0
    return Quaternion(
        x=(matrix[0][2] + matrix[2][0]) / scale,
        y=(matrix[1][2] + matrix[2][1]) / scale,
        z=0.25 * scale,
        w=(matrix[1][0] - matrix[0][1]) / scale,
    ).normalized()


def device_vector_to_reference_enu(attitude: CanonicalAttitude, device_vector: Vector3) -> EnuVector:
    """§9 ``deviceVectorToReferenceEnu``. Accepts only a canonical attitude (R49)."""
    return rotate_vector_by_quaternion(
        attitude.quaternion_device_to_reference_enu_xyzw, device_vector
    )


#: §3: the portrait top edge is device ``+y``; the outward screen normal is device ``+z``.
MODE_REFERENCE_VECTORS: dict[MeasurementMode, Vector3] = {
    MeasurementMode.FLAT_TOP_EDGE: Vector3(0.0, 1.0, 0.0),
    MeasurementMode.WALL_FLUSH_BACK: Vector3(0.0, 0.0, 1.0),
}


@dataclass(frozen=True)
class AxisBearing:
    """A reference axis projected into REFERENCE_ENU: its bearing and its conditioning."""

    heading_deg: float
    elevation_deg: float
    horizontal_norm: float


def enu_bearing_deg(vector: EnuVector) -> float:
    """§14: ``normalize360(degrees(atan2(east, north)))``.

    Raises :class:`SingularProjection` on an exactly-zero horizontal projection rather than
    returning the ``atan2(0, 0)`` zero that looks like north (failure mode 6).
    """
    if vector.horizontal_norm == 0.0:
        raise SingularProjection(
            "the axis is exactly vertical in REFERENCE_ENU; its horizontal bearing does not exist"
        )
    return normalize360(math.degrees(math.atan2(vector.east, vector.north)))


def enu_elevation_deg(vector: EnuVector) -> float:
    """Elevation above the horizontal plane, positive up, in ``[-90, 90]``.

    The engine compares this against the mode's configured pose gate; it is the conditioning
    measure the spec's `ill-conditioned` wording refers to.
    """
    norm = math.sqrt(vector.east**2 + vector.north**2 + vector.up**2)
    if norm == 0.0:
        raise FrameError("cannot take the elevation of a zero-length vector")
    return math.degrees(math.asin(max(-1.0, min(1.0, vector.up / norm))))


def mode_reference_vector_heading_deg(
    attitude: CanonicalAttitude, mode: MeasurementMode
) -> AxisBearing:
    """§9 ``modeReferenceVectorHeadingDeg`` — the active mode's axis, never another axis.

    §11.1: if the active reference-axis projection is singular or ill-conditioned, reject the
    pose; never resolve on a convenient different axis and transfer that label. In wall mode
    the top edge is close to vertical and its bearing is ill-conditioned, which is why the
    mode selects the axis rather than the caller.
    """
    device_vector = MODE_REFERENCE_VECTORS[mode]
    enu = device_vector_to_reference_enu(attitude, device_vector)
    return AxisBearing(
        heading_deg=enu_bearing_deg(enu),
        elevation_deg=enu_elevation_deg(enu),
        horizontal_norm=enu.horizontal_norm,
    )


#: §14/§18.5: the configured pose limit that bounds each mode's reference-axis elevation.
#: Flat mode's top edge tilts with pitch, so ``flatModePitchAbsMaxDeg`` bounds it; wall mode
#: states its own limit directly. No new constant is introduced — the conditioning gate is
#: the pose gate, read from configuration by name (§18.5 forbids a numeric literal in a gate).
MODE_AXIS_ELEVATION_GATE_KEY: dict[MeasurementMode, str] = {
    MeasurementMode.FLAT_TOP_EDGE: "flatModePitchAbsMaxDeg",
    MeasurementMode.WALL_FLUSH_BACK: "wallNormalElevationAbsMaxDeg",
}


def mode_axis_bearing_or_reject(
    attitude: CanonicalAttitude, mode: MeasurementMode, max_axis_elevation_abs_deg: float
) -> AxisBearing:
    """§14: reject an ill-conditioned reference-axis projection rather than bearing it.

    An exactly-vertical axis is the textbook singularity, but floating point rarely delivers
    it: rotating device ``+y`` into a wall pose leaves a horizontal projection of ``2.2e-16``,
    from which ``atan2`` returns a confident, arbitrary ``180°``. The conditioning decision is
    therefore made on the axis **elevation** against the mode's configured pose limit, and the
    exact-zero check remains only as the degenerate special case.
    """
    device_vector = MODE_REFERENCE_VECTORS[mode]
    enu = device_vector_to_reference_enu(attitude, device_vector)
    elevation = enu_elevation_deg(enu)
    if abs(elevation) > max_axis_elevation_abs_deg:
        raise SingularProjection(
            f"{mode.value} reference axis is {elevation:.6f}° from horizontal, beyond the "
            f"configured {max_axis_elevation_abs_deg}° limit; its horizontal bearing is "
            "ill-conditioned and MUST NOT be resolved on a different axis (§11.1, §14)"
        )
    return AxisBearing(
        heading_deg=enu_bearing_deg(enu),
        elevation_deg=elevation,
        horizontal_norm=enu.horizontal_norm,
    )


def transform_agreement_deg(attitude: CanonicalAttitude, mode: MeasurementMode) -> float:
    """§11.1/§16.1 ``transformAgreementDeg`` for two same-observation extraction routes.

    Direct quaternion-vector rotation versus the rotation-matrix route, over the **same**
    canonical attitude and the **same** physical axis. A large value is a code fault —
    frame transform, quaternion order, axis selection or remapping — and MUST NOT contribute
    to ``MagneticState``: telling a user to move away from metal because the wall-mode
    quaternion has a swapped axis is a failure that survives a long time in the field.
    """
    from .circular import absolute_circular_difference_deg  # local: keep the one §9 owner

    device_vector = MODE_REFERENCE_VECTORS[mode]
    quaternion = attitude.quaternion_device_to_reference_enu_xyzw
    direct = enu_bearing_deg(rotate_vector_by_quaternion(quaternion, device_vector))
    via_matrix = enu_bearing_deg(
        rotate_vector_by_matrix(rotation_matrix_from_quaternion(quaternion), device_vector)
    )
    return absolute_circular_difference_deg(direct, via_matrix)


def sitting_from_facing_deg(facing_deg: float) -> float:
    """§14: sitting (坐) is exactly ``normalize360(facing + 180)``, computed only on request.

    Kept as a named function so the derived opposite is always labelled as such. Reporting
    the wrong one is a 180° error that looks entirely plausible on a dial (failure mode 18).
    """
    return normalize360(facing_deg + 180.0)
