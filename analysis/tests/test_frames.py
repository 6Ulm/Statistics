"""SPEC.md §3 / §11.1 / §14 — frames, quaternion order and mode axis projection."""

from __future__ import annotations

import math

import pytest

from fscompass_analysis import fixtures, frames
from fscompass_analysis.enums import MeasurementMode, ProviderReferenceContract


@pytest.fixture(scope="module")
def attitude_fixture():
    return fixtures.load(fixtures.ATTITUDE_GOLDEN)


def _attitude(components: list[float]) -> frames.CanonicalAttitude:
    return frames.CanonicalAttitude(
        quaternion_device_to_reference_enu_xyzw=frames.Quaternion(*components),
        reference_contract=ProviderReferenceContract.TRUE,
        native_frame=frames.NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
    )


def test_fixture_declares_the_conventions_the_types_encode(attitude_fixture):
    assert attitude_fixture["quaternionComponentOrder"] == "XYZW"
    assert attitude_fixture["transformDirection"] == "DEVICE_TO_REFERENCE"
    assert attitude_fixture["canonicalFrame"] == "REFERENCE_ENU"
    assert attitude_fixture["modeReferenceVectors"]["FLAT_TOP_EDGE"] == [0.0, 1.0, 0.0]
    assert attitude_fixture["modeReferenceVectors"]["WALL_FLUSH_BACK"] == [0.0, 0.0, 1.0]


def test_golden_poses_project_to_the_expected_bearing(attitude_fixture, profile):
    """§11.1/§14: physical N/E/S/W golden poses, per mode, on the mode's own axis."""
    for case in attitude_fixture["cases"]:
        attitude = _attitude(case["quaternionDeviceToReferenceEnuXyzw"])
        mode = MeasurementMode(case["measurementMode"])
        gate = profile[attitude_fixture["modeAxisElevationGateKey"][mode.value]]
        if case.get("expectIllConditioned"):
            with pytest.raises(frames.SingularProjection):
                frames.mode_axis_bearing_or_reject(attitude, mode, gate)
            continue
        bearing = frames.mode_axis_bearing_or_reject(attitude, mode, gate)
        assert bearing.heading_deg == pytest.approx(case["expectedHeadingDeg"], abs=1e-9), case[
            "id"
        ]
        assert bearing.elevation_deg == pytest.approx(
            case["expectedElevationDeg"], abs=1e-9
        ), case["id"]


def test_only_an_exactly_vertical_axis_is_caught_by_the_zero_check(attitude_fixture):
    """The reason the conditioning gate is stated on elevation, not on a zero test.

    ``wall-axis-vertical-singular`` projects to an exact ``(0, 0)`` and the bearing function
    itself refuses. ``flat-axis-vertical-ill-conditioned`` projects to ``2.2e-16`` — the zero
    check passes it and ``atan2`` hands back a confident, arbitrary bearing. Only the
    configured elevation gate rejects the second, which is why the naive check is not the
    protection.
    """
    for case in attitude_fixture["cases"]:
        if not case.get("expectIllConditioned"):
            continue
        attitude = _attitude(case["quaternionDeviceToReferenceEnuXyzw"])
        mode = MeasurementMode(case["measurementMode"])
        enu = frames.device_vector_to_reference_enu(attitude, frames.MODE_REFERENCE_VECTORS[mode])
        if case["expectExactlySingular"]:
            assert enu.horizontal_norm == 0.0
            with pytest.raises(frames.SingularProjection):
                frames.enu_bearing_deg(enu)
        else:
            assert 0.0 < enu.horizontal_norm < 1e-12
            # The naive check would let this through with a plausible-looking bearing.
            assert 0.0 <= frames.enu_bearing_deg(enu) < 360.0
        assert abs(frames.enu_elevation_deg(enu)) == pytest.approx(90.0, abs=1e-6)


def test_the_two_extraction_routes_agree(attitude_fixture):
    """§11.1/§16.1: ``transformAgreementDeg`` should be ~0; non-zero is a code defect.

    Direct quaternion-vector rotation versus the rotation-matrix route, over the same
    canonical attitude and the same physical axis.
    """
    for case in attitude_fixture["cases"]:
        if case.get("expectIllConditioned"):
            continue
        attitude = _attitude(case["quaternionDeviceToReferenceEnuXyzw"])
        agreement = frames.transform_agreement_deg(attitude, MeasurementMode(case["measurementMode"]))
        assert agreement == pytest.approx(0.0, abs=1e-9), case["id"]


def test_wall_mode_does_not_fall_through_to_the_top_edge(attitude_fixture, profile):
    """R50/§11.1: in a wall pose the top edge is vertical and its bearing is ill-conditioned.

    Selecting the axis by mode rather than by caller is what makes that structural: asking
    for the flat axis in a wall pose raises instead of returning a plausible number.
    """
    wall_case = next(
        case for case in attitude_fixture["cases"] if case["id"] == "wall-outward-normal-east"
    )
    attitude = _attitude(wall_case["quaternionDeviceToReferenceEnuXyzw"])
    assert frames.mode_axis_bearing_or_reject(
        attitude, MeasurementMode.WALL_FLUSH_BACK, profile["wallNormalElevationAbsMaxDeg"]
    ).heading_deg == pytest.approx(90.0, abs=1e-9)
    with pytest.raises(frames.SingularProjection):
        frames.mode_axis_bearing_or_reject(
            attitude, MeasurementMode.FLAT_TOP_EDGE, profile["flatModePitchAbsMaxDeg"]
        )


def test_core_motion_native_conversion_matches_the_declared_convention(attitude_fixture):
    """R49: the adapter converts both axis convention **and** transform direction.

    Phase 1 pins the conversion *given* the declared convention. The convention itself is
    marked ``DECLARED_UNVERIFIED`` and Phase 2 must confirm it against the pinned SDK with
    physical poses — this test does not and cannot establish it.
    """
    for case in attitude_fixture["nativeFrameConversion"]:
        assert case["verification"] == "DECLARED_UNVERIFIED"
        convention = frames.NATIVE_CONVENTIONS[frames.NativeAttitudeFrame(case["nativeFrame"])]
        assert convention.verification is frames.ConventionVerification.DECLARED_UNVERIFIED
        canonical = frames.canonicalize_native_attitude(
            frames.Quaternion(*case["nativeQuaternionXyzw"]), convention
        )
        expected = case["expectedCanonicalQuaternionXyzw"]
        observed = canonical.quaternion_device_to_reference_enu_xyzw
        # A quaternion and its negation are the same rotation, so compare the rotation.
        for device_vector in (
            frames.Vector3(1.0, 0.0, 0.0),
            frames.Vector3(0.0, 1.0, 0.0),
            frames.Vector3(0.0, 0.0, 1.0),
        ):
            got = frames.rotate_vector_by_quaternion(observed, device_vector)
            want = frames.rotate_vector_by_quaternion(frames.Quaternion(*expected), device_vector)
            assert got.east == pytest.approx(want.east, abs=1e-12)
            assert got.north == pytest.approx(want.north, abs=1e-12)
            assert got.up == pytest.approx(want.up, abs=1e-12)
        assert frames.mode_reference_vector_heading_deg(
            canonical, MeasurementMode.FLAT_TOP_EDGE
        ).heading_deg == pytest.approx(case["expectedFlatHeadingDeg"], abs=1e-9)
        assert canonical.native_frame is convention.frame
        assert canonical.reference_contract is convention.reference_contract


def test_every_declared_native_convention_is_a_proper_rotation():
    """A reflection in the axis map would mirror every bearing without any error."""
    for convention in frames.NATIVE_CONVENTIONS.values():
        assert convention.permutation_determinant() == pytest.approx(1.0, abs=1e-12)


def test_a_reflecting_axis_map_is_rejected():
    with pytest.raises(frames.FrameError):
        frames.NativeAttitudeConvention(
            frame=frames.NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
            east=frames.AxisSelector(0, 1.0),
            north=frames.AxisSelector(1, 1.0),
            up=frames.AxisSelector(2, -1.0),  # flips handedness
            direction=frames.TransformDirection.DEVICE_TO_REFERENCE,
            reference_contract=ProviderReferenceContract.TRUE,
            verification=frames.ConventionVerification.DECLARED_UNVERIFIED,
        )


def test_transform_direction_is_converted_not_assumed():
    """Failure mode 5 includes active/passive inversion, which produces a plausible bearing."""
    device_to_reference = frames.Quaternion(0.0, 0.0, -0.7071067811865475, 0.7071067811865476)
    reference_to_device_convention = frames.NativeAttitudeConvention(
        frame=frames.NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
        east=frames.AxisSelector(0, 1.0),
        north=frames.AxisSelector(1, 1.0),
        up=frames.AxisSelector(2, 1.0),
        direction=frames.TransformDirection.REFERENCE_TO_DEVICE,
        reference_contract=ProviderReferenceContract.TRUE,
        verification=frames.ConventionVerification.DECLARED_UNVERIFIED,
    )
    # Feeding the *inverse* through a REFERENCE_TO_DEVICE convention must recover the same
    # canonical attitude the DEVICE_TO_REFERENCE convention produces from the original.
    canonical = frames.canonicalize_native_attitude(
        device_to_reference.conjugate(), reference_to_device_convention
    )
    heading = frames.mode_reference_vector_heading_deg(
        canonical, MeasurementMode.FLAT_TOP_EDGE
    ).heading_deg
    assert heading == pytest.approx(90.0, abs=1e-9)


def test_quaternion_and_matrix_round_trip(attitude_fixture):
    for case in attitude_fixture["cases"]:
        quaternion = frames.Quaternion(*case["quaternionDeviceToReferenceEnuXyzw"])
        recovered = frames.quaternion_from_rotation_matrix(
            frames.rotation_matrix_from_quaternion(quaternion)
        )
        for device_vector in (
            frames.Vector3(1.0, 0.0, 0.0),
            frames.Vector3(0.0, 1.0, 0.0),
            frames.Vector3(0.0, 0.0, 1.0),
        ):
            original = frames.rotate_vector_by_quaternion(quaternion, device_vector)
            again = frames.rotate_vector_by_quaternion(recovered, device_vector)
            assert original.east == pytest.approx(again.east, abs=1e-12), case["id"]
            assert original.north == pytest.approx(again.north, abs=1e-12), case["id"]
            assert original.up == pytest.approx(again.up, abs=1e-12), case["id"]


def test_canonical_attitude_requires_a_unit_quaternion():
    """§11.1: normalization happens at the adapter boundary, not inside core math."""
    with pytest.raises(frames.FrameError):
        frames.CanonicalAttitude(
            quaternion_device_to_reference_enu_xyzw=frames.Quaternion(0.0, 0.0, 0.0, 0.5),
            reference_contract=ProviderReferenceContract.TRUE,
            native_frame=frames.NativeAttitudeFrame.REPLAY_REFERENCE_ENU,
        )


def test_nonfinite_components_are_rejected():
    with pytest.raises(frames.FrameError):
        frames.Quaternion(float("nan"), 0.0, 0.0, 1.0)
    with pytest.raises(frames.FrameError):
        frames.Vector3(0.0, float("inf"), 0.0)
    with pytest.raises(frames.FrameError):
        frames.EnuVector(0.0, 0.0, float("nan"))


def test_bearing_uses_atan2_east_north_not_north_east():
    """§14: ``normalize360(degrees(atan2(east, north)))``. Swapping the arguments mirrors
    every bearing about north, which looks entirely plausible on a dial."""
    assert frames.enu_bearing_deg(frames.EnuVector(1.0, 0.0, 0.0)) == pytest.approx(90.0)
    assert frames.enu_bearing_deg(frames.EnuVector(0.0, 1.0, 0.0)) == pytest.approx(0.0)
    assert frames.enu_bearing_deg(frames.EnuVector(-1.0, 0.0, 0.0)) == pytest.approx(270.0)
    assert frames.enu_bearing_deg(frames.EnuVector(1.0, 1.0, 0.0)) == pytest.approx(45.0)


def test_a_vertical_axis_has_no_bearing():
    with pytest.raises(frames.SingularProjection):
        frames.enu_bearing_deg(frames.EnuVector(0.0, 0.0, 1.0))


def test_elevation_is_the_conditioning_measure():
    assert frames.enu_elevation_deg(frames.EnuVector(0.0, 1.0, 0.0)) == pytest.approx(0.0)
    assert frames.enu_elevation_deg(frames.EnuVector(0.0, 0.0, 1.0)) == pytest.approx(90.0)
    assert frames.enu_elevation_deg(
        frames.EnuVector(0.0, math.cos(math.radians(5.0)), math.sin(math.radians(5.0)))
    ) == pytest.approx(5.0, abs=1e-9)


def test_sitting_is_the_labelled_derived_opposite_of_facing():
    """§14 / failure mode 18: reporting the wrong one is a 180° error that looks plausible."""
    assert frames.sitting_from_facing_deg(90.0) == pytest.approx(270.0)
    assert frames.sitting_from_facing_deg(350.0) == pytest.approx(170.0)
    assert frames.sitting_from_facing_deg(0.0) == pytest.approx(180.0)
