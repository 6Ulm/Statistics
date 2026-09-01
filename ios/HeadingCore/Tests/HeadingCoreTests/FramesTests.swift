import XCTest
@testable import HeadingCore

/// SPEC.md §3 / §11.1 / §14 — frames, quaternion order and mode axis projection, iOS runtime.
///
/// > Warning: this file has never been compiled or executed — see
/// > `docs/IMPLEMENTATION_NOTES.md` D-3. On a macOS host, run `cd ios && swift test`.
final class FramesTests: XCTestCase {

    private func attitudeFixture() throws -> [String: Any] {
        let data = try Data(contentsOf: SharedArtifacts.attitudeGoldenFixtureURL())
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "fixture", code: 1)
        }
        return object
    }

    private func profile() throws -> [String: Any] {
        let data = try Data(contentsOf: SharedArtifacts.precisionProfileURL())
        guard let object = try JSONSerialization.jsonObject(with: data) as? [String: Any] else {
            throw NSError(domain: "profile", code: 1)
        }
        return object
    }

    private func attitude(_ components: [Double]) throws -> CanonicalAttitude {
        try CanonicalAttitude(
            quaternionDeviceToReferenceEnuXYZW: Quaternion(
                x: components[0], y: components[1], z: components[2], w: components[3]
            ),
            referenceContract: .trueReference,
            nativeFrame: .replayReferenceEnu
        )
    }

    private func gate(for mode: MeasurementMode) throws -> Double {
        let document = try attitudeFixture()
        let keys = try XCTUnwrap(document["modeAxisElevationGateKey"] as? [String: String])
        let key = try XCTUnwrap(keys[mode.wire])
        return try XCTUnwrap(profile()[key] as? Double)
    }

    func testFixtureDeclaresTheConventionsTheTypesEncode() throws {
        let document = try attitudeFixture()
        XCTAssertEqual(document["quaternionComponentOrder"] as? String, "XYZW")
        XCTAssertEqual(document["transformDirection"] as? String, "DEVICE_TO_REFERENCE")
        XCTAssertEqual(document["canonicalFrame"] as? String, "REFERENCE_ENU")
        let vectors = try XCTUnwrap(document["modeReferenceVectors"] as? [String: [Double]])
        XCTAssertEqual(vectors["FLAT_TOP_EDGE"], [0.0, 1.0, 0.0])
        XCTAssertEqual(vectors["WALL_FLUSH_BACK"], [0.0, 0.0, 1.0])
    }

    /// §11.1/§14: physical N/E/S/W golden poses, per mode, on the mode's own axis.
    func testGoldenPosesProjectToTheExpectedBearing() throws {
        for entry in try XCTUnwrap(attitudeFixture()["cases"] as? [[String: Any]]) {
            let id = try XCTUnwrap(entry["id"] as? String)
            let mode = try XCTUnwrap(
                MeasurementMode(rawValue: try XCTUnwrap(entry["measurementMode"] as? String))
            )
            let canonical = try attitude(
                try XCTUnwrap(entry["quaternionDeviceToReferenceEnuXyzw"] as? [Double])
            )
            let limit = try gate(for: mode)
            if (entry["expectIllConditioned"] as? Bool) == true {
                XCTAssertThrowsError(
                    try Frames.modeAxisBearingOrReject(
                        canonical, mode: mode, maxAxisElevationAbsDeg: limit
                    ),
                    id
                )
                continue
            }
            let bearing = try Frames.modeAxisBearingOrReject(
                canonical, mode: mode, maxAxisElevationAbsDeg: limit
            )
            XCTAssertEqual(bearing.headingDeg,
                           try XCTUnwrap(entry["expectedHeadingDeg"] as? Double),
                           accuracy: 1e-9, id)
            XCTAssertEqual(bearing.elevationDeg,
                           try XCTUnwrap(entry["expectedElevationDeg"] as? Double),
                           accuracy: 1e-9, id)
        }
    }

    /// The reason the conditioning gate is stated on elevation, not on a zero test:
    /// `wall-axis-vertical-singular` projects to an exact `(0, 0)`, while
    /// `flat-axis-vertical-ill-conditioned` projects to `2.2e-16` and `atan2` hands back a
    /// confident, arbitrary bearing.
    func testOnlyAnExactlyVerticalAxisIsCaughtByTheZeroCheck() throws {
        for entry in try XCTUnwrap(attitudeFixture()["cases"] as? [[String: Any]]) {
            guard (entry["expectIllConditioned"] as? Bool) == true else { continue }
            let mode = try XCTUnwrap(
                MeasurementMode(rawValue: try XCTUnwrap(entry["measurementMode"] as? String))
            )
            let canonical = try attitude(
                try XCTUnwrap(entry["quaternionDeviceToReferenceEnuXyzw"] as? [Double])
            )
            let enu = try Frames.deviceVectorToReferenceEnu(
                canonical, Frames.modeReferenceVectors[mode]!
            )
            if try XCTUnwrap(entry["expectExactlySingular"] as? Bool) {
                XCTAssertEqual(enu.horizontalNorm, 0.0)
                XCTAssertThrowsError(try Frames.enuBearingDeg(enu))
            } else {
                XCTAssertTrue(enu.horizontalNorm > 0.0 && enu.horizontalNorm < 1e-12)
                XCTAssertNoThrow(try Frames.enuBearingDeg(enu))
            }
            XCTAssertEqual(abs(try Frames.enuElevationDeg(enu)), 90.0, accuracy: 1e-6)
        }
    }

    /// §11.1/§16.1: `transformAgreementDeg` should be ~0; non-zero is a code defect.
    func testTheTwoExtractionRoutesAgree() throws {
        for entry in try XCTUnwrap(attitudeFixture()["cases"] as? [[String: Any]]) {
            guard (entry["expectIllConditioned"] as? Bool) != true else { continue }
            let mode = try XCTUnwrap(
                MeasurementMode(rawValue: try XCTUnwrap(entry["measurementMode"] as? String))
            )
            let canonical = try attitude(
                try XCTUnwrap(entry["quaternionDeviceToReferenceEnuXyzw"] as? [Double])
            )
            XCTAssertEqual(try Frames.transformAgreementDeg(canonical, mode: mode), 0.0,
                           accuracy: 1e-9, try XCTUnwrap(entry["id"] as? String))
        }
    }

    /// R50/§11.1: in a wall pose the top edge is vertical and its bearing is ill-conditioned.
    func testWallModeDoesNotFallThroughToTheTopEdge() throws {
        let cases = try XCTUnwrap(attitudeFixture()["cases"] as? [[String: Any]])
        let wallCase = try XCTUnwrap(
            cases.first { ($0["id"] as? String) == "wall-outward-normal-east" }
        )
        let canonical = try attitude(
            try XCTUnwrap(wallCase["quaternionDeviceToReferenceEnuXyzw"] as? [Double])
        )
        XCTAssertEqual(
            try Frames.modeAxisBearingOrReject(
                canonical,
                mode: .wallFlushBack,
                maxAxisElevationAbsDeg: try gate(for: .wallFlushBack)
            ).headingDeg,
            90.0, accuracy: 1e-9
        )
        XCTAssertThrowsError(
            try Frames.modeAxisBearingOrReject(
                canonical,
                mode: .flatTopEdge,
                maxAxisElevationAbsDeg: try gate(for: .flatTopEdge)
            )
        )
    }

    /// R49: the adapter converts both axis convention **and** transform direction.
    ///
    /// Phase 1 pins the conversion *given* the declared convention. The convention itself is
    /// `DECLARED_UNVERIFIED` and Phase 2 must confirm it against the pinned SDK with physical
    /// poses — this test does not and cannot establish it.
    func testCoreMotionNativeConversionMatchesTheDeclaredConvention() throws {
        let conversions = try XCTUnwrap(
            attitudeFixture()["nativeFrameConversion"] as? [[String: Any]]
        )
        for entry in conversions {
            XCTAssertEqual(entry["verification"] as? String, "DECLARED_UNVERIFIED")
            let frame = try XCTUnwrap(
                NativeAttitudeFrame(rawValue: try XCTUnwrap(entry["nativeFrame"] as? String))
            )
            let convention = try XCTUnwrap(NativeConventions.byFrame[frame])
            XCTAssertEqual(convention.verification, .declaredUnverified)

            let native = try XCTUnwrap(entry["nativeQuaternionXyzw"] as? [Double])
            let canonical = try Frames.canonicalizeNativeAttitude(
                Quaternion(x: native[0], y: native[1], z: native[2], w: native[3]),
                convention: convention
            )
            let expected = try XCTUnwrap(
                entry["expectedCanonicalQuaternionXyzw"] as? [Double]
            )
            let expectedQuaternion = try Quaternion(
                x: expected[0], y: expected[1], z: expected[2], w: expected[3]
            )
            // A quaternion and its negation are the same rotation, so compare the rotation.
            for deviceVector in [
                Vector3(x: 1.0, y: 0.0, z: 0.0),
                Vector3(x: 0.0, y: 1.0, z: 0.0),
                Vector3(x: 0.0, y: 0.0, z: 1.0),
            ] {
                let got = try Frames.rotateVectorByQuaternion(
                    canonical.quaternionDeviceToReferenceEnuXYZW, deviceVector
                )
                let want = try Frames.rotateVectorByQuaternion(expectedQuaternion, deviceVector)
                XCTAssertEqual(got.east, want.east, accuracy: 1e-12)
                XCTAssertEqual(got.north, want.north, accuracy: 1e-12)
                XCTAssertEqual(got.up, want.up, accuracy: 1e-12)
            }
            XCTAssertEqual(
                try Frames.modeReferenceVectorHeadingDeg(canonical, mode: .flatTopEdge).headingDeg,
                try XCTUnwrap(entry["expectedFlatHeadingDeg"] as? Double),
                accuracy: 1e-9
            )
            XCTAssertEqual(canonical.nativeFrame, convention.frame)
            XCTAssertEqual(canonical.referenceContract, convention.referenceContract)
        }
    }

    /// A reflection in the axis map would mirror every bearing without any error.
    func testEveryDeclaredConventionIsAProperRotation() {
        for convention in NativeConventions.byFrame.values {
            XCTAssertEqual(convention.permutationDeterminant(), 1.0, accuracy: 1e-12,
                           convention.frame.wire)
        }
    }

    func testAReflectingAxisMapIsRejected() {
        XCTAssertThrowsError(
            try NativeAttitudeConvention(
                frame: .replayReferenceEnu,
                east: AxisSelector(0, 1.0),
                north: AxisSelector(1, 1.0),
                up: AxisSelector(2, -1.0), // flips handedness
                direction: .deviceToReference,
                referenceContract: .trueReference,
                verification: .declaredUnverified
            )
        )
    }

    /// Failure mode 5 includes active/passive inversion, which produces a plausible bearing.
    func testTransformDirectionIsConvertedNotAssumed() throws {
        let deviceToReference = try Quaternion(
            x: 0.0, y: 0.0, z: -0.7071067811865475, w: 0.7071067811865476
        )
        let referenceToDevice = try NativeAttitudeConvention(
            frame: .replayReferenceEnu,
            east: AxisSelector(0, 1.0),
            north: AxisSelector(1, 1.0),
            up: AxisSelector(2, 1.0),
            direction: .referenceToDevice,
            referenceContract: .trueReference,
            verification: .declaredUnverified
        )
        let canonical = try Frames.canonicalizeNativeAttitude(
            deviceToReference.conjugate(), convention: referenceToDevice
        )
        XCTAssertEqual(
            try Frames.modeReferenceVectorHeadingDeg(canonical, mode: .flatTopEdge).headingDeg,
            90.0, accuracy: 1e-9
        )
    }

    func testCanonicalAttitudeRequiresAUnitQuaternion() {
        XCTAssertThrowsError(
            try CanonicalAttitude(
                quaternionDeviceToReferenceEnuXYZW: Quaternion(x: 0.0, y: 0.0, z: 0.0, w: 0.5),
                referenceContract: .trueReference,
                nativeFrame: .replayReferenceEnu
            )
        )
    }

    /// §14: swapping the `atan2` arguments mirrors every bearing about north, which looks
    /// entirely plausible on a dial.
    func testBearingUsesAtan2EastNorth() throws {
        XCTAssertEqual(try Frames.enuBearingDeg(EnuVector(east: 1.0, north: 0.0, up: 0.0)),
                       90.0, accuracy: 1e-12)
        XCTAssertEqual(try Frames.enuBearingDeg(EnuVector(east: 0.0, north: 1.0, up: 0.0)),
                       0.0, accuracy: 1e-12)
        XCTAssertEqual(try Frames.enuBearingDeg(EnuVector(east: -1.0, north: 0.0, up: 0.0)),
                       270.0, accuracy: 1e-12)
        XCTAssertEqual(try Frames.enuBearingDeg(EnuVector(east: 1.0, north: 1.0, up: 0.0)),
                       45.0, accuracy: 1e-12)
    }

    /// §14/failure mode 18: reporting the wrong one is a 180° error that looks plausible.
    func testSittingIsTheLabelledDerivedOpposite() throws {
        XCTAssertEqual(try Frames.sittingFromFacingDeg(90.0), 270.0, accuracy: 1e-12)
        XCTAssertEqual(try Frames.sittingFromFacingDeg(350.0), 170.0, accuracy: 1e-12)
        XCTAssertEqual(try Frames.sittingFromFacingDeg(0.0), 180.0, accuracy: 1e-12)
    }
}
