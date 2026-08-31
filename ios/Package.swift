// swift-tools-version:5.9
// SPEC.md §2: Swift, SwiftUI, Core Location, Core Motion; target iOS 17.0+.
//
// §4.1 module boundaries are declared here rather than only inside an Xcode project, so
// the pure targets build and test on any Swift toolchain and the boundaries are readable
// without Xcode. `HeadingCore` and `FengShuiCore` are pure: no UIKit, no SwiftUI, no
// framework singleton. `HeadingApple` wraps Core Location / Core Motion, `HeadingDiagnostics`
// owns raw sensor streams and never becomes the production estimator, and `BenchmarkMode`
// is internal-build only and depends on the same production core.
//
// The app target (ios/FengShuiCompass) and the Xcode project are assembled on a macOS host;
// see ios/XCODEPROJ.md and docs/IMPLEMENTATION_NOTES.md deviation D-3.
//
// No external package dependencies: SPEC.md §2.3 places a maintained pinned library below
// the platform primitives, and Phase 0 needs none. Adding one requires an exact version pin.

import PackageDescription

let package = Package(
    name: "FengShuiCompassCore",
    platforms: [
        .iOS(.v17),
        .macOS(.v13)   // host-side test execution only; no product ships for macOS.
    ],
    products: [
        .library(name: "HeadingCore", targets: ["HeadingCore"]),
        .library(name: "FengShuiCore", targets: ["FengShuiCore"]),
        .library(name: "HeadingApple", targets: ["HeadingApple"]),
        .library(name: "HeadingDiagnostics", targets: ["HeadingDiagnostics"]),
        .library(name: "BenchmarkMode", targets: ["BenchmarkMode"]),
    ],
    targets: [
        .target(name: "HeadingCore", path: "HeadingCore/Sources/HeadingCore"),
        .target(name: "FengShuiCore", dependencies: ["HeadingCore"], path: "FengShuiCore/Sources/FengShuiCore"),
        .target(name: "HeadingApple", dependencies: ["HeadingCore"], path: "HeadingApple/Sources/HeadingApple"),
        .target(name: "HeadingDiagnostics", dependencies: ["HeadingCore"], path: "HeadingDiagnostics/Sources/HeadingDiagnostics"),
        .target(name: "BenchmarkMode", dependencies: ["HeadingCore", "FengShuiCore"], path: "BenchmarkMode/Sources/BenchmarkMode"),
        .testTarget(name: "HeadingCoreTests", dependencies: ["HeadingCore"], path: "HeadingCore/Tests/HeadingCoreTests"),
        .testTarget(name: "FengShuiCoreTests", dependencies: ["FengShuiCore"], path: "FengShuiCore/Tests/FengShuiCoreTests"),
    ]
)
