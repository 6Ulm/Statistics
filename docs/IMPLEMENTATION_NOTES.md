# Implementation notes — deviations, substitutions, unsupported capabilities

SPEC.md §37 rules 4 and 8, and §37.1, require every API substitution, missing capability and
deviation to be recorded here rather than silently changing the architecture.

**Phase implemented: Phase 0** (SPEC.md §36 "repo, schemas, pinned tools").
**Status: `SCAFFOLDED`** (§36.1). See the bottom of this file for what that excludes.

---

## Deviations

### D-1 — Android SDK and Android Gradle Plugin unavailable; Android modules not configured

*Rule touched:* §36 Phase 0 exit, "both skeletons build".

`dl.google.com` is unreachable from the environment this scaffold was built in (the egress
proxy denies the CONNECT), so neither the Android SDK nor the Android Gradle Plugin can be
downloaded. `:app`, `:heading-google`, `:heading-diagnostics` and `:benchmark-mode` therefore
cannot be configured or compiled here.

What was done instead, and why it preserves the boundaries §4.1 requires:

- `heading-core` and `fengshui-core` are plain **Kotlin/JVM** modules, not Android library
  modules. §4.1 already requires them to be pure — "no UI or framework singleton" — so a JVM
  library is the honest shape for them, they are consumable unchanged by an Android module,
  and it lets the load-bearing Phase 0 gates actually execute on any JDK.
- The four Android modules exist with complete, pinned `build.gradle.kts` files and manifests,
  declaring their §4.1 dependency directions (including `:benchmark-mode` as a **debug-only**
  dependency of `:app`, per §23 and §29.7).
- `settings.gradle.kts` includes them by default. Passing `-PfscIncludeAndroidModules=false`
  excludes them and **prints exactly which modules were excluded and that the pure-core suites
  do not cover Android adapter code** — a visible degraded state rather than a silent skip
  (§37 conflict priority 6).

*Obligation:* on a host with the Android SDK, run
`cd android && ./gradlew :heading-core:test :fengshui-core:test :app:assembleDebug` (or set
`FSC_ANDROID_SDK_AVAILABLE=true` for `scripts/ci-phase0.sh`) and confirm the Android skeleton
builds before Phase 2 begins.

### D-2 — NOAA WMM sources not vendored; `ncei.noaa.gov` unreachable

*Rule touched:* §2.3 level 2, §10, §35.

`www.ncei.noaa.gov` is blocked from this environment. Vendoring the official C sources, both
coefficient sets and both error models is a **Phase 1** deliverable (§36), so its absence does
not fail a Phase 0 gate — but nothing was approximated. No coefficient, no error-model formula
and no hash was written from memory: §10.3 states that an implementation which "derives a sigma
from the coefficients, or substitutes a remembered global constant, has invented the quantity",
and §5 forbids interchanging *missing* with *zero*.

`third_party/noaa-wmm/UPSTREAM.md` records the declared `NOT_VENDORED` state and exactly what
Phase 1 must fetch and hash. `scripts/verify-artifacts.sh` reports that state explicitly and
fails the moment artifacts and hashes stop agreeing. The example telemetry carries the literal
`NOT_VENDORED` in `declinationCoefficientSha256` and `declinationErrorModelSha256`.

### D-3 — `ios/FengShuiCompass.xcodeproj` absent; Swift toolchain unavailable

*Rules touched:* §4.1 layout, §36 Phase 0 exit "both skeletons build".

There is no Swift compiler and no Xcode in this environment, and an `.xcodeproj` is a generated
artifact whose `project.pbxproj` encodes build settings, target membership and signing. Writing
one by hand on Linux produces a file nothing here can open, parse or validate — a placeholder
presented as a deliverable, which §37 forbids.

What exists instead: `ios/Package.swift` declares the §4.1 module boundaries — `HeadingCore`,
`FengShuiCore`, `HeadingApple`, `HeadingDiagnostics`, `BenchmarkMode` — with their dependency
directions and the iOS 17 platform floor, which is the part §4.1 says MUST NOT vary. See
`ios/XCODEPROJ.md`.

**Consequence, stated plainly: the iOS Swift sources in this change set have never been
compiled or executed.** They were written to mirror the Kotlin implementations case for case
against the same repository-root artifacts, but "mirrors the Kotlin tests" is not evidence that
they compile. Phase 0's "both skeletons build" criterion is **met for Android's pure core and
not met for iOS**.

*Obligation:* on a macOS host, run `cd ios && swift test`, fix whatever it reports, create the
app Xcode project referencing this package, and delete `ios/XCODEPROJ.md`, before reporting
Phase 0 complete for iOS.

### D-4 — Gradle wrapper has no `distributionSha256Sum`

*Rule touched:* §2, "Pin every dependency version."

The wrapper pins an exact distribution
(`https://services.gradle.org/distributions/gradle-8.14.3-bin.zip`), which satisfies version
pinning. The additional `distributionSha256Sum` supply-chain check is absent because
`downloads.gradle.org`, which `services.gradle.org` redirects to for the published `.sha256`
file, is blocked here, and inventing the digest is not an option.

`scripts/verify-artifacts.sh` emits a `WARN` naming this deviation — it does not fail, because
the version *is* pinned, and it does not pass silently either.

*Obligation:* fetch `gradle-8.14.3-bin.zip.sha256` on a networked host, add
`distributionSha256Sum=<value>` to `android/gradle/wrapper/gradle-wrapper.properties`, and
change the `WARN` in `verify-artifacts.sh` to a hard failure.

### D-5 — JDK 17 toolchain installed at build time

*Not a spec deviation; recorded for reproducibility.* The pure-core modules pin
`jvmToolchain(17)` to match the Android modules' `JavaVersion.VERSION_17`. The container shipped
only JDK 21, so `openjdk-17-jdk-headless` (17.0.20+8-1~24.04) was installed. Gradle's toolchain
auto-download is not configured, so a build host must provide a JDK 17.

---

## Findings

### F-1 — `normalize360` is not bit-exact for values already in `[0, 360)`

§9 mandates `((x % 360) + 360) % 360`, and §9 also names `359.9999999` as a required test case.
Those two requirements interact: for a value already in range, the round trip through `+360`
and `% 360` loses low bits, so

```
normalize360(359.9999999) == 359.9999998999999...
```

The residual is about `1e-10`. This is **not** a defect to fix by special-casing in-range
inputs — that would abandon the mandated formula, and §9's form exists because the language
remainder operator differs for negatives (failure mode 2). Two reasons it is safe as specified:

1. `1e-10` is three orders of magnitude inside the `1e-6°` cross-runtime tolerance §36 Phase 1
   declares, and roughly seven orders below the tightest gate in §8 (`transformAgreementMaxDeg`).
2. Kotlin, Swift and Python perform the same IEEE-754 operations in the same order, so all
   three runtimes produce the same bits; parity is unaffected.

Both `CircularMathTest.kt` and `CircularMathTests.swift` pin the observed value exactly, so a
change that enlarged the residual surfaces as a test failure rather than as a bearing.

### F-2 — No §8.1 invariant or §8.1.1 claim failed against the shipped constants

Every §8.1 invariant holds for `precision-v1-candidate-1` as published in SPEC.md §8, and every
grade claim in `testdata/grade-reachability-claims-v1.json` is arithmetically reachable. The
three §8.1.1 consequences reproduce exactly: flat-freehand instrument budget `2.0°` against a
`4.0°` unknown-device floor; wall-freehand budget `0.0°`; `suspectInterferenceBound95Deg = 3.0°`
exceeding both. Nothing was adjusted to reach that result.

Because a passing gate over a hand-written table proves little on its own, each runtime also
contains a **discrimination test**: the invariant checker, the reachability verifier and the
ruleset geometry check are each fed a deliberately broken in-memory copy and must report the
specific violation. No shipped artifact is mutated (§37 rule 12).

---

## API-signature verifications outstanding

§37 rule 4 requires verifying the **installed** API signature before use and recording any
substitution. Phase 0 uses no platform API: the pure cores touch only the standard library, and
the provider modules contain no code. The §12 / §13 signatures — `FusedOrientationProviderClient`,
`DeviceOrientationRequest.OUTPUT_PERIOD_DEFAULT`, `DeviceOrientation.getAttitude()`,
`CurrentLocationRequest.Builder`, `CLLocationManager`/`CLHeading`, `CMMotionManager`
`.xTrueNorthZVertical`, `availableAttitudeReferenceFrames()`, `CMCalibratedMagneticField` — must
be verified against the pinned SDKs in Phase 2, and any substitution recorded in this section.

---

## What `SCAFFOLDED` excludes

§36.1: `SCAFFOLDED` means "projects and schemas exist; provider/core behavior incomplete". It is
**not** `IMPLEMENTED_UNCERTIFIED`, which §37.1 gates on compiling native projects, provider
adapters, benchmark capture/export and a Precision UI — none of which exist. It is emphatically
not `CERTIFIED_FOR_LISTED_DEVICES`, which requires Phases 5–6 physical evidence.

No physical accuracy, battery or thermal result is claimed. Nothing in this change set has run
on a phone.
