# Implementation notes — deviations, substitutions, unsupported capabilities

SPEC.md §37 rules 4 and 8, and §37.1, require every API substitution, missing capability and
deviation to be recorded here rather than silently changing the architecture.

**Phase implemented: Phase 1** (SPEC.md §36 "pure core and WMM wrapper").
**Status: `SCAFFOLDED`** (§36.1) — unchanged from Phase 0, because two Phase 1 exit criteria
are unmet for reasons recorded below (D-2, D-3). See "What `SCAFFOLDED` excludes" at the end.

---

## Deviations

### D-1 — Android SDK and Android Gradle Plugin unavailable; Android app module not built

*Rule touched:* §36 Phase 0 exit, "both skeletons build". **Still open.**

`dl.google.com` is unreachable from this environment, so neither the Android SDK nor the
Android Gradle Plugin can be downloaded. `:app`, `:heading-google`, `:heading-diagnostics` and
`:benchmark-mode` cannot be configured or compiled here.

What is true as of Phase 1: `heading-core` and `fengshui-core` are plain **Kotlin/JVM** modules
(§4.1 already requires them to be pure — "no UI or framework singleton"), they compile, and
their 193 tests run and pass on JDK 17. The four Android modules exist with complete, pinned
`build.gradle.kts` files and manifests declaring their §4.1 dependency directions.
`-PfscIncludeAndroidModules=false` excludes them and prints exactly which modules were excluded
— a visible degraded state rather than a silent skip (§37 conflict priority 6).

*Obligation, unchanged:* on a host with the Android SDK, run
`cd android && ./gradlew :heading-core:test :fengshui-core:test :app:assembleDebug` (or set
`FSC_ANDROID_SDK_AVAILABLE=true` for `scripts/ci-phase1.sh`) and confirm the Android skeleton
builds before Phase 2 begins.

### D-2 — NOAA WMM sources not vendored; `ncei.noaa.gov` unreachable (Phase 1 blocker)

*Rules touched:* §2.3 level 2, §10, §35. **Still open, and now blocking.**

Phase 0 recorded this as deferred to Phase 1. Phase 1 retried and failed:
`www.ncei.noaa.gov`, `ncei.noaa.gov`, `www.ngdc.noaa.gov` and `noaa.gov` are all denied by the
egress policy (the agent proxy logs `connect_rejected` — "gateway answered 403 to CONNECT
(policy denial)" — for each). A third-party mirror was **not** substituted: §10 requires "the
exact NOAA package" with its own URL, date and per-file SHA-256, and a digest computed over a
mirror cannot be checked against NOAA's published one.

**Consequence, stated plainly: §36 Phase 1 cannot exit.** "Both platforms execute the same
hashed NOAA C with no home-grown model" is unmet, and no declination, no
`declinationSigma1Deg`, no `declinationModelBound95Deg`, no expected-field magnitude or
inclination, and therefore no §16 magnetic classification against a real model can be computed
by this repository today.

No coefficient value, no error-model formula and no hash was written from memory (§10.3, §5).

What was built instead of a guess — the typed surface exists and refuses:

- `Geomagnetic.vendoredArtifacts(...).requireVendored(op)` throws
  `VendoredModelUnavailableException` naming the missing artifact and the rule
  (`GeomagneticTest.noSigmaCanBeProducedWithoutAVendoredErrorModel`,
  `test_no_sigma_can_be_produced_without_a_vendored_error_model`).
- `GeomagneticModelUncertainty` cannot exist without an `errorModelHash`, and refuses any
  `sourceConfidenceLevel` but `ONE_STANDARD_DEVIATION` (failure mode 9).
- `boundFromSigma` is implemented and tested as §19.2's single conversion site, ready for a
  real sigma.
- `§10.2` altitude-datum conversion is implemented and refuses `MSL_ORTHOMETRIC` without a
  documented geoid separation and `UNKNOWN` always.
- `Geomagnetic.wmm2025Validity` encodes `[2025.0, 2030.0)` — a constant **stated in SPEC.md
  §10 itself**, not read from an absent artifact — and out-of-range dates throw
  `GeomagneticDateOutOfRangeException`.

*Obligation:* on a host that can reach NOAA NCEI, vendor both models per
`third_party/noaa-wmm/UPSTREAM.md`, add the official vectors under `testdata/wmm/`, build the
iOS C-interop and Android NDK/JNI wrappers over the same sources, regenerate `fixtures-v1`, and
re-run every suite.

### D-3 — no Swift toolchain; the iOS core has still never been compiled

*Rules touched:* §4.1 layout, §36 Phase 0 exit "both skeletons build", §36 Phase 1 exit "all
pure tests pass on both platforms". **Still open, and now blocking.**

Phase 0's obligation was to run `cd ios && swift test` and fix what it reported. That was
attempted and could not be done: there is no Swift compiler in this environment,
`download.swift.org` is denied by the egress policy, and the `swift` package in the distribution
repositories is OpenStack Swift, not the language toolchain.

**Consequence, stated plainly: the iOS Swift sources in this repository have never been
compiled or executed.** That was true of the Phase 0 sources and is true of the Phase 1 sources
added on top of them. They were written to mirror the Kotlin implementations case for case
against the same repository-root artifacts, and "mirrors the Kotlin tests" is not evidence that
they compile.

Phase 1 added, uncompiled: `CircularMath.swift` (extended to the full §9/§9.1/§15/§19.2
surface), `Estimators.swift`, `Frames.swift`, `Enums.swift` (the §6 subset the core consumes),
`FengShuiClassifier.swift`, `SharedArtifacts.swift` fixture accessors, `INV-11` in
`ConfigurationInvariants.swift`, and the `CircularMathFixtureTests.swift` /
`FramesTests.swift` suites that read the same frozen `fixtures-v1` files the other two runtimes
read.

**Deliberately not written:** Swift mirrors of §11 reference resolution, §16 magnetic
classification, §19 uncertainty composition, §24 certification, §19.3 deviation correction,
§22.2 telemetry codec and §18.2 the state reducer. Writing several thousand further lines that
no compiler in this environment can check would add unverified surface without adding evidence,
and §37 is explicit that a deliberate unsupported state with a reason beats a placeholder. The
frozen fixtures are what make those modules checkable the moment a toolchain exists; the
Kotlin and Python implementations are the reference for them.

*Obligation:* on a macOS host, run `cd ios && swift test`, fix whatever it reports, write the
remaining Phase 1 modules named above against the same frozen fixtures, create the app Xcode
project referencing this package, and delete `ios/XCODEPROJ.md`.

### D-4 — Gradle wrapper `distributionSha256Sum` — **resolved**

*Rule touched:* §2, "Pin every dependency version."

`android/gradle/wrapper/gradle-wrapper.properties` now carries
`distributionSha256Sum=bd71102213493060956ec229d946beee57158dbd89d0e62b91bca0fa2c5f3531`, and
the `WARN` in `scripts/verify-artifacts.sh` is a hard failure: a missing or malformed digest
fails the script, because the wrapper verifies the download against this value and a missing
entry silently removes that check.

*Substitution, recorded because it differs from D-4's stated obligation.* D-4 said to fetch
`gradle-8.14.3-bin.zip.sha256`. That sidecar is still denied by this environment's egress policy
(`services.gradle.org/...zip.sha256` → CONNECT 403) while the `.zip` itself returns 200. The
digest was therefore computed from the artifact fetched from the **pinned official
`distributionUrl`**, downloaded twice in separate connections with identical results. Both
routes share the same trust root — TLS to `services.gradle.org` through this environment's
proxy — so this is an equivalent pin, not a weaker one, but it is not an *independent* check.

The value is self-verifying and was verified: the wrapper's cached distribution was deleted and
`./gradlew --version` re-downloaded and validated the archive against this digest. A wrong value
would have failed the build immediately.

*Residual obligation:* on a networked host, fetch the published
`gradle-8.14.3-bin.zip.sha256` and confirm it matches, so the pin rests on NOAA-style
independent provenance rather than on a self-computed digest.

### D-5 — JDK 17 toolchain installed at build time

*Not a spec deviation; recorded for reproducibility.* The pure-core modules pin
`jvmToolchain(17)` to match the Android modules' `JavaVersion.VERSION_17`. The container ships
only JDK 21, so `openjdk-17-jdk-headless` (17.0.20+8-1~24.04) was installed. Gradle's toolchain
auto-download is not configured, so a build host must provide a JDK 17.

### D-6 — a comma-decimal locale was installed to make the §22.2 locale test real

*Not a spec deviation; recorded for reproducibility.* §22.2 requires "a test MUST run the export
path under a comma-decimal locale". The container shipped no such locale, so `de_DE.UTF-8` was
generated. The Python test **skips with an explicit message** rather than passing silently when
no comma-decimal locale exists; the Kotlin test sets `Locale.GERMANY` directly and needs no
system locale. A CI host must have one for the Python half of that check to run.

### D-7 — §6 vocabulary consolidated into one package (Kotlin)

*Rule touched:* §6, "There is exactly **one** measurement-state vocabulary" and one enum
vocabulary generally.

Phase 0 declared `PlacementMethod` and `MagneticState` inside
`com.fengshuicompass.headingcore.grade` because the §8.1.1 reachability analysis was the only
consumer. Phase 1 gives §6 a single home in `com.fengshuicompass.headingcore.model`. Rather than
leave two enums with identical case names in one runtime — which is how two parts of one binary
end up disagreeing about a wire value — the `grade` names are now `typealias` declarations
pointing at the `model` ones. No call site changed and no wire value changed.

---

## Findings

### F-1 — `normalize360` is not bit-exact for values already in `[0, 360)` (unchanged)

§9 mandates `((x % 360) + 360) % 360`, and §9 also names `359.9999999` as a required test case.
Those two requirements interact: for a value already in range, the round trip through `+360` and
`% 360` loses low bits, so `normalize360(359.9999999) == 359.9999998999999...`.

Re-read before touching §9 in Phase 1, and **left exactly as specified**. The residual is about
`1e-10`: three orders of magnitude inside the `1e-6°` cross-runtime tolerance §36 Phase 1
declares, and roughly seven below the tightest gate in §8 (`transformAgreementMaxDeg`). Kotlin,
Swift and Python perform the same IEEE-754 operations in the same order, so parity is
unaffected — Phase 1 now demonstrates that rather than asserting it: the value is stored in
`testdata/angles/circular-math-v1.json` and the Kotlin and Python runtimes both reproduce it.

Special-casing in-range inputs would abandon the mandated formula, whose form exists because the
language remainder operator differs for negatives (failure mode 2).

### F-2 — no §8.1 invariant or §8.1.1 claim failed against the shipped constants (unchanged)

Every §8.1 invariant holds for `precision-v1-candidate-1`, now including the new
`INV-11-EVENT-DRIVEN-ANCHOR-MINIMUM`, and every grade claim in
`testdata/grade-reachability-claims-v1.json` is arithmetically reachable. The three §8.1.1
consequences reproduce exactly. Nothing was adjusted to reach that result.

### F-3 — an exact-zero singularity check does **not** catch the ill-conditioned wall/flat axis

Found while writing the §14 golden poses, and it changed the implementation.

Rotating the device `+z` (wall) axis into a face-up pose yields an ENU horizontal projection of
*exactly* `(0, 0)`, which an `== 0.0` check catches. Rotating the device `+y` (flat top edge)
into a wall pose yields `2.220446049250313e-16` — not zero. The naive check passes it and
`atan2` returns a confident, entirely arbitrary bearing from an axis pointing straight up.

§14's wording is "reject if the horizontal projection is **ill-conditioned**", and §18.5 forbids
a gate comparing against a numeric literal. The conditioning decision is therefore made on the
axis **elevation** against the mode's existing configured pose limit —
`flatModePitchAbsMaxDeg` for flat, `wallNormalElevationAbsMaxDeg` for wall — via
`modeAxisBearingOrReject`. No new configuration key was invented; the exact-zero check remains
only as the degenerate special case. Both cases are in
`testdata/quaternions/attitude-golden-v1.json` and both runtimes assert the distinction.

### F-4 — the same shape appears in §15's circular mean, and only the config gate catches it

`atan2(0, 0)` returning `0.0` is the textbook statement of failure mode 6, but an antipodal
window never reaches it: `sin(0°) + sin(180°)` cancels to `1.2246e-16`, not to zero. The mean is
therefore *numerically defined* and comes back as a confident `90°` that means nothing.

This is exactly why §15 states decision 3 on `R` — "If `R < minCircularResultantLength`, emit
`CIRCULAR_MEAN_UNDEFINED` and reject" — rather than on the mean. `circularMeanIsUndefined(
aggregate, minCircularResultantLength)` implements that gate reading the config key, and the
frozen `testdata/angles/circular-aggregate-v1.json` records, per window, both whether the mean
is numerically defined and whether the gate rejects it. An implementation that only guarded
`atan2(0, 0)` would pass a naive test and ship the defect.

### F-5 — a walk-until-equal straddle set collapses a full-circle interval to one mountain

Found while writing the §21.4 straddle fixtures.

The obvious implementation walks sector indices from `sectorIndex(h - bound)` forward until it
reaches `sectorIndex(h + bound)`. At `bound = 179°` both endpoints land in the *same* sector, so
the walk terminates immediately and reports **one** mountain for an interval covering all 24 —
a maximally specific claim from a measurement that discriminates nothing, which is the precise
false-precision failure §21.3 warns about.

The shipped implementation counts from the **arc length** instead, using an offset derived from
the same wrapped quantity the index uses so the two cannot disagree by a rounding bit at a
boundary. `testdata/fengshui/classification-v1.json` carries the `(10°, 179°)` case with its
24-sector expectation, and both runtimes assert it.

### F-6 — the §22.1 example's `gradeLimitedBy` for a wall-freehand variant is
`PLACEMENT_UNCERTAINTY`, not `DEVICE_FLOOR`

While writing `testdata/uncertainty/composition-v1.json` the first draft asserted
`DEVICE_FLOOR` for a wall-freehand case with a `4.0°` unknown floor and a `5.0°` placement
bound. §19 names "the largest numeric uncertainty term", and `5.0 > 4.0`, so the correct value
is `PLACEMENT_UNCERTAINTY`. The fixture was corrected to the arithmetic rather than the
implementation being adjusted to the fixture.

The same case surfaces §19's tie rule: with flat-freehand placement at `3.0°` and a `SUSPECT`
interference term at `3.0°`, the two terms are exactly equal and `gradeLimitedBy` resolves by
stable enum order to `PLACEMENT_UNCERTAINTY`. Both runtimes are asserted against that, because
an unstable tie-break is how two platforms come to name different limiting factors for one
measurement.

---

## API-signature verifications outstanding

§37 rule 4 requires verifying the **installed** API signature before use. Phase 1 uses no
platform API: the pure cores touch only their standard libraries.

Two Phase 1 artifacts nevertheless *encode a claim about* a platform API, and both are marked
`DECLARED_UNVERIFIED` in code rather than assumed:

- **Core Motion `.xTrueNorthZVertical` axis convention and transform direction** (R49). The
  adapter conversion is implemented and pinned by golden vectors *given the declared
  convention*: native `+X` true north, `+Z` vertical, therefore right-handed native `+Y` west
  and `east = -native_y`, transform direction device→reference. §11.1 forbids inferring
  permutation, transpose or signs from yaw intuition, so Phase 2 MUST confirm the convention
  itself against the pinned SDK with physical N/E/S/W/up poses before any sample reaches the
  engine. `ConventionVerification.DECLARED_UNVERIFIED` marks this in the type system.
- **Google FOP `getAttitude()` `[qx, qy, qz, qw]` device→ENU** and **Android
  `TYPE_ROTATION_VECTOR` via `getRotationMatrixFromVector`**, likewise declared and likewise
  `DECLARED_UNVERIFIED`.

The remaining §12 / §13 signatures — `FusedOrientationProviderClient`,
`DeviceOrientationRequest.OUTPUT_PERIOD_DEFAULT`, `CurrentLocationRequest.Builder`,
`CLLocationManager`/`CLHeading`, `CMMotionManager`, `availableAttitudeReferenceFrames()`,
`CMCalibratedMagneticField` — must be verified against the pinned SDKs in Phase 2, and any
substitution recorded in this section.

---

## What `SCAFFOLDED` excludes

§36.1: `SCAFFOLDED` means "projects and schemas exist; provider/core behavior incomplete".

Phase 1 moved a great deal of core behaviour from absent to implemented-and-tested — §9/§9.1
utilities and pinned estimators, §3/§11.1 frames and quaternion conventions, §11 reference
resolution, §16 magnetic classification, §19 composition, §19.3 deviation types, §21
classification and straddle, §22.2 the telemetry codec, §24 certification keys, §18.2 the
reducer — but it did **not** meet Phase 1's exit criteria, so the status does not advance:

- **D-2**: no vendored NOAA WMM, so "both platforms execute the same hashed NOAA C" is unmet
  and no geomagnetic quantity can be produced at all.
- **D-3**: no Swift toolchain, so "all pure tests pass on both platforms" is unmet for iOS —
  the iOS sources have never been compiled.
- **D-1**: the Android application target has still never been built.

It is emphatically not `IMPLEMENTED_UNCERTIFIED`, which §37.1 gates on compiling native
projects, provider adapters, benchmark capture/export and a Precision UI — none of which exist.
It is not `CERTIFIED_FOR_LISTED_DEVICES`, which requires Phases 5–6 physical evidence.

No physical accuracy, battery or thermal result is claimed. Nothing in this repository has run
on a phone.
