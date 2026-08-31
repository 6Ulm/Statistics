# Feng Shui Precision Compass — Engineering Specification

Accuracy-first native iOS + Android compass for Feng Shui practitioners, with a mobile benchmark framework.

**Document authority:** this file is the v1 implementation source of truth. The earlier `feng-shui-precision-compass-spec-and-mobile-benchmark.md` is an archive of rationale only and MUST NOT override this file. Official documentation and vendored artifacts for the exact pinned SDK/provider/model version govern external API facts; if a symbol or documented behavior differs in the pinned toolchain, preserve the required behavior and record the substitution in `docs/IMPLEMENTATION_NOTES.md` rather than silently changing the architecture.

**Reading order for an implementing agent:** §1–§3 (contract) → §4–§11 (core, platform-independent) → §12–§14 (platform adapters) → §15–§22 (measurement pipeline) → §23–§25 (data contracts) → §26–§33 (benchmark) → §34–§36 (risks, checklist, phases).

`MUST` / `MUST NOT` / `SHOULD` / `MAY` carry RFC-2119 strength. Imperative and prohibitive forms — *never*, *always*, *do not*, *reject*, *record* — carry the same force as `MUST NOT` / `MUST` respectively; they are used where they read more naturally, not to signal a weaker requirement. Values marked *candidate* live in versioned configuration and change only through a benchmark result.

---

## 1. Purpose, framing, non-goals

Produce a true-north bearing whose error can be **measured, bounded, and refused**. Refusing is a feature: consumer phones cannot guarantee accurate absolute heading near steel, magnets, conductors, vehicles, or magnetic accessories, and those errors are locally unobservable.

Four framing facts drive the design:

1. **The measurement is a gesture, not a sensor reading.** Freehand alignment can contribute several degrees and can dominate clean-condition sensor error. The planning range `±2°`–`±5°` is a **candidate empirical range**, not a shipped fact: §29.5 measures the per-mode/operator distribution and is the only authority for production placement bounds. Placement uncertainty is a first-class term in the reported bound (§19), and the current candidate profile makes the top grades unreachable freehand (§20).
2. **Trust is per-measurement, not per-phone.** Device certification is prior evidence; every result is still conditioned on live sensor, environment, reference, pose, and placement state.
3. **"Calibration" means three unrelated things** and they MUST stay distinct: *sensor calibration* (OS magnetometer state), *deviation characterization* (azimuth-dependent residual experiment), *uncertainty calibration* (held-out coverage proof). §17.
4. **Ground truth is tiered and Tier 0 is executable by one developer** with no survey equipment. §27.

**Do not build a custom AHRS.** Use the platform fused provider. The app is an orchestrator and verifier: normalize, timestamp-check, resolve the north reference, reject, aggregate, bound, grade, classify.

Non-goals for v1: accounts/cloud/backend; custom EKF or magnetometer calibration; background heading; AR/camera sighting; external instruments; full map integration before heading gates pass; ML confidence models; the complete Luo Pan ring set; production deviation correction.

Explicitly not claimed: sub-degree accuracy on every phone; software removal of external distortion; that provider agreement, GNSS quality, or a stable dial proves heading accuracy; that a quality label is a probability unless empirically calibrated as one.

**Engineering priority:** (1) correct heading and north-reference label; (2) reject false confidence; (3) honest uncertainty, held-out coverage, and repeatability; (4) location/declination correctness; (5) responsiveness/stability; (6) battery/thermal efficiency. A lower-priority objective MUST NOT trade away a higher-priority one without an explicit benchmark decision.

---

## 2. Fixed v1 decisions

| Decision | Value |
|---|---|
| Shape | Two native apps in one repo. No shared runtime (no KMP/RN/Flutter). |
| iOS | Swift, SwiftUI, Core Location, Core Motion; target iOS 17.0+. |
| Android | Kotlin, Compose, coroutines/Flow; `minSdk 26`; SDK pinned at scaffold time. |
| Android heading | Capability-selected at runtime: Google Play services `FusedOrientationProviderClient`, else `TYPE_ROTATION_VECTOR`. Certified separately, never silently substituted. |
| Android location | Google `FusedLocationProviderClient`, else framework `LocationManager`. |
| iOS heading | Mode-specific Apple-native path: `FLAT_TOP_EDGE` uses valid `CLHeading.trueHeading`; `WALL_FLUSH_BACK` uses `CMDeviceMotion` in `.xTrueNorthZVertical` and projects the outward screen normal. Each mode/provider path is certified separately. |
| North reference | True north canonical internally. Feng Shui layer applies school reference/needle offset (§21.2). |
| Measurement modes | `FLAT_TOP_EDGE`, `WALL_FLUSH_BACK`. Separate axes, gates, bounds, certification. |
| Screen orientation | Portrait locked. Phones only; tablets/foldables → `UNSUPPORTED_DEVICE` (§2.1). |
| Geomagnetic model | Vendored NOAA WMM2025 (default) + WMMHR2025 (benchmarked). Same C core compiled both platforms. |
| Altitude datum | WGS84 ellipsoidal canonical; datum always explicit; `UNKNOWN` is a real state. |
| Deviation correction | `NONE` by default; any profile is separately hashed and evidence-gated. |
| Lifecycle | Foreground only. |
| Android request rate | `OUTPUT_PERIOD_DEFAULT` (requested 50 Hz). |
| Uncertainty label | `95%` only after held-out coverage passes for the exact certification key; otherwise "estimated error bound". |
| Storage | Local first; JSONL benchmark export. |
| Network | Not required for heading after install. |

Pin every dependency version. No `+`, `latest.release`, or unpinned branches.

### 2.1 Form-factor scope

Portrait-lock is coherent on a phone and incoherent on a tablet. v1 certifies handheld phones only. Tablets and unfolded foldables return `UNSUPPORTED_DEVICE` for Precision Mode until they appear in the certification database under a posture-qualified key. Lifting this requires the full §31 orientation programme and a versioned reference-axis contract, not a config change.

### 2.2 Why the no-GMS path is first-class

Mainland China is a major target market in which Google Play services availability cannot be assumed; many Huawei/HMS and de-Googled deployments elsewhere likewise operate without GMS. `AND-RV` therefore has its own certification gates (§30.4), device list, telemetry, and uncertainty floor. Provider selection comes from measured runtime availability, never a build flavour, and **the quality label never survives a provider switch**.

### 2.3 Reuse hierarchy

1. Official fused provider (Google FOP/FLP, Apple CL/CM).
2. Official reference implementation (NOAA WMM coefficients, C source, error model, test vectors).
3. Official platform primitive (`SensorManager` transforms, monotonic clocks, crypto, lifecycle, storage).
4. Maintained pinned library, only where no platform primitive exists.
5. Small project-owned pure function — domain glue only (circular math, state reduction, thresholds, classification).
6. New low-level algorithm — prohibited without explicit authorization.

If a preferred tool is missing: emit an explicit capability state, disable the affected tier, and use a fallback only when that exact fallback has its own benchmark evidence. Never silently switch algorithms while keeping a quality label.

Use these rather than recreating them:

| Need | Tool |
|---|---|
| Android absolute orientation | Google FOP when available, else `TYPE_ROTATION_VECTOR` via `SensorManager`. Both production-capable only after separate certification. |
| Android position | Google FLP when available, else framework location for the no-GMS variant. |
| Android frame transforms | `SensorManager.getRotationMatrixFromVector`, `getRotationMatrix`, `getOrientation`, `remapCoordinateSystem`. |
| iOS heading / true north | Core Location `CLLocationManager` / `CLHeading`. |
| iOS attitude, gravity, rotation, calibrated field | Core Motion `CMMotionManager` / `CMDeviceMotion`. |
| Geomagnetic model | Vendored NOAA C sources + coefficients + error model, compiled for both platforms. `GeomagneticField` is an Android cross-check only. |
| Android lifecycle / concurrency | AndroidX lifecycle, Kotlin coroutines, `StateFlow`/`SharedFlow`. |
| iOS lifecycle / concurrency | Swift structured concurrency, actors, SwiftUI observation for the pinned toolchain. |
| Android persistence | Room for saved records; streaming file I/O for benchmark JSONL. |
| iOS persistence | SwiftData for saved records on the iOS 17+ target; streaming file I/O for JSONL. |
| Android tests / profiling | JUnit, AndroidX Test, Macrobenchmark, Power Profiler, system tracing. |
| iOS tests / profiling | XCTest, Instruments, Xcode performance tools, MetricKit. |
| Hashing | CryptoKit on iOS; standard Java/Android cryptography on Android. |
| JSON | `Codable`/`JSONEncoder` on iOS; one pinned Kotlin serializer on Android, both configured per §22.2. |

The project-owned code may **orchestrate, validate, gate, aggregate, classify, and measure** provider output. It may not claim to improve physical orientation by reimplementing the provider. No mobile API guarantees accurate absolute heading in arbitrary magnetic environments — reuse-first reduces implementation risk, it does not remove the need for ground-truth benchmarking, interference rejection, and honest uncertainty.

---

## 3. Definitions and conventions

| Term | Definition |
|---|---|
| Heading | Horizontal azimuth of the mode's reference axis, clockwise from the stated north reference. |
| Reference axis | `FLAT_TOP_EDGE`: portrait top edge (camera/earpiece end). `WALL_FLUSH_BACK`: outward screen normal. |
| Heading error | `measured - truth`, shortest signed circular difference, range `(-180°, 180°]`. |
| Absolute error | `abs(heading error)`, `[0°, 180°]`. |
| `instrumentBound95Deg` | Bound on where the *device axis* points. Excludes placement. Never shown as the measurement uncertainty. |
| `placementBound95Deg` | Bound on aligning that axis to the physical plane. A property of method + operator, not device. |
| `reportedBound95Deg` | `instrument + placement`, capped 180°. The number shown, graded, and classified on. |
| Precision Lock | Aggregated measurement passing all gates with `reportedBound95Deg <= usableBound95MaxDeg`. |
| Degraded result | Valid but bound exceeds the lock ceiling. Never a lock, never styled as precision. |
| Effective heading sample count | Distinct accepted **absolute-heading** observations entering the locked circular mean, counting each source timestamp once. Duplicates/interpolated frames do not increment. |
| Periodic support sample count | Distinct accepted periodic pose/stability observations in the stable window. It may equal heading count on periodic heading providers; it is separate on iOS flat, where Core Motion supplies support and `CLHeading` supplies event-driven heading anchors. |
| Stable window | Low motion + compact circular dispersion for the required duration. **Not** identical repeated digits. |
| Clean field | No significant disturbance detected by the validated detector. Evidence, not proof. |
| Acceptance rate | Accepted attempts / eligible attempts. (Called "coverage" in selective-prediction literature; this document never uses the bare word that way.) |
| Empirical coverage | Fraction of observations whose absolute error falls inside the reported bound. Always qualified by which bound. |
| Certification key | The exact tuple in §24 that a measurement context must match for calibrated claims. |
| Candidate result | Produced before the exact certification key demonstrated held-out coverage. Numerically useful; MUST NOT be presented as certified. |
| Fresh data | Source-timestamp age within its class limit — not merely recently delivered. |
| `*95Deg` suffix | Names the v1 **target two-sided coverage semantics** of a bound. It does not assert demonstrated 95% coverage while `boundCalibrationState == CANDIDATE`; only held-out §32 evidence can earn that claim. |
| Provider error term | A provider-reported angular-error quantity with its exact semantics/provenance recorded. It may be absent; absence is never encoded as zero uncertainty. |

**Numeric conventions.** Degrees as `Double`, normalized `[0°, 360°)`, exactly `360.0 → 0.0`. Trig calls take radians; conversions explicit at boundaries. Signed difference follows `atan2` but is **not** raw `atan2`: the range is `(-180°, 180°]`, so the antipode is `+180`, never `-180` — a sign that flips on floating-point noise makes bias statistics and target guidance nondeterministic. Raw `atan2(sin(a-b), cos(a-b))` does not satisfy this by itself. It returns `-180.0` whenever `sin(radians(a-b))` evaluates to a tiny negative rather than exactly zero, which is what happens for ordinary inputs such as `a=0, b=180` and `a=90, b=270`. Every signed-difference implementation MUST therefore map an exact `-180.0` result to `+180.0` before returning, and the property tests MUST include both antipodal orderings of at least two distinct pairs, because a test that only checks `deltaDeg(180, 0)` passes on a broken implementation. Canonical orientation basis is **REFERENCE_ENU**: `+Z` up, `+Y` toward the north reference named by `providerReferenceContract`, and `+X = +Y × +Z` (east relative to that reference). With contract `TRUE` this is geographic ENU; with `MAGNETIC` it is magnetic-east/magnetic-north/up. Never relabel a magnetic basis as geographic ENU merely because its axis order is east/north/up. Quaternion types MUST name component order **and transform direction**; never pass a bare 4-element array beyond a provider adapter.

---

## 4. Architecture

```text
Android: capability resolver picks exactly one          iOS, mode-specific
  A) Google FOP + Google FLP                            FLAT: CLHeading + CLLocation
  B) TYPE_ROTATION_VECTOR + framework LocationManager  WALL: CMDeviceMotion (.xTrueNorthZVertical)
                                                        CLHeading/CM remain cross-check sources
                    |                                        |
                    +--------------------+-------------------+
                                         v
                          PrecisionHeadingEngine (pure, deterministic)
     normalize | freshness | reference resolution | mode projection | pose
     stability | magnetic state | optional certified correction
     uncertainty composition | grade | target guidance | lock
                                         |
                    +--------------------+-------------------+
                    v                                        v
          FengShuiDirectionEngine                  Telemetry / Diagnostics / Store
                    |
                    v
              LuoPanRenderer

Raw/native sensor streams -> diagnostics, interference features,
benchmark comparison, and the certified no-GMS provider only
```

Rules: the Feng Shui engine consumes only a gated `HeadingMeasurement`, never sensors. Platform types (`DeviceOrientation`, `SensorEvent`, `CLHeading`, `CLLocation`, `CMDeviceMotion`) convert to canonical samples at the adapter boundary and never escape it. `HeadingEngine` is deterministic for a given ordered event stream + config; time enters as event timestamps, never a wall-clock call inside decision logic.

**Concurrency.** One session object owns all subscriptions; starting a session cancels and awaits the previous one. Callbacks become immutable events immediately; the engine consumes them on one serialized executor/actor. UI observes immutable snapshots. Telemetry writes off the UI thread preserving sequence numbers. No unbounded queues — record drops and reset aggregation when ordering/freshness guarantees are lost. Cancellation, backgrounding, permission change, provider failure, and rotation are all explicit engine events. Isolation MUST prevent a torn-down session's callbacks entering a later session.

### 4.1 Repository layout

```text
/
|- SPEC.md                          this document
|- README.md
|- docs/{BENCHMARK,RISKS,IMPLEMENTATION_NOTES,TESTING,PRIVACY}.md
|- config/{precision-profile-v1.json, feng-shui-rules-v1.json}
|- schemas/{precision-profile-v1, feng-shui-rules-v1,
|           telemetry-event-v1, session-manifest-v1}.schema.json
|- testdata/{angles, quaternions, wmm, fengshui, replay}/
|- third_party/noaa-wmm/{UPSTREAM.md, LICENSES/, sha256.txt,
|                        src/, coefficients/, error-model/}
|- android/{settings.gradle.kts, gradle/libs.versions.toml, app/,
|           heading-core/, heading-google/, heading-diagnostics/,
|           benchmark-mode/, fengshui-core/}
|- ios/{FengShuiCompass.xcodeproj, FengShuiCompass/, HeadingCore/,
|       HeadingApple/, HeadingDiagnostics/, BenchmarkMode/, FengShuiCore/}
|- analysis/{pyproject.toml, src/, tests/}
`- scripts/{validate-fixtures.sh, verify-artifacts.sh, generate-scorecard.sh}
```

Names may follow local convention; boundaries MUST NOT. `heading-core`/`HeadingCore` are pure with no UI or framework singleton. Provider modules wrap SDKs. Diagnostic modules never become the production estimator. Benchmark modules are internal-build only and depend on the same production core. `analysis/` computes reports from exported telemetry and **never** changes acceptance outcomes after collection — a metric it cannot compute from exported fields is a telemetry defect, not licence to recompute the decision.

---

## 5. Core types

```text
ProviderHeadingSample                 // raw scalar heading observation
  providerHeadingDeg
  providerErrorTermDeg?                 // absent when provider exposes no degree error
  providerErrorSource                   // exact semantics/provenance
  conservativeHeadingErrorDeg?          // Google FOP only when known
  providerReferenceContract             // the API's documented promise only
  sourceMonotonicNs, arrivalMonotonicNs
  screenOrientation, providerId, providerSampleId?

ProviderAttitudeSample                 // adapter-normalized canonical attitude
  attitudeQuaternionDeviceToReferenceEnuXYZW
                                           // ALWAYS device -> project REFERENCE_ENU
  providerReferenceContract
  sourceMonotonicNs, arrivalMonotonicNs
  nativeAttitudeFrame, screenOrientation, providerId, providerSampleId?
  // raw/native quaternion + native axis convention are retained in telemetry for replay;
  // project core MUST NOT receive a provider-native quaternion directly.

LocationSample
  latitudeDeg, longitudeDeg, altitude: AltitudeSample
  horizontalAccuracyM
  authorizationAccuracy: PRECISE_FULL | APPROXIMATE_REDUCED
  sourceTimestamp, mappedMonotonicTimestamp
  locationProviderId, locationProviderRuntimeIdentity
  isMockOrSimulated: Boolean?         // benchmark diagnostics

PlacementProfileMetadata
  placementMethod
  placementProfileId, placementProfileHash
  bound95Deg
  evidenceId?                         // required for measured jig/operator profiles

AltitudeSample
  valueM, reference: WGS84_ELLIPSOID | MSL_ORTHOMETRIC | UNKNOWN
  verticalAccuracyM?

ReferenceResolutionResult             // one per active-mode stable window
  measurementMode, referenceAxis
  resolvedReference, referenceResolutionMethod
  referenceHypothesisResidualTrueDeg?     // rTrue; null when not hypothesis-tested
  referenceHypothesisResidualMagneticDeg? // rMag; null when not hypothesis-tested
  declinationDeg
  correctionDeg                       // exactly 0.0 or +declinationDeg
  referenceAmbiguityBound95Deg        // 0.0 unless TRUE_WITH_AMBIGUITY_BOUND
  geomagneticModelId
  sourceWindowStartMonotonicNs, sourceWindowEndMonotonicNs

MagneticDiagnosticSample
  calibratedMicroTeslaXYZ?, uncalibratedMicroTeslaXYZ?, biasMicroTeslaXYZ?
  calibrationState, deviceFrame, saturated, sourceMonotonicNs

MotionDiagnosticSample
  gravityGXYZ, userAccelerationGXYZ, rotationRateDegPerSecXYZ
  attitudeQuaternionDeviceToReferenceEnuXYZW, sourceMonotonicNs

SensorHealthSnapshot
  providerAvailable, requiredSensorsAvailable, osCalibrationState
  referenceMagneticPrecheckState, magneticState, providerErrorState
  transformAgreementDeg               // same provider, two extraction routes -> code fault
  pipelineAgreementDeg                // independent estimators -> environment signal
  locationFreshnessState, resolvedReference, boundCalibrationState
  chargingState, thermalState, trustAction, limitingReasons[]

CalibrationRequest  { requestId, entryReason, requestedAtMonotonicNs }
CalibrationResult   { requestId, outcome, beforeAssessment, afterAssessment,
                      completedAtMonotonicNs, limitingReasons[] }

TargetHeadingRequest   { requestId, targetHeadingDeg, reference }
TargetGuidanceSnapshot { targetHeadingDeg, liveHeadingDeg, signedDeltaDeg,
                         absoluteDeltaDeg, nearTarget, targetCentered,
                         referenceStatus, guidanceIsProvisional }

DeviationCorrectionProfileMetadata
  profileId, profileHash, scope: UNIT | MODEL_CLASS, structureClass
  correctionMethodId, measurementMode, placementMethod
  providerId, coveredProviderRuntimeIdentities[], coveredOsBuildIdentities[]
  geomagneticModelId, geomagneticCoefficientHash, precisionConfigHash
  heldOutResidualBound95Deg, trainingEvidenceId, heldOutEvidenceId
```

`GroundTruthSample` is a **benchmark-module** type, never in the production core:

```text
GroundTruthSample
  trueHeadingDeg, pitchDeg, rollDeg
  expandedUncertaintyDeg, coverageFactor
  referenceTier: TIER_0A | TIER_0B | TIER_0C | TIER_1 | TIER_2
  sourceClock, sourceTimestamp
```

### 5.1 `HeadingMeasurement` — canonical output

```text
mode/placement:   measurementMode, placementMethod,
                  placementProfileId, placementProfileHash, placementBound95Deg
provider:         providerId, providerRuntimeIdentity, providerHeadingDeg?,
                  providerErrorTermDeg?, providerErrorSource,
                  conservativeHeadingErrorDeg?, providerReferenceContract
reference:        resolvedReference, referenceResolutionMethod,
                  referenceHypothesisResidualTrueDeg,
                  referenceHypothesisResidualMagneticDeg,
                  referenceAmbiguityBound95Deg
heading:          magneticHeadingDeg, magneticHeadingSource,
                  uncorrectedTrueHeadingDeg, deviationCorrectionDeg,
                  deviationCorrectionState, deviationCorrectionProfileId,
                  deviationCorrectionProfileHash, trueHeadingDeg, declinationDeg
uncertainty:      instrumentBound95Deg, reportedBound95Deg,
                  uncertaintyCoverageTarget, uncertaintyCoverageEvidenceState,
                  boundCalibrationState, gradeLimitedBy
position:         latitudeDeg, longitudeDeg, altitudeM, altitudeReference,
                  horizontalAccuracyM, verticalAccuracyM, declinationEnvelopeDeg,
                  locationProviderId, locationProviderRuntimeIdentity
pose:             attitudeQuaternionDeviceToReferenceEnuXYZW, nativeAttitudeFrame,
                  pitchDeg, rollDeg, referenceAxis,
                  deviceOrientation, displayRotation
field:            calibratedMagneticFieldMicroTesla,
                  uncalibratedMagneticFieldMicroTesla, magneticBiasMicroTesla,
                  expectedWmmField, horizontalIntensityNanoTesla
health:           osCalibrationState, magneticState, calibrationEntryReason,
                  calibrationValidationOutcome, sensorHealthSnapshot,
                  transformAgreementDeg, pipelineAgreementDeg,
                  spaceWeatherState, chargingState, thermalState
timing:           sourceTimestamps, arrivalTimestamps, measurementTimestamp,
                  agesMs, effectiveHeadingSampleCount, periodicSupportSampleCount,
                  aggregationDurationMs, circularResultantLength
outcome:          measurementState, trustAction, provisionalQualityGrade,
                  displayQualityGrade?  (absent while CANDIDATE), rejectionReasons[]
classification:   fengShuiRuleSetVersion, fengShuiRuleSetHash,
                  fengShuiReferenceSelection, primaryFengShuiSector,
                  possibleFengShuiSectors[], boundaryStraddled,
                  signedOffsetFromSectorBoundaryDeg
provenance:       engineVersion, engineDecisionLogicHash, configVersion, configHash,
                  hardwareRuntimeIdentity, sensorRuntimeIdentity, osBuildIdentity
```

The type MUST distinguish missing / invalid / not-supported / stale. `0`, `-1`, `NaN`, `null` are not interchangeable.

**No opaque composite scores.** Record features (`circularResultantLength`, residual quantiles, angular speed, `relativeMagnitudeResidual`, `inclinationResidualDeg`, `stationaryFieldMadMicroTesla`, `pipelineAgreementDeg`); reports compute summaries. A future composite score arrives with a formula, range, direction, unit test, and a benchmark showing what it adds — an undefined `[0,1]` number in a canonical model is how an undefined quantity ends up in a gate.

---

## 6. Enums

Wire values are stable `UPPER_SNAKE_CASE` strings, everywhere, including examples and fixtures. Adding a case is backward compatible; renaming or reusing a stored value is a schema migration.

```text
ProviderId              GOOGLE_FOP | APPLE_CLHEADING | APPLE_CORE_MOTION_TRUE_NORTH
                        | ANDROID_ROTATION_VECTOR | ANDROID_HEADING (diag)
                        | ANDROID_ACCEL_MAG (diag) | REPLAY

LocationProviderId      GOOGLE_FLP | ANDROID_FRAMEWORK_LOCATION
                        | APPLE_CORE_LOCATION | REPLAY

ProviderErrorSource     GOOGLE_CONSERVATIVE | GOOGLE_ORDINARY
                        | APPLE_HEADING_ACCURACY
                        | ANDROID_ROTATION_VECTOR_HEADING_ACCURACY_95
                        | ANDROID_HEADING_ACCURACY_68
                        | NONE

ProviderReferenceContract
                        TRUE | MAGNETIC
                        | TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC | UNKNOWN

GeomagneticModelId      WMM2025 | WMMHR2025
MeasurementMode         FLAT_TOP_EDGE | WALL_FLUSH_BACK
TargetReference         TRUE | MAGNETIC
PlacementMethod         FREEHAND | WALL_FLUSH_FREEHAND
                        | NONMAGNETIC_ALIGNMENT_JIG | SURVEY_FIXTURE

ResolvedReference       TRUE_VERIFIED | TRUE_CORRECTED_FROM_MAGNETIC
                        | TRUE_WITH_AMBIGUITY_BOUND | MAGNETIC | UNVERIFIED

ReferenceResolutionMethod
                        PROVIDER_CONTRACT_EXPLICIT      // iOS trueHeading validity
                        | ATTITUDE_FRAME_EXPLICIT       // CM frame confirmed in use
                        | FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK
                        | APP_APPLIED_DECLINATION       // AND-RV, known by construction
                        | NOT_RESOLVED

MeasurementState        IDLE | ACQUIRING_LOCATION | ACQUIRING_ORIENTATION
                        | PROVIDER_INITIALIZING | CALIBRATION_CHECK
                        | MAGNETIC_FIELD_CHECK | TARGET_SEEKING | LEVEL_AND_HOLD
                        | STABILIZING | PRECISION_LOCKED | DEGRADED | INVALID | TIMED_OUT

ReferenceMagneticPrecheckState
                        CLEAN_FOR_REFERENCE | NOT_CLEAN_FOR_REFERENCE | UNKNOWN
MagneticState           CLEAN | SUSPECT | DISTURBED | INVALID | UNKNOWN
SpaceWeatherState       QUIET | ADVISORY | PROFESSIONAL_SUPPRESSED
                        | EXTREME_WMM_UNUSABLE | UNKNOWN
BoundCalibrationState   CANDIDATE | CALIBRATED
UncertaintyCoverageEvidenceState
                        TARGET_ONLY | EMPIRICALLY_CALIBRATED | UNDEFINED
CalibrationKind         SENSOR_CALIBRATION | DEVIATION_CHARACTERIZATION
                        | UNCERTAINTY_CALIBRATION
CalibrationEntryReason  AUTOMATIC_TRIGGER | USER_REQUESTED | BENCHMARK_PROTOCOL
CalibrationValidationOutcome
                        IMPROVED | ACCEPTABLE_NO_CHANGE | STILL_POOR
                        | ENVIRONMENT_DISTURBED | INVALID_OR_INCONCLUSIVE
DeviationCorrectionState NONE | EXPERIMENTAL | CERTIFIED_PROFILE
DeviationStructureClass UNIT_STABLE | MODEL_CLASS_STABLE | CALIBRATION_STATE_DEPENDENT
                        | SITE_DEPENDENT | TRANSIENT | NONREPEATABLE
ChargingState           NOT_CHARGING | WIRED | WIRELESS | UNKNOWN

TrustAction             READY_CALIBRATED | READY_CANDIDATE | SHOW_DEGRADED_RESULT
                        | HOLD_STEADY
                        | ROTATE_TO_INITIALIZE | CALIBRATE
                        | MOVE_AWAY_FROM_INTERFERENCE
                        | REACQUIRE_REFERENCE_OR_LOCATION | UNSUPPORTED_OR_REJECTED

QualityGrade            PROFESSIONAL | HIGH | USABLE | LOW_CONFIDENCE | INVALID

GradeLimitingFactor     NONE | PLACEMENT_UNCERTAINTY | PROVIDER_ERROR
                        | SAMPLE_DISPERSION | DEVICE_FLOOR | DECLINATION_MODEL
                        | LOCATION_TIME_SENSITIVITY | REFERENCE_AMBIGUITY
                        | INTERFERENCE_PENALTY | DEVIATION_PROFILE_RESIDUAL
                        | CERTIFICATION_CEILING | SPACE_WEATHER | CHARGING_STATE

RejectionReason         HEADING_UNAVAILABLE | HEADING_ERROR_INVALID
                        | PROVIDER_NOT_INITIALIZED | ORIENTATION_STALE
                        | LOCATION_PERMISSION_DENIED | LOCATION_STALE
                        | LOCATION_UNCERTAINTY_EXCEEDS_DECLINATION_BUDGET
                        | LOCATION_JUMP_REQUIRES_FRESH_FIX
                        | GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE | WEAK_HORIZONTAL_FIELD
                        | TRUE_REFERENCE_UNVERIFIED | MAGNETIC_CALIBRATION_INVALID
                        | MAGNETIC_FIELD_SUSPECT | MAGNETIC_FIELD_DISTURBED
                        | MAGNETIC_FIELD_UNKNOWN | TRANSFORM_DISAGREEMENT
                        | PIPELINE_DISAGREEMENT | CIRCULAR_MEAN_UNDEFINED
                        | DEVICE_MOVING | DEVICE_NOT_LEVEL
                        | UNSUPPORTED_SCREEN_ORIENTATION
                        | ORIENTATION_CHANGED_DURING_WINDOW | SENSOR_DISCONTINUITY
                        | APP_BACKGROUNDED | THERMAL_RESTRICTION
                        | WIRELESS_CHARGING_ACTIVE | PROVIDER_FAILURE
                        | SPACE_WEATHER_EXTREME
                        | UNSUPPORTED_DEVICE | ACQUISITION_TIMEOUT
                        | TARGET_REFERENCE_UNAVAILABLE | TARGET_NOT_STABLE
                        | DEVIATION_PROFILE_NOT_CERTIFIED | REPEAT_MEASUREMENT_INCONSISTENT
```

There is exactly **one** measurement-state vocabulary, `MeasurementState`. Any coarser UI vocabulary is derived in the view layer through a total tested mapping and never persisted as an independent fact.

---

## 7. Interfaces

```text
HeadingProvider          providerId; start(request, onHeading, onAttitude, onError); stop()
                         // may emit scalar heading, attitude, or both; never invent a missing form
LocationProvider         requestFreshLocation(request); startUpdates(...); stopUpdates()
DiagnosticSensorProvider start(onMagnetic, onMotion, onDiscontinuity); stop()

GeomagneticModel
  modelId; validityStartDecimalYear; validityEndDecimalYear
  evaluate(latDeg, lonDeg, altitude: AltitudeSample, utcInstant)
    -> declination, inclination, X/Y/Z/H/F, gridVariation?, metadata,
       uncertainty: GeomagneticModelUncertainty

GeomagneticModelUncertainty
  declinationSigma1Deg
  sourceConfidenceLevel = ONE_STANDARD_DEVIATION
  sourceModelId, errorModelId, errorModelHash, sourceDocumentReference

HeadingEngine            handle(event) -> [EngineEffect]; currentSnapshot(); reset(reason)
DeviationCorrectionProvider  lookup(liveContext) -> profile | NONE; apply(profile, deg)
CertificationDatabase    lookup(key: CertificationKey) -> CertificationRecord?
TelemetrySink            append(event); flush(); close()
FengShuiDirectionEngine  classify(canonicalTrueHeadingDeg, reportedBound95Deg, ruleSet)
```

Suggested names, to keep parallel implementations aligned:

| Role | Android | iOS |
|---|---|---|
| Production heading | `GoogleFusedHeadingProvider` | `AppleHeadingProvider` |
| No-GMS heading | `AndroidRotationVectorHeadingProvider` | — |
| Production location | `GoogleFusedLocationProvider` | `AppleLocationProvider` |
| No-GMS location | `AndroidFrameworkLocationProvider` | — |
| Diagnostics | `AndroidDiagnosticSensorProvider` | `AppleDiagnosticSensorProvider` |
| Engine | `PrecisionHeadingEngine` | `PrecisionHeadingEngine` |
| Session owner | `MeasurementSessionCoordinator` | `MeasurementSessionCoordinator` (actor) |
| WMM wrapper | `NoaaWmmGeomagneticModel` | `NoaaWmmGeomagneticModel` |
| Telemetry | `JsonlTelemetrySink` | `JsonlTelemetrySink` (actor) |
| Replay | `ReplayHeadingProvider` | `ReplayHeadingProvider` |

---

## 8. Configuration

`config/precision-profile-v1.json`, schema-validated at build and test time, bundled read-only, version + SHA-256 copied into every session manifest. Remote configuration is prohibited in certification builds.

```json
{
  "schemaVersion": "1.0.0",
  "configVersion": "precision-v1-candidate-1",

  "orientationMaxAgeMs": 100,
  "orientationInvalidAfterMs": 500,
  "freshLocationAtStartMaxAgeMs": 60000,
  "locationAtLockMaxAgeMs": 300000,
  "usableLocationMaxAgeMs": 1800000,
  "locationJumpRequiresFreshFixKm": 50.0,
  "declinationEnvelopeProfessionalMaxDeg": 0.10,
  "declinationEnvelopeUsableMaxDeg": 0.25,

  "stableWindowMinMs": 2000,
  "acquisitionTimeoutMs": 10000,
  "periodicOrientationRequestedHz": 50.0,
  "minPeriodicSupportSamples": 50,
  "clHeadingMinSamplesPerStableWindow": 1,
  "minCircularResultantLength": 0.995,
  "angularSpeedP95MaxDegPerSec": 3.0,
  "linearAccelerationP95MaxG": 0.05,
  "circularResidualP95MaxDeg": 3.0,

  "flatModePitchAbsMaxDeg": 5.0,
  "flatModeRollAbsMaxDeg": 5.0,
  "flatFreehandPlacementBound95Deg": 3.0,
  "wallNormalElevationAbsMaxDeg": 5.0,
  "wallTopAxisFromVerticalMaxDeg": 5.0,
  "wallFreehandPlacementBound95Deg": 5.0,

  "targetNearZoneDeg": 5.0,
  "targetCenteringToleranceDeg": 1.0,

  "providerCrossCheckMaxDeg": 5.0,
  "referenceSeparationMarginDeg": 2.0,
  "smallDeclinationAmbiguityMaxDeg": 2.0,
  "transformAgreementMaxDeg": 2.0,

  "magneticMagnitudeResidualSuspectFraction": 0.20,
  "magneticMagnitudeResidualDisturbedFraction": 0.50,
  "inclinationResidualSuspectDeg": 5.0,
  "inclinationResidualDisturbedDeg": 12.0,
  "stationaryFieldMadSuspectMicroTesla": 1.5,
  "stationaryFieldMadDisturbedMicroTesla": 4.0,
  "pipelineDisagreementSuspectDeg": 5.0,
  "pipelineDisagreementDisturbedDeg": 10.0,
  "suspectInterferenceBound95Deg": 3.0,
  "recoveryCleanWindowMs": 2000,

  "minHorizontalIntensityNanoTesla": 6000.0,

  "unknownDeviceFloor95Deg": 4.0,
  "professionalBound95MaxDeg": 2.0,
  "highBound95MaxDeg": 3.0,
  "usableBound95MaxDeg": 5.0,
  "lowConfidenceBound95MaxDeg": 10.0,

  "spaceWeatherAdvisoryKpMin": 5.0,
  "spaceWeatherProfessionalSuppressKpMin": 7.0,
  "spaceWeatherRejectKpMin": 9.0,
  "spaceWeatherCacheMaxAgeMs": 21600000,

  "thermalRestrictionBlocksLock": true,
  "wirelessChargingBlocksGradeAboveUsable": true,

  "precisionScreenOrientation": "PORTRAIT",
  "requireBoundaryStraddleReporting": true,
  "geomagneticModelId": "WMM2025",
  "canonicalAltitudeReference": "WGS84_ELLIPSOID",
  "declinationSigmaToBound95Factor": 1.96
}
```

### 8.1 Enforced invariants

Schema sets `"additionalProperties": false`. A build-time test MUST assert each of these; every one prevents a specific silent failure.

| Invariant | Prevents |
|---|---|
| No key matching `/calibrationState/i` exists anywhere in the profile | `boundCalibrationState` is derived from a certification lookup (§19). One editable value that turns every device Professional is the shortcut an agent under pressure takes; a schema that rejects it is stronger than prose. |
| `referenceSeparationMarginDeg <= smallDeclinationAmbiguityMaxDeg` | Since `rMag - rTrue <= abs(d)`, a margin above the ambiguity allowance creates a declination dead band that always resolves `UNVERIFIED` with no visible cause (§11). |
| `professionalBound95MaxDeg < highBound95MaxDeg < usableBound95MaxDeg < lowConfidenceBound95MaxDeg` | Grade function must be total and ordered. |
| `professionalBound95MaxDeg < flatFreehandPlacementBound95Deg` | Encodes in config that freehand cannot reach the top grade (§20). A future edit breaking this trips the intended alarm. |
| `declinationEnvelopeProfessionalMaxDeg <= declinationEnvelopeUsableMaxDeg` | Ordered gates. |
| suspect < disturbed for magnitude, inclination, stationary-MAD, pipeline pairs | A suspect threshold above disturbed makes `SUSPECT` unreachable. |
| `stableWindowMinMs * (periodicOrientationRequestedHz / 2) / 1000 >= minPeriodicSupportSamples` | Periodic support streams request 50 Hz; the candidate gate tolerates a 50% callback shortfall. This invariant does **not** apply to event-driven `CLHeading`; flat iOS has a separate in-window heading-anchor count. |
| `orientationMaxAgeMs < orientationInvalidAfterMs` | Drop and invalidate are different thresholds. |
| `freshLocationAtStartMaxAgeMs <= locationAtLockMaxAgeMs <= usableLocationMaxAgeMs` | Ordered freshness tiers. |
| `spaceWeatherAdvisoryKpMin <= spaceWeatherProfessionalSuppressKpMin < spaceWeatherRejectKpMin` | Ordered advisory/suppression/refusal tiers. |

`minHorizontalIntensityNanoTesla` is physics, not tuning: heading sensitivity to a transverse magnetic-field perturbation grows approximately as `1/H`, so at high magnetic latitude the same perturbation costs more angle. At `6000 nT`, a `50 nT` transverse perturbation is about `atan(50/6000) ≈ 0.48°`.

The candidate value is **not** arbitrary. NOAA defines a *Blackout Zone* where `H < 2000 nT`, in which WMM declination values are not accurate and compasses are unreliable, and a surrounding *Caution Zone* where `2000 nT <= H < 6000 nT`, in which compasses should be used with caution. `6000 nT` is the upper boundary of NOAA's own caution region, so the gate inherits the model authority's judgement rather than an invented threshold. The two zones SHOULD be distinguished in a later version — `H < 2000 nT` is an outright refusal for any north-referenced result, while the caution band is a candidate for downgrade-with-warning once benchmark evidence exists — but v1 conservatively refuses across the whole band via `WEAK_HORIZONTAL_FIELD`.

The gate MUST be evaluated from the **model's predicted `H`** at the fix, never the measured field, so that a disturbance cannot suppress its own gate.

#### 8.1.1 Grade reachability — a required build-time analysis

Because `reportedBound95Deg = instrumentBound95Deg + placementBound95Deg` and the lock ceiling is `usableBound95MaxDeg`, each placement method has a fixed **instrument budget** of `usableBound95MaxDeg - placementBound95Deg`. Any single uncertainty term larger than that budget makes a Precision Lock arithmetically impossible for that combination, no matter how good the sensor is. The candidate constants have three such consequences, and none of them is visible from reading the gate table:

| Combination | Instrument budget | Term that exceeds it | Consequence |
|---|---:|---|---|
| Flat freehand, uncertified device | `2.0°` | `unknownDeviceFloor95Deg = 4.0°` | No Precision Lock is possible on any uncertified device, in the ordinary user gesture. |
| Wall freehand, any device | `0.0°` | any positive instrument bound | `WALL_FLUSH_FREEHAND` can never lock. |
| Either freehand geometry, `SUSPECT` field | `2.0°` / `0.0°` | `suspectInterferenceBound95Deg = 3.0°` | `SUSPECT` never merely "caps the grade" freehand — it prevents locking outright. |

These are candidate-constant consequences, not permanent design intent, but they MUST be treated as facts by the implementation and by the benchmark plan:

- A build-time test MUST compute, for every `(PlacementMethod, certification state, MagneticState)` combination the product claims to support, whether the claimed maximum grade is arithmetically reachable, and MUST fail when the spec text claims a grade the constants forbid. This is the specific defect class that survives coverage review: two internally consistent sections, contradicted only by arithmetic.
- Code MUST NOT bypass the total-bound gate, special-case a term, or floor a bound to make a combination lock.
- Constants may change only from §29.5 placement evidence and §30 device-floor evidence, never to make a demo work.

**The certification bootstrap.** The first consequence above creates a real ordering problem that the delivery plan MUST respect. `AND-G1` and `IOS-A1` are defined as provider plus lock logic, so on uncertified hardware their freehand acceptance rate is zero; §30.1 requires comparison at matched acceptance rate and §30.2 requires `>= 95%` clean acceptance, and neither is reachable while every device is uncertified. Certification is what would fix the floor, and the benchmark is what produces certification.

The resolution is that **`deviceFloor95Deg` is an output of the benchmark, not an input to it**. Phase 5 therefore:

- sweeps the device floor as an explicit parameter alongside the provider-error threshold, rather than holding it at `unknownDeviceFloor95Deg`;
- reports acceptance and risk as functions of that floor, so the shipped floor is chosen from evidence;
- evaluates lock-gate behaviour primarily with `NONMAGNETIC_ALIGNMENT_JIG` placement, whose smaller placement term leaves a usable instrument budget, and reports freehand separately as the product-experience population (§30.2 already requires the freehand table to be published but not gated).

Until a device class is certified, the honest product behaviour is that freehand measurements return `DEGRADED` results with an explicit bound and no lock. That MUST be stated in release notes rather than engineered around.

All magnetic and space-weather values are candidate gates. Record all detector features; mark `SUSPECT` at the suspect threshold, reject `DISTURBED`. Do not add an absolute microtesla threshold on total magnitude without device/site evidence. Kp is a coarse planetary advisory and MUST NOT be added numerically to heading or declination.

---

## 9. Deterministic utilities

Pure functions, tested identically on both platforms:

```text
normalize360(deg)                      -> [0,360); exactly 360 -> 0
shortestSignedDifferenceDeg(a, b)      -> (-180,180]; atan2 convention, and
                                          MUST map an exact -180.0 to +180.0
                                          internally (§3). This is the single normative
                                          contract in the spec. Each executable runtime
                                          has exactly one allowlisted implementation;
                                          all other code calls that local shared utility.
shortestTargetDeltaDeg(cur, target)    = shortestSignedDifferenceDeg(target, cur)
                                          // (-180,180]; positive = clockwise
absoluteCircularDifferenceDeg(a, b)    = abs(shortestSignedDifferenceDeg(a, b))
                                          // [0,180]
circularMeanDeg(samples)               -> mean | UNDEFINED
circularResultantLength(samples)       -> [0,1]
circularResidualQuantileDeg(samples, q)
quantile(values, q) / median(values)   -> pinned estimator, §9.1
finiteAngle validation
quaternion normalization + explicit provider-order conversion
deviceVectorToReferenceEnu(attitude, deviceVector)
modeReferenceVectorHeadingDeg(attitude, mode)
wmmDecimalYear(utcInstant)
boundFromSigma(sigma1Deg)              -> declinationSigmaToBound95Factor * sigma1Deg
referenceResolution(window)            -> ReferenceResolutionResult  §11
magneticStateClassification(features)  -> MagneticState  §16
qualityGrade(reportedBound95Deg, gates)
targetGuidance(live, target, config)
trustAction(snapshot)
deviationCorrectionLookupAndApply(...) -> correctionDeg, default NONE
fengShuiSector(headingDeg, ruleSet)                    // half-open boundaries
fengShuiStraddleSet(headingDeg, boundDeg, ruleSet)
```

`normalize360` MUST be `((x % 360) + 360) % 360` with a finite check — language remainder differs for negatives. Test `-360`, `-0.0`, `359.9999999`, `360.0`.

### 9.1 Pinned quantile and median

Common library estimators disagree by a full sample position at these window sizes, so "P95" must be defined. Sorted ascending `x[0..n-1]`, **nearest-rank**:

```text
quantile(x, q) = x[ min(n-1, max(0, ceil(q*n) - 1)) ]
median(x)      = n odd  -> x[(n-1)/2]
                 n even -> (x[n/2 - 1] + x[n/2]) / 2
```

Circular residual quantiles apply the linear estimator to absolute residuals about the accepted circular mean. `quantile` and `median` on empty input MUST return a typed `UNDEFINED`/validation failure and MUST NOT index element zero; nonfinite members are rejected before sorting. Both platforms MUST be bit-identical on `testdata/angles/`, and `analysis/` MUST use the same definition so a device-computed P95 and a report-computed P95 of the same window agree exactly.

Property tests MUST cover random angles plus: negative multiples of 360, `0`, `360`, `359.999...`, antipodal values, empty window, one sample, two equal opposite samples, nonfinite inputs, very large finite inputs.

---

## 10. Geomagnetic model

```text
trueHeading = normalize360(magneticHeading + declination)   // declination east-positive
```

The sign convention MUST be validated against official model test vectors **and** real sites with positive and negative declination. Never infer the convention from one location.

Vendor the exact NOAA package under `third_party/noaa-wmm/`: C source, coefficients, error model, licences, URL/date, SHA-256 for each. Compile the **same C sources** on both platforms (Swift C interop; Android NDK/CMake + minimal JNI). Do not write separate Swift and Kotlin ports of the spherical-harmonic core. Wrappers still need unit, conversion, datum, date, and memory-safety tests.

The wrapper MUST:

- Store model name, coefficient hash, error-model hash, epoch, validity interval.
- Pass NOAA official vectors in CI for **every** vendored coefficient set.
- Take lat/lon in degrees, altitude in the model's datum, UTC converted to decimal year.
- Never accept a bare `altitudeM` — altitude enters as `AltitudeSample` and the wrapper converts or refuses. `UNKNOWN` downgrades quality; it is not a synonym for either datum.
- Reject dates outside validity with `GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE` and surface an explicit "model expired, update the app" state. An app installed near epoch end **will** outlive its coefficients on some devices.
- Handle longitude wrap, negative altitude, leap years, antimeridian.
- Return grid variation above `|lat| = 55°`, and treat weak horizontal field / pole proximity as high-risk, gated by `minHorizontalIntensityNanoTesla`.
- Not claim WMM predicts building- or accessory-induced fields.

WMM2025's v1 epoch interval is `2025.0 <= decimalYear < 2030.0` (the 2025 model expires at the end of 2029). Android `GeomagneticField` MAY be a platform cross-check, never the cross-platform source.

**Space weather.** Optionally fetch NOAA SWPC planetary K-index as a cached, nonblocking advisory. Heading MUST work offline. Fresh `Kp >= spaceWeatherAdvisoryKpMin` → `ADVISORY`; fresh `Kp >= spaceWeatherProfessionalSuppressKpMin` → `PROFESSIONAL_SUPPRESSED`; fresh `Kp >= spaceWeatherRejectKpMin` → `EXTREME_WMM_UNUSABLE`. Offline, parse failure, or age beyond `spaceWeatherCacheMaxAgeMs` → `UNKNOWN`, never `QUIET` and never zero.

The candidate thresholds map onto NOAA's published storm scale: `Kp = 5` is G1 (minor), `Kp = 7` is G3 (strong), and `Kp = 9` is G5 (extreme). NOAA's statement is specifically that **WMM should not be used during a G5 event**. The v1 app adopts the stricter project policy that a **fresh observed** G5 state refuses every Precision Lock, because every v1 precision path depends on WMM somewhere in its safety contract — magnetic→true conversion and/or expected-field/reference validation. Emit `SPACE_WEATHER_EXTREME`. This refusal is **not** evidence that Apple/Google's internal true-north conversion is itself invalid; it is an app-level rule caused by the app's WMM dependency. `UNKNOWN` never triggers the refusal, so loss of network alone does not block heading. Consequently this is a conditional observed-state protection, not a guarantee that an offline device can detect an ongoing G5 storm; telemetry/certification reports MUST stratify `SpaceWeatherState` and MUST NOT claim G5 protection for `UNKNOWN` periods.

### 10.1 WMM2025 vs WMMHR2025

WMM2025 resolves ~`3300 km`; WMMHR2025 extends to much higher degree, resolving ~`300 km` of long-wavelength crustal field. Both are vendored with their own hashes, error models, and official vectors. Model choice is a **configuration factor** (`geomagneticModelId`) crossed with the §26 variants, not a variant ID — it is orthogonal to the provider and adding it to the variant list would double the scorecard for no information.

The factor has two effect paths, and conflating them makes the benchmark look for the effect in the wrong place:

- **Direct heading effect** — only where the app owns the magnetic→true conversion: `AND-RV`, and `AND-G1` on the `TRUE_CORRECTED_FROM_MAGNETIC` branch. No direct effect on `IOS-A0/A1` in either mode (Apple's true-reference conversion is provider-owned) or on the `TRUE_VERIFIED` branch (Google already applied declination).
- **Indirect gating effect** — the model supplies expected magnitude and inclination to the §16 detector, so it shifts accept/reject decisions on **every** candidate using field residuals, including `IOS-A1` and `AND-G1/TRUE_VERIFIED`, which have no direct effect at all. A model change can alter which measurements iOS accepts while changing no iOS bearing.

Two couplings:

1. **The interference detector depends on the model.** Changing it silently re-tunes the `SUSPECT`/`DISTURBED` gates. Magnetic thresholds are versioned **jointly** with `geomagneticModelId`; a model change invalidates threshold calibration and requires re-running §30.3.
2. **The declination uncertainty term changes.** `declinationModelBound95Deg` MUST be derived from the *new* model's published one-sigma through `boundFromSigma`, never carried over. A higher-resolution model does not license a smaller bound until its bound is sourced and its coverage verified.

The model can also **hurt** reference disambiguation (§11), which compares the provider heading against `m + d`: if the provider's internal declination differs from the app's `d`, a more accurate `d` widens `rTrue`. Evaluate the model against reference-resolution correctness, not only declination accuracy.

The high-degree set is larger and slower — irrelevant per-session, but it MUST NOT be evaluated inside the sample path, and app-size impact belongs in the decision record. Adopt WMMHR2025 as default only on held-out evidence of improved accepted error or reference correctness where the app owns the conversion.

### 10.2 Altitude datum

Android `Location.getAltitude()` is WGS84 ellipsoidal. Apple `CLLocation.altitude` is approximately MSL, with `ellipsoidalAltitude` separate. Geoid separation exceeds `100 m` in places, so passing raw platform values into one model produces genuine cross-platform divergence.

Canonical input is **WGS84 ellipsoidal height** — one unambiguous shared representation matching the geodetic reference the model is built on. On iOS use `ellipsoidalAltitude` and mark `WGS84_ELLIPSOID`. Where only orthometric is available, mark `MSL_ORTHOMETRIC` and either convert with a documented geoid model or accept the uncertainty explicitly.

Declination varies negligibly with a `100 m` datum error. This rule exists because it is a **silent systematic cross-platform divergence in a shared numerical core** — the class of defect that survives unit tests and surfaces as an unexplained parity failure. Fix it by typing, not by measuring its effect. Parity fixtures MUST include one case per datum plus `UNKNOWN`, and wrapper tests MUST prove an orthometric input is converted or refused, never silently treated as ellipsoidal.

### 10.3 Where the declination sigma comes from

`declinationSigma1Deg` is **not** an output of evaluating the spherical-harmonic coefficients. It comes from NOAA's separately published error model for that exact coefficient set. An implementation that derives a sigma from the coefficients, or substitutes a remembered global constant, has invented the quantity.

- Vendor the error-model artifact/code/data per coefficient set under `third_party/noaa-wmm/error-model/`, hash it, and record `errorModelId` and `errorModelHash` in every measurement using a declination term.
- For the pinned 2025 artifacts, the NOAA one-sigma declination models are location-dependent through horizontal intensity `H`: WMM2025 `sqrt(0.26^2 + (5417/H)^2)°`; WMMHR2025 `sqrt(0.25^2 + (5205/H)^2)°`, with `H` in nT. NOAA publishes the WMM2025 figure alongside `I = 0.20°`, `H = 133 nT`, `X = 137 nT`, `Y = 89 nT`, `Z = 141 nT`, and `F = 138 nT`, all interpretable as one standard deviation covering commission **and** omission error, which is why §19.2's sigma-to-bound conversion is required and why the omission term MUST NOT be double-counted against the §16 local-anomaly detector. Treat these as **test oracles for the pinned 2025 artifacts**, not literals to carry into a future epoch.
- If a future official artifact changes representation, follow the pinned artifact and update tests/provenance; do not preserve the 2025 formula merely for compatibility.
- The published error model includes model/omission effects at its stated scope; it does not prove the present building/accessory environment is clean. Local anomaly detection remains §16's job, and an explicitly represented error term MUST NOT be counted twice.

---

## 11. North-reference resolution

Google FOP exposes the same ambiguous north-reference contract through both its scalar heading and attitude frame: true north when declination is available, magnetic north otherwise, with no per-sample flag. Resolve that ambiguity **without replacing Google's fusion and without using a geometrically ill-conditioned axis**.

Resolution is per **active measurement mode and stable window**. Form both hypotheses from the **same physical reference axis**:

```text
gAxis = aggregated Google FOP bearing of the active mode's reference axis
        FLAT_TOP_EDGE   -> getHeadingDegrees() in portrait, cross-checked against
                           the FOP-attitude-derived portrait top-edge bearing
        WALL_FLUSH_BACK -> outward screen normal projected through FOP attitude
                           into FOP ENU; do NOT use top-edge heading in this pose

mAxis = synchronized diagnostic magnetic-north bearing of that SAME physical axis,
        derived through an Android platform magnetic orientation path
        (gravity/accelerometer + magnetometer + SensorManager transforms);
        never raw magnetometer X/Y and never an axis whose horizontal projection
        is singular or fails the pose/conditioning gate

d = WMM declination at the accepted fix, east-positive
tAxis = normalize360(mAxis + d)

rTrue = absoluteCircularDifferenceDeg(gAxis, tAxis)
rMag  = absoluteCircularDifferenceDeg(gAxis, mAxis)

if rTrue <= providerCrossCheckMaxDeg
   and (rMag - rTrue) >= referenceSeparationMarginDeg:
    canonicalTrueHeading = gAxis; correctionDeg = 0.0
    resolvedReference = TRUE_VERIFIED; referenceAmbiguityBound95Deg = 0.0

else if rMag <= providerCrossCheckMaxDeg
        and (rTrue - rMag) >= referenceSeparationMarginDeg:
    canonicalTrueHeading = normalize360(gAxis + d); correctionDeg = +d
    resolvedReference = TRUE_CORRECTED_FROM_MAGNETIC; referenceAmbiguityBound95Deg = 0.0

else if abs(d) <= smallDeclinationAmbiguityMaxDeg:
    canonicalTrueHeading = gAxis; correctionDeg = 0.0
    resolvedReference = TRUE_WITH_AMBIGUITY_BOUND
    referenceAmbiguityBound95Deg = abs(d)

else:
    resolvedReference = UNVERIFIED; no true-north Precision Lock
```

The test is eligible only with fresh location/model evidence, valid synchronized source timestamps, a valid diagnostic magnetic orientation, and `referenceMagneticPrecheckState == CLEAN_FOR_REFERENCE`. This precheck is intentionally **not** the final `MagneticState`: it uses only finite/non-saturated/calibrated magnetic input plus fresh magnitude, positive-down inclination, and stationary-MAD features below their suspect thresholds. It MUST NOT read `pipelineAgreementDeg`, `resolvedReference`, or any feature whose construction needs a reference-resolved Google heading.

The dependency order is fixed and acyclic:

```text
raw synchronized magnetic/motion/WMM evidence
  -> referenceMagneticPrecheckState       // excludes pipelineAgreementDeg
  -> Google ReferenceResolutionResult     // §11
  -> true-referenced active-axis pipeline set and pipelineAgreementDeg // §15.1
  -> final MagneticState                  // §16
  -> lock decision                        // §18.5
```

If the precheck is not `CLEAN_FOR_REFERENCE`, §11 returns `UNVERIFIED`; the engine does not manufacture a Google pipeline reference in order to compute the evidence that would have been needed to resolve it. Because the diagnostic path shares hardware/environment with FOP, the precheck and hypothesis test provide **reference disambiguation and gross-failure detection**, not independent proof of heading correctness. Professional certification still requires §30.5 external ground truth.

Fixed rules:

- The ambiguity branch emits one uncertainty term; the resolver never writes or overwrites `reportedBound95Deg`.
- Because `tAxis = mAxis + d`, hypothesis separation cannot exceed `abs(d)`; the §8.1 ordering invariant prevents a dead band.
- `ReferenceResolutionResult` is bound to `measurementMode`, `referenceAxis`, and its source window. A flat result is not reusable for a wall pose or vice versa. Target guidance may reuse a still-fresh result only within the same unchanged mode/session and only as explicitly provisional (§18.2).
- `correctionDeg` is the **single** Google magnetic→true correction site: exactly `0.0` or `+declinationDeg`, applied once to the final active-axis heading. Double application yields a plausible but catastrophic `2 × declination` error.

Thresholds are candidates: tune on training devices/sites, freeze before held-out evaluation.

**iOS** does not use this hypothesis test for production true-reference paths: valid `CLHeading.trueHeading` is explicit (`PROVIDER_CONTRACT_EXPLICIT`), and `.xTrueNorthZVertical` is explicit when that frame is actually active (`ATTITUDE_FRAME_EXPLICIT`). **`AND-RV`** owns magnetic→true conversion itself (`APP_APPLIED_DECLINATION`).

### 11.1 Active-axis transform contract

`ProviderHeadingSample` remains a raw scalar observation, but **attitude does not remain provider-native past the adapter boundary**. `ProviderAttitudeSample.attitudeQuaternionDeviceToReferenceEnuXYZW` is always a device→project `REFERENCE_ENU` transform, while `providerReferenceContract` says whether the horizontal north/east axes are true- or magnetic-referenced. Google FOP already documents east/north/up axes but carries true-vs-magnetic ambiguity in that contract. Android rotation vector is normalized to the same axis order with an explicit `MAGNETIC` contract. Core Motion `.xTrueNorthZVertical` uses a different provider-native earth-axis convention; the iOS adapter MUST convert **both axis convention and transform direction** to project `REFERENCE_ENU` before emitting the canonical sample, retain the native frame/quaternion in telemetry, and prove the conversion with N/E/S/W/up golden vectors. Do not infer permutation, transpose/inversion, or signs from yaw intuition, and never feed a Core Motion native quaternion directly into `deviceVectorToReferenceEnu`.

Google may emit scalar heading and attitude from one `DeviceOrientation` callback with the same `providerSampleId` and occurrence time. Apple `CLHeading` and `CMDeviceMotion` are separate streams with separate timestamps; the adapter MUST NOT fabricate a common sample time.

`transformAgreementDeg` is a **code-correctness** signal and its routes depend on pose:

- `FLAT_TOP_EDGE`: compare FOP `getHeadingDegrees()` with the portrait top-edge bearing independently extracted from the same FOP attitude.
- `WALL_FLUSH_BACK`: top edge is close to vertical, so its horizontal bearing is ill-conditioned and MUST NOT be compared to the wall normal. Compare two independent implementations of the **same outward-screen-normal projection** from the same FOP attitude (for example direct quaternion-vector rotation versus a rotation matrix obtained through the documented `SensorManager.getRotationMatrixFromVector` route), plus physical N/E/S/W golden poses.
- A large value raises `TRANSFORM_DISAGREEMENT`, never a magnetic-interference reason; it MUST NOT feed `MagneticState`.

If the active reference-axis projection is singular or ill-conditioned, reject the pose; never resolve the north reference on a convenient different axis and transfer that label.

---

## 12. iOS adapter

```swift
let locationManager = CLLocationManager()
locationManager.delegate = headingDelegate
locationManager.desiredAccuracy = kCLLocationAccuracyBestForNavigation
locationManager.distanceFilter = kCLDistanceFilterNone
locationManager.headingFilter = kCLHeadingFilterNone
locationManager.startUpdatingLocation()
locationManager.startUpdatingHeading()

if CMMotionManager.availableAttitudeReferenceFrames().contains(.xTrueNorthZVertical) {
    motionManager.deviceMotionUpdateInterval = 1.0 / periodicOrientationRequestedHz
    motionManager.startDeviceMotionUpdates(using: .xTrueNorthZVertical,
                                           to: motionQueue, withHandler: motionHandler)
    // then read motionManager.attitudeReferenceFrame and record the frame in use
}
```

Initialize `CLLocationManager` on a thread with an active run loop; stop both services on lifecycle/measurement exit. Set `headingOrientation` deliberately for the portrait **CLHeading top-edge** contract; it cannot turn `CLHeading` into a wall-normal sensor. Changing it affects only subsequent heading samples, so reset the lock window at the same time. Before Core Motion start, require `isDeviceMotionAvailable`, verify `.xTrueNorthZVertical` is currently available, set `deviceMotionUpdateInterval` from `periodicOrientationRequestedHz`, and after start record both the observed timestamp-derived rate and `attitudeReferenceFrame`; the requested rate/frame are intentions, observed timestamps/active frame are facts.

`CLHeading` is **event-driven**, not a guaranteed 50 Hz orientation stream. `headingFilter = kCLHeadingFilterNone` requests all heading changes but does not turn it into a periodic source. In `FLAT_TOP_EDGE`, Core Motion supplies the periodic pose/stability support window, while valid `CLHeading.trueHeading` samples inside that same window are the absolute-heading anchors. The engine MUST NOT require 50 `CLHeading` callbacks in 2 seconds or apply the periodic `orientationMaxAgeMs` rule to a stationary `CLHeading` value. A flat-iOS stable window requires at least `clHeadingMinSamplesPerStableWindow` valid `CLHeading` source sample(s) whose timestamps fall inside the final stable window; the first such anchor starts/restarts the eligible window. The separately counted periodic Core Motion support samples satisfy `minPeriodicSupportSamples`.

Do **not** equate `attitude.yaw` with the active reference-axis heading. In `.xTrueNorthZVertical` the reference-frame X axis points true north, but the device vector still has to be transformed through the attitude and projected horizontally. Validate with physical N/E/S/W poses. `FLAT_TOP_EDGE` uses valid `CLHeading.trueHeading` as the production scalar; `WALL_FLUSH_BACK` uses the projected Core Motion true-north attitude as the production bearing and records `providerId = APPLE_CORE_MOTION_TRUE_NORTH`. The two paths are certified separately through the mode/provider key.

**Capture:** `CLHeading` magnetic/true/accuracy/x/y/z/timestamp; `CLLocation` coordinate, `ellipsoidalAltitude`, `altitude`, accuracies, timestamp; `CMDeviceMotion` attitude, rotation rate, gravity, user acceleration, calibrated field + accuracy; observed `attitudeReferenceFrame`; `accuracyAuthorization` and authorization status; interface and heading reference orientation; `ProcessInfo.thermalState`; charging state, wireless vs wired where distinguishable.

**Validity rules:**

- Negative `headingAccuracy` or otherwise invalid provider values MUST invalidate that sample.
- Negative `trueHeading` means undetermined → reject.
- `trueHeading` MUST NOT be accepted without valid, fresh location support.
- `headingAccuracy` is Apple's reported maximum deviation for the *magneticHeading* estimate; Apple does not state a percentile/coverage level for it. Do not relabel it `95%`, and do not assume it covers true-north conversion, wall-axis projection, placement, or environmental bias.
- `desiredAccuracy` is a request, not achieved accuracy, and has no effect under reduced authorization. Gate on fix age and the declination envelope, not a metre cutoff.
- `headingOrientation` MUST match the declared physical axis; transitions invalidate the aggregation window.
- `course` is direction of travel, not device direction.
- Suspend/foreground transitions MUST reset freshness and filter state. A cached pre-suspend heading MUST NOT be emitted as new.

**iOS clock mapping.** `CMLogItem.timestamp` is seconds since device boot, while `CLHeading.timestamp` and `CLLocation.timestamp` are `Date` values. Preserve both raw timestamps/source clocks, map them into the app's monotonic domain through a logged `clock_mapping`, and compute freshness only from the mapped source time. Never subtract a `Date` directly from a Core Motion boot timestamp or substitute callback-arrival time. Re-establish the mapping after detected wall-clock jumps or lifecycle/process discontinuity.

**Discontinuity detection** (iOS has no direct equivalent of Android's flags, so the inference is specified rather than left to each developer). Raise `SENSOR_DISCONTINUITY` and reset the window when: updates stop and restart for any reason including a frame change; consecutive `CMDeviceMotion` source timestamps gap beyond `orientationInvalidAfterMs`; `CMCalibratedMagneticField.accuracy` transitions to uncalibrated or invalidates the prior window; the app returns from background or the process is recreated; `attitudeReferenceFrame` differs from window start.

---

## 13. Android adapter

```kotlin
val orientationClient = LocationServices.getFusedOrientationProviderClient(context)
val orientationRequest = DeviceOrientationRequest.Builder(
    DeviceOrientationRequest.OUTPUT_PERIOD_DEFAULT   // 20,000 us / requested 50 Hz
).build()
orientationClient.requestOrientationUpdates(orientationRequest, orientationExecutor, orientationListener)
// lifecycle exit MUST pair with:
orientationClient.removeOrientationUpdates(orientationListener)
```

API contract details that are non-negotiable:

- `getFusedOrientationProviderClient` then `requestOrientationUpdates`; older orientation methods on `FusedLocationProviderClient` are deprecated.
- Updates are delivered only while the app is foreground.
- `OUTPUT_PERIOD_DEFAULT` = 50 Hz, `MEDIUM` = 100 Hz, `FAST` = 200 Hz. These are requested periods, not guaranteed observed rates. Google describes the faster presets as higher-precision update modes, but **rate alone does not establish better absolute-north accuracy**; 50/100/200 Hz remain separate benchmark factors for accuracy, latency, power, and thermal behavior.
- `getAttitude()` is `[qx, qy, qz, qw]` mapping device → ENU in Google's convention. Do not reorder into another library's convention.
- `getElapsedRealtimeNs()` is elapsed-realtime, not Unix time.
- Both error methods describe error in the API's reported **display-top scalar heading** and use `180°` for invalid/complete ignorance. For small angles Google describes that error cone as two-sigma, approximately a two-sided 95th-percentile interval. Conservative error exists only when `hasConservativeHeadingErrorDegrees()` and waits for sufficient rotation in a uniform field before leaving `180°` — a normal startup condition handled per §18.4 for the flat scalar path. Google does not document this scalar error as a bound on an arbitrary axis projected from `getAttitude()`; wall outward-normal uncertainty therefore MUST NOT inherit it. Record `providerErrorSource`; the ordinary and conservative branches are not certification-equivalent.
- Heading follows current screen rotation along the API's display top. v1 avoids confusing display top with the product axis by locking portrait.
- Heading is true north **when declination is available and magnetic otherwise**, with no per-sample flag. Never blindly label it `TRUE` (§11).

**Per-callback gates** before a sample may enter a window:

```text
finite normalized quaternion in documented [x,y,z,w] order
0 <= elapsedRealtimeNowNs - sampleElapsedRealtimeNs <= orientationMaxAgeNs
screen rotation unchanged since window start
foreground measurement session still owns the listener

FLAT_TOP_EDGE additionally requires finite getHeadingDegrees() in [0,360) and:
    if hasConservativeHeadingErrorDegrees():
        conservative error MUST be finite and < 180;
        180 => PROVIDER_INITIALIZING / ROTATE_TO_INITIALIZE; DO NOT fall through to ordinary
    else:
        ordinary error MUST be finite and < 180;
        providerErrorSource = GOOGLE_ORDINARY;
        certification earned under GOOGLE_CONSERVATIVE MUST NOT carry over

WALL_FLUSH_BACK derives its bearing from attitude; scalar heading/error are diagnostic only.
The wall path uses providerErrorSource = NONE and relies on the exact wall provider/mode
device floor plus held-out coverage. A scalar 180° does not itself initialize, reject, or
widen the wall result; it remains telemetry and may make scalar-based diagnostics absent.
```

In portrait-locked v1, do **not** re-apply screen remapping to `getHeadingDegrees()` — Google already reports the portrait display-top direction. Remapping via `SensorManager.remapCoordinateSystem()` (or an equivalent explicit matrix transform) *is* required for the raw `SensorManager` path, whose axes stay tied to the device's default screen orientation and do not follow UI rotation. Future landscape support applies one documented circular offset and passes the full orientation benchmark.

**No-GMS path.** `AndroidRotationVectorHeadingProvider` uses `TYPE_ROTATION_VECTOR` requested at `periodicOrientationRequestedHz`, `getRotationMatrixFromVector`, explicit device→magnetic-ENU transforms, and shared NOAA WMM correction applied exactly once. Android's rotation-vector event may expose estimated heading accuracy in `values[4]` (radians; `-1` means unavailable); when present convert to degrees and record `providerErrorSource = ANDROID_ROTATION_VECTOR_HEADING_ACCURACY_95`.

Its confidence level is **documented, not assumed**: the AOSP sensor-type contract requires the heading error to be less than the reported estimated accuracy 95% of the time. That is exactly the confidence level this project's bound model targets, so the term enters `providerReportedBoundTermDeg` directly with no `boundFromSigma` conversion.

This term cannot overcome the unknown-device floor. Because `baseHeadingBound95Deg` is a `max`, adding the RV term can only leave the base unchanged or increase it; it can never reduce `unknownDeviceFloor95Deg`, create reachability, or provide certification. Its value is to prevent an overconfident bound when the provider reports worse uncertainty than the floor and to supply evidence for later device/provider/mode certification. Reachability improves only through a certified lower `deviceFloor95Deg`, a smaller validated placement bound, or a changed lock ceiling supported by benchmark evidence.

Two caveats bound that use. The 95% figure is a compliance requirement placed on OEM sensor implementations, not a per-device guarantee, and Android's comparable accuracy contracts are known to be optimistic on real hardware; the term is therefore evidence subject to the same held-out coverage validation as every other term, never a substitute for it. And when `values[4]` is `-1` or the field is absent, the term is **absent, not zero**.

For `FLAT_TOP_EDGE`, project the portrait top-edge axis. For `WALL_FLUSH_BACK`, project the Android device `+Z` outward-screen normal through the rotation-vector transform; do not derive wall bearing from `getOrientation()` azimuth/top-edge. Apply WMM declination to the resulting magnetic-axis bearing. Both modes retain `providerId = ANDROID_ROTATION_VECTOR` and are certified separately by mode.

The path MUST have a distinct capability screen, telemetry, risk curve, device certification list, and uncertainty floor. If neither certified path is available → `UNSUPPORTED_DEVICE`. `TYPE_GAME_ROTATION_VECTOR` MUST NOT substitute (it excludes the magnetic field); `TYPE_GEOMAGNETIC_ROTATION_VECTOR` SHOULD NOT be the precision primary (no gyroscope, documented lower accuracy).

On Android 13/API 33+ the optional `TYPE_HEADING` sensor provides a **true-north scalar heading** in `[0.0, 360.0)` plus an accuracy value documented at **68% confidence**, which AOSP states is one standard deviation for a Gaussian distribution. Record it as diagnostic `ProviderId = ANDROID_HEADING` when present and benchmark it as a flat-mode cross-check; it supplies no attitude for wall-normal projection and is not a silent substitute for either v1 production path.

Its accuracy field is a **one-sigma** quantity, so §19.2 applies in full: any comparison against this project's 95% bounds, in a scorecard or anywhere else, MUST first convert it through `boundFromSigma` and record `providerErrorSource = ANDROID_HEADING_ACCURACY_68`. Placing a 68% number beside a 95% number in the same column is the same confidence-level conflation §19.2 exists to prevent, and it under-states this sensor's interval by roughly a factor of two.

**Diagnostic streams:** `TYPE_ROTATION_VECTOR`, `TYPE_MAGNETIC_FIELD`, `TYPE_MAGNETIC_FIELD_UNCALIBRATED`, `TYPE_GYROSCOPE`, `TYPE_ACCELEROMETER`/`TYPE_GRAVITY`. Use `getRotationMatrix()`/`getOrientation()` for the accel+mag baseline. Never blend a diagnostic path back into the fused heading with an unvalidated filter. Record public sensor descriptors such as name, vendor, version, resolution, range, power, reporting mode, min/max delay, and wake-up flag. On Android their canonical sorted hash is `sensorRuntimeIdentity` (§24); a physical-vs-synthesized label is recorded only when the runtime API actually supplies defensible evidence. Do not infer hidden hardware identity from performance.

**Timestamps.** Sensor and FOP elapsed-realtime timestamps MUST remain in the monotonic domain until aligned. Wall-clock UTC and `elapsedRealtimeNanos()` MUST NOT be subtracted without a measured, logged mapping, re-established across boot.

**Discontinuity:** raise on `onAccuracyChanged` transitions that invalidate the window, source-timestamp gaps beyond `orientationInvalidAfterMs`, unregister/re-register, and process recreation.

**Location.**

```kotlin
val currentRequest = CurrentLocationRequest.Builder()
    .setPriority(Priority.PRIORITY_HIGH_ACCURACY)
    .setGranularity(Granularity.GRANULARITY_FINE)
    .setMaxUpdateAgeMillis(0L)          // freshly derived, not history
    .setDurationMillis(10_000L)
    .build()
fusedLocationClient.getCurrentLocation(currentRequest, cancellationToken)
```

After the initial fix, request foreground updates only as long as the session needs them. Request parameters are not proof of quality — inspect delivered timestamp, accuracy, permission mode, mock status, provider result. Request coarse and fine per current rules; detect approximate-only grants including runtime downgrades that restart the process; treat one-time grants as expiring and re-check rather than caching a past "granted" state; detect disabled services and API unavailability; never require background location; stop reliably on exit/cancel/error/recreation; log Play services version and availability.

---

## 14. Measurement modes

The provider does production tilt compensation. The app validates pose and resulting error; it does **not** tilt-correct the fused provider a second time.

### `FLAT_TOP_EDGE`
- Phone approximately face-up; portrait top-edge vector indicates bearing.
- Android reads `getHeadingDegrees()` directly in portrait; iOS uses valid `CLHeading.trueHeading`.
- Pose gate: `abs(pitch) <= flatModePitchAbsMaxDeg`, `abs(roll) <= flatModeRollAbsMaxDeg`.

### `WALL_FLUSH_BACK`
- Phone back flush to the wall/door; screen faces the measured direction.
- Bearing is the horizontal projection of the **outward screen normal**, not the top edge.
- Google Android uses FOP canonicalized attitude (`providerId = GOOGLE_FOP`) and the §11 mode-axis reference resolver.
- No-GMS Android uses `TYPE_ROTATION_VECTOR` (`providerId = ANDROID_ROTATION_VECTOR`): project device `+Z` through the magnetic-ENU transform, then apply WMM declination exactly once.
- iOS uses Core Motion `.xTrueNorthZVertical` (`providerId = APPLE_CORE_MOTION_TRUE_NORTH`) **after the adapter converts the provider-native attitude to project canonical REFERENCE_ENU**.
- Convert the platform-documented outward-screen unit vector through the canonical device→REFERENCE_ENU transform, then `normalize360(degrees(atan2(east, north)))`.
- Outward normal within `wallNormalElevationAbsMaxDeg` of horizontal; portrait top axis within `wallTopAxisFromVerticalMaxDeg` of vertical (so the phone is not rolled sideways).
- Reject if the horizontal projection is ill-conditioned or true-north attitude/reference is unavailable.
- **Facing (向)** is the outward normal and is the default reported value. **Sitting (坐)** is exactly `normalize360(facing + 180°)`, computed only on request, and MUST be labelled as the derived opposite rather than presented interchangeably. Reporting the wrong one is a 180° error that looks entirely plausible on a dial.

Provider top-edge heading is not used for wall mode: a portrait top edge points roughly upward and its horizontal bearing is ill-conditioned. Every transform MUST pass physical golden poses with the screen facing true N/E/S/W. Do not reuse thresholds, provider fields, or transforms between modes without validation.

**Diagnostic/fallback magnetic path only:** never compute heading from magnetometer X/Y alone; transform the field into the horizontal earth plane using platform rotation matrices. Reject when gravity and magnetic samples are too far apart in time, acceleration makes gravity unreliable, inclination makes the projection ill-conditioned, horizontal intensity is below `minHorizontalIntensityNanoTesla`, or required transforms/display orientation are unknown.

---

## 15. Filtering and aggregation

Maintain four distinct values: `rawProviderHeading`, `rawIndependentHeading`, `liveFilteredHeading`, `lockedStatisticalHeading`.

```text
// Every signed circular difference here calls shortestSignedDifferenceDeg directly.
// Do not introduce a local alias or formula; see R67/R68 and §33.1.

C = sum(w_i * cos(radians(theta_i))) / sum(w_i)
S = sum(w_i * sin(radians(theta_i))) / sum(w_i)
mean = normalize360(degrees(atan2(S, C)))
R    = sqrt(C^2 + S^2)
```

Three decisions are fixed so two platform implementations cannot diverge:

1. **Weights are uniform.** Every accepted sample has `w_i = 1`. Weighting by provider error, recency, or dispersion is a plausible and untested improvement; it becomes a named benchmark variant, not a quiet implementation choice.
2. **No outlier trimming inside the window.** Rejection happens at the per-sample gate, before entry. Once accepted, a sample counts. `sampleBound95Deg` is therefore P95 of residuals over *all* accepted samples about the circular mean, and the same set feeds `circularResidualP95MaxDeg`. Trimming would let the window discard exactly the evidence that it is unreliable, and would make the dispersion gate and the dispersion-derived bound disagree about which samples exist. Any trimming rule is a versioned change needing evidence that it improves held-out coverage rather than merely narrowing the interval.
3. **A weak resultant is an explicit failure.** If `R < minCircularResultantLength`, emit `CIRCULAR_MEAN_UNDEFINED` and reject. Do not display `0° ± small` from a bimodal set — `atan2(0,0)` returns zero on both platforms and would disguise the condition as a north-facing measurement.

An adaptive live filter MAY smooth more while stationary and less while rotating, but filtering MUST be measured for latency, overshoot, settling, and the quality engine MUST inspect signals raw enough that a long time constant cannot hide instability.

Stability means **a compact cluster with low motion**, not a constant digit. `84.7, 85.3, 84.9, 85.4, 85.0` may lock if residual, angular-speed, acceleration, pose, and duration gates pass. The engine MUST NOT require the live display to equal a target exactly, and MUST NOT mistake visual interpolation for sensor stability.

### 15.1 Cross-validation

Cross-validation is **mode-axis-specific**. Every compared pipeline MUST represent the same physical reference axis at synchronized source times and the same north reference before a circular difference is computed. A top-edge scalar is not comparable to a wall-normal bearing merely because both are called “heading.”

Required candidate sets when fresh/valid:

| Platform / mode | Comparable true-referenced pipelines |
|---|---|
| iOS `FLAT_TOP_EDGE` | `CLHeading.trueHeading`; Core Motion canonical REFERENCE_ENU top-edge projection; `CLHeading.magneticHeading + WMM` |
| iOS `WALL_FLUSH_BACK` | Core Motion canonical REFERENCE_ENU outward-normal projection; diagnostic magnetic outward-normal projection + WMM. `CLHeading.trueHeading` is **not** a wall pipeline and is excluded. |
| Android Google `FLAT_TOP_EDGE` | FOP display-top heading after §11 resolution; rotation-vector top-edge + WMM; accel/gravity+mag top-edge + WMM; optional `TYPE_HEADING` diagnostic |
| Android Google `WALL_FLUSH_BACK` | FOP outward-normal projection after §11 resolution; rotation-vector outward-normal + WMM; accel/gravity+mag outward-normal + WMM. FOP display-top scalar and `TYPE_HEADING` are excluded. |
| Android no-GMS `FLAT_TOP_EDGE` | rotation-vector top-edge + WMM; accel/gravity+mag top-edge + WMM; optional `TYPE_HEADING` diagnostic |
| Android no-GMS `WALL_FLUSH_BACK` | rotation-vector outward-normal + WMM; accel/gravity+mag outward-normal + WMM |

`pipelineAgreementDeg` is the **maximum pairwise absolute circular difference** over the valid set actually available for the active mode, and telemetry MUST record the exact pipeline IDs/axes comprising that set. Fewer than two valid independent active-axis pipelines means the feature is `ABSENT`; because v1 makes pipeline disagreement a required interference feature, the magnetic classifier then resolves `UNKNOWN` and refuses a true-heading lock rather than pretending agreement.

Persistent disagreement MUST raise uncertainty or reject. Agreement is evidence, not proof — paths may share sensors, calibration, and vendor fusion. Cross-validation never transfers a reference label or certification between axes.


---

## 16. Magnetic interference detection

```text
M = sqrt(Bx^2 + By^2 + Bz^2)                            // microtesla
relativeMagnitudeResidual = abs(M - expectedM) / expectedM

measuredInclinationPositiveDownDeg
    = degrees(asin(clamp(-Bup / M, -1, 1)))              // input is ENU +up
expectedInclinationPositiveDownDeg = WMM inclination I   // WMM convention: +down
inclinationResidualDeg
    = measuredInclinationPositiveDownDeg - expectedInclinationPositiveDownDeg
                                                          // LINEAR; [-90,90]

stationaryFieldMadMicroTesla = median absolute deviation of M over a stationary window
gyroMagInnovationDeg = magneticHeadingDelta - integratedGyroDelta
```

The minus sign is mandatory: canonical REFERENCE_ENU has `Bup` positive upward, while WMM inclination `I` (and WMM vertical component `Z`) are positive downward. Comparing `asin(Bup/M)` directly with WMM `I` reverses the observed sign and can reject a clean northern-hemisphere field as disturbed. Inclination cannot wrap; a circular difference there is a category error that silently rescales the residual near the poles. Assert the input range and test northern/southern hemispheres plus `I = 0°`.

**The detector MUST NOT use magnitude alone** — a disturbance can rotate the field vector with little magnitude change, and that is precisely the case producing a confident wrong bearing. Conversely a magnitude mismatch alone may be model/site limitation. The v1 classifier runs **after** any Google §11 resolution and fuses magnitude, inclination, and stationary variability, plus independent-pipeline disagreement:

```text
if any magnetic value nonfinite, or sensor saturated, or OS calibration invalid:
    INVALID
else if relativeMagnitudeResidual    >= magneticMagnitudeResidualDisturbedFraction
     or abs(inclinationResidualDeg)  >= inclinationResidualDisturbedDeg
     or stationaryFieldMadMicroTesla >= stationaryFieldMadDisturbedMicroTesla
     or pipelineAgreementDeg         >= pipelineDisagreementDisturbedDeg:
    DISTURBED
else if relativeMagnitudeResidual    >= magneticMagnitudeResidualSuspectFraction
     or abs(inclinationResidualDeg)  >= inclinationResidualSuspectDeg
     or stationaryFieldMadMicroTesla >= stationaryFieldMadSuspectMicroTesla
     or pipelineAgreementDeg         >= pipelineDisagreementSuspectDeg:
    SUSPECT
else if every required feature is fresh and valid:
    CLEAN
else:
    UNKNOWN
```

`stationaryFieldMadMicroTesla` is evaluated only while motion gates indicate stationary; when motion disqualifies it the feature is **absent** and the classifier falls to `UNKNOWN` rather than treating an unmeasured feature as passing. The same rule applies when fewer than two valid independent **active-axis** pipelines exist for `pipelineAgreementDeg`: absent evidence is not zero disagreement.

`referenceMagneticPrecheckState` and final `MagneticState` MUST be recorded separately. The precheck is `CLEAN_FOR_REFERENCE` only when the three non-pipeline features required by final classification are fresh/valid and below their suspect thresholds; otherwise it is `NOT_CLEAN_FOR_REFERENCE`, or `UNKNOWN` when required evidence is absent. It is a narrow eligibility gate for §11, never a substitute for final classification. A Google lock still requires the post-resolution final `MagneticState`, including `pipelineAgreementDeg`.

`gyroMagInnovationDeg` is **recorded but not gated in v1**. It is informative — magnetic heading change disagreeing with integrated gyro rotation is strong anomaly evidence — but a threshold needs per-device gyro-bias characterization the benchmark has not produced. Recording it now lets the v2 threshold come from archived data rather than a new field campaign.

`UNKNOWN` cannot receive Professional and in v1 cannot produce a true-heading lock at all (`MAGNETIC_FIELD_UNKNOWN`). Whether a lower tier is defensible under `UNKNOWN` is a held-out benchmark question; the conservative pre-evidence behaviour is refusal.

Use hysteresis but bound recovery: returning to `CLEAN` requires a continuous clean interval of at least `recoveryCleanWindowMs` plus a new stable window. The app MUST NOT show a green/high-confidence reading while the engine considers the measurement disturbed. `CLEAN` means *no significant disturbance detected by the validated detector from the available evidence*; it MUST NOT be described as proof that the local field is perfect or free of all bias.

### 16.1 Two kinds of disagreement

| Metric | Compares | Large value means | Reason code |
|---|---|---|---|
| `transformAgreementDeg` | Two mathematically independent routes from the **same** provider observation and the **same well-conditioned physical axis** (§11.1) | Frame transform, quaternion order, axis selection, or remapping is wrong. Environment irrelevant. | `TRANSFORM_DISAGREEMENT` |
| `pipelineAgreementDeg` | **Independent** estimators (FOP, rotation vector, accel+mag, Core Motion) | Local field likely disturbed, or a sensor failing. Feeds the classifier above. | `PIPELINE_DISAGREEMENT` |

Only the second is environmental evidence. `transformAgreementDeg` MUST NOT contribute to `MagneticState`. Telling a user to move away from metal because the wall-mode quaternion has a swapped axis is a failure that survives a long time in the field — the advice is plausible and the user complies.

---

## 17. Calibration

Three unrelated concepts, kept distinct in code and UI:

1. **Sensor calibration** — OS/vendor magnetometer state and motion coverage. The app observes and guides; it never writes persistent offsets into the sensor stack.
2. **Deviation characterization** — external-ground-truth measurement of residual error vs azimuth (§29.3). Does not by itself authorize correction.
3. **Uncertainty calibration** — held-out proof that the reported bound attains its target coverage. This alone moves `boundCalibrationState` to `CALIBRATED`, via a certification-database hit.

Diagnostics MUST qualify the word: `sensor calibration: good` is not `uncertainty bound: calibrated`.

**Triggers and flow.** Automatic triggers: invalid/poor OS calibration, excessive provider uncertainty, inconsistent field ellipsoid/bias, sustained cross-pipeline disagreement, failure to recover after disturbance removal. The user MUST *always* be able to invoke **Check / Recalibrate** even when automatic evidence looks fine — automatic detection is a safety aid, not an oracle. Manual entry uses `USER_REQUESTED` and does not bypass checks or lower success criteria.

```text
calibration good              -> do not force movement; measure normally
poor/unknown + field clean    -> recommend 3D motion; reassess objectively
user requests Check/Recal     -> enter check even if state looks good; reassess objectively
field DISTURBED or INVALID    -> defer the motion, explain why, ask to move away first
```

Guidance when motion is appropriate: remove magnetic cases, wallets, chargers, mounts, ring holders, accessories; move away from vehicles, steel furniture, speakers, wiring, reinforced structures where practical; rotate slowly through multiple independent 3D orientations (figure-eight is a means, not proof); keep clear of the body and electronics; return to the measurement orientation and wait for stability.

**Success is measured, not animated.** Compare before/after where observable: OS calibration state/error, fitted hard-iron bias and soft-iron residual, ellipsoid orientation coverage, magnitude/inclination residual and stationary variance, provider uncertainty, cross-pipeline disagreement, absolute error in benchmark runs. Distinguish **device calibration failure** from **environmental interference** — repeated calibration cannot fix a steel beam, and an endless figure-eight loop is unacceptable. A user-requested recalibration may legitimately produce `ACCEPTABLE_NO_CHANGE`, `STILL_POOR`, `ENVIRONMENT_DISTURBED`, or `INVALID_OR_INCONCLUSIVE`. Never force a success state.

---

## 18. Measurement workflow

### 18.1 Practitioner protocol (implement as actionable UX, not documentation)

1. Remove magnetic cases, rings/mounts, chargers, magnetic wallets, earbuds/cases, watches/clasps, nearby electronics. Disconnect wireless charging.
2. Move away from vehicles, steel structures/furniture, speakers, electrical panels/cables, reinforced-concrete hot spots. There is no universal safe distance; the live detector decides.
3. Obtain fresh location/reference evidence and check Sensor Health before trusting the heading.
4. Calibrate when recommended or when manually requested. Do not calibrate through a disturbed field.
5. Select the correct mode and placement method; keep the declared reference axis on the direction actually being measured.
6. When seeking a target, rotate normally — directions passed through are never averaged into the lock.
7. Near the target, slow and hold **naturally steady**. Perfect immobility is not required.
8. Wait for Precision Lock. A moving live heading or a momentary exact target match is not a measurement.
9. If the app reports a limiting reason, fix that condition rather than using the number anyway.
10. Interpret the result with reference, bound, `CANDIDATE`/`CALIBRATED` state, magnetic state, placement method, and straddle set.
11. For important measurements, repeat with a full re-placement (§18.3).

### 18.2 Live tracking, target guidance, trust

Samples stream continuously; **only samples in the final accepted stable window enter the locked heading.**

```text
targetDeltaDeg = shortestTargetDeltaDeg(liveHeadingDeg, targetHeadingDeg)
// positive = clockwise/right, negative = counterclockwise/left
```

Sign convention MUST be documented and tested across north wrap. While rotating, guidance is provisional and implies no lock. On Android, if a fresh resolved true-reference is unavailable, true-target guidance MUST visibly say verification is pending, or reuse the most recent still-valid resolution from the same unchanged session as explicitly provisional. The final target result re-enters the normal gates.

A flashing `85.0°` is not a measurement. `targetNearZoneDeg` only switches the UI to fine guidance; it is not an accuracy claim. After a valid lock, set `targetCentered` only when `abs(shortestTargetDeltaDeg(locked, target)) <= targetCenteringToleranceDeg`; otherwise resume guidance. Even when centered, the app MUST still show the measurement uncertainty interval and any sector straddle.

State flow, with any failed invariant transitioning to a visible `DEGRADED` / `INVALID` / `TIMED_OUT` plus reason codes:

```text
IDLE -> ACQUIRING_LOCATION -> ACQUIRING_ORIENTATION -> PROVIDER_INITIALIZING?
     -> CALIBRATION_CHECK -> MAGNETIC_FIELD_CHECK -> TARGET_SEEKING?
     -> LEVEL_AND_HOLD -> STABILIZING -> PRECISION_LOCKED
```

`TrustAction` answers *is the present evidence good enough right now?* by combining certification prior, live sensor health, calibration status, magnetic state, reference/location validity, pose/stability, placement, and bound validity. Exactly one action is emitted per snapshot:

```text
READY_CALIBRATED                locked result with an earned calibrated-coverage label
READY_CANDIDATE                 a candidate lock may be shown only when the active candidate
                                constants make a lock arithmetically reachable; with the
                                shipped unknown-device floor, ordinary freehand is DEGRADED
SHOW_DEGRADED_RESULT            a finite result exists above the lock ceiling; show its bound
                                and limiting reason without lock styling or a certified badge
HOLD_STEADY                     more stable evidence required
ROTATE_TO_INITIALIZE            provider has not yet produced a usable error estimate (§18.4)
CALIBRATE                       sensor-calibration check/motion required
MOVE_AWAY_FROM_INTERFERENCE     environment/accessory problem dominates
REACQUIRE_REFERENCE_OR_LOCATION north/location evidence insufficient
UNSUPPORTED_OR_REJECTED         no trustworthy precision result
```
 A good device is rejected in a bad environment; a certified model behaving poorly is downgraded; an unknown healthy device can produce `READY_CANDIDATE` only in a geometry whose candidate terms are lock-reachable, and otherwise produces `SHOW_DEGRADED_RESULT` or rejection. Brand, price, age, and internal agreement never prove trust.

**State transitions.** Backgrounding, losing ownership, orientation change, north-reference change, sensor discontinuity, or permission/location-mode change resets the lock window. A transition to `DISTURBED` invalidates a live lock immediately. A timeout goes to `TIMED_OUT` and emits **no** measurement — it MUST NOT freeze the last number. Recovery needs the full `recoveryCleanWindowMs` plus a new stable window. Entering `Check / Recalibrate` invalidates the lock and requires fresh magnetic/stability checks on return. Changing target north reference invalidates reference-dependent guidance. The UI may animate between valid samples, but the data model retains raw occurrence time and never relabels interpolated frames as measurements.

### 18.3 Independent repeat confirmation

A repeat MUST be a genuinely new placement: release or move the device, re-establish alignment, acquire a new stable window. Multiple locks from one unchanged hold are correlated samples, not confirmation.

```text
repeatDeltaDeg = absoluteCircularDifferenceDeg(a.trueHeadingDeg, b.trueHeadingDeg)
repeatConsistencyEnvelopeDeg = min(180, a.reportedBound95Deg + b.reportedBound95Deg)
```

Exceeding the envelope raises `REPEAT_MEASUREMENT_INCONSISTENT` and prompts investigation of placement, environment, calibration, or device state. Overlap means consistent **within the app's current bounds** — a conservative screening comparison, not a hypothesis test and not proof of correctness. While either result is `CANDIDATE` it is diagnostic only.

Do **not** reduce uncertainty by `1/sqrt(n)` because repeats agree. Any rule combining repeats into a tighter bound needs its own held-out evidence; v1 preserves the individual results and their circular difference.

### 18.4 Provider initialization and the 180° conservative error

Google's conservative scalar-heading error deliberately reports `180°` until sufficient rotation occurs in a uniform field. This subsection gates Google `FLAT_TOP_EDGE`, whose production bearing is that scalar heading. It does not gate `WALL_FLUSH_BACK`: Google's documented error is for the display-top heading, not evidence for an outward-normal projection from attitude (§13). The Google-specific initialization contract also MUST NOT be projected onto Apple: Apple exposes negative/invalid `headingAccuracy` when heading is invalid but does not document the same `180°` initialization state. On the applicable Google flat path, this startup condition is neither a device fault nor proof of magnetic interference.

- On Google flat conservative-error startup, enter `PROVIDER_INITIALIZING`, emit `ROTATE_TO_INITIALIZE`.
- Give a concrete instruction ("rotate the phone slowly to initialize the compass"). A user shown `HEADING_ERROR_INVALID` concludes the phone is broken; a user given the instruction complies in seconds.
- Do **not** conflate with `CALIBRATE`. The sensor may be perfectly calibrated; the fusion has simply not observed enough rotation to bound its own error.
- Record time-in-initialization; its distribution across devices is a real usability metric (§30.3).
- If Google conservative error is unavailable but ordinary error is valid, record `providerErrorSource = GOOGLE_ORDINARY`; a certification record earned under `GOOGLE_CONSERVATIVE` MUST NOT carry over. On Apple, use Apple's own validity/calibration semantics rather than manufacturing a Google-style initialization state.

### 18.5 Lock gates

**Applicability is part of each gate.** A provider-specific prerequisite MUST NOT be applied to a path that cannot expose that signal. Before the common table, use this provider/mode contract:

| Path | Absolute-heading source | Periodic stability/pose support | Provider-error rule | Reference rule | Transform-agreement rule |
|---|---|---|---|---|---|
| Google FOP flat | FOP scalar | FOP attitude + diagnostics | Google conservative if advertised, otherwise ordinary | §11 active-axis hypothesis resolver | FOP scalar vs FOP-attitude top edge |
| Google FOP wall | FOP outward-normal attitude projection | FOP attitude + diagnostics | none; display-top scalar error is diagnostic and out of scope | §11 active-axis hypothesis resolver | two independent outward-normal extraction implementations |
| Android RV flat/wall | RV axis projection + WMM | RV + diagnostics | RV `values[4]` when available (AOSP-documented 95%); otherwise absent | `APP_APPLIED_DECLINATION` | deterministic transform/golden tests; no Google scalar comparison |
| iOS flat | `CLHeading.trueHeading` event anchors | Core Motion at requested periodic rate + diagnostics | `CLHeading.headingAccuracy` on anchors | `PROVIDER_CONTRACT_EXPLICIT` | no FOP-style scalar/attitude gate; physical/golden CL-vs-CM cross-check is diagnostic |
| iOS wall | Core Motion outward-normal projection | Core Motion + diagnostics | no provider degree-error term | active `.xTrueNorthZVertical` → `ATTITUDE_FRAME_EXPLICIT` | canonical-frame conversion/golden tests; no CLHeading scalar comparison |

Common gates then apply only where their required signal exists:

| Gate | Value | On failure |
|---|---:|---|
| Periodic orientation/support source age | `<= orientationMaxAgeMs` | Drop periodic sample; invalidate after `orientationInvalidAfterMs` without fresh periodic support. Does not age-reject a stationary event-driven `CLHeading` anchor at 100 ms. |
| iOS flat heading anchors | `>= clHeadingMinSamplesPerStableWindow` valid `CLHeading` source samples timestamped inside the final stable window | Continue/restart stable window; never synthesize callbacks. |
| Location evidence | prefer `<= 60 s`; up to `5 min` Professional, `30 min` Usable if no jump and envelope passes | Reacquire or downgrade. |
| Declination envelope | Versioned deterministic WMM envelope over accepted horizontal/altitude/time uncertainty; `<= 0.10°` Professional, `<= 0.25°` Usable | Downgrade or no true lock. A sparse stencil MUST NOT be called conservative unless dense-grid tests bound its under-estimation. |
| Location jump | `> 50 km` from prior accepted session needs a current fix | Reacquire; travel is not an error. |
| Permission label | precise/full or approximate/reduced always shown and logged | Reduced is not auto-rejected if its envelope passes. |
| Model validity | measurement date inside coefficient interval | `GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE`; prompt to update. |
| Horizontal field | model `H >= minHorizontalIntensityNanoTesla` | `WEAK_HORIZONTAL_FIELD`; no lock. |
| Google flat provider initialization | If FOP advertises conservative error, it must be `< 180°`; if it does not advertise the field, ordinary error may be used under a different key | `ROTATE_TO_INITIALIZE` only for the Google flat scalar-heading startup contract (§18.4). Not applicable to FOP wall. |
| Stable-window duration | `>= stableWindowMinMs`, max `acquisitionTimeoutMs` | Continue or time out with reason. |
| Periodic support samples | `>= minPeriodicSupportSamples` over the stable window for the path's periodic support stream | Continue; duplicate occurrence timestamps do not count. |
| Circular resultant | `>= 0.995` | `CIRCULAR_MEAN_UNDEFINED`. |
| Angular speed | window P95 `<= 3°/s` | Reset stable timer. |
| Linear acceleration | window P95 `<= 0.05 g` | Reset stable timer. |
| Flat-mode level | `abs(pitch), abs(roll) <= 5°` | Prompt to level; no lock. |
| Wall normal elevation | `<= 5°` from horizontal | Prompt to hold flush; no lock. |
| Wall top axis | `<= 5°` from vertical | Prompt to rotate upright; no lock. |
| Placement uncertainty | finite bound from method | Default to freehand bound; **never zero**. |
| Circular P95 residual | `<= 3°` | Reject unstable. |
| Transform agreement | Apply only where the provider/mode row above defines two same-observation extraction routes; threshold `transformAgreementMaxDeg` | `TRANSFORM_DISAGREEMENT` — code fault, not environment. Paths without such a route use their own frame/golden contract, not a fabricated comparison. |
| Reference check | FOP: §11 resolved; iOS: explicit provider/frame contract; AND-RV: `APP_APPLIED_DECLINATION` | No `TRUE` lock. `providerCrossCheckMaxDeg` applies only to the FOP hypothesis resolver. |
| Magnetic state | `CLEAN`, or `SUSPECT` only under the explicit v1 penalty rule | `SUSPECT` adds `suspectInterferenceBound95Deg`; with the candidate constants that term alone exceeds the freehand instrument budget, so `SUSPECT` prevents a freehand lock outright rather than merely capping the grade (§8.1.1). `UNKNOWN`/`DISTURBED`/`INVALID` reject. |
| Space weather | `UNKNOWN` allowed; advisory/suppression from config; fresh `Kp >= spaceWeatherRejectKpMin` → `EXTREME_WMM_UNUSABLE` | Network loss never blocks; a fresh extreme state emits `SPACE_WEATHER_EXTREME` and refuses Precision Lock under the v1 WMM-dependency policy. |
| Charging | wireless charging active | Always recorded; blocks any grade above `USABLE`. |
| Thermal | severe/critical restriction | `THERMAL_RESTRICTION`; no lock. |
| Orientation/lifecycle discontinuity | none in window | Discard the whole window. |
| Total bound | `reportedBound95Deg <= 5°` for a lock | Above 5° is a degraded result, never a lock. |

Every number here is the current value of a **named config key** (§8); the code reads keys, the table shows values. A test MUST assert no gate compares against a numeric literal. Rows that name only their value map to `stableWindowMinMs`, `acquisitionTimeoutMs`, `periodicOrientationRequestedHz`, `minPeriodicSupportSamples`, `clHeadingMinSamplesPerStableWindow`, `angularSpeedP95MaxDegPerSec`, `linearAccelerationP95MaxG`, `locationJumpRequiresFreshFixKm`, `precisionScreenOrientation`, `requireBoundaryStraddleReporting`, `canonicalAltitudeReference`, `spaceWeatherRejectKpMin`, `thermalRestrictionBlocksLock`, and `wirelessChargingBlocksGradeAboveUsable`.

**Lock vs degraded vs invalid.** `PRECISION_LOCKED` requires `reportedBound95Deg <= usableBound95MaxDeg`. Between `usableBound95MaxDeg` and `lowConfidenceBound95MaxDeg` the result is `DEGRADED` with grade `LOW_CONFIDENCE`: shown with its bound and limiting reason, never lock-styled, saveable only through an explicit user action that records the degraded flag. Above `lowConfidenceBound95MaxDeg`, or with an unknown bound, it is `INVALID` and produces no measurement. This resolves the otherwise-contradictory 5° lock ceiling and 5–10° grade tier.

Motion and residual gates intentionally allow normal hand tremor. `STABILIZING` is satisfied by low movement and a compact cluster over the required duration, not by identical digits. If motion stays too large, continue `STABILIZING` or reject — never manufacture stability by smoothing.

---

## 19. Uncertainty composition

```text
inputs     = accepted provider headings passing every per-sample check
candidate  = circularMeanDeg(inputs)                 // uniform weights, §15
residual_i = absoluteCircularDifferenceDeg(input_i, candidate)

providerReportedBoundTermDeg
                    = max valid provider error term in the accepted window when one exists;
                      ABSENT for provider/mode paths exposing no degree error
sampleBound95Deg    = circularResidualQuantileDeg(residuals, 0.95)
deviceFloor95Deg    = certified floor for the exact certification key,
                      else unknownDeviceFloor95Deg
baseHeadingBound95Deg
                    = max(all PRESENT values among providerReportedBoundTermDeg,
                          sampleBound95Deg, deviceFloor95Deg)

deviationCorrectionResidualBound95Deg = 0 when state is NONE,
                                        else the certified profile's held-out residual
declinationModelBound95Deg   = boundFromSigma(model.declinationSigma1Deg)
                               // only when the app performs/may need the conversion
locationTimeSensitivityBound95Deg
    = worst declination change over accepted position/altitude/time uncertainty
referenceAmbiguityBound95Deg = from ReferenceResolutionResult; 0 when verified
interferenceBound95Deg       = 0 when CLEAN
                             = suspectInterferenceBound95Deg when SUSPECT
                             = rejection when UNKNOWN / DISTURBED / INVALID
placementBound95Deg          = configured freehand bound for the mode, or a
                               repeatability-tested alignment-method bound

instrumentBound95Deg = min(180, baseHeadingBound95Deg
                                + declinationModelBound95Deg
                                + locationTimeSensitivityBound95Deg
                                + referenceAmbiguityBound95Deg
                                + deviationCorrectionResidualBound95Deg
                                + interferenceBound95Deg)

reportedBound95Deg   = min(180, instrumentBound95Deg + placementBound95Deg)

gradeLimitedBy:
  if a non-numeric policy ceiling lowers the grade, use fixed precedence
    CERTIFICATION_CEILING -> SPACE_WEATHER -> CHARGING_STATE
  otherwise use the largest numeric uncertainty term; exact ties use stable enum order
```

A missing provider error is **absent, never `0°` evidence**. `AND-RV` may expose a rotation-vector heading-accuracy term in event `values[4]`, whose AOSP-documented semantics are a 95% bound and which therefore enters `providerReportedBoundTermDeg` unconverted; when it is `-1`/unavailable the term is absent. iOS wall and Google FOP wall expose no documented provider degree-error term for their outward-normal projection. In particular, FOP's display-top scalar heading error MUST NOT enter a wall bound, gate, or certification key as though it covered the wall normal. On those wall paths the exact provider/mode device floor and held-out coverage carry the missing evidence. No provider term is silently relabelled as 95% unless its documented semantics and physical axis justify that label.

The asymmetry is deliberate: the three base terms combine with `max` because they estimate the *same* quantity; the rest add because they are *different, additive* error sources. This is a modelling choice, not a derivation, and is among the things held-out coverage tests.

`instrumentBound95Deg` says how well the pipeline knows where the device axis points, and is what §30.1 optimizes. `reportedBound95Deg` says how well the app knows where the *building plane* points, and is what the practitioner sees, what drives classification, and what determines the grade. **Never display `instrumentBound95Deg` as the measurement uncertainty** — it omits the largest term.

`sampleBound95Deg` is a **dispersion floor**, not an error estimate: it detects an unsteady hold and can never detect a steady wrong answer. Do not average provider errors as if independent samples reduced systematic magnetic error by `1/sqrt(n)` — repeated readings from one magnetometer are strongly correlated, and a long window reduces display noise, not unknown environmental bias.

### 19.1 CANDIDATE vs CALIBRATED

Summing several nominally-95% terms does **not** yield a 95% interval. Depending on correlation the result may be conservative or optimistic, and which cannot be reasoned out — it must be measured on held-out devices, sessions, and sites (§32.2). Until that evidence exists for the exact device class, provider path, mode, and config hash, `boundCalibrationState` is `CANDIDATE` and:

- Telemetry keeps the numeric fields unchanged so pre- and post-certification runs stay comparable. The **state field**, not a renamed field, records the distinction.
- UI MUST say **"estimated error bound"**. It MUST NOT display "95% confidence", "95% certain", or any phrasing asserting demonstrated coverage.
- A provisional internal `QualityGrade` MAY be computed for gating, but a `CANDIDATE` consumer result MUST NOT show a standalone `PROFESSIONAL`/`HIGH`/`USABLE` badge. Show `CANDIDATE`, the bound, and plain-language limiting evidence.
- Marketing, store listings, and exported reports inherit the restriction.
- `boundCalibrationState` is **derived at runtime from a §24 certification lookup**, never read from config. A hit yields the record's state; a miss on any key component yields `CANDIDATE`. There is no invalidation step to forget — changing model, config, provider path, mode, or placement changes the key and therefore misses.
- `precision-profile-v1.json` MUST NOT contain a calibration-state property at all, enforced by schema constraint plus test (§8.1).

Two invariants hold on every emitted result and MUST be asserted in tests:

```text
CALIBRATED  <=> uncertaintyCoverageEvidenceState == EMPIRICALLY_CALIBRATED
CANDIDATE    => uncertaintyCoverageEvidenceState in {TARGET_ONLY, UNDEFINED}
```

These two fields are near-redundant by design — one is the gate, the other the claim — and the redundancy is safe only while the invariant holds, because drift lets a `95%` label appear on a `CANDIDATE` measurement.

### 19.2 Sigma is not a bound

NOAA states declination uncertainty as **one standard deviation**; WMMHR2025 publishes its own values. Neither is a 95% interval, and neither is an empirical bound on *this app's* declination error. Adding the published number to a sum of nominal 95% terms under-covers by roughly a factor of two.

The wrapper returns `GeomagneticModelUncertainty` with `sourceConfidenceLevel = ONE_STANDARD_DEVIATION` (§10.3). Conversion happens once, in one named function:

```text
boundFromSigma(sigma1) = declinationSigmaToBound95Factor * sigma1
```

The candidate factor `1.96` is the Gaussian two-sided 95% multiplier. **This is a modelling assumption, not a property of the model** — NOAA's error model describes spatially varying unmodelled crustal field, not obviously Gaussian at a point. It is versioned configuration subject to the same coverage validation as every other term, MUST NOT appear as an unexplained literal, and MUST NOT be silently reused when the model changes.

### 19.3 Deviation correction

Default production state is `NONE`, `deviationCorrectionDeg = 0.0`, `trueHeadingDeg = uncorrectedTrueHeadingDeg`.

A profile may enter an **experimental** arm only when all hold: residual measured against external true-azimuth ground truth (provider agreement or another phone is insufficient); profile circular, deterministic, versioned, hashed; scope explicit (unit or model class, provider path, mode, placement, OS/provider range, model/config hashes, orientation convention); training separated from held-out by unit/session/site; raw uncorrected heading retained beside the correction; evaluated for benefit **and harm** at matched acceptance rate including severe false accepts; site-dependent or transient patterns classified as interference, never promoted to a portable correction.

If certified, apply **exactly once**, after reference resolution and before lock aggregation:

```text
correctionDeg  = profile.evaluate(uncorrectedTrueHeadingDeg)
trueHeadingDeg = normalize360(uncorrectedTrueHeadingDeg + correctionDeg)
```

The profile's held-out residual — not the smoothness of the curve — enters the budget. A profile repeatable on one unit but not across units is `UNIT` scope and stays experimental: v1's certification database does not bind to physical-unit identity, so a `UNIT` profile MUST NOT produce `CALIBRATED` output. Model-class claims need multi-unit evidence. A correction that changes materially after recalibration, travel to another clean site, OS/provider update, repair, or accessory change is not a stable model-class property. Adoption gates: §30.6.

---

## 20. Quality grades

Grades come from `reportedBound95Deg`. The engine MAY compute a provisional grade while `CANDIDATE`; consumer UI MUST withhold the label until the exact key is `CALIBRATED`. Grading on `instrumentBound95Deg` would advertise precision the practitioner cannot physically realize.

| Grade | `reportedBound95Deg` | Also required |
|---|---:|---|
| Professional | `<= 2°` | Clean field, calibrated, stable, fresh location, cross-check pass, device certified, not wireless charging, **repeatability-tested alignment method**. |
| High | `> 2°` and `<= 3°` | Same core gates plus a documented alignment method. |
| Usable | `> 3°` and `<= 5°` | No severe interference; limitation visible. Widest bound that can be a lock. Freehand flat reaches this **only on a certified device whose floor leaves a `2.0°` instrument budget**; on an uncertified device it cannot (§8.1.1). |
| Low confidence | `> 5°` and `<= 10°` | `DEGRADED` only; never a lock, never precision-styled, no single-sector claim. Explicit save only. Freehand wall normally lands here. |
| Invalid | `> 10°`, unknown bound, invalid provider, severe anomaly, stale prerequisites, failed cross-check | No measurement. |

Ranges are explicit half-open intervals so the grade function is total.

**Freehand cannot reach Professional or High, intentionally.** With candidate values `flatFreehandPlacementBound95Deg = 3.0` and `wallFreehandPlacementBound95Deg = 5.0`, the placement term alone meets or exceeds those tiers before any sensor error counts. The top grades require a non-magnetic alignment jig or another technique whose repeatability has been measured on the device/operator combination in use. An implementation reaching Professional freehand has dropped or falsified the placement term — a certification failure, not a feature. §8.1 encodes this as a build-time assertion.

Product consequence: a Professional tier requires an alignment accessory or validated technique, and that work belongs in the plan alongside the sensor work. Further magnetometer improvement while placement stays freehand cannot move the grade.

**Coverage requirements before the `95%` label is earned:** overall empirical coverage `>= 95%` against a prespecified CI criterion; audited by device model, environment, tilt bin, grade, provider path, mode, and placement — a bound covering well in flat-jig conditions and failing in freehand-wall has not passed. Audit `instrumentBound95Deg` against **jig-placed** error and `reportedBound95Deg` against **freehand** error; auditing the total against jig data credits the placement term for an error that was experimentally removed. Median `instrumentBound95Deg` SHOULD stay `<= 3°` and median `reportedBound95Deg` `<= 5°` in clean certified conditions, or coverage is achieved trivially through unusably wide intervals. Any high-grade locked reading with actual error `> 10°` is a severe false-confidence incident requiring root-cause review.

---

## 21. Feng Shui direction engine

Consumes a full-precision canonical heading and its bound; outputs cardinal direction, group/trigram, sector, signed boundary offset, and straddle set. Never round before classification. Boundaries are half-open `[start, end)` in increasing azimuth, tested with epsilon on both sides. The dial's visual rotation MUST use the same canonical reference and orientation mapping as the numeric value.

### 21.1 The versioned ruleset

`config/feng-shui-rules-v1.json` is a required, schema-validated, hashed artifact — not constants in the classifier. Its version and SHA-256 appear in every measurement, because a practitioner disputing a result needs to know which convention produced it, and a ruleset edit is a behavioural change that must trip regression tests. The block below is the **complete required v1 artifact**, not an abbreviated example: `sectors` MUST contain exactly 24 unique indices and `groups` exactly 8 unique trigrams. Ellipses, omitted entries, or an `excerpt` marker are schema errors and cannot ship.

```json
{
  "schemaVersion": "1.0.0",
  "ruleSetVersion": "fengshui-v1",
  "ruleSetName": "24 Mountains, zheng zhen",
  "referenceSelection": "TRUE",
  "needleOffsetDeg": 0.0,
  "sectorCount": 24,
  "sectorWidthDeg": 15.0,
  "firstSectorCenterDeg": 0.0,
  "sectors": [
    { "index": 0,  "centerDeg": 0.0,   "name": "zi",   "glyph": "子", "group": "KAN",  "groupGlyph": "坎" },
    { "index": 1,  "centerDeg": 15.0,  "name": "gui",  "glyph": "癸", "group": "KAN",  "groupGlyph": "坎" },
    { "index": 2,  "centerDeg": 30.0,  "name": "chou", "glyph": "丑", "group": "GEN",  "groupGlyph": "艮" },
    { "index": 3,  "centerDeg": 45.0,  "name": "gen",  "glyph": "艮", "group": "GEN",  "groupGlyph": "艮" },
    { "index": 4,  "centerDeg": 60.0,  "name": "yin",  "glyph": "寅", "group": "GEN",  "groupGlyph": "艮" },
    { "index": 5,  "centerDeg": 75.0,  "name": "jia",  "glyph": "甲", "group": "ZHEN", "groupGlyph": "震" },
    { "index": 6,  "centerDeg": 90.0,  "name": "mao",  "glyph": "卯", "group": "ZHEN", "groupGlyph": "震" },
    { "index": 7,  "centerDeg": 105.0, "name": "yi",   "glyph": "乙", "group": "ZHEN", "groupGlyph": "震" },
    { "index": 8,  "centerDeg": 120.0, "name": "chen", "glyph": "辰", "group": "XUN",  "groupGlyph": "巽" },
    { "index": 9,  "centerDeg": 135.0, "name": "xun",  "glyph": "巽", "group": "XUN",  "groupGlyph": "巽" },
    { "index": 10, "centerDeg": 150.0, "name": "si",   "glyph": "巳", "group": "XUN",  "groupGlyph": "巽" },
    { "index": 11, "centerDeg": 165.0, "name": "bing", "glyph": "丙", "group": "LI",   "groupGlyph": "離" },
    { "index": 12, "centerDeg": 180.0, "name": "wu",   "glyph": "午", "group": "LI",   "groupGlyph": "離" },
    { "index": 13, "centerDeg": 195.0, "name": "ding", "glyph": "丁", "group": "LI",   "groupGlyph": "離" },
    { "index": 14, "centerDeg": 210.0, "name": "wei",  "glyph": "未", "group": "KUN",  "groupGlyph": "坤" },
    { "index": 15, "centerDeg": 225.0, "name": "kun",  "glyph": "坤", "group": "KUN",  "groupGlyph": "坤" },
    { "index": 16, "centerDeg": 240.0, "name": "shen", "glyph": "申", "group": "KUN",  "groupGlyph": "坤" },
    { "index": 17, "centerDeg": 255.0, "name": "geng", "glyph": "庚", "group": "DUI",  "groupGlyph": "兌" },
    { "index": 18, "centerDeg": 270.0, "name": "you",  "glyph": "酉", "group": "DUI",  "groupGlyph": "兌" },
    { "index": 19, "centerDeg": 285.0, "name": "xin",  "glyph": "辛", "group": "DUI",  "groupGlyph": "兌" },
    { "index": 20, "centerDeg": 300.0, "name": "xu",   "glyph": "戌", "group": "QIAN", "groupGlyph": "乾" },
    { "index": 21, "centerDeg": 315.0, "name": "qian", "glyph": "乾", "group": "QIAN", "groupGlyph": "乾" },
    { "index": 22, "centerDeg": 330.0, "name": "hai",  "glyph": "亥", "group": "QIAN", "groupGlyph": "乾" },
    { "index": 23, "centerDeg": 345.0, "name": "ren",  "glyph": "壬", "group": "KAN",  "groupGlyph": "坎" }
  ],
  "groups": [
    { "name": "KAN",  "glyph": "坎", "cardinal": "N",  "centerDeg": 0.0,   "widthDeg": 45.0 },
    { "name": "GEN",  "glyph": "艮", "cardinal": "NE", "centerDeg": 45.0,  "widthDeg": 45.0 },
    { "name": "ZHEN", "glyph": "震", "cardinal": "E",  "centerDeg": 90.0,  "widthDeg": 45.0 },
    { "name": "XUN",  "glyph": "巽", "cardinal": "SE", "centerDeg": 135.0, "widthDeg": 45.0 },
    { "name": "LI",   "glyph": "離", "cardinal": "S",  "centerDeg": 180.0, "widthDeg": 45.0 },
    { "name": "KUN",  "glyph": "坤", "cardinal": "SW", "centerDeg": 225.0, "widthDeg": 45.0 },
    { "name": "DUI",  "glyph": "兌", "cardinal": "W",  "centerDeg": 270.0, "widthDeg": 45.0 },
    { "name": "QIAN", "glyph": "乾", "cardinal": "NW", "centerDeg": 315.0, "widthDeg": 45.0 }
  ]
}
```

Geometry is derived, never hand-typed as a boundary list:

```text
sectorIndex(h) = floor(normalize360(h - firstSectorCenterDeg + sectorWidthDeg/2)
                       / sectorWidthDeg) mod sectorCount
```

For the default ruleset this puts boundaries at `7.5° + 15k`, so `352.5°` separates 壬 and 子. A schema test MUST assert exact array cardinalities, unique/contiguous sector indices `0...23`, unique names/glyphs, group references that resolve, `sectorCount * sectorWidthDeg == 360`, and that each declared `centerDeg` equals `normalize360(firstSectorCenterDeg + index * sectorWidthDeg)`. An internally inconsistent or abbreviated ruleset fails the build rather than misclassifying quietly.

Golden fixtures in `testdata/fengshui/` MUST cover, per sector: centre, both boundaries, boundaries `± epsilon`, `± 0.1°`, `± 1.0°`, plus the north-wrap sector.

### 21.2 North reference and needle convention

A domain-correctness issue, not a preference: getting it wrong invalidates results for an entire school.

The canonical pipeline resolves **true north** because true north is what can be verified against survey ground truth — magnetic north cannot be verified without trusting the magnetometer under test. That is a validation choice, not doctrine. Many traditions measure with a magnetic Luo Pan, and some use plates at a constant offset. The engine applies the ruleset's reference and offset as a final, explicit, recorded step:

```text
classificationHeadingDeg = normalize360(
    (referenceSelection == TRUE     ? trueHeadingDeg
   : referenceSelection == MAGNETIC ? normalize360(trueHeadingDeg - declinationDeg)
   : error)
  + needleOffsetDeg)
```

- `referenceSelection` and `needleOffsetDeg` come from the ruleset and are stored with every result. A saved record whose reference is unknown is uninterpretable later.
- `needleOffsetDeg` expresses doctrinal plate conventions (the `±7.5°` conventions among them). It is a declared property of a named ruleset — never a user slider, never a correction for measurement error. Confusing a plate offset with an instrument correction would let a practitioner "fix" a bad compass by choosing a different school.
- A magnetic ruleset relaxes **no** true-north gate: the pipeline still resolves or explicitly ambiguity-bounds the true-reference point estimate and records `resolvedReference`; magnetic classification is derived from that same canonical measurement rather than substituting an unvalidated magnetic path.
- `TRUE_WITH_AMBIGUITY_BOUND` remains safe under either ruleset **only if the ambiguity term is retained after the reference transform**. Let `g` be Google output and `d` declination. If Google secretly emitted true north, TRUE uses `g` exactly and MAGNETIC uses `g-d` exactly. If Google secretly emitted magnetic north, the TRUE point `g` is wrong by `|d|`, and the derived MAGNETIC point `g-d` is also wrong by `|d|`; `referenceAmbiguityBound95Deg = |d|` covers either hidden hypothesis. Therefore subtracting `d` for a magnetic ruleset MUST NOT zero or remove the ambiguity term.
- Golden tests MUST enumerate both hidden Google hypotheses, positive and negative declination, TRUE and MAGNETIC rulesets, a sector boundary, and north wrap; sector straddle uses the transformed point estimate with the unchanged **total** reported bound.
- Switching rulesets invalidates the displayed classification and re-classifies the stored canonical measurement. It never requires re-measurement because the stored true-reference point estimate, declination, resolved-reference state, and bound contain the information needed for either v1 ruleset reference.
- v1 ships one ruleset. Others arrive as new hashed files with their own fixtures, not branches in the classifier.

### 21.3 Sector width sets hard limits

A sector is `15°` wide, so:

- `reportedBound95Deg > 7.5°` **guarantees** a two-sector straddle regardless of the point estimate.
- `> 15°` guarantees at least three.
- A `LOW_CONFIDENCE` result (up to `10°`) therefore has essentially no discriminating power; showing it beside a single mountain glyph is misleading even with a caveat.
- A `USABLE` result at `5°` straddles whenever the estimate is within `5°` of a boundary — two thirds of the sector width. **Straddles are the common case**, so the straddle presentation is a primary layout, not an error state.

Single-mountain answers arrive with any regularity only at the top tiers with a repeatability-tested placement method. The interface should embody that.

### 21.4 Boundary straddle

Classify the whole circular bound interval, not the point estimate. Semantics do not depend on calibration state — a `CANDIDATE` bound still straddles, because conservative ambiguity beats naming a false mountain — only the confidence wording differs.

```text
interval        = [classificationHeadingDeg - reportedBound95Deg,
                   classificationHeadingDeg + reportedBound95Deg]
possibleSectors = every ruleset sector intersecting that circular interval
```

- One sector → display normally.
- Two or more → `boundaryStraddled = true`, store all in azimuth order, show **all** prominently.
- Never pick one because the estimate lies a fraction of a degree to one side.
- Split intervals wrapping `0°/360°` correctly.
- More than two sectors → show the full set plus a low-specificity warning.
- `2 * reportedBound95Deg >= 360°` → report that no classification is possible rather than listing all 24.
- Measurement grade and classification certainty are separate: a low-noise heading can still straddle because placement and reference uncertainty remain.

Release-blocking; golden tests for every boundary, both sides, exact equality, wide bounds, north wrap.

### 21.5 Display

```text
182.4° TRUE
estimated error bound: ±7.0°
  instrument ±4.0° + freehand placement ±3.0°
SOUTH / selected Luo Pan classification
24 Mountains: 午 / 丁 (interval straddles the 187.5° boundary)

Location: age 0.8 s, precise      Magnetic field: clean
Space weather: Kp 3.7, quiet      Calibration: high
Device level: ±0.7°               Mode: flat, top edge · freehand
Ruleset: fengshui-v1 (true north, no needle offset)
Sensor Health: live conditions acceptable; device/bound evidence CANDIDATE
DEGRADED · LOW CONFIDENCE · CANDIDATE
No Precision Lock: the uncertified-device floor plus freehand placement exceeds 5°.
```

Never show more decimal places than the bound supports. The display MUST show the **total** bound as the headline number and MAY show the instrument/placement split beneath it. Showing only the instrument term is the specific dishonesty this rule prevents — it is the number that makes a phone look like a survey instrument and is not the number describing the measurement just taken. When placement caps the grade, say so via `gradeLimitedBy`; that tells the user what would actually improve the result.

The live unlocked heading may be shown continuously but MUST be visually distinct from a lock and MUST NOT carry a bound, grade, or sector claim. An expandable **Sensor Health / Measurement Diagnostics** view exposes calibration state, magnetic state, provider error, transform and pipeline disagreement, pose, freshness, certification evidence, correction state, and the exact limiting reason, with `Check / Recalibrate` reachable from it.

---

## 22. Telemetry

Append-only JSON Lines. Every event has a common envelope plus a typed payload.

```json
{
  "schemaVersion": "1.0.0",
  "sessionId": "uuid",
  "eventId": 123456,
  "eventType": "engine_output",
  "platform": "ANDROID",
  "appVersion": "1.4.0", "appBuild": "2401",
  "engineVersion": "heading-3.2.0",
  "configVersion": "precision-v1-candidate-1", "configHash": "sha256:...",
  "deviceAnonymousId": "salted-hash",
  "hardwareRuntimeIdentity": "ANDROID:example-runtime-tuple",
  "sensorRuntimeIdentity": "sha256:...", "osBuildIdentity": "example-exact-build",
  "wallTimeUtc": "2026-08-29T12:34:56.123456Z",
  "recordMonotonicTimeNs": 1234567890123,
  "sourceMonotonicTimeNs": 1234567889000,
  "arrivalMonotonicTimeNs": 1234567890100,
  "sourceClock": "ELAPSED_REALTIME",
  "clockMappingId": "IDENTITY",
  "sequence": 42,
  "payload": {}
}
```

Three canonical monotonic timestamps, three meanings, never interchangeable: `sourceMonotonicTimeNs` is the provider occurrence time **mapped into the app's monotonic domain**; `arrivalMonotonicTimeNs` is callback delivery; `recordMonotonicTimeNs` is record creation. **Freshness is always computed from mapped source time, never arrival.** Preserve the provider's raw timestamp in its typed payload and identify the original clock with `sourceClock` ∈ `ELAPSED_REALTIME | CORE_MOTION_BOOT_TIME | PROVIDER_DATE | FIXTURE_CLOCK`. `clockMappingId` identifies the logged mapping; Android elapsed realtime may use `IDENTITY`. Missing/invalid mapping makes the event ineligible for freshness-sensitive decisions rather than falling back to callback time.

**Event types** (`lower_snake_case` identifiers, a separate namespace from enum values): `session_start`, `session_end`, `app_lifecycle`; `clock_mapping`, `ground_truth`, `fixture_state`; `location_sample`, `location_authorization`, `location_provider_state`; `magnetometer_calibrated`, `magnetometer_uncalibrated`; `accelerometer`, `gravity`, `gyroscope`, `rotation_vector`, `device_motion`; `os_heading`, `fused_orientation`; `capability_resolution`; `wmm_output`, `reference_resolution`, `engine_output`, `state_transition`, `precision_lock`; `sensor_health`; `calibration_request`, `calibration_prompt`, `calibration_result`; `target_heading_request`, `target_guidance`; `deviation_profile_lookup`, `deviation_correction`; `certification_lookup`; `display_frame_marker`; `thermal_state`, `battery_state`, `charging_state`, `power_mode`; `space_weather_advisory`; `orientation_change`, `sensor_discontinuity`, `dropped_sample_summary`; `user_action`.

### 22.1 `engine_output` example

Deliberately a good-looking measurement that is still **uncertified**: fresh confident provider, clean field, verified reference, and a level device. Under the shipped constants that is still **not a Precision Lock**: `unknownDeviceFloor95Deg = 4.0°` plus `3.0°` flat-freehand placement produces a `7.0°` `LOW_CONFIDENCE` degraded result. The example is intentionally reachability-consistent with §8.1.1; a candidate state does not waive the floor.

```json
{
  "referenceAxis": "PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION",
  "attitudeEarthFrame": "REFERENCE_ENU",
  "canonicalHeadingReference": "TRUE",
  "implementationVariant": "AND-G1-50",
  "providerId": "GOOGLE_FOP", "providerErrorSource": "GOOGLE_CONSERVATIVE",
  "providerRuntimeIdentity": "GMS:example-exact-version",
  "locationProviderId": "GOOGLE_FLP",
  "locationProviderRuntimeIdentity": "GMS:example-exact-version",
  "hardwareRuntimeIdentity": "ANDROID:example-runtime-tuple",
  "sensorRuntimeIdentity": "sha256:...", "osBuildIdentity": "example-exact-build",
  "engineDecisionLogicHash": "sha256:...",
  "requestedOrientationPeriodUs": 20000, "observedOrientationRateHz": 48.7,
  "providerReferenceContract": "TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC",
  "resolvedReference": "TRUE_VERIFIED",
  "referenceResolutionMethod": "FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK",
  "referenceHypothesisResidualTrueDeg": 0.41,
  "referenceHypothesisResidualMagneticDeg": 7.88,
  "referenceAmbiguityBound95Deg": 0.0,
  "providerHeadingDeg": 189.00,
  "magneticHeadingDeg": 181.12, "magneticHeadingSource": "ANDROID_ACCEL_MAG",
  "uncorrectedTrueHeadingDeg": 189.00,
  "deviationCorrectionState": "NONE", "deviationCorrectionProfileId": null,
  "deviationCorrectionProfileHash": "NONE", "deviationCorrectionDeg": 0.0,
  "trueHeadingDeg": 189.00, "declinationDeg": 8.29,
  "geomagneticModelId": "WMM2025",
  "declinationCoefficientSha256": "...", "declinationErrorModelSha256": "...",
  "declinationSigma1Deg": 0.36, "declinationEnvelopeDeg": 0.03,
  "horizontalIntensityNanoTesla": 21450.0,
  "altitudeReference": "WGS84_ELLIPSOID",
  "locationAuthorizationAccuracy": "PRECISE_FULL",
  "spaceWeather": { "kp": 3.67, "observationTimeUtc": "2026-08-29T12:00:00Z",
                    "cacheAgeMs": 930000, "state": "QUIET" },
  "instrumentBound95Deg": 4.00, "placementBound95Deg": 3.00,
  "reportedBound95Deg": 7.00,
  "uncertaintyCoverageTarget": 0.95,
  "uncertaintyCoverageEvidenceState": "TARGET_ONLY",
  "boundCalibrationState": "CANDIDATE",
  "trustAction": "SHOW_DEGRADED_RESULT",
  "provisionalQualityGrade": "LOW_CONFIDENCE", "displayQualityGrade": null,
  "gradeLimitedBy": "DEVICE_FLOOR",
  "osHeadingErrorDeg": 1.50, "conservativeHeadingErrorDeg": 1.80,
  "sampleCircularDispersionDeg": 0.62, "circularResultantLength": 0.99987,
  "transformAgreementDeg": 0.11, "pipelineAgreementDeg": 0.54,
  "pipelineSet": ["GOOGLE_FOP", "ANDROID_ROTATION_VECTOR", "ANDROID_ACCEL_MAG"],
  "pitchDeg": 0.40, "rollDeg": -0.80,
  "fieldMagnitudeMicroTesla": 48.70, "expectedFieldMagnitudeMicroTesla": 47.90,
  "relativeMagnitudeResidual": 0.0167,
  "measuredInclinationPositiveDownDeg": 64.2,
  "expectedInclinationPositiveDownDeg": 63.3, "inclinationResidualDeg": 0.9,
  "stationaryFieldMadMicroTesla": 0.31,
  "referenceMagneticPrecheckState": "CLEAN_FOR_REFERENCE",
  "magneticState": "CLEAN", "chargingState": "NOT_CHARGING", "thermalState": "NOMINAL",
  "measurementMode": "FLAT_TOP_EDGE", "placementMethod": "FREEHAND",
  "placementProfileId": "flat-freehand-v1", "placementProfileHash": "sha256:...",
  "measurementState": "DEGRADED", "rejectionReasons": [],
  "fengShuiRuleSetVersion": "fengshui-v1", "fengShuiReferenceSelection": "TRUE",
  "primaryFengShuiSector": "ding", "possibleFengShuiSectors": ["wu", "ding"],
  "boundaryStraddled": true,
  "oldestInputAgeMs": 18.2, "locationAgeMs": 820,
  "aggregationDurationMs": 3200,
  "effectiveHeadingSampleCount": 146, "periodicSupportSampleCount": 146
}
```

### 22.2 Encoding rules

Cross-platform JSON differences are a recurring source of silent corruption, and exports that parse differently on two platforms cannot be pooled.

- **Casing.** Enum values `UPPER_SNAKE_CASE`; property keys `lowerCamelCase`; event-type identifiers `lower_snake_case`. No exceptions, including fixtures.
- **Nonfinite literals forbidden.** JSON has no `NaN`/`Infinity`. Unavailable → `null` plus a sibling status field. Encoders MUST fail rather than emit nonstandard literals; decoders MUST reject them.
- **Locale independence.** `.` decimal separator, no digit grouping, no localized number/date formatting, regardless of device locale. A test MUST run the export path under a comma-decimal locale.
- **Precision.** Doubles serialized with shortest round-trip (or 17 significant digits). Never serialize a `Double` through a `Float`.
- **Timestamps.** Wall clock is RFC 3339 UTC with explicit `Z`; monotonic is integer nanoseconds with a named clock domain. Never mixed in one field.
- **Units in names.** Numeric field names end with their unit (`Deg`, `Ms`, `Ns`, `MicroTesla`, `NanoTesla`, `M`, `Hz`, `G`) unless dimensionless and documented as such.

### 22.3 Privacy

Raw location is sensitive: obtain consent, minimize retention, keep lab telemetry separate from consumer analytics. Production export MUST be opt-in and MUST redact or quantize coordinates when exact values are unnecessary. **Consumer** builds MUST hash device identifiers with a *rotating* project salt and MUST NOT log advertising identifiers; **benchmark** builds MUST use a *fixed archived* salt so per-unit longitudinal analysis remains possible (§28). These serve different purposes and MUST be separate code paths, not one path with a flag that could ship in the wrong position. Sign manifests and hash raw files so analysis can prove inputs were not edited. Keep test labels separate until blind analysis completes where practical.

---

## 23. Mocking policy

Mocks and fakes are permitted for deterministic unit and UI tests, but: simulator/emulator sensor output can never satisfy physical accuracy acceptance; mock location MUST be marked in telemetry; replay data MUST preserve original timestamps and frames; a fake provider MUST NOT compile into a production release path without an explicit debug flag; UI previews may use synthetic headings but must visibly identify preview data.

---

## 24. Certification database

**Authoritative for the certification schema.** Every lookup elsewhere — the `boundCalibrationState` derivation and the device-floor term in §19 — refers here rather than restating the key. Two platform agents inventing two lookup schemas is a realistic failure mode.

```text
CertificationKey
  certificationSchemaVersion
  hardwareRuntimeIdentity      // deterministic public runtime tuple; platform-defined
  sensorRuntimeIdentity        // public runtime descriptor hash, or NOT_RUNTIME_OBSERVABLE
  osBuildIdentity              // exact build, never an open-ended version range
  providerId, providerRuntimeIdentity, providerErrorSource
  locationProviderId, locationProviderRuntimeIdentity
  measurementMode, placementMethod, placementProfileHash
  geomagneticModelId, geomagneticCoefficientHash, geomagneticErrorModelHash
  deviationCorrectionProfileHash    // literal NONE when correction is disabled
  engineDecisionLogicHash, precisionConfigHash

CertificationRecord
  key, boundCalibrationState: CALIBRATED
  deviceFloor95Deg
  supportedQualityGrade        // a ceiling, not a promise
  earnedUnderEngineVersion
  evidenceReportId, certificationDate
```

- A record exists **only** for `CALIBRATED`. Absence already means `CANDIDATE`; storing both invites writing a `CANDIDATE` record and editing its state field.
- Every key field MUST be derivable deterministically in the production process from a public runtime value or an app-bundled artifact hash. Lab-only facts such as sales-region SKU when the OS does not expose it, unit serial, repair history, device age, and operator name belong in the evidence inventory, **not** in the runtime lookup key. Never guess a missing value from locale, storefront, model naming, or benchmark notes.
- `hardwareRuntimeIdentity` is a versioned platform tuple (for example, the documented Android build identity fields selected by the project, or Apple hardware-machine identifier). `sensorRuntimeIdentity` hashes only public sensor descriptors actually obtainable in the production process. Where a platform does not expose physical sensor identity, use the literal `NOT_RUNTIME_OBSERVABLE`; certification evidence MUST then cover every known hardware/sensor variant sharing the remaining runtime identity, use a conservative worst-case floor, or issue no record. A hidden component swap cannot be made safe by inventing a signature.
- Lookup is exact on every component. `osBuildIdentity`, `providerRuntimeIdentity`, and `locationProviderRuntimeIdentity` are exact observed identities, never semantic or open-ended ranges that silently admit a future release. For Google paths, record the observable Play services package identity. For OS-owned Android or Apple providers without a separate public provider version, use `OS_BUILD:<osBuildIdentity>` rather than `UNKNOWN` or a marketing version. Evidence that covers several exact builds generates several records pointing to the same report.
- `engineDecisionLogicHash` covers the executable core code and generated gate/composition tables that can change a lock or bound. It is separate from human-readable `engineVersion`; changing behavior without changing config must miss the old record. `certificationSchemaVersion` prevents a newer client from reinterpreting an old tuple.
- Switching Google conservative↔ordinary↔`NONE` error semantics, changing FLP↔framework location provenance, changing the geomagnetic error-model artifact, changing engine decision logic, or changing the placement profile therefore changes the key.
- `placementProfileHash` identifies the actual measured placement contract, not merely the enum. A certified non-magnetic jig does not authorize another jig/operator procedure that also happens to be `NONMAGNETIC_ALIGNMENT_JIG`. Built-in freehand profiles are hashed artifacts/config subsets too.
- `geomagneticErrorModelHash` is separate from the coefficient hash because the error model changes the reported uncertainty even when coefficient evaluation is unchanged.
- In v1, a non-`NONE` `deviationCorrectionProfileHash` in a `CALIBRATED` record MUST identify a `MODEL_CLASS` profile that passed §30.6. Experimental `UNIT` profiles never match a calibrated production record.
- A miss returns nothing: the engine uses `CANDIDATE`, `unknownDeviceFloor95Deg`, and a provisional ceiling no higher than `USABLE`, with the consumer UI still withholding a certified label. Note that the ceiling is an upper limit, not a promise: with the candidate `unknownDeviceFloor95Deg` no freehand grade is arithmetically reachable on a miss at all (§8.1.1).
- `supportedQualityGrade` applies **after** the bound-based grade — a certified device with a poor live measurement still grades poorly.
- `earnedUnderEngineVersion` remains human-readable provenance; it does not replace `engineDecisionLogicHash`. A carry-forward to a byte-different but behavior-identical app is allowed only when the decision-logic hash is unchanged and §33.3 regression evidence is archived. A gate/composition change necessarily changes the hash and misses.
- Generated from benchmark evidence, versioned with the app, not remotely configurable in certification builds. `evidenceReportId` MUST resolve to archived raw telemetry.
- Adding a record is release-gated on §30 and §35 evidence for that exact key. An agent MUST NOT add records to make tests pass.

### 24.1 Live health is not certification

The database says what was demonstrated historically for an exact configuration; the live engine says whether the current measurement behaves acceptably now. Neither replaces the other.

```text
historical device/configuration evidence + live sensor health
  + sensor calibration + magnetic environment
  + reference/location freshness + pose/stability/placement
  -> TrustAction and final bound
```

A certified device behaving badly is downgraded or rejected. An uncertified device behaving well stays `CANDIDATE` — one healthy session cannot establish intrinsic magnetometer quality. Sensor age, brand, tier, and price are not evidence. Repair or component substitution invalidates historical expectations even under an unchanged marketing model. The user-visible summary answers *what action improves this measurement*, not raw metrics without interpretation.

---

## 25. Benchmark objectives

Answer: accuracy in clean conditions; repeatability across runs/headings/tilt/time/place/devices; how well the reported bound covers actual error; time to usable, tracking, recovery; interference rejection vs false alarms; whether calibration measurably helps; true-north correctness across declination/date/altitude/permission modes; accuracy/latency/battery/thermal under sustained use; which models certify; whether full-circle residual structure is repeatable and any correction portable; whether target guidance leads users correctly without hiding tremor/overshoot/ambiguity; whether automatic and manual calibration avoid both misses and harmful loops; whether Sensor Health distinguishes ready/calibrate/move-away/hold/initialize/candidate/reject; what freehand placement actually costs per mode and operator; whether a new build regresses.

The benchmark MUST prioritize actual angular error and false confidence over cosmetic stability. Physical devices only — simulators test UI and deterministic math, never sensor accuracy.

## 26. Implementations to compare

An A/B comparison of APIs and the thin app layer, not a contest between home-grown fusions. Log all variants simultaneously where APIs permit, so they share motion and field.

| ID | Platform | Implementation | Purpose |
|---|---|---|---|
| `AND-G0` | Android | Mode-native FOP output, otherwise unchanged: flat = `getHeadingDegrees()`; wall = FOP-attitude outward-normal projection; reference provenance retained | Required Google baseline |
| `AND-GE` | Android | Flat mode only: `AND-G0` accepted only when the FOP display-top scalar error is valid and `<= 5°`; wall = `NOT_APPLICABLE` because Google documents no error bound for the outward-normal attitude projection | Provider-error operating point for §30.1 |
| `AND-G1` | Android | `AND-G0` + freshness/orientation checks, mode-axis reference resolution, magnetic rejection, circular lock, app uncertainty model | Proposed Google production |
| `AND-RV` | Android | `TYPE_ROTATION_VECTOR` + platform transforms + shared NOAA WMM | First-class no-GMS candidate, certified independently |
| `AND-AM` | Android | accel/gravity + magnetometer via `getRotationMatrix`/`getOrientation` | Diagnostic magnetic-reference baseline only |
| `AND-HDG` | Android API 33+ where sensor exists | `TYPE_HEADING` true-north scalar + `boundFromSigma(accuracy68Deg)` for every 95%-bound comparison; raw 68%-confidence accuracy retained separately | Flat-mode diagnostic/current-platform benchmark only; no wall attitude |
| `IOS-A0` | iOS | Mode-native Apple output, otherwise unchanged: flat = valid `CLHeading.trueHeading`; wall = `.xTrueNorthZVertical` outward-normal projection | Required Apple baseline |
| `IOS-AE` | iOS | Flat mode only: valid `trueHeading` with `0 <= headingAccuracy <= 5°`; wall = `NOT_APPLICABLE` because Core Motion exposes no analogous degree-error field | Provider-error operating point |
| `IOS-A1` | iOS | `IOS-A0` + synchronized validation, checks, magnetic rejection, circular lock, app uncertainty model | Proposed iOS production |
| `IOS-CM-FLAT` | iOS | Core Motion `.xTrueNorthZVertical` portrait-top-edge projection in flat pose | Diagnostic cross-check; Core Motion is production in wall mode |

Every variant is compared against the **same external true-azimuth ground truth**. Product questions: does `G1`/`A1` reduce accepted error or tail risk vs the unchanged provider; what acceptance-rate and lock-time cost does gating impose; does the app reduce false acceptance under interference; is its uncertainty better calibrated than provider error alone; which devices can certify the no-GMS path. **The app layer passes only if it improves the accuracy/safety tradeoff on held-out data. Lower jitter alone is not an improvement.**

Deviation correction is an **orthogonal experimental factor**, not a new estimator ID: after §29.3 freezes a profile, compare `NONE` vs `PROFILE_<hash>` on the same candidate, on held-out data. Never fit and score on the same sweep.

Rate variants take suffixes `-50`, `-100`, `-200`. Because a new request replaces the previous listener and rate affects scheduling/power, run them as separate randomized/counterbalanced trials — do not pretend they were simultaneous. Report **observed** callback rate, not only the requested period.

---

## 27. Ground truth

Tier 0 is mandatory before buying equipment; Tier 2 is needed only for final certification.

**Tier 0A — relative rotation, no absolute north.** Non-magnetic turntable with a protractor and repeatable cradle; initial direction may be unknown. Commanded deltas `0 → +5 → +15 → +45 → +90 → +180`, both directions, plus `350 → 10` and `10 → 350` wrap crossings. Compare `shortestSignedDifferenceDeg(after, before)` with the mechanical delta. Tests scale, sign, wrap, hysteresis, lag, over-smoothing, tilt coupling, screen-axis mistakes. Does **not** test absolute bias or reference. Outputs: delta MAE/P95/max, CW-vs-CCW hysteresis, step latency, settling, wrap failures. Disproportionately valuable for its cost — a swapped quaternion component, a missing remap, an inverted sign, or a five-second filter all fail obviously here and would otherwise hide inside a merely-mediocre absolute number.

**Tier 0B — solar shadow.** Use the NREL Solar Position Algorithm with pinned source and version; it is the only solar-position source this specification accepts. Do not substitute a general-purpose web calculator — a reference whose provenance and version cannot be pinned cannot enter a reproducibility package. Astronomy libraries are acceptable only if pinned, versioned, and validated against SPA across the latitude/date/elevation range in use, with that validation archived. NREL's algorithmic uncertainty is **not** the benchmark uncertainty; the controlling term is physical.

Protocol: clear sky with the sun high enough for a sharp shadow (avoid refraction-sensitive low elevations); level a non-magnetic board with a plumb rod, measuring verticality; record lat/lon/altitude/UTC/pressure/temperature; at the exact timestamp mark rod base and shadow centre, marking repeatedly rather than trusting one penumbral edge; ideal shadow direction is `normalize360(sunAzimuth + 180°)`; fit the line and transfer it to the fixture, recording the transfer as its own step; archive inputs, SPA version/hash, photographs, raw marks, fitted lines, transfer method, and the budget.

The budget MUST be **empirical**: repeat the entire procedure — re-plumb, re-mark, refit, re-transfer — at least five times, ideally across two days and two sun elevations, and take the observed spread as the dominant term. A propagated budget can be made arbitrarily small by assuming a sharp edge and a perfect rod; the repeat spread cannot, because it contains penumbra, operator, level, and transfer at once. Combine with terms it does not contain (timestamp error, position error, independently measured rod verticality) and report an expanded uncertainty with its coverage factor. Target `<= 0.5°`; if the measured budget is larger, report the larger value and narrow the claims. Do not tighten by discarding the worst repeat, and do not support a device claim whose target error is within a factor of two of the reference.

**Tier 0C — long geodetic baseline.** Two visible points with coordinates and uncertainties from an authoritative source; compute the WGS84 forward azimuth with GeographicLib's inverse geodesic, never a home-grown spherical formula. Screening approximation for roughly isotropic endpoint sigmas: `azimuthStdRad ≈ sqrt(σ1² + σ2²) / L` (strictly the components perpendicular to the baseline, equal to the full sigmas when isotropic). Use a full calculation for the archived result. Longer baselines reduce angular uncertainty. Rendered imagery or cadastral visualization is not survey control — use the published accuracy of the actual coordinate dataset. Transfer to the fixture without magnetic tools, including alignment/parallax uncertainty.

**Tier 1 — characterized developer reference.** Two independent Tier 0 absolute methods, a non-magnetic indexed fixture, repeats across days, total expanded uncertainty `<= 0.5°`. The point is **not** the smaller number (two independent `0.5°` references give ~`0.35°`); it is the **cross-check**. A single method can be wrong by a large amount from one blunder — a mistyped longitude, a transfer line marked from the wrong end, a UTC/local confusion. Two methods disagreeing by `8°` reveal it immediately; no amount of repetition within one method would. Supports implementation selection and pre-release claims, not broad model certification.

**Tier 2 — certification reference.** Either two geodetic control points from dual-frequency survey GNSS with adequate baseline and QC plus total-station transfer, or a calibrated non-magnetic rotary table referenced to a surveyed meridian with documented encoder uncertainty. SHOULD achieve `<= 0.20°`, MUST achieve `<= 0.50°` for `2–5°` product claims. **Per-model bias gates require Tier 2**, because a `2.0°` bias gate against a `0.5°` reference is only four to one and the reference then contributes materially to the gated quantity.

Never use another phone, a car compass, a rendered map north arrow, an unverified wall, or a consumer hand compass as absolute ground truth.

**Budget contents.** Control-point coordinate uncertainty and baseline length; grid-to-true convergence if projected coordinates are used; total-station/theodolite or encoder calibration; fixture index and backlash; device placement and axis alignment repeatability; fixture level and local vertical; reference-line transfer and operator parallax; thermal/mechanical fixture movement. Always report and propagate the achieved value with its coverage factor. Never subtract reference uncertainty from observed phone error.

**Fixture.** Verified non-ferromagnetic low-current materials near the device; repeatable holding without magnets or steel springs near the magnetometer; yaw indexed at least every `15°`, preferably `<= 0.1°` readout; controlled pitch/roll or a two-axis non-magnetic goniometer; face-up **and** upright geometries, since the two modes need both; powered encoders either kept clear or characterized powered and unpowered; marked placement, height, orientation. Scan the empty fixture with a calibrated three-axis instrument at each position and angle first — a rotating fixture that changes the measured field appears in the data as exactly the azimuth-dependent residual §29.3 tries to attribute to the phone.

A calibrated three-axis fluxgate is strongly recommended for site and disturbance characterization but is **not** the azimuth reference. **Site qualification:** spatial grid at device height; ≥10 minutes of field stability; survey of nearby steel, conductors, speakers, vehicles, buried services; baseline with lab equipment on and off; comparison against WMM recognizing it models the main field; clean-site residual limits from measured baseline plus instrument uncertainty.

---

## 28. Device matrix and environments

**iOS:** current flagship and standard model, at least two prior hardware generations, devices with relevant magnetic accessory ecosystems, current and previous supported major iOS. Include one iPad only to verify it correctly reports `UNSUPPORTED_DEVICE` (§2.1).

**Android:** current and previous Pixel; current and previous Samsung flagship plus high-volume midrange; at least one high-volume device per supported OEM/SoC family; low/mid/high price tiers; **at least one genuinely no-GMS distribution** for `AND-RV`; devices whose rotation vector is synthesized from a sensor subset or that report no gyroscope, treated as a separate class; multiple sensor vendors where a marketing model has regional hardware variants; current and previous supported Android including OEM builds and patch levels.

Per-device inventory: manufacturer, marketing/hardware model, region SKU, serial hash, SoC, RAM tier, OS version/build/patch, baseband and Play services version, app/engine/config/schema versions, full sensor signature, device age, repair history, battery health, accessory state.

Use **at least three physical units** per high-volume certified model (five preferred where intra-model variation is suspected). One golden unit is the most common way a certification turns out to have measured one lucky magnetometer.

**Environments:** qualified open outdoor field; characterized indoor non-steel lab; residential interior; reinforced concrete; steel frame; dense urban street; beside and inside a vehicle; elevator lobby and car; near laptop/tablet/speaker/power supply; near current-carrying cable and electrical panel at safe distances; with approved nonmagnetic case; with magnetic case, wallet, mount, charger, ring, MagSafe-class accessories. Environments 3–8 are where practitioners work — accuracy gathered only outdoors describes a product nobody uses.

Record ambient and device temperature, humidity, weather, sky view, solar loading, vibration, nearby moving equipment, radio state, brightness, battery charge, charging state, time since boot. For comparable runs: fixed battery band (e.g. `60–80%`), no charging during magnetic or power tests, fixed brightness with auto-brightness off, controlled network mode, unrelated apps closed, required starting thermal state, and randomized or counterbalanced device/condition order.

---

## 29. Benchmark methodology

### 29.1 Static sweep

Per device unit, mode, and required environment: place without a magnetic case; start from a defined process state; at `0°, 15°, … 345°` wait the acquisition window; record continuously for ≥10 s after the app declares stable, or record the rejection outcome until timeout; repeat each heading ≥10 times across ≥3 sessions and ≥2 days for certification; randomize order in at least half the sessions and run both directions to reveal hysteresis; lift and reseat between selected repetitions to include placement repeatability.

Report **both** all-attempt and accepted-only metrics. Accepted-only accuracy without acceptance rate is misleading — it is the specific way a heavily-gating candidate makes itself look accurate.

### 29.2 Dynamic, boundary, and start-up

- **Boundary sweep:** every sector boundary at `±1.0°`, `±0.1°`, `±epsilon`, and exactly on it. Mathematics via deterministic replay over all 24 boundaries; physical tests on a representative subset including north wrap.
- **Dynamic yaw:** steps `5, 15, 45, 90, 180°` both directions; constant rates `5, 15, 30, 60, 120°/s` where the fixture permits; reversals and `350 → 10`, `10 → 350`; sinusoidal profiles. Measure onset latency, phase delay, 10–90% rise, overshoot, settling, maximum dynamic error, wrap discontinuities.
- **Cold/warm start:** app cold launch after termination; provider cold start after reboot; warm start with recent location; resume after 1, 5, 30 minutes backgrounded; resume after permission changes. Measure time in `PROVIDER_INITIALIZING` **separately** from time to lock — different causes, different fixes. State erasure users cannot reproduce must be labelled diagnostic-only.

### 29.3 Full-azimuth residual-deviation characterization

Makes the **shape** of residual error a mandatory analysis target rather than letting it vanish into pooled P95.

Per device unit, provider path, mode, placement method, and calibration state: collect signed error against external ground truth at least every `15°` around `0–345°` (`45°` is an exploratory subset, not a certification grid); repeat both directions across ≥3 sessions and ≥2 days, and across multiple qualified clean sites for certification claims; re-run after ordinary recalibration and after restart; for model-class claims repeat on multiple physical units of the same model/SKU/sensor signature; plot signed residual vs azimuth, session, site, and unit, quantifying repeatability **of the curve**, not only of individual headings; classify with `DeviationStructureClass`.

A candidate profile may come only from stable categories, trained on a prespecified subset, and the interpolation method MUST be circular, regularized against overfitting, deterministic, versioned, and selected before held-out scoring. The benchmark MUST preserve the raw point estimates and show the correction curve — a high-order fit oscillating between sparse headings is a failure even at low training RMSE. Repeatability at one site does not prove portability: a pattern changing materially across qualified clean sites is environmental bias, not a device profile. Note the §27 interaction — a fixture perturbing the field as it rotates produces an indistinguishable signature, so qualify the fixture before attributing structure to the phone.

### 29.4 Target and freehand behaviour

Randomize start and target headings including short moves, `90°/180°` moves, reversals, and `350 ↔ 10` wrap. Use representative operators holding freehand in each mode where freehand is allowed. Record the full live trajectory, target delta, angular speed, time entering the near zone, overshoot and reversal count, time to lock, locked heading and bound, final target delta. Include naturally shaky-but-competent holds (should lock) and deliberately excessive motion (should stay `STABILIZING` or reject). Repeat important measurements with full re-placement and measure between-repeat consistency. Run guidance with true, magnetic, and deliberately unavailable/unverified references — true-target guidance must never silently continue under a magnetic reference.

Report centring success and overshoot across candidate `targetNearZoneDeg`/`targetCenteringToleranceDeg`; select on training data and freeze before held-out scoring. These MUST NOT be tuned to make the heading look more accurate. The target UI passes only if it improves acquisition usability without increasing false lock, hiding latency, or weakening the lock gates. Report the rate at which independent repeats exceed the combined bound envelope, separately for clean freehand, jig, disturbed, and candidate/calibrated populations.

### 29.5 Placement study

The study that fixes two of the most consequential config values. The variable here is the **operator**, not the device.

Per mode: hold the fixture azimuth constant for a block, so only placement varies; the operator places, locks, fully removes, and repeats ≥20 times; repeat with ≥4 operators including both right- and left-handed; repeat with a non-magnetic alignment jig as the control, isolating the instrument term; repeat across ≥2 physical planes with different practical access (a flush wall and a recessed door frame behave differently).

Outputs: `placementRepeatability` (circular dispersion per operator per mode); `placementBias` (systematic freehand-vs-jig offset per operator per mode — a consistent operator offset is real, interesting, and **not correctable**, because the app does not know who is holding it); evidence-based `flatFreehandPlacementBound95Deg` and `wallFreehandPlacementBound95Deg`; and a documented repeatability bound for `NONMAGNETIC_ALIGNMENT_JIG`, without which no measurement can reach `HIGH` or `PROFESSIONAL`. Report the spread across operators, not only the pooled figure — a bound set from the best operator is not a product bound.

### 29.6 Interference, tilt, location, cross-validation, latency, power

**Interference (§30.3 gates).** Reproducible-distance fixtures for magnetic case/wallet/ring/mount, MagSafe-class accessory, **active wireless charging** (magnet plus time-varying current), speaker, laptop/tablet, USB/power cable at defined currents with electrical safety controls, ferromagnetic plate/bar with documented material and geometry, vehicle exterior/interior, reinforced-concrete and steel positions. Measure the actual three-axis field at the phone position with independent instrumentation; randomize present/absent and blind labels where practical. **Include at least one configuration that rotates the field vector with little magnitude change** — the case a magnitude-only detector misses and the one producing a confident wrong heading rather than an obviously broken one. Per interferer: clean baseline; distances `0, 2, 5, 10, 20, 50, 100 cm` as safe; multiple relative orientations (magnets are strongly directional); add while stationary, remove, and repeat during controlled yaw; anomalies changing direction with modest magnitude change; capture accept/reject, detection latency, actual error, recovery. Generate ROC and PR curves per feature **and** for the fused classifier, stratified by model, and report the incremental value of inclination and stationary variability over magnitude alone — if they add nothing measurable, that is a finding worth having explicitly. Select thresholds on a fixed training set, evaluate once on held-out devices/sessions/sites.

**Tilt and orientation.** Grid `pitch, roll ∈ {-80, -60, -45, -30, -15, 0, +15, +30, +45, +60, +80}` at representative yaws including cardinal, intercardinal, and wrap. Reduced factorial for regression, full for certification. **The grid extends far beyond the supported envelope on purpose and the regions score differently:** inside the pose gates it is an accuracy test; outside them it is a **rejection** test — the app must decline and say why. There is no requirement that a phone tilted 60° produce an accurate heading, and a build returning one is failing, not exceeding, the spec. Geometries where the reference axis projection is essentially zero MUST be explicitly rejected as singular rather than assigned an arbitrary heading. Screen cases: portrait, portrait-upside-down if supported, both landscapes, face-up/face-down transitions, auto-rotation locked and unlocked, multi-window, foldable/tablet postures — all non-portrait cases are **negative tests** in v1. Acceptance: no `90°`/`180°`/sign-flip errors; P95 additional tilt-induced error inside the envelope `<= 2°` vs level; unsupported/singular poses rejected within 500 ms with correct recovery; no stale pre-rotation sample contaminating the post-rotation window.

**Location and declination.** Model-level: official vectors plus a dense grid over both hemispheres, both declination signs, equator/high latitude/antimeridian/near-pole, sea level/negative/high altitude, all three altitude datums, start/middle/end of validity plus one date outside it to confirm refusal, leap day, year boundary, UTC/local transitions. This measures **numerical fidelity** — declination within `0.01°` of official vectors. It is a *different* test from the site-level check below (`0.05°`), which is end-to-end and includes position and datum handling; never report one tolerance as the other. Run the full suite for both models against their own vectors, record which coefficient hash produced which result, and report the inter-model declination difference across the grid and at every site — that difference is the maximum benefit the high-resolution model can deliver there, and where it sits well below the heading error floor the benchmark cannot resolve it. Verify each error-model artifact loads, is hashed, returns a one-sigma-labelled value, and that `boundFromSigma` is the only path from sigma to bound.

Physical: open sky, partial sky, urban canyon, indoor fallback; fresh vs cached fix; airplane/location off-on; approximate vs precise authorization; grant, denial, one-time grant, downgrade, revocation, process restart; marked mock location on Android; device time manually wrong while network and monotonic clocks differ; declination-gradient movement optional but deterministic replay required.

**Declination sensitivity:** use the same versioned deterministic envelope algorithm as production over the horizontal uncertainty region, altitude interval, and accepted time interval. Validate that algorithm against a substantially denser geodesic/altitude/time grid over the benchmark domain and report its worst under-estimation; a sparse set of perimeter points is not automatically a conservative bound. The resulting maximum circular difference is the production envelope used by location gates. A 5 m vs 20 m fix is usually immaterial; a stale fix from another region is not. Reduced authorization is a provenance label, not automatic failure, when its envelope stays inside the grade budget. Acceptance: app declination within `0.05°` of an independent computation at real fixes; no silent reference substitution; age/jumps/authorization labelled and gated by the envelope; cross-platform agreement including all three datum cases.

**Cross-validation.** Truth-labelled residuals for every provider pair: pairwise circular difference, each provider's error vs truth, error correlation (high correlation reveals non-independence), conditional error when providers agree vs disagree, cases where all pipelines agree but are wrong from a shared distorted magnetometer, and time-alignment sensitivity by shifting streams `±10, ±20, ±50, ±100 ms` in replay. Never award confidence from agreement alone. Report `transformAgreementDeg` separately and expect ~zero — a non-zero value is a bug report about frame math, not a finding about sensors.

**Sensor Health.** Truth-labelled clean, calibration-poor, provider-uninitialized, disturbed, moving, stale-reference, unsupported-sensor, and certified/uncertified cases; confusion matrix over the **full `TrustAction` enum**. Two cells matter more than the rest and should be named rather than buried: `CALIBRATE` when the truth was `ROTATE_TO_INITIALIZE` sends the user through an unnecessary procedure and teaches distrust of the prompt; `CALIBRATE` or `HOLD_STEADY` when the truth was `MOVE_AWAY_FROM_INTERFERENCE` is worse — the user complies, the field is still wrong, and the eventual lock is confidently incorrect. Include adversarial cases where internal pipelines agree but ground truth differs.

**Latency.** Fixture encoder/optical-marker timestamps synchronized to the logging host, with stated synchronization uncertainty. Measure event→callback, callback→engine output, engine→rendered frame, ground-truth motion→display, lock decision, interference detection and recovery. Monotonic end to end; on-device wall clock is insufficient for sub-100 ms claims; use high-frame-rate video or a photodiode for display latency. Report median, P90, P95, P99, max, and distributions by thermal state and device — averages hide dropped callbacks and scheduler stalls.

**Battery and thermal.** Modes: idle sensors-off; live compass at standard rate; active Precision Mode at max rate and high-accuracy location; map + live compass; raw diagnostic logging; background/suspended; a representative one-hour practitioner workflow; FOP 50/100/200 Hz as separate randomized runs. Controlled ambient, consistent starting thermal state and battery band, fixed brightness/radio/content/motion script, chargers disconnected during discharge measurement (wireless charging forbidden during accuracy tests unless testing interference), ≥30 min per mode (60 preferred), ≥3 repeats, alternating baseline and candidate builds to reduce aging and ambient drift. Record battery level/voltage/current, CPU/GPU time, location-active time, sensor rates, frame rate, thermal state.

### 29.7 Required in-app benchmark mode

Internal builds ship a non-consumer screen with identical concepts on both platforms:

```text
Session manifest: site, fixture, operator, device unit, app/config version
Ground truth: yaw, pitch, roll, uncertainty, coverage factor, tier, fixture timestamp
Variant: baseline, provider-error gate, production candidate, diagnostic
Controls: Arm -> Start -> Mark Stable -> Stop -> Export
Live: provider heading/error/reference, location age/accuracy, raw field, WMM field,
      pose, thermal, charging, sample/drop rate, sensor health, calibration entry
      reason, target/delta, deviation-correction state
Outcome: accepted/rejected/timeout, trust action, reason codes, lock heading/bound/time
```

One tap starts an immutable trial ID capturing the full pre-roll/acquisition/post-lock stream. **Ground truth is entered or received from the fixture and never copied from the phone result — the UI MUST make that impossible, not merely discouraged.** Show source timestamps and ages so stale data is visible during testing. Export is lossless JSONL plus manifest and SHA-256. Replay mode feeds telemetry back through the candidate engine without physical providers. Debug/export code MUST NOT change request rate, thread priority, filtering, or lifecycle relative to the measured production candidate; any unavoidable difference is a separately named variant. The build visibly identifies mock location, sensor/location developer options, and Play services/FOP availability.

---

## 30. Acceptance thresholds

Initial release gates, not universal sensor guarantees. Re-tuning requires a versioned benchmark report and must never hide a regression.

### 30.1 Provider vs application decision rule

Paired trials: `AND-G1` vs `AND-G0`, `IOS-A1` vs `IOS-A0`, `AND-RV` vs the Android baseline on no-GMS hardware. Accepted-only P95 at one fixed gate is insufficient — a candidate can look accurate merely by rejecting hard cases.

Sweep the provider-error/quality threshold across its full useful range; at each point report `acceptanceRate = accepted / eligible` and `risk = {P95 absolute error, MAE, Pr(|e| > 10°)}`. Plot risk vs acceptance rate with cluster-bootstrap bands, compare candidates **at the same acceptance rate** by interpolation or prespecified bins, and publish the region below the candidate's minimum usable point rather than extrapolating. `AND-HDG` exports both raw `accuracy68Deg` and `bound95FromAccuracyDeg = boundFromSigma(accuracy68Deg)`; only the converted field may appear in a threshold sweep, coverage calculation, or table column labelled as a 95% bound. `AND-GE` has no wall row because FOP's scalar error does not cover the wall-normal axis.

Adopt the app layer only if its held-out curve is non-inferior at matched acceptance rate, improves severe-error risk in the operating region, and: false accepts with `|e| > 10°` decrease or stay zero; matched-acceptance P95 accepted error worsens by no more than `0.25°` and preferably improves; empirical coverage reaches the declared level; clean acceptance rate stays `>= 95%` within 10 s; median lock time increases by no more than 2 s unless the severe-false-accept reduction justifies it; dynamic settling and display latency stay inside their gates.

**If raw provider output is already more accurate than the app-locked value, keep the provider heading** and retain only the safety checks that demonstrate benefit. Do not replace a superior platform result to preserve architectural symmetry.

Select **lexicographically**, never by one weighted score that can trade accuracy for cosmetics: (1) disqualify any variant violating false-confidence, reference, or maximum-accepted-error gates; (2) minimize severe-error risk and P95 at matched acceptance rate in the prespecified operating region; (3) then MAE, absolute bias, and azimuth-binned bias; (4) then maximize clean acceptance within 10 s; (5) then minimize lock time and tracking latency; (6) battery/thermal only after every accuracy and safety gate passes. Report the full table even after choosing. A single score such as "94% accurate" is prohibited.

### 30.2 Certified clean static measurement

Certification matrix, qualified clean sites, **device in a jig** so placement error is removed and the numbers describe the pipeline:

| Metric | Gate |
|---|---:|
| Median absolute true-heading error | `<= 2.0°` |
| MAE | `<= 2.5°` |
| P95 absolute error | `<= 5.0°` |
| Absolute circular bias per model | `<= 2.0°` (Tier 2 reference required) |
| Max absolute mean signed error in any 45° azimuth bin | `<= 3.0°` |
| Maximum accepted error | `<= 10.0°` |
| Clean acceptance rate within 10 s | `>= 95%` |
| Within-condition circular repeatability (SD-equivalent) | `<= 2.0°` |
| False accept with actual error `> 10°` | `< 0.5%`, no unexplained severe case |

The azimuth-binned gate exists because whole-circle bias is nearly blind to the most characteristic magnetometer defect: residual hard-iron error produces roughly sinusoidal signed error integrating to ~zero. A device with `±5°` of sinusoidal structure passes a `2.0°` bias gate comfortably while being `5°` wrong at two specific bearings — and a practitioner measuring one wall does not average over the circle.

A failing model is not "fixed" by excluding inconvenient headings: degrade it, deny certification, or give it a validated model-specific floor.

Report the table three more ways — separately for each mode, and once for freehand by a representative operator. The freehand table predicts what practitioners experience and will be several degrees worse. **Do not gate on it** (placement is an operator property) but publish it and use it with §29.5 to set the placement bounds from evidence rather than from the candidate values.

### 30.3 Interference and latency gates

| Metric | Gate |
|---|---:|
| Severe-disturbance rejection sensitivity | `>= 95%` |
| Clean-condition specificity | `>= 95%` |
| Severe-disturbance detection latency | `<= 1.0 s` P95 |
| Recovery after removal | `<= 5.0 s` P95, subject to recalibration need |
| Locked measurement during attached-magnet severe case | `0` in certification runs |
| Warm-start lock, clean outdoor | `<= 5 s` median, `<= 10 s` P95 |
| Time out of `PROVIDER_INITIALIZING` after guided rotation | `<= 10 s` P95, reported per model |
| 90° step display onset latency | `<= 150 ms` P95 |
| 90° step settling within `±3°` | `<= 1.5 s` P95 |
| Overshoot | `<= 5°` P95 |
| Stationary filtered jitter | peak-to-peak over a **rolling 2 s window**, P95 `<= 3°`, without violating the latency gate |

The jitter window is named explicitly because "P95 peak-to-peak" is meaningless without one, and two implementations measuring over different windows report different numbers for identical behaviour. If a provider cannot meet a threshold, report platform evidence and revise the certified tier rather than concealing lag with smoothing.

### 30.4 No-GMS gates

`AND-RV` ships, so it needs its own gates — looser on precision, **identical on safety**, because the failure this path most plausibly introduces is a wrong reference rather than a noisy one.

| Metric | Gate |
|---|---:|
| Median absolute error | `<= 3.0°` |
| P95 absolute error | `<= 7.0°` |
| Maximum accepted error | `<= 10.0°` |
| Max absolute mean signed error in any 45° bin | `<= 4.0°` |
| False accept with `|e| > 10°` | `< 0.5%` — identical to Google path |
| Empirical coverage of `reportedBound95Deg` | `>= 95%` — identical |
| Clean acceptance within 10 s | `>= 90%` |
| Severe-disturbance rejection sensitivity | `>= 95%` — identical |
| Max provisional internal grade before dedicated evidence | `USABLE`; display stays `CANDIDATE` until certified |

Certification is per exact §24 runtime identity — a rotation-vector implementation is a *vendor* implementation, and two phones sharing a marketing name may not share it. When regional or sensor variants cannot be distinguished by public runtime fields, the evidence must cover all variants that map to the same key with a conservative worst-case floor, or that key is not certified. `resolvedReference` MUST be `TRUE_CORRECTED_FROM_MAGNETIC` with `APP_APPLIED_DECLINATION`, or an explicit magnetic/unverified state; there is no `TRUE_VERIFIED` here without an independent reference check, and §11's ambiguity rule does not apply. `TYPE_GAME_ROTATION_VECTOR` MUST NOT substitute for a missing `TYPE_ROTATION_VECTOR` — that is an unsupported device. Devices defensibly identified as using synthesized rotation vectors or no gyroscope MUST be evaluated as a separate runtime-distinguishable class before certification; an unobservable distinction is handled by worst-case pooled evidence, not an invented key field. **The no-GMS build MUST run the full Tier 0 comparison independently on no-GMS hardware; reusing Google-path telemetry is not evidence.**

### 30.5 Reference challenge (hard Android certification test)

At a site with `|declination| >= 8°`, in **both** modes (§11.1):

1. With fresh precise location, record FOP heading, quaternion-derived heading, independent magnetic heading, WMM declination, surveyed true azimuth.
2. Revoke location, select approximate, disable system location, restart the process as the OS requires — each state separately.
3. Restore precise location and repeat through recovery.
4. Classify each callback as behaving true- or magnetic-referenced relative to truth.
5. Verify the app never labels or locks a magnetic/unknown sample as `TRUE`, in either mode.

Pass: zero silent mislabels in both modes; loss of reference evidence invalidates an active `TRUE` lock within 500 ms or the first affected callback, whichever is later; recovery requires a fresh precise fix, a new cross-check, and a new stable window; each mode derives its own mode/axis-bound `ReferenceResolutionResult`, with `correctionDeg` appearing exactly once in that active-mode heading — **a test MUST specifically look for the `2 × declination` double-correction signature**, roughly `16°` at this site and otherwise indistinguishable from a plausible bearing; telemetry preserves provider heading, both hypotheses, declination, permission state, and every state-change reason.

### 30.6 Deviation-correction adoption gates

`NONE` is the production default. Promotion to `CERTIFIED_PROFILE` requires all of, on held-out evidence for the declared scope: learned from external-ground-truth residuals and frozen before scoring; exact ID/hash in every trial and in the certification key; stable across the sessions/days/sites the scope claims, and across held-out physical units for model-class; no increase in severe false accepts or maximum accepted error; at matched acceptance rate, non-inferior on P95/MAE/bias and meeting a prespecified minimum worthwhile improvement (default `0.5°` P95 unless the plan justifies otherwise **before** data are examined); correction harm rate and worst degradation reported with no unexplained heading/site cluster; uncertainty model includes the held-out post-correction residual and still passes coverage and sharpness; profile survives ordinary recalibration within the certified protocol, or its scope reflects that instability; site-, accessory-, or transient-dependent profiles rejected as portable corrections — those belong to interference rejection, not compensation.

`UNIT` profiles remain experimental artifacts in v1 and MUST NOT produce a `CALIBRATED` consumer result, because the certification database intentionally does not bind to physical-unit identity. A per-unit correction product would need a privacy-safe stable unit-binding contract and a revised key first. Disabling or changing a certified profile changes the key and therefore invalidates the old calibration state automatically.

### 30.7 Calibration benchmark

States: recently calibrated clean; naturally low/unknown after reboot or inactivity; deliberately poor orientation coverage without permanent magnets; after transient controlled exposure once removed; known hard-iron behaviour; clean-accurate where the user requests recalibration despite no trigger; disturbed where the user requests it and the correct action is to defer. Never expose devices to fields risking damage or permanent bias.

Protocol: 60 s pre-calibration telemetry plus a static sweep subset; invoke separately via automatic trigger and `USER_REQUESTED`; scripted 3D motion with documented duration and coverage; record state changes, uncertainty, bias estimate, prompts; repeat the sweep subset; repeat after 10 minutes and after restart to test persistence.

KPIs: time to valid calibration; success within 30/60 s; change in absolute and P95 error; change in cross-pipeline disagreement; change in ellipsoid residual/bias; **false success** (declared calibrated, accuracy fails); **false failure** (clean accurate device trapped in calibration UX); prompt-loop count and recovery from cancellation/backgrounding; automatic-trigger miss and false-trigger rates; manual request from already-good state accepted as unchanged-good without inventing improvement; correct deferral in a disturbed field; correct `CALIBRATE` vs `ROTATE_TO_INITIALIZE` distinction. Accepted only if objective metrics improve or stay within certified limits.

### 30.8 Power and thermal gates

Report battery points per hour with CI, energy per measurement, CPU time, wakeups, location-active time, callback rate, time to thermal transition, accuracy/acceptance/latency/rate degradation by thermal state, dropped samples and frame misses. Gates: no sustained precision run enters severe/critical thermal state at standard ambient; P95 error and tracking latency after 30 minutes stay within 20% of cool-state values **and** inside absolute certification gates; background mode stops precision sensors/location within the lifecycle deadline unless an authorized user-visible feature requires otherwise; candidate energy MUST NOT regress more than 10% vs the approved baseline without documented accuracy benefit and sign-off; a faster FOP period is rejected when it produces no practically significant latency/lock improvement.

---

## 31. KPI definitions

```text
e_i = shortestSignedDifferenceDeg(m_i, g_i)   // shared §9/§15 function; (-180,180]
absoluteError_i = abs(e_i)                    // [0,180]
```

The shared function call is mandatory: restating raw `atan2(sin(m_i-g_i), cos(m_i-g_i))` here reintroduces `-180.0` for one antipodal ordering and contradicts the canonical convention. The conversions inside that shared function are not decorative — passing degree arguments to `sin`/`cos` produces plausible-looking small errors and has appeared in published compass evaluations.

| KPI | Definition |
|---|---|
| Signed circular bias | Circular mean of `e_i` with CI. Gates require Tier 2. |
| Azimuth-binned signed bias | Max absolute mean signed error in any 45° bin. Catches structure whole-circle bias cancels. |
| MAE / median / P90 / P95 / P99 / max absolute error | Using the §9.1 pinned estimator. |
| Repeatability | Within-device/session circular dispersion at identical truth and condition. |
| Between-unit variability | Variation among units of one model. |
| Acceptance rate | Accepted / eligible attempts before timeout. |
| Correct rejection / false accept / false reject rate | Against truth-labelled disturbed and clean sets. |
| Time to first valid heading / out of initialization / to Precision Lock | Three distinct start-relative times. |
| Tracking latency, settling time, drift, jitter | Dynamic behaviour vs fixture motion. |
| Placement repeatability / placement bias | §29.5, per operator and mode. |
| Empirical coverage | Per bound, against its matching population. |
| Sharpness | Median/quantiles of reported uncertainty; prevents trivial `±180°`. |
| Sector agreement | Fraction whose primary sector matches truth. **This, not degrees, is what the practitioner acts on.** |
| Straddle rate | Fraction reported as straddles, plus the fraction of those where truth fell in a non-primary sector. |
| Transform disagreement | Same-provider extraction routes; should be ~0, non-zero is a code defect. |
| Pipeline disagreement | Max pairwise circular difference over independent estimators, with the set recorded. |
| Declination error | App declination minus reference computation. |
| Azimuth-dependent residual structure | Signed error as a circular function of azimuth: first-harmonic amplitude, shape, repeatability across sessions/sites/units. |
| Deviation-profile transfer error / correction harm rate | Training-vs-held-out benefit gap; fraction of held-out observations materially worsened, severe cases separate. |
| Target acquisition error / time / overshoot | Requested vs final locked heading (always read with the bound); start-to-lock, max overshoot, reversals, wrap failures. |
| Repeat-measurement consistency | Circular difference between independently re-placed measurements. |
| Sensor Health action accuracy | Confusion matrix over the exact `TrustAction` enum. |
| Energy rate / thermal degradation | Per active precision minute; change vs thermal state and time. |

Never report linear mean or variance of raw headings near the wrap boundary.

---

## 32. Statistical analysis

### 32.1 Populations and estimation

Predefine: **all attempts** (includes timeouts and rejections; used for acceptance rate and safety); **accepted measurements** (stated accuracy, always with acceptance rate); **clean qualification set**; **disturbance set**; **jig-placed** and **freehand**, never pooled; per-model, per-unit, and pooled.

Use circular error definitions. Report sample count at every aggregation level. Use cluster/bootstrap CIs resampling at **device-unit and session** level, not high-rate sensor samples as if independent, with the resample count and seed pinned in the analysis configuration. Fit a hierarchical/mixed-effects model when comparing platform, model, heading, tilt, environment, session — unit and session as random effects, prespecified conditions as fixed. Inspect error vs azimuth for sinusoidal patterns and report first-harmonic amplitude explicitly, not just pooled dispersion. Inspect residuals vs pitch/roll, field magnitude, temperature, location age, sensor rate, charging state, and time since calibration. Correct for multiple comparisons or clearly label exploratory analyses.

### 32.2 Uncertainty calibration

For each predicted bound `u_i`, compute `I(|e_i| <= u_i)` and plot empirical vs nominal coverage, coverage by uncertainty bin, absolute error vs predicted uncertainty, reliability diagrams, sharpness distributions, and catastrophic miss rate (e.g. `|e| > max(10°, 2u)`).

Split by **device unit / session / site**, never random sensor samples, to prevent leakage. Tune on training units/sessions, freeze thresholds, report final performance on held-out devices and sites. Audit `instrumentBound95Deg` against **jig-placed** error and `reportedBound95Deg` against **freehand** error — auditing the total against jig data credits the placement term for an error that was experimentally removed, the easiest way to certify a bound that fails in the field.

### 32.3 Selective risk

Treat the acceptance gate as a selective predictor. Produce risk-vs-acceptance-rate curves separately for clean, disturbed, wall, flat, jig, freehand, per-model, and pooled populations, resampling whole units and sessions for bands. Compare variants only at overlapping acceptance rates; accepted-only metrics at unmatched rates MUST NOT decide the winner. Report AUC only as a secondary summary — it can hide a dangerous operating region.

### 32.4 Correction validation

Plot signed residual vs azimuth before and after correction per device/mode/provider scope; quantify curve repeatability across session, day, site, recalibration state, and unit using circular/periodic methods (do not treat `0°` and `360°` as separate endpoints). Separate training and evaluation: fit on training sessions/sites/units appropriate to the claimed scope; freeze model form, knots/harmonics/smoothing, parameters, and hash; evaluate on held-out sessions/sites and held-out units for a model-class claim; compare `NONE` vs profile at matched acceptance rate; report per-azimuth benefit and harm, P95/MAE/bias/max, severe false accepts, harm rate, coverage; reject a profile whose benefit disappears outside its training site or whose residuals form a new unexplained azimuth cluster. **Never average a site-specific correction into a device floor** — if the environment is the source, the safety response is rejection, not portable compensation.

### 32.5 Missingness and pass/fail

Timeouts, API errors, invalid headings, crashes, thermal shutdowns, and rejections are **outcomes**, not missing-at-random samples; report their rates. Exclude a trial only by a prespecified rule such as verified fixture failure, preserving it in the audit log.

A release passes only when: every hard safety gate passes; pooled gates pass with CIs; no certified high-volume model fails its model-level gate; there is no unexplained cluster of large errors at a heading, tilt, environment, or unit; coverage and sharpness both pass **for both bounds against their respective populations**; battery/thermal and latency gates pass or an explicitly approved tier downgrade is documented.

---

## 33. Regression strategy

### 33.1 Deterministic CI (every change)

Angle normalization and shortest-difference property tests over large random sets, including the `(-180,180]` antipode and `360 → 0` rules; a **per-runtime single-implementation check** with exactly one allowlisted `shortestSignedDifferenceDeg` implementation in each executable runtime that needs it (Android core, iOS core, and analysis tooling), while all other call sites use that runtime's shared utility (R67/R68). `shortestTargetDeltaDeg` and `absoluteCircularDifferenceDeg` MUST be thin delegating wrappers with the exact §9 definitions, and no local `deltaDeg` alias is permitted. CI MUST audit source-code `atan2` call sites outside those allowlisted implementation files and the separately allowlisted bearing-projection/circular-mean implementations; a new signed-difference formula or helper outside the approved implementation sites fails. Documentation and tests may quote a prohibited formula as text, so a blind repository-wide grep is not sufficient and MUST NOT reject explanatory prose. Cross-runtime golden/property fixtures enforce identical outputs, including both orderings of multiple antipodal pairs; wrap boundary, exact `0/360`, negative, `NaN`, infinity, antipodal cases; **pinned quantile/median parity** between platforms and the analysis tooling; quaternion/matrix golden vectors for every supported orientation; REFERENCE_ENU handedness/reference tagging, axis remapping, screen rotation; timestamp alignment, rollover, clock-jump, stale data; WMM official vectors for both models, error-model loading, `boundFromSigma`, cross-platform parity; configuration schema plus **every §8.1 invariant** including the no-calibration-state-key assertion, and the §8.1.1 grade-reachability analysis asserting that no grade claimed reachable is arithmetically forbidden by the constants; Feng Shui schema, derived-boundary consistency, every sector boundary; straddle sets including wide bounds, north wrap, and the full-circle degenerate case; target shortest-delta across quadrants and `359/0`, including reference unavailable/change; manual calibration entry and disturbed-field deferral state machine; `PROVIDER_INITIALIZING` entry/exit distinct from `CALIBRATE`; deviation-correction lookup, hash, circular interpolation, sign, wrap, and exactly-once application; certification-key construction and miss-yields-`CANDIDATE`; JSON encoding under a comma-decimal locale plus nonfinite rejection and round-trip precision; recorded telemetry replays with frozen expected outputs; out-of-order, duplicate, dropped, delayed, discontinuous sample fuzzing; permission/location/provider state machines.

### 33.2 Physical smoke (every release candidate)

8 yaw headings on one reference iPhone and one reference Android; level plus `±30°` pitch/roll subset scored as a rejection test outside the envelope; one attached magnetic accessory **and** one field-rotating magnitude-preserving interferer; one user-requested calibration from clean-good plus one disturbed-state deferral; one target acquisition crossing north wrap; one `WALL_FLUSH_BACK` measurement with transform-agreement active; cold/warm start and orientation transition; 30-minute sustained precision/thermal run.

### 33.3 Full recertification

New device support or form factor; major OS or Play services change; fusion/filter/confidence/interference algorithm change; deviation-profile/model/policy change; target-guidance changes affecting timing, stability, or acceptance; WMM coefficient/model/error-model change; Feng Shui ruleset geometry or reference change; SDK migration affecting location/orientation; any severe field incident or false-confidence bug.

Regression comparisons MUST use paired runs or replay where possible. Flag statistical **and** practical changes. Reduced jitter with increased tracking delay is a regression unless the tradeoff was approved. Treat a remote/provider behaviour change as a new implementation variant even when app source did not change.

---

## 34. Failure modes

Severity: **Critical** — can display or lock a confidently wrong reference or large heading without warning. **High** — material accuracy, rejection, timestamp, or lifecycle failure in common use. **Medium** — visible or recoverable degradation, or a narrow device/condition. **Low** — diagnostic or cosmetic, no incorrect locked measurement. Every Critical/High item needs a test, telemetry signature, owner, and mitigation before certification.

Critical failures share a shape: **the output remains a plausible bearing.** Double-applied declination, swapped sitting/facing, magnetic labelled `TRUE`, a wall-mode quaternion with a transposed axis — none makes the dial jump or the app crash. That is why this section is tests, not advice.

| # | Failure | Sev | Guard |
|---|---|---|---|
| 1 | Linear averaging across north (`359` and `1` → `180`) | Crit | Circular mean; wrap-crossing window tests |
| 2 | Wrong normalization; language `%` differs for negatives | High | `((x%360)+360)%360`, finite check, `360 → 0` pinned |
| 3 | Wrong subtraction direction flips signed bias | High | `measured - truth` defined once; property tests; absolute error separate |
| 4 | Degrees/radians confusion (rotation-vector accuracy may be radians) | High | Typed units; §31 shows conversions explicitly |
| 5 | Quaternion `wxyz`/`xyzw` swap, non-normalized, active/passive inversion, multiplication order, handedness, gimbal-lock Euler | Crit | Golden physical poses on **both** extraction routes |
| 6 | `R ≈ 0`; `atan2(0,0)` returns 0 on both platforms → confident false north | Crit | `minCircularResultantLength` gate, `CIRCULAR_MEAN_UNDEFINED`, clamp before `asin`/`sqrt` |
| 7 | Premature rounding moves a sector (`337.49°`) | High | Classify full precision; half-open derived boundaries; deterministic epsilon |
| 8 | Declination sign/unit/date/datum errors | Crit | Official vectors at diverse sites, both signs |
| 9 | **Confidence-level conflation** — one-sigma added to a sum of 95% terms under-covers ~2× | Crit | Explicit `sourceConfidenceLevel`; single `boundFromSigma` |
| 10 | Mixed clock domains create negative age or fresh-looking stale data | High | Source timestamp + clock ID; logged mapping with uncertainty; monotonic for intervals, UTC only for model date and audit |
| 11 | Callback arrival mistaken for sample time under batching | High | Freshness from `sourceMonotonicTimeNs` only (§22) |
| 12 | Cached location reused; restart does not make a fix fresh | High | Provider timestamp, reject by age |
| 13 | Out-of-order/duplicate samples inflate the window | Med | Sequence + source timestamps, explicit duplicate policy, duplicates do not increment `effectiveHeadingSampleCount` / `periodicSupportSampleCount` |
| 14 | Filters bridge reboot/suspend/provider discontinuity | High | Discontinuity flags (§13), iOS inference (§12), lifecycle resets |
| 15 | New magnetometer vector paired with old gravity while rotating | High | Interpolate only within a max gap, else reject |
| 16 | Android sensor axes do not rotate with UI → exact `90°/180°` errors | Crit | Explicit remap for the raw path; no double remap on FOP heading |
| 17 | iOS `headingOrientation` change mid-window corrupts it | High | Invalidate window on transition |
| 18 | Wall facing vs sitting confused → exact 180° | Crit | Facing is the default; sitting is labelled and derived (§14) |
| 19 | ENU/NED swap; map bearing convention differs from engine | High | One REFERENCE_ENU axis convention plus explicit north-reference tag; frame-tagged types |
| 20 | Magnetic heading labelled `TRUE` on the attitude route | Crit | §11/§11.1; per-mode same-axis resolution; §30.5 in both modes |
| 21 | Declination applied twice → exactly `2 × declination` | Crit | `correctionDeg` exists once; §30.5 looks for the signature |
| 22 | Fixed accessory bias that looks perfectly stable | Crit | Multi-feature detector; every dispersion gate would otherwise pass |
| 23 | Field rotated with normal magnitude defeats magnitude-only detection | Crit | Inclination + stationary variability in the classifier (§16) |
| 24 | Transform bug reported as magnetic advice; user complies, bug survives | High | `transformAgreementDeg` separate from `pipelineAgreementDeg` (§16.1) |
| 25 | Hysteresis after accessory removal; charger current drifting the field | Med | `recoveryCleanWindowMs`; charging state recorded and gated |
| 26 | Last valid heading retained after provider error; UI timer repaints it as live | Crit | Age on every value; `TIMED_OUT` emits no measurement |
| 27 | Over-smoothing hides lag and interference spikes; effective N ≪ raw N | High | Time-based filters; separate raw/filtered/locked streams; report `effectiveHeadingSampleCount` / `periodicSupportSampleCount`; step-response benchmark |
| 28 | Treating OS error as a guaranteed bound | Crit | Device floors, composed bound, held-out coverage |
| 29 | Train and test on the same device/session | Crit | Split by unit/session/site (§32.2) |
| 30 | Grading on `instrumentBound95Deg`, or auditing the total against jig data | Crit | §20; bound-to-population pairing enforced |
| 31 | Standalone certified grade shown while `CANDIDATE` | Crit | §19.1 invariants; UI test, not inspection |
| 32 | An editable config value that turns every device Professional | Crit | Schema forbids the key; §8.1 test |
| 33 | Fitting and scoring an azimuth correction on the same data | High | §30.6, §32.4 |
| 34 | Site pattern promoted to a portable device correction | Crit | `SITE_DEPENDENT` → rejection, never compensation |
| 35 | Endless calibration loop caused by environment, not the device | High | Distinguish device failure from interference; deferral path |
| 36 | Calibration declared successful because the animation finished | High | Measured before/after outcomes (§17) |
| 37 | User sent to calibration when the truth was provider initialization | Med | `ROTATE_TO_INITIALIZE` (§18.4); named confusion-matrix cell |
| 38 | Momentary exact target digit treated as a lock; traversed headings averaged in | High | Only the accepted stable window enters the lock (§18.2) |
| 39 | Linear target delta near `0/360` sends the user the long way | Med | `shortestTargetDeltaDeg` + wrap tests |
| 40 | True-target guidance continues after reference goes stale | High | Visible downgrade; reference re-gated at lock |
| 41 | Single mountain shown beside a `LOW_CONFIDENCE` bound | High | §21.3–21.4; no single-sector claim below `USABLE` |
| 42 | Result stored without ruleset version/reference → uninterpretable later | Med | Both persisted with every record |
| 43 | Play services absence treated as a crash path | High | Supported configuration with its own certification (§30.4) |
| 44 | Duplicate listeners, missing unregister, fusion on the UI thread, unbounded queues | High | §4 ownership rules |
| 45 | Regional SKU/repair/firmware changes the sensor under one marketing name | High | Runtime sensor identity where exposed; otherwise pooled worst-case variant evidence or no certification; ≥3 units per model |
| 46 | Remote threshold change without versioned telemetry; lock restored after restart without evidence | High | No remote config in certification builds; `configHash` in every record |
| 47 | Swift/Kotlin JSON `NaN` divergence; `Double` through `Float`; locale separators; differing default quantile estimators | High | §22.2 and §9.1 with explicit tests |
| 48 | Millisecond/nanosecond overflow; 32-bit counters on older targets | Med | Typed units; overflow tests |

---

### 34.5 Additional frame/applicability failure modes found by scenario tracing

| ID | Failure | Severity | Required prevention |
|---|---|---|---|
| R49 | Core Motion provider-native attitude is consumed as if its native axes/direction were project REFERENCE_ENU | Critical | Adapter-normalize to `attitudeQuaternionDeviceToReferenceEnuXYZW`; native frame retained only as provenance; golden vectors. |
| R50 | Wall `pipelineAgreementDeg` compares top-edge scalars with wall-normal bearings | Critical | Active-mode same-axis pipeline sets from §15.1; exclude ill-conditioned/mismatched axes. |
| R51 | Google-only initialization/reference/transform gates are applied universally | Critical | §18.5 provider/mode applicability matrix; N/A is not failure and is not fabricated evidence. |
| R52 | iOS flat is required to deliver 50 `CLHeading` events in 2 s or remain <100 ms old while stationary | Critical | Treat CLHeading as in-window event anchors; Core Motion supplies periodic support. |
| R53 | AND-RV wall falls through a top-edge/azimuth transform | High | Project device `+Z` through rotation-vector magnetic ENU and add WMM once; physical wall golden poses. |
| R54 | Certification survives an error-model, location-provider, or placement-fixture change | High | Complete §24 key with those hashes/providers. |
| R55 | FOP conservative error is 180 but implementation silently falls back to ordinary error | High | Branch on `hasConservative...`; advertised-but-180 stays `PROVIDER_INITIALIZING`. |
| R56 | Fresh G5 rule exists in prose but not config/state/gate/checklist | High | `spaceWeatherRejectKpMin`, explicit enum/reason, lock gate and tests. |
| R57 | `CHARGING_STATE` emitted as `gradeLimitedBy` but absent from enum | High | Enum includes it; schema/golden fixture test. |
| R58 | Rotation-vector provider accuracy is discarded, treated as zero, or re-derived at an assumed confidence | Medium | Capture `values[4]` when available, radians→degrees, use its AOSP-documented 95% semantics unconverted, and keep it absent (never `0°`) when `-1`. |
| R59 | Google reference resolution requires final `MagneticState`, while final state requires a reference-resolved Google pipeline | Critical | §11 precheck excludes pipeline agreement; enforce the staged dependency order through final lock. |
| R60 | ENU `Bup` is compared directly with WMM positive-down inclination | Critical | Negate `Bup` before `asin`; positive-down field names and hemisphere/equator golden tests. |
| R61 | FOP display-top scalar error is applied to wall outward-normal projection | Critical | FOP wall provider error is absent; separate wall floor/coverage; `AND-GE` wall is N/A. |
| R62 | Uncertified flat-freehand example emits Precision Lock despite the unknown floor | High | Reachability-consistent examples are executable fixtures; shipped example must degrade at `7.0°`. |
| R63 | RV 95% provider term is claimed to overcome the unknown floor | High | Property test for `max`: adding the term leaves base unchanged or increases it; certification alone may lower the floor. |
| R64 | `TYPE_HEADING` 68% accuracy is compared directly with 95% bounds | Critical | Preserve raw field; route every bound/coverage comparison through `boundFromSigma`. |
| R65 | A two-entry 24-Mountains excerpt is shipped as the required ruleset | High | Complete 24-sector/8-group artifact; exact cardinality, index, reference, and geometry validation. |
| R66 | Certification key depends on unobservable identity, admits future version ranges, or survives decision-logic drift | Critical | Runtime-derivable exact identities, explicit unobservable sentinel with pooled evidence, and `engineDecisionLogicHash`. |
| R67 | KPI signed error duplicates raw `atan2` and bypasses the canonical antipode normalization | High | §31 calls `shortestSignedDifferenceDeg`; forbid duplicate signed-difference formulas in validation. |
| R68 | A caller creates a second signed-difference definition or local alias, a derived wrapper implements independent angle math, or a global one-definition rule rejects required platform implementations | High | §9 owns one normative contract and exact delegating wrappers; each runtime has one allowlisted implementation; §15/§31 call it directly; §33.1 audits call sites plus cross-runtime parity. |

## 35. Verification checklist

Before a release is called accuracy-certified:

- [ ] Reference axis and north convention documented in code, UI, and tests per mode, including facing/sitting.
- [ ] Both geomagnetic models run from vendored NOAA C compiled identically on both platforms, pass their own vectors, and match cross-platform.
- [ ] Each model's published error model is vendored, hashed, and the only source of `declinationSigma1Deg`; sigma→bound applied exactly once and recorded as an assumption.
- [ ] Shipped `geomagneticModelId` justified by evidence, and magnetic thresholds calibrated against that same model.
- [ ] Model validity dates enforced; expiry produces an explicit update state, not extrapolation.
- [ ] Every altitude carries a datum; parity fixtures cover ellipsoidal, orthometric, unknown.
- [ ] Every provider-native attitude is converted at the adapter boundary to `attitudeQuaternionDeviceToReferenceEnuXYZW`; Core Motion's native frame is never fed directly into REFERENCE_ENU core math; N/E/S/W/up golden vectors pass.
- [ ] Google attitude-derived headings pass the same mode-axis reference resolution as Google scalar headings where the scalar axis is valid, with an explicit `2 × declination` test; iOS and AND-RV use their explicit reference contracts instead of the Google resolver.
- [ ] Google reference resolved once per **active-mode** stable window on the same well-conditioned physical axis in both hypotheses; result is mode/axis-bound, `correctionDeg` applied exactly once, ambiguity flows into composition rather than overwriting it.
- [ ] Google reference resolution is acyclic: `referenceMagneticPrecheckState` uses no pipeline/reference-dependent feature; final `MagneticState` is computed only after resolution and is re-gated at lock.
- [ ] All §8.1 configuration invariants pass, including the no-calibration-state key and the separation-margin ordering.
- [ ] The §8.1.1 grade-reachability analysis passes: every grade the spec text claims reachable is arithmetically reachable under the shipped constants, for every supported placement/certification/magnetic-state combination.
- [ ] Every executable/display example passes production bound composition and reachability checks; no uncertified flat-freehand fixture locks under the shipped `4° + 3°` minimum.
- [ ] Nothing claims `95%` without held-out coverage for that key; `CALIBRATED` only via a database hit; `CANDIDATE` UI shows no standalone certified grade.
- [ ] `boundCalibrationState` / `uncertaintyCoverageEvidenceState` invariant holds on every emitted result.
- [ ] Interference classifier uses magnitude **and** inclination **and** stationary variability; incremental feature value reported.
- [ ] Inclination conversion negates canonical ENU `Bup` before comparison with WMM positive-down `I`; northern, southern, and zero-inclination golden tests pass.
- [ ] Signed circular difference returns `+180` and never `-180` at the antipode, verified for both orderings of at least two distinct antipodal pairs; each executable runtime has one allowlisted implementation, target/absolute wrappers delegate exactly as §9 specifies, no local aliases exist, and cross-runtime golden outputs match.
- [ ] Transform and pipeline disagreement are separate metrics with separate reason codes; a transform fault never surfaces as magnetic advice; every `pipelineAgreementDeg` compares the same active physical axis and north reference, and fewer than two valid pipelines yields `ABSENT`/`UNKNOWN`, not zero disagreement.
- [ ] `ROTATE_TO_INITIALIZE` distinguished from `CALIBRATE` in code, UI, and the confusion matrix.
- [ ] Ground-truth budget complete, states tier and coverage factor, adequate for the claims; bias gates used Tier 2.
- [ ] At least three units per certified high-volume model.
- [ ] Static, wrap, dynamic, tilt, and orientation tests pass in both modes, with out-of-envelope poses scored as rejection tests.
- [ ] Azimuth-binned bias gate passes, not only whole-circle bias.
- [ ] Placement bounds measured via §29.5, not assumed; no grade above `USABLE` without a measured method.
- [ ] `instrumentBound95Deg` audited against jig error, `reportedBound95Deg` against freehand error — not the reverse.
- [ ] Lock, degraded, and invalid are distinct outcomes; nothing above the lock ceiling is lock-styled.
- [ ] `AND-RV` passed its own gates on no-GMS hardware, or the release declares it does not ship.
- [ ] RV `values[4]` enters the base `max` without sigma conversion and never lowers the unknown/certified floor; property tests assert monotonicity when the term is added or increased.
- [ ] FOP scalar heading errors gate/bound flat only; wall records them as diagnostic, uses provider error `NONE`, and relies on a wall-specific floor/coverage record.
- [ ] Raw `TYPE_HEADING` 68% accuracy is retained, and every 95%-bound threshold, coverage, or scorecard comparison uses `boundFromSigma` exactly once.
- [ ] Every certification lookup uses only exact runtime-derivable identities or the defined `NOT_RUNTIME_OBSERVABLE` sentinel, plus `engineDecisionLogicHash`; no open-ended version range or lab-only field can match.
- [ ] Heading provider path/error source, location provider path, mode, placement method/profile hash, engine decision-logic hash, WMM coefficient hash, and WMM error-model hash appear in every certification lookup and relevant telemetry/report metric.
- [ ] Feng Shui ruleset schema-validated, hashed, recorded with reference selection and needle offset; the shipped artifact has exactly 24 sectors/8 groups and all derived boundaries match declared centres.
- [ ] Straddle behaviour passes every boundary, both sides, wide bounds, north wrap, full-circle degenerate case; `TRUE_WITH_AMBIGUITY_BOUND` is tested under both hidden Google hypotheses and both TRUE/MAGNETIC ruleset references without dropping the ambiguity term.
- [ ] Space weather degrades to `UNKNOWN` offline without blocking; fresh configured G5/extreme state emits `SPACE_WEATHER_EXTREME` and refuses a Precision Lock under the documented v1 WMM-dependency policy.
- [ ] Weak-horizontal-field and charging-state gates implemented and tested.
- [ ] Automatic and manual calibration pass improvement/no-change, false-success, false-trigger/miss, and deferral tests.
- [ ] Interference sensitivity, specificity, latency, recovery pass, including a field-rotating magnitude-preserving interferer; `CLEAN` described as evidence, not proof.
- [ ] GNSS freshness and approximate/reduced permission paths pass; location-provider provenance changes miss the certification key.
- [ ] iOS flat treats `CLHeading` as event-driven anchors rather than a 50 Hz stream; periodic Core Motion support meets its separate sample/freshness gate. On Google flat, FOP conservative-180 never falls through to ordinary error; FOP wall does not inherit that scalar gate.
- [ ] AND-RV wall mode projects device `+Z`, applies WMM once, captures rotation-vector heading accuracy when available, and passes wall N/E/S/W physical golden poses.
- [ ] Cross-validation and shared-bias cases evaluated.
- [ ] Coverage and sharpness pass on held-out data.
- [ ] Target guidance uses shortest circular delta, does not weaken the lock, passes freehand/wrap/reference-loss tests.
- [ ] Practitioner protocol implemented as actionable UX; repeat inconsistency surfaced.
- [ ] Full-azimuth residual plots generated per device/mode; if a profile ships, §29.3, §30.6, §32.4 pass and its hash is in the key.
- [ ] Sensor Health evaluated against truth-labelled conditions using the full enum.
- [ ] Machine exports pass locale, nonfinite, precision, and casing tests.
- [ ] Battery, thermal, lifecycle, sustained-use gates pass.
- [ ] All attempts, rejections, timeouts, exclusions reported.
- [ ] Raw telemetry, manifests, analysis code, seeds, hashes archived.
- [ ] Deterministic replay and physical smoke regressions pass.
- [ ] No unresolved Critical/High false-confidence issue remains.

---

## 36. Delivery phases

Later phases depend on earlier exit evidence.

**Phase 0 — repo, schemas, pinned tools.** Native skeletons; pinned manifests; §4.1 module boundaries; `precision-profile-v1.json` + schema + all §8.1 invariant tests; `feng-shui-rules-v1.json` + schema + derived-boundary test; telemetry/session schemas; CI commands for schema validation and both pure-core suites; all `docs/` files.
*Exit:* both skeletons build; config, ruleset, and example telemetry validate; every invariant test runs and passes; no dynamic versions; fake/replay providers are debug-only.

**Phase 1 — pure core and WMM wrapper.** Named units/frames/quaternion order/enums; canonical `device→ENU` attitude type plus provider-native-frame provenance; circular utilities, pinned estimators, target deltas, property tests; event/state/effect reducer over the single `MeasurementState`; config parser/validator including periodic-vs-event-driven sampling invariants; vendored WMM2025 **and** WMMHR2025 source/coefficients/error models with provenance, licences, hashes, one iOS C-interop wrapper, one Android NDK/JNI wrapper, each model's own vectors; `GeomagneticModelUncertainty` from the vendored error model plus `boundFromSigma` and tests; positive-down inclination conversion and tests; staged reference precheck/resolution/final-classification; uncertainty composition producing both bounds; deviation-correction types with production state fixed to `NONE`; certification-key construction including exact runtime heading/location identities, engine-decision hash, placement profile, coefficient and error-model hashes, with miss → `CANDIDATE`; complete 24-sector Feng Shui loader, geometry derivation, classifier/straddle sets including ambiguity-reference tests; canonical telemetry codec under §22.2.
*Exit:* all pure tests pass on both platforms; identical fixtures agree within `1e-6°` for angle math and the declared WMM tolerance; `359/1`, negatives, `360 → 0`, `(-180,180]` antipode, nonfinite, quaternion order/direction, REFERENCE_ENU tagging, and sector boundaries pass; estimator parity holds across platforms and `analysis/`; both platforms execute the same hashed NOAA C with no home-grown model; shared schemas/config/fixtures frozen as `fixtures-v1` before platform agents split.

**Phase 2 — provider adapters.** *Android:* FOP and `TYPE_ROTATION_VECTOR` requested from `periodicOrientationRequestedHz`; FLP fresh/current plus foreground updates; framework location for the explicit no-GMS variant; capability resolver exposing heading **and location** provider IDs, emitting `capability_resolution`, never silently changing labels; RV `values[4]` heading-accuracy capture when available; API-33+ `TYPE_HEADING` diagnostic when present; no custom fusion; AndroidX lifecycle/coroutine cleanup. *iOS:* Core Location heading/location; Core Motion `.xTrueNorthZVertical` at the requested periodic interval with frame read-back; adapter-normalize Core Motion native attitude to canonical device→REFERENCE_ENU and retain native provenance; treat CLHeading as event-driven absolute-heading anchors, not periodic support; capture calibration, timestamps, authorization, pose, charging, thermal; §12 discontinuity inference; actor ownership and cleanup. *Both:* diagnostic adapters use platform transforms only; explicit unsupported/permission/failure states including `UNSUPPORTED_DEVICE` for out-of-scope form factors; portrait lock.
*Exit:* physical debug builds stream canonical events; backgrounding, cancellation, rotation, permission change, repeated start/stop leak no subscriptions and no cross-session callbacks; simulator builds work with replay only and are visibly marked nonphysical; no custom AHRS or silent fallback.

**Phase 3 — engine, telemetry, replay.** Serialized event consumption; provider/mode-applicable freshness and sampling gates (including CLHeading anchors vs periodic support), location, pose, stability, resultant-length, field-strength, thermal, lifecycle gates; acyclic Google reference precheck→resolution→pipeline-agreement→final-classification order with explicit iOS/RV reference contracts; positive-down WMM inclination comparison; both mode projections including AND-RV wall `+Z`; pose-valid same-axis transform checks; active-axis cross-pipeline sets; flat-only FOP scalar-error gating and wall-specific floors; lock calculation and composition with lock/degraded/invalid distinction; magnetic features/classifier with absent-pipeline→UNKNOWN; configured extreme-space-weather refusal; Sensor Health/`TrustAction`, flat-only FOP `PROVIDER_INITIALIZING`, manual calibration entry; target guidance; stable reason codes/transitions; lossless JSONL with complete certification provenance, manifests/hashes/export; deterministic replay provider and cross-platform fixtures.
*Exit:* fixtures replay deterministically; out-of-order, duplicate, stale, dropped, discontinuity tests pass; a timeout never freezes the last number as a lock; cross-platform outcomes match within declared tolerances; every lock traces to heading/location provider samples, placement profile hash, config hash, WMM coefficient/error-model hashes, ruleset hash, certification-lookup result, and decision features.

**Phase 4 — UI.** Precision screen: heading, reference, bound with calibration-aware labelling, instrument/placement split, level, state, trust action, `gradeLimitedBy`, actionable rejection reason; straddle as a primary layout; target guidance with provisional-vs-locked semantics; calibration supporting automatic and manual entry, validating outcomes rather than animation, distinguishing initialization; Sensor Health with plain-language action plus advanced diagnostics; practitioner guidance from §18.1; benchmark screen from §29.7; trial manifest entry, ground-truth input, capture and export; accessibility labels and localization-safe display with locale-independent machine exports.
*Exit:* UI tests cover every `MeasurementState` and rejection reason; north-wrap animation takes the shortest path; displayed precision never exceeds the bound; **the UI cannot display a certified grade while `CANDIDATE`, verified by test**; ground truth cannot be populated from the phone result; debug/export does not change the production variant's provider configuration; an exported session validates and replays.

**Phase 5 — comparison benchmark.** Freeze `precision-v1-candidate-1`. Run `AND-G0/G1` and `IOS-A0/A1` in both modes on the §29 minimum procedure; run `AND-GE`, `IOS-AE`, and `AND-HDG` only in flat mode, with `AND-HDG` raw-68% and converted-95% fields kept distinct; log `AND-AM`/`IOS-CM-FLAT` as diagnostics; run `AND-RV` and framework location on the no-GMS matrix; run 50/100/200 Hz as separate randomized trials. Run the model factor in **both** effect paths: cross with `AND-RV` and with `AND-G1` restricted to `TRUE_CORRECTED_FROM_MAGNETIC` for the direct effect (including `TRUE_VERIFIED` there dilutes the arm with bearings that mathematically cannot move), and with every field-residual candidate — `IOS-A1` and `AND-G1/TRUE_VERIFIED` included — for the indirect gating effect, measured as accept/reject and sensitivity/specificity change rather than heading change. This does not double the scorecard; the indirect arm reports gating metrics only. Re-calibrate interference thresholds against the selected model — thresholds tuned under one model are not evidence for the other. Run clean static, interference, tilt, wrap, step response, permission, and the §30.5 reference challenge in both modes. Run the §29.5 placement study, §29.3 characterization (freezing an experimental profile only if structure repeats, evaluated on held-out data under §30.6 and §32.4), §29.4 target/freehand tests, and both calibration paths including deferral. Generate the scorecard from immutable telemetry.
*Exit:* the §30.1 matched-acceptance rule selects a winner without accepted-only selection bias; training/held-out partitions fixed before tuning; failed/rejected/timed-out trials included; shipped placement bounds trace to measured repeatability, not defaults; the chosen configuration and every changed threshold get a new version/hash; no certified label while the combination is `CANDIDATE`; any certified profile has a frozen hash and passed transfer/harm gates, else production stays `NONE`.

**Phase 6 — certification.** Establish the Tier 2 reference, qualified sites, multi-unit matrix, and budget; run the complete static/dynamic/environmental/battery/thermal programme; freeze and evaluate on held-out units, sessions, days, sites; publish per-model supported/degraded/unsupported tiers and floors keyed by the complete §24 key; decide and document whether the no-GMS build ships, carrying its own certified list under §30.4.
*Exit:* every hard gate in §30 and §35 passes for each certified model, each shipping provider path, each certified mode; raw data, manifests, analysis environment, exclusions, hashes, reports archived; no unresolved Critical/High false-confidence defect.

**Phase 7 — expansion.** Only after the engine is validated: expand the Luo Pan ruleset, each as its own hashed file with fixtures; saved records with privacy/export controls, storing ruleset version and reference selection; maps as a consumer of canonical heading/location; notes/photos if authorized; `CAMERA_SIGHT`/AR only as a **new measurement mode** with its own axis, alignment uncertainty, benchmark, and certification — it does not inherit `FLAT_TOP_EDGE` accuracy; tablet/foldable support only through the full §2.1 programme; external instruments behind the existing provider interface. Product features MUST NOT change the certified heading path without a named regression variant and renewed evidence.

**Phase 8 — continuous regression.** §33.1 per commit, §33.2 per release candidate, §33.3 recertification on the listed triggers.

### 36.1 Status vocabulary

"Implementation complete" means Phases 0–4 pass on both platforms. It does **not** mean accuracy-certified; that means Phases 5–6 pass with retained physical evidence. An agent MUST report exactly one of:

```text
SCAFFOLDED                    projects and schemas exist; provider/core behavior incomplete
IMPLEMENTED_UNCERTIFIED       both apps and benchmark export work; physical gates incomplete
CERTIFIED_FOR_LISTED_DEVICES  full evidence exists only for explicitly listed model/SKU/OS combinations
```

Never use `complete`, `production-ready`, `professional-grade`, or `certified` without the corresponding exit evidence.

---

## 37. Agent operating rules

1. Inspect the repository, toolchains, and local instructions before editing.
2. State which phase you are implementing.
3. Map each requirement to concrete files and tests; do not answer with architecture prose.
4. Prefer the §2.3 reuse-first tool and verify the **installed** API signature — if a symbol here has changed, use the current non-deprecated equivalent preserving behaviour and record it in `docs/IMPLEMENTATION_NOTES.md`. Do not silently change architecture.
5. Implement the smallest complete vertical slice that preserves final module boundaries.
6. Run build, unit, schema, and replay tests after each phase.
7. Keep platforms aligned through shared fixtures, not copied assumptions.
8. Record every API substitution, missing capability, and deviation.
9. Leave maps, signing, and optional integrations feature-gated if credentials are unavailable; they must not block the heading core.
10. Never commit secrets, API keys, device identifiers, or raw private locations.
11. Never mark physical accuracy, battery, or thermal acceptance as passed from simulator results.
12. **Never add a certification record, widen a key range, or edit a golden fixture to make a test pass.** A failing gate is a finding, not an obstacle.
13. Finish with changed files, exact commands and results, deviations, remaining physical tests, and one §36.1 status.

Do not hide incomplete work behind placeholders — a deliberate unsupported state with a reason beats a production `TODO`, fake value, or silent fallback.

### 37.1 Required implementation handoff artifacts

An agent MUST NOT report `IMPLEMENTED_UNCERTIFIED` until the handoff includes: compiling native projects for the implemented platforms; pinned dependency/lock files; pure-core, schema, WMM-vector and replay tests; provider adapters with teardown/error handling; internal benchmark capture/export; a Precision UI showing reference, **total** bound, candidate/calibrated status, state/action/reason; target guidance; Sensor Health and user-invoked `Check / Recalibrate`; exact build/test commands and results; `IMPLEMENTATION_NOTES.md` deviations/unsupported capabilities; and a statement of remaining physical tests. Physical accuracy, battery, or thermal gates are never marked passed without retained real-device telemetry and the generated scorecard.

**Conflict priority:** (1) never emit a confidently wrong or mislabelled heading; (2) preserve provider fusion, no custom AHRS; (3) follow `MUST`/`MUST NOT`; (4) follow the §2 fixed decisions; (5) use versioned candidate constants until evidence replaces them; (6) prefer a visible degraded state over an undocumented assumption.

**Multi-agent handoff.** When separate agents implement iOS and Android, one integration owner completes Phases 0–1 and freezes the contract before platform work diverges: commit schemas, config, canonical types, angle/quaternion vectors, WMM vectors, replay fixtures, rule boundaries, and vendored NOAA hashes; tag that baseline `fixtures-v1`; give both agents the same files and tolerances, with neither editing shared fixtures unilaterally; a contract change requires one reviewed change set, regenerated fixtures, both test suites, and a new fixture version; cross-platform CI compares serialized outputs and decisions before either agent reports Phase 1 or Phase 3 complete. This ordering is mandatory because independently translated constants, C ports, quaternion conventions, quantile estimators, and JSON defaults are a common source of false parity.

### 37.2 Reproducibility package

Every benchmark release ships: test-plan version with prespecified hypotheses and gates; source commit and lockfiles; binaries and symbol identifiers; config/feature-flag snapshots; WMM coefficients, error models, and all hashes for every model used; Feng Shui ruleset and hash; device/OS/sensor inventory; site survey, fixture drawings, calibration certificates, photos, and the budget with coverage factors; environmental log and randomization seed; raw immutable telemetry and file hashes; an exclusion log with reason and author (never silently delete failed attempts); analysis code, environment, bootstrap seed, and generated report; golden replay datasets and expected outputs; known deviations and operator notes.

```yaml
protocol: static-sweep-v3
truth_reference: survey-line-2026-04
truth_tier: TIER_2
truth_expanded_uncertainty_deg: 0.18
truth_coverage_factor: 2
fixture: nonmag-jig-02
headings_deg: [0, 15, 30, 45, 60, 75, 90, 105, 120, 135, 150, 165,
               180, 195, 210, 225, 240, 255, 270, 285, 300, 315, 330, 345]
repetitions: 10
measurement_mode: FLAT_TOP_EDGE
placement_method: NONMAGNETIC_ALIGNMENT_JIG
settle_s: 5
capture_s: 10
randomization_seed: 834221
config_version: precision-v1-candidate-1
config_hash: sha256:...
```

`config_version` and `config_hash` MUST match the telemetry envelope for every event in the run. One identifier for the acceptance configuration — not a separate manifest name that can drift from the telemetry name.

---

## 38. References

Official platform documentation is normative **only for the exact SDK version a release pins**. Implementation agents MUST verify the installed toolchain rather than assuming a web page is unchanged. Archive upstream packages, coefficients, error models, vectors, and hashes with the reproducibility package.

**Apple.** [Getting heading and course information](https://developer.apple.com/documentation/corelocation/getting-heading-and-course-information) · [`CLHeading`](https://developer.apple.com/documentation/corelocation/clheading) · [`CLLocationManager`](https://developer.apple.com/documentation/corelocation/cllocationmanager) · [`headingOrientation`](https://developer.apple.com/documentation/corelocation/cllocationmanager/headingorientation) · [`accuracyAuthorization`](https://developer.apple.com/documentation/corelocation/cllocationmanager/accuracyauthorization) · [`CLLocationAccuracy`](https://developer.apple.com/documentation/corelocation/cllocationaccuracy) · [`ellipsoidalAltitude`](https://developer.apple.com/documentation/corelocation/cllocation/ellipsoidalaltitude) · [Processed device-motion data](https://developer.apple.com/documentation/coremotion/getting-processed-device-motion-data) · [`CMMotionManager`](https://developer.apple.com/documentation/coremotion/cmmotionmanager) · [`CMAttitudeReferenceFrame`](https://developer.apple.com/documentation/coremotion/cmattitudereferenceframe) · [`availableAttitudeReferenceFrames()`](https://developer.apple.com/documentation/coremotion/cmmotionmanager/availableattitudereferenceframes%28%29) · [`attitudeReferenceFrame`](https://developer.apple.com/documentation/coremotion/cmmotionmanager/attitudereferenceframe) · [`CMCalibratedMagneticField`](https://developer.apple.com/documentation/coremotion/cmcalibratedmagneticfield) · [`ProcessInfo.ThermalState`](https://developer.apple.com/documentation/foundation/processinfo/thermalstate-swift.enum) · [Analyzing battery use](https://developer.apple.com/documentation/xcode/analyzing-your-app-s-battery-use)

**Google / Android.** [`FusedOrientationProviderClient`](https://developers.google.com/android/reference/com/google/android/gms/location/FusedOrientationProviderClient) · [`DeviceOrientationRequest`](https://developers.google.com/android/reference/com/google/android/gms/location/DeviceOrientationRequest) · [`DeviceOrientation`](https://developers.google.com/android/reference/com/google/android/gms/location/DeviceOrientation) · [`LocationServices`](https://developers.google.com/android/reference/com/google/android/gms/location/LocationServices) · [`FusedLocationProviderClient`](https://developers.google.com/android/reference/com/google/android/gms/location/FusedLocationProviderClient) · [`CurrentLocationRequest`](https://developers.google.com/android/reference/com/google/android/gms/location/CurrentLocationRequest) · [Position sensors](https://developer.android.com/develop/sensors-and-location/sensors/sensors_position) · [`SensorManager`](https://developer.android.com/reference/android/hardware/SensorManager) · [`SensorEvent`](https://developer.android.com/reference/android/hardware/SensorEvent.html) · [`LocationManager`](https://developer.android.com/reference/android/location/LocationManager) · [Location permissions](https://developer.android.com/develop/sensors-and-location/location/permissions) · [`GeomagneticField`](https://developer.android.com/reference/android/hardware/GeomagneticField) (cross-check only) · [`PowerManager`](https://developer.android.com/reference/android/os/PowerManager.html) · [Battery profiling](https://developer.android.com/topic/performance/power/setup-battery-historian)

**Geomagnetic and space weather.** NOAA NCEI [World Magnetic Model](https://www.ncei.noaa.gov/products/world-magnetic-model) · [WMM High Resolution](https://www.ncei.noaa.gov/products/world-magnetic-model-high-resolution) · [Accuracy limitations and error model](https://www.ncei.noaa.gov/products/world-magnetic-model/accuracy-limitations-error-model) — the source of the one-sigma value feeding `boundFromSigma` · NOAA SWPC [planetary K-index](https://services.swpc.noaa.gov/products/noaa-planetary-k-index.json)

**Ground truth and uncertainty.** NREL [Solar Position Algorithm](https://midcdmz.nrel.gov/spa/) · [GeographicLib geodesics](https://geographiclib.sourceforge.io/html/python/code.html) · IGN [Géoplateforme](https://www.ign.fr/geoplateforme) and [geodetic documentation](https://geodesie.ign.fr/linformation-geodesique) (a rendered map is not survey control — use the source dataset's accuracy metadata) · JCGM [GUM 100:2008](https://doi.org/10.59161/JCGM100-2008E) · NIST [TN 1297](https://www.nist.gov/pml/nist-technical-note-1297). This project still requires empirical held-out coverage before converting its safety envelope into a probabilistic `95%` claim.

**Data contracts.** [JSON Schema 2020-12](https://json-schema.org/draft/2020-12) · [FIPS 180-4](https://csrc.nist.gov/pubs/fips/180-4/upd1/final) for SHA-256.

**Supporting literature — non-normative.** These support the premise that smartphone heading exhibits device-specific error and degrades under magnetic disturbance. They do **not** validate this project's thresholds, bounds, corrections, or claims. Novakova & Pavlis (2017), [*J. Struct. Geol.* 97, 93–103](https://doi.org/10.1016/j.jsg.2017.02.015) — apparently stable smartphone behaviour can still produce large azimuth errors. Fan, Li & Liu (2018), [*Sensors* 18(1), 76](https://doi.org/10.3390/s18010076) — systematic treatment of magnetic-disturbance effects. Ettlinger, Wieser & Neuner (2024), [*NAVIGATION* 71(1)](https://doi.org/10.33012/navi.632) — explicit anomaly detection and evaluation against accurate ground truth.

**Comparative products — non-normative.** Product pages and practitioner tools may be reviewed for UX hypotheses, but this specification does not rely on their advertised features or accuracy claims. Any external product behavior adopted here must first be restated as a project requirement and validated by this benchmark.

This project adopts external **questions**, not external accuracy claims. A measured residual pattern MUST NOT enter the production path merely because it can be fitted: any correction requires independent ground truth, training/held-out separation, repeated sessions/days/sites, scope identification, harm analysis, matched-acceptance improvement, uncertainty recalibration, and §24 certification. Unknown local distortion remains a rejection problem, not a compensation opportunity.

---

## 39. Final principle

The compass succeeds when it produces an accurate repeatable heading **and** refuses to certify measurements the evidence cannot support. Heading error, uncertainty coverage, false-accept rate, repeatability, and held-out transfer are the truth. Smooth animation, an exact-looking target digit, sample count, provider prestige, a fitted correction curve, and agreement between correlated APIs are not.

A practitioner holding a phone against a door frame adds a physical-alignment term whose size must be measured, not assumed. In the current candidate profile that term controls several grades. The honest product is therefore not the one with the smallest sensor-only error — it is the one that tells the practitioner exactly which mountain the **total measured bound** can and cannot distinguish, and what evidence or measurement method would have to change to distinguish more.
