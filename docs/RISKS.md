# Risks

SPEC.md §34 and §34.5 are the normative failure-mode registers, with severities and required
guards. This file records the project's live risk position: which guards exist today, which are
scheduled, and which risks the current phase does not address at all.

## The shape of the dangerous failures

§34: "Critical failures share a shape: **the output remains a plausible bearing.**" Double-applied
declination, swapped sitting/facing, magnetic labelled `TRUE`, a wall-mode quaternion with a
transposed axis — none makes the dial jump or the app crash. That is why §34 is a list of tests,
not advice, and why the Phase 0 gates were built before any measurement logic.

## Guarded in Phase 0

| Failure mode | Guard now in place |
|---|---|
| 32 — an editable config value that turns every device Professional | Schema `propertyNames` rejects any key matching `/calibrationState/i`, **and** an invariant test scans the whole document at every nesting depth in three runtimes. §19.1 requires both; a discrimination test proves the detector fires on an injected key. |
| 30 — grading on `instrumentBound95Deg`; auditing the total against jig data | The §8.1.1 reachability analysis operates only on `reportedBound95Deg` and refuses any placement method with no measured bound. §20's grade function is total over the documented half-open intervals. |
| 31 — a standalone certified grade shown while `CANDIDATE` | The §22.1 example fixture asserts `displayQualityGrade` is `null` while `boundCalibrationState` is `CANDIDATE`, and the §19.1 CALIBRATED ⇔ EMPIRICALLY_CALIBRATED invariant is checked on every emitted example. The UI-side test is Phase 4. |
| R62 — an uncertified flat-freehand example that locks | The shipped example is executable: it must compose `4.0 + 3.0 = 7.0°` and be `DEGRADED`, in both the Kotlin and Python suites. |
| R65 — a two-entry 24-Mountains excerpt shipped as the ruleset | Exact `minItems`/`maxItems` of 24 and 8 in the schema, plus derived-geometry checks that reject a truncated or hand-typed boundary list. |
| R57 — `CHARGING_STATE` emitted as `gradeLimitedBy` but absent from the enum | The §6 enum set is transcribed with `CHARGING_STATE` present; the fixture-level golden test lands with the telemetry codec in Phase 1. |
| 46 — remote threshold change without versioned telemetry | No remote configuration exists; `configHash` is in the envelope and is asserted to equal the SHA-256 of the shipped file. |
| 2 — wrong normalization; language `%` differs for negatives | `normalize360` implemented in the mandated form with a finite check and negative-zero assertions in two runtimes. See `IMPLEMENTATION_NOTES.md` F-1. |
| §37 rule 12 — adding a certification record to make a test pass | `scripts/verify-artifacts.sh` fails if any certification-database artifact exists at all; none can exist before Phase 6 evidence. |
| §23 — a fake provider compiled into a production release path | `:benchmark-mode` is a **debug-only** dependency of `:app`, asserted by both the layout test and `verify-artifacts.sh`, which also scans production source sets for `Replay*`/`Fake*`/`Mock*`/`Stub*` providers. |

## Explicitly not addressed yet

Everything else in §34 and §34.5 — including every Critical entry below — depends on code that
does not exist. Listing them keeps their absence a schedule fact:

- **1, 6** circular averaging across north, `atan2(0,0)` returning a confident false north —
  needs the §15 aggregation and the `minCircularResultantLength` gate (Phase 3).
- **5, R49, R53** quaternion order, active/passive inversion, handedness, Core Motion native-frame
  conversion, AND-RV wall `+Z` projection — needs the adapters and golden poses (Phase 1–2).
- **8, 21** declination sign/unit/date/datum errors and the `2 × declination` double-correction —
  needs the vendored NOAA model and the §30.5 reference challenge (Phase 1, 5).
- **9** confidence-level conflation (one-sigma summed with 95% terms) — needs
  `GeomagneticModelUncertainty` and the single `boundFromSigma` site (Phase 1).
- **16, 20** Android axes that do not rotate with the UI; magnetic heading labelled `TRUE` —
  needs the §11 resolver and the remapping contract (Phase 2–3).
- **18** wall facing vs sitting confused, an exact 180° error — needs mode projection (Phase 3).
- **22, 23** a fixed accessory bias that looks perfectly stable; a field rotated with normal
  magnitude — needs the multi-feature §16 classifier and a field-rotating interferer in the
  physical programme (Phase 3, 5).
- **24** a transform bug surfacing as magnetic advice the user complies with — needs
  `transformAgreementDeg` kept strictly separate from `pipelineAgreementDeg` (Phase 3).
- **26** a stale heading repainted as live after provider error — needs the lifecycle and
  `TIMED_OUT` semantics (Phase 3).
- **28, 29** treating OS error as a guaranteed bound; training and testing on the same
  device/session — needs held-out coverage (Phase 5–6).
- **34** a site pattern promoted to a portable device correction — production deviation state is
  fixed at `NONE`, but the §30.6 gates that would keep it there under pressure are Phase 5.
- **45** a regional SKU or repair changing the sensor under one marketing name — needs runtime
  sensor identity and ≥3 units per model (Phase 6).

## Standing project risks

- **The measurement is a gesture.** §1: placement can dominate clean-condition sensor error, and
  the candidate profile makes the top grades unreachable freehand. Further magnetometer work
  cannot move the grade while placement stays freehand. The product risk is building sensor
  sophistication that no practitioner can realize.
- **`CLEAN` is evidence, not proof.** §16: it means no significant disturbance was detected by the
  validated detector from the available evidence. It is not proof the local field is unbiased.
- **Agreement is not accuracy.** §15.1: pipelines share sensors, calibration and vendor fusion;
  all of them can be wrong together from one distorted magnetometer.
- **Certification ages.** §24.1: repair or component substitution invalidates historical
  expectations even under an unchanged marketing model.
- **The G5 refusal is conditional.** §10: `UNKNOWN` never triggers it, so an offline device cannot
  detect an ongoing storm. Reports MUST stratify `SpaceWeatherState` and MUST NOT claim G5
  protection for `UNKNOWN` periods.
