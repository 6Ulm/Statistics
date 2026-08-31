# Testing

SPEC.md §33 is the normative regression strategy. This file records what runs today, how to
run it, and which required tests belong to which later phase.

## Runtimes

The §8.1 configuration invariants and the §8.1.1 grade-reachability analysis are implemented
**three times** — Kotlin, Swift, Python — against the *same* repository-root artifacts. That is
not duplication for its own sake: §37.1 requires the platforms to stay aligned through shared
fixtures rather than copied assumptions, and §9.1 requires `analysis/` to agree with both
platforms exactly. Invariant identifiers (`INV-01-…` … `INV-10-…`) and ruleset check identifiers
(`RS-01-…` … `RS-11-…`) are identical across runtimes so a failure reads the same everywhere.

| Runtime | Location | Runs today |
|---|---|---|
| Kotlin / JVM | `android/heading-core/src/test`, `android/fengshui-core/src/test` | yes |
| Python | `analysis/tests` | yes |
| Swift | `ios/HeadingCore/Tests`, `ios/FengShuiCore/Tests` | **no** — no Swift toolchain here (deviation D-3) |

## Commands

```bash
# One-time: pinned analysis environment
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r analysis/requirements-lock.txt
.venv-analysis/bin/pip install -e analysis

# The whole Phase 0 exit gate
scripts/ci-phase0.sh

# Individually
.venv-analysis/bin/python -m pytest analysis/tests -q     # layout, §8.1, §8.1.1, §21.1, schemas, R62
scripts/validate-fixtures.sh                              # JSON Schema 2020-12 validation
scripts/verify-artifacts.sh                               # pinned versions, debug-only providers, NOAA hashes
cd android && ./gradlew :heading-core:test :fengshui-core:test -PfscIncludeAndroidModules=false
cd android && ./gradlew :heading-core:test :fengshui-core:test :app:assembleDebug   # needs the Android SDK
cd ios && swift test                                      # needs a Swift toolchain
```

`scripts/generate-scorecard.sh` deliberately **refuses** in Phase 0 and explains why: there is
no benchmark telemetry, so any output would be a fabricated accuracy claim (§30.1, §37 rule 11).

## What Phase 0 covers

- **§4.1 layout** — every required file and directory; the pure cores carry no Android plugin
  and no UIKit/SwiftUI/CoreLocation/CoreMotion import; `:benchmark-mode` is debug-only.
- **§8.1** — all ten enforced invariants, including the recursive `/calibrationState/i` key scan
  at any nesting depth, plus the schema-level `propertyNames` constraint (§19.1 requires *both*
  "schema constraint plus test").
- **§8.1.1** — the grade-reachability analysis over every `(PlacementMethod, certification state,
  MagneticState)` combination the product claims to support, with the three named consequences
  asserted directly and the certified device floor swept as an explicit parameter.
- **§9** — `normalize360` only. See `docs/IMPLEMENTATION_NOTES.md` F-1.
- **§20** — the bound-to-grade function is total over the documented half-open intervals.
- **§21.1** — ruleset completeness and derived-geometry consistency; boundaries at `7.5° + 15k`;
  `352.5°` separates 壬 and 子.
- **§22 / §36 exit** — config, ruleset, example telemetry and session manifest validate.
- **R62** — the §22.1 example is an executable fixture: it must compose `4.0 + 3.0 = 7.0°`, be
  `DEGRADED`/`LOW_CONFIDENCE`/`CANDIDATE`, and never lock.
- **§2** — no dynamic dependency versions in any build manifest.

### Discrimination tests

A gate that cannot fail is not a gate. Each runtime feeds a deliberately broken **in-memory**
copy to the invariant checker, the reachability verifier and the ruleset geometry check, and
asserts the specific violation identifier comes back. No shipped artifact is ever mutated
(§37 rule 12).

## Required tests that belong to later phases

Listed so their absence is a schedule fact rather than an oversight. From §33.1:

| Test | Phase |
|---|---|
| `shortestSignedDifferenceDeg` property tests incl. **both orderings** of at least two antipodal pairs; the per-runtime single-implementation audit and `atan2` call-site scan (R67/R68) | 1 |
| §9.1 pinned quantile/median parity across both platforms and `analysis/` | 1 |
| Quaternion/matrix golden vectors; REFERENCE_ENU handedness and tagging; Core Motion native-frame conversion (R49) | 1–2 |
| NOAA official vectors for both models; error-model loading; `boundFromSigma`; cross-platform parity; all three altitude datums | 1 |
| Positive-down inclination conversion, hemisphere and `I = 0°` cases (R60) | 1 |
| Certification-key construction and miss → `CANDIDATE` | 1 |
| Feng Shui straddle sets, every boundary from both sides, wide bounds, north wrap, full-circle degenerate case; §21.2 ambiguity under both hidden Google hypotheses | 1 |
| JSON encoding under a comma-decimal locale, nonfinite rejection, round-trip precision | 1 |
| Timestamp alignment, rollover, clock-jump, stale data | 2–3 |
| "No gate compares against a numeric literal" (§18.5) | 3 |
| Recorded-telemetry replay with frozen expected outputs; out-of-order, duplicate, dropped, delayed, discontinuous fuzzing | 3 |
| Permission/location/provider state machines; `PROVIDER_INITIALIZING` distinct from `CALIBRATE` | 3 |
| UI cannot display a certified grade while `CANDIDATE`; ground truth cannot be populated from the phone result | 4 |
| §33.2 physical smoke; §30.5 reference challenge incl. the `2 × declination` signature | 5–6 |

## Physical tests that no CI run can satisfy

§25: "Physical devices only — simulators test UI and deterministic math, never sensor accuracy."
§37 rule 11 forbids marking physical accuracy, battery or thermal acceptance as passed from
simulator or emulator results. Nothing in this repository has run on a phone.
