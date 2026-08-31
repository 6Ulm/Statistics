# Feng Shui Precision Compass

Accuracy-first native iOS + Android compass for Feng Shui practitioners, with a mobile benchmark
framework.

**`SPEC.md` is the sole normative authority.** It is v9 of the engineering specification; any
older spec file is historical, and where something conflicts with v9, v9 wins.

The design goal is a true-north bearing whose error can be **measured, bounded, and refused**.
Refusing is a feature: consumer phones cannot guarantee accurate absolute heading near steel,
magnets, conductors, vehicles or magnetic accessories, and those errors are locally
unobservable.

---

## Status: `SCAFFOLDED`

SPEC.md §36.1 defines exactly three status strings. `SCAFFOLDED` means *projects and schemas
exist; provider/core behavior incomplete*. Nothing here has run on a phone, and no accuracy,
battery or thermal claim is made.

**Phase 0 is implemented** (§36: repo, schemas, pinned tools). Phases 1–8 are not.

One consequence of the shipped candidate constants is worth stating on the front page, because
it is a product fact rather than a bug: with `unknownDeviceFloor95Deg = 4.0°` and
`flatFreehandPlacementBound95Deg = 3.0°`, an ordinary freehand measurement on an uncertified
device composes to a `7.0°` bound and therefore returns a **`DEGRADED` result with no Precision
Lock**. Wall freehand can never lock at all, at any device quality. This is §8.1.1's arithmetic,
enforced by a build-time analysis, not an implementation shortfall.

## Repository layout

Per SPEC.md §4.1. Names may follow local convention; the boundaries MUST NOT vary.

```
SPEC.md                 the normative specification (v9)
docs/                   BENCHMARK, RISKS, IMPLEMENTATION_NOTES, TESTING, PRIVACY
config/                 precision-profile-v1.json, feng-shui-rules-v1.json  (versioned + hashed)
schemas/                JSON Schema 2020-12 for config, ruleset, telemetry, session manifest
testdata/               shared fixtures read identically by all three runtimes
third_party/noaa-wmm/   vendored NOAA sources, coefficients, error models  (NOT_VENDORED — Phase 1)
android/                heading-core, fengshui-core (pure JVM) + app, heading-google,
                        heading-diagnostics, benchmark-mode (Android)
ios/                    HeadingCore, FengShuiCore (pure) + HeadingApple, HeadingDiagnostics,
                        BenchmarkMode, FengShuiCompass
analysis/               reports from exported telemetry; never changes acceptance outcomes
scripts/                validate-fixtures, verify-artifacts, generate-scorecard, ci-phase0
```

## Running the Phase 0 gate

```bash
python3 -m venv .venv-analysis
.venv-analysis/bin/pip install -r analysis/requirements-lock.txt
.venv-analysis/bin/pip install -e analysis
scripts/ci-phase0.sh
```

`docs/TESTING.md` lists each command individually and says exactly what each covers. The iOS
suites do not run without a Swift toolchain, and the Android app module does not build without
the Android SDK; `scripts/ci-phase0.sh` says so rather than skipping quietly. See
`docs/IMPLEMENTATION_NOTES.md` deviations D-1 and D-3.

## The load-bearing part of Phase 0

The §8.1 configuration invariants and the §8.1.1 grade-reachability analysis exist before any
core logic, deliberately. Each §8.1 invariant prevents one specific silent failure — most
sharply, the schema and three test runtimes all forbid any key matching `/calibrationState/i`,
because "one editable value that turns every device Professional is the shortcut an agent under
pressure takes".

The §8.1.1 analysis catches the defect class that survives coverage review: two internally
consistent sections contradicted only by arithmetic. It computes, for every
`(PlacementMethod, certification state, MagneticState)` combination the product claims to
support, whether the claimed maximum grade is arithmetically reachable, and fails when it is
not. The claim side is hand-authored from cited spec text in
`testdata/grade-reachability-claims-v1.json`; it is checked against the constants, never
generated from them.

Both are implemented three times — Kotlin, Swift, Python — against the same repository-root
artifacts, because independently translated constants are a common source of false parity
(§37.1).

## Rules that apply to every change

- Never add a certification-database record, widen a certification key range, or edit a golden
  fixture to make a test pass. **A failing gate is a finding.** (§37 rule 12)
- Never mark physical accuracy, battery or thermal acceptance as passed from simulator or
  emulator results. (§37 rule 11)
- Pin every dependency version. No `+`, `latest.release`, or unpinned branches. (§2)
- Do not build a custom AHRS, sensor fusion, location fusion, or geomagnetic model. Use the
  platform fused providers and the vendored NOAA sources. (§1, §2.3)
- Never use `complete`, `production-ready`, `professional-grade` or `certified` without the
  corresponding exit evidence. (§36.1)

---

## Pre-existing contents of this repository

This repository previously held unrelated statistics coursework, retained unchanged:

- `EMGauss/` — EM and K-means
- `HMM/` — Hidden Markov model with Viterbi algorithm
