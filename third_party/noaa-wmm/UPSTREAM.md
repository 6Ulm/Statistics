# Vendored NOAA World Magnetic Model — provenance record

SPEC.md §2.3 (reuse hierarchy level 2), §10, and the §35 checklist require this project to
compile the **same official NOAA C sources** on both platforms, with coefficients, the
separately published error model, licences, upstream URL and date, and a SHA-256 for each
file. §10: "Do not write separate Swift and Kotlin ports of the spherical-harmonic core."

## Status: NOT_VENDORED

Nothing under `src/`, `coefficients/`, `error-model/` or `LICENSES/` has been vendored, and
`sha256.txt` is correspondingly empty.

This was a **Phase 0** carry-over and is now a **Phase 1 blocker**: §36 places WMM wrapping in
Phase 1, so Phase 1 cannot exit while this file reads `NOT_VENDORED`.

### Retrieval attempts

| Phase | Host | Result |
|---|---|---|
| Phase 0 | `www.ncei.noaa.gov` | egress proxy denied the CONNECT |
| Phase 1 | `www.ncei.noaa.gov`, `ncei.noaa.gov`, `www.ngdc.noaa.gov`, `noaa.gov` | all denied; the agent proxy logs `connect_rejected` / `gateway answered 403 to CONNECT (policy denial)` for each |

A third-party mirror was **not** substituted. §2.3 level 2 names the *official reference
implementation*, §10 requires "the exact NOAA package" with its own URL, date and per-file
SHA-256, and a digest computed over a mirror cannot be checked against NOAA's published one.
Vendoring a mirror and recording its self-computed hash would satisfy the letter of "there is a
hash here" while abandoning the provenance the hash exists to establish.

No coefficient value, no error-model formula and no hash has been written from memory. §10.3 is
explicit that an implementation which "derives a sigma from the coefficients, or substitutes a
remembered global constant, has invented the quantity", and §5 forbids interchanging *missing*
with *zero* or with a plausible-looking value.

### What Phase 1 built around the absence

Rather than leave the contract unwritten, the typed surface exists and **refuses**:

- `Geomagnetic.vendoredArtifacts(modelId, repoRoot)` reads this directory's `sha256.txt` and
  reports the vendored state; `requireVendored(operation)` throws
  `VendoredModelUnavailableException` naming the missing artifact and the rule that forbids
  substituting for it. `analysis/` mirrors it as `VendoredModelUnavailable`.
- `GeomagneticModelUncertainty` cannot be constructed without an `errorModelHash`, and rejects
  any `sourceConfidenceLevel` other than `ONE_STANDARD_DEVIATION` (§10.3, failure mode 9).
- `boundFromSigma` exists and is tested as the single sigma→bound conversion site (§19.2), so
  the conversion is ready the moment a real sigma exists.
- `scripts/verify-artifacts.sh` reports `NOT_VENDORED` explicitly and
  `scripts/ci-phase1.sh` exits non-zero because of it.
- The `fixtures-v1` manifest freezes this file's hash, so a later vendoring is a visible
  contract change rather than a silent one (§37.1).

## What Phase 1 must vendor

Fetch from NOAA NCEI and record the retrieval URL and UTC date beside each file:

| Directory | Contents |
|---|---|
| `src/` | The official NOAA C sources for the spherical-harmonic evaluation, unmodified. Any local change is a patch file recorded here, never an edit in place. |
| `coefficients/` | The WMM2025 coefficient file **and** the WMMHR2025 coefficient file, each with its epoch and validity interval. §2: WMM2025 is the default, WMMHR2025 is benchmarked (§10.1). |
| `error-model/` | The separately published error model for **each** coefficient set — the only permitted source of `declinationSigma1Deg` (§10.3). It is hashed separately from the coefficients because it changes reported uncertainty even when coefficient evaluation is unchanged (§24). |
| `LICENSES/` | The upstream licence and use terms for every vendored artifact. |
| `sha256.txt` | `<sha256>  <path>` for every file above, in `sha256sum -c` format. |

Also required before Phase 1 exit, from §10 and §35:

- The official NOAA test vectors for **both** models, stored under `testdata/wmm/`, passing
  in CI for every vendored coefficient set.
- `validityStartDecimalYear` / `validityEndDecimalYear` recorded per model. For WMM2025 the
  v1 epoch interval is `2025.0 <= decimalYear < 2030.0`; a date outside it must produce
  `GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE` and an explicit "model expired, update the app"
  state rather than extrapolation. (`Geomagnetic.wmm2025Validity` already encodes this
  interval, which SPEC.md states directly; the coefficients it applies to are still absent.)
- One iOS C-interop wrapper and one Android NDK/CMake + JNI wrapper over the *same* sources.
- `GeomagneticModelUncertainty` populated from the vendored error-model artifact. The published
  one-sigma values are *test oracles for the pinned 2025 artifacts*, not literals to carry into
  a future epoch.

## Rules that outlive this file

- The vendored artifacts are inputs, never edited to make a test pass (§37 rule 12).
- `geomagneticCoefficientHash` and `geomagneticErrorModelHash` are separate components of the
  §24 certification key: changing either changes the key and invalidates prior certification.
- Changing the model re-tunes the §16 magnetic thresholds by construction (§10.1), which
  requires re-running §30.3 rather than carrying the old calibration forward.
- Vendoring is a `fixtures-v1` contract change: regenerate the fixtures, run every suite, and
  bump the fixture version (§37.1).
