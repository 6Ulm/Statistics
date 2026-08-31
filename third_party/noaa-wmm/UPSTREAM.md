# Vendored NOAA World Magnetic Model — provenance record

SPEC.md §2.3 (reuse hierarchy level 2), §10, and the §35 checklist require this project to
compile the **same official NOAA C sources** on both platforms, with coefficients, the
separately published error model, licences, upstream URL and date, and a SHA-256 for each
file. §10: "Do not write separate Swift and Kotlin ports of the spherical-harmonic core."

## Status: NOT_VENDORED

Nothing under `src/`, `coefficients/`, `error-model/` or `LICENSES/` has been vendored yet,
and `sha256.txt` is correspondingly empty. This is a **Phase 1** deliverable
(SPEC.md §36: "vendored WMM2025 **and** WMMHR2025 source/coefficients/error models with
provenance, licences, hashes"); Phase 0 establishes the directory boundary and this record.

Two independent reasons the artifacts are absent rather than approximated:

1. **Phase order.** §36 places WMM wrapping in Phase 1, after the shared fixtures freeze.
2. **Network policy.** `www.ncei.noaa.gov` is not reachable from the environment this
   scaffold was built in (the egress proxy denies the CONNECT). See
   `docs/IMPLEMENTATION_NOTES.md` deviation D-2.

No coefficient value, no error-model formula and no hash has been written from memory.
§10.3 is explicit that an implementation which "derives a sigma from the coefficients, or
substitutes a remembered global constant, has invented the quantity", and §5 forbids
interchanging *missing* with *zero* or with a plausible-looking value. `scripts/verify-artifacts.sh`
reports `NOT_VENDORED` for exactly this state and fails the moment a file appears whose hash
is absent from or disagrees with `sha256.txt`.

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
  state rather than extrapolation.
- One iOS C-interop wrapper and one Android NDK/CMake + JNI wrapper over the *same* sources.
- `GeomagneticModelUncertainty` carrying `sourceConfidenceLevel = ONE_STANDARD_DEVIATION`,
  with `boundFromSigma` as the single conversion site (§19.2). The published one-sigma values
  are *test oracles for the pinned 2025 artifacts*, not literals to carry into a future epoch.

## Rules that outlive this file

- The vendored artifacts are inputs, never edited to make a test pass (§37 rule 12).
- `geomagneticCoefficientHash` and `geomagneticErrorModelHash` are separate components of the
  §24 certification key: changing either changes the key and invalidates prior certification.
- Changing the model re-tunes the §16 magnetic thresholds by construction (§10.1), which
  requires re-running §30.3 rather than carrying the old calibration forward.
