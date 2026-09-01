#!/usr/bin/env bash
# SPEC.md §36 Phase 1 exit gate, run end to end.
#
#   Exit: all pure tests pass on both platforms; identical fixtures agree within 1e-6 deg for
#   angle math and the declared WMM tolerance; 359/1, negatives, 360 -> 0, (-180,180] antipode,
#   nonfinite, quaternion order/direction, REFERENCE_ENU tagging, and sector boundaries pass;
#   estimator parity holds across platforms and analysis/; both platforms execute the same
#   hashed NOAA C with no home-grown model; shared schemas/config/fixtures frozen as
#   `fixtures-v1` before platform agents split.
#
# Each step prints what it covered AND what it could not cover in this environment, because a
# gate that quietly skips a platform is indistinguishable from one that passed it. The script
# exits non-zero when a runnable gate fails; it exits non-zero at the end when a Phase 1 exit
# criterion is unmet, so "the script passed" can never be mistaken for "Phase 1 passed".
set -uo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

FSC_ANDROID_SDK_AVAILABLE="${FSC_ANDROID_SDK_AVAILABLE:-false}"
FSC_SWIFT_BIN="${FSC_SWIFT_BIN:-}"
PYTHON="${FSC_PYTHON:-$REPO_ROOT/.venv-analysis/bin/python}"

unmet=()
failures=0

section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }
run() { "$@" || failures=$((failures + 1)); }

section "1/7  Phase 0 gates still hold (§4.1 layout, §8.1, §8.1.1, §21.1, schemas)"
run scripts/validate-fixtures.sh
run scripts/verify-artifacts.sh

section "2/7  Analysis runtime: §9, §9.1, §11, §16, §19, §21, §22.2, §24, §18.2"
run "$PYTHON" -m pytest analysis/tests -q

section "3/7  §37.1 fixtures-v1 freeze is reproducible"
run "$PYTHON" scripts/generate-shared-fixtures.py --check

section "4/7  Android pure-core suites against the same frozen fixtures"
if [[ "$FSC_ANDROID_SDK_AVAILABLE" == "true" ]]; then
  (cd android && run ./gradlew :heading-core:test :fengshui-core:test :app:assembleDebug)
else
  echo "Android SDK not declared available (FSC_ANDROID_SDK_AVAILABLE != true)."
  echo "Running the pure Kotlin/JVM suites only; :app, :heading-google, :heading-diagnostics"
  echo "and :benchmark-mode are NOT configured or built. See docs/IMPLEMENTATION_NOTES.md D-1."
  (cd android && ./gradlew :heading-core:test :fengshui-core:test \
      -PfscIncludeAndroidModules=false) || failures=$((failures + 1))
  unmet+=("D-1: the Android app module was not built; §36 Phase 0's 'both skeletons build' "\
"remains unmet for the Android application target.")
fi

section "5/7  iOS pure-core suites against the same frozen fixtures"
if [[ -n "$FSC_SWIFT_BIN" ]]; then
  (cd ios && run "$FSC_SWIFT_BIN" test)
else
  echo "NOT RUN: no Swift toolchain (set FSC_SWIFT_BIN)."
  echo "ios/HeadingCore and ios/FengShuiCore have NOT been compiled or executed by this run."
  echo "See docs/IMPLEMENTATION_NOTES.md D-3."
  unmet+=("D-3: no Swift toolchain; the iOS core has never been compiled, so §36 Phase 1's "\
"'all pure tests pass on both platforms' is unmet for iOS.")
fi

section "6/7  §10 vendored NOAA WMM"
if [[ -s third_party/noaa-wmm/sha256.txt ]] && \
   grep -qv '^#' third_party/noaa-wmm/sha256.txt 2>/dev/null && \
   [[ -n "$(find third_party/noaa-wmm/coefficients -type f ! -name 'README.md' 2>/dev/null)" ]]; then
  echo "NOAA artifacts present; run their official test vectors before trusting a declination."
else
  echo "NOT VENDORED: no NOAA C sources, coefficients or error models."
  echo "§36 Phase 1 requires both models vendored with provenance, licences and hashes, an iOS"
  echo "C-interop wrapper, an Android NDK/JNI wrapper, and each model's own official vectors."
  unmet+=("D-2: ncei.noaa.gov is unreachable from this environment, so no WMM artifact was "\
"vendored and no declination, sigma or geomagnetic gate can be evaluated. §36 Phase 1 cannot "\
"exit. No coefficient, formula or hash was written from memory.")
fi

section "7/7  Phase 1 exit status"
if (( failures > 0 )); then
  echo "FAIL: $failures runnable gate(s) failed above."
fi
if (( ${#unmet[@]} > 0 )); then
  echo "Phase 1 exit criteria NOT met:"
  for item in "${unmet[@]}"; do printf '  - %s\n' "$item"; done
  echo
  echo "Status remains SCAFFOLDED (§36.1). Do not report IMPLEMENTED_UNCERTIFIED."
  exit 1
fi
if (( failures > 0 )); then exit 1; fi
echo "All Phase 1 exit criteria met."
