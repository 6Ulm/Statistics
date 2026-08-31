#!/usr/bin/env bash
# SPEC.md §36 Phase 0 exit gate, run end to end.
#
#   Exit: both skeletons build; config, ruleset, and example telemetry validate; every
#   invariant test runs and passes; no dynamic versions; fake/replay providers are debug-only.
#
# Each step prints what it covered AND what it could not cover in this environment, because
# a gate that quietly skips a platform is indistinguishable from one that passed it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

# Set to "true" on a host with the Android SDK and the Android Gradle Plugin available.
FSC_ANDROID_SDK_AVAILABLE="${FSC_ANDROID_SDK_AVAILABLE:-false}"
# Set to the swift executable on a host with a Swift toolchain.
FSC_SWIFT_BIN="${FSC_SWIFT_BIN:-}"

section() { printf '\n\033[1m== %s ==\033[0m\n' "$*"; }

section "1/5  §4.1 repository layout, §8.1 invariants, §8.1.1 reachability, §21.1 geometry (analysis runtime)"
"${FSC_PYTHON:-$REPO_ROOT/.venv-analysis/bin/python}" -m pytest analysis/tests -q

section "2/5  Schema validation (§36 Phase 0 exit: config, ruleset and example telemetry validate)"
scripts/validate-fixtures.sh

section "3/5  §2 pinned versions, §23 debug-only fake/replay providers, §10 NOAA hashes"
scripts/verify-artifacts.sh

section "4/5  Android pure-core suites (§8.1, §8.1.1, §9, §21.1)"
if [[ "$FSC_ANDROID_SDK_AVAILABLE" == "true" ]]; then
  (cd android && ./gradlew :heading-core:test :fengshui-core:test :app:assembleDebug)
else
  echo "Android SDK not declared available (FSC_ANDROID_SDK_AVAILABLE != true)."
  echo "Running the pure Kotlin/JVM suites only; :app, :heading-google, :heading-diagnostics"
  echo "and :benchmark-mode are NOT configured or built. See docs/IMPLEMENTATION_NOTES.md D-1."
  (cd android && ./gradlew :heading-core:test :fengshui-core:test -PfscIncludeAndroidModules=false)
fi

section "5/5  iOS pure-core suites (§8.1, §8.1.1, §9, §21.1)"
if [[ -n "$FSC_SWIFT_BIN" ]]; then
  (cd ios && "$FSC_SWIFT_BIN" test)
else
  echo "NOT RUN: no Swift toolchain (set FSC_SWIFT_BIN)."
  echo "ios/HeadingCore and ios/FengShuiCore, and their mirrored §8.1 / §8.1.1 / §9 / §21.1"
  echo "suites, are NOT compiled or executed by this run. See docs/IMPLEMENTATION_NOTES.md D-3."
  echo "Phase 0's exit criterion 'both skeletons build' is therefore NOT met for iOS."
fi

printf '\n\033[1mci-phase0: the steps above completed. Read section 5 before reporting a status.\033[0m\n'
