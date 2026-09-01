#!/usr/bin/env bash
# SPEC.md §36 Phase 0 exit: "no dynamic versions; fake/replay providers are debug-only."
# Plus the §10 / §35 requirement that every vendored NOAA artifact is hashed.
#
# Every check below either passes, or fails with the specific rule it enforces. A declared
# unsupported state (NOT_VENDORED) is reported explicitly and never silently treated as a
# pass for a check it does not satisfy.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

failures=0
fail() { echo "  FAIL: $*" >&2; failures=$((failures + 1)); }
warn() { echo "  WARN: $*"; }
ok()   { echo "  ok  $*"; }

echo "== §2: no dynamic dependency versions =="
# Gradle `1.2.+` / `latest.release` / `[1.0,2.0)`, Maven `LATEST`/`RELEASE`, snapshots, and
# Swift branch/`from:` requirements are all unpinned.
#
# Comments are stripped before scanning. SPEC.md §33.1 is explicit that "documentation and
# tests may quote a prohibited formula as text, so a blind repository-wide grep is not
# sufficient and MUST NOT reject explanatory prose" — and the version catalogue's own header
# quotes exactly these forms while forbidding them.
DYNAMIC_VERSION_RE='("[0-9][^"]*\+")|latest\.release|latest\.integration|"LATEST"|"RELEASE"|-SNAPSHOT|=[[:space:]]*"\[[0-9]|\.branch\(|branch:[[:space:]]*"|\.upToNextMajor|\.upToNextMinor|from:[[:space:]]*"'

strip_comments() {
  # Drop whole-line comments and trailing comments for `#` (TOML, properties, Python) and
  # `//` (Kotlin DSL, Swift). Line numbers are preserved so a hit can be located.
  sed -E -e 's@(^|[[:space:]])//.*$@@' -e 's@(^|[[:space:]])#.*$@@' "$1"
}

manifests=$(git ls-files \
  'android/**/*.gradle.kts' 'android/gradle/libs.versions.toml' 'android/gradle.properties' \
  'ios/Package.swift' 'ios/Package.resolved' \
  'analysis/pyproject.toml' 'analysis/requirements-lock.txt' 2>/dev/null || true)
for manifest in $manifests; do
  hits=$(strip_comments "$manifest" | grep -nE "$DYNAMIC_VERSION_RE" || true)
  if [[ -n "$hits" ]]; then
    fail "$manifest declares an unpinned version:"
    echo "$hits" >&2
  else
    ok "$manifest"
  fi
done

echo
echo "== §2: the Gradle wrapper pins an exact distribution =="
wrapper=android/gradle/wrapper/gradle-wrapper.properties
if grep -qE '^distributionUrl=.*gradle-[0-9]+\.[0-9]+(\.[0-9]+)?-(bin|all)\.zip$' "$wrapper"; then
  ok "$(grep '^distributionUrl=' "$wrapper")"
else
  fail "$wrapper does not pin an exact Gradle distribution"
fi
# §2 "Pin every dependency version" is satisfied by the exact distributionUrl; the checksum is
# the supply-chain half. Phase 1 resolved deviation D-4, so this is now a hard failure rather
# than a warning: the wrapper itself verifies the digest on every download, so a missing entry
# silently removes that check.
sha_line=$(grep '^distributionSha256Sum=' "$wrapper" || true)
if [[ -z "$sha_line" ]]; then
  fail "$wrapper has no distributionSha256Sum. The wrapper verifies the downloaded" \
       "distribution against this digest; without it the integrity check is absent (D-4)."
elif [[ ! "${sha_line#distributionSha256Sum=}" =~ ^[0-9a-f]{64}$ ]]; then
  fail "$wrapper distributionSha256Sum is not a 64-character lowercase hex SHA-256"
else
  ok "$sha_line"
fi

echo
echo "== §23 / §29.7: fake and replay providers are debug-only =="
# A ReplayHeadingProvider (§7) and any fake provider may exist only in a benchmark module or
# a test source set, never in a production source set of a shipping module.
production_offenders=$(git ls-files \
  'android/app/src/main/**' 'android/heading-core/src/main/**' 'android/fengshui-core/src/main/**' \
  'android/heading-google/src/main/**' 'android/heading-diagnostics/src/main/**' \
  'ios/HeadingCore/Sources/**' 'ios/FengShuiCore/Sources/**' \
  'ios/HeadingApple/Sources/**' 'ios/HeadingDiagnostics/Sources/**' 2>/dev/null \
  | xargs -r grep -lE 'class[[:space:]]+(Replay|Fake|Mock|Stub)[A-Za-z]*Provider|struct[[:space:]]+(Replay|Fake|Mock|Stub)[A-Za-z]*Provider' || true)
if [[ -n "$production_offenders" ]]; then
  fail "fake/replay provider in a production source set:"
  echo "$production_offenders" >&2
else
  ok "no fake/replay provider in any production source set"
fi
if grep -q 'debugImplementation(project(":benchmark-mode"))' android/app/build.gradle.kts \
   && ! grep -q '^\s*implementation(project(":benchmark-mode"))' android/app/build.gradle.kts; then
  ok ":benchmark-mode is a debug-only dependency of :app"
else
  fail ":benchmark-mode must be a debug-only dependency of :app (§23, §29.7)"
fi

echo
echo "== §10 / §35: vendored NOAA WMM artifacts are hashed =="
wmm_dir=third_party/noaa-wmm
artifacts=$(find "$wmm_dir/src" "$wmm_dir/coefficients" "$wmm_dir/error-model" "$wmm_dir/LICENSES" \
              -type f ! -name 'README.md' 2>/dev/null | sort || true)
entries=$(grep -vE '^\s*(#|$)' "$wmm_dir/sha256.txt" || true)
if [[ -z "$artifacts" && -z "$entries" ]]; then
  echo "  NOT_VENDORED: no NOAA artifacts and no hash entries."
  echo "                This is the state declared in $wmm_dir/UPSTREAM.md. Vendoring the"
  echo "                official C sources, both coefficient sets and both error models is a"
  echo "                Phase 1 exit requirement; no Phase 1 gate can pass in this state."
elif [[ -z "$artifacts" ]]; then
  fail "$wmm_dir/sha256.txt lists hashes but no artifact files are present"
elif [[ -z "$entries" ]]; then
  fail "NOAA artifacts are present but $wmm_dir/sha256.txt lists no hashes"
else
  if (cd "$wmm_dir" && sha256sum -c --quiet <(grep -vE '^\s*(#|$)' sha256.txt)); then
    ok "every vendored NOAA artifact matches its recorded SHA-256"
  else
    fail "vendored NOAA artifact hashes do not match $wmm_dir/sha256.txt"
  fi
  while IFS= read -r artifact; do
    rel="${artifact#"$wmm_dir/"}"
    grep -qF "  $rel" "$wmm_dir/sha256.txt" || fail "no SHA-256 entry for $rel"
  done <<< "$artifacts"
fi

echo
echo "== §24 / §37 rule 12: no certification records exist yet =="
# A CertificationRecord exists only for CALIBRATED and is release-gated on §30 and §35
# evidence for that exact key. None can exist before Phase 6, and an agent MUST NOT add one
# to make a test pass.
cert_files=$(git ls-files | grep -iE 'certification.*(record|database)' || true)
if [[ -n "$cert_files" ]]; then
  fail "certification database artifacts present before Phase 6 evidence exists:"
  echo "$cert_files" >&2
else
  ok "no certification records; every lookup misses and yields CANDIDATE (§19.1, §24)"
fi

echo
if (( failures > 0 )); then
  echo "verify-artifacts: $failures FAILED" >&2
  exit 1
fi
echo "verify-artifacts: OK"
