#!/usr/bin/env bash
# SPEC.md §36 Phase 0: "CI commands for schema validation and both pure-core suites."
# Exit: "config, ruleset, and example telemetry validate."
#
# Validates every shipped JSON artifact against its JSON Schema 2020-12 schema, and asserts
# the structural constraints §8.1 and §21.1 require of the schemas themselves. Implemented
# by delegating to the analysis test suite so there is exactly one definition of what
# "validates" means, rather than a second validator that can drift from it.
set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/.." && pwd)"
cd "$REPO_ROOT"

PYTHON_BIN="${FSC_PYTHON:-$REPO_ROOT/.venv-analysis/bin/python}"
if [[ ! -x "$PYTHON_BIN" ]]; then
  echo "validate-fixtures: no analysis interpreter at $PYTHON_BIN" >&2
  echo "  create it with:  python3 -m venv .venv-analysis \\" >&2
  echo "                   && .venv-analysis/bin/pip install -r analysis/requirements-lock.txt \\" >&2
  echo "                   && .venv-analysis/bin/pip install -e analysis" >&2
  echo "  or set FSC_PYTHON to an interpreter with the pinned dependencies installed." >&2
  exit 2
fi

echo "== JSON well-formedness =="
while IFS= read -r -d '' file; do
  "$PYTHON_BIN" -c 'import json,sys; json.load(open(sys.argv[1]))' "$file" \
    || { echo "  MALFORMED: $file" >&2; exit 1; }
  echo "  ok  $file"
done < <(find config schemas testdata -name '*.json' -print0 | sort -z)

echo
echo "== Schema validation and schema-constraint tests =="
"$PYTHON_BIN" -m pytest analysis/tests/test_schemas.py \
                       analysis/tests/test_fengshui_ruleset.py \
                       analysis/tests/test_example_telemetry.py -q

echo
echo "validate-fixtures: OK"
