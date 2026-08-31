#!/usr/bin/env bash
# SPEC.md §29.7 / §30.1 / §36 Phase 5: "Generate the scorecard from immutable telemetry."
#
# This script refuses in Phase 0, deliberately and loudly. There is no exported benchmark
# telemetry, no ground-truth reference, and no device matrix, so any output it produced
# would be a fabricated accuracy claim — the exact failure §30.1 and §37 rule 11 exist to
# prevent ("Never mark physical accuracy, battery, or thermal acceptance as passed from
# simulator results").
set -euo pipefail

cat >&2 <<'MSG'
generate-scorecard: REFUSED — no benchmark evidence exists.

SPEC.md §36 places scorecard generation in Phase 5, after:
  - Phase 2 provider adapters streaming canonical events from physical devices,
  - Phase 3 lossless JSONL telemetry with complete certification provenance,
  - a §27 ground-truth reference with a documented, empirical uncertainty budget,
  - the §28 device matrix (>= 3 physical units per certified high-volume model),
  - the §29 measurement programme run on physical hardware.

§30.1 additionally requires the risk-vs-acceptance-rate sweep with cluster-bootstrap bands
and matched-acceptance comparison; a single summary number such as "94% accurate" is
prohibited. §25: "Physical devices only — simulators test UI and deterministic math, never
sensor accuracy."

Current project status: SCAFFOLDED (§36.1).
MSG
exit 2
