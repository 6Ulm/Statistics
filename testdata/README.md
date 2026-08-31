# `testdata/` — shared fixtures

SPEC.md §37.1: the integration owner freezes schemas, config, canonical types, angle and
quaternion vectors, WMM vectors, replay fixtures and rule boundaries as `fixtures-v1` before
platform work diverges, and "neither [platform agent] edit[s] shared fixtures unilaterally".
§37 rule 12: **never edit a golden fixture to make a test pass.**

Every file here is read directly from the repository root by all three runtimes — the
Android pure core, the iOS pure core, and `analysis/` — never copied into a module, so a
constant cannot be independently translated into two different values.

## Present in Phase 0

| Path | Purpose |
|---|---|
| `grade-reachability-claims-v1.json` | The **claim** side of the §8.1.1 grade-reachability analysis: what maximum grade the product claims per `(PlacementMethod, certification state, MagneticState)`, hand-authored from the cited spec text. The analysis recomputes reachability from the shipped constants and fails on any disagreement. |
| `telemetry-event-engine-output-v1.example.json` | The §22.1 worked example in the §22 envelope. It is an **executable fixture** (R62): a good-looking uncertified flat-freehand measurement that must degrade at `4.0° + 3.0° = 7.0°` and must never lock. |
| `session-manifest-v1.example.json` | The §37.2 reproducibility-package manifest shape. Its `configVersion`/`configHash` must match the telemetry envelope. |

The two example files carry `sha256:` values computed over real repository artifacts where a
real artifact exists (config, ruleset). Where the referenced artifact does not exist yet —
the NOAA coefficient and error-model hashes — the field reads the literal `NOT_VENDORED`
rather than an invented digest (§5: missing, invalid and not-supported are distinct states).
Identity-like fields that have no real counterpart outside a device run are SHA-256 digests
of documented `FIXTURE:` strings, so they are well-formed and self-identifying rather than
mistakable for a captured runtime identity.

## Empty pending later phases

| Directory | Arrives in | Contents |
|---|---|---|
| `angles/` | Phase 1 | §9 / §9.1 cross-runtime golden and property fixtures: normalization, the `(-180,180]` antipode in **both** orderings of at least two distinct pairs, `359/1`, negatives, `360 → 0`, nonfinite, and pinned quantile/median parity between both platforms and `analysis/`. |
| `quaternions/` | Phase 1 | Quaternion/matrix golden vectors for every supported orientation, REFERENCE_ENU handedness and reference tagging, axis remapping, and the Core Motion native-frame conversion proved with N/E/S/W/up vectors (§11.1, R49). |
| `wmm/` | Phase 1 | The official NOAA test vectors for WMM2025 **and** WMMHR2025, one case per altitude datum plus `UNKNOWN` (§10.2), and dates inside and outside each validity interval. |
| `fengshui/` | Phase 1 | §21.1 golden fixtures: per sector, centre, both boundaries, boundaries `± epsilon`, `± 0.1°`, `± 1.0°`, plus the north-wrap sector; §21.4 straddle sets including wide bounds and the full-circle degenerate case; §21.2 ambiguity-reference cases under both hidden Google hypotheses. |
| `replay/` | Phase 3 | Recorded telemetry replayed through the candidate engine with frozen expected outputs, plus out-of-order, duplicate, dropped, delayed and discontinuous sample cases. |

These directories are empty because the code that would consume them does not exist. §37:
"Do not hide incomplete work behind placeholders — a deliberate unsupported state with a
reason beats a production `TODO`, fake value, or silent fallback."
