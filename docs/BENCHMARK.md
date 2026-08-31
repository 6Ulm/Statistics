# Benchmark

SPEC.md §25–§32 are normative. This file is the project-side index: what the benchmark must
produce, what exists today, and the ordering constraints that make the sequence non-obvious.

**Nothing in this section has been run.** No ground-truth reference, no device matrix, no
telemetry, no scorecard. `scripts/generate-scorecard.sh` refuses and says so.

## The ordering problem Phase 5 must respect

§8.1.1 "The certification bootstrap" is the part most likely to be discovered too late:

- `AND-G1` and `IOS-A1` are defined as provider **plus lock logic**, so on uncertified
  hardware their freehand acceptance rate is **zero**.
- §30.1 requires comparison at matched acceptance rate, and §30.2 requires `>= 95%` clean
  acceptance. Neither is reachable while every device is uncertified.
- Certification is what would fix the device floor; the benchmark is what produces
  certification.

The resolution is that **`deviceFloor95Deg` is an output of the benchmark, not an input**.
Phase 5 sweeps the floor as an explicit parameter alongside the provider-error threshold,
reports acceptance and risk as functions of it, evaluates lock-gate behaviour primarily with
`NONMAGNETIC_ALIGNMENT_JIG` placement (whose smaller placement term leaves a usable instrument
budget), and publishes the freehand table separately without gating on it.

Until a device class is certified, the honest product behaviour is that freehand measurements
return `DEGRADED` results with an explicit bound and no lock — stated in release notes, never
engineered around. The Phase 0 grade-reachability analysis
(`testdata/grade-reachability-claims-v1.json` plus the three runtime implementations) already
encodes this as an executable fact.

## Ground truth (§27)

Tier 0 is mandatory before buying equipment; Tier 2 is needed only for final certification, and
**per-model bias gates require Tier 2**.

- **Tier 0A** relative rotation — a non-magnetic turntable, no absolute north. Disproportionately
  valuable for its cost: a swapped quaternion component, a missing remap, an inverted sign or a
  five-second filter all fail obviously here.
- **Tier 0B** solar shadow — NREL SPA, pinned source and version, is the only accepted
  solar-position source. The uncertainty budget MUST be **empirical**: repeat the entire
  procedure at least five times and take the observed spread as the dominant term. A propagated
  budget can be made arbitrarily small by assuming a sharp edge and a perfect rod.
- **Tier 0C** long geodetic baseline — GeographicLib inverse geodesic, never a home-grown
  spherical formula; a rendered map is not survey control.
- **Tier 1** is two independent Tier 0 methods. The point is the **cross-check**, not the smaller
  number: a mistyped longitude or a UTC confusion shows as an 8° disagreement that no amount of
  repetition within one method would reveal.
- **Tier 2** SHOULD achieve `<= 0.20°` and MUST achieve `<= 0.50°` for 2–5° product claims.

Never use another phone, a car compass, a map north arrow, an unverified wall or a consumer hand
compass as absolute ground truth. Never subtract reference uncertainty from observed phone error.

## Implementations compared (§26)

`AND-G0`, `AND-GE` (flat only), `AND-G1`, `AND-RV`, `AND-AM` (diagnostic), `AND-HDG` (flat
diagnostic), `IOS-A0`, `IOS-AE` (flat only), `IOS-A1`, `IOS-CM-FLAT` (diagnostic). Rate variants
`-50/-100/-200` run as **separate randomized trials**, never pretended simultaneous.

Two conversion rules that are easy to get wrong and invalidate a whole column:

- `AND-HDG`'s accuracy field is **68% confidence (one sigma)**. Every 95%-bound comparison must
  route through `boundFromSigma` exactly once, and the raw field is kept separately (R64).
- `AND-RV`'s `values[4]` is **AOSP-documented 95%**, so it enters `providerReportedBoundTermDeg`
  **unconverted** — and because the base term is a `max`, it can never lower the device floor
  (R58, R63). When it is `-1`, the term is **absent, not zero**.

## Selection rule (§30.1)

Lexicographic, never one weighted score: (1) disqualify any variant violating false-confidence,
reference or maximum-accepted-error gates; (2) minimize severe-error risk and P95 at matched
acceptance rate; (3) then MAE, absolute bias, azimuth-binned bias; (4) then clean acceptance
within 10 s; (5) then lock time and tracking latency; (6) battery and thermal **only after every
accuracy and safety gate passes**. Report the full table even after choosing. A single score such
as "94% accurate" is prohibited.

If raw provider output is already more accurate than the app-locked value, **keep the provider
heading** and retain only the safety checks that demonstrate benefit.

## Gates

§30.2 certified clean static (device in a jig, so the numbers describe the pipeline), §30.3
interference and latency, §30.4 no-GMS (looser on precision, **identical on safety**), §30.5 the
reference challenge — which must specifically look for the `2 × declination` double-correction
signature, roughly 16° at an 8°-declination site and otherwise indistinguishable from a plausible
bearing — §30.6 deviation-correction adoption, §30.7 calibration, §30.8 power and thermal.

The azimuth-binned bias gate exists because whole-circle bias is nearly blind to the most
characteristic magnetometer defect: residual hard-iron error produces roughly sinusoidal signed
error integrating to ~zero. A device `±5°` wrong at two specific bearings passes a 2.0° bias gate
comfortably — and a practitioner measuring one wall does not average over the circle.

A failing model is not "fixed" by excluding inconvenient headings: degrade it, deny
certification, or give it a validated model-specific floor.

## Statistics (§32)

Cluster/bootstrap CIs resampling at **device-unit and session** level, never high-rate sensor
samples as if independent. Split by unit/session/site to prevent leakage; tune on training units,
freeze, report on held-out. Audit `instrumentBound95Deg` against **jig-placed** error and
`reportedBound95Deg` against **freehand** error — never the reverse, which credits the placement
term for an error that was experimentally removed (failure mode 30).

Timeouts, API errors, invalid headings and rejections are **outcomes**, not missing-at-random
samples; report their rates.

## Placement study (§29.5)

The study that fixes `flatFreehandPlacementBound95Deg` and `wallFreehandPlacementBound95Deg`,
and produces the `NONMAGNETIC_ALIGNMENT_JIG` repeatability bound without which no measurement
can reach `HIGH` or `PROFESSIONAL`. The variable is the **operator**, not the device: ≥20
placements per block, ≥4 operators including left- and right-handed, a jig control, ≥2 physical
planes. Report the spread across operators — a bound set from the best operator is not a product
bound.

The shipped values in `config/precision-profile-v1.json` are **candidates from SPEC.md §8**, not
measurements. §8.1.1: constants may change only from §29.5 placement evidence and §30 device-floor
evidence, never to make a demo work.

## Reproducibility package (§37.2)

Every benchmark release ships the test plan with prespecified hypotheses and gates, source commit
and lockfiles, binaries and symbol identifiers, config snapshots, all WMM coefficients/error
models/hashes, the ruleset and its hash, the device/OS/sensor inventory, the site survey and
uncertainty budget with coverage factors, the environmental log and randomization seed, raw
immutable telemetry with file hashes, an exclusion log with reason and author, analysis code and
seeds, golden replay datasets, and known deviations. `config_version` and `config_hash` MUST match
the telemetry envelope for every event in the run.
