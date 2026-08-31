"""SPEC.md §4.1 ``analysis/``.

This package computes reports from exported telemetry and **never** changes acceptance
outcomes after collection: "a metric it cannot compute from exported fields is a telemetry
defect, not licence to recompute the decision" (§4.1).

Phase 0 hosts three things here:

* :mod:`fscompass_analysis.artifacts` — locating the repository-root shared artifacts.
* :mod:`fscompass_analysis.config_invariants` — the §8.1 enforced invariants, third runtime.
* :mod:`fscompass_analysis.grade_reachability` — the §8.1.1 grade-reachability analysis.

The §9.1 pinned quantile/median estimators, which §9.1 requires ``analysis/`` to share with
both platforms, arrive in Phase 1 alongside the platform implementations so all three land
under one set of parity fixtures.
"""

__all__ = ["artifacts", "config_invariants", "grade_reachability"]
