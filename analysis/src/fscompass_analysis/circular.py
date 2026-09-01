"""SPEC.md §9 deterministic utilities and §9.1 pinned estimators — analysis runtime.

This module is the **single allowlisted home** for the signed circular difference in the
analysis runtime (§33.1, R67/R68). ``shortest_signed_difference_deg`` below is the one
implementation; ``shortest_target_delta_deg`` and ``absolute_circular_difference_deg`` are
exact delegating wrappers and contain no angle arithmetic of their own. No other file in
``analysis/`` may restate the formula, and ``tests/test_single_implementation_audit.py``
enforces that across all three runtimes.

Two ``atan2`` call sites live here and are the ones §33.1 allowlists: the signed-difference
implementation, and the circular mean/resultant (a bearing projection, a different
quantity).
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import Sequence


class AngleError(ValueError):
    """A nonfinite or otherwise invalid angular input.

    SPEC.md §5: ``0``, ``-1``, ``NaN`` and ``null`` are not interchangeable. An invalid
    angle raises rather than being silently normalized into a plausible bearing.
    """


class UndefinedResult(ValueError):
    """SPEC.md §9.1: the typed ``UNDEFINED`` outcome.

    ``quantile`` and ``median`` on empty input "MUST return a typed ``UNDEFINED``/validation
    failure and MUST NOT index element zero".
    """


def _require_finite(value: float, what: str) -> float:
    numeric = float(value)
    if not math.isfinite(numeric):
        raise AngleError(f"{what} requires a finite value, got {value!r}")
    return numeric


def normalize360(deg: float) -> float:
    """SPEC.md §9: ``((x % 360) + 360) % 360`` with a finite check; exactly ``360.0`` → ``0.0``.

    Written with the explicit double-modulo because the language remainder operator differs
    for negative operands (failure mode 2). Python's ``%`` already returns a non-negative
    result for a positive divisor, but the mandated form is kept verbatim so all three
    runtimes perform the same IEEE-754 operations in the same order — see
    ``docs/IMPLEMENTATION_NOTES.md`` F-1 for the ~1e-10 residual this costs and why it is
    not special-cased away.
    """
    value = _require_finite(deg, "normalize360")
    wrapped = ((value % 360.0) + 360.0) % 360.0
    return 0.0 if wrapped == 360.0 else wrapped + 0.0


def shortest_signed_difference_deg(a: float, b: float) -> float:
    """SPEC.md §9/§3: ``a - b`` as the shortest signed circular difference in ``(-180, 180]``.

    The ``atan2`` convention, **and** the mandatory antipode normalization: raw
    ``atan2(sin(a-b), cos(a-b))`` returns ``-180.0`` whenever ``sin`` evaluates to a tiny
    negative rather than exactly zero, which is what happens for ordinary inputs such as
    ``a=0, b=180`` and ``a=90, b=270``. An exact ``-180.0`` is therefore mapped to ``+180.0``
    before returning, so the antipode is ``+180`` and never ``-180`` and bias statistics stay
    deterministic.

    This is the one normative contract in the spec and the only implementation in this
    runtime.
    """
    left = _require_finite(a, "shortestSignedDifferenceDeg")
    right = _require_finite(b, "shortestSignedDifferenceDeg")
    radians = math.radians(left - right)
    delta = math.degrees(math.atan2(math.sin(radians), math.cos(radians)))
    return 180.0 if delta == -180.0 else delta


def shortest_target_delta_deg(current: float, target: float) -> float:
    """SPEC.md §9: ``shortestSignedDifferenceDeg(target, current)``; positive = clockwise.

    A thin delegating wrapper with the exact §9 definition. No local alias, no independent
    angle math (R68).
    """
    return shortest_signed_difference_deg(target, current)


def absolute_circular_difference_deg(a: float, b: float) -> float:
    """SPEC.md §9: ``abs(shortestSignedDifferenceDeg(a, b))``, range ``[0, 180]``.

    A thin delegating wrapper (R68).
    """
    return abs(shortest_signed_difference_deg(a, b))


@dataclass(frozen=True)
class CircularAggregate:
    """The result of aggregating a window of angles under SPEC.md §15.

    ``mean_deg`` is ``None`` exactly when the mean is ``UNDEFINED``: an empty window, or a
    resultant of exactly zero, where ``atan2(0, 0)`` returns ``0.0`` on every platform and
    would disguise a bimodal set as a north-facing measurement (failure mode 6).

    ``resultant_length`` is reported even when the mean is undefined, because §15's gate is
    stated on ``R`` and the engine records the feature either way.
    """

    mean_deg: float | None
    resultant_length: float
    count: int

    @property
    def is_defined(self) -> bool:
        return self.mean_deg is not None


def circular_aggregate(samples: Sequence[float]) -> CircularAggregate:
    """SPEC.md §15 circular mean and resultant length with **uniform weights**.

    ``w_i = 1`` for every accepted sample. Weighting by provider error, recency or
    dispersion is a plausible and untested improvement; §15 makes it a named benchmark
    variant, not a quiet implementation choice. There is no trimming here either — rejection
    happens at the per-sample gate, before entry.
    """
    values = [_require_finite(sample, "circularAggregate sample") for sample in samples]
    count = len(values)
    if count == 0:
        return CircularAggregate(mean_deg=None, resultant_length=0.0, count=0)

    cosine = sum(math.cos(math.radians(value)) for value in values) / count
    sine = sum(math.sin(math.radians(value)) for value in values) / count
    resultant = math.hypot(cosine, sine)
    # A resultant may exceed 1 only by floating-point rounding; §6 declares the range [0,1].
    resultant = min(1.0, max(0.0, resultant))
    if cosine == 0.0 and sine == 0.0:
        return CircularAggregate(mean_deg=None, resultant_length=resultant, count=count)
    mean = normalize360(math.degrees(math.atan2(sine, cosine)))
    return CircularAggregate(mean_deg=mean, resultant_length=resultant, count=count)


def circular_mean_deg(samples: Sequence[float]) -> float:
    """SPEC.md §9 ``circularMeanDeg``; raises :class:`UndefinedResult` for ``UNDEFINED``."""
    aggregate = circular_aggregate(samples)
    if aggregate.mean_deg is None:
        raise UndefinedResult(
            "circularMeanDeg is UNDEFINED: "
            + ("empty window" if aggregate.count == 0 else "zero resultant (CIRCULAR_MEAN_UNDEFINED)")
        )
    return aggregate.mean_deg


def circular_resultant_length(samples: Sequence[float]) -> float:
    """SPEC.md §9 ``circularResultantLength`` → ``[0, 1]``."""
    return circular_aggregate(samples).resultant_length


def circular_mean_is_undefined(
    aggregate: CircularAggregate, min_circular_resultant_length: float
) -> bool:
    """SPEC.md §15 decision 3: "A weak resultant is an explicit failure."

    The exactly-degenerate ``atan2(0, 0)`` case is only half the problem, and the smaller
    half: for an antipodal pair the two sines cancel to ``6.1e-17`` rather than to zero, so
    the mean is *numerically* defined and comes back as a confident, completely arbitrary
    bearing. Only the configured ``minCircularResultantLength`` gate catches that, which is
    why this decision reads a config key and never an epsilon invented here.

    Callers emit ``CIRCULAR_MEAN_UNDEFINED`` and reject when this returns ``True``.
    """
    return (
        not aggregate.is_defined
        or aggregate.resultant_length < min_circular_resultant_length
    )


def quantile(values: Sequence[float], q: float) -> float:
    """SPEC.md §9.1 pinned **nearest-rank** estimator.

    ``quantile(x, q) = x[min(n-1, max(0, ceil(q*n) - 1))]`` over ``x`` sorted ascending.
    Common library estimators disagree by a full sample position at these window sizes, so
    the estimator is pinned rather than inherited, and all three runtimes plus ``analysis/``
    use this one definition so a device-computed P95 and a report-computed P95 of the same
    window agree exactly.
    """
    if not 0.0 <= q <= 1.0:
        raise ValueError(f"quantile probability must lie in [0, 1], got {q!r}")
    ordered = _sorted_finite(values, "quantile")
    n = len(ordered)
    index = min(n - 1, max(0, math.ceil(q * n) - 1))
    return ordered[index]


def median(values: Sequence[float]) -> float:
    """SPEC.md §9.1: odd ``n`` → the middle element; even ``n`` → the mean of the two middles."""
    ordered = _sorted_finite(values, "median")
    n = len(ordered)
    if n % 2 == 1:
        return ordered[(n - 1) // 2]
    return (ordered[n // 2 - 1] + ordered[n // 2]) / 2.0


def _sorted_finite(values: Sequence[float], what: str) -> list[float]:
    if len(values) == 0:
        raise UndefinedResult(
            f"{what} is UNDEFINED on empty input; §9.1 forbids indexing element zero"
        )
    # Nonfinite members are rejected *before* sorting (§9.1) — NaN would otherwise corrupt
    # the ordering silently rather than failing.
    return sorted(_require_finite(value, f"{what} sample") for value in values)


def circular_residuals_deg(samples: Sequence[float], mean_deg: float) -> list[float]:
    """Absolute residuals about an accepted circular mean, in sample order (§19)."""
    return [absolute_circular_difference_deg(sample, mean_deg) for sample in samples]


def circular_residual_quantile_deg(samples: Sequence[float], q: float) -> float:
    """SPEC.md §9/§9.1: the linear estimator applied to absolute residuals about the mean.

    §15 fixes the sample set: **all** accepted samples, no trimming, so the dispersion gate
    and the dispersion-derived bound cannot disagree about which samples exist.
    """
    mean = circular_mean_deg(samples)
    return quantile(circular_residuals_deg(samples, mean), q)


def bound_from_sigma(sigma_1_deg: float, sigma_to_bound_95_factor: float) -> float:
    """SPEC.md §19.2: the single named conversion from one sigma to a 95% bound.

    ``boundFromSigma(sigma1) = declinationSigmaToBound95Factor * sigma1``. The candidate
    factor ``1.96`` is the Gaussian two-sided 95% multiplier — a **modelling assumption**,
    not a property of the model, which is why it lives in versioned configuration and is
    passed in here rather than written as a literal.
    """
    sigma = _require_finite(sigma_1_deg, "boundFromSigma sigma")
    factor = _require_finite(sigma_to_bound_95_factor, "boundFromSigma factor")
    if sigma < 0.0:
        raise AngleError(f"boundFromSigma requires a non-negative sigma, got {sigma_1_deg!r}")
    if factor <= 0.0:
        raise ValueError(
            f"declinationSigmaToBound95Factor must be positive, got {sigma_to_bound_95_factor!r}"
        )
    return factor * sigma
