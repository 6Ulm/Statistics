package com.fengshuicompass.headingcore.math

import kotlin.math.ceil
import kotlin.math.max
import kotlin.math.min

/**
 * SPEC.md §9.1 — the pinned quantile and median.
 *
 * Common library estimators disagree by a full sample position at these window sizes, so the
 * estimator is pinned rather than inherited: both platforms MUST be bit-identical on
 * `testdata/angles/`, and `analysis/` MUST use the same definition so a device-computed P95
 * and a report-computed P95 of the same window agree exactly (failure mode 47).
 */
public object Estimators {

    /**
     * SPEC.md §9.1 **nearest-rank**:
     * `quantile(x, q) = x[min(n-1, max(0, ceil(q*n) - 1))]` over `x` sorted ascending.
     */
    public fun quantile(values: List<Double>, q: Double): Double {
        require(q in 0.0..1.0) { "quantile probability must lie in [0, 1], got $q" }
        val ordered = sortedFinite(values, "quantile")
        val n = ordered.size
        val index = min(n - 1, max(0, ceil(q * n).toInt() - 1))
        return ordered[index]
    }

    /**
     * SPEC.md §9.1: odd `n` -> the middle element; even `n` -> the mean of the two middles.
     */
    public fun median(values: List<Double>): Double {
        val ordered = sortedFinite(values, "median")
        val n = ordered.size
        return if (n % 2 == 1) ordered[(n - 1) / 2] else (ordered[n / 2 - 1] + ordered[n / 2]) / 2.0
    }

    private fun sortedFinite(values: List<Double>, what: String): List<Double> {
        if (values.isEmpty()) {
            throw UndefinedResultException(
                "$what is UNDEFINED on empty input; §9.1 forbids indexing element zero"
            )
        }
        // Nonfinite members are rejected *before* sorting (§9.1) — NaN would otherwise
        // corrupt the ordering silently rather than failing.
        values.forEach {
            require(it.isFinite()) { "$what requires finite samples, got $it" }
        }
        return values.sorted()
    }
}
