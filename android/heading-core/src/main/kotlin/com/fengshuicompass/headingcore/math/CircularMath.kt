package com.fengshuicompass.headingcore.math

import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.hypot
import kotlin.math.sin

/**
 * SPEC.md §9 deterministic utilities.
 *
 * This object is the **single allowlisted home** for the signed circular difference in the
 * Android runtime (§33.1, R67/R68). [shortestSignedDifferenceDeg] below is the one
 * implementation; [shortestTargetDeltaDeg] and [absoluteCircularDifferenceDeg] are exact
 * delegating wrappers containing no angle arithmetic of their own. No other file in this
 * runtime may restate the formula, and the analysis-runtime audit enforces that across all
 * three runtimes.
 *
 * Two `atan2` call sites live here and are the ones §33.1 allowlists: the signed-difference
 * implementation, and the circular mean/resultant (a bearing projection, a different
 * quantity).
 */
public object CircularMath {

    /**
     * SPEC.md §9: `((x % 360) + 360) % 360` with a finite check; exactly `360.0` maps to
     * `0.0`. Written with the explicit double-modulo because the language remainder
     * operator differs for negative operands (failure mode 2).
     *
     * @throws IllegalArgumentException when [deg] is not finite. A nonfinite angle is an
     *   invalid input, never silently normalized (§5: `0`, `-1`, `NaN`, `null` are not
     *   interchangeable).
     */
    public fun normalize360(deg: Double): Double {
        require(deg.isFinite()) { "normalize360 requires a finite angle, got $deg" }
        val wrapped = ((deg % 360.0) + 360.0) % 360.0
        // ((-0.0 % 360) + 360) % 360 evaluates to 0.0 already, but 360.0 - epsilon
        // rounding can land exactly on 360.0; §9 pins that case to 0.0.
        return if (wrapped == 360.0) 0.0 else wrapped + 0.0
    }

    /**
     * SPEC.md §9/§3: `a - b` as the shortest signed circular difference in `(-180, 180]`.
     *
     * The `atan2` convention, **and** the mandatory antipode normalization: raw
     * `atan2(sin(a-b), cos(a-b))` returns `-180.0` whenever `sin` evaluates to a tiny
     * negative rather than exactly zero, which is what happens for ordinary inputs such as
     * `a=0, b=180` and `a=90, b=270`. An exact `-180.0` is therefore mapped to `+180.0`
     * before returning, so the antipode is `+180` and never `-180` and bias statistics stay
     * deterministic.
     *
     * This is the one normative contract in the spec and the only implementation in this
     * runtime.
     */
    public fun shortestSignedDifferenceDeg(a: Double, b: Double): Double {
        require(a.isFinite() && b.isFinite()) {
            "shortestSignedDifferenceDeg requires finite angles, got a=$a b=$b"
        }
        val radians = Math.toRadians(a - b)
        val delta = Math.toDegrees(atan2(sin(radians), cos(radians)))
        return if (delta == -180.0) 180.0 else delta
    }

    /**
     * SPEC.md §9: `shortestSignedDifferenceDeg(target, current)`; positive = clockwise.
     *
     * A thin delegating wrapper with the exact §9 definition. No local alias, no independent
     * angle math (R68).
     */
    public fun shortestTargetDeltaDeg(currentDeg: Double, targetDeg: Double): Double =
        shortestSignedDifferenceDeg(targetDeg, currentDeg)

    /**
     * SPEC.md §9: `abs(shortestSignedDifferenceDeg(a, b))`, range `[0, 180]`.
     * A thin delegating wrapper (R68).
     */
    public fun absoluteCircularDifferenceDeg(a: Double, b: Double): Double =
        abs(shortestSignedDifferenceDeg(a, b))

    /**
     * SPEC.md §15 circular mean and resultant length with **uniform weights**.
     *
     * `w_i = 1` for every accepted sample. Weighting by provider error, recency or
     * dispersion is a plausible and untested improvement; §15 makes it a named benchmark
     * variant, not a quiet implementation choice. There is no trimming here either —
     * rejection happens at the per-sample gate, before entry.
     */
    public fun circularAggregate(samples: List<Double>): CircularAggregate {
        samples.forEach {
            require(it.isFinite()) { "circularAggregate requires finite samples, got $it" }
        }
        if (samples.isEmpty()) {
            return CircularAggregate(meanDeg = null, resultantLength = 0.0, count = 0)
        }
        val count = samples.size
        val cosine = samples.sumOf { cos(Math.toRadians(it)) } / count
        val sine = samples.sumOf { sin(Math.toRadians(it)) } / count
        // A resultant may exceed 1 only by floating-point rounding; §6 declares the range [0,1].
        val resultant = hypot(cosine, sine).coerceIn(0.0, 1.0)
        if (cosine == 0.0 && sine == 0.0) {
            return CircularAggregate(meanDeg = null, resultantLength = resultant, count = count)
        }
        return CircularAggregate(
            meanDeg = normalize360(Math.toDegrees(atan2(sine, cosine))),
            resultantLength = resultant,
            count = count,
        )
    }

    /** SPEC.md §9 `circularMeanDeg`; throws [UndefinedResultException] for `UNDEFINED`. */
    public fun circularMeanDeg(samples: List<Double>): Double {
        val aggregate = circularAggregate(samples)
        return aggregate.meanDeg
            ?: throw UndefinedResultException(
                "circularMeanDeg is UNDEFINED: " +
                    if (aggregate.count == 0) {
                        "empty window"
                    } else {
                        "zero resultant (CIRCULAR_MEAN_UNDEFINED)"
                    }
            )
    }

    /** SPEC.md §9 `circularResultantLength` -> `[0, 1]`. */
    public fun circularResultantLength(samples: List<Double>): Double =
        circularAggregate(samples).resultantLength

    /**
     * SPEC.md §15 decision 3: "A weak resultant is an explicit failure."
     *
     * The exactly-degenerate `atan2(0, 0)` case is only half the problem, and the smaller
     * half: for an antipodal pair the two sines cancel to `6.1e-17` rather than to zero, so
     * the mean is *numerically* defined and comes back as a confident, completely arbitrary
     * bearing. Only the configured `minCircularResultantLength` gate catches that, which is
     * why this decision reads a config key and never an epsilon invented here.
     *
     * Callers emit `CIRCULAR_MEAN_UNDEFINED` and reject when this returns `true`.
     */
    public fun circularMeanIsUndefined(
        aggregate: CircularAggregate,
        minCircularResultantLength: Double,
    ): Boolean = aggregate.meanDeg == null || aggregate.resultantLength < minCircularResultantLength

    /** Absolute residuals about an accepted circular mean, in sample order (§19). */
    public fun circularResidualsDeg(samples: List<Double>, meanDeg: Double): List<Double> =
        samples.map { absoluteCircularDifferenceDeg(it, meanDeg) }

    /**
     * SPEC.md §9/§9.1: the linear estimator applied to absolute residuals about the mean.
     *
     * §15 fixes the sample set: **all** accepted samples, no trimming, so the dispersion gate
     * and the dispersion-derived bound cannot disagree about which samples exist.
     */
    public fun circularResidualQuantileDeg(samples: List<Double>, q: Double): Double =
        Estimators.quantile(circularResidualsDeg(samples, circularMeanDeg(samples)), q)

    /**
     * SPEC.md §19.2: the single named conversion from one sigma to a 95% bound.
     *
     * The candidate factor `1.96` is the Gaussian two-sided 95% multiplier — a **modelling
     * assumption**, not a property of the model, which is why it lives in versioned
     * configuration and is passed in here rather than written as a literal.
     */
    public fun boundFromSigma(sigma1Deg: Double, sigmaToBound95Factor: Double): Double {
        require(sigma1Deg.isFinite() && sigma1Deg >= 0.0) {
            "boundFromSigma requires a finite, non-negative sigma, got $sigma1Deg"
        }
        require(sigmaToBound95Factor.isFinite() && sigmaToBound95Factor > 0.0) {
            "declinationSigmaToBound95Factor must be positive, got $sigmaToBound95Factor"
        }
        return sigmaToBound95Factor * sigma1Deg
    }
}

/**
 * The result of aggregating a window of angles under SPEC.md §15.
 *
 * [meanDeg] is `null` exactly when the mean is `UNDEFINED`: an empty window, or a resultant
 * of exactly zero, where `atan2(0, 0)` returns `0.0` on every platform and would disguise a
 * bimodal set as a north-facing measurement (failure mode 6).
 *
 * [resultantLength] is reported even when the mean is undefined, because §15's gate is stated
 * on `R` and the engine records the feature either way.
 */
public data class CircularAggregate(
    val meanDeg: Double?,
    val resultantLength: Double,
    val count: Int,
) {
    val isDefined: Boolean get() = meanDeg != null
}

/**
 * SPEC.md §9.1's typed `UNDEFINED` outcome.
 *
 * `quantile` and `median` on empty input "MUST return a typed `UNDEFINED`/validation failure
 * and MUST NOT index element zero".
 */
public class UndefinedResultException(message: String) : IllegalStateException(message)
