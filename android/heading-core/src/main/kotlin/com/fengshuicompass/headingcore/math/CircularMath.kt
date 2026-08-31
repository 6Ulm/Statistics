package com.fengshuicompass.headingcore.math

/**
 * SPEC.md §9 deterministic utilities.
 *
 * Phase 0 defines only [normalize360], which the §21.1 derived-boundary check needs.
 * The remaining §9 utilities — `shortestSignedDifferenceDeg` and its exact delegating
 * wrappers, the circular mean/resultant, and the §9.1 pinned quantile/median — are
 * Phase 1 work and are deliberately absent rather than stubbed, so that no caller can
 * bind to a placeholder definition (§37 "Do not hide incomplete work behind
 * placeholders").
 *
 * This object is the single allowlisted home for circular math in the Android runtime
 * (§33.1, R67/R68). No other file may restate these formulas.
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
}
