package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.math.Estimators
import com.fengshuicompass.headingcore.math.UndefinedResultException
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs
import kotlin.math.atan2
import kotlin.math.cos
import kotlin.math.sin

/**
 * SPEC.md §9 / §9.1 / §15 — circular utilities and pinned estimators, Android runtime.
 *
 * Two kinds of check run here and both are needed. The **fixture** checks prove this runtime
 * agrees with the frozen `fixtures-v1` contract the analysis and iOS runtimes read. The
 * **spec-literal** checks assert the values SPEC.md states in prose, so a regenerated fixture
 * carrying a wrong expectation cannot make a broken implementation pass.
 */
class CircularMathTest {

    private val json = Json { ignoreUnknownKeys = false }

    private fun fixture(file: java.io.File): JsonObject =
        json.parseToJsonElement(file.readText()).jsonObject

    private val circularFixture: JsonObject get() = fixture(SharedArtifacts.circularMathFixture)
    private val estimatorFixture: JsonObject get() = fixture(SharedArtifacts.estimatorsFixture)
    private val aggregateFixture: JsonObject
        get() = fixture(SharedArtifacts.circularAggregateFixture)

    private val tolerance: Double
        get() = circularFixture["workingAngleToleranceDeg"]!!.jsonPrimitive.double

    // -----------------------------------------------------------------------------------
    // normalize360 (§9)
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§9: normalize360 matches the frozen fixture")
    fun normalize360MatchesTheFrozenFixture() {
        circularFixture["normalize360"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val input = case["inputDeg"]!!.jsonPrimitive.double
            val expected = case["expectedDeg"]!!.jsonPrimitive.double
            assertEquals(expected, CircularMath.normalize360(input), tolerance, "input=$input")
        }
    }

    @Test
    @DisplayName("§9: the named literal cases -360, -0.0, 360.0, 0, -1")
    fun normalize360SpecLiteralCases() {
        assertEquals(0.0, CircularMath.normalize360(-360.0))
        assertEquals(0.0, CircularMath.normalize360(-0.0))
        assertEquals(0.0, CircularMath.normalize360(360.0))
        assertEquals(0.0, CircularMath.normalize360(0.0))
        assertEquals(359.0, CircularMath.normalize360(-1.0))
    }

    @Test
    @DisplayName("§9 / F-1: 359.9999999 keeps the documented ~1e-10 residual")
    fun normalize360KeepsTheDocumentedResidual() {
        // docs/IMPLEMENTATION_NOTES.md F-1. The mandated ((x % 360) + 360) % 360 is not
        // bit-exact for a value already in range; the residual is three orders inside the
        // declared 1e-6 tolerance. Pinned so a change that *enlarged* it surfaces here.
        val observed = CircularMath.normalize360(359.9999999)
        assertTrue(abs(observed - 359.9999999) < 1e-9, "observed=$observed")
    }

    @Test
    @DisplayName("§5: a nonfinite angle is rejected, never silently normalized")
    fun normalize360RejectsNonFinite() {
        listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY).forEach {
            assertThrows<IllegalArgumentException> { CircularMath.normalize360(it) }
        }
    }

    // -----------------------------------------------------------------------------------
    // shortestSignedDifferenceDeg (§3, §9) — the single normative contract
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§9: the signed difference matches the frozen fixture")
    fun signedDifferenceMatchesTheFrozenFixture() {
        circularFixture["shortestSignedDifferenceDeg"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val a = case["aDeg"]!!.jsonPrimitive.double
            val b = case["bDeg"]!!.jsonPrimitive.double
            val expected = case["expectedDeg"]!!.jsonPrimitive.double
            assertEquals(
                expected,
                CircularMath.shortestSignedDifferenceDeg(a, b),
                tolerance,
                "a=$a b=$b",
            )
        }
    }

    @Test
    @DisplayName("§3/§35: the antipode is +180 in BOTH orderings of two distinct pairs")
    fun antipodeIsPlus180InBothOrderings() {
        // A test that only checks deltaDeg(180, 0) passes on a broken implementation, because
        // raw atan2 returns -180.0 only for the ordering whose sin lands on a tiny negative.
        listOf(
            0.0 to 180.0,
            180.0 to 0.0,
            90.0 to 270.0,
            270.0 to 90.0,
            120.0 to 300.0,
            300.0 to 120.0,
        ).forEach { (a, b) ->
            assertEquals(180.0, CircularMath.shortestSignedDifferenceDeg(a, b), "a=$a b=$b")
        }
    }

    @Test
    @DisplayName("§33.1: the prohibited raw formula demonstrably breaks the contract")
    fun rawAtan2WouldFailThisContract() {
        // §33.1 permits documentation and tests to *quote* a prohibited formula as text.
        // Computing it here proves the antipode normalization in the real implementation is
        // load-bearing rather than decorative.
        val radians = Math.toRadians(0.0 - 180.0)
        val prohibited = Math.toDegrees(atan2(sin(radians), cos(radians)))
        assertEquals(-180.0, prohibited)
        assertEquals(180.0, CircularMath.shortestSignedDifferenceDeg(0.0, 180.0))
    }

    @Test
    @DisplayName("§3: the range is (-180, 180], never -180")
    fun signedDifferenceRangeIsHalfOpenAtMinus180() {
        for (a in 0 until 360) {
            listOf(0, 37, 180, 359).forEach { b ->
                val delta = CircularMath.shortestSignedDifferenceDeg(a.toDouble(), b.toDouble())
                assertTrue(delta > -180.0 && delta <= 180.0, "a=$a b=$b delta=$delta")
            }
        }
    }

    @Test
    @DisplayName("§5: nonfinite inputs are rejected on either side")
    fun signedDifferenceRejectsNonFinite() {
        listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY).forEach {
            assertThrows<IllegalArgumentException> {
                CircularMath.shortestSignedDifferenceDeg(it, 0.0)
            }
            assertThrows<IllegalArgumentException> {
                CircularMath.shortestSignedDifferenceDeg(0.0, it)
            }
        }
    }

    // -----------------------------------------------------------------------------------
    // The exact delegating wrappers (§9, R68)
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§9: shortestTargetDeltaDeg is exactly the specified delegation")
    fun targetDeltaIsExactlyTheSpecifiedDelegation() {
        circularFixture["shortestTargetDeltaDeg"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val current = case["currentDeg"]!!.jsonPrimitive.double
            val target = case["targetDeg"]!!.jsonPrimitive.double
            val expected = case["expectedDeg"]!!.jsonPrimitive.double
            assertEquals(
                expected,
                CircularMath.shortestTargetDeltaDeg(current, target),
                tolerance,
                "current=$current target=$target",
            )
            assertEquals(
                CircularMath.shortestSignedDifferenceDeg(target, current),
                CircularMath.shortestTargetDeltaDeg(current, target),
            )
        }
    }

    @Test
    @DisplayName("§18.2: positive target delta means clockwise, across the north wrap")
    fun targetDeltaSignConventionAcrossNorthWrap() {
        assertTrue(CircularMath.shortestTargetDeltaDeg(359.0, 1.0) > 0.0)
        assertTrue(CircularMath.shortestTargetDeltaDeg(1.0, 359.0) < 0.0)
        assertEquals(2.0, abs(CircularMath.shortestTargetDeltaDeg(359.0, 1.0)), 1e-9)
    }

    @Test
    @DisplayName("§9: absoluteCircularDifferenceDeg is exactly the specified delegation")
    fun absoluteDifferenceIsExactlyTheSpecifiedDelegation() {
        circularFixture["absoluteCircularDifferenceDeg"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val a = case["aDeg"]!!.jsonPrimitive.double
            val b = case["bDeg"]!!.jsonPrimitive.double
            val expected = case["expectedDeg"]!!.jsonPrimitive.double
            val observed = CircularMath.absoluteCircularDifferenceDeg(a, b)
            assertEquals(expected, observed, tolerance, "a=$a b=$b")
            assertEquals(abs(CircularMath.shortestSignedDifferenceDeg(a, b)), observed)
            assertTrue(observed in 0.0..180.0)
        }
    }

    // -----------------------------------------------------------------------------------
    // §9.1 pinned estimators
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§9.1: quantile and median match the frozen fixture")
    fun estimatorsMatchTheFrozenFixture() {
        estimatorFixture["quantile"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val values = case["values"]!!.jsonArray.map { it.jsonPrimitive.double }
            val probability = case["probability"]!!.jsonPrimitive.double
            val expected = case["expected"]!!.jsonPrimitive.double
            assertEquals(expected, Estimators.quantile(values, probability), "q=$probability")
        }
        estimatorFixture["median"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val values = case["values"]!!.jsonArray.map { it.jsonPrimitive.double }
            assertEquals(case["expected"]!!.jsonPrimitive.double, Estimators.median(values))
        }
    }

    @Test
    @DisplayName("§9.1: nearest-rank, not an interpolating estimator")
    fun quantileIsNearestRank() {
        // With n = 20 the nearest-rank P95 is x[18]. A linear-interpolation estimator would
        // return 19.05; a device-computed P95 and a report-computed P95 must agree *exactly*.
        val values = (1..20).map { it.toDouble() }
        assertEquals(19.0, Estimators.quantile(values, 0.95))
        assertEquals(20.0, Estimators.quantile(values, 1.0))
        assertEquals(1.0, Estimators.quantile(values, 0.0))
    }

    @Test
    @DisplayName("§9.1: empty input is a typed UNDEFINED, never element zero")
    fun estimatorsAreUndefinedOnEmptyInput() {
        assertThrows<UndefinedResultException> { Estimators.quantile(emptyList(), 0.95) }
        assertThrows<UndefinedResultException> { Estimators.median(emptyList()) }
    }

    @Test
    @DisplayName("§9.1: nonfinite members are rejected before sorting")
    fun estimatorsRejectNonFiniteBeforeSorting() {
        assertThrows<IllegalArgumentException> {
            Estimators.quantile(listOf(1.0, Double.NaN, 3.0), 0.5)
        }
        assertThrows<IllegalArgumentException> {
            Estimators.median(listOf(1.0, Double.POSITIVE_INFINITY))
        }
    }

    // -----------------------------------------------------------------------------------
    // §15 circular aggregation
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§15: circular aggregation matches the frozen fixture")
    fun circularAggregateMatchesTheFrozenFixture() {
        val minimumResultant =
            aggregateFixture["minCircularResultantLength"]!!.jsonPrimitive.double
        aggregateFixture["windows"]!!.jsonArray.forEach { entry ->
            val window = entry.jsonObject
            val id = window["id"]!!.jsonPrimitive.content
            val samples = window["samples"]!!.jsonArray.map { it.jsonPrimitive.double }
            val aggregate = CircularMath.circularAggregate(samples)

            assertEquals(window["meanIsDefined"]!!.jsonPrimitive.content.toBoolean(), aggregate.isDefined, id)
            assertEquals(
                window["expectedResultantLength"]!!.jsonPrimitive.double,
                aggregate.resultantLength,
                1e-12,
                id,
            )
            assertEquals(
                window["expectedCircularMeanUndefinedUnderGate"]!!.jsonPrimitive.content.toBoolean(),
                CircularMath.circularMeanIsUndefined(aggregate, minimumResultant),
                id,
            )
            if (aggregate.isDefined) {
                assertEquals(
                    window["expectedMeanDeg"]!!.jsonPrimitive.double,
                    aggregate.meanDeg!!,
                    1e-9,
                    id,
                )
                assertEquals(
                    window["expectedResidualP95Deg"]!!.jsonPrimitive.double,
                    CircularMath.circularResidualQuantileDeg(samples, 0.95),
                    1e-9,
                    id,
                )
            }
        }
    }

    @Test
    @DisplayName("failure mode 1: the mean crosses north without linear averaging")
    fun circularMeanCrossesNorth() {
        val mean = CircularMath.circularMeanDeg(listOf(359.0, 359.5, 0.0, 0.5, 1.0))
        assertEquals(0.0, mean, 1e-9)
    }

    @Test
    @DisplayName("§15 decision 3: a weak resultant is an explicit failure, not a north reading")
    fun aWeakResultantIsAnExplicitFailure() {
        // atan2(0, 0) returning zero is the textbook case, but an antipodal window does not
        // reach it: sin(0) + sin(180) cancels to 1.2e-16, so the mean is numerically defined
        // and comes back as a confident 90 deg that means nothing. Only the configured
        // minCircularResultantLength gate catches that.
        val minimum = 0.995
        listOf(listOf(0.0, 180.0), listOf(0.0, 90.0, 180.0, 270.0)).forEach { samples ->
            val aggregate = CircularMath.circularAggregate(samples)
            assertEquals(0.0, aggregate.resultantLength, 1e-12)
            assertTrue(CircularMath.circularMeanIsUndefined(aggregate, minimum), "$samples")
        }
        val tight = CircularMath.circularAggregate(listOf(84.7, 85.3, 84.9, 85.4, 85.0))
        assertFalse(CircularMath.circularMeanIsUndefined(tight, minimum))
    }

    @Test
    @DisplayName("§9: an empty window has no mean")
    fun emptyWindowHasNoMean() {
        val aggregate = CircularMath.circularAggregate(emptyList())
        assertEquals(0, aggregate.count)
        assertFalse(aggregate.isDefined)
        assertThrows<UndefinedResultException> { CircularMath.circularMeanDeg(emptyList()) }
    }

    @Test
    @DisplayName("§15 decision 2: no trimming inside the window")
    fun noTrimmingInsideTheWindow() {
        // Trimming would let the window discard exactly the evidence that it is unreliable,
        // and would make the dispersion gate and the dispersion-derived bound disagree about
        // which samples exist.
        val steady = List(19) { 85.0 }
        val withOutlier = steady + 95.0
        assertTrue(
            CircularMath.circularResidualQuantileDeg(withOutlier, 0.95) >
                CircularMath.circularResidualQuantileDeg(steady, 0.95)
        )
    }

    // -----------------------------------------------------------------------------------
    // §19.2 boundFromSigma
    // -----------------------------------------------------------------------------------
    @Test
    @DisplayName("§19.2: boundFromSigma is the single conversion, factor from config")
    fun boundFromSigmaIsTheSingleConversion() {
        assertEquals(0.7056, CircularMath.boundFromSigma(0.36, 1.96), 1e-12)
        assertEquals(0.0, CircularMath.boundFromSigma(0.0, 1.96))
        assertThrows<IllegalArgumentException> { CircularMath.boundFromSigma(-0.1, 1.96) }
        assertThrows<IllegalArgumentException> { CircularMath.boundFromSigma(0.36, 0.0) }
        assertThrows<IllegalArgumentException> { CircularMath.boundFromSigma(Double.NaN, 1.96) }
    }
}
