package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.math.CircularMath.normalize360
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertThrows
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/**
 * SPEC.md §9: "`normalize360` MUST be `((x % 360) + 360) % 360` with a finite check —
 * language remainder differs for negatives. Test `-360`, `-0.0`, `359.9999999`, `360.0`."
 *
 * The full §9 property-test set arrives in Phase 1 with the rest of the circular
 * utilities; these are the cases §9 names explicitly.
 */
class CircularMathTest {

    /**
     * SPEC.md §33.1/§36 Phase 1 declare the cross-runtime angle tolerance as `1e-6°`;
     * these assertions use it rather than bit equality. See [inRangeValuesAreNotBitExact].
     */
    private val angleToleranceDeg = 1e-6

    @Test
    @DisplayName("§9: the four explicitly named cases")
    fun namedCases() {
        assertEquals(0.0, normalize360(-360.0), 0.0)
        assertEquals(0.0, normalize360(-0.0), 0.0)
        assertEquals(359.9999999, normalize360(359.9999999), angleToleranceDeg)
        assertEquals(0.0, normalize360(360.0), 0.0)
    }

    @Test
    @DisplayName("§9: the mandated double-modulo form is not bit-exact for in-range values")
    fun inRangeValuesAreNotBitExact() {
        // §9 mandates `((x % 360) + 360) % 360` because the language remainder operator
        // differs for negatives. A consequence, recorded here so it cannot drift unnoticed:
        // for a value already in [0,360) the round trip through +360 and %360 loses low
        // bits, so normalize360 is NOT the identity on in-range doubles.
        //   normalize360(359.9999999) == 359.9999998999999...
        // The residual is ~1e-10, three orders of magnitude inside the 1e-6 cross-runtime
        // tolerance, and every runtime performs the same IEEE-754 operations in the same
        // order, so parity is unaffected. This test pins the magnitude: a regression that
        // enlarged it would surface here rather than in a bearing.
        val inRange = listOf(359.9999999, 0.1, 123.456789, 359.5, 1e-8)
        inRange.forEach { value ->
            val residual = kotlin.math.abs(normalize360(value) - value)
            assertEquals(0.0, residual, 1e-9) { "normalize360($value) drifted by $residual" }
        }
        // Documented exact observed value at the time of writing, for cross-runtime parity.
        assertEquals(359.9999998999999, normalize360(359.9999999), 0.0)
    }

    @Test
    @DisplayName("§9: exactly 360 maps to 0, and the result carries no negative zero")
    fun exactly360MapsToZeroWithoutNegativeZero() {
        // -0.0 == 0.0 compares true, so assert the bit pattern: a negative zero escaping
        // into a bearing is the kind of value that reappears as a sign flip downstream.
        assertEquals(0.0.toRawBits(), normalize360(-0.0).toRawBits())
        assertEquals(0.0.toRawBits(), normalize360(360.0).toRawBits())
        assertEquals(0.0.toRawBits(), normalize360(-720.0).toRawBits())
    }

    @Test
    @DisplayName("§9: negative inputs wrap the way the double-modulo form requires")
    fun negativeInputsWrap() {
        assertEquals(359.0, normalize360(-1.0), 1e-12)
        assertEquals(1.0, normalize360(-719.0), 1e-12)
        assertEquals(180.0, normalize360(-180.0), 1e-12)
    }

    @Test
    @DisplayName("§9: nonfinite input is rejected, never silently normalized")
    fun nonfiniteRejected() {
        assertThrows(IllegalArgumentException::class.java) { normalize360(Double.NaN) }
        assertThrows(IllegalArgumentException::class.java) { normalize360(Double.POSITIVE_INFINITY) }
        assertThrows(IllegalArgumentException::class.java) { normalize360(Double.NEGATIVE_INFINITY) }
    }
}
