package com.fengshuicompass.fengshuicore

import com.fengshuicompass.headingcore.config.SharedArtifacts
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/** SPEC.md §21.1 required schema/derived-boundary test, plus R65's completeness rule. */
class FengShuiRuleSetGeometryTest {

    private val ruleSet = FengShuiRuleSet.load(SharedArtifacts.fengShuiRuleSetFile)

    @Test
    @DisplayName("§21.1: the shipped artifact is the complete 24-sector / 8-group ruleset")
    fun rulesetIsComplete() {
        assertEquals("fengshui-v1", ruleSet.ruleSetVersion)
        assertEquals(24, ruleSet.sectorCount)
        assertEquals(24, ruleSet.sectors.size)
        assertEquals(8, ruleSet.groups.size)
    }

    @Test
    @DisplayName("§21.1: every derived-geometry consistency check passes")
    fun geometryIsConsistent() {
        val violations = FengShuiRuleSetGeometry.check(ruleSet)
        assertTrue(violations.isEmpty()) {
            "SPEC.md §21.1 ruleset violations:\n" + violations.joinToString("\n") { "  $it" }
        }
    }

    @Test
    @DisplayName("§21.1: boundaries land at 7.5 + 15k, so 352.5 separates ren and zi")
    fun boundariesAreAtSevenPointFivePlusFifteenK() {
        // With firstSectorCenterDeg = 0 and sectorWidthDeg = 15, sector starts are the
        // 7.5° + 15k boundaries §21.1 names explicitly.
        val starts = (0 until 24).map { ruleSet.derivedSectorStartDeg(it) }.sorted()
        val expected = (0 until 24).map { (7.5 + 15.0 * it) % 360.0 }.sorted()
        starts.zip(expected).forEach { (actual, want) -> assertEquals(want, actual, 1e-9) }

        // The north-wrap boundary: 352.5 separates 壬 (ren, index 23) from 子 (zi, index 0).
        assertEquals(352.5, ruleSet.derivedSectorStartDeg(0), 1e-9)
        assertEquals("ren", ruleSet.sectors.first { it.index == 23 }.name)
        assertEquals("zi", ruleSet.sectors.first { it.index == 0 }.name)
    }

    @Test
    @DisplayName("§21.1: the geometry check detects an inconsistent ruleset")
    fun geometryCheckIsNotVacuous() {
        // Mutating a copy in memory; the shipped artifact is never edited to pass a test.
        val broken = ruleSet.copy(
            sectors = ruleSet.sectors.map {
                if (it.index == 7) it.copy(centerDeg = it.centerDeg + 1.0) else it
            }
        )
        val violations = FengShuiRuleSetGeometry.check(broken)
        assertTrue(violations.any { it.checkId == "RS-10-CENTER-DERIVED-7" }) { "$violations" }

        val truncated = ruleSet.copy(sectors = ruleSet.sectors.take(2))
        val truncatedViolations = FengShuiRuleSetGeometry.check(truncated)
        assertTrue(truncatedViolations.any { it.checkId == "RS-01-SECTOR-CARDINALITY" }) {
            "R65: a two-entry 24-Mountains excerpt must not ship. $truncatedViolations"
        }
    }
}
