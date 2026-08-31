package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.grade.CertificationState
import com.fengshuicompass.headingcore.grade.GradeReachability
import com.fengshuicompass.headingcore.grade.MagneticState
import com.fengshuicompass.headingcore.grade.PlacementMethod
import com.fengshuicompass.headingcore.grade.QualityGrade
import com.fengshuicompass.headingcore.grade.qualityGradeForReportedBound
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/**
 * SPEC.md §8.1.1 — the required build-time grade-reachability analysis.
 *
 * The three named consequences are asserted directly against the shipped constants, and
 * the whole declared claims table is checked. §8.1.1: "Code MUST NOT bypass the total-bound
 * gate, special-case a term, or floor a bound to make a combination lock."
 */
class GradeReachabilityTest {

    private val profile: PrecisionProfile = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)
    private val claims = GradeReachability.loadClaims(SharedArtifacts.gradeReachabilityClaimsFile)

    @Test
    @DisplayName("§8.1.1: no grade the product claims is arithmetically forbidden by the constants")
    fun everyClaimedGradeIsReachable() {
        val findings = GradeReachability.verify(claims, profile)
        assertTrue(findings.isEmpty()) {
            "SPEC.md §8.1.1 grade-reachability findings. A failing gate is a finding, not an " +
                "obstacle (§37 rule 12): fix the claim or the evidence, never the fixture.\n" +
                findings.joinToString("\n") { "  $it" }
        }
    }

    @Test
    @DisplayName("§8.1.1 row 1: flat freehand instrument budget is 2.0 deg and the unknown floor exceeds it")
    fun flatFreehandUncertifiedCannotLock() {
        val budget = GradeReachability.instrumentBudgetDeg(PlacementMethod.FREEHAND, profile)
        assertEquals(2.0, budget!!, 1e-12)
        assertTrue(profile.unknownDeviceFloor95Deg > budget) {
            "unknownDeviceFloor95Deg=${profile.unknownDeviceFloor95Deg} must exceed the ${budget} " +
                "budget; that is what makes a Precision Lock impossible on any uncertified device " +
                "in the ordinary user gesture."
        }
        val r = GradeReachability.compute(
            PlacementMethod.FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile
        )
        assertEquals(7.0, r.minimumReportedBound95Deg!!, 1e-12)
        assertFalse(r.lockReachable)
        assertEquals(GradeReachability.ClaimedGrade.LOW_CONFIDENCE, r.maxReachableGrade)
    }

    @Test
    @DisplayName("§8.1.1 row 2: wall freehand instrument budget is 0.0 deg, so it can never lock")
    fun wallFreehandCanNeverLock() {
        val budget = GradeReachability.instrumentBudgetDeg(PlacementMethod.WALL_FLUSH_FREEHAND, profile)
        assertEquals(0.0, budget!!, 1e-12)

        val uncertified = GradeReachability.compute(
            PlacementMethod.WALL_FLUSH_FREEHAND, CertificationState.UNCERTIFIED, MagneticState.CLEAN, profile
        )
        assertFalse(uncertified.lockReachable)

        // Certification cannot rescue it: the required floor is zero and a device floor is
        // strictly positive.
        val certified = GradeReachability.compute(
            PlacementMethod.WALL_FLUSH_FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile
        )
        assertEquals(0.0, certified.requiredDeviceFloorAtMostDeg!!, 1e-12)
        assertFalse(certified.lockReachable)
    }

    @Test
    @DisplayName("§8.1.1 row 3: SUSPECT prevents a freehand lock outright rather than capping the grade")
    fun suspectPreventsFreehandLockOutright() {
        val flatBudget = GradeReachability.instrumentBudgetDeg(PlacementMethod.FREEHAND, profile)!!
        val wallBudget = GradeReachability.instrumentBudgetDeg(PlacementMethod.WALL_FLUSH_FREEHAND, profile)!!
        assertTrue(profile.suspectInterferenceBound95Deg > flatBudget)
        assertTrue(profile.suspectInterferenceBound95Deg > wallBudget)

        for (method in listOf(PlacementMethod.FREEHAND, PlacementMethod.WALL_FLUSH_FREEHAND)) {
            for (certification in CertificationState.entries) {
                val r = GradeReachability.compute(method, certification, MagneticState.SUSPECT, profile)
                assertFalse(r.lockReachable) {
                    "$method/$certification under SUSPECT must not be lock-reachable: ${r.explanation}"
                }
            }
        }
    }

    @Test
    @DisplayName("§8.1.1 certification bootstrap: a certified flat-freehand lock needs a floor <= 2.0 deg")
    fun certifiedFlatFreehandRequiredFloor() {
        val r = GradeReachability.compute(
            PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN, profile
        )
        assertEquals(2.0, r.requiredDeviceFloorAtMostDeg!!, 1e-12)
        assertTrue(r.lockReachable)

        // Sweeping the floor as an explicit parameter, as §8.1.1 requires of Phase 5.
        assertTrue(
            GradeReachability.compute(
                PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN,
                profile, certifiedDeviceFloor95Deg = 2.0
            ).lockReachable
        )
        assertFalse(
            GradeReachability.compute(
                PlacementMethod.FREEHAND, CertificationState.CERTIFIED, MagneticState.CLEAN,
                profile, certifiedDeviceFloor95Deg = 2.0001
            ).lockReachable
        )
    }

    @Test
    @DisplayName("§29.5/§35: a placement method with no measured bound yields no grade, never a default")
    fun unmeasuredPlacementYieldsNoGrade() {
        for (method in listOf(PlacementMethod.NONMAGNETIC_ALIGNMENT_JIG, PlacementMethod.SURVEY_FIXTURE)) {
            assertNull(GradeReachability.placementBound95Deg(method, profile))
            val r = GradeReachability.compute(
                method, CertificationState.CERTIFIED, MagneticState.CLEAN, profile
            )
            assertEquals(GradeReachability.ClaimedGrade.NOT_SUPPORTED, r.maxReachableGrade)
            assertFalse(r.lockReachable)
        }
    }

    @Test
    @DisplayName("§16/§18.5: DISTURBED, INVALID and UNKNOWN produce no measurement and no grade")
    fun rejectingMagneticStatesProduceNoMeasurement() {
        for (state in listOf(MagneticState.DISTURBED, MagneticState.INVALID, MagneticState.UNKNOWN)) {
            assertNull(GradeReachability.interferenceBound95Deg(state, profile))
            val r = GradeReachability.compute(
                PlacementMethod.FREEHAND, CertificationState.CERTIFIED, state, profile
            )
            assertNull(r.minimumReportedBound95Deg)
            assertEquals(GradeReachability.ClaimedGrade.INVALID, r.maxReachableGrade)
        }
    }

    @Test
    @DisplayName("§20: the grade function is total and uses the documented half-open intervals")
    fun gradeFunctionIsTotalAndOrdered() {
        assertEquals(QualityGrade.PROFESSIONAL, qualityGradeForReportedBound(0.0, profile))
        assertEquals(QualityGrade.PROFESSIONAL, qualityGradeForReportedBound(2.0, profile))
        assertEquals(QualityGrade.HIGH, qualityGradeForReportedBound(2.0000001, profile))
        assertEquals(QualityGrade.HIGH, qualityGradeForReportedBound(3.0, profile))
        assertEquals(QualityGrade.USABLE, qualityGradeForReportedBound(3.0000001, profile))
        assertEquals(QualityGrade.USABLE, qualityGradeForReportedBound(5.0, profile))
        assertEquals(QualityGrade.LOW_CONFIDENCE, qualityGradeForReportedBound(5.0000001, profile))
        assertEquals(QualityGrade.LOW_CONFIDENCE, qualityGradeForReportedBound(10.0, profile))
        assertEquals(QualityGrade.INVALID, qualityGradeForReportedBound(10.0000001, profile))
        assertEquals(QualityGrade.INVALID, qualityGradeForReportedBound(180.0, profile))
    }

    @Test
    @DisplayName("§8.1.1: the verifier detects a claim the constants forbid")
    fun verifierIsNotVacuous() {
        // An in-memory claim asserting a Precision Lock for uncertified flat freehand - the
        // exact overreach §8.1.1 exists to catch. The shipped claims file is not modified.
        val overreaching = claims.copy(
            combinations = listOf(
                claims.combinations.first { it.id == "flat-freehand-uncertified-clean" }
                    .copy(claimedMaxGrade = "USABLE", claimedLockReachable = true)
            )
        )
        val findings = GradeReachability.verify(overreaching, profile)
        assertTrue(findings.size >= 2) { "expected a grade finding and a lock finding, got $findings" }
        assertTrue(findings.any { it.problem.contains("arithmetically forbidden") }) { "$findings" }
        assertTrue(findings.any { it.problem.contains("Precision Lock is reachable") }) { "$findings" }
    }
}
