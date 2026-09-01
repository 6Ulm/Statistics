package com.fengshuicompass.fengshuicore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.math.CircularMath
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertNotEquals
import org.junit.jupiter.api.Assertions.assertNull
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import kotlin.math.abs

/** SPEC.md §21 — sector geometry, straddle sets, and the §21.2 reference transform. */
class FengShuiClassifierTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val ruleSet: FengShuiRuleSet
        get() = FengShuiRuleSet.load(SharedArtifacts.fengShuiRuleSetFile)

    private val classificationFixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.fengShuiClassificationFixture.readText()
        ).jsonObject

    private val transformFixture: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.fengShuiReferenceTransformFixture.readText()
        ).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    /**
     * A ruleset variant built in memory from the shipped document. §37 rule 12 forbids editing
     * a shipped artifact to make a test pass, so the §21.2 cases that need a different
     * `referenceSelection` or `needleOffsetDeg` construct their own copy instead.
     */
    private fun variant(referenceSelection: String, needleOffsetDeg: Double): FengShuiRuleSet {
        val shipped = json.parseToJsonElement(
            SharedArtifacts.fengShuiRuleSetFile.readText()
        ).jsonObject
        val document = buildJsonObject {
            shipped.forEach { (key, value) ->
                when (key) {
                    "referenceSelection" -> put(key, JsonPrimitive(referenceSelection))
                    "needleOffsetDeg" -> put(key, JsonPrimitive(needleOffsetDeg))
                    else -> put(key, value)
                }
            }
        }
        return json.decodeFromString(FengShuiRuleSet.serializer(), document.toString())
    }

    @Test
    @DisplayName("§21.1: every sector probe matches the frozen fixture")
    fun everySectorProbeMatchesTheFrozenFixture() {
        // Per sector: centre, both boundaries, +/-epsilon, +/-0.1, +/-1.0, plus the north wrap.
        val cases = classificationFixture["sectorCases"]!!.jsonArray
        assertEquals(ruleSet.sectorCount * 9, cases.size)
        cases.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val index = FengShuiClassifier.sectorIndex(
                case["headingDeg"]!!.jsonPrimitive.double,
                ruleSet,
            )
            assertEquals(case["expectedSectorIndex"]!!.jsonPrimitive.content.toInt(), index, id)
            assertEquals(
                case["expectedSectorName"]!!.jsonPrimitive.content,
                ruleSet.sectors[index].name,
                id,
            )
        }
    }

    @Test
    @DisplayName("§21.1: boundaries are half-open at the start")
    fun boundariesAreHalfOpenAtTheStart() {
        fun nameAt(heading: Double) =
            ruleSet.sectors[FengShuiClassifier.sectorIndex(heading, ruleSet)].name
        assertEquals("zi", nameAt(352.5))
        assertEquals("ren", nameAt(352.5 - 1e-9))
        assertEquals("gui", nameAt(7.5))
        assertEquals("zi", nameAt(7.5 - 1e-9))
    }

    @Test
    @DisplayName("§21.1: the north-wrap sector is covered; 352.5 separates ren and zi")
    fun theNorthWrapSectorIsCovered() {
        assertEquals(352.5, ruleSet.derivedSectorStartDeg(0), 1e-12)
        listOf(352.5, 355.0, 359.9, 0.0, 0.1, 7.49).forEach { heading ->
            assertEquals(
                "zi",
                ruleSet.sectors[FengShuiClassifier.sectorIndex(heading, ruleSet)].name,
                "$heading",
            )
        }
    }

    @Test
    @DisplayName("failure mode 7: no premature rounding moves a sector")
    fun noPrematureRoundingMovesASector() {
        // 337.49 rounds to 337.5, which is a different sector.
        assertEquals("hai", ruleSet.sectors[FengShuiClassifier.sectorIndex(337.49, ruleSet)].name)
        assertEquals("ren", ruleSet.sectors[FengShuiClassifier.sectorIndex(337.5, ruleSet)].name)
    }

    @Test
    @DisplayName("§21.4: straddle sets match the frozen fixture")
    fun straddleSetsMatchTheFrozenFixture() {
        classificationFixture["straddleCases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val note = case["note"]!!.jsonPrimitive.content
            val indices = FengShuiClassifier.straddleIndices(
                case["headingDeg"]!!.jsonPrimitive.double,
                case["reportedBound95Deg"]!!.jsonPrimitive.double,
                ruleSet,
            )
            assertEquals(
                case["expectedSectorIndices"]!!.jsonArray.map { it.jsonPrimitive.content.toInt() },
                indices,
                note,
            )
            assertEquals(
                case["expectedSectorNames"]!!.jsonArray.map { it.jsonPrimitive.content },
                indices.map { ruleSet.sectors[it].name },
                note,
            )
        }
    }

    @Test
    @DisplayName("§21.3: a bound above half a sector guarantees a straddle, at every heading")
    fun aBoundAboveHalfASectorGuaranteesAStraddle() {
        for (tenths in 0 until 3600) {
            val heading = tenths / 10.0
            assertTrue(
                FengShuiClassifier.straddleIndices(heading, 7.6, ruleSet).size >= 2,
                "$heading",
            )
        }
    }

    @Test
    @DisplayName("§21.3: a bound above a full sector guarantees three")
    fun aBoundAboveAFullSectorGuaranteesThree() {
        for (tenths in 0 until 3600 step 7) {
            val heading = tenths / 10.0
            assertTrue(
                FengShuiClassifier.straddleIndices(heading, 15.1, ruleSet).size >= 3,
                "$heading",
            )
        }
    }

    @Test
    @DisplayName("§21.3: a LOW_CONFIDENCE bound has essentially no discriminating power")
    fun aLowConfidenceBoundHasNoDiscriminatingPower() {
        for (tenths in 0 until 3600 step 13) {
            val heading = tenths / 10.0
            assertTrue(
                FengShuiClassifier.straddleIndices(
                    heading,
                    profile.lowConfidenceBound95MaxDeg,
                    ruleSet,
                ).size >= 2,
                "$heading",
            )
        }
    }

    @Test
    @DisplayName("§21.4: the full-circle degenerate case reports no classification")
    fun theFullCircleDegenerateCaseReportsNoClassification() {
        listOf(180.0, 200.0, 360.0).forEach { bound ->
            assertTrue(FengShuiClassifier.straddleIndices(10.0, bound, ruleSet).isEmpty())
            val result = FengShuiClassifier.classify(10.0, 0.0, bound, ruleSet)
            assertFalse(result.classificationPossible)
            assertNull(result.primarySector)
            assertTrue(result.possibleSectors.isEmpty())
        }
    }

    @Test
    @DisplayName("§21.3: a nearly full circle is classifiable as every sector, not one")
    fun aNearlyFullCircleIsEverySector() {
        // The case a walk-until-equal implementation gets wrong: both endpoints land in the
        // same sector, so it would report one mountain for an interval covering all 24.
        val indices = FengShuiClassifier.straddleIndices(10.0, 179.0, ruleSet)
        assertEquals(ruleSet.sectorCount, indices.size)
        assertEquals(ruleSet.sectorCount, indices.toSet().size)
    }

    @Test
    @DisplayName("§21.4: straddle sets wrap north in azimuth order")
    fun straddleSetsWrapNorthInAzimuthOrder() {
        val result = FengShuiClassifier.classify(0.0, 0.0, 8.0, ruleSet)
        assertTrue(result.boundaryStraddled)
        assertEquals(listOf("ren", "zi", "gui"), result.possibleSectors)
    }

    @Test
    @DisplayName("§21: the signed boundary offset stays within half a sector")
    fun theSignedBoundaryOffsetStaysWithinHalfASector() {
        for (tenths in 0 until 3600 step 3) {
            val heading = tenths / 10.0
            val offset = FengShuiClassifier.signedOffsetFromSectorBoundaryDeg(heading, ruleSet)
            assertTrue(abs(offset) <= ruleSet.sectorWidthDeg / 2.0 + 1e-9, "$heading")
        }
    }

    @Test
    @DisplayName("§21.4: a negative or nonfinite bound is rejected")
    fun aNegativeBoundIsRejected() {
        assertThrows<IllegalArgumentException> {
            FengShuiClassifier.straddleIndices(10.0, -1.0, ruleSet)
        }
        assertThrows<IllegalArgumentException> {
            FengShuiClassifier.straddleIndices(10.0, Double.NaN, ruleSet)
        }
    }

    // ---------------------------------------------------------------------------------
    // §21.2 — reference selection, needle offset, and the ambiguity term
    // ---------------------------------------------------------------------------------
    @Test
    @DisplayName("§21.2: the reference transform matches the frozen fixture")
    fun theReferenceTransformMatchesTheFrozenFixture() {
        transformFixture["cases"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            val id = case["id"]!!.jsonPrimitive.content
            val rules = variant(
                case["referenceSelection"]!!.jsonPrimitive.content,
                case["needleOffsetDeg"]!!.jsonPrimitive.double,
            )
            val observed = FengShuiClassifier.classificationHeadingDeg(
                case["googleOutputDeg"]!!.jsonPrimitive.double,
                case["declinationDeg"]!!.jsonPrimitive.double,
                rules,
            )
            assertEquals(
                case["expectedClassificationHeadingDeg"]!!.jsonPrimitive.double,
                observed,
                1e-9,
                id,
            )
            assertEquals(
                case["expectedErrorDeg"]!!.jsonPrimitive.double,
                CircularMath.absoluteCircularDifferenceDeg(
                    observed,
                    case["truthUnderHypothesisDeg"]!!.jsonPrimitive.double,
                ),
                1e-9,
                id,
            )
        }
    }

    @Test
    @DisplayName("§21.2: the ambiguity term covers either hidden Google hypothesis")
    fun theAmbiguityTermCoversEitherHiddenHypothesis() {
        // If Google secretly emitted magnetic north, the TRUE point is wrong by |d| **and**
        // the derived MAGNETIC point g-d is wrong by |d| too. Subtracting d for a magnetic
        // ruleset MUST NOT zero or remove the term.
        val hypothesisCases = transformFixture["cases"]!!.jsonArray
            .map { it.jsonObject }
            .filter { it["caseKind"]!!.jsonPrimitive.content == "HIDDEN_HYPOTHESIS" }
        assertTrue(hypothesisCases.isNotEmpty())
        hypothesisCases.forEach { case ->
            assertTrue(
                case["expectedErrorDeg"]!!.jsonPrimitive.double <=
                    case["referenceAmbiguityBound95Deg"]!!.jsonPrimitive.double + 1e-9,
                case["id"]!!.jsonPrimitive.content,
            )
        }
        assertEquals(
            setOf("GOOGLE_EMITTED_TRUE", "GOOGLE_EMITTED_MAGNETIC"),
            hypothesisCases.map { it["hiddenHypothesis"]!!.jsonPrimitive.content }.toSet(),
        )
        assertEquals(
            setOf("TRUE", "MAGNETIC"),
            hypothesisCases.map { it["referenceSelection"]!!.jsonPrimitive.content }.toSet(),
        )
        assertEquals(
            setOf(true, false),
            hypothesisCases.map { it["declinationDeg"]!!.jsonPrimitive.double > 0 }.toSet(),
        )
        // And the term is not vacuously large: at least one case actually reaches |d|.
        assertTrue(
            hypothesisCases.any {
                abs(
                    it["expectedErrorDeg"]!!.jsonPrimitive.double -
                        it["referenceAmbiguityBound95Deg"]!!.jsonPrimitive.double
                ) < 1e-9
            }
        )
    }

    @Test
    @DisplayName("§21.2: a magnetic ruleset does not shrink the bound")
    fun theMagneticRuleSetDoesNotShrinkTheBound() {
        val magnetic = FengShuiClassifier.classify(189.0, 8.29, 7.0, variant("MAGNETIC", 0.0))
        val trueRules = FengShuiClassifier.classify(189.0, 8.29, 7.0, variant("TRUE", 0.0))
        assertNotEquals(magnetic.classificationHeadingDeg, trueRules.classificationHeadingDeg)
        assertTrue(magnetic.possibleSectors.size >= 2)
        assertTrue(trueRules.possibleSectors.size >= 2)
    }

    @Test
    @DisplayName("§21.2: the needle offset is a declared ruleset property, not a slider")
    fun theNeedleOffsetIsADeclaredRuleSetProperty() {
        assertEquals(
            107.5,
            FengShuiClassifier.classificationHeadingDeg(100.0, 0.0, variant("TRUE", 7.5)),
            1e-9,
        )
        assertEquals(
            100.0,
            FengShuiClassifier.classificationHeadingDeg(100.0, 0.0, variant("TRUE", 0.0)),
            1e-9,
        )
    }

    @Test
    @DisplayName("failure mode 42: the classification records its ruleset")
    fun theClassificationRecordsItsRuleSet() {
        val result = FengShuiClassifier.classify(189.0, 8.29, 7.0, ruleSet)
        assertEquals("fengshui-v1", result.ruleSetVersion)
        assertEquals("TRUE", result.referenceSelection)
    }

    @Test
    @DisplayName("R62/§22.1: the shipped example classification reproduces")
    fun theShippedExampleClassificationReproduces() {
        val event = json.parseToJsonElement(
            SharedArtifacts.exampleEngineOutputEventFile.readText()
        ).jsonObject["payload"]!!.jsonObject
        val result = FengShuiClassifier.classify(
            event["trueHeadingDeg"]!!.jsonPrimitive.double,
            event["declinationDeg"]!!.jsonPrimitive.double,
            event["reportedBound95Deg"]!!.jsonPrimitive.double,
            ruleSet,
        )
        assertEquals(event["primaryFengShuiSector"]!!.jsonPrimitive.content, result.primarySector)
        assertEquals(
            event["possibleFengShuiSectors"]!!.jsonArray.map { it.jsonPrimitive.content },
            result.possibleSectors,
        )
    }
}
