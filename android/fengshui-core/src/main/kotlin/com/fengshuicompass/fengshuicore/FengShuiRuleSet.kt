package com.fengshuicompass.fengshuicore

import com.fengshuicompass.headingcore.math.CircularMath.normalize360
import kotlinx.serialization.Serializable
import kotlinx.serialization.json.Json
import java.io.File

/**
 * SPEC.md §21.1 `config/feng-shui-rules-v1.json`, typed.
 *
 * The ruleset is a required, schema-validated, hashed artifact — not constants in the
 * classifier — because a practitioner disputing a result needs to know which convention
 * produced it, and a ruleset edit is a behavioural change that must trip regression tests.
 *
 * The classifier itself (`fengShuiSector`, straddle sets, the §21.2 reference transform)
 * is Phase 1 work. Phase 0 owns the loader and the derived-boundary consistency check,
 * because an internally inconsistent or abbreviated ruleset must fail the build rather
 * than misclassify quietly (R65).
 */
@Serializable
public data class FengShuiRuleSet(
    val schemaVersion: String,
    val ruleSetVersion: String,
    val ruleSetName: String,
    val referenceSelection: String,
    val needleOffsetDeg: Double,
    val sectorCount: Int,
    val sectorWidthDeg: Double,
    val firstSectorCenterDeg: Double,
    val sectors: List<Sector>,
    val groups: List<Group>,
) {
    @Serializable
    public data class Sector(
        val index: Int,
        val centerDeg: Double,
        val name: String,
        val glyph: String,
        val group: String,
        val groupGlyph: String,
    )

    @Serializable
    public data class Group(
        val name: String,
        val glyph: String,
        val cardinal: String,
        val centerDeg: Double,
        val widthDeg: Double,
    )

    public companion object {
        private val strictJson = Json {
            ignoreUnknownKeys = false
            isLenient = false
            allowSpecialFloatingPointValues = false
        }

        public fun load(file: File): FengShuiRuleSet =
            strictJson.decodeFromString(serializer(), file.readText())
    }

    /**
     * SPEC.md §21.1: "Geometry is derived, never hand-typed as a boundary list."
     * The declared `centerDeg` of each sector must equal the derived value.
     */
    public fun derivedCenterDeg(index: Int): Double =
        normalize360(firstSectorCenterDeg + index * sectorWidthDeg)

    /** Half-open `[start, end)` boundary in increasing azimuth for a sector index. */
    public fun derivedSectorStartDeg(index: Int): Double =
        normalize360(derivedCenterDeg(index) - sectorWidthDeg / 2.0)
}

/** A structural inconsistency §21.1 requires the build to reject. */
public data class RuleSetViolation(val checkId: String, val requirement: String, val detail: String) {
    override fun toString(): String = "[$checkId] $requirement -- $detail"
}

/**
 * SPEC.md §21.1's required schema test, expressed as code because the relationships are
 * cross-field and JSON Schema cannot state them: "A schema test MUST assert exact array
 * cardinalities, unique/contiguous sector indices 0...23, unique names/glyphs, group
 * references that resolve, sectorCount * sectorWidthDeg == 360, and that each declared
 * centerDeg equals normalize360(firstSectorCenterDeg + index * sectorWidthDeg)."
 */
public object FengShuiRuleSetGeometry {

    public fun check(ruleSet: FengShuiRuleSet): List<RuleSetViolation> {
        val violations = mutableListOf<RuleSetViolation>()
        fun require(id: String, holds: Boolean, requirement: String, detail: () -> String) {
            if (!holds) violations += RuleSetViolation(id, requirement, detail())
        }

        require(
            "RS-01-SECTOR-CARDINALITY", ruleSet.sectors.size == ruleSet.sectorCount,
            "sectors must contain exactly sectorCount entries; an excerpt cannot ship (R65)",
        ) { "sectorCount=${ruleSet.sectorCount}, sectors=${ruleSet.sectors.size}" }

        require(
            "RS-02-GROUP-CARDINALITY", ruleSet.groups.size == 8,
            "groups must contain exactly 8 unique trigrams",
        ) { "groups=${ruleSet.groups.size}" }

        val indices = ruleSet.sectors.map { it.index }
        require(
            "RS-03-INDICES-UNIQUE-CONTIGUOUS",
            indices.sorted() == (0 until ruleSet.sectorCount).toList(),
            "sector indices must be unique and contiguous 0..${ruleSet.sectorCount - 1}",
        ) { "indices=${indices.sorted()}" }

        val names = ruleSet.sectors.map { it.name }
        require(
            "RS-04-NAMES-UNIQUE", names.size == names.toSet().size,
            "sector names must be unique",
        ) { "duplicates=${names.groupingBy { it }.eachCount().filterValues { it > 1 }.keys}" }

        val glyphs = ruleSet.sectors.map { it.glyph }
        require(
            "RS-05-GLYPHS-UNIQUE", glyphs.size == glyphs.toSet().size,
            "sector glyphs must be unique",
        ) { "duplicates=${glyphs.groupingBy { it }.eachCount().filterValues { it > 1 }.keys}" }

        val groupsByName = ruleSet.groups.associateBy { it.name }
        require(
            "RS-06-GROUP-NAMES-UNIQUE", groupsByName.size == ruleSet.groups.size,
            "group names must be unique",
        ) { "groups=${ruleSet.groups.map { it.name }}" }

        ruleSet.sectors.forEach { sector ->
            val group = groupsByName[sector.group]
            require(
                "RS-07-GROUP-REFERENCE-RESOLVES-${sector.index}", group != null,
                "every sector's group reference must resolve to a declared group",
            ) { "sector ${sector.index} (${sector.name}) references ${sector.group}" }
            if (group != null) {
                require(
                    "RS-08-GROUP-GLYPH-AGREES-${sector.index}", group.glyph == sector.groupGlyph,
                    "a sector's groupGlyph must equal its group's glyph",
                ) { "sector ${sector.name}: ${sector.groupGlyph} vs group ${group.name}: ${group.glyph}" }
            }
        }

        require(
            "RS-09-FULL-CIRCLE",
            kotlin.math.abs(ruleSet.sectorCount * ruleSet.sectorWidthDeg - 360.0) <= 1e-9,
            "sectorCount * sectorWidthDeg must equal 360",
        ) { "${ruleSet.sectorCount} * ${ruleSet.sectorWidthDeg} = ${ruleSet.sectorCount * ruleSet.sectorWidthDeg}" }

        ruleSet.sectors.forEach { sector ->
            val derived = ruleSet.derivedCenterDeg(sector.index)
            require(
                "RS-10-CENTER-DERIVED-${sector.index}",
                kotlin.math.abs(derived - sector.centerDeg) <= 1e-9,
                "each declared centerDeg must equal normalize360(firstSectorCenterDeg + " +
                    "index * sectorWidthDeg) - geometry is derived, never hand-typed",
            ) { "sector ${sector.index} (${sector.name}) declares ${sector.centerDeg}, derived $derived" }
        }

        require(
            "RS-11-REFERENCE-SELECTION", ruleSet.referenceSelection in setOf("TRUE", "MAGNETIC"),
            "referenceSelection must be TRUE or MAGNETIC (§21.2)",
        ) { "referenceSelection=${ruleSet.referenceSelection}" }

        return violations
    }
}
