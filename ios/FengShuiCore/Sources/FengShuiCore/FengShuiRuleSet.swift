import Foundation
import HeadingCore

/// SPEC.md §21.1 `config/feng-shui-rules-v1.json`, typed.
///
/// The ruleset is a required, schema-validated, hashed artifact — not constants in the
/// classifier. The classifier itself (`fengShuiSector`, straddle sets, the §21.2 reference
/// transform) is Phase 1 work; Phase 0 owns the loader and the derived-boundary check,
/// because an internally inconsistent or abbreviated ruleset must fail the build rather
/// than misclassify quietly (R65).
public struct FengShuiRuleSet: Decodable {
    public struct Sector: Decodable {
        public let index: Int
        public let centerDeg: Double
        public let name: String
        public let glyph: String
        public let group: String
        public let groupGlyph: String
    }

    public struct Group: Decodable {
        public let name: String
        public let glyph: String
        public let cardinal: String
        public let centerDeg: Double
        public let widthDeg: Double
    }

    public let schemaVersion: String
    public let ruleSetVersion: String
    public let ruleSetName: String
    public let referenceSelection: String
    public let needleOffsetDeg: Double
    public let sectorCount: Int
    public let sectorWidthDeg: Double
    public let firstSectorCenterDeg: Double
    public let sectors: [Sector]
    public let groups: [Group]

    public static func load(contentsOf url: URL) throws -> FengShuiRuleSet {
        try JSONDecoder().decode(FengShuiRuleSet.self, from: Data(contentsOf: url))
    }

    /// SPEC.md §21.1: "Geometry is derived, never hand-typed as a boundary list."
    public func derivedCenterDeg(_ index: Int) throws -> Double {
        try CircularMath.normalize360(firstSectorCenterDeg + Double(index) * sectorWidthDeg)
    }

    /// Half-open `[start, end)` boundary in increasing azimuth for a sector index.
    public func derivedSectorStartDeg(_ index: Int) throws -> Double {
        try CircularMath.normalize360(derivedCenterDeg(index) - sectorWidthDeg / 2.0)
    }
}

public struct RuleSetViolation: CustomStringConvertible {
    public let checkId: String
    public let requirement: String
    public let detail: String
    public var description: String { "[\(checkId)] \(requirement) -- \(detail)" }
}

/// SPEC.md §21.1's required schema test, expressed as code because the relationships are
/// cross-field and JSON Schema cannot state them.
public enum FengShuiRuleSetGeometry {

    public static func check(_ ruleSet: FengShuiRuleSet) -> [RuleSetViolation] {
        var violations: [RuleSetViolation] = []
        func require(_ id: String, _ holds: Bool, _ requirement: String,
                     _ detail: @autoclosure () -> String) {
            if !holds {
                violations.append(RuleSetViolation(checkId: id, requirement: requirement, detail: detail()))
            }
        }

        require("RS-01-SECTOR-CARDINALITY", ruleSet.sectors.count == ruleSet.sectorCount,
                "sectors must contain exactly sectorCount entries; an excerpt cannot ship (R65)",
                "sectorCount=\(ruleSet.sectorCount), sectors=\(ruleSet.sectors.count)")

        require("RS-02-GROUP-CARDINALITY", ruleSet.groups.count == 8,
                "groups must contain exactly 8 unique trigrams", "groups=\(ruleSet.groups.count)")

        let indices = ruleSet.sectors.map(\.index).sorted()
        require("RS-03-INDICES-UNIQUE-CONTIGUOUS", indices == Array(0..<ruleSet.sectorCount),
                "sector indices must be unique and contiguous 0..\(ruleSet.sectorCount - 1)",
                "indices=\(indices)")

        let names = ruleSet.sectors.map(\.name)
        require("RS-04-NAMES-UNIQUE", Set(names).count == names.count,
                "sector names must be unique", "names=\(names)")

        let glyphs = ruleSet.sectors.map(\.glyph)
        require("RS-05-GLYPHS-UNIQUE", Set(glyphs).count == glyphs.count,
                "sector glyphs must be unique", "glyphs=\(glyphs)")

        let groupsByName = Dictionary(ruleSet.groups.map { ($0.name, $0) }, uniquingKeysWith: { a, _ in a })
        require("RS-06-GROUP-NAMES-UNIQUE", groupsByName.count == ruleSet.groups.count,
                "group names must be unique", "groups=\(ruleSet.groups.map(\.name))")

        for sector in ruleSet.sectors {
            let group = groupsByName[sector.group]
            require("RS-07-GROUP-REFERENCE-RESOLVES-\(sector.index)", group != nil,
                    "every sector's group reference must resolve to a declared group",
                    "sector \(sector.index) (\(sector.name)) references \(sector.group)")
            if let group {
                require("RS-08-GROUP-GLYPH-AGREES-\(sector.index)", group.glyph == sector.groupGlyph,
                        "a sector's groupGlyph must equal its group's glyph",
                        "\(sector.groupGlyph) vs \(group.glyph)")
            }
        }

        require("RS-09-FULL-CIRCLE",
                abs(Double(ruleSet.sectorCount) * ruleSet.sectorWidthDeg - 360.0) <= 1e-9,
                "sectorCount * sectorWidthDeg must equal 360",
                "\(ruleSet.sectorCount) * \(ruleSet.sectorWidthDeg)")

        for sector in ruleSet.sectors {
            let derived = (try? ruleSet.derivedCenterDeg(sector.index)) ?? Double.nan
            require("RS-10-CENTER-DERIVED-\(sector.index)",
                    abs(derived - sector.centerDeg) <= 1e-9,
                    "each declared centerDeg must equal normalize360(firstSectorCenterDeg + "
                        + "index * sectorWidthDeg) - geometry is derived, never hand-typed",
                    "sector \(sector.index) (\(sector.name)) declares \(sector.centerDeg), derived \(derived)")
        }

        require("RS-11-REFERENCE-SELECTION",
                ["TRUE", "MAGNETIC"].contains(ruleSet.referenceSelection),
                "referenceSelection must be TRUE or MAGNETIC (§21.2)",
                "referenceSelection=\(ruleSet.referenceSelection)")

        return violations
    }
}
