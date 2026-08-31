"""SPEC.md §21.1 derived-boundary test, analysis runtime.

'Geometry is derived, never hand-typed as a boundary list.' The schema cannot express these
cross-field relationships, so §21.1 requires a test that does.
"""

from __future__ import annotations

import pytest


def normalize360(deg: float) -> float:
    """§9. Kept local and tiny here because analysis/ shares the §9 definitions with both
    platforms from Phase 1; this Phase 0 use is the derived-centre check only."""
    if deg != deg or deg in (float("inf"), float("-inf")):
        raise ValueError(f"normalize360 requires a finite angle, got {deg}")
    wrapped = ((deg % 360.0) + 360.0) % 360.0
    return 0.0 if wrapped == 360.0 else wrapped


def test_ruleset_is_the_complete_artifact(rules):
    assert rules["ruleSetVersion"] == "fengshui-v1"
    assert rules["sectorCount"] == 24
    assert len(rules["sectors"]) == 24
    assert len(rules["groups"]) == 8


def test_sector_indices_unique_and_contiguous(rules):
    assert sorted(s["index"] for s in rules["sectors"]) == list(range(24))


def test_sector_names_and_glyphs_unique(rules):
    names = [s["name"] for s in rules["sectors"]]
    glyphs = [s["glyph"] for s in rules["sectors"]]
    assert len(set(names)) == len(names)
    assert len(set(glyphs)) == len(glyphs)


def test_group_references_resolve_and_glyphs_agree(rules):
    groups = {g["name"]: g for g in rules["groups"]}
    assert len(groups) == len(rules["groups"])
    for sector in rules["sectors"]:
        assert sector["group"] in groups, sector
        assert sector["groupGlyph"] == groups[sector["group"]]["glyph"], sector


def test_sectors_cover_the_full_circle(rules):
    assert rules["sectorCount"] * rules["sectorWidthDeg"] == pytest.approx(360.0)
    assert sum(g["widthDeg"] for g in rules["groups"]) == pytest.approx(360.0)


def test_declared_centres_equal_derived_centres(rules):
    """§21.1: centerDeg == normalize360(firstSectorCenterDeg + index * sectorWidthDeg)."""
    for sector in rules["sectors"]:
        derived = normalize360(
            rules["firstSectorCenterDeg"] + sector["index"] * rules["sectorWidthDeg"]
        )
        assert derived == pytest.approx(sector["centerDeg"], abs=1e-9), sector


def test_boundaries_land_at_seven_point_five_plus_fifteen_k(rules):
    """§21.1: 'this puts boundaries at 7.5° + 15k, so 352.5° separates 壬 and 子.'"""
    starts = sorted(
        normalize360(
            normalize360(rules["firstSectorCenterDeg"] + i * rules["sectorWidthDeg"])
            - rules["sectorWidthDeg"] / 2.0
        )
        for i in range(24)
    )
    expected = sorted(normalize360(7.5 + 15.0 * i) for i in range(24))
    for actual, want in zip(starts, expected):
        assert actual == pytest.approx(want, abs=1e-9)

    by_index = {s["index"]: s for s in rules["sectors"]}
    assert by_index[23]["name"] == "ren" and by_index[23]["glyph"] == "壬"
    assert by_index[0]["name"] == "zi" and by_index[0]["glyph"] == "子"


def test_reference_selection_and_needle_offset_are_declared(rules):
    """§21.2: both are stored with every result; a record whose reference is unknown is
    uninterpretable later (failure mode 42)."""
    assert rules["referenceSelection"] in ("TRUE", "MAGNETIC")
    assert isinstance(rules["needleOffsetDeg"], (int, float))
    assert rules["needleOffsetDeg"] == 0.0
