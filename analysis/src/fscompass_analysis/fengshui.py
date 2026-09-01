"""SPEC.md §21 — the Feng Shui direction engine: geometry, classification, straddle.

The engine consumes a full-precision canonical heading and its bound. It never rounds before
classification (failure mode 7: ``337.49°`` moves a sector), and boundaries are derived from
the ruleset rather than hand-typed (R65).

§21.3 sets the honest expectation this module is built around: a sector is ``15°`` wide, so a
``reportedBound95Deg`` above ``7.5°`` *guarantees* a two-sector straddle regardless of the
point estimate. **Straddles are the common case**, which is why
:class:`FengShuiClassification` returns the full set rather than a primary with a footnote.
"""

from __future__ import annotations

import math
from dataclasses import dataclass

from .circular import (
    absolute_circular_difference_deg,
    normalize360,
    shortest_signed_difference_deg,
)

__all__ = [
    "FengShuiClassification",
    "FengShuiRuleSet",
    "classification_heading_deg",
    "classify",
    "sector_index",
    "straddle_indices",
]


class RuleSetError(ValueError):
    """A ruleset that cannot be used for classification."""


@dataclass(frozen=True)
class FengShuiSector:
    index: int
    center_deg: float
    name: str
    glyph: str
    group: str
    group_glyph: str


@dataclass(frozen=True)
class FengShuiRuleSet:
    """The loaded, hashed ``config/feng-shui-rules-v1.json``.

    Its version and SHA-256 appear in every measurement: a practitioner disputing a result
    needs to know which convention produced it, and a ruleset edit is a behavioural change
    that must trip regression tests (§21.1).
    """

    rule_set_version: str
    rule_set_name: str
    reference_selection: str
    needle_offset_deg: float
    sector_count: int
    sector_width_deg: float
    first_sector_center_deg: float
    sectors: tuple[FengShuiSector, ...]

    @staticmethod
    def from_document(document: dict) -> "FengShuiRuleSet":
        return FengShuiRuleSet(
            rule_set_version=document["ruleSetVersion"],
            rule_set_name=document["ruleSetName"],
            reference_selection=document["referenceSelection"],
            needle_offset_deg=float(document["needleOffsetDeg"]),
            sector_count=int(document["sectorCount"]),
            sector_width_deg=float(document["sectorWidthDeg"]),
            first_sector_center_deg=float(document["firstSectorCenterDeg"]),
            sectors=tuple(
                FengShuiSector(
                    index=int(sector["index"]),
                    center_deg=float(sector["centerDeg"]),
                    name=sector["name"],
                    glyph=sector["glyph"],
                    group=sector["group"],
                    group_glyph=sector["groupGlyph"],
                )
                for sector in document["sectors"]
            ),
        )

    def __post_init__(self) -> None:
        if self.sector_count <= 0 or self.sector_width_deg <= 0.0:
            raise RuleSetError("sectorCount and sectorWidthDeg must be positive")
        if len(self.sectors) != self.sector_count:
            raise RuleSetError(
                f"ruleset declares {self.sector_count} sectors but carries {len(self.sectors)}"
            )
        if self.reference_selection not in ("TRUE", "MAGNETIC"):
            raise RuleSetError(f"unknown referenceSelection {self.reference_selection!r}")

    def derived_center_deg(self, index: int) -> float:
        return normalize360(self.first_sector_center_deg + index * self.sector_width_deg)

    def derived_start_deg(self, index: int) -> float:
        """The half-open ``[start, end)`` lower boundary in increasing azimuth."""
        return normalize360(self.derived_center_deg(index) - self.sector_width_deg / 2.0)

    def sector(self, index: int) -> FengShuiSector:
        return self.sectors[index]


def sector_index(heading_deg: float, rule_set: FengShuiRuleSet) -> int:
    """§21.1's derived index — the only place a boundary is computed.

    ``floor(normalize360(h - firstSectorCenterDeg + sectorWidthDeg/2) / sectorWidthDeg)
    mod sectorCount``. For the default ruleset this puts boundaries at ``7.5° + 15k``, so
    ``352.5°`` separates 壬 and 子. Half-open ``[start, end)``: a heading exactly on a
    boundary belongs to the sector that boundary starts.
    """
    offset = normalize360(
        heading_deg - rule_set.first_sector_center_deg + rule_set.sector_width_deg / 2.0
    )
    return int(math.floor(offset / rule_set.sector_width_deg)) % rule_set.sector_count


def offset_within_sector_deg(heading_deg: float, rule_set: FengShuiRuleSet) -> float:
    """How far past its own start boundary ``heading_deg`` sits, in ``[0, sectorWidthDeg)``.

    Derived from the *same* wrapped quantity :func:`sector_index` uses, so the index and the
    offset cannot disagree by a rounding bit at a boundary.
    """
    wrapped = normalize360(
        heading_deg - rule_set.first_sector_center_deg + rule_set.sector_width_deg / 2.0
    )
    return wrapped - math.floor(wrapped / rule_set.sector_width_deg) * rule_set.sector_width_deg


def classification_heading_deg(
    true_heading_deg: float,
    declination_deg: float,
    rule_set: FengShuiRuleSet,
) -> float:
    """§21.2's final, explicit, recorded reference step.

    ``TRUE`` uses the canonical true heading; ``MAGNETIC`` derives ``trueHeading -
    declination`` from that *same* canonical measurement rather than substituting an
    unvalidated magnetic path. ``needleOffsetDeg`` expresses doctrinal plate conventions and
    is a declared property of a named ruleset — never a user slider, never a correction for
    measurement error.
    """
    if rule_set.reference_selection == "TRUE":
        base = true_heading_deg
    elif rule_set.reference_selection == "MAGNETIC":
        base = normalize360(true_heading_deg - declination_deg)
    else:  # pragma: no cover - constructor rejects this
        raise RuleSetError(f"unknown referenceSelection {rule_set.reference_selection!r}")
    return normalize360(base + rule_set.needle_offset_deg)


def straddle_indices(
    classification_heading_deg_value: float,
    reported_bound_95_deg: float,
    rule_set: FengShuiRuleSet,
) -> tuple[int, ...]:
    """§21.4: every ruleset sector intersecting the circular interval, in azimuth order.

    The interval is the **closed** ``[h - bound, h + bound]`` while sectors are half-open, so
    an interval endpoint landing exactly on a boundary includes the sector that boundary
    starts. That asymmetry is deliberately conservative: naming one fewer mountain is a
    false-precision failure, naming one more is not.

    The count comes from the **arc length**, not from walking forward until the end index is
    met. At a bound approaching ``180°`` the interval wraps almost the whole circle and both
    endpoints land in the *same* sector; a walk-until-equal would then report one sector for
    an interval covering all 24 — a single-mountain claim from a measurement that discriminates
    nothing, which is the exact false-precision failure §21.3 warns about.
    """
    if reported_bound_95_deg < 0.0 or not math.isfinite(reported_bound_95_deg):
        raise ValueError("reportedBound95Deg must be a finite, non-negative bound")
    if 2.0 * reported_bound_95_deg >= 360.0:
        # §21.4: report that no classification is possible rather than listing all 24.
        return ()

    low = classification_heading_deg_value - reported_bound_95_deg
    start_index = sector_index(low, rule_set)
    offset = offset_within_sector_deg(low, rule_set)
    spanned = min(
        rule_set.sector_count,
        1
        + int(
            math.floor((offset + 2.0 * reported_bound_95_deg) / rule_set.sector_width_deg)
        ),
    )
    return tuple(
        (start_index + step) % rule_set.sector_count for step in range(spanned)
    )


def signed_offset_from_sector_boundary_deg(
    classification_heading_deg_value: float, rule_set: FengShuiRuleSet
) -> float:
    """Signed circular difference from the **nearest** sector boundary to the heading.

    Positive means the heading lies clockwise of that boundary. Magnitude never exceeds
    ``sectorWidthDeg / 2``. Reported so a practitioner can see how close to a boundary a
    result sits even when the bound happens not to straddle.
    """
    index = sector_index(classification_heading_deg_value, rule_set)
    candidates = (
        rule_set.derived_start_deg(index),
        rule_set.derived_start_deg((index + 1) % rule_set.sector_count),
    )
    nearest = min(
        candidates,
        key=lambda boundary: absolute_circular_difference_deg(
            classification_heading_deg_value, boundary
        ),
    )
    return shortest_signed_difference_deg(classification_heading_deg_value, nearest)


@dataclass(frozen=True)
class FengShuiClassification:
    """§5.1's ``classification:`` block for one measurement."""

    rule_set_version: str
    reference_selection: str
    classification_heading_deg: float
    primary_sector: str | None
    possible_sectors: tuple[str, ...]
    possible_sector_indices: tuple[int, ...]
    boundary_straddled: bool
    signed_offset_from_sector_boundary_deg: float
    classification_possible: bool


def classify(
    true_heading_deg: float,
    declination_deg: float,
    reported_bound_95_deg: float,
    rule_set: FengShuiRuleSet,
) -> FengShuiClassification:
    """§21: classify the whole circular bound interval, never the point estimate alone.

    ``reported_bound_95_deg`` is the **total** bound — instrument plus placement, including
    any ``referenceAmbiguityBound95Deg``. §21.2: subtracting declination for a magnetic
    ruleset MUST NOT zero or remove the ambiguity term, because if the provider secretly
    emitted the other reference the derived magnetic point is wrong by ``|d|`` too. This
    function therefore never touches the bound it is handed.
    """
    heading = classification_heading_deg(true_heading_deg, declination_deg, rule_set)
    indices = straddle_indices(heading, reported_bound_95_deg, rule_set)
    if not indices:
        return FengShuiClassification(
            rule_set_version=rule_set.rule_set_version,
            reference_selection=rule_set.reference_selection,
            classification_heading_deg=heading,
            primary_sector=None,
            possible_sectors=(),
            possible_sector_indices=(),
            boundary_straddled=True,
            signed_offset_from_sector_boundary_deg=(
                signed_offset_from_sector_boundary_deg(heading, rule_set)
            ),
            classification_possible=False,
        )

    primary_index = sector_index(heading, rule_set)
    return FengShuiClassification(
        rule_set_version=rule_set.rule_set_version,
        reference_selection=rule_set.reference_selection,
        classification_heading_deg=heading,
        primary_sector=rule_set.sector(primary_index).name,
        possible_sectors=tuple(rule_set.sector(index).name for index in indices),
        possible_sector_indices=indices,
        boundary_straddled=len(indices) > 1,
        signed_offset_from_sector_boundary_deg=(
            signed_offset_from_sector_boundary_deg(heading, rule_set)
        ),
        classification_possible=True,
    )
