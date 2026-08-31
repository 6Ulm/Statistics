"""SPEC.md §8.1.1 — the required build-time grade-reachability analysis, third runtime.

``reportedBound95Deg = instrumentBound95Deg + placementBound95Deg`` and the lock ceiling is
``usableBound95MaxDeg``, so each placement method has a fixed **instrument budget** of
``usableBound95MaxDeg - placementBound95Deg``. Any single uncertainty term larger than that
budget makes a Precision Lock arithmetically impossible for that combination, no matter how
good the sensor is.

The analysis computes the **infimum** of ``reportedBound95Deg``: every §19 term that can
legitimately be zero is taken at zero, so a claimed grade unreachable here is unreachable
everywhere.
"""

from __future__ import annotations

from dataclasses import dataclass
from enum import Enum
from typing import Any


class QualityGrade(Enum):
    """SPEC.md §6 ``QualityGrade`` plus the ``NOT_SUPPORTED`` claim-vocabulary value.

    Only the §6 cases may appear in telemetry or on a wire. The member value is the
    strength index: smaller is a stronger claim.
    """

    PROFESSIONAL = 0
    HIGH = 1
    USABLE = 2
    LOW_CONFIDENCE = 3
    INVALID = 4
    NOT_SUPPORTED = 5

    def is_stronger_than(self, other: "QualityGrade") -> bool:
        return self.value < other.value


class PlacementMethod(Enum):
    FREEHAND = "FREEHAND"
    WALL_FLUSH_FREEHAND = "WALL_FLUSH_FREEHAND"
    NONMAGNETIC_ALIGNMENT_JIG = "NONMAGNETIC_ALIGNMENT_JIG"
    SURVEY_FIXTURE = "SURVEY_FIXTURE"


class MagneticState(Enum):
    CLEAN = "CLEAN"
    SUSPECT = "SUSPECT"
    DISTURBED = "DISTURBED"
    INVALID = "INVALID"
    UNKNOWN = "UNKNOWN"


class CertificationState(Enum):
    UNCERTIFIED = "UNCERTIFIED"
    CERTIFIED = "CERTIFIED"


def quality_grade_for_reported_bound(
    reported_bound_95_deg: float, profile: dict[str, Any]
) -> QualityGrade:
    """SPEC.md §20: grades come from ``reportedBound95Deg`` on half-open intervals.

    Grading on ``instrumentBound95Deg`` would advertise precision the practitioner cannot
    physically realize (failure mode 30).
    """
    if not (reported_bound_95_deg == reported_bound_95_deg) or reported_bound_95_deg < 0:
        raise ValueError(
            f"reportedBound95Deg must be a finite non-negative angle, got {reported_bound_95_deg}"
        )
    if reported_bound_95_deg <= profile["professionalBound95MaxDeg"]:
        return QualityGrade.PROFESSIONAL
    if reported_bound_95_deg <= profile["highBound95MaxDeg"]:
        return QualityGrade.HIGH
    if reported_bound_95_deg <= profile["usableBound95MaxDeg"]:
        return QualityGrade.USABLE
    if reported_bound_95_deg <= profile["lowConfidenceBound95MaxDeg"]:
        return QualityGrade.LOW_CONFIDENCE
    return QualityGrade.INVALID


def placement_bound_95_deg(method: PlacementMethod, profile: dict[str, Any]) -> float | None:
    """The placement term, or ``None`` when the shipped profile has no measured bound.

    §18.5: "Placement uncertainty: finite bound from method ... **never zero**."
    §29.5 makes the jig and fixture bounds Phase 5 outputs; inventing one is the edit
    §8.1.1 forbids.
    """
    if method is PlacementMethod.FREEHAND:
        return profile["flatFreehandPlacementBound95Deg"]
    if method is PlacementMethod.WALL_FLUSH_FREEHAND:
        return profile["wallFreehandPlacementBound95Deg"]
    return None


def interference_bound_95_deg(state: MagneticState, profile: dict[str, Any]) -> float | None:
    """§19 interference term, or ``None`` when the magnetic state rejects outright."""
    if state is MagneticState.CLEAN:
        return 0.0
    if state is MagneticState.SUSPECT:
        return profile["suspectInterferenceBound95Deg"]
    return None


def instrument_budget_deg(method: PlacementMethod, profile: dict[str, Any]) -> float | None:
    placement = placement_bound_95_deg(method, profile)
    return None if placement is None else profile["usableBound95MaxDeg"] - placement


@dataclass(frozen=True)
class Reachability:
    minimum_reported_bound_95_deg: float | None
    max_reachable_grade: QualityGrade
    lock_reachable: bool
    required_device_floor_at_most_deg: float | None
    explanation: str


@dataclass(frozen=True)
class Finding:
    """A claim contradicted by the arithmetic. §37 rule 12: a finding, not an obstacle."""

    claim_id: str
    problem: str
    claimed: str
    computed: str

    def __str__(self) -> str:
        return (
            f"[{self.claim_id}] {self.problem} -- claimed: {self.claimed} "
            f"-- computed from the shipped constants: {self.computed}"
        )


def compute(
    placement_method: PlacementMethod,
    certification_state: CertificationState,
    magnetic_state: MagneticState,
    profile: dict[str, Any],
    certified_device_floor_95_deg: float | None = None,
) -> Reachability:
    placement = placement_bound_95_deg(placement_method, profile)
    if placement is None:
        return Reachability(
            None,
            QualityGrade.NOT_SUPPORTED,
            False,
            None,
            f"{placement_method.value} has no measured placement bound in "
            f"{profile['configVersion']}; §29.5 makes it a benchmark output and §18.5 forbids "
            "defaulting it to zero, so no grade is computable.",
        )

    interference = interference_bound_95_deg(magnetic_state, profile)
    if interference is None:
        return Reachability(
            None,
            QualityGrade.INVALID,
            False,
            None,
            f"MagneticState {magnetic_state.value} rejects outright in v1 (§16, §18.5); no "
            "measurement is produced, so no grade exists.",
        )

    budget = profile["usableBound95MaxDeg"] - placement
    required_floor = budget - interference

    if certification_state is CertificationState.UNCERTIFIED:
        floor = profile["unknownDeviceFloor95Deg"]
        min_reported = min(180.0, floor + interference + placement)
        return Reachability(
            min_reported,
            quality_grade_for_reported_bound(min_reported, profile),
            min_reported <= profile["usableBound95MaxDeg"],
            required_floor,
            f"unknownDeviceFloor95Deg={floor} + interference={interference} "
            f"+ placement={placement} = {min_reported}; instrument budget for "
            f"{placement_method.value} is {budget}.",
        )

    if certified_device_floor_95_deg is not None:
        min_reported = min(180.0, certified_device_floor_95_deg + interference + placement)
        return Reachability(
            min_reported,
            quality_grade_for_reported_bound(min_reported, profile),
            min_reported <= profile["usableBound95MaxDeg"],
            required_floor,
            f"certified floor={certified_device_floor_95_deg} + interference={interference} "
            f"+ placement={placement} = {min_reported}.",
        )

    # A device floor is strictly positive, so a required floor of zero or less means no
    # certification can make this combination lock.
    lock_possible = required_floor > 0.0
    best_case = min(180.0, interference + placement)
    return Reachability(
        None if lock_possible else best_case,
        QualityGrade.USABLE if lock_possible else quality_grade_for_reported_bound(best_case, profile),
        lock_possible,
        required_floor,
        f"instrument budget for {placement_method.value} is {budget}; after the "
        f"{magnetic_state.value} interference term {interference} a certified "
        f"deviceFloor95Deg must be <= {required_floor} to lock"
        + ("." if lock_possible else ", which no real device floor can satisfy."),
    )


def verify(claims: dict[str, Any], profile: dict[str, Any]) -> list[Finding]:
    findings: list[Finding] = []

    if claims["appliesToConfigVersion"] != profile["configVersion"]:
        findings.append(
            Finding(
                claims["claimsVersion"],
                "The claims document targets a different configuration version, so its rows "
                "were never checked against these constants.",
                f"appliesToConfigVersion={claims['appliesToConfigVersion']}",
                f"configVersion={profile['configVersion']}",
            )
        )

    for claim in claims["combinations"]:
        claimed_grade = QualityGrade[claim["claimedMaxGrade"]]
        method = PlacementMethod(claim["placementMethod"])
        declared_status = claim["placementBoundStatus"]
        actual_status = (
            "UNMEASURED" if placement_bound_95_deg(method, profile) is None else "CONFIGURED"
        )

        if declared_status != actual_status:
            findings.append(
                Finding(
                    claim["id"],
                    "The claim's placement-bound status disagrees with the shipped profile.",
                    f"placementBoundStatus={declared_status}",
                    f"profile {profile['configVersion']} has {actual_status} for {method.value}",
                )
            )
            continue

        if actual_status == "UNMEASURED":
            if claimed_grade is not QualityGrade.NOT_SUPPORTED or claim["claimedLockReachable"]:
                findings.append(
                    Finding(
                        claim["id"],
                        "A placement method with no measured bound may claim no grade and no "
                        "lock (§29.5; §35 'no grade above USABLE without a measured method').",
                        f"claimedMaxGrade={claimed_grade.name}, "
                        f"claimedLockReachable={claim['claimedLockReachable']}",
                        f"placement bound is UNMEASURED for {method.value}",
                    )
                )
            continue

        computed = compute(
            method,
            CertificationState(claim["certificationState"]),
            MagneticState(claim["magneticState"]),
            profile,
        )

        if claimed_grade.is_stronger_than(computed.max_reachable_grade):
            findings.append(
                Finding(
                    claim["id"],
                    "The claimed maximum grade is arithmetically forbidden by the shipped constants.",
                    f"claimedMaxGrade={claimed_grade.name} ({claim['specBasis']})",
                    f"maxReachableGrade={computed.max_reachable_grade.name}; {computed.explanation}",
                )
            )

        if claim["claimedLockReachable"] != computed.lock_reachable:
            findings.append(
                Finding(
                    claim["id"],
                    "The claim disagrees with the arithmetic about whether a Precision Lock is "
                    "reachable at all.",
                    f"claimedLockReachable={claim['claimedLockReachable']}",
                    f"lockReachable={computed.lock_reachable}; {computed.explanation}",
                )
            )

        declared_floor = claim.get("requiresDeviceFloorAtMostDeg")
        if (
            declared_floor is not None
            and computed.required_device_floor_at_most_deg is not None
            and abs(declared_floor - computed.required_device_floor_at_most_deg) > 1e-9
        ):
            findings.append(
                Finding(
                    claim["id"],
                    "The device floor the claim says is required does not match the instrument "
                    "budget the constants leave.",
                    f"requiresDeviceFloorAtMostDeg={declared_floor}",
                    f"required floor={computed.required_device_floor_at_most_deg}; "
                    f"{computed.explanation}",
                )
            )

        if (
            claim["certificationState"] == "CERTIFIED"
            and claim["claimedLockReachable"]
            and declared_floor is None
        ):
            findings.append(
                Finding(
                    claim["id"],
                    "A CERTIFIED lock claim must state the device floor it depends on; §8.1.1 "
                    "makes deviceFloor95Deg an output of the benchmark, not an assumption.",
                    "requiresDeviceFloorAtMostDeg absent",
                    f"required floor={computed.required_device_floor_at_most_deg}",
                )
            )

    return findings
