"""SPEC.md §18.2 — the event/state/effect reducer over the single ``MeasurementState``.

§4: "``HeadingEngine`` is deterministic for a given ordered event stream + config; time enters
as event timestamps, never a wall-clock call inside decision logic." This module therefore has
no clock, no I/O and no randomness: :func:`reduce` is a pure function of
``(snapshot, event) → (snapshot, effects)``, which is what makes the Phase 3 replay fixtures
able to reproduce an outcome exactly.

There is exactly **one** measurement-state vocabulary (§6). Any coarser UI vocabulary is
derived in the view layer through a total tested mapping and is never persisted as an
independent fact, so no such mapping lives here.

The rule this reducer exists to make unbreakable: **a timeout emits no measurement.** Failure
mode 26 is a Critical failure in which the last valid heading is retained after a provider
error and a UI timer repaints it as live. ``TIMED_OUT`` therefore carries
:attr:`EngineEffect.EMIT_NO_MEASUREMENT` and clears the aggregation window.
"""

from __future__ import annotations

from dataclasses import dataclass, field, replace
from enum import Enum

from .enums import MagneticState, MeasurementState, RejectionReason

__all__ = [
    "EngineEffect",
    "EngineEvent",
    "EngineEventType",
    "EngineSnapshot",
    "reduce",
    "reduce_sequence",
]


class EngineEventType(Enum):
    """The closed set of engine inputs.

    §4 requires cancellation, backgrounding, permission change, provider failure and rotation
    to be **explicit engine events** rather than ambient conditions read at decision time.
    """

    SESSION_STARTED = "SESSION_STARTED"
    LOCATION_ACQUIRED = "LOCATION_ACQUIRED"
    LOCATION_LOST = "LOCATION_LOST"
    ORIENTATION_STREAM_READY = "ORIENTATION_STREAM_READY"
    PROVIDER_INITIALIZATION_REQUIRED = "PROVIDER_INITIALIZATION_REQUIRED"
    PROVIDER_INITIALIZED = "PROVIDER_INITIALIZED"
    CALIBRATION_CHECK_REQUIRED = "CALIBRATION_CHECK_REQUIRED"
    CALIBRATION_CHECK_COMPLETED = "CALIBRATION_CHECK_COMPLETED"
    MAGNETIC_STATE_OBSERVED = "MAGNETIC_STATE_OBSERVED"
    TARGET_REQUESTED = "TARGET_REQUESTED"
    TARGET_CLEARED = "TARGET_CLEARED"
    POSE_VALID = "POSE_VALID"
    POSE_INVALID = "POSE_INVALID"
    STABILITY_PROGRESSED = "STABILITY_PROGRESSED"
    STABILITY_SATISFIED = "STABILITY_SATISFIED"
    BOUND_COMPOSED = "BOUND_COMPOSED"
    ACQUISITION_TIMED_OUT = "ACQUISITION_TIMED_OUT"
    APP_BACKGROUNDED = "APP_BACKGROUNDED"
    SENSOR_DISCONTINUITY = "SENSOR_DISCONTINUITY"
    SCREEN_ORIENTATION_CHANGED = "SCREEN_ORIENTATION_CHANGED"
    PERMISSION_CHANGED = "PERMISSION_CHANGED"
    PROVIDER_FAILURE = "PROVIDER_FAILURE"
    SESSION_CANCELLED = "SESSION_CANCELLED"


class EngineEffect(Enum):
    """Side effects the host performs; the reducer itself performs none."""

    RECORD_STATE_TRANSITION = "RECORD_STATE_TRANSITION"
    RESET_AGGREGATION_WINDOW = "RESET_AGGREGATION_WINDOW"
    START_ACQUISITION_TIMEOUT = "START_ACQUISITION_TIMEOUT"
    CLEAR_ACQUISITION_TIMEOUT = "CLEAR_ACQUISITION_TIMEOUT"
    EMIT_LOCKED_MEASUREMENT = "EMIT_LOCKED_MEASUREMENT"
    EMIT_DEGRADED_MEASUREMENT = "EMIT_DEGRADED_MEASUREMENT"
    EMIT_NO_MEASUREMENT = "EMIT_NO_MEASUREMENT"


@dataclass(frozen=True)
class EngineEvent:
    """One ordered input. ``source_monotonic_ns`` is the event's own occurrence time.

    §22: freshness is computed from mapped source time, never arrival, so the reducer is
    handed the source time and never asks a clock for "now".
    """

    type: EngineEventType
    source_monotonic_ns: int = 0
    magnetic_state: MagneticState | None = None
    reported_bound_95_deg: float | None = None
    lock_ceiling_deg: float | None = None
    low_confidence_ceiling_deg: float | None = None
    reason: RejectionReason | None = None


@dataclass(frozen=True)
class EngineSnapshot:
    """The immutable snapshot the UI observes (§4)."""

    state: MeasurementState = MeasurementState.IDLE
    has_location: bool = False
    has_orientation: bool = False
    target_requested: bool = False
    magnetic_state: MagneticState = MagneticState.UNKNOWN
    rejection_reasons: tuple[RejectionReason, ...] = field(default_factory=tuple)
    #: True only while a lock is currently held. A transition to ``DISTURBED`` clears it
    #: immediately (§18.2), which is what stops a stale green reading being repainted.
    lock_held: bool = False

    def with_reasons(self, *reasons: RejectionReason) -> "EngineSnapshot":
        merged = list(self.rejection_reasons)
        for reason in reasons:
            if reason not in merged:
                merged.append(reason)
        return replace(self, rejection_reasons=tuple(merged))


#: The states in which an aggregation window may be accumulating.
_WINDOW_STATES = frozenset(
    {
        MeasurementState.LEVEL_AND_HOLD,
        MeasurementState.STABILIZING,
        MeasurementState.PRECISION_LOCKED,
        MeasurementState.TARGET_SEEKING,
    }
)

#: Terminal states for one acquisition attempt.
_TERMINAL_STATES = frozenset(
    {MeasurementState.DEGRADED, MeasurementState.INVALID, MeasurementState.TIMED_OUT}
)


def _invalidate(
    snapshot: EngineSnapshot,
    to_state: MeasurementState,
    *reasons: RejectionReason,
) -> tuple[EngineSnapshot, tuple[EngineEffect, ...]]:
    """Reset the lock window and emit no measurement.

    Used by every §18.2 invalidating transition: backgrounding, losing ownership, orientation
    change, north-reference change, sensor discontinuity, permission or location-mode change.
    """
    return (
        replace(snapshot, state=to_state, lock_held=False).with_reasons(*reasons),
        (
            EngineEffect.RECORD_STATE_TRANSITION,
            EngineEffect.RESET_AGGREGATION_WINDOW,
            EngineEffect.EMIT_NO_MEASUREMENT,
        ),
    )


def reduce(  # noqa: C901 - a state machine reads better as one explicit table
    snapshot: EngineSnapshot, event: EngineEvent
) -> tuple[EngineSnapshot, tuple[EngineEffect, ...]]:
    """§7 ``HeadingEngine.handle(event) -> [EngineEffect]``, as a pure transition."""
    kind = event.type

    # --- Invalidating events, applicable from any state -----------------------------------
    if kind is EngineEventType.SESSION_CANCELLED:
        return _invalidate(EngineSnapshot(), MeasurementState.IDLE)

    if kind is EngineEventType.APP_BACKGROUNDED:
        return _invalidate(snapshot, MeasurementState.INVALID, RejectionReason.APP_BACKGROUNDED)

    if kind is EngineEventType.SENSOR_DISCONTINUITY:
        return _invalidate(snapshot, MeasurementState.INVALID, RejectionReason.SENSOR_DISCONTINUITY)

    if kind is EngineEventType.SCREEN_ORIENTATION_CHANGED:
        return _invalidate(
            snapshot,
            MeasurementState.INVALID,
            RejectionReason.ORIENTATION_CHANGED_DURING_WINDOW,
        )

    if kind is EngineEventType.PERMISSION_CHANGED:
        return _invalidate(
            replace(snapshot, has_location=False),
            MeasurementState.ACQUIRING_LOCATION,
            RejectionReason.LOCATION_PERMISSION_DENIED,
        )

    if kind is EngineEventType.PROVIDER_FAILURE:
        return _invalidate(snapshot, MeasurementState.INVALID, RejectionReason.PROVIDER_FAILURE)

    if kind is EngineEventType.ACQUISITION_TIMED_OUT:
        # Failure mode 26: a timeout goes to TIMED_OUT and emits **no** measurement. It MUST
        # NOT freeze the last number.
        return (
            replace(snapshot, state=MeasurementState.TIMED_OUT, lock_held=False).with_reasons(
                RejectionReason.ACQUISITION_TIMEOUT
            ),
            (
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.RESET_AGGREGATION_WINDOW,
                EngineEffect.EMIT_NO_MEASUREMENT,
            ),
        )

    if kind is EngineEventType.LOCATION_LOST:
        return _invalidate(
            replace(snapshot, has_location=False),
            MeasurementState.ACQUIRING_LOCATION,
            event.reason or RejectionReason.LOCATION_STALE,
        )

    # --- Ordinary progress ----------------------------------------------------------------
    if kind is EngineEventType.SESSION_STARTED:
        return (
            EngineSnapshot(state=MeasurementState.ACQUIRING_LOCATION),
            (
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.RESET_AGGREGATION_WINDOW,
                EngineEffect.START_ACQUISITION_TIMEOUT,
            ),
        )

    if kind is EngineEventType.LOCATION_ACQUIRED:
        updated = replace(snapshot, has_location=True, state=MeasurementState.ACQUIRING_ORIENTATION)
        return updated, (EngineEffect.RECORD_STATE_TRANSITION,)

    if kind is EngineEventType.ORIENTATION_STREAM_READY:
        updated = replace(
            snapshot, has_orientation=True, state=MeasurementState.CALIBRATION_CHECK
        )
        return updated, (EngineEffect.RECORD_STATE_TRANSITION,)

    if kind is EngineEventType.PROVIDER_INITIALIZATION_REQUIRED:
        # §18.4: distinct from CALIBRATE. The sensor may be perfectly calibrated; the fusion
        # has simply not observed enough rotation to bound its own error.
        return (
            replace(snapshot, state=MeasurementState.PROVIDER_INITIALIZING, lock_held=False)
            .with_reasons(RejectionReason.PROVIDER_NOT_INITIALIZED),
            (EngineEffect.RECORD_STATE_TRANSITION, EngineEffect.RESET_AGGREGATION_WINDOW),
        )

    if kind is EngineEventType.PROVIDER_INITIALIZED:
        return (
            replace(snapshot, state=MeasurementState.CALIBRATION_CHECK),
            (EngineEffect.RECORD_STATE_TRANSITION,),
        )

    if kind is EngineEventType.CALIBRATION_CHECK_REQUIRED:
        # Entering Check / Recalibrate invalidates the lock and requires fresh magnetic and
        # stability checks on return (§18.2).
        return (
            replace(snapshot, state=MeasurementState.CALIBRATION_CHECK, lock_held=False),
            (EngineEffect.RECORD_STATE_TRANSITION, EngineEffect.RESET_AGGREGATION_WINDOW),
        )

    if kind is EngineEventType.CALIBRATION_CHECK_COMPLETED:
        return (
            replace(snapshot, state=MeasurementState.MAGNETIC_FIELD_CHECK),
            (EngineEffect.RECORD_STATE_TRANSITION,),
        )

    if kind is EngineEventType.MAGNETIC_STATE_OBSERVED:
        observed = event.magnetic_state or MagneticState.UNKNOWN
        updated = replace(snapshot, magnetic_state=observed)
        if observed is MagneticState.DISTURBED:
            # §18.2: a transition to DISTURBED invalidates a live lock immediately.
            return _invalidate(
                updated, MeasurementState.INVALID, RejectionReason.MAGNETIC_FIELD_DISTURBED
            )
        if observed is MagneticState.INVALID:
            return _invalidate(
                updated, MeasurementState.INVALID, RejectionReason.MAGNETIC_CALIBRATION_INVALID
            )
        if observed is MagneticState.UNKNOWN:
            # §16: UNKNOWN cannot produce a true-heading lock at all in v1.
            return _invalidate(
                updated, MeasurementState.INVALID, RejectionReason.MAGNETIC_FIELD_UNKNOWN
            )
        next_state = (
            MeasurementState.TARGET_SEEKING
            if updated.target_requested
            else MeasurementState.LEVEL_AND_HOLD
        )
        return replace(updated, state=next_state), (EngineEffect.RECORD_STATE_TRANSITION,)

    if kind is EngineEventType.TARGET_REQUESTED:
        updated = replace(snapshot, target_requested=True)
        if updated.state in _WINDOW_STATES:
            return (
                replace(updated, state=MeasurementState.TARGET_SEEKING, lock_held=False),
                (EngineEffect.RECORD_STATE_TRANSITION, EngineEffect.RESET_AGGREGATION_WINDOW),
            )
        return updated, ()

    if kind is EngineEventType.TARGET_CLEARED:
        updated = replace(snapshot, target_requested=False)
        if updated.state is MeasurementState.TARGET_SEEKING:
            return (
                replace(updated, state=MeasurementState.LEVEL_AND_HOLD),
                (EngineEffect.RECORD_STATE_TRANSITION,),
            )
        return updated, ()

    if kind is EngineEventType.POSE_INVALID:
        return (
            replace(snapshot, state=MeasurementState.LEVEL_AND_HOLD, lock_held=False).with_reasons(
                event.reason or RejectionReason.DEVICE_NOT_LEVEL
            ),
            (EngineEffect.RECORD_STATE_TRANSITION, EngineEffect.RESET_AGGREGATION_WINDOW),
        )

    if kind is EngineEventType.POSE_VALID:
        return (
            replace(snapshot, state=MeasurementState.STABILIZING),
            (EngineEffect.RECORD_STATE_TRANSITION,),
        )

    if kind is EngineEventType.STABILITY_PROGRESSED:
        # Explicitly *not* a transition: "STABILIZING is satisfied by low movement and a
        # compact cluster over the required duration, not by identical digits" (§18.5).
        return replace(snapshot, state=MeasurementState.STABILIZING), ()

    if kind is EngineEventType.STABILITY_SATISFIED:
        return replace(snapshot, state=MeasurementState.STABILIZING), ()

    if kind is EngineEventType.BOUND_COMPOSED:
        return _reduce_bound(snapshot, event)

    raise ValueError(f"unhandled engine event {kind!r}")


def _reduce_bound(
    snapshot: EngineSnapshot, event: EngineEvent
) -> tuple[EngineSnapshot, tuple[EngineEffect, ...]]:
    """§18.5's lock / degraded / invalid distinction, evaluated on the **total** bound.

    ``PRECISION_LOCKED`` requires ``reportedBound95Deg <= usableBound95MaxDeg``. Between the
    lock ceiling and ``lowConfidenceBound95MaxDeg`` the result is ``DEGRADED``: shown with its
    bound and limiting reason, never lock-styled. Above that, or with an unknown bound, it is
    ``INVALID`` and produces no measurement.
    """
    bound = event.reported_bound_95_deg
    lock_ceiling = event.lock_ceiling_deg
    low_confidence_ceiling = event.low_confidence_ceiling_deg
    if bound is None or lock_ceiling is None or low_confidence_ceiling is None:
        return (
            replace(snapshot, state=MeasurementState.INVALID, lock_held=False).with_reasons(
                RejectionReason.HEADING_ERROR_INVALID
            ),
            (
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.RESET_AGGREGATION_WINDOW,
                EngineEffect.EMIT_NO_MEASUREMENT,
            ),
        )
    if bound <= lock_ceiling:
        return (
            replace(snapshot, state=MeasurementState.PRECISION_LOCKED, lock_held=True),
            (
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.CLEAR_ACQUISITION_TIMEOUT,
                EngineEffect.EMIT_LOCKED_MEASUREMENT,
            ),
        )
    if bound <= low_confidence_ceiling:
        return (
            replace(snapshot, state=MeasurementState.DEGRADED, lock_held=False),
            (
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.CLEAR_ACQUISITION_TIMEOUT,
                EngineEffect.EMIT_DEGRADED_MEASUREMENT,
            ),
        )
    return (
        replace(snapshot, state=MeasurementState.INVALID, lock_held=False),
        (
            EngineEffect.RECORD_STATE_TRANSITION,
            EngineEffect.RESET_AGGREGATION_WINDOW,
            EngineEffect.EMIT_NO_MEASUREMENT,
        ),
    )


def reduce_sequence(
    events: list[EngineEvent], snapshot: EngineSnapshot | None = None
) -> tuple[EngineSnapshot, list[tuple[EngineEffect, ...]]]:
    """Fold an ordered event stream — the shape the replay fixtures exercise."""
    current = snapshot or EngineSnapshot()
    trace: list[tuple[EngineEffect, ...]] = []
    for event in events:
        current, effects = reduce(current, event)
        trace.append(effects)
    return current, trace


def is_terminal(state: MeasurementState) -> bool:
    return state in _TERMINAL_STATES
