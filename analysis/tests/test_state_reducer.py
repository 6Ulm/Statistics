"""SPEC.md §18.2 / §18.5 — the event/state/effect reducer over the single ``MeasurementState``."""

from __future__ import annotations

import pytest

from fscompass_analysis import fixtures, state
from fscompass_analysis.enums import MagneticState, MeasurementState, RejectionReason


@pytest.fixture(scope="module")
def reducer_fixture():
    return fixtures.load(fixtures.STATE_REDUCER)


def _event(document) -> state.EngineEvent:
    return state.EngineEvent(
        type=state.EngineEventType(document["type"]),
        magnetic_state=(
            MagneticState(document["magneticState"]) if "magneticState" in document else None
        ),
        reported_bound_95_deg=document.get("reportedBound95Deg"),
        lock_ceiling_deg=document.get("lockCeilingDeg"),
        low_confidence_ceiling_deg=document.get("lowConfidenceCeilingDeg"),
    )


def test_sequences_match_the_frozen_fixture(reducer_fixture):
    for sequence in reducer_fixture["sequences"]:
        events = [_event(document) for document in sequence["events"]]
        snapshot, trace = state.reduce_sequence(events)
        assert snapshot.state.value == sequence["expectedFinalState"], sequence["id"]
        assert [effect.value for effect in trace[-1]] == sequence[
            "expectedFinalEffects"
        ], sequence["id"]
        assert snapshot.lock_held == sequence["expectedLockHeld"], sequence["id"]


def test_the_reducer_is_deterministic(reducer_fixture):
    """§4: deterministic for a given ordered event stream + config; time enters as event
    timestamps, never a wall-clock call inside decision logic."""
    for sequence in reducer_fixture["sequences"]:
        events = [_event(document) for document in sequence["events"]]
        first = state.reduce_sequence(events)
        second = state.reduce_sequence(events)
        assert first == second, sequence["id"]


def test_a_timeout_emits_no_measurement(reducer_fixture):
    """Failure mode 26 / §18.2: a timeout goes to ``TIMED_OUT`` and emits **no** measurement.

    It MUST NOT freeze the last number — the scenario where a UI timer keeps repainting a
    stale heading as live is Critical precisely because nothing looks wrong.
    """
    snapshot, effects = state.reduce(
        state.EngineSnapshot(state=MeasurementState.STABILIZING, lock_held=True),
        state.EngineEvent(state.EngineEventType.ACQUISITION_TIMED_OUT),
    )
    assert snapshot.state is MeasurementState.TIMED_OUT
    assert not snapshot.lock_held
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects
    assert state.EngineEffect.RESET_AGGREGATION_WINDOW in effects
    assert state.EngineEffect.EMIT_LOCKED_MEASUREMENT not in effects
    assert state.EngineEffect.EMIT_DEGRADED_MEASUREMENT not in effects
    assert RejectionReason.ACQUISITION_TIMEOUT in snapshot.rejection_reasons


def test_a_transition_to_disturbed_invalidates_a_live_lock_immediately(profile):
    """§18.2: the app MUST NOT show a green/high-confidence reading while the engine
    considers the measurement disturbed."""
    locked, _ = state.reduce(
        state.EngineSnapshot(state=MeasurementState.STABILIZING),
        state.EngineEvent(
            state.EngineEventType.BOUND_COMPOSED,
            reported_bound_95_deg=1.7,
            lock_ceiling_deg=profile["usableBound95MaxDeg"],
            low_confidence_ceiling_deg=profile["lowConfidenceBound95MaxDeg"],
        ),
    )
    assert locked.lock_held
    disturbed, effects = state.reduce(
        locked,
        state.EngineEvent(
            state.EngineEventType.MAGNETIC_STATE_OBSERVED,
            magnetic_state=MagneticState.DISTURBED,
        ),
    )
    assert not disturbed.lock_held
    assert disturbed.state is MeasurementState.INVALID
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects
    assert RejectionReason.MAGNETIC_FIELD_DISTURBED in disturbed.rejection_reasons


@pytest.mark.parametrize(
    ("event_type", "reason"),
    [
        (state.EngineEventType.APP_BACKGROUNDED, RejectionReason.APP_BACKGROUNDED),
        (state.EngineEventType.SENSOR_DISCONTINUITY, RejectionReason.SENSOR_DISCONTINUITY),
        (
            state.EngineEventType.SCREEN_ORIENTATION_CHANGED,
            RejectionReason.ORIENTATION_CHANGED_DURING_WINDOW,
        ),
        (state.EngineEventType.PERMISSION_CHANGED, RejectionReason.LOCATION_PERMISSION_DENIED),
        (state.EngineEventType.PROVIDER_FAILURE, RejectionReason.PROVIDER_FAILURE),
    ],
)
def test_every_invalidating_event_resets_the_lock_window(event_type, reason, profile):
    """§18.2: "Backgrounding, losing ownership, orientation change, north-reference change,
    sensor discontinuity, or permission/location-mode change resets the lock window"."""
    locked, _ = state.reduce(
        state.EngineSnapshot(state=MeasurementState.STABILIZING),
        state.EngineEvent(
            state.EngineEventType.BOUND_COMPOSED,
            reported_bound_95_deg=1.7,
            lock_ceiling_deg=profile["usableBound95MaxDeg"],
            low_confidence_ceiling_deg=profile["lowConfidenceBound95MaxDeg"],
        ),
    )
    assert locked.lock_held
    after, effects = state.reduce(locked, state.EngineEvent(event_type))
    assert not after.lock_held
    assert state.EngineEffect.RESET_AGGREGATION_WINDOW in effects
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects
    assert reason in after.rejection_reasons


def test_lock_degraded_and_invalid_are_three_distinct_outcomes(profile):
    """§18.5: this is what resolves the otherwise-contradictory 5° lock ceiling and 5–10°
    grade tier."""
    lock_ceiling = profile["usableBound95MaxDeg"]
    low_confidence_ceiling = profile["lowConfidenceBound95MaxDeg"]
    expectations = [
        (0.5, MeasurementState.PRECISION_LOCKED, state.EngineEffect.EMIT_LOCKED_MEASUREMENT),
        (lock_ceiling, MeasurementState.PRECISION_LOCKED, state.EngineEffect.EMIT_LOCKED_MEASUREMENT),
        (
            lock_ceiling + 0.001,
            MeasurementState.DEGRADED,
            state.EngineEffect.EMIT_DEGRADED_MEASUREMENT,
        ),
        (
            low_confidence_ceiling,
            MeasurementState.DEGRADED,
            state.EngineEffect.EMIT_DEGRADED_MEASUREMENT,
        ),
        (
            low_confidence_ceiling + 0.001,
            MeasurementState.INVALID,
            state.EngineEffect.EMIT_NO_MEASUREMENT,
        ),
    ]
    for bound, expected_state, expected_effect in expectations:
        snapshot, effects = state.reduce(
            state.EngineSnapshot(state=MeasurementState.STABILIZING),
            state.EngineEvent(
                state.EngineEventType.BOUND_COMPOSED,
                reported_bound_95_deg=bound,
                lock_ceiling_deg=lock_ceiling,
                low_confidence_ceiling_deg=low_confidence_ceiling,
            ),
        )
        assert snapshot.state is expected_state, bound
        assert expected_effect in effects, bound
        assert snapshot.lock_held == (expected_state is MeasurementState.PRECISION_LOCKED), bound


def test_an_unknown_bound_is_invalid_not_degraded(profile):
    """§18.5: "Above ``lowConfidenceBound95MaxDeg``, **or with an unknown bound**, it is
    ``INVALID`` and produces no measurement"."""
    snapshot, effects = state.reduce(
        state.EngineSnapshot(state=MeasurementState.STABILIZING),
        state.EngineEvent(state.EngineEventType.BOUND_COMPOSED, reported_bound_95_deg=None),
    )
    assert snapshot.state is MeasurementState.INVALID
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects


def test_unknown_magnetic_state_cannot_produce_a_true_heading_lock():
    """§16: "``UNKNOWN`` ... in v1 cannot produce a true-heading lock at all"."""
    snapshot, effects = state.reduce(
        state.EngineSnapshot(state=MeasurementState.MAGNETIC_FIELD_CHECK),
        state.EngineEvent(
            state.EngineEventType.MAGNETIC_STATE_OBSERVED, magnetic_state=MagneticState.UNKNOWN
        ),
    )
    assert snapshot.state is MeasurementState.INVALID
    assert RejectionReason.MAGNETIC_FIELD_UNKNOWN in snapshot.rejection_reasons
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects


def test_provider_initialization_is_distinct_from_calibration():
    """§18.4 / failure mode 37: sending a user to calibration when the truth was provider
    initialization teaches distrust of the prompt.

    The two are separate events reaching separate states, so the confusion cannot arise from
    the state machine.
    """
    initializing, _ = state.reduce(
        state.EngineSnapshot(state=MeasurementState.ACQUIRING_ORIENTATION),
        state.EngineEvent(state.EngineEventType.PROVIDER_INITIALIZATION_REQUIRED),
    )
    calibrating, _ = state.reduce(
        state.EngineSnapshot(state=MeasurementState.ACQUIRING_ORIENTATION),
        state.EngineEvent(state.EngineEventType.CALIBRATION_CHECK_REQUIRED),
    )
    assert initializing.state is MeasurementState.PROVIDER_INITIALIZING
    assert calibrating.state is MeasurementState.CALIBRATION_CHECK
    assert initializing.state is not calibrating.state
    assert RejectionReason.PROVIDER_NOT_INITIALIZED in initializing.rejection_reasons


def test_entering_a_calibration_check_invalidates_the_lock(profile):
    """§18.2: "Entering Check / Recalibrate invalidates the lock and requires fresh
    magnetic/stability checks on return"."""
    locked, _ = state.reduce(
        state.EngineSnapshot(state=MeasurementState.STABILIZING),
        state.EngineEvent(
            state.EngineEventType.BOUND_COMPOSED,
            reported_bound_95_deg=1.0,
            lock_ceiling_deg=profile["usableBound95MaxDeg"],
            low_confidence_ceiling_deg=profile["lowConfidenceBound95MaxDeg"],
        ),
    )
    after, effects = state.reduce(
        locked, state.EngineEvent(state.EngineEventType.CALIBRATION_CHECK_REQUIRED)
    )
    assert not after.lock_held
    assert state.EngineEffect.RESET_AGGREGATION_WINDOW in effects


def test_stability_progress_is_not_a_lock():
    """§18.5: ``STABILIZING`` is satisfied by low movement and a compact cluster over the
    required duration, not by identical digits — and never by progress alone."""
    snapshot = state.EngineSnapshot(state=MeasurementState.STABILIZING)
    for _ in range(100):
        snapshot, effects = state.reduce(
            snapshot, state.EngineEvent(state.EngineEventType.STABILITY_PROGRESSED)
        )
        assert snapshot.state is MeasurementState.STABILIZING
        assert not snapshot.lock_held
        assert state.EngineEffect.EMIT_LOCKED_MEASUREMENT not in effects


def test_cancelling_a_session_returns_to_idle_with_nothing_retained():
    """§4: starting a session cancels and awaits the previous one; a torn-down session's
    state must not enter a later one."""
    snapshot, effects = state.reduce(
        state.EngineSnapshot(
            state=MeasurementState.PRECISION_LOCKED,
            lock_held=True,
            rejection_reasons=(RejectionReason.DEVICE_MOVING,),
            magnetic_state=MagneticState.CLEAN,
            has_location=True,
        ),
        state.EngineEvent(state.EngineEventType.SESSION_CANCELLED),
    )
    assert snapshot.state is MeasurementState.IDLE
    assert not snapshot.lock_held
    assert snapshot.rejection_reasons == ()
    assert not snapshot.has_location
    assert state.EngineEffect.EMIT_NO_MEASUREMENT in effects


def test_every_event_type_is_handled():
    """A reducer that silently ignores an event is how an explicit engine event turns back
    into an ambient condition (§4)."""
    snapshot = state.EngineSnapshot(state=MeasurementState.STABILIZING, target_requested=True)
    for event_type in state.EngineEventType:
        event = state.EngineEvent(event_type)
        if event_type is state.EngineEventType.MAGNETIC_STATE_OBSERVED:
            event = state.EngineEvent(event_type, magnetic_state=MagneticState.CLEAN)
        result, _ = state.reduce(snapshot, event)
        assert isinstance(result, state.EngineSnapshot), event_type


def test_the_measurement_state_vocabulary_is_the_only_one():
    """§6: "There is exactly **one** measurement-state vocabulary, ``MeasurementState``"."""
    reachable = set()
    snapshot = state.EngineSnapshot(state=MeasurementState.STABILIZING, target_requested=True)
    for event_type in state.EngineEventType:
        event = state.EngineEvent(event_type)
        if event_type is state.EngineEventType.MAGNETIC_STATE_OBSERVED:
            event = state.EngineEvent(event_type, magnetic_state=MagneticState.CLEAN)
        result, _ = state.reduce(snapshot, event)
        reachable.add(result.state)
    assert reachable <= set(MeasurementState)
