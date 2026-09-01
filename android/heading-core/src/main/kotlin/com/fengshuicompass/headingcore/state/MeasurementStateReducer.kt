package com.fengshuicompass.headingcore.state

import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.MeasurementState
import com.fengshuicompass.headingcore.model.RejectionReason

/**
 * SPEC.md §18.2 — the event/state/effect reducer over the single `MeasurementState`.
 *
 * §4: "`HeadingEngine` is deterministic for a given ordered event stream + config; time enters
 * as event timestamps, never a wall-clock call inside decision logic." This reducer therefore
 * has no clock, no I/O and no randomness: [reduce] is a pure function of
 * `(snapshot, event) -> (snapshot, effects)`, which is what makes the Phase 3 replay fixtures
 * able to reproduce an outcome exactly.
 *
 * There is exactly **one** measurement-state vocabulary (§6). Any coarser UI vocabulary is
 * derived in the view layer through a total tested mapping and is never persisted as an
 * independent fact, so no such mapping lives here.
 *
 * The rule this reducer exists to make unbreakable: **a timeout emits no measurement.**
 * Failure mode 26 is a Critical failure in which the last valid heading is retained after a
 * provider error and a UI timer repaints it as live.
 */
public object MeasurementStateReducer {

    /** The states in which an aggregation window may be accumulating. */
    private val windowStates = setOf(
        MeasurementState.LEVEL_AND_HOLD,
        MeasurementState.STABILIZING,
        MeasurementState.PRECISION_LOCKED,
        MeasurementState.TARGET_SEEKING,
    )

    /** Terminal states for one acquisition attempt. */
    private val terminalStates = setOf(
        MeasurementState.DEGRADED,
        MeasurementState.INVALID,
        MeasurementState.TIMED_OUT,
    )

    public fun isTerminal(state: MeasurementState): Boolean = state in terminalStates

    /** §7 `HeadingEngine.handle(event) -> [EngineEffect]`, as a pure transition. */
    public fun reduce(
        snapshot: EngineSnapshot,
        event: EngineEvent,
    ): Pair<EngineSnapshot, List<EngineEffect>> = when (event.type) {
        // --- Invalidating events, applicable from any state ---------------------------
        EngineEventType.SESSION_CANCELLED -> invalidate(EngineSnapshot(), MeasurementState.IDLE)

        EngineEventType.APP_BACKGROUNDED ->
            invalidate(snapshot, MeasurementState.INVALID, RejectionReason.APP_BACKGROUNDED)

        EngineEventType.SENSOR_DISCONTINUITY ->
            invalidate(snapshot, MeasurementState.INVALID, RejectionReason.SENSOR_DISCONTINUITY)

        EngineEventType.SCREEN_ORIENTATION_CHANGED -> invalidate(
            snapshot,
            MeasurementState.INVALID,
            RejectionReason.ORIENTATION_CHANGED_DURING_WINDOW,
        )

        EngineEventType.PERMISSION_CHANGED -> invalidate(
            snapshot.copy(hasLocation = false),
            MeasurementState.ACQUIRING_LOCATION,
            RejectionReason.LOCATION_PERMISSION_DENIED,
        )

        EngineEventType.PROVIDER_FAILURE ->
            invalidate(snapshot, MeasurementState.INVALID, RejectionReason.PROVIDER_FAILURE)

        // Failure mode 26: a timeout goes to TIMED_OUT and emits **no** measurement. It MUST
        // NOT freeze the last number.
        EngineEventType.ACQUISITION_TIMED_OUT ->
            snapshot.copy(state = MeasurementState.TIMED_OUT, lockHeld = false)
                .withReasons(RejectionReason.ACQUISITION_TIMEOUT) to
                listOf(
                    EngineEffect.RECORD_STATE_TRANSITION,
                    EngineEffect.RESET_AGGREGATION_WINDOW,
                    EngineEffect.EMIT_NO_MEASUREMENT,
                )

        EngineEventType.LOCATION_LOST -> invalidate(
            snapshot.copy(hasLocation = false),
            MeasurementState.ACQUIRING_LOCATION,
            event.reason ?: RejectionReason.LOCATION_STALE,
        )

        // --- Ordinary progress --------------------------------------------------------
        EngineEventType.SESSION_STARTED ->
            EngineSnapshot(state = MeasurementState.ACQUIRING_LOCATION) to listOf(
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.RESET_AGGREGATION_WINDOW,
                EngineEffect.START_ACQUISITION_TIMEOUT,
            )

        EngineEventType.LOCATION_ACQUIRED ->
            snapshot.copy(hasLocation = true, state = MeasurementState.ACQUIRING_ORIENTATION) to
                listOf(EngineEffect.RECORD_STATE_TRANSITION)

        EngineEventType.ORIENTATION_STREAM_READY ->
            snapshot.copy(hasOrientation = true, state = MeasurementState.CALIBRATION_CHECK) to
                listOf(EngineEffect.RECORD_STATE_TRANSITION)

        // §18.4: distinct from CALIBRATE. The sensor may be perfectly calibrated; the fusion
        // has simply not observed enough rotation to bound its own error.
        EngineEventType.PROVIDER_INITIALIZATION_REQUIRED ->
            snapshot.copy(state = MeasurementState.PROVIDER_INITIALIZING, lockHeld = false)
                .withReasons(RejectionReason.PROVIDER_NOT_INITIALIZED) to
                listOf(
                    EngineEffect.RECORD_STATE_TRANSITION,
                    EngineEffect.RESET_AGGREGATION_WINDOW,
                )

        EngineEventType.PROVIDER_INITIALIZED ->
            snapshot.copy(state = MeasurementState.CALIBRATION_CHECK) to
                listOf(EngineEffect.RECORD_STATE_TRANSITION)

        // Entering Check / Recalibrate invalidates the lock and requires fresh magnetic and
        // stability checks on return (§18.2).
        EngineEventType.CALIBRATION_CHECK_REQUIRED ->
            snapshot.copy(state = MeasurementState.CALIBRATION_CHECK, lockHeld = false) to
                listOf(
                    EngineEffect.RECORD_STATE_TRANSITION,
                    EngineEffect.RESET_AGGREGATION_WINDOW,
                )

        EngineEventType.CALIBRATION_CHECK_COMPLETED ->
            snapshot.copy(state = MeasurementState.MAGNETIC_FIELD_CHECK) to
                listOf(EngineEffect.RECORD_STATE_TRANSITION)

        EngineEventType.MAGNETIC_STATE_OBSERVED -> reduceMagneticState(snapshot, event)

        EngineEventType.TARGET_REQUESTED -> {
            val updated = snapshot.copy(targetRequested = true)
            if (updated.state in windowStates) {
                updated.copy(state = MeasurementState.TARGET_SEEKING, lockHeld = false) to
                    listOf(
                        EngineEffect.RECORD_STATE_TRANSITION,
                        EngineEffect.RESET_AGGREGATION_WINDOW,
                    )
            } else {
                updated to emptyList()
            }
        }

        EngineEventType.TARGET_CLEARED -> {
            val updated = snapshot.copy(targetRequested = false)
            if (updated.state == MeasurementState.TARGET_SEEKING) {
                updated.copy(state = MeasurementState.LEVEL_AND_HOLD) to
                    listOf(EngineEffect.RECORD_STATE_TRANSITION)
            } else {
                updated to emptyList()
            }
        }

        EngineEventType.POSE_INVALID ->
            snapshot.copy(state = MeasurementState.LEVEL_AND_HOLD, lockHeld = false)
                .withReasons(event.reason ?: RejectionReason.DEVICE_NOT_LEVEL) to
                listOf(
                    EngineEffect.RECORD_STATE_TRANSITION,
                    EngineEffect.RESET_AGGREGATION_WINDOW,
                )

        EngineEventType.POSE_VALID ->
            snapshot.copy(state = MeasurementState.STABILIZING) to
                listOf(EngineEffect.RECORD_STATE_TRANSITION)

        // Explicitly *not* a transition: "STABILIZING is satisfied by low movement and a
        // compact cluster over the required duration, not by identical digits" (§18.5).
        EngineEventType.STABILITY_PROGRESSED, EngineEventType.STABILITY_SATISFIED ->
            snapshot.copy(state = MeasurementState.STABILIZING) to emptyList()

        EngineEventType.BOUND_COMPOSED -> reduceBound(snapshot, event)
    }

    /** Fold an ordered event stream — the shape the replay fixtures exercise. */
    public fun reduceSequence(
        events: List<EngineEvent>,
        initial: EngineSnapshot = EngineSnapshot(),
    ): Pair<EngineSnapshot, List<List<EngineEffect>>> {
        var current = initial
        val trace = mutableListOf<List<EngineEffect>>()
        events.forEach { event ->
            val (next, effects) = reduce(current, event)
            current = next
            trace += effects
        }
        return current to trace
    }

    private fun reduceMagneticState(
        snapshot: EngineSnapshot,
        event: EngineEvent,
    ): Pair<EngineSnapshot, List<EngineEffect>> {
        val observed = event.magneticState ?: MagneticState.UNKNOWN
        val updated = snapshot.copy(magneticState = observed)
        return when (observed) {
            // §18.2: a transition to DISTURBED invalidates a live lock immediately.
            MagneticState.DISTURBED -> invalidate(
                updated,
                MeasurementState.INVALID,
                RejectionReason.MAGNETIC_FIELD_DISTURBED,
            )
            MagneticState.INVALID -> invalidate(
                updated,
                MeasurementState.INVALID,
                RejectionReason.MAGNETIC_CALIBRATION_INVALID,
            )
            // §16: UNKNOWN cannot produce a true-heading lock at all in v1.
            MagneticState.UNKNOWN -> invalidate(
                updated,
                MeasurementState.INVALID,
                RejectionReason.MAGNETIC_FIELD_UNKNOWN,
            )
            MagneticState.CLEAN, MagneticState.SUSPECT -> {
                val next = if (updated.targetRequested) {
                    MeasurementState.TARGET_SEEKING
                } else {
                    MeasurementState.LEVEL_AND_HOLD
                }
                updated.copy(state = next) to listOf(EngineEffect.RECORD_STATE_TRANSITION)
            }
        }
    }

    /**
     * §18.5's lock / degraded / invalid distinction, evaluated on the **total** bound.
     *
     * `PRECISION_LOCKED` requires `reportedBound95Deg <= usableBound95MaxDeg`. Between the lock
     * ceiling and `lowConfidenceBound95MaxDeg` the result is `DEGRADED`: shown with its bound
     * and limiting reason, never lock-styled. Above that, or with an unknown bound, it is
     * `INVALID` and produces no measurement.
     */
    private fun reduceBound(
        snapshot: EngineSnapshot,
        event: EngineEvent,
    ): Pair<EngineSnapshot, List<EngineEffect>> {
        val bound = event.reportedBound95Deg
        val lockCeiling = event.lockCeilingDeg
        val lowConfidenceCeiling = event.lowConfidenceCeilingDeg
        if (bound == null || lockCeiling == null || lowConfidenceCeiling == null) {
            return snapshot.copy(state = MeasurementState.INVALID, lockHeld = false)
                .withReasons(RejectionReason.HEADING_ERROR_INVALID) to
                listOf(
                    EngineEffect.RECORD_STATE_TRANSITION,
                    EngineEffect.RESET_AGGREGATION_WINDOW,
                    EngineEffect.EMIT_NO_MEASUREMENT,
                )
        }
        return when {
            bound <= lockCeiling ->
                snapshot.copy(state = MeasurementState.PRECISION_LOCKED, lockHeld = true) to
                    listOf(
                        EngineEffect.RECORD_STATE_TRANSITION,
                        EngineEffect.CLEAR_ACQUISITION_TIMEOUT,
                        EngineEffect.EMIT_LOCKED_MEASUREMENT,
                    )
            bound <= lowConfidenceCeiling ->
                snapshot.copy(state = MeasurementState.DEGRADED, lockHeld = false) to
                    listOf(
                        EngineEffect.RECORD_STATE_TRANSITION,
                        EngineEffect.CLEAR_ACQUISITION_TIMEOUT,
                        EngineEffect.EMIT_DEGRADED_MEASUREMENT,
                    )
            else ->
                snapshot.copy(state = MeasurementState.INVALID, lockHeld = false) to
                    listOf(
                        EngineEffect.RECORD_STATE_TRANSITION,
                        EngineEffect.RESET_AGGREGATION_WINDOW,
                        EngineEffect.EMIT_NO_MEASUREMENT,
                    )
        }
    }

    /**
     * Reset the lock window and emit no measurement.
     *
     * Used by every §18.2 invalidating transition: backgrounding, losing ownership,
     * orientation change, north-reference change, sensor discontinuity, permission or
     * location-mode change.
     */
    private fun invalidate(
        snapshot: EngineSnapshot,
        toState: MeasurementState,
        vararg reasons: RejectionReason,
    ): Pair<EngineSnapshot, List<EngineEffect>> =
        snapshot.copy(state = toState, lockHeld = false).withReasons(*reasons) to
            listOf(
                EngineEffect.RECORD_STATE_TRANSITION,
                EngineEffect.RESET_AGGREGATION_WINDOW,
                EngineEffect.EMIT_NO_MEASUREMENT,
            )
}

/**
 * The closed set of engine inputs.
 *
 * §4 requires cancellation, backgrounding, permission change, provider failure and rotation to
 * be **explicit engine events** rather than ambient conditions read at decision time.
 */
public enum class EngineEventType {
    SESSION_STARTED,
    LOCATION_ACQUIRED,
    LOCATION_LOST,
    ORIENTATION_STREAM_READY,
    PROVIDER_INITIALIZATION_REQUIRED,
    PROVIDER_INITIALIZED,
    CALIBRATION_CHECK_REQUIRED,
    CALIBRATION_CHECK_COMPLETED,
    MAGNETIC_STATE_OBSERVED,
    TARGET_REQUESTED,
    TARGET_CLEARED,
    POSE_VALID,
    POSE_INVALID,
    STABILITY_PROGRESSED,
    STABILITY_SATISFIED,
    BOUND_COMPOSED,
    ACQUISITION_TIMED_OUT,
    APP_BACKGROUNDED,
    SENSOR_DISCONTINUITY,
    SCREEN_ORIENTATION_CHANGED,
    PERMISSION_CHANGED,
    PROVIDER_FAILURE,
    SESSION_CANCELLED,
}

/** Side effects the host performs; the reducer itself performs none. */
public enum class EngineEffect {
    RECORD_STATE_TRANSITION,
    RESET_AGGREGATION_WINDOW,
    START_ACQUISITION_TIMEOUT,
    CLEAR_ACQUISITION_TIMEOUT,
    EMIT_LOCKED_MEASUREMENT,
    EMIT_DEGRADED_MEASUREMENT,
    EMIT_NO_MEASUREMENT,
}

/**
 * One ordered input. [sourceMonotonicNs] is the event's own occurrence time.
 *
 * §22: freshness is computed from mapped source time, never arrival, so the reducer is handed
 * the source time and never asks a clock for "now".
 */
public data class EngineEvent(
    val type: EngineEventType,
    val sourceMonotonicNs: Long = 0L,
    val magneticState: MagneticState? = null,
    val reportedBound95Deg: Double? = null,
    val lockCeilingDeg: Double? = null,
    val lowConfidenceCeilingDeg: Double? = null,
    val reason: RejectionReason? = null,
)

/** The immutable snapshot the UI observes (§4). */
public data class EngineSnapshot(
    val state: MeasurementState = MeasurementState.IDLE,
    val hasLocation: Boolean = false,
    val hasOrientation: Boolean = false,
    val targetRequested: Boolean = false,
    val magneticState: MagneticState = MagneticState.UNKNOWN,
    val rejectionReasons: List<RejectionReason> = emptyList(),
    /**
     * True only while a lock is currently held. A transition to `DISTURBED` clears it
     * immediately (§18.2), which is what stops a stale green reading being repainted.
     */
    val lockHeld: Boolean = false,
) {
    public fun withReasons(vararg reasons: RejectionReason): EngineSnapshot {
        if (reasons.isEmpty()) return this
        val merged = rejectionReasons.toMutableList()
        reasons.forEach { if (it !in merged) merged += it }
        return copy(rejectionReasons = merged)
    }
}
