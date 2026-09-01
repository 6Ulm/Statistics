package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.PrecisionProfile
import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.model.MagneticState
import com.fengshuicompass.headingcore.model.MeasurementState
import com.fengshuicompass.headingcore.model.RejectionReason
import com.fengshuicompass.headingcore.state.EngineEffect
import com.fengshuicompass.headingcore.state.EngineEvent
import com.fengshuicompass.headingcore.state.EngineEventType
import com.fengshuicompass.headingcore.state.EngineSnapshot
import com.fengshuicompass.headingcore.state.MeasurementStateReducer
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.boolean
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertFalse
import org.junit.jupiter.api.Assertions.assertSame
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test

/** SPEC.md §18.2 / §18.5 — the reducer over the single `MeasurementState`. */
class MeasurementStateReducerTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val fixture: JsonObject
        get() = json.parseToJsonElement(SharedArtifacts.stateReducerFixture.readText()).jsonObject

    private val profile: PrecisionProfile
        get() = PrecisionProfile.load(SharedArtifacts.precisionProfileFile)

    private fun event(document: JsonObject) = EngineEvent(
        type = EngineEventType.valueOf(document["type"]!!.jsonPrimitive.content),
        magneticState = document["magneticState"]?.let {
            MagneticState.valueOf(it.jsonPrimitive.content)
        },
        reportedBound95Deg = document["reportedBound95Deg"]?.jsonPrimitive?.double,
        lockCeilingDeg = document["lockCeilingDeg"]?.jsonPrimitive?.double,
        lowConfidenceCeilingDeg = document["lowConfidenceCeilingDeg"]?.jsonPrimitive?.double,
    )

    private fun lockedSnapshot(): EngineSnapshot {
        val (locked, _) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.STABILIZING),
            EngineEvent(
                EngineEventType.BOUND_COMPOSED,
                reportedBound95Deg = 1.7,
                lockCeilingDeg = profile.usableBound95MaxDeg,
                lowConfidenceCeilingDeg = profile.lowConfidenceBound95MaxDeg,
            ),
        )
        check(locked.lockHeld)
        return locked
    }

    @Test
    @DisplayName("§18.2: sequences match the frozen fixture")
    fun sequencesMatchTheFrozenFixture() {
        fixture["sequences"]!!.jsonArray.forEach { entry ->
            val sequence = entry.jsonObject
            val id = sequence["id"]!!.jsonPrimitive.content
            val events = sequence["events"]!!.jsonArray.map { event(it.jsonObject) }
            val (snapshot, trace) = MeasurementStateReducer.reduceSequence(events)
            assertEquals(sequence["expectedFinalState"]!!.jsonPrimitive.content, snapshot.state.wire, id)
            assertEquals(
                sequence["expectedFinalEffects"]!!.jsonArray.map { it.jsonPrimitive.content },
                trace.last().map { it.name },
                id,
            )
            assertEquals(sequence["expectedLockHeld"]!!.jsonPrimitive.boolean, snapshot.lockHeld, id)
        }
    }

    @Test
    @DisplayName("§4: the reducer is deterministic for a given ordered event stream")
    fun theReducerIsDeterministic() {
        fixture["sequences"]!!.jsonArray.forEach { entry ->
            val events = entry.jsonObject["events"]!!.jsonArray.map { event(it.jsonObject) }
            assertEquals(
                MeasurementStateReducer.reduceSequence(events),
                MeasurementStateReducer.reduceSequence(events),
                entry.jsonObject["id"]!!.jsonPrimitive.content,
            )
        }
    }

    @Test
    @DisplayName("failure mode 26: a timeout emits no measurement and never freezes the last one")
    fun aTimeoutEmitsNoMeasurement() {
        val (snapshot, effects) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.STABILIZING, lockHeld = true),
            EngineEvent(EngineEventType.ACQUISITION_TIMED_OUT),
        )
        assertSame(MeasurementState.TIMED_OUT, snapshot.state)
        assertFalse(snapshot.lockHeld)
        assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects)
        assertTrue(EngineEffect.RESET_AGGREGATION_WINDOW in effects)
        assertFalse(EngineEffect.EMIT_LOCKED_MEASUREMENT in effects)
        assertFalse(EngineEffect.EMIT_DEGRADED_MEASUREMENT in effects)
        assertTrue(RejectionReason.ACQUISITION_TIMEOUT in snapshot.rejectionReasons)
    }

    @Test
    @DisplayName("§18.2: a transition to DISTURBED invalidates a live lock immediately")
    fun disturbedInvalidatesALiveLockImmediately() {
        val (disturbed, effects) = MeasurementStateReducer.reduce(
            lockedSnapshot(),
            EngineEvent(
                EngineEventType.MAGNETIC_STATE_OBSERVED,
                magneticState = MagneticState.DISTURBED,
            ),
        )
        assertFalse(disturbed.lockHeld)
        assertSame(MeasurementState.INVALID, disturbed.state)
        assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects)
        assertTrue(RejectionReason.MAGNETIC_FIELD_DISTURBED in disturbed.rejectionReasons)
    }

    @Test
    @DisplayName("§18.2: every invalidating event resets the lock window")
    fun everyInvalidatingEventResetsTheLockWindow() {
        mapOf(
            EngineEventType.APP_BACKGROUNDED to RejectionReason.APP_BACKGROUNDED,
            EngineEventType.SENSOR_DISCONTINUITY to RejectionReason.SENSOR_DISCONTINUITY,
            EngineEventType.SCREEN_ORIENTATION_CHANGED to
                RejectionReason.ORIENTATION_CHANGED_DURING_WINDOW,
            EngineEventType.PERMISSION_CHANGED to RejectionReason.LOCATION_PERMISSION_DENIED,
            EngineEventType.PROVIDER_FAILURE to RejectionReason.PROVIDER_FAILURE,
        ).forEach { (type, reason) ->
            val (after, effects) = MeasurementStateReducer.reduce(
                lockedSnapshot(),
                EngineEvent(type),
            )
            assertFalse(after.lockHeld, type.name)
            assertTrue(EngineEffect.RESET_AGGREGATION_WINDOW in effects, type.name)
            assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects, type.name)
            assertTrue(reason in after.rejectionReasons, type.name)
        }
    }

    @Test
    @DisplayName("§18.5: lock, degraded and invalid are three distinct outcomes")
    fun lockDegradedAndInvalidAreThreeDistinctOutcomes() {
        val lockCeiling = profile.usableBound95MaxDeg
        val lowConfidenceCeiling = profile.lowConfidenceBound95MaxDeg
        listOf(
            Triple(0.5, MeasurementState.PRECISION_LOCKED, EngineEffect.EMIT_LOCKED_MEASUREMENT),
            Triple(lockCeiling, MeasurementState.PRECISION_LOCKED, EngineEffect.EMIT_LOCKED_MEASUREMENT),
            Triple(
                lockCeiling + 0.001,
                MeasurementState.DEGRADED,
                EngineEffect.EMIT_DEGRADED_MEASUREMENT,
            ),
            Triple(
                lowConfidenceCeiling,
                MeasurementState.DEGRADED,
                EngineEffect.EMIT_DEGRADED_MEASUREMENT,
            ),
            Triple(
                lowConfidenceCeiling + 0.001,
                MeasurementState.INVALID,
                EngineEffect.EMIT_NO_MEASUREMENT,
            ),
        ).forEach { (bound, expectedState, expectedEffect) ->
            val (snapshot, effects) = MeasurementStateReducer.reduce(
                EngineSnapshot(state = MeasurementState.STABILIZING),
                EngineEvent(
                    EngineEventType.BOUND_COMPOSED,
                    reportedBound95Deg = bound,
                    lockCeilingDeg = lockCeiling,
                    lowConfidenceCeilingDeg = lowConfidenceCeiling,
                ),
            )
            assertSame(expectedState, snapshot.state, "$bound")
            assertTrue(expectedEffect in effects, "$bound")
            assertEquals(expectedState == MeasurementState.PRECISION_LOCKED, snapshot.lockHeld, "$bound")
        }
    }

    @Test
    @DisplayName("§18.5: an unknown bound is INVALID, not DEGRADED")
    fun anUnknownBoundIsInvalid() {
        val (snapshot, effects) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.STABILIZING),
            EngineEvent(EngineEventType.BOUND_COMPOSED, reportedBound95Deg = null),
        )
        assertSame(MeasurementState.INVALID, snapshot.state)
        assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects)
    }

    @Test
    @DisplayName("§16: UNKNOWN cannot produce a true-heading lock at all in v1")
    fun unknownMagneticStateCannotLock() {
        val (snapshot, effects) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.MAGNETIC_FIELD_CHECK),
            EngineEvent(
                EngineEventType.MAGNETIC_STATE_OBSERVED,
                magneticState = MagneticState.UNKNOWN,
            ),
        )
        assertSame(MeasurementState.INVALID, snapshot.state)
        assertTrue(RejectionReason.MAGNETIC_FIELD_UNKNOWN in snapshot.rejectionReasons)
        assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects)
    }

    @Test
    @DisplayName("§18.4/failure mode 37: provider initialization is distinct from calibration")
    fun providerInitializationIsDistinctFromCalibration() {
        val (initializing, _) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.ACQUIRING_ORIENTATION),
            EngineEvent(EngineEventType.PROVIDER_INITIALIZATION_REQUIRED),
        )
        val (calibrating, _) = MeasurementStateReducer.reduce(
            EngineSnapshot(state = MeasurementState.ACQUIRING_ORIENTATION),
            EngineEvent(EngineEventType.CALIBRATION_CHECK_REQUIRED),
        )
        assertSame(MeasurementState.PROVIDER_INITIALIZING, initializing.state)
        assertSame(MeasurementState.CALIBRATION_CHECK, calibrating.state)
        assertTrue(RejectionReason.PROVIDER_NOT_INITIALIZED in initializing.rejectionReasons)
    }

    @Test
    @DisplayName("§18.2: entering a calibration check invalidates the lock")
    fun enteringACalibrationCheckInvalidatesTheLock() {
        val (after, effects) = MeasurementStateReducer.reduce(
            lockedSnapshot(),
            EngineEvent(EngineEventType.CALIBRATION_CHECK_REQUIRED),
        )
        assertFalse(after.lockHeld)
        assertTrue(EngineEffect.RESET_AGGREGATION_WINDOW in effects)
    }

    @Test
    @DisplayName("§18.5: stability progress alone is never a lock")
    fun stabilityProgressIsNotALock() {
        var snapshot = EngineSnapshot(state = MeasurementState.STABILIZING)
        repeat(100) {
            val (next, effects) = MeasurementStateReducer.reduce(
                snapshot,
                EngineEvent(EngineEventType.STABILITY_PROGRESSED),
            )
            snapshot = next
            assertSame(MeasurementState.STABILIZING, snapshot.state)
            assertFalse(snapshot.lockHeld)
            assertFalse(EngineEffect.EMIT_LOCKED_MEASUREMENT in effects)
        }
    }

    @Test
    @DisplayName("§4: cancelling a session returns to IDLE with nothing retained")
    fun cancellingASessionReturnsToIdle() {
        val (snapshot, effects) = MeasurementStateReducer.reduce(
            EngineSnapshot(
                state = MeasurementState.PRECISION_LOCKED,
                lockHeld = true,
                rejectionReasons = listOf(RejectionReason.DEVICE_MOVING),
                magneticState = MagneticState.CLEAN,
                hasLocation = true,
            ),
            EngineEvent(EngineEventType.SESSION_CANCELLED),
        )
        assertSame(MeasurementState.IDLE, snapshot.state)
        assertFalse(snapshot.lockHeld)
        assertTrue(snapshot.rejectionReasons.isEmpty())
        assertFalse(snapshot.hasLocation)
        assertTrue(EngineEffect.EMIT_NO_MEASUREMENT in effects)
    }

    @Test
    @DisplayName("§4: every event type is handled — none is silently ignored")
    fun everyEventTypeIsHandled() {
        // A reducer that silently ignores an event is how an explicit engine event turns back
        // into an ambient condition.
        val snapshot = EngineSnapshot(state = MeasurementState.STABILIZING, targetRequested = true)
        EngineEventType.entries.forEach { type ->
            val event = if (type == EngineEventType.MAGNETIC_STATE_OBSERVED) {
                EngineEvent(type, magneticState = MagneticState.CLEAN)
            } else {
                EngineEvent(type)
            }
            val (result, _) = MeasurementStateReducer.reduce(snapshot, event)
            assertTrue(result.state in MeasurementState.entries, type.name)
        }
    }
}
