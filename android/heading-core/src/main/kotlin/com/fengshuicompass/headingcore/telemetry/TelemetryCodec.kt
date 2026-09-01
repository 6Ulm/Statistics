package com.fengshuicompass.headingcore.telemetry

import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.longOrNull

/**
 * SPEC.md §22 / §22.2 — the canonical telemetry codec.
 *
 * Cross-platform JSON differences are a recurring source of silent corruption, and exports
 * that parse differently on two platforms cannot be pooled (failure mode 47). Every rule
 * §22.2 states is enforced in **both** directions, because a decoder that quietly accepts
 * `NaN` makes a strict encoder pointless.
 *
 * The three canonical monotonic timestamps have three meanings and are never interchangeable.
 * **Freshness is always computed from mapped source time, never arrival** (failure mode 11),
 * so [TelemetryEnvelope] exposes the source/arrival distinction rather than one "timestamp".
 */
public object TelemetryCodec {

    /** §22 event types, `lower_snake_case`, a separate namespace from enum values. */
    public val eventTypes: Set<String> = setOf(
        "session_start", "session_end", "app_lifecycle",
        "clock_mapping", "ground_truth", "fixture_state",
        "location_sample", "location_authorization", "location_provider_state",
        "magnetometer_calibrated", "magnetometer_uncalibrated",
        "accelerometer", "gravity", "gyroscope", "rotation_vector", "device_motion",
        "os_heading", "fused_orientation",
        "capability_resolution",
        "wmm_output", "reference_resolution", "engine_output", "state_transition",
        "precision_lock",
        "sensor_health",
        "calibration_request", "calibration_prompt", "calibration_result",
        "target_heading_request", "target_guidance",
        "deviation_profile_lookup", "deviation_correction",
        "certification_lookup",
        "display_frame_marker",
        "thermal_state", "battery_state", "charging_state", "power_mode",
        "space_weather_advisory",
        "orientation_change", "sensor_discontinuity", "dropped_sample_summary",
        "user_action",
    )

    /** §22 `sourceClock` domain identifiers. */
    public val sourceClocks: Set<String> =
        setOf("ELAPSED_REALTIME", "CORE_MOTION_BOOT_TIME", "PROVIDER_DATE", "FIXTURE_CLOCK")

    /**
     * §22.2 unit suffixes. A numeric field name ends with one of these unless it is documented
     * dimensionless below.
     */
    public val unitSuffixes: List<String> =
        listOf("Deg", "Ms", "Ns", "Us", "MicroTesla", "NanoTesla", "Hz", "Km", "M", "G")

    /**
     * The documented dimensionless numeric fields. §22.2 permits a numeric name without a unit
     * suffix only when it is "dimensionless and documented as such"; this set *is* that
     * documentation, so adding a field here is a deliberate, reviewable act.
     */
    public val documentedDimensionlessFields: Set<String> = setOf(
        "eventId",
        "sequence",
        "kp",
        "uncertaintyCoverageTarget",
        "circularResultantLength",
        "relativeMagnitudeResidual",
        "effectiveHeadingSampleCount",
        "periodicSupportSampleCount",
        "sectorCount",
        "repetitions",
        "truthCoverageFactor",
    )

    private val lowerCamelCase = Regex("^[a-z][A-Za-z0-9]*$")
    private val lowerSnakeCase = Regex("^[a-z][a-z0-9_]*$")

    /** RFC 3339 UTC with an explicit `Z`; no offset form, no local time, no space separator. */
    private val rfc3339Utc = Regex("""^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$""")

    private val strictJson = Json {
        allowSpecialFloatingPointValues = false
        isLenient = false
        prettyPrint = false
        encodeDefaults = true
    }

    /** Every §22.2 structural rule, applied to a whole event document. */
    public fun assertEncodingRules(document: JsonObject) {
        walk(document, "") { key, path ->
            if (!lowerCamelCase.matches(key)) {
                throw TelemetryCodecException(
                    "property key $key at ${path.ifEmpty { "<root>" }} is not lowerCamelCase (§22.2)"
                )
            }
        }
    }

    /**
     * §22.2's units-in-names rule, made executable.
     *
     * Returns the dotted paths of numeric fields whose names carry neither a unit suffix nor a
     * place in [documentedDimensionlessFields].
     */
    public fun numericFieldsMissingUnitSuffix(document: JsonObject): List<String> {
        val offenders = mutableListOf<String>()
        walkNumbers(document, "") { path ->
            val name = path.substringAfterLast('.').substringBefore('[')
            if (name !in documentedDimensionlessFields &&
                unitSuffixes.none { name.endsWith(it) }
            ) {
                offenders += path
            }
        }
        return offenders
    }

    /**
     * Encode one JSONL line.
     *
     * `allowSpecialFloatingPointValues = false` makes the encoder **fail** rather than emit
     * `NaN`/`Infinity`. Kotlin's `Double.toString` is the shortest round-trip form §22.2
     * requires, and the serializer never applies locale formatting — the locale test proves
     * that rather than assuming it.
     */
    public fun encodeEvent(envelope: TelemetryEnvelope, payload: JsonObject): String {
        assertNoNonFinite(payload, "payload")
        val document = buildJsonObject {
            envelope.toDocument().forEach { (key, value) -> put(key, value) }
            put("payload", payload)
        }
        assertEncodingRules(document)
        return strictJson.encodeToString(JsonObject.serializer(), document)
    }

    /** Decode one JSONL line, rejecting anything the encoder would have refused to write. */
    public fun decodeEvent(line: String): Pair<TelemetryEnvelope, JsonObject> {
        // kotlinx.serialization already refuses NaN/Infinity literals with
        // allowSpecialFloatingPointValues = false, but the message is generic; the wrapper
        // keeps the failure attributable to §22.2 in either direction.
        val element = try {
            strictJson.parseToJsonElement(line)
        } catch (failure: IllegalArgumentException) {
            throw TelemetryCodecException(
                "decoder rejected the document; §22.2 forbids NaN/Infinity and malformed JSON " +
                    "in either direction: ${failure.message}"
            )
        }
        if (element !is JsonObject) {
            throw TelemetryCodecException("a telemetry event must be a JSON object")
        }
        assertNonFiniteTokensAbsent(line)
        assertEncodingRules(element)
        val payload = element["payload"]
        if (payload == null || payload !is JsonObject) {
            throw TelemetryCodecException("every event carries a typed object payload (§22)")
        }
        return TelemetryEnvelope.fromDocument(element) to payload
    }

    private fun assertNonFiniteTokensAbsent(line: String) {
        listOf("NaN", "Infinity", "-Infinity").forEach { token ->
            if (Regex("""(^|[\s:,\[])-?$token($|[\s,}\]])""").containsMatchIn(line)) {
                throw TelemetryCodecException(
                    "decoder rejected the nonstandard JSON literal $token; §22.2 forbids " +
                        "NaN/Infinity in either direction"
                )
            }
        }
    }

    private fun assertNoNonFinite(element: JsonElement, path: String) {
        walkNumbers(element, path) { }
        walkPrimitives(element, path) { primitive, at ->
            val value = primitive.doubleOrNull
            if (value != null && !value.isFinite()) {
                throw TelemetryCodecException(
                    "nonfinite number at $at: JSON has no NaN/Infinity. An unavailable value " +
                        "is null plus a sibling status field, never a nonstandard literal (§22.2)."
                )
            }
        }
    }

    private fun walk(element: JsonElement, path: String, visitKey: (String, String) -> Unit) {
        when (element) {
            is JsonObject -> element.forEach { (key, value) ->
                visitKey(key, path)
                walk(value, if (path.isEmpty()) key else "$path.$key", visitKey)
            }
            is JsonArray -> element.forEachIndexed { index, value ->
                walk(value, "$path[$index]", visitKey)
            }
            else -> Unit
        }
    }

    private fun walkPrimitives(
        element: JsonElement,
        path: String,
        visit: (JsonPrimitive, String) -> Unit,
    ) {
        when (element) {
            is JsonObject -> element.forEach { (key, value) ->
                walkPrimitives(value, if (path.isEmpty()) key else "$path.$key", visit)
            }
            is JsonArray -> element.forEachIndexed { index, value ->
                walkPrimitives(value, "$path[$index]", visit)
            }
            is JsonPrimitive -> visit(element, path)
        }
    }

    private fun walkNumbers(element: JsonElement, path: String, visit: (String) -> Unit) {
        walkPrimitives(element, path) { primitive, at ->
            val isNumber = !primitive.isString &&
                primitive.booleanOrNull == null &&
                primitive.content != "null" &&
                primitive.doubleOrNull != null
            if (isNumber) visit(at)
        }
    }
}

/** §22's common envelope. Typed so a timestamp cannot land in the wrong field. */
public data class TelemetryEnvelope(
    val schemaVersion: String,
    val sessionId: String,
    val eventId: Long,
    val eventType: String,
    val platform: String,
    val appVersion: String,
    val appBuild: String,
    val engineVersion: String,
    val configVersion: String,
    val configHash: String,
    val deviceAnonymousId: String,
    val hardwareRuntimeIdentity: String,
    val sensorRuntimeIdentity: String,
    val osBuildIdentity: String,
    val wallTimeUtc: String,
    val recordMonotonicTimeNs: Long,
    val sourceMonotonicTimeNs: Long,
    val arrivalMonotonicTimeNs: Long,
    val sourceClock: String,
    val clockMappingId: String,
    val sequence: Long,
) {
    init {
        if (eventType !in TelemetryCodec.eventTypes) {
            throw TelemetryCodecException(
                "unknown eventType $eventType; §22 fixes the event-type namespace"
            )
        }
        if (sourceClock !in TelemetryCodec.sourceClocks) {
            throw TelemetryCodecException(
                "unknown sourceClock $sourceClock; freshness cannot be computed from a " +
                    "timestamp whose clock domain is unidentified (failure mode 10)"
            )
        }
        if (!Regex("""^\d{4}-\d{2}-\d{2}T\d{2}:\d{2}:\d{2}(\.\d+)?Z$""").matches(wallTimeUtc)) {
            throw TelemetryCodecException(
                "wallTimeUtc $wallTimeUtc is not RFC 3339 UTC with an explicit Z"
            )
        }
    }

    public fun toDocument(): Map<String, JsonElement> = linkedMapOf(
        "schemaVersion" to JsonPrimitive(schemaVersion),
        "sessionId" to JsonPrimitive(sessionId),
        "eventId" to JsonPrimitive(eventId),
        "eventType" to JsonPrimitive(eventType),
        "platform" to JsonPrimitive(platform),
        "appVersion" to JsonPrimitive(appVersion),
        "appBuild" to JsonPrimitive(appBuild),
        "engineVersion" to JsonPrimitive(engineVersion),
        "configVersion" to JsonPrimitive(configVersion),
        "configHash" to JsonPrimitive(configHash),
        "deviceAnonymousId" to JsonPrimitive(deviceAnonymousId),
        "hardwareRuntimeIdentity" to JsonPrimitive(hardwareRuntimeIdentity),
        "sensorRuntimeIdentity" to JsonPrimitive(sensorRuntimeIdentity),
        "osBuildIdentity" to JsonPrimitive(osBuildIdentity),
        "wallTimeUtc" to JsonPrimitive(wallTimeUtc),
        "recordMonotonicTimeNs" to JsonPrimitive(recordMonotonicTimeNs),
        "sourceMonotonicTimeNs" to JsonPrimitive(sourceMonotonicTimeNs),
        "arrivalMonotonicTimeNs" to JsonPrimitive(arrivalMonotonicTimeNs),
        "sourceClock" to JsonPrimitive(sourceClock),
        "clockMappingId" to JsonPrimitive(clockMappingId),
        "sequence" to JsonPrimitive(sequence),
    )

    public companion object {
        public fun fromDocument(document: JsonObject): TelemetryEnvelope {
            fun text(key: String): String =
                (document[key] as? JsonPrimitive)?.content
                    ?: throw TelemetryCodecException("missing envelope field $key")

            fun integer(key: String): Long {
                val primitive = document[key] as? JsonPrimitive
                    ?: throw TelemetryCodecException("missing envelope field $key")
                return primitive.longOrNull ?: throw TelemetryCodecException(
                    "$key must be an integer; §22.2 keeps monotonic time in integer " +
                        "nanoseconds and never mixes it with a wall-clock string"
                )
            }

            return TelemetryEnvelope(
                schemaVersion = text("schemaVersion"),
                sessionId = text("sessionId"),
                eventId = integer("eventId"),
                eventType = text("eventType"),
                platform = text("platform"),
                appVersion = text("appVersion"),
                appBuild = text("appBuild"),
                engineVersion = text("engineVersion"),
                configVersion = text("configVersion"),
                configHash = text("configHash"),
                deviceAnonymousId = text("deviceAnonymousId"),
                hardwareRuntimeIdentity = text("hardwareRuntimeIdentity"),
                sensorRuntimeIdentity = text("sensorRuntimeIdentity"),
                osBuildIdentity = text("osBuildIdentity"),
                wallTimeUtc = text("wallTimeUtc"),
                recordMonotonicTimeNs = integer("recordMonotonicTimeNs"),
                sourceMonotonicTimeNs = integer("sourceMonotonicTimeNs"),
                arrivalMonotonicTimeNs = integer("arrivalMonotonicTimeNs"),
                sourceClock = text("sourceClock"),
                clockMappingId = text("clockMappingId"),
                sequence = integer("sequence"),
            )
        }

        /** Convenience for reading a whole event document, envelope plus payload. */
        public fun payloadOf(document: JsonObject): JsonObject =
            document["payload"]?.jsonObject
                ?: throw TelemetryCodecException("every event carries a typed object payload")
    }
}

/** A document that violates a §22.2 encoding rule, in either direction. */
public class TelemetryCodecException(message: String) : IllegalArgumentException(message)
