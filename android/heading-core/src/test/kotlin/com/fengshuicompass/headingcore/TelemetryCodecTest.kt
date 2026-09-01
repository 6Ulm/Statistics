package com.fengshuicompass.headingcore

import com.fengshuicompass.headingcore.config.SharedArtifacts
import com.fengshuicompass.headingcore.telemetry.TelemetryCodec
import com.fengshuicompass.headingcore.telemetry.TelemetryCodecException
import com.fengshuicompass.headingcore.telemetry.TelemetryEnvelope
import kotlinx.serialization.json.Json
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.buildJsonObject
import kotlinx.serialization.json.double
import kotlinx.serialization.json.jsonArray
import kotlinx.serialization.json.jsonObject
import kotlinx.serialization.json.jsonPrimitive
import org.junit.jupiter.api.Assertions.assertEquals
import org.junit.jupiter.api.Assertions.assertTrue
import org.junit.jupiter.api.DisplayName
import org.junit.jupiter.api.Test
import org.junit.jupiter.api.assertThrows
import java.security.MessageDigest
import java.util.Locale

/**
 * SPEC.md §22 / §22.2 — the canonical telemetry codec.
 *
 * Failure mode 47 names the specific defects: Swift/Kotlin `NaN` divergence, a `Double`
 * serialized through a `Float`, locale decimal separators, and differing default estimators.
 * Every rule is checked in both directions, and the locale rule is checked by *actually
 * switching the default locale*, not by trusting that the serializer is locale-independent.
 */
class TelemetryCodecTest {

    private val json = Json { ignoreUnknownKeys = false }

    private val exampleEvent: JsonObject
        get() = json.parseToJsonElement(
            SharedArtifacts.exampleEngineOutputEventFile.readText()
        ).jsonObject

    private val codecFixture: JsonObject
        get() = json.parseToJsonElement(SharedArtifacts.telemetryCodecFixture.readText()).jsonObject

    private val envelope: TelemetryEnvelope get() = TelemetryEnvelope.fromDocument(exampleEvent)

    private val payload: JsonObject get() = exampleEvent["payload"]!!.jsonObject

    private fun sha256Of(file: java.io.File): String =
        "sha256:" + MessageDigest.getInstance("SHA-256")
            .digest(file.readBytes())
            .joinToString("") { "%02x".format(it) }

    @Test
    @DisplayName("§22: the shipped example round-trips stably")
    fun theShippedExampleRoundTrips() {
        val line = TelemetryCodec.encodeEvent(envelope, payload)
        val (decodedEnvelope, decodedPayload) = TelemetryCodec.decodeEvent(line)
        assertEquals(envelope, decodedEnvelope)
        assertEquals(payload, decodedPayload)
        // Re-encoding the decoded form is stable, so a line hash is reproducible (§29.7, §37.2).
        assertEquals(line, TelemetryCodec.encodeEvent(decodedEnvelope, decodedPayload))
    }

    @Test
    @DisplayName("§22.2: the encoder fails rather than emitting a nonfinite literal")
    fun nonfiniteValuesAreRefusedByTheEncoder() {
        listOf(Double.NaN, Double.POSITIVE_INFINITY, Double.NEGATIVE_INFINITY).forEach { value ->
            assertThrows<TelemetryCodecException> {
                TelemetryCodec.encodeEvent(
                    envelope,
                    buildJsonObject { put("trueHeadingDeg", JsonPrimitive(value)) },
                )
            }
        }
    }

    @Test
    @DisplayName("§22.2: the decoder rejects what the encoder would refuse to write")
    fun rejectedDocumentsAreRejectedByTheDecoder() {
        // A permissive decoder makes a strict encoder pointless, because the corrupt file
        // still enters the analysis.
        codecFixture["rejectedDocuments"]!!.jsonArray.forEach { entry ->
            val case = entry.jsonObject
            assertThrows<TelemetryCodecException>({ case["id"]!!.jsonPrimitive.content }) {
                TelemetryCodec.decodeEvent(case["line"]!!.jsonPrimitive.content)
            }
        }
    }

    @Test
    @DisplayName("§22.2: unavailable is null plus a sibling status field")
    fun unavailableIsNullPlusASiblingStatusField() {
        val line = TelemetryCodec.encodeEvent(
            envelope,
            buildJsonObject {
                put("providerErrorTermDeg", kotlinx.serialization.json.JsonNull)
                put("providerErrorSource", JsonPrimitive("NONE"))
                put("displayQualityGrade", kotlinx.serialization.json.JsonNull)
                put("boundCalibrationState", JsonPrimitive("CANDIDATE"))
            },
        )
        val (_, decoded) = TelemetryCodec.decodeEvent(line)
        assertTrue(decoded["providerErrorTermDeg"] is kotlinx.serialization.json.JsonNull)
        assertEquals("NONE", decoded["providerErrorSource"]!!.jsonPrimitive.content)
    }

    @Test
    @DisplayName("§22.2: property keys must be lowerCamelCase, at any nesting depth")
    fun propertyKeysMustBeLowerCamelCase() {
        listOf("schema_version", "SchemaVersion", "TRUE_HEADING", "_leading").forEach { key ->
            assertThrows<TelemetryCodecException>({ key }) {
                TelemetryCodec.encodeEvent(
                    envelope,
                    buildJsonObject { put(key, JsonPrimitive(1.0)) },
                )
            }
        }
        assertThrows<TelemetryCodecException> {
            TelemetryCodec.encodeEvent(
                envelope,
                buildJsonObject {
                    put(
                        "spaceWeather",
                        buildJsonObject { put("observation_time_utc", JsonPrimitive("x")) },
                    )
                },
            )
        }
    }

    @Test
    @DisplayName("§22: event types come from the fixed lower_snake_case namespace")
    fun eventTypesComeFromTheFixedNamespace() {
        assertTrue(exampleEvent["eventType"]!!.jsonPrimitive.content in TelemetryCodec.eventTypes)
        codecFixture["unknownEventTypes"]!!.jsonArray.forEach { entry ->
            val document = buildJsonObject {
                exampleEvent.forEach { (key, value) ->
                    put(key, if (key == "eventType") entry else value)
                }
            }
            assertThrows<TelemetryCodecException>({ entry.jsonPrimitive.content }) {
                TelemetryEnvelope.fromDocument(document)
            }
        }
    }

    @Test
    @DisplayName("§6/§22.2: enum values are UPPER_SNAKE_CASE")
    fun enumValuesAreUpperSnakeCase() {
        listOf(
            "providerId", "providerErrorSource", "resolvedReference", "referenceResolutionMethod",
            "magneticState", "measurementState", "trustAction", "gradeLimitedBy",
            "boundCalibrationState", "uncertaintyCoverageEvidenceState", "measurementMode",
            "placementMethod", "altitudeReference", "chargingState",
        ).forEach { field ->
            val value = payload[field]!!.jsonPrimitive.content
            assertEquals(value.uppercase(), value, field)
            assertTrue(!value.contains(" ") && !value.contains("-"), field)
        }
    }

    @Test
    @DisplayName("§22.2: the export path runs under a comma-decimal locale")
    fun theExportPathRunsUnderACommaDecimalLocale() {
        val original = Locale.getDefault()
        try {
            Locale.setDefault(Locale.GERMANY)
            assertEquals(
                ',',
                java.text.DecimalFormatSymbols.getInstance().decimalSeparator,
                "the test needs a comma-decimal default locale to be meaningful",
            )
            val line = TelemetryCodec.encodeEvent(envelope, payload)
            // The serialized numbers still use "." — a comma would either corrupt the value or
            // split the JSON object, and both parse differently on the other platform.
            assertTrue(line.contains("\"declinationDeg\":8.29"), line.take(200))
            assertTrue(line.contains("\"observedOrientationRateHz\":48.7"))
            assertTrue(!line.contains("\"declinationDeg\":8,29"))
            val (_, decoded) = TelemetryCodec.decodeEvent(line)
            assertEquals(8.29, decoded["declinationDeg"]!!.jsonPrimitive.double, 1e-12)
            assertEquals(48.7, decoded["observedOrientationRateHz"]!!.jsonPrimitive.double, 1e-12)
        } finally {
            Locale.setDefault(original)
        }
    }

    @Test
    @DisplayName("§22.2: doubles round-trip at full precision, never through a Float")
    fun doublesRoundTripAtFullPrecision() {
        // 359.9999998999999 is in the set on purpose: a float32 round trip collapses it to
        // 360.0, which the §9 normalization then maps to 0.0 — a full-circle error from a
        // serialization choice.
        val values = codecFixture["roundTripDoubles"]!!.jsonArray.map { it.jsonPrimitive.double }
        val line = TelemetryCodec.encodeEvent(
            envelope,
            buildJsonObject {
                values.forEachIndexed { index, value -> put("value${index}Deg", JsonPrimitive(value)) }
            },
        )
        val (_, decoded) = TelemetryCodec.decodeEvent(line)
        values.forEachIndexed { index, value ->
            assertEquals(value, decoded["value${index}Deg"]!!.jsonPrimitive.double, "$value")
        }
        val throughFloat32 = 359.9999998999999f.toDouble()
        assertTrue(throughFloat32 != 359.9999998999999)
    }

    @Test
    @DisplayName("§22.2: numeric field names carry their unit")
    fun numericFieldNamesCarryTheirUnit() {
        assertEquals(emptyList<String>(), TelemetryCodec.numericFieldsMissingUnitSuffix(exampleEvent))
    }

    @Test
    @DisplayName("§22.2: the units rule is not vacuous")
    fun theUnitsRuleIsNotVacuous() {
        val document = buildJsonObject {
            exampleEvent.forEach { (key, value) ->
                if (key == "payload") {
                    put(
                        key,
                        buildJsonObject {
                            payload.forEach { (payloadKey, payloadValue) ->
                                put(payloadKey, payloadValue)
                            }
                            put("magneticDeclination", JsonPrimitive(8.29))
                        },
                    )
                } else {
                    put(key, value)
                }
            }
        }
        assertTrue(
            TelemetryCodec.numericFieldsMissingUnitSuffix(document)
                .contains("payload.magneticDeclination")
        )
    }

    @Test
    @DisplayName("§22: the three monotonic timestamps stay distinct")
    fun theThreeMonotonicTimestampsStayDistinct() {
        assertTrue(envelope.sourceMonotonicTimeNs < envelope.arrivalMonotonicTimeNs)
        assertTrue(envelope.arrivalMonotonicTimeNs < envelope.recordMonotonicTimeNs)
    }

    @Test
    @DisplayName("§22.2: wall clock is RFC 3339 UTC with an explicit Z")
    fun wallClockIsRfc3339Utc() {
        assertTrue(exampleEvent["wallTimeUtc"]!!.jsonPrimitive.content.endsWith("Z"))
        listOf(
            "2026-08-29T12:34:56.123456+02:00",
            "2026-08-29 12:34:56Z",
            "2026-08-29T12:34:56",
        ).forEach { bad ->
            val document = buildJsonObject {
                exampleEvent.forEach { (key, value) ->
                    put(key, if (key == "wallTimeUtc") JsonPrimitive(bad) else value)
                }
            }
            assertThrows<TelemetryCodecException>({ bad }) {
                TelemetryEnvelope.fromDocument(document)
            }
        }
    }

    @Test
    @DisplayName("failure mode 10: an unidentified source clock is rejected")
    fun anUnidentifiedSourceClockIsRejected() {
        val document = buildJsonObject {
            exampleEvent.forEach { (key, value) ->
                put(key, if (key == "sourceClock") JsonPrimitive("SYSTEM_CLOCK") else value)
            }
        }
        assertThrows<TelemetryCodecException> { TelemetryEnvelope.fromDocument(document) }
    }

    @Test
    @DisplayName("§37.2/failure mode 42: the envelope carries config and ruleset provenance")
    fun theEnvelopeCarriesProvenance() {
        assertEquals("precision-v1-candidate-1", exampleEvent["configVersion"]!!.jsonPrimitive.content)
        assertEquals(
            sha256Of(SharedArtifacts.precisionProfileFile),
            exampleEvent["configHash"]!!.jsonPrimitive.content,
        )
        assertEquals("fengshui-v1", payload["fengShuiRuleSetVersion"]!!.jsonPrimitive.content)
        assertEquals(
            sha256Of(SharedArtifacts.fengShuiRuleSetFile),
            payload["fengShuiRuleSetHash"]!!.jsonPrimitive.content,
        )
    }

    @Test
    @DisplayName("D-2: the unvendored WMM hashes are recorded as NOT_VENDORED")
    fun theUnvendoredWmmHashesAreRecordedAsNotVendored() {
        assertEquals("NOT_VENDORED", payload["declinationCoefficientSha256"]!!.jsonPrimitive.content)
        assertEquals("NOT_VENDORED", payload["declinationErrorModelSha256"]!!.jsonPrimitive.content)
    }
}
