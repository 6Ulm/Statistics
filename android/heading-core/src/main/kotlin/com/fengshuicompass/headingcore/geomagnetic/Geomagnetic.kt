package com.fengshuicompass.headingcore.geomagnetic

import com.fengshuicompass.headingcore.math.CircularMath
import com.fengshuicompass.headingcore.model.AltitudeReference
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import java.io.File
import java.time.Instant
import java.time.ZoneOffset
import java.time.ZonedDateTime

/**
 * SPEC.md §10, §10.2, §10.3, §19.2 — geomagnetic model contract and uncertainty.
 *
 * What this file **is**: the typed contract around the vendored NOAA model — altitude datum
 * handling, decimal-year conversion, validity enforcement, the [GeomagneticModelUncertainty]
 * shape, and `boundFromSigma` as the single conversion from one sigma to a 95% bound.
 *
 * What this file is **not**: a geomagnetic model. §10 requires the *same official NOAA C
 * sources* compiled on both platforms and forbids separate Swift and Kotlin ports of the
 * spherical-harmonic core; §10.3 states that an implementation which "derives a sigma from the
 * coefficients, or substitutes a remembered global constant, has invented the quantity". The
 * NOAA artifacts are not vendored (`docs/IMPLEMENTATION_NOTES.md` D-2), so every path that
 * would need a coefficient, an error-model formula or a hash throws
 * [VendoredModelUnavailableException] naming the missing artifact. A refusal is the honest
 * state; a plausible number is not.
 */
public object Geomagnetic {

    /**
     * §10: "WMM2025's v1 epoch interval is `2025.0 <= decimalYear < 2030.0`".
     *
     * This interval is stated in SPEC.md itself, so it is a specification constant rather than
     * a value read out of an unvendored artifact. The **coefficients** it applies to are still
     * absent (D-2), so nothing here can evaluate a field.
     */
    public val wmm2025Validity: ValidityInterval = ValidityInterval(2025.0, 2030.0)

    /**
     * Return WGS84 ellipsoidal height, or refuse (§10.2).
     *
     * Canonical model input is ellipsoidal height. An `MSL_ORTHOMETRIC` sample is accepted
     * only with an explicitly supplied, documented geoid separation; `UNKNOWN` is never
     * coerced, because `UNKNOWN` "is not a synonym for either datum".
     */
    public fun ellipsoidalAltitudeM(
        altitude: AltitudeSample,
        geoidSeparationM: Double? = null,
    ): Double = when (altitude.reference) {
        AltitudeReference.WGS84_ELLIPSOID -> altitude.valueM
        AltitudeReference.MSL_ORTHOMETRIC -> {
            if (geoidSeparationM == null) {
                throw AltitudeDatumUnconvertedException(
                    "MSL_ORTHOMETRIC altitude requires a documented geoid separation before " +
                        "it can enter the model; §10.2 forbids treating it as ellipsoidal"
                )
            }
            require(geoidSeparationM.isFinite()) { "geoidSeparationM must be finite" }
            altitude.valueM + geoidSeparationM
        }
        AltitudeReference.UNKNOWN -> throw AltitudeDatumUnconvertedException(
            "altitude reference is UNKNOWN; it downgrades quality and is not a datum (§10.2)"
        )
    }

    /**
     * §9/§10 `wmmDecimalYear` — UTC instant to decimal year, leap years included.
     *
     * `year + elapsed / length_of_that_year`, where the year length is measured from the two
     * UTC year boundaries so 2028 gets 366 days without a leap-year branch to get wrong.
     */
    public fun wmmDecimalYear(instant: Instant): Double {
        val utc = instant.atZone(ZoneOffset.UTC)
        val start = ZonedDateTime.of(utc.year, 1, 1, 0, 0, 0, 0, ZoneOffset.UTC)
        val end = ZonedDateTime.of(utc.year + 1, 1, 1, 0, 0, 0, 0, ZoneOffset.UTC)
        val elapsed = utc.toInstant().toEpochMilli() - start.toInstant().toEpochMilli()
        val length = end.toInstant().toEpochMilli() - start.toInstant().toEpochMilli()
        return utc.year + elapsed.toDouble() / length.toDouble()
    }

    public fun isWithinValidity(decimalYear: Double, validity: ValidityInterval): Boolean =
        validity.contains(decimalYear)

    public fun requireWithinValidity(decimalYear: Double, validity: ValidityInterval): Double {
        if (!validity.contains(decimalYear)) {
            throw GeomagneticDateOutOfRangeException(
                "decimal year $decimalYear is outside " +
                    "[${validity.startDecimalYear}, ${validity.endDecimalYear}); emit " +
                    "GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE and prompt to update the app"
            )
        }
        return decimalYear
    }

    /**
     * §19.2: the **only** path from a published sigma to this project's 95% bound.
     *
     * Applied exactly once, and recorded as an assumption (§35). The factor is versioned
     * configuration, passed in rather than written here.
     */
    public fun declinationBound95Deg(
        uncertainty: GeomagneticModelUncertainty,
        sigmaToBound95Factor: Double,
    ): Double = CircularMath.boundFromSigma(
        uncertainty.declinationSigma1Deg,
        sigmaToBound95Factor,
    )

    /**
     * Read the vendored state for [modelId] from `third_party/noaa-wmm/sha256.txt`.
     *
     * Absence is reported, never guessed. The manifest is the single source of truth for what
     * has actually been vendored; `scripts/verify-artifacts.sh` independently fails the build
     * if the files and the manifest ever stop agreeing.
     */
    public fun vendoredArtifacts(modelId: GeomagneticModelId, repoRoot: File): VendoredWmmArtifacts {
        val manifest = File(File(repoRoot, "third_party/noaa-wmm"), "sha256.txt")
        val entries = mutableMapOf<String, String>()
        if (manifest.isFile) {
            manifest.readLines().forEach { line ->
                val stripped = line.trim()
                if (stripped.isNotEmpty() && !stripped.startsWith("#")) {
                    val parts = stripped.split("  ", limit = 2)
                    if (parts.size == 2) entries[parts[1].trim()] = parts[0]
                }
            }
        }
        val token = modelId.wire.lowercase()
        fun find(directory: String): String? = entries.entries
            .firstOrNull { it.key.startsWith("$directory/") && it.key.lowercase().contains(token) }
            ?.let { "sha256:${it.value}" }

        return VendoredWmmArtifacts(
            modelId = modelId,
            coefficientHash = find("coefficients"),
            errorModelHash = find("error-model"),
            sourceHash = entries.entries
                .firstOrNull { it.key.startsWith("src/") }
                ?.let { "sha256:${it.value}" },
        )
    }
}

/** `[start, end)` in decimal years, per model (§10). */
public data class ValidityInterval(val startDecimalYear: Double, val endDecimalYear: Double) {
    public fun contains(decimalYear: Double): Boolean =
        decimalYear >= startDecimalYear && decimalYear < endDecimalYear
}

/** §5 `AltitudeSample`. The datum is always explicit; `UNKNOWN` is a real state. */
public data class AltitudeSample(
    val valueM: Double,
    val reference: AltitudeReference,
    val verticalAccuracyM: Double? = null,
) {
    init {
        require(valueM.isFinite()) { "AltitudeSample.valueM must be finite, got $valueM" }
        require(verticalAccuracyM == null || verticalAccuracyM.isFinite()) {
            "AltitudeSample.verticalAccuracyM must be finite when present"
        }
    }
}

/**
 * The confidence semantics of an error quantity.
 *
 * Failure mode 9 — confidence-level conflation — is Critical: adding a one-sigma number to a
 * sum of nominal 95% terms under-covers by roughly a factor of two. Every error term therefore
 * carries its level explicitly and conversion happens in exactly one function.
 */
public enum class ConfidenceLevel {
    ONE_STANDARD_DEVIATION,
    TWO_SIDED_95,
    SIXTY_EIGHT_PERCENT,
}

/**
 * §7/§10.3 `GeomagneticModelUncertainty`.
 *
 * [declinationSigma1Deg] comes from NOAA's separately published error model for that exact
 * coefficient set — never from evaluating the coefficients, never from a remembered global
 * constant. [errorModelHash] is therefore mandatory: §24 makes it a separate component of the
 * certification key because the error model changes reported uncertainty even when coefficient
 * evaluation is unchanged.
 */
public data class GeomagneticModelUncertainty(
    val declinationSigma1Deg: Double,
    val sourceModelId: GeomagneticModelId,
    val errorModelId: String,
    val errorModelHash: String,
    val sourceDocumentReference: String,
    val sourceConfidenceLevel: ConfidenceLevel = ConfidenceLevel.ONE_STANDARD_DEVIATION,
) {
    init {
        require(sourceConfidenceLevel == ConfidenceLevel.ONE_STANDARD_DEVIATION) {
            "§10.3 fixes sourceConfidenceLevel to ONE_STANDARD_DEVIATION for the NOAA " +
                "declination error model; a different level needs its own conversion, not a " +
                "relabelled field"
        }
        require(declinationSigma1Deg.isFinite() && declinationSigma1Deg >= 0.0) {
            "declinationSigma1Deg must be a finite, non-negative number of degrees"
        }
        require(errorModelHash.isNotBlank() && errorModelHash != "NONE") {
            "a declination sigma without an error-model hash is an invented quantity " +
                "(§10.3, §24)"
        }
    }
}

/**
 * The vendored artifact set for one model, or the declared absence of it.
 *
 * §10 requires the C source, the coefficient file, the separately published error model, the
 * licence, the retrieval URL/date and a SHA-256 per file, all under `third_party/noaa-wmm/`.
 * Phase 1 could not fetch them; this type makes the absence a typed, testable state rather
 * than a comment.
 */
public data class VendoredWmmArtifacts(
    val modelId: GeomagneticModelId,
    val coefficientHash: String?,
    val errorModelHash: String?,
    val sourceHash: String?,
) {
    val isVendored: Boolean
        get() = coefficientHash != null && errorModelHash != null && sourceHash != null

    public fun requireVendored(operation: String): VendoredWmmArtifacts {
        if (!isVendored) {
            throw VendoredModelUnavailableException(
                "$operation needs the vendored NOAA ${modelId.wire} artifacts (C source, " +
                    "coefficients, error model, hashes) under third_party/noaa-wmm/. They are " +
                    "NOT_VENDORED — see third_party/noaa-wmm/UPSTREAM.md and " +
                    "docs/IMPLEMENTATION_NOTES.md D-2. §10.3 forbids deriving a sigma from the " +
                    "coefficients or substituting a remembered constant, and §2.3 forbids a " +
                    "home-grown replacement for the official reference implementation."
            )
        }
        return this
    }

    public companion object {
        public const val NOT_VENDORED: String = "NOT_VENDORED"
    }
}

/**
 * The vendored NOAA artifact this operation needs does not exist. Thrown instead of returning
 * a remembered constant.
 */
public class VendoredModelUnavailableException(message: String) : IllegalStateException(message)

/**
 * §10: `GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE`. An app installed near epoch end **will**
 * outlive its coefficients on some devices; the model refuses and the app surfaces "model
 * expired, update the app" rather than extrapolating.
 */
public class GeomagneticDateOutOfRangeException(message: String) : IllegalArgumentException(message)

/**
 * §10.2: an orthometric or unknown altitude reached the model without conversion. Geoid
 * separation exceeds 100 m in places, so this is a silent systematic cross-platform
 * divergence — fixed by typing, not by measuring its effect.
 */
public class AltitudeDatumUnconvertedException(message: String) : IllegalArgumentException(message)
