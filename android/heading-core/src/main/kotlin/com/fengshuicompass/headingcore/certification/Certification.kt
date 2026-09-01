package com.fengshuicompass.headingcore.certification

import com.fengshuicompass.headingcore.grade.QualityGrade
import com.fengshuicompass.headingcore.model.BoundCalibrationState
import com.fengshuicompass.headingcore.model.GeomagneticModelId
import com.fengshuicompass.headingcore.model.LocationProviderId
import com.fengshuicompass.headingcore.model.MeasurementMode
import com.fengshuicompass.headingcore.model.PlacementMethod
import com.fengshuicompass.headingcore.model.ProviderErrorSource
import com.fengshuicompass.headingcore.model.ProviderId
import com.fengshuicompass.headingcore.model.UncertaintyCoverageEvidenceState

/**
 * SPEC.md §24 — the certification key, the database, and `miss -> CANDIDATE`.
 *
 * §24 is *authoritative* for this schema: two platform agents inventing two lookup schemas is
 * a realistic failure mode, so every lookup elsewhere refers here rather than restating the
 * key.
 *
 * Three properties are structural rather than procedural:
 *
 * * **A record exists only for `CALIBRATED`.** Absence already means `CANDIDATE`; storing both
 *   invites writing a `CANDIDATE` record and editing its state field.
 * * **Every key field is derivable in the production process** from a public runtime value or
 *   an app-bundled artifact hash. Lab-only facts belong in the evidence inventory, never in
 *   the runtime key (R66). A genuinely unobservable value uses the explicit
 *   [NOT_RUNTIME_OBSERVABLE] sentinel with pooled worst-case evidence.
 * * **Lookup is exact on every component.** `osBuildIdentity`, `providerRuntimeIdentity` and
 *   `locationProviderRuntimeIdentity` are exact observed identities, never semantic or
 *   open-ended ranges that silently admit a future release.
 *
 * §37 rule 12: an agent MUST NOT add records to make tests pass. The shipped database is
 * empty, so every lookup misses and every measurement is `CANDIDATE` with
 * `unknownDeviceFloor95Deg` — which, with the shipped constants, means no freehand grade is
 * arithmetically reachable at all (§8.1.1).
 */
public object Certification {

    /** §24: "prevents a newer client from reinterpreting an old tuple". */
    public const val CERTIFICATION_SCHEMA_VERSION: String = "certification-v1"

    /**
     * §24: the explicit sentinel for a component the runtime genuinely cannot observe. It is a
     * value, not a missing field, so evidence gathered under it is pooled worst-case evidence
     * rather than an invented key field (R66).
     */
    public const val NOT_RUNTIME_OBSERVABLE: String = "NOT_RUNTIME_OBSERVABLE"

    /**
     * §24: for OS-owned providers without a separate public version, the provider runtime
     * identity is derived from the OS build rather than left `UNKNOWN` or filled with a
     * marketing version.
     */
    public const val OS_BUILD_PROVIDER_IDENTITY_PREFIX: String = "OS_BUILD:"

    /**
     * Substrings that betray a semantic or open-ended version range. §24 requires exact
     * observed identities; a range "silently admits a future release" that was never measured.
     */
    internal val openEndedMarkers: List<String> =
        listOf("+", "*", "..", ">=", "<=", "latest", "any", "unknown")

    internal fun requireExact(name: String, value: String): String {
        if (value.isBlank()) {
            throw CertificationKeyException(
                "$name is required and must be a non-empty exact identity"
            )
        }
        if (value == NOT_RUNTIME_OBSERVABLE) return value
        val lowered = value.lowercase()
        openEndedMarkers.firstOrNull { lowered.contains(it) }?.let {
            throw CertificationKeyException(
                "$name=$value looks like an open-ended or semantic range. §24 requires an " +
                    "exact observed identity; evidence covering several exact builds generates " +
                    "several records pointing to the same report (R66)."
            )
        }
        return value
    }

    /**
     * §19.1's two invariants, asserted on every emitted result.
     *
     * `CALIBRATED  <=> uncertaintyCoverageEvidenceState == EMPIRICALLY_CALIBRATED`
     * `CANDIDATE    => uncertaintyCoverageEvidenceState in {TARGET_ONLY, UNDEFINED}`
     *
     * The two fields are near-redundant by design — one is the gate, the other the claim — and
     * the redundancy is safe only while the invariant holds, because drift lets a `95%` label
     * appear on a `CANDIDATE` measurement (failure mode 31).
     */
    public fun assertCalibrationInvariants(
        boundCalibrationState: BoundCalibrationState,
        coverageEvidenceState: UncertaintyCoverageEvidenceState,
    ) {
        val calibrated = boundCalibrationState == BoundCalibrationState.CALIBRATED
        val empirical =
            coverageEvidenceState == UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED
        require(calibrated == empirical) {
            "§19.1 invariant violated: boundCalibrationState=${boundCalibrationState.wire} " +
                "with uncertaintyCoverageEvidenceState=${coverageEvidenceState.wire}"
        }
        require(
            calibrated ||
                coverageEvidenceState in setOf(
                    UncertaintyCoverageEvidenceState.TARGET_ONLY,
                    UncertaintyCoverageEvidenceState.UNDEFINED,
                )
        ) {
            "§19.1 invariant violated: CANDIDATE requires TARGET_ONLY or UNDEFINED coverage " +
                "evidence, got ${coverageEvidenceState.wire}"
        }
    }
}

/** §24 `CertificationKey` — the exact tuple a measurement context must match. */
public data class CertificationKey(
    val certificationSchemaVersion: String,
    val hardwareRuntimeIdentity: String,
    val sensorRuntimeIdentity: String,
    val osBuildIdentity: String,
    val providerId: ProviderId,
    val providerRuntimeIdentity: String,
    val providerErrorSource: ProviderErrorSource,
    val locationProviderId: LocationProviderId,
    val locationProviderRuntimeIdentity: String,
    val measurementMode: MeasurementMode,
    val placementMethod: PlacementMethod,
    val placementProfileHash: String,
    val geomagneticModelId: GeomagneticModelId,
    val geomagneticCoefficientHash: String,
    val geomagneticErrorModelHash: String,
    val deviationCorrectionProfileHash: String,
    val engineDecisionLogicHash: String,
    val precisionConfigHash: String,
) {
    init {
        mapOf(
            "certificationSchemaVersion" to certificationSchemaVersion,
            "hardwareRuntimeIdentity" to hardwareRuntimeIdentity,
            "sensorRuntimeIdentity" to sensorRuntimeIdentity,
            "osBuildIdentity" to osBuildIdentity,
            "providerRuntimeIdentity" to providerRuntimeIdentity,
            "locationProviderRuntimeIdentity" to locationProviderRuntimeIdentity,
            "placementProfileHash" to placementProfileHash,
            "geomagneticCoefficientHash" to geomagneticCoefficientHash,
            "geomagneticErrorModelHash" to geomagneticErrorModelHash,
            "deviationCorrectionProfileHash" to deviationCorrectionProfileHash,
            "engineDecisionLogicHash" to engineDecisionLogicHash,
            "precisionConfigHash" to precisionConfigHash,
        ).forEach { (name, value) -> Certification.requireExact(name, value) }
    }
}

/** §24 `CertificationRecord`. Exists **only** for `CALIBRATED`. */
public data class CertificationRecord(
    val key: CertificationKey,
    val deviceFloor95Deg: Double,
    val supportedQualityGrade: QualityGrade,
    val earnedUnderEngineVersion: String,
    val evidenceReportId: String,
    val certificationDate: String,
    val boundCalibrationState: BoundCalibrationState = BoundCalibrationState.CALIBRATED,
) {
    init {
        require(boundCalibrationState == BoundCalibrationState.CALIBRATED) {
            "§24: a record exists only for CALIBRATED. Absence already means CANDIDATE; " +
                "storing both invites writing a CANDIDATE record and editing its state field."
        }
        require(deviceFloor95Deg.isFinite() && deviceFloor95Deg > 0.0) {
            "deviceFloor95Deg must be a finite, positive bound"
        }
        require(evidenceReportId.isNotBlank()) {
            "evidenceReportId MUST resolve to archived raw telemetry; an empty one makes the " +
                "record unauditable (§24)"
        }
    }
}

/**
 * What the engine consumes: a state, a floor, and a ceiling.
 *
 * §19.1: `boundCalibrationState` is derived at runtime from this lookup, never read from
 * config. There is no invalidation step to forget — changing model, config, provider path,
 * mode or placement changes the key and therefore misses.
 */
public data class CertificationLookupOutcome(
    val boundCalibrationState: BoundCalibrationState,
    val uncertaintyCoverageEvidenceState: UncertaintyCoverageEvidenceState,
    val deviceFloor95Deg: Double,
    val supportedQualityGrade: QualityGrade,
    val record: CertificationRecord?,
) {
    init {
        Certification.assertCalibrationInvariants(
            boundCalibrationState,
            uncertaintyCoverageEvidenceState,
        )
    }
}

/**
 * §7 `CertificationDatabase`. Generated from benchmark evidence, versioned with the app.
 *
 * The shipped instance is empty. §24 and §37 rule 12 forbid adding a record to make a test
 * pass; [withRecords] exists so a *test* can build its own in-memory database to exercise the
 * hit path, and it never touches a shipped artifact.
 */
public class CertificationDatabase private constructor(records: List<CertificationRecord>) {

    private val byKey: Map<CertificationKey, CertificationRecord> = records.associateBy { it.key }

    public val size: Int get() = byKey.size

    /**
     * Exact lookup. A miss on **any** component yields `CANDIDATE`.
     *
     * §24: a miss returns nothing, so the engine uses `CANDIDATE`, `unknownDeviceFloor95Deg`,
     * and a provisional ceiling no higher than `USABLE` — an upper limit, not a promise.
     */
    public fun lookup(
        key: CertificationKey,
        unknownDeviceFloor95Deg: Double,
    ): CertificationLookupOutcome {
        val record = byKey[key]
            ?: return CertificationLookupOutcome(
                boundCalibrationState = BoundCalibrationState.CANDIDATE,
                uncertaintyCoverageEvidenceState = UncertaintyCoverageEvidenceState.TARGET_ONLY,
                deviceFloor95Deg = unknownDeviceFloor95Deg,
                supportedQualityGrade = QualityGrade.USABLE,
                record = null,
            )
        return CertificationLookupOutcome(
            boundCalibrationState = BoundCalibrationState.CALIBRATED,
            uncertaintyCoverageEvidenceState =
                UncertaintyCoverageEvidenceState.EMPIRICALLY_CALIBRATED,
            deviceFloor95Deg = record.deviceFloor95Deg,
            supportedQualityGrade = record.supportedQualityGrade,
            record = record,
        )
    }

    public companion object {
        /** The database that actually ships in v1: empty, because no evidence exists. */
        public fun shipped(): CertificationDatabase = CertificationDatabase(emptyList())

        public fun withRecords(records: List<CertificationRecord>): CertificationDatabase =
            CertificationDatabase(records)
    }
}

/** A key component is missing, empty, or an open-ended range. */
public class CertificationKeyException(message: String) : IllegalArgumentException(message)
