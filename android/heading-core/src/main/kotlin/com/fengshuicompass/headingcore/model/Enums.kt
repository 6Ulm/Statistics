package com.fengshuicompass.headingcore.model

/**
 * SPEC.md §6 enumerations.
 *
 * Wire values are stable `UPPER_SNAKE_CASE` strings, everywhere, including examples and
 * fixtures (§22.2). Adding a case is backward compatible; renaming or reusing a stored value
 * is a schema migration, so every constant's [wire] value is written out literally rather
 * than derived from the Kotlin name.
 */
public interface WireValued {
    public val wire: String
}

public enum class ProviderId(override val wire: String) : WireValued {
    GOOGLE_FOP("GOOGLE_FOP"),
    APPLE_CLHEADING("APPLE_CLHEADING"),
    APPLE_CORE_MOTION_TRUE_NORTH("APPLE_CORE_MOTION_TRUE_NORTH"),
    ANDROID_ROTATION_VECTOR("ANDROID_ROTATION_VECTOR"),
    ANDROID_HEADING("ANDROID_HEADING"),
    ANDROID_ACCEL_MAG("ANDROID_ACCEL_MAG"),
    REPLAY("REPLAY"),
}

public enum class LocationProviderId(override val wire: String) : WireValued {
    GOOGLE_FLP("GOOGLE_FLP"),
    ANDROID_FRAMEWORK_LOCATION("ANDROID_FRAMEWORK_LOCATION"),
    APPLE_CORE_LOCATION("APPLE_CORE_LOCATION"),
    REPLAY("REPLAY"),
}

public enum class ProviderErrorSource(override val wire: String) : WireValued {
    GOOGLE_CONSERVATIVE("GOOGLE_CONSERVATIVE"),
    GOOGLE_ORDINARY("GOOGLE_ORDINARY"),
    APPLE_HEADING_ACCURACY("APPLE_HEADING_ACCURACY"),
    ANDROID_ROTATION_VECTOR_HEADING_ACCURACY_95("ANDROID_ROTATION_VECTOR_HEADING_ACCURACY_95"),
    ANDROID_HEADING_ACCURACY_68("ANDROID_HEADING_ACCURACY_68"),
    NONE("NONE"),
}

public enum class ProviderReferenceContract(override val wire: String) : WireValued {
    TRUE("TRUE"),
    MAGNETIC("MAGNETIC"),
    TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC("TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC"),
    UNKNOWN("UNKNOWN"),
}

public enum class GeomagneticModelId(override val wire: String) : WireValued {
    WMM2025("WMM2025"),
    WMMHR2025("WMMHR2025"),
}

public enum class MeasurementMode(override val wire: String) : WireValued {
    FLAT_TOP_EDGE("FLAT_TOP_EDGE"),
    WALL_FLUSH_BACK("WALL_FLUSH_BACK"),
}

public enum class TargetReference(override val wire: String) : WireValued {
    TRUE("TRUE"),
    MAGNETIC("MAGNETIC"),
}

public enum class PlacementMethod(override val wire: String) : WireValued {
    FREEHAND("FREEHAND"),
    WALL_FLUSH_FREEHAND("WALL_FLUSH_FREEHAND"),
    NONMAGNETIC_ALIGNMENT_JIG("NONMAGNETIC_ALIGNMENT_JIG"),
    SURVEY_FIXTURE("SURVEY_FIXTURE"),
}

public enum class ResolvedReference(override val wire: String) : WireValued {
    TRUE_VERIFIED("TRUE_VERIFIED"),
    TRUE_CORRECTED_FROM_MAGNETIC("TRUE_CORRECTED_FROM_MAGNETIC"),
    TRUE_WITH_AMBIGUITY_BOUND("TRUE_WITH_AMBIGUITY_BOUND"),
    MAGNETIC("MAGNETIC"),
    UNVERIFIED("UNVERIFIED"),
}

public enum class ReferenceResolutionMethod(override val wire: String) : WireValued {
    /** iOS `trueHeading` validity. */
    PROVIDER_CONTRACT_EXPLICIT("PROVIDER_CONTRACT_EXPLICIT"),

    /** Core Motion frame confirmed in use. */
    ATTITUDE_FRAME_EXPLICIT("ATTITUDE_FRAME_EXPLICIT"),
    FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK("FRESH_LOCATION_WMM_MAGNETIC_CROSSCHECK"),

    /** AND-RV, known by construction. */
    APP_APPLIED_DECLINATION("APP_APPLIED_DECLINATION"),
    NOT_RESOLVED("NOT_RESOLVED"),
}

/**
 * The **one** measurement-state vocabulary (§6). Any coarser UI vocabulary is derived in the
 * view layer through a total tested mapping and is never persisted as an independent fact.
 */
public enum class MeasurementState(override val wire: String) : WireValued {
    IDLE("IDLE"),
    ACQUIRING_LOCATION("ACQUIRING_LOCATION"),
    ACQUIRING_ORIENTATION("ACQUIRING_ORIENTATION"),
    PROVIDER_INITIALIZING("PROVIDER_INITIALIZING"),
    CALIBRATION_CHECK("CALIBRATION_CHECK"),
    MAGNETIC_FIELD_CHECK("MAGNETIC_FIELD_CHECK"),
    TARGET_SEEKING("TARGET_SEEKING"),
    LEVEL_AND_HOLD("LEVEL_AND_HOLD"),
    STABILIZING("STABILIZING"),
    PRECISION_LOCKED("PRECISION_LOCKED"),
    DEGRADED("DEGRADED"),
    INVALID("INVALID"),
    TIMED_OUT("TIMED_OUT"),
}

public enum class ReferenceMagneticPrecheckState(override val wire: String) : WireValued {
    CLEAN_FOR_REFERENCE("CLEAN_FOR_REFERENCE"),
    NOT_CLEAN_FOR_REFERENCE("NOT_CLEAN_FOR_REFERENCE"),
    UNKNOWN("UNKNOWN"),
}

public enum class MagneticState(override val wire: String) : WireValued {
    CLEAN("CLEAN"),
    SUSPECT("SUSPECT"),
    DISTURBED("DISTURBED"),
    INVALID("INVALID"),
    UNKNOWN("UNKNOWN"),
}

public enum class SpaceWeatherState(override val wire: String) : WireValued {
    QUIET("QUIET"),
    ADVISORY("ADVISORY"),
    PROFESSIONAL_SUPPRESSED("PROFESSIONAL_SUPPRESSED"),
    EXTREME_WMM_UNUSABLE("EXTREME_WMM_UNUSABLE"),
    UNKNOWN("UNKNOWN"),
}

public enum class BoundCalibrationState(override val wire: String) : WireValued {
    CANDIDATE("CANDIDATE"),
    CALIBRATED("CALIBRATED"),
}

public enum class UncertaintyCoverageEvidenceState(override val wire: String) : WireValued {
    TARGET_ONLY("TARGET_ONLY"),
    EMPIRICALLY_CALIBRATED("EMPIRICALLY_CALIBRATED"),
    UNDEFINED("UNDEFINED"),
}

public enum class CalibrationKind(override val wire: String) : WireValued {
    SENSOR_CALIBRATION("SENSOR_CALIBRATION"),
    DEVIATION_CHARACTERIZATION("DEVIATION_CHARACTERIZATION"),
    UNCERTAINTY_CALIBRATION("UNCERTAINTY_CALIBRATION"),
}

public enum class CalibrationEntryReason(override val wire: String) : WireValued {
    AUTOMATIC_TRIGGER("AUTOMATIC_TRIGGER"),
    USER_REQUESTED("USER_REQUESTED"),
    BENCHMARK_PROTOCOL("BENCHMARK_PROTOCOL"),
}

public enum class CalibrationValidationOutcome(override val wire: String) : WireValued {
    IMPROVED("IMPROVED"),
    ACCEPTABLE_NO_CHANGE("ACCEPTABLE_NO_CHANGE"),
    STILL_POOR("STILL_POOR"),
    ENVIRONMENT_DISTURBED("ENVIRONMENT_DISTURBED"),
    INVALID_OR_INCONCLUSIVE("INVALID_OR_INCONCLUSIVE"),
}

public enum class DeviationCorrectionState(override val wire: String) : WireValued {
    NONE("NONE"),
    EXPERIMENTAL("EXPERIMENTAL"),
    CERTIFIED_PROFILE("CERTIFIED_PROFILE"),
}

public enum class DeviationStructureClass(override val wire: String) : WireValued {
    UNIT_STABLE("UNIT_STABLE"),
    MODEL_CLASS_STABLE("MODEL_CLASS_STABLE"),
    CALIBRATION_STATE_DEPENDENT("CALIBRATION_STATE_DEPENDENT"),
    SITE_DEPENDENT("SITE_DEPENDENT"),
    TRANSIENT("TRANSIENT"),
    NONREPEATABLE("NONREPEATABLE"),
}

/** §5 `DeviationCorrectionProfileMetadata.scope`. */
public enum class DeviationCorrectionScope(override val wire: String) : WireValued {
    UNIT("UNIT"),
    MODEL_CLASS("MODEL_CLASS"),
}

public enum class ChargingState(override val wire: String) : WireValued {
    NOT_CHARGING("NOT_CHARGING"),
    WIRED("WIRED"),
    WIRELESS("WIRELESS"),
    UNKNOWN("UNKNOWN"),
}

public enum class TrustAction(override val wire: String) : WireValued {
    READY_CALIBRATED("READY_CALIBRATED"),
    READY_CANDIDATE("READY_CANDIDATE"),
    SHOW_DEGRADED_RESULT("SHOW_DEGRADED_RESULT"),
    HOLD_STEADY("HOLD_STEADY"),
    ROTATE_TO_INITIALIZE("ROTATE_TO_INITIALIZE"),
    CALIBRATE("CALIBRATE"),
    MOVE_AWAY_FROM_INTERFERENCE("MOVE_AWAY_FROM_INTERFERENCE"),
    REACQUIRE_REFERENCE_OR_LOCATION("REACQUIRE_REFERENCE_OR_LOCATION"),
    UNSUPPORTED_OR_REJECTED("UNSUPPORTED_OR_REJECTED"),
}

/**
 * §6 `GradeLimitingFactor`.
 *
 * `CHARGING_STATE` is present deliberately: R57 records a build in which the value was
 * emitted as `gradeLimitedBy` while being absent from the enum. Declaration order is
 * load-bearing — §19 resolves exact ties by stable enum order.
 */
public enum class GradeLimitingFactor(override val wire: String) : WireValued {
    NONE("NONE"),
    PLACEMENT_UNCERTAINTY("PLACEMENT_UNCERTAINTY"),
    PROVIDER_ERROR("PROVIDER_ERROR"),
    SAMPLE_DISPERSION("SAMPLE_DISPERSION"),
    DEVICE_FLOOR("DEVICE_FLOOR"),
    DECLINATION_MODEL("DECLINATION_MODEL"),
    LOCATION_TIME_SENSITIVITY("LOCATION_TIME_SENSITIVITY"),
    REFERENCE_AMBIGUITY("REFERENCE_AMBIGUITY"),
    INTERFERENCE_PENALTY("INTERFERENCE_PENALTY"),
    DEVIATION_PROFILE_RESIDUAL("DEVIATION_PROFILE_RESIDUAL"),
    CERTIFICATION_CEILING("CERTIFICATION_CEILING"),
    SPACE_WEATHER("SPACE_WEATHER"),
    CHARGING_STATE("CHARGING_STATE"),
}

public enum class RejectionReason(override val wire: String) : WireValued {
    HEADING_UNAVAILABLE("HEADING_UNAVAILABLE"),
    HEADING_ERROR_INVALID("HEADING_ERROR_INVALID"),
    PROVIDER_NOT_INITIALIZED("PROVIDER_NOT_INITIALIZED"),
    ORIENTATION_STALE("ORIENTATION_STALE"),
    LOCATION_PERMISSION_DENIED("LOCATION_PERMISSION_DENIED"),
    LOCATION_STALE("LOCATION_STALE"),
    LOCATION_UNCERTAINTY_EXCEEDS_DECLINATION_BUDGET(
        "LOCATION_UNCERTAINTY_EXCEEDS_DECLINATION_BUDGET"
    ),
    LOCATION_JUMP_REQUIRES_FRESH_FIX("LOCATION_JUMP_REQUIRES_FRESH_FIX"),
    GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE("GEOMAGNETIC_MODEL_DATE_OUT_OF_RANGE"),
    WEAK_HORIZONTAL_FIELD("WEAK_HORIZONTAL_FIELD"),
    TRUE_REFERENCE_UNVERIFIED("TRUE_REFERENCE_UNVERIFIED"),
    MAGNETIC_CALIBRATION_INVALID("MAGNETIC_CALIBRATION_INVALID"),
    MAGNETIC_FIELD_SUSPECT("MAGNETIC_FIELD_SUSPECT"),
    MAGNETIC_FIELD_DISTURBED("MAGNETIC_FIELD_DISTURBED"),
    MAGNETIC_FIELD_UNKNOWN("MAGNETIC_FIELD_UNKNOWN"),
    TRANSFORM_DISAGREEMENT("TRANSFORM_DISAGREEMENT"),
    PIPELINE_DISAGREEMENT("PIPELINE_DISAGREEMENT"),
    CIRCULAR_MEAN_UNDEFINED("CIRCULAR_MEAN_UNDEFINED"),
    DEVICE_MOVING("DEVICE_MOVING"),
    DEVICE_NOT_LEVEL("DEVICE_NOT_LEVEL"),
    UNSUPPORTED_SCREEN_ORIENTATION("UNSUPPORTED_SCREEN_ORIENTATION"),
    ORIENTATION_CHANGED_DURING_WINDOW("ORIENTATION_CHANGED_DURING_WINDOW"),
    SENSOR_DISCONTINUITY("SENSOR_DISCONTINUITY"),
    APP_BACKGROUNDED("APP_BACKGROUNDED"),
    THERMAL_RESTRICTION("THERMAL_RESTRICTION"),
    WIRELESS_CHARGING_ACTIVE("WIRELESS_CHARGING_ACTIVE"),
    PROVIDER_FAILURE("PROVIDER_FAILURE"),
    SPACE_WEATHER_EXTREME("SPACE_WEATHER_EXTREME"),
    UNSUPPORTED_DEVICE("UNSUPPORTED_DEVICE"),
    ACQUISITION_TIMEOUT("ACQUISITION_TIMEOUT"),
    TARGET_REFERENCE_UNAVAILABLE("TARGET_REFERENCE_UNAVAILABLE"),
    TARGET_NOT_STABLE("TARGET_NOT_STABLE"),
    DEVIATION_PROFILE_NOT_CERTIFIED("DEVIATION_PROFILE_NOT_CERTIFIED"),
    REPEAT_MEASUREMENT_INCONSISTENT("REPEAT_MEASUREMENT_INCONSISTENT"),
}

/**
 * §5 `AltitudeSample.reference`. `UNKNOWN` is a real state (§2): it downgrades quality and is
 * not a synonym for either datum.
 */
public enum class AltitudeReference(override val wire: String) : WireValued {
    WGS84_ELLIPSOID("WGS84_ELLIPSOID"),
    MSL_ORTHOMETRIC("MSL_ORTHOMETRIC"),
    UNKNOWN("UNKNOWN"),
}

public enum class LocationAuthorizationAccuracy(override val wire: String) : WireValued {
    PRECISE_FULL("PRECISE_FULL"),
    APPROXIMATE_REDUCED("APPROXIMATE_REDUCED"),
}

/** §22 `sourceClock` — which clock the provider's raw timestamp came from. */
public enum class SourceClock(override val wire: String) : WireValued {
    ELAPSED_REALTIME("ELAPSED_REALTIME"),
    CORE_MOTION_BOOT_TIME("CORE_MOTION_BOOT_TIME"),
    PROVIDER_DATE("PROVIDER_DATE"),
    FIXTURE_CLOCK("FIXTURE_CLOCK"),
}

/**
 * §3/§22.1 `referenceAxis` — the physical axis a bearing describes.
 *
 * Kept distinct from [MeasurementMode] because §15.1 compares pipelines by *axis*: "a
 * top-edge scalar is not comparable to a wall-normal bearing merely because both are called
 * heading".
 */
public enum class ReferenceAxis(override val wire: String) : WireValued {
    PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION("PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION"),
    OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION("OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION"),
}

/** §3: `FLAT_TOP_EDGE` -> portrait top edge; `WALL_FLUSH_BACK` -> outward screen normal. */
public fun referenceAxisForMode(mode: MeasurementMode): ReferenceAxis =
    when (mode) {
        MeasurementMode.FLAT_TOP_EDGE -> ReferenceAxis.PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION
        MeasurementMode.WALL_FLUSH_BACK ->
            ReferenceAxis.OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION
    }
