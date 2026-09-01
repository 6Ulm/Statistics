import Foundation

/// SPEC.md §6 enumerations — the subset the Phase 1 iOS core consumes.
///
/// Wire values are stable `UPPER_SNAKE_CASE` strings, everywhere, including examples and
/// fixtures (§22.2). Adding a case is backward compatible; renaming or reusing a stored value is
/// a schema migration, so every raw value is written out literally rather than derived from the
/// Swift case name.
///
/// > Warning: this file has never been compiled — see `docs/IMPLEMENTATION_NOTES.md` D-3.

public enum MeasurementMode: String, Sendable, CaseIterable {
    case flatTopEdge = "FLAT_TOP_EDGE"
    case wallFlushBack = "WALL_FLUSH_BACK"

    public var wire: String { rawValue }
}

public enum ProviderReferenceContract: String, Sendable, CaseIterable {
    /// `true` is a Swift keyword, so the case is named `trueReference`; the **wire value** is
    /// what §6 fixes and it is unchanged.
    case trueReference = "TRUE"
    case magnetic = "MAGNETIC"
    case trueIfDeclinationAvailableElseMagnetic = "TRUE_IF_DECLINATION_AVAILABLE_ELSE_MAGNETIC"
    case unknown = "UNKNOWN"

    public var wire: String { rawValue }
}

/// §3/§22.1 `referenceAxis` — the physical axis a bearing describes.
///
/// Kept distinct from `MeasurementMode` because §15.1 compares pipelines by *axis*: "a top-edge
/// scalar is not comparable to a wall-normal bearing merely because both are called heading".
public enum ReferenceAxis: String, Sendable, CaseIterable {
    case physicalTopEdgeHorizontalProjection = "PHYSICAL_TOP_EDGE_HORIZONTAL_PROJECTION"
    case outwardScreenNormalHorizontalProjection =
        "OUTWARD_SCREEN_NORMAL_HORIZONTAL_PROJECTION"

    public var wire: String { rawValue }
}

/// §3: `FLAT_TOP_EDGE` → portrait top edge; `WALL_FLUSH_BACK` → outward screen normal.
public func referenceAxisForMode(_ mode: MeasurementMode) -> ReferenceAxis {
    switch mode {
    case .flatTopEdge: return .physicalTopEdgeHorizontalProjection
    case .wallFlushBack: return .outwardScreenNormalHorizontalProjection
    }
}
