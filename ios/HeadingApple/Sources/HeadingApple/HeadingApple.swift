// SPEC.md §4.1: provider modules wrap SDKs. `HeadingApple` owns the Core Location and
// Core Motion adapters described in §12 — `CLLocationManager`/`CLHeading` for the
// FLAT_TOP_EDGE production scalar, and `CMDeviceMotion` in `.xTrueNorthZVertical` for the
// WALL_FLUSH_BACK outward-screen-normal projection — including the §12 clock mapping and
// discontinuity inference.
//
// Phase 2 owns that code. Phase 0 establishes the boundary and nothing else: an empty
// module is an honest statement that the adapters do not exist yet, where a stub provider
// would be a fake value this specification forbids (§23, §37).
