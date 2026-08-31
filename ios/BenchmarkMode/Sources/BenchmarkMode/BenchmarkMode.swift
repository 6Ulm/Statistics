// SPEC.md §4.1: "Benchmark modules are internal-build only and depend on the same
// production core." §29.7 specifies the in-app benchmark screen and its rule that
// debug/export code MUST NOT change request rate, thread priority, filtering, or lifecycle
// relative to the measured production candidate, and that ground truth can never be copied
// from the phone result.
//
// The dependency direction is deliberate: this target depends on the production core and
// nothing in the production path depends on it, so a replay or fake provider defined here
// cannot reach a release build (§23). Phase 3/4 work.
