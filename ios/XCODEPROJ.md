# `ios/FengShuiCompass.xcodeproj`

SPEC.md §4.1 lists an Xcode project at this path. It is **not** in the repository yet, and
this file records why rather than shipping something that cannot be opened.

An `.xcodeproj` is a generated artifact whose `project.pbxproj` encodes build settings,
target membership, entitlements and code-signing configuration. Hand-authoring one on a
Linux CI container produces a file that no toolchain in this environment can open, parse
or validate — a placeholder that looks like a deliverable. SPEC.md §37 forbids exactly
that ("Do not hide incomplete work behind placeholders — a deliberate unsupported state
with a reason beats a production `TODO`, fake value, or silent fallback").

What exists instead, and is verifiable:

- `ios/Package.swift` declares the §4.1 module boundaries — `HeadingCore`, `FengShuiCore`,
  `HeadingApple`, `HeadingDiagnostics`, `BenchmarkMode` — with their dependency directions
  and the iOS 17 platform floor. Those are the boundaries §4.1 says MUST NOT vary.
- `ios/FengShuiCompass/` holds the app target's sources.
- The Phase 0 configuration-invariant and grade-reachability suites live in
  `ios/HeadingCore/Tests/HeadingCoreTests/`, mirroring the Kotlin and Python
  implementations against the same repository-root artifacts.

**Phase 2 obligation.** On a macOS host with Xcode, create the app project referencing this
package as a local Swift package dependency, commit it, and remove this file. Until then the
iOS skeleton is not built or tested; see the Phase 0 status statement in
`docs/IMPLEMENTATION_NOTES.md`.
