package com.fengshuicompass.headingcore.config

import java.io.File

/**
 * Locates the repository-root shared artifacts (`config/`, `schemas/`, `testdata/`)
 * that the Android runtime, the iOS runtime and `analysis/` all read.
 *
 * SPEC.md §37.1 requires both platforms to consume the *same* files rather than copies,
 * because independently translated constants are a common source of false parity. In a
 * shipped app these files are bundled read-only; in the JVM test runtime they are read
 * from the checkout via the `fsc.repoRoot` system property set by the module build.
 */
public object SharedArtifacts {

    public const val REPO_ROOT_PROPERTY: String = "fsc.repoRoot"

    public val repoRoot: File
        get() {
            val configured = System.getProperty(REPO_ROOT_PROPERTY)
                ?: error(
                    "System property $REPO_ROOT_PROPERTY is not set. The Gradle test task " +
                        "sets it to the repository root; a runtime that cannot locate the " +
                        "shared artifacts must fail rather than fall back to a copy."
                )
            val root = File(configured)
            require(root.isDirectory) { "$REPO_ROOT_PROPERTY=$configured is not a directory" }
            return root
        }

    public val precisionProfileFile: File get() = resolve("config/precision-profile-v1.json")

    public val fengShuiRuleSetFile: File get() = resolve("config/feng-shui-rules-v1.json")

    public val gradeReachabilityClaimsFile: File
        get() = resolve("testdata/grade-reachability-claims-v1.json")

    public val exampleEngineOutputEventFile: File
        get() = resolve("testdata/telemetry-event-engine-output-v1.example.json")

    // ---------------------------------------------------------------------------------
    // The frozen `fixtures-v1` shared contract (SPEC.md §37.1).
    //
    // "give both agents the same files and tolerances, with neither editing shared fixtures
    // unilaterally; a contract change requires one reviewed change set, regenerated fixtures,
    // both test suites, and a new fixture version".
    // ---------------------------------------------------------------------------------

    public const val FIXTURE_VERSION: String = "fixtures-v1"

    public val circularMathFixture: File get() = resolve("testdata/angles/circular-math-v1.json")

    public val estimatorsFixture: File get() = resolve("testdata/angles/estimators-v1.json")

    public val circularAggregateFixture: File
        get() = resolve("testdata/angles/circular-aggregate-v1.json")

    public val attitudeGoldenFixture: File
        get() = resolve("testdata/quaternions/attitude-golden-v1.json")

    public val fengShuiClassificationFixture: File
        get() = resolve("testdata/fengshui/classification-v1.json")

    public val fengShuiReferenceTransformFixture: File
        get() = resolve("testdata/fengshui/reference-transform-v1.json")

    public val uncertaintyCompositionFixture: File
        get() = resolve("testdata/uncertainty/composition-v1.json")

    public val magneticClassificationFixture: File
        get() = resolve("testdata/magnetic/classification-v1.json")

    public val referenceResolutionFixture: File
        get() = resolve("testdata/reference/resolution-v1.json")

    public val stateReducerFixture: File get() = resolve("testdata/state/reducer-v1.json")

    public val certificationKeyFixture: File get() = resolve("testdata/certification/key-v1.json")

    public val telemetryCodecFixture: File get() = resolve("testdata/telemetry/codec-v1.json")

    public val fixturesManifest: File get() = resolve("testdata/fixtures-v1.manifest.json")

    public fun resolve(relativePath: String): File {
        val file = File(repoRoot, relativePath)
        require(file.isFile) { "required shared artifact is missing: $relativePath" }
        return file
    }
}
