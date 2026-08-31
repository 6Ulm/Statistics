// Root build script. No plugins are applied here; each module declares its own so a
// module that cannot be configured in a given environment does not break the others.
plugins {
    alias(libs.plugins.kotlin.jvm) apply false
    alias(libs.plugins.kotlin.serialization) apply false
}

// Absolute path to the repository root, one level above this Gradle build. Shared
// artifacts (config/, schemas/, testdata/) live there so the Android and iOS runtimes
// and analysis/ read byte-identical files rather than copies that can drift (§37.1
// multi-agent handoff).
val fscRepoRoot: String = rootProject.projectDir.parentFile.absolutePath
extra["fscRepoRoot"] = fscRepoRoot
