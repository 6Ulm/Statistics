// SPEC.md §4.1 module boundaries. `heading-core` and `fengshui-core` are the pure
// modules: no UI, no framework singleton, no Android SDK. They are plain Kotlin/JVM
// libraries so the Phase 0 configuration-invariant and grade-reachability suites can
// run on any JDK, and so an Android module can consume them unchanged.
//
// The Android modules require the Android Gradle Plugin and an installed SDK. Where
// neither is available, run with -PfscIncludeAndroidModules=false; the build then
// prints exactly which modules were excluded rather than silently degrading.
// See docs/IMPLEMENTATION_NOTES.md deviation D-1.

pluginManagement {
    repositories {
        gradlePluginPortal()
        google()
        mavenCentral()
    }
}

dependencyResolutionManagement {
    repositoriesMode.set(RepositoriesMode.FAIL_ON_PROJECT_REPOS)
    repositories {
        google()
        mavenCentral()
    }
}

rootProject.name = "fengshui-compass-android"

// Pure Kotlin/JVM modules — always included.
include(":heading-core")
include(":fengshui-core")

val includeAndroidModules =
    (providers.gradleProperty("fscIncludeAndroidModules").orNull ?: "true").toBoolean()

val androidModules = listOf(":app", ":heading-google", ":heading-diagnostics", ":benchmark-mode")

if (includeAndroidModules) {
    androidModules.forEach { include(it) }
} else {
    logger.lifecycle(
        "fscIncludeAndroidModules=false: excluded ${androidModules.joinToString(", ")}. " +
            "The Android SDK and the Android Gradle Plugin are required to build them; " +
            "the pure-core suites below do not cover Android adapter code."
    )
}
