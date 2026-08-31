// SPEC.md §4.1: heading-core is pure — no UI, no framework singleton, no Android SDK.
// It is therefore a plain Kotlin/JVM library, consumable unchanged by the Android
// modules and runnable on any JDK so the Phase 0 gates execute without an SDK.
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

dependencies {
    implementation(libs.kotlinx.serialization.json)
    testImplementation(libs.junit.jupiter)
    testRuntimeOnly(libs.junit.platform.launcher)
}

kotlin {
    jvmToolchain(17)
    compilerOptions {
        allWarningsAsErrors.set(true)
    }
}

tasks.test {
    useJUnitPlatform()
    // Shared artifacts are read from the repository root, never copied into the module.
    systemProperty("fsc.repoRoot", rootProject.projectDir.parentFile.absolutePath)
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}
