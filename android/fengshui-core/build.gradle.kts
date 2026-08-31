// SPEC.md §4.1: fengshui-core is pure. It consumes a gated HeadingMeasurement, never
// sensors (§4). It depends on heading-core only for the single normative circular-math
// utility set, so no second angle definition can appear (§9, R68).
plugins {
    alias(libs.plugins.kotlin.jvm)
    alias(libs.plugins.kotlin.serialization)
}

dependencies {
    api(project(":heading-core"))
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
    systemProperty("fsc.repoRoot", rootProject.projectDir.parentFile.absolutePath)
    testLogging {
        events("passed", "skipped", "failed")
        showStandardStreams = true
    }
}
