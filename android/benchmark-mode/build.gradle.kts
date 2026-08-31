// SPEC.md §4.1: "Benchmark modules are internal-build only and depend on the same
// production core." §29.7 defines the in-app benchmark screen; §29.7 also requires that
// debug/export code MUST NOT change request rate, thread priority, filtering, or
// lifecycle relative to the measured production candidate.
//
// The dependency direction is deliberate: this module depends on the production core, and
// nothing in the production path depends on it, so a replay or fake provider defined here
// cannot reach a release build (§23; scripts/verify-artifacts.sh asserts this).
//
// Requires the Android SDK and the Android Gradle Plugin (deviation D-1).
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.fengshuicompass.benchmarkmode"
    compileSdk = 35
    defaultConfig { minSdk = 26 }
    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin { jvmToolchain(17) }
}

dependencies {
    api(project(":heading-core"))
    api(project(":fengshui-core"))
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
