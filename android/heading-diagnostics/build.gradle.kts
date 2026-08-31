// SPEC.md §4.1: "Diagnostic modules never become the production estimator."
// This module owns the raw SensorManager streams (§13 diagnostic streams) and the
// accel/gravity + magnetometer baseline used for §11 reference disambiguation and §16
// interference features. It must never be blended back into the fused heading.
//
// Requires the Android SDK and the Android Gradle Plugin (deviation D-1).
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.fengshuicompass.headingdiagnostics"
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
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
