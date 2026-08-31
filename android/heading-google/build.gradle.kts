// SPEC.md §4.1: provider modules wrap SDKs. heading-google owns the Google Play services
// paths only — FusedOrientationProviderClient (§13) and FusedLocationProviderClient. The
// no-GMS TYPE_ROTATION_VECTOR path (§2.2, §30.4) is certified separately and MUST NOT be
// substituted silently, so it does not live here.
//
// Requires the Android SDK and the Android Gradle Plugin. See settings.gradle.kts and
// docs/IMPLEMENTATION_NOTES.md deviation D-1.
plugins {
    alias(libs.plugins.android.library)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.fengshuicompass.headinggoogle"
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
    implementation(libs.play.services.location)
    implementation(libs.androidx.lifecycle.runtime.ktx)
}
