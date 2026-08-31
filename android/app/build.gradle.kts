// SPEC.md §2: Kotlin, Compose, minSdk 26, portrait-locked, foreground-only.
// §4.1: the app module composes the pure core, the provider modules and the diagnostics
// module; it holds no heading logic of its own.
//
// Requires the Android SDK and the Android Gradle Plugin (deviation D-1).
plugins {
    alias(libs.plugins.android.application)
    alias(libs.plugins.kotlin.android)
}

android {
    namespace = "com.fengshuicompass.app"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.fengshuicompass.app"
        minSdk = 26
        targetSdk = 35
        versionCode = 1
        versionName = "0.1.0-phase0"
    }

    buildFeatures { compose = true }

    buildTypes {
        release {
            isMinifyEnabled = false
            // §8: "Remote configuration is prohibited in certification builds."
            // §23: a fake provider MUST NOT compile into a production release path.
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }
    kotlin { jvmToolchain(17) }
}

dependencies {
    implementation(project(":heading-core"))
    implementation(project(":fengshui-core"))
    implementation(project(":heading-google"))
    implementation(project(":heading-diagnostics"))
    // :benchmark-mode is intentionally absent from the release dependency set (§23, §29.7).
    debugImplementation(project(":benchmark-mode"))

    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.lifecycle.runtime.ktx)
    implementation(libs.androidx.activity.compose)
    implementation(platform(libs.androidx.compose.bom))
    implementation(libs.androidx.compose.ui)
    implementation(libs.androidx.compose.material3)
    implementation(libs.androidx.room.runtime)
}
