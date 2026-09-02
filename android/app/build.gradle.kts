import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
}

android {
    namespace = "com.bazi.qimen"
    compileSdk = 35

    defaultConfig {
        applicationId = "com.bazi.qimen"
        minSdk = 24
        targetSdk = 35
        // TĂNG versionCode mỗi lần dựng bản mới đem cài lên máy. Để nguyên thì
        // trong Settings → Apps hai bản trông y hệt nhau, mà cài đè thất bại
        // (thường do khác chữ ký) lại nhìn giống hệt cài thành công — không có
        // cách nào biết máy đang chạy bản nào.
        versionCode = 2
        versionName = "1.1"
    }

    buildTypes {
        release {
            isMinifyEnabled = true
            isShrinkResources = true
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
        debug {
            applicationIdSuffix = ".debug"
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    packaging {
        resources.excludes += setOf("/META-INF/{AL2.0,LGPL2.1}")
    }
}

// DSL mới thay cho `android { kotlinOptions { jvmTarget = "17" } }`: khối
// kotlinOptions đã bị bỏ ở Kotlin 2.2, để nguyên thì build đứt.
kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.13.1")
}
