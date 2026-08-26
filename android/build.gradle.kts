// Phiên bản để ở một chỗ duy nhất — đổi ở đây là đổi cả dự án.
//
// Gradle: 9.3.0 (xem gradle/wrapper/gradle-wrapper.properties). Bản wrapper mới
// là bản QUYẾT ĐỊNH: chạy ./gradlew thì Gradle cài sẵn trên máy không được dùng
// tới, nên hai bên không cần trùng nhau.
//
// AGP phải là bản có hỗ trợ Gradle 9 — AGP 8.7 (bản cũ của dự án) chỉ chạy tới
// Gradle 8.x và sẽ dừng ngay với thông báo "Minimum supported Gradle version".
// Nếu Android Studio báo AGP này quá cũ hoặc quá mới so với Gradle 9.3, mở
// Tools → AGP Upgrade Assistant, hoặc sửa thẳng con số dưới đây; không có gì
// khác trong dự án phụ thuộc vào nó.
plugins {
    id("com.android.application") version "8.13.0" apply false
    id("org.jetbrains.kotlin.android") version "2.2.21" apply false
}
