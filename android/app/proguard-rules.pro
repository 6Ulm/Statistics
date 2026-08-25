# Cầu nối JS ↔ Kotlin: R8 không được đổi tên/xoá các phương thức @JavascriptInterface,
# vì WebView gọi chúng bằng tên tại thời điểm chạy.
-keepclassmembers class com.bazi.qimen.WebAppBridge {
    @android.webkit.JavascriptInterface <methods>;
}
-keepattributes JavascriptInterface
