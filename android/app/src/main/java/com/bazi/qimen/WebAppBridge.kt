package com.bazi.qimen

import android.content.Context
import android.location.Location
import android.location.LocationManager
import android.os.Handler
import android.os.Looper
import android.webkit.JavascriptInterface
import android.webkit.WebView
import androidx.core.content.ContextCompat
import androidx.core.location.LocationManagerCompat
import androidx.core.os.CancellationSignal
import org.json.JSONObject
import java.util.TimeZone

/**
 * Cầu nối JavaScript ↔ Android.
 *
 * Mọi phương thức @JavascriptInterface được WebView gọi trên luồng riêng của
 * JS, KHÔNG phải luồng chính — nên tất cả thao tác UI/định vị đều được đẩy về
 * main looper.
 *
 * JS ↔ Android bridge. Interface methods run on the WebView's JS thread;
 * anything touching UI or location is posted to the main looper.
 */
class WebAppBridge(
    private val activity: MainActivity,
    private val webView: WebView
) {

    private val main = Handler(Looper.getMainLooper())
    private val prefs = activity.getSharedPreferences(PREFS, Context.MODE_PRIVATE)

    private var cancelSignal: CancellationSignal? = null
    private var timeoutRunnable: Runnable? = null
    private var fixDelivered = false

    /* ─────────────── Đọc tài nguyên đóng gói ─────────────── */

    /**
     * Đọc một file trong `assets/web/`. Dùng thay cho fetch() vì trang chạy
     * trên file:// nên XHR/fetch bị CORS chặn.
     */
    @JavascriptInterface
    fun readAsset(path: String): String? {
        if (path.contains("..") || path.startsWith("/")) return null
        return try {
            activity.assets.open("web/$path").use { it.readBytes().toString(Charsets.UTF_8) }
        } catch (e: Exception) {
            null
        }
    }

    /* ─────────────── Lưu tuỳ chọn ─────────────── */

    @JavascriptInterface
    fun getPref(key: String): String? = prefs.getString(key, null)

    @JavascriptInterface
    fun setPref(key: String, value: String) {
        prefs.edit().putString(key, value).apply()
    }

    /* ─────────────── Múi giờ máy ─────────────── */

    @JavascriptInterface
    fun deviceTimeZone(): String = TimeZone.getDefault().id

    @JavascriptInterface
    fun platform(): String = "android"

    /**
     * Mời hệ thống ghim widget Lịch ra màn hình chính.
     * @return "ok" nếu đã hiện hộp thoại ghim, "unsupported" nếu máy hoặc
     *         launcher không cho ghim tự động (người dùng phải tự kéo widget).
     */
    @JavascriptInterface
    fun pinCalendarWidget(): String {
        val done = java.util.concurrent.atomic.AtomicBoolean(false)
        main.post { done.set(CalendarWidgetProvider.requestPin(activity)) }
        // Hộp thoại do hệ thống dựng; ở đây chỉ cần biết máy có hỗ trợ hay không.
        return if (CalendarWidgetProvider.canPin(activity)) "ok" else "unsupported"
    }

    /* ─────────────── Định vị ─────────────── */

    @JavascriptInterface
    fun hasLocationPermission(): Boolean = activity.hasLocationPermission()

    /**
     * Bắt đầu lấy toạ độ. Kết quả trả về JS qua `window.__onNativeLocation`.
     * Xin quyền trước nếu chưa có.
     */
    @JavascriptInterface
    fun requestLocation() {
        main.post {
            fixDelivered = false
            if (!activity.hasLocationPermission()) activity.askLocationPermission()
            else startLocationFix()
        }
    }

    /**
     * Chiến lược lấy toạ độ, ưu tiên nhanh và chạy được khi không có mạng:
     *  1. Bản ghi gần nhất còn mới (< 5 phút) → trả ngay.
     *  2. Yêu cầu định vị mới từ GPS và NETWORK song song, cái nào xong trước thì lấy.
     *  3. Quá 20 giây → dùng bản ghi cũ nếu có, không thì báo lỗi.
     */
    fun startLocationFix() {
        main.post {
            val lm = activity.getSystemService(Context.LOCATION_SERVICE) as? LocationManager
            if (lm == null) { deliverLocationError("no-location-service"); return@post }

            val lastKnown = bestLastKnown(lm)
            if (lastKnown != null &&
                System.currentTimeMillis() - lastKnown.time < FRESH_MS
            ) {
                deliverLocation(lastKnown)
                return@post
            }

            val providers = listOf(LocationManager.GPS_PROVIDER, LocationManager.NETWORK_PROVIDER)
                .filter { runCatching { lm.isProviderEnabled(it) }.getOrDefault(false) }

            if (providers.isEmpty()) {
                if (lastKnown != null) deliverLocation(lastKnown)
                else deliverLocationError("location-disabled")
                return@post
            }

            cancelPendingFix()
            val signal = CancellationSignal()
            cancelSignal = signal
            val executor = ContextCompat.getMainExecutor(activity)

            providers.forEach { provider ->
                try {
                    LocationManagerCompat.getCurrentLocation(lm, provider, signal, executor) { loc ->
                        if (loc != null) deliverLocation(loc)
                    }
                } catch (e: SecurityException) {
                    deliverLocationError("permission-denied")
                }
            }

            val timeout = Runnable {
                if (fixDelivered) return@Runnable
                cancelPendingFix()
                val stale = bestLastKnown(lm)
                if (stale != null) deliverLocation(stale) else deliverLocationError("timeout")
            }
            timeoutRunnable = timeout
            main.postDelayed(timeout, TIMEOUT_MS)
        }
    }

    private fun bestLastKnown(lm: LocationManager): Location? =
        lm.allProviders
            .mapNotNull { p -> runCatching { lm.getLastKnownLocation(p) }.getOrNull() }
            .maxByOrNull { it.time }

    fun cancelPendingFix() {
        cancelSignal?.let { runCatching { it.cancel() } }
        cancelSignal = null
        timeoutRunnable?.let { main.removeCallbacks(it) }
        timeoutRunnable = null
    }

    private fun deliverLocation(loc: Location) {
        if (fixDelivered) return
        fixDelivered = true
        cancelPendingFix()
        val json = JSONObject().apply {
            put("lat", loc.latitude)
            put("lon", loc.longitude)
            if (loc.hasAccuracy()) put("accuracy", loc.accuracy.toDouble())
            put("provider", loc.provider ?: "")
            put("time", loc.time)
            put("tzId", TimeZone.getDefault().id)
        }
        pushToJs(json)
    }

    fun deliverLocationError(reason: String) {
        if (fixDelivered) return
        fixDelivered = true
        cancelPendingFix()
        pushToJs(JSONObject().put("error", reason))
    }

    private fun pushToJs(payload: JSONObject) {
        val script = "window.__onNativeLocation && window.__onNativeLocation($payload);"
        main.post { webView.evaluateJavascript(script, null) }
    }

    companion object {
        const val NAME = "QMDJNative"
        private const val PREFS = "qmdj_prefs"
        private const val FRESH_MS = 5 * 60_000L
        private const val TIMEOUT_MS = 20_000L
    }
}
