package com.bazi.qimen

import android.Manifest
import android.annotation.SuppressLint
import android.content.Intent
import android.content.pm.PackageManager
import android.graphics.Color
import android.os.Bundle
import android.view.View
import android.view.ViewGroup
import android.webkit.WebView
import android.webkit.WebViewClient
import android.widget.FrameLayout
import androidx.core.app.ActivityCompat
import androidx.core.content.ContextCompat
import androidx.core.view.ViewCompat
import androidx.core.view.WindowCompat
import androidx.core.view.WindowInsetsCompat
import androidx.core.view.updatePadding

/**
 * Vỏ Android cho ứng dụng Bát Tự & Kỳ Môn.
 *
 * Toàn bộ giao diện và engine tính toán nằm trong `assets/web/` và chạy trong
 * một WebView trỏ vào `file:///android_asset/`. Ứng dụng KHÔNG có quyền
 * INTERNET — mọi thứ chạy offline.
 *
 * Android shell: a single WebView over bundled assets. No INTERNET permission.
 */
class MainActivity : android.app.Activity() {

    private lateinit var webView: WebView
    private lateinit var bridge: WebAppBridge

    private val requestLocationCode = 4211

    /** Tab cần mở ngay khi trang nạp xong (do widget yêu cầu). */
    private var pendingTab: String? = null

    /** WebView đã bị huỷ (render process chết) — không được đụng vào nữa. */
    private var webViewGone = false

    @SuppressLint("SetJavaScriptEnabled")
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        WindowCompat.setDecorFitsSystemWindows(window, false)

        val root = FrameLayout(this).apply {
            layoutParams = ViewGroup.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
            setBackgroundColor(BACKGROUND)
        }

        webView = WebView(this).apply {
            setBackgroundColor(BACKGROUND)
            overScrollMode = View.OVER_SCROLL_NEVER
            layoutParams = FrameLayout.LayoutParams(
                ViewGroup.LayoutParams.MATCH_PARENT,
                ViewGroup.LayoutParams.MATCH_PARENT
            )
        }
        root.addView(webView)
        setContentView(root)

        // Chừa chỗ cho thanh trạng thái / thanh điều hướng (edge-to-edge của Android 15).
        ViewCompat.setOnApplyWindowInsetsListener(root) { view, insets ->
            val bars = insets.getInsets(
                WindowInsetsCompat.Type.systemBars() or WindowInsetsCompat.Type.ime()
            )
            view.updatePadding(bars.left, bars.top, bars.right, bars.bottom)
            insets
        }

        with(webView.settings) {
            javaScriptEnabled = true
            domStorageEnabled = true            // localStorage cho tuỳ chọn người dùng
            databaseEnabled = true
            allowFileAccess = false             // assets vẫn đọc được, file:// khác thì không
            allowContentAccess = false
            mediaPlaybackRequiresUserGesture = true
            // Bảng Kỳ Môn là lưới 3×3 dày đặc, được thiết kế theo px cố định:
            // giữ nguyên cỡ chữ để lưới không vỡ khi hệ thống phóng to chữ.
            textZoom = 100
            useWideViewPort = true
            loadWithOverviewMode = false
            setSupportZoom(false)
            builtInZoomControls = false
        }

        bridge = WebAppBridge(this, webView)
        webView.addJavascriptInterface(bridge, WebAppBridge.NAME)

        webView.webViewClient = object : WebViewClient() {
            // Ứng dụng offline: chặn mọi điều hướng ra ngoài assets.
            override fun shouldOverrideUrlLoading(
                view: WebView, request: android.webkit.WebResourceRequest
            ): Boolean = !request.url.toString().startsWith(ASSET_BASE)

            // Tiến trình render của WebView chết thì cả app chết theo nếu không
            // xử lý — dựng lại trang thay vì để hệ thống giết tiến trình.
            override fun onPageFinished(view: WebView, url: String) {
                pendingTab?.let { showTab(it); pendingTab = null }
            }

            override fun onRenderProcessGone(
                view: WebView, detail: android.webkit.RenderProcessGoneDetail
            ): Boolean {
                webViewGone = true
                (view.parent as? ViewGroup)?.removeView(view)
                view.destroy()
                recreate()
                return true          // true = đã xử lý, đừng giết tiến trình
            }
        }

        if (isDebuggable()) WebView.setWebContentsDebuggingEnabled(true)

        // Mở từ widget thì vào thẳng tab Lịch. Trang nạp xong mới gọi được JS,
        // nên phải chờ onPageFinished.
        val startTab = intent?.getStringExtra(EXTRA_TAB)
        if (startTab != null) pendingTab = startTab

        if (savedInstanceState != null) {
            webView.restoreState(savedInstanceState)
        } else {
            webView.loadUrl(START_URL)
        }
    }

    override fun onNewIntent(intent: Intent?) {
        super.onNewIntent(intent)
        val tab = intent?.getStringExtra(EXTRA_TAB) ?: return
        showTab(tab)
    }

    private fun showTab(tab: String) {
        if (webViewGone) return
        val safe = if (tab == "cal") "cal" else "qmdj"
        webView.evaluateJavascript("window.showTab && window.showTab('$safe');", null)
    }

    private fun isDebuggable(): Boolean =
        (applicationInfo.flags and android.content.pm.ApplicationInfo.FLAG_DEBUGGABLE) != 0

    override fun onSaveInstanceState(outState: Bundle) {
        super.onSaveInstanceState(outState)
        if (!webViewGone) webView.saveState(outState)
    }

    /** Nút Back: đóng bảng chọn đang mở trước, chỉ thoát khi không còn gì để đóng. */
    @Deprecated("Deprecated in Java")
    override fun onBackPressed() {
        webView.evaluateJavascript(
            "(function(){return !!(window.__onBackPressed && window.__onBackPressed());})()"
        ) { result ->
            // Không gọi super trong lambda được (Kotlin cấm) → đi qua hàm trung gian.
            if (result != "true") defaultBack()
        }

    }

    @Suppress("DEPRECATION")
    private fun defaultBack() = super.onBackPressed()

    /* ─────────────── Quyền vị trí ─────────────── */

    fun hasLocationPermission(): Boolean =
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_FINE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED ||
        ContextCompat.checkSelfPermission(this, Manifest.permission.ACCESS_COARSE_LOCATION) ==
            PackageManager.PERMISSION_GRANTED

    fun askLocationPermission() {
        ActivityCompat.requestPermissions(
            this,
            arrayOf(
                Manifest.permission.ACCESS_FINE_LOCATION,
                Manifest.permission.ACCESS_COARSE_LOCATION
            ),
            requestLocationCode
        )
    }

    override fun onRequestPermissionsResult(
        requestCode: Int, permissions: Array<out String>, grantResults: IntArray
    ) {
        super.onRequestPermissionsResult(requestCode, permissions, grantResults)
        if (requestCode != requestLocationCode) return
        if (grantResults.any { it == PackageManager.PERMISSION_GRANTED }) {
            bridge.startLocationFix()
        } else {
            bridge.deliverLocationError("permission-denied")
        }
    }

    override fun onDestroy() {
        bridge.cancelPendingFix()
        if (!webViewGone) {
            webView.removeJavascriptInterface(WebAppBridge.NAME)
            (webView.parent as? ViewGroup)?.removeView(webView)
            webView.destroy()
        }
        super.onDestroy()
    }

    companion object {
        const val EXTRA_TAB = "com.bazi.qimen.START_TAB"
        const val ASSET_BASE = "file:///android_asset/web/"
        const val START_URL = ASSET_BASE + "index.html"
        val BACKGROUND = Color.parseColor("#f4f7f9")
    }
}
