package com.bazi.qimen

import android.app.AlarmManager
import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.RectF
import android.graphics.Typeface
import android.os.Build
import android.util.TypedValue
import android.widget.RemoteViews
import java.util.Calendar

/**
 * Widget màn hình chính: CHỈ lịch âm dương, không có bàn Kỳ Môn.
 * Xem lịch mà không phải mở ứng dụng.
 *
 * RemoteViews không nhận WebView và cũng không dựng nổi lưới 7×6 gọn gàng,
 * nên tháng lịch được VẼ ra bitmap rồi gắn vào một ImageView.
 *
 * Home-screen widget showing only the lunar calendar, drawn to a bitmap.
 */
class CalendarWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(
        context: Context, manager: AppWidgetManager, ids: IntArray
    ) {
        LunarTable.ensureLoaded(context)
        ids.forEach { id -> render(context, manager, id) }
        scheduleMidnight(context)
    }

    override fun onAppWidgetOptionsChanged(
        context: Context, manager: AppWidgetManager, id: Int, newOptions: android.os.Bundle
    ) {
        LunarTable.ensureLoaded(context)
        render(context, manager, id)
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        // Sang ngày mới / đổi múi giờ thì phải vẽ lại, nếu không ô "hôm nay"
        // vẫn nằm ở ngày cũ.
        when (intent.action) {
            ACTION_REFRESH -> {
                val manager = AppWidgetManager.getInstance(context)
                val ids = manager.getAppWidgetIds(
                    ComponentName(context, CalendarWidgetProvider::class.java)
                )
                if (ids.isNotEmpty()) onUpdate(context, manager, ids)
            }
        }
    }

    override fun onEnabled(context: Context) = scheduleMidnight(context)

    override fun onDisabled(context: Context) {
        alarm(context)?.let { (am, pi) -> am.cancel(pi) }
    }

    /* ─────────────── Vẽ ─────────────── */

    private fun render(context: Context, manager: AppWidgetManager, id: Int) {
        val opts = manager.getAppWidgetOptions(id)
        val wDp = opts.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0)
            .takeIf { it > 0 } ?: 320
        val hDp = opts.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_HEIGHT, 0)
            .takeIf { it > 0 } ?: 220

        val bmp = drawMonth(context, wDp, hDp)
        val views = RemoteViews(context.packageName, R.layout.widget_calendar)
        views.setImageViewBitmap(R.id.widgetImage, bmp)

        // Chạm vào widget thì mở thẳng tab Lịch, không phải bàn Kỳ Môn.
        val open = Intent(context, MainActivity::class.java).apply {
            action = Intent.ACTION_MAIN
            putExtra(MainActivity.EXTRA_TAB, "cal")
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        val pi = PendingIntent.getActivity(
            context, 0, open, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        views.setOnClickPendingIntent(R.id.widgetImage, pi)
        manager.updateAppWidget(id, views)
    }

    private fun dp(context: Context, v: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v, context.resources.displayMetrics
    )

    private fun drawMonth(context: Context, wDp: Int, hDp: Int): Bitmap {
        val w = dp(context, wDp.toFloat()).toInt().coerceIn(240, 2000)
        val h = dp(context, hDp.toFloat()).toInt().coerceIn(160, 2000)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)

        val now = Calendar.getInstance()
        val year = now.get(Calendar.YEAR)
        val month = now.get(Calendar.MONTH) + 1
        val todayJdn = LunarTable.jdn(year, month, now.get(Calendar.DAY_OF_MONTH))

        val first = Calendar.getInstance().apply { set(year, month - 1, 1) }
        // Tuần bắt đầu từ Thứ 2, giống lịch trong ứng dụng.
        val lead = (first.get(Calendar.DAY_OF_WEEK) + 5) % 7
        val daysInMonth = first.getActualMaximum(Calendar.DAY_OF_MONTH)
        val weeks = Math.ceil((lead + daysInMonth) / 7.0).toInt().coerceAtLeast(1)

        val paint = Paint(Paint.ANTI_ALIAS_FLAG)
        val headH = h * 0.13f
        val dowH = h * 0.09f
        val gridTop = headH + dowH
        val cellW = w / 7f
        val cellH = (h - gridTop) / weeks

        // nền
        paint.color = Color.parseColor("#FDECEF")
        c.drawRoundRect(RectF(0f, 0f, w.toFloat(), h.toFloat()), dp(context, 10f), dp(context, 10f), paint)

        // thanh tiêu đề đỏ
        paint.color = Color.parseColor("#D32F2F")
        c.drawRoundRect(RectF(0f, 0f, w.toFloat(), headH + dp(context, 10f)), dp(context, 10f), dp(context, 10f), paint)
        c.drawRect(0f, headH - 1, w.toFloat(), headH, paint)

        paint.color = Color.WHITE
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = headH * 0.46f
        c.drawText("LỊCH ÂM THÁNG $month/$year", w / 2f, headH * 0.66f, paint)

        // hàng thứ
        val dows = arrayOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
        paint.textSize = dowH * 0.52f
        for (i in 0..6) {
            paint.color = if (i == 6) Color.parseColor("#C62828") else Color.parseColor("#7A2B33")
            c.drawText(dows[i], cellW * (i + 0.5f), headH + dowH * 0.68f, paint)
        }

        // các ô ngày
        for (d in 1..daysInMonth) {
            val idx = lead + d - 1
            val col = idx % 7
            val row = idx / 7
            val x = cellW * col
            val y = gridTop + cellH * row
            val jdn = LunarTable.jdn(year, month, d)
            val isToday = jdn == todayJdn

            if (isToday) {
                paint.color = Color.parseColor("#D32F2F")
                c.drawRect(x + 1, y + 1, x + cellW - 1, y + cellH - 1, paint)
            }

            val fg = if (isToday) Color.WHITE else Color.parseColor("#222222")
            val dim = if (isToday) Color.parseColor("#FFE0E3") else Color.parseColor("#6B6B6B")

            paint.typeface = Typeface.DEFAULT_BOLD
            paint.textAlign = Paint.Align.LEFT
            paint.textSize = cellH * 0.30f
            paint.color = fg
            c.drawText(d.toString(), x + cellW * 0.10f, y + cellH * 0.34f, paint)

            val lunar = LunarTable.lunarOf(jdn)
            paint.typeface = Typeface.DEFAULT
            paint.textAlign = Paint.Align.RIGHT
            paint.textSize = cellH * 0.21f
            paint.color = dim
            val lunarTxt = when {
                lunar == null -> ""
                lunar.day == 1 -> "${lunar.day}/${lunar.month}"
                else -> lunar.day.toString()
            }
            c.drawText(lunarTxt, x + cellW * 0.92f, y + cellH * 0.32f, paint)

            // can một dòng, chi một dòng — như lịch trong ứng dụng
            val (can, chi) = LunarTable.ganZhiOf(jdn)
            paint.textAlign = Paint.Align.CENTER
            paint.textSize = cellH * 0.21f
            paint.color = if (isToday) Color.WHITE else Color.parseColor("#555555")
            c.drawText(can, x + cellW / 2f, y + cellH * 0.63f, paint)
            c.drawText(chi, x + cellW / 2f, y + cellH * 0.88f, paint)
        }

        // lưới
        paint.color = Color.parseColor("#F3C6CD")
        paint.strokeWidth = 1f
        for (i in 1..6) c.drawLine(cellW * i, gridTop, cellW * i, h.toFloat(), paint)
        for (r in 0..weeks) {
            val y = gridTop + cellH * r
            c.drawLine(0f, y, w.toFloat(), y, paint)
        }
        return bmp
    }

    /* ─────────────── Tự làm mới lúc nửa đêm ─────────────── */

    private fun alarm(context: Context): Pair<AlarmManager, PendingIntent>? {
        val am = context.getSystemService(Context.ALARM_SERVICE) as? AlarmManager ?: return null
        val intent = Intent(context, CalendarWidgetProvider::class.java).setAction(ACTION_REFRESH)
        val pi = PendingIntent.getBroadcast(
            context, 1, intent, PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
        return am to pi
    }

    /**
     * Hẹn giờ vẽ lại ngay sau nửa đêm. Dùng báo thức lặp không chính xác —
     * chỉ cần đúng ngày, không cần đúng giây, và đỡ tốn pin hơn hẳn so với
     * để `updatePeriodMillis` đánh thức nửa tiếng một lần.
     */
    private fun scheduleMidnight(context: Context) {
        val (am, pi) = alarm(context) ?: return
        val next = Calendar.getInstance().apply {
            add(Calendar.DAY_OF_YEAR, 1)
            set(Calendar.HOUR_OF_DAY, 0)
            set(Calendar.MINUTE, 0)
            set(Calendar.SECOND, 30)
            set(Calendar.MILLISECOND, 0)
        }
        am.setInexactRepeating(
            AlarmManager.RTC, next.timeInMillis, AlarmManager.INTERVAL_DAY, pi
        )
    }

    companion object {
        const val ACTION_REFRESH = "com.bazi.qimen.WIDGET_REFRESH"

        /**
         * Mời người dùng ghim widget lịch ra màn hình chính.
         * @return false nếu máy/launcher không hỗ trợ ghim tự động.
         */
        /** Máy này có cho ghim widget bằng một cú chạm không? */
        fun canPin(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            return manager.isRequestPinAppWidgetSupported
        }

        fun requestPin(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            if (!manager.isRequestPinAppWidgetSupported) return false
            val provider = ComponentName(context, CalendarWidgetProvider::class.java)
            return manager.requestPinAppWidget(provider, null, null)
        }
    }
}
