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
import android.graphics.Typeface
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.util.TypedValue
import android.view.View
import android.widget.RemoteViews
import java.util.Calendar

/**
 * Widget màn hình chính: đúng những gì tab Lịch có — lưới lịch âm dương rồi hai
 * mục "Tiết khí" và "Lịch âm" — không có bàn Kỳ Môn, không có thanh tab, không
 * có nút ghim. Bố cục, màu sắc và CẢ TRẠNG THÁI GẬP/MỞ đều lấy theo tab Lịch.
 *
 * Widget này DÙNG ĐƯỢC chứ không chỉ để nhìn:
 *
 *   • hai mục CUỘN được bằng ngón tay — mỗi mục là một ListView do
 *     `WidgetSectionService` nuôi, giữ đủ 24 và 13 hàng;
 *   • chạm vào một ngày là CHỌN ngày ấy ngay trên màn hình chính: ô được viền
 *     màu nhấn, và hai mục tô lại hàng ứng với ngày đó rồi cuộn tới hàng ấy.
 *
 * Không chỗ nào trong widget mở ứng dụng nữa — mở bằng biểu tượng như mọi ứng
 * dụng khác. Trước đây chạm vào lưới là nhảy vào tab Lịch, nên không thể vừa
 * chạm vừa ở lại màn hình chính.
 *
 * Vì sao lưới vẫn là bitmap: dựng 42 ô bằng RemoteViews thì mỗi ô phải là ba
 * TextView lồng nhau, mà chiều cao ô lại không đặt được theo chiều cao widget
 * trước API 31 (`setViewLayoutHeight`), trong khi minSdk là 24. Bitmap thì vẽ
 * bao nhiêu cũng được; phần bắt chạm giao cho một lưới 6×7 View trong suốt nằm
 * đè lên nó (xem widget_calendar.xml).
 */
class CalendarWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(context: Context, manager: AppWidgetManager, ids: IntArray) {
        ids.forEach { render(context, manager, it) }
        scheduleMidnight(context)
    }

    override fun onAppWidgetOptionsChanged(
        context: Context, manager: AppWidgetManager, id: Int, newOptions: Bundle
    ) = render(context, manager, id)

    override fun onDeleted(context: Context, ids: IntArray) {
        ids.forEach { WidgetPrefs.clearWidget(context, it) }
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        when (intent.action) {
            ACTION_REFRESH -> refreshAll(context)
            ACTION_SEC -> {
                val key = intent.getStringExtra(EXTRA_SEC_KEY)
                if (key != WidgetPrefs.SEC_JQ) return
                WidgetPrefs.toggleSec(context, key)
                // Gập/mở là trạng thái CHUNG của ứng dụng chứ không của riêng
                // một widget (tab Lịch đọc cùng khoá ấy), nên vẽ lại tất cả.
                refreshAll(context)
            }
            ACTION_PREV, ACTION_NEXT, ACTION_TODAY, ACTION_PICK -> {
                val id = intent.getIntExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, 0)
                if (id == 0) return
                when (intent.action) {
                    ACTION_PREV ->
                        WidgetPrefs.setOffset(context, id, WidgetPrefs.offsetOf(context, id) - 1)
                    ACTION_NEXT ->
                        WidgetPrefs.setOffset(context, id, WidgetPrefs.offsetOf(context, id) + 1)
                    // Về tháng này thì bỏ luôn ngày đang chọn: quay lại đúng
                    // trạng thái ban đầu, hai mục tô theo hôm nay.
                    ACTION_TODAY -> {
                        WidgetPrefs.setOffset(context, id, 0)
                        WidgetPrefs.setSelected(context, id, 0)
                    }
                    ACTION_PICK -> {
                        val jdn = intent.getIntExtra(EXTRA_JDN, 0)
                        if (jdn <= 0) return
                        // Chạm lại đúng ngày đang chọn thì BỎ chọn — không thì
                        // không có đường nào quay về trạng thái "theo hôm nay".
                        val cur = WidgetPrefs.selectedJdn(context, id)
                        WidgetPrefs.setSelected(context, id, if (cur == jdn) 0 else jdn)
                    }
                }
                render(context, AppWidgetManager.getInstance(context), id)
            }
        }
    }

    override fun onEnabled(context: Context) = scheduleMidnight(context)

    override fun onDisabled(context: Context) {
        alarm(context)?.let { (am, pi) -> am.cancel(pi) }
    }

    private fun refreshAll(context: Context) {
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(
            ComponentName(context, CalendarWidgetProvider::class.java)
        )
        ids.forEach { render(context, manager, it) }
        // Đặt lại báo thức nửa đêm ở MỌI lần vẽ lại, không riêng onUpdate. Báo
        // thức là đường duy nhất widget tự biết đã sang ngày mới
        // (updatePeriodMillis = 0), mà nó thì mất khi khởi động lại máy hoặc khi
        // hệ thống dọn tiến trình. Đặt lại cùng một PendingIntent chỉ là thay
        // chỗ cũ, không chồng thêm báo thức nào.
        if (ids.isNotEmpty()) scheduleMidnight(context)
    }

    /* ─────────────── Dựng widget ─────────────── */

    private fun render(context: Context, manager: AppWidgetManager, id: Int) {
        LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)

        val opts = manager.getAppWidgetOptions(id)
        // Ở chế độ DỌC, bề ngang là MIN_WIDTH còn chiều cao là MAX_HEIGHT.
        // Lấy nhầm MIN_HEIGHT (chiều cao khi xoay NGANG, thấp hơn hẳn) thì
        // bitmap lùn hơn widget thật và mọi thứ trông sai tỉ lệ.
        val wDp = opts.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0)
            .takeIf { it > 0 } ?: 320
        val hDp = opts.getInt(AppWidgetManager.OPTION_APPWIDGET_MAX_HEIGHT, 0)
            .takeIf { it > 0 }
            ?: opts.getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_HEIGHT, 0)
                .takeIf { it > 0 } ?: 260

        val (year, month) = WidgetPrefs.viewMonth(context, id)
        val zh = WidgetPrefs.isZh(context)
        val todayJdn = WidgetPrefs.todayJdn()
        val selJdn = WidgetPrefs.selectedJdn(context, id)
        val secs = WidgetSections.build(context, year, month, WidgetPrefs.anchorJdn(context, id))
        val jqOpen = secs.firstOrNull { it.key == WidgetSections.JQ }?.open ?: false

        val views = RemoteViews(context.packageName, R.layout.widget_calendar)
        views.setTextViewText(
            R.id.widgetTitle,
            if (zh) "农历 ${year}年${month}月" else "LỊCH ÂM THÁNG $month/$year"
        )

        // hàng thứ
        val dows = if (zh) arrayOf("一", "二", "三", "四", "五", "六", "日")
                   else arrayOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
        for (i in 0..6) {
            views.setTextViewText(DOW_IDS[i], dows[i])
            views.setTextColor(
                DOW_IDS[i],
                if (i == 6) Color.parseColor("#C62828") else Color.parseColor("#7A2B33")
            )
        }

        // lưới lịch — bitmap vẽ vừa đúng khung mà layout_weight chừa cho nó
        val gridDp = WidgetLayout.gridHeightDp(hDp, jqOpen)
        views.setImageViewBitmap(
            R.id.widgetImage,
            drawGrid(context, wDp, gridDp, year, month, todayJdn, selJdn)
        )

        // chạm từng ngày
        val startJdn = firstCellJdn(year, month)
        for (idx in CELL_IDS.indices) {
            views.setOnClickPendingIntent(
                CELL_IDS[idx], pickIntent(context, id, idx, startJdn + idx)
            )
        }

        // hai mục — duyệt theo DANH SÁCH CỐ ĐỊNH, không duyệt theo `secs`.
        //
        // WidgetSections.build() bỏ hẳn một mục nếu lần này không dựng được
        // (bảng tra thiếu dữ liệu cho năm đang xem). Duyệt theo `secs` thì mục
        // ấy không được đụng tới một lần nào: ListView của nó giữ nguyên trạng
        // thái mặc định của XML — ĐANG HIỆN, KHÔNG có adapter — nên hiện ra
        // thành một mảng trắng chiếm đúng phần chiều cao của mình mà chẳng bao
        // giờ có hàng nào, còn hàng tiêu đề thì vẫn đủ chữ. Nhìn y như "mục này
        // mở không ra".
        for (sv in SECTION_VIEWS) {
            val sec = secs.firstOrNull { it.key == sv.key }
            val open = sec != null && sec.open
            views.setViewVisibility(sv.headId, if (sec == null) View.GONE else View.VISIBLE)
            views.setViewVisibility(sv.listId, if (open) View.VISIBLE else View.GONE)
            if (sec == null) continue
            for (i in 0..2) {
                // Dấu ▾/▸ ở cột đầu: hàng tiêu đề bấm được thì phải có dấu cho
                // biết, và cột đầu là cột căn trái duy nhất nên thêm vào đây
                // không đẩy hai cột kia lệch tâm.
                views.setTextViewText(
                    sv.headIds[i],
                    if (i == 0) sec.heads[0] + (if (open) "  ▾" else "  ▸") else sec.heads[i]
                )
                views.setTextColor(sv.headIds[i], sec.headFg)
                // Cùng luật co chữ với hàng giá trị, không thì tiêu đề to hơn
                // hẳn phần dưới trên widget bóp hẹp.
                views.setTextViewTextSize(
                    sv.headIds[i], TypedValue.COMPLEX_UNIT_SP, WidgetLayout.rowTextSp(wDp)
                )
            }
            views.setOnClickPendingIntent(sv.headId, secToggleIntent(context, id, sv.prefKey))
            views.setRemoteAdapter(sv.listId, sectionIntent(context, id, sv.key))
            // Cuộn tới hàng đang hiệu lực, y như tab Lịch tự cuộn tới tiết khí
            // của hôm nay khi mở mục.
            if (open && sec.active > 0) views.setScrollPosition(sv.listId, sec.active)
        }

        views.setOnClickPendingIntent(R.id.widgetPrev, navIntent(context, id, ACTION_PREV))
        views.setOnClickPendingIntent(R.id.widgetNext, navIntent(context, id, ACTION_NEXT))
        // Chạm tiêu đề: về tháng hiện tại, giống chạm tiêu đề trong ứng dụng.
        views.setOnClickPendingIntent(R.id.widgetTitle, navIntent(context, id, ACTION_TODAY))

        manager.updateAppWidget(id, views)
        // Bảo hai factory đọc lại: tháng, ngày đang chọn, ngôn ngữ hay địa điểm
        // đều có thể vừa đổi. Phải gọi SAU updateAppWidget.
        manager.notifyAppWidgetViewDataChanged(id, R.id.jqList)
    }

    /** Ngày Julius của ô ĐẦU TIÊN trong lưới (thường thuộc tháng trước). */
    private fun firstCellJdn(year: Int, month: Int): Int {
        val first = Calendar.getInstance().apply { set(year, month - 1, 1) }
        val lead = (first.get(Calendar.DAY_OF_WEEK) + 5) % 7   // tuần bắt đầu Thứ 2
        return LunarTable.jdn(year, month, 1) - lead
    }

    private fun navIntent(context: Context, id: Int, action: String): PendingIntent {
        val intent = Intent(context, CalendarWidgetProvider::class.java)
            .setAction(action)
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
        // requestCode phải khác nhau cho từng widget và từng nút, nếu không hệ
        // thống dùng lại cùng một PendingIntent và mọi nút cùng làm một việc.
        val code = id * CODES_PER_WIDGET + when (action) {
            ACTION_PREV -> 1
            ACTION_NEXT -> 2
            else -> 3
        }
        return PendingIntent.getBroadcast(
            context, code, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /**
     * PendingIntent cho MỘT ô ngày. Ngoài requestCode riêng còn đặt cả `data`
     * riêng: hệ thống so intent bằng `filterEquals`, vốn BỎ QUA extras — hai ô
     * chỉ khác extras thì có thể bị gộp làm một.
     */
    private fun pickIntent(context: Context, id: Int, idx: Int, jdn: Int): PendingIntent {
        val intent = Intent(context, CalendarWidgetProvider::class.java)
            .setAction(ACTION_PICK)
            .setData(Uri.parse("qmdj://widget/$id/cell/$idx"))
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
            .putExtra(EXTRA_JDN, jdn)
        return PendingIntent.getBroadcast(
            context, id * CODES_PER_WIDGET + CELL_CODE_BASE + idx, intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /**
     * PendingIntent cho hàng tiêu đề của một mục: bấm là gập/mở.
     *
     * `data` riêng cho từng widget và từng mục vì `filterEquals` bỏ qua extras —
     * để trong extras không thôi thì hai hàng tiêu đề dùng chung một
     * PendingIntent và bấm cái nào cũng lật cùng một mục.
     */
    private fun secToggleIntent(context: Context, id: Int, key: String): PendingIntent {
        val intent = Intent(context, CalendarWidgetProvider::class.java)
            .setAction(ACTION_SEC)
            .setData(Uri.parse("qmdj://widget/$id/toggle/$key"))
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
            .putExtra(EXTRA_SEC_KEY, key)
        return PendingIntent.getBroadcast(
            context,
            id * CODES_PER_WIDGET + SEC_TOGGLE_CODE,
            intent,
            PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
        )
    }

    /**
     * Intent trỏ tới nguồn hàng của MỘT mục. Khoá mục nằm trong `data` chứ
     * không chỉ trong extras, vì `filterEquals` bỏ qua extras: để trong extras
     * thì hai ListView dùng chung một factory và cùng hiện một bảng.
     */
    private fun sectionIntent(context: Context, id: Int, key: String): Intent =
        Intent(context, WidgetSectionService::class.java)
            .setData(Uri.parse("qmdj://widget/$id/sec/$key"))
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
            .putExtra(WidgetSectionService.EXTRA_SECTION, key)

    private fun dp(context: Context, v: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v, context.resources.displayMetrics
    )

    /**
     * Vẽ lưới lịch — CHỈ lưới, không còn hàng thứ lẫn hai bảng (nay đều là View
     * thật). Màu sắc và cách sắp xếp lấy đúng theo tab Lịch: hôm nay là viền đỏ
     * đậm trên nền sáng, ngày đang chọn là viền màu nhấn trên nền trắng, ngày
     * của tháng trước/sau tô mờ, can một dòng chi một dòng.
     *
     * LUÔN sáu hàng, kể cả tháng chỉ cần bốn hay năm: lưới bắt chạm đè lên trên
     * là 6×7 View cố định trong XML, nên số hàng của bitmap mà đổi theo tháng là
     * ô chạm lệch hẳn khỏi ô nhìn thấy. Sáu hàng cũng có nghĩa chiều cao ô không
     * nhảy khi lật tháng.
     */
    private fun drawGrid(
        context: Context, wDp: Int, hDp: Float, year: Int, month: Int,
        todayJdn: Int, selJdn: Int
    ): Bitmap {
        val w = dp(context, wDp.toFloat()).toInt().coerceIn(240, 2400)
        val h = dp(context, hDp).toInt().coerceIn(60, 2400)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        val tz = WidgetPrefs.timeZone(context)
        val zh = WidgetPrefs.isZh(context)
        val cellW = w / 7f
        val cellH = h / GRID_WEEKS.toFloat()

        paint.color = Color.parseColor("#FDECEF")
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), paint)

        // Cỡ chữ chặn theo dp tuyệt đối: thả trôi theo chiều cao widget thì
        // widget cao một chút là chữ phình, widget thấp là chữ bé không đọc nổi.
        val dayPx = minOf(cellH * 0.36f, dp(context, 17f))
        val lunPx = minOf(cellH * 0.27f, dp(context, 11.5f))
        val gzPx = minOf(cellH * 0.25f, dp(context, 11f))
        val showGanZhi = cellH >= dayPx + gzPx * 2 + dp(context, 6f)

        val startJdn = firstCellJdn(year, month)
        for (idx in 0 until GRID_WEEKS * 7) {
            val jdn = startJdn + idx
            val (cy, cm, cd) = LunarTable.civilOf(jdn)
            val outside = cm != month || cy != year
            val x = cellW * (idx % 7)
            val y = cellH * (idx / 7)
            val isToday = jdn == todayJdn
            val isSel = selJdn > 0 && jdn == selJdn

            if (isToday) {
                // Viền đỏ đậm + nền sáng, giống hệt tab Lịch.
                paint.style = Paint.Style.FILL
                paint.color = Color.parseColor("#FFFBEA")
                c.drawRect(x + 1, y + 1, x + cellW - 1, y + cellH - 1, paint)
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = dp(context, 1.6f)
                paint.color = Color.parseColor("#D32F2F")
                c.drawRect(x + 2, y + 2, x + cellW - 2, y + cellH - 2, paint)
                paint.style = Paint.Style.FILL
            } else if (isSel) {
                // Ngày đang chọn: nền trắng, viền màu nhấn — đúng
                // `.cal-sel:not(.cal-today)` bên CSS.
                paint.style = Paint.Style.FILL
                paint.color = Color.WHITE
                c.drawRect(x + 1, y + 1, x + cellW - 1, y + cellH - 1, paint)
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = dp(context, 1.8f)
                paint.color = Color.parseColor("#007BFF")
                c.drawRect(x + 2, y + 2, x + cellW - 2, y + cellH - 2, paint)
                paint.style = Paint.Style.FILL
            } else if (outside) {
                paint.color = Color.parseColor("#FDF5F6")
                c.drawRect(x + 1, y + 1, x + cellW - 1, y + cellH - 1, paint)
            }

            val fg = when {
                isToday -> Color.parseColor("#C62828")
                outside -> Color.parseColor("#B9A3A7")
                else -> Color.parseColor("#222222")
            }
            val dim = when {
                isToday -> Color.parseColor("#A06B30")
                outside -> Color.parseColor("#C2AEB2")
                else -> Color.parseColor("#6B6B6B")
            }

            paint.typeface = Typeface.DEFAULT_BOLD
            paint.textAlign = Paint.Align.LEFT
            paint.textSize = dayPx
            paint.color = fg
            val dayBase = y + dayPx + dp(context, 3f)
            c.drawText(cd.toString(), x + cellW * 0.09f, dayBase, paint)

            val lunar = LunarTable.lunarOf(jdn, tz)
            paint.typeface = Typeface.DEFAULT
            paint.textAlign = Paint.Align.RIGHT
            paint.textSize = lunPx
            paint.color = dim
            val lunarTxt = when {
                lunar == null -> ""
                lunar.day == 1 -> "${lunar.day}/${lunar.month}"
                else -> lunar.day.toString()
            }
            c.drawText(lunarTxt, x + cellW * 0.93f, dayBase, paint)

            if (showGanZhi) {
                val (can, chi) = LunarTable.ganZhiOf(jdn, zh)
                paint.textAlign = Paint.Align.CENTER
                paint.textSize = gzPx
                paint.color = if (isToday) Color.parseColor("#7A4A1C") else dim
                if (isToday) paint.typeface = Typeface.DEFAULT_BOLD
                // Can chi là một KHỐI hai dòng sát nhau, đặt cân giữa phần còn
                // lại của ô — đúng như `.cal-gz` trong tab Lịch (flex, căn giữa,
                // line-height 1,25). Đặt theo tỉ lệ 40%/84% của phần còn lại thì
                // ô càng cao hai chữ càng dạt xa nhau: ở widget 4×5 khoảng cách
                // giãn ra hơn gấp đôi cỡ chữ.
                val rest = cellH - (dayBase - y)
                val lineGap = gzPx * 1.28f
                val top = dayBase + (rest - (lineGap + gzPx)) / 2f
                val base1 = top + gzPx * 0.75f
                c.drawText(can, x + cellW / 2f, base1, paint)
                c.drawText(chi, x + cellW / 2f, base1 + lineGap, paint)
                paint.typeface = Typeface.DEFAULT
            }
        }

        paint.color = Color.parseColor("#F3C6CD")
        paint.strokeWidth = 1f
        paint.style = Paint.Style.STROKE
        for (i in 1..6) c.drawLine(cellW * i, 0f, cellW * i, h.toFloat(), paint)
        for (r in 0..GRID_WEEKS) {
            val y = cellH * r
            c.drawLine(0f, y, w.toFloat(), y, paint)
        }
        paint.style = Paint.Style.FILL
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
     * Hẹn giờ vẽ lại ngay sau nửa đêm. Báo thức lặp không chính xác — chỉ cần
     * đúng ngày, đỡ tốn pin hơn hẳn so với đánh thức nửa tiếng một lần.
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
        am.setInexactRepeating(AlarmManager.RTC, next.timeInMillis, AlarmManager.INTERVAL_DAY, pi)
    }

    companion object {
        const val ACTION_REFRESH = "com.bazi.qimen.WIDGET_REFRESH"
        const val ACTION_PREV = "com.bazi.qimen.WIDGET_PREV"
        const val ACTION_NEXT = "com.bazi.qimen.WIDGET_NEXT"
        const val ACTION_TODAY = "com.bazi.qimen.WIDGET_TODAY"
        const val ACTION_PICK = "com.bazi.qimen.WIDGET_PICK"
        /** Chạm hàng tiêu đề một mục: gập/mở đúng như trong tab Lịch. */
        const val ACTION_SEC = "com.bazi.qimen.WIDGET_SEC"

        private const val EXTRA_JDN = "com.bazi.qimen.JDN"
        private const val EXTRA_SEC_KEY = "com.bazi.qimen.SEC_KEY"

        /**
         * Lưới LUÔN sáu hàng — số hàng của tháng dài nhất — nên lưới bắt chạm
         * 6×7 trong XML lúc nào cũng đè đúng ô, và chiều cao ô không nhảy khi
         * lật tháng.
         */
        private const val GRID_WEEKS = 6

        /**
         * Mỗi widget chiếm ngần này requestCode: 1–3 cho ‹ › và tiêu đề, rồi 42
         * ô ngày kể từ CELL_CODE_BASE. Trùng mã giữa hai widget là hai cái cạnh
         * nhau cùng nhảy tháng với nhau.
         */
        private const val CODES_PER_WIDGET = 64
        private const val SEC_TOGGLE_CODE = 4
        private const val CELL_CODE_BASE = 8

        private val DOW_IDS = intArrayOf(
            R.id.dow0, R.id.dow1, R.id.dow2, R.id.dow3, R.id.dow4, R.id.dow5, R.id.dow6
        )
        private val JQ_HEAD_IDS = intArrayOf(R.id.jqHead0, R.id.jqHead1, R.id.jqHead2)

        /** Mọi id của MỘT mục, để vòng vẽ không phải rẽ nhánh theo khoá. */
        private class SecViews(
            val key: String, val prefKey: String,
            val headId: Int, val headIds: IntArray, val listId: Int,
        )

        private val SECTION_VIEWS = listOf(
            SecViews(WidgetSections.JQ, WidgetPrefs.SEC_JQ, R.id.jqHead, JQ_HEAD_IDS, R.id.jqList),
        )

        /** 42 ô của lưới bắt chạm, theo thứ tự đọc (hàng rồi cột). */
        private val CELL_IDS = intArrayOf(
            R.id.cell00, R.id.cell01, R.id.cell02, R.id.cell03, R.id.cell04, R.id.cell05, R.id.cell06,
            R.id.cell10, R.id.cell11, R.id.cell12, R.id.cell13, R.id.cell14, R.id.cell15, R.id.cell16,
            R.id.cell20, R.id.cell21, R.id.cell22, R.id.cell23, R.id.cell24, R.id.cell25, R.id.cell26,
            R.id.cell30, R.id.cell31, R.id.cell32, R.id.cell33, R.id.cell34, R.id.cell35, R.id.cell36,
            R.id.cell40, R.id.cell41, R.id.cell42, R.id.cell43, R.id.cell44, R.id.cell45, R.id.cell46,
            R.id.cell50, R.id.cell51, R.id.cell52, R.id.cell53, R.id.cell54, R.id.cell55, R.id.cell56
        )

        /** Máy này có cho ghim widget bằng một cú chạm không? */
        fun canPin(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            return manager.isRequestPinAppWidgetSupported
        }

        /**
         * Vẽ lại ngay mọi widget đang ghim — ứng dụng gọi sau khi đổi ngôn ngữ,
         * đổi địa điểm hoặc gập/mở một mục (xem WebAppBridge.refreshCalendarWidget).
         */
        fun refreshNow(context: Context) {
            context.sendBroadcast(
                Intent(context, CalendarWidgetProvider::class.java).setAction(ACTION_REFRESH)
            )
        }

        /**
         * Dựng lại báo thức nửa đêm RỒI vẽ lại ngay — dùng cho BootReceiver.
         *
         * Vẽ lại trước hết là để bắt kịp những ngày đã trôi qua trong lúc không
         * có báo thức nào (máy tắt, ứng dụng vừa cập nhật); `refreshAll` đặt
         * lại báo thức, nên từ đó trở đi nó tự theo kịp.
         */
        fun reviveNow(context: Context) = refreshNow(context)

        fun requestPin(context: Context): Boolean {
            if (!canPin(context)) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            val provider = ComponentName(context, CalendarWidgetProvider::class.java)
            return manager.requestPinAppWidget(provider, null, null)
        }
    }
}
