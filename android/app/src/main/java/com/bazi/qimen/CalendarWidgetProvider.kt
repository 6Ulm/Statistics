package com.bazi.qimen

import android.app.AlarmManager
import android.app.PendingIntent
import android.appwidget.AppWidgetManager
import android.appwidget.AppWidgetProvider
import android.content.ComponentName
import android.content.Context
import android.content.Intent
import android.content.res.Resources
import android.graphics.Bitmap
import android.graphics.Canvas
import android.graphics.Color
import android.graphics.Paint
import android.graphics.Typeface
import android.os.Build
import android.os.Bundle
import android.util.TypedValue
import android.widget.RemoteViews
import java.util.Calendar
import java.util.TimeZone
import org.json.JSONException
import org.json.JSONObject
import kotlin.math.sqrt

/**
 * Widget màn hình chính: CHỈ những gì tab Lịch có — lưới lịch âm dương rồi hai
 * mục gập được "Tiết khí" và "Lịch âm" — không có bàn Kỳ Môn, không có thanh
 * tab, không có nút ghim. Bố cục, màu sắc và CẢ TRẠNG THÁI GẬP/MỞ đều lấy theo
 * tab Lịch, nên hai chỗ không thể hiện hai thứ khác nhau.
 *
 * Thanh tiêu đề là View thật (xem widget_calendar.xml) để hai mũi tên ‹ › bấm
 * được mà lùi/tiến tháng; phần lưới và hai bảng bên dưới vẽ ra bitmap vì
 * RemoteViews không dựng nổi lưới 7×6 cho gọn.
 *
 * Home-screen widget: the lunar calendar and its jieqi table, nothing else.
 */
class CalendarWidgetProvider : AppWidgetProvider() {

    override fun onUpdate(context: Context, manager: AppWidgetManager, ids: IntArray) {
        LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)
        ids.forEach { render(context, manager, it) }
        scheduleMidnight(context)
    }

    override fun onAppWidgetOptionsChanged(
        context: Context, manager: AppWidgetManager, id: Int, newOptions: Bundle
    ) {
        LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)
        render(context, manager, id)
    }

    override fun onDeleted(context: Context, ids: IntArray) {
        val e = prefs(context).edit()
        ids.forEach { e.remove(offsetKey(it)) }
        e.apply()
    }

    override fun onReceive(context: Context, intent: Intent) {
        super.onReceive(context, intent)
        when (intent.action) {
            ACTION_REFRESH -> refreshAll(context)
            ACTION_PREV, ACTION_NEXT, ACTION_TODAY -> {
                val id = intent.getIntExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, 0)
                if (id == 0) return
                LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)
                val cur = prefs(context).getInt(offsetKey(id), 0)
                val next = when (intent.action) {
                    ACTION_PREV -> cur - 1
                    ACTION_NEXT -> cur + 1
                    else -> 0
                }
                prefs(context).edit().putInt(offsetKey(id), next.coerceIn(-1200, 1200)).apply()
                render(context, AppWidgetManager.getInstance(context), id)
            }
        }
    }

    override fun onEnabled(context: Context) = scheduleMidnight(context)

    override fun onDisabled(context: Context) {
        alarm(context)?.let { (am, pi) -> am.cancel(pi) }
    }

    private fun refreshAll(context: Context) {
        LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)
        val manager = AppWidgetManager.getInstance(context)
        val ids = manager.getAppWidgetIds(
            ComponentName(context, CalendarWidgetProvider::class.java)
        )
        ids.forEach { render(context, manager, it) }
        // Đặt lại báo thức nửa đêm ở MỌI lần vẽ lại, không riêng onUpdate.
        // Báo thức là đường duy nhất widget tự biết đã sang ngày mới
        // (updatePeriodMillis = 0), mà nó thì mất khi khởi động lại máy hoặc khi
        // hệ thống dọn tiến trình. Đặt lại cùng một PendingIntent chỉ là thay
        // chỗ cũ, không chồng thêm báo thức nào.
        if (ids.isNotEmpty()) scheduleMidnight(context)
    }

    /* ─────────────── Dựng widget ─────────────── */

    private fun render(context: Context, manager: AppWidgetManager, id: Int) {
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

        val offset = prefs(context).getInt(offsetKey(id), 0)
        val cal = Calendar.getInstance().apply {
            set(Calendar.DAY_OF_MONTH, 1)
            add(Calendar.MONTH, offset)
        }
        val year = cal.get(Calendar.YEAR)
        val month = cal.get(Calendar.MONTH) + 1

        val zh = LunarTable.langOf(context) == "zh"
        val views = RemoteViews(context.packageName, R.layout.widget_calendar)
        views.setTextViewText(
            R.id.widgetTitle,
            if (zh) "农历 ${year}年${month}月" else "LỊCH ÂM THÁNG $month/$year"
        )
        views.setImageViewBitmap(
            R.id.widgetImage,
            drawBody(context, wDp, (hDp - HEADER_DP).coerceAtLeast(90), year, month)
        )

        views.setOnClickPendingIntent(R.id.widgetPrev, navIntent(context, id, ACTION_PREV))
        views.setOnClickPendingIntent(R.id.widgetNext, navIntent(context, id, ACTION_NEXT))
        // Chạm tiêu đề: về tháng hiện tại, giống chạm tiêu đề trong ứng dụng.
        views.setOnClickPendingIntent(R.id.widgetTitle, navIntent(context, id, ACTION_TODAY))
        // Chạm vào lưới: mở ứng dụng ở tab Lịch.
        val open = Intent(context, MainActivity::class.java).apply {
            action = Intent.ACTION_MAIN
            putExtra(MainActivity.EXTRA_TAB, "cal")
            flags = Intent.FLAG_ACTIVITY_NEW_TASK or Intent.FLAG_ACTIVITY_CLEAR_TOP
        }
        views.setOnClickPendingIntent(
            R.id.widgetImage,
            PendingIntent.getActivity(
                context, 0, open,
                PendingIntent.FLAG_UPDATE_CURRENT or PendingIntent.FLAG_IMMUTABLE
            )
        )
        manager.updateAppWidget(id, views)
    }

    private fun navIntent(context: Context, id: Int, action: String): PendingIntent {
        val intent = Intent(context, CalendarWidgetProvider::class.java)
            .setAction(action)
            .putExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, id)
        // requestCode phải khác nhau cho từng widget và từng nút, nếu không hệ
        // thống dùng lại cùng một PendingIntent và mọi nút cùng làm một việc.
        val code = id * 8 + when (action) {
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
     * Múi giờ của địa điểm người dùng đã chọn trong ứng dụng. Ứng dụng ghi cả
     * cụm vị trí thành JSON dưới khoá `qmdj.location` (xem location.js).
     *
     * Chưa có khoá ấy thì dùng múi giờ CỦA MÁY — location.js cũng suy về múi
     * giờ máy khi mở app lần đầu mà chưa có vị trí đã lưu lẫn không mục nào
     * trong danh sách khớp múi giờ máy, nên đây là dự đoán gần nhất với cái
     * app đang thực sự dùng. Trước đây chốt cứng giờ Việt Nam — hợp lý với
     * CƠ SỞ LƯU TRỮ của lunar_months.txt/jieqi.txt (UTC+7), nhưng ứng dụng lại
     * không hề mặc định vào Việt Nam, nên widget cứ lệch giờ tiết khí ngay từ
     * lần mở app đầu tiên, trước khi ai kịp chọn địa điểm.
     */
    private fun selectedTimeZone(context: Context): TimeZone {
        val raw = context.getSharedPreferences("qmdj_prefs", Context.MODE_PRIVATE)
            .getString("qmdj.location", null) ?: return TimeZone.getDefault()
        val id = try {
            JSONObject(raw).optString("tzId", "")
        } catch (e: JSONException) {
            ""
        }
        if (id.isEmpty()) return TimeZone.getDefault()
        val tz = TimeZone.getTimeZone(id)
        // getTimeZone() trả về GMT cho id lạ thay vì báo lỗi — bắt lại ở đây,
        // không thì một id hỏng lặng lẽ đẩy mọi mốc về UTC.
        return if (tz.id == "GMT" && id != "GMT" && id != "UTC") {
            TimeZone.getDefault()
        } else tz
    }

    private fun dp(context: Context, v: Float): Float = TypedValue.applyDimension(
        TypedValue.COMPLEX_UNIT_DIP, v, context.resources.displayMetrics
    )

    /**
     * Bán kính góc bo thực tế của widget, tính bằng pixel.
     *
     * Android 12 trở lên TỰ bo góc mọi widget theo
     * `system_app_widget_background_radius` (One UI để khá rộng), không cần hỏi
     * ý ứng dụng; dưới mức đó thì góc là của `widget_bg.xml`, 16dp. Chặn trên
     * 32dp để một giá trị lạ của máy nào đó không nuốt mất cả hàng.
     */
    private fun cornerRadius(context: Context): Float {
        val sys = if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.S) {
            try {
                context.resources.getDimension(
                    android.R.dimen.system_app_widget_background_radius
                )
            } catch (e: Resources.NotFoundException) {
                0f
            }
        } else 0f
        return maxOf(sys, dp(context, 16f)).coerceAtMost(dp(context, 32f))
    }

    /**
     * Khoảng trắng phải chừa dưới hàng cuối cho góc bo — KHÔNG phải cả bán kính.
     *
     * Cung tròn chỉ ăn sâu nhất ở sát mép; chữ thì bắt đầu cách mép `inset`, mà
     * ở hoành độ ấy cung mới xuống tới `r − √(2r·inset − inset²)`. Với góc 16dp
     * và chữ bắt đầu ở 4dp thì chỉ 5,4dp chứ không phải 16dp — chừa cả bán kính
     * là hở ra một dải trắng to vô cớ dưới đáy bảng.
     *
     * Cộng thêm chút lề an toàn cho nét chữ và cho việc làm tròn pixel.
     */
    private fun cornerBottomPad(context: Context, radius: Float, inset: Float): Float {
        val margin = dp(context, 1.5f)
        if (inset >= radius) return margin
        val depth = radius - sqrt(2f * radius * inset - inset * inset)
        return depth + margin
    }

    /**
     * Vẽ lưới lịch + bảng tiết khí. Màu sắc và cách sắp xếp lấy đúng theo tab
     * Lịch: hôm nay là viền đỏ đậm trên nền sáng, ngày của tháng trước/sau tô
     * mờ, can một dòng chi một dòng.
     */
    private fun drawBody(
        context: Context, wDp: Int, hDp: Int, year: Int, month: Int
    ): Bitmap {
        val w = dp(context, wDp.toFloat()).toInt().coerceIn(240, 2400)
        val h = dp(context, hDp.toFloat()).toInt().coerceIn(90, 2400)
        val bmp = Bitmap.createBitmap(w, h, Bitmap.Config.ARGB_8888)
        val c = Canvas(bmp)
        val paint = Paint(Paint.ANTI_ALIAS_FLAG)

        val now = Calendar.getInstance()
        val todayJdn = LunarTable.jdn(
            now.get(Calendar.YEAR), now.get(Calendar.MONTH) + 1, now.get(Calendar.DAY_OF_MONTH)
        )

        val first = Calendar.getInstance().apply { set(year, month - 1, 1) }
        val lead = (first.get(Calendar.DAY_OF_WEEK) + 5) % 7   // tuần bắt đầu Thứ 2
        val daysInMonth = first.getActualMaximum(Calendar.DAY_OF_MONTH)
        val weeks = Math.ceil((lead + daysInMonth) / 7.0).toInt().coerceAtLeast(1)

        // Hai mục gập được, y như tab Lịch: "Tiết khí" (ba cột — tên · Dương
        // lịch · can chi tháng) rồi "Lịch âm" (Tháng âm · Sóc · Vọng). Hàng
        // tiêu đề của CẢ HAI luôn hiện; mục nào đang mở thì có thêm mấy hàng
        // giá trị bên dưới, đúng trạng thái người dùng để lại trong ứng dụng.
        //
        // Mười ba hàng là một khối lớn, nên chia theo TỈ LỆ của ứng dụng thay vì
        // cho nó một khoản cố định: ở tab Lịch trên A51, hàng lịch cao ~78dp còn
        // hàng bảng ~21dp, tức xấp xỉ 3,7 lần.
        // Giờ giao tiết và mốc Sóc/Vọng hiện theo múi giờ của ĐỊA ĐIỂM ĐANG
        // CHỌN trong ứng dụng, giống hệt tab Lịch — widget không có bảng chọn
        // nơi riêng.
        val tz = selectedTimeZone(context)
        val zh = LunarTable.langOf(context) == "zh"
        val secs = buildSections(context, year, month, todayJdn, tz, zh)
        val dowH = minOf(h * 0.09f, dp(context, 15f))
        val hasTable = secs.isNotEmpty()

        // Đáy widget bị góc bo cắt mất một cung tròn. Bảng chạm sát mép thì hàng
        // cuối mất chữ đầu và mất đuôi giờ. Chừa đúng chỗ cung ăn tới Ở HOÀNH ĐỘ
        // CHỮ BẮT ĐẦU — lấy lề hẹp nhất có thể xảy ra làm trường hợp xấu nhất,
        // vì lề thật chỉ rộng hơn.
        val safeBottom = if (!hasTable) 0f else
            cornerBottomPad(context, cornerRadius(context), dp(context, PAD_NARROWEST))

        // Khối bảng phải CỐ ĐỊNH: chia theo GRID_WEEKS (tháng dài nhất) và theo
        // NOMINAL_ROWS chứ không theo `weeks` của tháng đang xem hay theo số
        // hàng thật sự vẽ ra. Chia theo tháng đang xem thì tháng gọn 5 hàng làm
        // bảng phình ra ~12% — lật tháng một cái là cả khung lẫn cỡ chữ nhảy,
        // đúng thứ người dùng thấy chướng. Chỗ dôi ra của tháng 5 hàng đổ vào
        // lưới lịch, nơi ô cao thêm chỉ tốt lên.
        // 6 hàng lịch × 3,7 + 13 hàng bảng = 35,2 phần bằng nhau.
        // Kẹp hai đầu: dưới 9dp thì chữ không đọc nổi, trên 18dp thì bảng phình
        // ra nuốt mất lưới lịch trên widget cao.
        val jqRowH = if (!hasTable) 0f else
            ((h - dowH - safeBottom) / (GRID_WEEKS * ROW_RATIO + NOMINAL_ROWS))
                .coerceIn(dp(context, 9f), dp(context, 18f))

        // Bao nhiêu hàng GIÁ TRỊ thì vừa: lưới lịch giữ đúng phần của nó
        // (GRID_WEEKS hàng, mỗi hàng cao ROW_RATIO lần hàng bảng), phần còn lại
        // chia cho các mục đang mở. Trừ đi hàng tiêu đề của cả hai mục, vốn
        // luôn hiện dù mục có mở hay không.
        val gridKeep = if (!hasTable) 0f else jqRowH * ROW_RATIO * GRID_WEEKS
        val slots = if (!hasTable) 0 else
            (((h - dowH - safeBottom - gridKeep) / jqRowH).toInt() - secs.size)
                .coerceAtLeast(0)
        val shown = shareRows(secs, slots)
        val bodyRows = shown.sum()
        val jqH = if (!hasTable) 0f else
            jqRowH * (secs.size + bodyRows) + dp(context, 5f) + safeBottom
        val gridH = (h - dowH - jqH).coerceAtLeast(0f)
        val cellW = w / 7f
        val cellH = gridH / weeks

        paint.color = Color.parseColor("#FDECEF")
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), paint)

        // hàng thứ
        val dows = if (zh) arrayOf("一", "二", "三", "四", "五", "六", "日")
                   else arrayOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.CENTER
        paint.textSize = minOf(dowH * 0.62f, dp(context, 9f))
        for (i in 0..6) {
            paint.color = if (i == 6) Color.parseColor("#C62828") else Color.parseColor("#7A2B33")
            c.drawText(dows[i], cellW * (i + 0.5f), dowH * 0.70f, paint)
        }

        // Cỡ chữ chặn theo dp tuyệt đối: thả trôi theo chiều cao widget thì
        // widget cao một chút là chữ phình, widget thấp là chữ bé không đọc nổi.
        val dayPx = minOf(cellH * 0.36f, dp(context, 17f))
        val lunPx = minOf(cellH * 0.27f, dp(context, 11.5f))
        val gzPx = minOf(cellH * 0.25f, dp(context, 11f))
        val showGanZhi = cellH >= dayPx + gzPx * 2 + dp(context, 6f)

        val startJdn = LunarTable.jdn(year, month, 1) - lead
        for (idx in 0 until weeks * 7) {
            val jdn = startJdn + idx
            val (cy, cm, cd) = LunarTable.civilOf(jdn)
            val outside = cm != month || cy != year
            val x = cellW * (idx % 7)
            val y = dowH + cellH * (idx / 7)
            val isToday = jdn == todayJdn

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

        // lưới
        paint.color = Color.parseColor("#F3C6CD")
        paint.strokeWidth = 1f
        paint.style = Paint.Style.STROKE
        for (i in 1..6) c.drawLine(cellW * i, dowH, cellW * i, dowH + gridH, paint)
        for (r in 0..weeks) {
            val y = dowH + cellH * r
            c.drawLine(0f, y, w.toFloat(), y, paint)
        }
        paint.style = Paint.Style.FILL

        // ── Hai mục gập được, y như tab Lịch ──
        if (hasTable) {
            drawSections(context, c, paint, w, dowH + gridH + dp(context, 5f),
                jqRowH, safeBottom, secs, shown)
        }
        return bmp
    }

    /**
     * Một mục gập được của tab Lịch, đã dựng sẵn thành chữ: ba tiêu đề cột, các
     * hàng giá trị (mỗi hàng ba ô), hàng đang hiệu lực, và trạng thái mở/đóng.
     */
    private class Sec(
        val heads: Array<String>,
        val rows: List<Array<String>>,
        /**
         * Khuôn bề rộng của từng cột: những chuỗi RỘNG NHẤT cột ấy có thể phải
         * chứa, kể cả khi tháng đang xem không có chuỗi nào dài như thế. Không
         * có khuôn thì cột co giãn theo đúng dữ liệu của tháng đang xem, và lật
         * tháng một cái là chữ trượt ngang — "Giáp Tý" hẹp hơn "Nhâm Thân" khá
         * nhiều, mà cột can chi tháng thì đổi theo từng năm.
         */
        val samples: Array<List<String>>,
        val active: Int,
        val open: Boolean,
        val headBg: Int,
        val headFg: Int,
        val bar: Int,
    )

    /**
     * Dựng hai mục ĐÚNG NHƯ tab Lịch đang hiện — cùng cột, cùng chữ, cùng hàng
     * được tô, và cùng trạng thái gập/mở.
     *
     * Trạng thái gập/mở đọc thẳng từ kho tuỳ chọn mà calendar.js ghi
     * (`qmdj.calSecJq` / `qmdj.calSecAm`), nên đóng mục nào trong ứng dụng là
     * widget đóng đúng mục ấy. Mặc định khớp với calendar.js: Tiết khí mở, Lịch
     * âm đóng.
     */
    private fun buildSections(
        context: Context, year: Int, month: Int, todayJdn: Int, tz: TimeZone, zh: Boolean
    ): List<Sec> {
        val app = context.getSharedPreferences("qmdj_prefs", Context.MODE_PRIVATE)
        val ref = LunarTable.jdn(year, month, 15)
        val out = ArrayList<Sec>(2)

        val jieQi = LunarTable.jieQiYearOf(ref, zh).map { LunarTable.localize(it, tz) }
        if (jieQi.size == 24) {
            var active = -1
            for (i in jieQi.indices) if (jieQi[i].jdn <= todayJdn) active = i
            out.add(Sec(
                arrayOf(
                    if (zh) "节气" else context.getString(R.string.col_jieqi),
                    if (zh) "公历" else context.getString(R.string.col_solar),
                    if (zh) "月柱" else context.getString(R.string.col_ganzhi),
                ),
                jieQi.map { arrayOf(it.name, dateText(it), LunarTable.ganZhi60(it.gz, zh)) },
                arrayOf(emptyList<String>(), listOf(DATE_TEMPLATE), allGanZhi(zh)),
                active,
                app.getString(SEC_JQ, null) != "0",
                Color.parseColor("#FDEEEE"), Color.parseColor("#B71C1C"),
                Color.parseColor("#D32F2F"),
            ))
        }

        val lunarYear = LunarTable.lunarYearOf(ref, tz)
        val months = if (lunarYear == null) emptyList<LunarTable.Month>()
                     else LunarTable.monthsOfYear(lunarYear)
        if (months.isNotEmpty()) {
            // Tô tháng âm chứa HÔM NAY, và chỉ khi hôm nay còn nằm trong năm âm
            // đang hiện — lật tới năm khác thì không tô hàng nào, giống bảng
            // Tiết khí.
            val today = LunarTable.lunarOf(todayJdn, tz)
            val sameYear = LunarTable.lunarYearOf(todayJdn, tz) == lunarYear
            var active = -1
            if (today != null && sameYear) {
                for (i in months.indices) {
                    if (months[i].month == today.month && months[i].leap == today.leap) {
                        active = i; break
                    }
                }
            }
            val leap = if (zh) "闰" else context.getString(R.string.leap)
            out.add(Sec(
                arrayOf(
                    if (zh) "农历月" else context.getString(R.string.col_month),
                    if (zh) "朔" else context.getString(R.string.col_soc),
                    if (zh) "望" else context.getString(R.string.col_vong),
                ),
                months.map {
                    arrayOf(
                        monthLabel(context, it.month, it.leap, zh, leap),
                        LunarTable.stamp(it.socJdn, it.socMin, tz),
                        LunarTable.stamp(it.vongJdn, it.vongMin, tz),
                    )
                },
                arrayOf(allMonthLabels(context, zh, leap),
                        listOf(DATE_TEMPLATE), listOf(DATE_TEMPLATE)),
                active,
                app.getString(SEC_AM, null) == "1",
                Color.parseColor("#EEF1FD"), Color.parseColor("#283593"),
                Color.parseColor("#3949AB"),
            ))
        }
        return out
    }

    /** Nhãn một tháng âm, đúng chữ của tab Lịch: "Tháng 5 (Nhuận)" · "5月 (闰)". */
    private fun monthLabel(
        context: Context, month: Int, isLeap: Boolean, zh: Boolean, leap: String
    ): String =
        (if (zh) "${month}月" else context.getString(R.string.month_n, month)) +
            (if (isLeap) " ($leap)" else "")

    /**
     * Cả 60 trụ can chi, và cả 24 nhãn tháng âm có thể có — khuôn bề rộng cho
     * hai cột mà nội dung đổi theo từng năm. Đo trên toàn bộ khả năng thì cột
     * đứng yên khi lật tháng; đo trên riêng năm đang xem thì không.
     */
    private fun allGanZhi(zh: Boolean): List<String> =
        (0 until 60).map { LunarTable.ganZhi60(it, zh) }

    private fun allMonthLabels(context: Context, zh: Boolean, leap: String): List<String> {
        val out = ArrayList<String>(24)
        for (m in 1..12) {
            out.add(monthLabel(context, m, false, zh, leap))
            out.add(monthLabel(context, m, true, zh, leap))
        }
        return out
    }

    /**
     * Chia `slots` hàng giá trị cho các mục ĐANG MỞ, theo đúng tỉ lệ của tab
     * Lịch: Tiết khí lấy JQ_SHARE phần, Lịch âm lấy phần còn lại. Mục nào đóng
     * thì không lấy hàng nào; mục nào không dùng hết phần của mình thì nhường
     * lại cho mục kia thay vì bỏ trống.
     */
    private fun shareRows(secs: List<Sec>, slots: Int): IntArray {
        val out = IntArray(secs.size)
        val open = secs.indices.filter { secs[it].open }
        if (open.isEmpty() || slots <= 0) return out
        var left = slots
        if (open.size == 1) {
            out[open[0]] = minOf(secs[open[0]].rows.size, left)
            return out
        }
        val first = open[0]
        out[first] = minOf(secs[first].rows.size, maxOf(1, Math.round(slots * JQ_SHARE)))
        left -= out[first]
        for (i in 1 until open.size) {
            val k = open[i]
            out[k] = minOf(secs[k].rows.size, maxOf(0, left))
            left -= out[k]
        }
        // Còn dư thì trả lại cho mục đầu — nó là mục dài nhất (24 hàng).
        if (left > 0) out[first] = minOf(secs[first].rows.size, out[first] + left)
        return out
    }

    /**
     * Vẽ hai mục gập được — cùng hình dạng, cùng màu với tab Lịch: hàng tiêu đề
     * tô nền riêng của từng mục kèm vạch dọc bên trái, hàng chẵn tô nhạt, hàng
     * đang hiệu lực tô màu nhấn và in đậm.
     *
     * BA cột, không phải hai, và trải hết bề ngang thay vì gập đôi 12 hàng —
     * đúng bảng mà tab Lịch hiện từ lúc nó thêm cột can chi tháng. Đổi lại,
     * không còn nhét trọn 24 mục vào một màn: mục nào mở thì hiện một CỬA SỔ
     * hàng quanh hàng đang hiệu lực, y như tab Lịch tự cuộn tới hôm nay.
     *
     * Cột nằm ở CHỖ CỐ ĐỊNH, không nhích khi lật tháng: bề rộng đo trên TOÀN BỘ
     * số hàng (kể cả hàng đang không hiện) cộng khuôn ngày giờ dài nhất. Cột tên
     * căn trái, cột cuối căn giữa trong phần chừa sát mép phải, cột giữa căn
     * giữa khoảng trống còn lại — cùng luật với `.cal-jq-date` bên CSS.
     *
     * `bottomPad` là khoảng trắng chừa dưới hàng cuối cho góc bo của widget.
     */
    private fun drawSections(
        context: Context, c: Canvas, paint: Paint,
        w: Int, top: Float, rowH: Float, bottomPad: Float,
        secs: List<Sec>, shown: IntArray
    ) {
        // Lề co giãn: khi chật thì bóp về mức tối thiểu để dành chỗ cho CHỮ, khi
        // rộng thì nới ra cho thoáng. Vài dp lề đổi được thẳng thành cỡ chữ.
        val padMin = dp(context, PAD_NARROWEST); val padMax = dp(context, 8f)
        val gapMin = dp(context, 4f);  val gapMax = dp(context, 7f)
        val endMin = dp(context, 6f);  val endMax = dp(context, 14f)

        // Đo bằng chữ ĐẬM: hàng đang hiệu lực in đậm, rộng hơn hàng thường, nên
        // vừa cho nó thì vừa cho tất cả.
        paint.typeface = Typeface.DEFAULT_BOLD
        val txtPx = fitSections(
            paint, minOf(textSizeForRow(paint, rowH), dp(context, 12f)), dp(context, 7f),
            w - padMin - 2 * gapMin - endMin, secs
        )
        paint.textSize = txtPx
        // Đường cơ sở: đặt khối chữ CÂN GIỮA hàng theo metrics của chính cỡ chữ
        // ấy. Một hệ số ước chừng thì chữ nhỏ lệch lên, chữ to thì dấu của "Đại
        // Tuyết", "Bạch Lộ" thò lên hàng trên.
        val fm = paint.fontMetrics
        val baseDy = (rowH - (fm.descent - fm.ascent)) / 2f - fm.ascent

        // MỘT bộ bề rộng cột cho CẢ HAI mục: lấy cột rộng nhất của từng vị trí
        // rồi dùng chung. Để mỗi mục tự co theo dữ liệu của riêng nó thì "Dương
        // lịch" không thẳng hàng với "Sóc", "Can chi" không thẳng hàng với
        // "Vọng" — hai bảng nằm ngay trên dưới nhau nên lệch một chút là thấy
        // ngay. Cột cuối vì thế rộng bằng mốc ngày giờ của Vọng, và can chi
        // tháng đứng giữa cột ấy.
        val cols = sharedColWidths(paint, secs)
        // Chỗ chữ không dùng hết thì trả lại cho ba khoản lề, chia theo đúng tỉ
        // lệ dư địa của từng khoản — rộng rãi khi có chỗ, chật khi không.
        // Phần của `gap` không tiêu vào đâu cả: hai cột sau căn giữa hộp của
        // chúng, nên chỗ dôi ra tự nở thành khoảng hở giữa ba nhóm chữ. Vẫn
        // phải tính nó vào `room`, không thì padX/padEnd nuốt trọn chỗ dôi và
        // ba cột dính sát nhau ở giữa.
        val room = (padMax - padMin) + 2 * (gapMax - gapMin) + (endMax - endMin)
        val slack = (w - padMin - 2 * gapMin - endMin - cols.sum()).coerceIn(0f, room)
        val f = if (room <= 0f) 0f else slack / room
        val padX = padMin + (padMax - padMin) * f
        val padEnd = endMin + (endMax - endMin) * f

        // Ba mốc vẽ chữ, dùng chung cho cả hai mục: tên căn TRÁI, hai cột sau
        // căn GIỮA hộp của chúng. Cột giữa lấy trọn khoảng trống giữa hai cột
        // bên rồi đứng CHÍNH GIỮA khoảng ấy — hai khoản `gap` hai bên bằng nhau
        // nên triệt tiêu khỏi phép tính tâm.
        val lastCx = w - padEnd - cols[2] / 2f
        val midCx = (padX + cols[0] + (w - padEnd - cols[2])) / 2f

        var y = top
        val tableH = rowH * (secs.size + shown.sum()) + bottomPad
        paint.style = Paint.Style.FILL
        paint.color = Color.WHITE
        c.drawRect(0f, top, w.toFloat(), top + tableH, paint)

        for ((si, sec) in secs.withIndex()) {
            /* ── hàng tiêu đề ── */
            paint.color = sec.headBg
            c.drawRect(0f, y, w.toFloat(), y + rowH, paint)
            paint.color = sec.bar
            c.drawRect(0f, y, dp(context, 3f), y + rowH, paint)
            paint.typeface = Typeface.DEFAULT_BOLD
            paint.color = sec.headFg
            paint.textAlign = Paint.Align.LEFT
            c.drawText(sec.heads[0], padX, y + baseDy, paint)
            paint.textAlign = Paint.Align.CENTER
            c.drawText(sec.heads[1], midCx, y + baseDy, paint)
            c.drawText(sec.heads[2], lastCx, y + baseDy, paint)
            // Mũi tên gập/mở, nép sát mép phải — cùng dấu với tab Lịch. Căn
            // PHẢI chứ không căn giữa, không thì nửa dấu thò ra ngoài mép.
            paint.textAlign = Paint.Align.RIGHT
            paint.textSize = txtPx * 0.7f
            c.drawText(if (sec.open) "▾" else "▸", w - dp(context, 3f), y + baseDy, paint)
            paint.textSize = txtPx
            y += rowH

            /* ── cửa sổ hàng giá trị ── */
            val n = shown[si]
            val from = if (n >= sec.rows.size) 0 else
                (if (sec.active < 0) 0 else sec.active - n / 2)
                    .coerceIn(0, sec.rows.size - n)
            for (k in from until from + n) {
                val on = k == sec.active
                paint.style = Paint.Style.FILL
                if (on) {
                    paint.color = Color.parseColor("#E8EDFF")
                    c.drawRect(0f, y, w.toFloat(), y + rowH, paint)
                } else if (k % 2 == 0) {
                    paint.color = Color.parseColor("#FAFAFA")
                    c.drawRect(0f, y, w.toFloat(), y + rowH, paint)
                }
                paint.typeface = if (on) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                val base = y + baseDy
                paint.textAlign = Paint.Align.LEFT
                paint.color = Color.parseColor("#222222")
                c.drawText(sec.rows[k][0], padX, base, paint)
                paint.textAlign = Paint.Align.CENTER
                paint.color = if (on) Color.parseColor("#222222") else Color.parseColor("#666666")
                c.drawText(sec.rows[k][1], midCx, base, paint)
                paint.color = Color.parseColor(if (on) "#222222" else "#333333")
                c.drawText(sec.rows[k][2], lastCx, base, paint)

                if (k < from + n - 1) {
                    paint.style = Paint.Style.STROKE
                    paint.strokeWidth = 1f
                    paint.color = Color.parseColor("#EEEEEE")
                    c.drawLine(0f, y + rowH, w.toFloat(), y + rowH, paint)
                    paint.style = Paint.Style.FILL
                }
                y += rowH
            }

            // Vạch đóng mục: dưới hàng tiêu đề khi mục đóng, dưới hàng cuối khi
            // mục mở. Mục cuối cùng thì viền ngoài của bảng lo, khỏi kẻ thêm —
            // kẻ vào thì phần đệm góc bo bên dưới trông như một hàng trống.
            if (si < secs.size - 1) {
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = dp(context, 1.2f)
                paint.color = Color.parseColor("#DDDDDD")
                c.drawLine(0f, y, w.toFloat(), y, paint)
                paint.style = Paint.Style.FILL
            }
        }

        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 1f
        paint.color = Color.parseColor("#DDDDDD")
        c.drawRect(0f, top, w.toFloat(), top + tableH, paint)
        paint.style = Paint.Style.FILL
        paint.typeface = Typeface.DEFAULT
    }

    /**
     * Bề rộng ba cột DÙNG CHUNG cho mọi mục: cột rộng nhất của từng vị trí.
     * Nhờ nó mà "Dương lịch" thẳng hàng với "Sóc" và "Can chi" thẳng hàng với
     * "Vọng", dù hai bảng là hai bảng riêng.
     */
    private fun sharedColWidths(paint: Paint, secs: List<Sec>): FloatArray {
        val out = FloatArray(3)
        for (sec in secs) {
            val w = colWidths(paint, sec)
            for (i in 0..2) out[i] = maxOf(out[i], w[i])
        }
        return out
    }

    /**
     * Bề rộng ba cột của một mục, đo trên TOÀN BỘ số hàng — kể cả hàng đang bị
     * cửa sổ cắt ra ngoài — nên cột không nhích khi cuộn hay khi lật tháng.
     *
     * Cột ngày giờ còn lấy thêm khuôn `DATE_TEMPLATE` làm sàn: chữ số của Roboto
     * rộng bằng nhau nên mọi mốc bằng nhau, mà khuôn thì cố định — cột đứng yên
     * cả khi lật sang năm có mốc ngắn hơn.
     */
    private fun colWidths(paint: Paint, sec: Sec): FloatArray {
        val out = FloatArray(3)
        for (i in 0..2) {
            var v = paint.measureText(sec.heads[i])
            for (r in sec.rows) v = maxOf(v, paint.measureText(r[i]))
            for (t in sec.samples[i]) v = maxOf(v, paint.measureText(t))
            out[i] = v
        }
        return out
    }

    /**
     * Cỡ chữ lớn nhất — không quá `start`, không dưới `min` — mà MỌI mục đều
     * nằm trọn trong `avail`. Một cỡ dùng chung cho cả hai: hai mục chữ to nhỏ
     * khác nhau thì nhìn như hai bảng của hai ứng dụng.
     *
     * Chữ không co hoàn toàn tuyến tính theo textSize (hinting, làm tròn pixel)
     * nên tính tỉ lệ xong phải đo lại; vài vòng là hội tụ.
     */
    private fun fitSections(
        paint: Paint, start: Float, min: Float, avail: Float, secs: List<Sec>
    ): Float {
        var size = start
        for (i in 0 until 8) {
            paint.textSize = size
            // Đo theo bộ cột DÙNG CHUNG, không phải theo từng mục: ba cột rộng
            // nhất gộp lại mới là thứ phải nằm vừa bề ngang.
            val need = sharedColWidths(paint, secs).sum()
            if (need <= avail || size <= min) break
            // Hạ thêm 1% cho chắc: đo lại ở vòng sau vẫn có thể nhỉnh hơn tỉ lệ.
            size = maxOf(min, size * (avail / need) * 0.99f)
        }
        return size
    }
    /** Mốc ngày giờ của một tiết khí, đúng dạng hiện ở tab Lịch. */
    private fun dateText(item: LunarTable.JieQi): String {
        val (jy, jm, jd) = LunarTable.civilOf(item.jdn)
        return String.format(
            "%02d-%02d-%d %02d:%02d",
            jd, jm, jy, item.minutes / 60, item.minutes % 60
        )
    }

    /**
     * Cỡ chữ lớn nhất mà một dòng chữ còn nằm trọn trong hàng cao `rowH`.
     *
     * Đo bằng `FontMetrics` của chính phông đang dùng thay vì nhân với một hệ
     * số đoán chừng: tiếng Việt có dấu chồng (Ậ, Ổ, ế) nên phần trên đường cơ
     * sở cao hơn hẳn chữ Latin trơn, mà `ascent` đã tính sẵn khoản ấy. Hệ số
     * 0,66 cũ chừa thừa quá tay — trên widget 4×5 của S21 chữ chỉ còn ~7dp
     * trong khi hàng cao 10,7dp và bề ngang vẫn thừa hơn 40dp.
     */
    private fun textSizeForRow(paint: Paint, rowH: Float): Float {
        val probe = 100f
        val old = paint.textSize
        paint.textSize = probe
        val fm = paint.fontMetrics
        paint.textSize = old
        val lineRatio = (fm.descent - fm.ascent) / probe
        return if (lineRatio <= 0f) rowH * 0.66f else rowH / lineRatio
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

    private fun prefs(context: Context) =
        context.getSharedPreferences("qmdj_widget", Context.MODE_PRIVATE)

    private fun offsetKey(id: Int) = "w$id.offset"

    companion object {
        const val ACTION_REFRESH = "com.bazi.qimen.WIDGET_REFRESH"
        const val ACTION_PREV = "com.bazi.qimen.WIDGET_PREV"
        const val ACTION_NEXT = "com.bazi.qimen.WIDGET_NEXT"
        const val ACTION_TODAY = "com.bazi.qimen.WIDGET_TODAY"

        /** Chiều cao thanh tiêu đề trong widget_calendar.xml. */
        private const val HEADER_DP = 32

        /** Hàng lịch cao gấp ngần này lần hàng bảng — lấy theo tab Lịch. */
        private const val ROW_RATIO = 3.7f

        /**
         * Số hàng bảng dùng để CHIA chiều cao — không phải số hàng vẽ ra thật.
         * Chốt cứng thì khối bảng chiếm đúng một phần màn hình dù hai mục đang
         * mở hay đóng, nên gập/mở trong ứng dụng không làm lưới lịch của widget
         * đổi cỡ. Mười ba là con số của bản cũ (một tiêu đề + 12 hàng), giữ
         * nguyên để widget không đột ngột đổi tỉ lệ sau khi cập nhật.
         */
        private const val NOMINAL_ROWS = 13

        /**
         * Tiết khí lấy ngần này phần số hàng khi CẢ HAI mục cùng mở — đúng
         * `JQ_SHARE` của calendar.js, để hai bên chia chỗ giống nhau.
         */
        private const val JQ_SHARE = 0.65f

        /** Hai khoá trạng thái gập/mở mà calendar.js ghi ra kho tuỳ chọn. */
        private const val SEC_JQ = "qmdj.calSecJq"
        private const val SEC_AM = "qmdj.calSecAm"

        /**
         * Bảng tiết khí luôn chia theo ngần này hàng lịch — số hàng của tháng
         * DÀI nhất — chứ không theo tháng đang xem, để khung bảng và cỡ chữ
         * không nhảy mỗi lần bấm ‹ ›.
         */
        private const val GRID_WEEKS = 6

        /** Khuôn mốc ngày giờ dài nhất — dùng để chốt bề rộng cột "Dương lịch". */
        private const val DATE_TEMPLATE = "00-00-0000 00:00"

        /**
         * Lề hẹp nhất chữ có thể nằm cách mép bảng (khi bảng chật phải bóp lề).
         * Đệm đáy tránh góc bo tính theo đúng con số này — lề thật chỉ rộng hơn,
         * mà lề càng rộng thì cung góc càng ăn nông.
         */
        private const val PAD_NARROWEST = 4f

        /** Máy này có cho ghim widget bằng một cú chạm không? */
        fun canPin(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            return manager.isRequestPinAppWidgetSupported
        }

        /**
         * Vẽ lại ngay mọi widget đang ghim — ứng dụng gọi sau khi đổi ngôn ngữ
         * hoặc địa điểm (xem WebAppBridge.refreshCalendarWidget).
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
         * có báo thức nào (máy tắt, ứng dụng vừa cập nhật); đặt lại báo thức là
         * để từ đó trở đi nó tự theo kịp.
         */
        fun reviveNow(context: Context) {
            context.sendBroadcast(
                Intent(context, CalendarWidgetProvider::class.java).setAction(ACTION_REFRESH)
            )
        }

        fun requestPin(context: Context): Boolean {
            if (!canPin(context)) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            val provider = ComponentName(context, CalendarWidgetProvider::class.java)
            return manager.requestPinAppWidget(provider, null, null)
        }
    }
}
