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
 * Widget màn hình chính: CHỈ lịch âm dương và bảng tiết khí — không có bàn Kỳ
 * Môn, không có thanh tab, không có nút ghim. Bố cục và màu sắc giống hệt tab
 * Lịch trong ứng dụng.
 *
 * Thanh tiêu đề là View thật (xem widget_calendar.xml) để hai mũi tên ‹ › bấm
 * được mà lùi/tiến tháng; phần lưới và bảng tiết khí bên dưới vẽ ra bitmap vì
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

        val views = RemoteViews(context.packageName, R.layout.widget_calendar)
        views.setTextViewText(R.id.widgetTitle, "LỊCH ÂM THÁNG $month/$year")
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
     * cụm vị trí thành JSON dưới khoá `qmdj.location` (xem location.js); chưa
     * chọn gì thì dùng giờ Việt Nam, đúng như cơ sở lịch âm của widget.
     */
    private fun selectedTimeZone(context: Context): TimeZone {
        val raw = context.getSharedPreferences("qmdj_prefs", Context.MODE_PRIVATE)
            .getString("qmdj.location", null) ?: return TimeZone.getTimeZone(DEFAULT_TZ)
        val id = try {
            JSONObject(raw).optString("tzId", "")
        } catch (e: JSONException) {
            ""
        }
        if (id.isEmpty()) return TimeZone.getTimeZone(DEFAULT_TZ)
        val tz = TimeZone.getTimeZone(id)
        // getTimeZone() trả về GMT cho id lạ thay vì báo lỗi — bắt lại ở đây,
        // không thì một id hỏng lặng lẽ đẩy mọi mốc về UTC.
        return if (tz.id == "GMT" && id != "GMT" && id != "UTC") {
            TimeZone.getTimeZone(DEFAULT_TZ)
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

        // Bảng tiết khí: CẢ 24 mục của năm, xếp hai cột 12 — đúng hình dạng của
        // tab Lịch. Mười ba hàng (một hàng tiêu đề + 12) là một khối lớn, nên
        // chia theo TỈ LỆ của ứng dụng thay vì cho nó một khoản cố định: ở tab
        // Lịch trên A51, hàng lịch cao ~78dp còn hàng tiết khí ~21dp, tức xấp
        // xỉ 3,7 lần.
        // Giờ giao tiết hiện theo múi giờ của ĐỊA ĐIỂM ĐANG CHỌN trong ứng
        // dụng, giống hệt tab Lịch — widget không có bảng chọn nơi riêng.
        val tz = selectedTimeZone(context)
        val jieQi = LunarTable.jieQiYearOf(LunarTable.jdn(year, month, 15))
            .map { LunarTable.localize(it, tz) }
        val dowH = minOf(h * 0.09f, dp(context, 15f))
        val jqRows = if (jieQi.size == 24) 13 else 0

        // Đáy widget bị góc bo cắt mất một cung tròn. Bảng tiết khí chạm sát
        // mép thì hàng cuối (Mang Chủng · Đại Tuyết) mất chữ đầu và mất đuôi
        // giờ. Chừa đúng chỗ cung ăn tới Ở HOÀNH ĐỘ CHỮ BẮT ĐẦU — lấy lề hẹp
        // nhất có thể xảy ra làm trường hợp xấu nhất, vì lề thật chỉ rộng hơn.
        val safeBottom = if (jqRows == 0) 0f else
            cornerBottomPad(context, cornerRadius(context), dp(context, PAD_NARROWEST))

        // Bảng tiết khí phải CỐ ĐỊNH: chia theo GRID_WEEKS (tháng dài nhất) chứ
        // không theo `weeks` của tháng đang xem. Chia theo `weeks` thì tháng gọn
        // 5 hàng làm bảng phình ra ~12% — lật tháng một cái là cả khung lẫn cỡ
        // chữ nhảy, đúng thứ người dùng thấy chướng. Chỗ dôi ra của tháng 5 hàng
        // đổ vào lưới lịch, nơi ô cao thêm chỉ tốt lên.
        // 6 hàng lịch × 3,7 + 13 hàng tiết khí = 35,2 phần bằng nhau.
        // Kẹp hai đầu: dưới 9dp thì chữ tiết khí không đọc nổi, trên 18dp thì
        // bảng phình ra nuốt mất lưới lịch trên widget cao.
        val jqRowH = if (jqRows == 0) 0f else
            ((h - dowH - safeBottom) / (GRID_WEEKS * ROW_RATIO + jqRows))
                .coerceIn(dp(context, 9f), dp(context, 18f))
        val jqH = if (jqRows == 0) 0f else jqRowH * jqRows + dp(context, 5f) + safeBottom
        val gridH = (h - dowH - jqH).coerceAtLeast(0f)
        val cellW = w / 7f
        val cellH = gridH / weeks

        paint.color = Color.parseColor("#FDECEF")
        c.drawRect(0f, 0f, w.toFloat(), h.toFloat(), paint)

        // hàng thứ
        val dows = arrayOf("T2", "T3", "T4", "T5", "T6", "T7", "CN")
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
                val (can, chi) = LunarTable.ganZhiOf(jdn)
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

        // ── Bảng tiết khí: 12 hàng × 2 cặp cột, y như tab Lịch ──
        if (jqRows > 0) {
            drawJieQi(context, c, paint, w, dowH + gridH + dp(context, 5f),
                jqRowH, safeBottom, jieQi, todayJdn)
        }
        return bmp
    }

    /**
     * Bảng 24 tiết khí, 12 hàng × hai cặp cột — cùng hình dạng, cùng màu với
     * bảng ở tab Lịch: nền trắng, hàng lẻ tô nhạt, vách ngăn dọc giữa hai nửa,
     * mục đang hiệu lực tô màu nhấn.
     *
     * Hai cột nằm ở CHỖ CỐ ĐỊNH, giống hệt nhau ở cả hai nửa và ở mọi tháng:
     * cột tên rộng đúng bằng tên dài nhất (đo bằng measureText, y như
     * `width:1%` bên CSS), cột ngày bắt đầu ngay sau nó. Cỡ chữ hạ xuống vừa đủ
     * để dòng dài nhất còn nằm trong nửa bảng — thà chữ nhỏ hơn một chút còn
     * hơn để mốc ngày giờ tràn qua vách ngăn rồi bị mép widget cắt cụt.
     *
     * `bottomPad` là khoảng trắng chừa dưới hàng cuối cho góc bo của widget.
     */
    private fun drawJieQi(
        context: Context, c: Canvas, paint: Paint,
        w: Int, top: Float, rowH: Float, bottomPad: Float,
        jieQi: List<LunarTable.JieQi>, todayJdn: Int
    ) {
        val halfW = w / 2f
        // Lề của mỗi nửa bảng co giãn: khi chật thì bóp về mức tối thiểu để
        // dành chỗ cho CHỮ, khi rộng thì nới ra cho thoáng. Trên S21 nửa bảng
        // chỉ rộng ~165dp mà "Sương Giáng + 07-12-2026 09:52" đã gần kín, nên
        // vài dp lề ấy đổi được thẳng thành cỡ chữ.
        val padMin = dp(context, PAD_NARROWEST); val padMax = dp(context, 8f)  // lề trái
        val gapMin = dp(context, 4f);  val gapMax = dp(context, 6f)   // giữa hai cột
        val endMin = dp(context, 5f);  val endMax = dp(context, 10f)  // lề phải

        // Mục đang hiệu lực: mốc CUỐI CÙNG không muộn hơn hôm nay, giống tab
        // Lịch. Hôm nay nằm ngoài dãy đang hiện thì không tô mục nào.
        var active = -1
        for (i in jieQi.indices) if (jieQi[i].jdn <= todayJdn) active = i

        val dates = jieQi.map { dateText(it) }

        // Đo bằng chữ ĐẬM: dòng đang hiệu lực in đậm, cao và rộng hơn dòng
        // thường, nên hàng vừa cho nó thì vừa cho tất cả.
        paint.typeface = Typeface.DEFAULT_BOLD
        val txtPx = fitTextSize(
            paint, minOf(textSizeForRow(paint, rowH), dp(context, 12f)), dp(context, 7f),
            halfW - padMin - gapMin - endMin, jieQi, dates
        )
        paint.textSize = txtPx
        // Đường cơ sở: đặt khối chữ CÂN GIỮA hàng theo metrics của chính cỡ chữ
        // ấy. Hệ số 0,72 cũ là ước chừng — chữ nhỏ thì lệch lên, chữ to thì dấu
        // của "Đại Tuyết", "Bạch Lộ" thò lên hàng trên.
        val fm = paint.fontMetrics
        val baseDy = (rowH - (fm.descent - fm.ascent)) / 2f - fm.ascent
        val nameW = widestName(paint, jieQi)
        val dateW = widestDate(paint, dates)

        // Chỗ chữ không dùng hết thì trả lại cho ba khoản lề, chia theo đúng tỉ
        // lệ dư địa của từng khoản — rộng rãi khi có chỗ, chật khi không.
        val room = (padMax - padMin) + (gapMax - gapMin) + (endMax - endMin)
        val slack = (halfW - padMin - gapMin - endMin - nameW - dateW).coerceIn(0f, room)
        val f = if (room <= 0f) 0f else slack / room
        val padX = padMin + (padMax - padMin) * f
        val gap = gapMin + (gapMax - gapMin) * f
        val padEnd = endMin + (endMax - endMin) * f

        // Cột ngày bắt đầu ở đúng một chỗ: hai nửa và hàng tiêu đề thẳng hàng.
        val dateDx = padX + nameW + gap

        val headH = rowH
        val tableH = headH + rowH * 12 + bottomPad

        paint.style = Paint.Style.FILL
        paint.color = Color.WHITE
        c.drawRect(0f, top, w.toFloat(), top + tableH, paint)

        // hàng tiêu đề
        paint.typeface = Typeface.DEFAULT_BOLD
        paint.textAlign = Paint.Align.LEFT
        paint.color = Color.parseColor("#222222")
        val headBase = top + baseDy
        for (half in 0..1) {
            val x0 = halfW * half
            c.drawText(context.getString(R.string.col_jieqi), x0 + padX, headBase, paint)
            c.drawText(context.getString(R.string.col_solar), x0 + dateDx, headBase, paint)
        }
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = dp(context, 1.2f)
        paint.color = Color.parseColor("#DDDDDD")
        c.drawLine(0f, top + headH, w.toFloat(), top + headH, paint)
        paint.style = Paint.Style.FILL

        for (r in 0 until 12) {
            val y = top + headH + rowH * r
            if (r % 2 == 0) {
                paint.color = Color.parseColor("#FAFAFA")
                c.drawRect(0f, y, w.toFloat(), y + rowH, paint)
            }
            for (half in 0..1) {
                val k = r + half * 12
                val item = jieQi[k]
                val x0 = halfW * half
                val on = k == active
                if (on) {
                    paint.color = Color.parseColor("#E8EDFF")
                    c.drawRect(x0, y, x0 + halfW, y + rowH, paint)
                }
                val base = y + baseDy
                paint.textAlign = Paint.Align.LEFT
                paint.typeface = if (on) Typeface.DEFAULT_BOLD else Typeface.DEFAULT
                paint.color = Color.parseColor("#222222")
                c.drawText(item.name, x0 + padX, base, paint)

                paint.color = if (on) Color.parseColor("#222222") else Color.parseColor("#666666")
                c.drawText(dates[k], x0 + dateDx, base, paint)
            }
            // Vạch dưới hàng cuối là viền ngoài của bảng, không kẻ thêm — kẻ vào
            // thì phần đệm góc bo bên dưới trông như một hàng trống.
            if (r < 11) {
                paint.style = Paint.Style.STROKE
                paint.strokeWidth = 1f
                paint.color = Color.parseColor("#EEEEEE")
                c.drawLine(0f, y + rowH, w.toFloat(), y + rowH, paint)
                paint.style = Paint.Style.FILL
            }
        }

        // vách ngăn giữa hai nửa — kẻ suốt cả bảng, kể cả hàng tiêu đề
        paint.style = Paint.Style.STROKE
        paint.strokeWidth = 1f
        paint.color = Color.parseColor("#DDDDDD")
        c.drawLine(halfW, top, halfW, top + tableH, paint)
        c.drawRect(0f, top, w.toFloat(), top + tableH, paint)
        paint.style = Paint.Style.FILL
        paint.typeface = Typeface.DEFAULT
    }

    /** Mốc ngày giờ của một tiết khí, đúng dạng hiện ở tab Lịch. */
    private fun dateText(item: LunarTable.JieQi): String {
        val (jy, jm, jd) = LunarTable.civilOf(item.jdn)
        return String.format(
            "%02d-%02d-%d %02d:%02d",
            jd, jm, jy, item.minutes / 60, item.minutes % 60
        )
    }

    /** Tên tiết khí dài nhất — dãy 24 tên là cố định nên số này không đổi. */
    private fun widestName(paint: Paint, jieQi: List<LunarTable.JieQi>): Float {
        var v = 0f
        for (item in jieQi) v = maxOf(v, paint.measureText(item.name))
        return v
    }

    /**
     * Bề rộng cột ngày. Lấy theo KHUÔN `DATE_TEMPLATE` chứ không chỉ theo mốc
     * của năm đang xem: chữ số của Roboto rộng bằng nhau nên hai con số bằng
     * nhau, mà khuôn thì cố định — cột không nhích khi lật sang năm khác. Vẫn
     * so với mốc thật để phòng phông có chữ số rộng hẹp khác nhau.
     */
    private fun widestDate(paint: Paint, dates: List<String>): Float {
        var v = paint.measureText(DATE_TEMPLATE)
        for (d in dates) v = maxOf(v, paint.measureText(d))
        return v
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

    /**
     * Cỡ chữ lớn nhất — không quá `start`, không dưới `min` — mà tên tiết khí
     * dài nhất cộng mốc ngày giờ dài nhất vẫn nằm trong `avail`.
     *
     * Chữ không co hoàn toàn tuyến tính theo textSize (hinting, làm tròn pixel)
     * nên tính tỉ lệ xong phải đo lại; vài vòng là hội tụ.
     */
    private fun fitTextSize(
        paint: Paint, start: Float, min: Float, avail: Float,
        jieQi: List<LunarTable.JieQi>, dates: List<String>
    ): Float {
        var size = start
        for (i in 0 until 6) {
            paint.textSize = size
            val need = widestName(paint, jieQi) + widestDate(paint, dates)
            if (need <= avail || size <= min) break
            // Hạ thêm 1% cho chắc: đo lại ở vòng sau vẫn có thể nhỉnh hơn tỉ lệ.
            size = maxOf(min, size * (avail / need) * 0.99f)
        }
        return size
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

        /** Hàng lịch cao gấp ngần này lần hàng tiết khí — lấy theo tab Lịch. */
        private const val ROW_RATIO = 3.7f

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

        /** Chưa chọn địa điểm thì lấy giờ Việt Nam — cùng cơ sở với lịch âm. */
        private const val DEFAULT_TZ = "Asia/Ho_Chi_Minh"

        /** Máy này có cho ghim widget bằng một cú chạm không? */
        fun canPin(context: Context): Boolean {
            if (Build.VERSION.SDK_INT < Build.VERSION_CODES.O) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            return manager.isRequestPinAppWidgetSupported
        }

        fun requestPin(context: Context): Boolean {
            if (!canPin(context)) return false
            val manager = AppWidgetManager.getInstance(context) ?: return false
            val provider = ComponentName(context, CalendarWidgetProvider::class.java)
            return manager.requestPinAppWidget(provider, null, null)
        }
    }
}
