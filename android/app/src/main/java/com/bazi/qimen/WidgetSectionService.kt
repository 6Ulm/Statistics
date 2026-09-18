package com.bazi.qimen

import android.appwidget.AppWidgetManager
import android.content.Context
import android.content.Intent
import android.util.TypedValue
import android.widget.RemoteViews
import android.widget.RemoteViewsService

/**
 * Nguồn hàng cho hai ListView của widget.
 *
 * Widget chỉ CUỘN được bên trong một collection view, và collection view thì
 * chỉ nhận dữ liệu qua một `RemoteViewsService` như thế này — không có đường
 * nào khác. Mỗi mục (Tiết khí / Lịch âm) của mỗi widget là một factory riêng,
 * phân biệt bằng id widget cộng khoá mục nhét trong `Intent.data` (hệ thống
 * so intent bằng `filterEquals`, vốn BỎ QUA extras — hai mục mà chỉ khác extras
 * thì dùng chung một factory và cùng hiện một bảng).
 */
class WidgetSectionService : RemoteViewsService() {

    override fun onGetViewFactory(intent: Intent): RemoteViewsFactory =
        SectionFactory(
            applicationContext,
            intent.getIntExtra(AppWidgetManager.EXTRA_APPWIDGET_ID, AppWidgetManager.INVALID_APPWIDGET_ID),
            intent.getStringExtra(EXTRA_SECTION) ?: WidgetSections.JQ
        )

    companion object {
        const val EXTRA_SECTION = "com.bazi.qimen.SECTION"
    }
}

private class SectionFactory(
    private val context: Context,
    private val widgetId: Int,
    private val key: String,
) : RemoteViewsService.RemoteViewsFactory {

    private var rows: List<Array<String>> = emptyList()
    private var active = -1
    private var textSp = WidgetLayout.TEXT_SP

    override fun onCreate() = reload()

    /**
     * Hệ thống gọi lại mỗi lần provider bảo `notifyAppWidgetViewDataChanged`.
     * Đọc lại TỪ ĐẦU: tháng đang xem, ngày đang chọn, ngôn ngữ, địa điểm đều có
     * thể vừa đổi.
     */
    override fun onDataSetChanged() = reload()

    private fun reload() {
        // Cỡ chữ theo bề ngang widget — cột chia bằng weight nên chữ phải co,
        // không thì widget bóp hẹp là mất đuôi mốc ngày giờ.
        val wDp = AppWidgetManager.getInstance(context)
            .getAppWidgetOptions(widgetId)
            .getInt(AppWidgetManager.OPTION_APPWIDGET_MIN_WIDTH, 0)
        textSp = WidgetLayout.rowTextSp(if (wDp > 0) wDp else 320)
        val (y, m) = WidgetPrefs.viewMonth(context, widgetId)
        val sec = WidgetSections.of(context, y, m, WidgetPrefs.anchorJdn(context, widgetId), key)
        rows = sec?.rows ?: emptyList()
        active = sec?.active ?: -1
    }

    override fun onDestroy() {
        rows = emptyList()
    }

    override fun getCount(): Int = rows.size

    override fun getViewAt(position: Int): RemoteViews {
        val v = RemoteViews(context.packageName, R.layout.widget_sec_row)
        if (position < 0 || position >= rows.size) return v
        val row = rows[position]
        val on = position == active
        v.setInt(R.id.rowRoot, "setBackgroundColor", when {
            on -> WidgetSections.ACTIVE_BG
            position % 2 == 0 -> WidgetSections.ROW_ALT_BG
            else -> WidgetSections.ROW_BG
        })
        val ids = intArrayOf(R.id.rowCol0, R.id.rowCol1, R.id.rowCol2)
        val colors = intArrayOf(
            WidgetSections.TEXT_MAIN,
            if (on) WidgetSections.TEXT_MAIN else WidgetSections.TEXT_DIM,
            if (on) WidgetSections.TEXT_MAIN else WidgetSections.TEXT_LAST,
        )
        for (i in 0..2) {
            v.setTextViewText(ids[i], row[i])
            v.setTextColor(ids[i], colors[i])
            v.setTextViewTextSize(ids[i], TypedValue.COMPLEX_UNIT_SP, textSp)
        }
        return v
    }

    // Không có hàng "đang tải" riêng: hàng nào cũng dựng xong ngay trong bộ nhớ.
    override fun getLoadingView(): RemoteViews? = null

    // Mọi hàng dùng chung một layout.
    override fun getViewTypeCount(): Int = 1

    override fun getItemId(position: Int): Long = position.toLong()

    override fun hasStableIds(): Boolean = true
}
