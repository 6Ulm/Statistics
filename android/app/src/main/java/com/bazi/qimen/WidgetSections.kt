package com.bazi.qimen

import android.content.Context
import android.graphics.Color
import java.util.TimeZone

/**
 * Hai mục gập được của tab Lịch, dựng thành chữ để widget hiện: "Tiết khí"
 * (Tiết Khí · Dương lịch · Can chi) và "Lịch âm" (Tháng âm · Sóc · Vọng).
 *
 * Để RIÊNG khỏi provider vì cả hai phía đều cần: provider dựng hàng tiêu đề,
 * còn `WidgetSectionService` dựng từng hàng giá trị cho ListView. Hai bên gọi
 * chung một hàm thì không thể hiểu khác nhau.
 *
 * Khác bản cũ ở chỗ KHÔNG còn cắt cửa sổ hàng: ListView giữ đủ 24 và 13 hàng
 * rồi cuộn, y như mục trong tab Lịch cuộn trong khung của nó.
 */
object WidgetSections {

    const val JQ = "jq"
    const val AM = "am"

    class Sec(
        val key: String,
        val heads: Array<String>,
        val rows: List<Array<String>>,
        /** Hàng ứng với ngày đang chọn; −1 nghĩa là không tô hàng nào. */
        val active: Int,
        val open: Boolean,
        val headBg: Int,
        val headFg: Int,
        val bar: Int,
    )

    /** Màu tô hàng đang hiệu lực — đúng `--accent-light` của tab Lịch. */
    val ACTIVE_BG: Int = Color.parseColor("#E8EDFF")
    val TEXT_MAIN: Int = Color.parseColor("#222222")
    val TEXT_DIM: Int = Color.parseColor("#666666")
    val TEXT_LAST: Int = Color.parseColor("#333333")
    val ROW_ALT_BG: Int = Color.parseColor("#FAFAFA")
    val ROW_BG: Int = Color.WHITE

    /**
     * Dựng hai mục cho tháng đang xem, tô đậm hàng ứng với `anchorJdn` (ngày
     * người dùng vừa chạm, chưa chạm thì hôm nay).
     *
     * Nội dung theo THÁNG ĐANG XEM (lật ‹ › là đổi năm tiết khí / năm âm), còn
     * hàng được tô theo NGÀY ĐANG CHỌN — lật sang năm khác thì ngày ấy không
     * còn trong bảng nữa và không hàng nào được tô, đúng như tab Lịch.
     */
    fun build(context: Context, year: Int, month: Int, anchorJdn: Int): List<Sec> {
        LunarTable.ensureLoaded(context)
        LunarTable.loadAppCache(context)
        val tz = WidgetPrefs.timeZone(context)
        val zh = WidgetPrefs.isZh(context)
        val ref = LunarTable.jdn(year, month, 15)
        val out = ArrayList<Sec>(2)

        val jieQi = LunarTable.jieQiYearOf(ref, zh).map { LunarTable.localize(it, tz) }
        if (jieQi.size == 24) {
            // Tiết khí đang hiệu lực = mốc CUỐI CÙNG không muộn hơn ngày đang
            // chọn. Ngày ấy nằm ngoài dãy thì không tô mục nào.
            var active = -1
            if (anchorJdn >= jieQi[0].jdn) {
                for (i in jieQi.indices) if (jieQi[i].jdn <= anchorJdn) active = i
            }
            out.add(Sec(
                JQ,
                arrayOf(
                    if (zh) "节气" else context.getString(R.string.col_jieqi),
                    if (zh) "公历" else context.getString(R.string.col_solar),
                    if (zh) "月柱" else context.getString(R.string.col_ganzhi),
                ),
                jieQi.map { arrayOf(it.name, dateText(it), LunarTable.ganZhi60(it.gz, zh)) },
                active,
                WidgetPrefs.secOpen(context, WidgetPrefs.SEC_JQ),
                Color.parseColor("#FDEEEE"), Color.parseColor("#B71C1C"),
                Color.parseColor("#D32F2F"),
            ))
        }

        // Mục "Lịch âm" đã CHUYỂN sang tab Tra cứu của ứng dụng, và bỏ hẳn khỏi
        // lịch đã ghim — người dùng chốt: widget chỉ còn lưới lịch và bảng tiết
        // khí. Khoá SEC_AM đã bỏ theo; hằng số AM còn lại chỉ để một widget bản
        // cũ chưa kịp vẽ lại không vỡ khi đọc phải khoá mục lạ trong Intent.
        return out
    }

    /** Một mục theo khoá, hoặc null nếu lần này không dựng được. */
    fun of(context: Context, year: Int, month: Int, anchorJdn: Int, key: String): Sec? =
        build(context, year, month, anchorJdn).firstOrNull { it.key == key }

    /** Nhãn một tháng âm, đúng chữ của tab Lịch: "Tháng 5 (Nhuận)" · "5月 (闰)". */
    fun monthLabel(
        context: Context, month: Int, isLeap: Boolean, zh: Boolean, leap: String
    ): String =
        (if (zh) "${month}月" else context.getString(R.string.month_n, month)) +
            (if (isLeap) " ($leap)" else "")

    /** Mốc ngày giờ của một tiết khí, đúng dạng hiện ở tab Lịch. */
    fun dateText(item: LunarTable.JieQi): String {
        val (jy, jm, jd) = LunarTable.civilOf(item.jdn)
        return String.format(
            "%02d-%02d-%d %02d:%02d",
            jd, jm, jy, item.minutes / 60, item.minutes % 60
        )
    }
}
