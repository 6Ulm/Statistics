package com.bazi.qimen

import android.content.Context
import android.content.SharedPreferences
import java.util.Calendar
import java.util.TimeZone
import org.json.JSONException
import org.json.JSONObject

/**
 * Những con số chiều cao của widget, phải khớp KHÍT với `widget_calendar.xml`.
 *
 * Bố cục chia chiều cao bằng `layout_weight`, mà Kotlin thì phải tự biết khung
 * lưới lịch còn lại bao nhiêu để vẽ bitmap cho vừa — `scaleType="fitXY"` nên
 * lệch một chút là chữ trong lưới bị kéo giãn. Đặt XML một đằng Kotlin một nẻo
 * là hỏng âm thầm, nên `tools/test_widget_layout.mjs` đọc cả hai bên và canh.
 *
 * `setViewLayoutHeight` chỉ có từ API 31 mà minSdk là 24, nên không thể đi
 * đường ngược lại (Kotlin tính rồi bảo XML cao bao nhiêu).
 */
object WidgetLayout {
    const val HEADER_DP = 32
    const val DOW_DP = 16
    const val SEC_HEAD_DP = 22
    const val ROW_DP = 20
    const val CORNER_PAD_DP = 6

    /** Phần chia chiều cao giữa lưới lịch và thân mục Tiết khí. */
    const val W_GRID = 58
    const val W_JQ = 42

    /** Phần chia bề ngang ba cột — dùng chung cho hàng tiêu đề lẫn hàng giá trị. */
    const val COL0 = 27
    const val COL1 = 35
    const val COL2 = 38

    /** Cỡ chữ mặc định của hàng và hàng tiêu đề hai mục (sp), theo XML. */
    const val TEXT_SP = 11.5f
    private const val TEXT_SP_MIN = 8.5f
    /** Bề ngang mà ở đó chữ còn để nguyên cỡ; hẹp hơn thì co lại theo tỉ lệ.
     *  340 chứ không phải 300: cỡ chữ cơ bản vừa nâng từ 10sp lên 11,5sp, giữ
     *  nguyên 300 thì widget bị bóp về sàn 250dp có chữ to hơn trước và cột
     *  giữa cụt mất 1px đuôi mốc giờ. Nới mốc ra là widget cỡ thường vẫn 11,5sp
     *  mà widget bóp hẹp trở về đúng cỡ cũ. */
    private const val TEXT_FULL_DP = 340f

    /**
     * Cỡ chữ (sp) cho hai mục, theo bề ngang widget.
     *
     * Ba cột chia bằng `layout_weight` nên KHÔNG tự nới theo chữ: ở cỡ sàn
     * 250dp mà người dùng tự bóp tay, cột giữa chỉ còn ~87dp trong khi
     * "21-12-2025 22:03" cần ~92dp — `ellipsize` sẽ nuốt mất đuôi giờ. Thà chữ
     * nhỏ hơn một chút còn hơn mất chữ số.
     */
    fun rowTextSp(wDp: Int): Float =
        (TEXT_SP * wDp / TEXT_FULL_DP).coerceIn(TEXT_SP_MIN, TEXT_SP)

    /**
     * Chiều cao (dp) của khung lưới lịch, theo đúng luật `layout_weight`:
     * mục nào đang đóng thì ListView của nó là GONE, và LinearLayout KHÔNG tính
     * phần của View đã GONE — chỗ ấy chia lại cho những view còn lại.
     */
    /**
     * Mục "Lịch âm" đã bỏ hẳn khỏi lịch đã ghim (chuyển sang tab Tra cứu của
     * ứng dụng), nên chỉ còn MỘT hàng tiêu đề mục và một trọng số để chia.
     *
     * `jqOpen` chỉ còn nghĩa "mục có DỰNG ĐƯỢC không" — bảng tra thiếu dữ liệu
     * cho năm đang xem thì WidgetSections.build() bỏ hẳn mục ấy và cả phần
     * chiều cao của nó về cho lưới. Không còn nghĩa "người dùng có gập nó lại
     * không": mục không gập được nữa, ở cả ứng dụng lẫn lịch đã ghim. Nó vẫn
     * CUỘN được — thân mục là ListView do WidgetSectionService nuôi.
     */
    fun gridHeightDp(totalDp: Int, jqOpen: Boolean): Float {
        val fixed = HEADER_DP + DOW_DP + SEC_HEAD_DP + CORNER_PAD_DP
        val avail = (totalDp - fixed).coerceAtLeast(60).toFloat()
        var sum = W_GRID
        if (jqOpen) sum += W_JQ
        return avail * W_GRID / sum
    }
}

/**
 * Trạng thái của widget: tháng đang xem, ngày đang chọn, và những thứ nó đọc
 * ké từ ứng dụng (ngôn ngữ, địa điểm).
 *
 * Cả provider lẫn `WidgetSectionService` đều cần đúng những con số này — để
 * chung một chỗ thì hai bên không thể hiểu khác nhau.
 */
object WidgetPrefs {
    // KHÔNG còn khoá gập/mở nào. Mục Tiết khí luôn hiện, ở cả ứng dụng lẫn
    // lịch đã ghim; `qmdj.calSecJq` đã thôi được đọc và thôi được ghi.

    fun widget(context: Context): SharedPreferences =
        context.getSharedPreferences("qmdj_widget", Context.MODE_PRIVATE)

    fun app(context: Context): SharedPreferences =
        context.getSharedPreferences("qmdj_prefs", Context.MODE_PRIVATE)

    private fun offsetKey(id: Int) = "w$id.offset"
    private fun selKey(id: Int) = "w$id.sel"

    fun offsetOf(context: Context, id: Int): Int = widget(context).getInt(offsetKey(id), 0)

    fun setOffset(context: Context, id: Int, v: Int) {
        widget(context).edit().putInt(offsetKey(id), v.coerceIn(-1200, 1200)).apply()
    }

    /** Ngày người dùng vừa chạm trong widget; 0 nghĩa là chưa chọn gì. */
    fun selectedJdn(context: Context, id: Int): Int = widget(context).getInt(selKey(id), 0)

    fun setSelected(context: Context, id: Int, jdn: Int) {
        widget(context).edit().putInt(selKey(id), jdn).apply()
    }

    fun clearWidget(context: Context, id: Int) {
        widget(context).edit().remove(offsetKey(id)).remove(selKey(id)).apply()
    }

    fun todayJdn(): Int {
        val now = Calendar.getInstance()
        return LunarTable.jdn(
            now.get(Calendar.YEAR), now.get(Calendar.MONTH) + 1, now.get(Calendar.DAY_OF_MONTH)
        )
    }

    /**
     * Ngày mà hai mục lấy làm mốc tô đậm: ngày đang chọn, chưa chọn thì hôm nay
     * — đúng như tab Lịch, nơi `selected` mặc định là hôm nay.
     */
    fun anchorJdn(context: Context, id: Int): Int {
        val sel = selectedJdn(context, id)
        return if (sel > 0) sel else todayJdn()
    }

    /** Tháng dương đang xem (năm, tháng). */
    fun viewMonth(context: Context, id: Int): Pair<Int, Int> {
        val cal = Calendar.getInstance().apply {
            set(Calendar.DAY_OF_MONTH, 1)
            add(Calendar.MONTH, offsetOf(context, id))
        }
        return cal.get(Calendar.YEAR) to (cal.get(Calendar.MONTH) + 1)
    }

    fun isZh(context: Context): Boolean = LunarTable.langOf(context) == "zh"

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
    fun timeZone(context: Context): TimeZone {
        val raw = app(context).getString("qmdj.location", null) ?: return TimeZone.getDefault()
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
}
