package com.bazi.qimen

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent

/**
 * Dựng lại báo thức nửa đêm của widget sau những lúc nó bị mất.
 *
 * Widget tự vẽ lại nhờ MỘT báo thức lặp hằng ngày mà `scheduleMidnight()` đặt.
 * Báo thức của AlarmManager KHÔNG sống qua lần khởi động lại máy — và
 * `calendar_widget_info.xml` để `updatePeriodMillis="0"`, tức hệ thống cũng
 * không tự gọi `onUpdate` theo chu kỳ. Cộng lại: khởi động lại điện thoại là
 * lịch đã ghim đứng im ở ngày hôm ấy, không còn gì đánh thức nó, cho tới khi
 * người dùng tình cờ mở ứng dụng hay bấm ‹ ›. Đúng cái cảnh "widget hiện bản
 * cũ" mà không ai hiểu vì sao.
 *
 * Nên phải có một receiver RIÊNG, và phải `exported="true"`: receiver của
 * widget để `exported="false"` nên không nhận nổi broadcast của hệ thống (đó
 * là lý do ghi trong AndroidManifest.xml trước đây — đúng về mặt kỹ thuật,
 * nhưng kết luận "thôi không cần" thì sai).
 *
 * Ba mốc cần bắt lại:
 *   • BOOT_COMPLETED    — khởi động lại máy, mọi báo thức bay sạch;
 *   • MY_PACKAGE_REPLACED — cập nhật ứng dụng, tiến trình bị giết và dựng lại;
 *   • TIMEZONE_CHANGED  — đi máy bay sang múi giờ khác thì "hôm nay" đổi nghĩa,
 *     mà mốc nửa đêm cũng dời đi.
 *
 * Cả ba đều nằm trong danh sách ngoại lệ của luật chặn broadcast ngầm từ
 * Android 8, nên receiver khai trong manifest vẫn nhận được.
 */
class BootReceiver : BroadcastReceiver() {

    override fun onReceive(context: Context, intent: Intent) {
        when (intent.action) {
            Intent.ACTION_BOOT_COMPLETED,
            Intent.ACTION_MY_PACKAGE_REPLACED,
            Intent.ACTION_TIMEZONE_CHANGED -> CalendarWidgetProvider.reviveNow(context)
        }
    }
}
