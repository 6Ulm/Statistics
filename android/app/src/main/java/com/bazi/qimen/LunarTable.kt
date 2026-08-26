package com.bazi.qimen

import android.content.Context

/**
 * Tra âm lịch và can chi cho widget màn hình chính.
 *
 * Widget vẽ bằng RemoteViews nên không có WebView, không chạy được lunar.js.
 * Thay vì chép thuật toán tính điểm Sóc sang Kotlin (dễ sai lệch với phần còn
 * lại của ứng dụng), mọi mốc mùng 1 từ 1900–2100 được tính sẵn bằng chính
 * lunar.js rồi đóng gói vào `assets/lunar_months.txt`; ở đây chỉ tra bảng.
 *
 * Bảng và cách tra đã được đối chiếu với lunar.js từng ngày một trong
 * 73.414 ngày — xem `tools/test_lunar_table.mjs`.
 *
 * Widget lunar-date lookup, driven by a table precomputed from lunar.js.
 */
object LunarTable {

    val CAN = arrayOf("Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý")
    val CHI = arrayOf("Tý", "Sửu", "Dần", "Mão", "Thìn", "Tỵ", "Ngọ", "Mùi",
                      "Thân", "Dậu", "Tuất", "Hợi")

    /** Ngày âm lịch của một ngày dương lịch. */
    data class LunarDay(val day: Int, val month: Int, val year: Int, val leap: Boolean)

    private var starts: IntArray = IntArray(0)
    private var lYear: IntArray = IntArray(0)
    private var lMonth: IntArray = IntArray(0)
    private var lLeap: BooleanArray = BooleanArray(0)
    private var ganZhiEpoch = 0
    private var loaded = false

    @Synchronized
    fun ensureLoaded(context: Context) {
        if (loaded) return
        try {
            context.assets.open("lunar_months.txt").bufferedReader().use { reader ->
                val lines = reader.readLines().filter { it.isNotBlank() }
                ganZhiEpoch = lines[0].trim().toInt()
                val n = lines.size - 1
                starts = IntArray(n); lYear = IntArray(n)
                lMonth = IntArray(n); lLeap = BooleanArray(n)
                for (i in 1..n) {
                    val p = lines[i].split(' ')
                    starts[i - 1] = p[0].toInt()
                    lYear[i - 1] = p[1].toInt()
                    lMonth[i - 1] = p[2].toInt()
                    lLeap[i - 1] = p[3] == "1"
                }
            }
            loaded = starts.isNotEmpty()
        } catch (e: Exception) {
            loaded = false
        }
    }

    /** Số ngày Julius (Fliegel–Van Flandern) — giống hệt bản JavaScript. */
    fun jdn(year: Int, month: Int, day: Int): Int {
        val a = (14 - month) / 12
        val y = year + 4800 - a
        val m = month + 12 * a - 3
        return day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
    }

    /** Tra tháng âm chứa ngày này; null nếu ngoài khoảng 1900–2100. */
    fun lunarOf(jdn: Int): LunarDay? {
        if (!loaded || starts.isEmpty() || jdn < starts[0]) return null
        var lo = 0
        var hi = starts.size - 1
        var idx = -1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (starts[mid] <= jdn) { idx = mid; lo = mid + 1 } else { hi = mid - 1 }
        }
        if (idx < 0) return null
        return LunarDay(jdn - starts[idx] + 1, lMonth[idx], lYear[idx], lLeap[idx])
    }

    /** Can chi của ngày, suy thẳng từ JDN (không phụ thuộc múi giờ). */
    fun ganZhiOf(jdn: Int): Pair<String, String> {
        if (!loaded) return "" to ""
        val i = (((jdn - ganZhiEpoch) % 60) + 60) % 60
        return CAN[i % 10] to CHI[i % 12]
    }
}
