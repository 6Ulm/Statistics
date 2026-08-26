package com.bazi.qimen

import android.content.Context
import java.util.TimeZone

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

    /** Một mốc tiết khí: tên tiếng Việt, ngày, và phút kể từ 00:00 giờ địa phương. */
    data class JieQi(val name: String, val jdn: Int, val minutes: Int)

    val TIET_KHI = arrayOf(
        "Đông Chí", "Tiểu Hàn", "Đại Hàn", "Lập Xuân", "Vũ Thủy", "Kinh Trập",
        "Xuân Phân", "Thanh Minh", "Cốc Vũ", "Lập Hạ", "Tiểu Mãn", "Mang Chủng",
        "Hạ Chí", "Tiểu Thử", "Đại Thử", "Lập Thu", "Xử Thử", "Bạch Lộ",
        "Thu Phân", "Hàn Lộ", "Sương Giáng", "Lập Đông", "Tiểu Tuyết", "Đại Tuyết"
    )

    private var socMin: IntArray = IntArray(0)
    private var jqJdn: IntArray = IntArray(0)
    private var jqMin: IntArray = IntArray(0)
    private var jqIdx: IntArray = IntArray(0)

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
                socMin = IntArray(n)
                for (i in 1..n) {
                    val p = lines[i].split(' ')
                    starts[i - 1] = p[0].toInt()      // JDN điểm Sóc ở mốc UTC+7
                    socMin[i - 1] = p[1].toInt()      // phút trong ngày, mốc UTC+7
                    lYear[i - 1] = p[2].toInt()
                    lMonth[i - 1] = p[3].toInt()
                    lLeap[i - 1] = p[4] == "1"
                }
            }
            context.assets.open("jieqi.txt").bufferedReader().use { reader ->
                val lines = reader.readLines().filter { it.isNotBlank() }
                jqJdn = IntArray(lines.size)
                jqMin = IntArray(lines.size)
                jqIdx = IntArray(lines.size)
                lines.forEachIndexed { i, line ->
                    val p = line.split(' ')
                    jqJdn[i] = p[0].toInt()
                    jqMin[i] = p[1].toInt()
                    jqIdx[i] = p[2].toInt()
                }
            }
            loaded = starts.isNotEmpty()
        } catch (e: Exception) {
            loaded = false
        }
    }

    /**
     * Các tiết khí rơi vào một tháng dương lịch, đã sắp theo thời gian.
     * Bảng sắp sẵn theo JDN nên chỉ cần quét đoạn giữa hai mốc đầu/cuối tháng.
     */
    fun jieQiOfMonth(year: Int, month: Int): List<JieQi> {
        if (!loaded || jqJdn.isEmpty()) return emptyList()
        val from = jdn(year, month, 1)
        val to = from + 40
        val out = ArrayList<JieQi>(2)
        var lo = 0
        var hi = jqJdn.size - 1
        var start = jqJdn.size
        while (lo <= hi) {                      // mốc đầu tiên ≥ from
            val mid = (lo + hi) ushr 1
            if (jqJdn[mid] >= from) { start = mid; hi = mid - 1 } else lo = mid + 1
        }
        var i = start
        while (i < jqJdn.size && jqJdn[i] <= to) {
            val (cy, cm, _) = civilOf(jqJdn[i])
            if (cy == year && cm == month) {
                out.add(JieQi(TIET_KHI[jqIdx[i]], jqJdn[i], jqMin[i]))
            }
            i++
        }
        return out
    }

    /**
     * 24 tiết khí của "năm tiết khí" chứa ngày `ref` — đúng dãy mà bảng Sách Bổ
     * pháp và tab Lịch hiện: bắt đầu từ **Đông Chí** rồi chạy liền 24 mục tới
     * Đại Tuyết năm sau.
     *
     * Bảng jieqi.txt vốn đã là dãy phẳng đã sắp xếp, nên chỉ cần tìm mốc Đông
     * Chí (chỉ số tên 0) gần nhất không muộn hơn `ref` rồi lấy 24 mục kế tiếp.
     */
    fun jieQiYearOf(ref: Int): List<JieQi> {
        if (!loaded || jqJdn.isEmpty()) return emptyList()
        // mốc cuối cùng có jdn ≤ ref
        var lo = 0
        var hi = jqJdn.size - 1
        var at = -1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (jqJdn[mid] <= ref) { at = mid; lo = mid + 1 } else hi = mid - 1
        }
        if (at < 0) return emptyList()
        // lùi về Đông Chí gần nhất — cùng lắm 23 bước
        var start = at
        while (start >= 0 && jqIdx[start] != 0) start--
        if (start < 0 || start + 23 >= jqJdn.size) return emptyList()
        return (start until start + 24).map {
            JieQi(TIET_KHI[jqIdx[it]], jqJdn[it], jqMin[it])
        }
    }

    /**
     * Đổi một mốc tiết khí sang giờ của múi giờ khác.
     *
     * jieqi.txt lưu giờ **địa phương ở UTC+7** (xem build_lunar_table.mjs).
     * Bản thân mốc giao tiết là một thời điểm tuyệt đối, chỉ cách hiển thị mới
     * đổi theo nơi xem — nên tab Lịch hiện giờ của địa điểm đang chọn thì widget
     * cũng phải vậy, nếu không hai chỗ lệch nhau tới mấy tiếng cho cùng một
     * tiết khí. TimeZone.getOffset() tra đúng theo từng thời điểm nên giờ mùa
     * đông của nước có DST không bị cộng nhầm offset mùa hè.
     */
    fun localize(item: JieQi, tz: TimeZone): JieQi {
        val utcMs = (item.jdn - 2440588L) * 86_400_000L +
            item.minutes * 60_000L - 7L * 3_600_000L
        val local = utcMs + tz.getOffset(utcMs)
        // Chia làm tròn XUỐNG, tự viết: `/` của Kotlin cắt về 0 nên ngày trước
        // 1970 sẽ lệch một ngày, mà bảng chạy từ 1900. Math.floorDiv chỉ có từ
        // API 24 — minSdk vừa đúng 24, nhưng khỏi phụ thuộc cho chắc.
        var days = local / 86_400_000L
        var rem = local % 86_400_000L
        if (rem < 0) { days -= 1; rem += 86_400_000L }
        return JieQi(item.name, (days + 2440588L).toInt(), (rem / 60_000L).toInt())
    }

    /** Số ngày Julius (Fliegel–Van Flandern) — giống hệt bản JavaScript. */
    fun jdn(year: Int, month: Int, day: Int): Int {
        val a = (14 - month) / 12
        val y = year + 4800 - a
        val m = month + 12 * a - 3
        return day + (153 * m + 2) / 5 + 365 * y + y / 4 - y / 100 + y / 400 - 32045
    }

    /** Ngày dương lịch từ số ngày Julius — phép nghịch của [jdn]. */
    fun civilOf(jdn: Int): Triple<Int, Int, Int> {
        val a = jdn + 32044
        val b = (4 * a + 3) / 146097
        val c = a - 146097 * b / 4
        val d = (4 * c + 3) / 1461
        val e = c - 1461 * d / 4
        val m = (5 * e + 2) / 153
        val day = e - (153 * m + 2) / 5 + 1
        val month = m + 3 - 12 * (m / 10)
        val year = 100 * b + d - 4800 + m / 10
        return Triple(year, month, day)
    }

    /** Tra tháng âm chứa ngày này; null nếu ngoài khoảng 1900–2100. */
    /**
     * Bảng tháng âm do CHÍNH ỨNG DỤNG ghi ra (xem publishLunarCache trong
     * calendar.js): "<múi giờ phút>|<JDN mùng 1>,<tháng>,<nhuận>;…".
     *
     * Bảng đóng trong APK chỉ đúng tuyệt đối ở mốc UTC+7; suy sang múi giờ khác
     * bằng điểm Sóc còn lệch ~0,35% số tháng vì lunar.js không định mùng 1
     * thuần tuý bằng phép lấy phần nguyên. Ứng dụng thì có lunar.js, nên nó
     * ghi sẵn vài chục tháng quanh hôm nay ở đúng múi giờ đang chọn — widget
     * đọc cái này trước, hết tầm với mới quay về bảng đóng sẵn.
     */
    private var cacheTzMin: Int = Int.MIN_VALUE
    private var cacheStart: IntArray = IntArray(0)
    private var cacheMonth: IntArray = IntArray(0)
    private var cacheLeap: BooleanArray = BooleanArray(0)

    @Synchronized
    fun loadAppCache(context: Context) {
        val raw = context.getSharedPreferences("qmdj_prefs", Context.MODE_PRIVATE)
            .getString("qmdj.lunarCache", null)
        if (raw.isNullOrEmpty()) { cacheStart = IntArray(0); return }
        try {
            val bar = raw.indexOf('|')
            if (bar <= 0) { cacheStart = IntArray(0); return }
            cacheTzMin = raw.substring(0, bar).toInt()
            val parts = raw.substring(bar + 1).split(';').filter { it.isNotEmpty() }
            val st = IntArray(parts.size); val mo = IntArray(parts.size)
            val lp = BooleanArray(parts.size)
            parts.forEachIndexed { i, row ->
                val f = row.split(',')
                st[i] = f[0].toInt(); mo[i] = f[1].toInt(); lp[i] = f[2] == "1"
            }
            // Phải tăng dần thì tìm nhị phân mới đúng; ứng dụng ghi theo thứ tự
            // thời gian rồi, nhưng đây là dữ liệu ngoài nên vẫn kiểm.
            for (i in 1 until st.size) if (st[i] <= st[i - 1]) { cacheStart = IntArray(0); return }
            cacheStart = st; cacheMonth = mo; cacheLeap = lp
        } catch (e: Exception) {
            cacheStart = IntArray(0)
        }
    }

    /** Tra trong bảng của ứng dụng; trả null nếu không phủ tới ngày này. */
    private fun fromAppCache(jdn: Int, tzMin: Int): LunarDay? {
        if (cacheStart.size < 2 || cacheTzMin != tzMin) return null
        // Ngày nằm ngoài đoạn bảng phủ thì không đoán — để bảng đóng sẵn lo.
        if (jdn < cacheStart[0] || jdn >= cacheStart[cacheStart.size - 1]) return null
        var lo = 0; var hi = cacheStart.size - 1; var idx = -1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (cacheStart[mid] <= jdn) { idx = mid; lo = mid + 1 } else hi = mid - 1
        }
        if (idx < 0) return null
        return LunarDay(jdn - cacheStart[idx] + 1, cacheMonth[idx], 0, cacheLeap[idx])
    }

    fun lunarOf(jdn: Int, tz: TimeZone = TimeZone.getTimeZone("Asia/Ho_Chi_Minh")): LunarDay? {
        fromAppCache(jdn, tz.getOffset(jdnToMillis(jdn)) / 60000)?.let { return it }
        if (!loaded || starts.isEmpty()) return null
        // Mùng 1 = ngày CHỨA điểm Sóc, xét ở múi giờ đang xem. Bảng ghi điểm
        // Sóc ở mốc UTC+7 nên phải quy đổi trước rồi mới so — chốt cứng mùng 1
        // trong bảng thì ở Paris widget lệch ứng dụng đúng một ngày.
        var lo = 0
        var hi = starts.size - 1
        var idx = -1
        while (lo <= hi) {
            val mid = (lo + hi) ushr 1
            if (monthStart(mid, tz) <= jdn) { idx = mid; lo = mid + 1 } else { hi = mid - 1 }
        }
        if (idx < 0) return null
        return LunarDay(jdn - monthStart(idx, tz) + 1, lMonth[idx], lYear[idx], lLeap[idx])
    }

    /** Mốc epoch (ms) của 12:00 trưa ngày JDN — đủ để tra offset/DST. */
    private fun jdnToMillis(jdn: Int): Long =
        (jdn - 2440588L) * 86_400_000L + 43_200_000L

    /** Ngày dương (JDN) của mùng 1 tháng thứ `i`, xét ở múi giờ `tz`. */
    private fun monthStart(i: Int, tz: TimeZone): Int =
        localize(JieQi("", starts[i], socMin[i]), tz).jdn

    /** Can chi của ngày, suy thẳng từ JDN (không phụ thuộc múi giờ). */
    fun ganZhiOf(jdn: Int): Pair<String, String> {
        if (!loaded) return "" to ""
        val i = (((jdn - ganZhiEpoch) % 60) + 60) % 60
        return CAN[i % 10] to CHI[i % 12]
    }
}
