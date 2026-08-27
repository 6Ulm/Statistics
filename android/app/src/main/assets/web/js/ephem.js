/* ════════════════════════════════════════════════════════════════════
   ephem.js — Lịch thiên văn dùng chung

   MỘT nơi duy nhất tính các mốc thiên văn: điểm Sóc, điểm Vọng, tiết khí,
   Chính Ngọ / Chính Tý, và danh sách tháng âm. Cả tab Kỳ Môn lẫn tab Lịch
   đều gọi vào đây, nên không thể có chuyện hai màn hình cho hai con số.

   Vì sao gom lại:

     • Trước đây phương trình thời gian có HAI bản — getEquationOfTime trong
       app.js và equationOfTime trong astro.js — lệch nhau tới 8,8 giây. Cả
       hai đều dùng để tính Chính Ngọ, mà Chính Ngọ lại định ranh giới Chính
       Tý, tức định mùng 1. Nay chỉ còn một bản.

     • LunarYear.fromYear() của lunar.js chỉ nhớ ĐÚNG MỘT năm (_CACHE_YEAR).
       Mà mỗi lần vẽ, ứng dụng hỏi 3 năm ở mốc quy chiếu (zi_months), rồi hỏi
       lại ở mốc UTC+8 (bảng tiết khí), rồi lại ở mốc địa phương (bảng Âm
       Bàn) — lần nào cũng đá văng lần trước. Ở đây nhớ theo (năm, mốc) nên
       hết cảnh dựng đi dựng lại.

   Khi thay bộ tính thiên văn chính xác hơn, CHỈ cần thay ruột của tệp này.
   ════════════════════════════════════════════════════════════════════ */
(function (root) {
    'use strict';

    /** Bộ nhớ đệm giới hạn số mục, đủ dùng và không phình vô hạn. */
    function lru(limit, compute) {
        var map = new Map();
        return function () {
            var key = Array.prototype.join.call(arguments, '|');
            if (map.has(key)) return map.get(key);
            var v = compute.apply(null, arguments);
            if (map.size >= limit) map.delete(map.keys().next().value);
            map.set(key, v);
            return v;
        };
    }

    /* ─────────────── Mốc múi giờ toàn cục của lunar.js ─────────────── */

    /**
     * Chạy `fn` với ShouXingUtil ở mốc `basis`, rồi trả mốc cũ về nguyên
     * trạng — kể cả khi fn ném lỗi. Biến ấy là biến TOÀN CỤC: quên trả về là
     * mọi phép tính sau đó lệch múi giờ.
     */
    function atBasis(basis, fn) {
        var snap = ShouXingUtil.getTzOffsetHours();
        try {
            ShouXingUtil.setTzOffsetHours(basis);
            return fn();
        } finally {
            ShouXingUtil.setTzOffsetHours(snap);
        }
    }

    /**
     * LunarYear ở một mốc múi giờ, có nhớ. Đây là chỗ tốn kém nhất của cả
     * ứng dụng — dựng một năm mất chừng 10ms.
     *
     * Trả về mảng {jd, month, leap, year} thay vì chính đối tượng LunarYear:
     * đối tượng ấy đọc mốc múi giờ toàn cục ở thời điểm GỌI HÀM, nên nhớ nó
     * rồi dùng lại ở mốc khác thì ra số sai. Mảng thì bất biến.
     */
    var monthsAtBasis = lru(24, function (year, basis) {
        return atBasis(basis, function () {
            var out = [];
            var ms = LunarYear.fromYear(year).getMonths();
            for (var i = 0; i < ms.length; i++) {
                if (ms[i].getYear() !== year) continue;
                out.push({
                    jd: ms[i].getFirstJulianDay(),
                    month: Math.abs(ms[i].getMonth()),
                    leap: ms[i].getMonth() < 0,
                    year: year,
                });
            }
            return out;
        });
    });

    /** 25 mốc tiết khí (chỉ số 0 là Đông Chí năm trước) ở mốc múi giờ `basis`. */
    var jieQiJdAtBasis = lru(24, function (year, basis) {
        return atBasis(basis, function () {
            return LunarYear.fromYear(year).getJieQiJulianDays().slice();
        });
    });

    /* ─────────────── Mặt Trời ─────────────── */

    /**
     * Phương trình thời gian (phút) — BẢN DUY NHẤT của cả ứng dụng.
     * Uỷ quyền cho astro.js, nơi có sẵn nghiệm Mặt Trời theo Meeus.
     */
    function eotMinutes(jdUTC) {
        return root.Astro.equationOfTime(jdUTC);
    }

    /**
     * Chính Ngọ: số phút kể từ 00:00 giờ đồng hồ địa phương.
     * Lặp hai vòng vì phương trình thời gian phụ thuộc chính thời điểm ấy.
     */
    function solarNoonMinutes(y, m, d, lon, tz) {
        return root.Astro.solarNoonMinutes(y, m, d, lon, tz);
    }

    /** Chính Tý (nửa đêm Mặt Trời thật): Chính Ngọ − 12h, có thể âm. */
    function solarMidnightMinutes(y, m, d, lon, tz) {
        return solarNoonMinutes(y, m, d, lon, tz) - 720;
    }

    /** Đầu giờ Tý (mốc đổi TRỤ NGÀY): Chính Ngọ − 13h. */
    function ziHourStartMinutes(y, m, d, lon, tz) {
        return solarNoonMinutes(y, m, d, lon, tz) - 780;
    }

    /* ─────────────── Mặt Trăng ─────────────── */

    /**
     * Điểm Sóc chứa/gần một ngày đã làm tròn, trả Solar ở mốc `basis`.
     *
     * Gọi thẳng ShouXingUtil.shuoHigh chứ không chép lại công thức: shuoHigh
     * có thêm một bước mà bản chép trước đây bỏ mất — khi điểm Sóc rơi trong
     * vòng 30 phút quanh nửa đêm, nó giải lại bằng msaLonT (chính xác hơn
     * msaLonT2). Sát nửa đêm chính là lúc quyết định mùng 1 rơi ngày nào.
     */
    function socSolar(roundedSolar, basis) {
        var k = Math.round(
            (roundedSolar.getJulianDay() + 0.5 - Solar.J2000) / 29.5306);
        return Solar.fromJulianDay(
            ShouXingUtil.shuoHigh(k * 2 * Math.PI, basis) + Solar.J2000);
    }

    /**
     * Điểm Vọng (trăng tròn) của tuần trăng chứa `roundedSolar`.
     * Vọng là lúc hiệu kinh độ Mặt Trăng − Mặt Trời = 180°, tức nửa chu kỳ
     * sau Sóc — cùng nghiệm shuoHigh, chỉ lệch pha π.
     */
    function vongSolar(roundedSolar, basis) {
        var k = Math.round(
            (roundedSolar.getJulianDay() + 0.5 - Solar.J2000) / 29.5306);
        return Solar.fromJulianDay(
            ShouXingUtil.shuoHigh(k * 2 * Math.PI + Math.PI, basis) + Solar.J2000);
    }

    root.Ephem = {
        atBasis: atBasis,
        monthsAtBasis: monthsAtBasis,
        jieQiJdAtBasis: jieQiJdAtBasis,
        eotMinutes: eotMinutes,
        solarNoonMinutes: solarNoonMinutes,
        solarMidnightMinutes: solarMidnightMinutes,
        ziHourStartMinutes: ziHourStartMinutes,
        socSolar: socSolar,
        vongSolar: vongSolar,
    };
})(typeof window !== 'undefined' ? window : globalThis);
