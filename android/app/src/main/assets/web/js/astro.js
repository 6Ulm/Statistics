/* ════════════════════════════════════════════════════════════════════
   astro.js — Mặt Trời & Mặt Trăng thật tại một toạ độ bất kỳ
   Sun & Moon ephemeris for an arbitrary latitude/longitude.

   Thuật toán / Algorithms:
     • Mặt Trời: Meeus "Astronomical Algorithms" ch. 25 + NOAA solar calculator
     • Mặt Trăng: Meeus ch. 47 (chuỗi rút gọn) + ch. 48 (pha)
     • Mọc/lặn: quét độ cao theo giờ + nội suy tuyến tính điểm cắt

   Độ chính xác thực tế / Practical accuracy:
     • Chính Ngọ (true solar noon)  : < 1 giây
     • Mọc/lặn Mặt Trời             : ~ ±10 giây
     • Mọc/lặn Mặt Trăng            : ~ ±1 phút
     • Pha / độ chiếu sáng Trăng    : ~ ±0.3°

   Toàn bộ tính toán chạy offline, không phụ thuộc mạng.
   ════════════════════════════════════════════════════════════════════ */
(function (root) {
    'use strict';

    var RAD = Math.PI / 180, DEG = 180 / Math.PI;
    var J1970 = 2440587.5, DAY_MS = 86400000;

    function norm360(x) { return ((x % 360) + 360) % 360; }
    function sin(d) { return Math.sin(d * RAD); }
    function cos(d) { return Math.cos(d * RAD); }

    /** Julian Day từ mốc thời gian UTC (ms epoch). */
    function jdFromMs(ms) { return ms / DAY_MS + J1970; }
    function msFromJd(jd) { return (jd - J1970) * DAY_MS; }

    /** Julian Day của 00:00 UTC ngày dương lịch y-m-d. */
    function jdFromUTC(y, m, d, h, mi, s) {
        return jdFromMs(Date.UTC(y, m - 1, d, h || 0, mi || 0, s || 0));
    }

    /** Thế kỷ Julius từ J2000.0 */
    function tCent(jd) { return (jd - 2451545.0) / 36525.0; }

    /* ─────────────────── MẶT TRỜI / SUN ─────────────────── */

    /**
     * Vị trí biểu kiến Mặt Trời.
     * @returns {{lonApp:number, dec:number, ra:number, eot:number, eps:number}}
     *          lonApp = hoàng kinh biểu kiến (độ), dec = xích vĩ (độ),
     *          ra = xích kinh (độ), eot = phương trình thời gian (phút).
     */
    function sunPosition(jd) {
        var T = tCent(jd);
        var L0 = norm360(280.46646 + 36000.76983 * T + 0.0003032 * T * T);
        var M = norm360(357.52911 + 35999.05029 * T - 0.0001537 * T * T);
        var e = 0.016708634 - 0.000042037 * T - 0.0000001267 * T * T;
        var C = (1.914602 - 0.004817 * T - 0.000014 * T * T) * sin(M)
              + (0.019993 - 0.000101 * T) * sin(2 * M)
              + 0.000289 * sin(3 * M);
        var trueLon = L0 + C;
        var omega = 125.04 - 1934.136 * T;
        var lonApp = trueLon - 0.00569 - 0.00478 * sin(omega);
        var eps0 = 23.439291111 - 0.013004167 * T - 0.000000164 * T * T
                 + 0.000000504 * T * T * T;
        var eps = eps0 + 0.00256 * cos(omega);

        var dec = Math.asin(sin(eps) * sin(lonApp)) * DEG;
        var ra = norm360(Math.atan2(cos(eps) * sin(lonApp), cos(lonApp)) * DEG);

        // Phương trình thời gian (Meeus 28.3) — phút
        var y = Math.tan(eps / 2 * RAD); y *= y;
        var eotRad = y * sin(2 * L0)
                   - 2 * e * sin(M)
                   + 4 * e * y * sin(M) * cos(2 * L0)
                   - 0.5 * y * y * sin(4 * L0)
                   - 1.25 * e * e * sin(2 * M);
        return { lonApp: norm360(lonApp), dec: dec, ra: ra, eps: eps, eot: eotRad * 4 * DEG };
    }

    /** Phương trình thời gian (phút) tại thời điểm jd. */
    /**
     * Phương trình thời gian (phút) = giờ Mặt Trời BIỂU KIẾN trừ giờ Mặt Trời
     * TRUNG BÌNH. Đây là thứ duy nhất Chính Ngọ cần từ thiên văn: phần kinh độ
     * trong công thức Chính Ngọ là số học thuần tuý, nên MỘT bảng toàn cục dùng
     * được cho mọi địa điểm.
     *
     * Ưu tiên bảng DE423 (astro_table.js, nội suy bậc ba trên mẫu 4 ngày, sai
     * số 12 ms); không có bảng thì lùi về chuỗi Meeus, sai chừng 1–3 giây.
     *
     * Bảng đánh chỉ số theo TT nên phải cộng ΔT vào JD dân dụng. `dtT` của
     * ShouXingUtil có sẵn khi lunar.js đã nạp; thiếu nó thì bỏ qua, và cái giá
     * là ~16 ms (EoT đổi ~20 giây mỗi ngày, ΔT chừng 70 giây).
     */
    var _triedTable = false;

    function equationOfTime(jd) {
        var g = (typeof window !== 'undefined' ? window : globalThis);
        if (!g.AstroTable && !_triedTable) {
            // Trình duyệt nạp bằng thẻ <script>; trong Node thì tự require, bằng
            // không mỗi công cụ lại phải nhớ nạp đúng thứ tự (xem
            // ShouXingUtil._table trong lunar.js, cùng lý do).
            _triedTable = true;
            try { require('./astro_table.js'); } catch (e) { /* dùng Meeus */ }
        }
        var tab = g.AstroTable;
        if (tab && tab.eot) {
            // lunar.js đặt ShouXingUtil lên window trong trình duyệt, nhưng
            // trong Node thì chỉ xuất qua module.exports — tìm cả hai đường,
            // bằng không Node tra bảng theo UT còn trình duyệt tra theo TT.
            var sx = g.ShouXingUtil;
            if (!sx && typeof require === 'function') {
                try { sx = require('./lunar.js').ShouXingUtil; } catch (e) { sx = null; }
            }
            var dt = (sx && typeof sx.dtT === 'function') ? sx.dtT(jd - 2451545.0) : 0;
            var v = tab.eot(jd - 2451545.0 + dt);
            if (v !== null) return v;
        }
        return sunPosition(jd).eot;
    }

    /**
     * Giờ Mặt Trời thật (True Solar Time) tương ứng một mốc đồng hồ.
     * @param {number} clockMins  phút kể từ 00:00 giờ địa phương
     * @param {number} lon        kinh độ (độ, Đông dương)
     * @param {number} tzHours    lệch múi giờ (giờ, đã tính DST)
     * @param {number} eotMins    phương trình thời gian (phút)
     * @returns {number} phút kể từ 00:00 giờ Mặt Trời thật (chưa mod 1440)
     */
    function toTrueSolarMinutes(clockMins, lon, tzHours, eotMins) {
        return clockMins + (lon - tzHours * 15) * 4 + eotMins;
    }

    /** Chính Ngọ (true solar noon) theo giờ đồng hồ địa phương — phút. */
    function solarNoonMinutes(y, m, d, lon, tzHours) {
        // Lặp 2 vòng: EoT phụ thuộc chính thời điểm chính ngọ.
        var mins = 720 - (lon - tzHours * 15) * 4;
        for (var i = 0; i < 2; i++) {
            var jd = jdFromUTC(y, m, d) + (mins - tzHours * 60) / 1440;
            mins = 720 - (lon - tzHours * 15) * 4 - equationOfTime(jd);
        }
        return mins;
    }

    /**
     * Độ cao Mặt Trời (độ, đã hiệu chỉnh khúc xạ ở chân trời không tính).
     */
    function sunAltitude(jd, lat, lon) {
        var p = sunPosition(jd);
        return altitudeFromEquatorial(jd, p.ra, p.dec, lat, lon);
    }

    /** Giờ sao Greenwich (độ) — Meeus 12.4 */
    function gmst(jd) {
        var T = tCent(jd);
        return norm360(280.46061837 + 360.98564736629 * (jd - 2451545.0)
            + 0.000387933 * T * T - T * T * T / 38710000.0);
    }

    /** Độ cao thiên thể từ xích kinh/xích vĩ. */
    function altitudeFromEquatorial(jd, ra, dec, lat, lon) {
        var H = norm360(gmst(jd) + lon - ra);
        return Math.asin(sin(lat) * sin(dec) + cos(lat) * cos(dec) * cos(H)) * DEG;
    }

    /**
     * Quét mọi thời điểm thiên thể cắt một độ cao mục tiêu trong `span` ngày.
     * @param {function(number):number} altFn  jd → độ cao (độ)
     * @param {number} jdStart  JD của 00:00 giờ địa phương
     * @param {number} h0       độ cao mục tiêu (độ)
     * @param {number} span     số ngày cần quét (mặc định 1)
     * @returns {{rises:number[], sets:number[], alwaysUp:boolean, alwaysDown:boolean}}
     *          rises/sets = phút kể từ 00:00 giờ địa phương (có thể > 1440)
     */
    function findCrossings(altFn, jdStart, h0, span) {
        var STEP = 1 / 48;                 // quét mỗi 30 phút
        var end = span || 1;
        var rises = [], sets = [];
        var prevT = 0, prevA = altFn(jdStart) - h0;
        var maxA = prevA, minA = prevA;

        for (var t = STEP; t <= end + 1e-7; t += STEP) {
            var a = altFn(jdStart + t) - h0;
            if (a > maxA) maxA = a;
            if (a < minA) minA = a;
            if (prevA <= 0 && a > 0) rises.push(refine(altFn, jdStart, prevT, t, h0) * 1440);
            else if (prevA >= 0 && a < 0) sets.push(refine(altFn, jdStart, prevT, t, h0) * 1440);
            prevT = t; prevA = a;
        }
        return {
            rises: rises, sets: sets,
            alwaysUp: !rises.length && !sets.length && minA > 0,
            alwaysDown: !rises.length && !sets.length && maxA < 0
        };
    }

    /** Giao cắt đầu tiên nằm trong ngày [0, 1440) phút, hoặc null. */
    function firstInDay(arr) {
        for (var i = 0; i < arr.length; i++) if (arr[i] < 1440) return arr[i];
        return null;
    }
    /** Giao cắt đầu tiên xảy ra sau mốc `after` phút, hoặc null. */
    function firstAfter(arr, after) {
        for (var i = 0; i < arr.length; i++) if (arr[i] > after) return arr[i];
        return null;
    }

    /** Chia đôi khoảng để tìm điểm cắt chính xác (~0.5 giây). */
    function refine(altFn, jd0, t1, t2, h0) {
        for (var i = 0; i < 24; i++) {
            var tm = (t1 + t2) / 2;
            if ((altFn(jd0 + t1) - h0) * (altFn(jd0 + tm) - h0) <= 0) t2 = tm; else t1 = tm;
        }
        return (t1 + t2) / 2;
    }

    /**
     * Mọc / lặn / chính ngọ Mặt Trời cho một ngày dương lịch tại một toạ độ.
     * Mọi kết quả tính bằng PHÚT kể từ 00:00 giờ đồng hồ địa phương.
     */
    function sunTimes(y, m, d, lat, lon, tzHours) {
        var jdLocalMidnight = jdFromUTC(y, m, d) - tzHours / 24;
        var altFn = function (jd) { return sunAltitude(jd, lat, lon); };
        // Quét 2 ngày: ở vĩ độ cao, mặt trời lặn của "hôm nay" có thể rơi sang
        // sau nửa đêm — lặn phải là lần lặn TIẾP THEO sau lần mọc, không phải
        // lần lặn đầu tiên tính từ 00:00 (vốn thuộc về hôm qua).
        var res = findCrossings(altFn, jdLocalMidnight, -0.833, 2);   // rìa trên + khúc xạ
        var civil = findCrossings(altFn, jdLocalMidnight, -6, 2);
        var noon = solarNoonMinutes(y, m, d, lon, tzHours);
        var jdNoon = jdFromUTC(y, m, d) + (noon - tzHours * 60) / 1440;

        var sunrise = firstInDay(res.rises);
        var sunset = sunrise === null ? firstInDay(res.sets) : firstAfter(res.sets, sunrise);
        var dawn = firstInDay(civil.rises);
        var dusk = dawn === null ? firstInDay(civil.sets) : firstAfter(civil.sets, dawn);

        return {
            sunrise: sunrise, sunset: sunset,
            dawn: dawn, dusk: dusk,
            solarNoon: noon,
            dayLength: (sunrise !== null && sunset !== null) ? sunset - sunrise : null,
            polarDay: res.alwaysUp, polarNight: res.alwaysDown,
            equationOfTime: equationOfTime(jdNoon),
            declination: sunPosition(jdNoon).dec
        };
    }

    /* ─────────────────── MẶT TRĂNG / MOON ─────────────────── */

    /**
     * Vị trí Mặt Trăng (Meeus ch. 47, các số hạng chính).
     * @returns {{lon:number, lat:number, dist:number, ra:number, dec:number, parallax:number}}
     *          lon/lat = hoàng kinh/hoàng vĩ (độ), dist = km.
     */
    function moonPosition(jd) {
        var T = tCent(jd);
        var Lp = norm360(218.3164477 + 481267.88123421 * T - 0.0015786 * T * T
                       + T * T * T / 538841 - T * T * T * T / 65194000);   // kinh độ trung bình
        var D = norm360(297.8501921 + 445267.1114034 * T - 0.0018819 * T * T
                       + T * T * T / 545868);                              // độ giãn
        var M = norm360(357.5291092 + 35999.0502909 * T - 0.0001536 * T * T); // dị thường Mặt Trời
        var Mp = norm360(134.9633964 + 477198.8675055 * T + 0.0087414 * T * T
                       + T * T * T / 69699);                               // dị thường Mặt Trăng
        var F = norm360(93.2720950 + 483202.0175233 * T - 0.0036539 * T * T
                       - T * T * T / 3526000);                             // đối số vĩ độ
        var E = 1 - 0.002516 * T - 0.0000074 * T * T;

        // Σl (1e-6 độ), Σr (1e-3 km) — 24 số hạng lớn nhất của bảng 47.A
        var l = 6288774 * sin(Mp)
              + 1274027 * sin(2 * D - Mp)
              +  658314 * sin(2 * D)
              +  213618 * sin(2 * Mp)
              -  185116 * sin(M) * E
              -  114332 * sin(2 * F)
              +   58793 * sin(2 * D - 2 * Mp)
              +   57066 * sin(2 * D - M - Mp) * E
              +   53322 * sin(2 * D + Mp)
              +   45758 * sin(2 * D - M) * E
              -   40923 * sin(M - Mp) * E
              -   34720 * sin(D)
              -   30383 * sin(M + Mp) * E
              +   15327 * sin(2 * D - 2 * F)
              -   12528 * sin(Mp + 2 * F)
              +   10980 * sin(Mp - 2 * F)
              +   10675 * sin(4 * D - Mp)
              +   10034 * sin(3 * Mp)
              +    8548 * sin(4 * D - 2 * Mp)
              -    7888 * sin(2 * D + M - Mp) * E
              -    6766 * sin(2 * D + M) * E
              -    5163 * sin(D - Mp)
              +    4987 * sin(D + M) * E
              +    4036 * sin(2 * D - M + Mp) * E;

        var r = -20905355 * cos(Mp)
              -  3699111 * cos(2 * D - Mp)
              -  2955968 * cos(2 * D)
              -   569925 * cos(2 * Mp)
              +    48888 * cos(M) * E
              -     3149 * cos(2 * F)
              +   246158 * cos(2 * D - 2 * Mp)
              -   152138 * cos(2 * D - M - Mp) * E
              -   170733 * cos(2 * D + Mp)
              -   204586 * cos(2 * D - M) * E
              -   129620 * cos(M - Mp) * E
              +   108743 * cos(D)
              +   104755 * cos(M + Mp) * E
              +    10321 * cos(2 * D - 2 * F)
              +    79661 * cos(Mp - 2 * F)
              -    34782 * cos(4 * D - Mp)
              -    23210 * cos(3 * Mp)
              -    21636 * cos(4 * D - 2 * Mp)
              +    24208 * cos(2 * D + M - Mp) * E
              +    30824 * cos(2 * D + M) * E
              -     8379 * cos(D - Mp)
              -    16675 * cos(D + M) * E
              -    12831 * cos(2 * D - M + Mp) * E
              -    10445 * cos(2 * D + 2 * Mp);

        // Σb (1e-6 độ) — 16 số hạng lớn nhất của bảng 47.B
        var b = 5128122 * sin(F)
              +  280602 * sin(Mp + F)
              +  277693 * sin(Mp - F)
              +  173237 * sin(2 * D - F)
              +   55413 * sin(2 * D - Mp + F)
              +   46271 * sin(2 * D - Mp - F)
              +   32573 * sin(2 * D + F)
              +   17198 * sin(2 * Mp + F)
              +    9266 * sin(2 * D + Mp - F)
              +    8822 * sin(2 * Mp - F)
              +    8216 * sin(2 * D - M - F) * E
              +    4324 * sin(2 * D - 2 * Mp - F)
              +    4200 * sin(2 * D + Mp + F)
              -    3359 * sin(2 * D + M - F) * E
              +    2463 * sin(2 * D - M - Mp + F) * E
              +    2211 * sin(2 * D - M + F) * E;

        var lambda = norm360(Lp + l / 1000000);
        var beta = b / 1000000;
        var dist = 385000.56 + r / 1000;
        var parallax = Math.asin(6378.14 / dist) * DEG;

        var eps = (23.439291111 - 0.013004167 * T) + 0.00256 * cos(125.04 - 1934.136 * T);
        var ra = norm360(Math.atan2(sin(lambda) * cos(eps) - Math.tan(beta * RAD) * sin(eps),
                                    cos(lambda)) * DEG);
        var dec = Math.asin(sin(beta) * cos(eps) + cos(beta) * sin(eps) * sin(lambda)) * DEG;

        return { lon: lambda, lat: beta, dist: dist, ra: ra, dec: dec, parallax: parallax };
    }

    var PHASE_NAMES = {
        vi: ['Sóc (Trăng mới)', 'Trăng lưỡi liềm đầu', 'Thượng huyền', 'Trăng khuyết đầu',
             'Vọng (Trăng tròn)', 'Trăng khuyết cuối', 'Hạ huyền', 'Trăng lưỡi liềm cuối'],
        zh: ['朔（新月）', '蛾眉月', '上弦月', '盈凸月',
             '望（满月）', '亏凸月', '下弦月', '残月']
    };

    /**
     * Pha Mặt Trăng tại thời điểm jd.
     * @returns {{fraction:number, phase:number, angle:number, phaseIdx:number}}
     *          fraction = tỉ lệ chiếu sáng 0..1
     *          phase    = 0..1 (0 = Sóc, 0.5 = Vọng)
     *          angle    = góc pha (độ)
     */
    function moonIllumination(jd) {
        var s = sunPosition(jd), mo = moonPosition(jd);
        var sunDist = 149598000;                 // km, đủ chính xác cho góc pha
        var phi = Math.acos(cos(mo.lat) * cos(mo.lon - s.lonApp));
        var inc = Math.atan2(sunDist * Math.sin(phi), mo.dist - sunDist * Math.cos(phi));
        var fraction = (1 + Math.cos(inc)) / 2;
        // Dấu của hiệu hoàng kinh quyết định trăng đang tròn dần hay khuyết dần.
        var elong = norm360(mo.lon - s.lonApp);
        var phase = elong / 360;
        return {
            fraction: fraction,
            phase: phase,
            angle: inc * DEG,
            elongation: elong,
            phaseIdx: Math.floor(phase * 8 + 0.5) % 8
        };
    }

    function moonPhaseName(jd, lang) {
        return PHASE_NAMES[lang === 'zh' ? 'zh' : 'vi'][moonIllumination(jd).phaseIdx];
    }

    /**
     * Mọc / lặn Mặt Trăng cho một ngày dương lịch tại một toạ độ.
     * Ngưỡng độ cao: 0.7275·π − 0.34° (bán kính biểu kiến + khúc xạ).
     */
    function moonTimes(y, m, d, lat, lon, tzHours) {
        var jdLocalMidnight = jdFromUTC(y, m, d) - tzHours / 24;
        var altFn = function (jd) {
            var p = moonPosition(jd);
            var h0 = 0.7275 * p.parallax - 0.5667;
            return altitudeFromEquatorial(jd, p.ra, p.dec, lat, lon) - h0;
        };
        // Theo quy ước lịch thiên văn: mọc/lặn Trăng báo theo NGÀY địa phương —
        // có ngày Trăng không mọc, có ngày không lặn (chu kỳ ~24h50m).
        var res = findCrossings(altFn, jdLocalMidnight, 0, 1);
        return {
            moonrise: firstInDay(res.rises), moonset: firstInDay(res.sets),
            alwaysUp: res.alwaysUp, alwaysDown: res.alwaysDown
        };
    }

    /**
     * Thời điểm Sóc / Vọng gần một mốc jd (Meeus ch. 49, có hiệu chỉnh chính).
     * @param {number} jd     mốc tham chiếu
     * @param {number} target 0 = Sóc (new moon), 0.5 = Vọng (full moon)
     * @returns {number} JD (thang thời gian động lực ≈ UT với sai số < 1 phút cho thế kỷ 20–21)
     */
    function nearestPhase(jd, target) {
        var k = Math.round((jd - 2451550.09766) / 29.530588861 - target) + target;
        var T = k / 1236.85;
        var jde = 2451550.09766 + 29.530588861 * k
                + 0.00015437 * T * T - 0.000000150 * T * T * T
                + 0.00000000073 * T * T * T * T;
        var E = 1 - 0.002516 * T - 0.0000074 * T * T;
        var M = norm360(2.5534 + 29.10535670 * k - 0.0000014 * T * T);
        var Mp = norm360(201.5643 + 385.81693528 * k + 0.0107582 * T * T);
        var F = norm360(160.7108 + 390.67050284 * k - 0.0016118 * T * T);

        var corr;
        if (Math.abs(target) < 0.25) {          // Sóc
            corr = -0.40720 * sin(Mp) + 0.17241 * E * sin(M) + 0.01608 * sin(2 * Mp)
                 + 0.01039 * sin(2 * F) + 0.00739 * E * sin(Mp - M)
                 - 0.00514 * E * sin(Mp + M) + 0.00208 * E * E * sin(2 * M)
                 - 0.00111 * sin(Mp - 2 * F) - 0.00057 * sin(Mp + 2 * F)
                 + 0.00056 * E * sin(2 * Mp + M) - 0.00042 * sin(3 * Mp)
                 + 0.00042 * E * sin(M + 2 * F) + 0.00038 * E * sin(M - 2 * F);
        } else {                                 // Vọng
            corr = -0.40614 * sin(Mp) + 0.17302 * E * sin(M) + 0.01614 * sin(2 * Mp)
                 + 0.01043 * sin(2 * F) + 0.00734 * E * sin(Mp - M)
                 - 0.00515 * E * sin(Mp + M) + 0.00209 * E * E * sin(2 * M)
                 - 0.00111 * sin(Mp - 2 * F) - 0.00057 * sin(Mp + 2 * F)
                 + 0.00056 * E * sin(2 * Mp + M) - 0.00042 * sin(3 * Mp)
                 + 0.00042 * E * sin(M + 2 * F) + 0.00038 * E * sin(M - 2 * F);
        }
        var add = 0.000325 * sin(299.77 + 0.107408 * k - 0.009173 * T * T)
                + 0.000165 * sin(251.88 + 0.016321 * k)
                + 0.000164 * sin(251.83 + 26.651886 * k)
                + 0.000126 * sin(349.42 + 36.412478 * k)
                + 0.000110 * sin(84.66 + 18.206239 * k)
                + 0.000062 * sin(141.74 + 53.303771 * k)
                + 0.000060 * sin(207.14 + 2.453732 * k)
                + 0.000056 * sin(154.84 + 7.306860 * k)
                + 0.000047 * sin(34.52 + 27.261239 * k)
                + 0.000042 * sin(207.19 + 0.121824 * k)
                + 0.000040 * sin(291.34 + 1.844379 * k)
                + 0.000037 * sin(161.72 + 24.198154 * k)
                + 0.000035 * sin(239.56 + 25.513099 * k)
                + 0.000023 * sin(331.55 + 3.592518 * k);
        return jde + corr + add - deltaTDays(jde);
    }

    /** ΔT (TT − UT) quy đổi ra ngày — xấp xỉ Espenak & Meeus. */
    function deltaTDays(jd) {
        var y = 2000 + (jd - 2451545.0) / 365.25, t, dt;
        if (y >= 2005 && y < 2050) { t = y - 2000; dt = 62.92 + 0.32217 * t + 0.005589 * t * t; }
        else if (y >= 1986 && y < 2005) {
            t = y - 2000;
            dt = 63.86 + 0.3345 * t - 0.060374 * t * t + 0.0017275 * t * t * t
               + 0.000651814 * t * t * t * t + 0.00002373599 * t * t * t * t * t;
        } else if (y >= 1961 && y < 1986) { t = y - 1975; dt = 45.45 + 1.067 * t - t * t / 260 - t * t * t / 718; }
        else if (y >= 1941 && y < 1961) { t = y - 1950; dt = 29.07 + 0.407 * t - t * t / 233 + t * t * t / 2547; }
        else if (y >= 1920 && y < 1941) { t = y - 1920; dt = 21.20 + 0.84493 * t - 0.076100 * t * t + 0.0020936 * t * t * t; }
        else if (y >= 1900 && y < 1920) { t = y - 1900; dt = -2.79 + 1.494119 * t - 0.0598939 * t * t + 0.0061966 * t * t * t - 0.000197 * t * t * t * t; }
        else if (y >= 2050) { t = y - 2000; dt = 62.92 + 0.32217 * t + 0.005589 * t * t; }
        else { t = (y - 1820) / 100; dt = -20 + 32 * t * t; }
        return dt / 86400;
    }

    /* ─────────────────── TIỆN ÍCH / HELPERS ─────────────────── */

    /** Phút-trong-ngày → "HH:MM" (null → "—"). */
    function fmtMins(mins) {
        if (mins === null || mins === undefined || isNaN(mins)) return '—';
        var t = ((Math.round(mins) % 1440) + 1440) % 1440;
        var h = Math.floor(t / 60), mi = t % 60;
        return (h < 10 ? '0' : '') + h + ':' + (mi < 10 ? '0' : '') + mi;
    }

    root.Astro = {
        jdFromMs: jdFromMs, msFromJd: msFromJd, jdFromUTC: jdFromUTC,
        sunPosition: sunPosition, equationOfTime: equationOfTime,
        solarNoonMinutes: solarNoonMinutes, sunTimes: sunTimes,
        toTrueSolarMinutes: toTrueSolarMinutes,
        moonPosition: moonPosition, moonIllumination: moonIllumination,
        moonPhaseName: moonPhaseName, moonTimes: moonTimes,
        nearestPhase: nearestPhase, deltaTDays: deltaTDays,
        fmtMins: fmtMins
    };
})(typeof window !== 'undefined' ? window : globalThis);
