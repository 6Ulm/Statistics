/* ════════════════════════════════════════════════════════════════════
   core.js — BỘ TÍNH DÙNG CHUNG của cả bốn tab

   Luật của tệp này, ngắn gọn: MỌI con số mà từ hai tab trở lên cùng nhìn
   thấy đều phải tính Ở ĐÂY, đúng một lần. Bốn tab (Kỳ Môn · Lịch · Bát Tự ·
   Tra cứu) chỉ còn việc HIỆN ra — không tab nào được tự tính lại một thứ mà
   tệp này đã có, dù chỉ là đọc lại ô ngày giờ.

   Vì sao phải có luật ấy: trước đây ba tab mỗi tab tự đọc #inYear/#inMonth/
   #inDay/#solarHour/#solarMinute rồi tự tra countryData và tự gọi
   getTimezoneOffset — ba bản chép gần giống nhau, và "gần giống" là chỗ sinh
   ra hai màn hình cho một con số. tab Bát Tự còn tự gọi thẳng
   ShouXingUtil.qiAccurate để lấy mốc tiết khí, trong khi tab Lịch lấy cùng mốc
   ấy qua Ephem — hai đường, hai bộ nhớ đệm, và không gì bảo đảm chúng khớp.

   XẾP TẦNG (dưới không biết gì về trên):

     lunar.js      thư viện lịch Trung Hoa, không sửa
     astro.js      nghiệm Mặt Trời theo Meeus
     ephem.js      MỐC THIÊN VĂN: Sóc, Vọng, tiết khí, Chính Ngọ/Chính Tý
     core.js       SỰ KIỆN LỊCH PHÁP: ngày âm, bốn trụ, tiết khí đang giữ,
                   thần sát suy từ trụ   ← tệp này
     nguhanh.js    NGỮ NGHĨA can chi: ngũ hành, tàng can, thập thần, tuần không
     *.js của tab  chỉ dựng HTML

   `chart()` nhớ theo đúng bộ tham số vào, nên bốn tab gọi bốn lần trong một
   lần vẽ vẫn chỉ tính một lần.
   ════════════════════════════════════════════════════════════════════ */
(function (root) {
    'use strict';

    var CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
    var CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];
    var CAN_VI = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    var CHI_VI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];

    var TK_VI = [
    'Đông Chí','Tiểu Hàn','Đại Hàn',
    'Lập Xuân','Vũ Thủy','Kinh Trập',
    'Xuân Phân','Thanh Minh','Cốc Vũ',
    'Lập Hạ','Tiểu Mãn','Mang Chủng',
    'Hạ Chí','Tiểu Thử','Đại Thử',
    'Lập Thu','Xử Thử','Bạch Lộ',
    'Thu Phân','Hàn Lộ','Sương Giáng',
    'Lập Đông','Tiểu Tuyết','Đại Tuyết'
];
    var TK_ZH = [
    '冬至','小寒','大寒',
    '立春','雨水','惊蛰',
    '春分','清明','谷雨',
    '立夏','小满','芒种',
    '夏至','小暑','大暑',
    '立秋','处暑','白露',
    '秋分','寒露','霜降',
    '立冬','小雪','大雪'
];

    /** Tên tiết khí theo ngôn ngữ, tra từ tên chữ Hán. */
    var _tkIdx = {};
    for (var _i = 0; _i < TK_ZH.length; _i++) _tkIdx[TK_ZH[_i]] = _i;
    function tietKhiTen(zhName, isZH) {
        var k = _tkIdx[zhName];
        return (k === undefined) ? zhName : (isZH ? TK_ZH[k] : TK_VI[k]);
    }

    function pad2(n) { return (n < 10 ? '0' : '') + n; }

    /* ─────────────── Ô NGÀY GIỜ + VỊ TRÍ: đọc MỘT lần ───────────────
     *
     * Bốn tab dùng chung đúng một hàng điều khiển (xem #bottomDock), nên cũng
     * phải dùng chung đúng một cách đọc nó. Trả về null khi trang chưa dựng
     * xong — mọi tab đều phải chịu được điều đó vì thứ tự nạp script không
     * bảo đảm tab nào vẽ trước.
     */
    /**
     * Toạ độ + múi giờ của địa điểm đang chọn ở hàng dùng chung dưới đáy.
     *
     * Tách khỏi input() vì tab Lịch và tab Tra cứu cần ĐỊA ĐIỂM mà không cần
     * ngày giờ sinh: lịch tra theo tháng đang xem, Tra cứu theo năm đang
     * chọn. Trước đây mỗi tab tự viết lại đúng bốn dòng này — bốn bản gần
     * giống nhau là bốn chỗ để lệch.
     */
    function location() {
        if (typeof getDOM !== 'function' || typeof countryData === 'undefined') return null;
        var el = getDOM('country');
        var info = el ? countryData[el.value] : null;
        if (!info || !info.tzId) return null;
        return { info: info, tzId: info.tzId, lon: info.lon };
    }

    /**
     * Offset UTC (giờ) của địa điểm đang chọn TẠI một ngày dương lịch — hỏi
     * theo từng thời điểm nên đúng cả ở nước có giờ mùa hè. Không có địa
     * điểm thì 7 (giờ Việt Nam), đúng mặc định cũ của cả ba tab.
     */
    function tzAt(y, m, d, h) {
        var loc = location();
        if (!loc) return 7;
        return getTimezoneOffset(loc.tzId, new Date(y, m - 1, d, h === undefined ? 12 : h));
    }

    function input() {
        if (typeof getDOM !== 'function' || typeof countryData === 'undefined') return null;
        var el = function (id) { return getDOM(id); };
        if (!el('inYear') || !el('country')) return null;
        var y = parseInt(el('inYear').value, 10);
        var m = parseInt(el('inMonth').value, 10);
        var d = parseInt(el('inDay').value, 10);
        var h = parseInt(el('solarHour').value, 10);
        var mi = parseInt(el('solarMinute').value, 10);
        if (!isFinite(y) || !isFinite(m) || !isFinite(d)) return null;
        var loc = location();
        if (!loc) return null;
        h = isFinite(h) ? h : 0;
        mi = isFinite(mi) ? mi : 0;
        return {
            y: y, m: m, d: d, h: h, mi: mi,
            info: loc.info, tzId: loc.tzId, lon: loc.lon,
            tz: tzAt(y, m, d, h),
        };
    }

    /* ─────────────── Quy đổi mốc UTC+8 → giờ địa phương ───────────────
     *
     * jdUTC8 là một THỜI ĐIỂM UTC cố định (jdUTC = jdUTC8 − 8/24). Offset của
     * tzId lấy tại CHÍNH thời điểm ấy bằng Intl.DateTimeFormat, nên đúng cả
     * khi mốc rơi trúng giờ chuyển DST — chứ không suy từ "ngày UTC+8 đã làm
     * tròn", vốn lệch được 1 giờ ở những múi cách UTC+8 sáu, chín tiếng.
     */
    function tzOffsetAtJdUTC8(jdUTC8, tzId) {
        var unixMs = (jdUTC8 - 8 / 24 - 2440587.5) * 86400000;
        return getTimezoneOffset(tzId, new Date(unixMs));
    }
    function localSolarFromJdUTC8(jdUTC8, tzId) {
        return Solar.fromJulianDay(jdUTC8 + (tzOffsetAtJdUTC8(jdUTC8, tzId) - 8) / 24);
    }
    /** "DD-MM-YYYY HH:MM" của một mốc UTC+8, theo giờ địa phương. */
    function fmtJdUTC8Local(jdUTC8, tzId) {
        var s = localSolarFromJdUTC8(jdUTC8, tzId);
        return pad2(s.getDay()) + '-' + pad2(s.getMonth()) + '-' + s.getYear()
            + ' ' + pad2(s.getHour()) + ':' + pad2(s.getMinute());
    }

    /* ─────────────── CHÍNH NGỌ ───────────────
     *
     * Một con số, ba chỗ dùng: hộp thông tin tab Kỳ Môn hiện nó; ranh giới
     * ngày âm lịch (Chính Tý = Chính Ngọ − 12h) dựng trên nó; và trụ ngày
     * (đầu giờ Tý = Chính Ngọ − 13h) cũng vậy. Ba chỗ mà hai công thức là
     * ngày âm với giờ hiện ra cãi nhau — đã từng xảy ra, xem ghi chú đầu
     * ephem.js.
     */
    function chinhNgo(inp) {
        var lonOffsetMins = (inp.lon - inp.tz * 15) * 4;
        var minutes = Ephem.solarNoonMinutes(inp.y, inp.m, inp.d, inp.lon, inp.tz);
        return {
            minutes: minutes,
            lonOffsetMins: lonOffsetMins,
            // Suy ngược phương trình thời gian từ chính con số đã dùng, thay
            // vì gọi lại hàm EoT — hai đường thì hai kết quả làm tròn.
            eotMins: 720 - lonOffsetMins - minutes,
            hhmm: pad2(Math.floor(minutes / 60)) + ':' + pad2(Math.floor(minutes % 60)),
            gmt: 'GMT' + (inp.tz >= 0 ? '+' : '') + inp.tz,
        };
    }

    /* ─────────────── NGÀY ÂM LỊCH (ranh giới Chính Tý) ───────────────
     * Chuyển nguyên khối từ app.js — xem khối ghi chú "RANH GIỚI NGÀY ÂM
     * LỊCH: CHÍNH TÝ THIÊN VĂN" ở đó về lý do chọn nửa đêm THẬT.
     */
    function dayOf(local, lon, tzId) {
        var y = local.getYear(), m = local.getMonth(), d = local.getDay();
        var tz = getTimezoneOffset(tzId, new Date(y, m - 1, d, 12));
        var t = local.getHour() * 60 + local.getMinute();
        var shift = Math.floor((t - Ephem.solarMidnightMinutes(y, m, d, lon, tz)) / 1440);
        if (shift === 0) return { y: y, m: m, d: d };
        var dt = new Date(y, m - 1, d + shift);
        return { y: dt.getFullYear(), m: dt.getMonth() + 1, d: dt.getDate() };
    }

    /** Số ngày Julius của một ngày dương lịch (Fliegel–Van Flandern). */
    function jdn(y, m, d) {
        var a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
        return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
            + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
    }

    /**
     * Kinh tuyến quy chiếu cho NHÃN tháng: UTC+7 cho lịch ta, UTC+8 cho lịch
     * Tàu. Chính chỗ khác nhau này làm Tết ta và Tết Tàu thỉnh thoảng lệch
     * một ngày.
     */
    function labelBasis() {
        return (typeof currentLang !== 'undefined' && currentLang === 'zh') ? 8 : 7;
    }

    /** Mùng 1: ngày (Chính Tý → Chính Tý) chứa điểm Sóc. */
    function mung1FromJd(firstJd, lon, tzId) {
        var socUTC8 = Ephem.socSolar(Solar.fromJulianDay(firstJd), 8);
        return dayOf(localSolarFromJdUTC8(socUTC8.getJulianDay(), tzId), lon, tzId);
    }

    /**
     * Danh sách tháng âm quanh một năm dương: NHÃN lấy ở mốc quy chiếu, MỐC
     * BẮT ĐẦU neo theo Chính Tý địa phương. Trả mảng {jdn, month, year, leap}
     * tăng dần.
     *
     * Vì sao tách đôi như vậy:
     *
     *   • "Tháng này là tháng mấy, tháng nào nhuận" là QUY ƯỚC LỊCH, không
     *     phải sự kiện thiên văn tại chỗ người dùng đứng. Nó do luật "tháng
     *     không có trung khí là tháng nhuận" quyết, và luật ấy được định tại
     *     kinh tuyến quy chiếu. Lấy nhãn ở đó thì số tháng khớp lịch in, và
     *     xác định — không trôi.
     *
     *   • "Mùng 1 rơi vào ngày dương nào" thì mới là chuyện địa phương: ngày
     *     chứa điểm Sóc, đếm từ Chính Tý.
     *
     * Trước đây hỏi lunar.js ở ngay mốc địa phương, tức để chính luật trung
     * khí bị đánh giá trên lưới nửa đêm ĐỒNG HỒ ở một offset nguyên giờ. Mà
     * Chính Tý lại xê dịch tới ~30 phút trong năm theo phương trình thời gian,
     * nên một offset cố định không diễn tả nổi nó: đo ra chừng 4–8 tháng mỗi
     * thế kỷ đổi nhãn chỉ vì mốc lệch 15–30 phút. Nay nhãn không còn phụ thuộc
     * chuyện đó nữa.
     *
     * Ghép nhãn với mốc bắt đầu là an toàn: dãy tuần trăng giống hệt nhau ở
     * mọi mốc — đã kiểm 1900–2100, mọi mốc từ UTC−8 tới UTC+12 đều ra ĐÚNG
     * 2486 tháng, mốc bắt đầu lệch tối đa 1 ngày, không cặp nào lệch quá.
     */
    var _monthCache = new Map();
    function months(gregYear, lon, tzId) {
        var basis = labelBasis();
        var key = gregYear + '|' + tzId + '|' + lon + '|' + basis;
        if (_monthCache.has(key)) return _monthCache.get(key);
        var out = [];
        for (var ly = gregYear - 1; ly <= gregYear + 1; ly++) {
            var ms = Ephem.monthsAtBasis(ly, basis);
            for (var i = 0; i < ms.length; i++) {
                var g = mung1FromJd(ms[i].jd, lon, tzId);
                out.push({
                    jdn: jdn(g.y, g.m, g.d),
                    month: ms[i].month, year: ly, leap: ms[i].leap,
                });
            }
        }
        out.sort(function (a, b) { return a.jdn - b.jdn; });
        if (_monthCache.size > 24) _monthCache.clear();
        _monthCache.set(key, out);
        return out;
    }

    /** Ngày âm lịch của một ngày dương, theo ranh giới Chính Tý. */
    function lunarOf(y, m, d, lon, tzId) {
        var list = months(y, lon, tzId);
        var j = jdn(y, m, d), at = -1;
        for (var i = 0; i < list.length; i++) {
            if (list[i].jdn <= j) at = i; else break;
        }
        if (at < 0) return null;
        return {
            day: j - list[at].jdn + 1, month: list[at].month,
            year: list[at].year, leap: list[at].leap,
        };
    }

    /* ─────────────── TIẾT KHÍ đang giữ ─────────────── */

    /** Giờ Bắc Kinh (UTC+8) của một thời điểm giờ địa phương. */
    function solarBJof(inp) {
        var dUTC = new Date(Date.UTC(inp.y, inp.m - 1, inp.d, inp.h, inp.mi)
            - inp.tz * 3600000 + 8 * 3600000);
        return {
            solarBJ: Solar.fromYmdHms(
                dUTC.getUTCFullYear(), dUTC.getUTCMonth() + 1, dUTC.getUTCDate(),
                dUTC.getUTCHours(), dUTC.getUTCMinutes(), dUTC.getUTCSeconds()),
            dUTC: dUTC,
        };
    }
    function cmpNum(s) {
        return parseInt('' + s.getYear() + pad2(s.getMonth()) + pad2(s.getDay())
            + pad2(s.getHour()) + pad2(s.getMinute()), 10);
    }
    /** Tiết khí GẦN NHẤT ĐÃ QUA của một thời điểm (mốc UTC+8). */
    function prevJieQi(solarBJ, dUTC) {
        var jqObj = solarBJ.getLunar().getPrevJieQi(true);
        var jqSolar = jqObj.getSolar();
        if (cmpNum(solarBJ) < cmpNum(jqSolar)) {
            var prev = new Date(dUTC.getTime() - 86400000);
            jqObj = Solar.fromYmdHms(
                prev.getUTCFullYear(), prev.getUTCMonth() + 1, prev.getUTCDate(),
                prev.getUTCHours(), prev.getUTCMinutes(), prev.getUTCSeconds()
            ).getLunar().getPrevJieQi(true);
            jqSolar = jqObj.getSolar();
        }
        return { jqObj: jqObj, jqSolar: jqSolar };
    }

    /* ─────────────── THẦN SÁT SUY TỪ TRỤ ───────────────
     *
     * Ba bảng 60 dòng trong app.js (dayToTuanThu, hoaGiapToKhongVong,
     * hoaGiapToDichMa) đều chỉ là MỘT phép tính viết bung ra. Chép tay 60 dòng
     * là 60 chỗ gõ nhầm được, mà nhầm một dòng thì chỉ sai đúng một ngày trong
     * sáu mươi — không ai thấy. Nay tính, và test_core đối chiếu lại với đúng
     * ba bảng cũ ấy để chuyển đổi không lặng lẽ đổi con số nào.
     */

    /** Chỉ số tuần (0 = Giáp Tý … 5 = Giáp Dần) của một trụ. */
    function tuanIdx(ganIdx, chiIdx) {
        var đầu = ((chiIdx - ganIdx) % 12 + 12) % 12;   // chi của trụ Giáp mở tuần
        return ((12 - đầu) % 12) / 2;
    }
    var TUAN_THU = ['Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    /** Tuần thủ (can độn) của tuần chứa trụ — Giáp Tý tuần thì Mậu, … */
    function tuanThuOf(ganIdx, chiIdx) { return TUAN_THU[tuanIdx(ganIdx, chiIdx)]; }
    /** Trụ Giáp mở tuần, dạng "Giáp Tý". */
    function tuanGiapOf(ganIdx, chiIdx) {
        return 'Giáp ' + CHI_VI[((chiIdx - ganIdx) % 12 + 12) % 12];
    }
    /**
     * Dịch mã chỉ phụ thuộc CHI, theo tam hợp: Dần Ngọ Tuất → Thân,
     * Thân Tý Thìn → Dần, Tỵ Dậu Sửu → Hợi, Hợi Mão Mùi → Tỵ. Bốn nhóm ấy
     * chính là bốn lớp thặng dư của chi theo mod 4.
     */
    var DICH_MA = [2, 11, 8, 5];
    function dichMaOf(chiIdx) { return DICH_MA[((chiIdx % 4) + 4) % 4]; }
    /** Hai chi không vong của tuần chứa trụ — uỷ quyền cho nguhanh.js. */
    function khongVongOf(ganIdx, chiIdx) {
        return root.NguHanh.tuanKhongOf(CAN_VI[ganIdx], CHI_VI[chiIdx]);
    }

    /* ─────────────── LÁ SỐ ───────────────
     *
     * Trình tự dưới đây KHÔNG được xáo: mốc múi giờ toàn cục của lunar.js
     * (_tzOffsetHours) quyết định cả mốc Sóc lẫn mốc Lập Xuân, nên "tính gì ở
     * mốc nào" là một phần của phép tính chứ không phải chi tiết kỹ thuật.
     * Từng bước kèm ghi chú vì sao nó phải đứng đúng chỗ ấy.
     */
    var _chartCache = new Map();
    function chart(inp) {
        var key = [inp.y, inp.m, inp.d, inp.h, inp.mi, inp.tzId, inp.lon, inp.tz,
                   labelBasis()].join('|');
        if (_chartCache.has(key)) return _chartCache.get(key);

        var cn = chinhNgo(inp);

        // Giờ MẶT TRỜI THẬT, dịch sao cho đầu giờ Tý rơi đúng 00:00 — lunar.js
        // cắt ngày ở nửa đêm lịch, nên muốn nó cắt ở đầu giờ Tý thì phải dịch
        // đồng hồ chứ không sửa lunar.js.
        //   tyStart = noon − 780 = −60 − lonOff − eot  ⟹  shift = 60 + lonOff + eot
        var exact = new Date(inp.y, inp.m - 1, inp.d, inp.h, inp.mi);
        exact.setMinutes(exact.getMinutes() + cn.lonOffsetMins + cn.eotMins + 60);

        // TRỤ NGÀY + TRỤ GIỜ ở mốc UTC+8: can chi ngày là chu kỳ 60 ngày liên
        // tục, không phụ thuộc mốc; đặt mốc địa phương ở đây lại đẩy điểm Sóc
        // qua ranh giới ngày và làm hỏng độ dài tháng âm.
        var baziLocal = Ephem.atBasis(null, function () {
            return Solar.fromDate(exact).getLunar().getEightChar();
        });
        // Bản dự phòng khi không dựng nổi danh sách tháng âm.
        var lunarDisp = Ephem.atBasis(inp.tz, function () {
            return Solar.fromDate(exact).getLunar();
        });

        // NGÀY ÂM LỊCH: mùng 1 là ngày CHỨA điểm Sóc, đếm từ Chính Tý địa
        // phương — cùng hệ quy chiếu với giờ Sóc mà màn hình hiện ra.
        var ziDay = dayOf(Solar.fromYmdHms(inp.y, inp.m, inp.d, inp.h, inp.mi, 0),
            inp.lon, inp.tzId);
        var ziLunar = lunarOf(ziDay.y, ziDay.m, ziDay.d, inp.lon, inp.tzId);

        // TRỤ NĂM + TRỤ THÁNG so với mốc Lập Xuân / 12 Tiết, mà mốc ấy lấy từ
        // getJieQiJulianDays() theo mốc toàn cục — nên solarBJ (giờ Bắc Kinh)
        // và mốc tiết phải CÙNG ở UTC+8, bằng không so lệch nhau sáu tiếng.
        var bj = solarBJof(inp);
        var baziBJ = Ephem.atBasis(null, function () {
            return bj.solarBJ.getLunar().getEightChar();
        });
        var jq = Ephem.atBasis(null, function () {
            return prevJieQi(bj.solarBJ, bj.dUTC);
        });

        // TRỤ GIỜ theo giờ Tý thiên văn: mỗi thời thần 120 phút kể từ
        // Chính Ngọ − 13h. Can giờ theo Ngũ Thử Độn.
        var tyStart = cn.minutes - 780;
        var inputMins = inp.h * 60 + inp.mi;
        var gioChiIdx = Math.floor((((inputMins - tyStart) % 1440) + 1440) % 1440 / 120) % 12;
        var ngayGanIdx = CAN_ZH.indexOf(baziLocal.getDayGan());
        var gioGanIdx = (ngayGanIdx % 5 * 2 + gioChiIdx) % 10;

        var g = [CAN_ZH.indexOf(baziBJ.getYearGan()), CAN_ZH.indexOf(baziBJ.getMonthGan()),
                 ngayGanIdx, gioGanIdx];
        var z = [CHI_ZH.indexOf(baziBJ.getYearZhi()), CHI_ZH.indexOf(baziBJ.getMonthZhi()),
                 CHI_ZH.indexOf(baziLocal.getDayZhi()), gioChiIdx];

        var out = {
            input: inp,
            chinhNgo: cn,
            /** Bốn trụ: chỉ số can/chi (năm · tháng · ngày · giờ) và tên hai thứ tiếng. */
            gan: g, chi: z,
            ganZH: g.map(function (i) { return CAN_ZH[i]; }),
            chiZH: z.map(function (i) { return CHI_ZH[i]; }),
            ganVI: g.map(function (i) { return CAN_VI[i]; }),
            chiVI: z.map(function (i) { return CHI_VI[i]; }),
            /** Ngày âm lịch theo ranh giới Chính Tý (null → dùng bản dự phòng). */
            lunar: ziLunar || {
                day: lunarDisp.getDay(),
                month: Math.abs(lunarDisp.getMonth()),
                year: lunarDisp.getYear(),
                leap: lunarDisp.getMonth() < 0,
            },
            /** Tiết khí đang giữ: tên chữ Hán và mốc vào tiết (Solar ở UTC+8). */
            tietKhi: { zh: jq.jqObj.getName(), solarUTC8: jq.jqSolar },
            solarBJ: bj.solarBJ,
            _baziBJ: baziBJ, _baziLocal: baziLocal,
        };
        // Thần sát suy từ TRỤ GIỜ — bàn Kỳ Môn dùng cả ba.
        out.tuanThu = tuanThuOf(g[3], z[3]);
        out.tuanGiap = tuanGiapOf(g[3], z[3]);
        out.khongVongChi = khongVongOf(g[3], z[3]);
        out.dichMaChi = dichMaOf(z[3]);

        if (_chartCache.size > 16) _chartCache.clear();
        _chartCache.set(key, out);
        return out;
    }

    /** Xoá bộ nhớ đệm — ngôn ngữ đổi thì NHÃN tháng âm đổi theo (labelBasis). */
    function clearCache() { _chartCache.clear(); _monthCache.clear(); }

    root.Core = {
        CAN_ZH: CAN_ZH, CHI_ZH: CHI_ZH, CAN_VI: CAN_VI, CHI_VI: CHI_VI,
        TK_VI: TK_VI, TK_ZH: TK_ZH, tietKhiTen: tietKhiTen,
        location: location, tzAt: tzAt, input: input,
        tzOffsetAtJdUTC8: tzOffsetAtJdUTC8,
        localSolarFromJdUTC8: localSolarFromJdUTC8,
        fmtJdUTC8Local: fmtJdUTC8Local,
        chinhNgo: chinhNgo,
        dayOf: dayOf, jdn: jdn, labelBasis: labelBasis,
        mung1FromJd: mung1FromJd, months: months, lunarOf: lunarOf,
        prevJieQi: prevJieQi, solarBJof: solarBJof,
        tuanIdx: tuanIdx, tuanThuOf: tuanThuOf, tuanGiapOf: tuanGiapOf,
        dichMaOf: dichMaOf, khongVongOf: khongVongOf,
        chart: chart, clearCache: clearCache,
    };
})(typeof window !== 'undefined' ? window : globalThis);
