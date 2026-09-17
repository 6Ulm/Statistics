/* ════════════════════════════════════════════════════════════════════
   lenh.js — tab "Lệnh": nhân nguyên tư lệnh (人元司令分野)

   Mỗi tháng khí (từ TIẾT này tới TIẾT sau) không do một can duy nhất nắm.
   Chi của tháng tàng 2–3 can, và chúng thay nhau "cầm lệnh" theo thứ tự
   dư khí → trung khí → bản khí. Bảng này nói: tại thời điểm đang xem, can
   nào đang cầm lệnh.

   RANH GIỚI ĐO BẰNG ĐỘ HOÀNG KINH, KHÔNG PHẢI SỐ NGÀY.

   Sách xưa chép phần chia theo NGÀY ("Mậu 7 ngày, Bính 7 ngày, Giáp 16
   ngày"), vì một ngày Mặt Trời đi xấp xỉ một độ. Nhưng "xấp xỉ" ấy lệch tới
   3,4%: quanh cận nhật (tháng Giêng) Mặt Trời đi 1,019°/ngày, quanh viễn
   nhật (tháng Bảy) chỉ 0,953°/ngày. Đếm ngày thì tổng 30 ngày của một tháng
   khí không khớp khoảng cách thật giữa hai tiết (29,44 ngày mùa đông,
   31,44 ngày mùa hè) — cộng dồn ba đoạn là lệch cả ngày rưỡi, và đoạn cuối
   không rơi đúng vào tiết sau.

   Đo bằng độ thì hết hẳn: 7° + 7° + 16° = 30° = đúng khoảng cách hai tiết,
   theo định nghĩa. Ảnh mẫu người dùng gửi cũng ghi cột "Hoàng kinh" chứ
   không ghi số ngày.

   MỘT NGUỒN DUY NHẤT VỚI BẢNG TIẾT KHÍ.

   Mốc mở tháng (bội số của 15°) lấy THẲNG từ ShouXingUtil.qiAccurate — đúng
   hàm mà bảng tiết khí ở tab Lịch và bảng Sách Bổ ở tab Kỳ Môn đang dùng,
   nên nó tra bảng DE423 y hệt. Không có chuyện "Lập Xuân" ở tab Lịch một
   giờ mà "vào lệnh Mậu" ở tab này một giờ khác.

   Mốc giữa tháng (7°, 22°… không phải bội số của 15°) thì bảng DE423 không
   có — nó chỉ lưu 24 tiết khí. Chỗ ấy giải bằng chuỗi giải tích saLonT của
   chính lunar.js. Hai nguồn lệch nhau ≤ 2 giây tại các mốc dùng chung (đo
   trên cả năm 2026), tức không bao giờ đủ để đổi con số phút hiện ra.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var CAN_VI = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    var CHI_VI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
    var CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
    var CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

    var T = {
        tabLenh:  { vi: 'Lệnh',        zh: '令' },
        title:    { vi: 'LỆNH NĂM',    zh: '司令' },
        now:      { vi: 'Lệnh',        zh: '司令' },
        colMonth: { vi: 'Tháng',       zh: '月' },
        colCan:   { vi: 'Can',         zh: '天干' },
        colLon:   { vi: 'Hoàng kinh',  zh: '黄经' },
        colIn:    { vi: 'Vào lệnh',    zh: '入令' },
        colOut:   { vi: 'Hết lệnh',    zh: '退令' },
    };
    function isZH() { return typeof currentLang !== 'undefined' && currentLang === 'zh'; }
    function t(k) { return T[k][isZH() ? 'zh' : 'vi']; }
    function canName(i) { return isZH() ? CAN_ZH[i] : CAN_VI[i]; }
    function chiName(i) { return isZH() ? CHI_ZH[i] : CHI_VI[i]; }
    function jieName(k) {
        if (isZH()) return (typeof TK_ZH !== 'undefined') ? TK_ZH[k] : '';
        return (typeof TK_VI !== 'undefined') ? TK_VI[k] : '';
    }

    function esc(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }
    function pad2(n) { return (n < 10 ? '0' : '') + n; }

    /* ─────────────── Bảng phân dã ───────────────
     *
     * Khoá = chỉ số TIẾT mở tháng trong TK_VI/TK_ZH (0 = Đông Chí), luôn LẺ:
     * tháng khí mở bằng TIẾT (Tiểu Hàn, Lập Xuân…), còn KHÍ (Đại Hàn, Vũ
     * Thủy…) rơi vào giữa tháng và không phải ranh giới của gì cả.
     *
     * Giá trị = [chỉ số can, số độ] theo đúng thứ tự cầm lệnh. Tổng ba số độ
     * của mỗi tháng phải bằng 30 — buildYear() kiểm lại và ném lỗi nếu không,
     * vì gõ nhầm một số ở đây thì mọi mốc phía sau trong tháng ấy trôi theo mà
     * bảng vẫn trông bình thường.
     */
    var FEN = {
        1:  [[9, 9], [7, 3], [5, 18]],    // Sửu  (Tiểu Hàn):  Quý 9 · Tân 3 · Kỷ 18
        3:  [[4, 7], [2, 7], [0, 16]],    // Dần  (Lập Xuân):  Mậu 7 · Bính 7 · Giáp 16
        5:  [[0, 10], [1, 20]],           // Mão  (Kinh Trập): Giáp 10 · Ất 20
        7:  [[1, 9], [9, 3], [4, 18]],    // Thìn (Thanh Minh):Ất 9 · Quý 3 · Mậu 18
        9:  [[4, 5], [6, 9], [2, 16]],    // Tỵ   (Lập Hạ):    Mậu 5 · Canh 9 · Bính 16
        11: [[2, 10], [5, 9], [3, 11]],   // Ngọ  (Mang Chủng):Bính 10 · Kỷ 9 · Đinh 11
        13: [[3, 9], [1, 3], [5, 18]],    // Mùi  (Tiểu Thử):  Đinh 9 · Ất 3 · Kỷ 18
        15: [[4, 7], [8, 7], [6, 16]],    // Thân (Lập Thu):   Mậu 7 · Nhâm 7 · Canh 16
        17: [[6, 10], [7, 20]],           // Dậu  (Bạch Lộ):   Canh 10 · Tân 20
        19: [[7, 9], [3, 3], [4, 18]],    // Tuất (Hàn Lộ):    Tân 9 · Đinh 3 · Mậu 18
        21: [[4, 7], [0, 5], [8, 18]],    // Hợi  (Lập Đông):  Mậu 7 · Giáp 5 · Nhâm 18
        23: [[8, 10], [9, 20]],           // Tý   (Đại Tuyết): Nhâm 10 · Quý 20
    };
    /** Chi của tháng mà mỗi TIẾT mở ra. */
    var CHI_OF_JIE = { 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 0 };
    /** Thứ tự 12 tháng trong một NĂM DƯƠNG LỊCH: Sửu (tháng 1) → Tý (tháng 12). */
    var JIE_ORDER = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];

    var RAD = Math.PI / 180;
    /** Số ngày trung bình của một tiết khí — chỉ dùng để ĐOÁN chỉ số, rồi dò lại. */
    var MEAN_TERM_DAYS = 365.2422 / 24;

    /* ─────────────── Nghiệm Mặt Trời ─────────────── */

    /**
     * Mốc tiết khí số `n` (hoàng kinh biểu kiến 15n°), ngày Julius ở UTC+8.
     *
     * TRUYỀN THẲNG 8 chứ không để trống: ShouXingUtil giữ mốc múi giờ trong
     * một biến TOÀN CỤC mà tab Lịch đặt về múi giờ địa phương trước mỗi lần
     * vẽ. Bỏ trống là con số ra theo mốc của người gọi cuối cùng.
     */
    function termJd(n) {
        return ShouXingUtil.qiAccurate(n * Math.PI / 12, 8) + Solar.J2000;
    }

    /**
     * Thời điểm Mặt Trời đạt hoàng kinh `deg` (ĐỘ, đã mở vòng — 315° của năm
     * sau là 675°), ngày Julius ở UTC+8.
     *
     * Đây là ruột của qiAccurate, bỏ đi bước tra bảng DE423: bảng chỉ có 24
     * tiết khí nên mốc giữa tháng không tra được. `dtT` quy TT về UT, rồi
     * cộng 8 giờ ra giờ Bắc Kinh — đúng chuỗi phép mà qiAccurate làm.
     */
    function lonJd(deg) {
        var t = ShouXingUtil.saLonT(deg * RAD) * 36525;
        return t - ShouXingUtil.dtT(t) + 8 / 24 + Solar.J2000;
    }

    /**
     * Chỉ số tiết khí TUYỆT ĐỐI (n = 0 tại Xuân Phân 1999) của mốc `jd`.
     *
     * Đoán bằng số ngày trung bình rồi DÒ LẠI ±2: quỹ đạo Trái Đất là ellip
     * nên khoảng cách hai tiết lệch khỏi trung bình tới một ngày, và sau vài
     * chục năm phép đoán suông có thể lệch hẳn một chỉ số ngay sát ranh giới.
     */
    function termIndexNear(jd) {
        var n = Math.round((jd - 2451259.0) / MEAN_TERM_DAYS);
        for (var k = -2; k <= 2; k++) {
            if (Math.abs(termJd(n + k) - jd) < 3) return n + k;
        }
        return n;
    }

    /* ─────────────── Dựng bảng một năm ─────────────── */

    var _yearCache = {};

    /**
     * 12 tháng lệnh của NĂM DƯƠNG LỊCH `Y`, từ Sửu (Tiểu Hàn, tháng 1) tới Tý
     * (Đại Tuyết, tháng 12). Mọi mốc là ngày Julius ở UTC+8 — không phụ thuộc
     * địa điểm, nên nhớ theo năm là đủ; việc quy sang giờ địa phương để LÀ
     * việc của lúc vẽ.
     *
     * Mỗi phần tử: { chi, jie, n, parts: [{ can, from, to, jdFrom, jdTo }] }
     * với `from`/`to` là hoàng kinh ĐÃ MỞ VÒNG (cộng dồn qua các năm).
     */
    function buildYear(Y) {
        if (_yearCache[Y]) return _yearCache[Y];
        // Người dùng bấm tới bấm lui vài trăm năm thì bộ nhớ đệm phình mãi.
        // Dựng một năm chỉ mất chừng một mili giây, nên dọn sạch rồi dựng lại
        // là rẻ hơn hẳn so với việc giữ một danh sách LRU.
        if (Object.keys(_yearCache).length > 40) _yearCache = {};

        // Cùng dãy mốc mà bảng tiết khí của tab Lịch dùng: index 1..24 là
        // Đông Chí (tháng 12 năm Y−1) rồi Tiểu Hàn (tháng 1 năm Y)… tới Đại
        // Tuyết (tháng 12 năm Y). Nên TIẾT có chỉ số k nằm ở ô k+1.
        var jds = Ephem.jieQiJdAtBasis(Y, null);
        var out = [];
        for (var i = 0; i < JIE_ORDER.length; i++) {
            var k = JIE_ORDER[i];
            var n0 = termIndexNear(jds[k + 1]);
            var parts = [], off = 0;
            var fen = FEN[k];
            for (var j = 0; j < fen.length; j++) {
                var can = fen[j][0], span = fen[j][1];
                parts.push({
                    can: can,
                    from: 15 * n0 + off,
                    to: 15 * n0 + off + span,
                    // Mốc mở/đóng tháng là bội số của 15° → đi qua bảng DE423,
                    // trùng khít bảng tiết khí. Mốc giữa tháng thì giải chuỗi.
                    jdFrom: off === 0 ? termJd(n0) : lonJd(15 * n0 + off),
                    jdTo: (off + span) === 30 ? termJd(n0 + 2) : lonJd(15 * n0 + off + span),
                });
                off += span;
            }
            if (off !== 30) throw new Error('lenh.js: tháng ' + k + ' cộng ra ' + off + '°');
            out.push({ chi: CHI_OF_JIE[k], jie: k, n: n0, parts: parts });
        }
        _yearCache[Y] = out;
        return out;
    }

    /**
     * Can đang cầm lệnh tại thời điểm `jdUTC8`, hoặc null nếu tra không ra.
     *
     * Dò cả năm TRƯỚC nữa: tháng Tý mở ở Đại Tuyết (tháng 12) mà đóng ở Tiểu
     * Hàn năm sau, nên mọi ngày đầu tháng 1 đều thuộc bảng của năm trước.
     */
    function lenhAt(jdUTC8, y) {
        for (var dy = 0; dy >= -1; dy--) {
            var year = buildYear(y + dy);
            for (var i = 0; i < year.length; i++) {
                var parts = year[i].parts;
                for (var j = 0; j < parts.length; j++) {
                    if (jdUTC8 >= parts[j].jdFrom && jdUTC8 < parts[j].jdTo) {
                        return { month: year[i], part: parts[j] };
                    }
                }
            }
        }
        return null;
    }

    /* ─────────────── Đọc đầu vào ─────────────── */

    /** Ngày giờ đang chọn + địa điểm, hoặc null khi app.js chưa sẵn sàng. */
    function readInput() {
        if (typeof getDOM !== 'function' || typeof countryData === 'undefined') return null;
        var y = parseInt(getDOM('inYear').value, 10);
        var m = parseInt(getDOM('inMonth').value, 10);
        var d = parseInt(getDOM('inDay').value, 10);
        var h = parseInt(getDOM('solarHour').value, 10);
        var mi = parseInt(getDOM('solarMinute').value, 10);
        if (!isFinite(y) || !isFinite(m) || !isFinite(d)) return null;
        var info = countryData[getDOM('country').value];
        if (!info) return null;
        var tz = (typeof getTimezoneOffset === 'function')
            ? getTimezoneOffset(info.tzId, new Date(y, m - 1, d, h || 12)) : 7;
        return { y: y, m: m, d: d, h: h || 0, mi: mi || 0, info: info, tz: tz };
    }

    /**
     * Giờ địa phương của một mốc UTC+8, trả về đối tượng Solar.
     *
     * Dùng ĐÚNG đường mà bảng tiết khí dùng (_tzOffsetAtJdUTC8): offset lấy
     * tại CHÍNH THỜI ĐIỂM ấy chứ không phải theo ngày đang xem, nên mốc mùa
     * đông không bị cộng nhầm giờ mùa hè ở những nước có DST.
     */
    function toLocal(jdUTC8, tzId) {
        var tz = (typeof _tzOffsetAtJdUTC8 === 'function')
            ? _tzOffsetAtJdUTC8(jdUTC8, tzId) : 8;
        return Solar.fromJulianDay(jdUTC8 + (tz - 8) / 24);
    }

    /**
     * "05/01 16:23", hoặc "05/01/2027 22:09" khi mốc rơi ra ngoài năm của
     * bảng — đúng cách ảnh mẫu viết. Bỏ năm ở 31/33 hàng thì cột hẹp đi
     * chừng 30px, đủ để cả năm cột vừa màn hình 360px mà không phải kéo ngang.
     */
    function fmtLocal(jdUTC8, tzId, tableYear) {
        var s = toLocal(jdUTC8, tzId);
        var ymd = pad2(s.getDay()) + '/' + pad2(s.getMonth())
            + (s.getYear() === tableYear ? '' : '/' + s.getYear());
        return ymd + ' ' + pad2(s.getHour()) + ':' + pad2(s.getMinute());
    }

    /* ─────────────── Vẽ ─────────────── */

    var lastActive = null;

    function render() {
        var head = document.getElementById('lenhTitle');
        var nowBox = document.getElementById('lenhNow');
        var body = document.getElementById('lenhBody');
        if (!body) return;
        if (typeof Solar === 'undefined' || typeof ShouXingUtil === 'undefined' ||
            typeof Ephem === 'undefined' || typeof TK_VI === 'undefined') return;

        var inp = readInput();
        if (!inp) return;

        var year;
        try { year = buildYear(inp.y); } catch (e) { body.innerHTML = ''; return; }

        // Can đang cầm lệnh: hỏi ở ĐÚNG thời điểm UTC+8 mà trụ tháng dùng, nên
        // chi của tháng lệnh luôn khớp chi của trụ tháng trong bảng Bát Tự
        // ngay bên trên — hai con số cãi nhau trên cùng một màn hình là lỗi
        // nặng hơn cả sai vài phút.
        var active = null;
        if (typeof _readInputBJ === 'function') {
            try {
                var bj = _readInputBJ(inp.y, inp.m, inp.d, inp.h, inp.mi, inp.tz);
                active = lenhAt(bj.solarBJ.getJulianDay(), inp.y);
            } catch (e2) { active = null; }
        }
        lastActive = active;

        if (head) {
            head.textContent = isZH()
                ? (inp.y + '年' + t('title'))
                : (t('title') + ' ' + inp.y);
        }
        if (nowBox) {
            nowBox.innerHTML = esc(t('now')) + (isZH() ? '：' : ': ') +
                '<b id="lenhNowVal">' + esc(active ? canName(active.part.can) : '—') + '</b>';
        }

        var tzId = inp.info.tzId;
        var rows = '', alt = false;
        for (var i = 0; i < year.length; i++) {
            var mo = year[i];
            for (var j = 0; j < mo.parts.length; j++) {
                var p = mo.parts[j];
                var on = active && active.part === p;
                rows += '<tr class="' + (alt ? 'dp-row-alt' : '') + (on ? ' lenh-on' : '') + '"' +
                    (on ? ' id="lenhActive"' : '') + '>';
                if (j === 0) {
                    rows += '<td class="lenh-mon" rowspan="' + mo.parts.length + '">' +
                        esc(chiName(mo.chi)) +
                        ' <span class="lenh-jie">(' + esc(jieName(mo.jie)) + ')</span></td>';
                }
                rows += '<td class="c lenh-can">' + esc(canName(p.can)) + '</td>' +
                    '<td class="c dp-num lenh-lon">' +
                    esc(norm360(p.from) + '~' + norm360(p.to) + '°') + '</td>' +
                    '<td class="c dp-num">' + esc(fmtLocal(p.jdFrom, tzId, inp.y)) + '</td>' +
                    '<td class="c dp-num lenh-last">' + esc(fmtLocal(p.jdTo, tzId, inp.y)) + '</td>' +
                    '</tr>';
            }
            alt = !alt;   // đổi nền theo THÁNG, không theo hàng: ba hàng của
                          // một tháng phải cùng nền thì ô gộp mới liền khối.
        }

        body.innerHTML =
            '<table class="dp-table lenh-tb">' +
            '<colgroup><col><col><col><col><col></colgroup>' +
            '<thead><tr class="lenh-head">' +
            '<th>' + esc(t('colMonth')) + '</th>' +
            '<th class="c">' + esc(t('colCan')) + '</th>' +
            '<th class="c">' + esc(t('colLon')) + '</th>' +
            '<th class="c">' + esc(t('colIn')) + '</th>' +
            '<th class="c lenh-last">' + esc(t('colOut')) + '</th>' +
            '</tr></thead><tbody>' + rows + '</tbody></table>';

        fit();
        setTimeout(scrollToActive, 40);
    }

    function norm360(deg) { return ((deg % 360) + 360) % 360; }

    /**
     * Cuộn để hàng đang cầm lệnh nằm giữa khung — bảng dài 33 hàng, mở ra mà
     * phải tự đi tìm hàng của hôm nay thì bảng vô dụng một nửa.
     *
     * Chốt về đúng ranh giới hàng: hàng tiêu đề DÍNH ở mép trên, nên cuộn tới
     * một vị trí bất kỳ là để lại một hàng bị cắt ngang nằm ngay dưới nó.
     */
    function scrollToActive() {
        var box = document.getElementById('lenhBody');
        var row = document.getElementById('lenhActive');
        if (!box || !row) return;
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var thead = box.querySelector('thead');
        var headH = thead ? thead.getBoundingClientRect().height / zoom : 0;

        // Bước 1 — cuộn thô, đo TƯƠNG ĐỐI: "hàng này đang cách đỉnh khung bao
        // nhiêu" cộng vào scrollTop hiện thời. Không dựng lại hệ toạ độ nội
        // dung từ scrollTop và các mép: viền, đệm và chính hàng tiêu đề DÍNH
        // (getBoundingClientRect của <thead> vẫn trả vị trí lúc CHƯA dính, vì
        // chỉ mấy ô <th> mới sticky chứ không phải cả <thead>) — ba thứ ấy đủ
        // để lệch cả chục pixel.
        var rBox = box.getBoundingClientRect();
        var rRow = row.getBoundingClientRect();
        var mid = Math.max(0, (box.clientHeight - headH - rRow.height / zoom) / 2);
        box.scrollTop = Math.max(0, Math.round(
            box.scrollTop + (rRow.top - rBox.top) / zoom - headH - mid));

        alignUnderHead(box, headH, zoom);
    }

    /**
     * ĐO rồi chỉnh: kéo khung xuống vừa đủ để hàng đang bị hàng tiêu đề cắt
     * ngang lộ ra trọn vẹn.
     *
     * Các hàng KHÔNG cao bằng nhau — ô tháng gộp 2 hàng chứa hai dòng chữ nên
     * kéo hai hàng ấy cao hơn ba hàng của tháng bên cạnh — nên không thể chốt
     * bằng cách chia cho một "chiều cao hàng". Sau khi đã cuộn thì mọi thứ đã
     * nằm trên trang, hỏi thẳng là xong; kéo xuống chỉ giấu thêm phần trên,
     * không sinh ra hàng cụt mới.
     */
    function alignUnderHead(box, headH, zoom) {
        var bt = parseFloat(getComputedStyle(box).borderTopWidth) || 0;
        var edge = box.getBoundingClientRect().top / zoom + bt + headH;
        var rows = box.querySelectorAll('tbody tr');

        // Chốt về đầu THÁNG, không phải đầu hàng.
        //
        // Ô tháng gộp 2–3 hàng, nên dừng ở giữa một tháng là hàng đầu khung
        // hiện "(Kinh Trập)" mà mất chữ "Mão" nằm trên nó — một cái tên tiết
        // mồ côi, không biết của tháng nào. Một tháng chỉ cao 2–3 hàng mà
        // khung thì hiện hơn 20 hàng, nên kéo lên tới đầu tháng gần nhất
        // không hề đẩy hàng đang cầm lệnh ra khỏi khung.
        var top = 0;
        for (var i = 0; i < rows.length; i++) {
            var q = rows[i].getBoundingClientRect();
            if (q.top / zoom > edge + 0.5) break;
            if (rows[i].querySelector('.lenh-mon')) top = q.top / zoom;
        }
        if (!top) return;
        box.scrollTop = Math.max(0, box.scrollTop - Math.round(edge - top));
    }

    /* ─────────────── Chia chiều cao ─────────────── */

    /** Đệm chống tràn — phép làm tròn nửa pixel, không phải phép cộng lề. */
    var CHROME = 10;
    /** Khung bảng thấp hơn chừng này thì thà cuộn cả trang còn hơn. */
    var BOX_MIN = 120;

    /**
     * Kéo bảng lệnh cho lấp đúng phần màn hình còn lại.
     *
     * Cùng bài toán của fitGrid bên calendar.js: khung trên (ô ngày giờ, bảng
     * Bát Tự, dòng "Lệnh:") cao bao nhiêu là do nội dung, còn bảng thì dài vô
     * hạn — nên ĐO phần cố định rồi cấp phần còn lại cho bảng, chứ không đoán.
     * Không có bước này thì bảng 33 hàng đẩy trang cao gấp rưỡi màn hình,
     * viewport.js thu nhỏ cả trang để chữa, và bảng Bát Tự bé lại vô cớ.
     */
    function fit() {
        var sec = document.getElementById('lenhSec');
        var box = document.getElementById('lenhBody');
        var bar = document.getElementById('bottomDock');
        if (!sec || !box || !bar) return;
        if (!document.body.classList.contains('view-lenh')) return;

        // Khung bảng là khối CUỐI CÙNG của trang, nên chỗ nó được phép chiếm
        // chính là khoảng từ đỉnh nó tới thanh dưới — đo thẳng, khỏi phải cộng
        // lại chiều cao từng khối bên trên cùng mọi khe giữa chúng (bản đầu
        // làm vậy: mười mấy dòng, và sai ngay khi ai đó thêm một lề trong CSS).
        // Hạ chiều cao khung KHÔNG làm chính đỉnh nó nhúc nhích, nên con số đo
        // được vẫn đúng sau khi đặt.
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var top = sec.getBoundingClientRect().top / zoom;
        var floorY = bar.getBoundingClientRect().top / zoom;
        var room = Math.max(BOX_MIN, Math.floor(floorY - top - CHROME));

        // Bảng ngắn hơn chỗ được cấp (năm nào cũng 33 hàng nên hiếm, nhưng máy
        // tính bảng thì có) thì kẹp theo chính nó, đừng chừa một khoảng trắng.
        var tb = box.querySelector('table');
        var nat = tb ? Math.ceil(tb.getBoundingClientRect().height / zoom) + 2 : 0;
        box.style.maxHeight = (nat ? Math.min(nat, room) : room) + 'px';

        // Mép dưới không được cắt hàng cuối đúng chỗ có DẤU: "Mậu" cụt dấu
        // nặng thành "Mâu", "Bạch Lộ" thành "Bạch Lô" — chữ khác hẳn, không
        // phải hàng cụt. Phép canh nằm ở calendar.js (xem snapCut ở đó), dùng
        // chung một bản cho cả hai tab.
        if (typeof window.__snapCutRows === 'function') {
            try { window.__snapCutRows(box); } catch (e) {}
        }

        // Canh lại mép TRÊN sau MỌI lần đổi chiều cao. Lượt cuộn tới hàng đang
        // cầm lệnh chạy 40ms sau khi vẽ, còn viewport.js chỉnh tỉ lệ rồi gọi
        // fit() lần nữa SAU đó — chia lại khung xong mà không canh lại thì cú
        // canh trước đó tính trên một khung đã không còn nữa, và hàng đầu lại
        // nằm cụt dưới hàng tiêu đề.
        var thead = box.querySelector('thead');
        if (thead) alignUnderHead(box, thead.getBoundingClientRect().height / zoom, zoom);
    }

    /* ─────────────── Nhãn + móc nối ─────────────── */

    function refreshLabels() {
        var tab = document.getElementById('tabLenh');
        if (tab) tab.querySelector('.tab-lbl').textContent = t('tabLenh');
        if (document.body.classList.contains('view-lenh')) render();
    }

    window.__lenhRefreshLabels = refreshLabels;
    window.__lenhRender = render;
    window.__lenhFit = function () {
        if (!document.body.classList.contains('view-lenh')) return;
        try { fit(); } catch (e) {}
    };
    /** Chỉ dùng cho bộ kiểm thử: đọc thẳng bảng đã tính, khỏi phải đọc DOM. */
    window.__lenhData = function (Y) { return buildYear(Y); };
    window.__lenhActive = function () { return lastActive; };

    document.addEventListener('DOMContentLoaded', function () {
        refreshLabels();

        // Vẽ lại sau MỌI lần engine tính lại — đổi ngày, đổi địa điểm, đổi
        // ngôn ngữ đều đi qua processAll(). Cùng cách calendar.js móc vào, nên
        // ba tab luôn nói cùng một địa điểm và cùng một thời điểm.
        if (typeof processAll === 'function' && !processAll.__lenhWrapped) {
            var orig = processAll;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                if (document.body.classList.contains('view-lenh')) {
                    try { render(); } catch (err) { console.warn('lenh:', err); }
                }
                return r;
            };
            wrapped.__lenhWrapped = true;
            window.processAll = wrapped;
        }
    });

    window.addEventListener('resize', function () {
        if (document.body.classList.contains('view-lenh')) setTimeout(fit, 180);
    });
})();
