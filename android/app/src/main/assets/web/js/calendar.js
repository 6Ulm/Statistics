/* ════════════════════════════════════════════════════════════════════
   calendar.js — Lịch âm dương + thanh tab dưới màn hình
   Vietnamese lunar calendar tab.

   Mỗi ô hiển thị: ngày dương (to), ngày âm (nhỏ) và can chi — can một dòng,
   chi một dòng, giống hệt nhau ở mọi ngày.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var CAN_VI = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    var CHI_VI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
    var CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
    var CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

    var T = {
        tabQmdj:  { vi: 'Kỳ Môn',     zh: '奇门' },
        tabCal:   { vi: 'Lịch',       zh: '日历' },
        title:    { vi: 'LỊCH ÂM THÁNG', zh: '农历' },
        dows:     { vi: ['Thứ 2', 'Thứ 3', 'Thứ 4', 'Thứ 5', 'Thứ 6', 'Thứ 7', 'C.Nhật'],
                    zh: ['一', '二', '三', '四', '五', '六', '日'] },
        today:    { vi: 'Hôm nay',    zh: '今天' },
        colTk:    { vi: 'Tiết Khí',   zh: '节气' },
        colDate:  { vi: 'Dương lịch', zh: '公历' },
        pin:      { vi: '📌 Ghim lịch ra màn hình chính', zh: '📌 固定日历到主屏幕' },
        pinOk:    { vi: 'Hãy xác nhận trên hộp thoại vừa hiện ra.',
                    zh: '请在弹出的对话框中确认。' },
        pinManual:{ vi: 'Máy này không cho ghim tự động. Nhấn giữ khoảng trống trên màn hình chính → Tiện ích (Widget) → tìm "Lịch âm".',
                    zh: '此设备不支持一键固定。请长按主屏幕空白处 → 小部件 → 找到"农历"。' },
    };
    function t(k) {
        var zh = (typeof currentLang !== 'undefined' && currentLang === 'zh');
        return T[k][zh ? 'zh' : 'vi'];
    }
    function isZH() { return typeof currentLang !== 'undefined' && currentLang === 'zh'; }

    /**
     * Lịch âm được tính theo múi giờ NÀO là một quy ước, không phải tuỳ chọn:
     * lịch Việt Nam tính điểm Sóc ở UTC+7, lịch Trung Quốc ở UTC+8 — đó chính
     * là lý do Tết ta và Tết Tàu thỉnh thoảng lệch nhau một ngày.
     *
     * Bắt buộc phải đặt lại mỗi lần vẽ: processAll() để lại múi giờ của địa
     * điểm đang chọn trong biến toàn cục của lunar.js, nên nếu đang chọn Paris
     * (UTC+2) thì lịch sẽ lệch một ngày (26/08/2026 hoá ra 15/7 thay vì 14/7).
     */
    function setLunarBasis() {
        if (typeof ShouXingUtil === 'undefined' || !ShouXingUtil.setTzOffsetHours) return;
        ShouXingUtil.setTzOffsetHours(localTz());
    }

    /**
     * Múi giờ dùng làm mốc cho lịch âm = múi giờ của ĐỊA ĐIỂM ĐANG CHỌN, đúng
     * mốc mà tab Kỳ Môn dùng để tính ngày âm và hiện giờ Sóc. Trước đây chỗ này
     * cố định UTC+7, nên ở Paris lịch hiện mùng 1 = 13/08 trong khi bảng Sóc
     * ghi 12/08 19:37 — hai hệ quy chiếu trên cùng một ứng dụng.
     */
    function localTz() {
        try {
            var info = countryData[getDOM('country').value];
            if (info && info.tzId && typeof getTimezoneOffset === 'function') {
                var sel = selected || { y: viewY, m: viewM, d: 15 };
                return getTimezoneOffset(info.tzId, new Date(sel.y, sel.m - 1, sel.d, 12));
            }
        } catch (e) {}
        return 7;
    }
    /** Trả biến toàn cục về mặc định của thư viện cho phần còn lại của ứng dụng. */
    function clearLunarBasis() {
        if (typeof ShouXingUtil !== 'undefined' && ShouXingUtil.setTzOffsetHours) {
            ShouXingUtil.setTzOffsetHours(null);
        }
    }

    // Lề trên/dưới của trang cộng khoảng cách giữa các khối trong tab Lịch.
    var GRID_CHROME = 28;
    // Hàng vừa đủ chứa 3 dòng (ngày dương/âm, can, chi) mà không dềnh dàng.
    // Chỗ thừa của màn hình cao giờ đổ vào bảng tiết khí 24 dòng, không kéo
    // hàng lịch cao ra nữa.
    var ROW_MIN = 58, ROW_MAX = 80;
    // Bảng tiết khí mở ra thì phải cao đủ để đọc; dưới mức này thì thà để lưới
    // lịch tràn một chút rồi cuộn cả trang.
    // Dự phòng khi chưa đo được bảng tiết khí (lần vẽ đầu): 12 hàng hai cột cao
    // chừng ngần này.
    var JQ_FALLBACK = 300;

    /** Khoá kho tuỳ chọn: bảng tháng âm cho widget (xem publishLunarCache). */
    var K_LUNAR_CACHE = 'qmdj.lunarCache';

    var viewY, viewM;          // tháng đang xem (dương lịch)
    var selected = null;       // {y,m,d}
    var lastWeeks = 0;         // số hàng của lưới đang hiện, để đo lại khi gập

    /** Ngày âm theo ranh giới Chính Tý, dùng chung engine với tab Kỳ Môn. */
    function ziOf(y, m, d) {
        try {
            if (typeof zi_lunarOf !== 'function') return null;
            var info = countryData[getDOM('country').value];
            if (!info || !info.tzId) return null;
            var tz = getTimezoneOffset(info.tzId, new Date(y, m - 1, d, 12));
            return zi_lunarOf(y, m, d, info.lon, info.tzId, tz);
        } catch (e) { return null; }
    }

    /** Can chi tiếng Việt của một đối tượng Lunar. */
    function dayGanZhi(lunar) {
        var can = CAN_ZH.indexOf(lunar.getDayGan());
        var chi = CHI_ZH.indexOf(lunar.getDayZhi());
        if (can < 0 || chi < 0) return { can: '', chi: '', vi: '', chiVi: '' };
        var zh = isZH();
        return {
            can: zh ? lunar.getDayGan() : CAN_VI[can],
            chi: zh ? lunar.getDayZhi() : CHI_VI[chi],
            vi: CAN_VI[can] + ' ' + CHI_VI[chi],
            chiVi: CHI_VI[chi],
        };
    }

    /* Dùng chung kho tuỳ chọn với location.js (native prefs → localStorage). */
    function prefGet(k) {
        try {
            var n = window.QMDJNative;
            if (n && n.getPref) { var v = n.getPref(k); if (v !== null && v !== '') return v; }
        } catch (e) {}
        try { return localStorage.getItem(k); } catch (e) { return null; }
    }
    function prefSet(k, v) {
        try { var n = window.QMDJNative; if (n && n.setPref) n.setPref(k, v); } catch (e) {}
        try { localStorage.setItem(k, v); } catch (e) {}
    }

    function esc(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /** Dựng lưới lịch cho tháng viewY/viewM. */
    function render() {
        setLunarBasis();
        var head = document.getElementById('calTitle');
        if (head) {
            head.textContent = isZH()
                ? viewY + '年' + viewM + '月'
                : t('title') + ' ' + viewM + '/' + viewY;
        }

        var dows = T.dows[isZH() ? 'zh' : 'vi'];
        var html = '<div class="cal-row cal-dow">' +
            dows.map(function (d, i) {
                return '<div class="cal-cell cal-dow-cell' + (i === 6 ? ' cal-sun' : '') + '">' + d + '</div>';
            }).join('') + '</div>';

        var first = new Date(viewY, viewM - 1, 1);
        var lead = (first.getDay() + 6) % 7;              // tuần bắt đầu từ Thứ 2
        var daysInMonth = new Date(viewY, viewM, 0).getDate();
        var today = new Date();
        var tKey = today.getFullYear() + '-' + (today.getMonth() + 1) + '-' + today.getDate();

        // Ô đầu và ô cuối lưới không để trống: điền nốt ngày cuối tháng trước
        // và ngày đầu tháng sau, tô mờ. Chạm vào một ngày như vậy thì nhảy
        // luôn sang tháng đó.
        var prev = new Date(viewY, viewM - 2, 1);
        var prevY = prev.getFullYear(), prevM = prev.getMonth() + 1;
        var prevDays = new Date(prevY, prevM, 0).getDate();
        var next = new Date(viewY, viewM, 1);
        var nextY = next.getFullYear(), nextM = next.getMonth() + 1;

        var cells = [];
        for (var i = lead; i > 0; i--) {
            cells.push(cellHtml(prevY, prevM, prevDays - i + 1, cells.length, true, tKey));
        }
        for (var d = 1; d <= daysInMonth; d++) {
            cells.push(cellHtml(viewY, viewM, d, cells.length, false, tKey));
        }
        for (var nd = 1; cells.length % 7 !== 0; nd++) {
            cells.push(cellHtml(nextY, nextM, nd, cells.length, true, tKey));
        }

        for (var r = 0; r < cells.length; r += 7) {
            html += '<div class="cal-row">' + cells.slice(r, r + 7).join('') + '</div>';
        }
        document.getElementById('calGrid').innerHTML = html;

        renderJieQi();
        clearLunarBasis();
        lastWeeks = cells.length / 7;
        fitGrid(lastWeeks);
        publishLunarCache();
    }

    /**
     * Kéo cao các hàng cho lịch lấp đầy màn hình.
     * Bố cục bị chặn bởi bề ngang nên phóng to cả trang không được (sẽ tràn
     * ngang); phần thừa chiều cao phải rót vào chiều cao hàng thì mới hết
     * khoảng trống mênh mông ở đáy.
     */
    function fitGrid(weeks) {
        if (!weeks) return;
        var grid = document.getElementById('calGrid');
        var head = document.getElementById('calHead');
        var bar = document.getElementById('tabBar');
        var dow = grid ? grid.querySelector('.cal-dow') : null;
        // Nút "Ghim lịch ra màn hình chính" CHỈ hiện khi chạy trong ứng dụng
        // Android, nên mọi phép đo trên trình duyệt đều không thấy nó. Không trừ
        // ra thì trên máy thật lưới lịch với bảng tiết khí chiếm trọn màn hình
        // rồi đẩy nút xuống dưới, nằm khuất sau thanh tab cố định.
        var pin = document.getElementById('calPinBtn');
        var note = document.getElementById('calPinNote');
        if (!grid || !head || !dow) return;

        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var h = function (el) { return el ? el.getBoundingClientRect().height / zoom : 0; };

        // Đo phần cố định — KHÔNG tính thân bảng tiết khí, vì chính nó là thứ
        // ta sắp chia. Đo cả cụm rồi mới chia thì thành vòng luẩn quẩn.
        var vis = function (el) {
            return (el && getComputedStyle(el).display !== 'none') ? h(el) : 0;
        };
        var avail = window.innerHeight / zoom
            - h(head) - h(dow) - h(bar) - vis(pin) - vis(note) - GRID_CHROME;

        // Lưới lấy phần của nó trước (có trần), bảng tiết khí nhận toàn bộ
        // phần còn lại — nhờ vậy màn hình cao không còn hở một mảng ở đáy.
        // Đo chiều cao THẬT của bảng tiết khí thay vì giữ sẵn một khoản cố
        // định: giữ 320px mà bảng chỉ cao 280px thì 40px kia thành khoảng hở ở
        // đáy màn hình. Đo phần tử <table> chứ không phải khung cuộn — khung
        // đang bị max-height của lần chia trước cắt ngắn.
        var table = document.querySelector('#calJieQi table.cal-jq');
        var jqH = table ? Math.ceil(table.getBoundingClientRect().height / zoom) + 2
                        : JQ_FALLBACK;
        jqH = Math.min(jqH, Math.max(120, avail - ROW_MIN * weeks));

        var rowH = Math.max(ROW_MIN, Math.min(ROW_MAX, Math.floor((avail - jqH) / weeks)));
        document.documentElement.style.setProperty('--cal-row-h', rowH + 'px');
        document.documentElement.style.setProperty(
            '--cal-jq-h', Math.max(120, Math.floor(avail - rowH * weeks)) + 'px');
    }

    /**
     * Dựng HTML một ô ngày.
     * @param {number} idx    thứ tự ô trong lưới (để biết cột Chủ nhật)
     * @param {boolean} outside  ngày của tháng trước/sau — tô mờ
     */
    function cellHtml(y, m, d, idx, outside, tKey) {
        var lunar = Solar.fromYmd(y, m, d).getLunar();
        var gz = dayGanZhi(lunar);
        // Ngày âm lấy theo ranh giới CHÍNH TÝ (xem khối ghi chú trong app.js):
        // mùng 1 là ngày chứa điểm Sóc, đếm từ nửa đêm mặt trời thật chứ không
        // phải 00:00. Hỏng thì lùi về số của lunar.js còn hơn để trống cả lịch.
        var zl = ziOf(y, m, d);
        var lday = zl ? zl.day : lunar.getDay();
        var lmon = zl ? (zl.leap ? -zl.month : zl.month) : lunar.getMonth();
        // Mùng 1 thì ghi kèm tháng âm, như lịch giấy: "1/7"
        var lunarTxt = (lday === 1) ? (lday + '/' + Math.abs(lmon) + (lmon < 0 ? 'N' : '')) : String(lday);

        var cls = 'cal-cell cal-day';
        if (outside) cls += ' cal-out';
        if (y + '-' + m + '-' + d === tKey) cls += ' cal-today';
        if (selected && selected.y === y && selected.m === m && selected.d === d) cls += ' cal-sel';
        if (idx % 7 === 6) cls += ' cal-sun';

        // Can và chi luôn nằm trên HAI dòng riêng, mọi ngày như nhau —
        // để một chuỗi tự xuống dòng thì "Đinh Mùi" gãy đôi còn "Kỷ Dậu"
        // nằm một dòng, nhìn so le rất xấu.
        return '<div class="' + cls + '" data-y="' + y + '" data-m="' + m + '" data-d="' + d + '">' +
            '<div class="cal-top">' +
            '<span class="cal-solar">' + d + '</span>' +
            '<span class="cal-lunar">' + esc(lunarTxt) + '</span>' +
            '</div>' +
            '<div class="cal-gz">' +
            '<span>' + esc(gz.can) + '</span><span>' + esc(gz.chi) + '</span>' +
            '</div>' +
            '</div>';
    }

    /** "DD-MM-YYYY HH:MM" → số phút tuyệt đối, chỉ để so trước/sau. */
    function parseDt(str) {
        var m = /^(\d{2})-(\d{2})-(\d{4})[ T](\d{2}):(\d{2})/.exec(str || '');
        if (!m) return NaN;
        return Date.UTC(+m[3], +m[2] - 1, +m[1], +m[4], +m[5]) / 60000;
    }

    /**
     * Bảng 24 tiết khí của năm, DÙNG LẠI nguyên bảng Sách Bổ pháp ở tab Kỳ Môn:
     * cùng cột (Tiết Khí · Dương lịch · Độn · Số Cục), cùng lớp CSS, cùng cách
     * tô đậm tiết khí đang hiệu lực. Gọi thẳng `sb_getJieQiDates` và
     * `_mkRow`/`_donBadgeSm` của app.js thay vì chép lại — chép ra là hai bảng
     * sẽ lệch nhau ngay lần sửa đầu tiên.
     *
     * Mốc thời gian lấy theo NGÀY ĐANG CHỌN trong lịch (mặc định là hôm nay) và
     * theo địa điểm đang chọn, đúng như tab Kỳ Môn — tiết khí là mốc thiên văn,
     * giờ giao tiết khác nhau theo múi giờ.
     */
    function renderJieQi() {
        var box = document.getElementById('calJieQi');
        if (!box) return;
        var rows = null;
        try { rows = buildJieQiRows(); } catch (e) { rows = null; }
        if (!rows) { box.innerHTML = ''; box.className = ''; return; }

        // Không còn hộp tiêu đề "Tiết khí trong năm" với nút gập: xếp hai cột
        // xong thì cả 24 mục vừa một màn hình, chẳng còn gì để gập lại — thanh
        // tiêu đề chỉ tổ ăn mất chừng 32px mà không nói thêm được gì, vì hai
        // cột "Tiết Khí" đã tự giới thiệu chính nó.
        //
        // KHÔNG bọc thêm .dp-table-wrap: nó có overflow-x nên trở thành vùng
        // cuộn gần nhất của <th> sticky, mà chính nó lại không giới hạn chiều
        // cao — hàng tiêu đề vì thế trôi mất khi cuộn (màn hình thấp vẫn phải
        // cuộn). Cho .cal-jq-body cuộn cả hai chiều là xong.
        box.innerHTML =
            '<div class="cal-jq-body">' +
            '<table class="dp-table cal-jq"><thead><tr>' +
            '<th>' + t('colTk') + '</th><th>' + t('colDate') + '</th>' +
            '<th class="cal-jq-split">' + t('colTk') + '</th>' +
            '<th class="cal-jq-last">' + t('colDate') + '</th>' +
            '</tr></thead><tbody id="calJqBody">' + rows + '</tbody></table></div>';
        setTimeout(scrollToActiveJieQi, 40);
        return true;
    }

    /** Dựng thân bảng (12 hàng × 2 cột kép); trả null nếu app.js chưa sẵn sàng. */
    function buildJieQiRows() {
        if (typeof sb_getJieQiDates !== 'function' || typeof sb_findY !== 'function' ||
            typeof TK_ZH === 'undefined' || typeof TK_VI === 'undefined') return null;

        var sel = selected || { y: viewY, m: viewM, d: 1 };
        var info = (typeof countryData !== 'undefined' && typeof getDOM === 'function')
            ? countryData[getDOM('country').value] : null;
        if (!info) return null;
        var tzId = info.tzId;
        var tz = (typeof getTimezoneOffset === 'function')
            ? getTimezoneOffset(tzId, new Date(sel.y, sel.m - 1, sel.d, 12)) : 7;

        // Bảng Sách Bổ ở tab Kỳ Môn được dựng khi ShouXingUtil đang ở múi giờ
        // ĐỊA PHƯƠNG (app.js đặt trước bước 8). `findJieQi` bên trong sb_findY
        // đọc biến toàn cục đó, nên nếu chạy ở mốc UTC+7 của lịch âm thì hai
        // bảng có thể lệch nhau một giờ. Đặt đúng mốc rồi trả lại như cũ.
        var snap = ShouXingUtil.getTzOffsetHours();
        var Y, dates;
        try {
            ShouXingUtil.setTzOffsetHours(tz);
            Y = sb_findY(sel.y, sel.m, sel.d, 12, 0, tzId, tz);
            dates = sb_getJieQiDates(Y, tzId, tz);
        } finally {
            ShouXingUtil.setTzOffsetHours(snap);
        }

        // Tiết khí đang hiệu lực = mốc CUỐI CÙNG không muộn hơn ngày đang chọn.
        // Lấy 12:00 trưa làm mốc so: chọn 00:00 thì đúng ngày giao tiết sẽ rơi
        // về tiết trước, mà lịch chỉ có độ phân giải một ngày.
        var at = Date.UTC(sel.y, sel.m - 1, sel.d, 12, 0) / 60000;
        var active = 0;
        for (var i = 0; i < 24; i++) {
            var ts = parseDt(dates[i]);
            if (!isNaN(ts) && ts <= at) active = i;
        }

        // Bỏ Độn và Số Cục thì mỗi mục chỉ còn tên với ngày — hẹp bằng nửa
        // bề ngang. Xếp 12 mục đầu (Đông Chí → Mang Chủng) bên trái, 12 mục
        // sau (Hạ Chí → Đại Tuyết) bên phải: bảng thấp đi một nửa, gần như
        // không phải cuộn nữa, và ranh giới trái/phải trùng đúng ranh giới
        // Dương Độn / Âm Độn.
        var zh = isZH();
        // `right` chứ không phải `k === 12`: vách ngăn phải kẻ ở ô đầu của nửa
        // PHẢI trên MỌI hàng. Bám vào chỉ số 12 thì nó chỉ trúng hàng đầu tiên,
        // nên đường kẻ đứt ngay sau hàng ấy.
        var cell = function (k, right) {
            var on = k === active;
            return '<td class="cal-jq-name' + (on ? ' cal-jq-on' : '') +
                (right ? ' cal-jq-split' : '') + '"' +
                (on ? ' id="calJqActive"' : '') + '>' +
                esc(zh ? TK_ZH[k] : TK_VI[k]) + '</td>' +
                '<td class="dp-num cal-jq-date' + (on ? ' cal-jq-on' : '') +
                (right ? ' cal-jq-last' : '') + '">' +
                esc(dates[k] || '') + '</td>';
        };
        var rows = '';
        for (var k = 0; k < 12; k++) {
            rows += '<tr' + (k % 2 === 0 ? ' class="dp-row-alt"' : '') + '>' +
                cell(k, false) + cell(k + 12, true) + '</tr>';
        }
        return rows;
    }

    /** Bảng 24 dòng phải cuộn; đưa tiết khí đang hiệu lực vào giữa khung nhìn. */
    function scrollToActiveJieQi() {
        var row = document.getElementById('calJqActive');
        var body = document.querySelector('#calJieQi .cal-jq-body');
        if (!row || !body) return;
        body.scrollTop = Math.max(0,
            row.offsetTop - body.clientHeight / 2 + row.offsetHeight / 2);
    }


    function shiftMonth(delta) {
        var d = new Date(viewY, viewM - 1 + delta, 1);
        viewY = d.getFullYear();
        viewM = d.getMonth() + 1;
        render();
    }

    /* ─────────────── Chuyển tab ─────────────── */

    function showTab(which) {
        var cal = which === 'cal';
        document.body.classList.toggle('view-cal', cal);
        var tq = document.getElementById('tabQmdj');
        var tc = document.getElementById('tabCal');
        if (tq) tq.classList.toggle('tab-active', !cal);
        if (tc) tc.classList.toggle('tab-active', cal);
        if (cal) render();
        if (typeof window.__fitScreen === 'function') setTimeout(window.__fitScreen, 50);
        // Chỉ cuộn khi đang không ở đầu trang — gọi thừa vừa vô ích vừa làm
        // jsdom kêu "not implemented" trong bộ kiểm thử.
        if (window.scrollY) { try { window.scrollTo(0, 0); } catch (e) {} }
    }
    window.showTab = showTab;

    /**
     * Nút ghim widget Lịch ra màn hình chính. Chỉ hiện khi chạy trong ứng dụng
     * Android — mở bằng trình duyệt thì không có widget nào để ghim.
     */
    function setupPinButton() {
        var btn = document.getElementById('calPinBtn');
        var note = document.getElementById('calPinNote');
        var native = window.QMDJNative;
        if (!btn || !native || typeof native.pinCalendarWidget !== 'function') return;

        btn.style.display = 'block';
        btn.addEventListener('click', function () {
            var res = '';
            try { res = native.pinCalendarWidget(); } catch (e) { res = 'error'; }
            note.style.display = 'block';
            note.textContent = (res === 'ok')
                ? t('pinOk')
                : t('pinManual');
            // Ghi chú vừa hiện ra chiếm thêm chỗ — chia lại ngay, không thì nó
            // đẩy chính nó xuống dưới thanh tab.
            fitGrid(lastWeeks);
        });
    }

    /** Nhãn tab đổi theo ngôn ngữ. */
    function refreshLabels() {
        var tq = document.getElementById('tabQmdj');
        var tc = document.getElementById('tabCal');
        if (tq) tq.querySelector('.tab-lbl').textContent = t('tabQmdj');
        if (tc) tc.querySelector('.tab-lbl').textContent = t('tabCal');
        var pin = document.getElementById('calPinBtn');
        if (pin) pin.textContent = t('pin');
        if (document.body.classList.contains('view-cal')) render();
    }
    window.__calRefreshLabels = refreshLabels;

    /**
     * Ghi bảng tháng âm quanh hôm nay ra kho tuỳ chọn cho WIDGET dùng.
     *
     * Widget không chạy được lunar.js nên vẫn có bảng tra đóng sẵn trong APK —
     * nhưng bảng ấy chỉ đúng tuyệt đối ở mốc UTC+7. Suy mốc mùng 1 sang múi giờ
     * khác bằng điểm Sóc chỉ gần đúng: lunar.js không định mùng 1 thuần tuý
     * bằng "lấy phần nguyên của điểm Sóc theo múi giờ", nên còn lệch ~0,35% số
     * tháng.
     *
     * Chỗ duy nhất biết chắc câu trả lời là ỨNG DỤNG, vì nó có lunar.js. Nên
     * mỗi lần vẽ lịch, ghi luôn ra vài chục tháng quanh hôm nay ở đúng múi giờ
     * đang chọn; widget đọc bảng này trước, không có mới quay về bảng đóng sẵn.
     *
     * Định dạng: "<múi giờ phút>|<JDN mùng 1>,<tháng>,<nhuận>;…"
     */
    function publishLunarCache() {
        if (typeof ShouXingUtil === 'undefined' || typeof Solar === 'undefined') return;
        try {
            var tz = localTz();
            var now = new Date();
            var rows = [];
            // Lấy thẳng danh sách tháng đã chỉnh theo Chính Tý — cùng nguồn với
            // tab Lịch và tab Kỳ Môn, nên widget không thể lệch với ứng dụng.
            var info = countryData[getDOM('country').value];
            if (!info || !info.tzId || typeof zi_months !== 'function') return;
            var list = zi_months(now.getFullYear(), info.lon, info.tzId, tz);
            for (var i = 0; i < list.length; i++) {
                rows.push(list[i].jdn + ',' + list[i].month + ',' + (list[i].leap ? 1 : 0));
            }
            if (rows.length) prefSet(K_LUNAR_CACHE, Math.round(tz * 60) + '|' + rows.join(';'));
        } catch (e) {
            try { ShouXingUtil.setTzOffsetHours(null); } catch (e2) {}
        }
    }

    /** Số ngày Julius — cùng công thức với LunarTable.jdn bên Kotlin. */
    function jdnOf(y, m, d) {
        var a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
        return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
            + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
    }

    /** Nhảy tới một ngày cụ thể — dùng cho kiểm thử. */
    window.__calGoto = function (y, m, d) {
        viewY = y; viewM = m;
        selected = { y: y, m: m, d: d };
        render();
    };

    window.addEventListener('resize', function () {
        if (document.body.classList.contains('view-cal')) setTimeout(render, 180);
    });

    document.addEventListener('DOMContentLoaded', function () {
        var now = new Date();
        viewY = now.getFullYear();
        viewM = now.getMonth() + 1;
        selected = { y: now.getFullYear(), m: now.getMonth() + 1, d: now.getDate() };

        document.getElementById('tabQmdj').addEventListener('click', function () { showTab('qmdj'); });
        document.getElementById('tabCal').addEventListener('click', function () { showTab('cal'); });
        document.getElementById('calPrev').addEventListener('click', function () { shiftMonth(-1); });
        document.getElementById('calNext').addEventListener('click', function () { shiftMonth(1); });
        document.getElementById('calTitle').addEventListener('click', function () {
            var n = new Date();
            viewY = n.getFullYear(); viewM = n.getMonth() + 1;
            selected = { y: viewY, m: viewM, d: n.getDate() };
            render();
        });
        document.getElementById('calGrid').addEventListener('click', function (e) {
            var cell = e.target.closest ? e.target.closest('.cal-day') : null;
            if (!cell) return;
            selected = {
                y: parseInt(cell.getAttribute('data-y'), 10),
                m: parseInt(cell.getAttribute('data-m'), 10),
                d: parseInt(cell.getAttribute('data-d'), 10),
            };
            // Chạm vào ngày của tháng trước/sau thì chuyển hẳn sang tháng đó.
            viewY = selected.y;
            viewM = selected.m;
            render();
        });

        setupPinButton();
        refreshLabels();

        // Đổi ngôn ngữ thì vẽ lại nhãn tab và lưới lịch.
        if (typeof toggleLang === 'function' && !toggleLang.__calWrapped) {
            var orig = toggleLang;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                try { refreshLabels(); } catch (err) { console.warn('calendar:', err); }
                return r;
            };
            wrapped.__calWrapped = true;
            window.toggleLang = wrapped;
        }
    });
})();
