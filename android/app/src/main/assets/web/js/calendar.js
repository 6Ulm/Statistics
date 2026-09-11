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
        colGz:    { vi: 'Can chi',    zh: '月柱' },
        secJq:    { vi: 'Tiết khí',   zh: '节气' },
        secAm:    { vi: 'Lịch âm',    zh: '农历' },
        colMonth: { vi: 'Tháng âm',   zh: '农历月' },
        colSoc:   { vi: 'Sóc',        zh: '朔' },
        colVong:  { vi: 'Vọng',       zh: '望' },
        leap:     { vi: 'Nhuận',      zh: '闰' },
        pin:      { vi: '📌 Ghim lịch ra màn hình chính', zh: '📌 固定日历到主屏幕' },
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
    /** Mục đang mở không bao giờ thấp hơn chừng này — thấp quá thì vô dụng. */
    var SEC_MIN = 96;
    /** Lề trên #calSections cộng lề giữa hai mục (xem calendar.css). */
    var SEC_MARGIN = 11;

    /** Khoá kho tuỳ chọn: bảng tháng âm cho widget (xem publishLunarCache). */
    var K_LUNAR_CACHE = 'qmdj.lunarCache';
    /** Hai mục gập được của tab Lịch — mở/đóng độc lập, nhớ qua các lần mở app. */
    var K_SEC_JQ = 'qmdj.calSecJq';
    var K_SEC_AM = 'qmdj.calSecAm';
    /** Ngôn ngữ đang chọn, để widget vẽ đúng thứ tiếng (xem LunarTable.langOf). */
    var K_LANG = 'qmdj.lang';
    var openJq = true, openAm = false;

    var viewY, viewM;          // tháng đang xem (dương lịch)
    var selected = null;       // {y,m,d}
    var lastWeeks = 0;         // số hàng của lưới đang hiện, để đo lại khi gập

    /**
     * Bối cảnh cho MỘT lượt vẽ lịch: danh sách tháng âm và mốc can chi, dựng
     * đúng một lần rồi dùng cho cả 42 ô.
     *
     * Trước đây mỗi ô tự gọi getDOM('country'), getTimezoneOffset (Intl, đắt)
     * và dựng một đối tượng Lunar riêng — 42 lần mỗi lần vẽ, mà 41 lần trong
     * đó cho ra cùng một câu trả lời.
     */
    function buildCtx(anchorY, anchorM) {
        try {
            if (typeof zi_months !== 'function') return null;
            var info = countryData[getDOM('country').value];
            if (!info || !info.tzId) return null;
            var tz = getTimezoneOffset(info.tzId, new Date(anchorY, anchorM - 1, 15, 12));
            return { list: zi_months(anchorY, info.lon, info.tzId, tz) };
        } catch (e) { return null; }
    }

    /** Ngày âm của một ngày dương, tra trong bối cảnh đã dựng sẵn. */
    function ziFromCtx(ctx, jdn) {
        if (!ctx || !ctx.list) return null;
        var list = ctx.list, at = -1;
        for (var i = 0; i < list.length; i++) {
            if (list[i].jdn <= jdn) at = i; else break;
        }
        if (at < 0) return null;
        // Kèm cả NĂM âm: mục "Lịch âm" cần nó để dựng đúng 12-13 tháng của năm
        // đang xem. Bỏ sót thì Ephem.monthsAtBasis nhận undefined và ném lỗi.
        return { day: jdn - list[at].jdn + 1, month: list[at].month,
                 leap: list[at].leap, year: list[at].year };
    }

    /** Số ngày Julius — trùng công thức với zi_jdn bên app.js. */
    function jdnOfDate(y, m, d) {
        var a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
        return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
            + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
    }

    /**
     * Can chi ngày suy từ số ngày Julius. Mốc lấy MỘT LẦN từ lunar.js ở lượt
     * vẽ đầu tiên, nên không phải chép cứng hằng số vào đây.
     */
    var gzEpoch = null;        // {jdn, can, chi} của một ngày đã biết
    function ganZhiByJdn(ctx, jdn) {
        if (gzEpoch === null) {
            var s0 = Solar.fromYmd(2000, 1, 1);
            var l0 = s0.getLunar();
            var c0 = CAN_ZH.indexOf(l0.getDayGan()), z0 = CHI_ZH.indexOf(l0.getDayZhi());
            if (c0 < 0 || z0 < 0) return { can: '', chi: '' };
            gzEpoch = { jdn: jdnOfDate(2000, 1, 1), can: c0, chi: z0 };
        }
        var n = jdn - gzEpoch.jdn;
        var can = ((gzEpoch.can + n) % 10 + 10) % 10;
        var chi = ((gzEpoch.chi + n) % 12 + 12) % 12;
        var zh = isZH();
        return { can: zh ? CAN_ZH[can] : CAN_VI[can], chi: zh ? CHI_ZH[chi] : CHI_VI[chi] };
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

        // Một bối cảnh cho cả lưới: danh sách tháng âm + mốc can chi.
        var ctx = buildCtx(viewY, viewM);

        var cells = [];
        for (var i = lead; i > 0; i--) {
            cells.push(cellHtml(prevY, prevM, prevDays - i + 1, cells.length, true, tKey, ctx));
        }
        for (var d = 1; d <= daysInMonth; d++) {
            cells.push(cellHtml(viewY, viewM, d, cells.length, false, tKey, ctx));
        }
        for (var nd = 1; cells.length % 7 !== 0; nd++) {
            cells.push(cellHtml(nextY, nextM, nd, cells.length, true, tKey, ctx));
        }

        for (var r = 0; r < cells.length; r += 7) {
            html += '<div class="cal-row">' + cells.slice(r, r + 7).join('') + '</div>';
        }
        document.getElementById('calGrid').innerHTML = html;

        renderJieQi();
        clearLunarBasis();
        lastWeeks = cells.length / 7;
        fitGrid(lastWeeks);
        renderAmBan();
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
        // Cả THANH DƯỚI, không riêng hàng tab: hàng ngôn ngữ + địa điểm nằm
        // dưới hai tab trong cùng thanh ấy. Trừ thiếu nó thì lưới lịch cộng
        // bảng tiết khí chiếm quá chỗ và tab Lịch tràn khỏi một màn hình.
        var bar = document.getElementById('bottomDock') || document.getElementById('tabBar');
        var dow = grid ? grid.querySelector('.cal-dow') : null;
        // Nút "Ghim lịch ra màn hình chính" CHỈ hiện khi chạy trong ứng dụng
        // Android, nên mọi phép đo trên trình duyệt đều không thấy nó. Không trừ
        // ra thì trên máy thật lưới lịch với bảng tiết khí chiếm trọn màn hình
        // rồi đẩy nút xuống dưới, nằm khuất sau thanh tab cố định.
        var pin = document.getElementById('calPinBtn');
        if (!grid || !head || !dow) return;

        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var h = function (el) { return el ? el.getBoundingClientRect().height / zoom : 0; };

        // Đo phần cố định — KHÔNG tính thân bảng tiết khí, vì chính nó là thứ
        // ta sắp chia. Đo cả cụm rồi mới chia thì thành vòng luẩn quẩn.
        var vis = function (el) {
            return (el && getComputedStyle(el).display !== 'none') ? h(el) : 0;
        };
        var avail = window.innerHeight / zoom
            - h(head) - h(dow) - h(bar) - vis(pin) - GRID_CHROME;

        // Thanh tiêu đề nay là hàng <thead> NẰM TRONG khung cuộn của mỗi mục.
        // Mục đang MỞ thì chiều cao ấy đã nằm trong phần được chia; mục đã ĐÓNG
        // thì khung co lại vừa đúng hàng tiêu đề, và chừng ấy vẫn chiếm chỗ —
        // phải trừ ra trước khi chia, không thì tab Lịch tràn đúng bằng tổng
        // chiều cao hai hàng tiêu đề.
        var secs = [['calJieQi', openJq], ['calAmBan', openAm]];
        for (var si = 0; si < secs.length; si++) {
            if (!secs[si][1]) avail -= h(document.getElementById(secs[si][0]));
        }
        avail -= SEC_MARGIN;

        // Không mục nào mở: lưới lịch lấy hết phần còn lại.
        if (!openJq && !openAm) {
            var rowFull = Math.max(ROW_MIN, Math.min(ROW_MAX, Math.floor(avail / weeks)));
            document.documentElement.style.setProperty('--cal-row-h', rowFull + 'px');
            return;
        }

        // Lưới lấy phần của nó trước (có trần), phần còn lại chia cho các mục
        // đang mở — nhờ vậy màn hình cao không còn hở một mảng ở đáy. Đo chiều
        // cao THẬT của từng bảng thay vì giữ sẵn một khoản cố định: giữ 320px
        // mà bảng chỉ cao 280px thì 40px kia thành khoảng hở. Đo phần tử
        // <table> chứ không phải khung cuộn — khung đang bị max-height của lần
        // chia trước cắt ngắn.
        var natOf = function (id) {
            var tb = document.querySelector('#' + id + ' table.cal-jq');
            return tb ? Math.ceil(tb.getBoundingClientRect().height / zoom) + 2 : JQ_FALLBACK;
        };
        var want = (openJq ? natOf('calJieQi') : 0) + (openAm ? natOf('calAmBan') : 0);
        var secH = Math.min(want, Math.max(SEC_MIN, avail - ROW_MIN * weeks));

        var rowH = Math.max(ROW_MIN, Math.min(ROW_MAX, Math.floor((avail - secH) / weeks)));
        document.documentElement.style.setProperty('--cal-row-h', rowH + 'px');
        shareSectionHeight(Math.max(SEC_MIN, Math.floor(avail - rowH * weeks)));
    }

    /**
     * Dựng HTML một ô ngày.
     * @param {number} idx    thứ tự ô trong lưới (để biết cột Chủ nhật)
     * @param {boolean} outside  ngày của tháng trước/sau — tô mờ
     */
    function cellHtml(y, m, d, idx, outside, tKey, ctx) {
        var jdn = jdnOfDate(y, m, d);
        // Can chi ngày là chu kỳ 60 ngày liên tục theo số ngày Julius — suy
        // thẳng bằng số học, khỏi dựng một đối tượng Lunar cho mỗi ô.
        var gz = ganZhiByJdn(ctx, jdn);
        // Ngày âm lấy theo ranh giới CHÍNH TÝ (xem khối ghi chú trong app.js):
        // mùng 1 là ngày chứa điểm Sóc, đếm từ nửa đêm mặt trời thật chứ không
        // phải 00:00. Hỏng thì lùi về số của lunar.js còn hơn để trống cả lịch.
        var zl = ziFromCtx(ctx, jdn);
        var lunar = zl ? null : Solar.fromYmd(y, m, d).getLunar();
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

        // Một nhóm 24 hàng, ba cột. KHÔNG bọc thêm .dp-table-wrap: nó có
        // overflow-x nên trở thành vùng cuộn gần nhất của <th> sticky, mà chính
        // nó lại không giới hạn chiều cao — hàng tiêu đề vì thế trôi mất khi
        // cuộn. Chính .cal-sec-body cuộn là đủ.
        box.innerHTML =
            '<table class="dp-table cal-jq"><thead><tr class="cal-sec-head">' +
            '<th>' + t('colTk') + '</th>' +
            '<th class="c">' + t('colDate') + '</th>' +
            '<th class="c cal-jq-last">' + t('colGz') +
            '<span class="cal-sec-chev"></span></th>' +
            '</tr></thead><tbody id="calJqBody">' + rows + '</tbody></table>';
        setTimeout(scrollToActiveJieQi, 40);
        return true;
    }

    /**
     * Can chi THÁNG của tháng chứa mốc tiết khí ấy.
     *
     * Lấy từ chính engine (lunar.js) chứ không tự suy từ chỉ số tiết khí: can
     * tháng phụ thuộc can năm, mà năm can chi lại đổi ở Lập Xuân — tự dựng lại
     * luật ấy là mời thêm một nguồn sai lệch nữa với tab Kỳ Môn.
     *
     * Nhận thẳng NGÀY JULIUS Ở MỐC UTC+8, không nhận chuỗi giờ địa phương.
     * Quy ngược chuỗi ấy về UTC+8 cần offset ĐÚNG CỦA CHÍNH MỐC ĐÓ, trong khi
     * `tz` của bảng là offset của ngày đang chọn — ở nước có DST thì hai thứ
     * lệch nhau một giờ suốt nửa năm, đủ để Lập Xuân rơi về tháng Sửu thay vì
     * mở tháng Dần. Mà can chi tháng vốn là đại lượng ở UTC+8, nên đi thẳng.
     */
    function monthGanZhiAt(jdUTC8) {
        if (typeof Solar === 'undefined' || typeof Ephem === 'undefined') return '';
        if (!isFinite(jdUTC8)) return '';
        try {
            // Nhích 2 phút qua mốc giao tiết: đúng tại mốc, phép làm tròn trong
            // lunar.js có thể còn xếp về tháng cũ. 2 phút thì chắc chắn đã sang
            // tháng mới mà vẫn cách tiết sau cả nửa tháng.
            var gz = Ephem.atBasis(null, function () {
                return Solar.fromJulianDay(jdUTC8 + 2 / 1440)
                    .getLunar().getEightChar().getMonth();
            });
            if (!gz || gz.length < 2) return '';
            return isZH() ? gz
                : (getDisplayCan(gz[0]) + ' ' + getDisplayChi(gz[1]));
        } catch (e) { return ''; }
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
        var res = Ephem.atBasis(tz, function () {
            var Y = sb_findY(sel.y, sel.m, sel.d, 12, 0, tzId, tz);
            return { Y: Y, dates: sb_getJieQiDates(Y, tzId, tz) };
        });
        var Y = res.Y, dates = res.dates;
        // Cùng dãy mốc mà sb_getJieQiDates dùng, nhưng giữ nguyên ngày Julius ở
        // UTC+8 để suy can chi tháng (xem monthGanZhiAt).
        var jds = Ephem.jieQiJdAtBasis(Y + 1, null);

        // Tiết khí đang hiệu lực = mốc CUỐI CÙNG không muộn hơn ngày đang chọn.
        // Lấy 12:00 trưa làm mốc so: chọn 00:00 thì đúng ngày giao tiết sẽ rơi
        // về tiết trước, mà lịch chỉ có độ phân giải một ngày.
        var at = Date.UTC(sel.y, sel.m - 1, sel.d, 12, 0) / 60000;
        var active = 0;
        for (var i = 0; i < 24; i++) {
            var ts = parseDt(dates[i]);
            if (!isNaN(ts) && ts <= at) active = i;
        }

        // Một dãy 24 hàng liền, thêm cột can chi tháng. Can chi của một tháng
        // phủ đúng hai tiết khí (tiết rồi khí), nên hai hàng liền nhau lặp lại
        // cùng một giá trị — đó là đúng, không phải trùng lặp thừa.
        var zh = isZH();
        var rows = '';
        for (var k = 0; k < 24; k++) {
            var on = k === active;
            rows += '<tr' + (k % 2 === 0 ? ' class="dp-row-alt"' : '') + '>' +
                '<td class="cal-jq-name' + (on ? ' cal-jq-on' : '') + '"' +
                (on ? ' id="calJqActive"' : '') + '>' +
                esc(zh ? TK_ZH[k] : TK_VI[k]) + '</td>' +
                '<td class="dp-num cal-jq-date' + (on ? ' cal-jq-on' : '') + '">' +
                esc(dates[k] || '') + '</td>' +
                '<td class="cal-jq-gz cal-jq-last' + (on ? ' cal-jq-on' : '') + '">' +
                esc(monthGanZhiAt(jds[k + 1])) + '</td>' +
                '</tr>';
        }
        return rows;
    }

    /**
     * Mục "Lịch âm": đúng bảng của Âm Bàn pháp ở tab Kỳ Môn (Tháng âm · Sóc ·
     * Vọng), dựng lại tại đây từ CÙNG những hàm ấy để hai tab không thể lệch.
     */
    function renderAmBan() {
        var box = document.getElementById('calAmBan');
        if (!box) return;
        if (typeof Ephem === 'undefined' || typeof Solar === 'undefined' ||
            typeof formatPreciseSocLocal !== 'function' ||
            typeof formatPreciseVongLocal !== 'function') { box.innerHTML = ''; return; }
        try {
            var sel = selected || { y: viewY, m: viewM, d: 1 };
            var info = countryData[getDOM('country').value];
            if (!info || !info.tzId) { box.innerHTML = ''; return; }
            var tz = getTimezoneOffset(info.tzId, new Date(sel.y, sel.m - 1, sel.d, 12));

            // Tháng âm của ngày đang chọn, để tô đậm đúng hàng.
            var ctx = buildCtx(sel.y, sel.m);
            var cur = ctx ? ziFromCtx(ctx, jdnOfDate(sel.y, sel.m, sel.d)) : null;
            var curYear = cur ? cur.year : sel.y;
            var curMonth = cur ? (cur.leap ? -cur.month : cur.month) : 0;

            var months = Ephem.monthsAtBasis(curYear, tz);
            var rows = '';
            for (var i = 0; i < months.length; i++) {
                var mo = months[i];
                var moNum = mo.leap ? -mo.month : mo.month;
                var socSolar = Solar.fromJulianDay(mo.jd);
                var on = (moNum === curMonth);
                var label = (isZH() ? mo.month + '月' : 'Tháng ' + mo.month) +
                    (mo.leap ? ' (' + t('leap') + ')' : '');
                rows += '<tr' + (i % 2 === 0 ? ' class="dp-row-alt"' : '') + '>' +
                    '<td class="cal-jq-name' + (on ? ' cal-jq-on' : '') + '"' +
                    (on ? ' id="calAmActive"' : '') + '>' + esc(label) + '</td>' +
                    '<td class="dp-num cal-jq-date' + (on ? ' cal-jq-on' : '') + '">' +
                    esc(formatPreciseSocLocal(socSolar, info.tzId)) + '</td>' +
                    '<td class="dp-num cal-jq-date cal-jq-last' + (on ? ' cal-jq-on' : '') + '">' +
                    esc(formatPreciseVongLocal(socSolar, info.tzId)) + '</td>' +
                    '</tr>';
            }
            box.innerHTML =
                '<table class="dp-table cal-jq"><thead><tr class="cal-sec-head">' +
                '<th>' + t('colMonth') + '</th>' +
                '<th class="c">' + t('colSoc') + '</th>' +
                '<th class="c cal-jq-last">' + t('colVong') +
                '<span class="cal-sec-chev"></span></th>' +
                '</tr></thead><tbody>' + rows + '</tbody></table>';
        } catch (e) {
            console.warn('calAmBan:', e);
            box.innerHTML = '';
        }
    }

    /** Bảng 24 dòng phải cuộn; đưa tiết khí đang hiệu lực vào giữa khung nhìn. */
    /* ─────────────── Hai mục gập được ─────────────── */

    /** Áp trạng thái mở/đóng lên DOM (không chia lại chiều cao). */
    function applySections() {
        var pairs = [['calSecJq', openJq], ['calSecAm', openAm]];
        for (var i = 0; i < pairs.length; i++) {
            var sec = document.getElementById(pairs[i][0]);
            if (!sec) continue;
            sec.classList.toggle('cal-sec-open', pairs[i][1]);
            var chev = sec.querySelector('.cal-sec-chev');
            if (chev) chev.textContent = pairs[i][1] ? '▾' : '▸';
            // Mục đã đóng thì trả lại chiều cao tự nhiên (chỉ còn hàng tiêu đề),
            // không thì trần của lần chia trước còn chừa một khoảng trống.
            var body = sec.querySelector('.cal-sec-body');
            if (body && !pairs[i][1]) body.style.maxHeight = '';
        }
    }

    function toggleSection(which) {
        if (which === 'jq') { openJq = !openJq; prefSet(K_SEC_JQ, openJq ? '1' : '0'); }
        else                { openAm = !openAm; prefSet(K_SEC_AM, openAm ? '1' : '0'); }
        applySections();
        fitGrid(lastWeeks);
        if (which === 'jq' && openJq) setTimeout(scrollToActiveJieQi, 40);
    }

    /**
     * Chia chiều cao còn lại cho những mục ĐANG MỞ.
     *
     * Vừa đủ chỗ thì mỗi mục lấy đúng chiều cao thật của nó — không mục nào
     * phải cuộn. Chật thì chia theo tỉ lệ chiều cao thật, nên mục dài (24 tiết
     * khí) được phần lớn hơn mục ngắn (12-13 tháng âm), thay vì cưa đôi rồi
     * mục ngắn thừa chỗ còn mục dài cuộn mỏi tay.
     */
    function shareSectionHeight(avail) {
        var list = [];
        if (openJq) list.push(document.getElementById('calJieQi'));
        if (openAm) list.push(document.getElementById('calAmBan'));
        list = list.filter(Boolean);
        if (!list.length) return;
        // Trần phải tính cả hàng tiêu đề, vì nó nằm TRONG khung cuộn.


        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var nat = list.map(function (el) {
            var tb = el.querySelector('table');
            return tb ? Math.ceil(tb.getBoundingClientRect().height / zoom) + 2 : SEC_MIN;
        });
        var sum = nat.reduce(function (a, b) { return a + b; }, 0);
        for (var i = 0; i < list.length; i++) {
            var hgt = (sum <= avail) ? nat[i]
                : Math.max(SEC_MIN, Math.floor(avail * nat[i] / sum));
            list[i].style.maxHeight = hgt + 'px';
        }
    }

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
        var native = window.QMDJNative;
        if (!btn || !native || typeof native.pinCalendarWidget !== 'function') return;

        btn.style.display = 'block';
        btn.addEventListener('click', function () {
            // Hệ thống tự hiện hộp thoại xác nhận; không kèm dòng ghi chú nào.
            try { native.pinCalendarWidget(); } catch (e) {}
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
        publishLang();
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

    /**
     * Ghi ngôn ngữ đang chọn cho widget, rồi bảo nó vẽ lại NGAY.
     *
     * Widget đọc kho tuỳ chọn lúc vẽ, mà nó chỉ tự vẽ lại lúc nửa đêm hoặc khi
     * người dùng bấm ‹ › — nên nếu chỉ ghi khoá mà không gọi, đổi sang tiếng
     * Trung trong ứng dụng thì lịch đã ghim vẫn còn tiếng Việt hàng giờ liền.
     */
    function publishLang() {
        prefSet(K_LANG, isZH() ? 'zh' : 'vi');
        try {
            var n = window.QMDJNative;
            if (n && typeof n.refreshCalendarWidget === 'function') n.refreshCalendarWidget();
        } catch (e) {}
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

        // Trạng thái gập: mặc định mở Tiết khí, đóng Lịch âm — mở cả hai ngay
        // từ đầu thì trên máy thấp lưới lịch bị bóp về ROW_MIN.
        var sj = prefGet(K_SEC_JQ), sa = prefGet(K_SEC_AM);
        if (sj === '0' || sj === '1') openJq = sj === '1';
        if (sa === '0' || sa === '1') openAm = sa === '1';
        applySections();
        // Uỷ quyền: hàng tiêu đề nằm trong bảng, mà bảng thì dựng lại mỗi lần
        // vẽ — gắn thẳng vào nó thì cứ đổi tháng là mất người nghe.
        document.getElementById('calSections').addEventListener('click', function (e) {
            var head = e.target.closest ? e.target.closest('.cal-sec-head') : null;
            if (!head) return;
            var sec = head.closest('.cal-sec');
            if (sec) toggleSection(sec.id === 'calSecJq' ? 'jq' : 'am');
        });

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

        // Ô địa điểm giờ nằm ở #sharedBar, dùng chung hai tab — nên đổi được
        // ngay khi đang mở tab Lịch. Lịch vốn đọc countryData lúc vẽ, nhưng
        // không tự biết là vừa đổi; vẽ lại theo mỗi lần engine tính lại thì cả
        // hai tab luôn nói cùng một địa điểm. (applyLoc trong location.js gọi
        // processAll sau khi áp vị trí mới.)
        if (typeof processAll === 'function' && !processAll.__calRecalcWrapped) {
            var origCalc = processAll;
            var wrappedCalc = function () {
                var r = origCalc.apply(this, arguments);
                if (document.body.classList.contains('view-cal')) {
                    try { render(); } catch (err) { console.warn('calendar:', err); }
                }
                // Địa điểm đổi thì giờ tiết khí và bảng tháng của widget cũng
                // đổi theo — publishLunarCache đã ghi bảng mới, còn đây là chỗ
                // bảo widget vẽ lại.
                try {
                    var n = window.QMDJNative;
                    if (n && typeof n.refreshCalendarWidget === 'function') n.refreshCalendarWidget();
                } catch (e2) {}
                return r;
            };
            wrappedCalc.__calRecalcWrapped = true;
            window.processAll = wrappedCalc;
        }
    });
})();
