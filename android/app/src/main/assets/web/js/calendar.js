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

    /**
     * Đệm dự phòng, KHÔNG phải phép cộng lề — lề nay do measureChrome() đo.
     * Giữ vài pixel để phép làm tròn nửa pixel của trình duyệt không đẩy khối
     * cuối thò xuống dưới thanh tab.
     */
    var GRID_CHROME = 14;
    // SÀN chiều cao một hàng lịch — `min-height`, không phải chiều cao chốt.
    // Ô ngày chứa ba dòng (số ngày + ngày âm, can, chi); ba dòng ấy cần hơn thì
    // hàng tự cao thêm, và fitGrid ĐO lưới thật chứ không nhân ROW_MIN × số
    // tuần. Chỗ thừa của màn hình cao đổ vào hai mục, không kéo hàng lịch cao
    // ra nữa.
    var ROW_MIN = 52;
    /** Mục đang mở không bao giờ thấp hơn chừng này — thấp quá thì vô dụng. */
    var SEC_MIN = 96;

    /**
     * Chỗ để syncSectionColumns() đặt bề rộng ba cột. Hai mục là HAI bảng riêng
     * nên mỗi bảng tự co theo dữ liệu của mình — "Dương lịch" không thẳng hàng
     * với "Sóc", "Can chi" không thẳng hàng với "Vọng" (đo trên máy 393px: lệch
     * 9px và 89px). Hai bảng nằm ngay trên dưới nhau nên lệch là thấy ngay.
     */
    var COLGROUP = '<colgroup><col><col><col></colgroup>';
    /**
     * Phần khung cố định của tab Lịch, ĐO THẲNG trên DOM:
     *   outer  — lề trên/dưới của #calView cộng mọi khe giữa các khối con
     *            (đầu lịch, lưới, cụm hai mục, nút ghim);
     *   secGap — khe giữa hai mục.
     * Toàn là lề CSS nên không đổi theo chiều cao đang chia.
     */
    function measureChrome() {
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var view = document.getElementById('calView');
        var out = { outer: 0, secGap: 0 };
        if (!view) return out;
        var cs = getComputedStyle(view);
        out.outer = (parseFloat(cs.paddingTop) || 0) + (parseFloat(cs.paddingBottom) || 0);
        var kids = [];
        for (var i = 0; i < view.children.length; i++) {
            var el = view.children[i];
            if (getComputedStyle(el).display !== 'none') kids.push(el);
        }
        for (i = 1; i < kids.length; i++) {
            out.outer += Math.max(0,
                (kids[i].getBoundingClientRect().top
                 - kids[i - 1].getBoundingClientRect().bottom) / zoom);
        }
        var jq = document.getElementById('calSecJq');
        var am = document.getElementById('calSecAm');
        if (jq && am) {
            out.secGap = Math.max(0,
                (am.getBoundingClientRect().top - jq.getBoundingClientRect().bottom) / zoom);
        }
        return out;
    }

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
        // LUÔN đủ 6 hàng (42 ô), không phải "điền cho tròn tuần".
        //
        // Tháng dương có 4, 5 hay 6 hàng tuỳ ngày mùng 1 rơi vào thứ mấy —
        // tháng 2/2026 gọn đúng 4 hàng, tháng 11/2026 cần 6. Điền cho tròn
        // tuần thì lưới cao thấp theo từng tháng, kéo CẢ HAI tiêu đề bên dưới
        // nhảy 58px (4→5 hàng) tới 116px (4→6 hàng) mỗi lần bấm ‹ ›. Lịch đã
        // ghim ngoài màn hình chính vốn LUÔN vẽ 6 hàng (GRID_WEEKS trong
        // CalendarWidgetProvider.kt, vì 42 ô chạm là cố định), nên giữ nguyên
        // 6 hàng ở đây còn cho hai bên giống hệt nhau — thứ người dùng đòi
        // nhiều lần. Trả giá bằng một hàng ngày mờ thừa ở vài tháng; đổi lại
        // bố cục đứng yên quanh năm.
        for (var nd = 1; cells.length < 42; nd++) {
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
        // Hai bảng vừa dựng lại từ đầu: ghim lại bề rộng cột dùng chung, rồi
        // đặt lại trần kẹp của mục đang đóng — không thì mục đóng bung ra đủ
        // 24 hàng sau mỗi lần vẽ. Đúng thứ tự này: kẹp chiều cao đo <thead>,
        // mà <thead> chỉ đứng yên sau khi cột đã ghim xong.
        syncSectionColumns();
        applySections();
        // Chia lại chiều cao LẦN CUỐI, khi hai bảng đã dựng xong và cột đã
        // ghim. Lượt fitGrid ở trên chạy TRƯỚC renderAmBan(), nên lúc ấy hàng
        // tiêu đề Lịch âm còn là của lần vẽ trước (lần vẽ đầu tiên thì chưa
        // có) và `avail` dôi ra đúng một hàng tiêu đề. Đo trên A51: Tiết khí
        // được 224px lúc vừa mở tab rồi tụt còn 208px ngay khi người dùng chạm
        // vào Lịch âm — tiêu đề nhảy 16px ngay dưới ngón tay.
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
        // ĐO phần khung cố định thay vì cộng hằng số: lề trên/dưới của #calView
        // và các khe giữa bốn khối con của nó. Hai hằng số GRID_CHROME và
        // SEC_MARGIN trước đây phải gánh đúng những con số ấy bằng tay, mà chỉ
        // cần sửa một lề trong CSS là chúng lệch — lệch rồi thì nội dung tràn
        // khỏi màn hình, viewport.js thu nhỏ cả trang để chữa, và chữ bé lại
        // đúng lúc ta vừa cố phóng to. Lề là hằng số của CSS, không phụ thuộc
        // chiều cao đang chia, nên đo lúc nào cũng ra một kết quả.
        var chrome = measureChrome();

        // CHỐT chiều cao hàng lịch TRƯỚC, rồi ĐO lưới thật — không nhân
        // rowH × weeks nữa.
        //
        // rowH chỉ là SÀN: ô ngày chứa ba dòng (số ngày + ngày âm, can, chi) và
        // nếu ba dòng ấy cần hơn rowH thì hàng tự cao thêm, `min-height` không
        // cản. Đem rowH × weeks đi trừ là khai thiếu đúng phần chênh — đo trên
        // A51 với cỡ chữ mới: khai 6×52 = 312px trong khi lưới thật cao 342px,
        // hụt 30px. Hụt bao nhiêu thì cụm hai mục thò xuống dưới thanh tab bấy
        // nhiêu, và viewport.js thu nhỏ CẢ TRANG để chữa — chữ bé lại đúng lúc
        // vừa cố phóng to. Đo thẳng thì sai số ấy biến mất.
        document.documentElement.style.setProperty('--cal-row-h', ROW_MIN + 'px');
        var gridH = h(grid);

        // Phần còn lại là của HAI KHUNG MỤC (kể cả hàng tiêu đề nằm trong
        // chúng) cộng khe giữa hai mục.
        var budget = window.innerHeight / zoom
            - h(head) - gridH - h(bar) - vis(pin)
            - chrome.outer - chrome.secGap - GRID_CHROME;
        budget = Math.max(0, Math.floor(budget));

        shareSectionHeight(budget);
    }

    /**
     * Chừng nào người dùng chưa tự bấm: quyết hộ xem có mở sẵn Lịch âm hay
     * không, theo chỗ trống thật của máy.
     *
     * Trần của Tiết khí là JQ_SHARE phần trăm ngân sách và KHÔNG đổi theo việc
     * Lịch âm mở hay đóng (xem shareSectionHeight) — nên phần còn lại là phần
     * dành riêng cho Lịch âm, đóng nó lại thì phần ấy bỏ không. Trên máy cao
     * (A51 852px, S21 FE 790px) phần bỏ không ấy trên 100px, thành một dải
     * trống thấy rõ dưới hàng tiêu đề "Tháng âm"; ngay cả trên 360×640 nó cũng
     * là 66px. Còn đủ chỗ cho lấy một hàng đọc được thì mở, dải trống ấy hết.
     *
     * "Dùng được" ĐO trên chính bảng đang có, không phải một con số chọn bừa:
     * hàng tiêu đề cộng AM_OPEN_ROWS hàng dữ liệu, lấy chiều cao hàng thật của
     * bảng Lịch âm. Lấy SEC_MIN (96px) làm ngưỡng thì trên S21 FE — nơi phần
     * của Lịch âm là 88px, thừa sức chứa ba hàng — nó bị đóng lại một cách vô
     * lý và 71px đáy màn hình bỏ trống.
     *
     * Hai ngưỡng KHÁC NHAU, vì chừa rộng tay mà mở thì dễ tính:
     *
     *   AM_KEEP_ROWS — chỗ shareSectionHeight chừa lại cho Lịch âm khi chia
     *     với Tiết khí. HAI hàng, không phải ba: hàng bảng tiếng Trung cao hơn
     *     tiếng Việt vài pixel, nên ngưỡng ba hàng khiến CÙNG MỘT MÁY xử khác
     *     nhau ở hai ngôn ngữ (S21 FE: 9px thừa so với 85px).
     *
     *   AM_OPEN_ROWS — ngưỡng mở sẵn. MỘT hàng, vì mở hay đóng KHÔNG đổi chỗ
     *     Tiết khí được cấp (xem shareSectionHeight): đóng lại thì phần của
     *     Lịch âm bỏ không, thành dải trống ở đáy. Trên 360×640 phần ấy chỉ đủ
     *     một hàng rưỡi; một hàng đọc được cộng thanh cuộn vẫn hơn hẳn 66px
     *     trống trơn, nên chừng nào còn đủ MỘT hàng thì cứ mở.
     *
     * Chốt xong thì GHI vào kho tuỳ chọn chứ không giữ riêng trong bộ nhớ:
     * widget gập/mở theo đúng khoá này, mà widget thì phải khớp với tab Lịch.
     */
    var AM_KEEP_ROWS = 2;
    var AM_OPEN_ROWS = 1;
    /**
     * Trạng thái gập/mở của Lịch âm còn đang do máy tự quyết (chưa ai bấm).
     *
     * Quyết đi quyết lại chứ KHÔNG chốt một lần: chỗ trống còn đổi sau lượt
     * dựng đầu tiên — nút "Ghim lịch" chỉ hiện khi có cầu nối Android và hiện
     * muộn hơn lượt fitGrid đầu, lấy mất 32px; tháng 6 hàng thì lưới cao thêm
     * một hàng. Chốt một lần trên con số cũ là mở sẵn Lịch âm rồi lượt sau cấp
     * thật lại không đủ. Hai đầu vào của quyết định (chỗ trống, chiều cao hàng)
     * đều không phụ thuộc vào chính trạng thái ấy, nên quyết lại không sinh ra
     * vòng lật qua lật lại.
     */
    var amAuto = false;
    /**
     * Bảng Lịch âm đã dựng xong ÍT NHẤT MỘT LẦN chưa.
     *
     * fitGrid chạy hai lượt mỗi lần vẽ, và lượt ĐẦU đi trước renderAmBan() —
     * lúc ấy chiều cao hàng tiêu đề, chiều cao hàng và do đó cả phần chia cho
     * Lịch âm đều là số của lần vẽ trước. Chốt mặc định trên những con số ấy
     * thì trên máy thấp ra "đủ chỗ" rồi lượt sau cấp thật lại không đủ lấy một
     * hàng: mục mở ra mà chỉ thấy đúng hàng tiêu đề.
     */
    var amRendered = false;
    /**
     * Số đo MỘT hàng dữ liệu, đo trên chính bảng đang có (hàng tiếng Trung cao
     * hơn tiếng Việt vài pixel, nên không chốt cứng con số nào):
     *   h      — chiều cao cả hàng;
     *   trên   — mép TRÊN của hộp dòng chữ, tính từ đỉnh hàng;
     *   dưới   — mép DƯỚI của hộp dòng chữ. Hộp dòng đã gồm phần rơi xuống
     *            dưới đường cơ sở, tức cả dấu nặng của "Lộ", "Mậu".
     * Trả null khi bảng chưa dựng — người gọi hiểu là "chưa biết".
     */
    var inkCache = {};
    function rowMetrics(box) {
        var row = box && box.querySelector('tbody tr');
        if (!row || !row.cells.length) return null;
        var cell = row.cells[0];
        // Hộp DÒNG CHỮ, không phải hộp ô: ô còn có đệm trên/dưới. Range là
        // cách duy nhất hỏi được hộp dòng. jsdom (bộ kiểm thử test_app.mjs)
        // có Range nhưng KHÔNG có getBoundingClientRect trên nó — thiếu chỗ
        // canh này thì cả tab Lịch ném lỗi ngay lúc dựng.
        var t = null;
        try {
            var rg = document.createRange();
            rg.selectNodeContents(cell);
            if (typeof rg.getBoundingClientRect === 'function') t = rg.getBoundingClientRect();
        } catch (e) {}
        var r = row.getBoundingClientRect();
        if (!t || !t.height || !r.height) return null;
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var lineH = t.height / zoom;
        var trên = (t.top - r.top) / zoom;

        // ĐÁY MỰC, không phải đáy hộp dòng. Hộp dòng cao hơn chỗ chữ thật sự
        // chạm tới: trên S21 FE hộp dòng kết thúc ở 19px mà nét thấp nhất của
        // "Lộ"/"Mậu" chỉ tới 18px. Lấy nhầm đáy hộp thì vùng "thấy trọn chữ"
        // bị khai rộng ra 1px, và đúng 1px ấy là chỗ mép cắt hay rơi vào.
        // Lấy qua canvas với ĐÚNG phông đang dùng; nhớ theo chuỗi phông nên
        // mỗi lần đổi cỡ chữ/ngôn ngữ mới đo lại một lần.
        var cs = getComputedStyle(cell);
        var font = cs.font || (cs.fontStyle + ' ' + cs.fontWeight + ' ' +
                               cs.fontSize + '/' + cs.lineHeight + ' ' + cs.fontFamily);
        var ink = inkCache[font];
        if (ink === undefined) {
            ink = null;
            try {
                var cx = document.createElement('canvas').getContext('2d');
                cx.font = font;
                // Chuỗi mẫu gom đủ dấu rơi xuống dưới đường cơ sở của cả hai
                // thứ tiếng: dấu nặng, chữ có nét thòng, và một chữ Hán.
                var mt = cx.measureText('Lộ Mậu gy 露');
                if (mt.fontBoundingBoxAscent) {
                    ink = { lên: mt.fontBoundingBoxAscent + mt.fontBoundingBoxDescent,
                            cơsở: mt.fontBoundingBoxAscent,
                            xuống: mt.actualBoundingBoxDescent || 0 };
                }
            } catch (e) {}
            inkCache[font] = ink;
        }
        var dưới = (t.bottom - r.top) / zoom;
        if (ink) dưới = trên + (lineH - ink.lên) / 2 + ink.cơsở + ink.xuống;
        return { h: r.height / zoom, trên: trên, dưới: dưới };
    }
    function amRowHeight() {
        var m = rowMetrics(document.getElementById('calAmBan'));
        return m ? m.h : 0;
    }

    /**
     * Hàng cuối của một mục đang cuộn thì bị mép dưới cắt ngang — cắt ở ĐÂU
     * mới là chuyện.
     *
     * Đo trên S21 (360×740, tiếng Việt): hàng cuối hiện 90,9% chiều cao, đủ
     * trọn chữ nên đọc đúng. Nhưng trên S21 FE hàng cuối hiện 73,9%: vừa đúng
     * dưới đường cơ sở — thân chữ còn nguyên mà DẤU NẶNG thì mất, nên "Hàn Lộ"
     * đọc ra "Hàn Lô" và "Mậu Tuất" ra "Mâu Tuất". Đó không phải một hàng cụt,
     * đó là một chữ KHÁC. Tiếng Việt dày dấu dưới nên chỗ này là lỗi đọc sai,
     * không phải lỗi thẩm mỹ.
     *
     * Nên mép cắt chỉ được rơi vào một trong hai vùng an toàn:
     *   • từ `dưới` trở xuống — thấy trọn chữ, kể cả dấu;
     *   • từ `an` trở lên — cắt phạm vào THÂN chữ, nhìn là biết ngay hàng còn
     *     nữa, không ai đọc nhầm.
     * Rơi vào khoảng giữa thì HẠ chiều cao mục xuống đúng `an`. Trả giá tối đa
     * 28% chiều cao dòng chữ (đo được: 4-5px), và phần bị cắt của Tiết khí
     * chảy thẳng sang Lịch âm chứ không mất đi đâu.
     *
     * 0,72 là ước lượng đường cơ sở trong hộp dòng: mọi phông trong bộ này đều
     * có phần dưới đường cơ sở chừng một phần tư hộp dòng, nên cắt ở 72% là
     * chắc chắn phạm vào thân chữ.
     */
    var CUT_SAFE = 0.72;
    /**
     * Gọi SAU khi đã đặt max-height: ĐO xem mép cắt rơi vào đâu trong hàng
     * cuối, rồi hạ thêm vài pixel nếu nó rơi đúng vào chỗ dấu.
     *
     * Đo chứ không tính: bản tính tay phải tự dựng lại "chỗ các hàng bắt đầu"
     * từ max-height trừ viền trừ hàng tiêu đề, mà chuỗi ấy còn dính box-sizing,
     * `thead` sticky và phép làm tròn nửa pixel — chạy ra vẫn lệch 0,5-1px, tức
     * vẫn rơi vào đúng dải nguy hiểm ở ba cấu hình. Sau khi đã đặt chiều cao
     * thì mọi thứ đã nằm trên trang, hỏi thẳng là xong. Hạ chiều cao KHÔNG làm
     * các hàng nhúc nhích (chúng nằm trong phần cuộn), nên một lượt là đủ.
     *
     * @returns {number} chiều cao cuối cùng của khung, px chưa nhân zoom.
     */
    function snapCut(el) {
        if (!el) return 0;
        var m = rowMetrics(el);
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var box = el.getBoundingClientRect();
        var cao = box.height / zoom;
        if (!m || !m.h) return cao;
        var cs = getComputedStyle(el);
        var cắt = (box.bottom - (parseFloat(cs.borderBottomWidth) || 0)) / zoom;
        var rows = el.querySelectorAll('tbody tr');
        for (var i = 0; i < rows.length; i++) {
            var q = rows[i].getBoundingClientRect();
            if (q.top / zoom >= cắt - 0.5 || q.bottom / zoom <= cắt + 0.5) continue;
            var hiện = cắt - q.top / zoom;         // hàng cuối hiện được bấy nhiêu
            if (hiện >= m.dưới) return cao;        // thấy trọn chữ, kể cả dấu
            var an = m.trên + (m.dưới - m.trên) * CUT_SAFE;
            if (hiện <= an) return cao;            // đã cắt sâu vào thân chữ
            cao = Math.max(0, cao - (hiện - an) - 0.5);
            el.style.maxHeight = cao + 'px';
            return cao;
        }
        return cao;
    }
    function decideAmDefault(amRoom, amHeadH, measured) {
        if (!amAuto || !measured) return;
        var rowH = amRowHeight();
        if (!rowH) return;                      // bảng chưa dựng: quyết lần sau
        var want = amRoom >= amHeadH + AM_OPEN_ROWS * rowH;
        if (want === openAm) return;
        openAm = want;
        prefSet(K_SEC_AM, want ? '1' : '0');
        applySections();
        pokeWidget();
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
        // Luật chung: mỗi tiêu đề căn ĐÚNG KIỂU của giá trị bên dưới nó, không
        // thì hai bên không trùng tâm. "Tiết khí" căn trái theo tên tiết khí;
        // "Dương lịch" mang "c" vì giá trị của nó (.cal-jq-date) căn giữa —
        // lớp `.cal-jq-date` chỉ gắn cho <td>, nên bỏ "c" ở đây là tiêu đề một
        // đằng giá trị một nẻo, lệch ~60px trên máy 393px.
        box.innerHTML =
            '<table class="dp-table cal-jq">' + COLGROUP +
            '<thead><tr class="cal-sec-head">' +
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
                // Sóc/Vọng CĂN GIỮA (lớp "c"), khớp tiêu đề đã căn giữa — khác cột
                // "Dương lịch" của Tiết khí, vốn căn trái nên vẫn để nguyên.
                rows += '<tr' + (i % 2 === 0 ? ' class="dp-row-alt"' : '') + '>' +
                    '<td class="cal-jq-name' + (on ? ' cal-jq-on' : '') + '"' +
                    (on ? ' id="calAmActive"' : '') + '>' + esc(label) + '</td>' +
                    '<td class="dp-num cal-jq-date c' + (on ? ' cal-jq-on' : '') + '">' +
                    esc(formatPreciseSocLocal(socSolar, info.tzId)) + '</td>' +
                    '<td class="dp-num cal-jq-date c cal-jq-last' + (on ? ' cal-jq-on' : '') + '">' +
                    esc(formatPreciseVongLocal(socSolar, info.tzId)) + '</td>' +
                    '</tr>';
            }
            box.innerHTML =
                '<table class="dp-table cal-jq">' + COLGROUP +
                '<thead><tr class="cal-sec-head">' +
                '<th>' + t('colMonth') + '</th>' +
                '<th class="c">' + t('colSoc') + '</th>' +
                '<th class="c cal-jq-last">' + t('colVong') +
                '<span class="cal-sec-chev"></span></th>' +
                '</tr></thead><tbody>' + rows + '</tbody></table>';
            amRendered = true;
        } catch (e) {
            console.warn('calAmBan:', e);
            box.innerHTML = '';
        }
    }

    /** Bảng 24 dòng phải cuộn; đưa tiết khí đang hiệu lực vào giữa khung nhìn. */
    /* ─────────────── Hai mục gập được ─────────────── */

    /**
     * Bắt HAI bảng dùng CHUNG một bộ bề rộng cột, để "Dương lịch" thẳng hàng
     * với "Sóc" và "Can chi" thẳng hàng với "Vọng".
     *
     * Cột tên và cột cuối vốn co đúng bằng chữ (mẹo `width:1%`), nên bề rộng tự
     * nhiên của chúng CHÍNH LÀ nhu cầu thật — lấy cột rộng nhất của từng vị trí
     * giữa hai bảng rồi ghim vào <col>. Cột giữa thì không: bề rộng của nó là
     * phần CÒN LẠI, không phải nhu cầu, nên cứ để trống cho nó tự ăn chỗ thừa —
     * hai bảng rộng bằng nhau nên phần còn lại cũng bằng nhau.
     *
     * Đo ở chế độ `auto` rồi mới ghim: đọc bề rộng khi <col> còn giữ trần của
     * lần trước thì đo lại chính con số mình vừa đặt, và cột không bao giờ co
     * lại được khi đổi ngôn ngữ hay đổi tháng.
     */
    function syncSectionColumns() {
        var tables = [];
        var ids = ['calJieQi', 'calAmBan'];
        for (var i = 0; i < ids.length; i++) {
            var box = document.getElementById(ids[i]);
            var tb = box && box.querySelector('table');
            if (tb && tb.tHead && tb.tHead.rows.length &&
                tb.tHead.rows[0].cells.length >= 3 &&
                tb.querySelectorAll('col').length >= 3) tables.push(tb);
        }
        if (tables.length < 2) return;

        var t, k, cols;
        // 1. Trả cả hai bảng về `auto` để đo được bề rộng TỰ NHIÊN.
        for (t = 0; t < tables.length; t++) {
            tables[t].style.tableLayout = 'auto';
            cols = tables[t].querySelectorAll('col');
            for (k = 0; k < 3; k++) cols[k].style.width = '';
        }
        // 2. Cột tên và cột cuối co đúng bằng chữ (mẹo `width:1%`) nên bề rộng
        //    tự nhiên của chúng CHÍNH LÀ nhu cầu thật; lấy cột rộng nhất của
        //    từng vị trí giữa hai bảng.
        //
        //    CHIA CHO `zoom`: getBoundingClientRect trả về px ĐÃ PHÓNG, còn
        //    style.width nhận px CHƯA PHÓNG (viewport.js phóng cả trang bằng
        //    `zoom` trên <body>). Đo một đằng ghi một nẻo thì mọi bề rộng cột
        //    bị nhân thêm đúng hệ số phóng. Máy điện thoại đang ngắm có zoom
        //    = 1 nên không thấy gì, nhưng trên máy rộng (tablet 768px: zoom
        //    1,28) ba cột cộng lại thành 512px nhét vào khung 398px — bảng
        //    phình ra 656px, cột cuối (Can chi / Vọng) bị đẩy hẳn ra ngoài và
        //    phải kéo ngang 114px mới đọc được.
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var w0 = 0, w2 = 0, tableW = 0;
        for (t = 0; t < tables.length; t++) {
            var cells = tables[t].tHead.rows[0].cells;
            w0 = Math.max(w0, cells[0].getBoundingClientRect().width / zoom);
            w2 = Math.max(w2, cells[2].getBoundingClientRect().width / zoom);
            tableW = Math.max(tableW, tables[t].getBoundingClientRect().width / zoom);
        }
        w0 = Math.ceil(w0); w2 = Math.ceil(w2);
        // Cột giữa phải còn chỗ cho một mốc ngày giờ. Chật quá thì bóp hai cột
        // bên theo tỉ lệ chứ không để cột giữa âm — thà ba cột cùng hẹp còn hơn
        // bảng vỡ. Trên máy hẹp nhất đo được vẫn còn dư, nên đây là lưới an
        // toàn, không phải đường chạy thường ngày.
        var minMid = Math.min(tableW * 0.3, 120);
        if (w0 + w2 > tableW - minMid) {
            var scale = (tableW - minMid) / (w0 + w2);
            w0 = Math.floor(w0 * scale); w2 = Math.floor(w2 * scale);
        }
        // 3. Ghim cả ba cột rồi khoá `fixed`: ở chế độ `auto`, `width:1%` trên
        //    chính các ô vẫn tranh phần với <col> và bảng lại co theo dữ liệu
        //    của riêng nó; `fixed` thì <col> nói sao nghe vậy.
        for (t = 0; t < tables.length; t++) {
            cols = tables[t].querySelectorAll('col');
            cols[0].style.width = w0 + 'px';
            cols[1].style.width = Math.max(0, tableW - w0 - w2) + 'px';
            cols[2].style.width = w2 + 'px';
            tables[t].style.tableLayout = 'fixed';
        }
    }

    /**
     * Chiều cao kẹp một mục đã đóng: đúng hàng <thead>, cộng hai đường viền
     * của khung (box-sizing: border-box nên max-height tính cả viền).
     */
    function headOnlyHeight(body) {
        var thead = body.querySelector('thead');
        if (!thead) return 0;
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        return Math.ceil(thead.getBoundingClientRect().height / zoom) + 2;
    }

    /**
     * Mục đang mở mà chỗ được cấp không đủ MỘT hàng trọn vẹn thì hạ hẳn về
     * đúng hàng tiêu đề.
     *
     * Xoay ngang máy là thấy vì sao: màn A51 nằm ngang chỉ cao 412px, lưới
     * lịch ăn gần hết, mục Tiết khí được 28,5px mà riêng hàng tiêu đề đã 26,6px
     * — 1,9px còn lại hiện lên thành một DẢI NỬA CHỮ cắt ngang thân chữ. Trên
     * 320×568 thì 13px, đủ để thấy nửa trên của "Bạch Lộ 07-09-2026 14:41" mà
     * đọc không ra. Nửa hàng như thế tệ hơn không có hàng nào: nó chiếm chỗ,
     * trông như lỗi hiển thị, và vẫn không đọc được.
     *
     * Hạ về hàng tiêu đề thì mục trông đúng như đang đóng — sạch, vẫn chạm
     * được, và vài pixel nhả ra chảy sang mục kia.
     *
     * Việc này KHÔNG làm được trong snapCut: snapCut canh mép dưới theo chỗ
     * đang cuộn, mà cuộn thì đổi sau đó (scrollJqToActive chạy sau khi chia
     * chiều cao). Đây là quyết định về CỠ, phải chốt lúc cấp chiều cao.
     */
    function noPartialRow(body, used, headH) {
        var m = rowMetrics(body);
        if (!m || !m.h) return used;           // chưa dựng bảng: để nguyên
        return used >= headH + m.h ? used : headH;
    }

    /** Áp trạng thái mở/đóng lên DOM (không chia lại chiều cao). */
    function applySections() {
        var pairs = [['calSecJq', openJq], ['calSecAm', openAm]];
        for (var i = 0; i < pairs.length; i++) {
            var sec = document.getElementById(pairs[i][0]);
            if (!sec) continue;
            sec.classList.toggle('cal-sec-open', pairs[i][1]);
            var chev = sec.querySelector('.cal-sec-chev');
            if (chev) chev.textContent = pairs[i][1] ? '▾' : '▸';
            // Mục đã đóng: kẹp khung xuống đúng hàng tiêu đề. Bảng vẫn còn ĐỦ
            // tbody (xem calendar.css) để bề rộng ba cột không đổi khi gập/mở;
            // phần thừa bị cắt khuất. Trả khung về đầu trước khi kẹp, không thì
            // vị trí cuộn cũ còn đó và lúc mở lại bảng không ở đầu bảng.
            var body = sec.querySelector('.cal-sec-body');
            if (body && !pairs[i][1]) {
                body.scrollTop = 0;
                var capped = headOnlyHeight(body);
                body.style.maxHeight = capped ? capped + 'px' : '';
            }
        }
    }

    function toggleSection(which) {
        if (which === 'jq') { openJq = !openJq; prefSet(K_SEC_JQ, openJq ? '1' : '0'); }
        // Người dùng đã tự bấm thì thôi tự quyết — kể cả khi họ bấm đúng cái
        // trạng thái máy vừa chọn hộ.
        else                { openAm = !openAm; amAuto = false; prefSet(K_SEC_AM, openAm ? '1' : '0'); }
        // Widget gập/mở hai mục THEO ĐÚNG hai khoá này, nên ghi khoá xong phải
        // bảo nó vẽ lại — không thì lịch đã ghim còn hiện trạng thái cũ cho tới
        // nửa đêm. Chỉ gọi vẽ lại, KHÔNG công bố bảng tháng: bảng ấy không đổi
        // vì một cú bấm gập/mở, và việc công bố là của lúc mở ứng dụng.
        pokeWidget();
        applySections();
        fitGrid(lastWeeks);
        if (which === 'jq' && openJq) setTimeout(scrollToActiveJieQi, 40);
    }

    /**
     * Trần chiều cao của Tiết khí LUÔN CỐ ĐỊNH — `JQ_SHARE` phần trăm của
     * `avail`, không đổi dù Lịch âm có đang mở hay không. Lịch âm thì lấy hết
     * PHẦN CÒN LẠI sau khi trừ đúng phần Tiết khí đang dùng thật.
     *
     * Bất đối xứng CÓ CHỦ ĐÍCH — không phải chia đều "cho công bằng":
     *
     * Tiết khí đứng TRƯỚC Lịch âm trong trang, nên chiều cao THẬT của nó ảnh
     * hưởng tới vị trí tiêu đề của Lịch âm; còn Lịch âm đứng SAU CÙNG, chiều
     * cao của nó không ảnh hưởng tới bất cứ tiêu đề nào khác. Nếu chia theo tỉ
     * lệ `nat/sum` giữa các mục ĐANG MỞ (như bản trước), phần của Tiết khí sẽ
     * phụ thuộc vào việc Lịch âm CÓ đang mở hay không — bấm mở Lịch âm là Tiết
     * khí bị bớt lại NGAY LẬP TỨC dù bản thân nó không đổi trạng thái, kéo
     * tiêu đề Lịch âm nhảy đúng lúc người dùng vừa chạm vào nó. Khoá cứng phần
     * của Tiết khí thì triệt tiêu hẳn đường lây đó; nhường phần dư dôi ra cho
     * Lịch âm (mục cuối, không ai đứng sau nó) thì được ngay cái lợi cũ —
     * mở một mình thì Lịch âm vẫn chiếm trọn chỗ trống, không phải chừa vô cớ.
     */
    var JQ_SHARE = 0.55;
    function shareSectionHeight(avail) {
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var natOf = function (id) {
            var tb = document.querySelector('#' + id + ' table');
            return tb ? Math.ceil(tb.getBoundingClientRect().height / zoom) + 2 : SEC_MIN;
        };
        var jqEl = document.getElementById('calJieQi');
        var amEl = document.getElementById('calAmBan');
        // Chỗ TỐI THIỂU mỗi mục chiếm: đúng hàng tiêu đề của nó. Đo thẳng
        // <thead> nên con số này KHÔNG đổi theo việc mục đang mở hay đóng —
        // điều kiện sống còn của cả mục này (xem khối ghi chú bên trên).
        var jqMin = jqEl ? headOnlyHeight(jqEl) : 0;
        var amMin = amEl ? headOnlyHeight(amEl) : 0;

        // Tiết khí không bao giờ được lấn sang chỗ hàng tiêu đề của Lịch âm —
        // mục đóng vẫn choán đúng chừng ấy. Bỏ qua thì đóng Tiết khí mà mở
        // Lịch âm là cụm hai mục thò 17px xuống dưới thanh tab (đo trên
        // S21 FE), còn trên máy thấp thì tràn ngay cả ở trạng thái mặc định.
        // Trừ `amMin` LUÔN LUÔN, chứ không phải "khi Lịch âm đang đóng": trừ
        // có điều kiện là để trạng thái của Lịch âm quyết chiều cao Tiết khí,
        // đúng cái vòng lây mà cả hàm này sinh ra để cắt đứt.
        var jqRoom = Math.max(jqMin, avail - amMin);

        // Chỗ Lịch âm phải được CHỪA LẠI, dù nó đang mở hay đóng — lại là để
        // giữ đúng cái bất biến trên: trạng thái của Lịch âm không được quyết
        // chiều cao Tiết khí. Chừa hàng tiêu đề cộng AM_KEEP_ROWS hàng — rộng
        // hơn ngưỡng decideAmDefault lấy làm "dùng được", nên một khi Lịch âm
        // mở — tự quyết hay người dùng bấm — nó chắc chắn có hàng thật.
        // Thiếu chỗ này thì sàn SEC_MIN của Tiết khí vét sạch phần còn lại: đo
        // trên 360×640 (có nút Ghim lịch) Lịch âm mở ra mà cao đúng 28px, chỉ
        // thấy mỗi hàng tiêu đề — y như một cái nút bấm không ăn.
        //
        // Nhưng không bao giờ chừa quá NỬA phần thân chung: máy thấp tới mức
        // cả hai mục đều chật thì hai mục cùng ngắn, chứ không phải Tiết khí
        // co về đúng hàng tiêu đề để Lịch âm đủ hai hàng.
        var pool = Math.max(0, avail - jqMin - amMin);
        var amNeed = amMin + AM_KEEP_ROWS * amRowHeight();
        var amKeep = Math.min(amNeed, amMin + Math.floor(pool / 2));

        var jqUsed;
        if (openJq) {
            // Sàn SEC_MIN kẹp theo chỗ CÓ THẬT: máy thấp thì thà mục ngắn còn
            // hơn đẩy cả cụm xuống dưới thanh tab.
            var jqCap = Math.max(Math.min(SEC_MIN, jqRoom), Math.floor(jqRoom * JQ_SHARE));
            jqCap = Math.min(jqCap, Math.max(jqMin, Math.floor(avail - amKeep)));
            jqUsed = Math.min(natOf('calJieQi'), jqCap);
            if (jqEl) {
                jqUsed = noPartialRow(jqEl, jqUsed, jqMin);
                jqEl.style.maxHeight = jqUsed + 'px';
                // Hạ TRƯỚC khi chia phần cho Lịch âm: vài pixel Tiết khí nhả
                // ra phải chảy sang Lịch âm, chứ không thành khe trống.
                jqUsed = snapCut(jqEl);
            }
        } else {
            jqUsed = jqMin;
        }
        // Chốt mặc định của Lịch âm Ở ĐÂY, nơi biết CHỖ THẬT nó sẽ được cấp.
        // Trước đây chốt trong fitGrid bằng `budget − budget × JQ_SHARE`, mà đó
        // chỉ là phần theo tỉ lệ: trên máy thấp, sàn SEC_MIN kê phần của Tiết
        // khí lên cao hơn tỉ lệ ấy, nên Lịch âm được cấp ÍT hơn con số đem ra
        // quyết định — đo trên 360×640: mở sẵn rồi mà không đủ chỗ cho một
        // hàng nào.
        var amRoom = Math.max(0, avail - jqUsed);
        decideAmDefault(amRoom, amMin, amRendered);

        if (openAm) {
            var amCap = Math.max(amMin, amRoom);
            amCap = Math.min(natOf('calAmBan'), amCap);
            if (amEl) {
                amCap = noPartialRow(amEl, amCap, amMin);
                amEl.style.maxHeight = amCap + 'px';
                snapCut(amEl);
            }
        }
    }

    /**
     * Cuộn mục Tiết khí để hàng đang hiệu lực nằm giữa khung nhìn.
     *
     * `#calJieQi` CHÍNH LÀ khung cuộn (.cal-sec-body) từ khi tiêu đề chuyển
     * thành hàng <thead> — trước đó còn một lớp bọc `.cal-jq-body` riêng, nay
     * đã bỏ. Hàm này vẫn trỏ vào lớp bọc cũ nên querySelector luôn ra null và
     * lặng lẽ không làm gì; mở mục Tiết khí không còn tự cuộn tới hôm nay.
     */
    function scrollToActiveJieQi() {
        // Mục đang đóng thì khung chỉ còn cao đúng hàng tiêu đề mà bảng bên
        // trong vẫn đủ 24 hàng — cuộn lúc này là đẩy bảng lệch sẵn, để rồi mở
        // ra thấy giữa bảng. Chỉ cuộn khi mục đang mở.
        if (!openJq) return;
        var row = document.getElementById('calJqActive');
        var body = document.getElementById('calJieQi');
        if (!row || !body) return;
        // Hàng tiêu đề DÍNH nên nó che mất phần trên của khung cuộn: chỗ thấy
        // được chỉ là phần dưới nó.
        var thead = body.querySelector('thead');
        var headH = thead ? thead.getBoundingClientRect().height : 0;
        // Toạ độ trong KHÔNG GIAN NỘI DUNG của khung cuộn. Dùng rect chứ không
        // dùng offsetTop: offsetTop đo theo offsetParent, mà offsetParent của
        // một <tr> không chắc là khung cuộn này.
        var base = body.getBoundingClientRect().top - body.scrollTop;
        var yOf = function (el) { return el.getBoundingClientRect().top - base; };
        var rowH = row.getBoundingClientRect().height;
        var want = yOf(row) - headH - (body.clientHeight - headH - rowH) / 2;
        // NẮN về đúng mép một hàng. Thả tự do thì hàng trên cùng bị cắt ngang
        // ngay dưới hàng tiêu đề — nhìn ra một vệt chữ cụt, giống lỗi vẽ chứ
        // không giống "còn cuộn được nữa". Tiếng Trung hàng cao 22px, tiếng
        // Việt 20px, nên chỗ dừng tự do rơi giữa hàng ở thứ tiếng này mà lại
        // đúng mép ở thứ tiếng kia — cùng một máy, hai kiểu.
        var rows = body.querySelectorAll('tbody tr');
        var best = want, bestD = Infinity;
        for (var i = 0; i < rows.length; i++) {
            var top = yOf(rows[i]) - headH;      // scrollTop để hàng này nằm sát tiêu đề
            var d = Math.abs(top - want);
            if (d < bestD) { bestD = d; best = top; }
        }
        body.scrollTop = Math.max(0, Math.round(best));
    }


    function shiftMonth(delta) {
        var d = new Date(viewY, viewM - 1 + delta, 1);
        viewY = d.getFullYear();
        viewM = d.getMonth() + 1;
        render();
    }

    /* ─────────────── Chuyển tab ─────────────── */

    /**
     * Ba tab dùng CHUNG một danh sách, không rẽ nhánh if/else theo từng cái.
     * Thêm tab thứ ba bằng cách sửa hai chỗ (danh sách này và index.html) chứ
     * không phải rà lại mọi phép so `=== 'cal'` rải khắp tệp.
     *
     * Tên lớp của <body> là 'view-' + khoá, trừ tab Kỳ Môn: nó là tab MẶC
     * ĐỊNH, tức trạng thái "không lớp nào" — mọi luật ẩn hiện của hai tab kia
     * đều viết dưới dạng "ẩn tất cả trừ…", nên Kỳ Môn không cần lớp riêng.
     */
    var TABS = [
        { key: 'qmdj',   tab: 'tabQmdj',  cls: null },
        { key: 'cal',    tab: 'tabCal',   cls: 'view-cal' },
        { key: 'lenh',   tab: 'tabLenh',  cls: 'view-lenh' },
        { key: 'tracuu', tab: 'tabTraCuu', cls: 'view-tracuu' },
    ];

    function showTab(which) {
        for (var i = 0; i < TABS.length; i++) {
            var it = TABS[i], on = it.key === which;
            if (it.cls) document.body.classList.toggle(it.cls, on);
            var el = document.getElementById(it.tab);
            if (el) el.classList.toggle('tab-active', on);
        }
        if (which === 'cal') render();
        if (which === 'lenh' && typeof window.__lenhRender === 'function') {
            try { window.__lenhRender(); } catch (e) {}
        }
        if (which === 'tracuu' && typeof window.__tracuuRender === 'function') {
            try { window.__tracuuRender(); } catch (e) {}
        }
        if (typeof window.__fitScreen === 'function') setTimeout(window.__fitScreen, 50);
        // Chỉ cuộn khi đang không ở đầu trang — gọi thừa vừa vô ích vừa làm
        // jsdom kêu "not implemented" trong bộ kiểm thử.
        if (window.scrollY) { try { window.scrollTo(0, 0); } catch (e) {} }
    }
    window.showTab = showTab;

    /**
     * Chia lại chiều cao tab Lịch NGAY, ở đúng tỉ lệ phóng hiện thời.
     *
     * viewport.js gọi hàm này trước mỗi lần nó đo chiều cao nội dung. Không có
     * nó thì hai cơ chế cùng kéo một sợi dây: fitGrid nới nội dung cho vừa
     * MỌI tỉ lệ, còn viewport.js thấy nội dung vừa khít thì giữ nguyên tỉ lệ
     * đang có — nên mọi tỉ lệ trong [0,95; 1] đều "đúng" và app dừng ở đâu là
     * tuỳ thứ tự chạy. Cùng một máy, cùng một tháng, hai lần mở ra hai cỡ chữ.
     */
    /**
     * Đọc lại trạng thái gập/mở từ kho tuỳ chọn và áp lên trang.
     *
     * Widget cũng gập/mở được hai mục và ghi thẳng vào hai khoá này. Không có
     * hàm này thì quay lại ứng dụng sau khi bấm ở widget, tab Lịch vẫn hiện
     * trạng thái cũ — mà cú bấm tiếp theo trong ứng dụng lại lật từ trạng thái
     * sai ấy, thành ra hai bên cãi nhau. MainActivity.onResume() gọi hàm này.
     */
    window.__calSyncSections = function () {
        var sj = prefGet(K_SEC_JQ), sa = prefGet(K_SEC_AM);
        var nj = (sj === '0' || sj === '1') ? sj === '1' : openJq;
        var na = (sa === '0' || sa === '1') ? sa === '1' : openAm;
        if (nj === openJq && na === openAm) return;
        openJq = nj; openAm = na;
        // Đã có lựa chọn rõ ràng rồi thì đừng để fitGrid tự quyết lại nữa.
        amAuto = false;
        applySections();
        if (document.body.classList.contains('view-cal')) {
            fitGrid(lastWeeks);
            if (openJq) setTimeout(scrollToActiveJieQi, 40);
        }
    };

    /**
     * Cho tab Lệnh dùng chung phép canh mép cắt (xem snapCut). Bảng của nó
     * cũng đầy chữ có dấu dưới — "Mậu", "Tuất", "Bạch Lộ" — nên cũng gặp đúng
     * cái bẫy "cắt mất dấu thành chữ khác". Chép lại phép canh ấy sang tệp kia
     * là mời một bản thứ hai trôi khỏi bản này.
     */
    window.__snapCutRows = snapCut;

    window.__calFit = function () {
        if (!document.body.classList.contains('view-cal')) return;
        try { fitGrid(lastWeeks); } catch (e) {}
    };

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
        // Nhãn tab Lệnh và cả bảng của nó do lenh.js lo — nhưng nút đổi ngôn
        // ngữ nằm ở hàng dùng chung, bấm được từ BẤT CỨ tab nào, nên phải gọi
        // sang chứ không đợi tới lúc mở tab ấy.
        if (typeof window.__lenhRefreshLabels === 'function') {
            try { window.__lenhRefreshLabels(); } catch (e) {}
        }
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
    var lastCache = null;       // payload đã ghi lần trước, để khỏi ghi lại thừa

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
            // Ghi cả năm nay LẪN năm sau: tháng âm cuối năm dương tràn sang năm
            // sau, mà widget lật tháng bằng ‹ › thì đi xa hơn hôm nay nhiều.
            //
            // Hai năm thì danh sách CHỒNG NHAU (zi_months của năm sau vẫn chứa
            // mấy tháng cuối năm nay), mà widget đòi mốc mùng 1 tăng dần NGẶT —
            // trùng một mốc là nó chối cả bảng rồi lùi về bảng đóng sẵn. Gộp
            // rồi sắp và bỏ trùng theo JDN.
            var seen = Object.create(null), all = [];
            for (var y = now.getFullYear(); y <= now.getFullYear() + 1; y++) {
                var list = zi_months(y, info.lon, info.tzId, tz);
                for (var i = 0; i < list.length; i++) {
                    if (seen[list[i].jdn]) continue;
                    seen[list[i].jdn] = 1;
                    all.push(list[i]);
                }
            }
            all.sort(function (a, b) { return a.jdn - b.jdn; });
            for (var k = 0; k < all.length; k++) {
                rows.push(all[k].jdn + ',' + all[k].month + ',' + (all[k].leap ? 1 : 0));
            }
            if (!rows.length) return;

            // Khoá là MÃ MÚI GIỜ, không phải số phút lệch.
            //
            // Trước đây ghi Math.round(tz*60) — độ lệch của NGÀY ĐANG CHỌN — còn
            // widget thì so với độ lệch của ĐÚNG NGÀY nó đang tra. Ở nước có
            // DST hai con số ấy khác nhau suốt nửa năm (Paris: 120 vs 60), nên
            // bảng của ứng dụng bị chối và widget lặng lẽ lùi về bảng đóng sẵn
            // ở mốc UTC+7 — tức là hết đồng bộ, đúng lúc không ai ngờ.
            var payload = info.tzId + '|' + rows.join(';');
            if (payload === lastCache) return;
            lastCache = payload;
            prefSet(K_LUNAR_CACHE, payload);
            pokeWidget();
        } catch (e) {
            try { ShouXingUtil.setTzOffsetHours(null); } catch (e2) {}
        }
    }

    /** Bảo widget vẽ lại (không có lớp native thì im lặng bỏ qua). */
    function pokeWidget() {
        try {
            var n = window.QMDJNative;
            if (n && typeof n.refreshCalendarWidget === 'function') n.refreshCalendarWidget();
        } catch (e) {}
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
        pokeWidget();
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

        // Trạng thái gập: Tiết khí luôn mở sẵn. Lịch âm thì tuỳ máy — chưa có
        // lựa chọn cũ thì để fitGrid() chốt một lần theo chỗ trống đo được
        // (xem decideAmDefault); có rồi thì nghe người dùng.
        var sj = prefGet(K_SEC_JQ), sa = prefGet(K_SEC_AM);
        if (sj === '0' || sj === '1') openJq = sj === '1';
        if (sa === '0' || sa === '1') openAm = sa === '1';
        else amAuto = true;
        applySections();

        // Công bố ngay từ lúc MỞ ỨNG DỤNG — widget phải đúng kể cả khi người
        // dùng không chạm vào gì (không đổi ngôn ngữ, không đổi địa điểm,
        // không mở tab Lịch). processAll() ở app.js cũng tự publish sau
        // 100ms, nhưng CHỈ KHI countryData/Solar đã sẵn sàng; gọi thêm ở đây,
        // trễ hơn một khoảng an toàn, để không bỏ sót nếu thứ tự nạp lệch đi.
        setTimeout(function () {
            try { publishLunarCache(); } catch (e) {}
            try { publishLang(); } catch (e) {}
        }, 600);

        // Uỷ quyền: hàng tiêu đề nằm trong bảng, mà bảng thì dựng lại mỗi lần
        // vẽ — gắn thẳng vào nó thì cứ đổi tháng là mất người nghe.
        document.getElementById('calSections').addEventListener('click', function (e) {
            var head = e.target.closest ? e.target.closest('.cal-sec-head') : null;
            if (!head) return;
            var sec = head.closest('.cal-sec');
            if (sec) toggleSection(sec.id === 'calSecJq' ? 'jq' : 'am');
        });

        for (var ti = 0; ti < TABS.length; ti++) {
            (function (key) {
                var el = document.getElementById(TABS[ti].tab);
                if (el) el.addEventListener('click', function () { showTab(key); });
            })(TABS[ti].key);
        }
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
                // Ghi lại bảng tháng cho widget sau MỌI lần engine tính lại,
                // không riêng lúc vẽ tab Lịch.
                //
                // Trước đây publishLunarCache chỉ chạy trong render(), mà
                // render() chỉ chạy ở tab Lịch — trong khi ứng dụng mở ra ở tab
                // Kỳ Môn. Ai không bao giờ mở tab Lịch thì ứng dụng KHÔNG HỀ
                // đưa bảng tháng của mình cho widget, và widget đành dùng bảng
                // đóng sẵn ở mốc UTC+7 — ở Paris lệch với ứng dụng ~0,35% số
                // tháng và ~1% nhãn tháng. Đó chính là "lịch đã ghim không đồng
                // bộ" mà không đụng gì tới ngôn ngữ.
                try { publishLunarCache(); } catch (e2) {}
                pokeWidget();
                return r;
            };
            wrappedCalc.__calRecalcWrapped = true;
            window.processAll = wrappedCalc;
        }
    });
})();
