/* ════════════════════════════════════════════════════════════════════
   tracuu.js — tab "Tra cứu": ba bảng tra theo NĂM

   Gom về một chỗ ba bảng vốn nằm rải rác và vốn chỉ tra được theo ngày giờ
   đang nhập:

     · Trí Nhuận pháp  — trước ở tab Kỳ Môn, và chỉ hiện khi đang chọn đúng
                         phái Trí Nhuận
     · Sách Bổ pháp    — trước ở tab Kỳ Môn, và chỉ hiện khi đang chọn đúng
                         phái Sách Bổ
     · Lệnh năm        — trước ở cuối tab Bát Tự, luôn theo năm sinh trong lá số

   Cả ba nay tra theo MỘT NĂM người dùng tự chọn, độc lập với lá số đang xem.

   ─── Nguyên tắc: KHÔNG dựng lại cái gì cả ───
   Tệp này không tự sinh một dòng HTML bảng nào. Nó gọi đúng những hàm mà hai
   tab cũ vẫn gọi:

     tn_renderPanel()          (app.js)  → bảng Trí Nhuận
     sb_renderPanel()          (app.js)  → bảng Sách Bổ
     window.__lenhShared       (lenh.js) → bảng Lệnh năm + danh sách bộ số
     window.toggleDetailPanel  (app.js)  → gập/mở, chung với mọi bảng khác

   Ba bảng đã được CHUYỂN HẲN sang đây trong index.html chứ không nhân bản,
   nên mọi id (#trn-tbody, #sb-tbody, #lenhBody…) vẫn là duy nhất trên trang
   và mấy hàm trên chạy y nguyên, không phải sửa một dòng nào. Sửa cột, sửa
   cách tô hàng, sửa định dạng ngày giờ ở chỗ cũ là tab này đổi theo — đúng
   yêu cầu "đổi ở tab Bát Tự thì tab Tra cứu phải tự cập nhật".
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var K_YEAR = 'qmdj.tracuuYear';
    var K_RULE = 'qmdj.tracuuRule';
    /* Cùng khoảng mà ô năm của bảng chọn ngày giờ dùng — bảng tiết khí và
       bảng lệnh đều dựa trên DE423, ngoài khoảng này thì số liệu không còn
       đáng tin. */
    var Y_MIN = 1900, Y_MAX = 2100;

    var year = null;          // null = chưa chọn gì, chưa vẽ bảng nào
    var ruleKey = null;       // null = lấy bộ đang chọn ở tab Bát Tự

    var T = {
        pickYear:  { vi: 'Chọn năm',        zh: '选择年份' },
        yearPh:    { vi: 'Chọn năm…',       zh: '选择年份…' },
        hint:      { vi: 'Chọn một năm để xem ba bảng tra.',
                     zh: '选择一个年份以查看三张查询表。' },
        tabTraCuu: { vi: 'Tra cứu',         zh: '查询' },
    };
    function isZH() { return document.body.classList.contains('lang-zh'); }
    function t(k) { return T[k] ? (isZH() ? T[k].zh : T[k].vi) : k; }

    function store(k, v) { try { localStorage.setItem(k, v); } catch (e) {} }
    function load(k) { try { return localStorage.getItem(k); } catch (e) { return null; } }

    function shared() { return window.__lenhShared || null; }

    /** Bộ số đang dùng cho bảng Lệnh năm ở TAB NÀY. */
    function currentRule() {
        var sh = shared();
        if (!sh) return null;
        var list = sh.rules();
        for (var i = 0; i < list.length; i++) if (list[i].key === ruleKey) return list[i];
        // Chưa chọn riêng thì theo bộ của tab Bát Tự — mở tab lên thấy ngay
        // cái mình đang dùng, không phải cái mặc định của hệ thống.
        var cur = sh.ruleByKey(window.__lenhRule ? window.__lenhRule() : null);
        for (var j = 0; j < list.length; j++) if (list[j].key === cur.key) return list[j];
        return list[0];
    }

    /* ─────────────── Vẽ ─────────────── */

    var đãMởLầnĐầu = false;

    function render() {
        if (!document.body.classList.contains('view-tracuu')) return;
        var wrap = document.getElementById('traCuuView');
        if (!wrap) return;

        showYear();
        showRule();
        wrap.classList.toggle('tc-empty', year === null);
        if (year === null) return;

        renderTriNhuan();
        renderSachBo();
        renderLenh();

        // Mở tab lần đầu với năm nhớ từ phiên trước: cũng bung sẵn, đúng như
        // khi vừa chọn năm. Chỉ MỘT lần — sau đó tôn trọng trạng thái người
        // dùng tự đặt.
        if (!đãMởLầnĐầu) { đãMởLầnĐầu = true; openAll(); }
    }

    /**
     * Bảng Trí Nhuận của năm đã chọn.
     *
     * tn_renderPanel() cần một MỐC THỜI ĐIỂM để biết tô hàng nào là "đang hiệu
     * lực". Ở đây không có mốc nào cả — người dùng đang tra một năm bất kỳ,
     * không phải xem lá số — nên truyền `null` làm kết quả tính sẵn và một mốc
     * giữa năm chỉ để hàm có cái mà tính. Hệ quả: KHÔNG hàng nào được tô, đúng
     * ý: tô một hàng ở đây là nói dối rằng nó liên quan tới lá số đang mở.
     */
    function renderTriNhuan() {
        if (typeof tn_renderPanel !== 'function') return;
        var loc = locInfo();
        if (!loc) return;
        try {
            tn_renderPanel(year, loc.lon, loc.tz, loc.tzId, midYearJdFrac(), -1, -1, null);
        } catch (e) { console.warn('tracuu: Trí Nhuận', e); }
    }

    /**
     * Bảng Sách Bổ của năm đã chọn.
     *
     * sb_renderPanel() tự suy "năm mặt trời" từ ngày giờ truyền vào
     * (sb_findY), nên phải đưa một mốc GIỮA năm — 15/06 12:00 — để nó suy ra
     * đúng năm đang tra. Đưa 01/01 thì mốc ấy còn thuộc năm trước theo tiết
     * khí, và bảng hiện ra là bảng của năm liền trước.
     *
     * `jieQiZH` = null → không hàng nào được tô, cùng lý do với Trí Nhuận.
     */
    function renderSachBo() {
        if (typeof sb_renderPanel !== 'function') return;
        var loc = locInfo();
        if (!loc) return;
        try {
            sb_renderPanel(null, year, 6, 15, 12, 0, loc.tz, loc.tzId);
        } catch (e) { console.warn('tracuu: Sách Bổ', e); }
    }

    /** Bảng Lệnh năm — dựng bởi lenh.js, không phải bởi tệp này. */
    function renderLenh() {
        var sh = shared();
        var body = document.getElementById('lenhBody');
        var head = document.getElementById('lenhTitle');
        if (!sh || !body) return;
        var r = currentRule();
        try {
            body.innerHTML = sh.tableHtml(year, r ? r.key : null, sh.tzId());
            if (head) head.textContent = sh.title(year);
        } catch (e) {
            body.innerHTML = '';
            console.warn('tracuu: Lệnh năm', e);
        }
    }

    /** Toạ độ + múi giờ của địa điểm đang chọn ở hàng dùng chung dưới đáy. */
    function locInfo() {
        if (typeof getDOM !== 'function' || typeof countryData === 'undefined') return null;
        var el = getDOM('country');
        var info = el ? countryData[el.value] : null;
        if (!info) return null;
        var tz = (typeof getTimezoneOffset === 'function')
            ? getTimezoneOffset(info.tzId, new Date(year, 5, 15, 12)) : 7;
        return { lon: info.lon, tz: tz, tzId: info.tzId };
    }

    /** Ngày Julius của 15/06 12:00 năm đang tra — chỉ là một mốc trong năm. */
    function midYearJdFrac() {
        if (typeof Solar === 'undefined') return 0;
        try { return Solar.fromYmdHms(year, 6, 15, 12, 0, 0).getJulianDay(); }
        catch (e) { return 0; }
    }

    /* ─────────────── Hai ô chọn ─────────────── */

    function showYear() {
        var el = document.getElementById('tcYearText');
        if (el) el.textContent = year === null ? t('yearPh') : String(year);
    }
    function showRule() {
        var el = document.getElementById('tcRuleText');
        var r = currentRule();
        if (el) el.textContent = r ? r.label : '—';
    }

    /**
     * Mở cả ba bảng.
     *
     * Ba bảng mang theo trạng thái ĐÓNG SẴN từ chỗ cũ (.dp-body và #lenhSec
     * đều display:none), vốn hợp lý ở đó: chúng là chi tiết PHỤ của một màn
     * hình đã có nội dung chính. Ở tab này thì ngược hẳn — chúng LÀ nội dung,
     * và người dùng vừa chọn một năm chính là để xem chúng. Chọn năm xong mà
     * nhận được ba cái tiêu đề đóng im thì phải bấm thêm ba lần nữa.
     *
     * Chỉ gọi khi người dùng vừa CHỌN NĂM, không gọi ở mỗi lần vẽ lại: đổi
     * ngôn ngữ hay đổi địa điểm mà ba bảng tự bung ra là giật mất trạng thái
     * người dùng đã tự đóng.
     */
    function openAll() {
        [['trinhuanBody', 'trinhuanChevron'],
         ['sachboBody',   'sachboChevron'],
         ['lenhSec',      'lenhHeadChevron']].forEach(function (pair) {
            var body = document.getElementById(pair[0]);
            var chev = document.getElementById(pair[1]);
            if (!body) return;
            body.style.display = 'block';
            if (chev) chev.style.transform = 'rotate(180deg)';
        });
        var head = document.getElementById('lenhHead');
        if (head) head.classList.add('lenh-open');
    }

    function setYear(y) {
        y = parseInt(y, 10);
        if (!isFinite(y)) return;
        year = Math.min(Y_MAX, Math.max(Y_MIN, y));
        store(K_YEAR, String(year));
        render();
        openAll();
        if (typeof window.__fitScreen === 'function') setTimeout(window.__fitScreen, 60);
    }

    function setRule(k) {
        ruleKey = k;
        store(K_RULE, k);
        showRule();
        if (year !== null) renderLenh();
    }

    window.openTraCuuYearPicker = function () {
        if (typeof openOptionPicker !== 'function') return;
        var opts = [];
        // Năm gần đây trước: người dùng tra năm nay và vài năm quanh nó nhiều
        // hơn hẳn tra năm 1903, nên xếp GIẢM DẦN để khỏi phải vuốt qua cả thế
        // kỷ mới tới chỗ thường dùng.
        for (var y = Y_MAX; y >= Y_MIN; y--) opts.push({ value: String(y), label: String(y) });
        openOptionPicker(t('pickYear'), opts,
            String(year === null ? new Date().getFullYear() : year), setYear);
    };

    window.openTraCuuRulePicker = function () {
        var sh = shared();
        if (!sh || typeof openOptionPicker !== 'function') return;
        var list = sh.rules(), opts = [];
        // Bảng chọn dùng tên ĐẦY ĐỦ, ô ngoài dùng viết tắt — cùng quy ước với
        // ô bộ số ở tab Bát Tự (xem RULES trong lenh.js).
        for (var i = 0; i < list.length; i++) {
            opts.push({ value: list[i].key, label: list[i].labelFull });
        }
        var r = currentRule();
        openOptionPicker(sh.pickRuleTitle(), opts, r ? r.key : null, setRule);
    };

    /* ─────────────── Nhãn + móc nối ─────────────── */

    function refreshLabels() {
        var tab = document.getElementById('tabTraCuu');
        if (tab) tab.querySelector('.tab-lbl').textContent = t('tabTraCuu');
        var hint = document.getElementById('tcHint');
        if (hint) hint.textContent = t('hint');
        showYear();
        showRule();
        // Đổi ngôn ngữ thì ba bảng phải vẽ lại: tên tiết khí, tên can chi và
        // tiêu đề bảng đều theo ngôn ngữ.
        if (document.body.classList.contains('view-tracuu') && year !== null) render();
    }
    window.__tracuuRefreshLabels = refreshLabels;
    window.__tracuuRender = render;

    /* Chỉ dùng cho bộ kiểm thử. */
    window.__tracuuYear = function (y) { if (y !== undefined) setYear(y); return year; };
    window.__tracuuRule = function (k) { if (k) setRule(k); var r = currentRule(); return r && r.key; };

    document.addEventListener('DOMContentLoaded', function () {
        var sy = load(K_YEAR);
        if (sy !== null && /^\d{4}$/.test(sy)) {
            var n = parseInt(sy, 10);
            if (n >= Y_MIN && n <= Y_MAX) year = n;
        }
        var sr = load(K_RULE);
        if (sr) ruleKey = sr;

        // Dòng gợi ý khi chưa chọn năm — dựng bằng JS để nhãn theo được ngôn
        // ngữ mà không phải nhét hai bản vào index.html.
        var wrap = document.getElementById('traCuuView');
        if (wrap && !document.getElementById('tcHint')) {
            var h = document.createElement('div');
            h.id = 'tcHint';
            h.textContent = t('hint');
            var bar = document.getElementById('tcBar');
            if (bar && bar.nextSibling) wrap.insertBefore(h, bar.nextSibling);
            else wrap.appendChild(h);
        }
        refreshLabels();

        // Đổi địa điểm thì hai bảng tiết khí phải vẽ lại (mốc giờ địa phương
        // đổi theo), y như cách calendar.js và lenh.js móc vào processAll().
        if (typeof processAll === 'function' && !processAll.__tracuuWrapped) {
            var orig = processAll;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                if (document.body.classList.contains('view-tracuu')) {
                    try { render(); } catch (err) { console.warn('tracuu:', err); }
                }
                return r;
            };
            wrapped.__tracuuWrapped = true;
            window.processAll = wrapped;
        }
    });
})();
