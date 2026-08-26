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
        jieqi:    { vi: 'Tiết khí trong tháng', zh: '本月节气' },
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
        if (typeof ShouXingUtil !== 'undefined' && ShouXingUtil.setTzOffsetHours) {
            ShouXingUtil.setTzOffsetHours(isZH() ? 8 : 7);
        }
    }
    /** Trả biến toàn cục về mặc định của thư viện cho phần còn lại của ứng dụng. */
    function clearLunarBasis() {
        if (typeof ShouXingUtil !== 'undefined' && ShouXingUtil.setTzOffsetHours) {
            ShouXingUtil.setTzOffsetHours(null);
        }
    }

    // Lề trên/dưới của trang cộng khoảng cách giữa các khối trong tab Lịch.
    var GRID_CHROME = 28;
    var ROW_MIN = 52, ROW_MAX = 104;

    var viewY, viewM;          // tháng đang xem (dương lịch)
    var selected = null;       // {y,m,d}

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

        var cells = [];
        for (var i = 0; i < lead; i++) cells.push('<div class="cal-cell cal-empty"></div>');

        for (var d = 1; d <= daysInMonth; d++) {
            var lunar = Solar.fromYmd(viewY, viewM, d).getLunar();
            var gz = dayGanZhi(lunar);
            var lday = lunar.getDay();
            var lmon = lunar.getMonth();
            // Mùng 1 thì ghi kèm tháng âm, như lịch giấy: "1/7"
            var lunarTxt = (lday === 1) ? (lday + '/' + Math.abs(lmon) + (lmon < 0 ? 'N' : '')) : String(lday);
            var key = viewY + '-' + viewM + '-' + d;
            var cls = 'cal-cell cal-day';
            if (key === tKey) cls += ' cal-today';
            if (selected && selected.y === viewY && selected.m === viewM && selected.d === d) cls += ' cal-sel';
            if ((lead + d - 1) % 7 === 6) cls += ' cal-sun';

            // Can và chi luôn nằm trên HAI dòng riêng, mọi ngày như nhau —
            // để một chuỗi tự xuống dòng thì "Đinh Mùi" gãy đôi còn "Kỷ Dậu"
            // nằm một dòng, nhìn so le rất xấu.
            cells.push(
                '<div class="' + cls + '" data-d="' + d + '">' +
                '<div class="cal-top">' +
                '<span class="cal-solar">' + d + '</span>' +
                '<span class="cal-lunar">' + esc(lunarTxt) + '</span>' +
                '</div>' +
                '<div class="cal-gz">' +
                '<span>' + esc(gz.can) + '</span><span>' + esc(gz.chi) + '</span>' +
                '</div>' +
                '</div>'
            );
        }
        while (cells.length % 7 !== 0) cells.push('<div class="cal-cell cal-empty"></div>');

        for (var r = 0; r < cells.length; r += 7) {
            html += '<div class="cal-row">' + cells.slice(r, r + 7).join('') + '</div>';
        }
        document.getElementById('calGrid').innerHTML = html;

        renderJieQi();
        clearLunarBasis();
        fitGrid(cells.length / 7);
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
        var jq = document.getElementById('calJieQi');
        var bar = document.getElementById('tabBar');
        var dow = grid ? grid.querySelector('.cal-dow') : null;
        if (!grid || !head || !dow) return;

        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var h = function (el) { return el ? el.getBoundingClientRect().height / zoom : 0; };
        var avail = window.innerHeight / zoom - h(head) - h(dow) - h(jq) - h(bar) - GRID_CHROME;
        var rowH = Math.floor(avail / weeks);
        rowH = Math.max(ROW_MIN, Math.min(ROW_MAX, rowH));
        document.documentElement.style.setProperty('--cal-row-h', rowH + 'px');
    }

    /** Bảng tiết khí rơi vào tháng dương lịch đang xem. */
    function renderJieQi() {
        var box = document.getElementById('calJieQi');
        if (!box) return;
        var rows = '';
        try {
            var table = Solar.fromYmd(viewY, viewM, 15).getLunar().getJieQiTable();
            var items = [];
            for (var name in table) {
                var s = table[name];
                if (s.getYear() === viewY && s.getMonth() === viewM) {
                    items.push({ name: name, s: s });
                }
            }
            items.sort(function (a, b) { return a.s.getDay() - b.s.getDay(); });
            rows = items.map(function (it) {
                var vi = (typeof tietKhiMap !== 'undefined' && tietKhiMap[it.name]) || it.name;
                var p = function (n) { return (n < 10 ? '0' : '') + n; };
                return '<tr><td>' + esc(isZH() ? it.name : vi) + '</td><td>' +
                    p(it.s.getHour()) + ':' + p(it.s.getMinute()) + ' - ' +
                    p(it.s.getDay()) + '/' + p(it.s.getMonth()) + '/' + it.s.getYear() +
                    '</td></tr>';
            }).join('');
        } catch (e) { rows = ''; }
        box.innerHTML = rows
            ? '<div class="cal-jq-title">' + t('jieqi') + '</div><table class="cal-jq">' + rows + '</table>'
            : '';
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

    /** Nhãn tab đổi theo ngôn ngữ. */
    function refreshLabels() {
        var tq = document.getElementById('tabQmdj');
        var tc = document.getElementById('tabCal');
        if (tq) tq.querySelector('.tab-lbl').textContent = t('tabQmdj');
        if (tc) tc.querySelector('.tab-lbl').textContent = t('tabCal');
        if (document.body.classList.contains('view-cal')) render();
    }
    window.__calRefreshLabels = refreshLabels;

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
            selected = { y: viewY, m: viewM, d: parseInt(cell.getAttribute('data-d'), 10) };
            render();
        });

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
