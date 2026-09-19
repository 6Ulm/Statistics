/* ════════════════════════════════════════════════════════════════════
   nguhanh.js — Ngũ hành của can chi, tàng can, thập thần

   MỘT nguồn duy nhất cho mọi chỗ tô màu can chi trong ứng dụng. Bảng Tứ Trụ,
   bảng Đại Vận và dòng tóm tắt của tab Bát Tự đều hỏi ở đây, nên đổi một màu
   là đổi khắp nơi.

   Luật màu (người dùng chốt):
       Kim  xám bạc   Mộc  xanh lá     Thủy  xanh nước biển đậm
       Hỏa  đỏ        Thổ  nâu
   Mọi chữ KHÔNG PHẢI can chi đều màu đen — kể cả nạp âm ("Lộ Bàng Thổ") và
   thập thần ("Chính Ấn"), dù tên chúng có chứa tên một hành.

   Nhận cả tiếng Việt lẫn tiếng Trung: cùng một lá số, người dùng đổi ngôn ngữ
   lúc nào cũng được, nên tra theo tên hiển thị phải đúng ở cả hai.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var CAN_VI = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    var CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
    var CHI_VI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
    var CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

    /* Hành của 10 thiên can, theo thứ tự Giáp…Quý. */
    var CAN_HANH = ['moc', 'moc', 'hoa', 'hoa', 'tho', 'tho', 'kim', 'kim', 'thuy', 'thuy'];
    /* Hành của 12 địa chi, theo thứ tự Tý…Hợi. */
    var CHI_HANH = ['thuy', 'tho', 'moc', 'moc', 'tho', 'hoa',
                    'hoa', 'tho', 'kim', 'kim', 'tho', 'thuy'];

    /**
     * Tàng can — những thiên can ẩn trong mỗi địa chi, theo thứ tự
     * bản khí · trung khí · dư khí (chỉ số trong CAN_*).
     */
    var TANG_CAN = [
        [9],           // Tý   — Quý
        [5, 9, 7],     // Sửu  — Kỷ, Quý, Tân
        [0, 2, 4],     // Dần  — Giáp, Bính, Mậu
        [1],           // Mão  — Ất
        [4, 1, 9],     // Thìn — Mậu, Ất, Quý
        [2, 4, 6],     // Tỵ   — Bính, Mậu, Canh
        [3, 5],        // Ngọ  — Đinh, Kỷ
        [5, 3, 1],     // Mùi  — Kỷ, Đinh, Ất
        [6, 8, 4],     // Thân — Canh, Nhâm, Mậu
        [7],           // Dậu  — Tân
        [4, 7, 3],     // Tuất — Mậu, Tân, Đinh
        [8, 0],        // Hợi  — Nhâm, Giáp
    ];

    /* Vòng TƯƠNG SINH: mộc → hỏa → thổ → kim → thủy → mộc. */
    var SINH = { moc: 'hoa', hoa: 'tho', tho: 'kim', kim: 'thuy', thuy: 'moc' };
    /* Vòng TƯƠNG KHẮC: mộc → thổ → thủy → hỏa → kim → mộc. */
    var KHAC = { moc: 'tho', tho: 'thuy', thuy: 'hoa', hoa: 'kim', kim: 'moc' };

    /* Thập thần, theo quan hệ với NHẬT CHỦ (can ngày) và cùng/khác âm dương. */
    var THAN = {
        tyKien:    { vi: 'Tỷ Kiên',   zh: '比肩' },
        kiepTai:   { vi: 'Kiếp Tài',  zh: '劫財' },
        thucThan:  { vi: 'Thực Thần', zh: '食神' },
        thuongQuan:{ vi: 'Thương Quan', zh: '傷官' },
        thienTai:  { vi: 'Thiên Tài', zh: '偏財' },
        chinhTai:  { vi: 'Chính Tài', zh: '正財' },
        thatSat:   { vi: 'Thất Sát',  zh: '七殺' },
        chinhQuan: { vi: 'Chính Quan', zh: '正官' },
        thienAn:   { vi: 'Thiên Ấn',  zh: '偏印' },
        chinhAn:   { vi: 'Chính Ấn',  zh: '正印' },
    };

    /* ─────────────── Tra cứu ─────────────── */

    var canIdx = {}, chiIdx = {};
    for (var i = 0; i < 10; i++) { canIdx[CAN_VI[i]] = i; canIdx[CAN_ZH[i]] = i; }
    for (var j = 0; j < 12; j++) { chiIdx[CHI_VI[j]] = j; chiIdx[CHI_ZH[j]] = j; }

    /** Chuẩn hoá một mẩu chữ trước khi tra: bỏ khoảng trắng, giữ nguyên dấu. */
    function norm(s) { return String(s == null ? '' : s).trim(); }

    /** Chỉ số can (0–9) của một chữ, hoặc −1. */
    function canOf(s) { var k = canIdx[norm(s)]; return k === undefined ? -1 : k; }
    /** Chỉ số chi (0–11) của một chữ, hoặc −1. */
    function chiOf(s) { var k = chiIdx[norm(s)]; return k === undefined ? -1 : k; }

    /**
     * Hành của MỘT chữ can hoặc chi, hoặc null nếu không phải can chi.
     *
     * Chỉ nhận ĐÚNG một chữ. "Lộ Bàng Thổ" tra ra null dù có chữ "Thổ" ở
     * cuối — nạp âm không phải can chi, và người dùng chốt rõ: chữ nào không
     * phải can chi thì màu đen.
     */
    function hanhOf(s) {
        var a = canOf(s); if (a >= 0) return CAN_HANH[a];
        var b = chiOf(s); if (b >= 0) return CHI_HANH[b];
        return null;
    }

    /** Tàng can của một chi, trả về mảng TÊN theo ngôn ngữ đang chọn. */
    function tangCanOf(chi, zh) {
        var k = chiOf(chi);
        if (k < 0) return [];
        return TANG_CAN[k].map(function (n) { return zh ? CAN_ZH[n] : CAN_VI[n]; });
    }

    /**
     * Thập thần của can `target` khi nhật chủ là `dayMaster`.
     * Trả về tên theo ngôn ngữ, hoặc '' nếu một trong hai không phải can.
     */
    function thapThanOf(dayMaster, target, zh) {
        var d = canOf(dayMaster), t = canOf(target);
        if (d < 0 || t < 0) return '';
        var hd = CAN_HANH[d], ht = CAN_HANH[t];
        // Can chỉ số CHẴN là dương, LẺ là âm (Giáp dương, Ất âm, …).
        var cùngÂmDương = (d % 2) === (t % 2);
        var key;
        if (ht === hd)            key = cùngÂmDương ? 'tyKien'   : 'kiepTai';
        else if (SINH[hd] === ht) key = cùngÂmDương ? 'thucThan' : 'thuongQuan';
        else if (KHAC[hd] === ht) key = cùngÂmDương ? 'thienTai' : 'chinhTai';
        else if (KHAC[ht] === hd) key = cùngÂmDương ? 'thatSat'  : 'chinhQuan';
        else                      key = cùngÂmDương ? 'thienAn'  : 'chinhAn';
        return zh ? THAN[key].zh : THAN[key].vi;
    }

    /* ─────────────── Tô màu ─────────────── */

    /**
     * Bọc một chuỗi can chi thành HTML đã tô màu.
     *
     * Tách theo khoảng trắng rồi tra TỪNG chữ: "Canh Ngọ" ra hai chữ, hai màu
     * (Canh kim, Ngọ hỏa) — không phải một màu cho cả cụm. Chữ nào tra không
     * ra thì để nguyên, không bọc, nên nó thừa hưởng màu đen của khối chứa.
     *
     * Tiếng Trung không có khoảng trắng giữa hai chữ ("庚午"), nên tách thêm
     * theo TỪNG KÝ TỰ khi cả cụm không tra ra.
     */
    function paint(s) {
        var raw = norm(s);
        if (!raw) return '';
        var parts = raw.split(/(\s+)/);
        var out = '';
        for (var i = 0; i < parts.length; i++) {
            var p = parts[i];
            if (/^\s+$/.test(p)) { out += p; continue; }
            var h = hanhOf(p);
            if (h) { out += span(p, h); continue; }
            // Cụm liền không dấu cách: thử tách từng ký tự (tiếng Trung).
            var mỗiKýTự = p.split('');
            var tấtCảLàCanChi = mỗiKýTự.length > 1 &&
                mỗiKýTự.every(function (c) { return hanhOf(c) !== null; });
            if (tấtCảLàCanChi) {
                for (var k = 0; k < mỗiKýTự.length; k++) {
                    out += span(mỗiKýTự[k], hanhOf(mỗiKýTự[k]));
                }
            } else {
                out += esc(p);
            }
        }
        return out;
    }

    function span(txt, hanh) {
        return '<span class="nh nh-' + hanh + '">' + esc(txt) + '</span>';
    }

    function esc(s) {
        return String(s).replace(/&/g, '&amp;').replace(/</g, '&lt;')
                        .replace(/>/g, '&gt;').replace(/"/g, '&quot;');
    }

    /** Đặt nội dung ĐÃ TÔ MÀU cho một phần tử, theo id hoặc theo chính nó. */
    function paintInto(el, s) {
        if (typeof el === 'string') el = document.getElementById(el);
        if (!el) return;
        el.innerHTML = paint(s);
    }

    window.NguHanh = {
        hanhOf: hanhOf,
        canOf: canOf, chiOf: chiOf,
        tangCanOf: tangCanOf,
        thapThanOf: thapThanOf,
        paint: paint, paintInto: paintInto, esc: esc,
        CAN_VI: CAN_VI, CAN_ZH: CAN_ZH, CHI_VI: CHI_VI, CHI_ZH: CHI_ZH,
    };
})();
