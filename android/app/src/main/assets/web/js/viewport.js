/* ════════════════════════════════════════════════════════════════════
   viewport.js — Vừa khít mọi màn hình / Fit the layout to any screen

   Bản web gốc chỉ có một luật duy nhất:

       @media (min-width: 768px) { body { zoom: 1.25; } }

   Luật này chỉ nhìn CHIỀU RỘNG. Điện thoại xoay ngang (S21: 800×360) rộng
   hơn 768px nên bị phóng to 1,25 lần → bàn Kỳ Môn cao 500px trên màn hình
   chỉ cao 360px. Ngược lại, ở màn dọc S21 (360×800) nội dung chỉ cao ~693px,
   bỏ phí hơn 100px cuối màn hình trong khi chữ thì bé.

   Module này thay bằng một hệ số tỉ lệ tính từ CẢ hai chiều:

       tỉ lệ = min(rộng khả dụng / rộng thiết kế, cao khả dụng / cao nội dung)

   rồi kẹp trong khoảng an toàn. Bố cục không đổi một chút nào — chỉ to/nhỏ
   theo màn hình, nên vẫn giống hệt bản web gốc.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    // Không thu nhỏ quá mức đọc được: thà cuộn còn hơn chữ li ti.
    var MIN_SCALE = 0.95;
    // Không phóng quá to trên máy tính bảng / điện thoại gập.
    var MAX_SCALE = 1.6;
    // Khoảng cách gốc giữa các bảng (app.css) và mức giãn tối đa cho phép.
    var BASE_GAP = 3;
    var MAX_EXTRA_GAP = 14;

    var lastW = 0, lastH = 0, timer = null;

    /**
     * Đang bận thì đừng đo lại: bàn phím ảo làm innerHeight tụt xuống một nửa,
     * đo lúc đó sẽ thu bé cả giao diện; hộp thoại đang mở cũng vậy.
     */
    function isBusy() {
        var ae = document.activeElement;
        if (ae && /^(INPUT|TEXTAREA|SELECT)$/.test(ae.tagName)) return true;
        return !!document.querySelector('#locOverlay.open, #drumOverlay.open');
    }

    function apply() {
        if (isBusy()) return;
        var body = document.body;
        if (!body) return;


        // Thanh tab cố định che mất phần đáy, nên phải chừa ĐÚNG chiều cao của
        // nó. Đặt số cứng thì hoặc thừa (hở một khoảng ở đáy màn Kỳ Môn) hoặc
        // thiếu (che mất dòng cuối) — đo thẳng vẫn chắc hơn.
        var bar = document.getElementById('tabBar');
        if (bar) {
            var h = Math.round(bar.getBoundingClientRect().height);
            if (h > 0) document.documentElement.style.setProperty('--tabbar-h', h + 'px');
        }

        // Đo chiều cao tự nhiên ở tỉ lệ 1 VÀ khoảng cách mặc định, nếu không
        // phần thừa chia ở lần trước sẽ cộng dồn qua mỗi lần đo.
        var prev = body.style.zoom;
        body.style.zoom = '';
        body.style.gap = BASE_GAP + 'px';
        var natH = body.scrollHeight;
        var availW = document.documentElement.clientWidth;
        var availH = window.innerHeight;

        if (!natH || !availW || !availH) { body.style.zoom = prev; return; }

        // Bề rộng "tự nhiên" phải ĐO, không được đoán: bố cục vốn co giãn
        // (width:100%, max-width:400px) nên trên màn hẹp nó đã vừa khít rồi.
        // Lấy hằng số 412px mà chia sẽ ra tỉ lệ < 1 và thu bé giao diện một
        // cách vô cớ trên chính S21 (360px).
        //
        // Phải đo bảng ĐANG HIỆN. Đo cứng '.controls' thì sang tab Lịch nó bị
        // ẩn, bề rộng đọc ra 0 → tỉ lệ vọt lên và cả trang tràn ngang.
        var natW = 0;
        for (var k = 0; k < body.children.length; k++) {
            var kid = body.children[k];
            var kcs = getComputedStyle(kid);
            if (kcs.display === 'none' || kcs.position === 'fixed') continue;
            var kw = kid.getBoundingClientRect().width;
            if (kw > natW) natW = kw;
        }
        natW = natW ? natW + 12 : availW;      // + padding hai bên của body

        // Phóng to bị chặn bởi CẢ hai chiều: rộng ra thì tràn ngang, cao quá
        // thì phải cuộn.
        var scale = Math.min(availW / natW, availH / natH);
        scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
        body.style.zoom = scale.toFixed(3);

        spreadSlack(body, availH, natH * scale, scale);

        lastW = availW;
        lastH = availH;
    }

    /**
     * Bề ngang đã kịch trần thì không phóng to thêm được nữa, nên ở màn dọc
     * luôn còn thừa vài chục pixel chiều cao. Nếu cứ để nguyên, toàn bộ phần
     * thừa dồn xuống đáy thành một khoảng hở ngay trên thanh tab.
     * Chia đều phần thừa đó vào khoảng cách giữa các bảng — màn hình đầy đặn,
     * không còn khoảng hở ở đáy.
     */
    function spreadSlack(body, availH, usedH, scale) {
        var kids = [];
        for (var i = 0; i < body.children.length; i++) {
            var el = body.children[i];
            var cs = getComputedStyle(el);
            if (cs.display === 'none' || cs.position === 'fixed') continue;
            kids.push(el);
        }
        var gaps = kids.length - 1;
        var leftover = availH - usedH;
        if (gaps < 1 || leftover < 6) return;
        // gap tính theo px CHƯA phóng, nên phải chia lại cho hệ số.
        var extra = Math.min(MAX_EXTRA_GAP, leftover / scale / gaps);
        body.style.gap = (BASE_GAP + extra).toFixed(1) + 'px';
    }

    /** Gộp nhiều sự kiện resize liên tiếp thành một lần đo. */
    function schedule(delay) {
        clearTimeout(timer);
        timer = setTimeout(apply, delay || 120);
    }

    /**
     * Chỉ đo lại khi kích thước đổi thật sự (xoay máy, mở/đóng cửa sổ), bỏ qua
     * thay đổi nhỏ do thanh địa chỉ hoặc bàn phím.
     */
    function onResize() {
        var w = document.documentElement.clientWidth;
        var h = window.innerHeight;
        if (Math.abs(w - lastW) < 2 && h <= lastH) return;   // h tụt = bàn phím
        schedule(150);
    }

    window.addEventListener('resize', onResize);
    window.addEventListener('orientationchange', function () { schedule(250); });

    document.addEventListener('DOMContentLoaded', function () {
        // Nội dung đổi chiều cao khi đổi phái hoặc mở bảng Nhật–Nguyệt →
        // đo lại sau mỗi lần tính.
        if (typeof processAll === 'function' && !processAll.__fitWrapped) {
            var orig = processAll;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                schedule(60);
                return r;
            };
            wrapped.__fitWrapped = true;
            window.processAll = wrapped;
        }
        schedule(200);
    });

    window.addEventListener('load', function () { schedule(300); });

    // Cho phép gọi tay (ví dụ sau khi đóng bàn phím).
    window.__fitScreen = apply;
})();
