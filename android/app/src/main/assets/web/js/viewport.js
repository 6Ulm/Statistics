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
    // Khoảng cách giữa các bảng: đúng như bản web gốc, không nới thêm.
    // Từng thử chia phần thừa chiều cao vào các khe cho kín đáy màn hình,
    // nhưng các bảng rời rạc trông tệ hơn hẳn khoảng hở ở đáy.
    var BASE_GAP = 3;

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


        // Đo mọi thứ ở tỉ lệ 1 — kể cả thanh tab.
        var prev = body.style.zoom;
        body.style.zoom = '';
        body.style.gap = BASE_GAP + 'px';

        // Thanh tab cố định che mất phần đáy, nên phải chừa ĐÚNG chiều cao của
        // nó. Phải đo SAU khi bỏ zoom: đo lúc còn zoom thì con số là px đã
        // phóng, mà --tabbar-h lại được dùng như px chưa phóng — sai lệch đó
        // đủ để phép co giãn vượt quá màn hình vài pixel.
        var bar = document.getElementById('tabBar');
        if (bar) {
            var barH = Math.round(bar.getBoundingClientRect().height);
            if (barH > 0) document.documentElement.style.setProperty('--tabbar-h', barH + 'px');
        }

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
        // Trừ 2px dự phòng: làm tròn nửa pixel khi phóng đủ để đẩy trang dài
        // hơn màn hình 1px, thế là hiện thanh cuộn dù nội dung vừa khít.
        var scale = Math.min(availW / natW, (availH - 2) / natH);
        scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
        body.style.zoom = scale.toFixed(3);

        // Phóng to KHÔNG chỉ là nhân chiều cao lên: chữ ngắt dòng lại, mỗi phần
        // tử con làm tròn một ít, nên nội dung có thể cao hơn cả natH × tỉ lệ.
        // Vì vậy phải đo kết quả THẬT rồi hạ dần cho tới khi vừa — một nhịp
        // không đủ (đo được 1,091 → 1,088 → vẫn dôi ra 1px).
        for (var pass = 0; pass < 4 && scale > MIN_SCALE; pass++) {
            var realH = body.getBoundingClientRect().height;
            if (realH <= availH) break;
            scale = Math.max(MIN_SCALE, scale * (availH / realH) - 0.002);
            body.style.zoom = scale.toFixed(3);
        }

        lastW = availW;
        lastH = availH;
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
