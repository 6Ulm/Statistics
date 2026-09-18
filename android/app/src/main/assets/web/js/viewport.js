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
    // Khoảng cách giữa các bảng khi chưa chia gì thêm — đúng như bản web gốc.
    var BASE_GAP = 3;
    // Trần của khe sau khi rót phần thừa vào. Không có trần thì trên máy cao
    // (A51: dôi 82px cho 4 khe) các bảng rời rạc hẳn ra, xấu hơn cả khoảng hở.
    //
    // 28 chứ không 24 từ khi bảng chi tiết Âm Bàn pháp bị gỡ khỏi tab Kỳ Môn:
    // tab ấy mất một khối cao ~44px, nên phần dôi tăng đúng ngần ấy mà số KHE
    // lại giảm đi một (5 khối → 4 khe). Ở trần 24px, A51 852px chạm trần rồi
    // vẫn còn thừa 16,3px nằm chết ngay trên thanh dưới — đúng cái khoảng hở
    // mà cả cơ chế này sinh ra để xoá. 28px hấp thụ trọn 16,3px ấy (4 khe ×
    // 4,1px) mà vẫn còn là một khe, không phải một quãng trống.
    // S21 FE (19,3px) và S21 (17,6px) chưa chạm trần nên không đổi gì.
    var GAP_MAX = 28;

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

        // Thanh dưới cố định che mất phần đáy, nên phải chừa ĐÚNG chiều cao của
        // nó. Phải đo SAU khi bỏ zoom: đo lúc còn zoom thì con số là px đã
        // phóng, mà --tabbar-h lại được dùng như px chưa phóng — sai lệch đó
        // đủ để phép co giãn vượt quá màn hình vài pixel.
        //
        // Đo cả #bottomDock chứ không riêng #tabBar: hàng ngôn ngữ + địa điểm
        // nằm DƯỚI hai tab trong cùng thanh ấy, bỏ sót nó thì đáy trang bị
        // thanh che mất đúng chiều cao một hàng.
        var bar = document.getElementById('bottomDock') || document.getElementById('tabBar');
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
        //
        // TRỪ tab Lịch: chiều cao "tự nhiên" của nó KHÔNG phải một con số cố
        // định. fitGrid() trong calendar.js chia lại chiều cao từng khối theo
        // innerHeight, nên đo ở tỉ lệ nào nó cũng vừa khít đúng tỉ lệ ấy — đem
        // con số ấy đi tính tỉ lệ là hai cơ chế cùng kéo một sợi dây: mọi tỉ lệ
        // trong [MIN_SCALE; 1] đều tự nhất quán, và app dừng ở đâu là tuỳ thứ
        // tự chạy (đo được 0,976 lần này, 1,000 lần sau, trên cùng một máy).
        // Ở tab ấy chỉ lấy tỉ lệ theo BỀ NGANG — thứ không co giãn — rồi để
        // vòng hạ dần bên dưới lo nốt máy quá thấp (fitGrid có sàn SEC_MIN nên
        // không bóp mãi được).
        // Tab Lệnh (Bát Tự) TỪNG nằm trong diện này, vì bảng 33 hàng của nó
        // được kẹp chiều cao theo innerHeight. Nay bảng ấy bung đủ chiều cao
        // tự nhiên và cả trang cuộn (xem fitLenhSec trong lenh.js), nên chiều
        // cao của tab KHÔNG còn là hàm của tỉ lệ nữa — nó cư xử y như tab Kỳ
        // Môn, và phải được đo y như thế. Để lại trong diện co giãn thì phép
        // đo vẫn đúng nhưng vế chiều cao bị bỏ qua, và trang dài gấp đôi màn
        // hình vẫn được giữ nguyên tỉ lệ thay vì thu lại như tab Kỳ Môn.
        var fitTab = body.classList.contains('view-cal') ? window.__calFit : null;
        var elastic = typeof fitTab === 'function';
        var scale = elastic ? (availW / natW)
                            : Math.min(availW / natW, (availH - 2) / natH);
        scale = Math.max(MIN_SCALE, Math.min(MAX_SCALE, scale));
        body.style.zoom = scale.toFixed(3);

        // Phóng to KHÔNG chỉ là nhân chiều cao lên: chữ ngắt dòng lại, mỗi phần
        // tử con làm tròn một ít, nên nội dung có thể cao hơn cả natH × tỉ lệ.
        // Vì vậy phải đo kết quả THẬT rồi hạ dần cho tới khi vừa — một nhịp
        // không đủ (đo được 1,091 → 1,088 → vẫn dôi ra 1px).
        for (var pass = 0; pass < 4 && scale > MIN_SCALE; pass++) {
            // Tab Lịch phải được chia lại theo tỉ lệ vừa đặt TRƯỚC KHI đo,
            // bằng không ta đo một bố cục dựng cho tỉ lệ cũ.
            if (elastic) fitTab();
            var realH = body.getBoundingClientRect().height;
            if (realH <= availH) break;
            scale = Math.max(MIN_SCALE, scale * (availH / realH) - 0.002);
            body.style.zoom = scale.toFixed(3);
        }
        // Vòng trên thoát vì CHẠM SÀN thì tỉ lệ vừa đổi mà chưa kịp chia lại
        // (điều kiện lặp thành sai ngay), để lại một bố cục dựng cho tỉ lệ cũ.
        // Chỉ chia lại trong đúng trường hợp ấy: thoát vì ĐÃ VỪA thì bố cục
        // đang đúng rồi, chia thêm chỉ nới nội dung ra sát mép và làm trang
        // dôi ra vài pixel (đo trên tablet: 9px, đủ để hiện thanh cuộn).
        if (elastic && scale <= MIN_SCALE) fitTab();

        // ── Rót phần thừa chiều cao vào các khe ──
        // Phóng to bị CHẶN BỞI BỀ NGANG: trên S21 FE tỉ lệ đã kịch 1,0 vì rộng,
        // trong khi chiều cao còn dôi 39px (A51: 82px) nằm chết ở đáy màn hình
        // ngay trên thanh dưới. Bàn Kỳ Môn là lưới vuông nên không cao thêm
        // được nếu không rộng thêm, vậy chỗ duy nhất nhận được phần dôi ấy là
        // khe giữa các bảng.
        //
        // Tab Lệnh (Bát Tự) BỎ QUA bước này: lý luận trên chỉ đúng cho bàn Kỳ
        // Môn (lưới vuông, cỡ CỐ ĐỊNH, không thể cao thêm) — Đại Vận ở tab
        // Lệnh thì NGƯỢC LẠI, cao tuỳ nội dung của chính nó, không có "chỗ
        // dôi" nào cần lấp bằng khe cho vừa khít một màn hình. Người dùng
        // phản ánh đúng khe này (.controls → #tuTruPanel → #lenhView) từng bị
        // bơm gần kịch trần GAP_MAX (đo được 21.5px, so với BASE_GAP 3px),
        // nhìn thừa hẳn.
        var kids = [];
        for (var g = 0; g < body.children.length; g++) {
            var el = body.children[g];
            var cs = getComputedStyle(el);
            if (cs.display !== 'none' && cs.position !== 'fixed') kids.push(el);
        }
        if (kids.length > 1 && !body.classList.contains('view-lenh')) {
            var used = body.getBoundingClientRect().height;
            var spare = (availH - used) / (parseFloat(body.style.zoom) || 1);
            if (spare > 1) {
                var gap = Math.min(GAP_MAX, BASE_GAP + spare / (kids.length - 1));
                body.style.gap = gap.toFixed(1) + 'px';
            }
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
