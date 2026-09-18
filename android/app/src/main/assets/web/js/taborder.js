/* ════════════════════════════════════════════════════════════════════
   taborder.js — Cho phép người dùng tự sắp xếp thứ tự ba tab

   Ba tab (Kỳ Môn · Lịch · Bát Tự) trước đây chốt cứng theo thứ tự viết trong
   index.html. Ai chủ yếu xem lịch thì vẫn phải với qua tab giữa; ai chỉ dùng
   Bát Tự thì nó nằm tận mép phải — chỗ khó với nhất trên máy 6,5 inch.

   Cử chỉ: GIỮ LÂU rồi KÉO NGANG — đúng cách Android cho sắp xếp lại biểu
   tượng trên màn hình chính, nên không phải dạy. Chạm bình thường vẫn là
   chuyển tab như cũ.

   Thứ tự lưu trong localStorage nên sống qua các lần mở ứng dụng.

   ─── Vì sao ĐỔI CHỖ TRONG DOM chứ không dùng `order` của flex ───
   `order` chỉ đổi thứ tự VẼ, không đổi thứ tự DOM. Mà vạch ngăn giữa các tab
   vẽ bằng `.tab-item + .tab-item { border-left }` — một chọn tử theo DOM. Dùng
   `order` thì vạch ở lại trên tab vốn đứng thứ hai trong HTML, tức rơi vào
   giữa màn hình ở một chỗ tuỳ tiện. Đổi chỗ thật thì mọi chọn tử theo vị trí
   (`+`, `:first-child`) tự đúng, khỏi phải viết lại cái nào.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var KEY = 'qmdj.tabOrder';
    /** Giữ lâu bao nhiêu thì thành "kéo" chứ không còn là "chạm". */
    var HOLD_MS = 420;
    /** Nhúc nhích quá ngần này TRƯỚC khi đủ giờ giữ → đó là vuốt, không phải giữ. */
    var SLOP = 10;

    var bar, drag = null, holdTimer = null, justDragged = false, downX = 0;

    function tabBar() { return document.getElementById('tabBar'); }
    function tabs() {
        return bar ? [].slice.call(bar.querySelectorAll('.tab-item')) : [];
    }

    function readStore() {
        try { return localStorage.getItem(KEY); } catch (e) { return null; }
    }
    function writeStore(v) {
        try { localStorage.setItem(KEY, v); } catch (e) {}
    }

    /** Thứ tự hiện thời, dạng "tabCal,tabQmdj,tabLenh". */
    function currentOrder() {
        return tabs().map(function (el) { return el.id; }).join(',');
    }

    /**
     * Áp một thứ tự vào DOM.
     *
     * Bỏ QUA nếu danh sách lưu không khớp đúng tập tab đang có — phiên bản sau
     * có thể thêm hoặc bỏ tab, và một chuỗi cũ còn sót trong localStorage thì
     * hoặc thiếu tab mới (tab ấy biến mất khỏi thanh) hoặc trỏ tới tab đã gỡ.
     * Thà quay về thứ tự mặc định còn hơn.
     */
    function applyOrder(str) {
        if (!bar || !str) return false;
        var want = String(str).split(',').filter(Boolean);
        var have = tabs();
        if (want.length !== have.length) return false;
        var byId = {};
        have.forEach(function (el) { byId[el.id] = el; });
        for (var i = 0; i < want.length; i++) if (!byId[want[i]]) return false;
        want.forEach(function (id) { bar.appendChild(byId[id]); });
        return true;
    }

    /* ─────────────── Kéo ─────────────── */

    function startDrag(el, x) {
        drag = {
            el: el, startX: x, dx: 0,
            /* Chốt bề rộng một tab MỘT LẦN lúc bắt đầu: ba tab luôn rộng bằng
               nhau (flex: 1 1 0), và đo lại giữa chừng thì mỗi lần đổi chỗ là
               một lần đọc lại bố cục đang có transform — vừa chậm vừa nhiễu. */
            w: el.getBoundingClientRect().width || 1,
        };
        el.classList.add('tab-drag');
        bar.classList.add('tab-reordering');
    }

    function moveDrag(x) {
        if (!drag) return;
        drag.dx = x - drag.startX;

        // Kéo qua QUÁ NỬA một tab thì đổi chỗ với tab bên cạnh — rồi dời mốc
        // đi đúng một tab, để phần dịch còn lại tính tiếp từ chỗ mới. Không
        // dời mốc thì vừa đổi xong nó lại đủ điều kiện đổi tiếp, và hai tab
        // nhấp nháy qua lại suốt lúc đang kéo.
        var list = tabs();
        var i = list.indexOf(drag.el);
        if (drag.dx > drag.w / 2 && i < list.length - 1) {
            bar.insertBefore(list[i + 1], drag.el);
            drag.startX += drag.w;
            drag.dx -= drag.w;
        } else if (drag.dx < -drag.w / 2 && i > 0) {
            bar.insertBefore(drag.el, list[i - 1]);
            drag.startX -= drag.w;
            drag.dx += drag.w;
        }
        drag.el.style.transform = 'translateX(' + drag.dx + 'px)';
    }

    function endDrag() {
        if (!drag) return;
        drag.el.style.transform = '';
        drag.el.classList.remove('tab-drag');
        bar.classList.remove('tab-reordering');
        drag = null;
        justDragged = true;              // nuốt cú click sinh ra khi nhấc tay
        writeStore(currentOrder());
    }

    function cancelHold() {
        clearTimeout(holdTimer);
        holdTimer = null;
    }

    function onDown(e) {
        // XOÁ cờ nuốt-click ngay đầu mỗi lần chạm mới.
        //
        // Nhấc tay sau một cú kéo KHÔNG phải lúc nào cũng sinh ra `click`:
        // trình duyệt chỉ sinh nếu điểm nhấn và điểm nhả cùng nằm trên một
        // phần tử, mà kéo thì theo định nghĩa là không. Cờ vì thế nằm lại đó
        // chờ, rồi nuốt oan cú CHẠM KẾ TIẾP — đo thật: sắp xếp xong, bấm tab
        // Lịch một cái, không có gì xảy ra. Mỗi lần chạm mới bắt đầu sạch sẽ
        // thì cờ chỉ còn tác dụng trong đúng lần tương tác sinh ra nó.
        justDragged = false;
        if (drag) return;
        var el = e.target.closest ? e.target.closest('.tab-item') : null;
        if (!el || !bar.contains(el)) return;
        var x = e.clientX;
        cancelHold();
        downX = x;
        holdTimer = setTimeout(function () { startDrag(el, x); }, HOLD_MS);
    }

    function onMove(e) {
        if (drag) { moveDrag(e.clientX); return; }
        // Chưa đủ giờ giữ mà đã trượt đi → người dùng đang vuốt, không phải
        // định sắp xếp. Huỷ hẹn giờ.
        if (holdTimer && Math.abs(e.clientX - downX) > SLOP) cancelHold();
    }

    function onUp() {
        cancelHold();
        if (drag) endDrag();
    }

    function init() {
        bar = tabBar();
        if (!bar) return;
        applyOrder(readStore());

        // Pointer Events: một đường mã cho cả chạm lẫn chuột. WebView của
        // Android hỗ trợ từ lâu, và `touch-action: none` ở .tab-item
        // (calendar.css) giữ cho pointermove không bị trình duyệt cướp mất
        // giữa chừng để cuộn trang.
        bar.addEventListener('pointerdown', onDown);
        // Nghe trên WINDOW chứ không trên thanh tab: ngón tay kéo chệch lên
        // trên khỏi thanh là mất luôn sự kiện nếu chỉ nghe ở thanh, và tab
        // đang kéo kẹt lại giữa chừng.
        window.addEventListener('pointermove', onMove);
        window.addEventListener('pointerup', onUp);
        window.addEventListener('pointercancel', onUp);

        // Nhấc tay sau khi kéo thì trình duyệt vẫn sinh một `click` — mà
        // calendar.js gắn showTab() vào đúng sự kiện ấy, nên vừa sắp xếp xong
        // là nhảy sang tab vừa kéo. Chặn ở pha BẮT (capture) để chặn trước khi
        // nó tới người nghe kia.
        bar.addEventListener('click', function (e) {
            if (!justDragged) return;
            justDragged = false;
            e.stopPropagation();
            e.preventDefault();
        }, true);
    }

    if (document.readyState === 'loading') {
        document.addEventListener('DOMContentLoaded', init);
    } else {
        init();
    }

    /* Cho bộ kiểm thử — và cho bất cứ ai muốn đặt lại thứ tự bằng mã. */
    window.__tabOrder = function (str) {
        if (str) { if (applyOrder(str)) writeStore(currentOrder()); }
        return currentOrder();
    };
    window.__tabOrderReset = function () {
        try { localStorage.removeItem(KEY); } catch (e) {}
        applyOrder('tabQmdj,tabCal,tabLenh,tabTraCuu');
        return currentOrder();
    };
})();
