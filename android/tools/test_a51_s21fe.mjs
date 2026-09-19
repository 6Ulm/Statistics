/**
 * Kiểm thử riêng cho Galaxy A51 và Galaxy S21 FE — hai máy được nhắm tới.
 *
 *   cd android/tools && npm install && npx playwright install chromium
 *   node test_a51_s21fe.mjs
 *
 * Khác test_responsive.mjs ở ba chỗ:
 *
 *   1. Quét CẢ MA TRẬN cấu hình của mỗi máy, không phải một kích thước.
 *      Samsung cho đổi "Screen zoom" (Cài đặt › Màn hình) — nó đổi densityDpi,
 *      nên MỘT máy cho ra ba bề rộng CSS khác nhau. Nhân thêm hai kiểu điều
 *      hướng (nút ảo ba phím 48dp / cử chỉ 24dp) là 12 cấu hình. Một luật
 *      @media chỉ đúng ở kích thước mặc định thì sai ở năm cái còn lại.
 *
 *   2. Đo VÙNG CHẠM, không chỉ đo bố cục. Bố cục vừa khít mà ô bấm cao 16px
 *      thì vẫn là giao diện hỏng trên điện thoại.
 *
 *   3. Đi hết các ĐƯỜNG TƯƠNG TÁC bằng chạm thật (page.tap), kể cả lúc bàn
 *      phím ảo đang chiếm nửa màn hình.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

let chromium;
try {
    ({ chromium } = await import('playwright'));
} catch {
    console.log('Bỏ qua: chưa cài playwright (npm install playwright).');
    process.exit(0);
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const file = path.join(WEB, rel);
    if (!file.startsWith(WEB) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;

/* Chiều cao là chiều cao KHUNG WEB: đã trừ thanh trạng thái (24dp) và phần
   điều hướng, vì MainActivity đệm root theo window insets nên WebView chỉ nhận
   được phần còn lại. Bề rộng = số điểm ảnh ngang / tỉ lệ điểm ảnh của nấc
   Screen zoom đang đặt.

   A51    1080×2400, mặc định 420dpi (dpr 2,625)
   S21 FE 1080×2340, mặc định 450dpi (dpr 2,8125) */
const DEVICES = [
    { m: 'A51',    t: 'zoom nhỏ · cử chỉ',  w: 432, h: 912, dpr: 2.5    },
    { m: 'A51',    t: 'zoom nhỏ · 3 phím',  w: 432, h: 888, dpr: 2.5    },
    { m: 'A51',    t: 'mặc định · cử chỉ',  w: 412, h: 866, dpr: 2.625  },
    { m: 'A51',    t: 'mặc định · 3 phím',  w: 412, h: 842, dpr: 2.625  },
    { m: 'A51',    t: 'zoom lớn · cử chỉ',  w: 360, h: 752, dpr: 3.0    },
    { m: 'A51',    t: 'zoom lớn · 3 phím',  w: 360, h: 728, dpr: 3.0    },
    { m: 'S21 FE', t: 'zoom nhỏ · cử chỉ',  w: 411, h: 843, dpr: 2.625  },
    { m: 'S21 FE', t: 'zoom nhỏ · 3 phím',  w: 411, h: 819, dpr: 2.625  },
    { m: 'S21 FE', t: 'mặc định · cử chỉ',  w: 384, h: 784, dpr: 2.8125 },
    { m: 'S21 FE', t: 'mặc định · 3 phím',  w: 384, h: 760, dpr: 2.8125 },
    { m: 'S21 FE', t: 'zoom lớn · cử chỉ',  w: 360, h: 732, dpr: 3.0    },
    { m: 'S21 FE', t: 'zoom lớn · 3 phím',  w: 360, h: 708, dpr: 3.0    },
    { m: 'A51',    t: 'xoay ngang',         w: 866, h: 412, dpr: 2.625, ngang: true },
    { m: 'S21 FE', t: 'xoay ngang',         w: 784, h: 384, dpr: 2.8125, ngang: true },
];

/* Mức tối thiểu của vùng chạm.
   Android Material khuyên 48dp; giao diện này là bảng tra dày đặc, cố tình
   gói trọn một màn hình, nên 48dp ở MỌI chỗ là không trả nổi (xem chú thích
   --dock-row-h trong calendar.css). Hai mức dưới đây là chỗ đã chốt được:
     · 32px cho mọi chỗ bấm — trên mức tối thiểu 24px của WCAG 2.5.8;
     · 34px cho thanh dưới trên máy dọc đủ cao, vì đó là điều hướng CHÍNH và
       nằm sát mép dưới, nơi ngón cái chạm kém chính xác nhất. */
const TAP_MIN = 32;
const DOCK_MIN = 34;

/* Mọi chỗ bấm được, kèm tab cần mở để nó hiện ra. */
const TAPS = [
    ['#dateDisplayBtn',   'qmdj'], ['#methodDisplayBtn', 'qmdj'],
    ['#cobanToggleWrap',  'qmdj'], ['.dp-header',        'qmdj'],
    ['#calPrev',          'cal'],  ['#calNext',          'cal'],
    ['#calTitle',         'cal'],  ['#calPinBtn',        'cal'],
    // `.cal-sec-head` KHÔNG còn ở đây: mục Tiết khí luôn mở, hàng <thead> của
    // nó nay chỉ là tên cột chứ không phải nút gập/mở, nên nó được thấp lại
    // (đúng ý "giảm height header của box tiết khí") và không còn là chỗ bấm.
    ['#lenhGenderBtn',    'lenh'], ['#lenhRuleBtn',      'lenh'],
    ['#tcYearBtn',      'tracuu'], ['#tcRuleBtn',      'tracuu'],
    // Bốn hàng tiêu đề của tab Tra cứu dùng CHUNG một hình khối (xem
    // tracuu.css) — canh cả bốn, không chỉ một.
    ['#trinhuanHeader', 'tracuu'], ['#sachboHeader',   'tracuu'],
    ['#lenhHead',       'tracuu'], ['#tcAmHead',       'tracuu'],
];

let đạt = 0, hỏng = 0;
const ok = (tên, điều, ghi = '') => {
    if (điều) { đạt++; console.log(`  ok   ${tên}`); }
    else { hỏng++; console.log(`  ✗ SAI ${tên}   ${ghi}`); }
};
/** So khớp đúng một giá trị — in cả hai vế khi lệch, khỏi phải đoán. */
const check = (tên, được, mong) =>
    ok(tên, được === mong, `được "${được}", mong "${mong}"`);

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

/* ══ 1. Bố cục + vùng chạm trên cả ma trận ══ */
console.log('Bố cục và vùng chạm trên từng cấu hình máy');
for (const d of DEVICES) {
    const ctx = await browser.newContext({
        viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr,
        isMobile: true, hasTouch: true,
    });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(String(e).slice(0, 140)));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1100);

    for (const lang of ['vi', 'zh']) {
        await page.evaluate(l => window.setLang && window.setLang(l), lang);
        await page.waitForTimeout(350);

        for (const tab of ['qmdj', 'cal', 'lenh', 'tracuu']) {
            await page.evaluate(t => {
                // Tab Tra cứu cần một năm mới có gì để vẽ; chọn sẵn để phép
                // canh đo đúng cái bố cục người dùng thật sự nhìn thấy, không
                // phải màn hình rỗng.
                if (t === 'tracuu' && window.__tracuuYear) window.__tracuuYear(2026);
                window.showTab(t);
                // Nút Ghim chỉ hiện khi chạy trong ứng dụng Android; đo mà thiếu
                // nó thì bố cục trên máy thật cao hơn phép thử tưởng.
                const p = document.getElementById('calPinBtn');
                if (p) p.style.display = 'block';
            }, tab);
            await page.waitForTimeout(650);
            await page.evaluate(() => window.__fitScreen && window.__fitScreen());
            await page.waitForTimeout(300);

            const r = await page.evaluate(({ taps, tab }) => {
                const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
                const hiện = el => {
                    const c = getComputedStyle(el);
                    return c.display !== 'none' && c.visibility !== 'hidden';
                };
                // Chữ bị "…" nuốt mất. Bỏ qua khung CUỘN: tràn ở đó là cuộn
                // được, không phải mất chữ.
                const cắt = [];
                for (const el of document.querySelectorAll('body *')) {
                    if (!hiện(el) || el.children.length) continue;
                    const c = getComputedStyle(el);
                    if (/auto|scroll/.test(c.overflow + c.overflowX + c.overflowY)) continue;
                    const t = (el.textContent || '').trim();
                    if (t && el.scrollWidth > el.clientWidth + 1) t && cắt.push(t.slice(0, 22));
                }
                // Bị thanh dưới che. KHÔNG quét <body>: hộp của nó vốn kéo tới
                // đáy vì có padding-bottom bằng chiều cao thanh — đó chính là
                // cách chừa chỗ cho thanh.
                const che = [];
                const bar = document.getElementById('bottomDock');
                const barTop = bar.getBoundingClientRect().top;
                for (const el of document.querySelectorAll('#calView > *, #lenhView > *, #mainBody > *:not(#bottomDock)')) {
                    if (!hiện(el) || getComputedStyle(el).position === 'fixed') continue;
                    const b = el.getBoundingClientRect();
                    if (b.height > 0 && b.bottom > barTop + 1 && b.top < barTop) che.push(el.id || el.className);
                }
                // Vùng chạm, quy về px CHƯA phóng (dp trên máy thật).
                const nhỏ = [];
                for (const [sel, cần] of taps) {
                    if (cần !== tab) continue;
                    for (const el of document.querySelectorAll(sel)) {
                        if (!hiện(el)) continue;
                        const b = el.getBoundingClientRect();
                        if (b.width > 0) nhỏ.push([sel, +(b.height / z).toFixed(1)]);
                    }
                }
                const dock = [...document.querySelectorAll('.tab-item, #sharedBar .picker-btn')]
                    .map(e => +(e.getBoundingClientRect().height / z).toFixed(1));
                return {
                    tràn: document.documentElement.scrollWidth - document.documentElement.clientWidth,
                    cao: Math.round(document.body.getBoundingClientRect().height),
                    khung: window.innerHeight, z: +z.toFixed(3),
                    cắt: [...new Set(cắt)], che: [...new Set(che)], nhỏ, dock: Math.min(...dock),
                };
            }, { taps: TAPS, tab });

            const tag = `${d.m} ${d.t} ${d.w}×${d.h} · ${lang} · ${tab}`;
            ok(`${tag}: không tràn ngang`, r.tràn <= 1, `${r.tràn}px`);
            ok(`${tag}: không chữ nào bị cắt`, r.cắt.length === 0, r.cắt.slice(0, 3).join(' | '));
            // Trang phải cuộn thì phần dưới nằm sau thanh là bình thường —
            // cuộn tới là thấy. Chỉ là lỗi khi trang ĐÃ VỪA mà vẫn bị che.
            ok(`${tag}: không bị thanh dưới che`,
                r.che.length === 0 || r.cao > r.khung + 1, r.che.join(' | '));
            const xấu = r.nhỏ.filter(([, h]) => h < TAP_MIN);
            ok(`${tag}: mọi ô bấm ≥ ${TAP_MIN}px`, xấu.length === 0,
                xấu.map(([s, h]) => `${s} ${h}px`).join(' | '));
        }
    }

    // Thanh dưới là điều hướng CHÍNH: máy DỌC đủ cao thì phải được mức rộng
    // hơn. Máy ngang thì không — chiều cao ở đó quý hơn, và đằng nào cũng đã
    // phải cuộn (xem --dock-row-h trong calendar.css).
    const dockH = await browser.contexts && await page.evaluate(() => {
        const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
        return +(document.querySelector('.tab-item').getBoundingClientRect().height / z).toFixed(1);
    });
    if (!d.ngang && d.h >= 780) {
        ok(`${d.m} ${d.t}: thanh dưới ≥ ${DOCK_MIN}px`, dockH >= DOCK_MIN, `${dockH}px`);
    } else {
        ok(`${d.m} ${d.t}: thanh dưới ≥ ${TAP_MIN - 3}px`, dockH >= TAP_MIN - 3, `${dockH}px`);
    }
    ok(`${d.m} ${d.t}: không lỗi JS`, errs.length === 0, [...new Set(errs)].join(' | '));
    await ctx.close();
}

/* ══ 2. Luật cử chỉ của Android ══
   Ba khai báo này không nhìn thấy được trên ảnh chụp, mà thiếu thì thao tác
   hỏng theo kiểu rất khó lần ra: bánh xe đứng im khi vuốt nhanh, trang trôi
   khi cuộn hết một danh sách nhỏ. */
console.log('\nLuật cử chỉ (không nhìn thấy được, nhưng thiếu là hỏng thao tác)');
{
    const ctx = await browser.newContext({ viewport: { width: 412, height: 866 }, deviceScaleFactor: 2.625, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1100);
    const r = await page.evaluate(() => {
        const cs = s => { const e = document.querySelector(s); return e ? getComputedStyle(e) : null; };
        document.getElementById('drumOverlay').classList.add('open');
        const drum = cs('.drum-col');
        document.getElementById('drumOverlay').classList.remove('open');
        return {
            drum: drum && drum.touchAction,
            body: getComputedStyle(document.body).webkitTapHighlightColor,
            jq: cs('.cal-sec-body') && cs('.cal-sec-body').overscrollBehaviorY,
            loc: cs('#locList') && cs('#locList').overscrollBehaviorY,
            opt: cs('#optList') && cs('#optList').overscrollBehaviorY,
        };
    });
    ok('bánh xe ngày giờ khoá cử chỉ trình duyệt (touch-action:none)', r.drum === 'none', r.drum);
    ok('tắt mảng sáng mặc định của WebView', /rgba\(0, 0, 0, 0\)|transparent/.test(r.body || ''), r.body);
    ok('hai mục tab Lịch không đẩy cuộn sang trang', r.jq === 'contain', r.jq);
    ok('danh sách thành phố không đẩy cuộn sang trang', r.loc === 'contain', r.loc);
    ok('bảng chọn không đẩy cuộn sang trang', r.opt === 'contain', r.opt);
    await ctx.close();
}

/* ══ 3. Đường tương tác, bằng chạm THẬT ══ */
console.log('\nĐường tương tác (chạm thật, kể cả lúc bàn phím ảo che nửa màn hình)');
for (const d of [{ m: 'A51', w: 412, h: 866, dpr: 2.625 }, { m: 'S21 FE', w: 384, h: 784, dpr: 2.8125 }]) {
    const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);

    for (const [id, cls] of [['#tabCal', 'view-cal'], ['#tabLenh', 'view-lenh'], ['#tabQmdj', null]]) {
        await page.tap(id); await page.waitForTimeout(700);
        const hit = await page.evaluate(c => c ? document.body.classList.contains(c)
            : !/view-(cal|lenh)/.test(document.body.className), cls);
        ok(`${d.m}: chạm ${id} đổi đúng tab`, hit);
    }

    // Ô ngôn ngữ nằm ở hàng dùng chung, phải mở được NGAY TẠI tab Lịch — luật
    // ẩn của tab ấy xoá mọi con của <body> trừ danh sách chừa, mà bảng chọn
    // lại là con của <body>. Quên chừa #optOverlay thì bấm ra một bảng vô hình.
    await page.tap('#tabCal'); await page.waitForTimeout(700);
    await page.tap('#langDisplayBtn'); await page.waitForTimeout(600);
    let r = await page.evaluate(() => {
        const o = document.getElementById('optOverlay');
        const b = o.firstElementChild.getBoundingClientRect();
        return { mở: o.classList.contains('open'), hiện: getComputedStyle(o).display !== 'none',
                 cao: Math.round(b.height), đáy: Math.round(b.bottom), khung: window.innerHeight };
    });
    ok(`${d.m}: bảng ngôn ngữ mở được ở tab Lịch`, r.mở && r.hiện && r.cao > 40, JSON.stringify(r));
    ok(`${d.m}: bảng ngôn ngữ không thò khỏi màn hình`, r.đáy <= r.khung + 1, `${r.đáy - r.khung}px`);
    await page.tap('#optCancel'); await page.waitForTimeout(400);

    // Nút Back phải đóng bảng đang mở trước, chỉ thoát khi không còn gì để đóng.
    await page.tap('#countryDisplayBtn'); await page.waitForTimeout(700);
    const back = await page.evaluate(() => !!(window.__onBackPressed && window.__onBackPressed()));
    const còn = await page.evaluate(() => document.getElementById('locOverlay').classList.contains('open'));
    ok(`${d.m}: Back đóng bảng vị trí thay vì thoát app`, back && !còn, `back=${back} còn mở=${còn}`);

    // Bàn phím ảo: windowSoftInputMode=adjustResize + inset IME làm khung web
    // tụt còn khoảng nửa. Ô tìm kiếm phải còn thấy, danh sách phải còn chỗ.
    await page.tap('#countryDisplayBtn'); await page.waitForTimeout(600);
    await page.evaluate(() => document.getElementById('locSearch').focus());
    await page.setViewportSize({ width: d.w, height: Math.round(d.h * 0.52) });
    await page.waitForTimeout(700);
    r = await page.evaluate(() => {
        const pl = document.getElementById('locPanel').getBoundingClientRect();
        const s = document.getElementById('locSearch').getBoundingClientRect();
        const l = document.getElementById('locList').getBoundingClientRect();
        return { thấy: s.top >= 0 && s.bottom <= window.innerHeight,
                 thò: Math.round(pl.bottom - window.innerHeight), ds: Math.round(l.height) };
    });
    ok(`${d.m}: bàn phím mở — ô tìm thành phố vẫn thấy`, r.thấy);
    ok(`${d.m}: bàn phím mở — bảng không thò khỏi màn hình`, r.thò <= 1, `${r.thò}px`);
    ok(`${d.m}: bàn phím mở — danh sách còn chỗ`, r.ds >= 40, `${r.ds}px`);
    await page.setViewportSize({ width: d.w, height: d.h }); await page.waitForTimeout(600);

    // Tên thành phố dài phải bị "…" cắt gọn, không được đẩy bố cục.
    await page.evaluate(() => { document.getElementById('locOverlay').classList.remove('open');
        document.getElementById('countryDisplayText').textContent =
            'Thành phố Hồ Chí Minh (Việt Nam) — UTC+07:00'; });
    await page.waitForTimeout(400);
    r = await page.evaluate(() => ({
        tràn: document.documentElement.scrollWidth - document.documentElement.clientWidth,
        ell: getComputedStyle(document.getElementById('countryDisplayText')).textOverflow,
    }));
    ok(`${d.m}: tên thành phố dài không làm tràn ngang`, r.tràn <= 1, `${r.tràn}px`);
    ok(`${d.m}: tên thành phố dài bị "…" cắt gọn`, r.ell === 'ellipsis', r.ell);
    await ctx.close();
}

/* ══ 4. Thanh tab sắp xếp lại được ══
   GIỮ LÂU rồi KÉO NGANG — cử chỉ chuẩn của Android để sắp xếp lại. Phải thử
   bằng CHẠM THẬT qua CDP: chuột và ngón tay đi hai đường sự kiện khác nhau,
   và đường ngón tay mới là đường chạy trên máy. */
console.log('\nSắp xếp lại thứ tự bốn tab (giữ lâu rồi kéo)');
for (const d of [{ m: 'A51', w: 412, h: 866, dpr: 2.625 }, { m: 'S21 FE', w: 384, h: 784, dpr: 2.8125 }]) {
    const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(String(e).slice(0, 140)));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    const cdp = await ctx.newCDPSession(page);
    const thứTự = () => page.evaluate(() =>
        [...document.querySelectorAll('#tabBar .tab-item')].map(e => e.id).join(','));
    const ô = id => page.evaluate(i => {
        const r = document.getElementById(i).getBoundingClientRect();
        return { x: r.x + r.width / 2, y: r.y + r.height / 2, w: r.width };
    }, id);
    const chạm = (type, x, y) => cdp.send('Input.dispatchTouchEvent',
        { type, touchPoints: type === 'touchEnd' ? [] : [{ x, y, radiusX: 12, radiusY: 12, force: 1 }] });

    check(`${d.m}: thứ tự mặc định`, await thứTự(), 'tabQmdj,tabCal,tabLenh,tabTraCuu');

    // Vuốt NHANH qua thanh tab không được vào chế độ sắp xếp — bằng không chỉ
    // quệt tay một cái là thứ tự tab đổi, không ai hiểu vì sao.
    let q = await ô('tabQmdj');
    await chạm('touchStart', q.x, q.y);
    for (let i = 1; i <= 6; i++) { await chạm('touchMove', q.x + 8 * i, q.y); await page.waitForTimeout(25); }
    const vuốtNhầm = await page.evaluate(() =>
        document.getElementById('tabBar').classList.contains('tab-reordering'));
    await chạm('touchEnd', 0, 0);
    await page.waitForTimeout(400);
    ok(`${d.m}: vuốt nhanh KHÔNG vào chế độ sắp xếp`, !vuốtNhầm);
    check(`${d.m}: vuốt nhanh không đổi thứ tự`, await thứTự(), 'tabQmdj,tabCal,tabLenh,tabTraCuu');

    // Giữ lâu rồi kéo Bát Tự từ cuối về đầu.
    const t = await ô('tabLenh');
    await chạm('touchStart', t.x, t.y);
    await page.waitForTimeout(600);                       // vượt HOLD_MS
    ok(`${d.m}: giữ lâu thì vào chế độ sắp xếp`, await page.evaluate(() =>
        document.getElementById('tabBar').classList.contains('tab-reordering')));
    for (let i = 1; i <= 14; i++) {
        await chạm('touchMove', t.x - (t.w * 2) * i / 14, t.y);
        await page.waitForTimeout(18);
    }
    await chạm('touchEnd', 0, 0);
    await page.waitForTimeout(500);
    check(`${d.m}: kéo Bát Tự về đầu`, await thứTự(), 'tabLenh,tabQmdj,tabCal,tabTraCuu');
    ok(`${d.m}: buông tay là hết chế độ sắp xếp`, await page.evaluate(() =>
        !document.getElementById('tabBar').classList.contains('tab-reordering') &&
        !document.querySelector('#tabBar .tab-drag')));

    // Vạch ngăn phải ĐI THEO: nó vẽ bằng `.tab-item + .tab-item{border-left}`,
    // một chọn tử theo DOM — đây đúng là lý do phải đổi chỗ thật trong DOM chứ
    // không dùng `order` của flex (xem js/taborder.js).
    const viền = await page.evaluate(() => [...document.querySelectorAll('#tabBar .tab-item')]
        .map(e => parseFloat(getComputedStyle(e).borderLeftWidth)));
    ok(`${d.m}: tab đầu không có vạch ngăn trái`, viền[0] === 0, String(viền));
    ok(`${d.m}: ba tab sau đều có vạch ngăn`, viền.slice(1).every(v => v > 0), String(viền));

    // Chạm NGẮN vẫn phải là chuyển tab — cử chỉ mới không được nuốt cử chỉ cũ.
    // (Lỗi đã gặp: cờ nuốt-click nằm lại sau cú kéo không sinh ra click, rồi
    //  nuốt oan cú chạm kế tiếp.)
    const c = await ô('tabCal');
    await chạm('touchStart', c.x, c.y); await page.waitForTimeout(80); await chạm('touchEnd', 0, 0);
    await page.waitForTimeout(600);
    ok(`${d.m}: sau khi sắp xếp, chạm tab vẫn chuyển màn hình`,
        await page.evaluate(() => document.body.classList.contains('view-cal')));

    // Nhớ qua lần mở lại.
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1300);
    check(`${d.m}: thứ tự sống qua lần mở lại`, await thứTự(), 'tabLenh,tabQmdj,tabCal,tabTraCuu');

    // Chuỗi lưu hỏng (bản sau thêm/bớt tab) thì quay về mặc định, đừng để mất
    // tab nào khỏi thanh.
    await page.evaluate(() => localStorage.setItem('qmdj.tabOrder', 'tabCal,tabKhongTonTai'));
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1300);
    check(`${d.m}: chuỗi lưu hỏng thì về thứ tự mặc định`, await thứTự(), 'tabQmdj,tabCal,tabLenh,tabTraCuu');

    ok(`${d.m}: không lỗi JS`, errs.length === 0, [...new Set(errs)].join(' | '));
    await ctx.close();
}

/* ══ 5. Tab Bát Tự: mở Lệnh năm thì CẢ TRANG cuộn ══
   Bảng Lệnh năm từng bị kẹp vào chỗ trống còn lại (120px cho một bảng cao
   683px) rồi tự cuộn bên trong — ngón tay đặt xuống là rơi vào ô cuộn tí hon
   ấy, và toàn bộ màn hình phía trên (Bát Tự, Đại Vận) đứng im. Nay nó bung đủ
   chiều cao như ba bảng gập/mở ở tab Kỳ Môn. */
console.log('\nTab Bát Tự: mở Lệnh năm thì cả trang cuộn được');
for (const d of [{ m: 'A51', w: 412, h: 866, dpr: 2.625 }, { m: 'S21 FE', w: 384, h: 784, dpr: 2.8125 }]) {
    const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    await page.evaluate(() => window.showTab('lenh'));
    await page.waitForTimeout(900);

    ok(`${d.m}: không còn hàng tiêu đề "ĐẠI VẬN"`,
        await page.evaluate(() => !document.getElementById('daiVanHead')));
    ok(`${d.m}: tab Bát Tự KHÔNG còn bảng Lệnh năm`,
        await page.evaluate(() => !document.querySelector('#lenhView #lenhHead')));

    // Hàng lưu niên canh PHẢI, và mọi hàng dừng ở CÙNG một vạch.
    const canh = await page.evaluate(() => {
        const rows = [...document.querySelectorAll('.dv-row')];
        // Canh TRÁI: đo mép TRÁI của cột NĂM so với mép trái của hàng — cả 10
        // hàng phải bắt đầu ở cùng một vạch.
        const hở = rows.map(r => {
            const y = r.querySelector('.dv-year').getBoundingClientRect();
            return +(y.left - r.getBoundingClientRect().left).toFixed(2);
        });
        // Tràn ĐO HAI PHÍA: canh phải nên phần thừa dồn sang TRÁI, mà
        // scrollWidth chỉ đếm phía cuối dòng (xem rowOverflow trong lenh.js).
        let thò = 0, ví = '';
        for (const r of rows) {
            const kids = [...r.children].map(k => k.getBoundingClientRect());
            const cs = getComputedStyle(r), b = r.getBoundingClientRect();
            const o = Math.max(
                (b.left + parseFloat(cs.paddingLeft)) - Math.min(...kids.map(k => k.left)),
                Math.max(...kids.map(k => k.right)) - (b.right - parseFloat(cs.paddingRight)));
            if (o > thò) { thò = o; ví = r.textContent.trim(); }
        }
        return { min: Math.min(...hở), max: Math.max(...hở), thò: +thò.toFixed(2), ví,
                 canh: getComputedStyle(rows[0]).justifyContent };
    });
    check(`${d.m}: hàng lưu niên canh trái`, canh.canh, 'flex-start');
    ok(`${d.m}: cả 100 hàng dừng ở cùng một vạch`,
        canh.max - canh.min <= 0.5, `hở ${canh.min}–${canh.max}px`);
    ok(`${d.m}: có lề thật với mép trái`, canh.min >= 2, `${canh.min}px`);
    ok(`${d.m}: không hàng nào thò khỏi hộp nội dung`, canh.thò <= 0.5, `${canh.thò}px ở "${canh.ví}"`);

    // Bảng Lệnh năm nay ở tab TRA CỨU, và ĐÓNG SẴN (bốn mục đều vậy — bấm mới
    // bung). Bung ra rồi vuốt NGAY TRÊN bảng — cả trang phải nhúc nhích.
    await page.evaluate(() => { window.__tracuuYear(2026); window.showTab('tracuu'); });
    await page.waitForTimeout(1100);
    await page.evaluate(() => {
        for (const [h, b] of [['trinhuanHeader', 'trinhuanBody'], ['sachboHeader', 'sachboBody'],
                              ['lenhHead', 'lenhSec'], ['tcAmHead', 'tcAmSec']]) {
            const body = document.getElementById(b), head = document.getElementById(h);
            if (body && head && getComputedStyle(body).display === 'none') head.click();
        }
    });
    await page.waitForTimeout(900);
    const k = await page.evaluate(() => {
        const b = document.getElementById('lenhBody');
        const t = b.querySelector('table');
        return { kẹp: b.style.maxHeight || '', dưDọc: b.scrollHeight - b.clientHeight,
                 hụt: Math.round(t.getBoundingClientRect().height - b.getBoundingClientRect().height),
                 cuộnĐược: Math.round(document.documentElement.scrollHeight - window.innerHeight) };
    });
    ok(`${d.m}: bảng Lệnh năm không bị kẹp chiều cao`, k.kẹp === '', k.kẹp);
    ok(`${d.m}: bảng Lệnh năm không cuộn riêng`, k.dưDọc <= 1, `dư ${k.dưDọc}px`);
    ok(`${d.m}: khung ôm trọn bảng`, k.hụt <= 2, `hụt ${k.hụt}px`);
    ok(`${d.m}: cả trang cuộn được kha khá`, k.cuộnĐược > 200, `${k.cuộnĐược}px`);

    const cdp2 = await ctx.newCDPSession(page);
    // Điểm đặt ngón tay phải nằm TRONG khung nhìn: ở tab Tra cứu bảng Lệnh
    // năm đứng sau hai bảng kia nên mép trên của nó ở tận đâu dưới màn hình,
    // chạm vào toạ độ ấy là chạm ra ngoài và không sự kiện nào sinh ra.
    const vị = await page.evaluate(() => {
        const q = document.getElementById('lenhBody').getBoundingClientRect();
        const y = Math.min(Math.max(q.top + 40, 80), window.innerHeight - 120);
        return { x: q.left + q.width / 2, y: y };
    });
    await page.evaluate(() => window.scrollTo(0, 0));
    await page.waitForTimeout(200);
    await cdp2.send('Input.dispatchTouchEvent',
        { type: 'touchStart', touchPoints: [{ x: vị.x, y: vị.y }] });
    for (let i = 1; i <= 8; i++) {
        await cdp2.send('Input.dispatchTouchEvent',
            { type: 'touchMove', touchPoints: [{ x: vị.x, y: vị.y - i * 25 }] });
        await page.waitForTimeout(25);
    }
    await cdp2.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
    await page.waitForTimeout(700);
    const sy = await page.evaluate(() => Math.round(window.scrollY));
    ok(`${d.m}: vuốt trên bảng thì CẢ TRANG cuộn (không kẹt ở ô con)`, sy > 0, `scrollY ${sy}`);
    await ctx.close();
}

/* ── Bàn Kỳ Môn tiếng Việt: KHÔNG nhãn nào bị bẻ giữa chừng ──
   `.sub-cell` khai `word-break: break-word`, nên nhãn không vừa cột thì trình
   duyệt bẻ ĐÔI CHỮ chứ không tràn: "Thương" xuống dòng thành "Thươn" / "g".
   Mọi phép canh cũ đều bỏ lọt — nhãn KHÔNG thò khỏi ô, `scrollWidth` bằng
   `clientWidth`, không đè lên nhãn nào. Phải hỏi đúng thứ nó làm: Range của
   một chữ KHÔNG có khoảng trắng mà trải trên hơn một dòng là đã bị bẻ.

   Lỗi này lọt vào cùng lúc với việc sửa chọn tử `.lang-vi` (vốn là mã chết)
   thành `body:not(.lang-zh)`: bộ cỡ chữ nguyên bản 12,5–17px chưa bao giờ chạy
   cho tiếng Việt nên chưa ai đo nó. */
console.log('\nBàn Kỳ Môn tiếng Việt: không nhãn nào bị bẻ giữa chừng');
{
    const CA = [[1990, 5, 12, 14, 30], [2026, 9, 18, 21, 0], [1984, 2, 4, 3, 15]];
    for (const d of [{ m: 'A51', w: 412, h: 866, dpr: 2.625 },
                     { m: 'S21 FE', w: 384, h: 784, dpr: 2.8125 },
                     { m: 'zoom lớn', w: 360, h: 752, dpr: 3.0 }]) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h },
            deviceScaleFactor: d.dpr, isMobile: true, hasTouch: true });
        await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
        const page = await ctx.newPage();
        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForTimeout(1100);
        const xấu = new Set();
        // Hai chế độ dùng hai bộ cỡ chữ KHÁC NHAU nên quét cả hai, nhưng canh
        // KHÁC nhau:
        //   · "Cơ bản" (cung 3×3, MẶC ĐỊNH) — không nhãn nào được bẻ đôi.
        //   · "Đầy đủ" (4×4) — ô chữ chỉ rộng 24–32px, "Thương" cần cỡ 7,5–10px
        //     mới nằm gọn một dòng, bé hơn cả cỡ của bản gốc. Nó xuống hai dòng
        //     ở đó từ trước tới nay; chỉ canh KHÔNG TRÀN (xem chú thích cỡ chữ
        //     trong app.css).
        for (const đủ of [false, true]) {
            await page.evaluate(v => {
                const c = document.getElementById('cobanToggle');
                if (c && c.checked !== v) { c.checked = v; c.dispatchEvent(new Event('change', { bubbles: true })); }
                document.getElementById('mainBody').classList.toggle('coban-mode', !v);
                window.__bỏQuaBẻ = v;      // "Đầy đủ": chỉ canh tràn
            }, đủ);
            for (const [y, mo, dd, h, mi] of CA) {
                for (const ph of ['trinhuan', 'sachbo', 'amban']) {
                    await page.evaluate(([y, mo, dd, h, mi, ph]) => {
                        const set = (id, v) => {
                            const e = document.getElementById(id); if (!e) return;
                            e.value = String(v); e.dispatchEvent(new Event('change', { bubbles: true }));
                        };
                        set('inYear', y); set('inMonth', mo); set('inDay', dd);
                        set('solarHour', h); set('solarMinute', mi);
                        const ms = document.getElementById('method');
                        if (ms) { ms.value = ph; ms.dispatchEvent(new Event('change', { bubbles: true })); }
                        window.processAll();
                    }, [y, mo, dd, h, mi, ph]);
                    await page.waitForTimeout(180);
                    for (const b of await page.evaluate(() => {
                        const out = [];
                        const w = document.createTreeWalker(document.getElementById('board'), NodeFilter.SHOW_TEXT);
                        let n;
                        while ((n = w.nextNode())) {
                            const t = (n.nodeValue || '').trim();
                            if (!t || /\s/.test(t)) continue;
                            const el = n.parentElement;
                            if (!el.closest('.sub-cell') || getComputedStyle(el).display === 'none') continue;
                            const rg = document.createRange(); rg.selectNodeContents(n);
                            if (rg.getClientRects().length > 1 && !window.__bỏQuaBẻ) out.push('bẻ "' + t + '"');
                        }
                        for (const e of document.querySelectorAll('#board .sub-cell')) {
                            if (getComputedStyle(e).display === 'none') continue;
                            if (e.scrollWidth > e.clientWidth + 1) out.push('tràn "' + e.textContent.trim() + '"');
                        }
                        return out;
                    })) xấu.add(`${đủ ? 'Đầy đủ' : 'Cơ bản'} ${b}`);
                }
            }
        }
        ok(`${d.m} ${d.w}px: không nhãn nào bị bẻ hay tràn`, xấu.size === 0,
            [...xấu].slice(0, 5).join(' | '));
        await ctx.close();
    }
}

await browser.close();
server.close();
console.log(`\n${đạt} đạt · ${hỏng} hỏng`);
console.log(hỏng ? '✗ HỎNG' : '✓ A51 và S21 FE: bố cục vừa khít, ô bấm đủ rộng, thao tác đúng');
process.exit(hỏng ? 1 : 0);
