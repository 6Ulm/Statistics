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
    ['.cal-sec-head',     'cal'],
    ['#lenhGenderBtn',    'lenh'], ['#lenhRuleBtn',      'lenh'],
    ['#lenhHead',         'lenh'],
];

let đạt = 0, hỏng = 0;
const ok = (tên, điều, ghi = '') => {
    if (điều) { đạt++; console.log(`  ok   ${tên}`); }
    else { hỏng++; console.log(`  ✗ SAI ${tên}   ${ghi}`); }
};

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

        for (const tab of ['qmdj', 'cal', 'lenh']) {
            await page.evaluate(t => {
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

await browser.close();
server.close();
console.log(`\n${đạt} đạt · ${hỏng} hỏng`);
console.log(hỏng ? '✗ HỎNG' : '✓ A51 và S21 FE: bố cục vừa khít, ô bấm đủ rộng, thao tác đúng');
process.exit(hỏng ? 1 : 0);
