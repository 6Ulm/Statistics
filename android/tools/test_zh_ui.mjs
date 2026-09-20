/**
 * QUÉT TIẾNG TRUNG TRÊN CẢ BỐN TAB — tràn, cắt chữ, sát mép, chồng nhau.
 *
 *   node test_zh_ui.mjs
 *
 * Vì sao phải có riêng một phép thử cho tiếng Trung: chữ vuông và chữ Latin
 * hỏng theo hai kiểu khác nhau, nên một bộ số vừa vặn ở tiếng Việt không nói
 * gì về tiếng Trung.
 *
 *   · chữ Hán KHÔNG có chỗ ngắt từ — trình duyệt bẻ được giữa hai chữ bất kỳ,
 *     nên nhãn không tràn mà lặng lẽ xuống dòng, đẩy cả khối cao thêm;
 *   · ngược lại một chữ Hán rộng bằng 2 chữ Latin, nên nhãn ngắn hơn về SỐ
 *     CHỮ lại có thể rộng hơn về pixel ("值使" vs "Trực Sử" thì ngắn hơn,
 *     nhưng "正午时间:" vs "Chính Ngọ:" thì dài hơn);
 *   · phông CJK có vùng mực cao hơn phông Latin cùng cỡ, nên hàng nào canh
 *     sát chiều cao cũng phải đo lại.
 *
 * Quét MỌI tab ở MỌI trạng thái đáng kể (Kỳ Môn hai chế độ, Lịch, Bát Tự,
 * Tra cứu bốn mục đóng/mở), trên ba máy đích, và canh năm điều:
 *
 *   1. trang không cuộn ngang;
 *   2. không phần tử nào có chữ rộng hơn ô của nó (scrollWidth > clientWidth);
 *   3. không chữ nào bị bẻ giữa chừng ở chỗ KHÔNG ĐƯỢC PHÉP xuống dòng;
 *   4. không phần tử nào thò ra ngoài hoặc dính sát mép màn hình;
 *   5. không hai vùng chạm nào chồng lên nhau.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

/** Lề tối thiểu từ mép màn hình — hẹp hơn thì chữ trông như dính vào viền. */
const LỀ_TỐI_THIỂU = 2;
/** Tràn ô bao nhiêu pixel thì coi là lỗi (làm tròn nửa pixel của trình duyệt). */
const TRÀN_CHO_PHÉP = 1;

const MÁY = [
    { tên: 'S21', w: 360, h: 740, dpr: 3 },
    { tên: 'S21 FE', w: 393, h: 790, dpr: 2.75 },
    { tên: 'A51', w: 412, h: 852, dpr: 2.625 },
];

let pass = 0, fail = 0;
const ok = (ten, đúng, thêm) => {
    if (đúng) { pass++; }
    else { fail++; console.log('  ✗    ' + ten + (thêm ? '\n         ' + thêm : '')); }
};

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((q, r) => {
    const rel = decodeURIComponent(q.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const f = path.join(WEB, rel);
    if (!f.startsWith(WEB) || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
    r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
    fs.createReadStream(f).pipe(r);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;

const { chromium } = await import('playwright');
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

/** Phép quét chạy TRONG trang: trả về mọi vi phạm tìm được. */
const QUÉT = (LỀ, TRÀN) => {
    const vw = document.documentElement.clientWidth;
    const out = { tràn: [], bẻ: [], mép: [], chồng: [], cuộnNgang: 0 };
    const mô = el => {
        const t = (el.textContent || '').trim().replace(/\s+/g, ' ').slice(0, 24);
        return (el.id ? '#' + el.id : el.className ? '.' + String(el.className).split(' ')[0] : el.tagName)
            + (t ? ' "' + t + '"' : '');
    };
    const hiện = el => {
        const cs = getComputedStyle(el);
        if (cs.display === 'none' || cs.visibility === 'hidden' || cs.opacity === '0') return false;
        const b = el.getBoundingClientRect();
        return b.width > 0 && b.height > 0;
    };

    out.cuộnNgang = document.documentElement.scrollWidth - vw;

    for (const el of document.querySelectorAll('body *')) {
        if (!hiện(el)) continue;
        const cs = getComputedStyle(el);
        // 1. Chữ rộng hơn ô. Bỏ qua ô CỐ Ý cuộn (overflow auto/scroll).
        const tựCuộn = /auto|scroll/.test(cs.overflowX);
        if (!tựCuộn && el.scrollWidth - el.clientWidth > TRÀN && !el.children.length) {
            out.tràn.push(mô(el) + ` (+${(el.scrollWidth - el.clientWidth).toFixed(1)}px)`);
        }
        // 2. Sát / thò khỏi mép màn hình. Chỉ xét phần tử có CHỮ, và chỉ phần
        //    tử lá — khối cha rộng bằng màn hình là chuyện bình thường.
        if (!el.children.length && (el.textContent || '').trim()) {
            const b = el.getBoundingClientRect();
            if (b.width > 0 && (b.left < LỀ || b.right > vw - LỀ)) {
                out.mép.push(mô(el) + ` [${b.left.toFixed(1)}, ${b.right.toFixed(1)}] / ${vw}`);
            }
        }
    }

    // 3. Chữ bị BẺ giữa chừng ở chỗ khai `white-space: nowrap` (khai rồi mà
    //    vẫn bẻ nghĩa là bẻ bằng vũ lực) — và ở mọi nhãn của bàn Kỳ Môn.
    const w = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let n;
    while ((n = w.nextNode())) {
        const s = n.textContent;
        if (!s || !s.trim() || /\s/.test(s.trim())) continue;   // có dấu cách thì xuống dòng là hợp lệ
        const el = n.parentElement;
        if (!el || !hiện(el)) continue;
        const rg = document.createRange();
        rg.selectNodeContents(n);
        if (rg.getClientRects().length > 1) out.bẻ.push(mô(el));
    }

    // 4. Hai vùng CHẠM chồng lên nhau — bấm vào một thứ mà trúng thứ khác.
    //
    // KHÔNG tính cặp mà một bên nằm trong thanh CỐ ĐỊNH dưới đáy: nội dung
    // trang cuộn qua dưới thanh ấy là chuyện bình thường của mọi trang web,
    // thanh có nền đục nên nó che chứ không chồng. Chỉ hai thứ CÙNG cuộn (hoặc
    // cùng cố định) mà đè nhau mới là lỗi thật.
    const dock = document.getElementById('bottomDock');
    const trongDock = el => !!(dock && dock.contains(el));
    const chạm = [...document.querySelectorAll(
        '.tab-item, .picker-btn, #calPinBtn, .dp-header, #lenhHead, #tcAmHead, .opt-row')]
        .filter(hiện).map(el => ({ el, b: el.getBoundingClientRect(), dock: trongDock(el) }));
    for (let i = 0; i < chạm.length; i++) {
        for (let j = i + 1; j < chạm.length; j++) {
            if (chạm[i].dock !== chạm[j].dock) continue;
            const a = chạm[i].b, c = chạm[j].b;
            const w0 = Math.min(a.right, c.right) - Math.max(a.left, c.left);
            const h0 = Math.min(a.bottom, c.bottom) - Math.max(a.top, c.top);
            if (w0 > 1 && h0 > 1) out.chồng.push(mô(chạm[i].el) + ' ✗ ' + mô(chạm[j].el));
        }
    }
    // 5. Chữ VIỆT còn sót khi đang ở tiếng Trung — nhãn nào quên đường dịch
    //    thì đứng nguyên tiếng Việt, và không phép canh nào về bố cục bắt
    //    được (nó vẫn vừa ô, vẫn không tràn, chỉ là sai thứ tiếng).
    const VN = /[ăâđêôơưàáảãạằắẳẵặầấẩẫậèéẻẽẹềếểễệìíỉĩịòóỏõọồốổỗộờớởỡợùúủũụừứửữựỳýỷỹỵĂÂĐÊÔƠƯÀÁẢÃẠẰẮẲẴẶẦẤẨẪẬÈÉẺẼẸỀẾỂỄỆÌÍỈĨỊÒÓỎÕỌỒỐỔỖỘỜỚỞỠỢÙÚỦŨỤỪỨỬỮỰỲÝỶỸỴ]/;
    out.chữViệt = [];
    const w2 = document.createTreeWalker(document.body, NodeFilter.SHOW_TEXT);
    let n2;
    while ((n2 = w2.nextNode())) {
        const s = (n2.textContent || '').trim();
        if (!s || !VN.test(s)) continue;
        const el = n2.parentElement;
        if (!el || !hiện(el)) continue;
        out.chữViệt.push(mô(el));
    }
    out.chữViệt = [...new Set(out.chữViệt)];
    return out;
};

for (const máy of MÁY) {
    console.log(`\n── ${máy.tên} ${máy.w}×${máy.h} @${máy.dpr}x · TIẾNG TRUNG ──`);
    const ctx = await browser.newContext({
        viewport: { width: máy.w, height: máy.h }, deviceScaleFactor: máy.dpr,
        isMobile: true, hasTouch: true,
    });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1400);
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(700);
    // Nút Ghim chỉ hiện trong ứng dụng Android — bật tay để nó cũng được soi.
    await page.evaluate(() => {
        const p = document.getElementById('calPinBtn');
        if (p) p.style.display = 'block';
    });
    // Một lá số có đủ thứ để hiện: ba tàng can, thần sát, đại vận.
    await page.evaluate(() => {
        const set = (id, v) => {
            const el = document.getElementById(id);
            el.value = String(v);
            el.dispatchEvent(new Event('change', { bubbles: true }));
        };
        set('inYear', 1991); set('inMonth', 7); set('inDay', 16);
        set('solarHour', 16); set('solarMinute', 0);
        window.processAll();
    });
    await page.waitForTimeout(700);

    const CẢNH = [
        ['tab Kỳ Môn · Đầy đủ', async () => {
            await page.evaluate(() => window.showTab('qmdj'));
            await page.evaluate(() => document.getElementById('mainBody').classList.remove('coban-mode'));
        }],
        ['tab Kỳ Môn · Cơ bản', async () => {
            await page.evaluate(() => document.getElementById('mainBody').classList.add('coban-mode'));
        }],
        ['tab Lịch', async () => {
            await page.evaluate(() => document.getElementById('mainBody').classList.remove('coban-mode'));
            await page.evaluate(() => window.showTab('cal'));
        }],
        ['tab Bát Tự', async () => { await page.evaluate(() => window.showTab('lenh')); }],
        ['tab Tra cứu · đóng', async () => { await page.evaluate(() => window.showTab('tracuu')); }],
        ['tab Tra cứu · mở hết', async () => {
            await page.evaluate(() => window.__tracuuYear(1991));
            await page.waitForTimeout(400);
            await page.evaluate(() => {
                for (const h of document.querySelectorAll('#traCuuView .dp-header, #lenhHead, #tcAmHead')) {
                    h.click();
                }
            });
        }],
    ];

    for (const [tên, vào] of CẢNH) {
        await vào();
        await page.waitForTimeout(700);
        const r = await page.evaluate(
            ([L, T]) => (new Function('LỀ', 'TRÀN', 'return (' + window.__quét + ')(LỀ, TRÀN)'))(L, T),
            [LỀ_TỐI_THIỂU, TRÀN_CHO_PHÉP]).catch(() => null);
        const v = r || await page.evaluate(
            ({ src, L, T }) => (new Function('return ' + src)())(L, T),
            { src: QUÉT.toString(), L: LỀ_TỐI_THIỂU, T: TRÀN_CHO_PHÉP });
        const nhãn = `${máy.tên} · ${tên}`;
        ok(`${nhãn}: trang không cuộn ngang`, v.cuộnNgang <= 1, `dôi ${v.cuộnNgang}px`);
        ok(`${nhãn}: không ô nào bị tràn chữ`, v.tràn.length === 0, v.tràn.slice(0, 4).join(' | '));
        ok(`${nhãn}: không chữ nào bị bẻ giữa chừng`, v.bẻ.length === 0, v.bẻ.slice(0, 4).join(' | '));
        ok(`${nhãn}: không chữ nào sát/thò khỏi mép màn hình`, v.mép.length === 0, v.mép.slice(0, 4).join(' | '));
        ok(`${nhãn}: không hai vùng chạm nào chồng nhau`, v.chồng.length === 0, v.chồng.slice(0, 3).join(' | '));
        ok(`${nhãn}: không còn chữ Việt nào trên màn`, v.chữViệt.length === 0,
            v.chữViệt.slice(0, 6).join(' | '));
        console.log(`  ${tên.padEnd(24)} tràn ${v.tràn.length} · bẻ ${v.bẻ.length}`
            + ` · sát mép ${v.mép.length} · chồng ${v.chồng.length}`
            + ` · chữ Việt ${v.chữViệt.length} · ngang ${v.cuộnNgang}px`);
    }
    ok(`${máy.tên}: không lỗi JS`, errs.length === 0, errs.join('; '));
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
if (fail) process.exit(1);
console.log('✓ Tiếng Trung: bốn tab, ba máy — không tràn, không cắt chữ, không sát mép, không chồng');
