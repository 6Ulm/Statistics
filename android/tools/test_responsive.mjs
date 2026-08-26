/**
 * Kiểm thử bố cục trên nhiều kích thước màn hình thật, bằng Chromium
 * (WebView của Android cũng là Chromium nên kết quả sát thực tế).
 *
 *   cd android/tools && npm install && npx playwright install chromium
 *   node test_responsive.mjs
 *
 * Bắt ba lỗi bố cục:
 *   1. tràn ngang  — phải cuộn sang phải mới đọc hết
 *   2. chữ bị cắt  — nội dung dài hơn ô chứa, bị "…" nuốt mất
 *   3. phóng sai   — luật `zoom: 1.25` của bản gốc chỉ nhìn chiều rộng, nên
 *                    điện thoại xoay ngang (800×360) bị phóng to đến mức bàn
 *                    Kỳ Môn cao 500px trên màn hình chỉ cao 360px
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

// Chiều cao ở đây là chiều cao WebView thật (đã trừ thanh trạng thái và thanh
// điều hướng), không phải chiều cao màn hình — vỏ Android chừa lề theo
// window insets nên trang chỉ nhận được phần còn lại.
const DEVICES = [
    { name: 'Galaxy S21',          w: 360, h: 740, dpr: 3 },
    { name: 'Galaxy S21 FE',       w: 393, h: 790, dpr: 2.75 },
    { name: 'Galaxy A51',          w: 412, h: 852, dpr: 2.625 },
    { name: 'Galaxy S21 Ultra',    w: 384, h: 794, dpr: 3.5 },
    { name: 'Galaxy S21 ngang',    w: 800, h: 330, dpr: 3 },
    { name: 'điện thoại nhỏ',      w: 320, h: 520, dpr: 2 },
    { name: 'Z Fold (mở)',         w: 673, h: 800, dpr: 2.6 },
];

let fail = 0;
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

for (const d of DEVICES) {
    const ctx = await browser.newContext({
        viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr,
        isMobile: true, hasTouch: true,
    });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);
    if (process.env.TAB === 'cal') {
        await page.evaluate(() => {
            window.showTab('cal');
            // Nút Ghim chỉ hiện khi chạy trong ứng dụng Android. Đo mà thiếu nó
            // thì bố cục trên máy thật cao hơn phép thử tưởng, và nút bị thanh
            // tab cố định che mất — đúng lỗi đã lọt ra máy thật một lần.
            document.getElementById('calPinBtn').style.display = 'block';
        });
        await page.waitForTimeout(200);
        // Vẽ lại: chỉ gọi __fitScreen thì viewport.js co giãn cả trang, nhưng
        // fitGrid() — thứ chia chiều cao cho lưới và bảng — chỉ chạy trong
        // render(). Trên máy thật nút Ghim đã hiện sẵn từ lần vẽ đầu.
        await page.evaluate(() => window.showTab('cal'));
        await page.waitForTimeout(600);
    }

    const r = await page.evaluate(() => {
        // Thanh tab cố định nổi trên mọi thứ; phần tử nào thò xuống dưới mép
        // trên của nó là bị che, người dùng không bấm được.
        const hidden = [];
        const bar = document.getElementById('tabBar');
        if (bar) {
            const top = bar.getBoundingClientRect().top;
            // KHÔNG quét <body> (id mainBody): hộp của nó vốn kéo tới đáy màn
            // hình vì có padding-bottom bằng chiều cao thanh tab — luôn "thò
            // xuống", mà đó chính là cách chừa chỗ cho thanh tab.
            for (const el of document.querySelectorAll('#calView > *, #mainBody > *:not(#tabBar)')) {
                const cs = getComputedStyle(el);
                if (cs.display === 'none' || cs.visibility === 'hidden') continue;
                const b = el.getBoundingClientRect();
                if (b.height > 0 && b.bottom > top + 1 && b.top < top) {
                    hidden.push((el.id || el.className) + ' thò xuống ' +
                        Math.round(b.bottom - top) + 'px');
                }
            }
        }
        const trunc = [];
        for (const el of document.querySelectorAll('body *')) {
            const cs = getComputedStyle(el);
            if (cs.display === 'none' || cs.visibility === 'hidden') continue;
            if (el.children.length === 0 && el.scrollWidth > el.clientWidth + 1) {
                const t = (el.textContent || '').trim();
                if (t) trunc.push(t.slice(0, 30));
            }
        }
        const board = document.getElementById('board');
        // Phải đọc zoom ĐÃ TÍNH: luật `zoom: 1.25` của bản gốc nằm trong CSS,
        // đọc style nội tuyến sẽ luôn thấy rỗng và bỏ lọt lỗi.
        const zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        return {
            docW: document.documentElement.scrollWidth,
            innerW: window.innerWidth,
            innerH: window.innerHeight,
            contentH: Math.round(document.body.getBoundingClientRect().height),
            zoom,
            boardH: board ? Math.round(board.getBoundingClientRect().height) : 0,
            trunc,
            hidden,
        };
    });

    const problems = [];
    if (r.docW > r.innerW + 1) problems.push(`tràn ngang ${r.docW - r.innerW}px`);
    if (r.trunc.length) problems.push(`${r.trunc.length} chỗ chữ bị cắt: ${r.trunc.slice(0, 3).join(' | ')}`);
    // Trang phải cuộn thì phần dưới nằm sau thanh tab là bình thường — cuộn tới
    // là thấy (body đã chừa padding-bottom đúng bằng chiều cao thanh tab).
    // Chỉ là lỗi khi trang VỪA màn hình mà vẫn có thứ bị che.
    if (r.hidden.length && r.contentH <= r.innerH + 1) {
        problems.push(`bị thanh tab che: ${r.hidden.join(' | ')}`);
    }
    // Tab Lịch trên điện thoại dựng đứng thì PHẢI vừa một màn hình — fitGrid()
    // chia chiều cao chính là để thế. Tràn ra là dấu hiệu nó quên trừ một khối
    // nào đó (nút Ghim từng bị quên đúng như vậy), và viewport.js sẽ che lỗi
    // bằng cách thu nhỏ cả trang tới đáy 0,95.
    // Ngưỡng 640px: dưới mức đó thì 6 hàng × ROW_MIN (58px) cộng đầu lịch, bảng
    // tiết khí và thanh tab đã vượt màn hình rồi — máy 320×520 buộc phải cuộn,
    // không phải lỗi chia chiều cao.
    if (process.env.TAB === 'cal' && d.h > d.w && d.h >= 640 && r.contentH > r.innerH + 1) {
        problems.push(`tab Lịch tràn dọc ${r.contentH - r.innerH}px (phải vừa một màn hình)`);
    }
    // Luật cốt lõi: nội dung đã cao quá màn hình thì TUYỆT ĐỐI không được phóng
    // to thêm. Bản gốc phóng 1,25 lần chỉ vì màn rộng ≥768px, nên điện thoại
    // xoay ngang bị phóng trong khi đã phải cuộn dọc.
    if (r.contentH > r.innerH && r.zoom > 1.001) {
        problems.push(`đã phải cuộn (${r.contentH}px > ${r.innerH}px) mà còn phóng ×${r.zoom}`);
    }

    if (problems.length) { fail++; console.log(`  FAIL ${d.name.padEnd(18)} ${problems.join('; ')}`); }
    else console.log(`  ok   ${d.name.padEnd(18)} zoom ${r.zoom.toFixed(3)}  bàn ${r.boardH}px  ` +
                     `nội dung ${r.contentH}/${r.innerH}px  không tràn, không cắt chữ`);

    await ctx.close();
}

await browser.close();
server.close();
console.log(fail ? `\n✗ ${fail}/${DEVICES.length} kích thước có lỗi bố cục` : `\n✓ ${DEVICES.length}/${DEVICES.length} kích thước đạt`);
process.exit(fail ? 1 : 0);
