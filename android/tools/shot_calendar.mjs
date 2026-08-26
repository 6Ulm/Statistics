/**
 * Chụp màn hình tab Lịch trên nhiều điện thoại để xem trước mà không cần build.
 *
 *   node shot_calendar.mjs
 *
 * Ảnh ra thư mục ../../shots/.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const OUT = process.env.OUT || path.join(HERE, '..', '..', 'shots');
fs.mkdirSync(OUT, { recursive: true });

const { chromium } = await import('playwright');

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

const DEVICES = [
    { name: 'S21',    w: 360, h: 740, dpr: 3 },
    { name: 'S21FE',  w: 393, h: 790, dpr: 2.75 },
    { name: 'A51',    w: 412, h: 852, dpr: 2.625 },
];

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

for (const d of DEVICES) {
    const ctx = await browser.newContext({
        viewport: { width: d.w, height: d.h }, deviceScaleFactor: d.dpr,
        isMobile: true, hasTouch: true,
    });
    // Ứng dụng mặc định tiếng Trung; đặt sẵn tuỳ chọn để chụp bản tiếng Việt.
    await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);
    await page.evaluate(() => window.showTab('cal'));
    await page.waitForTimeout(700);

    // 1. tiết khí đang MỞ (mặc định)
    await page.screenshot({ path: path.join(OUT, `cal-${d.name}-mo.png`) });

    // 2. gập bảng tiết khí lại
    await page.click('#calJqHead');
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUT, `cal-${d.name}-gap.png`) });

    // 3. mở lại + sang tháng sau (kiểm tra ô "hôm nay" biến mất đúng)
    await page.click('#calJqHead');
    await page.waitForTimeout(300);
    await page.click('#calNext');
    await page.waitForTimeout(500);
    await page.screenshot({ path: path.join(OUT, `cal-${d.name}-thangsau.png`) });

    // 4. tab Kỳ Môn để đối chiếu
    await page.evaluate(() => window.showTab('qmdj'));
    await page.waitForTimeout(700);
    await page.screenshot({ path: path.join(OUT, `kmdj-${d.name}.png`) });

    await ctx.close();
    console.log(`✓ ${d.name} (${d.w}×${d.h} @${d.dpr}x)`);
}
await browser.close();
server.close();
console.log('→ ' + OUT);
