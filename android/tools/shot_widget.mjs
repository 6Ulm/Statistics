/**
 * Chụp bản xem trước widget (widget_preview.html) để kiểm tra bố cục mà không
 * cần build APK.
 *
 *   node shot_widget.mjs
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');
const OUT = process.env.OUT || path.join(HERE, '..', '..', 'shots');
fs.mkdirSync(OUT, { recursive: true });

const { chromium } = await import('playwright');

const MIME = { '.html': 'text/html', '.txt': 'text/plain', '.js': 'text/javascript' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'widget_preview.html';
    const file = rel.startsWith('assets/')
        ? path.join(ASSETS, rel.slice('assets/'.length))
        : path.join(HERE, rel);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/widget_preview.html`;

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);
const ctx = await browser.newContext({ viewport: { width: 760, height: 900 }, deviceScaleFactor: 2 });
const page = await ctx.newPage();
page.on('pageerror', e => console.error('lỗi trang:', e.message));

for (const [q, name] of [['', 'thangnay'], ['?month=2026-9', 'thangsau']]) {
    await page.goto(base + q, { waitUntil: 'networkidle' });
    await page.waitForSelector('body[data-ready="1"]');
    await page.screenshot({ path: path.join(OUT, `widget-${name}.png`), fullPage: true });
    console.log('✓ widget-' + name);
}
await browser.close();
server.close();
console.log('→ ' + OUT);
