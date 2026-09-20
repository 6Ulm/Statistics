/**
 * Chụp "ảnh vàng" mọi CON SỐ mà bốn tab hiện ra, để đối chiếu trước/sau một
 * lần tái cấu trúc.
 *
 *   node golden_snapshot.mjs truoc.json      # chụp
 *   node golden_snapshot.mjs sau.json truoc.json   # chụp rồi so với bản cũ
 *
 * Đây KHÔNG phải phép thử đúng/sai — nó không biết con số nào mới là đúng. Nó
 * chỉ trả lời đúng một câu: "sau khi dọn mã, có con số nào đổi không?". Tái
 * cấu trúc mà đổi một con số là hỏng, kể cả khi con số mới nhìn hợp lý hơn.
 *
 * Quét MA TRẬN ngày × nơi × thứ tiếng, và ở mỗi ô lấy hết những gì đọc được:
 * hộp thông tin Kỳ Môn (chính ngọ, tiết khí, cục, tuần thủ, trực phù, trực
 * sử), cả bàn Kỳ Môn 9 cung, hộp Bát Tự (8 chữ + tàng can + phó tinh + nạp âm
 * + thần sát), dòng Lệnh và Nhập vận, bảng Đại Vận, lưới lịch và bảng tiết khí
 * của tab Lịch, và ba bảng của tab Tra cứu.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const OUT = process.argv[2] || 'golden.json';
const SO_VOI = process.argv[3] || null;

const NOI = [
    ['Hà Nội', 21.0278, 105.8342, 'Asia/Ho_Chi_Minh'],
    ['Paris', 48.8566, 2.3522, 'Europe/Paris'],
    ['Bắc Kinh', 39.9042, 116.4074, 'Asia/Shanghai'],
    ['New York', 40.7128, -74.0060, 'America/New_York'],
];
/** Ngày rải đều, kèm mấy ca sát ranh giới: Lập Xuân, giao tiết, Sóc, Tết. */
const NGAY = [
    [1991, 7, 16, 16, 0], [1996, 8, 8, 18, 0], [2026, 2, 4, 3, 0], [2026, 2, 4, 5, 0],
    [2026, 9, 19, 9, 21], [2026, 1, 1, 0, 30], [2026, 6, 21, 23, 45], [2026, 12, 31, 23, 59],
    [1975, 4, 30, 11, 30], [2000, 1, 1, 0, 0], [2035, 11, 7, 12, 0], [1930, 3, 15, 7, 7],
];

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
const ctx = await browser.newContext({
    viewport: { width: 412, height: 852 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true,
});
const page = await ctx.newPage();
const errs = [];
page.on('pageerror', e => errs.push(e.message));
await page.goto(base, { waitUntil: 'networkidle' });
await page.waitForTimeout(1500);

/** Gom mọi chữ đọc được của cả bốn tab, sau khi đã ở đúng tab. */
const GOM = () => ({
    // ── Kỳ Môn ──
    info: ['out-chinhngo', 'out-tietkhi', 'out-cuc', 'out-tuan', 'out-tp', 'out-ts']
        .map(id => { const e = document.getElementById(id); return e ? e.innerText.trim() : null; }),
    ban: [...document.querySelectorAll('.grid .cell')].map(c => c.innerText.replace(/\s+/g, ' ').trim()),
    // ── Hộp Bát Tự (dùng chung hai tab) ──
    bazi: ['Nam', 'Thang', 'Ngay', 'Gio'].map(k => [
        'ttCan', 'ttChi', 'ttTang', 'ttThan', 'ttKV', 'ttTS',
    ].map(p => { const e = document.getElementById(p + k); return e ? e.innerText.replace(/\s+/g, ' ').trim() : null; })),
    ring: ['Nam', 'Thang', 'Ngay', 'Gio']
        .map(k => !!document.querySelector('#ttChi' + k + ' .tt-kv-ring')),
    lunarRow: (document.getElementById('out-lunar-table') || {}).innerText || null,
    via: (document.getElementById('out-via') || {}).innerText || null,
});
const GOM_LENH = () => ({
    now: (document.getElementById('lenhNow') || {}).innerText || null,
    daiVan: [...document.querySelectorAll('#daiVanSec .dv-card')]
        .map(c => c.innerText.replace(/\s+/g, ' ').trim()),
});
const GOM_LICH = () => ({
    tieuDe: (document.getElementById('calTitle') || {}).innerText || null,
    o: [...document.querySelectorAll('.cal-day')].map(c => c.innerText.replace(/\s+/g, ' ').trim()),
    jq: [...document.querySelectorAll('#calJieQi tbody tr')]
        .map(r => [...r.cells].map(c => c.innerText.trim()).join('|')),
});
/** Chạy TRONG trang, nên mọi thứ nó cần phải nằm gọn bên trong. */
const GOM_TRACUU = () => {
    const hàng = sel => [...document.querySelectorAll(sel)]
        .map(r => [...r.cells].map(c => c.innerText.trim()).join('|'));
    return {
        lenh: hàng('#lenhBody tbody tr'),
        am: hàng('#tcAmBody tbody tr'),
        // Hai bảng Trí Nhuận / Sách Bổ: cả tiêu đề cột lẫn ruột, vì chúng có
        // bản tiếng Trung riêng — đúng chỗ dễ trôi khi sửa phần dịch.
        trn: hàng('#trn-table thead tr').concat(hàng('#trn-tbody tr')),
        sb: hàng('#sb-table thead tr').concat(hàng('#sb-tbody tr')),
        tieuDe: ['lblTrinhuanTitle', 'lblSachboTitle']
            .map(id => { const e = document.getElementById(id); return e ? e.textContent.trim() : null; }),
    };
};

const kq = {};
for (const [tên, lat, lon, tzId] of NOI) {
    await page.evaluate(({ tên, lat, lon, tzId }) =>
        window.QMDJLocation.apply(window.QMDJLocation.makeLoc(tên, lat, lon, tzId)),
        { tên, lat, lon, tzId });
    await page.waitForTimeout(500);
    for (const lang of ['vi', 'zh']) {
        await page.evaluate(l => window.setLang(l), lang);
        await page.waitForTimeout(400);
        for (const [y, m, d, h, mi] of NGAY) {
            await page.evaluate(({ y, m, d, h, mi }) => {
                const set = (id, v) => {
                    const el = document.getElementById(id);
                    el.value = String(v);
                    el.dispatchEvent(new Event('change', { bubbles: true }));
                };
                set('inYear', y); set('inMonth', m); set('inDay', d);
                set('solarHour', h); set('solarMinute', mi);
                if (window.processAll) window.processAll();
            }, { y, m, d, h, mi });
            await page.waitForTimeout(260);
            const khoá = `${tên}|${lang}|${y}-${m}-${d} ${h}:${mi}`;
            await page.evaluate(() => window.showTab('qmdj'));
            await page.waitForTimeout(160);
            const o = { qm: await page.evaluate(GOM) };
            await page.evaluate(() => window.showTab('lenh'));
            await page.waitForTimeout(260);
            o.lenh = await page.evaluate(GOM_LENH);
            o.baziLenh = await page.evaluate(GOM);
            await page.evaluate(() => window.showTab('cal'));
            await page.waitForTimeout(260);
            o.lich = await page.evaluate(GOM_LICH);
            kq[khoá] = o;
        }
    }
}
// Tab Tra cứu tra theo NĂM người dùng chọn, không theo ngày sinh — quét riêng.
await page.evaluate(() => window.showTab('tracuu'));
await page.waitForTimeout(500);
for (const lang of ['vi', 'zh']) {
    await page.evaluate(l => window.setLang(l), lang);
    await page.waitForTimeout(500);
    for (const Y of [1991, 2026, 2035]) {
        await page.evaluate(y => window.__tracuuYear(y), Y);
        await page.waitForTimeout(700);
        // Bung hết bốn mục — bảng đang đóng thì không đọc được gì.
        await page.evaluate(() => {
            for (const h of document.querySelectorAll('#traCuuView .dp-header, #lenhHead, #tcAmHead')) {
                const b = h.parentElement.querySelector('.dp-body, .cal-sec-body');
                if (b && getComputedStyle(b).display === 'none') h.click();
            }
        });
        await page.waitForTimeout(600);
        kq[`tracuu|${lang}|${Y}`] = await page.evaluate(GOM_TRACUU);
    }
}

await browser.close();
server.close();
fs.writeFileSync(OUT, JSON.stringify(kq, null, 1));
const sốÔ = Object.keys(kq).length;
console.log(`✓ chụp ${sốÔ} ô → ${OUT}` + (errs.length ? `\n⚠ lỗi JS: ${errs.join('; ')}` : ''));

if (SO_VOI) {
    const cũ = JSON.parse(fs.readFileSync(SO_VOI, 'utf8'));
    let lệch = 0;
    const khoá = new Set([...Object.keys(cũ), ...Object.keys(kq)]);
    for (const k of khoá) {
        const a = JSON.stringify(cũ[k]), b = JSON.stringify(kq[k]);
        if (a !== b) {
            lệch++;
            if (lệch <= 5) {
                console.log(`\n✗ LỆCH ${k}`);
                console.log('  cũ : ' + String(a).slice(0, 400));
                console.log('  mới: ' + String(b).slice(0, 400));
            }
        }
    }
    if (lệch) { console.log(`\n✗ ${lệch}/${khoá.size} ô LỆCH so với ${SO_VOI}`); process.exit(1); }
    console.log(`✓ trùng khít ${SO_VOI} ở cả ${khoá.size} ô — tái cấu trúc không đổi con số nào`);
}
