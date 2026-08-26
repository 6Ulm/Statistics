/**
 * Quy tắc lịch âm: **mùng 1 là ngày CHỨA điểm Sóc** — xét ở đúng mốc múi giờ mà
 * ứng dụng dùng để hiện giờ Sóc.
 *
 *   node test_soc_parity.mjs
 *
 * Trước đây ứng dụng trộn hai hệ quy chiếu trên cùng một màn hình: ngày âm tính
 * ở UTC+8 còn giờ Sóc lại quy sang giờ địa phương. Ở Paris, Sóc hiện
 * 12/08/2026 19:37 nhưng mùng 1 lại là 13/08 (ở UTC+8 thì Sóc rơi vào 13/08
 * 00:37). Cùng một thời điểm, hai cách đọc — nhìn như tính sai.
 *
 * Phép thử mở ứng dụng thật ở nhiều múi giờ và canh ba điều:
 *   1. tab Kỳ Môn: ngày Sóc hiện ra phải TRÙNG ngày dương của mùng 1;
 *   2. tab Lịch:   ô mùng 1 phải rơi đúng vào ngày ấy;
 *   3. hai tab phải nói cùng một ngày âm cho cùng một ngày dương;
 *   4. bảng mà ứng dụng ghi ra cho WIDGET (publishLunarCache) phải khớp luôn —
 *      widget không chạy được lunar.js nên nó sống bằng bảng này.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.log('Bỏ qua: chưa cài playwright.'); process.exit(0); }

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const f = path.join(WEB, rel);
    if (!f.startsWith(WEB) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
    fs.createReadStream(f).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

// Ngày thử: gồm đúng ca người dùng nêu (Sóc 12/08/2026 17:37 UTC — rơi hai bên
// nửa đêm tuỳ múi giờ) và vài ca rải rác trong năm.
const DATES = [[2026, 8, 12], [2026, 8, 13], [2026, 2, 17], [2026, 5, 16], [2025, 12, 20]];

const ctx0 = await browser.newContext();
const p0 = await ctx0.newPage();
await p0.goto(base, { waitUntil: 'networkidle' });
const CITIES = (await p0.evaluate(
    () => [...document.getElementById('country').options].map(o => o.value)
)).filter(v => v && v !== '__loc');
await ctx0.close();
const PICK = [CITIES[0], ...CITIES.slice(1).filter((_, i) => i % 5 === 0)].slice(0, 6);

let fail = 0, checks = 0;
for (const city of PICK) {
    const ctx = await browser.newContext({ viewport: { width: 412, height: 900 } });
    await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);

    const rows = await page.evaluate(({ city, DATES }) => {
        document.getElementById('country').value = city;
        selectMethod('amban');          // bảng Sóc chỉ dựng ở phái Âm Bàn
        const out = [];
        for (const [y, m, d] of DATES) {
            getDOM('inYear').value = y;
            getDOM('inMonth').value = m;
            getDOM('inDay').value = d;
            getDOM('solarHour').value = 12;
            getDOM('solarMinute').value = 0;
            processAll();
            const lunar = (getDOM('out-lunar-table')?.innerText || '').trim();
            const lmon = parseInt((lunar.split('-')[1] || '').trim(), 10);
            // Bảng Âm Bàn: Tháng | Sóc | Mùng 1 | Rằm — lấy đúng dòng đang tô đậm
            const tr = [...document.querySelectorAll('#ab-tbody tr')]
                .find(t => t.classList.contains('dp-row-active')) ||
                [...document.querySelectorAll('#ab-tbody tr')]
                    .find(t => new RegExp(`Tháng ${lmon}$`).test(t.cells[0].textContent.trim()));
            out.push({
                y, m, d, lunar,
                soc: tr ? tr.cells[1].textContent.trim() : '',
                mung1: tr ? tr.cells[2].textContent.trim() : '',
                ram: tr ? tr.cells[3].textContent.trim() : '',
            });
        }
        // tab Lịch: ngày âm của chính những ngày dương ấy
        const cal = {};
        window.showTab('cal');
        for (const [y, m, d] of DATES) {
            window.__calGoto && window.__calGoto(y, m, d);
            const c = document.querySelector(
                `#calGrid .cal-day[data-y="${y}"][data-m="${m}"][data-d="${d}"]`);
            cal[`${y}-${m}-${d}`] = c
                ? (c.querySelector('.cal-lunar')?.textContent || '').trim() : null;
        }
        window.showTab('qmdj');
        let cache = null;
        try { cache = localStorage.getItem('qmdj.lunarCache'); } catch (e) {}
        return { rows: out, cal, cache, name: countryData[city] && countryData[city].name_vi };
    }, { city, DATES });
    const cal = rows.cal;

    const bad = [];
    const p = n => String(n).padStart(2, '0');
    for (const r of rows.rows) {
        checks++;
        const lday = parseInt((r.lunar.split('-')[0] || '').trim(), 10);
        if (!lday || !/^\d{2}-\d{2}-\d{4}/.test(r.soc)) {
            bad.push(`${r.d}/${r.m}/${r.y}: không đọc được (âm "${r.lunar}", sóc "${r.soc}")`);
            continue;
        }
        // 1. Mùng 1 phải là NGÀY CHỨA điểm Sóc — cùng một dòng của bảng.
        if (r.mung1 !== r.soc.slice(0, 10)) {
            bad.push(`${r.d}/${r.m}/${r.y}: Sóc ${r.soc} nhưng cột Mùng 1 ghi ${r.mung1}`);
        }
        // 2. Rằm = mùng 1 + 14 ngày.
        const [d1, m1, y1] = r.mung1.split('-').map(Number);
        const ram = new Date(y1, m1 - 1, d1 + 14);
        const ramStr = `${p(ram.getDate())}-${p(ram.getMonth() + 1)}-${ram.getFullYear()}`;
        if (r.ram !== ramStr) {
            bad.push(`${r.d}/${r.m}/${r.y}: Rằm ghi ${r.ram}, mùng 1 ${r.mung1} + 14 = ${ramStr}`);
        }
        // 3. Ngày âm của chính ngày đang xét phải khớp khoảng cách tới mùng 1.
        const gap = Math.round(
            (Date.UTC(r.y, r.m - 1, r.d) - Date.UTC(y1, m1 - 1, d1)) / 86400000) + 1;
        if (gap >= 1 && gap !== lday) {
            bad.push(`${r.d}/${r.m}/${r.y}: âm ${lday} nhưng cách mùng 1 (${r.mung1}) ${gap} ngày`);
        }
        // 4. Tab Lịch phải nói cùng ngày âm với tab Kỳ Môn.
        const c = cal[`${r.y}-${r.m}-${r.d}`];
        if (c != null) {
            const cday = parseInt(String(c).split('/')[0], 10);
            if (cday !== lday) {
                bad.push(`${r.d}/${r.m}/${r.y}: tab Kỳ Môn âm ${lday}, tab Lịch âm ${cday}`);
            }
        }
    }
    // ── 4. Bảng cho widget ──
    if (!rows.cache) {
        bad.push('ứng dụng không ghi ra bảng tháng âm cho widget');
    } else {
        const [tzMin, body] = rows.cache.split('|');
        const tbl = body.split(';').filter(Boolean).map(x => x.split(',').map(Number));
        if (tbl.length < 20) bad.push(`bảng widget chỉ có ${tbl.length} tháng`);
        for (let i = 1; i < tbl.length; i++) {
            if (tbl[i][0] <= tbl[i - 1][0]) { bad.push('bảng widget không tăng dần'); break; }
        }
        // Tra bằng chính cách LunarTable.fromAppCache tra, so với tab Lịch.
        const look = j => {
            let lo = 0, hi = tbl.length - 1, idx = -1;
            while (lo <= hi) { const m = (lo + hi) >> 1; if (tbl[m][0] <= j) { idx = m; lo = m + 1; } else hi = m - 1; }
            return idx < 0 ? null : { day: j - tbl[idx][0] + 1, month: tbl[idx][1], leap: tbl[idx][2] === 1 };
        };
        const jd = (y, m, d) => {
            const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
            return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
                + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
        };
        for (const r of rows.rows) {
            const c = cal[`${r.y}-${r.m}-${r.d}`];
            if (c == null) continue;
            const got = look(jd(r.y, r.m, r.d));
            const calDay = parseInt(String(c).split('/')[0], 10);
            if (!got) { bad.push(`bảng widget không phủ ${r.d}/${r.m}/${r.y}`); continue; }
            if (got.day !== calDay) {
                bad.push(`${r.d}/${r.m}/${r.y}: bảng widget âm ${got.day}, tab Lịch âm ${calDay}`);
            }
        }
    }

    if (errs.length) bad.push('lỗi JS: ' + errs.slice(0, 2).join('; '));

    const label = (rows.name || city).padEnd(24);
    if (bad.length) { fail++; console.log(`  LỆCH ${label}`); bad.forEach(b => console.log('    ' + b)); }
    else console.log(`  ok   ${label} ${DATES.length} ngày · mùng 1 = ngày chứa Sóc, Rằm = +14, hai tab + widget khớp`);
    await ctx.close();
}

await browser.close();
server.close();
console.log(fail
    ? `\n✗ ${fail}/${PICK.length} ca lệch`
    : `\n✓ ${PICK.length}/${PICK.length} ca (${checks} phép so): mùng 1 = ngày chứa Sóc, hai tab khớp nhau`);
process.exit(fail ? 1 : 0);
