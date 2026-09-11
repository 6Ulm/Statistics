/**
 * Lịch đã ghim có đồng bộ với ứng dụng không — đo thật, không suy đoán.
 *
 *   node test_widget_sync.mjs
 *
 * Chạy ứng dụng thật trong Chromium, giả lập lớp native để hứng đúng những gì
 * nó ghi ra kho tuỳ chọn, rồi dựng lại ĐƯỜNG TRA CỦA WIDGET (bản sao của
 * LunarTable.kt) trên chính dữ liệu ấy và so từng ngày với ứng dụng.
 *
 * Ca quan trọng nhất: KHÔNG mở tab Lịch. Ứng dụng mở ra ở tab Kỳ Môn, và đã có
 * lúc bảng tháng chỉ được ghi trong render() của tab Lịch — ai không bao giờ mở
 * tab ấy thì widget không hề nhận được bảng của ứng dụng.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');

let chromium;
try { ({ chromium } = await import('playwright')); }
catch (e) { console.log('Bỏ qua: chưa cài playwright.'); process.exit(0); }

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

/* ── Bản sao đường tra của LunarTable.kt ── */
const lines = fs.readFileSync(path.join(ASSETS, 'lunar_months.txt'), 'utf8').trim().split('\n');
const packStart = [], packSec = [], packMonth = [], packLeap = [];
for (let i = 1; i < lines.length; i++) {
    const p = lines[i].split(' ');
    packStart.push(+p[0]); packSec.push(+p[1]); packMonth.push(+p[3]); packLeap.push(p[4] === '1');
}
function jdnOf(y, m, d) {
    const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}
const offsetMin = (tzId, ms) => {
    const f = new Intl.DateTimeFormat('en-US', { timeZone: tzId, hour12: false, year: 'numeric',
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const p = Object.fromEntries(f.formatToParts(new Date(ms)).map(x => [x.type, x.value]));
    return (Date.UTC(+p.year, +p.month - 1, +p.day, (+p.hour) % 24, +p.minute, +p.second) - ms) / 60000;
};
/** LunarTable.monthStart — mốc mùng 1 của tháng thứ i, xét ở múi giờ tzId. */
function packMonthStart(i, tzId) {
    const utc = (packStart[i] - 2440588) * 86400000 + packSec[i] * 1000 - 7 * 3600000;
    return Math.floor((utc + offsetMin(tzId, utc) * 60000) / 86400000) + 2440588;
}
/** LunarTable.lunarOf — bảng của ứng dụng trước, hết tầm mới về bảng đóng sẵn. */
function widgetLunar(jdn, tzId, cache) {
    if (cache && cache.tzId === tzId && cache.start.length >= 2 &&
        jdn >= cache.start[0] && jdn < cache.start[cache.start.length - 1]) {
        let at = -1;
        for (let i = 0; i < cache.start.length; i++) if (cache.start[i] <= jdn) at = i; else break;
        if (at >= 0) return { day: jdn - cache.start[at] + 1, month: cache.month[at], from: 'app' };
    }
    let at = -1;
    for (let i = 0; i < packStart.length; i++) if (packMonthStart(i, tzId) <= jdn) at = i; else break;
    if (at < 0) return null;
    return { day: jdn - packMonthStart(at, tzId) + 1, month: packMonth[at], from: 'apk' };
}

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

let pass = 0, fail = 0;
function ok(what, cond, detail) {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(52)} ${cond ? '' : (detail || '')}`);
}

/** Mở ứng dụng với lớp native giả; trả về trang + kho tuỳ chọn đã ghi. */
async function boot(tzId, openCalTab) {
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2 });
    await ctx.addInitScript(() => {
        window.__prefs = {};
        window.__pokes = 0;
        window.QMDJNative = {
            getPref: k => (k in window.__prefs ? window.__prefs[k] : null),
            setPref: (k, v) => { window.__prefs[k] = v; },
            deviceTimeZone: () => 'Europe/Paris',
            platform: () => 'android',
            refreshCalendarWidget: () => { window.__pokes++; },
        };
    });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    // Chọn địa điểm đúng như người dùng làm, rồi ĐỂ NGUYÊN ở tab Kỳ Môn.
    await page.evaluate(async id => {
        const sel = document.getElementById('country');
        for (const o of sel.options) {
            const info = countryData[o.value];
            if (info && info.tzId === id) { sel.value = o.value; break; }
        }
        processAll();
    }, tzId);
    await page.waitForTimeout(900);
    if (openCalTab) { await page.click('#tabCal'); await page.waitForTimeout(900); }
    return { ctx, page, errs };
}

for (const tzId of ['Europe/Paris', 'Asia/Ho_Chi_Minh', 'America/New_York']) {
    for (const openCal of [false, true]) {
        const tag = `${tzId.split('/')[1]} · ${openCal ? 'có mở tab Lịch' : 'KHÔNG mở tab Lịch'}`;
        const { ctx, page, errs } = await boot(tzId, openCal);

        const raw = await page.evaluate(() => window.__prefs['qmdj.lunarCache'] || '');
        const pokes = await page.evaluate(() => window.__pokes);
        ok(`${tag}: ứng dụng có ghi bảng tháng cho widget`, !!raw, 'không ghi gì');
        ok(`${tag}: có bảo widget vẽ lại`, pokes > 0, `${pokes} lần`);

        let cache = null;
        if (raw) {
            const bar = raw.indexOf('|');
            cache = { tzId: raw.slice(0, bar), start: [], month: [], leap: [] };
            for (const row of raw.slice(bar + 1).split(';')) {
                if (!row) continue;
                const f = row.split(',');
                cache.start.push(+f[0]); cache.month.push(+f[1]); cache.leap.push(f[2] === '1');
            }
            ok(`${tag}: khoá bảng là mã múi giờ đang chọn`, cache.tzId === tzId, cache.tzId);
        }

        // So từng ngày trong 120 ngày quanh hôm nay — ứng dụng vs widget.
        const days = await page.evaluate(() => {
            const out = [];
            const now = new Date();
            for (let k = -60; k < 60; k++) {
                const d = new Date(now.getFullYear(), now.getMonth(), now.getDate() + k);
                const ctx2 = window.__calCtxFor(d.getFullYear(), d.getMonth() + 1);
                out.push({ y: d.getFullYear(), m: d.getMonth() + 1, d: d.getDate(), z: ctx2 });
            }
            return out;
        }).catch(() => null);

        // Không có móc riêng thì đọc thẳng lưới lịch của tab Lịch.
        const appDays = await page.evaluate(() => {
            const out = [];
            const now = new Date();
            for (let k = -60; k < 60; k++) {
                const dt = new Date(now.getFullYear(), now.getMonth(), now.getDate() + k);
                const info = countryData[document.getElementById('country').value];
                const tz = getTimezoneOffset(info.tzId, new Date(dt.getFullYear(), dt.getMonth(), dt.getDate(), 12));
                const list = zi_months(dt.getFullYear(), info.lon, info.tzId, tz);
                const a = Math.floor((14 - (dt.getMonth() + 1)) / 12);
                const yy = dt.getFullYear() + 4800 - a, mm = (dt.getMonth() + 1) + 12 * a - 3;
                const jdn = dt.getDate() + Math.floor((153 * mm + 2) / 5) + 365 * yy
                    + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
                let at = -1;
                for (let i = 0; i < list.length; i++) if (list[i].jdn <= jdn) at = i; else break;
                out.push(at < 0 ? null : { jdn, day: jdn - list[at].jdn + 1, month: list[at].month });
            }
            return out;
        });

        let bad = 0, first = '';
        let fromApp = 0;
        for (const a of appDays) {
            if (!a) continue;
            const w = widgetLunar(a.jdn, tzId, cache);
            if (w && w.from === 'app') fromApp++;
            if (!w || w.day !== a.day || w.month !== a.month) {
                bad++;
                if (!first) first = `JDN ${a.jdn}: app ${a.day}/${a.month} · widget ` +
                    (w ? `${w.day}/${w.month} (${w.from})` : 'không tra được');
            }
        }
        ok(`${tag}: 120 ngày khớp từng ngày`, bad === 0, `${bad} ngày lệch — ${first}`);
        ok(`${tag}: widget dùng bảng của ứng dụng`, fromApp >= appDays.length - 2,
            `chỉ ${fromApp}/${appDays.length} ngày tra được trong bảng app`);
        ok(`${tag}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ LỊCH ĐÃ GHIM KHÔNG ĐỒNG BỘ' : '✓ Lịch đã ghim khớp ứng dụng từng ngày một');
process.exit(fail ? 1 : 0);
