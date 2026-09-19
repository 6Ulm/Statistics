/**
 * Hai bảng tra cứu — "Tiết khí" (tab Lịch, cũng là mục duy nhất của widget) và
 * "Lịch âm" (nay ở tab Tra cứu) — phải khớp từng ô với ĐƯỜNG TRA CỦA WIDGET.
 *
 *   node test_widget_sections.mjs
 *
 * Widget không chạy được lunar.js: nó tra hai bảng đóng trong APK. Bảng ấy nay
 * mang thêm ba thứ mà trước đây không có — can chi THÁNG của từng tiết khí, và
 * mốc Sóc/Vọng để hiện — nên đây là chỗ chứng minh ba cột mới không trôi khỏi
 * thứ ứng dụng in ra.
 *
 * Phép thử chạy ứng dụng thật trong Chromium, đọc đúng ba cột của từng bảng,
 * rồi dựng lại ĐƯỜNG TRA CỦA WIDGET (bản sao của LunarTable.kt) trên chính hai
 * tệp assets và so từng ô một, ở nhiều múi giờ.
 *
 * Vì sao mốc Sóc phải là CỘT RIÊNG, không dùng lại cột giây đầu dòng: cột ấy là
 * ngưỡng bậc thang định NGÀY mùng 1, không phải điểm Sóc thiên văn. Đo trên
 * 2000–2050 thì hai con số lệch nhau ở 2,4% số tháng, có ca lệch 17 giờ.
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

/* ─────────── Bản sao đường tra của LunarTable.kt ─────────── */

const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
const TIET_KHI = [
    'Đông Chí', 'Tiểu Hàn', 'Đại Hàn', 'Lập Xuân', 'Vũ Thủy', 'Kinh Trập',
    'Xuân Phân', 'Thanh Minh', 'Cốc Vũ', 'Lập Hạ', 'Tiểu Mãn', 'Mang Chủng',
    'Hạ Chí', 'Tiểu Thử', 'Đại Thử', 'Lập Thu', 'Xử Thử', 'Bạch Lộ',
    'Thu Phân', 'Hàn Lộ', 'Sương Giáng', 'Lập Đông', 'Tiểu Tuyết', 'Đại Tuyết',
];

const lmLines = fs.readFileSync(path.join(ASSETS, 'lunar_months.txt'), 'utf8').trim().split('\n');
const mo = { start: [], year: [], month: [], leap: [], socJdn: [], socMin: [], vongJdn: [], vongMin: [] };
for (let i = 1; i < lmLines.length; i++) {
    const p = lmLines[i].split(' ').map(Number);
    mo.start.push(p[0]); mo.year.push(p[2]); mo.month.push(p[3]); mo.leap.push(p[4] === 1);
    mo.socJdn.push(p[0] + p[5]); mo.socMin.push(p[6]);
    mo.vongJdn.push(p[0] + p[5] + p[7]); mo.vongMin.push(p[8]);
}
const jqLines = fs.readFileSync(path.join(ASSETS, 'jieqi.txt'), 'utf8').trim().split('\n');
const jq = { jdn: [], min: [], idx: [], gz: [] };
for (const line of jqLines) {
    const p = line.split(' ').map(Number);
    jq.jdn.push(p[0]); jq.min.push(p[1]); jq.idx.push(p[2]); jq.gz.push(p[3]);
}

function jdnOf(y, m, d) {
    const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}
function civilOf(j) {
    const a = j + 32044, b = Math.floor((4 * a + 3) / 146097);
    const c = a - Math.floor(146097 * b / 4), d = Math.floor((4 * c + 3) / 1461);
    const e = c - Math.floor(1461 * d / 4), m = Math.floor((5 * e + 2) / 153);
    return [100 * b + d - 4800 + Math.floor(m / 10),
            m + 3 - 12 * Math.floor(m / 10),
            e - Math.floor((153 * m + 2) / 5) + 1];
}
const offsetMin = (tzId, ms) => {
    const f = new Intl.DateTimeFormat('en-US', { timeZone: tzId, hour12: false, year: 'numeric',
        month: '2-digit', day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit' });
    const p = Object.fromEntries(f.formatToParts(new Date(ms)).map(x => [x.type, x.value]));
    return (Date.UTC(+p.year, +p.month - 1, +p.day, (+p.hour) % 24, +p.minute, +p.second) - ms) / 60000;
};
const pad = n => String(n).padStart(2, '0');
/** LunarTable.localize + dateText: mốc ghi ở UTC+7, in ra theo múi giờ tzId. */
function stamp(jdnLocal7, minute, tzId) {
    const utc = (jdnLocal7 - 2440588) * 86400000 + minute * 60000 - 7 * 3600000;
    const local = utc + offsetMin(tzId, utc) * 60000;
    let days = Math.floor(local / 86400000);
    let rem = local - days * 86400000;
    const [y, m, d] = civilOf(days + 2440588);
    const mins = Math.floor(rem / 60000);
    return `${pad(d)}-${pad(m)}-${y} ${pad(Math.floor(mins / 60))}:${pad(mins % 60)}`;
}

/** LunarTable.jieQiYearOf — 24 mục từ Đông Chí gần nhất không muộn hơn `ref`. */
function widgetJieQi(ref, tzId) {
    let at = -1;
    for (let i = 0; i < jq.jdn.length; i++) if (jq.jdn[i] <= ref) at = i; else break;
    let start = at;
    while (start >= 0 && jq.idx[start] !== 0) start--;
    const out = [];
    for (let i = start; i < start + 24; i++) {
        out.push([TIET_KHI[jq.idx[i]], stamp(jq.jdn[i], jq.min[i], tzId),
                  CAN[jq.gz[i] % 10] + ' ' + CHI[jq.gz[i] % 12]]);
    }
    return out;
}

/** Các tháng âm của một năm âm, kèm mốc Sóc/Vọng để hiện. */
function widgetMonths(lunarYear, tzId) {
    const out = [];
    for (let i = 0; i < mo.start.length; i++) {
        if (mo.year[i] !== lunarYear) continue;
        out.push(['Tháng ' + mo.month[i] + (mo.leap[i] ? ' (nhuận)' : ''),
                  stamp(mo.socJdn[i], mo.socMin[i], tzId),
                  stamp(mo.vongJdn[i], mo.vongMin[i], tzId)]);
    }
    return out;
}

/* ─────────── Chạy ứng dụng thật ─────────── */

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);
let pass = 0, fail = 0;
function ok(what, cond, detail) {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(50)} ${cond ? '' : (detail || '')}`);
}

for (const tzId of ['Asia/Ho_Chi_Minh', 'Europe/Paris', 'America/New_York']) {
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2 });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    await page.click('#langDisplayBtn');
    await page.waitForSelector('#optOverlay.open .opt-row[data-value="vi"]');
    await page.click('.opt-row[data-value="vi"]');
    await page.waitForTimeout(700);
    await page.evaluate(id => {
        const sel = document.getElementById('country');
        for (const o of sel.options) {
            const info = countryData[o.value];
            if (info && info.tzId === id) { sel.value = o.value; break; }
        }
        processAll();
    }, tzId);
    await page.waitForTimeout(900);
    await page.click('#tabCal');
    await page.waitForTimeout(900);
    // Mục Tiết khí: mục DUY NHẤT còn lại ở tab Lịch (và ở widget).
    await page.evaluate(() => {
        const s = document.getElementById('calSecJq');
        if (s && !s.classList.contains('cal-sec-open')) s.querySelector('.cal-sec-head').click();
    });
    await page.waitForTimeout(700);

    const jqAndDay = await page.evaluate(() => {
        const rows = sel => [...document.querySelectorAll(sel)]
            .map(tr => [...tr.cells].map(c => c.textContent.trim()));
        const now = new Date();
        return {
            jq: rows('#calJqBody tr'),
            today: [now.getFullYear(), now.getMonth() + 1, now.getDate()],
        };
    });

    // Bảng Lịch âm đã CHUYỂN sang tab Tra cứu và bỏ hẳn khỏi widget — nhưng nó
    // vẫn đọc CÙNG hai tệp assets mà LunarTable.kt tra, nên vẫn phải khớp từng
    // ô với đường tra ấy. Ở đó bảng tra theo NĂM người dùng gõ, không theo
    // tháng đang xem, nên phải đặt năm tường minh.
    await page.click('#tabTraCuu');
    await page.waitForTimeout(600);
    await page.evaluate(y => window.__tracuuYear(y), jqAndDay.today[0]);
    await page.waitForTimeout(700);
    await page.evaluate(() => {
        const sec = document.getElementById('tcAmSec');
        if (sec && getComputedStyle(sec).display === 'none') window.toggleDetailPanel('tcam');
    });
    await page.waitForTimeout(600);
    const app = {
        ...jqAndDay,
        am: await page.evaluate(() => [...document.querySelectorAll('#tcAmBody tbody tr')]
            .map(tr => [...tr.cells].map(c => c.textContent.trim()))),
    };

    console.log(`\n${tzId}`);
    const ref = jdnOf(app.today[0], app.today[1], app.today[2]);
    const wJq = widgetJieQi(ref, tzId);
    ok('bảng Tiết khí: đủ 24 dòng cả hai bên',
        app.jq.length === 24 && wJq.length === 24, `${app.jq.length} vs ${wJq.length}`);
    let badName = null, badDate = null, badGz = null;
    for (let i = 0; i < Math.min(app.jq.length, wJq.length); i++) {
        if (!badName && app.jq[i][0] !== wJq[i][0]) badName = `#${i} "${app.jq[i][0]}" vs "${wJq[i][0]}"`;
        if (!badDate && app.jq[i][1] !== wJq[i][1]) badDate = `#${i} ${app.jq[i][1]} vs ${wJq[i][1]}`;
        if (!badGz && app.jq[i][2] !== wJq[i][2]) badGz = `#${i} "${app.jq[i][2]}" vs "${wJq[i][2]}"`;
    }
    ok('…tên tiết khí khớp từng dòng', !badName, badName);
    ok('…mốc Dương lịch khớp từng dòng', !badDate, badDate);
    ok('…CAN CHI THÁNG khớp từng dòng (cột mới)', !badGz, badGz);

    // Năm âm mà bảng đang hiện: suy từ chính nhãn tháng đầu bảng của ứng dụng.
    let wAm = null;
    for (let y = app.today[0] - 1; y <= app.today[0] + 1 && !wAm; y++) {
        const cand = widgetMonths(y, tzId);
        if (cand.length === app.am.length && cand[0][0] === app.am[0][0]) wAm = cand;
    }
    ok('bảng Lịch âm: tìm được đúng năm âm và đủ số tháng', !!wAm,
        `ứng dụng ${app.am.length} dòng, đầu bảng "${app.am[0] && app.am[0][0]}"`);
    if (wAm) {
        let badMo = null, badSoc = null, badVong = null;
        for (let i = 0; i < app.am.length; i++) {
            if (!badMo && app.am[i][0] !== wAm[i][0]) badMo = `#${i} "${app.am[i][0]}" vs "${wAm[i][0]}"`;
            if (!badSoc && app.am[i][1] !== wAm[i][1]) badSoc = `#${i} ${app.am[i][1]} vs ${wAm[i][1]}`;
            if (!badVong && app.am[i][2] !== wAm[i][2]) badVong = `#${i} ${app.am[i][2]} vs ${wAm[i][2]}`;
        }
        ok('…nhãn tháng khớp từng dòng', !badMo, badMo);
        ok('…mốc SÓC khớp từng dòng (cột mới)', !badSoc, badSoc);
        ok('…mốc VỌNG khớp từng dòng (cột mới)', !badVong, badVong);
    }
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Widget và tab Lịch nói cùng một bảng');
process.exit(fail ? 1 : 0);
