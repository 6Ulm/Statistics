/**
 * Đối chiếu N ca giữa bản web gốc (1 file HTML) và bản assets Android.
 *
 *   cd android/tools && npm install
 *   node diff_vs_original.mjs /đường/dẫn/QMDJ_1_1.html 1000
 * Differential test: original single-file webapp vs. the split Android assets.
 *
 * So sánh TOÀN BỘ đầu ra hiển thị, không chỉ Tứ Trụ:
 *   - 4 trụ (can/chi), nạp âm, nhãn cột
 *   - bảng thông tin (Chính Ngọ, Tiết khí, Cục, Tuần thủ, Trực Phù/Sử, lịch âm)
 *   - toàn bộ HTML của bàn Kỳ Môn (9 cung)
 *   - bảng chi tiết của cả 3 phái (Trí Nhuận / Sách Bổ / Âm Bàn)
 */
import fs from 'fs';
import path from 'path';
import { JSDOM, VirtualConsole } from 'jsdom';

import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

// Đường dẫn tới file HTML gốc (bản web một-file) và số ca / hạt giống.
const ORIG = process.argv[2];
const N_CASES = Number(process.argv[3] || 1000);
if (!ORIG || !fs.existsSync(ORIG)) {
    console.error('Dùng: node diff_vs_original.mjs <QMDJ_goc.html> [số ca] [hạt giống]');
    console.error('Ví dụ: node diff_vs_original.mjs ~/QMDJ_1_1.html 1000 20260825');
    process.exit(2);
}

/* ── PRNG có hạt giống, để chạy lại ra đúng bộ ca cũ ── */
let seed = Number(process.argv[4] || 20260825);
const rnd = () => {
    seed = (seed * 1103515245 + 12345) & 0x7fffffff;
    return seed / 0x7fffffff;
};
const pick = a => a[Math.floor(rnd() * a.length)];
const int = (lo, hi) => lo + Math.floor(rnd() * (hi - lo + 1));

const COUNTRIES = ['FR', 'VN-HN', 'VN-HCM', 'CN', 'JP', 'KR', 'TH', 'SG', 'MY', 'ID', 'PH',
    'HK', 'TW', 'KH', 'LA', 'MM', 'RU', 'DE', 'GB', 'IT', 'ES', 'NL', 'SE', 'NO', 'CH',
    'PL', 'US_ET', 'CA'];
const METHODS = ['amban', 'bophap', 'trinhuan'];
const LANGS = ['vi', 'zh'];

/* ── Bộ ca thử ── */
const cases = [];
// Nếu N_CASES nhỏ hơn số ca biên, cắt bớt ở cuối.

// 1. Ranh giới giờ Tý / nửa đêm / Chính Ngọ — nơi giờ Mặt Trời thật quyết định ngày
cases.length = 0;
for (const h of [22, 23, 0, 1, 11, 12, 13]) {
    for (const mi of [0, 1, 29, 30, 59]) {
        cases.push({ y: int(1950, 2050), m: int(1, 12), d: int(1, 28), h, mi,
                     country: pick(COUNTRIES), method: pick(METHODS), lang: pick(LANGS) });
    }
}
// 2. Quanh Lập Xuân (đổi trụ năm) và Đông Chí (đổi độn)
for (const [m, d] of [[2, 3], [2, 4], [2, 5], [12, 21], [12, 22], [12, 23], [6, 21], [6, 22]]) {
    for (let k = 0; k < 8; k++) {
        cases.push({ y: int(1920, 2080), m, d, h: int(0, 23), mi: int(0, 59),
                     country: pick(COUNTRIES), method: pick(METHODS), lang: pick(LANGS) });
    }
}
// 3. Chuyển giờ mùa hè ở châu Âu / Bắc Mỹ (DST)
for (const c of ['FR', 'DE', 'GB', 'US_ET', 'CA', 'ES', 'IT', 'NL', 'PL', 'SE', 'NO', 'CH']) {
    for (const [m, d] of [[3, 29], [3, 30], [3, 31], [10, 25], [10, 26], [11, 2], [11, 3]]) {
        cases.push({ y: int(2000, 2050), m, d, h: int(0, 5), mi: int(0, 59),
                     country: c, method: pick(METHODS), lang: pick(LANGS) });
    }
}
// 4. Cuối/đầu tháng, năm nhuận
for (const [m, d] of [[1, 1], [1, 31], [2, 28], [2, 29], [3, 1], [12, 31], [4, 30], [8, 31]]) {
    for (let k = 0; k < 6; k++) {
        cases.push({ y: pick([1996, 2000, 2004, 2020, 2024, 2028, 2100, 1900, 2048]), m, d,
                     h: int(0, 23), mi: int(0, 59), country: pick(COUNTRIES),
                     method: pick(METHODS), lang: pick(LANGS) });
    }
}
// 5. Ngẫu nhiên phủ toàn dải 1900–2100 cho đủ 1000
while (cases.length < N_CASES) {
    cases.push({ y: int(1900, 2100), m: int(1, 12), d: int(1, 28), h: int(0, 23),
                 mi: int(0, 59), country: pick(COUNTRIES), method: pick(METHODS),
                 lang: pick(LANGS) });
}

/* ── Nạp một trang ── */
async function boot(html, url, tag, warns, native) {
    const vc = new VirtualConsole();
    vc.on('jsdomError', e => warns.push(`${tag}/jsdomError: ${e.message}`));
    vc.on('error', (...a) => warns.push(`${tag}/error: ${a.join(' ')}`));
    vc.on('warn', (...a) => {
        const s = a.join(' ');
        if (/astro panel|KMDG error|loc row/i.test(s)) warns.push(`${tag}/warn: ${s}`);
    });
    const dom = new JSDOM(html, {
        url, runScripts: 'dangerously', resources: 'usable',
        pretendToBeVisual: true, virtualConsole: vc,
    });
    // jsdom không có innerText — app.js dùng nó cho bảng thông tin.
    // Vá cho cả hai bên như nhau để những trường đó cũng được so sánh.
    Object.defineProperty(dom.window.Element.prototype, 'innerText', {
        get() { return this.textContent; },
        set(v) { this.textContent = v; },
        configurable: true,
    });
    if (native) dom.window.QMDJNative = native;
    await new Promise(r => dom.window.addEventListener('load', r));
    await new Promise(r => setTimeout(r, 400));
    return dom;
}

const TEXT_IDS = [
    'ttCanNam', 'ttChiNam', 'ttKVNam', 'ttValNam',
    'ttCanThang', 'ttChiThang', 'ttKVThang', 'ttValThang',
    'ttCanNgay', 'ttChiNgay', 'ttKVNgay', 'ttValNgay',
    'ttCanGio', 'ttChiGio', 'ttKVGio', 'ttValGio',
    'out-lunar-table', 'out-via', 'out-chinhngo', 'out-tietkhi',
    'out-cuc', 'out-tuan', 'out-tp', 'out-ts',
    'trn-pttn1', 'trn-pttn2', 'trn-pttn3', 'trn-tk1', 'trn-tk2', 'trn-tk3', 'trn-d-summary',
];
const HTML_IDS = ['board', 'trn-tbody', 'sb-tbody', 'ab-tbody'];

function snap(dom, c) {
    const doc = dom.window.document, w = dom.window;
    if (w.currentLang !== c.lang) w.setLang(c.lang);
    for (const [id, v] of [['inYear', c.y], ['inMonth', c.m], ['inDay', c.d],
                           ['solarHour', c.h], ['solarMinute', c.mi],
                           ['country', c.country], ['methodSelect', c.method]]) {
        doc.getElementById(id).value = String(v);
    }
    w.processAll();
    const o = {};
    for (const id of TEXT_IDS) {
        const el = doc.getElementById(id);
        o[id] = el ? el.textContent.replace(/\s+/g, ' ').trim() : '<missing>';
    }
    for (const id of HTML_IDS) {
        const el = doc.getElementById(id);
        o[id] = el ? el.innerHTML.replace(/\s+/g, ' ').trim() : '<missing>';
    }
    return o;
}

/* ── Chạy ── */
const warnsA = [], warnsB = [];
const domA = await boot(fs.readFileSync(ORIG, 'utf8'), 'file://' + ORIG, 'orig', warnsA, null);
const domB = await boot(fs.readFileSync(path.join(WEB, 'index.html'), 'utf8'),
    'file://' + path.join(WEB, 'index.html'), 'new', warnsB, {
        readAsset: p => fs.readFileSync(path.join(WEB, p), 'utf8'),
        getPref: () => null, setPref: () => {},
        deviceTimeZone: () => 'Asia/Ho_Chi_Minh',
        hasLocationPermission: () => false, requestLocation: () => {},
        platform: () => 'android',
    });

let diffs = 0, compared = 0;
const t0 = Date.now();
const failures = [];

if (cases.length > N_CASES) cases.length = N_CASES;
for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    const a = snap(domA, c), b = snap(domB, c);
    const keys = Object.keys(a);
    compared += keys.length;
    const bad = keys.filter(k => a[k] !== b[k]);
    if (bad.length) {
        diffs++;
        if (failures.length < 5) {
            failures.push({ c, bad: bad.map(k => ({ k, a: a[k].slice(0, 200), b: b[k].slice(0, 200) })) });
        }
    }
    if ((i + 1) % 200 === 0) {
        process.stdout.write(`  ${i + 1}/${cases.length} ca — ${diffs} khác biệt\n`);
    }
}

console.log(`\n${cases.length} ca · ${compared} trường được so sánh · ${((Date.now() - t0) / 1000).toFixed(1)}s`);
console.log(`Khác biệt: ${diffs}`);
for (const f of failures) {
    console.log(`\nCA LỆCH ${JSON.stringify(f.c)}`);
    for (const d of f.bad) console.log(`  ${d.k}\n    gốc=${d.a}\n    mới=${d.b}`);
}
console.log('\nCảnh báo (gốc):', warnsA.length ? warnsA.slice(0, 3) : 'không có');
console.log('Cảnh báo (mới):', warnsB.length ? warnsB.slice(0, 3) : 'không có');

const ok = diffs === 0 && warnsB.length === 0;
console.log(ok ? '\n✓ KHỚP HOÀN TOÀN' : '\n✗ CÓ KHÁC BIỆT');
process.exit(ok ? 0 : 1);
