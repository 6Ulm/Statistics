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

/**
 * Bảng riêng của từng phái chỉ được DỰNG LẠI khi đang ở đúng phái đó **và**
 * ngôn ngữ không phải tiếng Trung (app.js: `if (method === 'amban' && notZH)`).
 * Ngoài điều kiện ấy nó giữ nguyên nội dung của ca trước — so vào đó là so rác
 * còn sót lại, và kết quả phụ thuộc thứ tự chạy ca.
 */
const PANEL_OF = { 'ab-tbody': 'amban', 'sb-tbody': 'bophap', 'trn-tbody': 'trinhuan' };
const trnPanel = k => k.startsWith('trn-');
function relevantKey(cc) {
    return k => {
        const owner = trnPanel(k) ? 'trinhuan' : PANEL_OF[k];
        return !owner || (owner === cc.method && cc.lang !== 'zh');
    };
}

/**
 * Những trường phụ thuộc NGÀY ÂM LỊCH — nơi bản Android cố ý khác bản web gốc.
 *
 * Bản gốc tính ngày âm ở mốc UTC+8 rồi lại hiện giờ Sóc theo giờ địa phương,
 * nên ở Paris nó ghi "Sóc 12-08-2026 19:37" ngay cạnh "Mùng 1 13-08-2026" —
 * hai hệ quy chiếu trong cùng một dòng, trong khi quy tắc là mùng 1 phải là
 * ngày CHỨA điểm Sóc. Bản Android tính ngày âm ở mốc địa phương nên khớp.
 *
 * Cục Âm Bàn = (chi năm + tháng âm + ngày âm + chi giờ) % 9 nên cũng đổi theo,
 * và bảng Âm Bàn (ab-tbody) hiện cột Mùng 1 / Rằm lấy từ đó.
 *
 * Từ khi mốc mùng 1 đổi sang CHÍNH TÝ thiên văn (nửa đêm mặt trời thật), khác
 * biệt xảy ra ở mọi nơi lệch khỏi kinh tuyến múi giờ của mình — kể cả nơi đúng
 * UTC+8, vì Bắc Kinh ở 116,4°Đ chứ không phải 120°Đ.
 *
 * Nên chỗ đáng canh không còn là "bằng 0", mà là HÌNH DẠNG của khác biệt: ngày
 * âm chỉ được lệch ĐÚNG MỘT NGÀY (hoặc chỉ đổi nhãn tháng, khi mốc kinh tuyến
 * làm cấu trúc tháng nhuận khác đi). Lệch hơn thế là hỏng thật.
 */
const LUNAR_DEPENDENT = new Set(['out-lunar-table', 'out-cuc', 'ab-tbody']);

/**
 * Hệ quả kéo theo khi cục đổi: bàn Kỳ Môn được dựng TỪ cục, nên Thiên Bàn /
 * Trực Phù / Trực Sử đổi theo. Chỉ miễn cho những trường này KHI cục thật sự
 * khác — cục giống mà bàn khác thì đó là hồi quy thật, phải đỏ.
 */
const CUC_DEPENDENT = new Set(['board', 'out-tp', 'out-ts']);

/**
 * Chính Ngọ: bản gốc tính phương trình thời gian tại **12:00 UTC** của ngày
 * đó, trong khi nó phải được tính tại CHÍNH thời điểm Chính Ngọ địa phương.
 * Với nơi lệch xa UTC (Nhật +9, Mỹ −5) thì chênh tới 9 giờ, đủ để phương
 * trình thời gian đổi ~9 giây — thỉnh thoảng lật một phút hiển thị.
 *
 * Đã đối chiếu Meeus ví dụ 28.b: hai công thức khớp nhau tới 0,04 giây, và
 * khi đánh giá tại cùng một thời điểm thì chúng khớp tới 0,00 giây. Nên khác
 * biệt duy nhất là thời điểm đánh giá, và bản mới mới là bản đúng.
 *
 * Cho phép lệch TỐI ĐA MỘT PHÚT — lệch hơn thế vẫn là hồi quy.
 */
const CHINH_NGO = 'out-chinhngo';

/**
 * Trụ GIỜ đổi tại các mốc thời thần, cách nhau 120 phút kể từ đầu giờ Tý
 * (Chính Ngọ − 13h). Sửa mốc đánh giá EoT dịch Chính Ngọ vài giây, nên một
 * thời điểm nhập nằm SÁT ranh giới thời thần có thể lật sang chi kế bên.
 *
 * Chỉ miễn khi thời điểm nhập cách ranh giới DƯỚI MỘT PHÚT — đó là chữ ký của
 * ca sát ranh, không phải của một lỗi tính giờ. Xa hơn thế vẫn là hồi quy.
 */
const GIO_KEYS = new Set(['ttCanGio', 'ttChiGio', 'ttKVGio', 'ttValGio']);
/** Bàn Kỳ Môn và tuần thủ dựng TỪ can/chi giờ, nên lật theo. */
const GIO_DEPENDENT = new Set(['board', 'out-tp', 'out-ts', 'out-tuan', 'out-cuc']);
function gioMarginMinutes(dom, c) {
    try {
        const w = dom.window;
        const lon = w.eval(`countryData[${JSON.stringify(c.country)}].lon`);
        const tzId = tzIdOf(dom, c.country);
        const tz = tzHoursAt(tzId, c.y, c.m, c.d);
        const ty = w.Astro.solarNoonMinutes(c.y, c.m, c.d, lon, tz) - 780;
        const pos = (((c.h * 60 + c.mi - ty) % 1440) + 1440) % 1440;
        return Math.abs(pos - Math.round(pos / 120) * 120);
    } catch (e) {
        return Infinity;
    }
}
function chinhNgoGapMinutes(a, b) {
    const mm = v => { const m = String(v).match(/(\d{1,2}):(\d{2})/); return m ? +m[1] * 60 + +m[2] : null; };
    const x = mm(a), y = mm(b);
    return (x === null || y === null) ? Infinity : Math.abs(x - y);
}

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

let diffs = 0, compared = 0, deliberate = 0, lunarZi = 0, lunarMeridian = 0, lunarShapeBad = 0;
let chinhNgoFixed = 0, gioEdge = 0;
const shapeSamples = [];
const t0 = Date.now();
const failures = [];

/**
 * Múi giờ THẬT của thành phố tại đúng thời điểm của ca thử.
 *
 * Không dùng danh sách "nước ở UTC+8" được: Malaysia từng ở UTC+7:30 tới 1982,
 * Đài Loan và Philippines cũng đổi múi giờ trong quá khứ — mà bộ ca thử chạy từ
 * 1900 đến 2100. Phải tra theo từng thời điểm.
 *
 * countryData khai bằng `const` nên không nằm trên window; eval trong chính
 * realm của jsdom thì thấy được binding đó.
 */
const tzIdCache = new Map();
function tzIdOf(dom, country) {
    if (!tzIdCache.has(country)) {
        let id = null;
        try { id = dom.window.eval(`countryData[${JSON.stringify(country)}].tzId`); }
        catch (e) { id = null; }
        tzIdCache.set(country, id);
    }
    return tzIdCache.get(country);
}
function tzHoursAt(tzId, y, m, d) {
    const utcMs = Date.UTC(y, m - 1, d, 12);
    const f = new Intl.DateTimeFormat('en-US', {
        timeZone: tzId, hour12: false, year: 'numeric', month: '2-digit',
        day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    const p = Object.fromEntries(f.formatToParts(new Date(utcMs)).map(x => [x.type, x.value]));
    return (Date.UTC(+p.year, +p.month - 1, +p.day, (+p.hour) % 24, +p.minute, +p.second)
        - utcMs) / 3600000;
}

if (cases.length > N_CASES) cases.length = N_CASES;
for (let i = 0; i < cases.length; i++) {
    const c = cases[i];
    const a = snap(domA, c), b = snap(domB, c);
    const keys = Object.keys(a);
    compared += keys.filter(relevantKey(c)).length;
    // Ở múi giờ UTC+8 hai bản phải trùng khít. Nơi khác, các trường phụ thuộc
    // ngày âm lịch được phép khác — đó là chỗ bản Android sửa lỗi trộn hệ quy
    // chiếu của bản gốc (xem LUNAR_DEPENDENT), không phải hồi quy.
    // Mốc địa phương trùng UTC+8 thì hai bản phải trùng khít — đó là chỗ đáng
    // canh. Lệch khỏi UTC+8 thì ngày âm khác là CÓ CHỦ Ý.
    const tzId = tzIdOf(domB, c.country);
    const tzH = tzId ? tzHoursAt(tzId, c.y, c.m, c.d) : null;
    const allowLunar = true;   // xem ghi chú ở LUNAR_DEPENDENT
    const sameMeridian = tzH !== null && Math.abs(tzH - 8) < 1e-9;
    const relevant = relevantKey(c);
    const rawBad = keys.filter(k => relevant(k) && a[k] !== b[k]);
    // Ngày âm lệch quá một ngày là hỏng thật. Ngày giống nhau mà nhãn tháng khác
    // thì không phải lệch ngày — đó là cấu trúc tháng nhuận đổi theo mốc kinh
    // tuyến, chuyện đã biết.
    const dayOf = v => parseInt(String(v).split('-')[0].trim(), 10);
    if (a['out-lunar-table'] !== b['out-lunar-table']) {
        const da = dayOf(a['out-lunar-table']), db = dayOf(b['out-lunar-table']);
        const gap = Math.abs(da - db);
        const rollover = (da === 1 && db >= 29) || (db === 1 && da >= 29);
        if (!(gap <= 1 || rollover)) {
            lunarShapeBad++;
            if (shapeSamples.length < 4) {
                shapeSamples.push(`${JSON.stringify(c)}  gốc=${a['out-lunar-table']}  mới=${b['out-lunar-table']}`);
            }
        } else if (sameMeridian) lunarZi++; else lunarMeridian++;
    }
    // Chính Ngọ lệch ≤1 phút là do sửa thời điểm đánh giá EoT — xem CHINH_NGO.
    const chinhNgoOk = a[CHINH_NGO] === b[CHINH_NGO] ||
        chinhNgoGapMinutes(a[CHINH_NGO], b[CHINH_NGO]) <= 1;
    if (a[CHINH_NGO] !== b[CHINH_NGO] && chinhNgoOk) chinhNgoFixed++;

    // Trụ giờ chỉ được lệch khi thời điểm nhập sát ranh giới thời thần.
    const gioDiffers = [...GIO_KEYS].some(k => a[k] !== b[k]);
    const gioOk = gioDiffers && gioMarginMinutes(domB, c) < 1;
    if (gioOk) gioEdge++;

    const cucChanged = allowLunar && a['out-cuc'] !== b['out-cuc'];
    const bad = (!allowLunar ? rawBad : rawBad.filter(k =>
        !LUNAR_DEPENDENT.has(k) && !(cucChanged && CUC_DEPENDENT.has(k))))
        .filter(k => !(k === CHINH_NGO && chinhNgoOk))
        .filter(k => !(gioOk && (GIO_KEYS.has(k) || GIO_DEPENDENT.has(k))));
    if (rawBad.length !== bad.length) deliberate++;
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
console.log(`Ca khác CÓ CHỦ Ý (ngày âm + hệ quả): ${deliberate}`);
console.log(`  ngày âm khác ở nơi đúng UTC+8 (do Chính Tý): ${lunarZi}`);
console.log(`  ngày âm lệch do đổi MỐC KINH TUYẾN (ngoài UTC+8): ${lunarMeridian}`);
console.log(`  ngày âm lệch QUÁ MỘT NGÀY (phải bằng 0): ${lunarShapeBad}`);
console.log(`  Chính Ngọ lệch 1 phút (sửa mốc đánh giá EoT): ${chinhNgoFixed}`);
console.log(`  trụ giờ lật vì nhập SÁT ranh giới thời thần (<1 phút): ${gioEdge}`);
shapeSamples.forEach(x => console.log('    ' + x));
for (const f of failures) {
    console.log(`\nCA LỆCH ${JSON.stringify(f.c)}`);
    for (const d of f.bad) console.log(`  ${d.k}\n    gốc=${d.a}\n    mới=${d.b}`);
}
console.log('\nCảnh báo (gốc):', warnsA.length ? warnsA.slice(0, 3) : 'không có');
console.log('Cảnh báo (mới):', warnsB.length ? warnsB.slice(0, 3) : 'không có');

const ok = diffs === 0 && lunarShapeBad === 0 && warnsB.length === 0;
console.log(ok ? '\n✓ KHỚP HOÀN TOÀN' : '\n✗ CÓ KHÁC BIỆT');
process.exit(ok ? 0 : 1);
