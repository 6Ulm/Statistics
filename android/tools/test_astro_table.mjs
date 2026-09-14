/**
 * Bảng thiên văn DE423 (js/astro_table.js) có thật sự đang được dùng không.
 *
 *   node test_astro_table.mjs
 *
 * Bảng này thay chuỗi giải tích của ShouXing cho tiết khí và điểm Sóc/Vọng.
 * Nếu thẻ <script> bị xoá khỏi index.html, hoặc tệp bảng biến mất, ứng dụng
 * KHÔNG hỏng — nó lặng lẽ rơi về chuỗi cũ, sai tới ~4 phút ở điểm Vọng. Im
 * lặng đúng là lý do phải có phép thử này.
 *
 * Mốc vàng lấy từ tools/almanac/build_astro_table.py (DE423 + IAU 2006/2000A),
 * đơn vị: ngày TT kể từ J2000.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const require = createRequire(import.meta.url);
const { LunarYear, ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));
require(path.join(WEB, 'js', 'astro.js'));
const { Astro } = globalThis;

let fail = 0;
const ok = (cond, msg, extra = '') => {
    if (cond) console.log(`  ok    ${msg} ${extra}`);
    else { fail++; console.log(`  FAIL  ${msg} ${extra}`); }
};

/** ShouXing trả giờ dân dụng địa phương; đổi ngược về TT để so với bảng. */
const TZ = 7, TZDAY = TZ / 24;
const toTT = (local) => { const t = local - TZDAY; return t + ShouXingUtil.dtT(t); };

// Gọi một lần cho lunar.js tự nạp bảng (xem ShouXingUtil._table).
ShouXingUtil.setTzOffsetHours(TZ);
ShouXingUtil.shuoHigh(0, TZ);
const T = globalThis.AstroTable;

console.log('1. Bảng có mặt và phủ đủ 1900-2100');
ok(!!T, 'astro_table.js nạp được');
if (!T) { console.log('\n✗ không có bảng — dừng'); process.exit(1); }
ok(T.newMoon(-1237) !== null && T.newMoon(1250) !== null, 'điểm Sóc phủ hai đầu');
ok(T.term(-2381) !== null && T.term(2442) !== null, 'tiết khí phủ hai đầu');
ok(T.newMoon(-99999) === null && T.term(99999) === null, 'ngoài khoảng trả null');

console.log('\n2. Quy ước chỉ số khớp ShouXing');
// k=0 là điểm Sóc 2000-01-06; n=0 là Xuân Phân 1999 (JD 2451259).
ok(Math.abs(T.newMoon(0) - 5.26) < 0.05, 'k=0 rơi vào 2000-01-06',
   `(${T.newMoon(0).toFixed(3)} ngày sau J2000)`);
ok(Math.abs(T.term(0) - (-286.0)) < 1.0, 'n=0 rơi vào Xuân Phân 1999',
   `(${T.term(0).toFixed(3)})`);

console.log('\n3. Các seam THẬT SỰ tra bảng (không phải chỉ nạp rồi bỏ đó)');
for (const [name, w, expect] of [
    ['shuoHigh Sóc  k=0', 0, T.newMoon(0)],
    ['shuoHigh Vọng k=0', Math.PI, T.fullMoon(0)],
    ['shuoHigh Vọng k=-3', -3 * 2 * Math.PI + Math.PI, T.fullMoon(-3)],
    ['qiAccurate n=0', 0, T.term(0)],
    ['qiAccurate n=40', 40 * Math.PI / 12, T.term(40)],
]) {
    const got = name.startsWith('qi')
        ? toTT(ShouXingUtil.qiAccurate(w, TZ))
        : toTT(ShouXingUtil.shuoHigh(w, TZ));
    ok(Math.abs(got - expect) * 86400 < 0.2, name,
       `lệch ${((got - expect) * 86400).toFixed(3)}s so với bảng`);
}

console.log('\n4. Mốc vàng từ oracle DE423 (ngày TT kể từ J2000)');
const GOLD = [
    // n suy từ mốc TT: n = round((t + 286)/15.2184); hoàng kinh 15n mod 360
    ['term', -2383, -36549.205205],   // 255 deg
    ['term', -1683, -25898.202003],   // 315 deg
    ['term',  -983, -15246.199399],   //  15 deg
    ['term',  -283,  -4592.277853],   //  75 deg
    ['term',   417,   6062.579287],   // 135 deg
    ['newMoon', -1237, -36523.922269], ['newMoon', -737, -21758.454642],
    ['newMoon',  -237,  -6993.636541], ['newMoon',  263,   7771.605556],
    ['fullMoon', -1237, -36509.703171], ['fullMoon', -637, -18791.394789],
    ['fullMoon',   -37,  -1072.866840], ['fullMoon',  563,  16645.424890],
];
for (const [fn, idx, want] of GOLD) {
    const got = T[fn](idx);
    ok(got !== null && Math.abs(got - want) * 86400 < 0.1,
       `${fn}(${idx})`, `lệch ${got === null ? 'null' : ((got - want) * 86400).toFixed(3) + 's'}`);
}

console.log('\n5. Phương trình thời gian — nền của Chính Ngọ');
// EoT là thứ DUY NHẤT Chính Ngọ cần từ thiên văn: phần kinh độ trong công thức
// là số học thuần tuý, nên một bảng toàn cục phục vụ mọi địa điểm.
{
    ok(typeof T.eot === 'function', 'bảng có mục EoT');
    ok(T.eot(-36600) !== null && T.eot(36900) !== null, 'EoT phủ 1900-2100');
    ok(T.eot(-99999) === null, 'ngoài khoảng trả null');

    // Mốc trong implementation_prompt.md: EoT = biểu kiến trừ trung bình,
    // ~ +16,4 phút đầu tháng 11 và ~ -14,2 phút giữa tháng 2.
    const dayOf = (y, m, d) => {
        const jd = Date.UTC(y, m - 1, d, 12) / 86400000 + 2440587.5;
        return jd - 2451545.0;
    };
    const nov = T.eot(dayOf(2024, 11, 3));
    const feb = T.eot(dayOf(2024, 2, 12));
    ok(Math.abs(nov - 16.4) < 0.2, 'cực đại đầu tháng 11 ~ +16,4 phút',
       `(${nov.toFixed(2)})`);
    ok(Math.abs(feb + 14.2) < 0.2, 'cực tiểu giữa tháng 2 ~ -14,2 phút',
       `(${feb.toFixed(2)})`);

    // astro.js phải THẬT SỰ dùng bảng chứ không lặng lẽ rơi về Meeus.
    const jd = 2451545.0 + dayOf(2024, 11, 3);
    const viaAstro = Astro.equationOfTime(jd);
    ok(Math.abs(viaAstro - T.eot(dayOf(2024, 11, 3))) < 0.001,
       'Astro.equationOfTime tra bảng', `(lệch ${((viaAstro - nov) * 60).toFixed(3)}s)`);
    ok(Math.abs(viaAstro - Astro.sunPosition(jd).eot) > 1e-6,
       'và khác hẳn chuỗi Meeus cũ',
       `(Meeus lệch ${((Astro.sunPosition(jd).eot - nov) * 60).toFixed(3)}s)`);
}

console.log('\n6. Kết cấu LỊCH không đổi — bảng chỉ được đổi MỐC');
// Bảng DE423 dịch tiết khí vài giây và điểm Sóc vài phút. Chừng ấy KHÔNG được
// phép đổi tháng nhuận hay mùng 1: tháng nhuận là tháng không có trung khí, một
// quy tắc của lunar.js áp lên mốc, và mốc chính xác hơn chỉ làm đầu vào tốt hơn.
// Dãy dưới đây chốt cứng: mỗi năm 1900-2100 một chữ số, 0 = không nhuận.
const LEAP_1900_2100 =
    '8005004002060050020700500400206005003070060040020700500308006004003070050040800600400207005003' +
    '0800500400207005004090060040020600500301100600500207005003080060040030700500408006004003070050' +
    '04080060040020';
{
    const got = [];
    for (let y = 1900; y <= 2100; y++) {
        let L = 0;
        for (const m of LunarYear.fromYear(y).getMonths()) {
            if (m.getYear() === y && m.getMonth() < 0) L = -m.getMonth();
        }
        got.push(L);
    }
    const s = got.join('');
    ok(s.length === LEAP_1900_2100.length && s === LEAP_1900_2100,
       'tháng nhuận 1900-2100 y nguyên',
       s === LEAP_1900_2100 ? `(${got.filter(Boolean).length} năm nhuận)`
                            : `lệch ở năm ${1900 + [...s].findIndex((c, i) => c !== LEAP_1900_2100[i])}`);
}

console.log('\n7. index.html nạp bảng TRƯỚC lunar.js');
const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
const iTable = html.indexOf('js/astro_table.js');
const iLunar = html.indexOf('js/lunar.js');
ok(iTable >= 0, 'index.html có thẻ script cho astro_table.js');
ok(iTable >= 0 && iTable < iLunar, 'đứng trước lunar.js');

console.log(fail ? `\n✗ ${fail} phép canh hỏng` : '\n✓ tất cả đạt — bảng DE423 đang chạy');
process.exit(fail ? 1 : 0);
