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
const { ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));

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

console.log('\n5. index.html nạp bảng TRƯỚC lunar.js');
const html = fs.readFileSync(path.join(WEB, 'index.html'), 'utf8');
const iTable = html.indexOf('js/astro_table.js');
const iLunar = html.indexOf('js/lunar.js');
ok(iTable >= 0, 'index.html có thẻ script cho astro_table.js');
ok(iTable >= 0 && iTable < iLunar, 'đứng trước lunar.js');

console.log(fail ? `\n✗ ${fail} phép canh hỏng` : '\n✓ tất cả đạt — bảng DE423 đang chạy');
process.exit(fail ? 1 : 0);
