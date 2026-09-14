/**
 * Kiểm thử astro.js — đối chiếu với giá trị tham chiếu tính bằng PyEphem
 * (thư viện thiên văn độ chính xác cao, độc lập hoàn toàn với mã trong repo).
 *
 * Chạy:  node android/tools/test_astro.mjs
 *
 * Giá trị tham chiếu sinh bằng:
 *     pip install ephem
 *     ephem.Observer(lat, lon) → previous_rising / next_setting / moon_phase
 *     ephem.next_new_moon / next_full_moon
 * Đơn vị: phút kể từ 00:00 giờ ĐỊA PHƯƠNG của toạ độ đó.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASTRO = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web', 'js', 'astro.js');

const g = globalThis;
new Function('window', fs.readFileSync(ASTRO, 'utf8'))(g);
const A = g.Astro;

// date, lat, lon, tz, {sunrise, sunset, moonrise, moonset, illum}  ← PyEphem
const CASES = [
  ['Hanoi',     [2026, 8, 25],  21.0278,  105.8342,  7, 338.182, 1099.230,  991.325,  166.594, 0.915],
  ['HCMC',      [1988, 3, 15],  10.8231,  106.6297,  7, 360.533, 1084.474,  212.090,  926.300, 0.118],
  ['Paris',     [2026, 6, 21],  48.8566,    2.3522,  2, 346.570, 1318.240,  809.755,   83.363, 0.452],
  ['New York',  [2026, 12, 21], 40.7128,  -74.0060, -5, 436.216,  992.162,  839.696,  275.089, 0.918],
  ['Beijing',   [2000, 2, 4],   39.9042,  116.4074,  8, 440.568, 1056.306,  385.943,  983.977, 0.018],
  ['Sydney',    [2026, 9, 30], -33.8688,  151.2093, 10, 333.983, 1077.128, 1322.537,  433.009, 0.859],
  ['Reykjavik', [2026, 6, 21],  64.1466,  -21.9426,  0, 173.867, 1445.298,  793.096,   71.777, 0.460],
  ['Nairobi',   [2026, 11, 11], -1.2921,   36.8219,  3, 371.350, 1102.110,  459.888, 1210.706, 0.043],
  ['Anchorage', [2026, 3, 20],  61.2181, -149.9003, -8, 479.481, 1215.818,  473.382, 1434.192, 0.043],
];

// Ở vĩ độ cao mặt trời cắt chân trời rất thoải, nên chênh lệch mô hình khúc xạ
// biến thành vài chục giây; nới dung sai cho hai mốc đó.
const TOL_SUN = 1.5, TOL_MOON = 2.0, TOL_ILLUM = 0.01;

let fail = 0, checks = 0;
function check(label, got, want, tol, unit) {
    checks++;
    const d = got - want;
    const ok = Math.abs(d) <= tol;
    if (!ok) fail++;
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(9)} got=${got.toFixed(3).padStart(9)}` +
                ` ref=${want.toFixed(3).padStart(9)} Δ=${d >= 0 ? '+' : ''}${d.toFixed(3)} ${unit}`);
}

for (const [name, [y, m, d], lat, lon, tz, sr, ss, mr, ms, illum] of CASES) {
    console.log(name);
    const st = A.sunTimes(y, m, d, lat, lon, tz);
    const mt = A.moonTimes(y, m, d, lat, lon, tz);
    const jdNoon = A.jdFromUTC(y, m, d, 12, 0, 0) - tz / 24;
    check('sunrise', st.sunrise, sr, TOL_SUN, 'min');
    check('sunset', st.sunset, ss, TOL_SUN, 'min');
    check('moonrise', mt.moonrise, mr, TOL_MOON, 'min');
    check('moonset', mt.moonset, ms, TOL_MOON, 'min');
    check('illum', A.moonIllumination(jdNoon).fraction, illum, TOL_ILLUM, 'frac');
}

// Chính Ngọ phải luôn khớp định nghĩa: giờ Mặt Trời thật tại đó đúng 12:00.
console.log('Chính Ngọ ↔ giờ Mặt Trời thật');
for (const [name, [y, m, d], , lon, tz] of CASES) {
    const noon = A.solarNoonMinutes(y, m, d, lon, tz);
    const eot = A.equationOfTime(A.jdFromUTC(y, m, d) + (noon - tz * 60) / 1440);
    const tst = A.toTrueSolarMinutes(noon, lon, tz, eot);
    check(name.slice(0, 9), tst, 720, 0.02, 'min');
}

console.log(`\n${checks - fail}/${checks} phép kiểm đạt.`);
process.exit(fail ? 1 : 0);
