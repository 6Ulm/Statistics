/**
 * Đối chiếu bảng tra âm lịch của widget với lunar.js, TỪNG NGÀY một trong
 * 1900–2100 (~73.400 ngày).
 *
 *   node test_lunar_table.mjs
 *
 * Thuật toán dưới đây là bản sao đúng từng bước của LunarTable.kt. Widget
 * không chạy được lunar.js, nên đây là chỗ duy nhất chứng minh được bảng tra
 * và cách tra cho ra cùng kết quả với engine trong ứng dụng.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');
const require = createRequire(import.meta.url);
const { Solar, ShouXingUtil } = require(path.join(ASSETS, 'web', 'js', 'lunar.js'));
ShouXingUtil.setTzOffsetHours(7);

const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
const CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
const CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

/* ── bản sao của LunarTable.kt ── */
const lines = fs.readFileSync(path.join(ASSETS, 'lunar_months.txt'), 'utf8').trim().split('\n');
const GANZHI_EPOCH = parseInt(lines[0], 10);
const starts = new Int32Array(lines.length - 1);
const lyear = new Int16Array(lines.length - 1);
const lmonth = new Int8Array(lines.length - 1);
const leap = new Int8Array(lines.length - 1);
for (let i = 1; i < lines.length; i++) {
    const p = lines[i].split(' ');
    starts[i - 1] = +p[0]; lyear[i - 1] = +p[1]; lmonth[i - 1] = +p[2]; leap[i - 1] = +p[3];
}

function jdnOf(y, m, d) {
    const a = Math.floor((14 - m) / 12);
    const yy = y + 4800 - a;
    const mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}

/** Tìm tháng âm chứa ngày jdn — tìm nhị phân mốc mùng 1 lớn nhất mà ≤ jdn. */
function lookup(jdn) {
    let lo = 0, hi = starts.length - 1, idx = -1;
    while (lo <= hi) {
        const mid = (lo + hi) >> 1;
        if (starts[mid] <= jdn) { idx = mid; lo = mid + 1; } else { hi = mid - 1; }
    }
    if (idx < 0) return null;
    return {
        day: jdn - starts[idx] + 1,
        month: lmonth[idx],
        year: lyear[idx],
        leap: leap[idx] === 1,
    };
}

function ganZhi(jdn) {
    const i = (((jdn - GANZHI_EPOCH) % 60) + 60) % 60;
    return { can: i % 10, chi: i % 12 };
}

/* ── so từng ngày ── */
let checked = 0, bad = 0;
const samples = [];
for (let y = 1900; y <= 2100; y++) {
    for (let m = 1; m <= 12; m++) {
        const days = new Date(y, m, 0).getDate();
        for (let d = 1; d <= days; d++) {
            const j = jdnOf(y, m, d);
            const got = lookup(j);
            const gz = ganZhi(j);
            const ref = Solar.fromYmd(y, m, d).getLunar();
            checked++;
            const refLeap = ref.getMonth() < 0;
            const ok = got &&
                got.day === ref.getDay() &&
                got.month === Math.abs(ref.getMonth()) &&
                got.year === ref.getYear() &&
                got.leap === refLeap &&
                CAN_ZH[gz.can] === ref.getDayGan() &&
                CHI_ZH[gz.chi] === ref.getDayZhi();
            if (!ok) {
                bad++;
                if (samples.length < 6) {
                    samples.push(`${d}/${m}/${y}: bảng ${got ? got.day + '/' + got.month + '/' + got.year +
                        (got.leap ? 'N' : '') + ' ' + CAN[gz.can] + ' ' + CHI[gz.chi] : 'không tra được'}` +
                        ` | lunar.js ${ref.getDay()}/${Math.abs(ref.getMonth())}/${ref.getYear()}` +
                        `${refLeap ? 'N' : ''} ${ref.getDayGan()}${ref.getDayZhi()}`);
                }
            }
        }
    }
}

console.log(`Đã so ${checked.toLocaleString()} ngày (1900–2100)`);
console.log(`Lệch: ${bad}`);
samples.forEach(s => console.log('  ' + s));
console.log(bad ? '\n✗ BẢNG TRA SAI' : '\n✓ Bảng tra khớp lunar.js từng ngày một');
process.exit(bad ? 1 : 0);
