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
const { Solar, LunarYear, ShouXingUtil } = require(path.join(ASSETS, 'web', 'js', 'lunar.js'));
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

/** Ngày dương lịch từ JDN — bản sao của LunarTable.civilOf, phép nghịch của jdnOf. */
function civilOf(jdn) {
    const a = jdn + 32044;
    const b = Math.floor((4 * a + 3) / 146097);
    const c = a - Math.floor(146097 * b / 4);
    const d = Math.floor((4 * c + 3) / 1461);
    const e = c - Math.floor(1461 * d / 4);
    const m = Math.floor((5 * e + 2) / 153);
    return {
        d: e - Math.floor((153 * m + 2) / 5) + 1,
        m: m + 3 - 12 * Math.floor(m / 10),
        y: 100 * b + d - 4800 + Math.floor(m / 10),
    };
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
            const back = civilOf(j);
            const ok = got &&
                back.y === y && back.m === m && back.d === d &&
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

/* ── Bảng tiết khí của widget ── */
const TK_ZH = ['冬至','小寒','大寒','立春','雨水','惊蛰','春分','清明','谷雨','立夏','小满','芒种',
    '夏至','小暑','大暑','立秋','处暑','白露','秋分','寒露','霜降','立冬','小雪','大雪'];
const jqLines = fs.readFileSync(path.join(ASSETS, 'jieqi.txt'), 'utf8').trim().split('\n');
const jqMap = new Map();                       // "jdn:idx" -> phút
for (const line of jqLines) {
    const [j, mins, idx] = line.split(' ').map(Number);
    jqMap.set(j + ':' + idx, mins);
}

let jqChecked = 0, jqBad = 0;

// Đối chiếu theo ĐÚNG nguyên tắc của bảng Sách Bổ pháp (sb_getJieQiDates trong
// js/app.js): getJieQiJulianDays() ở mốc UTC+8, rồi quy sang giờ địa phương
// bằng jdLocal = jdUTC8 + (tz − 8)/24. Nếu dựng bảng bằng đường khác
// (getJieQiTable ở mốc UTC+7 chẳng hạn) thì widget và ứng dụng có thể hiện hai
// giờ khác nhau cho cùng một tiết khí — chính là thứ phép thử này canh.
const TZ_WIDGET = 7;
ShouXingUtil.setTzOffsetHours(null);
for (let Y = 1899; Y <= 2100; Y++) {
    const jds = LunarYear.fromYear(Y + 1).getJieQiJulianDays();
    for (let i = 0; i < 24; i++) {
        const sol = Solar.fromJulianDay(jds[i + 1] + (TZ_WIDGET - 8) / 24);
        if (sol.getYear() < 1900 || sol.getYear() > 2100) continue;
        jqChecked++;
        const j = jdnOf(sol.getYear(), sol.getMonth(), sol.getDay());
        const want = sol.getHour() * 60 + sol.getMinute();
        const got = jqMap.get(j + ':' + i);
        if (got !== want) {
            jqBad++;
            if (jqBad <= 4) {
                console.log(`  LỆCH tiết khí ${TK_ZH[i]} ${sol.toYmdHms()}: bảng=${got} lunar.js=${want}`);
            }
        }
    }
}
ShouXingUtil.setTzOffsetHours(7);

console.log(`Đã so ${jqChecked.toLocaleString()} mốc tiết khí · lệch ${jqBad}`);
bad += jqBad;

console.log(`Đã so ${checked.toLocaleString()} ngày (1900–2100)`);
console.log(`Lệch: ${bad}`);
samples.forEach(s => console.log('  ' + s));
console.log(bad ? '\n✗ BẢNG TRA SAI' : '\n✓ Bảng tra khớp lunar.js từng ngày một');
process.exit(bad ? 1 : 0);
