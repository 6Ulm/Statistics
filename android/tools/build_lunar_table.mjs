/**
 * Sinh bảng tra âm lịch cho widget màn hình chính.
 *
 *   node build_lunar_table.mjs
 *
 * Widget chạy bằng RemoteViews nên KHÔNG dùng được WebView — toàn bộ lunar.js
 * không với tới được. Thay vì chép lại thuật toán tính điểm Sóc sang Kotlin
 * (dễ sai, khó kiểm), ta tính sẵn ở đây rồi đóng gói thành bảng: mỗi tháng âm
 * một dòng, Kotlin chỉ việc tra.
 *
 * Định dạng (assets/lunar_months.txt):
 *   dòng 1 : GANZHI_EPOCH  — số ngày Julius có can chi "Giáp Tý"
 *   dòng 2+: <JDN mùng 1> <năm âm> <tháng âm> <1 nếu nhuận>
 *
 * Mốc múi giờ: UTC+7, đúng như lịch Việt Nam (xem setLunarBasis trong
 * calendar.js). Ngày can chi không phụ thuộc múi giờ nên tính thẳng từ JDN.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const require = createRequire(import.meta.url);
const { Solar, ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));

ShouXingUtil.setTzOffsetHours(7);          // cơ sở lịch Việt Nam

const Y0 = 1900, Y1 = 2100;

/** Số ngày Julius của một ngày dương lịch (thuật toán Fliegel–Van Flandern). */
function jdn(y, m, d) {
    const a = Math.floor((14 - m) / 12);
    const yy = y + 4800 - a;
    const mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}

/* ── Quét từng ngày, ghi lại mốc mùng 1 của mỗi tháng âm ── */
const months = [];
let ganziEpoch = null;

for (let y = Y0; y <= Y1; y++) {
    for (let m = 1; m <= 12; m++) {
        const days = new Date(y, m, 0).getDate();
        for (let d = 1; d <= days; d++) {
            const lunar = Solar.fromYmd(y, m, d).getLunar();
            const j = jdn(y, m, d);
            if (lunar.getDay() === 1) {
                const lm = lunar.getMonth();
                months.push([j, lunar.getYear(), Math.abs(lm), lm < 0 ? 1 : 0]);
            }
            if (ganziEpoch === null && lunar.getDayInGanZhi() === '甲子') {
                ganziEpoch = j;      // mốc để suy can chi từ JDN
            }
        }
    }
}

const out = [String(ganziEpoch)]
    .concat(months.map(r => r.join(' ')))
    .join('\n') + '\n';
const dest = path.join(WEB, '..', 'lunar_months.txt');
fs.writeFileSync(dest, out);

/* ── Tiết khí ──
   Widget cũng phải hiện bảng tiết khí y như tab Lịch, mà giờ giao tiết thì
   lunar.js mới tính được. Tính sẵn ra bảng: mỗi dòng một tiết khí.
   Định dạng: <JDN> <phút kể từ 00:00 giờ địa phương> <số thứ tự tên>       */
const TK_ZH = ['冬至', '小寒', '大寒', '立春', '雨水', '惊蛰', '春分', '清明',
    '谷雨', '立夏', '小满', '芒种', '夏至', '小暑', '大暑', '立秋',
    '处暑', '白露', '秋分', '寒露', '霜降', '立冬', '小雪', '大雪'];
const idxOf = Object.fromEntries(TK_ZH.map((n, i) => [n, i]));

const seen = new Set();
const jq = [];
for (let y = Y0; y <= Y1; y++) {
    const table = Solar.fromYmd(y, 6, 15).getLunar().getJieQiTable();
    for (const name in table) {
        const idx = idxOf[name];
        if (idx === undefined) continue;          // bỏ các mốc phụ của lunar.js
        const s = table[name];
        if (s.getYear() < Y0 || s.getYear() > Y1) continue;
        const j = jdn(s.getYear(), s.getMonth(), s.getDay());
        const key = j + ':' + idx;
        if (seen.has(key)) continue;
        seen.add(key);
        jq.push([j, s.getHour() * 60 + s.getMinute(), idx]);
    }
}
jq.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
const jqDest = path.join(WEB, '..', 'jieqi.txt');
fs.writeFileSync(jqDest, jq.map(r => r.join(' ')).join('\n') + '\n');

console.log(`${months.length} tháng âm (${Y0}–${Y1})`);
console.log(`mốc can chi Giáp Tý: JDN ${ganziEpoch}`);
console.log(`${fs.statSync(dest).size.toLocaleString()} bytes → ${path.basename(dest)}`);
console.log(`${jq.length} tiết khí · ${fs.statSync(jqDest).size.toLocaleString()} bytes → ${path.basename(jqDest)}`);
