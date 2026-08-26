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
const { Solar, LunarYear, ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));

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
   Widget cũng phải hiện bảng tiết khí, mà giờ giao tiết thì lunar.js mới tính
   được. Tính sẵn ra bảng: mỗi dòng một tiết khí.
   Định dạng: <JDN> <phút kể từ 00:00 giờ địa phương> <số thứ tự tên>

   NGUYÊN TẮC TÍNH phải trùng KHÍT với bảng Sách Bổ pháp trong ứng dụng
   (`sb_getJieQiDates` trong js/app.js) — cả app lẫn widget đều hiện tiết khí
   thì không được có hai con số khác nhau. Nghĩa là:

     1. `LunarYear.getJieQiJulianDays()` chứ không phải `getJieQiTable()`;
     2. tính ở mốc UTC+8 (`setTzOffsetHours(null)`), không phải UTC+7;
     3. rồi mới quy đổi sang giờ địa phương bằng đúng phép của
        `_formatJdUTC8ToLocal`: jdLocal = jdUTC8 + (tz − 8)/24.

   Việt Nam là UTC+7 quanh năm, không có DST, nên bước 3 luôn là trừ đúng một
   giờ. Nếu đổi widget sang múi giờ khác thì phải tra DST theo từng mốc như
   `_tzOffsetAtJdUTC8` làm.                                                  */
const TZ_WIDGET = 7;

ShouXingUtil.setTzOffsetHours(null);          // mốc UTC+8 như Sách Bổ pháp
const jq = [];
for (let Y = Y0 - 1; Y <= Y1; Y++) {
    const jds = LunarYear.fromYear(Y + 1).getJieQiJulianDays();
    // chỉ số 1..24 = 冬至(Y), 小寒, …, 大雪(Y)
    for (let i = 0; i < 24; i++) {
        const local = Solar.fromJulianDay(jds[i + 1] + (TZ_WIDGET - 8) / 24);
        const y = local.getYear();
        if (y < Y0 || y > Y1) continue;
        jq.push([
            jdn(y, local.getMonth(), local.getDay()),
            local.getHour() * 60 + local.getMinute(),
            i,
        ]);
    }
}
ShouXingUtil.setTzOffsetHours(7);             // trả lại mốc lịch Việt Nam

jq.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
const jqDest = path.join(WEB, '..', 'jieqi.txt');
fs.writeFileSync(jqDest, jq.map(r => r.join(' ')).join('\n') + '\n');

console.log(`${months.length} tháng âm (${Y0}–${Y1})`);
console.log(`mốc can chi Giáp Tý: JDN ${ganziEpoch}`);
console.log(`${fs.statSync(dest).size.toLocaleString()} bytes → ${path.basename(dest)}`);
console.log(`${jq.length} tiết khí · ${fs.statSync(jqDest).size.toLocaleString()} bytes → ${path.basename(jqDest)}`);
