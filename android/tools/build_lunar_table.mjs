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
 *   dòng 2+: <JDN điểm Sóc> <giây trong ngày> <năm âm> <tháng âm> <1 nếu nhuận>
 *            <lệch ngày Sóc hiện> <phút trong ngày> <lệch ngày Vọng> <phút>
 *
 * Bốn cột cuối là mốc Sóc/Vọng ĐỂ HIỆN trong bảng "Lịch âm" — KHÔNG phải cột
 * <giây trong ngày> ở đầu dòng. Hai thứ khác nhau thật, không phải chép thừa:
 *
 *   • Cột giây đầu dòng là NGƯỠNG BẬC THANG của lunar.js: mốc mà tại đó mùng 1
 *     nhảy sang ngày khác khi đổi múi giờ. Nó định NGÀY nào là mùng 1.
 *   • Bốn cột cuối là điểm Sóc/Vọng thiên văn (`shuoHigh` ở mốc UTC+8, đúng
 *     hàm mà Ephem.socSolar/vongSolar của ứng dụng gọi) — thứ tab Lịch IN RA.
 *
 * Đo thử 2000–2050: hai con số lệch nhau ở 2,4% số tháng, có ca lệch tới 17
 * giờ. Dùng nhầm cột là widget in giờ Sóc khác hẳn ứng dụng.
 *
 * Ghi ĐIỂM SÓC chứ không ghi sẵn ngày mùng 1: mùng 1 là ngày CHỨA điểm Sóc, mà
 * "ngày" nào thì tuỳ múi giờ người xem. Ứng dụng tính lịch âm theo múi giờ của
 * địa điểm đang chọn, nên widget cũng phải vậy — nếu chốt cứng mùng 1 ở đây thì
 * ở Paris widget hiện mùng 1 = 13/08 trong khi ứng dụng hiện 12/08.
 *
 * Mốc ghi trong tệp là UTC+7; Kotlin quy đổi sang múi giờ đang chọn (xem
 * LunarTable.localize). Ngày can chi không phụ thuộc múi giờ nên tính thẳng
 * từ JDN.
 */
import fs from 'fs';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const require = createRequire(import.meta.url);
const { Solar, LunarYear, ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));

/* ── Mốc Sóc / Vọng ĐỂ HIỆN ──
   Chép đúng hai hàm của ephem.js (socSolar / vongSolar) chứ không gọi vào đó:
   ephem.js là tệp trình duyệt, nạp được ở node nhưng kéo theo cả astro.js.
   Hai hàm này chỉ có ngần này, và `test_lunar_table.mjs` canh lại kết quả với
   chính ephem.js nên bản chép không thể lặng lẽ trôi đi. */
const J2000 = Solar.J2000;
const lunation = solar => Math.round((solar.getJulianDay() + 0.5 - J2000) / 29.5306);
const socSolar = (solar, basis) =>
    Solar.fromJulianDay(ShouXingUtil.shuoHigh(lunation(solar) * 2 * Math.PI, basis) + J2000);
const vongSolar = (solar, basis) =>
    Solar.fromJulianDay(
        ShouXingUtil.shuoHigh(lunation(solar) * 2 * Math.PI + Math.PI, basis) + J2000);

/** Chỉ số 0–59 của một trụ can chi viết bằng chữ Hán, vd "丙寅" → 2. */
const CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
const CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];
function ganZhiIndex(gz) {
    const can = CAN_ZH.indexOf(gz[0]);
    const chi = CHI_ZH.indexOf(gz[1]);
    if (can < 0 || chi < 0) return -1;
    // Chu kỳ 60: chỉ số i thoả i%10 == can và i%12 == chi. Chỉ có một nghiệm.
    for (let i = 0; i < 60; i++) if (i % 10 === can && i % 12 === chi) return i;
    return -1;
}

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

/* ── Điểm Sóc của từng tháng âm ──
   KHÔNG tính lại điểm Sóc bằng công thức riêng. Thử rồi: cách đó lệch với mốc
   mùng 1 của chính lunar.js ở 0,81% số tháng (luôn là các ca Sóc rơi sát nửa
   đêm, kiểu 23:39 hay 00:04) — tức bảng của widget sẽ đặt mùng 1 khác ứng dụng
   đúng những tháng ấy.

   Thay vào đó, HỎI CHÍNH lunar.js. Mốc mùng 1 mà nó trả về là một hàm bậc
   thang theo múi giờ: giảm dần offset thì tới đúng lúc điểm Sóc lùi qua nửa
   đêm, mùng 1 tụt một ngày. Chia đôi khoảng để tìm ngưỡng ấy là ra giờ-phút của
   điểm Sóc, khớp với lunar.js theo đúng định nghĩa.                          */
const TZ_BASE = 7;

/** Ngày dương (JDN) của mùng 1 tháng âm (lunarYear, lunarMonth) ở mốc `tz`. */
function monthStartAt(lunarYear, lunarMonth, tz) {
    ShouXingUtil.setTzOffsetHours(tz);
    let jd = null;
    for (const mo of LunarYear.fromYear(lunarYear).getMonths()) {
        if (mo.getYear() === lunarYear && mo.getMonth() === lunarMonth) {
            jd = mo.getFirstJulianDay();
            break;
        }
    }
    ShouXingUtil.setTzOffsetHours(null);
    if (jd === null) return null;
    const s = Solar.fromJulianDay(jd);
    return jdn(s.getYear(), s.getMonth(), s.getDay());
}

/**
 * Giờ-phút của điểm Sóc ở mốc TZ_BASE, suy từ ngưỡng bậc thang của lunar.js.
 *
 * monthStart(u) = base + floor((socSec + (u − 7)·3600) / 86400), nên chỉ cần tìm
 * MỘT chỗ nhảy là ra socMin. Có hai chỗ:
 *   nhảy −1→0 tại u = 7 − socSec/3600        (trong [−11, 7] khi socSec ≤ 18h)
 *   nhảy  0→1 tại u = 7 + (86400−socSec)/3600 (trong [7, 14] khi socSec ≥ 17h)
 * Hai khoảng phủ kín 24 giờ, và cả hai đều nằm trong dải offset mà lunar.js
 * còn tính đúng — dò thẳng xuống −17 như lần đầu thì nó trả về rỗng và phép
 * chia đôi bám mãi ở 23:59.
 */
const clampSec = v => Math.min(86399, Math.max(0, v));
function socMinuteOfDay(lunarYear, lunarMonth, base) {
    const at = u => monthStartAt(lunarYear, lunarMonth, u);
    if (at(-11) < base) {
        let lo = -11, hi = TZ_BASE;               // lo: base−1, hi: base
        for (let i = 0; i < 28; i++) {          // 24h / 2^28 ≈ 0,3 ms
            const mid = (lo + hi) / 2;
            if (at(mid) < base) lo = mid; else hi = mid;
        }
        return clampSec(Math.round((TZ_BASE - (lo + hi) / 2) * 3600));
    }
    let lo = TZ_BASE, hi = 14;                    // lo: base, hi: base+1
    for (let i = 0; i < 28; i++) {
        const mid = (lo + hi) / 2;
        if (at(mid) > base) hi = mid; else lo = mid;
    }
    return clampSec(Math.round(86400 - ((lo + hi) / 2 - TZ_BASE) * 3600));
}

const months = [];
const seenMonth = new Set();
for (let y = Y0 - 1; y <= Y1 + 1; y++) {
    ShouXingUtil.setTzOffsetHours(TZ_BASE);
    const list = LunarYear.fromYear(y).getMonths()
        .filter(mo => mo.getYear() === y)
        .map(mo => [mo.getMonth(), mo.getFirstJulianDay()]);
    ShouXingUtil.setTzOffsetHours(null);
    for (const [lm, jdFirst] of list) {
        const key = y + ':' + lm;
        if (seenMonth.has(key)) continue;
        seenMonth.add(key);
        const s = Solar.fromJulianDay(jdFirst);
        if (s.getYear() < Y0 - 1 || s.getYear() > Y1 + 1) continue;
        const base = jdn(s.getYear(), s.getMonth(), s.getDay());
        // Mốc để HIỆN: giải ở UTC+8 như ứng dụng, rồi hạ về mốc UTC+7 của tệp.
        // Cắt xuống PHÚT ngay ở đây được, vì mọi múi giờ đều lệch tròn phút nên
        // cộng offset sau này không đổi phần giây.
        const at7 = solar8 => {
            const local = Solar.fromJulianDay(solar8.getJulianDay() - 1 / 24);
            return [jdn(local.getYear(), local.getMonth(), local.getDay()),
                    local.getHour() * 60 + local.getMinute()];
        };
        const [socJdn, socMin] = at7(socSolar(s, 8));
        const [vongJdn, vongMin] = at7(vongSolar(s, 8));
        months.push([
            base, socMinuteOfDay(y, lm, base), y, Math.abs(lm), lm < 0 ? 1 : 0,
            socJdn - base, socMin, vongJdn - socJdn, vongMin,
        ]);
    }
}
months.sort((a, b) => a[0] - b[0]);

/* Mốc can chi Giáp Tý — không phụ thuộc múi giờ. */
let ganziEpoch = null;
outer:
for (let m = 1; m <= 3; m++) {
    const dim = new Date(1900, m, 0).getDate();
    for (let d = 1; d <= dim; d++) {
        if (Solar.fromYmd(1900, m, d).getLunar().getDayInGanZhi() === '甲子') {
            ganziEpoch = jdn(1900, m, d);
            break outer;
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
              <chỉ số 0–59 của can chi THÁNG>

   Cột can chi tháng để widget dựng được cột thứ ba của bảng Tiết khí, đúng như
   tab Lịch. Hỏi thẳng lunar.js (getEightChar().getMonth()) chứ không suy từ chỉ
   số tiết khí: can tháng phụ thuộc can năm, mà năm can chi lại đổi ở Lập Xuân —
   dựng lại luật ấy bằng tay là mời thêm một nguồn lệch nữa với ứng dụng.
   Nhích 2 phút qua mốc giao tiết y như monthGanZhiAt trong calendar.js: đúng
   tại mốc, phép làm tròn trong lunar.js còn có thể xếp về tháng cũ.

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
        const gz = ganZhiIndex(
            Solar.fromJulianDay(jds[i + 1] + 2 / 1440).getLunar().getEightChar().getMonth());
        if (gz < 0) throw new Error(`can chi tháng lạ ở tiết khí ${i} năm ${Y}`);
        jq.push([
            jdn(y, local.getMonth(), local.getDay()),
            local.getHour() * 60 + local.getMinute(),
            i,
            gz,
        ]);
    }
}
ShouXingUtil.setTzOffsetHours(7);             // trả lại mốc lịch Việt Nam

jq.sort((a, b) => a[0] - b[0] || a[1] - b[1]);
const jqDest = path.join(WEB, '..', 'jieqi.txt');
fs.writeFileSync(jqDest, jq.map(r => r.join(' ')).join('\n') + '\n');

console.log(`${months.length} tháng âm (điểm Sóc, mốc UTC+7)`);
console.log(`mốc can chi Giáp Tý: JDN ${ganziEpoch}`);
console.log(`${fs.statSync(dest).size.toLocaleString()} bytes → ${path.basename(dest)}`);
console.log(`${jq.length} tiết khí · ${fs.statSync(jqDest).size.toLocaleString()} bytes → ${path.basename(jqDest)}`);
