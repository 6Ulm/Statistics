/**
 * Mọi bảng tiết khí trong ứng dụng phải tính theo CÙNG MỘT NGUYÊN TẮC — nguyên
 * tắc của bảng Sách Bổ pháp.
 *
 *   node test_jieqi_parity.mjs
 *
 * Phép thử so bảng tiết khí ở tab Lịch với bảng Sách Bổ pháp ở tab Kỳ Môn,
 * từng tên và từng mốc giờ một, ở vài múi giờ khác nhau.
 *
 * Hai bảng dùng chung `sb_getJieQiDates` nên đáng lẽ không thể lệch — nhưng
 * chúng chạy ở hai trạng thái múi giờ TOÀN CỤC khác nhau (tab Lịch đặt
 * ShouXingUtil về UTC+7 để lịch âm ra đúng lịch Việt Nam), mà `findJieQi` bên
 * trong lại đọc chính biến toàn cục đó. Phép thử này canh đúng chỗ ấy.
 *
 * Ca thứ ba là WIDGET: nó không chạy JavaScript mà tra bảng jieqi.txt, và bảng
 * đó lưu giờ ở UTC+7 rồi mới quy sang múi giờ của địa điểm đang chọn. Phép thử
 * dựng lại đúng phép quy đổi ấy rồi so với bảng trên màn hình — ba nơi hiện
 * tiết khí thì cả ba phải ra cùng một con số.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');
const WEB = path.join(ASSETS, 'web');

/* ── Bản dựng lại LunarTable.jieQiYearOf + localize bằng JavaScript ── */
const TK_VI_TABLE = ['Đông Chí', 'Tiểu Hàn', 'Đại Hàn', 'Lập Xuân', 'Vũ Thủy', 'Kinh Trập',
    'Xuân Phân', 'Thanh Minh', 'Cốc Vũ', 'Lập Hạ', 'Tiểu Mãn', 'Mang Chủng',
    'Hạ Chí', 'Tiểu Thử', 'Đại Thử', 'Lập Thu', 'Xử Thử', 'Bạch Lộ',
    'Thu Phân', 'Hàn Lộ', 'Sương Giáng', 'Lập Đông', 'Tiểu Tuyết', 'Đại Tuyết'];
const JQ = fs.readFileSync(path.join(ASSETS, 'jieqi.txt'), 'utf8')
    .trim().split('\n').map(l => l.split(' ').map(Number));

function jdnOf(y, m, d) {
    const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}
function civilOf(j) {
    const a = j + 32044, b = Math.floor((4 * a + 3) / 146097), c = a - Math.floor(146097 * b / 4),
        d = Math.floor((4 * c + 3) / 1461), e = c - Math.floor(1461 * d / 4),
        m = Math.floor((5 * e + 2) / 153);
    return [100 * b + d - 4800 + Math.floor(m / 10), m + 3 - 12 * Math.floor(m / 10),
        e - Math.floor((153 * m + 2) / 5) + 1];
}
/** Lệch múi giờ (ms) của tzId tại đúng thời điểm utcMs — có tính DST. */
function tzOffsetMs(tzId, utcMs) {
    const f = new Intl.DateTimeFormat('en-US', {
        timeZone: tzId, hour12: false, year: 'numeric', month: '2-digit',
        day: '2-digit', hour: '2-digit', minute: '2-digit', second: '2-digit',
    });
    const p = Object.fromEntries(f.formatToParts(new Date(utcMs)).map(x => [x.type, x.value]));
    return Date.UTC(+p.year, +p.month - 1, +p.day, (+p.hour) % 24, +p.minute, +p.second) - utcMs;
}
/** 24 mục của năm tiết khí chứa `ref`, đã quy sang giờ tzId. */
function widgetJieQi(ref, tzId) {
    let lo = 0, hi = JQ.length - 1, at = -1;
    while (lo <= hi) { const mid = (lo + hi) >> 1; if (JQ[mid][0] <= ref) { at = mid; lo = mid + 1; } else hi = mid - 1; }
    if (at < 0) return [];
    let start = at;
    while (start >= 0 && JQ[start][2] !== 0) start--;
    if (start < 0 || start + 23 >= JQ.length) return [];
    const p = n => (n < 10 ? '0' : '') + n;
    return JQ.slice(start, start + 24).map(r => {
        const utcMs = (r[0] - 2440588) * 86400000 + r[1] * 60000 - 7 * 3600000;
        const local = utcMs + tzOffsetMs(tzId, utcMs);
        let days = Math.floor(local / 86400000);
        let rem = local - days * 86400000;
        const [y, m, d] = civilOf(days + 2440588);
        const mins = Math.floor(rem / 60000);
        return {
            name: TK_VI_TABLE[r[2]],
            date: `${p(d)}-${p(m)}-${y} ${p(Math.floor(mins / 60))}:${p(mins % 60)}`,
        };
    });
}

let chromium;
try { ({ chromium } = await import('playwright')); }
catch { console.log('Bỏ qua: chưa cài playwright.'); process.exit(0); }

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const f = path.join(WEB, rel);
    if (!f.startsWith(WEB) || !fs.existsSync(f)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
    fs.createReadStream(f).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;

const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

// Múi giờ mới là thứ dễ làm hai bảng lệch, nên thử nhiều nơi.
const ctx0 = await browser.newContext();
const p0 = await ctx0.newPage();
await p0.goto(base, { waitUntil: 'networkidle' });
const CITIES = (await p0.evaluate(
    () => [...document.getElementById('country').options].map(o => o.value)
)).filter(v => v && v !== '__loc');
await ctx0.close();
const PICK = [CITIES[0], ...CITIES.slice(1).filter((_, i) => i % 7 === 0)].slice(0, 5);

let fail = 0;
for (const city of PICK) {
    const ctx = await browser.newContext({ viewport: { width: 412, height: 900 } });
    await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);

    const r = await page.evaluate((city) => {
        document.getElementById('country').value = city;
        selectMethod('bophap');
        processAll();
        // Bảng Sách Bổ: mỗi dòng một tiết khí, cột 1 tên, cột 2 ngày giờ.
        const kmdj = [...document.querySelectorAll('#sb-tbody tr')].map(tr => ({
            name: tr.cells[0].textContent.trim(),
            date: tr.cells[1].textContent.trim(),
            on: tr.classList.contains('dp-row-active'),
        }));
        // Bảng tab Lịch: 12 dòng × 2 cặp cột; cặp trái là mục 0–11, phải 12–23.
        window.showTab('cal');
        const rows = [...document.querySelectorAll('#calJqBody tr')];
        const cal = [];
        for (const half of [0, 2]) {
            for (const tr of rows) {
                cal.push({
                    name: tr.cells[half].textContent.trim(),
                    date: tr.cells[half + 1].textContent.trim(),
                    on: tr.cells[half].classList.contains('cal-jq-on'),
                });
            }
        }
        window.showTab('qmdj');
        return {
            kmdj, cal,
            name: countryData[city] && countryData[city].name_vi,
            tzId: countryData[city] && countryData[city].tzId,
        };
    }, city);

    const diffs = [];
    if (r.kmdj.length !== 24 || r.cal.length !== 24) {
        diffs.push(`số mục: Kỳ Môn ${r.kmdj.length}, Lịch ${r.cal.length}`);
    } else {
        for (let i = 0; i < 24; i++) {
            if (r.kmdj[i].name !== r.cal[i].name || r.kmdj[i].date !== r.cal[i].date) {
                diffs.push(`mục ${i + 1}: Kỳ Môn "${r.kmdj[i].name} ${r.kmdj[i].date}" ` +
                    `≠ Lịch "${r.cal[i].name} ${r.cal[i].date}"`);
            }
        }
    }
    const ai = r.kmdj.findIndex(x => x.on), bi = r.cal.findIndex(x => x.on);
    if (ai < 0) diffs.push('bảng Kỳ Môn không tô đậm mục nào');
    if (bi < 0) diffs.push('bảng Lịch không tô đậm mục nào');
    // Đúng ngày giao tiết thì hai bên có thể lệch một mục: tab Kỳ Môn lấy cả giờ
    // phút đang nhập, tab Lịch chỉ có độ phân giải một ngày (lấy mốc 12:00 trưa).
    if (ai >= 0 && bi >= 0 && Math.abs(ai - bi) > 1) {
        diffs.push(`mục tô đậm lệch: Kỳ Môn ${ai + 1} (${r.kmdj[ai].name}), Lịch ${bi + 1} (${r.cal[bi].name})`);
    }
    if (errs.length) diffs.push('lỗi JS: ' + errs.join('; '));

    // ── Widget: tra bảng jieqi.txt rồi quy sang múi giờ đang chọn ──
    const now = new Date();
    const wid = widgetJieQi(jdnOf(now.getFullYear(), now.getMonth() + 1, now.getDate()), r.tzId);
    if (wid.length !== 24) {
        diffs.push(`widget dựng được ${wid.length} mục, cần 24`);
    } else if (r.cal.length === 24) {
        for (let i = 0; i < 24; i++) {
            if (wid[i].name !== r.cal[i].name || wid[i].date !== r.cal[i].date) {
                diffs.push(`widget mục ${i + 1}: "${wid[i].name} ${wid[i].date}" ` +
                    `≠ Lịch "${r.cal[i].name} ${r.cal[i].date}"`);
            }
        }
    }

    const label = (r.name || city).padEnd(22);
    if (diffs.length) {
        fail++;
        console.log(`  LỆCH  ${label}`);
        diffs.slice(0, 5).forEach(d => console.log('    ' + d));
    } else {
        console.log(`  ok    ${label} Kỳ Môn = Lịch = widget, 24 mục · đang ở ${r.cal[bi].name} ${r.cal[bi].date}`);
    }
    await ctx.close();
}

await browser.close();
server.close();
console.log(fail
    ? `\n✗ ${fail}/${PICK.length} ca lệch`
    : `\n✓ ${PICK.length}/${PICK.length} ca: Sách Bổ pháp, tab Lịch và widget cho cùng một bảng tiết khí`);
process.exit(fail ? 1 : 0);
