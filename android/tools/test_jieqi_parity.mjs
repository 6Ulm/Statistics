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
 * (Bảng tiết khí của widget được canh riêng trong test_lunar_table.mjs, cũng
 * theo nguyên tắc Sách Bổ pháp.)
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

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
        return { kmdj, cal, name: countryData[city] && countryData[city].name_vi };
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

    const label = (r.name || city).padEnd(22);
    if (diffs.length) {
        fail++;
        console.log(`  LỆCH  ${label}`);
        diffs.slice(0, 5).forEach(d => console.log('    ' + d));
    } else {
        console.log(`  ok    ${label} 24 mục giống hệt · đang ở ${r.cal[bi].name} ${r.cal[bi].date}`);
    }
    await ctx.close();
}

await browser.close();
server.close();
console.log(fail
    ? `\n✗ ${fail}/${PICK.length} ca lệch`
    : `\n✓ ${PICK.length}/${PICK.length} ca: tab Lịch và Sách Bổ pháp cho cùng một bảng tiết khí`);
process.exit(fail ? 1 : 0);
