/**
 * Quy tắc lịch âm: **mùng 1 là ngày CHỨA điểm Sóc** — xét ở đúng mốc múi giờ mà
 * ứng dụng dùng để hiện giờ Sóc.
 *
 *   node test_soc_parity.mjs
 *
 * Trước đây ứng dụng trộn hai hệ quy chiếu trên cùng một màn hình: ngày âm tính
 * ở UTC+8 còn giờ Sóc lại quy sang giờ địa phương. Ở Paris, Sóc hiện
 * 12/08/2026 19:37 nhưng mùng 1 lại là 13/08 (ở UTC+8 thì Sóc rơi vào 13/08
 * 00:37). Cùng một thời điểm, hai cách đọc — nhìn như tính sai.
 *
 * Phép thử mở ứng dụng thật ở nhiều múi giờ và canh ba điều:
 *   1. tab Kỳ Môn: mùng 1 phải là ngày CHỨA điểm Sóc, đếm từ CHÍNH TÝ tới
 *      Chính Tý — nửa đêm mặt trời thật (Chính Ngọ − 12h), không phải 00:00
 *      đồng hồ. Phép thử tự tính lại mốc ấy từ Astro.solarNoonMinutes, tức đi
 *      đường khác với zi_dayOf trong app.js chứ không chép lại nó;
 *   2. cột Vọng phải đúng là lúc trăng tròn (đối chiếu độc lập bằng % chiếu
 *      sáng của astro.js, phải ≥ 99,5%);
 *   3. hai tab phải nói cùng một ngày âm cho cùng một ngày dương;
 *   4. bảng mà ứng dụng ghi ra cho WIDGET (publishLunarCache) phải khớp luôn —
 *      widget không chạy được lunar.js nên nó sống bằng bảng này.
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

// Ngày thử. Ba ca đầu là ngày thường; năm ca sau là các tháng mà điểm Sóc rơi
// SÁT Chính Tý, tức đúng chỗ luật này khác luật nửa đêm đồng hồ — cả hai chiều:
//   Paris     06/07/2024 Sóc 00:57 → mùng 1 lùi về 05/07 (chưa tới Chính Tý)
//   Paris     07/04/2027 Sóc 01:51 → mùng 1 lùi về 06/04
//   Paris     03/04/2030 Sóc 00:02 → mùng 1 lùi về 02/04
//   Hồng Kông 15/12/2020 Sóc 00:18 → mùng 1 lùi về 14/12
//   Hà Nội    21/04/2031 Sóc 23:58 → mùng 1 tiến tới 22/04 (đã qua Chính Tý)
// Không có mấy ca này thì phép thử xanh mà chẳng đụng tới luật mới lần nào.
const DATES = [
    [2026, 8, 12], [2026, 8, 13], [2026, 2, 17],
    [2024, 7, 6], [2027, 4, 7], [2030, 4, 3], [2020, 12, 15], [2031, 4, 21],
];

const ctx0 = await browser.newContext();
const p0 = await ctx0.newPage();
await p0.goto(base, { waitUntil: 'networkidle' });
const CITIES = (await p0.evaluate(
    () => [...document.getElementById('country').options].map(o => o.value)
)).filter(v => v && v !== '__loc');
await ctx0.close();
const PICK = [CITIES[0], ...CITIES.slice(1).filter((_, i) => i % 5 === 0)].slice(0, 6);

let fail = 0, checks = 0, ziCases = 0, outOfRange = 0;
for (const city of PICK) {
    const ctx = await browser.newContext({ viewport: { width: 412, height: 900 } });
    await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(800);

    const rows = await page.evaluate(({ city, DATES }) => {
        document.getElementById('country').value = city;
        // Bảng Sóc/Vọng nay ở mục "Lịch âm" của tab TRA CỨU: bảng chi tiết Âm
        // Bàn pháp ở tab Kỳ Môn đã bỏ hẳn, và mục Lịch âm cũng đã rời tab Lịch.
        // Cùng một dãy số (cùng hàm __calShared.amBanTableHtml dựng ra), chỉ
        // khác chỗ hiện — và ở đây nó tra theo NĂM ÂM người dùng gõ, không theo
        // tháng đang xem, nên mỗi ca phải đặt năm rồi mới đọc.
        const mởLịchÂm = () => {
            const sec = document.getElementById('tcAmSec');
            if (sec && getComputedStyle(sec).display === 'none') window.toggleDetailPanel('tcam');
        };
        const out = [];
        for (const [y, m, d] of DATES) {
            getDOM('inYear').value = y;
            getDOM('inMonth').value = m;
            getDOM('inDay').value = d;
            getDOM('solarHour').value = 12;
            getDOM('solarMinute').value = 0;
            processAll();
            // Ngày âm của chính ngày dương này: "dd - mm - yyyy", tháng nhuận
            // viết "mm (Nhuận)". Cả ba mảnh đều cần — NĂM âm để lái bảng tra,
            // THÁNG và cờ nhuận để chọn đúng dòng.
            const lunar = (getDOM('out-lunar-table')?.innerText || '').trim();
            const mảnh = lunar.split('-').map(x => x.trim());
            const lmon = parseInt(mảnh[1], 10);
            const lleap = /Nhuận|闰/.test(mảnh[1] || '');
            const lyear = parseInt(mảnh[2], 10);
            // Bảng Lịch âm ở tab Tra cứu liệt kê 12–13 tháng của NĂM ÂM được
            // gõ vào ô năm — phải đặt đúng năm ấy, bằng không mọi ca đều đọc
            // ra dãy Sóc của năm mặc định.
            window.showTab('tracuu');
            window.__tracuuYear(lyear);
            mởLịchÂm();
            // Tháng | Sóc | Vọng. Tra một năm bất kỳ thì KHÔNG dòng nào được
            // tô "đang hiệu lực" (activeMon = 0), nên dò theo nhãn tháng — và
            // phải phân biệt "Tháng 3" với "Tháng 3 (Nhuận)", hai dòng khác
            // hẳn nhau trong cùng một năm.
            const trs = [...document.querySelectorAll('#tcAmBody tbody tr')];
            const tr = trs.find(t => {
                const nh = t.cells[0].textContent.trim();
                const n = parseInt((/\d+/.exec(nh) || [])[0], 10);
                return n === lmon && /Nhuận|闰/.test(nh) === lleap;
            });
            const soc = tr ? tr.cells[1].textContent.trim() : '';
            const vong = tr ? tr.cells[2].textContent.trim() : '';
            // Mốc mùng 1 mong đợi, tính ĐỘC LẬP từ giờ Sóc đang hiện:
            // Chính Tý = Chính Ngọ − 12h, rồi xem giờ Sóc rơi vào ngày nào.
            let want = null, inWindow = false;
            const mm = soc.match(/^(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2})$/);
            if (mm && window.Astro) {
                const [, dd, mo2, yy, hh, mi2] = mm.map(Number);
                const info = countryData[city];
                const tzS = getTimezoneOffset(info.tzId, new Date(yy, mo2 - 1, dd, 12));
                const zi = Astro.solarNoonMinutes(yy, mo2, dd, info.lon, tzS) - 720;
                const t = hh * 60 + mi2;
                const shift = Math.floor((t - zi) / 1440);
                const dt = new Date(yy, mo2 - 1, dd + shift);
                const p2 = n => String(n).padStart(2, '0');
                want = `${p2(dt.getDate())}-${p2(dt.getMonth() + 1)}-${dt.getFullYear()}`;
                inWindow = shift !== 0;
            }
            // % Mặt Trăng được chiếu sáng tại thời điểm Vọng — đường kiểm
            // ĐỘC LẬP, không đi qua công thức đã dựng ra cột ấy.
            let vongIll = null;
            const vm = vong.match(/^(\d{2})-(\d{2})-(\d{4}) (\d{2}):(\d{2})$/);
            if (vm && window.Astro) {
                const [, vd, vmo, vy, vh, vmi] = vm.map(Number);
                const info = countryData[city];
                const tzV = getTimezoneOffset(info.tzId, new Date(vy, vmo - 1, vd, 12));
                const jdUTC = (Date.UTC(vy, vmo - 1, vd, vh, vmi) / 86400000 + 2440587.5) - tzV / 24;
                vongIll = Astro.moonIllumination(jdUTC).fraction;
            }
            out.push({ y, m, d, lunar, soc, vong, vongIll, want, inWindow });
        }
        // tab Lịch: ngày âm của chính những ngày dương ấy
        const cal = {};
        window.showTab('cal');
        for (const [y, m, d] of DATES) {
            window.__calGoto && window.__calGoto(y, m, d);
            const c = document.querySelector(
                `#calGrid .cal-day[data-y="${y}"][data-m="${m}"][data-d="${d}"]`);
            cal[`${y}-${m}-${d}`] = c
                ? (c.querySelector('.cal-lunar')?.textContent || '').trim() : null;
        }
        window.showTab('qmdj');
        let cache = null;
        try { cache = localStorage.getItem('qmdj.lunarCache'); } catch (e) {}
        return { rows: out, cal, cache, name: countryData[city] && countryData[city].name_vi };
    }, { city, DATES });
    const cal = rows.cal;

    const bad = [];
    const p = n => String(n).padStart(2, '0');
    for (const r of rows.rows) {
        checks++;
        const lday = parseInt((r.lunar.split('-')[0] || '').trim(), 10);
        if (!lday || !/^\d{2}-\d{2}-\d{4}/.test(r.soc)) {
            bad.push(`${r.d}/${r.m}/${r.y}: không đọc được (âm "${r.lunar}", sóc "${r.soc}")`);
            continue;
        }
        // 1. Mùng 1 = ngày chứa điểm Sóc, đếm từ Chính Tý.
        //    Bảng Âm Bàn không còn cột Mùng 1, nên suy nó từ NGÀY ÂM đang hiện:
        //    mùng 1 = ngày đang xét lùi lại (ngày âm − 1) ngày.
        const first = new Date(r.y, r.m - 1, r.d - (lday - 1));
        const mung1 = `${p(first.getDate())}-${p(first.getMonth() + 1)}-${first.getFullYear()}`;
        if (!r.want) {
            bad.push(`${r.d}/${r.m}/${r.y}: không tính được mốc mong đợi từ Sóc "${r.soc}"`);
        } else if (mung1 !== r.want) {
            bad.push(`${r.d}/${r.m}/${r.y}: Sóc ${r.soc} → mùng 1 phải là ${r.want}` +
                ` (theo Chính Tý), nhưng ngày âm ${lday} suy ra ${mung1}`);
        }
        if (r.inWindow) ziCases++;

        // 2. Cột Vọng phải là một THỜI ĐIỂM (có giờ phút, như Sóc) và phải
        //    đúng lúc trăng tròn — đối chiếu bằng % chiếu sáng của astro.js,
        //    tức đường khác hẳn công thức đã dựng ra cột ấy.
        if (!/^\d{2}-\d{2}-\d{4} \d{2}:\d{2}$/.test(r.vong)) {
            bad.push(`${r.d}/${r.m}/${r.y}: cột Vọng "${r.vong}" không phải thời điểm DD-MM-YYYY HH:MM`);
        } else if (r.vongIll === null) {
            bad.push(`${r.d}/${r.m}/${r.y}: không kiểm được % chiếu sáng của Vọng`);
        } else if (r.vongIll < 0.995) {
            bad.push(`${r.d}/${r.m}/${r.y}: Vọng ${r.vong} chỉ sáng ` +
                `${(r.vongIll * 100).toFixed(2)}% — chưa phải trăng tròn`);
        }
        // 3. Tab Lịch phải nói cùng ngày âm với tab Kỳ Môn.
        const c = cal[`${r.y}-${r.m}-${r.d}`];
        if (c != null) {
            const cday = parseInt(String(c).split('/')[0], 10);
            if (cday !== lday) {
                bad.push(`${r.d}/${r.m}/${r.y}: tab Kỳ Môn âm ${lday}, tab Lịch âm ${cday}`);
            }
        }
    }
    // ── 4. Bảng cho widget ──
    if (!rows.cache) {
        bad.push('ứng dụng không ghi ra bảng tháng âm cho widget');
    } else {
        const [tzMin, body] = rows.cache.split('|');
        const tbl = body.split(';').filter(Boolean).map(x => x.split(',').map(Number));
        if (tbl.length < 20) bad.push(`bảng widget chỉ có ${tbl.length} tháng`);
        for (let i = 1; i < tbl.length; i++) {
            if (tbl[i][0] <= tbl[i - 1][0]) { bad.push('bảng widget không tăng dần'); break; }
        }
        // Tra bằng CHÍNH cách LunarTable.fromAppCache tra — kể cả phép chặn hai
        // đầu: bảng chỉ phủ chừng 40 tháng quanh hôm nay, ra ngoài thì trả null
        // để bảng đóng trong APK lo, chứ không suy bừa.
        const look = j => {
            if (tbl.length < 2 || j < tbl[0][0] || j >= tbl[tbl.length - 1][0]) return null;
            let lo = 0, hi = tbl.length - 1, idx = -1;
            while (lo <= hi) { const m = (lo + hi) >> 1; if (tbl[m][0] <= j) { idx = m; lo = m + 1; } else hi = m - 1; }
            return idx < 0 ? null : { day: j - tbl[idx][0] + 1, month: tbl[idx][1], leap: tbl[idx][2] === 1 };
        };
        const jd = (y, m, d) => {
            const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
            return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
                + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
        };
        for (const r of rows.rows) {
            const c = cal[`${r.y}-${r.m}-${r.d}`];
            if (c == null) continue;
            const got = look(jd(r.y, r.m, r.d));
            const calDay = parseInt(String(c).split('/')[0], 10);
            // Ngoài tầm phủ là đúng thiết kế, không phải lỗi.
            if (!got) { outOfRange++; continue; }
            if (got.day !== calDay) {
                bad.push(`${r.d}/${r.m}/${r.y}: bảng widget âm ${got.day}, tab Lịch âm ${calDay}`);
            }
        }
    }

    if (errs.length) bad.push('lỗi JS: ' + errs.slice(0, 2).join('; '));

    const label = (rows.name || city).padEnd(24);
    if (bad.length) { fail++; console.log(`  LỆCH ${label}`); bad.forEach(b => console.log('    ' + b)); }
    else console.log(`  ok   ${label} ${DATES.length} ngày · mùng 1 = ngày chứa Sóc, Vọng = trăng tròn, hai tab + widget khớp`);
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\nSố ca mà Chính Tý ĐẨY mùng 1 lệch khỏi ngày dương của Sóc: ${ziCases}`);
console.log(`Ngày nằm ngoài tầm phủ của bảng widget (bỏ qua, đúng thiết kế): ${outOfRange}`);
if (ziCases === 0) {
    // Xanh mà không đụng tới luật giờ Tý thì phép thử vô nghĩa.
    console.log('✗ không ca nào chạm ranh giới Chính Tý — bộ ngày thử đã mất tác dụng');
    fail++;
}
console.log(fail
    ? `\n✗ ${fail}/${PICK.length} ca lệch`
    : `\n✓ ${PICK.length}/${PICK.length} ca (${checks} phép so): mùng 1 = ngày chứa Sóc, hai tab khớp nhau`);
process.exit(fail ? 1 : 0);
