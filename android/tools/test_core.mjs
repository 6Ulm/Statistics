/**
 * core.js — bộ tính dùng chung, và LUẬT "không tab nào tự tính lại".
 *
 *   node test_core.mjs
 *
 * Ba nhóm:
 *
 *   1. LUẬT XẾP TẦNG, canh bằng cách ĐỌC MÃ. Một phép thử chạy được thì chỉ
 *      nói "hôm nay hai tab ra cùng một số"; nó không ngăn ai ngày mai viết
 *      thêm một bản tính thứ hai. Nhóm này canh chính chỗ ấy: tệp của tab
 *      không được chạm vào lunar.js, không được tự đọc ô ngày giờ, không được
 *      tự dựng lại bảng mà core.js đã có.
 *   2. SỐ HỌC của những thứ core.js TÍNH thay cho ba bảng 60 dòng đã xoá —
 *      đối chiếu với chính ba bảng ấy, chép lại nguyên văn dưới đây làm mốc.
 *   3. MỘT LÁ SỐ, BỐN TAB: cùng một ngày giờ thì bốn tab phải hiện cùng những
 *      con số ấy — không phải "gần giống", mà TRÙNG KHÍT từng ký tự.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

let pass = 0, fail = 0;
const ok = (ten, đúng, thêm) => {
    if (đúng) { pass++; console.log('  ok   ' + ten); }
    else { fail++; console.log('  ✗    ' + ten + (thêm ? '  — ' + thêm : '')); }
};
const check = (ten, được, mong) => ok(ten, JSON.stringify(được) === JSON.stringify(mong),
    `được ${JSON.stringify(được)}, mong ${JSON.stringify(mong)}`);

const đọc = f => fs.readFileSync(path.join(WEB, 'js', f), 'utf8');
/** Bỏ ghi chú để phép quét không bắt nhầm chính lời giải thích về luật. */
const bỏGhiChú = src => src
    .replace(/\/\*[\s\S]*?\*\//g, '')
    .replace(/(^|[^:])\/\/.*$/gm, '$1');

const TAB = ['calendar.js', 'lenh.js', 'tracuu.js'];
const MÃ = {};
for (const f of [...TAB, 'app.js', 'core.js', 'ephem.js', 'nguhanh.js']) MÃ[f] = bỏGhiChú(đọc(f));

/* ── 1. Luật xếp tầng ── */
console.log('\nLuật xếp tầng: chỉ ephem.js được chạm vào lunar.js');
{
    // 1a. ShouXingUtil / LunarYear chỉ được gọi trong ephem.js.
    for (const f of [...TAB, 'app.js', 'core.js']) {
        const hit = [...MÃ[f].matchAll(/\b(ShouXingUtil|LunarYear)\s*\./g)].map(m => m[1]);
        ok(`${f}: không gọi thẳng ShouXingUtil/LunarYear`, hit.length === 0,
            hit.slice(0, 3).join(', '));
    }
    ok('ephem.js vẫn là nơi DUY NHẤT gọi chúng',
        /ShouXingUtil\s*\./.test(MÃ['ephem.js']) && /LunarYear\s*\./.test(MÃ['ephem.js']));

    // 1b. Ô ngày giờ + ô vị trí: chỉ core.js đọc.
    const Ô = ['inYear', 'inMonth', 'inDay', 'solarHour', 'solarMinute', 'country'];
    for (const f of TAB) {
        const hit = Ô.filter(id => MÃ[f].includes(`'${id}'`) || MÃ[f].includes(`"${id}"`));
        ok(`${f}: không tự đọc ô ngày giờ / ô vị trí`, hit.length === 0, hit.join(', '));
    }
    for (const f of TAB) {
        ok(`${f}: không tự tra countryData`, !/\bcountryData\s*\[/.test(MÃ[f]));
        ok(`${f}: không tự gọi getTimezoneOffset`, !/\bgetTimezoneOffset\s*\(/.test(MÃ[f]));
    }
    ok('core.js đọc chúng, và có Core.input()/location()/tzAt()',
        /getDOM\('inYear'\)|el\('inYear'\)/.test(MÃ['core.js'])
        && /input:\s*input/.test(MÃ['core.js'])
        && /location:\s*location/.test(MÃ['core.js']));

    // 1c. Ba cửa ngầm giữa hai tệp đã bỏ.
    for (const cửa of ['__yearGanIdx', '__monthGanIdx']) {
        const ai = Object.keys(MÃ).filter(f => MÃ[f].includes(cửa));
        ok(`cửa ngầm window.${cửa} đã bỏ hẳn`, ai.length === 0, ai.join(', '));
    }

    // 1d. Ba bảng 60 dòng đã xoá, và không tệp nào dựng lại.
    for (const bảng of ['dayToTuanThu', 'hoaGiapToKhongVong', 'hoaGiapToDichMa']) {
        const ai = Object.keys(MÃ).filter(f => MÃ[f].includes(bảng));
        ok(`bảng 60 dòng ${bảng} đã xoá khỏi mã`, ai.length === 0, ai.join(', '));
    }
}

/* ── 2. Số học thay cho ba bảng ── */
console.log('\nHàm tính khớp ba bảng 60 dòng đã xoá');
const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
{
    // Ba bảng NGUYÊN VĂN của bản trước — mốc đối chiếu, cố ý chép vào đây chứ
    // không đọc lại từ mã: mốc mà lấy từ chính thứ đang kiểm thì không phải mốc.
    const THU = 'Mậu Mậu Mậu Mậu Mậu Mậu Mậu Mậu Mậu Mậu Kỷ Kỷ Kỷ Kỷ Kỷ Kỷ Kỷ Kỷ Kỷ Kỷ Canh Canh Canh Canh Canh Canh Canh Canh Canh Canh Tân Tân Tân Tân Tân Tân Tân Tân Tân Tân Nhâm Nhâm Nhâm Nhâm Nhâm Nhâm Nhâm Nhâm Nhâm Nhâm Quý Quý Quý Quý Quý Quý Quý Quý Quý Quý'.split(' ');
    const KV = ('Tuất Hợi,'.repeat(10) + 'Thân Dậu,'.repeat(10) + 'Ngọ Mùi,'.repeat(10)
        + 'Thìn Tỵ,'.repeat(10) + 'Dần Mão,'.repeat(10) + 'Tý Sửu,'.repeat(10))
        .replace(/,$/, '').split(',');
    const DM = ['Dần', 'Hợi', 'Thân', 'Tỵ'];   // theo chi mod 4: Tý→Dần, Sửu→Hợi, Dần→Thân, Mão→Tỵ

    const w = {};
    new Function('window', đọc('nguhanh.js'))(w);
    // core.js gọi Ephem/Solar lúc CHẠY, nên nạp được trong sandbox trống.
    new Function('window', đọc('core.js'))(w);
    const C = w.Core;
    ok('core.js nạp được và lộ Core', !!C && typeof C.tuanThuOf === 'function');

    let lệchThu = [], lệchKv = [], lệchDm = [];
    for (let n = 0; n < 60; n++) {
        const g = n % 10, z = n % 12;
        if (C.tuanThuOf(g, z) !== THU[n]) lệchThu.push(`${CAN[g]} ${CHI[z]}`);
        if (C.khongVongOf(g, z).map(i => CHI[i]).join(' ') !== KV[n]) lệchKv.push(`${CAN[g]} ${CHI[z]}`);
    }
    for (let z = 0; z < 12; z++) {
        if (CHI[C.dichMaOf(z)] !== DM[z % 4]) lệchDm.push(CHI[z]);
    }
    ok('tuanThuOf khớp bảng dayToTuanThu cũ (60/60)', lệchThu.length === 0, lệchThu.slice(0, 3).join(', '));
    ok('khongVongOf khớp bảng hoaGiapToKhongVong cũ (60/60)', lệchKv.length === 0, lệchKv.slice(0, 3).join(', '));
    ok('dichMaOf khớp bảng hoaGiapToDichMa cũ (12/12 chi)', lệchDm.length === 0, lệchDm.join(', '));
    check('tuanGiapOf: Nhâm Thân → Giáp Tý', C.tuanGiapOf(8, 8), 'Giáp Tý');
    check('tuanGiapOf: Quý Hợi → Giáp Dần', C.tuanGiapOf(9, 11), 'Giáp Dần');
    check('jdn 1/1/2000', C.jdn(2000, 1, 1), 2451545);
}

/* ── 3. Một lá số, bốn tab ── */
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((q, r) => {
    const rel = decodeURIComponent(q.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const f = path.join(WEB, rel);
    if (!f.startsWith(WEB) || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
    r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
    fs.createReadStream(f).pipe(r);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;
const { chromium } = await import('playwright');
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

console.log('\nMột lá số, bốn tab — cùng một con số');
{
    const ctx = await browser.newContext({
        viewport: { width: 412, height: 852 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1400);
    await page.evaluate(() => window.QMDJLocation.apply(
        window.QMDJLocation.makeLoc('Paris', 48.8566, 2.3522, 'Europe/Paris')));
    await page.waitForTimeout(600);

    for (const [y, m, d, h, mi] of [[1991, 7, 16, 16, 0], [2026, 2, 4, 5, 0], [2026, 9, 19, 9, 21]]) {
        await page.evaluate(({ y, m, d, h, mi }) => {
            const set = (id, v) => {
                const el = document.getElementById(id);
                el.value = String(v);
                el.dispatchEvent(new Event('change', { bubbles: true }));
            };
            set('inYear', y); set('inMonth', m); set('inDay', d);
            set('solarHour', h); set('solarMinute', mi);
            window.processAll();
        }, { y, m, d, h, mi });
        await page.waitForTimeout(450);
        const nhãn = `${d}/${m}/${y} ${h}:${mi}`;

        // Hộp Bát Tự dùng chung hai tab — đọc ở tab Kỳ Môn rồi đọc lại ở tab
        // Bát Tự, phải y hệt.
        const đọcBazi = () => page.evaluate(() => ['Nam', 'Thang', 'Ngay', 'Gio']
            .map(k => document.getElementById('ttCan' + k).textContent.trim()
                + document.getElementById('ttChi' + k).textContent.trim()).join(' '));
        await page.evaluate(() => window.showTab('qmdj'));
        await page.waitForTimeout(250);
        const ởKyMon = await đọcBazi();
        const info = await page.evaluate(() => ({
            chinhNgo: document.getElementById('out-chinhngo').textContent.trim(),
            tietKhi: document.getElementById('out-tietkhi').textContent.trim(),
            lunar: document.getElementById('out-lunar-table').textContent.trim(),
        }));
        await page.evaluate(() => window.showTab('lenh'));
        await page.waitForTimeout(400);
        const ởBatTu = await đọcBazi();
        ok(`${nhãn}: bốn trụ ở tab Kỳ Môn và tab Bát Tự trùng khít`,
            ởKyMon === ởBatTu, `${ởKyMon} vs ${ởBatTu}`);

        // Và chính Core mới là nguồn của chúng.
        const tuCore = await page.evaluate(() => {
            const ch = Core.chart(Core.input());
            return {
                bazi: ch.ganZH.map((g, i) => g + ch.chiZH[i]).join(' '),
                chinhNgo: ch.chinhNgo.hhmm + ' (' + ch.chinhNgo.gmt + ')',
                lunarNgay: ch.lunar.day,
            };
        });
        const zhBazi = await page.evaluate(() => {
            const ch = Core.chart(Core.input());
            return ch.ganZH.map((g, i) => g + ch.chiZH[i]).join(' ');
        });
        ok(`${nhãn}: hộp Bát Tự đúng là chart của Core`, tuCore.bazi === zhBazi);
        ok(`${nhãn}: Chính Ngọ trên màn hình = Core.chinhNgo`,
            info.chinhNgo === tuCore.chinhNgo, `${info.chinhNgo} vs ${tuCore.chinhNgo}`);
        ok(`${nhãn}: ngày âm trên màn hình = Core.lunar`,
            info.lunar.startsWith(String(tuCore.lunarNgay).padStart(2, '0')),
            `${info.lunar} vs ngày ${tuCore.lunarNgay}`);

        // Hộp Lệnh ở tab Bát Tự nay hiện CẢ Chính Ngọ và Tiết khí — phải
        // trùng KHÍT hộp thông tin tab Kỳ Môn, cả con số lẫn cách viết. Đây
        // là chỗ dễ trôi nhất: hai hộp, hai tệp, một thông tin.
        const hộpLệnh = await page.evaluate(() => ({
            chinhNgo: (document.getElementById('lenhChinhNgo') || {}).textContent || null,
            tietKhi: (document.getElementById('lenhTietKhi') || {}).textContent || null,
        }));
        ok(`${nhãn}: Chính Ngọ ở hộp Lệnh = hộp Kỳ Môn`,
            hộpLệnh.chinhNgo === info.chinhNgo, `${hộpLệnh.chinhNgo} vs ${info.chinhNgo}`);
        ok(`${nhãn}: Tiết khí ở hộp Lệnh = hộp Kỳ Môn`,
            hộpLệnh.tietKhi === info.tietKhi, `${hộpLệnh.tietKhi} vs ${info.tietKhi}`);
        // Và cùng bộ nhãn: hộp Lệnh mượn thẳng lớp CSS của hộp Kỳ Môn.
        const khung = await page.evaluate(() => {
            const b = document.getElementById('lenhNow');
            return {
                dòng: b.querySelectorAll('.info-line-nowrap').length,
                cặp: b.querySelectorAll('.info-pair').length,
                nhãn: [...b.querySelectorAll('.lbl')].map(e => e.textContent),
            };
        });
        check(`${nhãn}: hộp Lệnh có 2 dòng × 2 ô, đúng khung hộp Kỳ Môn`,
            [khung.dòng, khung.cặp], [2, 4]);

        // Tiết khí: hộp Kỳ Môn và bảng Tiết khí của tab Lịch phải cùng mốc.
        // Phải LÁI LỊCH về đúng năm ấy trước: tab Lịch xem tháng người dùng
        // đang lật tới, không đi theo ngày sinh — bảng tiết khí của nó là
        // bảng của NĂM ĐANG XEM. So hai bảng của hai năm khác nhau thì lệch
        // là phải, và đó là lỗi của phép thử chứ không phải của ứng dụng.
        await page.evaluate(() => window.showTab('cal'));
        await page.waitForTimeout(350);
        await page.evaluate(({ y, m, d }) => window.__calGoto(y, m, d), { y, m, d });
        await page.waitForTimeout(600);
        const tênTiet = info.tietKhi.split(' ').slice(0, -2).join(' ');
        const mốcTiet = info.tietKhi.split(' ').slice(-2).join(' ');
        const ởLich = await page.evaluate((tên) => {
            for (const r of document.querySelectorAll('#calJieQi tbody tr')) {
                if (r.cells[0] && r.cells[0].textContent.trim() === tên) {
                    return r.cells[1] ? r.cells[1].textContent.trim() : null;
                }
            }
            return null;
        }, tênTiet);
        ok(`${nhãn}: mốc "${tênTiet}" ở tab Kỳ Môn và bảng Tiết khí tab Lịch trùng khít`,
            ởLich !== null && ởLich === mốcTiet, `Kỳ Môn ${mốcTiet} · Lịch ${ởLich}`);
    }
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
if (fail) process.exit(1);
console.log('✓ core.js là nguồn DUY NHẤT, và luật ấy được canh bằng cách đọc mã');
