/**
 * Thần sát trong hộp Bát Tự — nay mới có KHÔNG VONG.
 *
 *   node test_thansat.mjs
 *
 * Ba nhóm, ba mức độc lập khác nhau:
 *
 *   1. SỐ HỌC — NguHanh.tuanKhongOf() TÍNH ra hai chi không vong của một trụ;
 *      app.js thì có sẵn bảng `hoaGiapToKhongVong` chép đủ 60 dòng cho bàn Kỳ
 *      Môn. Hai nguồn hoàn toàn khác nhau về cách dựng, nên canh chúng khớp
 *      nhau từng dòng là canh được cả hai cùng lúc.
 *   2. HAI LÁ SỐ NGƯỜI DÙNG CHO — kết quả mong đợi do người dùng chốt, không
 *      phải do ứng dụng tự sinh.
 *   3. QUÉT RỘNG — hàng chục lá số, mỗi lá dựng kỳ vọng TẠI ĐÂY từ bốn trụ
 *      đọc được trên màn hình, rồi so với ô thần sát. Đây là chỗ bắt được
 *      luật sai: nếu ứng dụng lỡ lấy cả trụ tháng hay trụ giờ làm MỐC (luật
 *      chỉ cho trụ ngày và trụ năm làm mốc), phép quét sẽ đỏ ở những lá số mà
 *      hai luật cho ra kết quả khác nhau — và nhóm 3b canh rằng có thật những
 *      lá số như thế trong mẫu, không thì phép quét chỉ là canh suông.
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
const check = (ten, được, mong) => ok(ten + ` = ${JSON.stringify(mong)}`,
    JSON.stringify(được) === JSON.stringify(mong), `được ${JSON.stringify(được)}`);

const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];

/* ── 1. Số học: hàm tính khớp bảng 60 dòng CHÉP TAY ──
 * Bảng `hoaGiapToKhongVong` từng nằm trong app.js và phép thử này đọc thẳng
 * từ đó. Bảng ấy nay đã xoá (core.js tính ra nó), nên MỐC ĐỐI CHIẾU chuyển
 * vào chính phép thử: sáu tuần × hai chi, chép nguyên văn bản cũ. Mốc mà lấy
 * từ chính thứ đang kiểm thì không phải là mốc. */
console.log('\nKhông vong: hàm tính so bảng 60 dòng chép tay');
{
    // Chạy nguhanh.js trong một `window` giả — nó là IIFE gán window.NguHanh.
    const src = fs.readFileSync(path.join(WEB, 'js', 'nguhanh.js'), 'utf8');
    const w = {};
    new Function('window', src)(w);
    const NH = w.NguHanh;
    ok('nguhanh.js có lộ tuanKhongOf', typeof NH.tuanKhongOf === 'function');

    // Bản cũ của hoaGiapToKhongVong: mười trụ đầu tuần Giáp Tý thì "Tuất
    // Hợi", mười trụ tuần Giáp Tuất thì "Thân Dậu", … sáu tuần như vậy.
    const BẢNG = {};
    const TUẦN = ['Tuất Hợi', 'Thân Dậu', 'Ngọ Mùi', 'Thìn Tỵ', 'Dần Mão', 'Tý Sửu'];
    for (let n = 0; n < 60; n++) {
        BẢNG[CAN[n % 10] + ' ' + CHI[n % 12]] = TUẦN[Math.floor(n / 10)];
    }
    check('bảng đối chiếu đủ 60 dòng', Object.keys(BẢNG).length, 60);

    let lệch = [];
    for (let n = 0; n < 60; n++) {
        const can = CAN[n % 10], chi = CHI[n % 12];
        const tính = NH.tuanKhongOf(can, chi).map(k => CHI[k]).join(' ');
        const bảng = BẢNG[can + ' ' + chi];
        if (tính !== bảng) lệch.push(`${can} ${chi}: tính ${tính} ≠ bảng ${bảng}`);
    }
    ok('cả 60 trụ: hàm tính khớp bảng', lệch.length === 0, lệch.slice(0, 4).join('; '));

    // Tính chất phải đúng của mọi tuần không, không phụ thuộc nguồn nào:
    let hỏng = [];
    for (let n = 0; n < 60; n++) {
        const g = n % 10, z = n % 12;
        const kv = NH.tuanKhongOf(CAN[g], CHI[z]);
        if (kv.length !== 2) { hỏng.push(`${CAN[g]} ${CHI[z]}: ${kv.length} chi`); continue; }
        // Hai chi không vong phải LIỀN NHAU trên vòng 12 chi.
        if ((kv[0] + 1) % 12 !== kv[1]) hỏng.push(`${CAN[g]} ${CHI[z]}: ${CHI[kv[0]]}/${CHI[kv[1]]} không liền nhau`);
        // Và không bao giờ trùng chi của chính trụ ấy.
        if (kv.indexOf(z) >= 0) hỏng.push(`${CAN[g]} ${CHI[z]}: tự rơi vào không vong của mình`);
    }
    ok('mọi trụ: đúng HAI chi, liền nhau, và không trùng chi của chính trụ',
        hỏng.length === 0, hỏng.slice(0, 4).join('; '));

    // Mười trụ cùng một tuần phải cho CÙNG một cặp không vong.
    let tuầnLệch = [];
    for (let t = 0; t < 6; t++) {
        const đầu = t * 10;
        const gốc = NH.tuanKhongOf(CAN[đầu % 10], CHI[đầu % 12]).join(',');
        for (let k = 1; k < 10; k++) {
            const n = đầu + k;
            const nó = NH.tuanKhongOf(CAN[n % 10], CHI[n % 12]).join(',');
            if (nó !== gốc) tuầnLệch.push(`tuần ${t}, trụ ${k}`);
        }
    }
    ok('mười trụ cùng một tuần cho cùng một cặp không vong', tuầnLệch.length === 0,
        tuầnLệch.slice(0, 4).join('; '));
    check('Giáp Tý → Tuất Hợi', NH.tuanKhongOf('Giáp', 'Tý').map(k => CHI[k]), ['Tuất', 'Hợi']);
    check('Quý Hợi → Tý Sửu', NH.tuanKhongOf('Quý', 'Hợi').map(k => CHI[k]), ['Tý', 'Sửu']);
    check('tra bằng chữ Hán cũng ra', NH.tuanKhongOf('甲', '子').map(k => CHI[k]), ['Tuất', 'Hợi']);
    check('chữ không phải can chi → rỗng', NH.tuanKhongOf('Lộ', 'Bàng'), []);
}

/* ── Máy chủ tĩnh + trình duyệt ── */
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

const CỘT = ['năm', 'tháng', 'ngày', 'giờ'];

async function mở(w = 412, h = 852, dpr = 2.625) {
    const ctx = await browser.newContext({
        viewport: { width: w, height: h }, deviceScaleFactor: dpr, isMobile: true, hasTouch: true,
    });
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1300);
    await page.click('#tabLenh');
    await page.waitForTimeout(800);
    return { ctx, page, errs };
}
async function đặtNgày(page, y, m, d, h, mi) {
    await page.evaluate(({ y, m, d, h, mi }) => {
        const set = (id, v) => {
            const el = document.getElementById(id);
            el.value = String(v);
            el.dispatchEvent(new Event('change', { bubbles: true }));
        };
        set('inYear', y); set('inMonth', m); set('inDay', d);
        set('solarHour', h); set('solarMinute', mi);
        if (window.processAll) window.processAll();
    }, { y, m, d, h, mi });
    await page.waitForTimeout(330);
}
const đọc = page => page.evaluate(() => {
    const K = ['Nam', 'Thang', 'Ngay', 'Gio'];
    const t = id => document.getElementById(id).textContent.trim();
    return {
        can: K.map(k => t('ttCan' + k)),
        chi: K.map(k => t('ttChi' + k)),
        ts: K.map(k => document.getElementById('ttTS' + k).innerText.trim()),
        ring: K.map(k => !!document.getElementById('ttChi' + k).querySelector('.tt-kv-ring')),
    };
});

/** Luật người dùng chốt, dựng lại ĐỘC LẬP tại đây: chỉ trụ NGÀY và trụ NĂM
 *  làm mốc, soi vào chi của ba trụ còn lại. */
function kỳVọng(can, chi) {
    const iC = s => CAN.indexOf(s), iZ = s => CHI.indexOf(s);
    const out = [false, false, false, false];
    for (const mốc of [2, 0]) {
        const g = iC(can[mốc]), z = iZ(chi[mốc]);
        if (g < 0 || z < 0) continue;
        const đầu = ((z - g) % 12 + 12) % 12;
        const kv = [(đầu + 10) % 12, (đầu + 11) % 12];
        for (let i = 0; i < 4; i++) if (i !== mốc && kv.indexOf(iZ(chi[i])) >= 0) out[i] = true;
    }
    return out;
}
/** Cùng luật ấy nhưng lấy CẢ BỐN trụ làm mốc — dùng để chứng minh mẫu quét có
 *  phân biệt được hai luật, chứ không phải lúc nào cũng trùng nhau. */
function kỳVọngBốnMốc(can, chi) {
    const iC = s => CAN.indexOf(s), iZ = s => CHI.indexOf(s);
    const out = [false, false, false, false];
    for (let mốc = 0; mốc < 4; mốc++) {
        const g = iC(can[mốc]), z = iZ(chi[mốc]);
        if (g < 0 || z < 0) continue;
        const đầu = ((z - g) % 12 + 12) % 12;
        const kv = [(đầu + 10) % 12, (đầu + 11) % 12];
        for (let i = 0; i < 4; i++) if (i !== mốc && kv.indexOf(iZ(chi[i])) >= 0) out[i] = true;
    }
    return out;
}

/* ── 2. Hai lá số người dùng cho ── */
console.log('\nHai lá số người dùng cho');
{
    const { ctx, page, errs } = await mở();
    // `tru` chốt cứng để phép canh dưới đứng được một mình: nếu mai này phép
    // tính bốn trụ đổi, ô thần sát có thể vẫn "đúng luật" trên bộ trụ SAI mà
    // không ai hay — canh cả hai đầu thì không có chỗ nấp.
    const CA = [
        { y: 1991, m: 7, d: 16, h: 16, ten: '16/7/1991 giờ Thân',
          tru: 'Tân Mùi · Ất Mùi · Đinh Hợi · Mậu Thân', mong: ['năm', 'tháng', 'ngày'] },
        { y: 1996, m: 8, d: 8, h: 18, ten: '8/8/1996 giờ Dậu',
          tru: 'Bính Tý · Bính Thân · Đinh Sửu · Kỷ Dậu', mong: ['tháng', 'giờ'] },
    ];
    for (const c of CA) {
        await đặtNgày(page, c.y, c.m, c.d, c.h, 0);
        const r = await đọc(page);
        const có = r.ts.map((x, i) => x ? CỘT[i] : null).filter(Boolean);
        check(`${c.ten}: bốn trụ`, r.can.map((x, i) => x + ' ' + r.chi[i]).join(' · '), c.tru);
        check(`${c.ten}: cột có Không Vong`, có, c.mong);
        ok(`${c.ten}: ô nào ghi Không Vong thì đúng ô ấy có vòng tròn`,
            r.ring.every((v, i) => v === !!r.ts[i]), JSON.stringify(r.ring) + ' vs ' + JSON.stringify(r.ts));
        ok(`${c.ten}: chữ đúng là "Không Vong"`,
            r.ts.filter(Boolean).every(x => x === 'Không Vong'), JSON.stringify(r.ts));
    }
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 3. Quét rộng: luật chỉ lấy trụ NGÀY và trụ NĂM làm mốc ── */
console.log('\nQuét 48 lá số: ô thần sát khớp luật dựng lại độc lập');
{
    const { ctx, page, errs } = await mở();
    const NGÀY = [];
    let seed = 20260919;
    const rnd = (n) => { seed = (seed * 1103515245 + 12345) & 0x7fffffff; return seed % n; };
    for (let k = 0; k < 48; k++) {
        NGÀY.push([1930 + rnd(96), 1 + rnd(12), 1 + rnd(28), rnd(24)]);
    }
    let lệch = [], phânBiệt = 0, sốCóKV = 0;
    for (const [y, m, d, h] of NGÀY) {
        await đặtNgày(page, y, m, d, h, 0);
        const r = await đọc(page);
        const được = r.ts.map(Boolean);
        const mong = kỳVọng(r.can, r.chi);
        if (JSON.stringify(được) !== JSON.stringify(mong)) {
            lệch.push(`${d}/${m}/${y} ${h}h [${r.can.map((c, i) => c + r.chi[i]).join(' ')}] `
                + `được ${JSON.stringify(được)} mong ${JSON.stringify(mong)}`);
        }
        if (JSON.stringify(kỳVọngBốnMốc(r.can, r.chi)) !== JSON.stringify(mong)) phânBiệt++;
        if (được.some(Boolean)) sốCóKV++;
        // Vòng tròn luôn đi kèm chữ, không bao giờ lệch nhau.
        if (JSON.stringify(r.ring) !== JSON.stringify(được)) {
            lệch.push(`${d}/${m}/${y} ${h}h: vòng tròn ${JSON.stringify(r.ring)} ≠ chữ ${JSON.stringify(được)}`);
        }
    }
    ok(`cả ${NGÀY.length} lá số khớp luật (và vòng tròn khớp chữ)`, lệch.length === 0,
        lệch.slice(0, 3).join(' | '));
    // 3b. Mẫu quét phải THỰC SỰ phân biệt được hai luật, không thì phép canh
    //     trên chỉ là canh suông.
    ok('mẫu quét có lá số mà luật "bốn mốc" cho kết quả KHÁC — phép canh trên có răng',
        phânBiệt >= 5, `mới ${phânBiệt} lá số phân biệt được`);
    ok('mẫu quét có đủ lá số DÍNH không vong để canh (không phải toàn ô trống)',
        sốCóKV >= 20, `mới ${sốCóKV}/${NGÀY.length} lá dính`);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 4. Hiện chỗ nào, ẩn chỗ nào, và tiếng Trung ── */
console.log('\nChỗ hiện, chỗ ẩn, và tiếng Trung');
{
    const { ctx, page, errs } = await mở();
    await đặtNgày(page, 1991, 7, 16, 16, 0);

    const ở = async () => page.evaluate(() => {
        const hàng = document.getElementById('ttThanSatRow');
        const ring = document.querySelector('#tuTruPanel .tt-kv-ring');
        return {
            hàngHiện: hàng ? getComputedStyle(hàng).display !== 'none' : null,
            ringHiện: ring ? getComputedStyle(ring).display !== 'none' : false,
            sốRing: document.querySelectorAll('#tuTruPanel .tt-kv-ring').length,
        };
    });
    const tabBatTu = await ở();
    ok('tab Bát Tự: hàng thần sát hiện', tabBatTu.hàngHiện === true);
    ok('tab Bát Tự: vòng tròn hiện', tabBatTu.ringHiện === true);
    check('tab Bát Tự: đúng ba vòng tròn', tabBatTu.sốRing, 3);

    await page.click('#tabQmdj');
    await page.waitForTimeout(700);
    const tabKyMon = await ở();
    ok('tab Kỳ Môn: hàng thần sát ẨN (hộp Bát Tự ở đó gọn hơn)', tabKyMon.hàngHiện === false);
    ok('tab Kỳ Môn: vòng tròn cũng ẨN, không để một dấu lẻ không lời giải thích',
        tabKyMon.ringHiện === false);
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(800);
    const zh = await đọc(page);
    check('tiếng Trung: chữ là 空亡', zh.ts.filter(Boolean)[0], '空亡');
    check('tiếng Trung: vẫn đúng ba cột', zh.ts.map(Boolean), [true, true, true, false]);
    check('tiếng Trung: vẫn đúng ba vòng tròn', zh.ring, [true, true, true, false]);
    await page.evaluate(() => window.setLang('vi'));
    await page.waitForTimeout(600);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 5. Ba máy đích: chữ không tràn, không bẻ giữa chừng, vòng tròn nằm gọn ── */
console.log('\nS21 · S21 FE · A51 — hàng thần sát và vòng tròn');
{
    for (const [ten, w, h, dpr] of [['S21', 360, 740, 3], ['S21 FE', 393, 790, 2.75], ['A51', 412, 852, 2.625]]) {
        const { ctx, page, errs } = await mở(w, h, dpr);
        for (const lang of ['vi', 'zh']) {
            if (lang === 'zh') { await page.evaluate(() => window.setLang('zh')); await page.waitForTimeout(700); }
            await đặtNgày(page, 1991, 7, 16, 16, 0);
            const m = await page.evaluate(() => {
                const out = { tràn: [], bẻ: [], ringNgoài: [], cao: 0 };
                document.querySelectorAll('#ttThanSatRow td').forEach(td => {
                    if (td.scrollWidth > td.clientWidth + 1) out.tràn.push(td.textContent.trim());
                });
                document.querySelectorAll('#tuTruPanel .tt-ts-item').forEach(el => {
                    const rg = document.createRange();
                    rg.selectNodeContents(el);
                    if (rg.getClientRects().length > 1) out.bẻ.push(el.textContent.trim());
                });
                document.querySelectorAll('#tuTruPanel .tt-kv-ring').forEach(r => {
                    const b = r.getBoundingClientRect(), c = r.parentElement.getBoundingClientRect();
                    if (b.top < c.top - 0.5 || b.right > c.right + 0.5 || b.bottom > c.bottom + 0.5) {
                        out.ringNgoài.push(r.parentElement.id);
                    }
                });
                const hàng = document.getElementById('ttThanSatRow');
                out.cao = +hàng.getBoundingClientRect().height.toFixed(1);
                out.bảngTràn = document.getElementById('tuTruPanel').scrollWidth
                    - document.getElementById('tuTruPanel').clientWidth;
                return out;
            });
            ok(`${ten} ${lang}: chữ thần sát không tràn khỏi ô`, m.tràn.length === 0, JSON.stringify(m.tràn));
            ok(`${ten} ${lang}: chữ thần sát không bị bẻ giữa chừng`, m.bẻ.length === 0, JSON.stringify(m.bẻ));
            ok(`${ten} ${lang}: vòng tròn nằm gọn trong ô chi`, m.ringNgoài.length === 0, JSON.stringify(m.ringNgoài));
            ok(`${ten} ${lang}: hộp Bát Tự không rộng quá khung`, m.bảngTràn <= 1, `${m.bảngTràn}px`);
            console.log(`       ${ten} ${lang}: hàng thần sát cao ${m.cao}px`);
        }
        ok(`${ten}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
if (fail) process.exit(1);
console.log('✓ Thần sát: không vong đúng luật, đúng chỗ, đúng hai thứ tiếng');
