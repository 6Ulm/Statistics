/**
 * Tab "Lệnh" — nhân nguyên tư lệnh (人元司令分野).
 *
 *   node test_lenh.mjs
 *
 * Bốn nhóm:
 *   1. SỐ HỌC   — cả BA bộ số phân dã cộng đủ 30° mỗi tháng, các đoạn nối
 *                 liền nhau không hở không chồng, và mốc mở tháng TRÙNG KHÍT
 *                 bảng tiết khí mà tab Lịch với tab Kỳ Môn đang dùng.
 *   2. ĐỐI CHIẾU— bộ Uyên Hải so với bảng mẫu người dùng gửi (2026, UTC+8),
 *                 và ba bộ số so với NGUYÊN VĂN chữ Hán chép trong lenh.js.
 *   3. NHẤT QUÁN— chi của tháng lệnh luôn trùng chi của TRỤ THÁNG trong bảng
 *                 Bát Tự ngay bên trên, quét cả năm ở nhiều múi giờ, cả ba bộ.
 *   4. GIAO DIỆN— tab thứ ba dùng lại ĐÚNG ô ngày giờ và ĐÚNG bảng Bát Tự của
 *                 tab Kỳ Môn (cùng một phần tử, không phải bản chép); ô chọn
 *                 bộ số đứng CÙNG HÀNG với ô ngày giờ và đổi thì bảng đổi
 *                 theo; đổi địa điểm và ngôn ngữ thì bảng theo kịp; và không
 *                 máy nào phải kéo ngang.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { createRequire } from 'module';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');
const WEB = path.join(ASSETS, 'web');

let pass = 0, fail = 0;
function check(what, got, want) {
    const good = String(got) === String(want);
    good ? pass++ : fail++;
    console.log(`  ${good ? 'ok  ' : '✗ SAI'} ${what.padEnd(52)} ${good ? '' : `được "${got}", cần "${want}"`}`);
}
function ok(what, cond, detail) {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(52)} ${cond ? '' : (detail || '')}`);
}

/* ══════════════════════════════════════════════════════════════════════
   Phần 1–3 chạy thẳng trên Node: dựng lại đúng phép tính của lenh.js từ
   cùng những hàm mà nó gọi. Không mở trình duyệt cho những phép này —
   chúng là chuyện số học, và chạy trên Node thì nhanh hơn ba chục lần.
   ══════════════════════════════════════════════════════════════════════ */
const require = createRequire(import.meta.url);
require(path.join(WEB, 'js', 'astro_table.js'));
const { Solar, LunarYear, ShouXingUtil } = require(path.join(WEB, 'js', 'lunar.js'));

const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
const CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];
const CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];

/**
 * BA bảng phân dã, ĐỌC THẲNG TỪ lenh.js — không chép lại.
 *
 * Chép lại là dựng một nguồn sự thật thứ hai: sửa bảng trong lenh.js mà quên
 * sửa ở đây thì phép kiểm vẫn xanh trong khi ứng dụng đã đổi hành vi. Đọc từ
 * tệp thì phép kiểm luôn nói về ĐÚNG bộ số đang chạy, và việc đối chiếu với
 * cổ thư dồn hết vào nhóm 2 (so với nguyên văn chữ Hán).
 */
const SRC = fs.readFileSync(path.join(WEB, 'js', 'lenh.js'), 'utf8');
function readTable(name) {
    const i = SRC.indexOf('var ' + name + ' = {');
    if (i < 0) return null;
    const j = SRC.indexOf('\n    };', i);
    const body = SRC.slice(i, j);
    const out = {};
    for (const m of body.matchAll(/^\s*(\d+):\s*(\[\[[^\]]*\](?:,\s*\[[^\]]*\])*\])/gm)) {
        out[m[1]] = JSON.parse(m[2].replace(/\s+/g, ''));
    }
    return out;
}
const TABLES = {
    yhzp: { ten: 'Uyên Hải Tử Bình', fen: readTable('FEN_YHZP') },
    smth: { ten: 'Tam Mệnh Thông Hội', fen: readTable('FEN_SMTH') },
    zpzq: { ten: 'Tử Bình Chân Thuyên', fen: readTable('FEN_ZPZQ') },
};
const CHI_OF_JIE = { 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 0 };
const JIE_ORDER = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];
/**
 * BẢN KHÍ của mỗi chi — đoạn CUỐI của tháng phải đúng can này.
 *
 * KHÔNG canh hai đoạn đầu theo bảng tàng can: đoạn đầu (dư khí) là bản khí
 * của tháng TRƯỚC nên thường không nằm trong tàng can của chi này (Mão dư khí
 * là Giáp, Tý dư khí là Nhâm), còn đoạn giữa thì 三命通会 cố ý lấy can DƯƠNG
 * của mộ khố (Thìn dùng Nhâm chứ không Quý) — cả hai đều đúng sách, nên canh
 * theo tàng can là canh nhầm chỗ.
 */
const BAN_KHI = { 0: 9, 1: 5, 2: 0, 3: 1, 4: 4, 5: 2, 6: 3, 7: 5, 8: 6, 9: 7, 10: 4, 11: 8 };
const RAD = Math.PI / 180;

const atUTC8 = fn => {
    const snap = ShouXingUtil.getTzOffsetHours();
    ShouXingUtil.setTzOffsetHours(null);
    try { return fn(); } finally { ShouXingUtil.setTzOffsetHours(snap); }
};
const termJd = n => ShouXingUtil.qiAccurate(n * Math.PI / 12, 8) + Solar.J2000;
function lonJd(deg) {
    const t = ShouXingUtil.saLonT(deg * RAD) * 36525;
    return t - ShouXingUtil.dtT(t) + 8 / 24 + Solar.J2000;
}
function termIndexNear(jd) {
    let n = Math.round((jd - 2451259.0) / (365.2422 / 24));
    for (let k = -2; k <= 2; k++) if (Math.abs(termJd(n + k) - jd) < 3) return n + k;
    return n;
}
function buildYear(Y, fen) {
    const jds = atUTC8(() => LunarYear.fromYear(Y).getJieQiJulianDays().slice());
    return JIE_ORDER.map(k => {
        const n0 = termIndexNear(jds[k + 1]);
        let off = 0;
        const parts = fen[k].map(([can, span]) => {
            const p = {
                can, span, from: 15 * n0 + off, to: 15 * n0 + off + span,
                jdFrom: off === 0 ? termJd(n0) : lonJd(15 * n0 + off),
                jdTo: (off + span) === 30 ? termJd(n0 + 2) : lonJd(15 * n0 + off + span),
            };
            off += span;
            return p;
        });
        return { chi: CHI_OF_JIE[k], jie: k, n: n0, parts, tổng: off };
    });
}
const fmt8 = jd => {
    const s = Solar.fromJulianDay(jd), p = n => String(n).padStart(2, '0');
    return `${p(s.getDay())}/${p(s.getMonth())} ${p(s.getHour())}:${p(s.getMinute())}`;
};

/* ── 1. Số học, cả ba bộ ── */
console.log('Ba bảng phân dã: cộng đủ, nối liền, và trùng bảng tiết khí');
for (const [key, T] of Object.entries(TABLES)) {
    ok(`${T.ten}: lenh.js khai đủ 12 tháng`, T.fen && Object.keys(T.fen).length === 12,
        T.fen ? Object.keys(T.fen).length + ' tháng' : 'không đọc được bảng');
    if (!T.fen || Object.keys(T.fen).length !== 12) continue;

    let bad30 = [], badChain = [], badJie = [], badCan = [];
    for (const k of JIE_ORDER) {
        const tổng = T.fen[k].reduce((a, x) => a + x[1], 0);
        if (tổng !== 30) bad30.push(`${CHI[CHI_OF_JIE[k]]}=${tổng}°`);
        const cuối = T.fen[k][T.fen[k].length - 1][0];
        if (cuối !== BAN_KHI[CHI_OF_JIE[k]]) {
            badCan.push(`${CHI[CHI_OF_JIE[k]]} kết bằng ${CAN[cuối]}`);
        }
    }
    for (const Y of [2024, 2026, 2030, 2041]) {
        const year = buildYear(Y, T.fen);
        const jds = atUTC8(() => LunarYear.fromYear(Y).getJieQiJulianDays().slice());
        for (let i = 0; i < year.length; i++) {
            const mo = year[i];
            if (Math.abs(mo.parts[0].jdFrom - jds[mo.jie + 1]) > 1e-9) badJie.push(`${Y} ${CHI[mo.chi]}`);
            for (let j = 1; j < mo.parts.length; j++) {
                if (Math.abs(mo.parts[j].jdFrom - mo.parts[j - 1].jdTo) > 1e-9) {
                    badChain.push(`${Y} ${CHI[mo.chi]} đoạn ${j}`);
                }
            }
            const next = year[i + 1];
            if (next && Math.abs(mo.parts[mo.parts.length - 1].jdTo - next.parts[0].jdFrom) > 1e-9) {
                badChain.push(`${Y} ${CHI[mo.chi]}→${CHI[next.chi]}`);
            }
        }
    }
    ok(`${T.ten}: mỗi tháng cộng đủ 30°`, bad30.length === 0, bad30.join(', '));
    ok(`${T.ten}: đoạn cuối mỗi tháng là BẢN KHÍ của chi`, badCan.length === 0, badCan.join(', '));
    ok(`${T.ten}: các đoạn nối liền, không hở không chồng`, badChain.length === 0, badChain.join(', '));
    ok(`${T.ten}: mốc mở tháng trùng khít bảng tiết khí`, badJie.length === 0, badJie.join(', '));
}

/* ── 2a. Ba bộ số so với NGUYÊN VĂN chữ Hán chép ngay trong lenh.js ──
 *
 * Nguyên văn nằm trong khối ghi chú phía trên mỗi bảng. Phép kiểm này đọc
 * chữ Hán ấy, dịch ra số, rồi so với bảng bên dưới — nên gõ sai một số trong
 * bảng mà quên sửa nguyên văn (hoặc ngược lại) là ĐỎ. Không có nguyên văn
 * cho bộ Uyên Hải: nó là bản thông hành, và phép đối chiếu của nó là ảnh mẫu
 * người dùng gửi (nhóm 2b).
 */
console.log('\nBa bộ số khớp nguyên văn chữ Hán chép trong lenh.js');
{
    const SỐ = { 一: 1, 二: 2, 三: 3, 四: 4, 五: 5, 六: 6, 七: 7, 八: 8, 九: 9, 十: 10 };
    const hán = txt => {                       // "二十三" → 23, "十八" → 18
        let n = 0, cur = 0;
        for (const c of txt) {
            if (c === '十') { cur = (cur || 1) * 10; n += cur; cur = 0; }
            else { cur = SỐ[c]; }
        }
        return n + cur;
    };
    // "艮土"/"坤土" trong 三命通会 đều là MẬU (xem ghi chú trong lenh.js).
    const ganIdx = c => ({ 艮: 4, 坤: 4 })[c] ?? CAN_ZH.indexOf(c);
    const quote = name => {
        const i = SRC.indexOf('var ' + name + ' = {');
        const doc = SRC.lastIndexOf('/**', i);
        // Bỏ HẾT khoảng trắng: nguyên văn bị ngắt dòng trong khối ghi chú,
        // mà một cặp "can + số + 日" có thể nằm vắt qua hai dòng (三命通会:
        // "…丁火用事七日，甲木" / "墓库五日…") — để nguyên xuống dòng thì mất
        // đúng bốn cặp ấy mà vẫn trông như đọc được.
        return SRC.slice(doc, i).replace(/^\s*\*\s?/gm, '').replace(/\s+/g, '');
    };
    // Nguyên văn xếp theo tháng GIÊNG trở đi (寅→丑); bảng thì xếp từ Sửu.
    const THEO_SACH = [3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23, 1];
    for (const [key, ten] of [['smth', 'Tam Mệnh Thông Hội'], ['zpzq', 'Tử Bình Chân Thuyên']]) {
        const doc = quote(key === 'smth' ? 'FEN_SMTH' : 'FEN_ZPZQ');
        // Mỗi cặp "can + số + 日" trong nguyên văn, theo đúng thứ tự xuất hiện.
        const seg = [];
        for (const g of doc.matchAll(
            /([甲乙丙丁戊己庚辛壬癸艮坤])[土火木金水己]*[^日，；。\n]{0,6}?([一二三四五六七八九十]+)日/g)) {
            seg.push([ganIdx(g[1]), hán(g[2])]);
        }
        const bảng = [];
        for (const k of THEO_SACH) for (const pr of TABLES[key].fen[k]) bảng.push(pr);
        check(`${ten}: nguyên văn tách ra đủ số đoạn`, seg.length, bảng.length);
        ok(`${ten}: bảng khớp nguyên văn từng đoạn`,
            JSON.stringify(seg) === JSON.stringify(bảng),
            `nguyên văn ${JSON.stringify(seg)}`);
    }
}

/* ── 2b. Bộ Uyên Hải so với ảnh mẫu ──
 *
 * Ảnh mẫu người dùng gửi, năm 2026 ở mốc UTC+8. Nguồn ấy LÀM TRÒN tới phút
 * gần nhất, còn cả ứng dụng này thì CẮT (Solar.fromJulianDay bỏ phần giây) —
 * nên cho phép lệch một phút, nhưng KHÔNG hơn.
 */
console.log('\nBộ Uyên Hải đối chiếu ảnh mẫu (2026, UTC+8)');
{
    const REF = ['05/01 16:23', '14/01 12:23', '17/01 11:03', '04/02 04:02', '11/02 01:53',
        '18/02 00:05', '05/03 21:59', '15/03 22:12', '05/04 02:40', '14/04 06:31',
        '17/04 08:01', '05/05 19:49', '10/05 23:50', '20/05 07:41', '05/06 23:48',
        '16/06 10:45', '25/06 21:02', '07/07 09:57', '16/07 20:22', '19/07 23:48',
        '07/08 19:43', '15/08 02:46', '22/08 09:24', '07/09 22:41', '18/09 05:15',
        '08/10 14:29', '17/10 16:42', '20/10 17:13', '07/11 17:52', '14/11 16:58',
        '19/11 16:02', '07/12 10:52', '17/12 06:56'];
    const got = [], độ = [];
    for (const mo of buildYear(2026, TABLES.yhzp.fen)) for (const p of mo.parts) {
        got.push(fmt8(p.jdFrom));
        độ.push(`${CAN[p.can]} ${p.span}`);
    }
    check('bảng 2026 có đúng 33 đoạn', got.length, REF.length);
    let lệch = [];
    for (let i = 0; i < REF.length; i++) {
        const a = Date.parse('2026-' + got[i].slice(3, 5) + '-' + got[i].slice(0, 2) + 'T' + got[i].slice(6) + 'Z');
        const b = Date.parse('2026-' + REF[i].slice(3, 5) + '-' + REF[i].slice(0, 2) + 'T' + REF[i].slice(6) + 'Z');
        if (!(Math.abs(a - b) / 60000 <= 1)) lệch.push(`${got[i]} ≠ ${REF[i]}`);
    }
    ok('mọi mốc khớp ảnh mẫu trong vòng một phút', lệch.length === 0, lệch.join('; '));
    check('ba đoạn tháng Sửu', độ.slice(0, 3).join(' · '), 'Quý 9 · Tân 3 · Kỷ 18');
    check('ba đoạn tháng Dần', độ.slice(3, 6).join(' · '), 'Mậu 7 · Bính 7 · Giáp 16');
    check('hai đoạn tháng Dậu', độ.slice(23, 25).join(' · '), 'Canh 10 · Tân 20');
    check('hai đoạn tháng Tý', độ.slice(31, 33).join(' · '), 'Nhâm 10 · Quý 20');
}

/* ── 2c. Ba bộ phải THỰC SỰ khác nhau ──
 * Bằng không ô chọn chỉ là cái nút không làm gì.
 */
console.log('\nBa bộ số khác nhau thật');
{
    const j = k => JSON.stringify(TABLES[k].fen);
    ok('Uyên Hải ≠ Tam Mệnh', j('yhzp') !== j('smth'));
    ok('Uyên Hải ≠ Tử Bình Chân Thuyên', j('yhzp') !== j('zpzq'));
    ok('Tam Mệnh ≠ Tử Bình Chân Thuyên', j('smth') !== j('zpzq'));
    // Nét riêng của Tam Mệnh: bốn tháng mộ khố lấy can DƯƠNG làm trung khí.
    const trung = k => [7, 13, 19, 1].map(x => CAN[TABLES[k].fen[x][1][0]]).join(',');
    // Sửu là mộ khố KIM, nên can dương của nó là Canh (bản thông hành dùng Tân).
    check('Tam Mệnh: trung khí bốn tháng mộ khố', trung('smth'), 'Nhâm,Giáp,Bính,Canh');
    check('Uyên Hải: trung khí bốn tháng mộ khố', trung('yhzp'), 'Quý,Ất,Đinh,Tân');
}

/* ── 3. Chi của tháng lệnh phải trùng TRỤ THÁNG, với MỌI bộ số ── */
console.log('\nQuét cả năm: chi tháng lệnh trùng trụ tháng của bảng Bát Tự');
{
    function lenhAt(jdUTC8, y, fen) {
        for (let dy = 0; dy >= -1; dy--) {
            for (const mo of buildYear(y + dy, fen)) for (const p of mo.parts) {
                if (jdUTC8 >= p.jdFrom && jdUTC8 < p.jdTo) return { mo, p };
            }
        }
        return null;
    }
    for (const [key, T] of Object.entries(TABLES)) {
        for (const tz of [7, -5]) {
            let n = 0, sai = [];
            for (let d = 0; d < 365; d += 5) {
                const dt = new Date(Date.UTC(2026, 0, 1 + d, 13, 17));
                const bj = new Date(dt.getTime() - tz * 3600000 + 8 * 3600000);
                const s = Solar.fromYmdHms(bj.getUTCFullYear(), bj.getUTCMonth() + 1,
                    bj.getUTCDate(), bj.getUTCHours(), bj.getUTCMinutes(), 0);
                const trụ = atUTC8(() => s.getLunar().getEightChar().getMonth());
                const r = lenhAt(s.getJulianDay(), 2026, T.fen);
                n++;
                if (!r) { sai.push(s.toYmd() + ' không tra ra'); continue; }
                if (CHI_ZH[r.mo.chi] !== trụ[1]) sai.push(`${s.toYmd()} trụ ${trụ} ≠ ${CHI[r.mo.chi]}`);
            }
            ok(`${T.ten} · UTC${tz >= 0 ? '+' : ''}${tz}: ${n} thời điểm đều khớp`,
                sai.length === 0, sai.slice(0, 3).join('; '));
        }
    }
}

/* ══════════════════════════════════════════════════════════════════════
   Phần 4: giao diện
   ══════════════════════════════════════════════════════════════════════ */
let chromium;
try { ({ chromium } = await import('playwright')); }
catch (e) {
    console.log('\nBỏ qua phần giao diện: chưa cài playwright.');
    console.log(`\n${pass} đạt · ${fail} hỏng`);
    process.exit(fail ? 1 : 0);
}

const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
    const file = path.join(WEB, rel);
    if (!file.startsWith(WEB) || !fs.existsSync(file)) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/index.html`;
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

async function open(ctx) {
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await ctx.addInitScript(() => { try { localStorage.setItem('defaultLang', 'vi'); } catch (e) {} });
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    return { page, errs };
}

console.log('\nTab thứ ba: đúng chỗ, và dùng lại đúng hai khối của tab Kỳ Môn');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);

    const tabs = await page.$$eval('#tabBar .tab-item', els => els.map(e => e.id));
    check('thanh tab có đúng ba mục, Lệnh đứng cuối', tabs.join(','), 'tabQmdj,tabCal,tabLenh');

    // Đánh dấu bảng Bát Tự lúc đang ở tab Kỳ Môn, rồi sang tab Lệnh xem CÓ
    // CÒN đúng cái đã đánh dấu không: chỉ cách này mới phân biệt "dùng lại
    // đúng phần tử" với "dựng một bản trông giống".
    await page.evaluate(() => { document.getElementById('tuTruPanel').dataset.moc = 'x'; });
    const qm = await page.evaluate(() => ({
        bazi: document.getElementById('tuTruPanel').innerText.replace(/\s+/g, ' ').trim(),
        ngày: document.getElementById('dateDisplayText').textContent,
    }));

    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    const ln = await page.evaluate(() => {
        const tt = document.getElementById('tuTruPanel');
        const cs = getComputedStyle(tt);
        return {
            moc: tt.dataset.moc,
            hiện: cs.display !== 'none',
            bazi: tt.innerText.replace(/\s+/g, ' ').trim(),
            ngày: document.getElementById('dateDisplayText').textContent,
            ngàyHiện: getComputedStyle(document.getElementById('dateDisplayBtn')).display !== 'none',
            pháiẨn: getComputedStyle(document.getElementById('methodDisplayBtn')).display === 'none',
            // Hỏi CHIỀU CAO THẬT chứ không hỏi `display` của chính nó: lúc
            // chạy, app.js bọc #board vào #boardFlipScene (hiệu ứng lật thẻ),
            // nên thứ bị luật `body.view-lenh > *` ẩn đi là cái bọc — còn
            // display của #board vẫn là "grid" như thường.
            bànẨn: document.getElementById('board').getBoundingClientRect().height === 0,
            lịchẨn: document.getElementById('calView').getBoundingClientRect().height === 0,
        };
    });
    ok('bảng Bát Tự ở tab Lệnh là CHÍNH phần tử của tab Kỳ Môn', ln.moc === 'x');
    ok('…và đang hiện', ln.hiện);
    check('…với đúng nội dung ấy', ln.bazi, qm.bazi);
    ok('ô ngày giờ cũng là chính ô của tab Kỳ Môn, đang hiện', ln.ngàyHiện);
    check('…và đúng chuỗi ngày giờ ấy', ln.ngày, qm.ngày);
    ok('ô chọn phái đã ẩn (chuyện của riêng Kỳ Môn)', ln.pháiẨn);
    ok('bàn Kỳ Môn đã ẩn', ln.bànẨn);
    ok('tab Lịch đã ẩn', ln.lịchẨn);

    // "Lệnh: X" phải khớp hàng đang tô đậm trong bảng.
    const now = await page.evaluate(() => ({
        val: document.getElementById('lenhNowVal').textContent.trim(),
        hàng: document.getElementById('lenhActive')
            ? document.getElementById('lenhActive').cells[
                document.getElementById('lenhActive').cells.length - 4].textContent.trim()
            : null,
        sốHàng: document.querySelectorAll('#lenhBody tbody tr').length,
        cột: [...document.querySelectorAll('#lenhBody thead th')].map(e => e.textContent.trim()),
    }));
    check('bảng đủ 33 đoạn của năm', now.sốHàng, 33);
    check('năm cột đúng như ảnh mẫu', now.cột.join('|'), 'Tháng|Can|Số độ|Vào lệnh|Hết lệnh');
    ok('"Lệnh: X" khớp hàng đang tô đậm', now.val && now.val === now.hàng,
        `dòng "${now.val}" vs hàng "${now.hàng}"`);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

console.log('\nÔ chọn bộ số: cùng hàng với ô ngày giờ, và đổi thì bảng đổi theo');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);

    // Ở tab Kỳ Môn ô này phải ẨN: hàng ấy vốn đã khít (ngày giờ · phái · Đầy đủ).
    const qm = await page.evaluate(() =>
        getComputedStyle(document.getElementById('lenhRuleBtn')).display === 'none');
    ok('ở tab Kỳ Môn ô chọn bộ số ẩn hẳn', qm);

    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    const hàng = await page.evaluate(() => {
        const r = id => document.getElementById(id).getBoundingClientRect();
        const d = r('dateDisplayBtn'), l = r('lenhRuleBtn'), row = r('qmRow');
        return {
            cùngHàng: Math.abs(d.top - l.top) < 2 && Math.abs(d.bottom - l.bottom) < 2,
            phủHàng: Math.abs((d.width + l.width) - row.width) < 12,
            nhãn: document.getElementById('lenhRuleText').textContent.trim(),
            cắt: document.getElementById('lenhRuleText').scrollWidth >
                 document.getElementById('lenhRuleText').clientWidth + 1,
        };
    });
    ok('ô chọn bộ số nằm ĐÚNG hàng với ô ngày giờ', hàng.cùngHàng);
    ok('hai ô chia nhau trọn hàng', hàng.phủHàng);
    ok('nhãn bộ số không bị cắt chữ', !hàng.cắt, hàng.nhãn);
    check('mặc định là bản thông hành', hàng.nhãn, 'Uyên Hải Tử Bình');

    // Bảng chọn có đúng ba sách.
    await page.click('#lenhRuleBtn');
    await page.waitForTimeout(400);
    const opts = await page.$$eval('#optList .opt-row',
        els => els.map(e => e.getAttribute('data-value')));
    check('bảng chọn có đúng ba sách', opts.join(','), 'yhzp,smth,zpzq');

    // Đổi sang 三命通会: tháng Dần phải từ 7·7·16 thành 5·5·20, và giờ vào
    // lệnh của đoạn giữa phải dịch theo — đổi mỗi cái nhãn thì vô nghĩa.
    const dan = () => page.evaluate(() => {
        const rows = [...document.querySelectorAll('#lenhBody tbody tr')];
        let mon = null, out = [];
        for (const r of rows) {
            const c = r.querySelector('.lenh-mon');
            if (c) mon = c.textContent.replace(/\s+/g, '');
            if (!/^Dần|^寅/.test(mon || '')) continue;
            const cells = [...r.cells].map(e => e.textContent.trim());
            out.push(cells.slice(-4).join('|'));
        }
        return { dan: out, lenh: document.getElementById('lenhNowVal').textContent.trim(),
                 sốHàng: rows.length };
    });
    const trước = await dan();
    await page.click('.opt-row[data-value="smth"]');
    await page.waitForTimeout(900);
    const sau = await dan();
    check('Uyên Hải: tháng Dần 7·7·16', trước.dan.map(x => x.split('|')[1]).join('·'), '7·7·16');
    check('Tam Mệnh: tháng Dần 5·5·20', sau.dan.map(x => x.split('|')[1]).join('·'), '5·5·20');
    ok('giờ vào lệnh đoạn giữa cũng dịch theo',
        trước.dan[1].split('|')[2] !== sau.dan[1].split('|')[2],
        `vẫn ${sau.dan[1].split('|')[2]}`);
    check('Tam Mệnh có 32 đoạn (bốn tháng chỉ hai đoạn)', sau.sốHàng, 32);
    check('nhãn ô đã đổi', await page.textContent('#lenhRuleText'), 'Tam Mệnh Thông Hội');

    // Nhớ lựa chọn qua lần mở sau — người dùng theo một phái, không chọn lại
    // mỗi lần mở app.
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    check('mở lại vẫn nhớ bộ số đã chọn', await page.textContent('#lenhRuleText'), 'Tam Mệnh Thông Hội');
    check('…và bảng vẫn là bộ ấy', (await dan()).dan.map(x => x.split('|')[1]).join('·'), '5·5·20');

    // Tên sách sang tiếng Trung.
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(800);
    check('tên sách sang tiếng Trung', await page.textContent('#lenhRuleText'), '三命通会');
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

console.log('\nĐịa điểm và ngôn ngữ dùng chung cho CẢ BA tab');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);

    const trước = await page.evaluate(() =>
        document.querySelector('#lenhBody tbody tr td:nth-last-child(2)').textContent.trim());

    // Đổi địa điểm NGAY TẠI tab Lệnh (hàng dùng chung nằm dưới cả ba tab).
    await page.evaluate(() => window.QMDJLocation.apply(
        window.QMDJLocation.makeLoc('Hà Nội', 21.0278, 105.8342, 'Asia/Ho_Chi_Minh')));
    await page.waitForTimeout(900);
    const sau = await page.evaluate(() => ({
        giờ: document.querySelector('#lenhBody tbody tr td:nth-last-child(2)').textContent.trim(),
        nước: document.getElementById('countryDisplayText').textContent,
    }));
    ok('đổi địa điểm thì giờ vào lệnh đổi theo', sau.giờ !== trước, `vẫn ${trước}`);
    ok('ô địa điểm hiện tên mới ngay tại tab Lệnh', /Hà Nội|21\.03|105\.83/i.test(sau.nước), sau.nước);

    // Giờ vào lệnh của mốc mở tháng phải TRÙNG bảng tiết khí ở tab Lịch — cùng
    // một thời điểm thiên văn thì không được hiện hai con số.
    const lệnh = await page.evaluate(() => {
        const hàng = [...document.querySelectorAll('#lenhBody tbody tr')];
        const out = {};
        for (const r of hàng) {
            const mon = r.querySelector('.lenh-mon');
            if (!mon) continue;
            const tiết = mon.textContent.match(/\(([^)]+)\)/)[1];
            out[tiết] = r.cells[r.cells.length - 2].textContent.trim();
        }
        return out;
    });
    await page.click('#tabCal');
    await page.waitForTimeout(900);
    const lịch = await page.evaluate(() => {
        const out = {};
        for (const r of document.querySelectorAll('#calJieQi tbody tr')) {
            out[r.cells[0].textContent.trim()] = r.cells[1].textContent.trim();
        }
        return out;
    });
    let lệch = [];
    for (const [tiết, giờ] of Object.entries(lệnh)) {
        const jq = lịch[tiết];
        if (!jq) continue;                       // tiết của năm khác, bảng Lịch không có
        // Tab Lệnh viết "05/03 21:58", tab Lịch viết "05-03-2026 21:58".
        const a = giờ.replace(/\//g, '-').replace(/^(\d\d-\d\d)(?:-(\d{4}))? /, '$1 ');
        const b = jq.replace(/^(\d\d-\d\d)-\d{4} /, '$1 ');
        if (a !== b) lệch.push(`${tiết}: Lệnh ${a} ≠ Lịch ${b}`);
    }
    ok('giờ mở tháng trùng khít bảng tiết khí ở tab Lịch', lệch.length === 0, lệch.join('; '));

    // Đổi ngôn ngữ ở tab Lịch rồi quay lại: nhãn tab và bảng phải sang tiếng Trung.
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(800);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    const zh = await page.evaluate(() => ({
        nhãn: document.querySelector('#tabLenh .tab-lbl').textContent,
        cột: [...document.querySelectorAll('#lenhBody thead th')].map(e => e.textContent.trim()).join('|'),
        tháng: document.querySelector('#lenhBody .lenh-mon').textContent.replace(/\s+/g, ''),
    }));
    check('nhãn tab sang tiếng Trung', zh.nhãn, '令');
    check('tên cột sang tiếng Trung', zh.cột, '月|天干|度数|入令|退令');
    ok('tên tháng và tiết sang tiếng Trung', /^[子丑寅卯辰巳午未申酉戌亥]\(.+\)$/.test(zh.tháng), zh.tháng);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

console.log('\nVừa khít trên từng máy (không cắt chữ, không kéo ngang)');
for (const d of [{ n: 'S21', w: 360, h: 740 }, { n: 'S21 FE', w: 393, h: 790 }, { n: 'A51', w: 412, h: 852 }]) {
    for (const lang of ['vi', 'zh']) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await open(ctx);
        if (lang === 'zh') { await page.evaluate(() => window.setLang('zh')); await page.waitForTimeout(600); }
        await page.click('#tabLenh');
        await page.waitForTimeout(900);
        const g = await page.evaluate(() => {
            const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
            const box = document.getElementById('lenhBody');
            const dock = document.getElementById('bottomDock');
            const kids = [...document.body.children].filter(e => {
                const c = getComputedStyle(e);
                return c.display !== 'none' && c.position !== 'fixed';
            });
            const low = Math.max(...kids.map(e => e.getBoundingClientRect().bottom / z));
            const cut = [];
            document.querySelectorAll('#lenhBody th, #lenhBody td, #lenhNow, #lenhTitle')
                .forEach(c => { if (c.scrollWidth > c.clientWidth + 1) cut.push((c.textContent || '').trim().slice(0, 14)); });
            // Hàng đầu KHÔNG được nằm cụt dưới hàng tiêu đề dính.
            const th = box.querySelector('thead');
            const edge = box.getBoundingClientRect().top / z
                + (parseFloat(getComputedStyle(box).borderTopWidth) || 0)
                + th.getBoundingClientRect().height / z;
            let cụt = null;
            for (const r of box.querySelectorAll('tbody tr')) {
                const q = r.getBoundingClientRect();
                if (q.top / z < edge - 0.5 && q.bottom / z > edge + 0.5) {
                    cụt = (r.cells[0].textContent || '').trim().slice(0, 14);
                }
            }
            return {
                thừa: +(dock.getBoundingClientRect().top / z - low).toFixed(1),
                cắt: cut, cụt,
                ngang: box.scrollWidth > box.clientWidth + 1,
                trang: document.documentElement.scrollWidth > window.innerWidth + 1,
                hiện: [...box.querySelectorAll('tbody tr')].filter(r => {
                    const q = r.getBoundingClientRect(), c = box.getBoundingClientRect();
                    return q.top >= c.top - 1 && q.bottom <= c.bottom + 1;
                }).length,
            };
        });
        const tag = `${d.n} ${d.w}px · ${lang}`;
        ok(`${tag}: không ô nào bị cắt chữ`, g.cắt.length === 0, g.cắt.join(', '));
        ok(`${tag}: bảng không phải kéo ngang`, !g.ngang);
        ok(`${tag}: trang không tràn ngang`, !g.trang);
        ok(`${tag}: không tràn xuống dưới thanh tab`, g.thừa >= 0, `${g.thừa}px`);
        ok(`${tag}: không còn dải trống ở đáy`, g.thừa <= 16, `còn thừa ${g.thừa}px`);
        ok(`${tag}: hàng đầu không nằm cụt dưới hàng tiêu đề`, !g.cụt, g.cụt);
        ok(`${tag}: hiện được ít nhất 12 đoạn`, g.hiện >= 12, `${g.hiện} đoạn`);

        // ── Kéo được tới đủ 12 tháng ──
        // Bảng 33 đoạn không bao giờ vừa một màn, nên KÉO ĐƯỢC là chuyện sống
        // còn, không phải tiện nghi. Và phải thử bằng NGÓN TAY thật: lỗi đã gặp
        // là `overflow-y: hidden` (thừa kế luật "mục đang đóng" của tab Lịch),
        // mà nó vẫn cho JS đặt scrollTop — kiểm bằng `box.scrollTop = n` thì
        // đạt, người dùng vẫn chịu chết.
        const cuộn = await page.evaluate(() => {
            const b = document.getElementById('lenhBody');
            return { của: getComputedStyle(b).overflowY, dư: b.scrollHeight - b.clientHeight };
        });
        ok(`${tag}: khung bảng mở quyền cuộn dọc`,
            cuộn.của === 'auto' || cuộn.của === 'scroll', cuộn.của);

        const hộp = await page.evaluate(() => {
            const q = document.getElementById('lenhBody').getBoundingClientRect();
            return { x: q.left + q.width / 2, y: q.top + q.height / 2, h: q.height };
        });
        await page.evaluate(() => { document.getElementById('lenhBody').scrollTop = 0; });
        await page.waitForTimeout(150);
        const cdp = await ctx.newCDPSession(page);
        await cdp.send('Input.dispatchTouchEvent',
            { type: 'touchStart', touchPoints: [{ x: hộp.x, y: hộp.y + hộp.h / 3 }] });
        for (let i = 1; i <= 8; i++) {
            await cdp.send('Input.dispatchTouchEvent',
                { type: 'touchMove', touchPoints: [{ x: hộp.x, y: hộp.y + hộp.h / 3 - i * 25 }] });
            await page.waitForTimeout(25);
        }
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
        await page.waitForTimeout(600);
        const sauKéo = await page.evaluate(() => document.getElementById('lenhBody').scrollTop);
        ok(`${tag}: ngón tay kéo được bảng`, sauKéo > 0, `scrollTop vẫn ${sauKéo}`);

        // Kéo hết cỡ: đủ 12 tháng phải với tới được, tháng cuối (Sửu) hiện trọn.
        await page.evaluate(() => {
            const b = document.getElementById('lenhBody');
            b.scrollTop = b.scrollHeight;
        });
        await page.waitForTimeout(250);
        const đáy = await page.evaluate(() => {
            const b = document.getElementById('lenhBody');
            const c = b.getBoundingClientRect();
            const cuối = [...b.querySelectorAll('tbody tr')].pop();
            return {
                tháng: new Set([...b.querySelectorAll('.lenh-mon')]
                    .map(e => e.textContent.replace(/\s+/g, ' ').trim())).size,
                cuốiTrọn: cuối.getBoundingClientRect().bottom <= c.bottom + 1,
            };
        });
        ok(`${tag}: bảng có đủ 12 tháng`, đáy.tháng === 12, `${đáy.tháng} tháng`);
        ok(`${tag}: kéo tới đáy thì đoạn cuối hiện trọn`, đáy.cuốiTrọn);

        ok(`${tag}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
if (fail) process.exit(1);
console.log('✓ Tab Lệnh đúng bảng, đúng giờ, và dùng chung địa điểm với hai tab kia');
