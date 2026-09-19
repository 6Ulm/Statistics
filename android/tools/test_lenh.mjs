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
    for (const [key, ten] of [['smth', 'Tam Mệnh Thông Hội']]) {
        const doc = quote('FEN_SMTH');
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

/* ── 2c. Hai bộ phải THỰC SỰ khác nhau ──
 * Bằng không ô chọn chỉ là cái nút không làm gì.
 *
 * Tử Bình Chân Thuyên ĐÃ BỎ (xem khối ghi chú đầu js/lenh.js) — canh luôn
 * rằng nó đi hẳn, cả bảng phân dã lẫn dòng trong RULES: bỏ nửa vời thì ô chọn
 * vẫn hiện một sách mà bảng phân dã của nó không còn, và người dùng bấm vào là
 * ra bộ số của sách khác mà không hay.
 */
console.log('\nHai bộ số khác nhau thật');
{
    const j = k => JSON.stringify(TABLES[k].fen);
    ok('Uyên Hải ≠ Tam Mệnh', j('yhzp') !== j('smth'));
    ok('bảng FEN_ZPZQ đã xoá khỏi lenh.js', readTable('FEN_ZPZQ') === null);
    ok('RULES không còn dòng zpzq', !/key:\s*'zpzq'/.test(SRC));
    ok('không còn nhãn TBCT nào trong mã', !/\bTBCT\b/.test(SRC));
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

/**
 * Mở tab TRA CỨU ở năm `Y` — nơi bảng Lệnh năm nay nằm.
 *
 * Bảng đã chuyển hẳn khỏi tab Bát Tự (xem #traCuuView trong index.html): ở tab
 * ấy nó tra theo năm sinh trong lá số, ở đây nó tra theo một năm người dùng tự
 * chọn. Cùng MỘT hàm dựng bảng cho cả hai (lenhTableHtml trong lenh.js, lộ ra
 * qua window.__lenhShared), nên mọi phép canh về cột, về số đoạn, về giờ vào
 * lệnh dưới đây vẫn đo đúng cái bảng mà tab Bát Tự từng hiện.
 *
 * Khác một điểm: KHÔNG có hàng nào được tô "đang cầm lệnh" — tra một năm bất
 * kỳ thì không có thời điểm sinh nào để mà cầm lệnh.
 */
/** Bốn mục của tab Tra cứu, theo thứ tự trên xuống: [tiêu đề, thân]. */
const TC_MỤC = [['trinhuanHeader', 'trinhuanBody'], ['sachboHeader', 'sachboBody'],
                ['lenhHead', 'lenhSec'], ['tcAmHead', 'tcAmSec']];

/**
 * Mở tab Tra cứu ở một năm, rồi (mặc định) bung hết các mục ra.
 *
 * Bốn mục nay ĐÓNG SẴN — người dùng chốt "hiển thị cả ba tab… click vào thì
 * expand hoặc hide" — nên phép đo nào cần thấy nội dung phải tự mở lấy.
 * Truyền `mở = false` khi muốn đo đúng trạng thái vừa chọn năm xong.
 */
async function mởTraCuu(page, Y, mở = true) {
    await page.evaluate(y => {
        if (y !== undefined && y !== null) window.__tracuuYear(y);
        window.showTab('tracuu');
    }, Y);
    await page.waitForTimeout(900);
    if (!mở) return;
    await page.evaluate(mục => {
        for (const [head, body] of mục) {
            const b = document.getElementById(body), h = document.getElementById(head);
            if (b && h && getComputedStyle(b).display === 'none') h.click();
        }
    }, TC_MỤC);
    await page.waitForTimeout(800);
}

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
    check('thanh tab có đúng bốn mục, Bát Tự thứ ba', tabs.join(','),
        'tabQmdj,tabCal,tabLenh,tabTraCuu');

    // Đánh dấu bảng Bát Tự lúc đang ở tab Kỳ Môn, rồi sang tab Lệnh xem CÓ
    // CÒN đúng cái đã đánh dấu không: chỉ cách này mới phân biệt "dùng lại
    // đúng phần tử" với "dựng một bản trông giống".
    await page.evaluate(() => { document.getElementById('tuTruPanel').dataset.moc = 'x'; });
    // `chung` = những hàng CẢ HAI tab đều hiện (can · chi · nạp âm · lịch âm).
    // Ba hàng .tt-bazi-only là của RIÊNG tab Bát Tự, nên đo tách ra.
    const đoBảng = () => page.evaluate(() => {
        const tt = document.getElementById('tuTruPanel');
        const hiện = el => el.getBoundingClientRect().height > 0;
        const txt = tr => [...tr.cells].map(c => c.textContent.trim()).join('|');
        const hàng = [...tt.querySelectorAll('tbody tr')];
        return {
            moc: tt.dataset.moc,
            chung: hàng.filter(tr => !tr.classList.contains('tt-bazi-only'))
                       .map(txt).join(' / '),
            riêng: hàng.filter(tr => tr.classList.contains('tt-bazi-only') && hiện(tr)).length,
            ngày: document.getElementById('dateDisplayText').textContent,
        };
    });
    const qm = await đoBảng();

    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    const ln = { ...(await đoBảng()), ...(await page.evaluate(() => {
        const tt = document.getElementById('tuTruPanel');
        const cs = getComputedStyle(tt);
        return {
            hiện: cs.display !== 'none',
            ngàyHiện: getComputedStyle(document.getElementById('dateDisplayBtn')).display !== 'none',
            pháiẨn: getComputedStyle(document.getElementById('methodDisplayBtn')).display === 'none',
            // Hỏi CHIỀU CAO THẬT chứ không hỏi `display` của chính nó: lúc
            // chạy, app.js bọc #board vào #boardFlipScene (hiệu ứng lật thẻ),
            // nên thứ bị luật `body.view-lenh > *` ẩn đi là cái bọc — còn
            // display của #board vẫn là "grid" như thường.
            bànẨn: document.getElementById('board').getBoundingClientRect().height === 0,
            lịchẨn: document.getElementById('calView').getBoundingClientRect().height === 0,
        };
    })) };
    ok('bảng Bát Tự ở tab Lệnh là CHÍNH phần tử của tab Kỳ Môn', ln.moc === 'x');
    ok('…và đang hiện', ln.hiện);
    // Cùng MỘT phần tử ⇒ những hàng dùng chung phải in ra y hệt nhau.
    check('…với đúng nội dung ấy ở các hàng dùng chung', ln.chung, qm.chung);
    // …nhưng ba hàng tàng can · phó tinh · hàng trống là CỦA RIÊNG tab Bát Tự:
    // hộp bát tự bên Kỳ Môn phải y như trước khi thêm chúng.
    check('hộp Bát Tự ở tab KỲ MÔN không mọc thêm hàng nào', qm.riêng, 0);
    check('…còn ở tab Bát Tự thì có đủ ba hàng riêng ấy', ln.riêng, 3);
    ok('ô ngày giờ cũng là chính ô của tab Kỳ Môn, đang hiện', ln.ngàyHiện);
    check('…và đúng chuỗi ngày giờ ấy', ln.ngày, qm.ngày);
    ok('ô chọn phái đã ẩn (chuyện của riêng Kỳ Môn)', ln.pháiẨn);
    ok('bàn Kỳ Môn đã ẩn', ln.bànẨn);
    ok('tab Lịch đã ẩn', ln.lịchẨn);

    // "Lệnh: X" ở tab Bát Tự phải khớp can đang cầm lệnh lúc sinh — nay đối
    // chiếu với chính bảng ở tab TRA CỨU của cùng năm ấy, vì bảng đã chuyển
    // sang đó. Cùng một hàm dựng bảng cho cả hai (lenhTableHtml trong lenh.js),
    // nên hai bên phải nói cùng một con số.
    const nămSinh = await page.$eval('#inYear', e => parseInt(e.value, 10));
    const tómTắt = await page.evaluate(() =>
        document.getElementById('lenhNowVal').textContent.trim());
    await mởTraCuu(page, nămSinh);
    const now = await page.evaluate(() => ({
        sốHàng: document.querySelectorAll('#lenhBody tbody tr').length,
        cột: [...document.querySelectorAll('#lenhBody thead th')].map(e => e.textContent.trim()),
        // Mọi can có mặt trong bảng — "Lệnh: X" phải là một trong số đó.
        can: [...document.querySelectorAll('#lenhBody .lenh-can')].map(e => e.textContent.trim()),
    }));
    check('bảng đủ 33 đoạn của năm', now.sốHàng, 33);
    check('năm cột đúng như ảnh mẫu', now.cột.join('|'), 'Tháng|Can|Số độ|Vào lệnh|Hết lệnh');
    ok('"Lệnh: X" của tab Bát Tự là một can CÓ THẬT trong bảng cùng năm',
        !!tómTắt && tómTắt !== '—' && now.can.indexOf(tómTắt) >= 0,
        `dòng "${tómTắt}" không có trong ${now.can.length} đoạn của năm ${nămSinh}`);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

console.log('\nÔ chọn bộ số: cùng hàng với ô ngày giờ, và đổi thì bảng đổi theo');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);

    // Ở tab Kỳ Môn hai ô này phải ẨN: hàng ấy vốn đã khít (ngày giờ · phái · Đầy đủ).
    const qm = await page.evaluate(() => ({
        rule: getComputedStyle(document.getElementById('lenhRuleBtn')).display === 'none',
        gender: getComputedStyle(document.getElementById('lenhGenderBtn')).display === 'none',
    }));
    ok('ở tab Kỳ Môn ô chọn bộ số ẩn hẳn', qm.rule);
    ok('ở tab Kỳ Môn ô Giới tính ẩn hẳn', qm.gender);

    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    const hàng = await page.evaluate(() => {
        const r = id => document.getElementById(id).getBoundingClientRect();
        const d = r('dateDisplayBtn'), g = r('lenhGenderBtn'), l = r('lenhRuleBtn'), row = r('qmRow');
        return {
            cùngHàng: Math.abs(d.top - l.top) < 2 && Math.abs(d.bottom - l.bottom) < 2
                   && Math.abs(g.top - l.top) < 2 && Math.abs(g.bottom - l.bottom) < 2,
            phủHàng: Math.abs((d.width + g.width + l.width) - row.width) < 14,
            // Ba ô đúng thứ tự trái→phải theo vạch 50%|75% của bảng Bát Tự.
            đúngThứTự: d.right <= g.left + 1 && g.right <= l.left + 1,
            nhãn: document.getElementById('lenhRuleText').textContent.trim(),
            nhãnGiới: document.getElementById('lenhGenderText').textContent.trim(),
            cắt: document.getElementById('lenhRuleText').scrollWidth >
                 document.getElementById('lenhRuleText').clientWidth + 1,
            cắtGiới: document.getElementById('lenhGenderText').scrollWidth >
                     document.getElementById('lenhGenderText').clientWidth + 1,
        };
    });
    ok('ô chọn bộ số nằm ĐÚNG hàng với ô ngày giờ và ô Giới tính', hàng.cùngHàng);
    ok('ba ô chia nhau trọn hàng', hàng.phủHàng);
    ok('ba ô đúng thứ tự: ngày giờ → Giới tính → Quy tắc', hàng.đúngThứTự);
    ok('nhãn bộ số không bị cắt chữ', !hàng.cắt, hàng.nhãn);
    ok('nhãn Giới tính không bị cắt chữ', !hàng.cắtGiới, hàng.nhãnGiới);
    check('mặc định là bản thông hành, viết tắt UHTB', hàng.nhãn, 'UHTB');
    check('mặc định Giới tính là Nam', hàng.nhãnGiới, 'Nam');

    // Bảng chọn có đúng hai sách, tiêu đề rút gọn còn "Quy tắc".
    await page.click('#lenhRuleBtn');
    await page.waitForTimeout(400);
    check('tiêu đề bảng chọn là "Quy tắc" (không phải "Chọn quy tắc")',
        await page.textContent('#optTitle'), 'Quy tắc');
    const opts = await page.$$eval('#optList .opt-row',
        els => els.map(e => e.getAttribute('data-value')));
    check('bảng chọn có đúng hai sách', opts.join(','), 'yhzp,smth');
    const optLabels = await page.$$eval('#optList .opt-name', els => els.map(e => e.textContent.trim()));
    // BẢNG CHỌN dùng tên ĐẦY ĐỦ, Ô NGOÀI dùng viết tắt — hai chỗ, hai bản.
    // "UHTB" thì không ai đoán ra là sách nào nếu chưa quen, mà ô ngoài chỉ
    // rộng một phần tư hàng nên tên đầy đủ vào đó là bị "…" nuốt.
    check('bảng chọn hiện TÊN ĐẦY ĐỦ', optLabels.join(','),
        'Uyên Hải Tử Bình,Tam Mệnh Thông Hội');

    await page.click('.opt-row[data-value="smth"]');
    await page.waitForTimeout(900);
    check('ô ngoài vẫn là VIẾT TẮT', await page.textContent('#lenhRuleText'), 'TMTH');

    // Bộ số phải đổi THẬT chứ không đổi mỗi cái nhãn. Ở tab Bát Tự bảng Lệnh
    // năm đã chuyển đi (sang tab Tra cứu), nên chỗ duy nhất còn thấy tác dụng
    // ở đây là dòng tóm tắt "Lệnh: X" — can đang cầm lệnh lúc sinh, vốn tra từ
    // chính bảng phân dã của bộ số.
    const lệnhTheoBộ = () => page.evaluate(() =>
        document.getElementById('lenhNowVal').textContent.trim());
    const lệnhTMTH = await lệnhTheoBộ();
    ok('dòng tóm tắt có giá trị thật', lệnhTMTH && lệnhTMTH !== '—', lệnhTMTH);

    // Nhớ lựa chọn qua lần mở sau — người dùng theo một phái, không chọn lại
    // mỗi lần mở app.
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    check('mở lại vẫn nhớ bộ số đã chọn', await page.textContent('#lenhRuleText'), 'TMTH');

    // ── Bộ số ĐỔI THẬT BẢNG — kiểm ở tab Tra cứu, nơi bảng Lệnh năm nay nằm.
    // Ô bộ số ở đó là ô RIÊNG (#tcRuleBtn): ở tab Bát Tự bộ số thuộc về lá số
    // của người dùng, còn ở đây là một phép tra độc lập.
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
        return { dan: out, sốHàng: rows.length };
    });
    await mởTraCuu(page, 2026);
    await page.evaluate(() => window.__tracuuRule('yhzp'));
    await page.waitForTimeout(700);
    const trước = await dan();
    await page.evaluate(() => window.__tracuuRule('smth'));
    await page.waitForTimeout(700);
    const sau = await dan();
    check('Uyên Hải: tháng Dần 7·7·16', trước.dan.map(x => x.split('|')[1]).join('·'), '7·7·16');
    check('Tam Mệnh: tháng Dần 5·5·20', sau.dan.map(x => x.split('|')[1]).join('·'), '5·5·20');
    ok('giờ vào lệnh đoạn giữa cũng dịch theo',
        trước.dan[1].split('|')[2] !== sau.dan[1].split('|')[2],
        `vẫn ${sau.dan[1].split('|')[2]}`);
    check('Tam Mệnh có 32 đoạn (bốn tháng chỉ hai đoạn)', sau.sốHàng, 32);
    check('ô bộ số của Tra cứu hiện viết tắt', await page.textContent('#tcRuleText'), 'TMTH');

    // Ô bộ số của hai tab ĐỘC LẬP: đổi ở Tra cứu không kéo theo tab Bát Tự.
    await page.evaluate(() => window.__tracuuRule('yhzp'));
    await page.waitForTimeout(600);
    check('đổi bộ số ở Tra cứu…', await page.textContent('#tcRuleText'), 'UHTB');
    check('…không đụng tới bộ số của tab Bát Tự',
        await page.textContent('#lenhRuleText'), 'TMTH');
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    // Tên sách sang tiếng Trung — KHÔNG viết tắt (giữ nguyên chuyện đã kiểm
    // ở nhóm số học phía trên: chỉ tiếng Việt được rút gọn).
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(800);
    check('tên sách sang tiếng Trung', await page.textContent('#lenhRuleText'), '三命通会');
    // BẢNG CHỌN tiếng Trung cũng phải còn ĐÚNG HAI sách: bỏ một bộ mà chỉ dọn
    // phía tiếng Việt thì người dùng tiếng Trung vẫn bấm được vào 子平真诠.
    await page.click('#lenhRuleBtn');
    await page.waitForTimeout(400);
    check('bảng chọn tiếng Trung còn đúng hai sách',
        (await page.$$eval('#optList .opt-row', els => els.map(e => e.getAttribute('data-value')))).join(','),
        'yhzp,smth');
    check('tên sách tiếng Trung KHÔNG viết tắt',
        (await page.$$eval('#optList .opt-name', els => els.map(e => e.textContent.trim()))).join(','),
        '渊海子平,三命通会');
    await page.evaluate(() => window.closeOptionPicker());
    await page.waitForTimeout(300);
    await page.evaluate(() => window.setLang('vi'));
    await page.waitForTimeout(600);

    // ── Ô Giới tính: mở picker, tiêu đề đúng, đổi giá trị, nhớ qua lần mở sau.
    await page.click('#lenhGenderBtn');
    await page.waitForTimeout(400);
    check('tiêu đề bảng chọn Giới tính là "Giới tính"', await page.textContent('#optTitle'), 'Giới tính');
    const genderOpts = await page.$$eval('#optList .opt-row', els => els.map(e => e.getAttribute('data-value')));
    check('bảng chọn Giới tính có đúng hai giá trị', genderOpts.join(','), 'nam,nu');
    const genderLabels = await page.$$eval('#optList .opt-name', els => els.map(e => e.textContent.trim()));
    check('hai giá trị là Nam/Nữ', genderLabels.join(','), 'Nam,Nữ');
    await page.click('.opt-row[data-value="nu"]');
    await page.waitForTimeout(500);
    check('chọn Nữ thì ô hiện Nữ', await page.textContent('#lenhGenderText'), 'Nữ');
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(600);
    check('Nữ sang tiếng Trung là 女', await page.textContent('#lenhGenderText'), '女');
    await page.evaluate(() => window.setLang('vi'));
    await page.waitForTimeout(600);
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    check('mở lại vẫn nhớ Giới tính đã chọn', await page.textContent('#lenhGenderText'), 'Nữ');

    // ── Máy đang chọn bộ số ĐÃ BỎ: phải tự về bản thông hành, không treo.
    // Người dùng nào từng chọn Tử Bình Chân Thuyên thì khoá 'zpzq' còn nằm
    // trong localStorage sau khi cập nhật — ruleByKey() không tìm thấy thì
    // trả RULES[0], nên chỉ cần canh rằng đường ấy thật sự chạy (cả ở tab Bát
    // Tự lẫn tab Tra cứu, hai ô bộ số độc lập nhau).
    await page.evaluate(() => {
        localStorage.setItem('qmdj.lenhRule', 'zpzq');
        localStorage.setItem('qmdj.tracuuRule', 'zpzq');
    });
    await page.reload({ waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.click('#tabLenh');
    await page.waitForTimeout(900);
    check('khoá cũ "zpzq" rơi về bản thông hành', await page.textContent('#lenhRuleText'), 'UHTB');
    check('__lenhRule("zpzq") cũng trả về yhzp',
        await page.evaluate(() => window.__lenhRule('zpzq')), 'yhzp');

    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

console.log('\nĐịa điểm và ngôn ngữ dùng chung cho CẢ BA tab');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    // Bảng Lệnh năm nay ở tab Tra cứu — giờ vào/ra lệnh vẫn quy về giờ ĐỊA
    // PHƯƠNG của địa điểm đang chọn, nên đây vẫn là chỗ thấy tác dụng rõ nhất
    // của hàng dùng chung.
    await mởTraCuu(page, 2026);

    const trước = await page.evaluate(() =>
        document.querySelector('#lenhBody tbody tr td:nth-last-child(2)').textContent.trim());

    // Đổi địa điểm NGAY TẠI tab Tra cứu (hàng dùng chung nằm dưới cả bốn tab).
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
    await mởTraCuu(page, 2026);
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
        await mởTraCuu(page, 2026);
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
                // Nhân z: `thừa`/`low` ở trên đã chia cho z (px CHƯA phóng),
                // còn scrollHeight là px của TÀI LIỆU (đã phóng). So thẳng hai
                // bên là lệch đúng một hệ số zoom — đo ở tab Tra cứu (trang
                // dài nên z = 0,95) thấy ngay: 1488 vs 1414 = 1488 × 0,95.
                cuộnĐược: Math.round((document.documentElement.scrollHeight - window.innerHeight) / z),
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
        // Mở Lệnh năm ra thì trang DÀI HƠN một màn hình — đó là chủ ý, đúng
        // như tab Kỳ Môn khi mở Âm Bàn (đo: 1234px trên màn 866px). Nên không
        // canh "nội dung không được thò xuống dưới thanh tab" nữa; canh thứ
        // THẬT SỰ quan trọng: phần thò ra phải CUỘN TỚI ĐƯỢC. <body> có
        // padding-bottom đúng bằng chiều cao thanh dưới, nên cuộn hết trang là
        // thấy trọn hàng cuối, không bị thanh che.
        // Dung sai 1px: `thừa` giữ một chữ số thập phân còn `cuộnĐược` làm
        // tròn về số nguyên, nên hai bên lệch nhau tới nửa pixel chỉ vì cách
        // đọc, không phải vì thiếu chỗ cuộn (đo được 1488,3 vs 1488).
        ok(`${tag}: phần dôi ra cuộn tới được`,
            g.thừa >= 0 || g.cuộnĐược + 1 >= -g.thừa,
            `thò ${(-g.thừa).toFixed(1)}px, cuộn được ${g.cuộnĐược}px`);
        ok(`${tag}: không còn dải trống ở đáy`, g.thừa <= 16, `còn thừa ${g.thừa}px`);
        ok(`${tag}: hàng đầu không nằm cụt dưới hàng tiêu đề`, !g.cụt, g.cụt);
        // Ngưỡng hạ từ 8 xuống 5: ĐẠI VẬN giờ đứng TRƯỚC bảng này và không
        // còn maxHeight (luôn hiện trọn, xem fitDaiVan cũ đã bỏ) — bảng LỆNH
        // NĂM vì thế chỉ còn được đúng phần CÒN LẠI xuống thanh dưới, thường
        // chạm sàn BOX_MIN=120px (trước đây hiếm khi chạm, vì Đại Vận từng
        // chỉ xin một sàn nhỏ DV_MIN chừa TRƯỚC nó). Đo lại trên cả 3 máy:
        // thấp nhất 5 đoạn (kể cả tiêu đề tháng), vẫn đủ thấy ít nhất một
        // tháng trọn vẹn — đánh đổi có chủ đích của người dùng (Đại Vận luôn
        // đủ, LỆNH NĂM lùi thành chi tiết phụ), không phải hồi quy.
        ok(`${tag}: hiện được ít nhất 5 đoạn`, g.hiện >= 5, `${g.hiện} đoạn`);

        // ── Kéo được tới đủ 12 tháng — bằng CẢ TRANG, không phải bằng một ô
        // cuộn riêng ──
        // Bảng 33 đoạn không bao giờ vừa một màn, nên KÉO ĐƯỢC là chuyện sống
        // còn. Nhưng nó phải kéo đúng CÁCH của tab Kỳ Môn: bảng bung đủ chiều
        // cao tự nhiên và cả trang cuộn (xem fitLenhSec trong lenh.js).
        //
        // Bản trước kẹp bảng vào chỗ trống còn lại rồi cho nó tự cuộn bên
        // trong. Chỗ ấy chỉ còn 120px cho một bảng cao 683px, nên ngón tay đặt
        // xuống gần như chắc chắn rơi vào ô cuộn tí hon ấy và nó nuốt trọn cú
        // vuốt — Bát Tự, Đại Vận, mọi thứ phía trên đứng im. Đo thật trên A51:
        // cả trang chỉ cuộn được 46px, so với 368px khi mở Âm Bàn ở tab Kỳ Môn.
        const cuộn = await page.evaluate(() => {
            const b = document.getElementById('lenhBody');
            const t = b.querySelector('table');
            return {
                // KHÔNG đọc getComputedStyle().overflowY: lenh.css khai
                // `visible`, nhưng .cal-sec-body dùng chung đặt `overflow:auto`
                // cho trục NGANG (lưới an toàn), và theo đúng đặc tả CSS thì
                // `visible` ở một trục sẽ tính thành `auto` khi trục kia không
                // phải `visible`. Con số đọc ra vì thế luôn là 'auto' dù khai
                // gì. Đo THỨ THẬT SỰ quan trọng: có gì để cuộn dọc hay không.
                dưDọc: b.scrollHeight - b.clientHeight,
                kẹp: b.style.maxHeight || '',
                // Hộp phải cao ĐÚNG BẰNG bảng bên trong: không kẹp, không cắt.
                hụt: t ? Math.round(t.getBoundingClientRect().height
                                    - b.getBoundingClientRect().height) : 0,
            };
        });
        ok(`${tag}: khung bảng KHÔNG còn bị kẹp chiều cao`, cuộn.kẹp === '', cuộn.kẹp);
        ok(`${tag}: khung bảng KHÔNG còn gì để cuộn dọc bên trong`,
            cuộn.dưDọc <= 1, `còn dư ${cuộn.dưDọc}px`);
        ok(`${tag}: khung cao đủ ôm trọn bảng`, cuộn.hụt <= 2, `hụt ${cuộn.hụt}px`);

        // Đặt ngón tay lên GIỮA BẢNG rồi vuốt — thứ phải nhúc nhích là CẢ
        // TRANG. Đây chính là thao tác từng chết: người dùng kéo mà màn hình
        // trên đứng im.
        // Điểm đặt ngón tay phải nằm TRONG khung nhìn. Ở tab Tra cứu, bảng
        // Lệnh năm đứng sau hai bảng kia nên mép trên của nó ở tận đâu dưới
        // màn hình — chạm vào toạ độ ấy là chạm ra ngoài, không sự kiện nào
        // sinh ra và phép canh xanh/đỏ vô nghĩa. Kẹp vào giữa màn hình.
        const hộp = await page.evaluate(() => {
            const q = document.getElementById('lenhBody').getBoundingClientRect();
            const y = Math.min(Math.max(q.top + 40, 80), window.innerHeight - 120);
            return { x: q.left + q.width / 2, y: y };
        });
        await page.evaluate(() => window.scrollTo(0, 0));
        await page.waitForTimeout(150);
        const cdp = await ctx.newCDPSession(page);
        await cdp.send('Input.dispatchTouchEvent',
            { type: 'touchStart', touchPoints: [{ x: hộp.x, y: hộp.y }] });
        for (let i = 1; i <= 8; i++) {
            await cdp.send('Input.dispatchTouchEvent',
                { type: 'touchMove', touchPoints: [{ x: hộp.x, y: hộp.y - i * 25 }] });
            await page.waitForTimeout(25);
        }
        await cdp.send('Input.dispatchTouchEvent', { type: 'touchEnd', touchPoints: [] });
        await page.waitForTimeout(700);
        const sauKéo = await page.evaluate(() => Math.round(window.scrollY));
        ok(`${tag}: vuốt trên bảng thì CẢ TRANG cuộn`, sauKéo > 0, `scrollY vẫn ${sauKéo}`);

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

console.log('\nTuổi nhập đại vận: chiều thuận/nghịch chéo Giới tính × Can năm');
{
    // ── Công thức đổi tuổi (không đổi khi thêm chiều thuận/nghịch) ──
    function daiVanTuoi(diffDays) {
        const ageY = diffDays / 3, years = Math.floor(ageY + 1e-9);
        const monthsDec = (ageY - years) * 12, months = Math.floor(monthsDec + 1e-9);
        let days = Math.round((monthsDec - months) * 30);
        let Y = years, M = months;
        if (days >= 30) { days -= 30; M += 1; }
        if (M >= 12) { M -= 12; Y += 1; }
        return { years: Y, months: M, days };
    }
    const ex = daiVanTuoi(10.5);
    ok('công thức: 10,5 ngày / 3 = 3,5 tuổi = 3 tuổi 6 tháng',
        ex.years === 3 && ex.months === 6 && ex.days === 0, JSON.stringify(ex));

    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    /** Đặt ngày giờ sinh qua chính các <select> ẩn — cùng cách bàn tay người
     *  dùng chạm, không lách qua API riêng. */
    async function setBirth(y, m, d, h, mi) {
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
        await page.waitForTimeout(450);
    }

    // ── Can năm: đối chiếu với SỰ THẬT ĐÃ BIẾT, không phụ thuộc phép tính
    // của chính ứng dụng — 2024 Giáp Thìn (DƯƠNG), 2025 Ất Tỵ (ÂM), 2026
    // Bính Ngọ (DƯƠNG). Giữa năm (tháng 6) để khỏi vướng ranh giới Lập Xuân
    // đầu năm dương lịch.
    for (const [y, idx, ten] of [[2024, 0, 'Giáp (dương)'], [2025, 1, 'Ất (âm)'], [2026, 2, 'Bính (dương)']]) {
        await setBirth(y, 6, 15, 12, 0);
        const got = await page.evaluate(() => window.__yearGanIdx);
        check(`can năm ${y} đúng là ${ten}`, got, idx);
    }

    // ── Chéo Giới tính × can năm ra đúng thuận/nghịch, đúng công thức người
    // dùng cho: Nam+Dương hoặc Nữ+Âm → THUẬN; còn lại → NGHỊCH.
    const CASES = [
        { y: 2024, gender: 'nam', wantThuan: true,  ten: '2024 Giáp(dương) + Nam → thuận' },
        { y: 2024, gender: 'nu',  wantThuan: false, ten: '2024 Giáp(dương) + Nữ → nghịch' },
        { y: 2025, gender: 'nam', wantThuan: false, ten: '2025 Ất(âm) + Nam → nghịch' },
        { y: 2025, gender: 'nu',  wantThuan: true,  ten: '2025 Ất(âm) + Nữ → thuận' },
        { y: 2026, gender: 'nam', wantThuan: true,  ten: '2026 Bính(dương) + Nam → thuận' },
        { y: 2026, gender: 'nu',  wantThuan: false, ten: '2026 Bính(dương) + Nữ → nghịch' },
    ];
    for (const c of CASES) {
        await setBirth(c.y, 6, 15, 12, 0);
        await page.evaluate(g => window.__lenhGender(g), c.gender);
        await page.waitForTimeout(350);
        const dv = await page.evaluate(() => window.__lenhDaiVan());
        ok(c.ten, dv && dv.thuan === c.wantThuan, JSON.stringify(dv));
    }

    // ── Đối chiếu mốc mở/đóng tháng bằng MỘT NGUỒN KHÁC ngay trong chính
    // lunar.js: getPrevJie()/getNextJie() — chỉ 12 "tiết", khác API với
    // getPrevJieQi()/getNextJieQi() mà app.js dùng cho ô Tiết Khí (lấy CẢ
    // "khí"). Đây là phép dò ĐỘC LẬP với monthBounds() của lenh.js, không
    // gọi lại đúng những dòng code đang được kiểm.
    //
    // PHẢI đặt ShouXingUtil.setTzOffsetHours(8) ngay trước khi gọi — đúng cái
    // bẫy mà lenh.js tự ghi chú ở đầu termJd(): ShouXingUtil giữ múi giờ
    // trong một biến TOÀN CỤC, và getPrevJie/getNextJie (không như termJd)
    // không tự truyền "8", nên lấy nguyên mốc múi giờ của LẦN GỌI CUỐI CÙNG
    // — ở đây là múi giờ hiển thị hiện tại (0/UTC trong môi trường kiểm thử
    // này), lệch đúng 8 giờ so với mốc UTC+8 mà lenh.js dùng. Đo tay xác nhận
    // rồi mới sửa: thiếu bước này thì "đối chiếu độc lập" tự nó sai, chứ
    // không phải monthBounds() sai (đã thử: KHÔNG setTzOffsetHours(8) thì
    // lệch đúng 8,000 giờ ở cả hai mốc, đúng bằng offset UTC+8 — dấu hiệu
    // kinh điển của lỗi múi giờ, không phải sai số thiên văn).
    await setBirth(2026, 9, 17, 21, 0);
    const doc = await page.evaluate(() => {
        const inp = {
            y: parseInt(document.getElementById('inYear').value, 10),
            m: parseInt(document.getElementById('inMonth').value, 10),
            d: parseInt(document.getElementById('inDay').value, 10),
            h: parseInt(document.getElementById('solarHour').value, 10),
            mi: parseInt(document.getElementById('solarMinute').value, 10),
        };
        const info = countryData[document.getElementById('country').value];
        const tz = getTimezoneOffset(info.tzId, new Date(inp.y, inp.m - 1, inp.d, inp.h || 12));
        const bj = window._readInputBJ(inp.y, inp.m, inp.d, inp.h, inp.mi, tz);
        const dv = window.__lenhDaiVan();
        ShouXingUtil.setTzOffsetHours(8);
        const lunar = bj.solarBJ.getLunar();
        const prevJie = lunar.getPrevJie(false).getSolar().getJulianDay();
        const nextJie = lunar.getNextJie(false).getSolar().getJulianDay();
        ShouXingUtil.setTzOffsetHours(tz);  // trả lại đúng múi giờ hiển thị,
                                             // cho các bước sau của bài kiểm.
        return { prevJie, nextJie, dv };
    });
    // Dung sai 3 giây, không phải 0: getPrevJie/getNextJie ở đây là MỘT
    // ĐƯỜNG TÍNH KHÁC (không chắc cùng đi qua bảng DE423 như qiAccurate của
    // lenh.js) — README đã đo và chép lại đúng mức lệch giữa hai đường tính
    // độc lập cho cùng một tiết khí là "≤ 2 giây tại các mốc dùng chung".
    // Đo thực tế ở đây ra 0,37 giây, khớp đúng cỡ đó.
    const DUNG_SAI_NGAY = 3 / 86400;
    ok('mốc mở tháng HIỆN TẠI khớp getPrevJie() — nguồn khác trong lunar.js',
        doc.dv && Math.abs(doc.dv.mb.start - doc.prevJie) < DUNG_SAI_NGAY,
        `monthBounds=${doc.dv && doc.dv.mb.start} vs getPrevJie=${doc.prevJie} ` +
        `(lệch ${doc.dv && ((doc.dv.mb.start - doc.prevJie) * 86400).toFixed(2)}s)`);
    ok('mốc mở tháng KẾ TIẾP khớp getNextJie() — nguồn khác trong lunar.js',
        doc.dv && Math.abs(doc.dv.mb.end - doc.nextJie) < DUNG_SAI_NGAY,
        `monthBounds=${doc.dv && doc.dv.mb.end} vs getNextJie=${doc.nextJie} ` +
        `(lệch ${doc.dv && ((doc.dv.mb.end - doc.nextJie) * 86400).toFixed(2)}s)`);

    // ── Nội bộ: thuận + nghịch từ CÙNG một khoảnh khắc phải cộng lại đúng
    // bằng bề rộng cả tháng (mb.end − mb.start) — và giống nhau ở cả HAI
    // sách, vì n0 (mốc mở/đóng tháng) không phụ thuộc bộ số.
    let mbTheoSách = null;
    for (const rk of ['yhzp', 'smth']) {
        await page.evaluate(k => window.__lenhRule(k), rk);
        await page.evaluate(() => window.__lenhGender('nam'));
        await page.waitForTimeout(300);
        const a = await page.evaluate(() => window.__lenhDaiVan());
        await page.evaluate(() => window.__lenhGender('nu'));
        await page.waitForTimeout(300);
        const b = await page.evaluate(() => window.__lenhDaiVan());
        const tong = a.diffDays + b.diffDays, rong = a.mb.end - a.mb.start;
        ok(`${rk}: thuận + nghịch cộng đúng bề rộng tháng (bộ số không đổi việc này)`,
            Math.abs(tong - rong) < 1e-6, `${tong} vs ${rong}`);
        if (mbTheoSách === null) mbTheoSách = a.mb;
        else ok(`${rk}: mốc mở/đóng tháng giống hệt sách trước (cùng một ngày sinh)`,
            a.mb.start === mbTheoSách.start && a.mb.end === mbTheoSách.end,
            `${JSON.stringify(a.mb)} vs ${JSON.stringify(mbTheoSách)}`);
    }
    await page.evaluate(() => window.__lenhRule('yhzp'));
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(300);

    // Tuổi không âm, và trong khoảng jié dài nhất có thể (README: tối đa
    // 31,44 ngày → dưới 31,44/3 ≈ 10,48 tuổi — KHÔNG còn trần 6 tuổi của bản
    // cũ, vì đó là trần của "tiết khí gần nhất trong 24 mốc", còn nay là bề
    // rộng CẢ THÁNG, dài gấp đôi).
    for (const c of CASES) {
        await setBirth(c.y, 6, 15, 12, 0);
        await page.evaluate(g => window.__lenhGender(g), c.gender);
        await page.waitForTimeout(300);
        const dv = await page.evaluate(() => window.__lenhDaiVan());
        ok(`tuổi nhập vận không âm (${c.ten})`,
            dv.dv.years >= 0 && dv.dv.months >= 0 && dv.dv.days >= 0, JSON.stringify(dv.dv));
        ok(`tuổi nhập vận dưới 11 (${c.ten})`, dv.dv.years < 11, JSON.stringify(dv.dv));
    }

    // ── Hiện đúng trên màn: dòng "Nhập vận" khớp CHÍNH dv vừa đọc, kèm ngày
    // bắt đầu = sinh + tuổi (cộng lịch, không phải cộng ngày thô).
    await setBirth(2026, 9, 18, 8, 28);
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(400);
    const chk = await page.evaluate(() => {
        const dv = window.__lenhDaiVan().dv;
        const inp = {
            y: parseInt(document.getElementById('inYear').value, 10),
            m: parseInt(document.getElementById('inMonth').value, 10),
            d: parseInt(document.getElementById('inDay').value, 10),
        };
        const dt = new Date(inp.y, inp.m - 1, inp.d);
        dt.setFullYear(dt.getFullYear() + dv.years);
        dt.setMonth(dt.getMonth() + dv.months);
        dt.setDate(dt.getDate() + dv.days);
        const pad2 = x => (x < 10 ? '0' : '') + x;
        const ngàyBắtĐầu = pad2(dt.getDate()) + '/' + pad2(dt.getMonth() + 1) + '/' + dt.getFullYear();
        return {
            hiện: document.getElementById('lenhDaiVanVal').textContent,
            dv, ngàyBắtĐầu,
            // Không còn <div id="lenhDaiVan"> riêng — "Nhập vận: " giờ là một
            // text node NẰM CHUNG dòng với "Lệnh: " trong #lenhNow, ngay
            // trước <b id="lenhDaiVanVal">; tách dấu phân cách " · " ra để
            // còn lại đúng nhãn.
            nhãn: document.getElementById('lenhDaiVanVal').previousSibling.textContent
                .replace(/^\s*·\s*/, '').trim(),
        };
    });
    const wantAge = [
        chk.dv.years  > 0 ? `${chk.dv.years} tuổi`   : '',
        chk.dv.months > 0 ? `${chk.dv.months} tháng` : '',
        chk.dv.days   > 0 ? `${chk.dv.days} ngày`    : '',
    ].filter(Boolean).join(' ') || '0 ngày';
    check('số hiện trên màn khớp đúng dv nội bộ vừa đọc', chk.hiện, wantAge + ' · ' + chk.ngàyBắtĐầu);
    check('nhãn tiếng Việt là "Nhập vận:"', chk.nhãn, 'Nhập vận:');
    ok('có kèm ngày bắt đầu đại vận dd/mm/yyyy', /\d{2}\/\d{2}\/\d{4}$/.test(chk.hiện), chk.hiện);

    // ── Đổi ngôn ngữ: nhãn/đơn vị sang tiếng Trung, ngày bắt đầu KHÔNG đổi.
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(700);
    const zh = await page.evaluate(() => ({
        nhãn: document.getElementById('lenhDaiVanVal').previousSibling.textContent
            .replace(/^\s*·\s*/, '').trim(),
        hiện: document.getElementById('lenhDaiVanVal').textContent,
    }));
    check('nhãn tiếng Trung là "起运："', zh.nhãn, '起运：');
    ok('đơn vị tiếng Trung dùng 岁/个月/天, không lẫn chữ Việt',
        /^[0-9岁个月天]+ · \d{2}\/\d{2}\/\d{4}$/.test(zh.hiện), zh.hiện);
    ok('ngày bắt đầu KHÔNG đổi theo ngôn ngữ', zh.hiện.endsWith(chk.hiện.split('· ')[1]),
        `vi "${chk.hiện}" vs zh "${zh.hiện}"`);
    await page.evaluate(() => window.setLang('vi'));
    await page.waitForTimeout(500);

    // ── Đổi ngày sinh: số phải đổi theo (không phải một chuỗi tĩnh).
    const truoc = await page.textContent('#lenhDaiVanVal');
    const dNay = parseInt(await page.$eval('#inDay', e => e.value), 10);
    await setBirth(2026, 9, (dNay % 27) + 1, 8, 28);
    const sau = await page.textContent('#lenhDaiVanVal');
    ok('đổi ngày sinh thì tuổi nhập vận đổi theo', sau !== truoc, `"${truoc}" → "${sau}"`);

    // ── Đổi Giới tính: số phải đổi theo (chiều đổi, không phải chỉ nhãn).
    const truocGT = await page.textContent('#lenhDaiVanVal');
    await page.evaluate(() => window.__lenhGender('nu'));
    await page.waitForTimeout(400);
    const sauGT = await page.textContent('#lenhDaiVanVal');
    ok('đổi Giới tính thì tuổi nhập vận đổi theo (chiều đổi)', sauGT !== truocGT, `"${truocGT}" → "${sauGT}"`);
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(300);

    // ── Hình học: "Lệnh" + "Nhập vận" chung một dòng, không tràn, không cắt chữ.
    for (const d of [{ n: 'S21', w: 360 }, { n: 'S21 FE', w: 393 }, { n: 'A51', w: 412 }]) {
        const c2 = await browser.newContext({ viewport: { width: d.w, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page: p2 } = await open(c2);
        await p2.click('#tabLenh');
        await p2.waitForTimeout(700);
        const g = await p2.evaluate(() => {
            const e = document.getElementById('lenhNow');
            return e ? { cắt: e.scrollWidth > e.clientWidth + 1, cao: e.getBoundingClientRect().height } : null;
        });
        ok(`${d.n}: dòng "Lệnh · Nhập vận" không cắt chữ`, g && !g.cắt, JSON.stringify(g));
        await c2.close();
    }

    ok('không lỗi JS suốt lượt kiểm tuổi nhập vận', errs.length === 0, errs.join(' ; '));
    await ctx.close();
}

console.log('\nBốn mục của tab Tra cứu: hiện đủ tiêu đề, ĐÓNG SẴN, bấm là gập/mở');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);

    // Chưa chọn năm: không bảng nào hiện, chỉ một dòng gợi ý — ba cái tiêu đề
    // rỗng trông như bảng hỏng.
    await page.evaluate(() => window.showTab('tracuu'));
    await page.waitForTimeout(700);
    const chưaChọn = await page.evaluate(() => ({
        gợiÝ: getComputedStyle(document.getElementById('tcHint')).display !== 'none',
        bảngHiện: ['trinhuanPanel', 'sachboPanel', 'lenhHead', 'tcAmHead']
            .filter(i => getComputedStyle(document.getElementById(i)).display !== 'none'),
        nhãnNăm: document.getElementById('tcYearText').textContent.trim(),
    }));
    ok('chưa chọn năm: hiện dòng gợi ý', chưaChọn.gợiÝ);
    check('chưa chọn năm: không bảng nào hiện', chưaChọn.bảngHiện.join(','), '');

    // Chọn năm: hiện ĐỦ BỐN tiêu đề, nhưng THÂN ĐÓNG SẴN — bung cả bốn ra là
    // một trang gần 2000px, người dùng phải cuộn qua ba bảng mới tới bảng thứ
    // tư. Người dùng chốt: "hiển thị cả 3 tab… click vào thì expand hoặc hide".
    await mởTraCuu(page, 2026, false);
    const vừaChọn = await page.evaluate(() => ({
        gợiÝ: getComputedStyle(document.getElementById('tcHint')).display !== 'none',
        đầuHiện: ['trinhuanHeader', 'sachboHeader', 'lenhHead', 'tcAmHead']
            .filter(i => document.getElementById(i).getBoundingClientRect().height > 0).length,
        trn: getComputedStyle(document.getElementById('trinhuanBody')).display,
        sb: getComputedStyle(document.getElementById('sachboBody')).display,
        lenh: getComputedStyle(document.getElementById('lenhSec')).display,
        am: getComputedStyle(document.getElementById('tcAmSec')).display,
        // Đóng hết thì cả tab lọt gọn một màn hình: không có gì để cuộn.
        cuộnĐược: document.documentElement.scrollHeight - window.innerHeight,
    }));
    ok('chọn năm rồi: dòng gợi ý biến mất', !vừaChọn.gợiÝ);
    check('hiện đủ BỐN hàng tiêu đề', vừaChọn.đầuHiện, 4);
    check('bảng Trí Nhuận đóng sẵn', vừaChọn.trn, 'none');
    check('bảng Sách Bổ đóng sẵn', vừaChọn.sb, 'none');
    check('bảng Lệnh năm đóng sẵn', vừaChọn.lenh, 'none');
    check('mục Lịch âm đóng sẵn', vừaChọn.am, 'none');
    ok('đóng hết thì tab gọn trong một màn hình', vừaChọn.cuộnĐược <= 2,
        `còn cuộn được ${vừaChọn.cuộnĐược}px`);

    // Bung cả bốn: nội dung đủ, và vượt màn hình thì CẢ TRANG cuộn được —
    // không mục nào có khung cuộn con nuốt cú vuốt.
    await mởTraCuu(page, null);
    const mởSẵn = await page.evaluate(() => ({
        lớpMở: document.getElementById('lenhHead').classList.contains('lenh-open'),
        chev: document.getElementById('lenhHeadChevron').style.transform,
        lớpMởAm: document.getElementById('tcAmHead').classList.contains('lenh-open'),
        hàngTrn: document.querySelectorAll('#trn-tbody tr').length,
        hàngSb: document.querySelectorAll('#sb-tbody tr').length,
        hàngLenh: document.querySelectorAll('#lenhBody tbody tr').length,
        hàngAm: document.querySelectorAll('#tcAmBody tbody tr').length,
        tiêuĐề: document.getElementById('lenhTitle').textContent.trim(),
        cuộnĐược: document.documentElement.scrollHeight - window.innerHeight,
        // Không mục nào được giữ phần cuộn cho riêng mình.
        cuộnCon: ['trinhuanBody', 'sachboBody', 'lenhBody', 'tcAmBody']
            .filter(i => {
                const e = document.getElementById(i);
                return e && e.scrollHeight - e.clientHeight > 1;
            }),
    }));
    ok('đầu bảng Lệnh năm có lớp .lenh-open', mởSẵn.lớpMở);
    ok('mũi tên đã xoay', mởSẵn.chev === 'rotate(180deg)', mởSẵn.chev);
    ok('đầu mục Lịch âm cũng có lớp .lenh-open', mởSẵn.lớpMởAm);
    check('Trí Nhuận đủ 25 hàng', mởSẵn.hàngTrn, 25);
    check('Sách Bổ đủ 24 hàng', mởSẵn.hàngSb, 24);
    check('Lệnh năm đủ 33 đoạn', mởSẵn.hàngLenh, 33);
    ok('Lịch âm có đủ 12–13 tháng', mởSẵn.hàngAm >= 12 && mởSẵn.hàngAm <= 13,
        `${mởSẵn.hàngAm} hàng`);
    check('tiêu đề mang đúng năm đã chọn', mởSẵn.tiêuĐề, 'LỆNH NĂM 2026');
    ok('mở hết thì CẢ TRANG cuộn được', mởSẵn.cuộnĐược > 100,
        `chỉ cuộn được ${mởSẵn.cuộnĐược}px`);
    check('không mục nào có khung cuộn riêng', mởSẵn.cuộnCon.join(','), '');

    // Tra một năm bất kỳ thì KHÔNG có thời điểm sinh nào để mà "đang cầm
    // lệnh" — tô một hàng ở đây là nói dối rằng nó liên quan tới lá số đang mở.
    check('không hàng nào bị tô "đang hiệu lực"',
        await page.evaluate(() =>
            document.querySelectorAll('#traCuuView .dp-row-active, #traCuuView .lenh-on').length), 0);

    // Gập/mở từng mục, độc lập với nhau.
    for (const [head, body, tên] of [['#trinhuanHeader', 'trinhuanBody', 'Trí Nhuận'],
                                     ['#sachboHeader', 'sachboBody', 'Sách Bổ'],
                                     ['#lenhHead', 'lenhSec', 'Lệnh năm'],
                                     ['#tcAmHead', 'tcAmSec', 'Lịch âm']]) {
        await page.click(head);
        await page.waitForTimeout(450);
        check(`bấm một lần: ${tên} đóng lại`,
            await page.evaluate(b => getComputedStyle(document.getElementById(b)).display, body), 'none');
        await page.click(head);
        await page.waitForTimeout(450);
        check(`bấm lần nữa: ${tên} mở ra`,
            await page.evaluate(b => getComputedStyle(document.getElementById(b)).display, body), 'block');
    }

    // Đóng bảng rồi ĐỔI NĂM rồi mở lại: nội dung phải là của năm MỚI, không
    // phải một bản dựng từ lúc trước khi đổi.
    await page.click('#lenhHead');
    await page.waitForTimeout(400);
    await page.evaluate(() => window.__tracuuYear(1984));
    await page.waitForTimeout(800);
    const sauĐổiNăm = await page.evaluate(() => ({
        tiêuĐề: document.getElementById('lenhTitle').textContent.trim(),
        hàng: document.querySelectorAll('#lenhBody tbody tr').length,
        // Đổi năm KHÔNG được tự bung lại: gập/mở là lựa chọn của người dùng,
        // và tự mở ra thì cả bốn mục cùng bung, đẩy trang dài gấp ba.
        lenh: getComputedStyle(document.getElementById('lenhSec')).display,
    }));
    check('đổi năm lúc bảng đang đóng: tiêu đề theo năm mới', sauĐổiNăm.tiêuĐề, 'LỆNH NĂM 1984');
    check('…và bảng dựng lại đủ hàng', sauĐổiNăm.hàng, 33);
    check('…nhưng vẫn ĐÓNG: đổi năm không tự bung mục người dùng đã gập',
        sauĐổiNăm.lenh, 'none');
    // …và bung ra thì đúng là nội dung năm mới, không phải bản dựng cũ.
    await page.click('#lenhHead');
    await page.waitForTimeout(450);
    check('mở lại: đúng nội dung năm mới',
        await page.evaluate(() =>
            document.querySelector('#lenhBody tbody tr td').textContent.trim().slice(0, 40)
            && getComputedStyle(document.getElementById('lenhSec')).display), 'block');

    // Dòng tóm tắt "Lệnh: X" vẫn thuộc tab BÁT TỰ — nó nói về lá số, không
    // phải về năm đang tra.
    await page.click('#tabLenh');
    await page.waitForTimeout(700);
    const tómTắt = await page.evaluate(() =>
        document.getElementById('lenhNowVal').textContent.trim());
    ok('tab Bát Tự vẫn có dòng tóm tắt "Lệnh:"', tómTắt && tómTắt !== '—', tómTắt);
    ok('tab Bát Tự KHÔNG còn bảng Lệnh năm',
        await page.evaluate(() => !document.querySelector('#lenhView #lenhHead')));

    ok('không lỗi JS', errs.length === 0, errs.join(' ; '));
    await ctx.close();
}

console.log('\nĐại Vận (can chi Đại Vận đầu tiên) — bước một nấc từ trụ tháng');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    // Đúng ví dụ người dùng cho: tháng Đinh Dậu → Nam thuận → Mậu Tuất;
    // Nữ nghịch → Bính Thân. Đặt ngày để chắc chắn trụ tháng là Đinh Dậu.
    await page.evaluate(() => {
        const set = (id, v) => { const el = document.getElementById(id); el.value = String(v);
            el.dispatchEvent(new Event('change', { bubbles: true })); };
        set('inYear', 2026); set('inMonth', 9); set('inDay', 18);
        set('solarHour', 9); set('solarMinute', 0);
        window.processAll();
    });
    await page.waitForTimeout(500);
    const thángHiện = await page.evaluate(() =>
        document.getElementById('ttCanThang').textContent + document.getElementById('ttChiThang').textContent);
    check('trụ tháng đúng là Đinh Dậu (tiền đề của ví dụ)', thángHiện, 'ĐinhDậu');

    // Không còn dòng "Đại Vận: …" riêng trong ô Lệnh (bỏ theo yêu cầu — bảng
    // ĐẠI VẬN đầy đủ ngay dưới đã nói việc này rồi, nhắc lại là thừa) — đọc
    // thẳng dữ liệu qua window.__lenhDaiVan().pillar, không scrape một dòng
    // DOM không còn tồn tại.
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(400);
    let p = await page.evaluate(() => window.__lenhDaiVan().pillar);
    check('Nam (thuận): Đại Vận đầu = Mậu Tuất', CAN[p.can] + ' ' + CHI[p.chi], 'Mậu Tuất');

    await page.evaluate(() => window.__lenhGender('nu'));
    await page.waitForTimeout(400);
    p = await page.evaluate(() => window.__lenhDaiVan().pillar);
    check('Nữ (nghịch): Đại Vận đầu = Bính Thân', CAN[p.can] + ' ' + CHI[p.chi], 'Bính Thân');

    // Ghép LIỀN không khoảng trắng trong tiếng Trung (quy ước can chi tiếng
    // Trung trong cả ứng dụng, xem "Tuần thủ" ở tab Kỳ Môn) — kiểm trên thẻ
    // đầu tiên của bảng ĐẠI VẬN (.dv-line2, cùng đúng phép ghép chuỗi với
    // dòng đã bỏ), vì đó là chỗ DUY NHẤT còn hiện can chi đại vận đầu tiên.
    await page.evaluate(() => window.setLang('zh'));
    await page.waitForTimeout(600);
    const zhGiá = await page.evaluate(() =>
        document.querySelector('#daiVanBody .dv-card .dv-line2').textContent.trim());
    ok('can chi tiếng Trung KHÔNG có khoảng trắng ở giữa', !zhGiá.includes(' '), zhGiá);
    ok('…và đúng hai chữ Hán', /^[甲乙丙丁戊己庚辛壬癸][子丑寅卯辰巳午未申酉戌亥]$/.test(zhGiá), zhGiá);
    await page.evaluate(() => window.setLang('vi'));
    await page.waitForTimeout(500);

    ok('không lỗi JS', errs.length === 0, errs.join(' ; '));
    await ctx.close();
}

console.log('\nMột cỡ chữ duy nhất trong khối "Lệnh · Nhập vận", không đậm');
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    const kiểu = await page.evaluate(() => {
        const ids = ['lenhNowVal', 'lenhDaiVanVal'];
        return ids.map(id => {
            const e = document.getElementById(id);
            const cs = getComputedStyle(e);
            return { id, cỡ: cs.fontSize, đậm: cs.fontWeight };
        });
    });
    const cỡs = new Set(kiểu.map(k => k.cỡ));
    ok('cả hai đoạn cùng MỘT cỡ chữ', cỡs.size === 1, JSON.stringify(kiểu));
    for (const k of kiểu) {
        ok(`${k.id}: không đậm (weight ${k.đậm}, cần < 700)`, parseInt(k.đậm, 10) < 700, k.đậm);
    }
    ok('không lỗi JS', errs.length === 0, errs.join(' ; '));
    await ctx.close();
}

console.log('\nBảng ĐẠI VẬN: 10 đại vận × 10 năm, lấp chỗ trống dưới Lệnh năm');
{
    // ── Lưu niên: đối chiếu SỰ THẬT ĐÃ BIẾT (độc lập với ứng dụng) ──
    // 1952 Nhâm Thìn, 2022 Nhâm Dần, 2024 Giáp Thìn — cả ba đọc thẳng từ
    // ảnh mẫu người dùng gửi / lịch vạn niên phổ biến.
    function luuNienCanChi(year) {
        return { can: ((year - 4) % 10 + 10) % 10, chi: ((year - 4) % 12 + 12) % 12 };
    }
    for (const [y, canTen, chiTen] of [[1952, 8, 4], [2022, 8, 2], [2024, 0, 4], [2026, 2, 6]]) {
        const cc = luuNienCanChi(y);
        check(`lưu niên ${y}: đúng can`, cc.can, canTen);
        check(`lưu niên ${y}: đúng chi`, cc.chi, chiTen);
    }

    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await open(ctx);
    await page.click('#tabLenh');
    await page.waitForTimeout(700);

    // Đúng ví dụ người dùng cho: Nam sinh Bính Ngọ, tháng Đinh Dậu.
    await page.evaluate(() => {
        const set = (id, v) => { const el = document.getElementById(id); el.value = String(v);
            el.dispatchEvent(new Event('change', { bubbles: true })); };
        set('inYear', 2026); set('inMonth', 9); set('inDay', 18);
        set('solarHour', 9); set('solarMinute', 0);
        window.processAll();
    });
    await page.waitForTimeout(500);
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(500);

    const bang = await page.evaluate(() => window.__lenhDaiVanBang());
    ok('bảng có đủ 10 đại vận', Array.isArray(bang) && bang.length === 10, JSON.stringify(bang && bang.length));

    // Tên: Mậu Tuất, Kỷ Hợi, Canh Tý, Tân Sửu… — đúng nguyên văn ví dụ.
    const CAN = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    const CHI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
    const tên = bang.map(b => CAN[b.can] + ' ' + CHI[b.chi]);
    check('4 đại vận đầu đúng ví dụ', tên.slice(0, 4).join(', '),
        'Mậu Tuất, Kỷ Hợi, Canh Tý, Tân Sửu');
    // Mỗi đại vận bước ĐÚNG MỘT nấc so với đại vận trước (thuận: Nam+Bính(dương)).
    for (let i = 1; i < 10; i++) {
        check(`đại vận #${i}: can bước đúng +1 từ #${i - 1}`, bang[i].can, (bang[i - 1].can + 1) % 10);
        check(`đại vận #${i}: chi bước đúng +1 từ #${i - 1}`, bang[i].chi, (bang[i - 1].chi + 1) % 12);
    }

    // Mỗi đại vận bắt đầu ĐÚNG 10 năm dương lịch sau đại vận trước, cùng
    // tháng — và mỗi đại vận phủ ĐÚNG 10 năm liên tục, không hở không chồng.
    for (let i = 0; i < 10; i++) {
        check(`đại vận #${i}: phủ đúng 10 năm liên tục`, bang[i].years.length, 10);
        for (let j = 1; j < 10; j++) {
            check(`đại vận #${i}: năm #${j} = năm #${j - 1} + 1`,
                bang[i].years[j].y, bang[i].years[j - 1].y + 1);
        }
        if (i > 0) {
            check(`đại vận #${i}: bắt đầu đúng 10 năm sau #${i - 1}`,
                bang[i].startY, bang[i - 1].startY + 10);
            check(`đại vận #${i}: cùng tháng với #${i - 1}`, bang[i].startM, bang[i - 1].startM);
        }
    }
    // Tuổi ghi ở đầu mỗi thẻ tăng đúng 10 mỗi đại vận.
    for (let i = 1; i < 10; i++) {
        check(`đại vận #${i}: tuổi = tuổi #${i - 1} + 10`, bang[i].tuoi, bang[i - 1].tuoi + 10);
    }

    // Lưu niên của TỪNG năm trong bảng khớp công thức độc lập ở trên — không
    // chỉ tin phép tính của chính ứng dụng.
    let lệchLưuNiên = 0;
    for (const đv of bang) for (const yr of đv.years) {
        const muốn = luuNienCanChi(yr.y);
        if (yr.cc.can !== muốn.can || yr.cc.chi !== muốn.chi) lệchLưuNiên++;
    }
    ok('cả 100 năm khớp công thức lưu niên độc lập', lệchLưuNiên === 0, `${lệchLưuNiên} năm sai`);

    // ── DOM: 10 thẻ, đúng cấu trúc, đúng định dạng đầu thẻ ──
    const dom = await page.evaluate(() => {
        const cards = [...document.querySelectorAll('#daiVanBody .dv-card')];
        return {
            soThẻ: cards.length,
            mỗiThẻ10Hàng: cards.every(c => c.querySelectorAll('.dv-row').length === 10),
            dòng1Thẻ0: cards[0].querySelector('.dv-line1').textContent.trim(),
            dòng2Thẻ0: cards[0].querySelector('.dv-line2').textContent.trim(),
            cóTiêuĐềRiêng: !!document.getElementById('daiVanHead'),
        };
    });
    check('DOM: đủ 10 thẻ', dom.soThẻ, 10);
    ok('DOM: mỗi thẻ đủ 10 hàng năm', dom.mỗiThẻ10Hàng);
    ok('đầu thẻ đúng định dạng "mm/yyyy - Nt"', /^\d{2}\/\d{4} - \d+t$/.test(dom.dòng1Thẻ0), dom.dòng1Thẻ0);
    check('dòng 2 đầu thẻ là can chi đại vận', dom.dòng2Thẻ0, tên[0]);
    // Hàng tiêu đề "ĐẠI VẬN" đã gỡ hẳn (xem index.html): nó chỉ nhắc lại điều
    // mà 10 thẻ can chi kèm khoảng tuổi đã nói rõ, trong khi ngốn 37px của tab
    // chật nhất và đẩy "LỆNH NĂM" xuống quá sâu.
    ok('không còn hàng tiêu đề "ĐẠI VẬN" riêng', !dom.cóTiêuĐềRiêng);

    // ── Cỡ chữ THẬT phải đủ đọc — chính lỗi user bắt được lúc đầu: clamp()
    // co theo bề rộng màn hình từng tụt xuống 7.5–8.6px ĐỀU KHẮP cả bảng
    // (nhỏ hơn mọi chữ khác trong app), mà lúc soi ảnh chụp phóng to 2× lại
    // tưởng ổn. Đầu thẻ (dòng mốc + dòng can chi) không bao giờ cần co động
    // (đo thực nghiệm: luôn đủ chỗ, xem tools/_grid.mjs) nên khoá cứng
    // ngưỡng tối thiểu ở đây — lỗi này không được lặp lại lặng lẽ. ──
    const cỡChữĐầuThẻ = await page.evaluate(() => {
        const px = sel => { const el = document.querySelector(sel); return el ? parseFloat(getComputedStyle(el).fontSize) : null; };
        return { line1: px('#daiVanBody .dv-line1'), line2: px('#daiVanBody .dv-line2') };
    });
    ok('cỡ chữ dòng mốc+tuổi đầu thẻ đủ đọc (≥8px)', cỡChữĐầuThẻ.line1 >= 8, cỡChữĐầuThẻ.line1);
    ok('cỡ chữ dòng can chi đại vận đầu thẻ đủ đọc (≥9px)', cỡChữĐầuThẻ.line2 >= 9, cỡChữĐầuThẻ.line2);

    // ── Hàng lưu niên (năm + can chi): ĐA SỐ phải ở cỡ thoải mái (~9–10.5px,
    // xem lenh.css) — chỉ THIỂU SỐ tổ hợp can chi DÀI NHẤT (can VÀ chi đều 4
    // chữ: Giáp/Bính/Đinh/Canh/Nhâm × Thìn/Thân/Tuất, ví dụ "Nhâm Thân") mới
    // cần shrinkDaiVanRows() co thêm — vẫn phải có SÀN, không được co xuống
    // dưới ngưỡng đọc được luôn. ──
    const cỡHàng = await page.evaluate(() => {
        const px = el => parseFloat(getComputedStyle(el).fontSize);
        const rows = [...document.querySelectorAll('#daiVanBody .dv-row')];
        const sizes = rows.map(px);
        return {
            tổng: sizes.length,
            nhỏNhất: Math.min(...sizes),
            đaSốĐủ9px: sizes.filter(s => s >= 9).length,
        };
    });
    ok('mọi hàng lưu niên vẫn có sàn đọc được (≥7.5px) dù có bị co để hết tràn',
        cỡHàng.nhỏNhất >= 7.5, cỡHàng.nhỏNhất);
    ok('đa số hàng lưu niên (≥70/100) giữ cỡ chữ thoải mái ≥9px, không phải co toàn bộ',
        cỡHàng.đaSốĐủ9px >= 70, `${cỡHàng.đaSốĐủ9px}/100`);

    // ── KHÔNG hàng lưu niên nào bị cắt chữ thật — kiểm tra chính .dv-row (là
    // flex container), KHÔNG chỉ .dv-year/.dv-cc con bên trong: con flex mặc
    // định min-width:auto nên không tự co dưới kích thước chữ của chính nó
    // (scrollWidth riêng luôn bằng clientWidth riêng, "sạch" giả tạo dù cha
    // đang tràn thật) — .dv-card có overflow:hidden nên cha tràn là cắt cụt
    // chữ thật, phải bắt đúng ở cha. ──
    const cắtHàng = await page.evaluate(() => {
        const bad = [];
        document.querySelectorAll('#daiVanBody .dv-row').forEach(r => {
            if (r.scrollWidth > r.clientWidth + 0.5) bad.push(r.textContent.trim());
        });
        return bad;
    });
    ok('không hàng lưu niên nào bị cắt chữ (đo trên .dv-row, không chỉ span con)',
        cắtHàng.length === 0, cắtHàng.join(' | '));

    // ── Đổi Giới tính: chiều bước đổi (Nữ → nghịch), tên 10 đại vận đổi theo ──
    await page.evaluate(() => window.__lenhGender('nu'));
    await page.waitForTimeout(500);
    const bangNu = await page.evaluate(() => window.__lenhDaiVanBang());
    check('Nữ (nghịch): đại vận đầu = Bính Thân (đối xứng ví dụ Nam)',
        CAN[bangNu[0].can] + ' ' + CHI[bangNu[0].chi], 'Bính Thân');
    check('Nữ: đại vận #2 bước LÙI một nấc', bangNu[1].can, (bangNu[0].can - 1 + 10) % 10);
    await page.evaluate(() => window.__lenhGender('nam'));
    await page.waitForTimeout(500);

    // ── Highlight "năm nay" — dùng ĐÚNG "năm nay" thật (đồng hồ máy chạy
    // kiểm thử này), không mock giả — hỏi độc lập trong Node rồi so. ──
    const nayThật = new Date().getFullYear();
    const nằmTrongBảng = bang.some(đv => nayThật >= đv.years[0].y && nayThật <= đv.years[9].y);
    const hl = await page.evaluate(() => {
        const card = document.getElementById('daiVanCurrent');
        const yr = document.getElementById('daiVanYearOn');
        return {
            cóThẻ: !!card, cóNăm: !!yr,
            nộiDungNăm: yr ? yr.querySelector('.dv-year').textContent.trim() : null,
            scrollTop: document.getElementById('daiVanBody').scrollTop,
        };
    });
    if (nằmTrongBảng) {
        ok('năm nay NẰM TRONG 100 năm của bảng này → phải highlight', hl.cóThẻ && hl.cóNăm, JSON.stringify(hl));
        check('đúng năm được highlight', hl.nộiDungNăm, String(nayThật));
        ok('đã cuộn tới thẻ đang sống (không đứng yên ở đỉnh)', hl.scrollTop > 0, hl.scrollTop);
    } else {
        // Sinh 2026 — 10 đại vận đầu chỉ phủ 2033–2132, không chứa năm nay.
        // Trường hợp NÀY cũng phải đúng: KHÔNG highlight gì, không phải lỗi.
        ok('năm nay KHÔNG nằm trong 100 năm của bảng → KHÔNG highlight gì',
            !hl.cóThẻ && !hl.cóNăm, JSON.stringify(hl));
    }

    // ── Người 50 tuổi: BẮT BUỘC "năm nay" rơi vào bảng (nhập vận luôn trước
    // tuổi 15, nên bảng 100 năm từ đó chắc chắn phủ qua "năm nay") — lá số ở
    // trên (sinh 2026) không bao giờ tự đi vào nhánh CÓ highlight, nên phải
    // dựng riêng một lá số khác mới thật sự kiểm được scrollToCurrentDaiVan()
    // đưa đúng thẻ vào khung nhìn (không chỉ "có cuộn", đo VỊ TRÍ THẬT). ──
    {
        const tuổi50 = new Date().getFullYear() - 50;
        await page.evaluate((y) => {
            const set = (id, v) => { const el = document.getElementById(id); el.value = String(v);
                el.dispatchEvent(new Event('change', { bubbles: true })); };
            set('inYear', y); set('inMonth', 9); set('inDay', 18);
            set('solarHour', 9); set('solarMinute', 0);
            window.processAll();
        }, tuổi50);
        await page.waitForTimeout(500);
        await page.evaluate(() => window.__lenhGender('nam'));
        await page.waitForTimeout(500);

        const hl2 = await page.evaluate(() => {
            const within = (el, anc) => {
                const r = el.getBoundingClientRect(), a = anc.getBoundingClientRect();
                return r.top >= a.top - 0.5 && r.bottom <= a.bottom + 0.5 &&
                       r.left >= a.left - 0.5 && r.right <= a.right + 0.5;
            };
            const card = document.getElementById('daiVanCurrent');
            const yr = document.getElementById('daiVanYearOn');
            const body = document.getElementById('daiVanBody');
            const grid = card ? card.closest('.dv-grid') : null;
            return {
                cóThẻ: !!card, cóNăm: !!yr,
                dọcThấy: (card && body) ? within(card, body) : null,
                ngangThấy: (card && grid) ? within(card, grid) : null,
            };
        });
        ok('người 50 tuổi: năm nay chắc chắn rơi vào bảng 100 năm → có thẻ + có năm highlight',
            hl2.cóThẻ && hl2.cóNăm, JSON.stringify(hl2));
        ok('đã cuộn DỌC tới đúng chỗ: thẻ đang sống nằm TRỌN trong khung nhìn dọc',
            hl2.dọcThấy === true, JSON.stringify(hl2));
        ok('đã cuộn NGANG tới đúng chỗ (nếu cần): thẻ đang sống nằm TRỌN trong khung nhìn ngang',
            hl2.ngangThấy === true, JSON.stringify(hl2));

        // Khôi phục đúng lá số gốc (2026, ví dụ user cho) cho các phép đo sau.
        await page.evaluate(() => {
            const set = (id, v) => { const el = document.getElementById(id); el.value = String(v);
                el.dispatchEvent(new Event('change', { bubbles: true })); };
            set('inYear', 2026); set('inMonth', 9); set('inDay', 18);
            set('solarHour', 9); set('solarMinute', 0);
            window.processAll();
        });
        await page.waitForTimeout(500);
        await page.evaluate(() => window.__lenhGender('nam'));
        await page.waitForTimeout(500);
    }

    // ── Màu trong bảng Đại Vận: CHỈ can chi được tô ──
    // Người dùng chốt: "TẤT CẢ các chữ CAN CHI có màu tương ứng với ngũ hành…
    // TẤT CẢ các chữ KO PHẢI CAN CHI thì đều màu đen". Bảng Đại Vận vốn đơn
    // sắc, nên ở đây phép canh tách đôi: mỗi <span.nh> phải mang ĐÚNG màu của
    // hành ghi trong lớp của nó, còn mọi thứ khác vẫn phải R=G=B.
    const màu = await page.evaluate(() => {
        const HÀNH = {
            kim: 'rgb(110, 119, 129)', moc: 'rgb(30, 125, 52)',
            thuy: 'rgb(18, 64, 143)', hoa: 'rgb(198, 40, 40)', tho: 'rgb(138, 90, 43)',
        };
        const lát = s => (s.match(/\d+(\.\d+)?/g) || []).map(Number);
        const xám = v => v.length >= 3 && v[0] === v[1] && v[1] === v[2];
        const sai = [], saiHành = [];
        let sốNh = 0;
        for (const el of document.querySelectorAll('#daiVanBody, #daiVanBody *')) {
            const cs = getComputedStyle(el);
            // Nền và viền KHÔNG bao giờ được có màu, kể cả trên ô can chi.
            // Viền chỉ xét khi THỰC SỰ có viền: border-color mặc định là
            // `currentColor`, nên một <span> không viền vẫn khai đúng màu chữ
            // của nó — đọc con số ấy là bắt nhầm chính màu ngũ hành.
            if (!xám(lát(cs.backgroundColor))) sai.push(`${el.className}·nền:${cs.backgroundColor}`);
            if (parseFloat(cs.borderTopWidth) > 0 && !xám(lát(cs.borderTopColor))) {
                sai.push(`${el.className}·viền:${cs.borderTopColor}`);
            }
            const hành = (/\bnh-(\w+)\b/.exec(el.className || '') || [])[1];
            if (hành) {
                sốNh++;
                if (cs.color !== HÀNH[hành]) saiHành.push(`${hành}:${cs.color}`);
            } else if (!xám(lát(cs.color))) {
                sai.push(`${el.className}·color:${cs.color}`);
            }
        }
        return { sai, saiHành, sốNh };
    });
    ok('bảng Đại Vận có tô can chi theo ngũ hành', màu.sốNh > 0, `${màu.sốNh} chữ`);
    check('…mỗi chữ đúng màu của hành ấy', màu.saiHành.slice(0, 5).join(' | '), '');
    ok('…còn mọi thứ KHÔNG PHẢI can chi vẫn đen/trắng/ghi (R=G=B)',
        màu.sai.length === 0, màu.sai.slice(0, 5).join(' | '));

    // ── Đại Vận KHÔNG còn maxHeight, KHÔNG tự cuộn riêng — nó là nội dung
    // CHÍNH của tab (như bàn Kỳ Môn), luôn hiện TRỌN dù có tràn xuống dưới
    // thanh tab hay không (trang cuộn dọc lo phần đó, không phải hộp này). ──
    const khôngCuộnRiêng = await page.evaluate(() => {
        const box = document.getElementById('daiVanBody');
        return {
            maxHeight: getComputedStyle(box).maxHeight,
            khôngCầnCuộnNộiBộ: box.scrollHeight <= box.clientHeight + 1,
        };
    });
    ok('Đại Vận không còn bị ép maxHeight (luôn hiện trọn, không tuỳ chỗ còn lại)',
        khôngCuộnRiêng.maxHeight === 'none', khôngCuộnRiêng.maxHeight);
    ok('hộp Đại Vận không cần tự cuộn dọc nội bộ (khớp đúng chiều cao tự nhiên)',
        khôngCuộnRiêng.khôngCầnCuộnNộiBộ, JSON.stringify(khôngCuộnRiêng));

    // ── Bảng Lệnh năm đã RỜI HẲN tab này (sang tab Tra cứu) ──
    // Bản trước canh "mở Lệnh năm ra không được làm Đại Vận co lại" — hai khối
    // từng tranh chung một ngân sách chiều cao, mở cái này là cái kia teo đi.
    // Nay chúng còn không ở chung một tab, nên phép canh mạnh hơn hẳn: tab Bát
    // Tự KHÔNG chứa bảng Lệnh năm, và Đại Vận là nội dung duy nhất của nó.
    const mộtMình = await page.evaluate(() => {
        const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
        const dock = document.getElementById('bottomDock').getBoundingClientRect();
        const dv = document.getElementById('daiVanSec').getBoundingClientRect();
        const lv = document.getElementById('lenhView');
        return {
            lệnhNămTrongTabNày: !!lv.querySelector('#lenhHead, #lenhSec, #lenhBody'),
            dvCao: +(dv.height / z).toFixed(1),
            dockỞĐúngĐáyMànHình: Math.abs(dock.bottom - window.innerHeight) < 2,
            trangChoPhépCuộnKhiCần: document.documentElement.scrollHeight >= dock.bottom - 1,
        };
    });
    ok('tab Bát Tự KHÔNG còn chứa bảng Lệnh năm', !mộtMình.lệnhNămTrongTabNày);
    ok('Đại Vận vẫn hiện trọn chiều cao của nó', mộtMình.dvCao > 100, JSON.stringify(mộtMình));
    ok('#bottomDock vẫn đứng cố định đúng đáy màn hình dù trang có dài hơn một màn',
        mộtMình.dockỞĐúngĐáyMànHình, JSON.stringify(mộtMình));
    ok('trang cho phép cuộn xuống hết nội dung (không khoá overflow ở đâu đó)',
        mộtMình.trangChoPhépCuộnKhiCần, JSON.stringify(mộtMình));

    // ── Hình học trên máy hẹp nhất: không cắt chữ, không kéo ngang ──
    ok('không lỗi JS', errs.length === 0, errs.join(' ; '));
    await ctx.close();

    for (const d of [{ n: 'S21', w: 360 }, { n: 'S21 FE', w: 393 }, { n: 'A51', w: 412 }]) {
        const c2 = await browser.newContext({ viewport: { width: d.w, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page: p2, errs: errs2 } = await open(c2);
        await p2.click('#tabLenh');
        await p2.waitForTimeout(700);
        // Dùng lại đúng lá số 1975/Nam — 100 năm của lá số này CHỨA SẴN nhiều
        // tổ hợp can chi DÀI NHẤT (Canh Thân, Nhâm Tuất, Nhâm Thân, Nhâm
        // Thìn…, đo thật ở tools/_grid.mjs), nên vòng lặp này thật sự kiểm
        // tra đúng trường hợp khó, không phải trạng thái mặc định trống.
        await p2.evaluate(() => {
            const set = (id, v) => { const el = document.getElementById(id); el.value = String(v);
                el.dispatchEvent(new Event('change', { bubbles: true })); };
            set('inYear', 1975); set('inMonth', 6); set('inDay', 15);
            set('solarHour', 9); set('solarMinute', 0);
            window.processAll();
        });
        await p2.waitForTimeout(500);
        await p2.evaluate(() => window.__lenhGender('nam'));
        await p2.waitForTimeout(500);
        const g = await p2.evaluate(() => {
            const cắt = [];
            // Đầu thẻ vẫn canh giữa nên scrollWidth đủ dùng.
            document.querySelectorAll('#daiVanBody .dv-line1, #daiVanBody .dv-line2')
                .forEach(e => { if (e.scrollWidth > e.clientWidth + 1) cắt.push((e.textContent || '').trim().slice(0, 16)); });

            // .dv-row thì KHÔNG: từ khi hàng canh PHẢI (xem .dv-row trong
            // lenh.css) phần thừa dồn hết sang TRÁI, mà scrollWidth chỉ đếm
            // phần tràn về phía CUỐI dòng — nó báo 0 trong khi "2052 Nhâm
            // Thân" đang thò 13,9px sang trái và mấy chữ số đầu của NĂM bị
            // .dv-card{overflow:hidden} cắt mất. Đo hai phía, so với hộp NỘI
            // DUNG của hàng, đúng như rowOverflow() trong lenh.js.
            //
            // Đo ở hàng CHA (flex container), không phải ở .dv-year/.dv-cc:
            // con flex mặc định min-width:auto nên không tự co dưới kích
            // thước chữ của nó, chỉ đo con thì luôn "sạch" giả tạo.
            document.querySelectorAll('#daiVanBody .dv-row').forEach(e => {
                const kids = [...e.children].map(k => k.getBoundingClientRect());
                if (!kids.length) return;
                const cs = getComputedStyle(e), r = e.getBoundingClientRect();
                const thò = Math.max(
                    (r.left + parseFloat(cs.paddingLeft)) - Math.min(...kids.map(k => k.left)),
                    Math.max(...kids.map(k => k.right)) - (r.right - parseFloat(cs.paddingRight)));
                if (thò > 1) cắt.push((e.textContent || '').trim().slice(0, 16));
            });
            return { cắt, ngang: document.documentElement.scrollWidth > window.innerWidth + 1 };
        });
        ok(`${d.n}: bảng Đại Vận không cắt chữ`, g.cắt.length === 0, g.cắt.join(' | '));
        ok(`${d.n}: trang không tràn ngang`, !g.ngang);
        ok(`${d.n}: không lỗi JS`, errs2.length === 0, errs2.join(' ; '));
        await c2.close();
    }
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
if (fail) process.exit(1);
console.log('✓ Tab Lệnh đúng bảng, đúng giờ, và dùng chung địa điểm với hai tab kia');
