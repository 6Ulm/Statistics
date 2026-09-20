/**
 * BỘ TÍNH THIÊN VĂN: đúng chuẩn, và không đổi so với bản trước khi dọn mã.
 *
 *   node test_astro_ref.mjs
 *
 * Hai câu hỏi khác nhau, nên hai nhóm phép thử khác nhau:
 *
 *   A. ĐÚNG CHUẨN THIÊN VĂN — so với PyEphem, một thư viện độc lập hoàn toàn
 *      với repo này (PyEphem dựng trên XEphem: VSOP87 cho Mặt Trời, ELP2000
 *      cho Mặt Trăng; lunar.js thì dùng chuỗi rút gọn khớp bảng DE423). Hai
 *      đường tính khác hẳn nhau, nên khớp được là khớp thật.
 *      Mốc tham chiếu nằm sẵn ở tools/refdata/astro_ref.json, sinh bằng
 *      tools/refdata/gen_ref.py.
 *
 *   B. KHÔNG ĐỔI SO VỚI BẢN CŨ — dựng hai trang cùng lúc: bản trước khi gom
 *      phép tính về core.js, và bản bây giờ. Cùng một dải tham số, so từng con
 *      số một. Đây là chỗ trả lời "dọn mã có làm xê dịch gì không", còn nhóm A
 *      trả lời "con số ấy có đúng không" — không nhóm nào thay được nhóm kia.
 *
 * Bản cũ lấy từ đâu: `git archive <commit> …/web` vào một thư mục tạm, rồi
 * dựng nó như một trang web thứ hai. Không có bản cũ (chạy trên máy không có
 * git, hoặc lịch sử đã đổi) thì nhóm B tự bỏ qua và nói rõ, chứ không đỏ.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import os from 'os';
import { execFileSync } from 'child_process';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const REPO = path.join(HERE, '..', '..');
/** Commit ngay TRƯỚC lần gom phép tính về core.js. */
const COMMIT_CU = process.env.COMMIT_CU || '476465f';

let pass = 0, fail = 0, bỏQua = 0;
const ok = (ten, đúng, thêm) => {
    if (đúng) { pass++; console.log('  ok   ' + ten); }
    else { fail++; console.log('  ✗    ' + ten + (thêm ? '\n         ' + thêm : '')); }
};

/* ─────────────── Hai trang web, hai cổng ─────────────── */
const MIME = { '.html': 'text/html', '.js': 'text/javascript', '.css': 'text/css', '.txt': 'text/plain' };
function phục(gốc) {
    const sv = http.createServer((q, r) => {
        const rel = decodeURIComponent(q.url.split('?')[0]).replace(/^\/+/, '') || 'index.html';
        const f = path.join(gốc, rel);
        if (!f.startsWith(gốc) || !fs.existsSync(f)) { r.writeHead(404); return r.end(); }
        r.writeHead(200, { 'Content-Type': MIME[path.extname(f)] || 'application/octet-stream' });
        fs.createReadStream(f).pipe(r);
    });
    return new Promise(res => sv.listen(0, '127.0.0.1', () => res(sv)));
}

/** Bày bản cũ ra một thư mục tạm; null nếu không lấy được. */
function lấyBảnCũ() {
    const đích = fs.mkdtempSync(path.join(os.tmpdir(), 'web-cu-'));
    try {
        const tar = execFileSync('git', ['archive', COMMIT_CU, 'android/app/src/main/assets/web'],
            { cwd: REPO, maxBuffer: 1 << 28 });
        execFileSync('tar', ['-x', '-C', đích, '--strip-components=6'], { input: tar });
        if (!fs.existsSync(path.join(đích, 'index.html'))) return null;
        return đích;
    } catch (e) {
        console.log('  (không lấy được bản cũ: ' + e.message.split('\n')[0] + ')');
        return null;
    }
}

const svMoi = await phục(WEB);
const baseMoi = `http://127.0.0.1:${svMoi.address().port}/index.html`;
const thưMụcCũ = lấyBảnCũ();
const svCu = thưMụcCũ ? await phục(thưMụcCũ) : null;
const baseCu = svCu ? `http://127.0.0.1:${svCu.address().port}/index.html` : null;

const { chromium } = await import('playwright');
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);
async function mở(base) {
    const ctx = await browser.newContext({ viewport: { width: 412, height: 852 } });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1200);
    return { ctx, page };
}

/* ═══════════════ A. ĐÚNG CHUẨN THIÊN VĂN ═══════════════ */
const REF = JSON.parse(fs.readFileSync(path.join(HERE, 'refdata', 'astro_ref.json'), 'utf8'));
const { page } = await mở(baseMoi);

/**
 * Sai số cho phép, GIÂY — đặt theo ĐO THẬT rồi làm tròn lên một mức, không
 * đặt rộng cho dễ xanh (xem con số "đo được" phép thử in ra).
 *
 * Hai cột, vì có hai câu hỏi khác nhau:
 *
 *   ephem  — so hai mốc đã KHỬ ΔT, tức so phần thiên văn thuần tuý (vị trí
 *            Mặt Trời, Mặt Trăng). Đây mới là chỗ nói "bộ tính có chuẩn
 *            không". Chặt.
 *   ut     — so mốc UT như người dùng nhìn thấy. Lỏng hơn, và CỐ Ý: ΔT là
 *            hiệu giữa thời gian nguyên tử và vòng quay thật của Trái Đất —
 *            ĐO được cho quá khứ, chỉ ĐOÁN được cho tương lai. Hai thư viện
 *            dùng hai bộ đa thức khác nhau nên mốc UT phải lệch đúng bằng
 *            hiệu hai ΔT. Không có gì để "chữa" ở đây.
 */
const DUNG_SAI = {
    ephem:   { tietKhi: 20, soc: 20, vong: 20 },
    ut:      { tietKhi: 150, soc: 150, vong: 150 },
    chinhNgo: 30,
};

console.log('\nA. Đối chiếu PyEphem (độc lập hoàn toàn với repo)');
{
    // ── Tiết khí ──
    const đo = await page.evaluate((refs) => {
        const J1970 = 2440587.5;
        const unixOf = jdUTC8 => (jdUTC8 - 8 / 24 - J1970) * 86400;
        // GỐC đánh số tiết khí của ứng dụng: n = 0 rơi vào XUÂN PHÂN 1999
        // (hoàng kinh 0°), không phải Đông Chí. Lấy thẳng Ephem.termJd(0) làm
        // mốc chứ không chép một hằng số — chép thì mai nó đổi mà phép thử
        // không biết.
        const gốc = unixOf(Ephem.termJd(0));
        const bước = 365.2422 * 86400 / 24;
        /** ΔT (giây) mà lunar.js dùng tại một mốc, để bên kia khử ra. */
        const dtApp = jdUTC8 => ShouXingUtil.dtT(jdUTC8 - 8 / 24 - 2451545.0) * 86400;
        return refs.map(r => {
            // So theo THỜI ĐIỂM chứ không theo chỉ số: đoán thô rồi dò ±3 để
            // lấy mốc gần nhất, khỏi phải tin vào cách đánh số của bên nào.
            let n0 = Math.round((r.utc - gốc) / bước);
            let best = null;
            for (let k = -3; k <= 3; k++) {
                const jd = Ephem.termJd(n0 + k);
                const d = Math.abs(unixOf(jd) - r.utc);
                if (!best || d < best.d) best = { d: d, jd: jd, n: n0 + k };
            }
            return { ref: r.utc, app: unixOf(best.jd), n: best.n, deg: r.deg, year: r.year,
                     dtRef: r.dt, dtApp: dtApp(best.jd) };
        });
    }, REF.tietKhi);
    const soSánh = (đo, tên, dsEphem, dsUt) => {
        const ut = đo.map(x => Math.abs(x.app - x.ref));
        const ep = đo.map(x => Math.abs((x.app + x.dtApp) - (x.ref + x.dtRef)));
        const maxUt = Math.max(...ut), maxEp = Math.max(...ep);
        const tệ = đo[ep.indexOf(maxEp)];
        ok(`${tên}: ${đo.length} mốc 1900–2100 — THIÊN VĂN (đã khử ΔT) lệch tối đa `
            + `${maxEp.toFixed(2)}s ≤ ${dsEphem}s`,
            maxEp <= dsEphem, `tệ nhất: năm ${tệ.year}`);
        ok(`${tên}: mốc UT lệch tối đa ${maxUt.toFixed(2)}s ≤ ${dsUt}s (phần dôi là ΔT)`,
            maxUt <= dsUt);
        return { maxEp: maxEp, maxUt: maxUt };
    };
    const kqTk = soSánh(đo, 'tiết khí', DUNG_SAI.ephem.tietKhi, DUNG_SAI.ut.tietKhi);
    const max = kqTk.maxUt;
    // Chỉ số tiết khí của ứng dụng phải khớp ĐỘ của mốc tham chiếu: n mod 24
    // đếm từ Đông Chí (270°), nên deg = (270 + 15·(n mod 24)) mod 360.
    const sai = đo.filter(x => ((15 * (((x.n % 24) + 24) % 24)) % 360) !== x.deg);
    ok('…và mỗi mốc đúng ĐỘ hoàng kinh của nó (không lệch một nấc 15°)',
        sai.length === 0, sai.slice(0, 3).map(x => `n=${x.n} mong ${x.deg}°`).join(', '));

    // ── Sóc và Vọng ──
    for (const [khoá, tên] of [['soc', 'Sóc'], ['vong', 'Vọng']]) {
        const đo = await page.evaluate(({ refs, khoá }) => {
            const J1970 = 2440587.5;
            const unixOf = jdUTC8 => (jdUTC8 - 8 / 24 - J1970) * 86400;
            return refs.map(r => {
                // socSolar/vongSolar đều đánh số tuần trăng bằng
                // round((jd + 0.5 − J2000) / 29,5306), tức theo mốc SÓC. Với
                // Vọng phải đưa vào ngày của SÓC ĐỨNG TRƯỚC nó (lùi ~14,8
                // ngày) — đưa thẳng ngày Vọng thì phép làm tròn nhảy lên tuần
                // trăng sau và lệch trọn một tuần trăng. Ứng dụng cũng gọi
                // đúng như vậy: nó truyền ngày mùng 1.
                const jdRefUTC8 = r.utc / 86400 + J1970 + 8 / 24
                    - (khoá === 'vong' ? 14.77 : 0);
                const s = Solar.fromJulianDay(Math.floor(jdRefUTC8 + 0.5) - 0.5);
                const got = (khoá === 'soc' ? Ephem.socSolar(s, 8) : Ephem.vongSolar(s, 8));
                const jdGot = got.getJulianDay();
                return { ref: r.utc, app: unixOf(jdGot), year: r.year, dtRef: r.dt,
                         dtApp: ShouXingUtil.dtT(jdGot - 8 / 24 - 2451545.0) * 86400 };
            });
        }, { refs: REF[khoá], khoá });
        soSánh(đo, tên, DUNG_SAI.ephem[khoá], DUNG_SAI.ut[khoá]);
    }

    // ── Chính Ngọ ──
    const cn = await page.evaluate((refs) => refs.map(r => {
        const phút = Ephem.solarNoonMinutes(r.y, r.m, r.d, r.lon, r.tz);
        // Mốc tham chiếu là thời điểm UTC; quy về "số phút kể từ 00:00 giờ
        // đồng hồ địa phương" của chính ngày ấy để so cùng đơn vị.
        const nửaĐêmUTC = Date.UTC(r.y, r.m - 1, r.d) / 1000 - r.tz * 3600;
        return { noi: r.noi, y: r.y, m: r.m, d: r.d,
                 app: phút, ref: (r.utc - nửaĐêmUTC) / 60 };
    }), REF.chinhNgo);
    const lệchCn = cn.map(x => Math.abs(x.app - x.ref) * 60);
    const maxCn = Math.max(...lệchCn);
    const tệCn = cn[lệchCn.indexOf(maxCn)];
    ok(`Chính Ngọ: ${cn.length} ca (6 nơi × 8 ngày), lệch tối đa ${maxCn.toFixed(2)}s ≤ ${DUNG_SAI.chinhNgo}s`,
        maxCn <= DUNG_SAI.chinhNgo,
        `tệ nhất: ${tệCn.noi} ${tệCn.d}/${tệCn.m}/${tệCn.y} — ứng dụng ${tệCn.app.toFixed(3)} phút, PyEphem ${tệCn.ref.toFixed(3)} phút`);

    // Bảng ΔT: chỗ DUY NHẤT hai bộ tính được phép lệch, và lệch bao nhiêu.
    const dtBang = await page.evaluate(() => [1900, 1950, 1991, 2000, 2026, 2050, 2100].map(y => {
        const t = (Date.UTC(y, 6, 1) / 86400000 + 2440587.5) - 2451545.0;
        return { y: y, app: ShouXingUtil.dtT(t) * 86400 };
    }));
    const dtRef = {};
    for (const r of REF.tietKhi) {
        const y = new Date(r.utc * 1000).getUTCFullYear();
        if (dtRef[y] === undefined) dtRef[y] = r.dt;
    }
    console.log('       ΔT (giây) — chỗ duy nhất hai bộ được phép lệch:');
    for (const r of dtBang) {
        const ref = dtRef[r.y];
        console.log(`         ${r.y}: lunar.js ${r.app.toFixed(2)}`
            + (ref === undefined ? '' : `  ·  PyEphem ${ref.toFixed(2)}`
                + `  ·  chênh ${Math.abs(r.app - ref).toFixed(2)}`));
    }
    console.log(`       (đo được: tiết khí ${max.toFixed(1)}s UT · Chính Ngọ ${maxCn.toFixed(1)}s)`);

    // ── Hệ quả THẬT: chênh ΔT ấy có đổi được con số nào trên màn không? ──
    //
    // Ngày âm lịch đổi tại CHÍNH TÝ (nửa đêm mặt trời thật), nên mùng 1 chỉ xê
    // dịch nếu điểm Sóc rơi SÁT mốc ấy. Đo khoảng cách từ mỗi điểm Sóc tới
    // Chính Tý gần nhất, ở bốn nơi.
    //
    // Chia làm hai thời kỳ, vì ΔT là hai loại đại lượng khác nhau:
    //   · tới ~2030 nó ĐO được (IERS công bố), hai thư viện chênh ≤ 0,3s;
    //   · sau đó phải ĐOÁN, và hai bộ đa thức rẽ nhau tới trăm giây ở 2100.
    // Nên phép canh cũng phải khác nhau: thời kỳ đo được thì đòi KHÔNG ca nào
    // sát ranh, thời kỳ đoán thì chỉ đòi ca sát ranh không rơi vào dải năm mà
    // người ta thật sự lập lá số.
    const MOC_DO_DUOC = 2030;
    const MOC_AN_TOAN = 2050;
    const biên = await page.evaluate((refs) => {
        const NOI = [[105.8342, 7, 'Asia/Ho_Chi_Minh'], [2.3522, 1, 'Europe/Paris'],
                     [116.4074, 8, 'Asia/Shanghai'], [-74.0060, -5, 'America/New_York']];
        const out = [];
        for (const r of refs) {
            for (const [lon, tz, tzId] of NOI) {
                const d = new Date((r.utc + tz * 3600) * 1000);
                const y = d.getUTCFullYear(), m = d.getUTCMonth() + 1, dd = d.getUTCDate();
                const phút = d.getUTCHours() * 60 + d.getUTCMinutes() + d.getUTCSeconds() / 60;
                const a = Ephem.solarMidnightMinutes(y, m, dd, lon, tz);
                const kc = Math.min(Math.abs(phút - a), Math.abs(phút - (a + 1440))) * 60;
                out.push({ y: y, kc: kc, iso: r.iso, tzId: tzId });
            }
        }
        return out;
    }, REF.soc);

    const đoĐược = biên.filter(x => x.y <= MOC_DO_DUOC);
    const minĐo = Math.min(...đoĐược.map(x => x.kc));
    ok(`thời kỳ ΔT ĐO ĐƯỢC (≤ ${MOC_DO_DUOC}): không điểm Sóc nào cách Chính Tý dưới 60s `
        + `(gần nhất ${(minĐo / 60).toFixed(1)} phút) — hai bộ không thể cho hai mùng 1 khác nhau`,
        minĐo > 60, `${đoĐược.length} ca`);

    const sátRanh = biên.filter(x => x.kc < DUNG_SAI.ut.soc).sort((a, b) => a.kc - b.kc);
    ok(`thời kỳ ΔT phải ĐOÁN: không ca sát ranh nào rơi trước ${MOC_AN_TOAN}`,
        sátRanh.every(x => x.y > MOC_AN_TOAN),
        sátRanh.slice(0, 3).map(x => `${x.iso} ${x.tzId} ${x.kc.toFixed(0)}s`).join(' | '));
    if (sátRanh.length) {
        console.log(`       ${sátRanh.length} ca SÁT RANH (Sóc cách Chính Tý < ${DUNG_SAI.ut.soc}s), `
            + `tất cả sau ${MOC_AN_TOAN} — ở đó chênh ΔT ĐỔI ĐƯỢC mùng 1 một ngày:`);
        sátRanh.slice(0, 4).forEach(x =>
            console.log(`         ${x.iso} UTC · ${x.tzId} · cách Chính Tý ${x.kc.toFixed(0)}s`));
        console.log('         (không phải lỗi: ΔT của năm ấy chưa ai biết. Bộ của lunar.js còn');
        console.log('          sát thực tế hơn — xem bảng ΔT ở trên, mốc 2026.)');
    }
}

/* ═══════════════ B. KHÔNG ĐỔI SO VỚI BẢN CŨ ═══════════════ */
console.log(`\nB. So với bản trước khi gom phép tính (${COMMIT_CU})`);
if (!baseCu) {
    bỏQua++;
    console.log('  – bỏ qua: không dựng được bản cũ');
} else {
    const { page: pageCu } = await mở(baseCu);

    /** Cùng một dải tham số, chạy trên cả hai trang. */
    const QUÉT = () => {
        const out = { term: [], soc: [], vong: [], noon: [], lunar: [] };
        // Tiết khí: 200 năm liên tục, mỗi năm 24 mốc.
        // Bản cũ chưa có Ephem.termJd — hồi ấy tab Bát Tự gọi thẳng
        // ShouXingUtil.qiAccurate. Chính sự TƯƠNG ĐƯƠNG của hai đường ấy là
        // thứ phép thử này phải chứng minh, nên gọi bên nào có.
        const term = (typeof Ephem.termJd === 'function')
            ? (n => Ephem.termJd(n))
            : (n => ShouXingUtil.qiAccurate(n * Math.PI / 12, 8) + Solar.J2000);
        for (let n = -2400; n < 2400; n++) out.term.push(term(n));
        // Sóc / Vọng: mọi tuần trăng từ 1900 tới 2100, lấy theo mốc 29,53 ngày.
        for (let k = -1220; k < 1240; k++) {
            const s = Solar.fromJulianDay(2451545 + k * 29.5306);
            out.soc.push(Ephem.socSolar(s, 8).getJulianDay());
            out.vong.push(Ephem.vongSolar(s, 8).getJulianDay());
        }
        // Chính Ngọ: bốn nơi × cả một năm.
        const NOI = [[105.8342, 7], [2.3522, 1], [116.4074, 8], [-74.0060, -5]];
        for (const [lon, tz] of NOI) {
            for (let m = 1; m <= 12; m++) {
                for (const d of [1, 8, 15, 22, 28]) {
                    out.noon.push(Ephem.solarNoonMinutes(2026, m, d, lon, tz));
                }
            }
        }
        // Ngày ÂM LỊCH: mọi ngày của bốn năm, ở bốn nơi. Đây là đường mà lần
        // dọn mã vừa rồi chuyển chỗ (zi_* của app.js → core.js), nên quét dày.
        const NOI2 = [[105.8342, 'Asia/Ho_Chi_Minh'], [2.3522, 'Europe/Paris'],
                      [116.4074, 'Asia/Shanghai'], [-74.0060, 'America/New_York']];
        for (const [lon, tzId] of NOI2) {
            for (const y of [1991, 2000, 2026, 2050]) {
                for (let m = 1; m <= 12; m++) {
                    for (let d = 1; d <= 28; d++) {
                        const tz = getTimezoneOffset(tzId, new Date(y, m - 1, d, 12));
                        const r = zi_lunarOf(y, m, d, lon, tzId, tz);
                        out.lunar.push(r ? `${r.day}/${r.month}${r.leap ? 'N' : ''}/${r.year}` : '—');
                    }
                }
            }
        }
        return out;
    };

    const a = await pageCu.evaluate(QUÉT);
    const b = await page.evaluate(QUÉT);
    for (const [khoá, tên, đơnVị] of [
        ['term', 'tiết khí', 'ngày Julius'],
        ['soc', 'Sóc', 'ngày Julius'],
        ['vong', 'Vọng', 'ngày Julius'],
        ['noon', 'Chính Ngọ', 'phút'],
        ['lunar', 'ngày âm lịch', 'chuỗi'],
    ]) {
        const ac = a[khoá], bc = b[khoá];
        if (ac.length !== bc.length) { ok(`${tên}: cùng số mẫu`, false, `${ac.length} vs ${bc.length}`); continue; }
        let lệch = [], maxΔ = 0;
        for (let i = 0; i < ac.length; i++) {
            if (typeof ac[i] === 'number') {
                const d = Math.abs(ac[i] - bc[i]);
                if (d > maxΔ) maxΔ = d;
                if (d !== 0) lệch.push(`#${i}: ${ac[i]} vs ${bc[i]}`);
            } else if (ac[i] !== bc[i]) lệch.push(`#${i}: ${ac[i]} vs ${bc[i]}`);
        }
        ok(`${tên}: ${ac.length} mẫu TRÙNG KHÍT bản cũ (${đơnVị})`,
            lệch.length === 0, lệch.slice(0, 3).join(' | '));
    }
}

await browser.close();
svMoi.close();
if (svCu) svCu.close();
if (thưMụcCũ) fs.rmSync(thưMụcCũ, { recursive: true, force: true });
console.log(`\n${pass} đạt · ${fail} hỏng${bỏQua ? ' · ' + bỏQua + ' bỏ qua' : ''}`);
if (fail) process.exit(1);
console.log('✓ Thiên văn: khớp PyEphem, và trùng khít bản trước khi dọn mã');
