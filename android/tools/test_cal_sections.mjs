/**
 * Hai mục gập được của tab Lịch, và việc widget theo kịp ngôn ngữ.
 *
 *   node test_cal_sections.mjs
 *
 * Bốn nhóm:
 *   1. Mục "Tiết khí": một dãy 24 hàng (không còn chia đôi), ba cột.
 *   2. Mục "Lịch âm": đúng bảng Sóc/Vọng của Âm Bàn pháp ở tab Kỳ Môn.
 *   3. Gập/mở: độc lập, mở được CẢ HAI, mục dài thì cuộn, nhớ qua lần mở sau.
 *   4. Ngôn ngữ: ứng dụng ghi khoá qmdj.lang và gọi widget vẽ lại.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

let chromium;
try { ({ chromium } = await import('playwright')); }
catch (e) { console.log('Bỏ qua: chưa cài playwright.'); process.exit(0); }

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

let pass = 0, fail = 0;
function check(what, got, want) {
    const ok = String(got) === String(want);
    ok ? pass++ : fail++;
    console.log(`  ${ok ? 'ok  ' : '✗ SAI'} ${what.padEnd(46)} ${ok ? '' : `được "${got}", cần "${want}"`}`);
}
function ok(what, cond, detail) {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(46)} ${cond ? '' : (detail || '')}`);
}

/** Trang mới, đã sang tiếng Việt và đang ở tab Lịch. */
async function openCal(ctx) {
    const page = await ctx.newPage();
    const errs = [];
    page.on('pageerror', e => errs.push(e.message));
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1000);
    await page.click('#langDisplayBtn');
    await page.waitForSelector('#optOverlay.open .opt-row[data-value="vi"]');
    await page.click('.opt-row[data-value="vi"]');
    await page.waitForTimeout(800);
    await page.click('#tabCal');
    await page.waitForTimeout(800);
    return { page, errs };
}
const setOpen = (page, which, want) => page.evaluate(([w, v]) => {
    const sec = document.getElementById(w === 'jq' ? 'calSecJq' : 'calSecAm');
    // Thanh tiêu đề nay là hàng <thead> của chính bảng.
    if (sec.classList.contains('cal-sec-open') !== v) sec.querySelector('.cal-sec-head').click();
}, [which, want]);

/* ── 1. Mục Tiết khí: 24 hàng liền, ba cột ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);
    await setOpen(page, 'jq', true);
    await page.waitForTimeout(500);

    console.log('\nMục "Tiết khí" — một dãy 24, thêm cột can chi');
    const r = await page.evaluate(() => {
        var textCx = el => {
            if (!el) return null;
            var w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
            var n = w.nextNode();
            if (!n) return null;
            var rg = document.createRange();
            rg.selectNodeContents(n);
            var b = rg.getBoundingClientRect();
            return { left: b.left, cx: (b.left + b.right) / 2 };
        };
        var heads = [...document.querySelectorAll('#calJieQi thead th')];
        var row1 = document.querySelector('#calJqBody tr');
        return {
            // Tiêu đề mục CHÍNH LÀ hàng tên cột — không còn nhãn riêng, nên tên
            // "Tiết khí" không bị lặp hai lần như trước.
            dupTitle: !!document.getElementById('calJqTitle'),
            headIsThead: !!document.querySelector('#calJieQi thead tr.cal-sec-head'),
            cols: heads.map(x => x.textContent.replace(/[▾▸]/g, '').trim()),
            rows: document.querySelectorAll('#calJqBody tr').length,
            cells: [...document.querySelectorAll('#calJqBody tr')].map(tr => tr.cells.length),
            // Không còn vách ngăn giữa hai nửa bảng — dấu hiệu của bố cục cũ.
            split: document.querySelectorAll('#calJieQi .cal-jq-split').length,
            gz: [...document.querySelectorAll('#calJqBody tr')].map(tr => tr.cells[2].textContent.trim()),
            // "Dương lịch" CĂN GIỮA — như giá trị của nó. Căn trái (bản trước)
            // dán cột ngày vào sát cột tên trong khi phía Can chi hở ra một
            // mảng trống rộng gấp năm: đo trên máy 393px là 33px một bên và
            // 169px bên kia.
            dateHeadAlign: getComputedStyle(heads[1]).textAlign,
            dateValAlign: getComputedStyle(row1.children[1]).textAlign,
            dateHeadCx: textCx(heads[1])?.cx,
            dateValCx: textCx(row1.children[1])?.cx,
            // Ba khoảng hở của hàng tiêu đề, đo theo CHỮ chứ không theo ô.
            gaps: (() => {
                const b = el => {
                    const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
                    const n = w.nextNode();
                    if (!n) return null;
                    const rg = document.createRange();
                    rg.selectNodeContents(n);
                    const r = rg.getBoundingClientRect();
                    return { l: r.left, r: r.right };
                };
                const t = heads.map(b);
                const box = document.getElementById('calJieQi').getBoundingClientRect();
                return t.every(Boolean) ? {
                    ho1: t[1].l - t[0].r, ho2: t[2].l - t[1].r, phai: box.right - t[2].r,
                } : null;
            })(),
        };
    });
    ok('không còn nhãn tiêu đề riêng (hết lặp tên)', !r.dupTitle);
    ok('hàng tên cột đóng luôn vai tiêu đề', r.headIsThead);
    check('đúng 24 hàng', r.rows, 24);
    ok('mọi hàng đều 3 ô', r.cells.every(n => n === 3), r.cells.join(','));
    check('ba cột', r.cols.join(' | '), 'Tiết Khí | Dương lịch | Can chi');
    check('không còn chia đôi', r.split, 0);
    check('tiêu đề "Dương lịch" căn giữa', r.dateHeadAlign, 'center');
    check('giá trị "Dương lịch" cũng căn giữa', r.dateValAlign, 'center');
    ok('tiêu đề "Dương lịch" trùng tâm giá trị',
        Math.abs(r.dateHeadCx - r.dateValCx) <= 1.5,
        `tiêu đề ${r.dateHeadCx?.toFixed(1)} vs giá trị ${r.dateValCx?.toFixed(1)}`);
    // Cột giữa phải nằm GIỮA hai cột bên, không dán vào cột tên. KHÔNG đòi hai
    // khoảng hở bằng nhau: ba cột nay dùng chung một bộ bề rộng với mục Lịch âm
    // (xem phần "Cột tiêu đề không trượt ngang"), nên cột cuối rộng bằng mốc
    // ngày giờ của Vọng chứ không bằng can chi, và chữ căn giữa cột ấy tất
    // nhiên lệch khỏi tâm hình học của hàng. Chênh gấp đôi thì còn tự nhiên;
    // gấp năm như bản căn trái (33px một bên, 169px bên kia) thì phải đỏ.
    ok('cột giữa không dán vào cột tên',
        r.gaps && Math.max(r.gaps.ho1, r.gaps.ho2) <= 2 * Math.min(r.gaps.ho1, r.gaps.ho2),
        `trái ${r.gaps?.ho1.toFixed(1)} · phải ${r.gaps?.ho2.toFixed(1)}`);
    ok('"Can chi" không dán vào mép phải', r.gaps && r.gaps.phai >= 18,
        `cách mép ${r.gaps?.phai.toFixed(1)}px`);
    ok('cột can chi không ô nào trống', r.gz.every(x => x.length > 0));
    // Mỗi trụ tháng phủ đúng hai tiết khí liền nhau (tiết mở tháng, rồi khí).
    let pairs = true;
    for (let i = 1; i + 1 < 24; i += 2) if (r.gz[i] !== r.gz[i + 1]) pairs = false;
    ok('mỗi trụ tháng phủ đúng hai mục', pairs, r.gz.join(','));
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 2. Mục Lịch âm = bảng Âm Bàn pháp ở tab Kỳ Môn ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);

    console.log('\nMục "Lịch âm" — khớp bảng chi tiết Âm Bàn pháp');
    // Bảng gốc ở tab Kỳ Môn (phái Âm Bàn mới dựng bảng này).
    const km = await page.evaluate(() => {
        window.showTab('qmdj');
        selectMethod('amban');
        processAll();
        return [...document.querySelectorAll('#ab-tbody tr')]
            .map(tr => [...tr.cells].map(c => c.textContent.trim()).join(' | '));
    });
    await page.click('#tabCal'); await page.waitForTimeout(700);
    await setOpen(page, 'am', true);
    await page.waitForTimeout(600);
    const cal = await page.evaluate(() => {
        // Tâm ngang của phần TEXT thật sự (Range trên node văn bản đầu tiên),
        // không phải tâm của cả ô — ô thường rộng hơn hẳn nội dung, đo theo ô
        // thì sai lệch dù chữ đã canh đúng.
        var textCx = el => {
            if (!el) return null;
            var w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
            var n = w.nextNode();
            if (!n) return null;
            var rg = document.createRange();
            rg.selectNodeContents(n);
            var b = rg.getBoundingClientRect();
            return (b.left + b.right) / 2;
        };
        var heads = [...document.querySelectorAll('#calAmBan thead th')];
        var row1 = document.querySelector('#calAmBan tbody tr');
        var socTd = row1 ? row1.children[1] : null;
        var vongTd = row1 ? row1.children[2] : null;
        return {
            dupTitle: !!document.getElementById('calAmTitle'),
            headIsThead: !!document.querySelector('#calAmBan thead tr.cal-sec-head'),
            // Tiêu đề Sóc và Vọng phải CĂN GIỮA — VÀ giá trị bên dưới cũng vậy,
            // không như cột "Dương lịch" của Tiết khí (vẫn căn trái).
            headCentred: heads.slice(1).every(x => getComputedStyle(x).textAlign === 'center'),
            valCentred: socTd && vongTd &&
                getComputedStyle(socTd).textAlign === 'center' &&
                getComputedStyle(vongTd).textAlign === 'center',
            socHeadCx: textCx(heads[1]), socValCx: textCx(socTd),
            vongHeadCx: textCx(heads[2]), vongValCx: textCx(vongTd),
            cols: heads.map(x => x.textContent.replace(/[▾▸]/g, '').trim()),
            rows: [...document.querySelectorAll('#calAmBan tbody tr')]
                .map(tr => [...tr.cells].map(c => c.textContent.trim()).join(' | ')),
            active: (document.getElementById('calAmActive') || {}).textContent,
        };
    });
    ok('không còn nhãn tiêu đề riêng', !cal.dupTitle);
    ok('hàng tên cột đóng luôn vai tiêu đề', cal.headIsThead);
    check('ba cột', cal.cols.join(' | '), 'Tháng âm | Sóc | Vọng');
    ok('tiêu đề Sóc và Vọng căn giữa', cal.headCentred);
    ok('giá trị Sóc và Vọng cũng căn giữa', cal.valCentred);
    // Tâm chữ tiêu đề phải trùng tâm chữ giá trị — đây chính là yêu cầu
    // "Sóc/Vọng đang bị right aligned so với values, hãy fix để center so với
    // values". Ngưỡng 1,5px: chừa cho lệch làm tròn nửa pixel của trình duyệt.
    ok('tiêu đề "Sóc" trùng tâm giá trị', Math.abs(cal.socHeadCx - cal.socValCx) <= 1.5,
        `tiêu đề ${cal.socHeadCx?.toFixed(1)} vs giá trị ${cal.socValCx?.toFixed(1)}`);
    ok('tiêu đề "Vọng" trùng tâm giá trị', Math.abs(cal.vongHeadCx - cal.vongValCx) <= 1.5,
        `tiêu đề ${cal.vongHeadCx?.toFixed(1)} vs giá trị ${cal.vongValCx?.toFixed(1)}`);
    ok('số tháng khớp bảng gốc', cal.rows.length === km.length, `${cal.rows.length} vs ${km.length}`);
    ok('từng dòng khớp bảng gốc', cal.rows.join('#') === km.join('#'),
        'Lịch: ' + (cal.rows[0] || '—') + ' · Kỳ Môn: ' + (km[0] || '—'));
    ok('có tô đậm tháng đang xem', !!cal.active, String(cal.active));
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 3. Gập/mở độc lập, cuộn được, nhớ trạng thái ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page } = await openCal(ctx);

    console.log('\nGập/mở hai mục');
    const state = () => page.evaluate(() => ({
        jq: document.getElementById('calSecJq').classList.contains('cal-sec-open'),
        am: document.getElementById('calSecAm').classList.contains('cal-sec-open'),
        jqH: Math.round(document.getElementById('calJieQi').clientHeight),
        amH: Math.round(document.getElementById('calAmBan').clientHeight),
        jqScroll: document.getElementById('calJieQi').scrollHeight,
        amScroll: document.getElementById('calAmBan').scrollHeight,
        over: document.documentElement.scrollHeight > window.innerHeight + 1,
    }));
    await setOpen(page, 'jq', true); await setOpen(page, 'am', true);
    await page.waitForTimeout(700);
    const both = await state();
    ok('mở được CẢ HAI cùng lúc', both.jq && both.am);
    ok('cả hai đều có chiều cao thật', both.jqH > 40 && both.amH > 40, `${both.jqH} / ${both.amH}`);
    ok('mục dài thì cuộn được', both.jqScroll > both.jqH, `${both.jqScroll} ≤ ${both.jqH}`);
    ok('trang không tràn dọc khi mở cả hai', !both.over);

    await setOpen(page, 'jq', false);
    await page.waitForTimeout(600);
    const oneOpen = await state();
    ok('đóng mục này không đụng mục kia', !oneOpen.jq && oneOpen.am);
    // Đóng lại thì chỉ còn HÀNG TIÊU ĐỀ (trước đây thân bị display:none nên
    // cao 0; nay tiêu đề nằm trong thân nên còn đúng chiều cao một hàng).
    ok('mục đã đóng chỉ còn hàng tiêu đề', oneOpen.jqH > 10 && oneOpen.jqH < 46,
        String(oneOpen.jqH));
    ok('mục còn mở được rộng thêm', oneOpen.amH > both.amH, `${oneOpen.amH} ≤ ${both.amH}`);

    // Trạng thái phải sống sót qua lần mở sau (ghi vào kho tuỳ chọn).
    const kept = await page.evaluate(() => ({
        jq: localStorage.getItem('qmdj.calSecJq'), am: localStorage.getItem('qmdj.calSecAm'),
    }));
    check('nhớ trạng thái mục Tiết khí', kept.jq, '0');
    check('nhớ trạng thái mục Lịch âm', kept.am, '1');
    await ctx.close();
}

/* ── 3b. Vị trí tiêu đề CỐ ĐỊNH bất kể gập hay mở ──
   Tiêu đề của MỘT mục không được di chuyển do CHÍNH mục đó (hay mục khác)
   đổi trạng thái, TRỪ khi có mục nào NẰM TRÊN nó vừa đổi chiều cao thật (đó
   là phản xạ bình thường của một accordion — đóng mục Tiết khí thì dĩ nhiên
   đẩy tiêu đề Lịch âm bên dưới nó lên, không ai coi đó là lỗi).
   Bug đã sửa: fitGrid từng có HAI công thức khác hẳn nhau — "còn mục nào mở"
   dùng ROW_MIN, "không mục nào mở" cho lưới ăn hết phần dư (có thể chạm
   ROW_MAX) — nên bấm gập/mở là lưới lịch đổi cỡ, kéo CẢ HAI tiêu đề nhảy hơn
   100px dù người dùng chỉ đóng/mở một mục. Nay chỉ còn MỘT công thức, không
   rẽ nhánh theo trạng thái, nên lưới (và do đó tiêu đề Tiết khí — mục ĐẦU
   TIÊN, không có gì phía trên ngoài lưới) đứng yên tuyệt đối. */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);

    console.log('\nVị trí tiêu đề cố định bất kể gập/mở');
    const pos = () => page.evaluate(() => {
        var top = id => {
            var el = document.getElementById(id);
            return el ? +el.getBoundingClientRect().top.toFixed(1) : null;
        };
        return {
            jq: top('calSecJq'), am: top('calSecAm'),
            rowH: getComputedStyle(document.documentElement).getPropertyValue('--cal-row-h').trim(),
        };
    });

    // Mục ĐẦU TIÊN (Tiết khí): không có gì phía trên ngoài lưới lịch cố định,
    // nên tiêu đề của nó phải đứng đúng MỘT chỗ qua toàn bộ 5 trạng thái sau,
    // bất kể trạng thái của MỤC KIA.
    const seq = [];
    seq.push(['jq mở · am đóng (ban đầu)', await pos()]);
    await setOpen(page, 'jq', false); await page.waitForTimeout(500);
    seq.push(['jq đóng · am đóng', await pos()]);
    await setOpen(page, 'am', true); await page.waitForTimeout(500);
    seq.push(['jq đóng · am mở', await pos()]);
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở · am mở', await pos()]);
    await setOpen(page, 'am', false); await page.waitForTimeout(500);
    seq.push(['jq mở · am đóng (quay lại)', await pos()]);

    const jqTops = seq.map(([, p]) => p.jq);
    const rowHs = seq.map(([, p]) => p.rowH);
    ok('tiêu đề Tiết khí đứng đúng một chỗ ở cả 5 trạng thái',
        jqTops.every(t => Math.abs(t - jqTops[0]) < 1),
        seq.map(([label, p]) => `${label}: ${p.jq}px`).join(' · '));
    ok('chiều cao hàng lưới (--cal-row-h) không đổi theo trạng thái gập/mở',
        rowHs.every(h => h === rowHs[0]), rowHs.join(' → '));

    // Đóng/mở MỤC LỊCH ÂM (mục cuối) không được tự dịch chuyển TIÊU ĐỀ CỦA
    // CHÍNH NÓ — vì tiêu đề luôn nằm TRÊN thân của chính mục ấy. Giữ nguyên
    // trạng thái Tiết khí (đang mở) trong suốt phép so này.
    const beforeAmToggle = await pos();
    await setOpen(page, 'am', true); await page.waitForTimeout(500);
    const amOpened = await pos();
    await setOpen(page, 'am', false); await page.waitForTimeout(500);
    const amClosed = await pos();
    ok('gập/mở CHÍNH MÌNH không dịch chuyển tiêu đề của mục Lịch âm',
        Math.abs(amOpened.am - beforeAmToggle.am) < 1 && Math.abs(amClosed.am - beforeAmToggle.am) < 1,
        `trước ${beforeAmToggle.am}px · lúc mở ${amOpened.am}px · đóng lại ${amClosed.am}px`);

    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 3c. Cột tiêu đề CỐ ĐỊNH NGANG bất kể gập hay mở ──
   Anh em của 3b, nhưng theo trục NGANG: bấm gập/mở thì ba ô tiêu đề không
   được co giãn hay trượt sang chỗ khác.
   Bug đã sửa: mục đóng lại thì tbody bị `display: none`, mà như vậy là gỡ hẳn
   24 hàng khỏi phép chia bề rộng cột của bảng — còn mỗi ba ô tiêu đề tự chia
   nhau 100% bề ngang. Mở ra thì bề rộng lại do nội dung 24 hàng quyết định,
   nên mỗi lần bấm là cột tiêu đề nhảy ngang (đo trên máy 393px: cột giữa của
   mục Lịch âm lệch 106px). Nay mục đóng chỉ bị KẸP chiều cao xuống đúng hàng
   tiêu đề, bảng vẫn còn đủ hàng, nên bề rộng cột không đổi. */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);

    console.log('\nCột tiêu đề không trượt ngang khi gập/mở');
    // Mép trái từng ô tiêu đề, đo THEO KHUNG của chính mục (không theo màn
    // hình) để phép so không dính vào chuyện mục nằm cao hay thấp.
    const cols = () => page.evaluate(() => {
        const one = id => {
            const body = document.getElementById(id);
            const left = body.getBoundingClientRect().left;
            return [...body.querySelectorAll('thead th')]
                .map(x => +(x.getBoundingClientRect().left - left).toFixed(1));
        };
        return { jq: one('calJieQi'), am: one('calAmBan') };
    });

    const seq = [];
    seq.push(['jq mở · am đóng (ban đầu)', await cols()]);
    await setOpen(page, 'am', true); await page.waitForTimeout(500);
    seq.push(['jq mở · am mở', await cols()]);
    await setOpen(page, 'jq', false); await page.waitForTimeout(500);
    seq.push(['jq đóng · am mở', await cols()]);
    await setOpen(page, 'am', false); await page.waitForTimeout(500);
    seq.push(['jq đóng · am đóng', await cols()]);
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở · am đóng (quay lại)', await cols()]);

    for (const k of ['jq', 'am']) {
        const first = seq[0][1][k];
        const worst = Math.max(...seq.map(([, p]) => Math.max(...p[k].map((v, i) => Math.abs(v - first[i])))));
        ok(`${k === 'jq' ? 'Tiết khí' : 'Lịch âm'}: ba cột tiêu đề đứng yên ở cả 5 trạng thái`,
            worst < 1,
            seq.map(([label, p]) => `${label}: [${p[k].join(',')}]`).join(' · '));
    }

    // Hai mục là HAI bảng riêng, nhưng nằm ngay trên dưới nhau nên phải thẳng
    // cột với nhau: "Dương lịch" trên "Sóc", "Can chi" trên "Vọng". Để mỗi bảng
    // tự co theo dữ liệu của mình thì lệch 9px và 89px trên máy 393px.
    await setOpen(page, 'jq', true); await setOpen(page, 'am', true);
    await page.waitForTimeout(700);
    const pair = await page.evaluate(() => {
        const cx = el => {
            const w = document.createTreeWalker(el, NodeFilter.SHOW_TEXT);
            const n = w.nextNode();
            if (!n) return null;
            const rg = document.createRange();
            rg.selectNodeContents(n);
            const b = rg.getBoundingClientRect();
            return (b.left + b.right) / 2;
        };
        const one = id => {
            const body = document.getElementById(id);
            const bl = body.getBoundingClientRect().left;
            const ths = [...body.querySelectorAll('thead th')];
            let spill = 0;
            for (const cell of body.querySelectorAll('th, td')) {
                spill = Math.max(spill, cell.scrollWidth - cell.clientWidth);
            }
            return {
                edges: ths.map(x => +(x.getBoundingClientRect().left - bl).toFixed(1)),
                headCx: ths.map(cx),
                spill,
            };
        };
        return { jq: one('calJieQi'), am: one('calAmBan') };
    });
    ok('mép ba cột của hai mục trùng nhau',
        pair.jq.edges.every((v, i) => Math.abs(v - pair.am.edges[i]) < 1),
        `Tiết khí [${pair.jq.edges}] · Lịch âm [${pair.am.edges}]`);
    ok('"Dương lịch" thẳng cột với "Sóc"',
        Math.abs(pair.jq.headCx[1] - pair.am.headCx[1]) <= 1.5,
        `${pair.jq.headCx[1]?.toFixed(1)} vs ${pair.am.headCx[1]?.toFixed(1)}`);
    ok('"Can chi" thẳng cột với "Vọng"',
        Math.abs(pair.jq.headCx[2] - pair.am.headCx[2]) <= 1.5,
        `${pair.jq.headCx[2]?.toFixed(1)} vs ${pair.am.headCx[2]?.toFixed(1)}`);
    // Ghim bề rộng cột thì phải ghim ĐỦ RỘNG: cột hẹp hơn chữ là chữ tràn sang
    // ô bên hoặc bị cắt, mà `white-space: nowrap` thì không xuống dòng được.
    ok('không ô nào có chữ rộng hơn cột của nó',
        Math.max(pair.jq.spill, pair.am.spill) <= 0.5,
        `thừa ${Math.max(pair.jq.spill, pair.am.spill).toFixed(1)}px`);
    await setOpen(page, 'am', false);
    await page.waitForTimeout(500);

    // Đóng lại vẫn phải là ĐÓNG: kẹp chiều cao chứ không phải để lộ cả bảng.
    const shut = await page.evaluate(() => {
        const b = document.getElementById('calAmBan');
        return { h: Math.round(b.clientHeight), scroll: b.scrollHeight, rows: b.querySelectorAll('tbody tr').length };
    });
    ok('mục đóng vẫn chỉ cao đúng hàng tiêu đề', shut.h > 10 && shut.h < 46, String(shut.h));
    ok('…dù bảng bên trong vẫn còn đủ hàng (thứ giữ bề rộng cột)',
        shut.rows > 6 && shut.scroll > shut.h, `${shut.rows} hàng, scrollHeight ${shut.scroll}`);
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 4. Ngôn ngữ cho widget ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    // Giả lập cầu native để đếm số lần widget được bảo vẽ lại.
    await ctx.addInitScript(() => {
        window.__widgetRefreshes = 0;
        window.QMDJNative = {
            getPref: k => { try { return localStorage.getItem(k); } catch (e) { return null; } },
            setPref: (k, v) => { try { localStorage.setItem(k, v); } catch (e) {} },
            deviceTimeZone: () => 'Europe/Paris',
            platform: () => 'android',
            refreshCalendarWidget: () => { window.__widgetRefreshes++; },
        };
    });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(1100);

    console.log('\nNgôn ngữ truyền sang widget');
    const read = () => page.evaluate(() => ({
        lang: localStorage.getItem('qmdj.lang'),
        n: window.__widgetRefreshes,
    }));
    const a = await read();
    ok('ghi khoá ngay từ lúc mở app', a.lang === 'vi' || a.lang === 'zh', String(a.lang));

    const pick = async v => {
        await page.click('#langDisplayBtn');
        await page.waitForSelector(`#optOverlay.open .opt-row[data-value="${v}"]`);
        await page.click(`.opt-row[data-value="${v}"]`);
        await page.waitForTimeout(800);
    };
    // Ứng dụng mở ra ở TIẾNG VIỆT, nên phải sang tiếng Trung TRƯỚC thì lần
    // chọn tiếng Việt sau mới là một thay đổi thật — chọn đúng thứ đang dùng
    // thì setLang() thoát sớm và chẳng có gì để canh.
    check('mở app ra là tiếng Việt', a.lang, 'vi');

    await pick('zh');
    const b = await read();
    check('chọn tiếng Trung thì khoá đổi', b.lang, 'zh');
    ok('và widget được bảo vẽ lại', b.n > a.n, `${a.n} → ${b.n}`);

    await pick('vi');
    const c = await read();
    check('chọn lại tiếng Việt thì khoá đổi', c.lang, 'vi');
    ok('widget lại được bảo vẽ lại', c.n > b.n, `${b.n} → ${c.n}`);
    await ctx.close();
}

/* ── 4b. Cuộn: cả ba ô tiêu đề phải dính lại CÙNG NHAU ──
   Từng có lúc ô tiêu đề cuối bị đè position:relative (để đặt dấu mũi), mà
   relative thì GỠ MẤT position:sticky của riêng ô ấy: hai cột đầu vẫn dính, cột
   thứ ba trôi theo nội dung — cuộn xuống là giá trị cột cuối đè lên chỗ hàng
   tiêu đề. Trên máy thật trông như bảng vỡ đôi. */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);
    await setOpen(page, 'jq', true);
    await setOpen(page, 'am', true);
    await page.waitForTimeout(600);

    console.log('\nCuộn: hàng tiêu đề dính lại');
    await page.evaluate(() => {
        document.getElementById('calJieQi').scrollTop = 200;
        document.getElementById('calAmBan').scrollTop = 120;
    });
    await page.waitForTimeout(400);
    const r = await page.evaluate(() => {
        const chk = id => {
            const body = document.getElementById(id);
            const br = body.getBoundingClientRect();
            const ths = [...body.querySelectorAll('thead th')];
            const tops = ths.map(x => +x.getBoundingClientRect().top.toFixed(1));
            return {
                scrolled: body.scrollTop,
                sameTop: new Set(tops).size === 1,
                atTop: Math.abs(tops[0] - br.top) < 2,
                sticky: ths.every(x => getComputedStyle(x).position === 'sticky'),
                tops,
            };
        };
        return { jq: chk('calJieQi'), am: chk('calAmBan') };
    });
    for (const [name, x] of [['Tiết khí', r.jq], ['Lịch âm', r.am]]) {
        ok(`${name}: thật sự đã cuộn`, x.scrolled > 50, String(x.scrolled));
        ok(`${name}: mọi ô tiêu đề đều sticky`, x.sticky);
        ok(`${name}: ba ô tiêu đề cùng một hàng`, x.sameTop, x.tops.join(','));
        ok(`${name}: tiêu đề dính mép trên khung`, x.atTop, x.tops.join(','));
    }
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 5. Hợp đồng đồng bộ widget — soi thẳng vào mã Kotlin ──
   Widget vẽ bằng Kotlin, mà ở đây không chạy được Kotlin: phần kiểm ở trên chỉ
   chứng minh được NỬA phía JavaScript (ghi khoá, gọi cầu). Nửa kia từng hỏng
   mà cả bộ kiểm thử vẫn xanh, nên đọc thẳng mã nguồn cho từng mắt xích. */
{
    console.log('\nHợp đồng đồng bộ widget (đọc mã Kotlin)');
    const read = p => fs.readFileSync(path.join(HERE, '..', p), 'utf8');
    const lunarKt = read('app/src/main/java/com/bazi/qimen/LunarTable.kt');
    const provKt  = read('app/src/main/java/com/bazi/qimen/CalendarWidgetProvider.kt');
    const bridge  = read('app/src/main/java/com/bazi/qimen/WebAppBridge.kt');
    const mainKt  = read('app/src/main/java/com/bazi/qimen/MainActivity.kt');
    const manifest= read('app/src/main/AndroidManifest.xml');
    const appJs   = read('app/src/main/assets/web/js/app.js');
    const calJs   = read('app/src/main/assets/web/js/calendar.js');
    const secKt   = read('app/src/main/java/com/bazi/qimen/WidgetSections.kt');
    const supKt   = read('app/src/main/java/com/bazi/qimen/WidgetSupport.kt');
    const svcKt   = read('app/src/main/java/com/bazi/qimen/WidgetSectionService.kt');
    const layXml  = read('app/src/main/res/layout/widget_calendar.xml');
    const rowXml  = read('app/src/main/res/layout/widget_sec_row.xml');
    // Phần dựng widget nay trải ra bốn tệp; phần lớn phép canh dưới đây chỉ cần
    // biết "có ở đâu đó trong mã widget".
    const widgetKt = provKt + secKt + supKt + svcKt;

    // Mặc định hai bên PHẢI trùng nhau: ngay sau khi cài mới, chưa ai ghi khoá
    // qmdj.lang, mà ứng dụng đã hiện một thứ tiếng rồi.
    const appDefault = (/const initLang = .*?: '(vi|zh)'/.exec(appJs) || [])[1];
    const ktDefault = (/DEFAULT_LANG = "(vi|zh)"/.exec(lunarKt) || [])[1];
    ok('mặc định ngôn ngữ của app đọc được', !!appDefault, String(appDefault));
    check('widget mặc định cùng thứ tiếng với app', ktDefault, appDefault);

    // Từng chỗ widget vẽ chữ phải theo ngôn ngữ, không được chốt cứng.
    ok('tiêu đề widget theo ngôn ngữ', /if \(zh\) "农历/.test(provKt));
    ok('thứ trong tuần theo ngôn ngữ', /if \(zh\) arrayOf\("一"/.test(provKt));
    // Đối số zh nằm sau một lời gọi lồng nhau — mẫu không được dừng ở ")" đầu.
    ok('tên tiết khí theo ngôn ngữ', /jieQiYearOf\(.*\bzh\)/.test(widgetKt));
    ok('can chi theo ngôn ngữ', /ganZhiOf\(jdn, zh\)/.test(widgetKt));
    ok('tiêu đề hai cột theo ngôn ngữ', /if \(zh\) "节气"/.test(widgetKt));
    ok('đọc khoá qmdj.lang', /LunarTable\.langOf\(context\)/.test(widgetKt));

    // Đường báo cho widget vẽ lại.
    ok('cầu native có refreshCalendarWidget', /@JavascriptInterface\s+fun refreshCalendarWidget/.test(bridge));
    ok('provider có refreshNow', /fun refreshNow\(context: Context\)/.test(provKt));
    ok('manifest khai báo WIDGET_REFRESH', /com\.bazi\.qimen\.WIDGET_REFRESH/.test(manifest));
    ok('onReceive xử lý ACTION_REFRESH', /ACTION_REFRESH -> refreshAll/.test(provKt));
    // Không treo toàn bộ việc đồng bộ vào một lời gọi từ trang web.
    ok('rời ứng dụng thì widget vẽ lại', /fun onStop\(\)[\s\S]{0,160}refreshNow\(this\)/.test(mainKt));

    // Phía JavaScript: đổi ngôn ngữ VÀ đổi địa điểm đều phải báo.
    ok('có một chỗ dùng chung để bảo widget vẽ lại',
        /function pokeWidget\(\)[\s\S]{0,300}refreshCalendarWidget/.test(calJs));
    ok('đổi ngôn ngữ thì bảo widget vẽ lại',
        /function publishLang\(\)[\s\S]{0,200}pokeWidget\(\)/.test(calJs));
    ok('ghi bảng tháng xong cũng bảo widget vẽ lại',
        /function publishLunarCache\(\)[\s\S]{0,3000}pokeWidget\(\)/.test(calJs));
    ok('ghi bảng tháng ngoài cả tab Lịch',
        /__calRecalcWrapped[\s\S]{0,1600}publishLunarCache\(\)/.test(calJs));
    ok('publishLang được gọi khi đổi nhãn', /\n\s*publishLang\(\);/.test(calJs));
    ok('đổi địa điểm cũng bảo widget vẽ lại',
        /__calRecalcWrapped[\s\S]{0,1700}pokeWidget\(\)/.test(calJs));

    // Lỗi từng lọt lưới: khối "công bố lúc mở app" bị đặt NHẦM vào trong
    // toggleSection (chỉ chạy khi người dùng bấm mở/đóng mục), thay vì vào
    // đúng chỗ chạy một lần lúc khởi động — nên ai ghim widget rồi không đụng
    // gì tới app thì widget không bao giờ nhận được bảng cả. Đo vị trí thật
    // trong mã nguồn, không suy đoán qua hành vi (dễ bị một đường công bố
    // khác của app.js che khuất mất sự khác biệt).
    const toggleBody = (/function toggleSection\(which\) \{[\s\S]*?\n    \}/.exec(calJs) || [''])[0];
    ok('toggleSection không tự công bố (đó là việc của lúc mở app, không phải lúc bấm)',
        toggleBody.length > 0 && !/publishLunarCache/.test(toggleBody));
    ok('có khối công bố kèm hẹn giờ Ở NGOÀI toggleSection (chạy lúc mở app)',
        /setTimeout\(function \(\) \{[\s\S]{0,40}try \{ publishLunarCache\(\); \}/.test(calJs.replace(toggleBody, '')));

    // Lỗi khác từng lọt lưới: chưa chọn địa điểm (hoặc múi giờ máy không
    // khớp mục nào trong countryData) thì Kotlin xưa chốt cứng giờ VIỆT NAM,
    // trong khi app lại mặc định/suy ra một nơi khác — widget và app lệch
    // giờ tiết khí ngay từ lần mở đầu tiên. Nay cả hai lượt "chưa có gì" đều
    // phải rơi về múi giờ MÁY (TimeZone.getDefault()), không phải một nước
    // chốt cứng.
    ok('widget KHÔNG còn chốt cứng giờ Việt Nam khi chưa có địa điểm',
        !/getTimeZone\(DEFAULT_TZ\)/.test(widgetKt) && !/"Asia\/Ho_Chi_Minh"/.test(widgetKt));
    const selectedTzFn = (/fun timeZone\(context: Context\)[\s\S]*?\n    \}/.exec(supKt) || [''])[0];
    check('cả ba lượt "chưa có gì" đều rơi về múi giờ máy',
        (selectedTzFn.match(/TimeZone\.getDefault\(\)/g) || []).length, 3);

    /* ── Widget phải hiện ĐÚNG hai mục của tab Lịch ── */
    console.log('\nWidget dựng đúng hai mục của tab Lịch');
    ok('mục Tiết khí có cột CAN CHI THÁNG', /LunarTable\.ganZhi60\(it\.gz, zh\)/.test(secKt));
    ok('có mục Lịch âm với Sóc và Vọng',
        /stamp\(it\.socJdn, it\.socMin, tz\)/.test(secKt) &&
        /stamp\(it\.vongJdn, it\.vongMin, tz\)/.test(secKt));
    // Trạng thái gập/mở phải ĐỌC TỪ chính hai khoá mà calendar.js ghi.
    check('widget đọc khoá gập/mở của mục Tiết khí',
        /SEC_JQ = "([^"]+)"/.exec(supKt)?.[1], 'qmdj.calSecJq');
    check('widget đọc khoá gập/mở của mục Lịch âm',
        /SEC_AM = "([^"]+)"/.exec(supKt)?.[1], 'qmdj.calSecAm');
    ok('…và mặc định khớp calendar.js (Tiết khí mở, Lịch âm đóng)',
        /getString\(SEC_JQ, null\) != "0"/.test(supKt) &&
        /getString\(SEC_AM, null\) == "1"/.test(supKt));
    // Ghi khoá thôi chưa đủ: widget chỉ tự vẽ lại lúc nửa đêm.
    const toggleFn = (/function toggleSection\(which\) \{[\s\S]*?\n    \}/.exec(calJs) || [''])[0];
    ok('bấm gập/mở trong ứng dụng thì bảo widget vẽ lại ngay',
        /pokeWidget\(\)/.test(toggleFn));

    /* ── Widget dùng được bằng ngón tay, không phải ảnh tĩnh ── */
    console.log('\nWidget cuộn được và chạm được');
    // Cuộn chỉ có trong collection view do RemoteViewsService nuôi.
    ok('hai mục là ListView, không còn vẽ vào bitmap',
        /<ListView[\s\S]{0,200}@\+id\/jqList/.test(layXml) &&
        /<ListView[\s\S]{0,200}@\+id\/amList/.test(layXml) &&
        !/fun drawSections\(/.test(widgetKt));
    ok('có RemoteViewsService nuôi hàng cho hai mục',
        /class WidgetSectionService : RemoteViewsService\(\)/.test(svcKt) &&
        /RemoteViewsFactory/.test(svcKt));
    ok('manifest khai service kèm quyền BIND_REMOTEVIEWS (thiếu là mục trống trơn)',
        /<service[\s\S]{0,260}\.WidgetSectionService[\s\S]{0,260}android\.permission\.BIND_REMOTEVIEWS/
            .test(manifest));
    ok('provider nối adapter cho cả hai ListView',
        /setRemoteAdapter\(listId, sectionIntent\(context, id, sec\.key\)\)/.test(provKt));
    // filterEquals bỏ qua extras: hai mục chỉ khác extras thì dùng chung một
    // factory và cùng hiện một bảng.
    ok('khoá mục nằm trong Intent.data, không chỉ trong extras',
        /setData\(Uri\.parse\("qmdj:\/\/widget\/\$id\/sec\/\$key"\)\)/.test(provKt));
    ok('bảo factory đọc lại sau mỗi lần vẽ',
        /notifyAppWidgetViewDataChanged\(id, R\.id\.jqList\)/.test(provKt) &&
        /notifyAppWidgetViewDataChanged\(id, R\.id\.amList\)/.test(provKt));
    ok('mục đang mở thì cuộn tới hàng đang hiệu lực',
        /setScrollPosition\(listId, sec\.active\)/.test(provKt));

    // Chạm ngày: 42 ô, mỗi ô một PendingIntent riêng.
    const cellIds = (layXml.match(/@\+id\/cell\d\d/g) || []).length;
    check('lưới bắt chạm đủ 42 ô trong layout', cellIds, 42);
    const cellRefs = (provKt.match(/R\.id\.cell\d\d/g) || []).length;
    check('…và Kotlin trỏ đủ 42 ô ấy', cellRefs, 42);
    ok('chạm ngày là CHỌN ngày, không mở ứng dụng',
        /ACTION_PICK/.test(provKt) && /setSelected\(context, id,/.test(provKt));
    ok('không còn chỗ nào trong widget mở ứng dụng',
        !/MainActivity::class\.java/.test(provKt) && !/EXTRA_TAB/.test(provKt));
    ok('ngày đang chọn có viền riêng, khác hôm nay',
        /isSel/.test(provKt) && /#007BFF/.test(provKt));
    // Lưới bitmap phải luôn 6 hàng, không thì ô chạm lệch khỏi ô nhìn thấy.
    ok('lưới luôn 6 hàng cho khớp lưới bắt chạm',
        /GRID_WEEKS \* 7/.test(provKt) && /GRID_WEEKS = 6/.test(provKt));

    // Ba cột của hàng giá trị phải cùng bộ weight với hàng tiêu đề, và hai mục
    // phải cùng bộ ấy — đó là thứ giữ "Dương lịch" thẳng hàng với "Sóc".
    const weights = x => (x.match(/android:layout_weight="(\d+)"/g) || [])
        .map(w => +/\d+/.exec(w)[0]);
    const rowW = weights(rowXml).slice(0, 3);
    const headW = (layXml.match(/android:layout_weight="(27|35|38)"/g) || [])
        .map(w => +/\d+/.exec(w)[0]);
    ok('hàng giá trị có đúng ba cột theo weight', rowW.length === 3, rowW.join(':'));
    ok('hai hàng tiêu đề dùng ĐÚNG bộ weight ấy',
        headW.length === 6 && headW.slice(0, 3).join() === rowW.join() &&
        headW.slice(3).join() === rowW.join(),
        `hàng ${rowW.join(':')} · tiêu đề ${headW.join(':')}`);
    // Kotlin tính chiều cao bitmap lưới từ chính những con số của XML.
    const dimens = read('app/src/main/res/values/dimens.xml');
    const dimen = n => +(new RegExp(`name="${n}">(\\d+)dp`).exec(dimens) || [])[1];
    const konst = n => +(new RegExp(`${n} = (\\d+)`).exec(supKt) || [])[1];
    for (const [d, k] of [['widget_header', 'HEADER_DP'], ['widget_dow', 'DOW_DP'],
                          ['widget_sec_head', 'SEC_HEAD_DP'], ['widget_row', 'ROW_DP'],
                          ['widget_corner_pad', 'CORNER_PAD_DP']]) {
        check(`Kotlin biết đúng ${d} của XML`, konst(k), dimen(d));
    }
    const blockW = (open, id) => {
        const re = new RegExp(`<${open}[\\s\\S]{0,400}?@\\+id\\/${id}[\\s\\S]{0,400}?android:layout_weight="(\\d+)"`);
        const m = re.exec(layXml);
        return m ? +m[1] : null;
    };
    const frame = /<FrameLayout[\s\S]{0,240}?android:layout_weight="(\d+)"/.exec(layXml);
    check('Kotlin biết đúng phần chia của lưới lịch', konst('W_GRID'), frame ? +frame[1] : null);
    check('…của mục Tiết khí', konst('W_JQ'), blockW('ListView', 'jqList'));
    check('…của mục Lịch âm', konst('W_AM'), blockW('ListView', 'amList'));

    /* ── Báo thức nửa đêm phải sống sót qua reboot ── */
    console.log('\nWidget không kẹt ở ngày cũ');
    const bootKt = (() => { try { return read('app/src/main/java/com/bazi/qimen/BootReceiver.kt'); }
                            catch (e) { return ''; } })();
    ok('có receiver dựng lại báo thức sau khi khởi động máy',
        /ACTION_BOOT_COMPLETED/.test(bootKt) && /reviveNow\(context\)/.test(bootKt));
    ok('…và bắt cả lúc cập nhật ứng dụng lẫn lúc đổi múi giờ',
        /ACTION_MY_PACKAGE_REPLACED/.test(bootKt) && /ACTION_TIMEZONE_CHANGED/.test(bootKt));
    ok('manifest khai receiver ấy, và khai exported (không thì không nhận được)',
        /<receiver[\s\S]{0,200}\.BootReceiver[\s\S]{0,200}android:exported="true"/.test(manifest));
    ok('manifest xin quyền RECEIVE_BOOT_COMPLETED',
        /android\.permission\.RECEIVE_BOOT_COMPLETED/.test(manifest));
    // Khai THẬT, không phải chữ INTERNET trong khối ghi chú ngay đầu manifest.
    ok('vẫn KHÔNG xin quyền INTERNET',
        !/<uses-permission[^>]*android\.permission\.INTERNET/.test(manifest));
    const refreshFn = (/private fun refreshAll\(context: Context\) \{[\s\S]*?\n    \}/.exec(provKt) || [''])[0];
    ok('mỗi lần vẽ lại cũng đặt lại báo thức nửa đêm',
        /scheduleMidnight\(context\)/.test(refreshFn));
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Hai mục của tab Lịch đúng, và widget theo kịp ngôn ngữ');
process.exit(fail ? 1 : 0);
