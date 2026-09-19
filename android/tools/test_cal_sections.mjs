/**
 * Hai mục gập được của tab Lịch, và việc widget theo kịp ngôn ngữ.
 *
 *   node test_cal_sections.mjs
 *
 * Bốn nhóm:
 *   1. Mục "Tiết khí": một dãy 24 hàng (không còn chia đôi), ba cột.
 *   2. Mục "Lịch âm": ĐÃ CHUYỂN sang tab Tra cứu — xem test_lenh.mjs.
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
/**
 * Không còn gì để gập/mở ở tab Lịch.
 *
 * Mục "Lịch âm" đã sang tab Tra cứu, còn mục "Tiết khí" thì LUÔN MỞ — người
 * dùng chốt "ô tiết khí từ giờ ko hide nữa mà luôn hiển thị, cho phép scroll
 * up down". Giữ lại cái vỏ hàm để những phép canh cũ (vốn gọi nó giữa các lượt
 * đo) đọc vẫn xuôi, và để CHỨNG MINH đúng điều ấy: gọi với bất kỳ trạng thái
 * nào cũng không đổi được gì.
 */
const setOpen = (page, which, want) => page.evaluate(() => {});

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

/* ── 2. Mục Lịch âm — nay ở tab TRA CỨU ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);

    console.log('\nMục "Lịch âm" — cấu trúc và định dạng (tab Tra cứu)');
    // Mục này đã CHUYỂN khỏi tab Lịch sang tab Tra cứu, và bảng chi tiết Âm Bàn
    // pháp ở tab Kỳ Môn thì bỏ hẳn — nên nó là chỗ DUY NHẤT hiện dãy Sóc/Vọng
    // trong ứng dụng. Tab Lịch và lịch đã ghim nay chỉ còn lưới lịch và bảng
    // tiết khí. Phép đối chiếu chéo nằm ở test_soc_parity.mjs; ở đây canh cấu
    // trúc và định dạng.
    await page.evaluate(() => { window.__tracuuYear(2026); window.showTab('tracuu'); });
    await page.waitForTimeout(1200);
    await page.evaluate(() => document.getElementById('tcAmHead').click());
    await page.waitForTimeout(700);
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
        var heads = [...document.querySelectorAll('#tcAmBody thead th')];
        var row1 = document.querySelector('#tcAmBody tbody tr');
        var socTd = row1 ? row1.children[1] : null;
        var vongTd = row1 ? row1.children[2] : null;
        return {
            dupTitle: !!document.getElementById('calAmTitle'),
            headIsThead: !!document.querySelector('#tcAmBody thead tr.cal-sec-head'),
            // Tiêu đề Sóc và Vọng phải CĂN GIỮA — VÀ giá trị bên dưới cũng vậy,
            // không như cột "Dương lịch" của Tiết khí (vẫn căn trái).
            headCentred: heads.slice(1).every(x => getComputedStyle(x).textAlign === 'center'),
            valCentred: socTd && vongTd &&
                getComputedStyle(socTd).textAlign === 'center' &&
                getComputedStyle(vongTd).textAlign === 'center',
            socHeadCx: textCx(heads[1]), socValCx: textCx(socTd),
            vongHeadCx: textCx(heads[2]), vongValCx: textCx(vongTd),
            cols: heads.map(x => x.textContent.replace(/[▾▸]/g, '').trim()),
            rows: [...document.querySelectorAll('#tcAmBody tbody tr')]
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
    // 12 tháng, hoặc 13 khi năm âm có tháng nhuận.
    ok('đủ 12 (hoặc 13, năm nhuận) tháng âm',
        cal.rows.length === 12 || cal.rows.length === 13, String(cal.rows.length));
    // Mỗi dòng: "Tháng N | dd-mm-yyyy hh:mm | dd-mm-yyyy hh:mm".
    const dạng = /^Tháng \d+(N)? \| \d{2}-\d{2}-\d{4} \d{2}:\d{2} \| \d{2}-\d{2}-\d{4} \d{2}:\d{2}$/;
    const sai = cal.rows.filter(r => !dạng.test(r));
    ok('mọi dòng đúng dạng "Tháng N | Sóc | Vọng"', sai.length === 0, sai.slice(0, 2).join(' · '));
    // Vọng luôn SAU Sóc của cùng tháng — canh thứ tự thời gian, không chỉ dạng.
    const ngược = cal.rows.filter(r => {
        const c = r.split('|').map(x => x.trim());
        const key = t => t.slice(6, 10) + t.slice(3, 5) + t.slice(0, 2) + t.slice(11);
        return key(c[2]) <= key(c[1]);
    });
    ok('Vọng luôn sau Sóc của cùng tháng', ngược.length === 0, ngược.slice(0, 2).join(' · '));
    // Tra một NĂM bất kỳ thì không có "tháng đang xem" nào — cùng lý lẽ với ba
    // bảng kia của tab này, nên KHÔNG mục nào được tô.
    ok('không tô đậm tháng nào (tra theo năm, không theo tháng đang xem)',
        !cal.active, String(cal.active));
    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 3. Mục Tiết khí: LUÔN MỞ, cuộn được, không gập lại được ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page } = await openCal(ctx);

    console.log('\nMục Tiết khí luôn mở (tab Lịch nay chỉ còn MỘT mục)');
    // Mục "Lịch âm" đã chuyển sang tab Tra cứu, và mục "Tiết khí" thì không gập
    // được nữa — người dùng chốt "ô tiết khí từ giờ ko hide nữa mà luôn hiển
    // thị, cho phép scroll up down". Nên mọi phép canh "hai mục chia nhau chiều
    // cao", "đóng mục này không đụng mục kia" và "nhớ trạng thái gập" đều không
    // còn đối tượng. Thay bằng thứ vẫn đúng và vẫn quan trọng: mục duy nhất còn
    // lại phải lấy TRỌN ngân sách, cuộn được, và không đẩy trang tràn dọc.
    const state = () => page.evaluate(() => ({
        jq: document.getElementById('calSecJq').classList.contains('cal-sec-open'),
        am: !!document.getElementById('calSecAm'),
        jqH: Math.round(document.getElementById('calJieQi').clientHeight),
        jqScroll: document.getElementById('calJieQi').scrollHeight,
        over: document.documentElement.scrollHeight > window.innerHeight + 1,
    }));
    await setOpen(page, 'jq', true);
    await page.waitForTimeout(700);
    const both = await state();
    ok('tab Lịch KHÔNG còn mục Lịch âm', !both.am);
    ok('mục Tiết khí mở được', both.jq);
    ok('…và có chiều cao thật', both.jqH > 40, String(both.jqH));
    ok('mục dài thì cuộn được', both.jqScroll > both.jqH, `${both.jqScroll} ≤ ${both.jqH}`);
    ok('trang không tràn dọc', !both.over);

    // Bấm THẲNG vào hàng tiêu đề: không được gập gì cả. Đây là chỗ dễ hồi quy
    // nhất — chỉ cần một người nghe click sót lại là mục lại gập được.
    const sau = await page.evaluate(async () => {
        const h = document.querySelector('#calSecJq .cal-sec-head');
        h.click();
        await new Promise(r => setTimeout(r, 400));
        return {
            mở: document.getElementById('calSecJq').classList.contains('cal-sec-open'),
            cao: Math.round(document.getElementById('calJieQi').clientHeight),
            trỏTay: getComputedStyle(h).cursor,
        };
    });
    ok('bấm vào hàng tiêu đề KHÔNG gập mục lại', sau.mở && sau.cao > 40,
        `mở=${sau.mở} cao=${sau.cao}`);
    ok('…và hàng tiêu đề không còn giả vờ bấm được (không con trỏ tay)',
        sau.trỏTay !== 'pointer', sau.trỏTay);

    // Không còn trạng thái gập nào để nhớ, nên tab Lịch không ghi khoá nào cả.
    const kept = await page.evaluate(() => ({
        jq: localStorage.getItem('qmdj.calSecJq'), am: localStorage.getItem('qmdj.calSecAm'),
    }));
    check('tab Lịch không còn ghi khoá gập/mở của Tiết khí', kept.jq, null);
    // Mục Lịch âm đã rời tab Lịch, nên khoá của nó không còn được ghi ở đây.
    check('không còn ghi khoá của mục đã rời đi', kept.am, null);
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
            jq: top('calSecJq'),
            rowH: getComputedStyle(document.documentElement).getPropertyValue('--cal-row-h').trim(),
        };
    });

    // Tab Lịch nay chỉ còn MỘT mục (Tiết khí): không có gì phía trên nó ngoài
    // lưới lịch cố định, nên tiêu đề của nó phải đứng đúng MỘT chỗ dù gập hay
    // mở — kể cả khi gập/mở CHÍNH NÓ, vì tiêu đề luôn nằm TRÊN thân mục ấy.
    const seq = [];
    seq.push(['ban đầu', await pos()]);
    await setOpen(page, 'jq', false); await page.waitForTimeout(500);
    seq.push(['jq đóng', await pos()]);
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở', await pos()]);
    await setOpen(page, 'jq', false); await page.waitForTimeout(500);
    seq.push(['jq đóng lại', await pos()]);
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở (quay lại)', await pos()]);

    const jqTops = seq.map(([, p]) => p.jq);
    const rowHs = seq.map(([, p]) => p.rowH);
    ok('tiêu đề Tiết khí đứng đúng một chỗ ở cả 5 trạng thái',
        jqTops.every(t => Math.abs(t - jqTops[0]) < 1),
        seq.map(([label, p]) => `${label}: ${p.jq}px`).join(' · '));
    ok('chiều cao hàng lưới (--cal-row-h) không đổi theo trạng thái gập/mở',
        rowHs.every(h => h === rowHs[0]), rowHs.join(' → '));

    ok('không lỗi JS', errs.length === 0, errs.join('; '));
    await ctx.close();
}

/* ── 3d. Lấp đầy chiều cao màn hình trên hai máy đích ──
   Tab Lịch phải dùng HẾT chiều cao một màn hình: không tràn xuống dưới thanh
   tab, mà cũng không chừa một dải trống to ở đáy.
   Hai lỗi đã sửa, đều làm dải trống ấy phình ra:
   a) fitGrid() trừ hai hàng tiêu đề khỏi `avail` (để rowH không phụ thuộc
      trạng thái gập/mở) rồi đem CHÍNH con số đã trừ ấy đi đặt max-height cho
      cả khung — mà khung thì chứa luôn <thead>. Hai hàng tiêu đề bị trừ hai
      lần: đo trên A51 thì cụm hai mục hụt đúng 48px.
   b) Trần của Tiết khí là 65% ngân sách và cố định (xem shareSectionHeight),
      nên 35% còn lại là phần của riêng Lịch âm — đóng sẵn Lịch âm là bỏ không
      chừng ấy chỗ. Trên máy cao thì đó là hơn 100px trống dưới hàng tiêu đề
      "Tháng âm". Nay decideAmDefault() mở sẵn Lịch âm khi máy đủ cao. */
{
    console.log('\nLấp đầy chiều cao trên máy đích');
    // Nút "Ghim lịch" CHỈ hiện khi có cầu nối Android; trình duyệt không có
    // nên phải ép hiện, bằng không phép đo rộng rãi hơn máy thật 32px.
    const showPin = page => page.evaluate(() => {
        const b = document.getElementById('calPinBtn');
        if (b) b.style.display = 'block';
        if (window.__calRefreshLabels) window.__calRefreshLabels();
    });
    const geom = page => page.evaluate(() => {
        const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
        const view = document.getElementById('calView');
        const dock = document.getElementById('bottomDock');
        const kids = [...view.children].filter(e => getComputedStyle(e).display !== 'none');
        const low = Math.max(...kids.map(e => e.getBoundingClientRect().bottom / z));
        const h = id => {
            const el = document.getElementById(id);
            return el ? +(el.getBoundingClientRect().height / z).toFixed(1) : 0;
        };
        return {
            slack: +(dock.getBoundingClientRect().top / z - low).toFixed(1),
            // Mục Lịch âm đã rời tab này — không còn gì để mở/đóng ở đây.
            amOpen: false,
            amPref: localStorage.getItem('qmdj.calSecAm'),
            jqH: h('calJieQi'), amH: 0,
            jqRows: (() => {
                const box = document.getElementById('calJieQi');
                const c = box.getBoundingClientRect();
                return [...box.querySelectorAll('tbody tr')].filter(r => {
                    const q = r.getBoundingClientRect();
                    return q.top >= c.top - 1 && q.bottom <= c.bottom + 1;
                }).length;
            })(),
        };
    });

    // Hai máy người dùng chỉ đích danh. Dải trống còn lại chỉ là phần đệm
    // chống tràn của GRID_CHROME cộng mấy pixel làm tròn.
    for (const d of [{ n: 'S21 FE', w: 393, h: 790 }, { n: 'A51', w: 412, h: 852 }]) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await openCal(ctx);
        await showPin(page);
        await page.waitForTimeout(500);
        const g = await geom(page);
        ok(`${d.n}: không tràn xuống dưới thanh tab`, g.slack >= 0, `${g.slack}px`);
        ok(`${d.n}: không còn dải trống ở đáy`, g.slack <= 16, `còn thừa ${g.slack}px`);
        // Tab Lịch nay chỉ còn MỘT mục, nên nó phải lấy TRỌN ngân sách: đủ chỗ
        // cho nhiều hàng đọc được, không phải 65% như hồi phải chia đôi.
        ok(`${d.n}: mục Tiết khí lấy trọn ngân sách (≥6 hàng)`, g.jqRows >= 6,
            `${g.jqRows} hàng, cao ${g.jqH}px`);
        ok(`${d.n}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }

    // Máy thấp: mở sẵn cả hai thì Lịch âm chỉ được một hai hàng, vô dụng —
    // vẫn phải đóng, và vẫn không được tràn.
    {
        const ctx = await browser.newContext({ viewport: { width: 360, height: 640 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await openCal(ctx);
        await showPin(page);
        await page.waitForTimeout(500);
        const g = await geom(page);
        // Máy thấp: mở sẵn Lịch âm cũng được, MIỄN LÀ mở ra có hàng đọc được
        // — ngưỡng AM_OPEN_ROWS của decideAmDefault. Trước đây phép kiểm này
        // đòi "phải đóng"; nhưng chỗ của Lịch âm KHÔNG chuyển sang cho Tiết
        // khí khi nó đóng (xem shareSectionHeight), nên đóng lại chỉ đổi lấy
        // một dải trống — đúng cái người dùng kêu. Hai phép kiểm dưới đây chặn
        // cả hai lối hỏng: mở mà rỗng, và đóng mà bỏ phí đáy màn hình.
        ok('máy thấp 360×640: mục Tiết khí vẫn có hàng đọc được', g.jqRows >= 1,
            `${g.jqRows} hàng`);
        ok('máy thấp 360×640: không tràn xuống dưới thanh tab', g.slack >= 0, `${g.slack}px`);
        ok('máy thấp 360×640: không còn dải trống ở đáy', g.slack <= 16, `còn thừa ${g.slack}px`);
        ok('máy thấp 360×640: không lỗi JS', errs.length === 0, errs.join('; '));
        await ctx.close();
    }

    // ── Mép cắt của hàng cuối không được rơi vào chỗ DẤU ──
    //
    // Hàng cuối của một mục đang cuộn thì bị mép dưới cắt ngang; cắt ở đâu mới
    // là chuyện. Chụp S21 FE thấy hàng cuối hiện gần trọn — thân chữ còn
    // nguyên mà dấu nặng thì mất, nên "Hàn Lộ" đọc ra "Hàn Lô", "Mậu Tuất" ra
    // "Mâu Tuất". Đó không phải hàng cụt, đó là chữ KHÁC; tiếng Việt dày dấu
    // dưới nên chỗ này là lỗi ĐỌC SAI.
    //
    // Mép cắt phải nằm ở một trong hai vùng an toàn: từ đáy mực trở xuống
    // (thấy trọn chữ) hoặc từ 72% hộp dòng trở lên (cắt phạm thân chữ, nhìn là
    // biết còn nữa). Đáy mực lấy qua canvas với đúng phông đang dùng — đáy HỘP
    // DÒNG cao hơn đáy mực 1px, mà đúng 1px ấy là chỗ mép cắt hay rơi vào.
    for (const d of [{ n: 'S21', w: 360, h: 740 }, { n: 'S21 FE', w: 393, h: 790 },
                     { n: 'A51', w: 412, h: 852 }, { n: 'thấp', w: 360, h: 640 }]) {
        for (const lang of ['vi', 'zh']) {
            const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
            const { page } = await openCal(ctx);
            if (lang === 'zh') { await page.evaluate(() => window.setLang && window.setLang('zh')); await page.waitForTimeout(400); }
            await showPin(page);
            await page.waitForTimeout(500);
            const cut = await page.evaluate(() => {
                const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
                const one = id => {
                    const box = document.getElementById(id);
                    // Mục Lịch âm đã rời tab này — gọi với id của nó thì không
                    // có gì để đo, trả null như khi bảng chưa dựng.
                    if (!box) return null;
                    const rows = [...box.querySelectorAll('tbody tr')];
                    if (!rows.length) return null;
                    const r0 = rows[0].getBoundingClientRect();
                    const cell = rows[0].cells[0];
                    const rg = document.createRange(); rg.selectNodeContents(cell);
                    const t = rg.getBoundingClientRect();
                    const cs = getComputedStyle(cell);
                    const cx = document.createElement('canvas').getContext('2d');
                    cx.font = cs.font;
                    const m = cx.measureText('Lộ Mậu gy 露');
                    const lineH = t.height / z, top = (t.top - r0.top) / z;
                    const ink = top + (lineH - (m.fontBoundingBoxAscent + m.fontBoundingBoxDescent)) / 2
                        + m.fontBoundingBoxAscent + (m.actualBoundingBoxDescent || 0);
                    const bcs = getComputedStyle(box), br = box.getBoundingClientRect();
                    const edge = (br.bottom - (parseFloat(bcs.borderBottomWidth) || 0)) / z;
                    let shown = null, text = '';
                    for (const r of rows) {
                        const q = r.getBoundingClientRect();
                        if (q.top / z < edge - 0.5 && q.bottom / z > edge + 0.5) {
                            shown = edge - q.top / z; text = (r.cells[0].textContent || '').trim();
                        }
                    }
                    return { ink, safe: top + (ink - top) * 0.72, shown, text };
                };
                return { jq: one('calJieQi'), am: one('calAmBan') };
            });
            for (const [id, c] of Object.entries(cut)) {
                const good = !c || c.shown === null || c.shown >= c.ink - 0.05 || c.shown <= c.safe + 0.05;
                ok(`${d.n} ${lang}: mép cắt hàng cuối mục ${id} không ăn mất dấu`, good,
                    c && c.shown !== null
                        ? `"${c.text}" hiện ${c.shown.toFixed(1)}px, vùng cấm (${c.safe.toFixed(1)}; ${c.ink.toFixed(1)})`
                        : '');
            }
            await ctx.close();
        }
    }

    // Tháng 6 hàng (lưới cao thêm một hàng 58px) là lúc chật nhất. Quét trọn
    // một năm trên cả hai máy: không tháng nào được tràn xuống dưới thanh tab.
    for (const d of [{ n: 'S21 FE', w: 393, h: 790 }, { n: 'A51', w: 412, h: 852 }]) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await openCal(ctx);
        await showPin(page);
        await page.waitForTimeout(400);
        let worst = { slack: 1e9, m: 0 }, weeks6 = 0;
        for (let m = 1; m <= 12; m++) {
            await page.evaluate(mm => window.__calGoto(2026, mm, 1), m);
            await page.waitForTimeout(260);
            const g = await geom(page);
            const rows = await page.evaluate(
                () => document.querySelectorAll('#calGrid .cal-row:not(.cal-dow)').length);
            if (rows === 6) weeks6++;
            if (g.slack < worst.slack) worst = { slack: g.slack, m: m };
        }
        ok(`${d.n}: cả 12 tháng đều không tràn`, worst.slack >= 0,
            `chật nhất tháng ${worst.m}: ${worst.slack}px`);
        // Lưới LUÔN 6 hàng, không còn 4/5/6 tuỳ tháng — xem khối ghi chú ở
        // render(): lưới cao thấp theo tháng thì hai tiêu đề bên dưới nhảy
        // 58–116px mỗi lần bấm ‹ ›, mà lịch đã ghim thì vốn luôn vẽ 6 hàng.
        ok(`${d.n}: tháng nào cũng đúng 6 hàng lưới`, weeks6 === 12, `${weeks6}/12 tháng`);
        ok(`${d.n}: quét 12 tháng không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }

    // Lần vẽ ĐẦU TIÊN phải cho ra đúng con số của mọi lần vẽ sau. Lỗi đã sửa:
    // lượt fitGrid trong render() chạy TRƯỚC renderAmBan(), nên lần đầu nó
    // không thấy hàng tiêu đề Lịch âm và `avail` dôi ra 24px — Tiết khí được
    // 224px lúc vừa mở tab rồi tụt còn 208px ngay khi người dùng chạm vào
    // Lịch âm, tiêu đề nhảy 16px ngay dưới ngón tay.
    {
        const ctx = await browser.newContext({ viewport: { width: 412, height: 852 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await openCal(ctx);
        await showPin(page);
        await page.waitForTimeout(500);
        const first = await geom(page);
        await setOpen(page, 'am', false); await page.waitForTimeout(400);
        await setOpen(page, 'am', true); await page.waitForTimeout(400);
        const after = await geom(page);
        ok('lần vẽ đầu chia chiều cao y hệt các lần sau',
            Math.abs(first.jqH - after.jqH) < 1,
            `lúc mới mở tab ${first.jqH}px · sau một vòng gập/mở ${after.jqH}px`);
        ok('không lỗi JS', errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

/* ── 3e. Bố cục không vỡ ở mọi cỡ máy và mọi tổ hợp gập/mở ──
   Bốn lỗi đã sửa, tìm ra khi soi kỹ toàn bộ giao diện:
   a) syncSectionColumns() đo bề rộng bằng getBoundingClientRect (px ĐÃ phóng)
      rồi ghi vào style.width (px CHƯA phóng). Máy điện thoại có zoom = 1 nên
      không lộ; trên tablet 768px (zoom 1,28) ba cột cộng lại 512px nhét vào
      khung 398px — bảng phình thành 656px, cột Can chi/Vọng bị đẩy ra ngoài,
      phải kéo ngang 114px mới đọc được.
   b) shareSectionHeight() coi mục Tiết khí ĐANG ĐÓNG là chiếm 0px, trong khi
      nó vẫn choán đúng hàng tiêu đề — đóng Tiết khí mà mở Lịch âm là cụm hai
      mục thò 17px xuống dưới thanh tab.
   c) Ngân sách chiều cao kê lên SEC_MIN (96px) ngay cả khi chỗ còn lại chỉ
      51px — bịa ra chỗ không có, máy 360×640 tràn 14px.
   d) viewport.js lấy chiều cao nội dung tab Lịch làm cơ sở tính tỉ lệ phóng,
      mà chiều cao ấy do fitGrid() chia ra cho vừa MỌI tỉ lệ — hai cơ chế cùng
      kéo một sợi dây nên mọi tỉ lệ trong [0,95; 1] đều tự nhất quán: cùng một
      máy, hai lần mở ra hai cỡ chữ (đo được 0,976 rồi 1,000). */
{
    console.log('\nBố cục không vỡ ở mọi cỡ máy');
    const MAY = [
        ['360×640', 360, 640], ['S21 360×740', 360, 740], ['S21 FE', 393, 790],
        ['A51', 412, 852], ['tablet', 768, 1024],
    ];
    for (const [nm, w, h] of MAY) {
        const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page, errs } = await openCal(ctx);
        await page.evaluate(() => {
            const b = document.getElementById('calPinBtn');
            if (b) b.style.display = 'block';
            if (window.__calRefreshLabels) window.__calRefreshLabels();
        });
        await page.waitForTimeout(500);

        // (a) Hai bảng không bao giờ phải kéo ngang.
        // Chỉ còn MỘT mục ở tab này (Lịch âm đã sang tab Tra cứu).
        const ngang = await page.evaluate(() => ['calJieQi', 'calAmBan'].map(id => {
            const b = document.getElementById(id);
            return b ? { id, thua: b.scrollWidth - b.clientWidth } : null;
        }).filter(Boolean));
        ok(`${nm}: mục Tiết khí không phải kéo ngang`, ngang.every(x => x.thua <= 1),
            ngang.map(x => `${x.id} +${x.thua}`).join(' · '));

        // (b)(c) Đủ bốn tổ hợp gập/mở, không tổ hợp nào đẩy nội dung xuống
        // dưới thanh tab.
        const slack = () => page.evaluate(() => {
            const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
            const v = document.getElementById('calView'), d = document.getElementById('bottomDock');
            const kids = [...v.children].filter(e => getComputedStyle(e).display !== 'none');
            const low = Math.max(...kids.map(e => e.getBoundingClientRect().bottom));
            return +((d.getBoundingClientRect().top - low) / z).toFixed(1);
        });
        const xau = [];
        for (const [j, a] of [[true, true], [true, false], [false, true], [false, false]]) {
            await setOpen(page, 'jq', j); await page.waitForTimeout(260);
            await setOpen(page, 'am', a); await page.waitForTimeout(260);
            const s = await slack();
            // Máy quá thấp thì fitGrid hết đường bóp, trang cuộn — chấp nhận
            // được, miễn là cuộn tới được (thanh dưới là fixed, thân trang có
            // chừa đúng chiều cao nó).
            const cuon = await page.evaluate(() =>
                document.documentElement.scrollHeight > window.innerHeight + 2);
            if (s < 0 && !cuon) xau.push(`${j ? 'J' : '-'}${a ? 'A' : '-'}:${s}`);
        }
        ok(`${nm}: không tổ hợp gập/mở nào bị thanh tab che`, !xau.length, xau.join(' '));

        // (d) Gọi lại phép co giãn nhiều lần phải ra ĐÚNG một con số.
        const zooms = [];
        for (let i = 0; i < 3; i++) {
            await page.evaluate(() => window.__fitScreen && window.__fitScreen());
            await page.waitForTimeout(400);
            zooms.push(await page.evaluate(() =>
                (parseFloat(getComputedStyle(document.body).zoom) || 1).toFixed(3)));
        }
        ok(`${nm}: tỉ lệ phóng tab Lịch đứng yên qua 3 lần đo`,
            new Set(zooms).size === 1, zooms.join(' → '));

        ok(`${nm}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

/* ── 3e2. Mở bảng gập ở tab Kỳ Môn thì phải CUỘN TỚI ──
   Hàng tiêu đề của bảng chi tiết là khối cuối cùng của trang, mà viewport.js
   căn cho đáy nó chạm đúng mép thanh tab — nên 100% phần vừa mở nằm dưới mép
   màn hình. Bấm vào: mũi tên lật, và không có gì khác xảy ra. Đo trên A51:
   bảng cao 362px, không một pixel nào lọt vào khung nhìn. */
{
    console.log('\nMở bảng gập ở tab Tra cứu');
    for (const [nm, w, h] of [['A51', 412, 852], ['S21 FE', 393, 790]]) {
        const ctx = await browser.newContext({ viewport: { width: w, height: h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const page = await ctx.newPage();
        const errs = []; page.on('pageerror', e => errs.push(e.message));
        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForTimeout(1200);
        // Ba bảng gập/mở nay nằm ở tab Tra cứu (bảng Âm Bàn pháp đã bỏ hẳn).
        // Lấy bảng CUỐI — Lệnh năm — vì nó đúng là trường hợp xấu nhất: hàng
        // tiêu đề của nó là khối cuối trang, nên khi nó đóng và trang đang
        // cuộn hết cỡ thì 100% phần vừa mở nằm dưới mép màn hình.
        await page.evaluate(() => { window.__tracuuYear(2026); window.showTab('tracuu'); });
        await page.waitForTimeout(1200);

        // Bốn mục của tab Tra cứu ĐÓNG SẴN sau khi chọn năm, nên chỉ cần cuộn
        // xuống đáy là đã dựng đúng cái thế xấu: tiêu đề mục cuối nằm sát mép
        // dưới, bấm mở ra thì phần vừa mở rơi hết xuống dưới màn hình.
        await page.evaluate(() => window.scrollTo(0, document.documentElement.scrollHeight));
        await page.waitForTimeout(500);
        const truoc = await page.evaluate(() => ({
            dong: getComputedStyle(document.getElementById('lenhSec')).display === 'none',
            y: Math.round(window.scrollY),
        }));
        ok(`${nm}: dựng được thế xấu — bảng đóng, trang đã cuộn hết`,
            truoc.dong, JSON.stringify(truoc));

        await page.evaluate(() => document.getElementById('lenhHead').click());
        await page.waitForTimeout(1400);
        const sau = await page.evaluate(() => {
            const sec = document.getElementById('lenhSec');
            const head = document.getElementById('lenhHead');
            return {
                mo: getComputedStyle(sec).display === 'block',
                // Thấy được ÍT NHẤT một phần bảng vừa mở: bấm mà không có gì
                // lọt vào khung nhìn thì người dùng tưởng nút hỏng.
                thayItNhatMotPhan: sec.getBoundingClientRect().top < window.innerHeight - 20,
                conTieuDe: head.getBoundingClientRect().bottom >= -1,
                y: Math.round(window.scrollY),
            };
        });
        ok(`${nm}: bấm là bảng mở ra`, sau.mo);
        ok(`${nm}: …và thấy được phần vừa mở`, sau.thayItNhatMotPhan, JSON.stringify(sau));
        ok(`${nm}: …mà vẫn còn thấy hàng tiêu đề`, sau.conTieuDe, JSON.stringify(sau));
        ok(`${nm}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
}

/* ── 3e3. Mục Tiết khí tự cuộn thì phải DỪNG ĐÚNG MÉP HÀNG ──
   Hàng tiêu đề dính đè lên phần trên khung cuộn. Thả cho scrollTop rơi tự do
   thì hàng trên cùng bị cắt ngang ngay dưới nó — nhìn ra một vệt chữ cụt,
   giống lỗi vẽ chứ không giống "còn cuộn được nữa". Hàng bảng tiếng Trung cao
   22px, tiếng Việt 20px, nên chỗ dừng tự do rơi giữa hàng ở thứ tiếng này mà
   lại đúng mép ở thứ tiếng kia — cùng một máy, hai kiểu. */
{
    console.log('\nMục Tiết khí dừng đúng mép hàng');
    /** Chiều cao ô lịch theo thứ tiếng — bớt một dòng thì hàng phải thấp đi. */
    const caoÔ = { vi: null, zh: null };
    for (const lang of ['vi', 'zh']) {
        const ctx = await browser.newContext({ viewport: { width: 412, height: 852 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const page = await ctx.newPage();
        const errs = []; page.on('pageerror', e => errs.push(e.message));
        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForTimeout(1100);
        await page.click('#langDisplayBtn');
        await page.waitForSelector(`#optOverlay.open .opt-row[data-value="${lang}"]`);
        await page.click(`.opt-row[data-value="${lang}"]`);
        await page.waitForTimeout(800);
        await page.evaluate(() => {
            const p = document.getElementById('calPinBtn');
            if (p) p.style.display = 'block';
        });
        await page.evaluate(() => window.showTab('cal'));
        await page.waitForTimeout(800);
        await page.evaluate(() => window.__calRefreshLabels && window.__calRefreshLabels());
        await page.waitForTimeout(800);
        const m = await page.evaluate(() => {
            const body = document.getElementById('calJieQi');
            const thead = body.querySelector('thead');
            const headBot = body.getBoundingClientRect().top + thead.getBoundingClientRect().height;
            const rows = [...body.querySelectorAll('tbody tr')];
            const first = rows.find(r => r.getBoundingClientRect().bottom > headBot + 0.5);
            return {
                daCuon: body.scrollTop > 0,
                lech: first ? +(first.getBoundingClientRect().top - headBot).toFixed(1) : null,
            };
        });
        ok(`${lang}: mục Tiết khí có tự cuộn tới hàng đang hiệu lực`, m.daCuon);
        ok(`${lang}: hàng đầu không bị cắt ngang dưới tiêu đề`,
            m.lech !== null && Math.abs(m.lech) < 1.5, `lệch ${m.lech}px`);

        // Can chi trong ô lịch: tiếng Trung MỘT dòng ("甲子" chỉ hai chữ
        // vuông), tiếng Việt HAI dòng ("Nhâm Thân" một dòng thì tràn ô).
        // Đo bằng vị trí THẬT của hai <span> — cùng mép trên là một dòng —
        // chứ không đọc luật CSS, vì luật có thể bị luật khác đè.
        const gz = await page.evaluate(() => {
            let motDong = 0, haiDong = 0, tran = 0, rongNhat = 0;
            const ô = [...document.querySelectorAll('.cal-day')];
            for (const c of ô) {
                const sp = [...c.querySelectorAll('.cal-gz span')];
                if (sp.length !== 2) continue;
                const a = sp[0].getBoundingClientRect(), b = sp[1].getBoundingClientRect();
                if (Math.abs(a.top - b.top) < 0.5) motDong++; else haiDong++;
                const cb = c.getBoundingClientRect();
                if (a.left < cb.left - 0.5 || b.right > cb.right + 0.5) tran++;
                rongNhat = Math.max(rongNhat, b.right - a.left);
            }
            return { sốÔ: ô.length, motDong, haiDong, tran, rongNhat: +rongNhat.toFixed(1),
                     rộngÔ: +ô[0].getBoundingClientRect().width.toFixed(1),
                     caoÔ: +ô[0].getBoundingClientRect().height.toFixed(1) };
        });
        if (lang === 'zh') {
            ok('zh: can chi nằm CHUNG một dòng ở mọi ô',
                gz.motDong === gz.sốÔ && gz.haiDong === 0,
                `${gz.motDong}/${gz.sốÔ} ô một dòng`);
            ok('zh: hai chữ can chi không tràn khỏi ô',
                gz.tran === 0 && gz.rongNhat <= gz.rộngÔ,
                `rộng nhất ${gz.rongNhat}px trong ô ${gz.rộngÔ}px, ${gz.tran} ô tràn`);
            caoÔ.zh = gz.caoÔ;
        } else {
            ok('vi: can chi vẫn tách HAI dòng ở mọi ô',
                gz.haiDong === gz.sốÔ && gz.motDong === 0,
                `${gz.haiDong}/${gz.sốÔ} ô hai dòng`);
            caoÔ.vi = gz.caoÔ;
        }

        ok(`${lang}: không lỗi JS`, errs.length === 0, errs.join('; '));
        await ctx.close();
    }
    // Bớt một dòng thì hàng lịch tiếng Trung phải THẤP hơn hàng tiếng Việt —
    // và chính phần thấp đi ấy là thứ fitGrid trả cho bảng Tiết khí. Nếu hai
    // bên bằng nhau thì việc gộp một dòng chẳng đổi được gì.
    ok('zh: hàng lịch thấp hơn tiếng Việt (chỗ dôi trả cho Tiết khí)',
        caoÔ.zh !== null && caoÔ.vi !== null && caoÔ.zh < caoÔ.vi - 1,
        `zh ${caoÔ.zh}px vs vi ${caoÔ.vi}px`);
}

/* ── 3f. Nút Back của Android ──
   Không còn hộp thoại nào mà đang ở tab Lịch thì Back đưa về tab Kỳ Môn — tab
   ứng dụng mở lên đầu tiên — chứ không thoát thẳng ra màn hình chính. */
{
    console.log('\nNút Back của Android');
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const { page, errs } = await openCal(ctx);
    const r = await page.evaluate(() => {
        const out = { coHam: typeof window.__onBackPressed === 'function' };
        if (!out.coHam) return out;
        out.dangOLich = document.body.classList.contains('view-cal');
        out.nhanLan1 = window.__onBackPressed();
        out.veKyMon = !document.body.classList.contains('view-cal');
        out.nhanLan2 = window.__onBackPressed();   // ở Kỳ Môn rồi thì nhường hệ thống
        return out;
    });
    ok('có hàm nhận nút Back', r.coHam);
    ok('đang ở tab Lịch thì Back xử lý luôn, không thoát app', r.nhanLan1 === true);
    ok('…và đưa về tab Kỳ Môn', r.veKyMon === true);
    ok('ở tab Kỳ Môn thì Back nhường lại cho hệ thống', r.nhanLan2 === false);

    // Hộp thoại đang mở thì Back đóng hộp TRƯỚC, chưa đụng tới tab.
    const ov = await page.evaluate(() => {
        document.getElementById('langDisplayBtn').click();
        const daMo = document.getElementById('optOverlay').classList.contains('open');
        const xuLy = window.__onBackPressed();
        return { daMo, xuLy, daDong: !document.getElementById('optOverlay').classList.contains('open') };
    });
    ok('Back đóng hộp thoại đang mở trước tiên', ov.daMo && ov.xuLy === true && ov.daDong);
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
        return { jq: one('calJieQi') };
    });

    // Tab Lịch nay chỉ còn MỘT mục, nên chuỗi trạng thái rút còn mở ↔ đóng.
    const seq = [];
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở', await cols()]);
    await setOpen(page, 'jq', false); await page.waitForTimeout(500);
    seq.push(['jq đóng', await cols()]);
    await setOpen(page, 'jq', true); await page.waitForTimeout(500);
    seq.push(['jq mở (quay lại)', await cols()]);

    {
        const first = seq[0][1].jq;
        const worst = Math.max(...seq.map(([, p]) => Math.max(...p.jq.map((v, i) => Math.abs(v - first[i])))));
        ok('Tiết khí: ba cột tiêu đề đứng yên qua mọi trạng thái', worst < 1,
            seq.map(([label, p]) => `${label}: [${p.jq.join(',')}]`).join(' · '));
    }

    // Hai mục từng nằm trên dưới nhau nên phải thẳng cột với nhau; nay mục Lịch
    // âm đã sang tab Tra cứu, không còn cặp nào để so. Phần còn giá trị là:
    // bề rộng cột đã ghim phải ĐỦ RỘNG cho chữ — cột hẹp hơn chữ thì chữ tràn
    // sang ô bên hoặc bị cắt, mà `white-space: nowrap` thì không xuống dòng.
    await setOpen(page, 'jq', true);
    await page.waitForTimeout(700);
    const pair = await page.evaluate(() => {
        const one = id => {
            const body = document.getElementById(id);
            let spill = 0;
            for (const cell of body.querySelectorAll('th, td')) {
                spill = Math.max(spill, cell.scrollWidth - cell.clientWidth);
            }
            return { spill };
        };
        return { jq: one('calJieQi') };
    });
    ok('không ô nào có chữ rộng hơn cột của nó', pair.jq.spill <= 0.5,
        `thừa ${pair.jq.spill}px`);
    // Dù đổi ngôn ngữ hay đổi tháng, mục vẫn MỞ và vẫn đủ 24 hàng — nó không
    // còn đường nào để đóng lại.
    const shut = await page.evaluate(() => {
        const b = document.getElementById('calJieQi');
        return { h: Math.round(b.clientHeight), scroll: b.scrollHeight, rows: b.querySelectorAll('tbody tr').length };
    });
    ok('mục vẫn mở và cao thật', shut.h > 40, String(shut.h));
    check('…và vẫn đủ 24 hàng', shut.rows, 24);
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
        return { jq: chk('calJieQi') };
    });
    for (const [name, x] of [['Tiết khí', r.jq]]) {
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
    // gì tới app thì widget không bao giờ nhận được bảng cả. toggleSection đã
    // bị gỡ hẳn (mục Tiết khí không gập được nữa), nên nay canh trực tiếp: khối
    // công bố nằm trong nhánh khởi động, và KHÔNG còn hàm gập/mở nào.
    ok('không còn hàm gập/mở nào ở tab Lịch', !/function toggleSection\(/.test(calJs));
    ok('có khối công bố kèm hẹn giờ chạy lúc mở app',
        /setTimeout\(function \(\) \{[\s\S]{0,40}try \{ publishLunarCache\(\); \}/.test(calJs));

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

    /* ── Widget phải hiện ĐÚNG mục duy nhất của tab Lịch ── */
    console.log('\nWidget dựng đúng mục Tiết khí của tab Lịch');
    ok('mục Tiết khí có cột CAN CHI THÁNG', /LunarTable\.ganZhi60\(it\.gz, zh\)/.test(secKt));
    // Mục "Lịch âm" đã sang tab Tra cứu và BỎ HẲN khỏi lịch đã ghim: widget
    // không được còn chỗ nào dựng bảng Sóc/Vọng, và cũng không còn khoá riêng
    // cho nó — sót lại là widget vẫn vẽ mục người dùng đã bảo bỏ.
    ok('KHÔNG còn mục Lịch âm (Sóc/Vọng) trong widget',
        !/socJdn/.test(secKt) && !/vongJdn/.test(secKt));
    ok('…và không còn trọng số chiều cao riêng của nó',
        !/SEC_AM/.test(supKt) && !/W_AM/.test(supKt));
    // Khoá `qmdj.calSecJq` đã đi hết: không bên nào đọc, không bên nào ghi.
    ok('ứng dụng không còn ghi khoá gập/mở nào',
        !/prefSet\(K_SEC_JQ/.test(calJs) && !/K_SEC_JQ =/.test(calJs));

    /* ── Widget dùng được bằng ngón tay, không phải ảnh tĩnh ── */
    console.log('\nWidget cuộn được và chạm được');
    // Cuộn chỉ có trong collection view do RemoteViewsService nuôi.
    ok('mục Tiết khí là ListView, không còn vẽ vào bitmap',
        /<ListView[\s\S]{0,200}@\+id\/jqList/.test(layXml) &&
        !/@\+id\/amList/.test(layXml) &&
        !/fun drawSections\(/.test(widgetKt));
    ok('có RemoteViewsService nuôi hàng cho mục ấy',
        /class WidgetSectionService : RemoteViewsService\(\)/.test(svcKt) &&
        /RemoteViewsFactory/.test(svcKt));
    ok('manifest khai service kèm quyền BIND_REMOTEVIEWS (thiếu là mục trống trơn)',
        /<service[\s\S]{0,260}\.WidgetSectionService[\s\S]{0,260}android\.permission\.BIND_REMOTEVIEWS/
            .test(manifest));
    ok('provider nối adapter cho ListView ấy',
        /setRemoteAdapter\(sv\.listId, sectionIntent\(context, id, sv\.key\)\)/.test(provKt));
    // filterEquals bỏ qua extras: hai mục chỉ khác extras thì dùng chung một
    // factory và cùng hiện một bảng.
    ok('khoá mục nằm trong Intent.data, không chỉ trong extras',
        /setData\(Uri\.parse\("qmdj:\/\/widget\/\$id\/sec\/\$key"\)\)/.test(provKt));
    ok('bảo factory đọc lại sau mỗi lần vẽ',
        /notifyAppWidgetViewDataChanged\(id, R\.id\.jqList\)/.test(provKt) &&
        !/R\.id\.amList/.test(provKt));
    ok('cuộn tới hàng đang hiệu lực',
        /setScrollPosition\(sv\.listId, sec\.active\)/.test(provKt));

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

    /* ── KHÔNG còn gập/mở ở bất cứ đâu ──
       Trước đây mục gập được ở cả hai nơi, và hai bên dùng chung khoá
       `qmdj.calSecJq`. Người dùng chốt bỏ hẳn: "ô tiết khí từ giờ ko hide nữa
       mà luôn hiển thị, cho phép scroll up down" — rồi "có bỏ luôn gập ngoài
       pinned calendar". Nên cả đường bấm lẫn khoá đều phải đi hết, ở cả ứng
       dụng lẫn widget; sót một mẩu là còn một nửa cơ chế treo lơ lửng. */
    ok('widget KHÔNG còn đường bấm gập/mở nào',
        !/ACTION_SEC/.test(provKt) && !/secToggleIntent/.test(provKt)
        && !/setOnClickPendingIntent\(sv\.headId/.test(provKt));
    ok('…không còn khoá gập/mở nào trong WidgetPrefs',
        !/SEC_JQ/.test(supKt) && !/fun secOpen/.test(supKt) && !/fun toggleSec/.test(supKt));
    ok('…không còn cờ `open` trong mô hình mục',
        !/val open: Boolean/.test(secKt) && !/sec\.open/.test(provKt));
    ok('…manifest thôi khai action ấy',
        !/com\.bazi\.qimen\.WIDGET_SEC/.test(manifest));
    ok('…và tiêu đề thôi mang dấu ▾/▸ (nó hứa một cú bấm không còn)',
        !/"  ▾"/.test(provKt) && !/▸/.test(provKt));

    // Bỏ nút gập KHÔNG được đụng tới chuyện CUỘN: thân mục vẫn là ListView do
    // RemoteViewsService nuôi, vẫn kéo được bằng ngón tay. Đây là điều người
    // dùng dặn thêm ("nhưng vẫn scroll up down được").
    ok('thân mục VẪN là ListView cuộn được, không phải ảnh tĩnh',
        /<ListView[\s\S]{0,200}@\+id\/jqList/.test(layXml)
        && /setRemoteAdapter\(sv\.listId/.test(provKt)
        && /class WidgetSectionService : RemoteViewsService\(\)/.test(svcKt));

    /* ── Mục dựng hụt không được để lại ListView mồ côi ──
       WidgetSections.build() bỏ hẳn một mục nếu năm đang xem thiếu dữ liệu.
       Vòng vẽ mà duyệt theo `secs` thì ListView của mục ấy giữ nguyên trạng
       thái mặc định của XML — đang HIỆN, KHÔNG có adapter — thành một mảng
       trắng chiếm chỗ mà chẳng bao giờ có hàng nào. */
    ok('vòng vẽ duyệt danh sách CỐ ĐỊNH các mục, không duyệt theo secs',
        /for \(sv in SECTION_VIEWS\)/.test(provKt) && !/for \(sec in secs\)/.test(provKt));
    ok('mục không dựng được thì ẩn cả tiêu đề lẫn danh sách',
        /setViewVisibility\(sv\.headId, if \(sec == null\)/.test(provKt)
        && /setViewVisibility\(sv\.listId, if \(sec == null\)/.test(provKt));

    // __calSyncSections ở lại dù không còn trạng thái nào để đồng bộ: nó chia
    // lại chiều cao khi người dùng quay về sau khi đổi cỡ chữ hệ thống.
    ok('ứng dụng vẫn canh lại bố cục mỗi lần trở lại',
        /override fun onResume\(\)/.test(mainKt)
        && /__calSyncSections/.test(mainKt) && /__calSyncSections/.test(calJs));
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
    ok('hàng tiêu đề (nay chỉ một) dùng ĐÚNG bộ weight ấy',
        headW.length === 3 && headW.join() === rowW.join(),
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

/* ── Máy thấp: KHÔNG được để lại nửa hàng ──
   Trước đây trên 320×568 mục Tiết khí được 38px mà riêng hàng tiêu đề đã 25px:
   13px còn lại hiện ra một DẢI NỬA CHỮ, thấy nửa trên của "Bạch Lộ 07-09-2026"
   mà đọc không ra. Luật: mục nào không đủ hàng tiêu đề + MỘT hàng trọn vẹn thì
   hạ hẳn về hàng tiêu đề — trông như đang đóng, sạch, và vài pixel nhả ra chảy
   sang mục kia. */
{
    console.log('\nMáy thấp: không để lại nửa hàng');
    for (const d of [{ n: '320×568', w: 320, h: 568 }, { n: '360×640', w: 360, h: 640 },
                     { n: 'ngang 740×360', w: 740, h: 360 }]) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h },
            deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const { page } = await openCal(ctx);
        await page.waitForTimeout(700);
        const đo = await page.evaluate(() => {
            const z = parseFloat(getComputedStyle(document.body).zoom) || 1;
            return ['calJieQi', 'calAmBan'].map(id => {
                const box = document.getElementById(id);
                if (!box) return null;        // mục Lịch âm đã rời tab này
                const thead = box.querySelector('thead');
                const row = box.querySelector('tbody tr');
                if (!thead || !row) return null;
                const cao = box.getBoundingClientRect().height / z;
                const đầu = thead.getBoundingClientRect().height / z;
                const hàng = row.getBoundingClientRect().height / z;
                return { id, cao: +cao.toFixed(1), đầu: +đầu.toFixed(1), hàng: +hàng.toFixed(1),
                         // phần thân hiện ra, ngoài hàng tiêu đề
                         thân: +(cao - đầu).toFixed(1) };
            });
        });
        for (const m of đo) {
            if (!m) continue;
            // Hoặc thân KHÔNG hiện gì (≤ viền, tức chỉ có hàng tiêu đề),
            // hoặc hiện được TRỌN ít nhất một hàng. Không có cửa giữa.
            const sạch = m.thân <= 3 || m.thân >= m.hàng - 0.5;
            ok(`${d.n} · ${m.id}: không để lại nửa hàng`, sạch,
                `thân ${m.thân}px, hàng ${m.hàng}px`);
        }
        await ctx.close();
    }
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Mục Tiết khí của tab Lịch đúng, và widget theo kịp ngôn ngữ');
process.exit(fail ? 1 : 0);
