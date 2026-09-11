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
    const r = await page.evaluate(() => ({
        // Tiêu đề mục CHÍNH LÀ hàng tên cột — không còn nhãn riêng, nên tên
        // "Tiết khí" không bị lặp hai lần như trước.
        dupTitle: !!document.getElementById('calJqTitle'),
        headIsThead: !!document.querySelector('#calJieQi thead tr.cal-sec-head'),
        cols: [...document.querySelectorAll('#calJieQi thead th')]
            .map(x => x.textContent.replace(/[▾▸]/g, '').trim()),
        rows: document.querySelectorAll('#calJqBody tr').length,
        cells: [...document.querySelectorAll('#calJqBody tr')].map(tr => tr.cells.length),
        // Không còn vách ngăn giữa hai nửa bảng — dấu hiệu của bố cục cũ.
        split: document.querySelectorAll('#calJieQi .cal-jq-split').length,
        gz: [...document.querySelectorAll('#calJqBody tr')].map(tr => tr.cells[2].textContent.trim()),
    }));
    ok('không còn nhãn tiêu đề riêng (hết lặp tên)', !r.dupTitle);
    ok('hàng tên cột đóng luôn vai tiêu đề', r.headIsThead);
    check('đúng 24 hàng', r.rows, 24);
    ok('mọi hàng đều 3 ô', r.cells.every(n => n === 3), r.cells.join(','));
    check('ba cột', r.cols.join(' | '), 'Tiết Khí | Dương lịch | Can chi');
    check('không còn chia đôi', r.split, 0);
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
    const cal = await page.evaluate(() => ({
        dupTitle: !!document.getElementById('calAmTitle'),
        headIsThead: !!document.querySelector('#calAmBan thead tr.cal-sec-head'),
        // Tiêu đề Sóc và Vọng phải CĂN GIỮA cột (giá trị vẫn căn trái).
        centred: [...document.querySelectorAll('#calAmBan thead th')]
            .slice(1).every(x => getComputedStyle(x).textAlign === 'center'),
        cols: [...document.querySelectorAll('#calAmBan thead th')]
            .map(x => x.textContent.replace(/[▾▸]/g, '').trim()),
        rows: [...document.querySelectorAll('#calAmBan tbody tr')]
            .map(tr => [...tr.cells].map(c => c.textContent.trim()).join(' | ')),
        active: (document.getElementById('calAmActive') || {}).textContent,
    }));
    ok('không còn nhãn tiêu đề riêng', !cal.dupTitle);
    ok('hàng tên cột đóng luôn vai tiêu đề', cal.headIsThead);
    check('ba cột', cal.cols.join(' | '), 'Tháng âm | Sóc | Vọng');
    ok('tiêu đề Sóc và Vọng căn giữa', cal.centred);
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
    // Ứng dụng mở ra ở tiếng Trung, nên phải sang tiếng Việt TRƯỚC thì lần
    // chọn tiếng Trung sau mới là một thay đổi thật — chọn đúng thứ đang dùng
    // thì setLang() thoát sớm và chẳng có gì để canh.
    await pick('vi');
    const b = await read();
    check('chọn tiếng Việt thì khoá đổi', b.lang, 'vi');
    ok('và widget được bảo vẽ lại', b.n > a.n, `${a.n} → ${b.n}`);

    await pick('zh');
    const c = await read();
    check('chọn tiếng Trung thì khoá đổi', c.lang, 'zh');
    ok('widget lại được bảo vẽ lại', c.n > b.n, `${b.n} → ${c.n}`);
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
    ok('tên tiết khí theo ngôn ngữ', /jieQiYearOf\(.*\bzh\)/.test(provKt));
    ok('can chi theo ngôn ngữ', /ganZhiOf\(jdn, zh\)/.test(provKt));
    ok('tiêu đề hai cột theo ngôn ngữ', /if \(zh\) "节气"/.test(provKt));
    ok('đọc khoá qmdj.lang', /LunarTable\.langOf\(context\)/.test(provKt));

    // Đường báo cho widget vẽ lại.
    ok('cầu native có refreshCalendarWidget', /@JavascriptInterface\s+fun refreshCalendarWidget/.test(bridge));
    ok('provider có refreshNow', /fun refreshNow\(context: Context\)/.test(provKt));
    ok('manifest khai báo WIDGET_REFRESH', /com\.bazi\.qimen\.WIDGET_REFRESH/.test(manifest));
    ok('onReceive xử lý ACTION_REFRESH', /ACTION_REFRESH -> refreshAll/.test(provKt));
    // Không treo toàn bộ việc đồng bộ vào một lời gọi từ trang web.
    ok('rời ứng dụng thì widget vẽ lại', /fun onStop\(\)[\s\S]{0,160}refreshNow\(this\)/.test(mainKt));

    // Phía JavaScript: đổi ngôn ngữ VÀ đổi địa điểm đều phải báo.
    ok('đổi ngôn ngữ thì gọi refreshCalendarWidget',
        /function publishLang\(\)[\s\S]{0,400}refreshCalendarWidget/.test(calJs));
    ok('publishLang được gọi khi đổi nhãn', /\n\s*publishLang\(\);/.test(calJs));
    ok('đổi địa điểm cũng gọi refreshCalendarWidget',
        (calJs.match(/refreshCalendarWidget/g) || []).length >= 2);
}

await browser.close();
server.close();
console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Hai mục của tab Lịch đúng, và widget theo kịp ngôn ngữ');
process.exit(fail ? 1 : 0);
