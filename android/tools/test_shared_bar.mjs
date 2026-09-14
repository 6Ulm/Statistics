/**
 * Hàng dùng chung dưới hai tab: ngôn ngữ + địa điểm.
 *
 *   node test_shared_bar.mjs
 *
 * Canh ba điều mà mắt thường dễ bỏ sót:
 *   1. VỊ TRÍ  — ba ô nằm DƯỚI hai tab, trên cùng một hàng, ở CẢ hai tab.
 *   2. VỪA KHÍT — không ô nào bị cắt chữ, trên máy hẹp nhất và ở cả hai
 *      ngôn ngữ (nhãn tiếng Việt dài gần gấp đôi nhãn tiếng Trung).
 *   3. ĐỒNG BỘ  — đổi ở tab này thì tab kia đổi theo, không phải mở lại.
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
/** Chọn một mục qua bảng chọn — thay cho việc bấm thẳng vào nút như trước. */
async function pick(page, boxId, value) {
    await page.click('#' + boxId);
    await page.waitForSelector('#optOverlay.open .opt-row[data-value="' + value + '"]', { timeout: 4000 });
    await page.click('.opt-row[data-value="' + value + '"]');
    await page.waitForTimeout(700);
}

function ok(what, cond, detail) {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(46)} ${cond ? '' : (detail || '')}`);
}

/* ── 1. Cấu trúc: hàng dùng chung nằm DƯỚI hai tab, trong cùng thanh dưới ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);

    console.log('\nCấu trúc thanh dưới');
    const st = await page.evaluate(() => {
        const dock = document.getElementById('bottomDock');
        const bar = document.getElementById('tabBar');
        const shared = document.getElementById('sharedBar');
        return {
            dockIsBodyChild: dock && dock.parentElement === document.body,
            tabInDock: bar && bar.parentElement === dock,
            sharedInDock: shared && shared.parentElement === dock,
            sharedBelowTabs: shared.getBoundingClientRect().top >= bar.getBoundingClientRect().bottom - 1,
            dockFixed: getComputedStyle(dock).position,
            // Hàng điều khiển của tab Kỳ Môn giờ chỉ còn ĐÚNG một hàng.
            controlRows: document.querySelectorAll('.controls .control-item').length,
        };
    });
    ok('#bottomDock là con trực tiếp của body', st.dockIsBodyChild);
    ok('#tabBar nằm trong #bottomDock', st.tabInDock);
    ok('#sharedBar nằm trong #bottomDock', st.sharedInDock);
    ok('hàng dùng chung nằm DƯỚI hai tab', st.sharedBelowTabs);
    check('thanh dưới vẫn dính đáy', st.dockFixed, 'fixed');
    check('tab Kỳ Môn chỉ còn MỘT hàng điều khiển', st.controlRows, 1);
    await ctx.close();
}

/* ── 2. Vừa khít: không ô nào bị cắt chữ, hai ngôn ngữ × ba máy ── */
const DEVICES = [
    { name: 'S21',    w: 360, h: 740 },
    { name: 'S21 FE', w: 393, h: 790 },
    { name: 'A51',    w: 412, h: 852 },
];
console.log('\nVừa khít trên từng máy (không cắt chữ, đúng một hàng)');
for (const d of DEVICES) {
    for (const lang of ['zh', 'vi']) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const page = await ctx.newPage();
        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForTimeout(800);
        await pick(page, 'langDisplayBtn', lang);

        const m = await page.evaluate(() => {
            const clipped = [];
            const ids = ['dateDisplayBtn', 'methodDisplayBtn', 'cobanToggleWrap',
                         'langDisplayBtn', 'countryDisplayBtn',
                         'lblHienThiCoBan', 'dateDisplayText',
                         'countryDisplayText', 'methodDisplayText', 'langDisplayText'];
            for (const id of ids) {
                const e = document.getElementById(id);
                if (!e) { clipped.push(id + '(thiếu)'); continue; }
                if (e.scrollWidth > e.clientWidth + 1) clipped.push(id);
            }
            const oneRow = el => {
                const kids = [...el.children];
                if (!kids.length) return false;
                // Cùng một hàng = các ô chồng lấn nhau theo chiều dọc.
                const first = kids[0].getBoundingClientRect();
                return kids.every(k => {
                    const r = k.getBoundingClientRect();
                    return r.top < first.bottom && r.bottom > first.top;
                });
            };
            const row = document.getElementById('qmRow');
            const shared = document.getElementById('sharedBar');
            return {
                clipped,
                qmOneRow: oneRow(row), qmKids: row.children.length,
                sharedOneRow: oneRow(shared), sharedKids: shared.children.length,
                bodyOverflow: document.documentElement.scrollWidth > document.documentElement.clientWidth + 1,
            };
        });
        const tag = `${d.name} ${d.w}px · ${lang}`;
        ok(`${tag}: không ô nào bị cắt chữ`, m.clipped.length === 0, 'bị cắt: ' + m.clipped.join(', '));
        ok(`${tag}: hàng Kỳ Môn đúng 1 hàng, 3 ô`, m.qmOneRow && m.qmKids === 3, `${m.qmKids} ô, cùng hàng=${m.qmOneRow}`);
        ok(`${tag}: hàng dùng chung đúng 1 hàng, 2 ô`, m.sharedOneRow && m.sharedKids === 2, `${m.sharedKids} ô`);
        ok(`${tag}: trang không tràn ngang`, !m.bodyOverflow);
        await ctx.close();
    }
}

/* ── 3. Hiện ở CẢ hai tab, và không đè lên nội dung ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);

    console.log('\nHiện ở cả hai tab');
    const seen = async () => page.evaluate(() => {
        const s = document.getElementById('sharedBar');
        const r = s.getBoundingClientRect();
        return { vis: getComputedStyle(s).display !== 'none' && r.height > 0, top: r.top };
    });
    const a = await seen();
    ok('tab Kỳ Môn: hàng dùng chung hiện', a.vis);
    await page.click('#tabCal'); await page.waitForTimeout(700);
    const b = await seen();
    ok('tab Lịch: hàng dùng chung hiện', b.vis);
    // Không so toạ độ tuyệt đối: viewport.js đặt zoom riêng cho từng tab nên
    // thanh cao lên/thấp xuống vài pixel. Điều phải đúng là nó DÍNH ĐÁY.
    const docked = await page.evaluate(() => {
        const r = document.getElementById('bottomDock').getBoundingClientRect();
        return Math.abs(r.bottom - window.innerHeight);
    });
    ok('thanh dưới vẫn dính đáy ở tab Lịch', docked < 2, `lệch ${docked.toFixed(1)}px`);

    // Thanh dưới che đáy màn hình — nội dung phải dừng TRƯỚC nó.
    const clear = await page.evaluate(() => {
        const dockTop = document.getElementById('bottomDock').getBoundingClientRect().top;
        const cal = document.getElementById('calView').getBoundingClientRect().bottom;
        return { dockTop, cal, gap: dockTop - cal };
    });
    ok('nội dung tab Lịch không chui xuống dưới thanh', clear.gap >= -1, `dôi ${(-clear.gap).toFixed(1)}px`);
    await ctx.close();
}

/* ── 4. Đồng bộ: đổi ở tab Lịch thì tab Kỳ Môn đổi theo ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);

    console.log('\nĐồng bộ hai tab: NGÔN NGỮ đổi từ tab Lịch');
    await pick(page, 'langDisplayBtn', 'zh');
    await page.click('#tabCal');     await page.waitForTimeout(700);
    check('lịch đang là tiếng Trung', await page.textContent('#tabCal .tab-lbl'), '日历');
    // Đổi ngôn ngữ NGAY TRONG tab Lịch.
    await pick(page, 'langDisplayBtn', 'vi');
    check('nhãn tab đổi ngay tại tab Lịch', await page.textContent('#tabCal .tab-lbl'), 'Lịch');
    const calVi = await page.textContent('#calTitle');
    ok('tiêu đề lịch sang tiếng Việt', /THÁNG/i.test(calVi), calVi);
    await page.click('#tabQmdj'); await page.waitForTimeout(700);
    const ttNam = await page.textContent('#lblTTNamTxt');
    check('tab Kỳ Môn cũng đã sang tiếng Việt', ttNam, 'Năm');

    console.log('\nĐồng bộ hai tab: ĐỊA ĐIỂM đổi từ tab Lịch');
    // Mốc ban đầu, đọc ở tab Kỳ Môn.
    const gmtBefore = (await page.textContent('#out-chinhngo')).match(/GMT[+-]\d+/)?.[0];
    await page.click('#tabCal'); await page.waitForTimeout(700);
    const jqBefore = await page.textContent('#calJieQi');

    // Danh sách múi giờ được đổ SAU khi mở hộp thoại (chờ cities.txt), nên
    // phải đợi có mục Asia/Ho_Chi_Minh rồi mới chọn — gán sớm thì <select> bỏ
    // qua, giá trị còn rỗng và onManualApply lặng lẽ lấy múi giờ của máy.
    await page.evaluate(() => window.openCountryPicker());
    await page.waitForFunction(
        () => [...document.getElementById('locTz').options].some(o => o.value === 'Asia/Ho_Chi_Minh'),
        null, { timeout: 5000 });
    await page.evaluate(() => {
        document.getElementById('locLat').value = '21.0278';
        document.getElementById('locLon').value = '105.8342';
        document.getElementById('locTz').value = 'Asia/Ho_Chi_Minh';
        document.getElementById('locApply').dispatchEvent(new Event('click'));
    });
    await page.waitForTimeout(900);

    const jqAfter = await page.textContent('#calJieQi');
    ok('bảng tiết khí của tab Lịch vẽ lại ngay', jqAfter !== jqBefore, 'không đổi gì');
    const ctry = await page.textContent('#countryDisplayText');
    ok('ô địa điểm hiện tên mới ngay tại tab Lịch', /Hà Nội|21\.03|105\.83/i.test(ctry), ctry);

    await page.click('#tabQmdj'); await page.waitForTimeout(700);
    const gmtAfter = (await page.textContent('#out-chinhngo')).match(/GMT[+-]\d+/)?.[0];
    check('tab Kỳ Môn nhận múi giờ mới', gmtAfter, 'GMT+7');
    ok('múi giờ đã thật sự đổi', gmtBefore !== gmtAfter, `trước ${gmtBefore}, sau ${gmtAfter}`);

    console.log('\nĐồng bộ hai tab: đổi từ tab Kỳ Môn thì tab Lịch theo');
    await pick(page, 'langDisplayBtn', 'zh');
    await page.click('#tabCal');     await page.waitForTimeout(700);
    check('lịch quay lại tiếng Trung', await page.textContent('#tabCal .tab-lbl'), '日历');
    await ctx.close();
}

/* ── 5. Hai ô chọn mới: mở, đánh dấu mục đang chọn, Hủy thì không đổi ── */
{
    const ctx = await browser.newContext({ viewport: { width: 393, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
    const page = await ctx.newPage();
    await page.goto(base, { waitUntil: 'networkidle' });
    await page.waitForTimeout(900);

    console.log('\nÔ chọn ngôn ngữ và ô chọn phái');
    // Cả hai ô phải mở ra CÙNG một kiểu bảng như ô địa điểm.
    await pick(page, 'langDisplayBtn', 'vi');
    check('ô ngôn ngữ hiện mục đã chọn', await page.textContent('#langDisplayText'), 'Tiếng Việt');
    check('ô phái theo ngôn ngữ mới', await page.textContent('#methodDisplayText'), 'Âm Bàn');

    await page.click('#methodDisplayBtn');
    await page.waitForSelector('#optOverlay.open', { timeout: 4000 });
    const sheet = await page.evaluate(() => {
        const rows = [...document.querySelectorAll('#optList .opt-row')];
        return {
            n: rows.length,
            labels: rows.map(r => r.querySelector('.opt-name').textContent),
            active: rows.filter(r => r.classList.contains('opt-row-active'))
                        .map(r => r.getAttribute('data-value')),
            title: document.getElementById('optTitle').textContent,
        };
    });
    check('bảng phái có đủ ba mục', sheet.n, 3);
    check('đúng thứ tự cũ', sheet.labels.join(' · '), 'Trí Nhuận · Âm Bàn · Sách Bổ');
    check('đánh dấu đúng mục đang chọn', sheet.active.join(','), 'amban');
    check('tiêu đề bảng theo ngôn ngữ', sheet.title, 'Phái');

    // Hủy thì KHÔNG được đổi gì.
    await page.click('#optCancel'); await page.waitForTimeout(500);
    check('bấm Hủy thì bảng đóng', await page.evaluate(() => document.getElementById('optOverlay').classList.contains('open')), 'false');
    check('bấm Hủy thì phái giữ nguyên', await page.inputValue('#methodSelect'), 'amban');

    // Chọn thật thì đổi cả ô, cả engine, cả bảng chi tiết.
    await pick(page, 'methodDisplayBtn', 'bophap');
    check('chọn Sách Bổ: ô hiện đúng', await page.textContent('#methodDisplayText'), 'Sách Bổ');
    check('chọn Sách Bổ: engine nhận', await page.inputValue('#methodSelect'), 'bophap');
    check('chọn Sách Bổ: bảng chi tiết đổi theo',
        await page.evaluate(() => getComputedStyle(document.getElementById('sachboPanel')).display), 'block');

    // Nút Back của Android phải đóng bảng chọn, không thoát ứng dụng.
    await page.click('#langDisplayBtn');
    await page.waitForSelector('#optOverlay.open', { timeout: 4000 });
    const handled = await page.evaluate(() => window.__onBackPressed());
    check('nút Back đóng bảng chọn', String(handled), 'true');
    check('bảng chọn đã đóng', await page.evaluate(() => document.getElementById('optOverlay').classList.contains('open')), 'false');
    await ctx.close();
}

/* ── 6. Chia bề rộng: ô phái vừa đúng nhãn dài nhất, phần thừa cho ngày giờ ── */
{
    console.log('\nChia bề rộng hàng Kỳ Môn');
    for (const d of [{ name: 'S21', w: 360 }, { name: 'A51', w: 412 }]) {
        for (const lang of ['zh', 'vi']) {
            const ctx = await browser.newContext({ viewport: { width: d.w, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
            const page = await ctx.newPage();
            await page.goto(base, { waitUntil: 'networkidle' });
            await page.waitForTimeout(900);
            await pick(page, 'langDisplayBtn', lang);

            const seen = [];
            for (const m of ['trinhuan', 'amban', 'bophap']) {
                await pick(page, 'methodDisplayBtn', m);
                seen.push(await page.evaluate(() => ({
                    date: +document.getElementById('dateDisplayBtn').getBoundingClientRect().width.toFixed(1),
                    meth: +document.getElementById('methodDisplayBtn').getBoundingClientRect().width.toFixed(1),
                    dateH: +document.getElementById('dateDisplayBtn').getBoundingClientRect().height.toFixed(1),
                    methH: +document.getElementById('methodDisplayBtn').getBoundingClientRect().height.toFixed(1),
                    // Nhãn đang hiện phải vừa trong ô, không bị cắt.
                    fits: (() => {
                        const t = document.getElementById('methodDisplayText');
                        const b = document.getElementById('methodDisplayBtn');
                        return t.scrollWidth <= t.clientWidth + 1 &&
                               b.scrollWidth <= b.clientWidth + 1;
                    })(),
                    // Bề rộng THẬT của chữ, đo bằng phạm vi chọn — khác bề
                    // rộng hộp, vì <span> giãn kín ô lưới.
                    label: (() => {
                        const t = document.getElementById('methodDisplayText');
                        const r = document.createRange();
                        r.selectNodeContents(t);
                        return +r.getBoundingClientRect().width.toFixed(1);
                    })(),
                })));
            }
            const tag = `${d.name} ${d.w}px · ${lang}`;
            const widths = [...new Set(seen.map(x => x.meth))];
            ok(`${tag}: ô phái không đổi bề rộng theo phái`, widths.length === 1, widths.join(' / '));
            ok(`${tag}: nhãn phái nào cũng vừa ô`, seen.every(x => x.fits));
            // Ô ngày giờ phải là ô ĂN phần thừa, không phải ô bị bóp.
            ok(`${tag}: ngày giờ vẫn rộng hơn ô phái`, seen[0].date > seen[0].meth,
                `ngày giờ ${seen[0].date} vs phái ${seen[0].meth}`);
            // Ô phái nới thêm 50% bằng zoom trên ba nhãn ẩn (xem .opt-ghost).
            // Nếu WebView bỏ qua zoom thì ô co về bề rộng chữ trần — canh cho
            // chắc là phần lề quanh chữ thật sự có.
            ok(`${tag}: ô phái nới rộng quanh nhãn`, seen[0].meth >= seen[0].label * 1.35,
                `ô ${seen[0].meth} vs nhãn ${seen[0].label}`);
            // Nới RỘNG, không nới CAO: `zoom` phóng cả hai chiều, nên ba nhãn
            // ẩn phải bị ép height:0, nếu không ô phái cao hơn ô ngày giờ.
            ok(`${tag}: ô phái CAO BẰNG ô ngày giờ`, Math.abs(seen[0].methH - seen[0].dateH) < 1.5,
                `phái ${seen[0].methH}px vs ngày giờ ${seen[0].dateH}px`);
            await ctx.close();
        }
    }
}

/* ── 7. Canh theo vạch cột của bảng Tứ Trụ ──
   Khe giữa ô ngày giờ và ô phái phải rơi đúng vạch Tháng|Ngày của bảng Tứ Trụ;
   còn hai ô của hàng dùng chung phải trùng khít hai tab ngay trên nó. Chỉ cần
   một bên đổi đệm là lệch ngay — canh bằng số đo, không bằng mắt. */
{
    console.log('\nCanh theo vạch cột bảng Tứ Trụ');
    // Ca "phông rộng": phông hệ thống trên máy thật (Samsung) rộng hơn phông
    // mặc định của Chromium ở máy dựng, nên một hàng vừa khít ở đây vẫn có thể
    // tràn trên máy. Nống cỡ chữ 12% để bắt trước cái tràn ấy — chính nó đã cắt
    // mất chữ "Đầy đủ" ngoài đời.
    const WIDE = `.picker-btn { font-size: 15.2px !important; }
                  #cobanToggleWrap { font-size: 15.7px !important; }`;
    for (const d of [{ name: 'S21', w: 360 }, { name: 'S21 FE', w: 393 },
                     { name: 'A51', w: 412 }, { name: 'rộng', w: 520 },
                     { name: 'S21 FE phông rộng', w: 393, wide: true },
                     { name: 'S21 phông rộng', w: 360, wide: true }]) {
        for (const lang of ['zh', 'vi']) {
            const ctx = await browser.newContext({ viewport: { width: d.w, height: 790 }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
            const page = await ctx.newPage();
            await page.goto(base, { waitUntil: 'networkidle' });
            await page.waitForTimeout(1000);
            if (d.wide) await page.addStyleTag({ content: WIDE });
            await pick(page, 'langDisplayBtn', lang);

            const r = await page.evaluate(() => {
                const th = [...document.querySelectorAll('#tuTruPanel thead th')]
                    .map(x => x.getBoundingClientRect());
                const box = id => document.getElementById(id).getBoundingClientRect();
                const dt = box('dateDisplayBtn'), mb = box('methodDisplayBtn');
                const tabs = [...document.querySelectorAll('.tab-item')]
                    .map(x => x.getBoundingClientRect());
                return {
                    seam: (dt.right + mb.left) / 2,   // tâm khe giữa hai ô
                    edgeTN: th[1].right,              // vạch Tháng | Ngày
                    // Hàng dùng chung phải trùng khít hàng tab ngay trên nó.
                    lang: [box('langDisplayBtn').left, box('langDisplayBtn').right],
                    ctry: [box('countryDisplayBtn').left, box('countryDisplayBtn').right],
                    tab0: [tabs[0].left, tabs[0].right],
                    tab1: [tabs[1].left, tabs[1].right],
                    tabH: tabs[0].height,
                    boxH: box('langDisplayBtn').height,
                    // Khe giữa hàng tab và hàng dùng chung.
                    rowGap: box('sharedBar').top -
                            document.getElementById('tabBar').getBoundingClientRect().bottom,
                    // Mép phải ô địa điểm, tính theo phần trăm bề ngang.
                    rightPct: box('countryDisplayBtn').right / window.innerWidth * 100,
                    // Nửa phải của hàng vẫn phải đủ chỗ cho ô phái và ô Đầy đủ.
                    fits: document.getElementById('qmRow').scrollWidth <=
                          document.getElementById('qmRow').clientWidth + 1,
                    // Ô "Đầy đủ" nằm cuối hàng — tràn là nó bị cắt mất trước tiên.
                    cobanOver: box('cobanToggleWrap').right -
                               document.getElementById('qmRow').getBoundingClientRect().right,
                    dateFits: (() => {
                        const t = document.getElementById('dateDisplayText');
                        return t.scrollWidth <= t.clientWidth + 1;
                    })(),
                };
            });
            const tag = `${d.name} ${d.w}px · ${lang}`;
            // Ngưỡng 1px: bảng Tứ Trụ dùng border-collapse nên vạch dày 1px,
            // tâm vạch và mép ô lệch nhau tối đa nửa viền.
            // Phông rộng thì ô ngày giờ NHƯỜNG chỗ nên khe lùi trái — cố ý:
            // thà lệch vạch còn hơn đẩy ô "Đầy đủ" ra khỏi màn hình.
            if (!d.wide) {
                ok(`${tag}: khe trùng vạch Tháng|Ngày`, Math.abs(r.seam - r.edgeTN) <= 1,
                    `khe ${r.seam.toFixed(1)} vs vạch ${r.edgeTN.toFixed(1)}`);
            }
            const near = (a, b) => Math.abs(a[0] - b[0]) <= 1 && Math.abs(a[1] - b[1]) <= 1;
            ok(`${tag}: ô ngôn ngữ trùng khít tab Kỳ Môn`, near(r.lang, r.tab0),
                `[${r.lang.map(v => v.toFixed(1))}] vs [${r.tab0.map(v => v.toFixed(1))}]`);
            ok(`${tag}: ô địa điểm trùng khít tab Lịch`, near(r.ctry, r.tab1),
                `[${r.ctry.map(v => v.toFixed(1))}] vs [${r.tab1.map(v => v.toFixed(1))}]`);
            ok(`${tag}: ô cao đúng bằng tab`, Math.abs(r.boxH - r.tabH) <= 1,
                `ô ${r.boxH.toFixed(1)}px vs tab ${r.tabH.toFixed(1)}px`);
            ok(`${tag}: có khe giữa hai hàng`, r.rowGap >= 3 && r.rowGap <= 10,
                `khe ${r.rowGap.toFixed(1)}px`);
            ok(`${tag}: mép phải ô địa điểm ≥ 75%`, r.rightPct >= 75,
                `mới ${r.rightPct.toFixed(1)}%`);
            ok(`${tag}: nửa phải vẫn đủ chỗ`, r.fits);
            ok(`${tag}: ô Đầy đủ không tràn khỏi hàng`, r.cobanOver <= 1,
                `thò ${r.cobanOver.toFixed(1)}px`);
            ok(`${tag}: ngày giờ không bị cắt`, r.dateFits);
            await ctx.close();
        }
    }
}

/* ── 8. Không còn khoảng hở ở đáy màn Kỳ Môn ──
   Tỉ lệ phóng bị chặn bởi BỀ NGANG, nên trên máy cao phần dôi chiều cao nằm
   chết ngay trên thanh dưới (S21 FE: 39px, A51: 82px). viewport.js rót phần ấy
   vào các khe giữa các bảng — canh cho nó thật sự rót. */
{
    console.log('\nKhoảng hở ở đáy màn Kỳ Môn');
    for (const d of [{ name: 'S21', w: 360, h: 740 }, { name: 'S21 FE', w: 393, h: 790 },
                     { name: 'S21 Ultra', w: 384, h: 794 }, { name: 'A51', w: 412, h: 852 }]) {
        const ctx = await browser.newContext({ viewport: { width: d.w, height: d.h }, deviceScaleFactor: 2, isMobile: true, hasTouch: true });
        const page = await ctx.newPage();
        await page.goto(base, { waitUntil: 'networkidle' });
        await page.waitForTimeout(1400);
        const r = await page.evaluate(() => {
            const dock = document.getElementById('bottomDock').getBoundingClientRect();
            const kids = [...document.body.children].filter(e => {
                const cs = getComputedStyle(e);
                return cs.display !== 'none' && cs.position !== 'fixed';
            });
            const last = kids[kids.length - 1].getBoundingClientRect();
            return { gap: dock.top - last.bottom,
                     over: document.documentElement.scrollHeight > window.innerHeight + 1 };
        });
        ok(`${d.name} ${d.w}×${d.h}: hở ≤ 16px trên thanh dưới`, r.gap <= 16,
            `hở ${r.gap.toFixed(1)}px`);
        ok(`${d.name}: vẫn không tràn dọc`, !r.over);
        await ctx.close();
    }
}

await browser.close();
server.close();

console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Hàng dùng chung đúng chỗ, vừa khít và đồng bộ hai tab');
process.exit(fail ? 1 : 0);
