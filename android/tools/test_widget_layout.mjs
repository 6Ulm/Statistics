/**
 * Đo bố cục bảng tiết khí của widget ở đúng cấu hình S21 và A51.
 *
 *   node test_widget_layout.mjs
 *
 * Không chụp ảnh mà ĐỌC SỐ: `widget_preview.html` (bản mô phỏng 1:1 của
 * drawBody() trong CalendarWidgetProvider.kt) trả về vị trí cột, cỡ chữ, chiều
 * cao bảng và đệm đáy; phép thử canh bốn điều mà video của người dùng bắt được:
 *
 *   1. Giá trị cột "Dương lịch" không tràn qua vách ngăn / mép widget.
 *   2. Chữ tiết khí không nhỏ đến mức không đọc nổi.
 *   3. Hàng cuối (Mang Chủng · Đại Tuyết) nằm trên cung góc bo, không bị cắt.
 *   4. Khung bảng và cỡ chữ KHÔNG đổi khi lật tháng — tháng 5 hàng lịch và
 *      tháng 6 hàng lịch phải cho ra cùng một bảng.
 *
 * Lưu ý về phông: Chromium ở đây dùng phông sans mặc định của máy chủ, rộng hơn
 * Roboto của Android, nên cỡ chữ đo được là PHÍA AN TOÀN — trên máy thật chữ chỉ
 * có thể to hơn con số ở đây, không nhỏ hơn.
 */
import fs from 'fs';
import http from 'http';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const ASSETS = path.join(HERE, '..', 'app', 'src', 'main', 'assets');

/**
 * Hai máy đích, kèm dải cỡ widget mà người dùng dựng được trên lưới màn hình
 * chính One UI.
 *
 * Bề ngang: widget rộng 4 cột ≈ bề ngang màn hình trừ lề hai bên (~0,92×) —
 * S21 (360dp) ra ~330dp, A51 (412dp) ra ~380dp. Bề dọc lấy từ sàn khai báo
 * trong calendar_widget_info.xml (320dp) lên tới gần kín màn hình.
 *
 * Kèm cả 250dp: `minResizeWidth` cho phép bóp tới đó, nên nó vẫn phải đứng được.
 */
const DEVICES = [
    {
        name: 'S21',  screenDp: 360, density: 3,
        sizes: [[250, 320, 'sàn bóp tay'], [330, 320], [330, 400],
                [330, 440], [330, 530], [330, 620]],
    },
    {
        // 1080×2340 @2,75x → 393×851dp. Nằm giữa S21 và A51 cả bề ngang lẫn
        // mật độ, nên là ca canh xem luật co giãn có mượt giữa hai đầu không.
        name: 'S21 FE', screenDp: 393, density: 2.75,
        sizes: [[250, 320, 'sàn bóp tay'], [360, 320], [360, 400],
                [360, 450], [360, 545], [360, 650]],
    },
    {
        name: 'A51',  screenDp: 412, density: 2.625,
        sizes: [[250, 320, 'sàn bóp tay'], [380, 320], [380, 400],
                [380, 460], [380, 560], [380, 680]],
    },
];

/** Tháng dùng để thử: 2/2027 có 4 hàng lịch, 9/2026 có 5, 8/2026 có 6. */
const MONTHS = ['2026-8', '2026-9', '2027-2', '2026-11', '2027-5'];
const TODAY = '2026-08-27';

/**
 * Cỡ chữ tối thiểu. Hai mức, vì hai loại cỡ widget khác hẳn nhau:
 *
 *   * `MIN_TEXT_DP` — cho mọi cỡ mà LƯỚI MÀN HÌNH tự dựng ra (widget 4 cột:
 *     330dp trên S21, 380dp trên A51). Đây là cái người dùng thực sự nhìn.
 *   * `FLOOR_TEXT_DP` — cho cỡ sàn 250dp mà người dùng phải tự tay bóp lại mới
 *     có (`minResizeWidth`). Ở đó nửa bảng chỉ còn 125dp cho cả tên lẫn mốc
 *     ngày giờ, chữ buộc phải nhỏ; README đã ghi rõ là "đọc được nhưng nhỏ".
 *     Không nâng được sàn này bằng manifest: `minWidth` mà quá 250dp thì công
 *     thức ô của Android đòi 5 cột, widget hết đặt được lên lưới 4 cột.
 */
const MIN_TEXT_DP = 8;
const FLOOR_TEXT_DP = 7.5;
/**
 * Góc bo của bản xem trước (widget_bg.xml; widget thật hỏi hệ thống, có thể
 * rộng hơn). Đệm đáy KHÔNG cần cả bán kính — cung tròn chỉ ăn sâu
 * `r − √(2r·x − x²)` ở hoành độ x mà chữ bắt đầu — nên phép canh dựng lại đúng
 * công thức ấy thay vì so với một con số chết.
 */
/** Lưới lịch phải giữ được ngần này phần thân widget, không cho hai mục nuốt hết. */
const MIN_GRID_SHARE = 0.35;
/** Chiều cao thanh tiêu đề — phải khớp values/dimens.xml. */
const HEADER_DP = 32;
/** Mục đang mở mà hiện chưa nổi ngần này hàng thì cuộn cũng chẳng để làm gì. */
const MIN_ROWS_SEEN = 2;
/**
 * Ba cột của hai mục chia bằng `layout_weight` chứ không co theo chữ, nên chữ
 * dài có thể bị cắt vài pixel. Ngần này thì mắt không thấy; hơn nữa là phải
 * chỉnh lại bộ weight trong widget_calendar.xml và widget_sec_row.xml.
 */
const MAX_SPILL_PX = 2;
/**
 * Khe dọc tối thiểu trong ô lịch (dp): trên số ngày, giữa số ngày và can chi,
 * dưới chi. Người dùng bắt được đúng hai chỗ này ở bản trước — "can quá sát số
 * lịch dương, chi quá sát vạch đáy" — nên nay canh THẲNG ba con số ấy thay vì
 * chỉ canh cỡ chữ. Bản cũ cho ra 0,4dp và 0,5dp, tức chữ gần như chạm nhau.
 */
const MIN_CELL_PAD_DP = 2;
/** Ba khe phải BẰNG NHAU (chia đều phần dôi), sai số cho phép vì làm tròn pixel. */
const PAD_EVEN_DP = 0.75;

const MIME = { '.html': 'text/html', '.txt': 'text/plain', '.js': 'text/javascript' };
const server = http.createServer((req, res) => {
    const rel = decodeURIComponent(req.url.split('?')[0]).replace(/^\/+/, '') || 'widget_preview.html';
    const file = rel.startsWith('assets/')
        ? path.join(ASSETS, rel.slice('assets/'.length))
        : path.join(HERE, rel);
    if (!fs.existsSync(file) || fs.statSync(file).isDirectory()) { res.writeHead(404); return res.end(); }
    res.writeHead(200, { 'Content-Type': MIME[path.extname(file)] || 'application/octet-stream' });
    fs.createReadStream(file).pipe(res);
});
await new Promise(r => server.listen(0, '127.0.0.1', r));
const base = `http://127.0.0.1:${server.address().port}/widget_preview.html`;

const { chromium } = await import('playwright');
const browser = await chromium.launch(
    fs.existsSync('/opt/pw-browsers/chromium') ? { executablePath: '/opt/pw-browsers/chromium' } : {}
);

let fails = 0, checks = 0;
const bad = (msg) => { fails++; console.log('  ✗ ' + msg); };
const ok = () => { checks++; };

for (const dev of DEVICES) {
    console.log(`\n── ${dev.name} — màn hình ${dev.screenDp}dp @${dev.density}x ──`);
    const ctx = await browser.newContext({ viewport: { width: 1400, height: 900 } });
    const page = await ctx.newPage();
    page.on('pageerror', e => { bad(`lỗi trang: ${e.message}`); });

    const sizesParam = dev.sizes.map(([w, h]) => `${w}x${h}`).join(',');
    const đo = async (m, lang) => {
        await page.goto(`${base}?today=${TODAY}&month=${m}&density=${dev.density}`
            + `&sizes=${sizesParam}` + (lang ? `&lang=${lang}` : ''),
            { waitUntil: 'networkidle' });
        await page.waitForSelector('body[data-ready="1"]');
        return page.evaluate(() => window.__wlayout);
    };
    /** layout[month][i] — số đo của cỡ widget thứ i ở tháng ấy. */
    const byMonth = {};
    for (const m of MONTHS) byMonth[m] = await đo(m, null);
    /**
     * Cùng widget ấy nhưng TIẾNG TRUNG. Can chi tiếng Trung là chữ vuông, vùng
     * mực khác hẳn chữ Việt có dấu, nên ba khe dọc phải đo lại chứ không suy ra
     * được từ bản tiếng Việt. Một tháng là đủ: khung lưới không phụ thuộc tháng
     * (chính phép canh 6 bên dưới canh điều đó).
     */
    const zh = await đo(MONTHS[0], 'zh');

    dev.sizes.forEach(([wDp, hDp, floorNote], i) => {
        const L = byMonth[MONTHS[0]][i];
        const tag = `${dev.name} ${wDp}×${hDp}dp${floorNote ? ' (' + floorNote + ')' : ''}`;
        const minText = floorNote ? FLOOR_TEXT_DP : MIN_TEXT_DP;

        if (!L) { bad(`${tag}: không dựng được widget`); return; }

        // 1. Số ngày trong ô lịch còn đọc được.
        if (L.dayPx < minText) {
            bad(`${tag}: số ngày chỉ ${L.dayPx.toFixed(1)}dp (< ${minText}dp)`);
        } else ok();

        // 2. Lưới lịch không bị hai mục nuốt mất.
        const share = L.gridDp / (hDp - HEADER_DP);
        if (share < MIN_GRID_SHARE) {
            bad(`${tag}: lưới lịch chỉ còn ${(share * 100).toFixed(0)}% thân widget`);
        } else ok();

        // 3. Không tràn khỏi chiều cao widget.
        if (L.over) bad(`${tag}: nội dung tràn khỏi chiều cao widget`); else ok();

        // 4. Mục LUÔN HIỆN (không gập được nữa) nên phải luôn đủ mấy hàng,
        //    không thì cuộn cũng vô nghĩa.
        if (L.jqOpen && L.jqSeen < MIN_ROWS_SEEN) {
            bad(`${tag}: mục Tiết khí chỉ hiện ${L.jqSeen} hàng`);
        } else ok();

        // 5. Ba cột chia bằng layout_weight nên KHÔNG tự nới theo chữ — cột hẹp
        //    hơn chữ là chữ bị cắt. Đây là cái giá của việc bỏ bitmap, nên phải
        //    canh: chỗ nào cắt quá vài pixel là phải chỉnh lại bộ weight.
        if (L.spill > MAX_SPILL_PX) {
            bad(`${tag}: "${L.spillText}" bị cắt ${L.spill.toFixed(1)}px`);
        } else ok();

        // 6. Khung CỐ ĐỊNH: mọi tháng cho ra cùng một lưới, cùng cỡ chữ. Lưới
        //    luôn 6 hàng nên tháng 4 hay 6 tuần đều phải y hệt nhau.
        for (const m of MONTHS.slice(1)) {
            const o = byMonth[m][i];
            const same = ['gridDp', 'cellH', 'dayPx', 'gzPx']
                .every(k => Math.abs(L[k] - o[k]) < 0.01);
            if (!same) {
                bad(`${tag}: tháng ${m} cho lưới khác tháng ${MONTHS[0]} `
                    + `(cao ${o.gridDp.toFixed(1)} vs ${L.gridDp.toFixed(1)}dp, `
                    + `chữ ${o.dayPx.toFixed(1)} vs ${L.dayPx.toFixed(1)}dp)`);
            } else ok();
        }

        // 7. Số ngày và can chi đứng CÂN ĐỐI trong ô: ba khe dọc đều nhau và
        //    không khe nào mỏng tới mức chữ dính vào nhau hay dính vạch lưới.
        //    Chỉ xét ô có can chi — ô không có thì dòng số ngày canh giữa, và
        //    padMid không có nghĩa gì.
        if (L.showGanZhi) {
            const pads = [L.padTop, L.padMid, L.padBot];
            const mỏng = Math.min(...pads);
            if (mỏng < MIN_CELL_PAD_DP) {
                bad(`${tag}: khe dọc trong ô chỉ ${mỏng.toFixed(2)}dp `
                    + `(trên ${L.padTop.toFixed(2)} · giữa ${L.padMid.toFixed(2)} `
                    + `· dưới ${L.padBot.toFixed(2)})`);
            } else ok();
            const lệch = Math.max(...pads) - mỏng;
            if (lệch > PAD_EVEN_DP) {
                bad(`${tag}: ba khe dọc không đều, lệch ${lệch.toFixed(2)}dp `
                    + `(trên ${L.padTop.toFixed(2)} · giữa ${L.padMid.toFixed(2)} `
                    + `· dưới ${L.padBot.toFixed(2)})`);
            } else ok();
        } else {
            // Không can chi thì dòng số ngày phải canh GIỮA ô, không dính đỉnh.
            const lệch = Math.abs(L.padTop - L.padBot);
            if (lệch > PAD_EVEN_DP) {
                bad(`${tag}: ô không có can chi mà số ngày lệch tâm ${lệch.toFixed(2)}dp`);
            } else ok();
        }

        // 8. TIẾNG TRUNG phải đứng được y hệt: cùng khung lưới, cùng cỡ chữ,
        //    can chi bật/tắt giống nhau, và ba khe dọc cũng đủ rộng. Đổi ngôn
        //    ngữ mà lưới đổi hình là lỗi — hai mặt của cùng một widget.
        const Z = zh[i];
        if (!Z) { bad(`${tag}: không dựng được widget tiếng Trung`); }
        else {
            const khung = ['gridDp', 'cellH', 'dayPx', 'gzPx']
                .every(k => Math.abs(L[k] - Z[k]) < 0.01);
            if (!khung) {
                bad(`${tag}: tiếng Trung cho lưới khác tiếng Việt `
                    + `(ô ${Z.cellH.toFixed(1)} vs ${L.cellH.toFixed(1)}dp, `
                    + `chữ ${Z.dayPx.toFixed(1)} vs ${L.dayPx.toFixed(1)}dp)`);
            } else ok();
            if (Z.showGanZhi !== L.showGanZhi) {
                bad(`${tag}: can chi ${Z.showGanZhi ? 'hiện' : 'tắt'} ở tiếng Trung `
                    + `mà ${L.showGanZhi ? 'hiện' : 'tắt'} ở tiếng Việt`);
            } else ok();
            const zPads = Z.showGanZhi ? [Z.padTop, Z.padMid, Z.padBot] : [Z.padTop, Z.padBot];
            const zMỏng = Math.min(...zPads);
            if (zMỏng < MIN_CELL_PAD_DP) {
                bad(`${tag}: khe dọc tiếng Trung chỉ ${zMỏng.toFixed(2)}dp `
                    + `(trên ${Z.padTop.toFixed(2)} · giữa ${Z.padMid.toFixed(2)} `
                    + `· dưới ${Z.padBot.toFixed(2)})`);
            } else ok();
            if (Math.max(...zPads) - zMỏng > PAD_EVEN_DP) {
                bad(`${tag}: ba khe dọc tiếng Trung không đều `
                    + `(${Z.padTop.toFixed(2)}/${Z.padMid.toFixed(2)}/${Z.padBot.toFixed(2)}dp)`);
            } else ok();
            if (Z.spill > MAX_SPILL_PX) {
                bad(`${tag}: tiếng Trung — "${Z.spillText}" bị cắt ${Z.spill.toFixed(1)}px`);
            } else ok();
        }

        console.log(`  ${tag}: lưới ${L.gridDp.toFixed(0)}dp · ô ${L.cellH.toFixed(1)}dp`
            + ` · số ngày ${L.dayPx.toFixed(1)}dp · can chi ${L.showGanZhi ? 'có' : 'tắt'}`
            + ` · hàng hiện ${L.jqSeen}`
            + ` · khe ${L.padTop.toFixed(1)}/${L.padMid.toFixed(1)}/${L.padBot.toFixed(1)}dp`
            + ` (中 ${zh[i] ? zh[i].padTop.toFixed(1) + '/' + zh[i].padMid.toFixed(1)
                + '/' + zh[i].padBot.toFixed(1) : '—'})`
            + ` · cắt chữ ${L.spill.toFixed(1)}px · lưới ${(share * 100).toFixed(0)}%`);
    });
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\n${fails === 0 ? '✓ ĐẠT' : '✗ HỎNG'} — ${checks} phép canh đạt, ${fails} hỏng`);
process.exit(fails === 0 ? 0 : 1);
