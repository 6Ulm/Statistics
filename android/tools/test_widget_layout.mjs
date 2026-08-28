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
const CORNER_DP = 16;
const PAD_NARROWEST_DP = 4;
const arcDepth = (r, x) => (x >= r ? 0 : r - Math.sqrt(2 * r * x - x * x));
/** Lưới lịch phải giữ được ngần này phần thân widget, không cho bảng nuốt hết. */
const MIN_GRID_SHARE = 0.35;

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
    /** layout[month][i] — số đo của cỡ widget thứ i ở tháng ấy. */
    const byMonth = {};
    for (const m of MONTHS) {
        await page.goto(`${base}?today=${TODAY}&month=${m}&density=${dev.density}&sizes=${sizesParam}`,
            { waitUntil: 'networkidle' });
        await page.waitForSelector('body[data-ready="1"]');
        byMonth[m] = await page.evaluate(() => window.__wlayout);
    }

    dev.sizes.forEach(([wDp, hDp, floorNote], i) => {
        const L = byMonth[MONTHS[0]][i];
        const px = v => v / dev.density;               // px canvas → dp
        const tag = `${dev.name} ${wDp}×${hDp}dp${floorNote ? ' (' + floorNote + ')' : ''}`;
        const minText = floorNote ? FLOOR_TEXT_DP : MIN_TEXT_DP;

        if (!L.jq) { bad(`${tag}: không dựng được bảng tiết khí`); return; }
        const q = L.jq;

        // 1. Không tràn: mép phải của giá trị dài nhất phải nằm trong nửa bảng.
        const right = q.dateDx + q.maxDateW;
        const limit = q.halfW - q.padEnd;
        if (right > limit + 0.5) {
            bad(`${tag}: giá trị tràn ${px(right - limit).toFixed(1)}dp qua vách ngăn`);
        } else ok();

        // 2. Chữ còn đọc được.
        if (px(q.txtPx) < minText) {
            bad(`${tag}: chữ tiết khí chỉ ${px(q.txtPx).toFixed(1)}dp (< ${minText}dp)`);
        } else ok();

        // 3. Hàng cuối nằm trên cung góc bo, và bảng không vượt đáy bitmap.
        // Đệm đáy phải phủ được chỗ cung góc ăn tới ở hoành độ chữ bắt đầu.
        const needPad = arcDepth(CORNER_DP, PAD_NARROWEST_DP);
        if (px(q.bottomPad) < needPad - 0.01) {
            bad(`${tag}: đệm đáy ${px(q.bottomPad).toFixed(1)}dp < cung góc ăn `
                + `${needPad.toFixed(1)}dp — hàng cuối sẽ bị cắt`);
        } else ok();
        // ...nhưng cũng không được hở hơn mức cần quá 4dp: chừa thừa là một dải
        // trắng vô cớ dưới đáy bảng, đúng thứ người dùng kêu.
        if (px(q.bottomPad) > needPad + 4) {
            bad(`${tag}: đệm đáy ${px(q.bottomPad).toFixed(1)}dp, thừa `
                + `${(px(q.bottomPad) - needPad).toFixed(1)}dp so với cung góc`);
        } else ok();
        const hPx = (hDp - 32) * dev.density;
        if (q.top + q.tableH > hPx + 0.5) {
            bad(`${tag}: bảng thò ${px(q.top + q.tableH - hPx).toFixed(1)}dp khỏi đáy widget`);
        } else ok();

        // 4. Lưới lịch không bị bảng nuốt.
        const share = L.gridH / (hPx - L.dowH);
        if (share < MIN_GRID_SHARE) {
            bad(`${tag}: lưới lịch chỉ còn ${(share * 100).toFixed(0)}% thân widget`);
        } else ok();

        // 5. Bảng CỐ ĐỊNH: mọi tháng cho ra cùng một khung, cùng cỡ chữ.
        for (const m of MONTHS.slice(1)) {
            const o = byMonth[m][i], oq = o.jq;
            const same = ['jqRowH', 'jqH'].every(k => Math.abs(L[k] - o[k]) < 0.01)
                && ['txtPx', 'nameW', 'dateDx', 'top', 'tableH']
                    .every(k => Math.abs(q[k] - oq[k]) < 0.01);
            if (!same) {
                bad(`${tag}: tháng ${m} cho bảng khác tháng ${MONTHS[0]} `
                    + `(cao ${px(oq.tableH).toFixed(1)} vs ${px(q.tableH).toFixed(1)}dp, `
                    + `chữ ${px(oq.txtPx).toFixed(1)} vs ${px(q.txtPx).toFixed(1)}dp)`);
            } else ok();
        }

        console.log(`  ${tag}: hàng ${px(L.jqRowH).toFixed(1)}dp · chữ ${px(q.txtPx).toFixed(1)}dp`
            + ` · cột ngày ở ${px(q.dateDx).toFixed(0)}dp · thừa ${px(limit - right).toFixed(1)}dp`
            + ` · đệm đáy ${px(q.bottomPad).toFixed(0)}dp · lưới ${(share * 100).toFixed(0)}%`);
    });
    await ctx.close();
}

await browser.close();
server.close();
console.log(`\n${fails === 0 ? '✓ ĐẠT' : '✗ HỎNG'} — ${checks} phép canh đạt, ${fails} hỏng`);
process.exit(fails === 0 ? 0 : 1);
