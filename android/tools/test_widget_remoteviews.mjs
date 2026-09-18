/**
 * Bố cục widget chỉ được dùng những lớp view mà RemoteViews dựng được.
 *
 *   node test_widget_remoteviews.mjs
 *
 * RemoteViews không dựng bằng LayoutInflater thường: nó đặt một bộ lọc chỉ cho
 * qua các lớp CÓ chú thích @RemoteView. Một thẻ không nằm trong danh sách ấy
 * làm cả widget hỏng NGAY TỪ LÚC DỰNG, mà lỗi thì nằm trong tiến trình
 * launcher chứ không phải ứng dụng — người dùng chỉ thấy "Couldn't add widget"
 * trong hộp thoại ghim và một ô trống trơn trên màn hình chính. Không có phép
 * kiểm nào khác trong kho này bắt được chuyện đó: aapt2 dựng bình thường, mọi
 * phép đo trên bản mô phỏng HTML vẫn đúng, APK vẫn cài được.
 *
 * Lỗi đã sửa: lưới bắt chạm 6×7 và ba vạch ngăn dùng thẻ <View> trần — 45 thẻ.
 * `android.view.View` KHÔNG có @RemoteView (cả `android.widget.Space` cũng
 * không), nên widget chưa bao giờ dựng nổi.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const RES = path.join(HERE, '..', 'app', 'src', 'main', 'res');

/**
 * Danh sách lấy THẲNG từ khung Android: quét mọi lớp trong android.view,
 * android.widget và android.app của android.jar API 34 (bản android-all của
 * Robolectric) tìm chú thích RemoteViews$RemoteView. Bỏ đi mấy lớp nội bộ của
 * hệ thống (NotificationHeaderView, NotificationTopLineView, chính RemoteViews)
 * vì ứng dụng không dùng tới.
 *
 * Đáng nhớ hai cái VẮNG MẶT: android.view.View và android.widget.Space.
 */
const ALLOWED = new Set([
    'ViewStub',
    'AbsoluteLayout', 'AdapterViewFlipper', 'AnalogClock', 'Button', 'CheckBox',
    'Chronometer', 'DateTimeView', 'FrameLayout', 'GridLayout', 'GridView',
    'ImageButton', 'ImageView', 'LinearLayout', 'ListView', 'ProgressBar',
    'RadioButton', 'RadioGroup', 'RelativeLayout', 'StackView', 'Switch',
    'TextClock', 'TextView', 'ViewFlipper',
]);

let pass = 0, fail = 0;
const ok = (what, cond, detail) => {
    cond ? pass++ : fail++;
    console.log(`  ${cond ? 'ok  ' : '✗ SAI'} ${what.padEnd(52)} ${cond ? '' : (detail || '')}`);
};

/** Tên thẻ của mọi phần tử trong một tệp bố cục, bỏ ghi chú. */
function tagsOf(xml) {
    const body = xml.replace(/<!--[\s\S]*?-->/g, '');
    return [...body.matchAll(/<([A-Za-z][\w.]*)/g)].map(m => m[1]);
}

const layouts = fs.readdirSync(path.join(RES, 'layout'))
    .filter(f => f.startsWith('widget_') && f.endsWith('.xml'));

console.log('Bố cục widget chỉ dùng lớp RemoteViews dựng được');
ok('có tìm thấy tệp bố cục widget', layouts.length >= 2, layouts.join(', '));

for (const f of layouts) {
    const xml = fs.readFileSync(path.join(RES, 'layout', f), 'utf8');
    const tags = tagsOf(xml);
    const bad = [...new Set(tags.filter(t => !ALLOWED.has(t)))];
    ok(`${f}: mọi thẻ đều được phép`, bad.length === 0,
        `thẻ không dựng được: ${bad.join(', ')}`);
    // <merge> không có gốc duy nhất, <include> thì lớp gốc của tệp được nhúng
    // vẫn phải qua bộ lọc — cả hai đều là bẫy, cấm hẳn cho gọn.
    ok(`${f}: không có <merge> hay <include>`,
        !/<\s*(merge|include)\b/.test(xml.replace(/<!--[\s\S]*?-->/g, '')));
    // Lớp tự viết (có dấu chấm trong tên) chắc chắn không có @RemoteView.
    ok(`${f}: không có lớp view tự viết`,
        !tags.some(t => t.includes('.')), tags.filter(t => t.includes('.')).join(', '));
}

// Bố cục mà appwidget-provider trỏ tới phải nằm trong số vừa kiểm.
const info = fs.readFileSync(path.join(RES, 'xml', 'calendar_widget_info.xml'), 'utf8');
for (const attr of ['initialLayout', 'previewLayout']) {
    const m = new RegExp(`android:${attr}="@layout/([\\w]+)"`).exec(info);
    ok(`appwidget-provider khai ${attr}`, !!m);
    if (m) ok(`  …trỏ tới bố cục đã kiểm ở trên`, layouts.includes(m[1] + '.xml'), m[1]);
}

// Hàng của ListView do RemoteViewsFactory dựng cũng đi qua đúng bộ lọc ấy.
const factory = fs.readFileSync(
    path.join(HERE, '..', 'app', 'src', 'main', 'java', 'com', 'bazi', 'qimen',
              'WidgetSectionService.kt'), 'utf8');
const rowLayout = /R\.layout\.(\w+)/.exec(factory);
ok('factory dựng hàng từ một bố cục đã kiểm',
    !!rowLayout && layouts.includes(rowLayout[1] + '.xml'), rowLayout && rowLayout[1]);

console.log(`\n${pass} đạt · ${fail} hỏng`);
console.log(fail ? '✗ HỎNG' : '✓ Widget dựng được: không thẻ nào ngoài danh sách RemoteViews');
process.exit(fail ? 1 : 0);
