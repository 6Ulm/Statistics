/**
 * Kiểm thử toàn bộ trang web đóng gói trong assets, chạy headless bằng jsdom
 * với cầu native giả lập.
 *
 * Chạy:  cd android/tools && npm install && node test_app.mjs
 *
 * Các giá trị Tứ Trụ mong đợi dưới đây được đối chiếu với bản web gốc (một
 * file HTML duy nhất) trước khi tách thành assets — bảo đảm việc tách file và
 * lớp vị trí mới KHÔNG làm đổi kết quả của engine.
 */
import fs from 'fs';
import path from 'path';
import { fileURLToPath } from 'url';
import { JSDOM, VirtualConsole } from 'jsdom';

const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');

const PILLARS = [
    // y, m, d, h, min, vị trí, phương pháp, Tứ Trụ mong đợi
    [1988, 3, 15, 7, 25, 'VN-HCM', 'amban', '戊辰乙卯己巳戊辰'],
    [2026, 8, 25, 23, 40, 'VN-HN', 'trinhuan', '丙午丙申壬申庚子'],
    [2000, 2, 4, 12, 0, 'CN', 'bophap', '己卯丁丑壬辰丙午'],
    [1975, 12, 31, 0, 5, 'FR', 'amban', '乙卯戊子辛亥戊子'],
    [2019, 6, 21, 18, 30, 'US_ET', 'trinhuan', '己亥庚午己丑癸酉'],
    [2033, 11, 3, 4, 15, 'JP', 'bophap', '癸丑壬戌戊午甲寅'],
];

const SEARCHES = [
    ['ha noi', 'Hanoi'], ['hà nội', 'Hanoi'], ['saigon', 'Ho Chi Minh City'],
    ['北京', 'Beijing'], ['da lat', 'Ðà Lạt'], ['tokyo', 'Tokyo'],
    ['new york', 'New York City'], ['vung tau', 'Vũng Tàu'],
];

const TZ_GUESSES = [
    [21.0278, 105.8342, 'Asia/Ho_Chi_Minh'],
    [48.8566, 2.3522, 'Europe/Paris'],
    [-33.8688, 151.2093, 'Australia/Sydney'],
    [40.7128, -74.0060, 'America/New_York'],
    [35.6895, 139.6917, 'Asia/Tokyo'],
];

let fail = 0, checks = 0;
function check(label, got, want) {
    checks++;
    const ok = got === want;
    if (!ok) fail++;
    console.log(`  ${ok ? 'ok  ' : 'FAIL'} ${label.padEnd(34)} ${ok ? got : `got=${got}  want=${want}`}`);
}

const errors = [];
const vc = new VirtualConsole();
vc.on('jsdomError', e => errors.push(e.message));
vc.on('error', (...a) => errors.push(a.join(' ')));

const dom = new JSDOM(fs.readFileSync(path.join(WEB, 'index.html'), 'utf8'), {
    url: 'file://' + path.join(WEB, 'index.html'),
    runScripts: 'dangerously', resources: 'usable', pretendToBeVisual: true,
    virtualConsole: vc,
});
const store = {};
let gpsCalls = 0;
dom.window.QMDJNative = {
    readAsset: p => fs.readFileSync(path.join(WEB, p), 'utf8'),
    getPref: k => (k in store ? store[k] : null),
    setPref: (k, v) => { store[k] = v; },
    deviceTimeZone: () => 'Asia/Ho_Chi_Minh',
    hasLocationPermission: () => true,
    platform: () => 'android',
    requestLocation: () => {
        gpsCalls++;
        setTimeout(() => dom.window.__onNativeLocation(
            { lat: 16.0678, lon: 108.2208, accuracy: 12, tzId: 'Asia/Ho_Chi_Minh' }), 5);
    },
};
await new Promise(r => dom.window.addEventListener('load', r));
await new Promise(r => setTimeout(r, 400));
const w = dom.window, doc = w.document;
const val = id => { const e = doc.getElementById(id); return e ? (e.innerText ?? e.textContent) : ''; };

console.log('Tứ Trụ (đối chiếu bản web gốc)');
for (const [y, m, d, h, mi, loc, method, want] of PILLARS) {
    for (const [id, v] of [['inYear', y], ['inMonth', m], ['inDay', d], ['solarHour', h],
                           ['solarMinute', mi], ['country', loc], ['methodSelect', method]]) {
        doc.getElementById(id).value = String(v);
    }
    w.processAll();
    const got = ['ttCanNam', 'ttChiNam', 'ttCanThang', 'ttChiThang',
                 'ttCanNgay', 'ttChiNgay', 'ttCanGio', 'ttChiGio'].map(val).join('');
    check(`${y}-${m}-${d} ${h}:${mi} ${loc}/${method}`, got, want);
}

console.log('\nTra cứu thành phố offline');
await w.QMDJLocation.loadCities();
for (const [q, want] of SEARCHES) {
    const r = w.QMDJLocation.searchCities(q, 5);
    check(`search ${JSON.stringify(q)}`, r.length ? r[0].name : '(none)', want);
}

console.log('\nSuy múi giờ từ toạ độ (offline, không cần mạng)');
for (const [lat, lon, want] of TZ_GUESSES) {
    check(`${lat},${lon}`, w.QMDJLocation.guessTz(lat, lon), want);
}

console.log('\nGPS → tính lại lá số');
doc.getElementById('locGpsBtn').dispatchEvent(new w.Event('click'));
await new Promise(r => setTimeout(r, 300));
check('gọi cầu native', String(gpsCalls), '1');
check('toạ độ hiển thị', val('out-astro-coord'), '16.07°N 108.22°E');
check('vị trí được lưu', String(JSON.parse(store['qmdj.location']).tzId), 'Asia/Ho_Chi_Minh');

console.log('\nNhập toạ độ thủ công');
w.openCountryPicker();
await new Promise(r => setTimeout(r, 50));
doc.getElementById('locLat').value = '21.0278';
doc.getElementById('locLon').value = '105.8342';
doc.getElementById('locTz').value = 'Asia/Ho_Chi_Minh';
doc.getElementById('locApply').dispatchEvent(new w.Event('click'));
await new Promise(r => setTimeout(r, 100));
check('múi giờ áp dụng', val('out-chinhngo').split(' ')[1], '(GMT+7)');

console.log('\nBảng Nhật–Nguyệt (mặc định ẩn để giống bản web gốc)');
check('mặc định ẩn', doc.getElementById('astroPanel').style.display, 'none');
doc.querySelector('.info-pair-chinhngo').dispatchEvent(new w.Event('click'));
check('chạm Chính Ngọ thì hiện', doc.getElementById('astroPanel').style.display, 'block');
check('có dữ liệu Mặt Trời', val('out-astro-sun').length > 5 ? 'yes' : 'no', 'yes');
check('có dữ liệu Mặt Trăng', val('out-astro-moon').length > 5 ? 'yes' : 'no', 'yes');
check('Back đóng bảng', String(w.__onBackPressed()), 'true');
check('đã đóng lại', doc.getElementById('astroPanel').style.display, 'none');

console.log('\nĐổi ngôn ngữ');
doc.querySelector('.info-pair-chinhngo').dispatchEvent(new w.Event('click'));  // mở lại
w.setLang('vi');
check('nhãn Mặt Trời (vi)', val('lblAstroSun'), 'Mặt Trời:');
check('giá trị đổi theo (vi)', val('out-astro-sun').slice(0, 3), 'mọc');
w.setLang('zh');
check('nhãn Mặt Trời (zh)', val('lblAstroSun'), '太阳:');
check('giá trị đổi theo (zh)', val('out-astro-sun').slice(0, 1), '出');

console.log('\nNút Back của Android');
w.openCountryPicker();
check('1. đóng hộp thoại vị trí trước', String(w.__onBackPressed()), 'true');
check('   hộp thoại đã đóng', String(doc.getElementById('locOverlay').classList.contains('open')), 'false');
check('   bảng Nhật–Nguyệt còn mở', doc.getElementById('astroPanel').style.display, 'block');
check('2. rồi mới đóng bảng Nhật–Nguyệt', String(w.__onBackPressed()), 'true');
check('3. không còn gì để đóng', String(w.__onBackPressed()), 'false');

console.log('\nLỗi JS trong lúc chạy:', errors.length ? errors.slice(0, 5) : 'không có');
if (errors.length) fail++;
console.log(`\n${checks - fail}/${checks} phép kiểm đạt.`);
process.exit(fail ? 1 : 0);
