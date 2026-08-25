# Bát Tự & Kỳ Môn — ứng dụng Android

Bản chuyển từ web app một-file (`QMDJ_1_1.html`) sang ứng dụng Android chạy
**hoàn toàn offline**, dùng **giờ Mặt Trời thật** và **dữ liệu Mặt Trăng thật**
của **bất kỳ toạ độ nào** để lập lá số.

*Android port of the single-file Qi Men Dun Jia / BaZi web app. Fully offline,
true-solar-time and real lunar data for any coordinate on Earth.*

---

## Điểm khác so với bản web

| | Bản web gốc | Bản Android |
|---|---|---|
| Vị trí | 28 thành phố cố định | GPS · 34.006 thành phố · nhập toạ độ tay |
| Múi giờ | theo danh sách cứng | IANA đầy đủ, có DST, suy được từ toạ độ khi offline |
| Mặt Trời | chỉ Chính Ngọ | Chính Ngọ, mọc/lặn, độ dài ngày, xích vĩ, lệch giờ MT thật |
| Mặt Trăng | điểm Sóc (bảng Âm Bàn) | thêm mọc/lặn, pha, % chiếu sáng, Sóc kế tiếp theo giờ địa phương |
| Màn hình | — | **giống hệt bản gốc**; bảng Nhật–Nguyệt ẩn, chạm "Chính Ngọ" mới hiện |
| Mạng | tải trong trình duyệt | **không có quyền INTERNET** |

Engine Bát Tự / Kỳ Môn **không bị sửa một dòng nào**. Lớp vị trí mới ghi toạ độ
đã chọn vào `countryData['__loc']` rồi trỏ `#country` sang khoá đó, nên
`processAll()` và toàn bộ ba phái (Trí Nhuận / Sách Bổ / Âm Bàn) chạy y hệt cũ,
chỉ khác là kinh độ và múi giờ giờ đây là của đúng nơi người dùng chọn.
Bộ kiểm thử đối chiếu từng lá số với bản web gốc để bảo đảm điều đó.

## Bảng Nhật–Nguyệt

Màn hình chính phải giống **y hệt** bản web gốc, nên bảng Mặt Trời / Mặt Trăng
mặc định **ẩn**. Chạm vào ô **Chính Ngọ** để mở hoặc đóng — Chính Ngọ chính là
giờ Mặt Trời thật mà bảng này diễn giải chi tiết. Lựa chọn được ghi nhớ.

Nút Back đóng theo thứ tự: hộp thoại phủ toàn màn hình trước, rồi mới tới bảng
Nhật–Nguyệt, cuối cùng mới thoát ứng dụng.

## Vì sao chạy được offline

Không có `android.permission.INTERNET` trong `AndroidManifest.xml` — đây là
bằng chứng kỹ thuật, không phải lời hứa: hệ điều hành sẽ chặn mọi kết nối ra
ngoài kể cả khi có mã cố tình gọi.

* `lunar.js`, CSS, JS, và CSDL thành phố nằm trong `assets/` của APK.
* Trang chạy trên `file:///android_asset/` nên `fetch()` bị CORS chặn — dữ liệu
  được đọc qua cầu native `QMDJNative.readAsset()`.
* GPS không cần mạng. Múi giờ của một toạ độ được suy ra bằng cách tra thành
  phố gần nhất trong CSDL đóng gói sẵn (kiểm thử đúng cho mọi ca thử).

## Tốc độ

Đo trên Node 22 (WebView của điện thoại nhanh hơn hoặc tương đương):

| Việc | Thời gian |
|---|---|
| Lập một lá số (`processAll`) | ~12 ms |
| Nạp CSDL 34.006 thành phố | ~70 ms, **nạp lười** — chỉ khi mở bảng chọn vị trí |
| Tìm kiếm thành phố | 1–5 ms |
| Tra thành phố gần nhất | ~9 ms |

CSDL thành phố không đụng tới lúc khởi động, nên màn hình đầu tiên chỉ tốn
thời gian phân tích `lunar.js` + một lần `processAll`.

## Độ chính xác thiên văn

`js/astro.js` được đối chiếu với **PyEphem** trên 9 toạ độ khắp thế giới
(xem `tools/test_astro.mjs`):

| Đại lượng | Sai lệch so với PyEphem |
|---|---|
| Chính Ngọ (giờ Mặt Trời thật = 12:00) | < 0,02 phút |
| Mặt Trời mọc / lặn | < 0,5 phút (< 1,5 phút ở vĩ độ ≥ 64°) |
| Mặt Trăng mọc / lặn | < 0,7 phút |
| Tỉ lệ chiếu sáng Mặt Trăng | < 0,3 % |
| Thời điểm Sóc / Vọng | < 0,6 phút |

Các ca ở vùng cực (Tromsø tháng 1, Reykjavík hạ chí) được xử lý riêng: ngày
không có mặt trời mọc/lặn, và ngày mặt trời lặn *sau* nửa đêm.

---

## Dựng ứng dụng

Cần **Android Studio** (hoặc Android SDK command-line tools) và **JDK 17+**.

```bash
cd android
./gradlew :app:assembleDebug      # APK gỡ lỗi
./gradlew :app:assembleRelease    # APK phát hành (chưa ký)
./gradlew :app:installDebug       # cài thẳng vào máy đang cắm USB
```

APK nằm ở `app/build/outputs/apk/`.

Mở bằng Android Studio: **File → Open** rồi chọn thư mục `android/`.

| | |
|---|---|
| `minSdk` | 24 (Android 7.0) |
| `targetSdk` / `compileSdk` | 35 (Android 15) |
| Ngôn ngữ | Kotlin, AGP 8.7.3, Gradle 8.14.3 |
| Phụ thuộc | chỉ `androidx.core:core-ktx` |

`assembleRelease` tạo APK **chưa ký**. Muốn ký thì thêm `signingConfigs` vào
`app/build.gradle.kts` hoặc dùng **Build → Generate Signed Bundle / APK**.

## Kiểm thử

Chạy headless bằng jsdom, không cần thiết bị hay giả lập:

```bash
cd android/tools
npm install
npm test
```

* `test_astro.mjs` — đối chiếu Mặt Trời/Mặt Trăng với giá trị PyEphem.
* `test_app.mjs` — nạp cả trang web trong jsdom với cầu native giả lập; kiểm
  Tứ Trụ (so với bản web gốc), tìm kiếm thành phố, suy múi giờ, luồng GPS,
  nhập toạ độ tay, nút Back.

### Đối chiếu 1-1 với bản web gốc

Nạp CẢ HAI bản trong jsdom rồi so từng trường hiển thị một:

```bash
node diff_vs_original.mjs /đường/dẫn/QMDJ_1_1.html 1000
node diff_vs_original.mjs /đường/dẫn/QMDJ_1_1.html 1000 987654321   # hạt giống khác
```

Bộ ca gồm các mốc dễ sai — ranh giới giờ Tý (22h–1h), quanh Lập Xuân và
Đông/Hạ Chí, ngày đổi giờ mùa hè ở châu Âu và Bắc Mỹ, 29/2 năm nhuận, cuối
tháng — cộng phần ngẫu nhiên phủ 1900–2100 × 28 vị trí × 3 phái × 2 ngôn ngữ.

So sánh 35 trường mỗi ca: 4 trụ can/chi, nạp âm, Chính Ngọ, Tiết khí, Cục,
Tuần thủ, Trực Phù/Trực Sử, lịch âm, **toàn bộ HTML của bàn Kỳ Môn 9 cung**, và
bảng chi tiết của cả ba phái.

Kết quả: **2000 ca (2 hạt giống), 70.000 trường, 0 khác biệt.**

### Đối chiếu MÀN HÌNH với bản web gốc

```bash
node test_visual_parity.mjs /đường/dẫn/QMDJ_1_1.html
```

Dựng cây DOM chỉ gồm phần **thực sự nhìn thấy** (theo computed style, nên phần
bị CSS ẩn không tính) của cả hai bản rồi so từng phần tử một — bắt được mọi
thứ thừa hoặc thiếu. Kết quả: **246–249 phần tử, giống hệt** trên cả ba ca
(hai ngôn ngữ × ba phái).

## Sinh lại CSDL thành phố

```bash
pip install geonamescache pytz
python3 android/tools/build_cities.py
```

---

## Cấu trúc

```
android/
├── app/src/main/
│   ├── AndroidManifest.xml          không có quyền INTERNET
│   ├── java/com/bazi/qimen/
│   │   ├── MainActivity.kt          WebView + window insets + nút Back
│   │   └── WebAppBridge.kt          @JavascriptInterface: assets, prefs, GPS
│   ├── res/                         icon, theme, quy tắc sao lưu
│   └── assets/web/
│       ├── index.html               khung trang + bảng chọn vị trí + bảng Nhật–Nguyệt
│       ├── css/app.css              CSS của bản gốc, giữ nguyên
│       ├── css/location.css         phần giao diện mới
│       ├── js/lunar.js              thư viện lịch âm của 6tail, giữ nguyên
│       ├── js/app.js                engine Bát Tự / Kỳ Môn của bản gốc
│       ├── js/astro.js              MỚI — Mặt Trời & Mặt Trăng theo toạ độ
│       ├── js/location.js           MỚI — GPS, tra thành phố, toạ độ tay
│       └── data/cities.txt          34.006 thành phố + múi giờ IANA
└── tools/                           bộ sinh dữ liệu và kiểm thử
```

Ghi công thư viện và dữ liệu bên thứ ba: xem [`NOTICE.md`](NOTICE.md).
