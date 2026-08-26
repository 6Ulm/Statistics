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

## Vừa khít màn hình

Bản gốc chỉ có một luật: `@media (min-width: 768px) { body { zoom: 1.25 } }`.
Luật này chỉ nhìn **chiều rộng**, nên S21 **xoay ngang** (800×360) rộng hơn
768px và bị phóng to 1,25 lần — trong khi màn hình chỉ cao 360px.

`js/viewport.js` thay bằng hệ số tính từ **cả hai chiều**, đo thực tế chứ không
đoán, rồi kẹp trong khoảng 0,95–1,6. Bố cục không đổi, chỉ to/nhỏ theo màn hình.

| Máy | Trước | Sau |
|---|---|---|
| S21 dọc 360×800 | zoom 1 · bàn 348px | **không đổi** — vốn đã vừa khít |
| S21 ngang 800×360 | zoom 1,25 · bàn **500px** | zoom 0,95 · bàn **380px** |
| Z Fold mở 673×841 | zoom 1 · bàn 400px, thừa 273px hai bên | zoom 1,13 · bàn **452px**, lấp đầy |
| Điện thoại nhỏ 320×568 | zoom 1 | zoom 0,95 |

S21 dọc **không cần sửa gì**: đo bằng Chromium cho thấy không tràn ngang, không
có chữ nào bị cắt. Khoảng trống ~13% ở đáy màn hình là do bàn Kỳ Môn hình vuông
và đã chiếm trọn bề ngang — phóng to nữa sẽ tràn ngang, nên giữ nguyên tỉ lệ 1.

Bàn phím ảo làm `innerHeight` tụt một nửa; module bỏ qua lúc đó để giao diện
không co lại khi đang gõ tìm thành phố.

## Tab Lịch

Thanh tab đáy màn hình có hai mục: **Kỳ Môn** (bàn Kỳ Môn, đúng như bản web
gốc) và **Lịch** (lịch âm dương).

* Mỗi ô ghi ngày dương (to), ngày âm (nhỏ, mùng 1 kèm tháng) và can chi —
  **can một dòng, chi một dòng** ở mọi ngày, không phụ thuộc độ dài tên.
* Ô trống đầu/cuối lưới được điền bằng ngày của **tháng trước / tháng sau**, tô
  mờ; chạm vào là nhảy sang tháng đó.
* **Hôm nay** có viền đỏ đậm trên nền vàng nhạt — tô đặc màu đỏ thì nổi hơn
  thật, nhưng chữ phải đảo sang trắng và ô hoá thành một mảng đặc, đọc ngày âm
  với can chi khó hơn hẳn.
* Bảng **tiết khí trong năm** nằm ngay dưới lưới, không có hộp tiêu đề gập/mở:
  xếp hai cột thì cả 24 mục vừa một màn hình trên S21, S21 FE, A51, S21 Ultra và
  Z Fold, chẳng còn gì để gập. `fitGrid()` đo chiều cao THẬT của bảng rồi mới
  chia phần còn lại cho các hàng lịch — giữ sẵn một khoản cố định thì phần dư
  hoá thành khoảng hở ở đáy màn hình. Máy quá thấp (320×520) thì bảng tự cuộn
  và hàng tiêu đề dính lại.
* Hai tab ngăn nhau bằng một vạch dọc, tab đang mở có **nền màu nhấn** chứ
  không chỉ đổi màu chữ.

Lịch âm được tính theo **UTC+7** như quy ước lịch Việt Nam (bản tiếng Trung
dùng UTC+8) — đó chính là lý do Tết ta và Tết Tàu thỉnh thoảng lệch một ngày.
Phải đặt lại mốc này mỗi lần vẽ: `processAll()` để lại múi giờ của địa điểm
đang chọn trong biến toàn cục của `lunar.js`, nên nếu đang chọn Paris thì
26/08/2026 hoá ra 15/7 thay vì 14/7.

## Tiết khí: một nguyên tắc duy nhất

Tiết khí xuất hiện ở ba chỗ — bảng Sách Bổ pháp (tab Kỳ Môn), bảng Tiết khí
(tab Lịch) và widget — và cả ba phải ra **cùng một con số**. Nguyên tắc lấy theo
bảng Sách Bổ pháp:

1. `LunarYear.getJieQiJulianDays()`, không phải `getJieQiTable()`;
2. tính ở mốc **UTC+8** (`setTzOffsetHours(null)`);
3. rồi mới quy sang giờ địa phương: `jdLocal = jdUTC8 + (tz − 8)/24`, với `tz`
   tra theo DST tại **chính thời điểm** của mốc đó.

Bước 2 và 3 không phải chuyện vặt: tính thẳng ở múi giờ địa phương thì các tiết
khí mùa đông của một nước có DST bị cộng nhầm offset mùa hè, lệch đúng một giờ.

Tab Lịch gọi thẳng `sb_getJieQiDates` của tab Kỳ Môn thay vì chép lại — nhưng nó
chạy khi `ShouXingUtil` đang ở mốc UTC+7 (để lịch âm ra đúng lịch Việt Nam), mà
`findJieQi` bên trong lại đọc chính biến toàn cục ấy, nên phải đặt lại mốc rồi
trả về như cũ. `tools/test_jieqi_parity.mjs` so hai bảng từng mục một ở năm múi
giờ khác nhau; `tools/test_lunar_table.mjs` canh bảng của widget theo cùng
nguyên tắc.

Ở tab Lịch, bảng bỏ hai cột **Độn** và **Số Cục** (đó là chuyện của bàn Kỳ Môn)
nên mỗi mục chỉ còn tên với ngày giờ — hẹp bằng nửa bề ngang. 24 mục vì thế xếp
thành **hai cột kép**, 12 mục mỗi bên: bảng thấp đi một nửa, cả năm hiện gọn
trong một màn hình, và ranh giới trái/phải trùng luôn ranh giới Dương Độn / Âm
Độn. Cột tên co đúng bằng chữ và cột ngày căn trái ngay sau nó, để tiêu đề
"Dương lịch" thẳng hàng với giá trị bên dưới thay vì bị đẩy sát mép máy.

## Widget lịch trên màn hình chính

Ghim riêng **lịch âm** ra màn hình chính, không cần mở ứng dụng. Trong tab
Lịch, bấm **📌 Ghim lịch ra màn hình chính** (Android 8 trở lên; launcher cũ thì
nhấn giữ màn hình chính → Tiện ích → "Lịch âm"). Chạm vào widget mở thẳng tab
Lịch, không phải bàn Kỳ Môn.

Widget có **đúng thiết kế của tab Lịch** — cùng màu, cùng cách sắp chữ, cùng
kiểu đánh dấu hôm nay — chỉ bỏ thanh tab và nút ghim. Nội dung gồm lưới lịch và
bảng tiết khí, không có gì khác.

Hai mũi tên **‹ ›** lùi/tiến tháng, chạm tiêu đề thì về tháng hiện tại; tháng
đang xem được nhớ riêng cho **từng widget** (`qmdj_widget` / `w<id>.offset`), nên
ghim hai cái cạnh nhau vẫn xem được hai tháng khác nhau. Ba nút dùng ba
`requestCode` khác nhau (`id * 8 + 1|2|3`), nếu không hệ thống dùng lại cùng một
`PendingIntent` và cả ba cùng làm một việc.

Thanh tiêu đề là **View thật** (`widget_calendar.xml`) để hai mũi tên bấm được;
phần lưới và bảng tiết khí vẽ ra bitmap vì RemoteViews không dựng nổi lưới 7×6
cho gọn.

Sàn kích thước là **4×4 ô** (`minResizeWidth/Height` = `minWidth/Height` =
250dp): kéo to ra thì được, thu nhỏ hơn thì không. Ở 3×2 hay 4×3 ô, mỗi ô lịch
chỉ cao chừng 15dp nên can chi tự tắt và số ngày còn khoảng 5dp — không đọc nổi,
nên đơn giản là chặn hẳn thay vì để người dùng dựng ra một widget vô dụng.

Can chi được đặt thành một **khối hai dòng sát nhau, cân giữa** phần còn lại của
ô — đúng như `.cal-gz` trong tab Lịch. Đặt theo tỉ lệ phần trăm của phần còn lại
thì ô càng cao hai chữ càng dạt xa nhau: ở widget 4×5 khoảng cách giãn ra hơn
gấp đôi cỡ chữ.

Widget vẽ bằng RemoteViews nên **không có WebView** — `lunar.js` không với tới
được. Thay vì chép thuật toán tính điểm Sóc và giờ giao tiết sang Kotlin (dễ
lệch với phần còn lại của ứng dụng), mọi mốc mùng 1 và mọi tiết khí từ
1900–2100 được tính sẵn bằng chính `lunar.js` rồi đóng gói thành hai bảng tra
(43 KB + 71 KB); Kotlin chỉ tìm nhị phân. Can chi suy thẳng từ số ngày Julius.

```bash
node tools/build_lunar_table.mjs   # sinh assets/lunar_months.txt + jieqi.txt
node tools/test_lunar_table.mjs    # đối chiếu với lunar.js
```

Bảng tra và cách tra đã đối chiếu **từng ngày một trong 73.414 ngày (1900–2100)
và 4.824 mốc tiết khí, lệch 0**.

Widget chỉ hiện tiết khí **của tháng đang xem** (một hoặc hai mục), không hiện
cả 24 mục như tab Lịch: widget không cuộn và không gập được, nhét 24 dòng vào
đó thì chữ bé đến mức vô dụng.

Xem trước widget mà không cần dựng APK:

```bash
node tools/shot_widget.mjs         # ảnh widget ở 4 kích thước
```

`tools/widget_preview.html` vẽ lại y hệt `drawBody()` bằng Canvas của trình
duyệt (cùng mô hình vẽ với Canvas của Android) và đọc **chính hai tệp assets mà
widget dùng**, nên bản xem trước không thể lệch với widget thật.

Widget tự vẽ lại sau nửa đêm bằng một báo thức lặp không chính xác — chỉ cần
đúng ngày, đỡ tốn pin hơn nhiều so với đánh thức nửa tiếng một lần.

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

### Bố cục trên nhiều kích thước màn hình

```bash
npm install playwright && npx playwright install chromium
node test_responsive.mjs
```

Mở trang bằng Chromium thật (WebView Android cũng là Chromium) ở 7 kích thước —
S21, **S21 FE** (393×790 @2,75x), **A51** (412×852 @2,625x), S21 Ultra, S21 xoay
ngang, một máy nhỏ 320×520 và Z Fold mở — rồi bắt: tràn ngang, chữ bị cắt bởi
ellipsis, và phóng to trong khi nội dung đã phải cuộn. Chiều cao ở đây là chiều
cao **WebView thật** (đã trừ thanh trạng thái và thanh điều hướng), không phải
chiều cao màn hình.

`TAB=cal node test_responsive.mjs` chạy lại cùng bộ đó trên tab Lịch, và **bật
nút Ghim lên trước khi đo**. Nút ấy `display:none` ngoài ứng dụng Android, nên
mọi phép đo trên trình duyệt vốn không thấy nó — bố cục trên máy thật vì thế cao
hơn phép thử tưởng và nút bị thanh tab cố định che mất. Phép thử giờ canh thêm
hai điều: không phần tử nào bị thanh tab che khi trang vừa màn hình, và **tab
Lịch trên điện thoại dựng đứng phải vừa đúng một màn hình** — tràn ra là dấu
hiệu `fitGrid()` quên trừ một khối nào đó, mà `viewport.js` sẽ che lỗi ấy bằng
cách thu nhỏ cả trang xuống đáy 0,95.

Gỡ `viewport.js` ra thì ca "S21 ngang" lập tức đỏ — nên phép thử này có thật,
không phải lúc nào cũng xanh. Trước đây nó **đọc `body.style.zoom`** (thuộc tính
inline, luôn rỗng) nên vẫn xanh với cả bản hỏng; giờ đọc computed style.

### Tiết khí giữa các bảng

```bash
node test_jieqi_parity.mjs
```

Mở ứng dụng ở năm múi giờ khác nhau, đọc bảng Sách Bổ pháp ở tab Kỳ Môn và bảng
tiết khí ở tab Lịch, rồi so **từng tên và từng mốc giờ**. Cũng kiểm mục được tô
đậm: hai bên chỉ được lệch tối đa một mục, đúng vào ngày giao tiết (tab Kỳ Môn
lấy cả giờ phút đang nhập, tab Lịch chỉ có độ phân giải một ngày).

### Ảnh chụp màn hình

```bash
node tools/shot_calendar.mjs       # tab Lịch + tab Kỳ Môn trên S21 / S21 FE / A51
```

Chụp bằng Chromium ở đúng kích thước WebView của từng máy: tiết khí mở, tiết khí
gập, tháng sau, và tab Kỳ Môn — đủ để kiểm bằng mắt mà không phải cài APK.

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
│   │   ├── WebAppBridge.kt          @JavascriptInterface: assets, prefs, GPS
│   │   ├── LunarTable.kt            tra âm lịch + tiết khí cho widget
│   │   └── CalendarWidgetProvider.kt  widget màn hình chính
│   ├── res/                         icon, theme, layout widget, quy tắc sao lưu
│   ├── assets/lunar_months.txt      mốc mùng 1, 1900–2100 (sinh sẵn)
│   ├── assets/jieqi.txt             4.823 mốc tiết khí (sinh sẵn)
│   └── assets/web/
│       ├── index.html               khung trang + bảng chọn vị trí + bảng Nhật–Nguyệt
│       ├── css/app.css              CSS của bản gốc, giữ nguyên
│       ├── css/location.css         phần giao diện mới
│       ├── css/calendar.css         MỚI — thanh tab + lịch âm dương
│       ├── js/lunar.js              thư viện lịch âm của 6tail, giữ nguyên
│       ├── js/app.js                engine Bát Tự / Kỳ Môn của bản gốc
│       ├── js/astro.js              MỚI — Mặt Trời & Mặt Trăng theo toạ độ
│       ├── js/location.js           MỚI — GPS, tra thành phố, toạ độ tay
│       ├── js/viewport.js           MỚI — vừa khít mọi kích thước màn hình
│       ├── js/calendar.js           MỚI — tab Lịch: lưới, tiết khí, ghim widget
│       └── data/cities.txt          34.006 thành phố + múi giờ IANA
└── tools/                           bộ sinh dữ liệu và kiểm thử
```

Ghi công thư viện và dữ liệu bên thứ ba: xem [`NOTICE.md`](NOTICE.md).
