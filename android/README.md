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

## Ngày âm lịch bắt đầu lúc nào

Hai câu hỏi tách rời nhau: **mốc nào** (kinh tuyến nào) và **lúc mấy giờ** (ranh
giới ngày).

### Ranh giới: Chính Tý, không phải 00:00

Mùng 1 là ngày CHỨA điểm Sóc, và ngày ở đây đếm từ **Chính Tý tới Chính Tý** —
nửa đêm MẶT TRỜI THẬT (Chính Ngọ − 12h) tại nơi người dùng đứng.

Đây là chuyện **quy ước, không phải đúng/sai**. Lịch pháp Trung–Việt định ngày
từ nửa đêm đồng hồ tới nửa đêm đồng hồ tại kinh tuyến quy chiếu, và mọi cuốn
lịch in đều theo luật ấy. Ứng dụng này chọn nửa đêm thật, cùng hệ với Chính Ngọ
mà nó vẫn hiển thị.

Không lấy ranh giới **đầu giờ Tý** (Chính Ngọ − 13h): đó là quy ước của mệnh lý
cho trụ ngày, và bản thân nó còn hai phái (早子時 / 夜子時). Nửa đêm thật thì chỉ
có một.

Chính Ngọ lệch khỏi 12:00 vì kinh độ + phương trình thời gian + giờ mùa hè, nên
Chính Tý lệch khỏi 00:00 đúng chừng ấy:

| Nơi | Chính Ngọ | Chính Tý | Cửa sổ lệch |
|---|---|---|---|
| Hà Nội | 12:01 | **00:01** | 1 phút |
| Paris (hè) | 13:55 | **01:55** | 115 phút |

Cửa sổ lệch không đứng yên: phương trình thời gian kéo Chính Tý của Hà Nội dao
động từ **−17,5 phút đến +6 phút** quanh 00:00 trong năm. Nên ngay cả ở Việt
Nam vẫn có **0,96% số tháng** (36/3741, quét 1960–2060) rơi mùng 1 khác lịch in;
nơi lệch xa kinh tuyến múi giờ của mình thì chừng 8%. Ví dụ Paris: Sóc
06/07/2024 lúc 00:57 vẫn còn **trước** Chính Tý (01:55), nên mùng 1 là 05/07 chứ
không phải 06/07.

Phần dôi ra ấy là do CHÍNH LUẬT NÀY, không phải sai số tính toán: điểm Sóc mà
ứng dụng hiển thị rơi đúng ngày mùng 1 của `lunar.js` ở **1744/1744 tháng** từ
1960 trở đi (mốc UTC+7).

### Điểm Sóc được TÍNH, và tính bằng đúng hàm của lunar.js

`getPreciseSocSolarUTC8()` không tra bảng: nó gọi thẳng `ShouXingUtil.shuoHigh`,
tức chính hàm mà `lunar.js` dùng để định mốc Sóc. `mo.getFirstJulianDay()` chỉ
dùng để chọn số thứ tự tuần trăng, không cung cấp giờ.

Trước đây chỗ này CHÉP LẠI công thức của `shuoHigh` — và chép thiếu một bước:

```js
var v = ((t + 0.5) % 1) * SECOND_PER_DAY;
if (v < 1800 || v > SECOND_PER_DAY - 1800) {
    t = this.msaLonT(w) * 36525 - this.dtT(t) + tzDay;   // chính xác hơn
}
```

Khi điểm Sóc rơi trong vòng 30 phút quanh nửa đêm, `shuoHigh` giải lại bằng
`msaLonT` thay cho `msaLonT2`. Mà sát nửa đêm chính là lúc quyết định mùng 1 rơi
ngày nào — bỏ bước ấy là sai đúng chỗ có hại nhất. Gọi thẳng hàm gốc đưa số
tháng lệch từ 1960 trở đi **từ 2 về 0**.

### Trước 1960: thiên văn khác sử liệu

`ShouXingUtil.calcShuo` đổi chế độ tại **JD 2436935 = 01/01/1960**:

* **từ 1960**: `shuoHigh` — tính thiên văn thuần;
* **trước đó**: tra bảng `SHUO_KB` và `shuoLow` kèm chuỗi sửa `SB` — tức chép
  lại LỊCH SỬ, ghi đúng những gì lịch chính thức ngày ấy đã ban, kể cả chỗ nó
  sai so với thiên văn.

Ứng dụng luôn tính thiên văn, nên với ngày trước 1960 mốc mùng 1 có thể lệch
**18/742 tháng (2,43%)** so với cấu trúc tháng của `lunar.js`. Không hàm thiên
văn nào khớp được chỗ đó — đấy là sử liệu, không phải phép tính.

### Nhãn tháng lấy ở mốc quy chiếu, không lấy ở chỗ đứng

Hai câu hỏi tách bạch:

* **"Tháng này là tháng mấy, tháng nào nhuận"** là QUY ƯỚC LỊCH. Nó do luật
  "tháng không có trung khí là tháng nhuận" quyết, và luật ấy được định tại
  kinh tuyến quy chiếu — **UTC+7 cho lịch ta, UTC+8 cho lịch Tàu** (chính chỗ
  này làm Tết ta và Tết Tàu thỉnh thoảng lệch một ngày).
* **"Mùng 1 rơi vào ngày dương nào"** mới là chuyện địa phương: ngày chứa điểm
  Sóc, đếm từ Chính Tý.

Trước đây hỏi `lunar.js` ngay ở mốc địa phương, tức để luật trung khí bị đánh
giá trên lưới nửa đêm ĐỒNG HỒ ở một offset nguyên giờ. Mà Chính Tý lại xê dịch
tới ~30 phút trong năm, nên một offset cố định không diễn tả nổi nó — đo được
**4–8 tháng mỗi thế kỷ đổi nhãn** chỉ vì mốc lệch 15–30 phút. Nay nhãn không còn
phụ thuộc chuyện đó.

Ghép nhãn (mốc quy chiếu) với mốc bắt đầu (địa phương) là an toàn vì **dãy tuần
trăng giống hệt nhau ở mọi mốc**: quét 1900–2100, mọi mốc từ UTC−8 tới UTC+12
đều ra **đúng 2486 tháng**, mốc bắt đầu lệch **tối đa 1 ngày**, không cặp nào
lệch quá. Độ dài tháng vẫn 29 hoặc 30 ngày ở mọi thành phố đã thử.

## Mốc kinh tuyến

Quy tắc: **mùng 1 là ngày CHỨA điểm Sóc**. Nhưng "ngày" nào thì tuỳ mốc quy
chiếu — và mốc ấy phải trùng với mốc dùng để HIỆN giờ Sóc, nếu không một màn
hình có hai hệ quy chiếu.

Bản web gốc mắc đúng chỗ này: nó tính ngày âm ở mốc UTC+8 nhưng lại quy giờ Sóc
sang giờ địa phương, nên ở Paris bảng Âm Bàn ghi **Sóc 12-08-2026 19:37** ngay
cạnh **Mùng 1 13-08-2026** — hai con số cùng một thời điểm (17:37 UTC) nhưng đọc
ở hai hệ khác nhau.

Bản Android tính ngày âm ở **mốc múi giờ của địa điểm đang chọn**. Ở Paris mùng
1 là 12/08, khớp với giờ Sóc đang hiện; ở Việt Nam vẫn là 13/08 như cũ.

Đã kiểm lại lời cảnh báo trong mã gốc rằng mốc địa phương làm hỏng độ dài tháng
29/30: **không đúng**. Quét 2020–2035 ở các mốc từ UTC−8 tới UTC+12, mọi tháng
đều 29 hoặc 30 ngày, số ngày âm liên tục, và mùng 1 luôn chứa Sóc (198/198 tháng
mỗi mốc).

Chỉ NGÀY ÂM LỊCH đổi mốc.

### Giờ Bắc Kinh ở đây KHÔNG phải một quy ước

Dễ hiểu nhầm chỗ này, nên nói cho rõ. Trụ **năm** và trụ **tháng** đổi tại Lập
Xuân và 12 mốc Tiết — mà đó đều là những THỜI ĐIỂM tuyệt đối, không phải ngày
trên lịch. Thời điểm bạn nhập cũng vậy. So thời điểm với thời điểm thì kết quả
**không phụ thuộc hệ quy chiếu**.

`_readInputBJ()` đổi giờ bạn nhập sang giờ Bắc Kinh
(`Date.UTC(...) − tz·3600000 + 8·3600000`) chỉ để đặt cả hai vế về CÙNG một hệ
cho tiện so; làm ở giờ địa phương cũng ra y hệt. Kiểm chứng: cùng một thời điểm
tuyệt đối đọc từ hai nơi cho cùng trụ năm/tháng, ngay hai bên mốc Lập Xuân 2026.

| Thời điểm | Nơi | Năm | Tháng | Ngày | Giờ |
|---|---|---|---|---|---|
| ngay sau Lập Xuân | Hà Nội 04/02 03:30 | Bính Ngọ | Canh Dần | Kỷ Dậu | Bính Dần |
| ngay sau Lập Xuân | Paris 03/02 21:30 | Bính Ngọ | Canh Dần | Mậu Thân | Nhâm Tuất |
| ngay trước Lập Xuân | Hà Nội 04/02 02:30 | Ất Tỵ | Kỷ Sửu | Kỷ Dậu | Ất Sửu |
| ngay trước Lập Xuân | Paris 03/02 20:30 | Ất Tỵ | Kỷ Sửu | Mậu Thân | Nhâm Tuất |

Năm và tháng trùng khít; **ngày và giờ mới khác** — đúng như thiết kế, vì hai
trụ ấy dùng giờ Mặt Trời thật tại chỗ.

Bảng tiết khí cũng vậy: **tính** ở mốc UTC+8 rồi **hiện ra ở giờ địa phương**.
Cùng mốc Lập Xuân 2026, Hà Nội ghi `04-02-2026 03:02` còn Paris ghi
`03-02-2026 21:02` — chênh đúng 6 giờ, cùng một thời điểm.

### Cái giá phải trả (đổi mốc kinh tuyến)

Cục Âm Bàn = `(chi năm + tháng âm + ngày âm + chi giờ) % 9`, nên đổi mốc ngày âm
là đổi cả kết quả Kỳ Môn ở nơi lệch khỏi UTC+8:

| Nơi | Ngày âm khác bản gốc | Cục Âm Bàn khác |
|---|---|---|
| Việt Nam UTC+7 | 0 % | **0 %** |
| Paris (hè) | 22,9 % | **22,9 %** |
| Paris (đông) | 29,1 % | **29,1 %** |
| New York | 56,1 % | **55,6 %** |

`diff_vs_original.mjs` vì thế đòi **trùng khít tuyệt đối ở nơi có mốc UTC+8**
(tra múi giờ thật theo từng thời điểm — Malaysia từng ở UTC+7:30 tới 1982), còn
nơi khác thì miễn cho các trường phụ thuộc ngày âm, và chỉ miễn cho bàn Kỳ Môn
**khi chính cục đã khác** — cục giống mà bàn khác vẫn là hồi quy.

## Engine thiên văn dùng chung (`js/ephem.js`)

Tab Kỳ Môn và tab Lịch cần cùng những mốc thiên văn — Sóc, Vọng, tiết khí,
Chính Ngọ — nên tất cả nằm ở **một chỗ**, và cả hai gọi vào đó.

Trước khi gom, có hai vấn đề thật:

* **Phương trình thời gian có HAI bản** — `getEquationOfTime` trong `app.js` và
  `equationOfTime` trong `astro.js`. Đối chiếu Meeus ví dụ 28.b thì cả hai đều
  đúng tới **0,04 giây**; chúng lệch nhau vì **thời điểm đánh giá**: bản cũ tính
  EoT tại **12:00 UTC** của ngày đó thay vì tại chính lúc Chính Ngọ địa phương.
  Với Nhật (UTC+9) hay Mỹ (UTC−5) thì lệch tới 9 giờ, đủ đổi EoT ~9 giây. Nay
  chỉ còn một bản, đánh giá đúng chỗ.

* **`LunarYear.fromYear()` chỉ nhớ MỘT năm** (`_CACHE_YEAR`). Mỗi lần vẽ, ứng
  dụng hỏi 3 năm ở mốc quy chiếu, rồi hỏi lại ở UTC+8 cho bảng tiết khí, rồi lại
  ở mốc địa phương cho bảng Âm Bàn — lần nào cũng đá văng lần trước. `Ephem` nhớ
  theo **(năm, mốc)** nên hết cảnh dựng đi dựng lại.

Tab Lịch còn dựng **một bối cảnh cho cả lưới** thay vì lặp lại 42 lần: trước
đây mỗi ô tự gọi `getDOM`, `getTimezoneOffset` (Intl, đắt) và dựng một `Lunar`
riêng. Can chi ngày nay suy thẳng từ số ngày Julius (chu kỳ 60 liên tục), mốc
lấy một lần từ `lunar.js`.

`tools/test_perf.mjs` canh các đường nóng khỏi tụt lại. Nó dọn SẠCH mọi bộ nhớ
đệm trước mỗi phép đo "nguội" — kể cả `_sbCache` riêng của `sb_getJieQiDates` —
vì phép đo đầu tiên viết ra không làm thế: nó quay vòng qua 40 năm với bộ đệm
24 mục, nên trộn lẫn trúng đệm với trượt đệm và cho ra con số nhảy gấp bốn giữa
hai lần chạy. Ngưỡng để rộng tay: chúng là lưới chặn hồi quy, không phải phép
đo chính xác.

Số đo hiện tại (máy chạy test, đã dọn đệm): `processAll` ~13 ms, `render()` tab
Lịch ~8,5 ms, `zi_months` nguội ~17 ms, `sb_getJieQiDates` nguội ~5,5 ms và
~0 ms khi ấm.

`Ephem` cũng là **chỗ duy nhất** cần thay khi đổi sang bộ tính thiên văn khác:
`socSolar`, `vongSolar`, `jieQiJdAtBasis`, `solarNoonMinutes` là toàn bộ bề mặt.

## Bảng Âm Bàn: Tháng | Sóc | Vọng

Bảng bỏ hai cột **Mùng 1** và **Rằm**, thay bằng **Thời điểm Vọng** (trăng
tròn). Lý do: mùng 1 và rằm đều suy được từ ngày âm lịch đang hiện ngay bên
trên, còn thời điểm Vọng thì không — nó là một mốc thiên văn riêng.

Vọng là lúc hiệu kinh độ Mặt Trăng − Mặt Trời đạt **180°**, cùng nghiệm với Sóc
nhưng lệch pha π (`Ephem.vongSolar`). **Không** được cộng nửa tuần trăng vào
Sóc: quỹ đạo Mặt Trăng là ellip nên khoảng Sóc→Vọng xê dịch quanh 14,765 ngày —
tháng 8/2026 chẳng hạn là **15,45 ngày**.

Kiểm chứng độc lập trong `test_soc_parity.mjs`: tại thời điểm Vọng, `astro.js`
tính Mặt Trăng được chiếu sáng **100,00%**, còn trước và sau đó một ngày là
98,97% và 98,91% — một cực đại sạch. Phép thử đòi ≥ 99,5% ở mọi ca.

## Mốc thiên văn: bảng DE423, một nguồn duy nhất

Tiết khí, điểm Sóc và điểm Vọng **không còn tính bằng chuỗi giải tích của
ShouXing** mà tra `assets/web/js/astro_table.js` — bảng sinh sẵn từ JPL DE423 với
chuỗi tuế sai–chương động IAU 2006/2000A (xem `tools/almanac/`).

Chỉ có ba chỗ nối, tất cả nằm trong `ShouXingUtil`: `qiAccurate`, `qiHigh` và
`shuoHigh`. Mọi thứ phía sau — `LunarYear`, cấu trúc tháng âm, tháng nhuận,
`ephem.js`, cả hai tab, và bảng của widget do `build_lunar_table.mjs` sinh ra —
thừa hưởng giá trị mới mà không phải sửa gì. Ba hàm ấy chỉ được gọi ở nhánh
**sau 1959**; trước mốc đó `calcQi`/`calcShuo` vẫn tra bảng lịch sử `QI_KB`/
`SHUO_KB`, và bảng mới không đụng vào — lịch Trung Quốc trước 1959 là **dữ liệu
đã công bố**, không phải thứ để tính lại.

Bảng ghi mốc theo TT; chỗ nối quy sang giờ dân dụng bằng đúng `dtT` cũ, nên chỉ
phần thiên văn đổi, cách xử lý ΔT giữ nguyên.

Đo trên toàn dải 1900–2100 (9.798 mốc), so bản cũ với bản mới **trong TT** để ΔT
triệt tiêu:

| Mốc | trung vị lệch | p95 | lớn nhất | vượt 60 s |
|---|---|---|---|---|
| Tiết khí | 0,45 s | 1,37 s | **2,83 s** | 0 / 4.824 |
| Điểm Sóc | 27,9 s | 103 s | **180 s** | 456 / 2.487 |
| Điểm Vọng | 28,7 s | 146 s | **234 s** | 601 / 2.487 |

Tức là **tiết khí vốn đã đúng** — chuỗi ShouXing cho tiết khí sai không tới 3
giây — còn **điểm Sóc và điểm Vọng mới là chỗ sai thật**, tới gần 4 phút. Chính
`lunar.js` cũng ghi nhận điều đó trong một ghi chú tối ưu cũ ("Sóc chính xác hơn
~2–3 phút").

Dù vậy, **không một ngày lịch nào đổi**: dựng lại `jieqi.txt` và
`lunar_months.txt` cho ra 0/4.824 tiết khí và 0/2.510 mùng 1 nhảy sang ngày
khác; chỉ trường giây trong ngày đổi (40 dòng tiết khí, 1.721 dòng tháng âm).
Sửa cỡ vài phút chỉ dời được ranh giới ngày khi mốc rơi sát nửa đêm địa phương,
và trong 200 năm không có mốc nào rơi đủ sát.

`node tools/test_astro_table.mjs` canh rằng bảng **đang thật sự được dùng**: mất
tệp bảng hay mất thẻ `<script>` thì ứng dụng không hỏng, nó lặng lẽ rơi về chuỗi
cũ — im lặng đúng là lý do phải có phép thử ấy.

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

Ngày âm trong widget lấy từ **bảng do chính ứng dụng ghi ra**
(`publishLunarCache` trong `calendar.js`): mỗi lần vẽ lịch, ứng dụng ghi 40
tháng quanh hôm nay ở đúng múi giờ đang chọn vào SharedPreferences (~500 ký
tự). Widget đọc bảng ấy trước.

Lý do không tự suy từ bảng đóng trong APK: bảng ấy ghi điểm Sóc ở mốc UTC+7, mà
`lunar.js` **không** định mùng 1 thuần tuý bằng "lấy phần nguyên của điểm Sóc
theo múi giờ" — suy như vậy còn lệch ~0,35% số tháng và ~1% nhãn tháng ở mốc xa
UTC+7. Bảng đóng sẵn vẫn giữ làm đường lùi (đúng tuyệt đối ở UTC+7, và vẫn hơn
hẳn cách cũ là chốt cứng mùng 1, vốn lệch tới 16% ở UTC+2).

Widget hiện **cả 24 tiết khí của năm**, xếp hai cột 12 — đúng hình dạng bảng ở
tab Lịch, kể cả vách ngăn giữa hai nửa và ô tô màu cho tiết khí đang hiệu lực.

Chừng ấy nội dung cần chiều cao thật: 6 hàng lịch cộng 13 hàng bảng. Chiều cao
được chia theo đúng tỉ lệ của tab Lịch (hàng lịch cao gấp ~3,7 lần hàng tiết
khí), rồi kẹp hàng tiết khí trong khoảng 9–18dp.

Bảng tiết khí là một khối **cố định**, không đổi khi lật tháng:

* Phần chia luôn tính theo **6 hàng lịch** (`GRID_WEEKS`) — số hàng của tháng dài
  nhất — chứ không theo số hàng của tháng đang xem. Tính theo tháng đang xem thì
  tháng gọn 5 hàng làm bảng phình thêm ~12%: bấm ‹ › một cái là cả khung lẫn cỡ
  chữ nhảy. Chỗ dôi ra của tháng ngắn đổ vào lưới lịch, nơi ô cao thêm chỉ tốt lên.
* Hai cột nằm ở **chỗ cố định**: cột tên rộng đúng bằng tên dài nhất (đo bằng
  `measureText`, y như `width:1%` bên CSS), cột ngày bắt đầu ngay sau nó — giống
  nhau ở cả hai nửa bảng và ở mọi tháng.
* **Cỡ chữ lấy hai ràng buộc, không lấy hệ số đoán chừng.** Theo chiều cao:
  `Paint.FontMetrics` của chính phông đang dùng cho biết một dòng chiếm bao
  nhiêu, chia ra là được cỡ lớn nhất còn nằm trọn trong hàng — tiếng Việt dấu
  chồng (Ậ, Ổ, ế) cao hơn Latin trơn, mà `ascent` đã tính sẵn khoản ấy. Theo bề
  ngang: hạ tiếp cho tới khi "tên dài nhất + mốc ngày giờ dài nhất" nằm gọn
  trong nửa bảng (đo bằng chữ đậm, vì dòng đang hiệu lực in đậm và rộng hơn).
  Trần vẫn là 12dp cho khớp bảng ở tab Lịch. Hệ số 0,66 cũ chừa thừa quá tay:
  trên widget 4×5 của S21 chữ chỉ còn 7,1dp trong khi hàng cao 10,7dp và bề
  ngang vẫn dư hơn 40dp.
* **Lề mỗi nửa bảng co giãn**: chật thì bóp về mức tối thiểu (4 · 4 · 5dp) để
  dành chỗ cho chữ, dư ra bao nhiêu trả lại cho lề tới mức rộng rãi
  (8 · 6 · 10dp — lề trái · khoảng hở giữa hai cột · lề phải, gần đúng bằng đệm
  của bảng bên tab Lịch). Trên S21 nửa bảng chỉ rộng ~165dp nên vài dp lề ấy đổi
  thẳng thành cỡ chữ.
* Bề rộng cột ngày chốt theo **khuôn `00-00-0000 00:00`**, không theo mốc của
  năm đang xem, nên cột không nhích khi lật sang năm khác.
* **Đệm đáy tính theo chỗ cung góc thật sự ăn tới, không phải cả bán kính.**
  Android 12 trở lên tự bo góc mọi widget theo
  `system_app_widget_background_radius` (One UI để khá rộng); bảng chạm sát mép
  thì cung tròn ăn mất chữ đầu của hàng cuối (Mang Chủng) và đuôi giờ của nửa
  phải (Đại Tuyết). Nhưng cung chỉ sâu nhất ở SÁT mép: ở hoành độ *x* mà chữ bắt
  đầu, nó mới xuống tới `d = r − √(2r·x − x²)`. Với góc 16dp và lề hẹp nhất 4dp
  thì d = 5,4dp, không phải 16dp — chừa cả bán kính là hở ra một dải trắng vô cớ
  dưới đáy bảng. Chừa đúng d cộng 1,5dp lề an toàn, tính theo lề HẸP NHẤT có thể
  xảy ra vì lề thật chỉ rộng hơn, mà lề càng rộng thì cung càng ăn nông.
  Bán kính đọc thẳng từ hệ thống, không dưới 16dp của `widget_bg.xml` và không
  quá 32dp.

Cỡ chữ bảng tiết khí đo trên ba máy đích (`node tools/test_widget_layout.mjs`):

| Máy · cỡ widget | Trước | Sau | Ràng buộc |
|---|---|---|---|
| S21 · 4×5 (330×440dp) | 7,1dp | **9,8dp** | chiều cao hàng |
| S21 · 4×6 (330×530dp) | 8,8dp | **10,5dp** | bề ngang nửa bảng |
| S21 FE · 4×5 (360×450dp) | 7,3dp | **10,1dp** | chiều cao hàng |
| S21 FE · 4×6 (360×545dp) | 9,0dp | **11,5dp** | bề ngang nửa bảng |
| A51 · 4×5 (380×460dp) | 7,4dp | **10,3dp** | chiều cao hàng |
| A51 · 4×6 (380×560dp) | 9,3dp | **12,0dp** | chạm trần 12dp |

Ba máy chỉ khác nhau ở bề ngang màn hình (360 · 393 · 412dp) và mật độ
(3 · 2,75 · 2,625). Bố cục tính hết theo dp và theo `measureText` của chính phông
đang dùng, nên **không có nhánh riêng cho máy nào** — cùng một luật cho ra ba kết
quả xếp đúng theo bề ngang. Mật độ lẻ 2,75 của S21 FE cũng không gây lệch: mọi
mốc đều là số thực, chỉ có bề rộng bitmap mới làm tròn.

Ở cỡ 4×6 trở lên, **S21 và S21 FE bị bề ngang chặn** chứ không phải chiều cao:
nửa bảng rộng ~165dp (S21) và ~180dp (S21 FE), mà "Sương Giáng" cộng
"07-12-2026 09:52" đã chiếm gần hết, nên trần thật là ~10,5dp và ~11,5dp — kéo
widget cao thêm không làm chữ to hơn nữa. Chỉ A51 (nửa bảng ~190dp) mới chạm được
trần 12dp. Muốn phá trần ấy thì phải đổi cách hiện mốc ngày giờ, mà như thế lại
lệch với bảng ở tab Lịch.

Sàn 250dp (`minResizeWidth`, phải tự tay bóp mới có) cho chữ 7,8dp. Không nâng
sàn ấy bằng manifest được: `minWidth` quá 250dp thì công thức ô của Android đòi
5 cột, widget hết đặt được lên lưới 4 cột của One UI.

Sàn đặt ở 4×5 để còn đặt được trên lưới màn hình chính 5 hàng mặc định của One
UI; kéo cao thêm một hàng là khác hẳn.

Giờ giao tiết hiện theo **múi giờ của địa điểm đang chọn trong ứng dụng**, đọc
từ `qmdj.location` trong SharedPreferences. Bảng `jieqi.txt` lưu giờ ở UTC+7 nên
widget quy đổi lại bằng `TimeZone.getOffset()` — tra theo từng thời điểm, nên
giờ mùa đông của nước có DST không bị cộng nhầm offset mùa hè. Không có bước
này thì cùng một tiết khí, widget và ứng dụng lệch nhau tới mấy tiếng.

Bố cục bảng có phép thử riêng, đo số chứ không nhìn ảnh:

```bash
node tools/test_widget_layout.mjs
```

Nó dựng `widget_preview.html` ở đúng cấu hình S21 (360dp @3x), S21 FE (393dp
@2,75x) và A51 (412dp @2,625x), quét dải cỡ widget mà lưới One UI dựng ra, rồi
canh năm điều: giá trị
không tràn qua vách ngăn, chữ không nhỏ dưới ngưỡng đọc được, hàng cuối nằm trên
cung góc bo mà cũng không hở thừa quá 4dp, lưới lịch không bị bảng nuốt, và
**bảng không đổi khi lật tháng** —
tháng 4, 5 và 6 hàng lịch phải cho ra cùng một khung, cùng cỡ chữ, cùng vị trí
cột. Phông của Chromium rộng hơn Roboto nên cỡ chữ đo được là phía an toàn: trên
máy thật chữ chỉ có thể to hơn con số ấy.

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
| Ngôn ngữ | Kotlin 2.2.21, AGP 8.13.0, Gradle 9.3.0 |
| Phụ thuộc | chỉ `androidx.core:core-ktx` |

`assembleRelease` tạo APK **chưa ký**. Muốn ký thì thêm `signingConfigs` vào
`app/build.gradle.kts` hoặc dùng **Build → Generate Signed Bundle / APK**.

### Về Gradle 9

Bản Gradle nằm trong `gradle/wrapper/gradle-wrapper.properties` mới là bản
quyết định: chạy `./gradlew` thì Gradle cài sẵn trên máy không được dùng tới,
nên hai bên không cần trùng nhau.

Lên Gradle 9 kéo theo hai thứ bắt buộc:

* **AGP 8.7 không chạy được** — nó chỉ hỗ trợ tới Gradle 8.x và dừng ngay với
  "Minimum supported Gradle version". Phải là bản AGP có hỗ trợ Gradle 9.
* **`kotlinOptions { }` đã bị bỏ ở Kotlin 2.2**, thay bằng
  `kotlin { compilerOptions { jvmTarget.set(JvmTarget.JVM_17) } }`.

Cả hai số phiên bản đều nằm gọn trong `build.gradle.kts` ở thư mục gốc. Nếu
Android Studio báo AGP quá cũ hoặc quá mới so với Gradle bạn đang dùng, mở
**Tools → AGP Upgrade Assistant** hoặc sửa thẳng con số đó — không có chỗ nào
khác trong dự án phụ thuộc vào nó.

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

Phép thử còn đo riêng **bảng tiết khí trong tab Lịch**. Bảng nằm trong khối cuộn
của chính nó (`.cal-jq-body`, `overflow:auto`), nên nội dung rộng quá thì nó cuộn
ngang BÊN TRONG: trang không tràn, không ô lá nào bị "…" nuốt, hai phép quét
chung đều không thấy — mà người dùng thì mất đuôi cột "Dương lịch". Giờ đo thẳng
`scrollWidth` của khối ấy và mép phải của cột cuối. Nhân đây cũng xác nhận: ở
cả bảy kích thước, kể cả S21 (360px) và A51 (412px), bảng **không** tràn — cột
ngày trong app vốn đã vừa, khác hẳn bảng vẽ tay của widget.

Gỡ `viewport.js` ra thì ca "S21 ngang" lập tức đỏ — nên phép thử này có thật,
không phải lúc nào cũng xanh. Trước đây nó **đọc `body.style.zoom`** (thuộc tính
inline, luôn rỗng) nên vẫn xanh với cả bản hỏng; giờ đọc computed style.

### Mùng 1 và điểm Sóc

```bash
node test_soc_parity.mjs
```

Mở ứng dụng ở sáu múi giờ, canh bốn điều: mùng 1 đúng là ngày chứa điểm Sóc
**tính theo Chính Tý**, Rằm = mùng 1 + 14, hai tab nói cùng một ngày âm, và
bảng mà ứng dụng ghi ra cho widget khớp luôn. Mốc mong đợi được tính lại **độc
lập** từ `Astro.solarNoonMinutes` chứ không gọi `zi_dayOf` của `app.js`, nên
sai cùng chiều thì vẫn đỏ.

Bộ ngày thử gồm năm tháng mà điểm Sóc rơi sát Chính Tý, đủ cả hai chiều — và
phép thử **tự kiểm** rằng ít nhất một ca chạm ranh giới, để nó không lặng lẽ
hoá vô nghĩa khi đổi ngày thử. Lùi `js/app.js` về bản cũ thì 3/6 ca đỏ ngay.

### Tiết khí giữa các bảng

```bash
node test_jieqi_parity.mjs
```

Mở ứng dụng ở năm múi giờ khác nhau, đọc bảng Sách Bổ pháp ở tab Kỳ Môn và bảng
tiết khí ở tab Lịch, dựng lại bảng của **widget** từ chính `jieqi.txt` (kể cả
bước quy đổi múi giờ), rồi so cả ba **từng tên và từng mốc giờ**. Cũng kiểm mục được tô
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
