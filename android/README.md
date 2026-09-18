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
| Màn hình | — | **giống hệt bản gốc** |
| Mạng | tải trong trình duyệt | **không có quyền INTERNET** |

Engine Bát Tự / Kỳ Môn **không bị sửa một dòng nào**. Lớp vị trí mới ghi toạ độ
đã chọn vào `countryData['__loc']` rồi trỏ `#country` sang khoá đó, nên
`processAll()` và toàn bộ ba phái (Trí Nhuận / Sách Bổ / Âm Bàn) chạy y hệt cũ,
chỉ khác là kinh độ và múi giờ giờ đây là của đúng nơi người dùng chọn.
Bộ kiểm thử đối chiếu từng lá số với bản web gốc để bảo đảm điều đó.

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

## Hàng điều khiển và hàng dùng chung

Tab Kỳ Môn chỉ còn **một** hàng điều khiển: ngày giờ · phái · ô "Đầy đủ". Ngôn
ngữ và địa điểm tách hẳn ra một hàng riêng nằm **dưới hai tab**, trong cùng
`#bottomDock` với chúng, nên hiện ở **cả hai tab** và đổi ở đâu cũng có tác dụng
như nhau.

Ngôn ngữ và phái vốn là hai dãy nút (2 nút + 3 nút) chiếm gần trọn một hàng; nay
mỗi thứ gói vào **một ô** giống hệt ô địa điểm — ô hiện mục đang chọn, chạm vào
thì mở bảng trượt từ đáy (`#optOverlay`, dùng chung cho cả hai). Ba ô cạnh nhau
thì mở ra cùng một thứ.

Vài chỗ phải để ý:

* **Ô phái chiếm đúng phần tư thứ ba của bảng Tứ Trụ** — mép trái trên vạch
  `Tháng|Ngày`, mép phải trên vạch `Ngày|Giờ`. Bề rộng là một PHẦN TRĂM nên
  **giống hệt nhau ở hai ngôn ngữ**; bản trước để ô co đúng bằng nhãn dài nhất
  (ba nhãn ẩn xếp chồng trong một ô lưới, cộng `zoom` để nới) nên cùng một màn
  hình mà ô rộng 77px ở tiếng Việt và chỉ 44px ở tiếng Trung — hai bố cục khác
  hẳn nhau, và mép phải rơi vào những chỗ khác nhau.
  Phép tính ra phần trăm: `.controls` lọt vào 6px (đệm 5 + viền 1) mỗi bên so
  với bảng, nên hàng rộng `w = W − 12`; vạch 50% của bảng nằm ở `W/2 − 6 = w/2`
  tính từ mép hàng, vạch 75% nằm ở `3W/4 − 6 = 3w/4 + 3`. Ô phái vì thế rộng
  `calc(25% + 3px)`, còn ô ngày giờ dừng ở `calc(50% − 4px)` để chừa đúng một
  khe. Đo được lệch **0,0px** ở mép trái và **0,2–0,3px** ở mép phải, trên
  320/360/393px × hai ngôn ngữ.
* **Ô ngày giờ và ô "Đầy đủ" giữ nguyên bề rộng chữ của mình.** Cho ô ngày giờ
  co được thì nó bị bóp còn `0-09-2026 18:3`; ép cỡ chữ nhỏ đi thì "Sách Bổ" bị
  cắt còn chữ "S" khi đang ở tiếng Việt. Đo trên 360/393/412px × hai ngôn ngữ:
  không ô nào bị cắt, mà cỡ chữ vẫn y như cũ.
* **`#optOverlay` phải nằm trong danh sách chừa của `body.view-cal`.** Luật ẩn
  của tab Lịch xoá mọi con của `<body>` trừ danh sách ấy, mà bảng chọn lại là
  con của `<body>` — quên thì ô ngôn ngữ ở hàng dùng chung mở ra một bảng vô
  hình, đúng cái cảnh mà hàng dùng chung sinh ra để phục vụ.
* **Địa điểm phải kéo được tab Lịch vẽ lại.** Ô này trước chỉ có ở tab Kỳ Môn
  nên không đổi được khi đang xem lịch; nay `calendar.js` bọc `processAll()` để
  vẽ lại khi đang ở tab Lịch (`applyLoc` gọi `processAll` sau khi áp vị trí).
* **`--tabbar-h` đo cả `#bottomDock`**, không riêng `#tabBar` — bỏ sót hàng dùng
  chung thì đáy trang bị thanh che mất đúng chiều cao một hàng.

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
* Dưới lưới là **hai mục gập được**: "Tiết khí" (24 mục trong năm) và "Lịch âm"
  (Sóc/Vọng từng tháng) — xem mục riêng bên dưới. `fitGrid()` chốt chiều cao
  hàng lịch ở `ROW_MIN` rồi nhường trọn phần dư cho hai mục ấy; mục nào dài hơn
  chỗ được chia thì tự cuộn trong khung của nó, hàng tiêu đề dính lại.
* Hai tab ngăn nhau bằng một vạch dọc, tab đang mở có **nền màu nhấn** chứ
  không chỉ đổi màu chữ.

Lịch âm được tính theo **UTC+7** như quy ước lịch Việt Nam (bản tiếng Trung
dùng UTC+8) — đó chính là lý do Tết ta và Tết Tàu thỉnh thoảng lệch một ngày.
Phải đặt lại mốc này mỗi lần vẽ: `processAll()` để lại múi giờ của địa điểm
đang chọn trong biến toàn cục của `lunar.js`, nên nếu đang chọn Paris thì
26/08/2026 hoá ra 15/7 thay vì 14/7.

## Hai mục gập được của tab Lịch

Dưới lưới lịch là hai mục, mở/đóng **độc lập** nên mở được cả hai cùng lúc;
trạng thái nhớ qua các lần mở ứng dụng.

Thanh tiêu đề **chính là hàng `<thead>`** của bảng: ba tên cột vừa làm tiêu đề
mục vừa làm tên cột. Trước đây có một nhãn riêng nên "Tiết khí" hiện hai lần —
một ở tiêu đề, một ở tên cột — mà tên cột lại phải canh tay cho thẳng với giá
trị. Nay chính bảng lo việc ấy, và gập lại thì `tbody` ẩn đi, còn đúng hàng tiêu
đề. Đổi lại, `fitGrid` phải trừ chiều cao hàng tiêu đề của những mục ĐANG ĐÓNG
(mục đang mở thì đã nằm trong phần được chia) — quên là tab Lịch tràn đúng bằng
tổng hai hàng ấy.

**Tiết khí** — một dãy **24 hàng liền**, ba cột: tên · ngày giờ · **can chi
tháng**. Trước đây bảng chia đôi thành hai nhóm 12 cho vừa một màn hình; nay có
thêm cột thứ ba nên xếp thẳng một dãy rồi cho cuộn, đọc theo thứ tự thời gian
cũng tự nhiên hơn.

Can chi tháng lấy **từ chính engine** (`lunar.js`), không tự suy từ chỉ số tiết
khí: can tháng phụ thuộc can năm, mà năm can chi lại đổi ở Lập Xuân — dựng lại
luật ấy bằng tay là mời thêm một nguồn lệch nữa với tab Kỳ Môn. Mỗi trụ tháng
phủ đúng **hai** mục liền nhau (tiết mở tháng, rồi khí nằm giữa tháng), nên hai
hàng lặp cùng một giá trị là đúng chứ không thừa.

Hàm nhận thẳng **ngày Julius ở mốc UTC+8**, không nhận chuỗi giờ địa phương.
Quy ngược chuỗi ấy cần offset đúng của **chính mốc đó**, trong khi offset của
bảng là của ngày đang chọn — ở nước có DST hai thứ lệch nhau một giờ suốt nửa
năm, đủ để Lập Xuân rơi về tháng Sửu thay vì mở tháng Dần. Mà can chi tháng vốn
là đại lượng ở UTC+8, nên đi thẳng.

**Lịch âm** — đúng bảng chi tiết của Âm Bàn pháp ở tab Kỳ Môn (Tháng âm · Sóc ·
Vọng), dựng lại bằng **cùng những hàm ấy** (`Ephem.monthsAtBasis`,
`formatPreciseSocLocal`, `formatPreciseVongLocal`) để hai tab không thể lệch —
`test_cal_sections.mjs` so từng dòng một.

Giá trị Sóc/Vọng CĂN GIỮA (lớp `.c`, giống hệt tiêu đề) — và cột "Dương lịch"
của Tiết khí nay cũng vậy. Cột này rộng hơn hẳn nội dung (Sóc/Vọng chỉ chiếm
non nửa cột, phần dư nhường cho cột Tháng âm hẹp bên trái theo mẹo `width:1%`),
nên căn GIỮA tiêu đề trong khi giá trị căn TRÁI khiến tiêu đề trông như bị đẩy
sang phải cả 20-30px so với nơi giá trị thật sự nằm — đo được: tâm chữ "Sóc"
cách tâm chữ giá trị tới 30px. Cho giá trị cũng căn giữa thì cả hai chia sẻ
đúng một tâm — không cần đo/canh gì thêm, tự động khớp theo đúng nghĩa hình
học. Còn dấu mũi gập/mở (▾/▸) của cột
cuối: trước đây ô TIÊU ĐỀ cuối có lệ riêng "padding-right lớn hơn ô giá trị"
để chừa chỗ cho dấu mũi — làm tiêu đề và giá trị không còn cùng một hộp nội
dung, lệch thêm vài px nữa. Bỏ lệ riêng ấy, thu nhỏ dấu mũi (9px, nép sát mép)
để vừa gọn trong đúng phần đệm chung với ô giá trị ở MỌI bề rộng màn hình
(kể cả nấc thu gọn dưới 375px) — `test_cal_sections.mjs` đo tâm chữ (không phải
tâm ô) của tiêu đề so với giá trị, lệch ≤ 1,5px.

### Vị trí hai tiêu đề CỐ ĐỊNH, bất kể gập hay mở

Lưới lịch giữ **đúng một chiều cao hàng** (`ROW_MIN`), không đổi theo việc mục
nào đang mở hay đóng. Bản trước có HAI công thức khác hẳn nhau: "còn mục nào mở
thì lưới chỉ lấy `ROW_MIN`, phần dư nhường cho mục" so với "không mục nào mở thì
lưới lấy hết phần dư" (có thể chạm `ROW_MAX`) — hai công thức lệch nhau tới
20px/hàng, nhân với 5–6 hàng thì cả lưới lẫn hai tiêu đề bên dưới nhảy hơn
100px mỗi lần bấm gập/mở, dù người dùng chỉ đóng/mở MỘT mục. Nay chỉ còn MỘT
công thức, luôn chạy: lưới luôn giữ `ROW_MIN`, mọi phần dư luôn "nhường" cho
mục — không mục nào mở thì phần dư ấy hoá thành một khoảng trống đứng yên dưới
hai hàng tiêu đề, thay vì kéo lưới phình ra. Đổi lại `fitGrid` phải đo CHIỀU
CAO CỦA `<thead>` (không phải cả DIV): `<thead>` cao như nhau bất kể `tbody`
bên trong đang hiện hay ẩn, nên phép trừ này không đổi theo trạng thái gập/mở —
đo cả DIV (như trước) sẽ khiến "avail" trôi theo mục nào đang mở.

Chia phần dư giữa hai mục thì KHÔNG chia theo tỉ lệ chiều cao thật của những
mục ĐANG MỞ lúc này (bản trước làm vậy) — phần của Tiết khí sẽ phụ thuộc vào
việc Lịch âm CÓ đang mở hay không, nên bấm mở Lịch âm là Tiết khí bị bớt lại
ngay lập tức dù bản thân nó không đổi trạng thái, kéo tiêu đề Lịch âm nhảy
đúng lúc người dùng vừa chạm vào nó. Nay BẤT ĐỐI XỨNG có chủ đích: Tiết khí
(đứng trước) luôn được nhắm tới **55% phần dư**, không đổi dù Lịch âm mở hay
đóng; Lịch âm (đứng sau cùng) lấy hết PHẦN CÒN LẠI sau khi trừ đúng phần Tiết
khí đang dùng thật. Vì Tiết khí đứng trước Lịch âm nên chiều cao thật của nó
ảnh hưởng tới vị trí tiêu đề Lịch âm — khoá cứng phần của nó triệt tiêu hẳn
đường lây; còn Lịch âm không đứng trước ai nên nhường phần dư dôi ra cho nó
không ảnh hưởng tới bất cứ tiêu đề nào — mở một mình thì nó vẫn chiếm trọn chỗ
trống, không phải chừa vô cớ (bản test `test_cal_sections.mjs` canh cả hai:
tiêu đề Tiết khí đứng yên qua đủ 5 tổ hợp gập/mở, và gập/mở CHÍNH Lịch âm không
tự dịch chuyển tiêu đề của chính nó).

### Lấp kín chiều cao trên A51 và S21 FE

Hai máy này cao hơn hẳn S21 (852px và 790px so với 740px), và chính chỗ cao
thêm ấy lại bỏ không: đo trong app (có tính nút "Ghim lịch", thứ chỉ hiện khi
chạy trên Android) thì tab Lịch chừa **128px trống ở đáy trên A51, 107px trên
S21 FE**. Hai nguyên nhân, đều nằm trong phép chia chiều cao:

**1. Hai hàng tiêu đề bị trừ hai lần.** `fitGrid` trừ chiều cao `<thead>` của
cả hai mục ra khỏi `avail` để `rowH` không phụ thuộc trạng thái gập/mở (xem
mục trên) — rồi đem CHÍNH con số đã trừ ấy đi chia cho hai mục. Nhưng thứ
`shareSectionHeight` đặt là `max-height` của CẢ KHUNG, mà khung thì chứa luôn
`<thead>`: hai hàng tiêu đề bị tính vào phần trừ một lần rồi lại nằm trong
phần chia một lần nữa, nên cụm hai mục luôn thấp hơn chỗ nó được phép chiếm
đúng `jqHead + amHead` = **48px**. Nay cộng lại hai hàng ấy trước khi chia.

**2. Đóng sẵn Lịch âm là bỏ không phần của nó.** Trần của Tiết khí cố định ở
55% ngân sách và KHÔNG đổi theo việc Lịch âm mở hay đóng — đó là điều kiện để
tiêu đề Lịch âm không nhảy dưới ngón tay, và không được đụng tới. Hệ quả: phần
còn lại là của riêng Lịch âm, đóng nó lại thì chẳng ai nhận. Trên máy cao, phần
ấy là hơn 100px. Nên `decideAmDefault()` tự quyết hộ khi chưa có lựa chọn cũ
nào: còn đủ chỗ cho hàng tiêu đề cộng ÍT NHẤT MỘT hàng dữ liệu thì mở sẵn cả
hai mục. Quyết theo CHỖ TRỐNG chứ không theo cú bấm nào, nên không sinh ra cú
nhảy nào. Chốt xong ghi thẳng vào `qmdj.calSecAm` rồi gọi widget vẽ lại —
widget gập/mở theo đúng khoá ấy, mà widget thì phải khớp tab Lịch.

Nhân tiện lộ ra một cú nhảy còn sót: trong `render()`, lượt `fitGrid` chạy
TRƯỚC `renderAmBan()` nên lần vẽ đầu tiên nó chưa thấy hàng tiêu đề Lịch âm và
`avail` dôi ra đúng 24px — Tiết khí được 224px lúc vừa mở tab rồi tụt còn 208px
ngay khi người dùng chạm vào Lịch âm, tiêu đề nhảy **16px ngay dưới ngón tay**.
Nay `render()` chia lại chiều cao LẦN CUỐI, sau khi hai bảng đã dựng xong và
cột đã ghim.

Kết quả (đo bằng Chromium, ép hiện nút Ghim để giống lúc chạy trong app, cả
tiếng Việt lẫn tiếng Trung):

| Máy | Trước | Sau |
|---|---|---|
| S21 FE 393×790 @2,75x | thừa 107px | thừa **2,4px** |
| A51 412×852 @2,625x | thừa 128px | thừa **2,4px** |

Chỗ thừa còn lại là phần đệm chống tràn của `GRID_CHROME` cộng mấy pixel làm
tròn (9px ở vòng này, 2,4px sau khi `fitGrid` ĐO phần khung thay vì cộng tay —
xem "Chữ to hơn, đáy khít hơn" bên dưới), cộng tối đa 3px nữa ở những cấu hình
phải hạ mục để mép cắt không ăn mất dấu.
Tab Kỳ Môn vốn đã khít sẵn (hở 0,1px) và không đụng tới. `test_cal_sections.mjs`
canh cả ba điều: không tràn xuống dưới thanh tab, không còn dải trống (≤16px),
và quét trọn 12 tháng của một năm — kể cả tháng 6 hàng, lúc lưới cao thêm trọn
một hàng — trên cả hai máy.

### Soi kỹ toàn bộ giao diện — năm lỗi nữa

Quét tĩnh (9 cỡ máy × 2 thứ tiếng × 2 tab) rồi bấm thật (đổi 90 lượt tháng,
chọn ngày, gập/mở đủ tổ hợp, đổi tiếng, xoay máy, mở cả bốn hộp chọn, quét
1900–2101) lôi ra thêm năm chỗ:

**1. Bảng hai mục phình ra khi trang bị phóng.** `syncSectionColumns()` đo bề
rộng cột bằng `getBoundingClientRect` — px ĐÃ phóng — rồi ghi vào `style.width`
— px CHƯA phóng. Điện thoại có `zoom = 1` nên không lộ ra; trên tablet 768px
(`zoom` 1,28) ba cột cộng lại thành 512px nhét vào khung 398px, bảng phình
thành 656px và cột cuối (Can chi / Vọng) bị đẩy hẳn ra ngoài — phải kéo ngang
114px mới đọc được. Nay chia cho `zoom` đúng như mọi phép đo khác trong tệp.

**2. Mục đã đóng vẫn choán chỗ mà bị tính là 0.** `shareSectionHeight()` cho
Lịch âm lấy trọn phần còn lại khi Tiết khí đóng, trong khi Tiết khí đóng vẫn
choán đúng hàng tiêu đề của nó — đóng Tiết khí mà mở Lịch âm là cụm hai mục
thò 17px xuống dưới thanh tab. Nay mỗi mục luôn được trừ sẵn phần tối thiểu
(hàng tiêu đề) của mục kia, và trừ VÔ ĐIỀU KIỆN — trừ có điều kiện là để
trạng thái mục này quyết chiều cao mục kia, đúng cái vòng lây phải cắt.

**3. Ngân sách chiều cao bịa ra chỗ không có.** Công thức cũ kê phần còn lại
lên `SEC_MIN` (96px) ngay cả khi thật ra chỉ còn 51px, nên trên máy 360×640
nội dung tràn 14px xuống dưới thanh tab. `SEC_MIN` là MONG MUỐN ("mục mở ra
thấp quá thì vô dụng"), không phải chỗ có thật: nay khai đúng chỗ còn lại, và
sàn `SEC_MIN` chỉ áp ở chỗ chia, có kẹp theo chỗ có thật.

**4. Tỉ lệ phóng tab Lịch không xác định.** `viewport.js` tính tỉ lệ từ chiều
cao nội dung, mà chiều cao ấy do `fitGrid()` chia ra cho vừa MỌI tỉ lệ — hai
cơ chế cùng kéo một sợi dây, nên mọi tỉ lệ trong [0,95; 1] đều tự nhất quán và
app dừng ở đâu là tuỳ thứ tự chạy: cùng một máy, cùng một tháng, hai lần mở ra
hai cỡ chữ (đo được 0,976 rồi 1,000). Nay ở tab Lịch chỉ lấy tỉ lệ theo BỀ
NGANG — thứ không co giãn — rồi để vòng hạ dần lo nốt máy quá thấp;
`calendar.js` mở thêm `window.__calFit()` để `viewport.js` bảo nó chia lại
trước mỗi lần đo.

**5. Ô chạm hai mũi tên đổi tháng quá bé.** Chữ ‹ › chỉ cao 20px nên ô chạm
cũng chỉ 26×20 — chưa tới nửa mức Android khuyên (48dp), mà đây lại là chỗ bấm
nhiều nhất tab Lịch. Nay đệm ra đúng bằng phần đệm sẵn có của `#calHead` rồi
kéo lại bằng lề âm: ô chạm 38×36, còn thanh tiêu đề không cao thêm pixel nào.

Kèm hai chỗ nhỏ: `<body>` không còn đặt sẵn lớp `lang-zh` (mặc định nay là
tiếng Việt, để sẵn lớp ấy thì khung hình ĐẦU TIÊN vẽ bằng cỡ chữ tiếng Trung
rồi mới nhảy về), và nút **Back** của Android ở tab Lịch nay quay về tab Kỳ Môn
— tab mở lên đầu tiên — thay vì thoát thẳng ra màn hình chính.

### Lưới lịch LUÔN 6 hàng

Tháng dương có 4, 5 hay 6 hàng tuỳ mùng 1 rơi vào thứ mấy (tháng 2/2026 gọn
đúng 4 hàng, tháng 11/2026 cần 6). Điền cho tròn tuần thì lưới cao thấp theo
từng tháng, kéo CẢ HAI tiêu đề bên dưới nhảy **58px** (4→5 hàng) tới **116px**
(4→6 hàng) mỗi lần bấm ‹ ›. Lịch đã ghim ngoài màn hình chính thì vốn LUÔN vẽ
6 hàng (`GRID_WEEKS` trong `CalendarWidgetProvider.kt`, vì 42 ô bắt chạm là cố
định), nên hai bên còn hiện khác nhau ở những tháng ngắn.

Nay tab Lịch cũng luôn 42 ô. Trả giá bằng một hàng ngày mờ thừa ở vài tháng và
58px chỗ của hai mục; đổi lại bố cục đứng yên quanh năm và tab khớp widget
từng ô một.

Mất 58px ấy làm ngưỡng mở sẵn Lịch âm phải đo lại cho đúng: lấy `SEC_MIN`
(96px) làm ngưỡng thì trên S21 FE — nơi phần của Lịch âm là 88px, thừa sức
chứa vài hàng — nó bị đóng lại một cách vô lý và 71px đáy màn hình bỏ trống.
Nay ngưỡng ĐO trên chính bảng đang có: hàng tiêu đề cộng **một** hàng dữ liệu,
lấy chiều cao hàng thật của chính bảng ấy (hàng tiếng Trung cao hơn tiếng Việt
vài pixel, nên một con số px chốt cứng sẽ xử khác nhau ở hai thứ tiếng).

Chỗ thừa ở đáy tab Lịch sau tất cả (đo trong app, cả hai thứ tiếng):

| Máy | Thừa | Lịch âm |
|---|---|---|
| 360×640 | 2,5–5,4px | mở sẵn (1 hàng) |
| S21 360×740 | 2,4–5,4px | mở sẵn (3–4 hàng) |
| S21 FE 393×790 | 2,4–2,5px | mở sẵn (4–5 hàng) |
| A51 412×852 | 2,4px | mở sẵn (5–6 hàng) |
| tablet 768×1024 | 11–12px | mở sẵn |

Máy quá thấp (320×520) và màn hình ngang (800×360) vẫn phải cuộn — `MIN_SCALE`
0,95 là cố ý ("thà cuộn còn hơn chữ li ti") — nhưng cuộn tới đáy là thấy hết,
không có gì kẹt dưới thanh tab.

### Vệt chữ rò trên đỉnh mục khi đang cuộn

Hàng tiêu đề là `<th>` `position: sticky` NẰM TRONG khung cuộn. Trên máy có tỉ
lệ điểm ảnh **lẻ** — đúng hai máy đang ngắm: A51 (2,625) và S21 FE (2,75) —
Chromium chốt vị trí đã "dính" của `<th>` và mép cắt của khung cuộn về hai số
nguyên điểm ảnh khác nhau, nên 1–2px đầu của hàng đang trượt bên dưới ló lên
phía TRÊN hàng tiêu đề: nhìn ra một vệt chữ cụt mờ mờ ngay dưới viền trên của
mục. Không phải lỗi bố cục — `getBoundingClientRect` nói `<th>` nằm đúng mép
khung (369px, trùng khít) — nên đẩy `top`, dày viền hay bỏ bo góc đều không
chữa được (đã thử đủ). Cách chữa: dán một dải đục cao 4px đúng màu tiêu đề lên
đỉnh mục bằng `.cal-sec::after`. Dải nằm trên `.cal-sec` chứ KHÔNG phải trên
khung cuộn nên không bị chính khung ấy cắt mất; thụt vào 1px để viền của mục
vẫn hiện nguyên, và `pointer-events: none` để bấm vào vẫn gập/mở được.

## Widget theo kịp ngôn ngữ

Widget vẽ bằng Kotlin nên không dùng được từ điển của trang web. Ứng dụng ghi
ngôn ngữ đang chọn vào khoá `qmdj.lang` (`publishLang` trong `calendar.js`) rồi
gọi `refreshCalendarWidget`; `LunarTable.langOf` đọc khoá ấy và widget đổi cả
tiêu đề, thứ trong tuần, tên tiết khí, tiêu đề hai cột lẫn can chi.

**Mặc định là TIẾNG VIỆT, và hai bên phải trùng nhau.** Ứng dụng mặc định tiếng
Việt (`initLang` trong `app.js`), nên `LunarTable.DEFAULT_LANG` cũng phải là
`"vi"` — lệch nhau thì ngay sau khi cài mới, trước khi ai kịp ghi khoá, ứng dụng
hiện một thứ tiếng còn lịch đã ghim hiện thứ tiếng khác.
`test_cal_sections.mjs` đọc cả hai mặc định từ mã nguồn rồi so.

**Không treo việc đồng bộ vào một lời gọi.** `MainActivity.onStop` cũng vẽ lại
widget: lỡ một nhịp nào đó (trang chưa nạp xong, broadcast rơi) thì cứ rời ứng
dụng về màn hình chính là widget đã đúng.

### Bốn chỗ từng làm lịch đã ghim lệch khỏi ứng dụng

**Bảng tháng chỉ được ghi ở tab Lịch.** `publishLunarCache()` xưa chỉ chạy trong
`render()`, mà `render()` chỉ chạy ở tab Lịch — trong khi ứng dụng **mở ra ở tab
Kỳ Môn**. Ai không bao giờ mở tab Lịch thì ứng dụng KHÔNG HỀ đưa bảng tháng của
mình cho widget, và widget đành dùng bảng đóng sẵn ở mốc UTC+7: ở Paris lệch
~0,35% số tháng và ~1% nhãn tháng. Nay bảng được ghi sau **mọi** lần engine tính
lại, và ghi luôn một lần lúc mở ứng dụng.

**Khoá bảng là số phút lệch.** Ứng dụng ghi độ lệch của *ngày đang chọn*, widget
lại so với độ lệch của *đúng ngày nó đang vẽ*. Ở nước có DST hai con số ấy khác
nhau suốt nửa năm (Paris: 120 và 60), nên bảng của ứng dụng bị chối oan và
widget lặng lẽ lùi về bảng đóng sẵn. Nay khoá là **mã múi giờ**, không đổi theo
mùa.

Cả hai đều lặng lẽ: widget vẫn hiện một con số trông hợp lý, chỉ là không phải
con số của ứng dụng — nên chỉ đo mới thấy, xem `test_widget_sync.mjs`.

**Khối "công bố lúc mở app" lại bị đặt trong tay bấm.** Có một hẹn giờ mang
đúng ý định "công bố ngay khi mở app, kể cả không đụng gì" — nhưng khối đó nằm
NHẦM bên trong `toggleSection()`, hàm chỉ chạy khi người dùng bấm mở/đóng tiêu
đề Tiết khí hay Lịch âm, thay vì nằm ở `DOMContentLoaded`, nơi chạy đúng một
lần lúc khởi động. Ai ghim widget rồi không bao giờ đụng tới hai tiêu đề ấy thì
khối này không bao giờ chạy. Bug lọt qua nhiều vòng kiểm trước đó vì `app.js`
đã có sẵn một lời gọi `processAll()` 100ms sau khi mở app, tình cờ phủ kín gần
hết các ca thực tế — `test_widget_sync.mjs` dù thử đúng kịch bản "khởi động
lạnh, không chạm gì" vẫn xanh trên cả bản lỗi lẫn bản đã sửa, vì đường công bố
kia của `app.js` che mất sự khác biệt. Nên bản kiểm bổ sung không đo hành vi mà
soi thẳng VỊ TRÍ trong mã nguồn: `test_cal_sections.mjs` trích hẳn thân hàm
`toggleSection` ra và khẳng định nó không gọi `publishLunarCache`, đồng thời
khối hẹn giờ công bố ấy phải tồn tại Ở NGOÀI hàm đó.

Phải gọi cho widget vẽ lại chứ không chỉ ghi khoá: widget chỉ tự vẽ lại lúc nửa
đêm hoặc khi người dùng bấm ‹ ›, nên nếu chỉ ghi thì đổi sang tiếng Trung trong
ứng dụng mà lịch đã ghim vẫn tiếng Việt hàng giờ liền. Đổi **địa điểm** cũng
gọi, vì giờ giao tiết và bảng tháng của widget đều theo nơi đang chọn.

**Chưa từng chọn địa điểm thì app và widget đoán HAI NƠI KHÁC NHAU.** Đây là
chỗ lệch nặng nhất trong cả bốn, vì không cần điều kiện đặc biệt gì — chỉ cần
CHƯA TỪNG mở bảng chọn vị trí, đúng cảnh mở app lần đầu sau khi cài. `countryData`
chỉ có hơn chục nước; múi giờ máy không khớp mục nào trong đó thì `location.js`
xưa BỎ QUA LUÔN, không ghi `qmdj.location`, còn `app.js` lặng lẽ đứng ở mặc định
chốt cứng `'FR'` (Pháp). Widget đọc thấy khoá ấy trống lại tự chốt cứng sang một
nước KHÁC — Việt Nam (`DEFAULT_TZ = "Asia/Ho_Chi_Minh"`) — nên hai bên tính giờ
giao tiết lệch nguyên số giờ (Paris–Hà Nội lệch 5-6 tiếng tuỳ mùa) ngay từ lần mở
app đầu tiên, trước khi ai kịp làm gì cả. Hai chỗ chốt cứng này còn KHÁC NHAU,
nên không phải chỉnh một hằng số cho khớp hằng số kia là xong.

Sửa ở gốc, không phải sửa cho khớp: `location.js` giờ LUÔN ghi lại một vị trí
đúng múi giờ máy khi không mục nào khớp (kinh độ suy từ độ lệch UTC hiện tại,
nhãn "GMT±N" cho biết đây là suy đoán) — không bỏ qua nữa. Phòng thêm một lớp,
`CalendarWidgetProvider.kt` đổi mặc định "chưa có gì" từ MỘT NƯỚC chốt cứng
sang `TimeZone.getDefault()` — múi giờ của chính cái máy đang chạy — để dù JS
có lỡ chưa kịp ghi gì thì widget vẫn đoán gần đúng nhất có thể, thay vì nhảy
sang một nước bất kỳ không liên quan. `test_app.mjs` dựng lại đúng cảnh "múi
giờ máy không khớp mục nào có sẵn", đối chiếu `qmdj.location` VÀ vị trí app
đang dùng đều phải khớp múi giờ máy; `test_cal_sections.mjs` soi mã Kotlin,
khẳng định không còn chỗ nào chốt cứng "Asia/Ho_Chi_Minh" nữa.

Xem trước bản tiếng Trung mà không cần dựng APK:
`node tools/shot_widget.mjs` rồi mở `widget_preview.html?lang=zh`.

## Canh hàng theo thứ bên dưới / bên trên

**Mép trái** ô phái rơi đúng vạch **Tháng|Ngày** của bảng Tứ Trụ ngay dưới, và
mép phải đúng vạch **Ngày|Giờ** — ô chiếm trọn phần tư thứ ba của bảng (xem
phép tính ở mục trên). Bản trước canh TÂM KHE giữa hai ô vào vạch Tháng|Ngày;
đổi vì ô phái nay phải chạy từ vạch 50% tới vạch 75% và rộng như nhau ở hai
ngôn ngữ. Đo được lệch **0,0px** trên 320/360/393/412px × hai ngôn ngữ.

Hai ô của hàng dùng chung thì canh theo **hàng tab ngay trên nó**: rộng hết bề
ngang, chia đôi, mép trái / vạch giữa / mép phải trùng khít mép hai tab Kỳ Môn
và Lịch. Muốn trùng thì không được có đệm ngoài hay khe giữa — vạch ngăn chính
là viền trái của ô thứ hai, y như `.tab-item + .tab-item`.

Hai hàng **cao bằng nhau** nhờ cùng chốt vào `--dock-row-h`, chứ không thả theo
chữ: chữ Hán cao hơn chữ Latin nên một tab tự nhiên cao 27px ở tiếng Trung mà
chỉ 25px ở tiếng Việt — pin một con số cho riêng hàng dưới thì không tài nào
bằng ở cả hai thứ tiếng. Giữa hai hàng chừa 3px, và nền thanh dưới lấy màu nền
TRANG (không phải trắng) nên cái khe ấy hiện ra thành một vạch, hai hàng tách
bạch chứ không dính liền một khối.

Nhãn tab là **chữ không, in hoa, in đậm, 14,5px** — trước có ký hiệu ▦ ▤ ◷
đứng cạnh và chữ chỉ 11,5px, bé hơn mọi chữ khác trên màn trong khi đây lại là
thứ người dùng chạm nhiều nhất. Ba chi tiết phải giữ:

* In hoa bằng `text-transform`, **không** viết hoa sẵn trong HTML: `refreshLabels()`
  ghi lại nhãn mỗi lần đổi ngôn ngữ, và các phép kiểm so `textContent` với
  "Lịch" / "日历". Chữ Hán không có hoa/thường nên luật này không đụng tiếng Trung.
* Chữ hoa tiếng Việt **đội dấu** (Ỳ, Ô, Ệ) nên dòng chữ cao hơn hẳn chữ thường —
  `--dock-row-h` nâng 24px → 29px theo. Hạ lại mà không đo là cụt dấu.
* `--dock-row-h` dùng chung cho cả hàng dưới, nên nâng một chỗ là hai hàng vẫn
  cao bằng nhau (đo lại: 29px/29px ở cả hai thứ tiếng, ba máy).

Thanh dưới dày thêm 5px mà màn Kỳ Môn **không phải thu nhỏ** (đo ở S21/S21FE/A51,
hai thứ tiếng: tỉ lệ vẫn 1, thừa 0,1–0,2px).

Trên máy hẹp nhất (S21, 360px) ô ngày giờ chốt ở 50% và ô phái chốt ở một phần
tư, nên chỗ dự phòng chỉ còn lấy được từ ô "Đầy đủ": dưới 412px nó bóp lề trái
và khe trong còn 2–3px. Phông trên máy thật (Samsung) rộng hơn phông ở máy
dựng, mà cái tràn ấy thì cắt mất chữ "Đầy đủ".

## Không để hở đáy màn Kỳ Môn

Tỉ lệ phóng của `viewport.js` bị chặn bởi **bề ngang**: trên S21 FE nó đã kịch
1,0 vì rộng, trong khi chiều cao còn dôi 39px (A51: 82px) nằm chết ngay trên
thanh dưới. Bàn Kỳ Môn là lưới vuông nên không cao thêm được nếu không rộng
thêm, vậy chỗ duy nhất nhận được phần dôi ấy là **khe giữa các bảng**: sau khi
chốt tỉ lệ, phần thừa được chia đều vào các khe, kẹp trần ở 21px (không có trần
thì trên máy cao các bảng rời rạc hẳn ra, xấu hơn cả khoảng hở).

Kết quả: S21, S21 FE và S21 Ultra khít đáy (hở ≤ 0,3px), A51 còn 10px.

## Tab Lệnh: nhân nguyên tư lệnh

*(Đổi tên nhãn tab sau đó: "Lệnh" → **"Bát Tự"** — `T.tabLenh.vi` trong
lenh.js, đúng hơn với những gì tab này thật sự bày ra nay đã có thêm cả Bát Tự
đầy đủ và bảng Đại Vận, không chỉ mỗi bảng lệnh. ID/class nội bộ (`tabLenh`,
`view-lenh`, `lenhSec`, …) và nhãn tiếng Trung (`令`) giữ nguyên — chỉ đổi
CHỮ hiện tiếng Việt trên tab. Mục này giữ nguyên tên gốc vì mô tả lúc xây tab.)*

Tab thứ ba, đứng cạnh tab Lịch. Nó trả lời đúng một câu: **tại thời điểm đang
xem, can nào đang cầm lệnh** — và bày cả bảng lệnh của năm để xem trước xem sau.

Mỗi tháng khí (từ TIẾT này tới TIẾT sau) không do một can duy nhất nắm: chi của
tháng tàng 2–3 can, và chúng thay nhau cầm lệnh theo thứ tự dư khí → trung khí
→ bản khí. Tháng Dần chẳng hạn: Mậu, rồi Bính, rồi Giáp.

### Ba bộ số, ba cuốn sách

Cổ thư không thống nhất phần chia này, và chênh nhau không phải vài phút mà là
cả tuần. Tháng Dần: bản thông hành chia **7·7·16** (Mậu tới ngày 7, Bính tới
ngày 14, rồi Giáp), 《三命通会》 chia **5·5·20** (Mậu tới ngày 5, Bính tới ngày
10, rồi Giáp) — hai mốc đổi can lệch nhau 2 và 4 ngày, nên có những ngày cùng
một người ra hai can khác nhau. Đo trên chính ứng dụng (Hà Nội, Lập Xuân 2026
rơi 04/02 03:02):

| Sinh | Uyên Hải Tử Bình | Tam Mệnh Thông Hội |
|---|---|---|
| 10/02 (≈6 ngày sau Lập Xuân) | **Mậu** | **Bính** |
| 16/02 (≈12 ngày sau) | **Bính** | **Giáp** |

Không có cách nào trung hoà; chỉ có cách nói rõ đang dùng bộ nào. Ô chọn nằm **cùng hàng với ô ngày giờ**, chỉ hiện ở tab Lệnh,
và nhớ lựa chọn qua lần mở sau (`qmdj.lenhRule`).

| Sách | Dần | Mão | Thìn | Thân | Nét riêng |
|---|---|---|---|---|---|
| Uyên Hải Tử Bình *(mặc định)* | 7·7·16 | 10·20 | Ất 9 · **Quý** 3 · Mậu 18 | 7·7·16 | bản thông hành |
| Tam Mệnh Thông Hội | 5·5·20 | 7·23 | Ất 7 · **Nhâm** 5 · Mậu 18 | 5·5·20 | mộ khố lấy can DƯƠNG |
| Tử Bình Chân Thuyên | 7·7·16 | 10·20 | Ất 9 · **Quý** 3 · Mậu 18 | 10·3·17 | chỉ khác tháng Thân |

Xuất xứ, và mức tin cậy của từng bộ — khác nhau, nên nói thẳng:

* **Tam Mệnh Thông Hội** — chép thẳng từ nguyên văn 卷二「论人元司事」, đối chiếu
  sáu bản độc lập. Nguyên văn nằm ngay trong khối ghi chú trên bảng ở
  `lenh.js`, và `test_lenh.mjs` ĐỌC chữ Hán ấy rồi so với bảng bên dưới — sửa
  một bên mà quên bên kia là đỏ. ("艮土"/"坤土" đều là Mậu.)
* **Tử Bình Chân Thuyên** — chép thẳng từ 「十二月令人元司令分野表」 trong
  《子平真诠评注》, cũng canh bằng nguyên văn như trên. Chỉ khác bản thông hành ở
  **đúng một tháng**: Thân 10·3·17 thay vì 7·7·16 ("戊己土十日，壬水三日，庚金
  十七日").
* **Uyên Hải Tử Bình** — là **bản thông hành** mà giới mệnh lý ngày nay quy cho
  hệ Uyên Hải, và cũng đúng bộ số trong ảnh mẫu người dùng gửi (nên nó là mặc
  định). Bản 《渊海子平》 tìm được chỉ có bài 「论天地干支暗藏总诀」 chia theo
  nửa tháng, không phải bảng ba đoạn này — chỗ quy cho ấy là **theo tập quán**,
  không phải một dòng đọc được trong sách.

Cả ba bộ đều phải qua cùng bộ phép canh: mỗi tháng cộng đủ 30, các đoạn nối
liền không hở không chồng, đoạn cuối là bản khí của chi, mốc mở tháng trùng
khít bảng tiết khí, và chi của tháng lệnh trùng trụ tháng ở mọi múi giờ.

### Đo bằng ĐỘ hoàng kinh, không phải số ngày

Sách xưa chép phần chia theo **ngày** ("Mậu 7 ngày, Bính 7 ngày, Giáp 16 ngày"),
vì một ngày Mặt Trời đi xấp xỉ một độ. Nhưng "xấp xỉ" ấy lệch tới **3,4%**:
quanh cận nhật (tháng Giêng) Mặt Trời đi 1,019°/ngày, quanh viễn nhật (tháng
Bảy) chỉ 0,953°/ngày. Đếm ngày thì tổng 30 ngày của một tháng khí không khớp
khoảng cách thật giữa hai tiết — **29,44 ngày** mùa đông, **31,44 ngày** mùa hè
— nên cộng dồn ba đoạn là lệch cả ngày rưỡi, và đoạn cuối không rơi đúng vào
tiết sau.

Đo bằng độ thì hết hẳn: 7° + 7° + 16° = 30° = đúng khoảng cách hai tiết, **theo
định nghĩa**. `test_lenh.mjs` canh tổng ấy cho cả 12 tháng của cả ba bộ, và
canh các đoạn nối liền nhau không hở không chồng.

Cột thứ ba của bảng vì thế hiện **số độ của từng đoạn** (7 · 7 · 16), không
phải khoảng hoàng kinh (285~294°): bộ số là thứ phân biệt ba sách, nên đó mới
là con số người dùng cần liếc thấy.

### Một nguồn duy nhất với bảng tiết khí

Mốc **mở tháng** (bội số của 15°) lấy thẳng từ `ShouXingUtil.qiAccurate` — đúng
hàm mà bảng tiết khí ở tab Lịch và bảng Sách Bổ ở tab Kỳ Môn đang dùng, nên nó
tra bảng DE423 y hệt. Không thể có chuyện "Lập Xuân" ở tab Lịch một giờ mà "vào
lệnh Mậu" ở tab Lệnh một giờ khác; phép kiểm so hai bảng từng mốc một.

Mốc **giữa tháng** (7°, 22°… không phải bội số của 15°) thì bảng DE423 không có
— nó chỉ lưu 24 tiết khí. Chỗ ấy giải bằng chuỗi giải tích `saLonT` của chính
lunar.js. Hai nguồn lệch nhau **≤ 2 giây** tại các mốc dùng chung (đo trên cả
năm 2026: 19:42:43 so với 19:42:45), tức không bao giờ đủ để đổi con số phút
hiện ra.

Đối chiếu bảng mẫu người dùng gửi (năm 2026, mốc UTC+8): **33/33 mốc khớp**.
22 mốc trùng đến từng phút, 11 mốc lệch đúng một phút — và cả 11 đều có phần
giây ≥ 30 (49, 44, 57, 43, 34…), tức nguồn kia **làm tròn** tới phút gần nhất
trong khi cả ứng dụng này thì **cắt**. Giữ phép cắt, vì đó là phép mà bảng tiết
khí và bảng Sóc/Vọng đang dùng — đổi riêng một bảng là mời hai con số khác nhau
cho cùng một thời điểm.

### Dùng lại ĐÚNG hai khối của tab Kỳ Môn

Ô ngày giờ và bảng Bát Tự ở tab Lệnh **không phải bản dựng lại**: chúng là
chính `.controls` và `#tuTruPanel` của tab Kỳ Môn, chỉ ẩn bớt ô chọn phái và ô
"Đầy đủ" (hai thứ chỉ có nghĩa khi có bàn để bày). Dựng lại thì sớm muộn hai
bảng cũng trôi khỏi nhau. Phép kiểm đánh dấu `data-moc` lên bảng ở tab Kỳ Môn
rồi sang tab Lệnh xem dấu ấy có còn không — chỉ cách đó mới phân biệt "dùng lại
đúng phần tử" với "dựng một bản trông giống".

Hệ quả: địa điểm và ngôn ngữ **tự động** dùng chung cho cả ba tab, vì cả ba đọc
cùng `#country` và cùng `currentLang`; `lenh.js` bọc `processAll()` đúng cách
`calendar.js` bọc, nên đổi ngày, đổi nơi hay đổi tiếng đều vẽ lại cả ba.

### Bảng của năm

Cấu trúc theo đúng ảnh mẫu: **Tháng · Can · Số độ · Vào lệnh · Hết lệnh**,
12 tháng từ Sửu (Tiểu Hàn, tháng 1) tới Tý (Đại Tuyết, tháng 12) — tức 12 TIẾT
rơi vào năm dương lịch đang chọn. Bản thông hành và Tử Bình Chân Thuyên ra 33
đoạn; Tam Mệnh Thông Hội ra 32, vì bốn tháng tứ chính (Mão Ngọ Dậu Tý) chỉ chia
hai đoạn. Ô tháng gộp 2–3 hàng.

Giờ viết "05/01 16:23", chỉ thêm năm khi mốc rơi ra ngoài năm của bảng
("05/01/2027 14:09" ở hàng cuối). Bỏ năm ở 32/33 hàng thì cột hẹp đi chừng
30px — đủ để cả năm cột vừa màn hình 360px mà không phải kéo ngang.

Hàng đang cầm lệnh được tô đậm và **tự cuộn vào giữa khung**; bảng 33 hàng mà
mở ra phải tự đi tìm hàng của hôm nay thì bảng vô dụng một nửa. Mép trên khung
chốt về **đầu THÁNG**, không phải đầu hàng: ô tháng gộp 2–3 hàng nên dừng giữa
một tháng là hàng đầu hiện "(Kinh Trập)" mà mất chữ "Mão" nằm trên nó — một cái
tên tiết mồ côi. Mép dưới thì dùng chung phép canh của tab Lịch
(`window.__snapCutRows`, xem "Hàng cuối bị cắt mất DẤU"), vì bảng này cũng đầy
"Mậu", "Tuất", "Bạch Lộ".

### Hàng tab chia ba

Thêm tab thứ ba thì hàng tab chia **ba**, nên vạch ngăn của hàng dùng chung bên
dưới (ngôn ngữ | địa điểm) chia đôi 50/50 như cũ sẽ rơi vào **giữa tab Lịch** —
hai vạch lệch nhau ngay cạnh nhau, nhìn là thấy. Nay ngôn ngữ lấy một phần ba
("Tiếng Việt" / "中文" chỉ cần chừng 60px) và địa điểm lấy hai phần ba, nơi tên
thành phố dài mới thật sự cần chỗ — vạch ngăn về đúng vạch Kỳ Môn|Lịch.

Kèm một lỗi cũ lộ ra khi sửa: cả `.tab-item` lẫn `#sharedBar .picker-btn` đều
là `content-box` (mặc định của `<div>`), nên "một phần ba" là một phần ba của
phần CÒN LẠI sau khi trừ đệm, chứ không phải của hàng — ô ngôn ngữ phình ra
123,7px thay vì 120px. Cho cả hai `box-sizing: border-box` thì hai vạch trùng
nhau trong 0,7px.

### Không bao giờ để lại NỬA HÀNG

Hai mục của tab Lịch nhận chiều cao còn lại sau khi lưới lịch lấy phần của nó.
Trên máy rất thấp phần còn lại nhỏ hơn một hàng: đo trên 320×568, mục Tiết khí
được 38px mà riêng hàng tiêu đề đã 25px — 13px còn lại hiện ra một **dải nửa
chữ**, thấy được nửa trên của "Bạch Lộ 07-09-2026 14:41" mà đọc không ra. Nửa
hàng như thế tệ hơn không có hàng nào: nó chiếm chỗ và trông như lỗi hiển thị.

`noPartialRow()` chốt luật: mục đang mở mà chỗ được cấp **không đủ hàng tiêu đề
cộng MỘT hàng trọn vẹn** thì hạ hẳn về đúng hàng tiêu đề. Mục trông như đang
đóng — sạch, vẫn chạm được — và vài pixel nhả ra chảy sang mục kia (đo lại trên
320×568: Lịch âm từ chỗ bị đóng hẳn nay mở ra với một hàng thật).

Luật này **không** đặt được trong `snapCut`. `snapCut` canh mép dưới theo chỗ
ĐANG cuộn, mà chỗ cuộn đổi sau đó (`scrollJqToActive` chạy sau khi chia chiều
cao) — canh xong là hỏng lại. Đây là quyết định về CỠ nên phải chốt đúng lúc
cấp chiều cao. Cũng vì thế mà không được "sửa" `snapCut` để nó giấu hàng bị cắt
sâu: ở khung cao bình thường, một hàng hiện dở dang dưới đáy là **dấu hiệu còn
cuộn được**, không phải lỗi.

Xoay ngang thì hai mục chỉ còn hàng tiêu đề (lưới lịch 6 tuần ăn gần hết 412px)
— nhưng trang **cuộn được** 117px nên vẫn với tới. Đó là chỗ đã đo, không phải
chỗ đoán.

### Khung bảng phải tự mở quyền cuộn

`#lenhSec` mượn `.cal-sec`/`.cal-sec-body` của tab Lịch, nên mượn luôn luật
`.cal-sec:not(.cal-sec-open) .cal-sec-body { overflow-y: hidden }`. Ở tab Lịch
luật ấy đúng — mục nào người dùng chưa bấm mở thì đừng cuộn. Nhưng tab Lệnh
**không có nút đóng/mở**, nên `#lenhSec` không bao giờ nhận lớp `cal-sec-open`,
và lĩnh đủ luật của mục đang đóng.

Lỗi này khó thấy vì `overflow-y: hidden` **vẫn cho JS đặt `scrollTop`**: bảng
vẫn tự cuộn tới tháng đang cầm lệnh lúc mở tab, ảnh chụp trông bình thường, đo
hình học cũng không bắt được. Chỉ NGÓN TAY là không kéo được — 12 tháng chỉ với
tới chừng một nửa. Nên `lenh.css` ghi đè thẳng `#lenhSec .cal-sec-body
{ overflow-y: auto }`, không mượn lớp `cal-sec-open` (lớp ấy nghĩa là "người
dùng vừa bấm mở mục này", còn mục này không bao giờ đóng).

Phép kiểm trong `test_lenh.mjs` vì thế phải **kéo bằng ngón tay thật**
(`Input.dispatchTouchEvent` qua CDP) rồi đọc lại `scrollTop`, chứ kiểm bằng
`box.scrollTop = n` thì bản hỏng vẫn đạt.

### Một màu chữ duy nhất: ĐEN

Bảng Lệnh ban đầu mượn màu chàm của mục Lịch âm: tiêu đề "LỆNH NĂM ….." là dải
đặc màu chàm chữ trắng, hàng tiêu đề bảng nền xanh nhạt chữ chàm, cột tháng chữ
chàm có vạch chàm, hàng đang cầm lệnh nền xanh chữ chàm. Nay **bỏ hết**: mọi chữ
trong tab này — dòng "Lệnh: Canh", tiêu đề, tên tiết trong ngoặc, cột số độ —
đều là `--text-main` (#000). Màu chỉ còn ở **nền và viền**, và chỉ là xám trung
tính (#f0f0f0 hàng tiêu đề, #f7f7f7 nền xen kẽ, #dcdcdc hàng đang cầm lệnh) —
#f9f9fb của `--bg-alt` cũng phải thay vì nó ngả xanh.

Hai hệ quả phải xử lý, không phải chuyện thẩm mỹ:

* **Tiêu đề mất dải màu thì mất luôn hình khối.** Chữ đen trên nền trắng không
  tự tách khỏi nền trang, nên `#lenhHead` phải nhận `border: 1px solid` khớp
  viền khung bảng, và **bỏ viền dưới** để chỗ nối với khung bảng không dày gấp
  đôi (khung bảng đã có viền trên của nó).
* **Miếng vá vệt rò phải đổi màu theo.** `#lenhSec::after` là dải đục dán lên
  3px đầu khung để che vệt rò một điểm ảnh của hàng tiêu đề dính (xem
  `.cal-sec::after`). Nó phải trùng **đúng** nền hàng tiêu đề, nên đổi #eef1fd
  → #f0f0f0 cùng lúc; để sót là hiện một vạch xanh mảnh ngay trên chữ.

Và bỏ luôn **vạch dọc bên trái cột chi**: ô tháng gộp 2–3 hàng (`rowspan`) đã
tự nói nó là một khối — viền dưới chỉ cắt ở hàng cuối của tháng, nền xen kẽ
đổi theo THÁNG chứ không theo hàng. Vạch ấy là vẽ lại thứ đã có.

Việc này chỉ đụng `css/lenh.css`. Mục Lịch âm ở tab Lịch và widget ghim **vẫn
giữ màu chàm** — chúng là một bộ khác và phải khớp nhau (xem phần widget).

### Tuổi nhập đại vận

Dòng "Lệnh: Canh" có thêm một dòng phụ: **"Nhập vận: 1 tuổi 8 tháng 17 ngày ·
17/06/2028"**. Công thức là quy ước "tam nhật nhất tuế" (三日一岁) của Tử Bình,
và **có chiều** — thuận hay nghịch chéo giữa Giới tính và âm dương của can
năm, đúng luật cổ điển (không phải luôn thuận như bản đầu tiên làm nhầm):

1. **Chiều**: Nam sinh năm can DƯƠNG (Giáp Bính Mậu Canh Nhâm) hoặc Nữ sinh
   năm can ÂM (Ất Đinh Kỷ Tân Quý) → **THUẬN**; còn lại (Nam+Âm hoặc Nữ+
   Dương) → **NGHỊCH** (`isThuanHanh()` trong `lenh.js`).
2. **Khoảng cách**, tính bằng NGÀY thập phân (giờ/phút gộp vào phần lẻ):
   * Thuận: mốc MỞ THÁNG KẾ TIẾP trừ giờ sinh.
   * Nghịch: giờ sinh trừ mốc MỞ THÁNG HIỆN TẠI.

   "Tháng" ở đây là ranh giới THẬT của tháng lệnh (`monthBounds()`), tức mốc
   TIẾT (12 "tiết" mở tháng — Lập Xuân, Kinh Trập…, KHÔNG phải 24 tiết khí:
   "khí" giữa tháng như Vũ Thuỷ, Xuân Phân không phải ranh tháng). Hai mốc
   này rule-independent: `n0` (mốc mở tháng) tính một lần trước khi chia
   theo `rule.fen`, nên tuổi nhập vận giống nhau ở cả ba sách dù can cầm
   lệnh khác nhau.
3. Chia 3 → tuổi nhập vận, dạng thập phân (`daiVanTuoi()`). Phần lẻ của TUỔI
   quy ra THÁNG (1 tuổi = 12 tháng), phần lẻ còn lại của THÁNG quy ra NGÀY
   (1 tháng = 30 ngày) — dừng ở ngày, không xuống giờ/phút. Ví dụ đối chiếu:
   cách mốc mở tháng 10,5 ngày → 10,5⁄3 = 3,5 tuổi = 3 tuổi + 6 tháng
   (không dư ngày).
4. **Ngày bắt đầu đại vận** = ngày sinh (dương lịch, đúng ngày người dùng đã
   chọn) **cộng LỊCH** ba con số năm/tháng/ngày ấy (`addYMD()`, dùng thẳng
   `Date` gốc của JS để việc tràn tháng/năm tự đúng) — **không phải** cộng
   thẳng số-ngày-thập-phân đã dùng để RA ba con số ở bước 3 (tháng ở bước 3
   là ước lệ 30 ngày, còn ở bước 4 là tháng thật, dài ngắn khác nhau). Luôn
   **cộng tới**, dù thuận hay nghịch: "tuổi nhập vận" luôn dương, và ngày bắt
   đầu đại vận luôn ở SAU ngày sinh. Hiện cố định `dd/mm/yyyy`, không đổi
   theo ngôn ngữ — mọi cột ngày tháng khác trong tab này (Vào lệnh/Hết lệnh)
   đã theo đúng quy ước ấy bất kể tiếng Việt hay tiếng Trung.

Chỉ số Can năm (`window.__yearGanIdx`, 0=Giáp…9=Quý) app.js lộ ra mỗi lần
`processAll()` chạy — cùng lúc bảng Bát Tự phía trên vẽ lại, nên không lệch
pha với can năm đang hiển thị.

`test_lenh.mjs` đối chiếu bằng MỘT NGUỒN KHÁC ngay trong chính `lunar.js`:
`getPrevJie()`/`getNextJie()` (chỉ 12 "tiết", khác API với `getPrevJieQi()`/
`getNextJieQi()` mà app.js dùng cho ô Tiết Khí — API ấy lấy CẢ "khí"). Khớp
`monthBounds()` trong vòng 3 giây — dung sai này không tuỳ tiện, mà đúng mức
lệch đã đo giữa hai đường tính độc lập ở mục "Một nguồn duy nhất với bảng
tiết khí" bên trên ("≤ 2 giây tại các mốc dùng chung"). Và phải đặt
`ShouXingUtil.setTzOffsetHours(8)` ngay trước khi gọi — đúng cái bẫy mà
`termJd()` tự ghi chú: ShouXingUtil giữ múi giờ trong một biến TOÀN CỤC, và
`getPrevJie`/`getNextJie` không tự truyền "8" như `termJd` làm, nên lấy
nguyên múi giờ của lần gọi cuối cùng — thiếu bước này thì phép đối chiếu tự
nó lệch đúng 8,000 giờ (dấu hiệu kinh điển của lỗi múi giờ, không phải sai số
thiên văn — đã đo tay xác nhận trước khi sửa).

### Ô Giới tính (Nam/Nữ)

Cùng hàng với ô ngày giờ và ô Quy tắc, ở phần tư thứ ba của bảng Bát Tự (vạch
50%–75%, đúng cột "Ngày"). Quyết CHIỀU đại vận (thuận/nghịch) khi chéo với âm
dương của can năm — xem mục "Tuổi nhập đại vận" bên trên. Không đụng gì tới
bảng Lệnh của cả năm (bảng ấy chỉ phụ thuộc bộ số ở `RULES`), chỉ đụng dòng
"Nhập vận: …" phía trên bảng.

### Can chi Đại Vận đầu tiên

Dòng thứ ba trong khối: **"Đại Vận: Mậu Tuất"**. Bước ĐÚNG MỘT NẤC trong lục
thập hoa giáp kể từ trụ THÁNG — tới nếu thuận, lùi nếu nghịch
(`daiVanPillarOf()` trong `lenh.js`). Ví dụ đối chiếu: tháng Đinh Dậu, thuận
→ Mậu Tuất (Đinh+1=Mậu, Dậu+1=Tuất); nghịch → Bính Thân (Đinh−1=Bính,
Dậu−1=Thân).

Can và chi CÙNG bước một nấc, không phải chọn riêng rồi ghép: lục thập hoa
giáp chỉ có 60 cặp hợp lệ trong 120 cặp có thể (can và chi phải cùng tính
chẵn/lẻ), và bước cả hai chỉ số cùng lúc luôn giữ đúng tính chất ấy — cách
DUY NHẤT sinh ra cặp kế tiếp/trước đó hợp lệ mà khỏi phải dò qua bảng 60 cặp.

Cần chỉ số CAN của trụ tháng, mà lenh.js trước đó chưa có biến nào giữ số này
(chỉ có `active.part.can` — can đang CẦM LỆNH, một khái niệm khác hẳn). Thêm
`window.__monthGanIdx` vào app.js, ngay cạnh `window.__yearGanIdx` đã có sẵn,
cùng cách lộ ra và cùng nhịp cập nhật (mỗi lần `processAll()` chạy).

Tiếng Trung ghép LIỀN không khoảng trắng ("戊戌", không phải "戊 戌") — đúng quy
ước can chi tiếng Trung đã dùng ở khắp ứng dụng (ví dụ cột "Tuần thủ" ở tab Kỳ
Môn: `${arrGanZH[0]}${chi}` không có khoảng trắng). Tiếng Việt thì cần khoảng
trắng ("Mậu Tuất" là hai tiếng).

### Một cỡ chữ, không đậm

Khối "Lệnh: X / Nhập vận: … / Đại Vận: …" trước đó có phân bậc: dòng "Lệnh:"
đậm và to hơn hẳn, dòng "Nhập vận:" nhỏ hơn. Nay CẢ BA dòng cùng một cỡ chữ
(`clamp(12.5px, 3.4vw, 14px)`), không chữ nào đậm — ba dòng thông tin ngang
hàng nhau, không phải một tiêu điểm với chi tiết phụ đi kèm. Các thẻ `<b>` mà
lenh.js vẫn dựng trong HTML (không xoá, đỡ sửa hai chỗ) nhận `font-weight` và
`font-size` là `inherit`, ngả hẳn về chữ thường của dòng chứa nó.

**Cập nhật sau đó: bỏ dòng "Đại Vận: …", gộp "Lệnh"/"Nhập vận" còn lại thành
MỘT dòng.** Ba dòng ban đầu nay chỉ còn MỘT — dòng "Đại Vận: …" (can chi đại
vận đầu tiên) bỏ hẳn, vì bảng ĐẠI VẬN đầy đủ ngay dưới đã nói đúng việc này,
nhắc lại ở khối tóm tắt là thừa. "Lệnh: X" và "Nhập vận: …" không còn mỗi thứ
một `<div>` riêng — nối thẳng trên cùng một dòng, ngăn cách bằng " · ". Không
còn phần tử `#lenhDaiVan`/`#lenhDaiVanPillar` (div bọc) trong DOM nữa —
`#lenhNow` giờ chỉ chứa hai thẻ `<b>` (`lenhNowVal`, `lenhDaiVanVal`) nằm
CHUNG một dòng văn bản, không phải hai `<div>` xếp chồng. Bộ kiểm thử đọc
nhãn "Nhập vận:" qua `previousSibling` của `#lenhDaiVanVal` (text node ngay
trước nó) thay vì `.firstChild` của một `<div>` không còn tồn tại; can chi
đại vận đầu tiên (trước đọc qua `#lenhDaiVanPillarVal`) nay đọc thẳng qua
`window.__lenhDaiVan().pillar` — dữ liệu vẫn tính y hệt, chỉ không còn dòng
hiển thị riêng cho nó nữa.

### Bảng "LỆNH NĂM" gập/mở được, như Trí Nhuận/Sách Bổ/Âm Bàn

Đóng SẴN lúc mở tab — bảng 33 hàng không phải thứ ai cũng cần thấy ngay, còn
khối "Lệnh/Nhập vận/Đại Vận" phía trên đã là phần tóm tắt. Bấm vào tiêu đề
"LỆNH NĂM 2026" để mở/đóng, y hệt cách ba bảng chi tiết ở tab Kỳ Môn hoạt
động — và dùng lại ĐÚNG hàm `toggleDetailPanel()` của app.js, chỉ khai thêm
một khoá `lenh: { bodyId: 'lenhSec', chevId: 'lenhHeadChevron' }` ở
`_panelIds`, không viết hàm mở/đóng riêng.

`toggleDetailPanel()` vẫn hoàn toàn không biết "Lệnh" là gì — hành vi RIÊNG
(đo lại khung cuộn sau khi mở) nằm ở `lenh.js`, BỌC lấy hàm ấy đúng cách
`render()` đã bọc `processAll()` từ trước, không sửa app.js để nó biết về
từng bảng cụ thể:

* **`fit()`/`scrollToActive()` đo ra RÁC lúc bảng còn đóng.** Phần tử
  `display:none` trả `getBoundingClientRect()` toàn số 0, nên `maxHeight` và
  `scrollTop` tính lúc ấy là vô nghĩa. Hai hàm tự BỎ QUA khi `#lenhSec` đang
  đóng (đỡ tính rác mỗi lần `render()` chạy trong lúc ẩn), và lớp bọc quanh
  `toggleDetailPanel()` gọi lại CẢ HAI ngay sau khi mở — không đợi tới lần
  `render()` kế tiếp (có thể rất lâu sau, tới khi đổi ngày/địa điểm).
* **Hai hình dạng khác nhau tuỳ trạng thái.** Đóng: `#lenhHead` đứng một mình,
  bo tròn cả bốn góc, có viền dưới — một cái hộp đầy đủ. Mở: dính liền khung
  bảng bên dưới thành một khối (bỏ bo góc dưới, bỏ viền dưới — khung bảng đã
  có viền trên của nó). Lớp bọc `toggleDetailPanel()` gắn/gỡ lớp `.lenh-open`
  trên `#lenhHead` sau mỗi lần bấm để CSS biết chọn hình nào.
* **Tiêu đề lệch trái, mũi tên lệch phải** — bỏ cách canh giữa cũ (thêm mũi
  tên vào một tiêu đề canh giữa thì "đẹp" nhưng lệch tâm thật, vì không có gì
  đối xứng bên trái) — đúng cách ba bảng kia đã trình bày.

Chỗ trống để lại lúc bảng đóng (tab Lệnh không có bàn cờ lớn như tab Kỳ Môn
để lấp chỗ) nay có bảng ĐẠI VẬN lấp vào — xem mục ngay dưới.

### Bảng ĐẠI VẬN: 10 đại vận × 10 năm

Lấp đúng chỗ trống dưới "LỆNH NĂM" lúc bảng ấy đang đóng. KHÔNG gập/mở như
"LỆNH NĂM" — luôn hiện, vì việc của nó chính là lấp chỗ trống.

**Tên các đại vận** bước đúng một nấc lục thập hoa giáp mỗi lần, kể từ chính
đại vận ĐẦU TIÊN đã tính ở khối tóm tắt phía trên (`daiVanPillarOf()`) —
`daiVanSequence()` chỉ lặp lại phép bước ấy 10 lần, cùng chiều thuận/nghịch.
Đối chiếu đúng ví dụ người dùng cho: Nam sinh Bính Ngọ, tháng Đinh Dậu →
Mậu Tuất, Kỷ Hợi, Canh Tý, Tân Sửu…

**Mỗi đại vận dài đúng 10 năm dương lịch**, kể từ mốc "ngày bắt đầu đại vận"
đã tính ở khối tóm tắt (`addYMD`) — đại vận thứ k bắt đầu ở
`addYMD(ngày sinh, {years: dv.years + 10k, …})`, tức CÙNG tháng/ngày, năm
+10k mỗi bước; "số tuổi" ghi ở đầu thẻ (`dv.years + 10k`) vì thế cũng tăng
đúng 10 mỗi thẻ. Đầu thẻ hai dòng: `mm/yyyy - Nt` rồi tới can chi đại vận.

**Lưu niên** (can chi của TỪNG năm trong 10 năm một thẻ) là công thức lục
thập hoa giáp CHUẨN theo năm dương lịch trơn (`luuNienCanChi()`: Giáp Tý ≡
năm 4 (mod 60), ví dụ 1984, 2044) — KHÔNG phải trụ năm Bát Tự (trụ năm đổi
tại Lập Xuân, ~4/2 — hai khái niệm khác nhau, "lưu niên" luôn theo đúng nghĩa
quen thuộc "năm 2024 là Giáp Thìn", không lùi lại vài ngày đầu tháng 2). Đối
chiếu ba mốc đã biết: 1952→Nhâm Thìn, 2022→Nhâm Dần, 2024→Giáp Thìn — cả ba
khớp ảnh mẫu người dùng gửi và sự thật lịch vạn niên.

**Bố cục ĐÚNG 2 hàng × 5 cột như ảnh mẫu** (`.dv-grid` > 2 `.dv-grid-row` × 5
`.dv-card`, `document.querySelectorAll('.dv-grid-row')` luôn trả về 2 phần
tử, mỗi phần tử luôn đủ 5 `.dv-card`) — ép vừa đúng bề rộng màn hình
(`grid-template-columns: repeat(5, 1fr)`), không cuộn ngang. Đầu thẻ hai dòng
(mốc bắt đầu + tuổi, rồi can chi đại vận) dùng `clamp()` co theo bề rộng máy;
mỗi hàng lưu niên (năm + can chi) nằm CHUNG MỘT DÒNG như ảnh mẫu ("1952
Nhâm Thìn"), cũng `clamp()` nhưng có SÀN — xem đoạn "cỡ chữ" ngay dưới.

Bản đầu tiên thử ép cứng 5 cột vào bề rộng thật đã làm cỡ chữ tụt xuống
7.5–8.6px THẬT (đo bằng `getComputedStyle`, không phải nhìn ảnh chụp phóng
to trên máy tính) — nhỏ hơn mọi chữ khác trong app, không đọc nổi trên máy
thật. Bản kế tiếp đổi sang cột rộng cố định (116–128px) + cuộn ngang đồng bộ
hai hàng, đọc rất rõ (11.5–13px) nhưng chỉ hiện ~3/5 cột cùng lúc trên máy
hẹp — người dùng chọn ngược lại: **thấy trọn 5 cột ngay, chấp nhận chữ nhỏ
hơn cuộn ngang nhưng không nhỏ như bản clamp() gốc.** Cỡ chữ cuối cùng
(`clamp(9px, 2.4vw, 10.5px)` cho hàng lưu niên) đứng giữa hai thái cực đó.

**Cỡ chữ có SÀN — và một số ít hàng cần co THÊM để khỏi cắt chữ.** Ngay cả
với sàn `clamp()` hợp lý, khoảng 1/4 trong 60 tổ hợp can chi (can VÀ chi đều
dài 4 chữ: Giáp/Bính/Đinh/Canh/Nhâm ghép với Thìn/Thân/Tuất — ví dụ "Nhâm
Thân", "Giáp Thìn") vẫn tràn vài px ở đúng hai máy hẹp nhất (320–360px), vì
năm+can chi CHUNG MỘT DÒNG mà cột chỉ còn ~55–65px. `.dv-card` có
`overflow:hidden` nên tràn là CẮT CỤT chữ thật (mất chữ cái cuối), không
phải chuyện vô hại — không thể lờ đi trong một app mà đúng tên can chi là
cốt lõi. `shrinkDaiVanRows()` (gọi từ `fitDaiVan()` mỗi lần `fit()` chạy) đo
từng `.dv-row` sau khi render, CHỈ co font-size (bước 0.2px, sàn 7.5px) của
ĐÚNG những hàng đang tràn — khoảng 90/100 hàng còn lại vẫn giữ nguyên cỡ
thoải mái của `clamp()`. Luôn XOÁ inline style trước khi đo lại mỗi lần
`fit()` chạy, để một hàng từng phải co lúc màn hẹp được LỚN LẠI đúng cỡ nếu
sau đó màn xoay ngang/rộng ra — co không phải vĩnh viễn.

Bẫy khi tự kiểm tra điều này: `.dv-year`/`.dv-cc` là con `flex` bên trong
`.dv-row`, mặc định `min-width: auto` khiến CON không bao giờ tự co dưới
kích thước chữ của chính nó — `scrollWidth` riêng của con luôn bằng
`clientWidth` riêng, "sạch" giả tạo dù CHA (`.dv-row`) đang tràn thật. Phải
đo tràn trên chính `.dv-row`, không chỉ trên các span con — `tools/test_lenh.mjs`
và `tools/_grid.mjs` đều từng mắc lỗi này trước khi phát hiện ra.

`.dv-grid { overflow-x: auto }` vẫn còn đó nhưng chỉ là LƯỚI AN TOÀN (bình
thường không cuộn gì — `grid.scrollWidth === grid.clientWidth`), phòng một tổ
hợp cực đoan nào đó trong tương lai vẫn tràn dù đã co hết cỡ; không phải cơ
chế chính của bố cục này.

**Chỉ đen/trắng/ghi**, đúng yêu cầu — không màu nào khác trong cả khối (kiểm
bằng máy: mọi `color`/`background-color`/`border-color` phải có R=G=B).
Đại vận ĐANG SỐNG (chứa "năm nay" — xem dưới) và lưu niên "năm nay" bên
trong nó tô đậm bằng NỀN GHI + CHỮ ĐẬM, không phải màu, mượn đúng cách bảng
LỆNH NĂM đã tô hàng đang cầm lệnh (`.lenh-on`).

**"Năm nay"** là năm THẬT lúc xem bảng (`new Date().getFullYear()`, đồng hồ
máy) — KHÔNG phải năm đang nhập ở ô ngày giờ (đó là năm SINH, một khái niệm
khác). Bảng này trả lời "đời người này đang ở đâu", không phụ thuộc đang xem
lá số ở thời điểm nào trong quá khứ. Vì thế sinh năm 2026 mà "năm nay" cũng
là 2026 thì KHÔNG highlight gì cả — 10 đại vận đầu chỉ phủ 2033–2132, chưa
tới lượt; đó là kết quả ĐÚNG, không phải lỗi thiếu highlight. Có thẻ đang
sống thì tự cuộn tới nó (`scrollToCurrentDaiVan()`) — không có hàng tiêu đề
dính phải bù trừ như `scrollToActive()`, nên đơn giản hơn: đưa thẳng đỉnh
thẻ lên đỉnh khung DỌC (`#daiVanBody`), và (phòng khi lưới an toàn cuộn
ngang phải dùng tới — xem đoạn trên) đưa thẻ ra giữa khung nhìn NGANG
(`.dv-grid`) nữa; bình thường 5 cột vừa khít nên vế ngang này là số 0, không
việc gì để làm.

**Chia chỗ với "LỆNH NĂM" — không phải shareSectionHeight() của tab Lịch.**
Hai mục Tiết Khí/Lịch âm ở tab Lịch nằm CẠNH NHAU (chia theo tỉ lệ phần
trăm một ngân sách chung); "LỆNH NĂM" và "ĐẠI VẬN" nằm CHỒNG (dòng chảy tài
liệu bình thường), nên đơn giản hơn nhiều: `fitLenhSec()` CHỪA TRƯỚC một sàn
(`DV_MIN = 170px` + chiều cao đầu khối Đại Vận) khi chia chỗ cho Lệnh năm,
rồi `fitDaiVan()` ĐO LẠI vị trí thật của `#daiVanHead` (đã dịch xuống đúng
chỗ sau khi Lệnh năm định hình) để cấp NỐT phần còn lại. Thứ tự bắt buộc:
`fitLenhSec()` phải chạy trước `fitDaiVan()` trong `fit()`.

`window.__monthGanIdx` (app.js) và `window.__lenhDaiVanBang()` (dữ liệu 10
đại vận của lần vẽ gần nhất, chỉ dùng cho bộ kiểm thử) là hai chỗ lộ ra mới
cho bảng này.

**Cập nhật sau đó: bỏ chữ năm mờ và nền xen kẽ trong mỗi hàng lưu niên.**
`.dv-row .dv-year` trước tô màu mờ (`var(--text-dim)`, #666) để phân biệt với
can chi đậm bên cạnh — nay về màu chữ THƯỜNG (`var(--text-main)`, #000, y hệt
mọi chữ khác) theo đúng yêu cầu, chữ năm không cần mờ đi mới đọc được. Nền
xen kẽ trắng/xám nhạt giữa các hàng (`.dv-row:nth-child(odd)`, cả bản thường
lẫn bản trong thẻ đang sống) cũng bỏ hẳn — không còn phân biệt hàng chẵn/lẻ
bằng màu nền nữa, mỗi thẻ giờ chỉ còn MỘT nền đồng nhất (viền ngăn cách giữa
các hàng vẫn còn, `border-top`), đỡ rối mắt hơn khi nhìn cả bảng 100 dòng.

### "Chọn quy tắc" → "Quy tắc", và ba tên viết tắt

Hai đổi riêng, cùng một chỗ:

* Tiêu đề bảng chọn (`T.pickRule`) rút từ "Chọn quy tắc" còn **"Quy tắc"** —
  ngắn hơn, và khớp cách đặt tên "Giới tính" (không phải "Chọn giới tính") của
  ô mới bên cạnh.
* **Chỉ tiếng Việt**: tên ba sách trong `RULES[].vi` đổi từ tên đầy đủ thành
  chữ đầu viết tắt — **UHTB** (Uyên Hải Tử Bình), **TMTH** (Tam Mệnh Thông
  Hội), **TBCT** (Tử Bình Chân Thuyên). Lý do là chỗ: ô Quy tắc bị đẩy từ nửa
  hàng (50%) xuống một phần tư hàng (25%) để nhường chỗ cho ô Giới tính, và
  "Tam Mệnh Thông Hội" không lọt nổi ở cỡ chữ đọc được trên máy 360px. Áp dụng
  cho **cả** ô đóng lẫn danh sách trong bảng chọn — người dùng "replace ALL".
  Tiếng Trung không đổi (渊海子平/三命通会/子平真诠 vốn đã ngắn). Tên đầy đủ vẫn
  còn nguyên trong khối ghi chú xuất xứ ở đầu `lenh.js` và trong README —
  viết tắt chỉ là CÁCH HIỆN, không đổi dữ liệu `FEN_*` hay việc đối chiếu
  nguyên văn ở `test_lenh.mjs`.

### Hàng ba ô: ngày giờ 50% · Giới tính 25% · Quy tắc 25%

Trước đó ô Quy tắc `flex: 1 1 0` chiếm TRỌN nửa còn lại sau ô ngày giờ (50%–
100%). Thêm ô Giới tính vào GIỮA thì đổi thành ba phần: ô ngày giờ giữ nguyên
trần 50% (Năm+Tháng), ô Giới tính chốt cứng `flex: 0 0 calc(25% + 3px)` —
ĐÚNG công thức `#qmRow #methodDisplayBtn` của tab Kỳ Môn đã dùng cho cùng cột
ấy (xem mục "Hàng điều khiển và hàng dùng chung" — cùng một cột của cùng một
bảng thì cùng một công thức, không suy luận lại từ đầu), còn ô Quy tắc vẫn để
`flex: 1 1 0` nhận phần CÒN LẠI. Với hai ô kia đã chốt cứng, phần còn lại tự
nhiên khớp đúng cột cuối (cột "Giờ") — đo lại: mép phải ô Quy tắc trùng đúng
mép phải của `.controls` (kém mép bảng Bát Tự 6px, đúng độ lọt đã có từ trước,
xem mục "Canh hàng theo thứ bên dưới/bên trên"), không cần tính bù trừ lần
hai.

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

**Nguyên tắc: mốc thì đo, ngày thì tra.** Điểm Sóc, điểm Vọng và tiết khí là
những sự kiện có thật trên trời — với chúng, chuẩn duy nhất là bầu trời, nên
tính bằng ephemeris tốt nhất có được. Còn "mùng 1 là ngày nào" thì không phải
một đại lượng thiên văn: đó là một QUY TẮC áp lên cái mốc ấy, và quy tắc thì
cần múi giờ với mốc nửa đêm mới ra được ngày. Mốc chính xác hơn làm đầu vào của
quy tắc tốt hơn — luôn luôn có lợi — nhưng đầu ra vẫn là quy ước.

Hệ quả cho phần trước 1959: **đừng đụng vào `QI_KB` / `SHUO_KB`.** Hai bảng ấy
không phải phép xấp xỉ bầu trời mà là ghi chép lịch Trung Quốc **đã thật sự
dùng**, kể cả những quyết định hành chính mà không ephemeris nào dựng lại được.
Tính lại chúng là lấy một con số đẹp hơn thay cho một sự thật lịch sử — sai theo
đúng nghĩa. Đó là lý do ba chỗ nối chỉ nằm ở nhánh sau 1959.

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

### Chính Ngọ: cũng lấy từ DE423, qua phương trình thời gian

Chính Ngọ (trưa Mặt Trời thật) chỉ cần **một** đại lượng thiên văn: phương trình
thời gian, tức giờ Mặt Trời biểu kiến trừ giờ Mặt Trời trung bình. Phần kinh độ
trong công thức là số học thuần tuý:

```
Chính Ngọ (phút kể từ 0h giờ đồng hồ) = 720 − (kinh độ − múi giờ×15)×4 − EoT
```

EoT là hàm của **thời gian, không phải nơi chốn**, nên MỘT bảng toàn cục phục vụ
mọi địa điểm — khác hẳn mọc/lặn, vốn phụ thuộc vĩ độ và không tra bảng được.
Bảng lấy mẫu 4 ngày một, nội suy bậc ba; sai số nội suy lớn nhất 12 ms (bước 8
ngày cho 172 ms, bước 2 ngày chỉ hơn được 0,65 ms mà tốn gấp đôi).

Đo trên 1.946 mốc rải khắp 1901–2098, so với DE423:

| | trung vị | p95 | lớn nhất |
|---|---|---|---|
| Meeus (bản cũ) | 0,900 s | 2,555 s | **3,624 s** |
| bảng DE423 (mới) | 0,0018 s | 0,0080 s | **0,0133 s** |

Tức Chính Ngọ đi từ sai vài giây xuống sai vài phần trăm giây — đạt mức "dưới
một giây" mà `implementation_prompt.md` đặt ra. Bảng cũng khớp hai mốc kiểm
trong tài liệu ấy: cực đại **+16,45 phút** ngày 03-11 và cực tiểu **−14,19 phút**
ngày 12-02 (tài liệu nêu +16,4 và −14,2).

**Còn một khoản chưa khử: ΔUT1.** EoT tự nó thuần hình học nên không cần biết
Trái Đất quay nhanh chậm ra sao. Nhưng Chính Ngọ là đại lượng ấy **quy ra giờ
đồng hồ dân dụng**, mà đồng hồ dân dụng chạy theo UTC còn giờ Mặt Trời trung
bình chạy theo UT1 — công thức trên ngầm coi UT1 = UTC. Đấy đúng là điều
*stage 2* của tài liệu nói tới. Phần dư là UT1 − UTC, đo trên 19.955 bản ghi
IERS được **−0,676 … +0,808 giây**. Không nhét được vào bảng: giá trị ấy không
đoán trước được cho ngày mai, không tồn tại trước 1972 (khi UTC ra đời), mà ứng
dụng thì chạy ngoại tuyến, không tải được EOP. Nó cũng nhỏ hơn khoảng 60 lần so
với đơn vị PHÚT mà ứng dụng hiển thị Chính Ngọ.

### Bảng chỉ đổi MỐC — tháng nhuận và Bát Tự vẫn theo luật của lunar.js

Bảng chỉ thay các con số; mọi quy tắc suy ra từ chúng vẫn là của `lunar.js`.
Đã kiểm bằng cách chạy hai lần (có bảng / giấu bảng) rồi so từng ký tự:

* **Kết cấu tháng âm 1900–2100 — giống hệt.** 3.711 dòng gồm số tháng, cờ
  nhuận và mùng 1 của mọi tháng: lệch 0. Cả 74 năm nhuận rơi vào đúng tháng cũ.
  `tools/test_astro_table.mjs` chốt cứng dãy tháng nhuận ấy, nên lần sau có ai
  đổi bảng mà xê dịch tháng nhuận thì phép thử đỏ ngay.
* **Bát Tự ở giờ bình thường — giống hệt.**

Có **một** chỗ đổi, và không tránh được: người sinh trong đúng **1–3 giây** mà
mốc giao tiết dịch đi thì trụ tháng (hoặc trụ năm, nếu là Lập Xuân) lật sang trụ
kia. Quét ±10 giây quanh 432 mốc giao tiết: 135/9.072 truy vấn đổi kết quả, cửa
sổ rộng 1 giây (trung vị), 3 giây (lớn nhất); ra ngoài ±4 giây thì khớp 100%.

Đấy là hệ quả số học của việc sửa mốc, không phải lỗi: trụ tháng đổi tại đúng
khoảnh khắc giao tiết, nên mốc dịch bao nhiêu thì cửa sổ lật rộng bấy nhiêu. Và
giá trị MỚI mới là giá trị đúng — theo đúng nguyên tắc "mốc thì đo" ở trên.

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
và thêm cột **Can chi** (trụ tháng mà tiết khí ấy mở ra), thành một dãy 24 hàng
liền × ba cột, cuộn trong khung của mục.

Cột tên căn trái; cột "Dương lịch" và cột "Can chi" **căn GIỮA** — cả tiêu đề
lẫn giá trị, nên hai bên chung một tâm. Căn trái (bản trước) dán cột ngày vào
sát cột tên trong khi phía Can chi hở ra một mảng trống rộng gấp năm: đo trên
máy 393px là 33px một bên và 169px bên kia.

**Hai mục dùng CHUNG một bộ bề rộng cột** (`syncSectionColumns`), nên "Dương
lịch" thẳng hàng với "Sóc" và "Can chi" thẳng hàng với "Vọng". Hai mục là hai
`<table>` riêng, mỗi bảng tự co theo dữ liệu của mình thì lệch nhau 9px và 89px
trên máy 393px — mà chúng nằm ngay trên dưới nhau nên lệch là thấy ngay.

Cách làm: cột tên và cột cuối vốn co đúng bằng chữ (mẹo `width:1%`) nên bề rộng
tự nhiên của chúng CHÍNH LÀ nhu cầu thật; đo ở chế độ `auto`, lấy cột rộng nhất
của từng vị trí giữa hai bảng, ghim vào `<col>` rồi khoá `table-layout: fixed`.
Phải khoá `fixed`: ở chế độ `auto`, `width:1%` trên chính các ô vẫn tranh phần
với `<col>` và bảng lại co theo dữ liệu của riêng nó. Cột giữa nhận phần còn
lại — bề rộng của nó không phải nhu cầu mà là chỗ thừa, nên lấy `max` của hai
bảng là sai.

Hệ quả có chủ đích: cột cuối rộng bằng mốc ngày giờ của Vọng chứ không bằng can
chi, nên "Can chi" đứng giữa một cột rộng hơn nó nhiều. Ba nhãn vì thế không
chia đều tuyệt đối bề ngang (đo được 64px và 94px) — đổi lại hai bảng thẳng cột
với nhau, và đó là thứ nhìn vào là thấy.

## Widget lịch trên màn hình chính

Ghim riêng **lịch âm** ra màn hình chính, không cần mở ứng dụng. Trong tab
Lịch, bấm **📌 Ghim lịch ra màn hình chính** (Android 8 trở lên; launcher cũ thì
nhấn giữ màn hình chính → Tiện ích → "Lịch âm").

**Widget DÙNG ĐƯỢC, không chỉ để nhìn.** Hai mục CUỘN được bằng ngón tay, và
chạm vào một ngày là CHỌN ngày ấy ngay trên màn hình chính: ô được viền màu
nhấn, hai mục tô lại hàng ứng với ngày đó rồi cuộn tới hàng ấy. Không chỗ nào
trong widget mở ứng dụng nữa — mở bằng biểu tượng như mọi ứng dụng khác. Trước
đây chạm vào lưới là nhảy thẳng vào tab Lịch, nên không thể vừa chạm vừa ở lại
màn hình chính.

Chạm lại đúng ngày đang chọn thì BỎ chọn, quay về tô theo hôm nay; chạm tiêu đề
cũng vậy (về tháng này và bỏ chọn).

Widget có **đúng thiết kế của tab Lịch** — cùng màu, cùng cách sắp chữ, cùng
kiểu đánh dấu hôm nay — chỉ bỏ thanh tab và nút ghim. Nội dung gồm lưới lịch rồi
**cả hai mục gập được**: "Tiết khí" (Tiết Khí · Dương lịch · Can chi) và "Lịch
âm" (Tháng âm · Sóc · Vọng).

**Gập/mở theo đúng ứng dụng.** Hai mục ấy trong tab Lịch nhớ trạng thái vào
`qmdj.calSecJq` / `qmdj.calSecAm`; widget đọc chính hai khoá đó, nên đóng mục
nào trong ứng dụng là widget đóng đúng mục ấy, mở cũng vậy. `toggleSection`
gọi luôn `pokeWidget()` sau khi ghi khoá — chỉ ghi không thôi thì widget còn
hiện trạng thái cũ tới tận nửa đêm.

Hai mũi tên **‹ ›** lùi/tiến tháng, chạm tiêu đề thì về tháng hiện tại; tháng
đang xem được nhớ riêng cho **từng widget** (`qmdj_widget` / `w<id>.offset`), nên
ghim hai cái cạnh nhau vẫn xem được hai tháng khác nhau. Ba nút dùng ba
`requestCode` khác nhau (`id * 8 + 1|2|3`), nếu không hệ thống dùng lại cùng một
`PendingIntent` và cả ba cùng làm một việc.

Widget KHÔNG còn là một tấm ảnh duy nhất. Hai ràng buộc của App Widget quyết
định bố cục:

* **Cuộn** chỉ có bên trong một *collection view* (`ListView`/`GridView`) do
  `RemoteViewsService` nuôi — vẽ vào bitmap thì không tài nào cuộn. Nên mỗi mục
  nay là một `ListView` giữ đủ 24 và 13 hàng (`WidgetSectionService`).
* **Chạm** chỉ nhận qua `PendingIntent` gắn vào từng View — RemoteViews không
  có cách nào biết toạ độ ngón tay. Nên trên bitmap lưới có một lưới **6×7 View
  trong suốt**, mỗi ô một PendingIntent riêng. Khoá mục và chỉ số ô nằm trong
  `Intent.data` chứ không chỉ trong extras: hệ thống so intent bằng
  `filterEquals`, vốn BỎ QUA extras, nên hai thứ chỉ khác extras có thể bị gộp
  làm một.

Lưới lịch vẫn là bitmap: dựng 42 ô bằng RemoteViews thì mỗi ô phải là ba
TextView lồng nhau, mà chiều cao ô lại không đặt được theo chiều cao widget
trước API 31 (`setViewLayoutHeight`), trong khi minSdk là 24.

Lưới LUÔN **sáu hàng**, kể cả tháng chỉ cần bốn hay năm: lưới bắt chạm đè lên
trên là 6×7 View cố định trong XML, nên số hàng của bitmap mà đổi theo tháng là
ô chạm lệch hẳn khỏi ô nhìn thấy.

Chiều cao chia bằng `layout_weight`, mà Kotlin phải tự biết khung lưới còn lại
bao nhiêu để vẽ bitmap cho vừa (`scaleType="fitXY"` nên lệch một chút là chữ bị
kéo giãn). Những con số ấy để ở `WidgetLayout.kt` và `values/dimens.xml`;
`test_cal_sections.mjs` đọc cả hai bên và canh khớp nhau.

Ba cột của hai mục chia bằng `layout_weight` **27:35:38**, dùng chung cho hàng
tiêu đề lẫn hàng giá trị và cho cả hai mục — đó là thứ giữ "Dương lịch" thẳng
hàng với "Sóc" và "Can chi" thẳng hàng với "Vọng", không phải đo gì. Đổi lại,
cột không tự nới theo chữ, nên cỡ chữ co theo bề ngang widget
(`WidgetLayout.rowTextSp`): ở cỡ sàn 250dp mà người dùng tự bóp tay, cột giữa
chỉ còn ~87dp trong khi "21-12-2025 22:03" cần ~92dp — thà chữ nhỏ hơn một chút
còn hơn `ellipsize` nuốt mất đuôi giờ.

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
(92 KB + 85 KB); Kotlin chỉ tìm nhị phân. Can chi ngày suy thẳng từ số ngày
Julius.

Hai bảng ấy mang thêm ba cột để dựng được đúng hai mục của tab Lịch:

* **Can chi THÁNG** của từng tiết khí (`jieqi.txt`, cột 4, chỉ số 0–59). Hỏi
  thẳng `getEightChar().getMonth()` chứ không suy từ chỉ số tiết khí: can tháng
  phụ thuộc can năm, mà năm can chi lại đổi ở Lập Xuân.
* **Mốc Sóc và Vọng để HIỆN** (`lunar_months.txt`, bốn cột cuối). Đây KHÔNG
  phải cột giây ở đầu dòng: cột ấy là *ngưỡng bậc thang* định xem NGÀY nào là
  mùng 1 khi đổi múi giờ, còn bốn cột cuối là điểm Sóc/Vọng thiên văn
  (`shuoHigh` ở mốc UTC+8 — đúng hàm mà `Ephem.socSolar`/`vongSolar` gọi) mà tab
  Lịch in ra. Đo trên 2000–2050 thì hai con số **lệch nhau ở 2,4% số tháng, có
  ca lệch 17 giờ** — dùng nhầm cột là widget in giờ Sóc khác hẳn ứng dụng.
  `tools/test_widget_sections.mjs` so từng ô của cả hai bảng với ứng dụng thật,
  ở ba múi giờ.

```bash
node tools/build_lunar_table.mjs   # sinh assets/lunar_months.txt + jieqi.txt
node tools/test_lunar_table.mjs    # đối chiếu với lunar.js
node tools/test_widget_sections.mjs # hai bảng của widget khớp tab Lịch từng ô
node tools/test_astro_table.mjs    # bảng DE423 còn đúng, còn được dùng, và
                                   # tháng nhuận 1900-2100 chưa xê dịch
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

### Cỡ chữ hai mục: thôi dùng ké cỡ của bảng phụ

Hai mục của tab Lịch dựng lại trên `.dp-table` của bảng Sách Bổ pháp ở tab Kỳ
Môn, nên thừa hưởng luôn cỡ chữ của nó: **12px** cho bảng và **11px** cho
`.dp-num` — tức mọi mốc ngày giờ. Nhưng hai chỗ ấy khác nhau về việc: bảng phụ
tab Kỳ Môn để liếc qua, còn đây là chỗ người dùng ĐỌC SỐ (giờ giao tiết, giờ
Sóc/Vọng). Ở 11px thì "22-07-2026 19:13" phải nhíu mắt mới đọc được.

Nay hai mục có cỡ chữ riêng: **13,5px** cho bảng, **13px** cho mốc ngày giờ,
**14px** cho hàng tiêu đề (máy hẹp dưới 375px: 12 / 11,5 / 12,5). Đổi lại mỗi
mục hiện được ít hơn khoảng một hàng — `fitGrid` vẫn chia đúng vì nó ĐO chiều
cao thật chứ không đoán, và phép quét lại cho thấy không chỗ nào cắt chữ, không
chỗ nào phải kéo ngang, đáy vẫn thừa đúng 9px trên cả hai máy đích. (Vòng sau
còn nâng tiếp — xem "Chữ to hơn, đáy khít hơn".)

Widget cũng nâng theo cho khớp: **11,5sp** thay cho 10sp, hàng cao 20dp (18) và
hàng tiêu đề 22dp (20). Kèm một chỉnh nhỏ: mốc `TEXT_FULL_DP` — bề ngang mà ở
đó chữ còn giữ nguyên cỡ — nới từ 300 lên **340dp**. Giữ 300 thì widget bị bóp
về sàn 250dp có chữ to hơn trước và cột giữa cụt mất 1px đuôi mốc giờ; nới ra
thì widget cỡ thường vẫn 11,5sp mà widget bóp hẹp trở về đúng cỡ cũ.

Ba nơi phải cùng một con số — `dimens.xml`, `WidgetLayout.kt` và
`tools/widget_preview.html` — và `tools/test_widget_layout.mjs` canh chúng khớp
nhau ở 18 cỡ widget.

### Chữ to hơn, đáy khít hơn

13,5px vẫn còn bé. Vòng này nâng lên **16,5px** cho hàng tiêu đề (Tiết Khí /
Tháng âm) và **16px** cho cả tên lẫn mốc ngày giờ — nhưng KHÔNG chốt cứng:

```css
table.cal-jq { font-size: clamp(13px, 4.05vw, 16px); }
table.cal-jq thead tr.cal-sec-head th { font-size: clamp(13.5px, 4.18vw, 16.5px); }
```

Ba cột chia nhau bề ngang theo bề rộng ĐO ĐƯỢC của chữ, nên "cỡ lớn nhất còn
vừa" tăng dần theo bề ngang máy. Dò từng nấc 0,25px cho tới lúc có ô đầu tiên
phải cuộn ngang (bên chặn luôn là **tiếng Việt** — chữ dài hơn tiếng Trung):

| Bề ngang | Cỡ lớn nhất còn vừa | = vw |
|---|---|---|
| 360px (S21) | 15px | 4,17 |
| 384px (S21 Ultra) | 16px | 4,17 |
| 393px (S21 FE) | 16,5px | 4,20 |
| 412px (A51) | 17,5px | 4,25 |

Một mốc `@media` cứng thì hoặc cụt chữ ở máy hẹp, hoặc chữ bé vô cớ ở máy rộng.
Hệ số **4,05vw** bám sát đường ấy mà vẫn chừa ~3% phòng khi phông máy thật
(Samsung) rộng hơn phông ở máy đo, và 4,18 = 4,05 × 16,5/16 để hai cỡ giữ đúng
tỉ lệ. Kết quả: A51 được đúng **16,5 / 16px** như yêu cầu, S21 FE 16,4 / 15,9,
S21 15,0 / 14,6. Nhân tiện bỏ luôn nấc `@media (max-width: 375px)` cho cỡ chữ
(phần đệm vẫn giữ) — `clamp()` đã lo.

Chữ to lên thì mỗi mục mất một hàng, nên phải moi chỗ ở nơi khác:

**1. Thanh dưới gọn lại 63px → 52px.** `--dock-row-h` 28 → 24 (nội dung một
hàng chỉ cần 22px: biểu tượng 13px + nhãn 11,5px + đệm), khe giữa hai hàng 6 →
3, đệm tab 5 → 3, đệm `#calHead` 8 → 6, nút Ghim lịch 7/6 → 5/4. Thanh dưới ăn
THẲNG vào chiều cao hai mục, nên 11px ấy về hết cho chúng.

**2. `fitGrid` ĐO phần khung thay vì cộng tay.** Bản trước cộng nhẩm lề/khe
bằng một hằng số, và nhân `ROW_MIN × số tuần` để đoán chiều cao lưới — sai 30px
(ô lịch tự cao thêm vì nội dung). Nay `measureChrome()` đọc đệm của `#calView`
và các khe giữa bốn khối con của nó, còn chiều cao lưới thì đo thẳng
`getBoundingClientRect` sau khi đã đặt `--cal-row-h`. `GRID_CHROME` tụt xuống
còn 14px và chỉ còn là đệm chống tràn. Đáy màn hình từ thừa 9px xuống **2,4px**
— dôi thêm gần một hàng nữa.

`JQ_SHARE` 0,65 → **0,55** và `ROW_MIN` 58 → **52** chia lại chỗ vừa moi được
cho đều hai mục. Kết quả (đo trong app, ép hiện nút Ghim lịch):

| Máy | Tiết khí | Lịch âm | Thừa đáy |
|---|---|---|---|
| A51 412×852 vi / zh | 7 / 6 hàng | 6 / 5 hàng | 2,4px |
| S21 FE 393×790 vi / zh | 5 / 5 hàng | 5 / 4 hàng | 2,5px |
| S21 360×740 vi / zh | 4 / 4 hàng | 4 / 3 hàng | 2,4 / 5,4px |
| 360×640 vi / zh | 2 / 2 hàng | 1 / 1 hàng | 5,4 / 2,5px |

**Hai ngưỡng cho Lịch âm, không phải một.** `AM_KEEP_ROWS = 2` là chỗ
`shareSectionHeight` CHỪA LẠI cho Lịch âm khi chia với Tiết khí — chừa vô điều
kiện, dù Lịch âm đang mở hay đóng, đúng như cái bất biến "trạng thái Lịch âm
không được quyết chiều cao Tiết khí". Thiếu nó thì sàn `SEC_MIN` (96px) của
Tiết khí vét sạch phần còn lại: đo trên 360×640 (có nút Ghim lịch) Lịch âm mở
ra mà cao đúng 28px, thấy mỗi hàng tiêu đề — y như một cái nút bấm không ăn.
Nhưng không bao giờ chừa quá NỬA phần thân chung, để máy quá thấp thì hai mục
cùng ngắn chứ không phải Tiết khí co về đúng hàng tiêu đề.

`AM_OPEN_ROWS = 1` là ngưỡng MỞ SẴN, dễ tính hơn hẳn: vì chỗ của Lịch âm không
chuyển sang cho Tiết khí khi nó đóng, đóng lại chỉ đổi lấy một dải trống ở đáy
— đúng thứ người dùng kêu. Trên 360×640 phần ấy chỉ đủ một hàng rưỡi; một hàng
đọc được cộng thanh cuộn vẫn hơn hẳn 66px trống trơn.

**Và quyết đi quyết lại, chứ không chốt một lần.** Chỗ trống còn đổi SAU lượt
dựng đầu tiên: nút "Ghim lịch" chỉ hiện khi có cầu nối Android và hiện muộn hơn
lượt `fitGrid` đầu, lấy mất 32px; tháng 6 hàng thì lưới cao thêm một hàng. Chốt
một lần trên con số cũ là mở sẵn Lịch âm rồi lượt sau cấp thật lại không đủ —
đúng cái bẫy `amRendered` đã chặn một nửa. Nay `amAuto` giữ quyền tự quyết cho
tới khi người dùng (hoặc widget) tự bấm, và `decideAmDefault` chạy lại mỗi lượt
chia. Không sinh vòng lật qua lật lại, vì cả hai đầu vào của nó — chỗ trống và
chiều cao hàng — đều KHÔNG phụ thuộc vào chính trạng thái đang quyết.

**Hàng cuối bị cắt mất DẤU — thành chữ khác.** Chụp S21 FE rồi phóng to đáy
mục Tiết khí: hàng cuối hiện 73,9% chiều cao, tức vừa đúng dưới đường cơ sở —
thân chữ còn nguyên mà dấu nặng thì mất, nên **"Hàn Lộ" đọc ra "Hàn Lô"** và
**"Mậu Tuất" ra "Mâu Tuất"**. Đó không phải một hàng cụt, đó là một chữ KHÁC:
tiếng Việt dày dấu dưới nên chỗ này là lỗi đọc sai, không phải lỗi thẩm mỹ. Cỡ
chữ cũ cũng có, chỉ là bé nên ít ai để ý.

Mép cắt nay chỉ được rơi vào một trong hai vùng an toàn:

* từ **đáy mực** trở xuống — thấy trọn chữ, kể cả dấu;
* từ **72% hộp dòng** trở lên — cắt phạm vào thân chữ, nhìn là biết ngay hàng
  còn nữa, không ai đọc nhầm.

Rơi vào khoảng giữa thì hạ chiều cao mục xuống đúng mốc 72%. Trả giá tối đa
28% một dòng chữ (đo được 3px ở đáy màn hình, vì phần Tiết khí nhả ra chảy
thẳng sang Lịch âm chứ không mất đi đâu).

Hai chi tiết khiến nó CHẠY ĐÚNG, mà bản đầu tiên thiếu cả hai:

1. **Đáy MỰC, không phải đáy hộp dòng.** Hộp dòng cao hơn chỗ nét chữ thật sự
   chạm tới: trên S21 FE hộp dòng kết thúc ở 19px mà nét thấp nhất của
   "Lộ"/"Mậu" chỉ tới 18px. Lấy nhầm đáy hộp thì vùng "thấy trọn chữ" bị khai
   rộng ra 1px — và đúng 1px ấy là chỗ mép cắt hay rơi vào. Lấy qua
   `canvas.measureText().actualBoundingBoxDescent` với đúng phông đang dùng,
   nhớ theo chuỗi phông nên mỗi lần đổi cỡ chữ/ngôn ngữ mới đo lại một lần.
2. **ĐO sau khi đặt, chứ không tính trước.** Bản tính tay phải dựng lại "chỗ
   các hàng bắt đầu" từ `max-height` trừ viền trừ hàng tiêu đề, mà chuỗi ấy
   còn dính `box-sizing`, `thead` sticky và phép làm tròn nửa pixel — chạy ra
   vẫn lệch 0,5–1px, tức vẫn rơi vào đúng dải nguy hiểm ở 3/10 cấu hình. Đặt
   chiều cao xong thì mọi thứ đã nằm trên trang, hỏi thẳng là xong; hạ chiều
   cao không làm các hàng nhúc nhích (chúng nằm trong phần cuộn) nên một lượt
   là đủ.

`test_cal_sections.mjs` canh cả 10 cấu hình (4 máy × 2 thứ tiếng × 2 mục).

Hộp dòng chữ chỉ hỏi được qua `Range.getBoundingClientRect()` — mà jsdom (nền
của `test_app.mjs`) có `Range` nhưng KHÔNG có hàm ấy, nên lượt đầu cả tab Lịch
ném `TypeError` ngay lúc dựng. Bọc try/catch và kiểm tra hàm có tồn tại không:
thiếu thì `rowMetrics` trả `null` và phép canh này tự tắt, chứ không kéo sập
cả tab.

**Vệt rò trên đỉnh mục, lần thứ hai.** Dải `::after` dán lên đỉnh mục để che
vệt rò của `<thead>` sticky (xem "Soi bằng ẢNH CHỤP" bên dưới) vẫn HỤT đúng một
điểm ảnh thiết bị, và cỡ chữ lớn làm nó lộ hẳn ra: chụp A51 ở 2,625x rồi dò
từng điểm ảnh thì hàng 1094 còn 13 điểm ảnh chữ đen nằm TRÊN hàng tiêu đề.
Nguyên do là dải thụt vào 1px, nên mép của nó chốt theo hộp đã cộng 1px — mà ở
tỉ lệ lẻ 1px không rơi đúng số nguyên. Nay dải phủ CẢ VIỀN (`top/left/right:
0`) và tự vẽ lại viền bằng `border` của chính nó; `.cal-sec` với `.cal-sec-body`
trùng khít nhau từng phần nghìn pixel nên không còn chỗ cho phép làm tròn chen
vào. Dò lại: 0 điểm ảnh rò trên cả ba máy, cả hai mục.

(Thử vá từ bên trong khung cuộn trước — kéo dài hàng tiêu đề lên bằng
`box-shadow` đặc — chỉ bớt được 2 trong 13 điểm ảnh: phần còn lại nằm ở lớp hợp
thành của chính khung cuộn, không thứ gì vẽ BÊN TRONG nó che được.)

Kèm hai lỗi canh hàng cùng một gốc: **`text-align: center` căn theo hộp NỘI
DUNG, không phải hộp ô** — nên chỉ cần `<th>` và `<td>` của một cột đệm ngang
khác nhau là tâm chữ hai bên lệch, và chữ càng to thì lệch càng lộ (ngưỡng
`test_cal_sections.mjs` canh là 1,5px). Cột giữa: `thead th + th` đệm trái 8px
trong khi ô giá trị đệm 6px; và `.cal-jq-date` — lớp CHỈ gắn cho `<td>` — còn
thêm `padding-left: 8px` nữa, di sản của thời cột này căn TRÁI, đẩy tâm giá trị
lệch 2,2px khỏi tâm tiêu đề. Bỏ cả hai lệ riêng: mọi ô của cột đều 6px.

### Soi bằng ẢNH CHỤP, không suy từ mã — hai lỗi nữa

Mọi phép soi trước đều đo DOM rồi suy ra; lượt này chụp thật 28 tấm (2 máy × 2
thứ tiếng × 7 trạng thái) rồi nhìn từng tấm một. Hai chỗ chỉ lộ ra khi nhìn:

**1. Mở bảng chi tiết ở tab Kỳ Môn thì chẳng thấy gì xảy ra.** Hàng tiêu đề
"Bảng chi tiết — …" là khối CUỐI CÙNG của trang, mà `viewport.js` thì căn cho
đáy nó chạm đúng mép thanh tab — nên toàn bộ phần vừa mở nằm dưới mép màn
hình. Đo trên A51: bảng cao 362px, **không một pixel nào** lọt vào khung nhìn.
Người dùng bấm, mũi tên lật, trang đứng im — tưởng nút hỏng, mà thật ra phải
tự vuốt xuống mới thấy. Nay mở xong thì trang cuộn tới, vừa đủ để thấy hết
bảng mà vẫn còn hàng tiêu đề trên màn hình.

**2. Mục Tiết khí tự cuộn xong để lại một hàng cụt dưới hàng tiêu đề.** Hàng
tiêu đề là `<th>` dính, đè lên phần trên khung cuộn; `scrollToActiveJieQi()`
thả `scrollTop` rơi tự do nên hàng trên cùng bị cắt ngang ngay dưới nó — nhìn
ra một vệt chữ cụt, giống lỗi vẽ chứ không giống "còn cuộn được nữa". Hàng
bảng tiếng Trung cao 22px, tiếng Việt 20px, nên chỗ dừng tự do rơi giữa hàng ở
thứ tiếng này mà lại đúng mép ở thứ tiếng kia — cùng một máy, hai kiểu. Nay
nắn `scrollTop` về đúng mép một hàng (đo bằng rect chứ không bằng `offsetTop`:
`offsetTop` của một `<tr>` đo theo `offsetParent`, mà `offsetParent` không chắc
là khung cuộn).

Còn một chỗ KHÔNG phải lỗi nhưng đáng biết: **bản tiếng Trung không có ba bảng
chi tiết** (Trí Nhuận / Sách Bổ / Âm Bàn). Mã gác chúng sau điều kiện `notZH`
ở cả hai chỗ — cố ý, vì nội dung ba bảng ấy chưa dịch (ngay nhãn cũng còn là
"Bảng chi tiết - Âm Bàn pháp"). Đổi sang tiếng Trung là mất hẳn một tính năng
mà không có dấu hiệu gì.

### Gập/mở hai mục ngay trên widget

Triệu chứng người dùng gặp: trong lịch đã ghim, mục **Tiết khí không mở ra
được**, chỉ có Lịch âm là mở sẵn.

Widget vốn chỉ **soi** trạng thái của ứng dụng: nó đọc `qmdj.calSecJq` và
`qmdj.calSecAm` rồi vẽ theo, không có chỗ nào bấm được để lật. Ai lỡ gập Tiết
khí trong tab Lịch thì ngoài màn hình chính đành chịu — mà hai hàng tiêu đề
trông y hệt trong ứng dụng, nơi bấm vào là gập/mở được, nên chẳng có gì gợi ý
rằng ở đây chúng chỉ để nhìn.

Nay hàng tiêu đề của mỗi mục là một `PendingIntent`: bấm là lật đúng khoá mà
tab Lịch đọc, rồi vẽ lại **mọi** widget — trạng thái này là của chung, không
phải của riêng một widget. Cột đầu thêm dấu `▾`/`▸` cho biết bấm được (cột đầu
là cột căn trái duy nhất, thêm vào đấy không đẩy hai cột kia lệch tâm).

Kèm chiều ngược lại: trang web chỉ đọc hai khoá ấy MỘT LẦN lúc nạp, nên bấm ở
widget rồi quay lại ứng dụng thì tab Lịch vẫn hiện trạng thái cũ — và cú bấm
tiếp theo trong ứng dụng sẽ lật từ trạng thái sai ấy. `MainActivity.onResume()`
nay gọi `window.__calSyncSections()` để trang đọc lại.

Và một lỗi nữa lòi ra khi soi chỗ này: `WidgetSections.build()` **bỏ hẳn** một
mục nếu năm đang xem thiếu dữ liệu trong bảng tra (ví dụ lật tới mép bảng, nơi
`jieQiYearOf` không gom đủ 24 mốc). Vòng vẽ duyệt theo danh sách trả về nên mục
bị bỏ ấy không được đụng tới một lần nào: `ListView` của nó giữ nguyên trạng
thái mặc định của XML — **đang hiện, không có adapter** — thành một mảng trắng
chiếm đúng phần chiều cao của mình mà chẳng bao giờ có hàng nào, còn hàng tiêu
đề thì vẫn đủ chữ. Nhìn đúng như "mục này mở không ra". Nay vòng vẽ duyệt một
danh sách CỐ ĐỊNH hai mục và luôn đặt trạng thái cho cả hai; mục nào không dựng
được thì ẩn cả tiêu đề lẫn danh sách.

### Thẻ `<View>` làm widget hỏng hẳn — không có lỗi nào hiện ra

Triệu chứng người dùng gặp: bấm "Ghim lịch ra màn hình chính", hộp thoại của
hệ thống hiện lên đúng, nhưng chỗ xem trước chỉ có dòng **"Couldn't add
widget"** trên một ô đen.

`RemoteViews` KHÔNG dựng bố cục bằng `LayoutInflater` thường: nó cài một bộ lọc
chỉ cho qua các lớp có chú thích `@RemoteView`. Quét `android.jar` API 34 thì
danh sách ấy là `FrameLayout`, `LinearLayout`, `RelativeLayout`, `GridLayout`,
`TextView`, `ImageView`, `Button`, `ImageButton`, `ListView`, `GridView`,
`StackView`, `ViewFlipper`, `AdapterViewFlipper`, `ProgressBar`, `Chronometer`,
`AnalogClock`, `TextClock`, `ViewStub`, `CheckBox`, `RadioButton`, `RadioGroup`,
`Switch`, `AbsoluteLayout`, `DateTimeView` — và **không** có
`android.view.View`, cũng **không** có `android.widget.Space`.

Bố cục widget dùng `<View>` trần 45 lần: 42 ô trong suốt của lưới bắt chạm và
ba vạch ngăn 3dp. Nghĩa là widget CHƯA BAO GIỜ dựng nổi — vừa xem trước vừa đặt
thật đều hỏng. Nay tất cả là `<FrameLayout>`.

Chỗ khó chịu của lỗi này là nó im lặng với mọi phép kiểm sẵn có: `aapt2` dựng
bình thường (bố cục hợp lệ với `LayoutInflater` thường), APK cài được, bản mô
phỏng HTML của widget vẫn đo ra đúng từng pixel, và việc dựng thì xảy ra trong
tiến trình **launcher** nên không có dòng log nào của ứng dụng. `aapt2` không
biết bố cục này dành cho RemoteViews.

Nên có thêm `tools/test_widget_remoteviews.mjs`: đọc mọi `res/layout/widget_*.xml`
và soi từng thẻ theo đúng danh sách trên, chặn luôn `<merge>`, `<include>` và
lớp view tự viết, rồi kiểm cả bố cục mà `appwidget-provider` với
`RemoteViewsFactory` trỏ tới.

Kèm theo: các ô chữ trong bố cục nay có `android:text` mặc định (tên thứ, tên
ba cột của hai mục, tên widget). Bố cục này cũng là `previewLayout`, mà lúc xem
trước thì không có tiến trình nào chạy để rót dữ liệu vào — không có chữ mặc
định thì hộp thoại ghim hiện một khung trống trơn. Kotlin ghi đè mọi ô ấy ở MỖI
lần vẽ nên lúc chạy thật không đổi gì.

### Hai bảng, hai đơn vị — chỗ đã sai một lần

`jieqi.txt` ghi cột 2 là **phút** trong ngày (≤ 1439); `lunar_months.txt` ghi
**giây** (≤ 86 399). `LunarTable.monthStart` từng dùng lại `localize()` — vốn
viết cho `jieqi.txt` nên nhân 60 000 — cho mốc Sóc, tức nhân thừa đúng 60 lần.
Mùng 1 vì thế trôi `floor(giây/1440)` ngày, tới 59 ngày: 2 481/2 510 tháng lệch
và 340 chỗ mất tính tăng dần nên tìm nhị phân cũng sai theo. Sai ở **cả mốc gốc
UTC+7**, không riêng múi giờ xa.

Con bọ chỉ lộ ra khi widget phải dùng đường lùi, vì đường chính (bảng do ứng
dụng ghi ra) vẫn đúng — và `test_lunar_table.mjs` thì vẫn xanh, do bản sao
JavaScript trong đó nhân đúng 1000: **bản sao đã lặng lẽ trôi khỏi thứ nó mô
phỏng**. Nay phép thử soi thẳng vào mã Kotlin — đơn vị của từng bảng,
`monthStart` không được đi qua `localize` — và kiểm cả dải giá trị thật trong
hai tệp dữ liệu, nên một bản sao lệch nữa sẽ đỏ chứ không im.

Hai bảng trải hết bề ngang, **ba cột**, một dãy liền — đúng hình dạng bảng ở tab
Lịch từ lúc nó thêm cột can chi tháng. Trước đây widget gập bảng tiết khí thành
**hai nửa 12 hàng × hai cột**, tức hình dạng CŨ của tab, và cũng vì thế mà không
có chỗ cho cột thứ ba: nửa bảng chỉ rộng ~165dp.

Mỗi mục giữ ĐỦ hàng (24 và 13) và cuộn trong khung của nó, y như mục trong tab
Lịch. Mở mục thì nó tự cuộn tới hàng đang hiệu lực
(`RemoteViews.setScrollPosition`), đúng như tab Lịch cuộn tới tiết khí của ngày
đang chọn.

Chiều cao chia bằng `layout_weight`: lưới lịch **58**, mục Tiết khí **26**, mục
Lịch âm **16**, sau khi trừ các khối cố định (tiêu đề 32dp, hàng thứ 16dp, hai
hàng tiêu đề mục 20dp, đệm đáy 6dp). Mục nào đang ĐÓNG thì `ListView` của nó là
`GONE`, mà `LinearLayout` không tính phần của View đã `GONE` — chỗ ấy tự chia
lại cho lưới và mục còn lại. `WidgetLayout.gridHeightDp` dựng lại đúng luật ấy
để biết vẽ bitmap lưới cao bao nhiêu.

Lưới lịch là một khối **cố định**, không đổi khi lật tháng:

* Luôn **6 hàng** (`GRID_WEEKS`) — số hàng của tháng dài nhất — nên chiều cao ô
  không nhảy khi bấm ‹ ›, và lưới bắt chạm 6×7 trong XML lúc nào cũng đè đúng ô.
* **Cỡ chữ chặn theo dp tuyệt đối** (số ngày ≤ 17dp, ngày âm ≤ 11,5dp, can chi
  ≤ 11dp): thả trôi theo chiều cao widget thì widget cao một chút là chữ phình,
  widget thấp là chữ bé không đọc nổi. Can chi tự tắt khi ô không đủ cao.
* **Đệm đáy 6dp** (`widget_corner_pad`) chừa cho góc bo mà Android 12 trở lên tự
  áp cho mọi widget: hàng cuối của `ListView` chạm sát mép thì bị cung tròn gặm
  mất chữ. Trước đây khoản này tính bằng lượng giác trong Kotlin vì bảng nằm
  trong bitmap; nay bảng là View thật nên chỉ cần một lề tĩnh.

Bố cục đo trên ba máy đích (`node tools/test_widget_layout.mjs`):

| Máy · cỡ widget | Khung lưới | Ô lịch | Số ngày | Can chi | Hàng mục hiện |
|---|---|---|---|---|---|
| S21 · 4×5 (330×440dp) | 239dp | 39,8dp | 14,3dp | tắt | 6 |
| S21 · 4×6 (330×530dp) | 301dp | 50,2dp | **17,0dp** | có | 8 |
| S21 FE · 4×5 (360×450dp) | 246dp | 41,0dp | 14,7dp | tắt | 6 |
| S21 FE · 4×6 (360×545dp) | 311dp | 51,9dp | **17,0dp** | có | 8 |
| A51 · 4×5 (380×460dp) | 253dp | 42,1dp | 15,2dp | tắt | 6 |
| A51 · 4×6 (380×560dp) | 322dp | 53,7dp | **17,0dp** | có | 8 |

Số hàng hiện là số hàng NHÌN THẤY của mục đang mở; mục vẫn giữ đủ 24 hàng và
cuộn. Ở cỡ 4×5 ô lịch chưa đủ cao cho can chi nên nó tự tắt — đúng luật cũ, chỉ
khác là nay lưới luôn 6 hàng nên ngưỡng ấy không còn nhảy theo tháng.

Ba máy chỉ khác nhau ở bề ngang màn hình (360 · 393 · 412dp) và mật độ
(3 · 2,75 · 2,625). Bố cục tính hết theo dp và theo `layout_weight`, nên **không
có nhánh riêng cho máy nào** — cùng một luật cho ra ba kết quả xếp đúng theo bề
ngang. Mật độ lẻ 2,75 của S21 FE cũng không gây lệch: mọi mốc đều là số thực,
chỉ có bề rộng bitmap mới làm tròn.

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

### Widget không được kẹt ở ngày cũ

Widget tự vẽ lại sau nửa đêm bằng một báo thức lặp không chính xác — chỉ cần
đúng ngày, đỡ tốn pin hơn nhiều so với đánh thức nửa tiếng một lần.

Mà báo thức của `AlarmManager` **không sống qua lần khởi động lại máy**, và
`calendar_widget_info.xml` để `updatePeriodMillis="0"` nên hệ thống cũng không
tự gọi `onUpdate` theo chu kỳ. Cộng lại: reboot một cái là lịch đã ghim đứng im
ở ngày hôm ấy, không còn gì đánh thức nó, cho tới khi người dùng tình cờ mở ứng
dụng hay bấm ‹ ›. Đúng cái cảnh "widget hiện bản cũ" mà không ai hiểu vì sao.

Nên có ba đường bù lại:

* `BootReceiver` (`exported="true"`, quyền `RECEIVE_BOOT_COMPLETED`) bắt
  `BOOT_COMPLETED`, `MY_PACKAGE_REPLACED` và `TIMEZONE_CHANGED` rồi dựng lại
  báo thức và vẽ lại ngay. Phải là receiver RIÊNG: receiver của widget để
  `exported="false"` nên không nhận nổi broadcast của hệ thống. Cả ba action
  đều nằm trong danh sách ngoại lệ của luật chặn broadcast ngầm từ Android 8.
* `refreshAll()` đặt lại báo thức ở **mọi** lần vẽ lại, không riêng `onUpdate`.
  Đặt lại cùng một `PendingIntent` chỉ là thay chỗ cũ, không chồng thêm.
* Ứng dụng gọi `refreshCalendarWidget` mỗi khi đổi ngôn ngữ, đổi địa điểm,
  gập/mở một mục, và khi rời ứng dụng (`MainActivity.onStop`).

`RECEIVE_BOOT_COMPLETED` là quyền **thường** (normal): không hỏi người dùng,
không mở đường ra mạng. Ứng dụng vẫn không có `INTERNET`.

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

Mặc định chạy **cả hai tab** (14 lượt đo). Trước đây tab Lịch chỉ được đo khi
đặt `TAB=cal`, mà phép canh "vừa một màn hình" thì chỉ tab ấy mới có — nên một
lần `fitGrid()` quên trừ hàng dùng chung dưới hai tab đã lọt qua trọn bộ kiểm
thử. `TAB=cal` / `TAB=qmdj` vẫn chạy riêng một tab được.

Lượt tab Lịch **bật nút Ghim lên trước khi đo**. Nút ấy `display:none` ngoài ứng dụng Android, nên
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

### Hàng dùng chung và hai ô chọn

```bash
node test_shared_bar.mjs
```

Canh bốn nhóm: **vị trí** (hàng dùng chung nằm dưới hai tab, đúng một hàng, hiện
ở cả hai tab, thanh dưới vẫn dính đáy và không đè lên nội dung); **bề rộng**
(không ô nào bị cắt chữ, trang không tràn ngang, ở 360/393/412px × hai ngôn
ngữ); **đồng bộ** (đổi ngôn ngữ lẫn địa điểm từ tab Lịch thì tab Kỳ Môn theo, và
ngược lại); **hai ô chọn** (đủ mục, đúng thứ tự cũ, đánh dấu đúng mục đang chọn,
bấm Hủy không đổi gì, chọn thật thì cả ô lẫn engine lẫn bảng chi tiết đổi theo,
nút Back đóng bảng) — cộng phần chia bề rộng: ô phái giữ nguyên bề rộng qua cả
ba phái, và ô ngày giờ phải rộng hơn hẳn, tức nó là ô **ăn** phần thừa chứ không
phải ô bị bóp.

Chính phép thử này bắt được lỗi `#optOverlay` bị luật ẩn của tab Lịch xoá mất.

### Lịch đã ghim có khớp ứng dụng không

```bash
node test_widget_sync.mjs
```

Chạy ứng dụng thật trong Chromium, giả lập lớp native để hứng đúng những gì nó
ghi ra kho tuỳ chọn, rồi **dựng lại đường tra của widget** (bản sao
`LunarTable.lunarOf`, cả bảng của ứng dụng lẫn bảng đóng sẵn) trên chính dữ liệu
ấy và so **từng ngày** trong 120 ngày quanh hôm nay.

Ca quan trọng nhất là **KHÔNG mở tab Lịch** — đúng thói quen thật, vì ứng dụng
mở ra ở tab Kỳ Môn. Phép thử canh cả việc widget có THẬT SỰ dùng bảng của ứng
dụng hay không, chứ không chỉ canh con số cuối: bảng đóng sẵn thường cho cùng
đáp án, nên nếu chỉ so số thì lỗi "ứng dụng không ghi gì cho widget" vẫn lọt.
Gỡ bản sửa ra thì phép thử đỏ ngay: *0/120 ngày* tra được trong bảng của ứng
dụng.

### Hai mục của tab Lịch và ngôn ngữ của widget

```bash
node test_cal_sections.mjs
```

Bốn nhóm: mục **Tiết khí** (đúng 24 hàng liền, ba ô mỗi hàng, không còn vách
ngăn chia đôi, cột can chi không ô nào trống và mỗi trụ tháng phủ đúng hai mục);
mục **Lịch âm** (so **từng dòng một** với bảng chi tiết Âm Bàn pháp ở tab Kỳ
Môn); **gập/mở** (mở được cả hai, đóng mục này không đụng mục kia, mục dài thì
cuộn được, trang không tràn dọc, trạng thái được nhớ); **ngôn ngữ** (đổi ngôn
ngữ thì khoá `qmdj.lang` đổi theo VÀ widget được bảo vẽ lại — cầu native được
giả lập để đếm số lần gọi).

`test_jieqi_parity.mjs` canh thêm cột can chi: dòng đang hiệu lực phải khớp đúng
trụ tháng mà tab Kỳ Môn đang hiện, ở cả năm múi giờ.

### Tab Lệnh

```bash
node test_lenh.mjs
```

Bốn nhóm. **Số học**: bảng phân dã cộng đủ 30° mỗi tháng, các đoạn nối liền
nhau không hở không chồng qua sáu năm mẫu, và mốc mở tháng trùng khít tới
`1e-9` ngày với bảng tiết khí dùng chung — kèm một phép so bảng trong
`test_lenh.mjs` với chính bảng trong `lenh.js`, để sửa một bên mà quên bên kia
là đỏ. **Đối chiếu** với ảnh mẫu người dùng gửi (2026, UTC+8): cả 33 mốc, cho
lệch tối đa một phút vì nguồn kia làm tròn còn ứng dụng thì cắt. **Nhất quán**:
quét 244 thời điểm của năm ở bốn múi giờ (UTC+7/+8/+2/−5), chi của tháng lệnh
phải trùng chi của trụ tháng trong bảng Bát Tự, và can cầm lệnh phải là một
trong những can tàng trong chi ấy. **Giao diện**: bảng Bát Tự ở tab Lệnh phải
là CHÍNH phần tử của tab Kỳ Môn (đánh dấu `data-moc` rồi chuyển tab xem dấu còn
không); ô chọn bộ số đứng đúng hàng với ô ngày giờ, ẩn hẳn ở tab Kỳ Môn, đổi
sang Tam Mệnh thì tháng Dần từ 7·7·16 thành 5·5·20 VÀ giờ vào lệnh dịch theo
(đổi mỗi cái nhãn thì vô nghĩa), và nhớ lựa chọn qua lần tải lại; đổi địa điểm
ngay tại tab Lệnh thì giờ vào lệnh đổi theo; giờ mở tháng trùng bảng tiết khí ở
tab Lịch; và sáu cấu hình máy × tiếng đều không cắt chữ, không kéo ngang, không
tràn xuống dưới thanh tab.

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
│   ├── assets/lunar_months.txt      2.510 mốc Sóc, 1900–2100 (sinh sẵn)
│   ├── assets/jieqi.txt             4.824 mốc tiết khí (sinh sẵn)
│   └── assets/web/
│       ├── index.html               khung trang + bảng chọn (vị trí, ngôn ngữ,
│       │                            phái)
│       ├── css/app.css              CSS của bản gốc, giữ nguyên
│       ├── css/location.css         phần giao diện mới
│       ├── css/calendar.css         MỚI — thanh dưới (tab + hàng dùng chung)
│       │                            + lịch âm dương
│       ├── css/lenh.css             MỚI — tab Lệnh (nhân nguyên tư lệnh)
│       ├── js/astro_table.js        MỚI — mốc tiết khí/Sóc/Vọng từ JPL DE423
│       ├── js/lunar.js              thư viện lịch âm của 6tail — ĐÃ SỬA: ba chỗ
│       │                            nối tra astro_table.js (xem NOTICE.md)
│       ├── js/app.js                engine Bát Tự / Kỳ Môn của bản gốc
│       ├── js/astro.js              MỚI — Mặt Trời & Mặt Trăng theo toạ độ
│       ├── js/ephem.js              MỚI — engine thiên văn dùng chung hai tab
│       ├── js/location.js           MỚI — GPS, tra thành phố, toạ độ tay
│       ├── js/viewport.js           MỚI — vừa khít mọi kích thước màn hình
│       ├── js/calendar.js           MỚI — tab Lịch: lưới, tiết khí, ghim widget
│       ├── js/lenh.js               MỚI — tab Lệnh: bảng phân dã theo hoàng kinh
│       └── data/cities.txt          34.006 thành phố + múi giờ IANA
└── tools/                           bộ sinh dữ liệu và kiểm thử
    └── almanac/                     MỚI — oracle Python sinh astro_table.js
```

Ghi công thư viện và dữ liệu bên thứ ba: xem [`NOTICE.md`](NOTICE.md).
