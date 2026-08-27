# Bản quyền & ghi công / Third-party notices

## lunar-javascript (`app/src/main/assets/web/js/lunar.js`)

Thư viện lịch âm – tiết khí – can chi của **6tail**, giấy phép **MIT**.
<https://github.com/6tail/lunar-javascript>

Tệp trong repo này là bản đã được nhúng sẵn trong ứng dụng web gốc, giữ nguyên
không sửa đổi.

## GeoNames (`app/src/main/assets/web/data/cities.txt`)

Dữ liệu 34.006 thành phố (tên, toạ độ, múi giờ IANA, dân số, tên chữ Hán) lấy
từ **GeoNames**, giấy phép **Creative Commons Attribution 4.0 International
(CC BY 4.0)**.
<https://www.geonames.org> · <https://creativecommons.org/licenses/by/4.0/>

Trích xuất bằng `tools/build_cities.py` (qua gói `geonamescache`). Múi giờ của
các quốc gia chỉ có một múi giờ được chuẩn hoá lại theo `zone.tab` của IANA —
xem chú thích trong `canonical_tz()`.

## Thuật toán thiên văn (`app/src/main/assets/web/js/astro.js`)

Cài đặt lại từ **Jean Meeus, _Astronomical Algorithms_ (2nd ed.)**, các chương
25 (Mặt Trời), 28 (phương trình thời gian), 47 (vị trí Mặt Trăng), 48 (pha
Mặt Trăng), 49 (Sóc/Vọng). Công thức toán học không thuộc phạm vi bảo hộ bản
quyền; phần mã trong tệp này là mã tự viết cho dự án.
