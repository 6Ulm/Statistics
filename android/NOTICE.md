# Bản quyền & ghi công / Third-party notices

## lunar-javascript (`app/src/main/assets/web/js/lunar.js`)

Thư viện lịch âm – tiết khí – can chi của **6tail**, giấy phép **MIT**.
<https://github.com/6tail/lunar-javascript>

Tệp trong repo này lấy từ bản nhúng sẵn của ứng dụng web gốc và **đã được sửa**:
ba chỗ nối trong `ShouXingUtil` — `qiAccurate`, `qiHigh`, `shuoHigh` — nay tra
bảng `js/astro_table.js` thay cho chuỗi giải tích, và lùi về chuỗi cũ khi không
thấy bảng. Ngoài ba hàm ấy (cùng hai hàm phụ `_table`, `_tableLocal`) thì giữ
nguyên. Giấy phép MIT cho phép sửa; sửa ở đâu ghi ngay tại chỗ trong mã.

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

## Bảng thiên văn JPL (`app/src/main/assets/web/js/astro_table.js`)

Mốc tiết khí, điểm Sóc và điểm Vọng tính sẵn từ **JPL Development Ephemeris
DE423** (Jet Propulsion Laboratory, California Institute of Technology, 2010),
lấy qua gói PyPI `de423` của Brandon Rhodes. Ephemeris của JPL là **tài liệu do
chính phủ Hoa Kỳ tạo ra**, thuộc phạm vi công cộng và tự do sử dụng lại; theo
thông lệ, ghi công NASA/JPL–Caltech.
<https://ssd.jpl.nasa.gov/planets/eph_export.html>

Bảng do `tools/almanac/build_astro_table.py` sinh ra — bản thân tệp sinh ra chỉ
chứa **số**, không chứa mã của JPL.

Công cụ sinh bảng (chỉ chạy lúc dựng, **không** đóng gói vào ứng dụng):

* **pyerfa** / **ERFA** — thư viện chuẩn thiên văn IAU SOFA đã đổi tên, giấy
  phép BSD 3 điều khoản. <https://github.com/liberfa/erfa>
* **jplephem** — Brandon Rhodes, giấy phép MIT.
  <https://github.com/brandon-rhodes/python-jplephem>

`tools/almanac/almanac_core.py` là bản cài đặt tham chiếu do người dùng cung
cấp kèm yêu cầu, đã sửa một lỗi (`jieqi_seed` lệch trọn một năm) — ghi rõ trong
`tools/almanac/README.md`.
