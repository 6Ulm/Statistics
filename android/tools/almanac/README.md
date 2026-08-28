# Bộ tính thiên văn tham chiếu (oracle)

Sinh ra `app/src/main/assets/web/js/astro_table.js` — bảng **mốc tiết khí, điểm
Sóc và điểm Vọng** mà cả ứng dụng lẫn widget dùng chung.

## Vì sao là bảng, không phải bộ tính chạy trên máy

Chuỗi apparent-place nghiêm chỉnh cần chuỗi nutation IAU 2000A (1365 hạng) và
một ephemeris Chebyshev. Đóng gói ngần ấy **hai lần** — Kotlin cho widget,
JavaScript cho WebView — không đổi lấy được gì so với tính sẵn một lần ở đây, mà
hai bản port thì có thể trôi khỏi nhau; một bảng thì không.

Bảng ghi mốc theo **TT** (giờ động lực). Bên tiêu thụ tự quy sang giờ dân dụng
bằng đúng mô hình ΔT nó vẫn dùng, nên thay bảng này chỉ đổi phần **thiên văn**,
không đụng tới cách xử lý ΔT.

## Chạy

```bash
pip install pyerfa numpy jplephem
# DE423 (1800-2200) từ PyPI; sdist cũ cần distutils nên giải nén tay:
pip download --no-deps -d /tmp/de423 de423
mkdir -p /tmp/eph && tar xzf /tmp/de423/*.tar.gz -C /tmp/eph

python3 build_astro_table.py /tmp/eph \
    > ../../app/src/main/assets/web/js/astro_table.js
node ../build_lunar_table.mjs        # sinh lại bảng của widget theo giá trị mới
node ../test_astro_table.mjs         # canh bảng còn đúng và còn được dùng
```

## Ephemeris: vì sao DE423 chứ không phải DE440

`implementation_prompt.md` yêu cầu `de440s.bsp` từ `naif.jpl.nasa.gov`. Máy dựng
này bị chính sách mạng chặn host ấy (403 ở bước CONNECT), nên dùng **DE423**
(2010, phủ 1800–2200) lấy từ PyPI.

Thay thế này không đáng kể, và đã **đo** chứ không phải phỏng đoán: DE423, DE421
và DE405 cho ra cùng một mốc trong **≤ 0,014 giây** ở mọi tiết khí và mọi tuần
trăng đã thử. 1″ hoàng kinh Mặt Trời là 24,4 giây tiết khí, nên chênh lệch cỡ
mili-giây-cung giữa các bản DE nằm sâu dưới mọi ngưỡng ta quan tâm. Nút thắt độ
chính xác nằm ở chỗ khác — xem ghi chú về stage 1 dưới đây.

## Bốn cái bẫy (giữ nguyên trong `almanac_core.py`)

1. **Thời gian truyền sáng bất đối xứng** — thiên thể ở thời điểm phát, người
   quan sát ở thời điểm nhận. Sai chỗ này là ~20,5″, mà chồng thêm quang sai thì
   thành đếm hai lần: **8 phút** lệch tiết khí.
2. **Quang sai áp đúng một lần.**
3. **Hoàng đạo THẬT của ngày**, không phải hoàng đạo trung bình. Bỏ nutation lệch
   tới 17″ ≈ 7 phút tiết khí.
4. **Trái Đất là NAIF 399**, không phải 3 (khối tâm Trái Đất–Mặt Trăng). Đo
   khoảng cách KHÔNG bắt được lỗi này: cả hai đều ~1 AU, sai lệch là sai lệch góc.

Cả bốn đã đo lại trên chính mã này: bỏ nutation +17,6″ (7,1 phút), quang sai hai
lần −20,6″ (8,4 phút), dùng khối tâm thay Trái Đất tới 6,6″ (2,7 phút) — khớp
đúng bậc độ lớn tài liệu nêu.

## Hai chỗ đã sửa so với bản `almanac_core.py` nhận được

* **`jieqi_seed` lệch trọn một năm.** Bản gốc đo hoàng kinh từ J2000 rồi cộng
  một số nguyên năm chí tuyến, nên với mọi hoàng kinh dưới 280,47° — tức 22
  trong 24 tiết khí — nó rơi vào tháng Chạp năm TRƯỚC: lệch 365,7 ngày ở Đông
  Chí. Bộ giải Newton kẹp mỗi bước 3 ngày trong 8 vòng, tức với tới 24 ngày, nên
  không thể gỡ lại: nó lặng lẽ hội tụ sang tiết khí khác. Dòng `n` trong bản gốc
  là mã chết, dấu vết của một bước phân định năm chưa làm xong.

* **Tiền đề của stage 1 sai.** Tài liệu bảo hai mốc đầu lệch là do ephemeris độ
  chính xác thấp, và cắm DE440 vào thì xuống dưới một giây. Không phải vậy:
  quy hai mốc "đã công bố" ấy về UTC thì ra **đúng phút tròn**
  (2000-01-06 18:14:00 UTC và 2024-12-21 09:21:00 UTC) — chúng là giá trị độ
  phân giải PHÚT mặc áo giây, nên sai số ±30 giây của chúng chính là "phần dư"
  mà tài liệu quy cho ephemeris. Đổi ephemeris không bao giờ siết được, và đo
  cho thấy đúng thế: DE423 dịch mốc Đông Chí đi 0,01 giây so với bản dự phòng.
  Mốc thứ ba (Xuân Phân 2024) có giây thật, và bộ tính khớp nó trong **2,2 giây**.
