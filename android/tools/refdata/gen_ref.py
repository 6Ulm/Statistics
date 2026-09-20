# -*- coding: utf-8 -*-
"""
Sinh MỐC THAM CHIẾU thiên văn bằng PyEphem — độc lập hoàn toàn với mã trong
repo (PyEphem dựng trên XEphem, tức VSOP87 cho Mặt Trời và ELP2000 cho Mặt
Trăng; lunar.js thì dùng chuỗi rút gọn khớp DE423).

    python3 gen_ref.py > tools/refdata/astro_ref.json

Ba loại mốc:
  · TIẾT KHÍ — lúc hoàng kinh BIỂU KIẾN (xích đạo/hoàng đạo của ngày, đã kể
    quang sai và chương động) của Mặt Trời đạt đúng bội số của 15°;
  · SÓC / VỌNG — hợp sóc và vọng, đúng định nghĩa lịch Trung Hoa;
  · CHÍNH NGỌ — lúc Mặt Trời qua kinh tuyến trên của nơi quan sát.

Mọi mốc ghi bằng GIÂY UNIX (UTC), để bên kiểm thử khỏi phải tin vào bất cứ
phép quy đổi lịch nào của chính repo.
"""
import json
import math
import ephem

EPOCH = ephem.Date('1970/01/01 00:00:00')

def to_unix(d):
    return (float(d) - float(EPOCH)) * 86400.0

def dt_of(d):
    """ΔT (giây) mà PyEphem dùng tại thời điểm ấy.

    Ghi kèm từng mốc vì đây là chỗ hai bộ tính KHÔNG THỂ khớp nhau: ΔT là hiệu
    giữa thời gian nguyên tử và vòng quay thật của Trái Đất — một đại lượng ĐO
    ĐƯỢC cho quá khứ và chỉ ĐOÁN được cho tương lai. Hai thư viện dùng hai bộ
    đa thức khác nhau, nên mốc UT của cùng một sự kiện thiên văn lệch nhau
    đúng bằng hiệu hai ΔT ấy. Có con số này thì bên kiểm thử khử được nó ra và
    so phần THIÊN VĂN THẬT."""
    return float(ephem.delta_t(d))

def sun_lon_deg(d):
    """Hoàng kinh BIỂU KIẾN của Mặt Trời (độ), xuân phân của CHÍNH NGÀY ấy.

    Phải đi vòng qua Equatorial(epoch=d) chứ không viết thẳng
    `ephem.Ecliptic(ephem.Sun(d))`: hàm ấy lấy toạ độ TRẮC LƯỢNG của thiên thể,
    quy về xuân phân J2000. Tiết khí thì định nghĩa trên hoàng kinh biểu kiến
    của ngày, mà hai thứ lệch nhau đúng phần TUẾ SAI kể từ 2000 — 50,3 giây
    cung mỗi năm, tức 1,4° ở năm 1900 hay 2100, đổi thành 1,45 NGÀY chuyển
    động của Mặt Trời. Dùng nhầm thì chính mốc tham chiếu sai ngày rưỡi, mà
    con số vẫn trông đường hoàng.

    `s.ra`/`s.dec` sau khi compute(d) là toạ độ biểu kiến địa tâm ở xuân phân
    của ngày (đã kể quang sai và chương động) — đúng thứ cần.
    """
    s = ephem.Sun(d)
    eq = ephem.Equatorial(s.ra, s.dec, epoch=d)
    return math.degrees(float(ephem.Ecliptic(eq).lon)) % 360.0

def solve_term(target_deg, guess):
    """Tìm thời điểm hoàng kinh Mặt Trời = target_deg, quanh `guess` (ephem.Date).

    Chia đôi trên hiệu góc đã mở vòng — hiệu ấy tăng đơn điệu theo thời gian
    (Mặt Trời không bao giờ lùi trên hoàng đạo), nên chia đôi luôn hội tụ.
    """
    def f(d):
        return (sun_lon_deg(d) - target_deg + 180.0) % 360.0 - 180.0
    lo, hi = guess - 12.0, guess + 12.0
    flo = f(lo)
    if flo > 0:
        lo, hi = lo - 12.0, hi - 12.0
        flo = f(lo)
    for _ in range(80):
        mid = (lo + hi) / 2.0
        if f(mid) < 0:
            lo = mid
        else:
            hi = mid
    return (lo + hi) / 2.0

out = {'tietKhi': [], 'soc': [], 'vong': [], 'chinhNgo': []}

# ── Tiết khí: 24 mốc × các năm rải từ 1900 tới 2100 ──
# Đông Chí là 270°, rồi cứ 15° một. Dò từ mốc gần đúng theo ngày trung bình.
for year in [1900, 1950, 1991, 2000, 2026, 2050, 2100]:
    for i in range(24):
        deg = (270.0 + 15.0 * i) % 360.0
        # Đoán thô: Xuân Phân (0°) rơi ~20/3. Mỗi 15° ≈ 15.2 ngày.
        guess = ephem.Date('%d/03/20 12:00:00' % year) + deg * 365.2422 / 360.0
        if deg > 285:                      # 300°…355° thuộc đầu năm ấy
            guess -= 365.2422
        d = solve_term(deg, guess)
        out['tietKhi'].append({
            'year': year, 'i': i, 'deg': deg,
            'utc': to_unix(d), 'iso': str(ephem.Date(d)), 'dt': dt_of(d),
        })

# ── Sóc và Vọng: mọi tuần trăng trong vài năm ──
for year in [1900, 1991, 2000, 2026, 2050, 2100]:
    d = ephem.Date('%d/01/01 00:00:00' % year)
    end = ephem.Date('%d/01/01 00:00:00' % (year + 1))
    while True:
        nm = ephem.next_new_moon(d)
        if nm > end:
            break
        out['soc'].append({'year': year, 'utc': to_unix(nm), 'iso': str(nm), 'dt': dt_of(nm)})
        d = ephem.Date(nm + 1)
    d = ephem.Date('%d/01/01 00:00:00' % year)
    while True:
        fm = ephem.next_full_moon(d)
        if fm > end:
            break
        out['vong'].append({'year': year, 'utc': to_unix(fm), 'iso': str(fm), 'dt': dt_of(fm)})
        d = ephem.Date(fm + 1)

# ── Chính Ngọ: Mặt Trời qua kinh tuyến trên, tại vài nơi và vài ngày ──
NOI = [
    ('Ha Noi', 21.0278, 105.8342, 7),
    ('Paris', 48.8566, 2.3522, 1),
    ('Bac Kinh', 39.9042, 116.4074, 8),
    ('New York', 40.7128, -74.0060, -5),
    ('Reykjavik', 64.1466, -21.9426, 0),
    ('Sydney', -33.8688, 151.2093, 10),
]
NGAY = [(1991, 7, 16), (2026, 1, 15), (2026, 2, 4), (2026, 4, 1),
        (2026, 6, 21), (2026, 9, 19), (2026, 11, 3), (2026, 12, 21)]
for ten, lat, lon, tz in NOI:
    obs = ephem.Observer()
    obs.lat = str(lat)
    obs.lon = str(lon)
    obs.elevation = 0
    obs.pressure = 0            # tắt khúc xạ: qua kinh tuyến không phụ thuộc nó
    for (y, m, dd) in NGAY:
        # Nửa đêm ĐỊA PHƯƠNG của ngày ấy, viết bằng UTC — rồi lấy lần qua
        # kinh tuyến KẾ TIẾP. Lùi nửa ngày theo UTC như trước là sai ở những
        # múi âm: với New York (UTC−5), 00:00 UTC ngày 21 đã là 19:00 ngày 20
        # giờ địa phương, nên "lần kế tiếp" rơi vào trưa NGÀY HÔM TRƯỚC.
        obs.date = ephem.Date('%04d/%02d/%02d 00:00:00' % (y, m, dd)) - tz / 24.0
        tr = obs.next_transit(ephem.Sun())
        out['chinhNgo'].append({
            'noi': ten, 'lat': lat, 'lon': lon, 'tz': tz,
            'y': y, 'm': m, 'd': dd,
            'utc': to_unix(tr), 'iso': str(tr), 'dt': dt_of(tr),
        })

print(json.dumps(out))
