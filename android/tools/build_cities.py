#!/usr/bin/env python3
"""
Sinh `app/src/main/assets/web/data/cities.txt` — cơ sở dữ liệu thành phố offline.
Builds the offline city database bundled with the app.

    pip install geonamescache pytz
    python3 android/tools/build_cities.py

Định dạng / Format (phân tách bằng TAB):
    dòng 1 : bảng múi giờ IANA
    dòng 2 : bảng quốc gia, "MÃ|Tên tiếng Anh"
    dòng 3+: tên  vĩ độ  kinh độ  chỉ-số-múi-giờ  chỉ-số-quốc-gia  dân-số  [tên chữ Hán]

Các dòng được sắp theo dân số giảm dần, nên phần đầu file cũng chính là danh
sách "thành phố lớn" và kết quả tìm kiếm không cần sắp xếp lại.

Nguồn dữ liệu: GeoNames (https://www.geonames.org) — CC BY 4.0.
"""
import os
import re
import sys

try:
    import geonamescache
    import pytz
except ImportError:  # pragma: no cover
    sys.exit("Cần cài: pip install geonamescache pytz")

OUT = os.path.join(
    os.path.dirname(os.path.abspath(__file__)),
    "..", "app", "src", "main", "assets", "web", "data", "cities.txt",
)

# Chỉ nhận tên gồm toàn chữ Hán/Kanji — dùng làm từ khoá tìm kiếm cho bản 中文.
CJK = re.compile(r"^[㐀-䶿一-鿿々〆]+$")


def cjk_alias(city):
    for alt in city.get("alternatenames") or []:
        if CJK.match(alt):
            return alt
    return ""


def canonical_tz(tz, cc):
    """
    GeoNames gán nhầm múi giờ cho một số nơi — ví dụ Hà Nội và Hải Phòng bị ghi
    là Asia/Bangkok. Hai múi này cùng UTC+7 hôm nay nhưng KHÁC nhau trong quá
    khứ, nên lá số của người sinh trước 1975 sẽ sai giờ.

    Với quốc gia chỉ có đúng một múi giờ trong zone.tab, ép về múi giờ đó.
    Quốc gia nhiều múi giờ thì giữ nguyên (không đủ dữ liệu để phân định).
    """
    zones = pytz.country_timezones.get(cc, [])
    if len(zones) == 1 and tz != zones[0]:
        return zones[0]
    return tz


def main():
    cache = geonamescache.GeonamesCache()
    cities = sorted(cache.get_cities().values(), key=lambda c: -c["population"])
    countries = cache.get_countries()

    tz_list, tz_idx = [], {}
    cc_list, cc_idx = [], {}
    rows = []
    fixed = 0

    for c in cities:
        tz, cc = c["timezone"], c["countrycode"]
        if not tz:
            continue
        canon = canonical_tz(tz, cc)
        if canon != tz:
            fixed += 1
            tz = canon
        if tz not in tz_idx:
            tz_idx[tz] = len(tz_list)
            tz_list.append(tz)
        if cc not in cc_idx:
            cc_idx[cc] = len(cc_list)
            cc_list.append(cc)
        rows.append("\t".join((
            c["name"],
            f"{c['latitude']:.4f}".rstrip("0").rstrip("."),
            f"{c['longitude']:.4f}".rstrip("0").rstrip("."),
            str(tz_idx[tz]), str(cc_idx[cc]), str(c["population"]),
            cjk_alias(c),
        )))

    header_cc = "\t".join(
        cc + "|" + countries.get(cc, {}).get("name", cc) for cc in cc_list
    )
    out = "\n".join(["\t".join(tz_list), header_cc] + rows) + "\n"
    path = os.path.normpath(OUT)
    with open(path, "w", encoding="utf-8") as fh:
        fh.write(out)

    print(f"{len(rows)} thành phố · {len(tz_list)} múi giờ · {len(cc_list)} quốc gia")
    print(f"{fixed} múi giờ được chuẩn hoá lại theo zone.tab")
    print(f"{os.path.getsize(path):,} bytes → {path}")


if __name__ == "__main__":
    main()
