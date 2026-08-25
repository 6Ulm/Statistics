/* ════════════════════════════════════════════════════════════════════
   location.js — Vị trí thật (GPS / 34.000 thành phố / toạ độ thủ công)
   Real-location layer: GPS, offline city database, manual coordinates.

   Vai trò / Role
   ──────────────
   Bản web gốc chỉ có 28 thành phố cố định. Module này thay bằng MỌI toạ độ:
     • GPS của máy (offline, không cần mạng)
     • Tra cứu ~34.000 thành phố kèm múi giờ IANA (data/cities.txt)
     • Nhập tay vĩ độ / kinh độ / múi giờ

   Toạ độ đã chọn được nạp vào `countryData['__loc']` rồi trỏ
   `#country` sang khoá đó — nhờ vậy TOÀN BỘ engine Bát Tự / Kỳ Môn
   (processAll, tn_*, ab_*, sb_*) chạy nguyên vẹn, chỉ khác là kinh độ và
   múi giờ giờ đây là của đúng nơi người dùng chọn.

   The chosen coordinate is injected as `countryData['__loc']` and
   `#country` is pointed at it, so the untouched bazi/qimen engine
   automatically computes true solar time for that exact place.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var NATIVE = window.QMDJNative || null;
    var CUSTOM_KEY = '__loc';
    var K_LOC = 'qmdj.location';
    var K_RECENT = 'qmdj.recent';
    var MAX_RECENT = 8;

    /* ─────────────── Lưu trữ (native prefs → localStorage) ─────────────── */
    function prefGet(k) {
        try { if (NATIVE && NATIVE.getPref) { var v = NATIVE.getPref(k); if (v !== null && v !== '') return v; } } catch (e) {}
        try { return localStorage.getItem(k); } catch (e) { return null; }
    }
    function prefSet(k, v) {
        try { if (NATIVE && NATIVE.setPref) NATIVE.setPref(k, v); } catch (e) {}
        try { localStorage.setItem(k, v); } catch (e) {}
    }
    function jsonGet(k, dflt) {
        var raw = prefGet(k);
        if (!raw) return dflt;
        try { return JSON.parse(raw); } catch (e) { return dflt; }
    }

    /* ─────────────── Chuẩn hoá chuỗi tìm kiếm ─────────────── */
    /**
     * Bỏ dấu, bỏ khoảng trắng và dấu câu, thường hoá — nhờ vậy "hà nội",
     * "ha noi" và "Hanoi" đều khớp cùng một mục; "sai gon" khớp "Saigon".
     * Strips diacritics and separators so spacing/accents never block a match.
     */
    function fold(s) {
        return s.normalize('NFD')
            .replace(/[\u0300-\u036f]/g, '')
            // Các chữ cái KHÔNG tách dấu được bằng NFD, phải thay tay.
            // GeoNames viết "Ðà Lạt" bằng Ð (U+00D0, chữ eth) chứ không phải
            // Đ (U+0110) — thiếu dòng này thì gõ "da lat" không ra kết quả.
            .replace(/[đĐðÐ]/g, 'd')
            .replace(/[øØ]/g, 'o')
            .replace(/[łŁ]/g, 'l')
            .replace(/[æÆ]/g, 'ae')
            .replace(/[œŒ]/g, 'oe')
            .replace(/[ßẞ]/g, 'ss')
            .toLowerCase()
            .replace(/[^a-z0-9]/g, '');
    }

    /* ─────────────── CSDL thành phố offline ─────────────── */
    var cityDB = null;          // {tz:[], cc:[], ccName:{}, rows:[]}
    var cityLoading = null;

    /**
     * Nạp data/cities.txt (~1,3 MB, 34.006 thành phố).
     * Chỉ nạp khi người dùng mở bảng chọn vị trí lần đầu → không làm chậm khởi động.
     */
    function loadCities() {
        if (cityDB) return Promise.resolve(cityDB);
        if (cityLoading) return cityLoading;
        cityLoading = readAsset('data/cities.txt').then(function (text) {
            var lines = text.split('\n');
            var tz = lines[0].split('\t');
            var cc = [], ccName = {};
            lines[1].split('\t').forEach(function (pair) {
                var i = pair.indexOf('|');
                var code = pair.slice(0, i);
                cc.push(code);
                ccName[code] = pair.slice(i + 1);
            });
            var rows = new Array(lines.length - 2);
            var n = 0;
            for (var i = 2; i < lines.length; i++) {
                var L = lines[i];
                if (!L) continue;
                var p = L.split('\t');
                rows[n++] = {
                    name: p[0], f: fold(p[0]),
                    lat: +p[1], lon: +p[2],
                    tzId: tz[+p[3]], cc: cc[+p[4]], pop: +p[5],
                    z: p[6] || ''          // tên chữ Hán (nếu có) — chỉ dùng để tìm kiếm
                };
            }
            rows.length = n;
            cityDB = { tz: tz, cc: cc, ccName: ccName, rows: rows };
            return cityDB;
        });
        return cityLoading;
    }

    /** Đọc file trong assets: qua cầu native (file://) hoặc fetch (trình duyệt). */
    function readAsset(path) {
        if (NATIVE && NATIVE.readAsset) {
            try {
                var s = NATIVE.readAsset(path);
                if (s) return Promise.resolve(s);
            } catch (e) {}
        }
        return fetch(path).then(function (r) { return r.text(); });
    }

    /**
     * Tên gọi quen thuộc mà CSDL GeoNames không dùng làm tên chính.
     * Danh sách ngắn, chọn tay: tên tự động trích từ alternatenames cho kết quả
     * hú hoạ (Tokyo ra "Edo", HCM ra "Cathair Ho Chi Minh") nên không dùng.
     */
    var ALIASES = {
        saigon: 'hochiminhcity', sg: 'hochiminhcity', hcm: 'hochiminhcity',
        peking: 'beijing', canton: 'guangzhou', bombay: 'mumbai',
        calcutta: 'kolkata', madras: 'chennai', rangoon: 'yangon',
        edo: 'tokyo', kiev: 'kyiv', constantinople: 'istanbul',
        batavia: 'jakarta', danang: 'danang', hue: 'hue'
    };

    /**
     * Tìm thành phố theo tên Latinh (không dấu, không khoảng trắng) hoặc theo
     * tên chữ Hán — gõ "hà nội", "hanoi" hay "河內" đều ra cùng một kết quả.
     * Xếp hạng: khớp đầu chuỗi trước, rồi theo dân số (rows đã sắp sẵn).
     */
    function searchCities(q, limit) {
        if (!cityDB) return [];
        var raw = q.trim();
        var fq = fold(raw);
        if (ALIASES[fq]) fq = ALIASES[fq];
        var hasCJK = /[\u3400-\u9fff]/.test(raw);
        if (!fq && !hasCJK) return cityDB.rows.slice(0, limit || 40);
        var starts = [], contains = [];
        var rows = cityDB.rows, cap = limit || 60;
        for (var i = 0; i < rows.length; i++) {
            var r = rows[i], idx = -1;
            if (hasCJK && r.z) idx = r.z.indexOf(raw);
            if (idx < 0 && fq) idx = r.f.indexOf(fq);
            if (idx === 0) starts.push(r);
            else if (idx > 0 && contains.length < cap) contains.push(r);
            if (starts.length >= cap) break;
        }
        return starts.concat(contains).slice(0, cap);
    }

    /**
     * Thành phố gần một toạ độ nhất — dùng để suy ra múi giờ IANA cho điểm GPS
     * mà không cần mạng.
     * @param {number} [minPop] chỉ xét nơi có dân số ≥ ngưỡng này; hữu ích khi
     *        cần một cái TÊN dễ đọc (không lấy tên phường/quận nhỏ).
     */
    function nearestCity(lat, lon, minPop) {
        if (!cityDB) return null;
        var rows = cityDB.rows, best = null, bestD = Infinity;
        var cosLat = Math.cos(lat * Math.PI / 180);
        var floor = minPop || 0;
        for (var i = 0; i < rows.length; i++) {
            if (rows[i].pop < floor) continue;
            var dLat = rows[i].lat - lat;
            var dLon = (rows[i].lon - lon) * cosLat;
            var d = dLat * dLat + dLon * dLon;   // bình phương khoảng cách phẳng — đủ để so sánh
            if (d < bestD) { bestD = d; best = rows[i]; }
        }
        return best ? { city: best, distKm: Math.sqrt(bestD) * 111.32 } : null;
    }

    /* ─────────────── Mô hình vị trí ─────────────── */

    /**
     * @typedef {{name:string, nameZh:string, lat:number, lon:number,
     *            tzId:string, source:string}} Loc
     */

    function makeLoc(name, lat, lon, tzId, source, nameZh) {
        return {
            name: name, nameZh: nameZh || name,
            lat: Math.round(lat * 1e6) / 1e6,
            lon: Math.round(lon * 1e6) / 1e6,
            tzId: tzId, source: source || 'city'
        };
    }

    function locFromCity(c) {
        var label = c.name + ', ' + (c.cc || '');
        return makeLoc(label, c.lat, c.lon, c.tzId, 'city');
    }

    /** Vị trí đang dùng (null = đang dùng một mục có sẵn trong countryData). */
    var currentLoc = null;

    /**
     * Nạp vị trí vào engine: ghi vào countryData['__loc'] và trỏ #country sang đó.
     * Engine gốc (processAll) đọc `countryData[#country.value].lon / .tzId`
     * nên không cần sửa một dòng nào của app.js.
     */
    function applyLoc(loc, recalc) {
        currentLoc = loc;
        countryData[CUSTOM_KEY] = {
            name_vi: loc.name, name_zh: loc.nameZh || loc.name,
            tzId: loc.tzId, lon: loc.lon, lat: loc.lat
        };
        var sel = getDOM('country');
        if (sel) {
            var opt = null;
            for (var i = 0; i < sel.options.length; i++) {
                if (sel.options[i].value === CUSTOM_KEY) { opt = sel.options[i]; break; }
            }
            if (!opt) { opt = new Option(loc.name, CUSTOM_KEY); sel.add(opt); }
            opt.text = loc.name;
            sel.value = CUSTOM_KEY;
        }
        prefSet(K_LOC, JSON.stringify(loc));
        pushRecent(loc);
        updateCountryDisplay();
        if (recalc !== false && typeof Solar !== 'undefined' && typeof processAll === 'function') processAll();
    }

    function pushRecent(loc) {
        var list = jsonGet(K_RECENT, []);
        list = list.filter(function (l) {
            return !(Math.abs(l.lat - loc.lat) < 1e-4 && Math.abs(l.lon - loc.lon) < 1e-4);
        });
        list.unshift(loc);
        if (list.length > MAX_RECENT) list.length = MAX_RECENT;
        prefSet(K_RECENT, JSON.stringify(list));
    }

    /* ─────────────── GPS qua cầu native ─────────────── */
    var gpsPending = null;

    /**
     * Lấy toạ độ hiện tại. Ưu tiên LocationManager của Android (chạy được cả khi
     * không có mạng); nếu chạy trong trình duyệt thì dùng navigator.geolocation.
     */
    function requestGPS() {
        if (gpsPending) return gpsPending;
        gpsPending = new Promise(function (resolve, reject) {
            var done = false;
            var timer = setTimeout(function () {
                if (!done) { done = true; gpsPending = null; reject(new Error('timeout')); }
            }, 25000);

            window.__onNativeLocation = function (payload) {
                if (done) return;
                done = true; clearTimeout(timer); gpsPending = null;
                var p = typeof payload === 'string' ? JSON.parse(payload) : payload;
                if (p && p.error) reject(new Error(p.error)); else resolve(p);
            };

            if (NATIVE && NATIVE.requestLocation) {
                try { NATIVE.requestLocation(); return; } catch (e) {}
            }
            if (navigator.geolocation) {
                navigator.geolocation.getCurrentPosition(function (pos) {
                    window.__onNativeLocation({
                        lat: pos.coords.latitude, lon: pos.coords.longitude,
                        accuracy: pos.coords.accuracy, tzId: deviceTz()
                    });
                }, function (err) {
                    window.__onNativeLocation({ error: err.message || 'geolocation failed' });
                }, { enableHighAccuracy: true, timeout: 20000, maximumAge: 60000 });
                return;
            }
            done = true; clearTimeout(timer); gpsPending = null;
            reject(new Error('no-geolocation'));
        });
        return gpsPending;
    }

    function deviceTz() {
        try { if (NATIVE && NATIVE.deviceTimeZone) return NATIVE.deviceTimeZone(); } catch (e) {}
        try { return Intl.DateTimeFormat().resolvedOptions().timeZone; } catch (e) { return 'UTC'; }
    }

    /**
     * Suy ra múi giờ IANA cho một toạ độ, hoàn toàn offline:
     * lấy múi giờ của thành phố gần nhất trong CSDL; nếu điểm GPS ở rất gần
     * (< 150 km) và múi giờ máy đang cùng độ lệch UTC thì ưu tiên múi giờ máy
     * (chính xác hơn ở vùng giáp ranh).
     */
    function guessTz(lat, lon, hintTz) {
        var near = nearestCity(lat, lon);
        var cityTz = near ? near.city.tzId : 'UTC';
        if (hintTz && near && near.distKm < 150) {
            var now = new Date();
            if (getTimezoneOffset(hintTz, now) === getTimezoneOffset(cityTz, now)) return hintTz;
        }
        return cityTz;
    }

    /* ─────────────── Hiển thị & bảng chọn vị trí ─────────────── */

    function isZH() { return typeof currentLang !== 'undefined' && currentLang === 'zh'; }

    var T = {
        title:    { vi: 'Vị trí',                 zh: '位置' },
        cancel:   { vi: 'Hủy',                    zh: '取消' },
        ok:       { vi: 'Chọn',                   zh: '选择' },
        search:   { vi: 'Tìm thành phố…',         zh: '搜索城市…' },
        gps:      { vi: 'Dùng vị trí hiện tại',   zh: '使用当前位置' },
        gpsWait:  { vi: 'Đang định vị…',          zh: '定位中…' },
        gpsFail:  { vi: 'Không lấy được vị trí',  zh: '无法获取位置' },
        recent:   { vi: 'Gần đây',                zh: '最近' },
        presets:  { vi: 'Mặc định',               zh: '预设' },
        popular:  { vi: 'Thành phố lớn',          zh: '主要城市' },
        results:  { vi: 'Kết quả',                zh: '搜索结果' },
        manual:   { vi: 'Nhập toạ độ',            zh: '手动输入坐标' },
        lat:      { vi: 'Vĩ độ',                  zh: '纬度' },
        lon:      { vi: 'Kinh độ',                zh: '经度' },
        tz:       { vi: 'Múi giờ',                zh: '时区' },
        apply:    { vi: 'Áp dụng',                zh: '应用' },
        badCoord: { vi: 'Toạ độ không hợp lệ',    zh: '坐标无效' },
        loading:  { vi: 'Đang tải dữ liệu…',      zh: '正在加载…' },
        noResult: { vi: 'Không tìm thấy',         zh: '未找到' }
    };
    function t(k) { return T[k][isZH() ? 'zh' : 'vi']; }

    function fmtCoord(lat, lon) {
        return Math.abs(lat).toFixed(2) + '°' + (lat >= 0 ? 'N' : 'S') + ' ' +
               Math.abs(lon).toFixed(2) + '°' + (lon >= 0 ? 'E' : 'W');
    }

    function esc(s) {
        return String(s).replace(/[&<>"']/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;', "'": '&#39;' }[c];
        });
    }

    /** Nút hiển thị vị trí trên thanh điều khiển. */
    window.updateCountryDisplay = function () {
        var el = getDOM('countryDisplayText');
        if (!el) return;
        var key = getDOM('country') ? getDOM('country').value : null;
        var info = key ? countryData[key] : null;
        if (!info) return;
        el.textContent = info[isZH() ? 'name_zh' : 'name_vi'];
    };

    var searchTimer = null;

    function renderList(html) {
        var box = getDOM('locList');
        if (box) box.innerHTML = html;
    }

    function sectionHead(label) {
        return '<div class="loc-sec">' + esc(label) + '</div>';
    }

    function cityRow(name, sub, data) {
        return '<div class="loc-row" data-loc=\'' + esc(JSON.stringify(data)) + '\'>' +
               '<span class="loc-name">' + esc(name) + '</span>' +
               '<span class="loc-sub">' + esc(sub) + '</span></div>';
    }

    /** Danh sách mặc định: gần đây + các mục có sẵn + thành phố đông dân nhất. */
    function renderDefaultList() {
        var html = '';
        var recent = jsonGet(K_RECENT, []);
        if (recent.length) {
            html += sectionHead(t('recent'));
            recent.forEach(function (l) {
                html += cityRow(isZH() ? (l.nameZh || l.name) : l.name,
                    fmtCoord(l.lat, l.lon) + ' · ' + l.tzId, l);
            });
        }
        html += sectionHead(t('presets'));
        Object.keys(countryData).forEach(function (k) {
            if (k === CUSTOM_KEY) return;
            var c = countryData[k];
            if (typeof c.lat !== 'number') return;
            html += cityRow(c[isZH() ? 'name_zh' : 'name_vi'],
                fmtCoord(c.lat, c.lon) + ' · ' + c.tzId,
                makeLoc(c.name_vi, c.lat, c.lon, c.tzId, 'preset', c.name_zh));
        });
        if (cityDB) {
            html += sectionHead(t('popular'));
            cityDB.rows.slice(0, 30).forEach(function (c) {
                html += cityRow(c.name + ', ' + c.cc,
                    fmtCoord(c.lat, c.lon) + ' · ' + c.tzId, locFromCity(c));
            });
        }
        renderList(html);
    }

    function renderSearch(q) {
        if (!cityDB) { renderList(sectionHead(t('loading'))); return; }
        var res = searchCities(q, 60);
        if (!res.length) { renderList(sectionHead(t('noResult'))); return; }
        var html = sectionHead(t('results'));
        res.forEach(function (c) {
            html += cityRow(c.name + (c.z ? ' · ' + c.z : '') + ', ' + (cityDB.ccName[c.cc] || c.cc),
                fmtCoord(c.lat, c.lon) + ' · ' + c.tzId, locFromCity(c));
        });
        renderList(html);
    }

    function setStatus(msg, isError) {
        var el = getDOM('locStatus');
        if (!el) return;
        el.textContent = msg || '';
        el.style.display = msg ? 'block' : 'none';
        el.style.color = isError ? '#d44' : 'var(--text-dim, #555)';
    }

    window.openCountryPicker = function () {
        var ov = getDOM('locOverlay');
        if (!ov) return;
        ov.classList.add('open');
        getDOM('locTitle').textContent = t('title');
        getDOM('locCancel').textContent = t('cancel');
        getDOM('locGpsBtn').textContent = '📍 ' + t('gps');
        getDOM('locSearch').placeholder = t('search');
        getDOM('locSearch').value = '';
        getDOM('locManualToggle').textContent = '⌖ ' + t('manual');
        getDOM('locApply').textContent = t('apply');
        getDOM('lblLocLat').textContent = t('lat');
        getDOM('lblLocLon').textContent = t('lon');
        getDOM('lblLocTz').textContent = t('tz');
        setStatus('');
        if (currentLoc) {
            getDOM('locLat').value = currentLoc.lat;
            getDOM('locLon').value = currentLoc.lon;
        }
        renderDefaultList();
        loadCities().then(function () {
            fillTzOptions();
            if (!getDOM('locSearch').value) renderDefaultList();
        }).catch(function () { setStatus(t('noResult'), true); });
    };

    function closePicker() {
        var ov = getDOM('locOverlay');
        if (ov) ov.classList.remove('open');
    }

    var tzFilled = false;
    /**
     * Đổ danh sách múi giờ. Hợp nhất Intl.supportedValuesOf với bảng múi giờ
     * trong cities.txt và múi giờ đang dùng — thiếu một mục nào đó thì <select>
     * sẽ trả về chuỗi rỗng và ta lặng lẽ tính sai múi giờ.
     */
    function fillTzOptions() {
        var sel = getDOM('locTz');
        if (!sel) return;
        var wanted = (currentLoc && currentLoc.tzId) || deviceTz();
        if (!tzFilled) {
            var seen = Object.create(null), zones = [];
            var add = function (z) { if (z && !seen[z]) { seen[z] = 1; zones.push(z); } };
            try { if (Intl.supportedValuesOf) Intl.supportedValuesOf('timeZone').forEach(add); } catch (e) {}
            if (cityDB) cityDB.tz.forEach(add);
            add(deviceTz());
            add(wanted);
            add('UTC');
            zones.sort();
            sel.innerHTML = zones.map(function (z) {
                return '<option value="' + esc(z) + '">' + esc(z) + '</option>';
            }).join('');
            tzFilled = true;
        } else if (wanted && !sel.querySelector('option[value="' + wanted.replace(/"/g, '') + '"]')) {
            sel.add(new Option(wanted, wanted));
        }
        sel.value = wanted;
    }

    function onGpsClick() {
        setStatus(t('gpsWait'));
        loadCities()
            .then(function () { return requestGPS(); })
            .then(function (p) {
                var tzId = guessTz(p.lat, p.lon, p.tzId || deviceTz());
                // Tên hiển thị lấy từ đô thị đủ lớn gần nhất (tránh tên phường/xã).
                var named = nearestCity(p.lat, p.lon, 50000);
                var label = named && named.distKm < 60
                    ? named.city.name + ' (GPS)'
                    : fmtCoord(p.lat, p.lon);
                applyLoc(makeLoc(label, p.lat, p.lon, tzId, 'gps'));
                closePicker();
            })
            .catch(function (e) {
                setStatus(t('gpsFail') + (e && e.message ? ' — ' + e.message : ''), true);
            });
    }

    function onManualApply() {
        var lat = parseFloat(getDOM('locLat').value);
        var lon = parseFloat(getDOM('locLon').value);
        var tzId = getDOM('locTz').value || deviceTz();
        if (!isFinite(lat) || !isFinite(lon) || lat < -90 || lat > 90 || lon < -180 || lon > 180) {
            setStatus(t('badCoord'), true);
            return;
        }
        applyLoc(makeLoc(fmtCoord(lat, lon), lat, lon, tzId, 'manual'));
        closePicker();
    }

    /* ─────────────── Bảng Nhật – Nguyệt ─────────────── */

    var AT = {
        sun:     { vi: 'Mặt Trời',   zh: '太阳' },
        moon:    { vi: 'Mặt Trăng',  zh: '月亮' },
        rise:    { vi: 'mọc',        zh: '出' },
        set:     { vi: 'lặn',        zh: '落' },
        daylen:  { vi: 'Ngày dài',   zh: '昼长' },
        phase:   { vi: 'Pha',        zh: '月相' },
        lit:     { vi: 'sáng',       zh: '照度' },
        newmoon: { vi: 'Sóc kế',     zh: '下次朔' },
        coord:   { vi: 'Toạ độ',     zh: '坐标' },
        offset:  { vi: 'Lệch giờ MT thật', zh: '真太阳时差' },
        polarD:  { vi: 'Ngày vùng cực (không lặn)', zh: '极昼' },
        polarN:  { vi: 'Đêm vùng cực (không mọc)',  zh: '极夜' },
        noRise:  { vi: 'không mọc',  zh: '不出' },
        noSet:   { vi: 'không lặn',  zh: '不落' }
    };
    function at(k) { return AT[k][isZH() ? 'zh' : 'vi']; }

    function fmtDur(mins) {
        if (mins === null || mins === undefined) return '—';
        var h = Math.floor(mins / 60), mi = Math.round(mins % 60);
        return h + 'h' + (mi < 10 ? '0' : '') + mi;
    }

    /** Phút-trong-ngày, đánh dấu (+1) nếu rơi sang ngày hôm sau. */
    function fmtT(mins) {
        if (mins === null || mins === undefined) return '—';
        var s = Astro.fmtMins(mins);
        return mins >= 1440 ? s + ' (+1)' : (mins < 0 ? s + ' (−1)' : s);
    }

    /**
     * Vẽ bảng Nhật–Nguyệt cho ngày & vị trí đang chọn.
     * Gọi ngay sau processAll() để dùng đúng bộ input hiện hành.
     */
    function renderAstroPanel() {
        var panel = getDOM('astroPanel');
        if (!panel || typeof Astro === 'undefined') return;
        var key = getDOM('country').value;
        var info = countryData[key];
        if (!info) return;

        var y = parseInt(getDOM('inYear').value, 10);
        var m = parseInt(getDOM('inMonth').value, 10);
        var d = parseInt(getDOM('inDay').value, 10);
        var lat = typeof info.lat === 'number' ? info.lat : 0;
        var lon = info.lon;
        var tzH = getTimezoneOffset(info.tzId, new Date(y, m - 1, d, 12));

        var st = Astro.sunTimes(y, m, d, lat, lon, tzH);
        var mt = Astro.moonTimes(y, m, d, lat, lon, tzH);
        var jdNoon = Astro.jdFromUTC(y, m, d, 12, 0, 0) - tzH / 24;
        var ill = Astro.moonIllumination(jdNoon);

        // Sóc kế tiếp, đổi sang giờ địa phương của vị trí đã chọn.
        var nm = Astro.nearestPhase(jdNoon, 0);
        if (nm < jdNoon) nm = Astro.nearestPhase(jdNoon + 20, 0);
        var nmLocal = new Date(Astro.msFromJd(nm));

        var sunTxt = st.polarDay ? at('polarD')
                   : st.polarNight ? at('polarN')
                   : at('rise') + ' ' + fmtT(st.sunrise) + ' · ' + at('set') + ' ' + fmtT(st.sunset);
        var moonTxt = mt.alwaysUp ? at('noSet')
                    : mt.alwaysDown ? at('noRise')
                    : at('rise') + ' ' + (mt.moonrise === null ? at('noRise') : fmtT(mt.moonrise)) +
                      ' · ' + at('set') + ' ' + (mt.moonset === null ? at('noSet') : fmtT(mt.moonset));

        // Lệch giữa giờ đồng hồ và giờ Mặt Trời thật (kinh độ + phương trình thời gian)
        var offMins = (lon - tzH * 15) * 4 + st.equationOfTime;
        var offTxt = (offMins >= 0 ? '+' : '−') + Math.abs(offMins).toFixed(1) + (isZH() ? '分' : ' phút');

        setTxt('lblAstroSun', at('sun') + ':');
        setTxt('out-astro-sun', sunTxt);
        setTxt('lblAstroDaylen', at('daylen') + ':');
        setTxt('out-astro-daylen', st.polarDay ? '24h' : st.polarNight ? '0h' : fmtDur(st.dayLength));
        setTxt('lblAstroMoon', at('moon') + ':');
        setTxt('out-astro-moon', moonTxt);
        setTxt('lblAstroPhase', at('phase') + ':');
        setTxt('out-astro-phase', Astro.moonPhaseName(jdNoon, isZH() ? 'zh' : 'vi') +
            ' (' + Math.round(ill.fraction * 100) + '% ' + at('lit') + ')');
        setTxt('lblAstroNewMoon', at('newmoon') + ':');
        setTxt('out-astro-newmoon', fmtDateTimeInTz(nmLocal, info.tzId));
        setTxt('lblAstroCoord', at('coord') + ':');
        setTxt('out-astro-coord', fmtCoord(lat, lon));
        setTxt('lblAstroOffset', at('offset') + ':');
        setTxt('out-astro-offset', offTxt);

        panel.style.display = 'block';
    }

    function setTxt(id, s) { var el = getDOM(id); if (el) el.textContent = s; }

    function fmtDateTimeInTz(date, tzId) {
        try {
            return new Intl.DateTimeFormat('en-GB', {
                timeZone: tzId, day: '2-digit', month: '2-digit',
                hour: '2-digit', minute: '2-digit', hour12: false
            }).format(date).replace(',', '');
        } catch (e) {
            return date.toISOString().slice(5, 16).replace('T', ' ');
        }
    }

    /* ─────────────── Khởi tạo ─────────────── */

    document.addEventListener('DOMContentLoaded', function () {
        // Bọc processAll để vẽ thêm bảng Nhật–Nguyệt sau mỗi lần tính.
        if (typeof processAll === 'function' && !processAll.__wrapped) {
            var orig = processAll;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                try { renderAstroPanel(); } catch (e) { console.warn('astro panel:', e); }
                return r;
            };
            wrapped.__wrapped = true;
            window.processAll = wrapped;
        }

        // Đổi ngôn ngữ khi bảng chưa vẽ thì toggleLang() không gọi processAll(),
        // nhãn Nhật–Nguyệt sẽ kẹt ở ngôn ngữ cũ — vẽ lại cho chắc.
        if (typeof toggleLang === 'function' && !toggleLang.__wrapped) {
            var origLang = toggleLang;
            var wrappedLang = function () {
                var r = origLang.apply(this, arguments);
                if (getDOM('astroPanel') && getDOM('astroPanel').style.display === 'block') {
                    try { renderAstroPanel(); } catch (e) { console.warn('astro panel:', e); }
                }
                return r;
            };
            wrappedLang.__wrapped = true;
            window.toggleLang = wrappedLang;
        }

        getDOM('locGpsBtn').addEventListener('click', onGpsClick);
        getDOM('locCancel').addEventListener('click', closePicker);
        getDOM('locOverlay').addEventListener('click', function (e) { if (e.target === this) closePicker(); });
        getDOM('locApply').addEventListener('click', onManualApply);
        getDOM('locManualToggle').addEventListener('click', function () {
            var box = getDOM('locManualBox');
            var open = box.style.display === 'block';
            box.style.display = open ? 'none' : 'block';
            if (!open) fillTzOptions();
        });
        getDOM('locSearch').addEventListener('input', function () {
            var q = this.value;
            clearTimeout(searchTimer);
            searchTimer = setTimeout(function () {
                loadCities().then(function () {
                    if (q.trim()) renderSearch(q); else renderDefaultList();
                });
            }, 90);
        });
        getDOM('locList').addEventListener('click', function (e) {
            var row = e.target.closest ? e.target.closest('.loc-row') : null;
            if (!row) return;
            try {
                applyLoc(JSON.parse(row.getAttribute('data-loc')));
                closePicker();
            } catch (err) { console.warn('loc row:', err); }
        });

        // Khôi phục vị trí đã lưu; nếu chưa có thì suy từ múi giờ máy.
        var saved = jsonGet(K_LOC, null);
        if (saved && typeof saved.lat === 'number' && typeof saved.lon === 'number') {
            applyLoc(saved, false);
        } else {
            var tzId = deviceTz();
            var match = null;
            Object.keys(countryData).forEach(function (k) {
                if (k !== CUSTOM_KEY && countryData[k].tzId === tzId && !match) match = countryData[k];
            });
            if (match) {
                applyLoc(makeLoc(match.name_vi, match.lat, match.lon, match.tzId, 'preset', match.name_zh), false);
            }
        }
    });

    /**
     * Nút Back của Android: đóng bảng đang mở thay vì thoát ứng dụng.
     * Trả về true nếu đã xử lý.
     */
    window.__onBackPressed = function () {
        var ids = ['locOverlay', 'drumOverlay'];
        for (var i = 0; i < ids.length; i++) {
            var ov = getDOM(ids[i]);
            if (ov && ov.classList.contains('open')) { ov.classList.remove('open'); return true; }
        }
        return false;
    };

    // Cho phép lớp native đẩy vị trí vào bất cứ lúc nào (ví dụ sau khi cấp quyền).
    window.QMDJLocation = {
        apply: applyLoc, makeLoc: makeLoc, current: function () { return currentLoc; },
        requestGPS: requestGPS, guessTz: guessTz, loadCities: loadCities,
        searchCities: searchCities, nearestCity: nearestCity,
        renderAstroPanel: renderAstroPanel
    };
})();
