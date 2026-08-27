// --- GLOBAL ERROR HANDLER (tránh crash âm thầm trên Samsung/iOS WebView) ---
window.onerror = function(msg, src, line, col, err) {
    console.warn('KMDG error:', msg, 'line:', line);
    return true; // ngăn crash toàn trang
};
window.addEventListener('unhandledrejection', function(e) {
    console.warn('KMDG unhandled promise:', e.reason);
    e.preventDefault();
});

// --- SAFE STORAGE WRAPPER (tránh crash trên iOS standalone/private mode) ---
const safeStorage = {
    getItem: function(key) {
        try { return localStorage.getItem(key); } catch(e) { return null; }
    },
    setItem: function(key, val) {
        try { localStorage.setItem(key, val); } catch(e) {}
    },
    removeItem: function(key) {
        try { localStorage.removeItem(key); } catch(e) {}
    }
};

// --- GLOBAL UTILS (dùng toàn file) ---
/** Padding helper */
const pad = n => String(n).padStart(2, '0');

/** DOM cache — tra 1 lần, tái dùng mọi nơi */
const DOM = {};
function getDOM(id) { return DOM[id] || (DOM[id] = document.getElementById(id)); }

// --- COLLAPSIBLE DETAIL PANEL (Trí Nhuận / Sách Bổ / Âm Bàn) ---
const _panelIds = {
    trinhuan: { bodyId: 'trinhuanBody', chevId: 'trinhuanChevron' },
    sachbo:   { bodyId: 'sachboBody',   chevId: 'sachboChevron'   },
    amban:    { bodyId: 'ambanBody',    chevId: 'ambanChevron'    },
};
window.toggleDetailPanel = function(which) {
    const cfg  = _panelIds[which];
    if (!cfg) return;
    const body = getDOM(cfg.bodyId);
    const chev = getDOM(cfg.chevId);
    if (!body) return;
    const isOpen = body.style.display === 'block';
    body.style.display = isOpen ? 'none' : 'block';
    if (chev) chev.style.transform = isOpen ? 'rotate(0deg)' : 'rotate(180deg)';
};

// --- COUNTRY DATA & TIMEZONE HELPER ---
const countryData = {
    "FR": { name_vi: "Pháp (Paris)", name_zh: "法国（巴黎）", tzId: "Europe/Paris", lon: 2.352222, lat: 48.856614 },
    "VN-HN": { name_vi: "Việt Nam (Hà Nội)", name_zh: "越南（河内）", tzId: "Asia/Ho_Chi_Minh", lon: 105.834160, lat: 21.027764 },
    "VN-HCM": { name_vi: "Việt Nam (Hồ Chí Minh)", name_zh: "越南（胡志明市）", tzId: "Asia/Ho_Chi_Minh", lon: 106.660172, lat: 10.823099 },
    "CN": { name_vi: "Trung Quốc (Bắc Kinh)", name_zh: "中国（北京）", tzId: "Asia/Shanghai", lon: 116.407395, lat: 39.9042 },
    "JP": { name_vi: "Nhật Bản (Tokyo)", name_zh: "日本（东京）", tzId: "Asia/Tokyo", lon: 139.69, lat: 35.6895 },
    "KR": { name_vi: "Hàn Quốc (Seoul)", name_zh: "韩国（首尔）", tzId: "Asia/Seoul", lon: 126.98, lat: 37.5665 },
    "TH": { name_vi: "Thái Lan (Bangkok)", name_zh: "泰国（曼谷）", tzId: "Asia/Bangkok", lon: 100.52, lat: 13.7563 },
    "SG": { name_vi: "Singapore", name_zh: "新加坡", tzId: "Asia/Singapore", lon: 103.82, lat: 1.3521 },
    "MY": { name_vi: "Malaysia (Kuala Lumpur)", name_zh: "马来西亚（吉隆坡）", tzId: "Asia/Kuala_Lumpur", lon: 101.69, lat: 3.139 },
    "ID": { name_vi: "Indonesia (Jakarta)", name_zh: "印尼（雅加达）", tzId: "Asia/Jakarta", lon: 106.85, lat: -6.2088 },
    "PH": { name_vi: "Philippines (Manila)", name_zh: "菲律宾（马尼拉）", tzId: "Asia/Manila", lon: 120.98, lat: 14.5995 },
    "HK": { name_vi: "Hồng Kông", name_zh: "香港", tzId: "Asia/Hong_Kong", lon: 114.16, lat: 22.3193 },
    "TW": { name_vi: "Đài Loan (Taipei)", name_zh: "台湾（台北）", tzId: "Asia/Taipei", lon: 121.57, lat: 25.033 },
    "KH": { name_vi: "Campuchia (Phnom Penh)", name_zh: "柬埔寨（金边）", tzId: "Asia/Phnom_Penh", lon: 104.93, lat: 11.5564 },
    "LA": { name_vi: "Lào (Vientiane)", name_zh: "老挝（万象）", tzId: "Asia/Vientiane", lon: 102.63, lat: 17.9757 },
    "MM": { name_vi: "Myanmar (Naypyidaw)", name_zh: "缅甸（内比都）", tzId: "Asia/Rangoon", lon: 96.13, lat: 19.7633 },
    "RU": { name_vi: "Nga (Moscow)", name_zh: "俄罗斯（莫斯科）", tzId: "Europe/Moscow", lon: 37.62, lat: 55.7558 },
    "DE": { name_vi: "Đức (Berlin)", name_zh: "德国（柏林）", tzId: "Europe/Berlin", lon: 13.41, lat: 52.52 },
    "GB": { name_vi: "Anh (London)", name_zh: "英国（伦敦）", tzId: "Europe/London", lon: -0.13, lat: 51.5074 },
    "IT": { name_vi: "Italy (Rome)", name_zh: "意大利（罗马）", tzId: "Europe/Rome", lon: 12.50, lat: 41.9028 },
    "ES": { name_vi: "Tây Ban Nha (Madrid)", name_zh: "西班牙（马德里）", tzId: "Europe/Madrid", lon: -3.70, lat: 40.4168 },
    "NL": { name_vi: "Hà Lan (Amsterdam)", name_zh: "荷兰（阿姆斯特丹）", tzId: "Europe/Amsterdam", lon: 4.90, lat: 52.3676 },
    "SE": { name_vi: "Thụy Điển (Stockholm)", name_zh: "瑞典（斯德哥尔摩）", tzId: "Europe/Stockholm", lon: 18.07, lat: 59.3293 },
    "NO": { name_vi: "Na Uy (Oslo)", name_zh: "挪威（奥斯陆）", tzId: "Europe/Oslo", lon: 10.75, lat: 59.9139 },
    "CH": { name_vi: "Thụy Sĩ (Bern)", name_zh: "瑞士（伯尔尼）", tzId: "Europe/Zurich", lon: 7.45, lat: 46.948 },
    "PL": { name_vi: "Ba Lan (Warsaw)", name_zh: "波兰（华沙）", tzId: "Europe/Warsaw", lon: 21.01, lat: 52.2297 },
    "US_ET": { name_vi: "Mỹ (New York, ET)", name_zh: "美国（纽约, ET）", tzId: "America/New_York", lon: -74.01, lat: 40.7128 },
    "CA": { name_vi: "Canada (Toronto)", name_zh: "加拿大（多伦多）", tzId: "America/Toronto", lon: -79.38, lat: 43.6532 },
};

function getTimezoneOffset(tzId, date) {
    const fmt = (tz) => new Intl.DateTimeFormat('en-CA', {
        timeZone: tz, year: 'numeric', month: '2-digit', day: '2-digit',
        hour: '2-digit', minute: '2-digit', second: '2-digit', hour12: false
    }).formatToParts(date);
    const toMs = (parts) => {
        const g = (t) => parseInt(parts.find(p => p.type === t).value);
        return Date.UTC(g('year'), g('month') - 1, g('day'), g('hour'), g('minute'), g('second'));
    };
    try {
        const utcMs = toMs(fmt('UTC'));
        const localMs = toMs(fmt(tzId));
        return (localMs - utcMs) / 3600000;
    } catch(e) {
        return 0;
    }
}

// --- DICTIONARIES & MAPPING ---

// ── Mảng nguồn (source of truth) ──
const arrGanZH = ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"];
const arrZhiZH = ["子", "丑", "寅", "卯", "辰", "巳", "午", "未", "申", "酉", "戌", "亥"];
const chiVI    = ["Tý","Sửu","Dần","Mão","Thìn","Tỵ","Ngọ","Mùi","Thân","Dậu","Tuất","Hợi"];

/** 24 tiết khí theo thứ tự bắt đầu từ Đông Chí */
const TK_VI = [
    'Đông Chí','Tiểu Hàn','Đại Hàn',
    'Lập Xuân','Vũ Thủy','Kinh Trập',
    'Xuân Phân','Thanh Minh','Cốc Vũ',
    'Lập Hạ','Tiểu Mãn','Mang Chủng',
    'Hạ Chí','Tiểu Thử','Đại Thử',
    'Lập Thu','Xử Thử','Bạch Lộ',
    'Thu Phân','Hàn Lộ','Sương Giáng',
    'Lập Đông','Tiểu Tuyết','Đại Tuyết'
];
const TK_ZH = [
    '冬至','小寒','大寒',
    '立春','雨水','惊蛰',
    '春分','清明','谷雨',
    '立夏','小满','芒种',
    '夏至','小暑','大暑',
    '立秋','处暑','白露',
    '秋分','寒露','霜降',
    '立冬','小雪','大雪'
];

// ── Computed maps (built from source arrays — không lặp data) ──
const tietKhiMap    = Object.fromEntries(TK_ZH.map((zh, i) => [zh, TK_VI[i]]));  // ZH→VI
const chiMapping    = Object.fromEntries(chiVI.map((v, i) => [v, arrZhiZH[i]])); // VI→ZH
const chiToStt      = Object.fromEntries(arrZhiZH.map((z, i) => [z, i + 1]));    // ZH→1-based index
const TK_AM_DON     = new Set(TK_ZH.slice(12)); // Hạ Chí → Đại Tuyết (âm độn)

const NT1 = [4, 9, 2, 7, 6, 1, 8, 3];
const ltNumberMap = {4: "4 5 3 8", 9: "9 3 2 7", 2: "2 8 5 10", 7: "7 2 4 9", 6: "6 1 4 9", 1: "1 6 1 6", 8: "8 7 5 10", 3: "3 4 3 8"}
// Mảng đã tách sẵn (tránh .split(' ') lặp lại ở mỗi lần render cell)
const ltNumberArrMap = Object.fromEntries(
    Object.entries(ltNumberMap).map(([lt, s]) => [lt, s.split(' ')])
);
const hauThienMapCung = { 1: "☵", 2: "☷", 3: "☳", 4: "☴", 6: "☰", 7: "☱", 8: "☶", 9: "☲" };
const tienThienMapCung = { 1: "☷", 2: "☴", 3: "☲", 4: "☱", 6: "☶", 7: "☵", 8: "☳", 9: "☰" };

const dataBase = {
    vi: { tinh: ["Phụ", "Anh", "Nhuế", "Trụ", "Tâm", "Bồng", "Nhậm", "Xung"], mon: ["Đỗ", "Cảnh", "Tử", "Kinh", "Khai", "Hưu", "Sinh", "Thương"], thanD: ["Phù", "Xà", "Âm", "Hợp", "Hổ", "Vũ", "Địa", "Thiên"], thanA: ["Phù", "Thiên", "Địa", "Vũ", "Hổ", "Hợp", "Âm", "Xà"], can: ["Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý", "Đinh", "Bính", "Ất"], canFull: ["Giáp", "Ất", "Bính", "Đinh", "Mậu", "Kỷ", "Canh", "Tân", "Nhâm", "Quý"] },
    zh: { tinh: ["辅", "英", "芮", "柱", "心", "蓬", "任", "冲"], mon: ["杜", "景", "死", "惊", "开", "休", "生", "伤"], thanD: ["符", "蛇", "阴", "合", "虎", "玄", "地", "天"], thanA: ["符", "天", "地", "玄", "虎", "合", "阴", "蛇"], can: ["戊", "己", "庚", "辛", "壬", "癸", "丁", "丙", "乙"], canFull: ["甲", "乙", "丙", "丁", "戊", "己", "庚", "辛", "壬", "癸"] }
};

const dayToTuanThu = { "Giáp Tý": "Mậu", "Ất Sửu": "Mậu", "Bính Dần": "Mậu", "Đinh Mão": "Mậu", "Mậu Thìn": "Mậu", "Kỷ Tỵ": "Mậu", "Canh Ngọ": "Mậu", "Tân Mùi": "Mậu", "Nhâm Thân": "Mậu", "Quý Dậu": "Mậu", "Giáp Tuất": "Kỷ", "Ất Hợi": "Kỷ", "Bính Tý": "Kỷ", "Đinh Sửu": "Kỷ", "Mậu Dần": "Kỷ", "Kỷ Mão": "Kỷ", "Canh Thìn": "Kỷ", "Tân Tỵ": "Kỷ", "Nhâm Ngọ": "Kỷ", "Quý Mùi": "Kỷ", "Giáp Thân": "Canh", "Ất Dậu": "Canh", "Bính Tuất": "Canh", "Đinh Hợi": "Canh", "Mậu Tý": "Canh", "Kỷ Sửu": "Canh", "Canh Dần": "Canh", "Tân Mão": "Canh", "Nhâm Thìn": "Canh", "Quý Tỵ": "Canh", "Giáp Ngọ": "Tân", "Ất Mùi": "Tân", "Bính Thân": "Tân", "Đinh Dậu": "Tân", "Mậu Tuất": "Tân", "Kỷ Hợi": "Tân", "Canh Tý": "Tân", "Tân Sửu": "Tân", "Nhâm Dần": "Tân", "Quý Mão": "Tân", "Giáp Thìn": "Nhâm", "Ất Tỵ": "Nhâm", "Bính Ngọ": "Nhâm", "Đinh Mùi": "Nhâm", "Mậu Thân": "Nhâm", "Kỷ Dậu": "Nhâm", "Canh Tuất": "Nhâm", "Tân Hợi": "Nhâm", "Nhâm Tý": "Nhâm", "Quý Sửu": "Nhâm", "Giáp Dần": "Quý", "Ất Mão": "Quý", "Bính Thìn": "Quý", "Đinh Tỵ": "Quý", "Mậu Ngọ": "Quý", "Kỷ Mùi": "Quý", "Canh Thân": "Quý", "Tân Dậu": "Quý", "Nhâm Tuất": "Quý", "Quý Hợi": "Quý" };
const thuToTuanGiap = { "Mậu": "Giáp Tý", "Kỷ": "Giáp Tuất", "Canh": "Giáp Thân", "Tân": "Giáp Ngọ", "Nhâm": "Giáp Thìn", "Quý": "Giáp Dần" };
const hoaGiapToKhongVong = { "Giáp Tý": "Tuất Hợi", "Ất Sửu": "Tuất Hợi", "Bính Dần": "Tuất Hợi", "Đinh Mão": "Tuất Hợi", "Mậu Thìn": "Tuất Hợi", "Kỷ Tỵ": "Tuất Hợi", "Canh Ngọ": "Tuất Hợi", "Tân Mùi": "Tuất Hợi", "Nhâm Thân": "Tuất Hợi", "Quý Dậu": "Tuất Hợi", "Giáp Tuất": "Thân Dậu", "Ất Hợi": "Thân Dậu", "Bính Tý": "Thân Dậu", "Đinh Sửu": "Thân Dậu", "Mậu Dần": "Thân Dậu", "Kỷ Mão": "Thân Dậu", "Canh Thìn": "Thân Dậu", "Tân Tỵ": "Thân Dậu", "Nhâm Ngọ": "Thân Dậu", "Quý Mùi": "Thân Dậu", "Giáp Thân": "Ngọ Mùi", "Ất Dậu": "Ngọ Mùi", "Bính Tuất": "Ngọ Mùi", "Đinh Hợi": "Ngọ Mùi", "Mậu Tý": "Ngọ Mùi", "Kỷ Sửu": "Ngọ Mùi", "Canh Dần": "Ngọ Mùi", "Tân Mão": "Ngọ Mùi", "Nhâm Thìn": "Ngọ Mùi", "Quý Tỵ": "Ngọ Mùi", "Giáp Ngọ": "Thìn Tỵ", "Ất Mùi": "Thìn Tỵ", "Bính Thân": "Thìn Tỵ", "Đinh Dậu": "Thìn Tỵ", "Mậu Tuất": "Thìn Tỵ", "Kỷ Hợi": "Thìn Tỵ", "Canh Tý": "Thìn Tỵ", "Tân Sửu": "Thìn Tỵ", "Nhâm Dần": "Thìn Tỵ", "Quý Mão": "Thìn Tỵ", "Giáp Thìn": "Dần Mão", "Ất Tỵ": "Dần Mão", "Bính Ngọ": "Dần Mão", "Đinh Mùi": "Dần Mão", "Mậu Thân": "Dần Mão", "Kỷ Dậu": "Dần Mão", "Canh Tuất": "Dần Mão", "Tân Hợi": "Dần Mão", "Nhâm Tý": "Dần Mão", "Quý Sửu": "Dần Mão", "Giáp Dần": "Tý Sửu", "Ất Mão": "Tý Sửu", "Bính Thìn": "Tý Sửu", "Đinh Tỵ": "Tý Sửu", "Mậu Ngọ": "Tý Sửu", "Kỷ Mùi": "Tý Sửu", "Canh Thân": "Tý Sửu", "Tân Dậu": "Tý Sửu", "Nhâm Tuất": "Tý Sửu", "Quý Hợi": "Tý Sửu" };
const hoaGiapToDichMa = { "Giáp Tý": "Dần", "Ất Sửu": "Hợi", "Bính Dần": "Thân", "Đinh Mão": "Tỵ", "Mậu Thìn": "Dần", "Kỷ Tỵ": "Hợi", "Canh Ngọ": "Thân", "Tân Mùi": "Tỵ", "Nhâm Thân": "Dần", "Quý Dậu": "Hợi", "Giáp Tuất": "Thân", "Ất Hợi": "Tỵ", "Bính Tý": "Dần", "Đinh Sửu": "Hợi", "Mậu Dần": "Thân", "Kỷ Mão": "Tỵ", "Canh Thìn": "Dần", "Tân Tỵ": "Hợi", "Nhâm Ngọ": "Thân", "Quý Mùi": "Tỵ", "Giáp Thân": "Dần", "Ất Dậu": "Hợi", "Bính Tuất": "Thân", "Đinh Hợi": "Tỵ", "Mậu Tý": "Dần", "Kỷ Sửu": "Hợi", "Canh Dần": "Thân", "Tân Mão": "Tỵ", "Nhâm Thìn": "Dần", "Quý Tỵ": "Hợi", "Giáp Ngọ": "Thân", "Ất Mùi": "Tỵ", "Bính Thân": "Dần", "Đinh Dậu": "Hợi", "Mậu Tuất": "Thân", "Kỷ Hợi": "Tỵ", "Canh Tý": "Dần", "Tân Sửu": "Hợi", "Nhâm Dần": "Thân", "Quý Mão": "Tỵ", "Giáp Thìn": "Dần", "Ất Tỵ": "Hợi", "Bính Ngọ": "Thân", "Đinh Mùi": "Tỵ", "Mậu Thân": "Dần", "Kỷ Dậu": "Hợi", "Canh Tuất": "Thân", "Tân Hợi": "Tỵ", "Nhâm Tý": "Dần", "Quý Sửu": "Hợi", "Giáp Dần": "Thân", "Ất Mão": "Tỵ", "Bính Thìn": "Dần", "Đinh Tỵ": "Hợi", "Mậu Ngọ": "Thân", "Kỷ Mùi": "Tỵ", "Canh Thân": "Dần", "Tân Dậu": "Hợi", "Nhâm Tuất": "Thân", "Quý Hợi": "Tỵ" };

// Nạp Âm Ngũ Hành cho 60 Can-Chi
const canChiToNapAm = {
    "Giáp Tý":   "Hải Trung Kim",  "Ất Sửu":   "Hải Trung Kim",
    "Bính Dần":  "Lô Trung Hỏa",   "Đinh Mão":  "Lô Trung Hỏa",
    "Mậu Thìn":  "Đại Lâm Mộc",    "Kỷ Tỵ":    "Đại Lâm Mộc",
    "Canh Ngọ":  "Lộ Bàng Thổ",    "Tân Mùi":   "Lộ Bàng Thổ",
    "Nhâm Thân": "Kiếm Phong Kim",  "Quý Dậu":   "Kiếm Phong Kim",
    "Giáp Tuất": "Sơn Đầu Hỏa",    "Ất Hợi":    "Sơn Đầu Hỏa",
    "Bính Tý":   "Giản Hạ Thủy",   "Đinh Sửu":  "Giản Hạ Thủy",
    "Mậu Dần":   "Thành Đầu Thổ",  "Kỷ Mão":    "Thành Đầu Thổ",
    "Canh Thìn": "Bạch Lạp Kim",    "Tân Tỵ":    "Bạch Lạp Kim",
    "Nhâm Ngọ":  "Dương Liễu Mộc", "Quý Mùi":   "Dương Liễu Mộc",
    "Giáp Thân": "Tuyền Trung Thủy","Ất Dậu":    "Tuyền Trung Thủy",
    "Bính Tuất": "Ốc Thượng Thổ",  "Đinh Hợi":  "Ốc Thượng Thổ",
    "Mậu Tý":    "Tích Lịch Hỏa",  "Kỷ Sửu":    "Tích Lịch Hỏa",
    "Canh Dần":  "Tùng Bách Mộc",  "Tân Mão":   "Tùng Bách Mộc",
    "Nhâm Thìn": "Trường Lưu Thủy","Quý Tỵ":    "Trường Lưu Thủy",
    "Giáp Ngọ":  "Sa Trung Kim",    "Ất Mùi":    "Sa Trung Kim",
    "Bính Thân": "Sơn Hạ Hỏa",     "Đinh Dậu":  "Sơn Hạ Hỏa",
    "Mậu Tuất":  "Bình Địa Mộc",   "Kỷ Hợi":    "Bình Địa Mộc",
    "Canh Tý":   "Bích Thượng Thổ","Tân Sửu":   "Bích Thượng Thổ",
    "Nhâm Dần":  "Kim Bạch Kim",    "Quý Mão":   "Kim Bạch Kim",
    "Giáp Thìn": "Phú Đăng Hỏa",   "Ất Tỵ":     "Phú Đăng Hỏa",
    "Bính Ngọ":  "Thiên Hà Thủy",  "Đinh Mùi":  "Thiên Hà Thủy",
    "Mậu Thân":  "Đại Dịch Thổ",   "Kỷ Dậu":    "Đại Dịch Thổ",
    "Canh Tuất": "Thoa Xuyến Kim",  "Tân Hợi":   "Thoa Xuyến Kim",
    "Nhâm Tý":   "Tang Chá Mộc",   "Quý Sửu":   "Tang Chá Mộc",
    "Giáp Dần":  "Đại Khê Thủy",   "Ất Mão":    "Đại Khê Thủy",
    "Bính Thìn": "Sa Trung Thổ",    "Đinh Tỵ":   "Sa Trung Thổ",
    "Mậu Ngọ":   "Thiên Thượng Hỏa","Kỷ Mùi":   "Thiên Thượng Hỏa",
    "Canh Thân": "Thạch Lựu Mộc",  "Tân Dậu":   "Thạch Lựu Mộc",
    "Nhâm Tuất": "Đại Hải Thủy",   "Quý Hợi":   "Đại Hải Thủy"
};

// Dịch tên Nạp Âm sang tiếng Trung
const napAmToZH = {
    "Hải Trung Kim":   "海中金",
    "Lô Trung Hỏa":   "炉中火",
    "Đại Lâm Mộc":    "大林木",
    "Lộ Bàng Thổ":    "路旁土",
    "Kiếm Phong Kim":  "剑锋金",
    "Sơn Đầu Hỏa":    "山头火",
    "Giản Hạ Thủy":   "涧下水",
    "Thành Đầu Thổ":  "城头土",
    "Bạch Lạp Kim":   "白蜡金",
    "Dương Liễu Mộc": "杨柳木",
    "Tuyền Trung Thủy":"泉中水",
    "Ốc Thượng Thổ":  "屋上土",
    "Tích Lịch Hỏa":  "霹雳火",
    "Tùng Bách Mộc":  "松柏木",
    "Trường Lưu Thủy":"长流水",
    "Sa Trung Kim":    "沙中金",
    "Sơn Hạ Hỏa":     "山下火",
    "Bình Địa Mộc":   "平地木",
    "Bích Thượng Thổ":"壁上土",
    "Kim Bạch Kim":    "金箔金",
    "Phú Đăng Hỏa":   "覆灯火",
    "Thiên Hà Thủy":  "天河水",
    "Đại Dịch Thổ":   "大驿土",
    "Thoa Xuyến Kim":  "钗钏金",
    "Tang Chá Mộc":   "桑柘木",
    "Đại Khê Thủy":   "大溪水",
    "Sa Trung Thổ":    "沙中土",
    "Thiên Thượng Hỏa":"天上火",
    "Thạch Lựu Mộc":  "石榴木",
    "Đại Hải Thủy":   "大海水"
};
const chiToLT = {"Tý":1, "子":1, "Sửu":8, "丑":8, "Dần":8, "寅":8, "Mão":3, "卯":3, "Thìn":4, "辰":4, "Tỵ":4, "巳":4, "Ngọ":9, "午":9, "Mùi":2, "未":2, "Thân":2, "申":2, "Dậu":7, "酉":7, "Tuất":6, "戌":6, "Hợi":6, "亥":6};

// mapToVi: ZH can/chi → VI, built from arrGanZH+dataBase+chiVI (no duplicate data)
const mapToVi = Object.fromEntries([
    ...arrGanZH.map((z, i) => [z, dataBase.vi.canFull[i]]),
    ...arrZhiZH.map((z, i) => [z, chiVI[i]])
]);

// quaiMap: tên tinh/môn (cả VI+ZH) → quẻ, built from dataBase
const _tinhQuai = ["☴","☲","☷","☱","☰","☵","☶","☳"];
const quaiMap = Object.fromEntries([
    ...dataBase.vi.tinh.map((n, i) => [n, _tinhQuai[i]]),
    ...dataBase.zh.tinh.map((n, i) => [n, _tinhQuai[i]]),
    ...dataBase.vi.mon .map((n, i) => [n, _tinhQuai[i]]),
    ...dataBase.zh.mon .map((n, i) => [n, _tinhQuai[i]])
]);

// redList: các tên cát (VI+ZH), built from dataBase
// vi: mon[5]=Hưu, mon[4]=Khai, mon[6]=Sinh | thanA[5]=Hợp, thanA[0]=Phù, thanA[6]=Âm, thanD[7]=Thiên, thanD[6]=Địa | tinh[4]=Tâm, tinh[6]=Nhậm, tinh[0]=Phụ
const _redIdx = { mon:[5,4,6], thanA:[5,0,6], thanD:[7,6], tinh:[4,6,0] };
const redSet = new Set(Object.entries(_redIdx).flatMap(([k, idxs]) => [
    ...idxs.map(i => dataBase.vi[k][i]),
    ...idxs.map(i => dataBase.zh[k][i])
]));
/** Lớp "cát" (tốt) → class CSS txt-red. O(1) qua Set, dùng chung cho mọi cell. */
const getRC = v => redSet.has(v) ? 'txt-red' : '';

// UI TRANSLATION DICTIONARY
const uiDict = {
    vi: {
        header: "NHẬP THÔNG TIN", chinhNgo: "Chính Ngọ:", gmt: "Múi giờ:",
        lunar: "Lịch âm:", viaDay: "Ngày vía:", soc: "Sóc:", baziLunar: "Bát tự (Âm):", baziSolar: "Bát tự (Dương):",
        tietkhi: "Tiết khí:", cuc: "Cục:", don: "Độn:", tuan: "Tuần thủ:",
        tp: "Trực Phù:", ts: "Trực Sử:", dm: "Mã:", kv: "Nạp âm:",
        donD: "Dương", donA: "Âm", solarDate: "Lịch dương:",
        methodAmBan: "Âm Bàn", methodBoPhap: "Sách Bổ", methodTriNhuan: "Trí Nhuận", coban: "Đầy đủ"
    },
    zh: {
        header: "输入信息", chinhNgo: "正午时间:", gmt: "时区:",
        lunar: "农历:", viaDay: "圣诞:", soc: "朔点:", baziLunar: "八字 (农历):", baziSolar: "八字 (阳历):",
        tietkhi: "节气:", cuc: "局数:", don: "遁:", tuan: "旬首:",
        tp: "值符:", ts: "值使:", dm: "马:", kv: "纳音:",
        donD: "阳", donA: "阴", solarDate: "公历:",
        methodAmBan: "阴盘", methodBoPhap: "拆补", methodTriNhuan: "置润", coban: "完整"
    }
};

let currentLang = 'zh';

/* ======================================================================
   KHỞI TẠO DOM
   ====================================================================== */
document.addEventListener("DOMContentLoaded", function() {
    const ySel   = getDOM('inYear');
    const mSel   = getDOM('inMonth');
    const dSel   = getDOM('inDay');
    const hSel   = getDOM('solarHour');
    const minSel = getDOM('solarMinute');

    for(let i=1900; i<=2100; i++) ySel.add(new Option("Năm " + i, i));
    for(let i=1; i<=12; i++)  mSel.add(new Option("Tháng " + i, i));
    for(let i=1; i<=31; i++)  dSel.add(new Option("Ngày " + i, i));
    for(let i=0; i<24; i++)   hSel.add(new Option(i + "h", i));
    for(let i=0; i<60; i++)   minSel.add(new Option(i + "p", i));

    const today = new Date();
    ySel.value   = today.getFullYear();
    mSel.value   = today.getMonth() + 1;
    dSel.value   = today.getDate();
    hSel.value   = today.getHours();
    minSel.value = today.getMinutes();

    if (typeof updateDateDisplay === 'function')    updateDateDisplay();
    if (typeof updateCountryDisplay === 'function') updateCountryDisplay();
    // Khởi tạo UI labels theo ngôn ngữ mặc định
    // Dùng setTimeout để đảm bảo tất cả DOMContentLoaded (drum picker override) đã chạy xong
    const savedLang   = safeStorage.getItem('defaultLang');
    const savedMethod = safeStorage.getItem('defaultMethod');
    const initLang = (savedLang === 'vi' || savedLang === 'zh') ? savedLang : 'zh';
    setTimeout(() => {
        if (typeof setLang === 'function') { currentLang = initLang === 'zh' ? 'vi' : 'zh'; setLang(initLang); }
        // Load saved default method
        if (savedMethod) { selectMethod(savedMethod); }
        // Restore co-ban filter — default is coban-mode (basic); '1' = full view (Đầy đủ)
        const savedCoBan = safeStorage.getItem('chkHienThiCoBan');
        const chk = getDOM('chkHienThiCoBan');
        if (savedCoBan === '1') {
            // Full view: check box, no coban-mode
            if (chk) { chk.checked = true; getDOM('mainBody').classList.remove('coban-mode'); }
        } else {
            // Basic view (default): uncheck box, add coban-mode
            if (chk) { chk.checked = false; getDOM('mainBody').classList.add('coban-mode'); }
        }
    }, 0);

    // Build country select
    const countrySel = getDOM('country');
    for (const key in countryData) {
        countrySel.add(new Option(countryData[key].name_vi, key));
    }
    // Load default country from localStorage if saved
    const savedCountry = safeStorage.getItem('defaultCountry');
    if (savedCountry && countryData[savedCountry]) {
        countrySel.value = savedCountry;
    } else {
        countrySel.value = 'FR';
    }

    // Auto-recalculate on any input change
    const methodSel = getDOM('methodSelect');
    [ySel, mSel, dSel, hSel, minSel, countrySel, methodSel].forEach(input => {
        if (input) input.addEventListener('change', () => {
            if (typeof Solar !== 'undefined') processAll();
        });
    });

    // Render board on load (100ms delay to ensure DOM + Lunar.js are ready)
    setTimeout(() => { if (typeof Solar !== 'undefined') processAll(); }, 100);
});

function toggleLang() {
    currentLang = currentLang === 'vi' ? 'zh' : 'vi';
    const isZH = currentLang === 'zh';
    getDOM('mainBody').classList.toggle('lang-zh', isZH);

    const u = uiDict[currentLang];

    const labelMap = {
        lblChinhNgo: u.chinhNgo, lblLunarInTable: u.lunar, lblViaDay: u.viaDay,
        lblTietkhi:  u.tietkhi,  lblCuc:   u.cuc,    lblTuan: u.tuan,
        lblTP:       u.tp,       lblTS:    u.ts,
        lblMethodAmBan:  u.methodAmBan,
        lblMethodBoPháp: u.methodBoPhap,
        lblMethodTriNhuan: u.methodTriNhuan,
        lblHienThiCoBan: u.coban,
        lblTTNamTxt:   isZH ? '年' : 'Năm',
        lblTTThangTxt: isZH ? '月' : 'Tháng',
        lblTTNgayTxt:  isZH ? '日' : 'Ngày',
        lblTTGioTxt:   isZH ? '时' : 'Giờ',
    };
    for (const [id, text] of Object.entries(labelMap)) getDOM(id).textContent = text;

    const optFmts = [
        ['inYear',      v => (isZH ? '' : 'Năm ')   + v + (isZH ? '年' : '')],
        ['inMonth',     v => (isZH ? '' : 'Tháng ') + v + (isZH ? '月' : '')],
        ['inDay',       v => (isZH ? '' : 'Ngày ')  + v + (isZH ? '日' : '')],
        ['solarHour',   v => v + (isZH ? '时' : 'h')],
        ['solarMinute', v => v + (isZH ? '分' : 'p')],
    ];
    for (const [id, fmt] of optFmts) {
        for (const opt of getDOM(id).options) opt.text = fmt(opt.value);
    }

    const methodSel = getDOM('methodSelect');
    if (methodSel) {
        for (const opt of methodSel.options) {
            if (opt.value === 'amban')    opt.text = u.methodAmBan;
            if (opt.value === 'bophap')   opt.text = u.methodBoPhap;
            if (opt.value === 'trinhuan') opt.text = u.methodTriNhuan;
        }
    }

    const langKey = isZH ? 'name_zh' : 'name_vi';
    for (const opt of getDOM('country').options) {
        const info = countryData[opt.value];
        if (info) opt.text = info[langKey];
    }

    if (getDOM('board').style.display !== 'none') processAll();
    if (typeof updateCountryDisplay === 'function') updateCountryDisplay();
    // Toggle legend language
    const leg = getDOM('tuTruLegend');
    if (leg) {
        const vi = leg.querySelector('.lbl-vi'), zh = leg.querySelector('.lbl-zh');
        if (vi) vi.style.display = isZH ? 'none' : 'inline';
        if (zh) zh.style.display = isZH ? 'inline' : 'none';
    }
}

function setLang(lang) {
    if (currentLang === lang) return;
    // Temporarily set to opposite so toggleLang() flips to the target
    currentLang = lang === 'zh' ? 'vi' : 'zh';
    toggleLang();
    getDOM('langBtn_vi').classList.toggle('method-toggle-active', lang === 'vi');
    getDOM('langBtn_zh').classList.toggle('method-toggle-active', lang === 'zh');
    // Always save lang to localStorage
    safeStorage.setItem('defaultLang', lang);
}

const _translateRE = new RegExp(Object.keys(mapToVi).join('|'), 'g');
function translateStr(text) {
    if (!text) return "";
    return text.replace(_translateRE, m => mapToVi[m] + ' ').trim().replace(/\s+/g, ' ');
}

// Pre-built lookup: can char → {type:'full'|'short', idx}
// Priority: zh.canFull > zh.can > vi.canFull > vi.can (last-write-wins, highest priority added last)
const _canLookup = new Map([
    ...dataBase.vi.can    .map((c, i) => [c, { type: 'short', idx: i }]),
    ...dataBase.vi.canFull.map((c, i) => [c, { type: 'full',  idx: i }]),
    ...dataBase.zh.can    .map((c, i) => [c, { type: 'short', idx: i }]),
    ...dataBase.zh.canFull.map((c, i) => [c, { type: 'full',  idx: i }]),
]);

function getDisplayCan(c) {
    const entry = _canLookup.get(c);
    if (!entry) return c;
    return entry.type === 'full'
        ? dataBase[currentLang].canFull[entry.idx]
        : dataBase[currentLang].can[entry.idx];
}

function getDisplayChi(c) {
    const idx = arrZhiZH.indexOf(c);
    if (idx !== -1) return currentLang === 'zh' ? c : chiVI[idx];
    return translateStr(c); // fallback
}

function updateTuTru(yPillar, mPillar, dPillar, hPillar) {
    const cans = [yPillar[0], mPillar[0], dPillar[0], hPillar[0]];
    const chis = [yPillar[yPillar.length-1], mPillar[mPillar.length-1], dPillar[dPillar.length-1], hPillar[hPillar.length-1]];
    ['ttCanNam','ttCanThang','ttCanNgay','ttCanGio'].forEach((id,i) =>
        getDOM(id).textContent = getDisplayCan(cans[i]));
    ['ttChiNam','ttChiThang','ttChiNgay','ttChiGio'].forEach((id,i) =>
        getDOM(id).textContent = getDisplayChi(chis[i]));
    getDOM('tuTruPanel').style.display = 'table';
    getDOM('tuTruLegend').style.display = 'block';
}

function selectMethod(val) {
    getDOM('methodSelect').value = val;
    getDOM('methodBtn_amban')   .classList.toggle('method-toggle-active', val === 'amban');
    getDOM('methodBtn_bophap')  .classList.toggle('method-toggle-active', val === 'bophap');
    getDOM('methodBtn_trinhuan').classList.toggle('method-toggle-active', val === 'trinhuan');
    const notZH = currentLang !== 'zh';
    const trnPanel = getDOM('trinhuanPanel');
    if (trnPanel) trnPanel.style.display = (val === 'trinhuan' && notZH) ? 'block' : 'none';
    const sbPanel = getDOM('sachboPanel');
    if (sbPanel)  sbPanel.style.display  = (val === 'bophap'   && notZH) ? 'block' : 'none';
    const abPanel = getDOM('ambanPanel');
    if (abPanel)  abPanel.style.display  = (val === 'amban'    && notZH) ? 'block' : 'none';
    // Always save method to localStorage
    safeStorage.setItem('defaultMethod', val);
    if (typeof Solar !== 'undefined') processAll();
}

function applyCobanFilter() {
    const checked = getDOM('chkHienThiCoBan').checked;
    // checked = Đầy đủ (full view, no coban-mode); unchecked = cơ bản (coban-mode)
    getDOM('mainBody').classList.toggle('coban-mode', !checked);
    safeStorage.setItem('chkHienThiCoBan', checked ? '1' : '0');
    if (typeof Solar !== 'undefined') processAll();
}

/* ======================================================================
   CORE LOGIC: KỲ MÔN ĐỘN GIÁP
   ====================================================================== */
function calculateQMDJ(cuc, don, canTuan, canGioRaw, lang) {
    // Chuẩn hóa số Lạc Thư về [1..9] — nhanh hơn while-loop
    const wrapLT = n => ((n - 1) % 9 + 9) % 9 + 1;

    // ── 1. An Địa Bàn + bảng tra ngược canToLt (1 vòng lặp) ──
    const canList = dataBase.vi.can;           // 9 can (không gồm Giáp)
    const diaBan  = {1:[],2:[],3:[],4:[],5:[],6:[],7:[],8:[],9:[]};
    const canToLt = {};
    const isDuong = don === 'duong';
    for (let i = 0; i < canList.length; i++) {
        const lt = wrapLT(isDuong ? cuc + i : cuc - i);
        diaBan[lt].push(canList[i]);
        canToLt[canList[i]] = lt;
    }

    // ── 2. Dải Địa Bàn theo NT1 (cung 2 hợp cung 5) ──
    const dsdb = NT1.map(lt => lt === 2 ? [...diaBan[5], ...diaBan[2]] : [...diaBan[lt]]);

    // ── 3. Tra LT của Tuần Thủ + Can Giờ thực ──
    const canGioTB  = canGioRaw === 'Giáp' ? canTuan : canGioRaw;
    const ltTuan    = canToLt[canTuan];
    const ltGioDB   = canToLt[canGioTB];

    // ── 4. Index khởi đầu (O(1) qua indexOf) ──
    const idxStartDSDB  = NT1.indexOf(ltTuan   === 5 ? 2 : ltTuan);
    const idxStartHT    = NT1.indexOf(ltGioDB  === 5 ? 2 : ltGioDB);
    // idxStartAnCan ≡ idxStartHT luôn đúng (đã kiểm chứng toàn ánh xạ): dsdb[idxStartHT]
    // luôn chứa canGioTB vì idxStartHT được suy trực tiếp từ ltGioDB = canToLt[canGioTB]
    // → bỏ findIndex(dsdb...) thừa, dùng lại idxStartHT.
    // idxTP ≡ idxTS ≡ idxStartDSDB (Trực Phù Tinh và Trực Sử Môn cùng gốc)
    const idxTP = idxStartDSDB;
    const idxTS = idxStartDSDB;

    // ── 5. LT Trực Sử → index khởi đầu Môn ──
    const thuTuGio = dataBase.vi.canFull.indexOf(canGioRaw) + 1;
    let ltTrucSu = wrapLT(isDuong ? ltTuan + thuTuGio - 1 : ltTuan - thuTuGio + 1);
    if (ltTrucSu === 5) ltTrucSu = 2;
    const idxStartMon = NT1.indexOf(ltTrucSu);

    // ── 6. An 7 bảng song song trong 1 vòng lặp (8 cung phi cung 5) ──
    const listThan  = isDuong ? dataBase[lang].thanD : dataBase[lang].thanA;
    const thienCan  = {}, thienTinh = {}, thanTB = {};
    const thienMon  = {}, anCan     = {}, thanDB = {}, amCan = {};
    for (let i = 0; i < 8; i++) {
        const cungHT    = NT1[(idxStartHT   + i) % 8];
        const cungMon   = NT1[(idxStartMon  + i) % 8];
        const cungDB    = NT1[(idxStartDSDB + i) % 8];
        const dsdbSlice = dsdb[(idxStartDSDB + i) % 8];

        thienCan[cungHT]  = dsdbSlice;
        thienTinh[cungHT] = dataBase[lang].tinh[(idxTP + i) % 8];
        thanTB[cungHT]    = listThan[i];

        thienMon[cungMon] = dataBase[lang].mon[(idxTS + i) % 8];
        anCan[cungMon]    = dsdb[(idxStartHT + i) % 8];
        amCan[cungMon]    = dsdbSlice;
        thanDB[cungDB]    = listThan[i];
    }

    // ── 7. Phụ Tinh ở cung 4 (Tốn): an Ẩn Can theo đường lộ ──
    if (thienTinh[4] === dataBase[lang].tinh[0] && thienMon[4] === dataBase[lang].mon[0]) {
        const baseCan = dataBase[lang].can;
        const idxGio  = dataBase.vi.can.indexOf(canGioTB);
        if (idxGio !== -1) {
            const ltxPath = [5, 6, 7, 8, 9, 1, 2, 3, 4];
            for (let i = 0; i < 9; i++) {
                const idx = isDuong ? (idxGio + i) % 9 : (idxGio - i + 9) % 9;
                anCan[ltxPath[i]] = [baseCan[idx]];
            }
            // Cung Khôn (2) nhận thêm ẩn can cung 5 làm ký can
            anCan[2] = [...anCan[5], ...anCan[2]];
        }
    }

    return { diaBan, thienCan, thienTinh, thanTB, thienMon, anCan, thanDB, idxTP, idxTS, amCan };
}

function renderCanPair(canArr, highlightCan, posMain) {
    if (!canArr || canArr.length === 0)
        return `<div class="sub-cell ${posMain}"></div>`;

    const hasKy   = canArr.length > 1;
    const kycan   = hasKy ? canArr[0] : '';
    const mainArr = hasKy ? canArr.slice(1) : canArr;

    // Hàm helper render 1 can với highlight
    const renderOne = (c, extraCls = '') => {
        const d = getDisplayCan(c);
        return (highlightCan && c === highlightCan)
            ? `<span class="can-gio-box${extraCls}">${d}</span>`
            : d;
    };

    const mainHtml = mainArr.map(c => renderOne(c)).join('');

    if (!hasKy)
        return `<div class="sub-cell ${posMain}">${mainHtml}</div>`;

    const kyHtml = renderOne(kycan, ' box-small');

    // Tiếng Việt: ký can thẳng dưới can chính (flex-column)
    if (currentLang === 'vi') {
        return `<div class="sub-cell ${posMain}" style="flex-direction:column;align-items:center;justify-content:center;">` +
               `<span style="font-size:.9em;line-height:1.2;">${mainHtml}</span>` +
               `<span style="font-size:.8em;line-height:1.2;">${kyHtml}</span>` +
               `</div>`;
    }
    // Tiếng Trung: layout ngang, ký can xiên xuống
    return `<div class="sub-cell ${posMain} can-pair-ky">` +
           `<span class="can-pair-main"><span class="can-pair-chinh">${mainHtml}</span><span class="can-pair-sub">${kyHtml}</span></span>` +
           `</div>`;
}

/* Lookup: "ThiênCan|ĐịaCan" (tiếng Việt) → tên cách cục */
const cachCucMap = {
    "Ất|Ất": { vi: "Nhật kỳ phục ngâm", zh: "日奇伏吟" },
    "Ất|Bính": { vi: "Kỳ nghi thuận toại", zh: "奇仪顺遂" },
    "Ất|Đinh": { vi: "Kỳ nghi tương tá", zh: "奇仪相佐" },
    "Ất|Mậu": { vi: "Âm hại dương môn", zh: "阴害阳门" },
    "Ất|Kỷ": { vi: "Nhật kỳ nhập mộ", zh: "日奇入墓" },
    "Ất|Canh": { vi: "Nhật kỳ bị hình", zh: "日奇被刑" },
    "Ất|Tân": { vi: "Thanh Long đào tẩu", zh: "青龙逃走" },
    "Ất|Nhâm": { vi: "Nhật kỳ nhập thiên lao", zh: "日奇入天牢" },
    "Ất|Quý": { vi: "Nhật kỳ nhập địa võng", zh: "日奇入地网" },
    "Bính|Ất": { vi: "Nhật nguyệt bình hành", zh: "日月并行" },
    "Bính|Bính": { vi: "Nguyệt kỳ bội sư (Bội cách)", zh: "月奇悖师" },
    "Bính|Đinh": { vi: "Tinh kỳ Chu Tước", zh: "星奇朱雀" },
    "Bính|Mậu": { vi: "Phi điểu điệt huyệt", zh: "飞鸟跌穴" },
    "Bính|Kỷ": { vi: "Hỏa bội nhập hình", zh: "火悖入刑" },
    "Bính|Canh": { vi: "Huỳnh nhập Thái Bạch (Tặc tất khứ)", zh: "荧入太白" },
    "Bính|Tân": { vi: "Nguyệt kỳ tương hợp", zh: "月奇相合" },
    "Bính|Nhâm": { vi: "Hỏa nhập thiên lao", zh: "火入天牢" },
    "Bính|Quý": { vi: "Nguyệt kỳ địa võng", zh: "月奇地网" },
    "Đinh|Ất": { vi: "Ngọc nữ kỳ sinh", zh: "玉女奇生" },
    "Đinh|Bính": { vi: "Tinh tùy nguyệt chuyển", zh: "星随月转" },
    "Đinh|Đinh": { vi: "Tinh kỳ phục ngâm", zh: "星奇伏吟" },
    "Đinh|Mậu": { vi: "Thanh Long chuyển quang", zh: "青龙转光" },
    "Đinh|Kỷ": { vi: "Hỏa nhập Câu Trận", zh: "火入勾陈" },
    "Đinh|Canh": { vi: "Tinh kỳ thụ trở", zh: "星奇受阻" },
    "Đinh|Tân": { vi: "Chu Tước nhập ngục", zh: "朱雀入狱" },
    "Đinh|Nhâm": { vi: "Kỳ nghi tương hợp", zh: "奇仪相合" },
    "Đinh|Quý": { vi: "Chu Tước đầu giang", zh: "朱雀投江" },
    "Mậu|Ất": { vi: "Thanh Long hòa hội", zh: "青龙和会" },
    "Mậu|Bính": { vi: "Thanh Long phản thủ", zh: "青龙返首" },
    "Mậu|Đinh": { vi: "Thanh Long diệu minh", zh: "青龙耀明" },
    "Mậu|Mậu": { vi: "Phục Ngâm", zh: "伏吟" },
    "Mậu|Kỷ": { vi: "Quý nhân nhập ngục", zh: "贵人入狱" },
    "Mậu|Canh": { vi: "Trực Phù phi cung", zh: "直符飞宫" },
    "Mậu|Tân": { vi: "Thanh Long chiết túc", zh: "青龙折足" },
    "Mậu|Nhâm": { vi: "Thanh Long nhập thiên lao", zh: "青龙入天牢" },
    "Mậu|Quý": { vi: "Thanh Long hoa cái", zh: "青龙华盖" },
    "Kỷ|Ất": { vi: "Địa hộ phùng tinh", zh: "地户逢星" },
    "Kỷ|Bính": { vi: "Hỏa bội địa hộ", zh: "火悖地户" },
    "Kỷ|Đinh": { vi: "Chu Tước nhập mộ", zh: "朱雀入墓" },
    "Kỷ|Mậu": { vi: "Khuyển ngộ Thanh Long", zh: "犬遇青龙" },
    "Kỷ|Kỷ": { vi: "Địa hộ phùng quỷ", zh: "地户逢鬼" },
    "Kỷ|Canh": { vi: "Hình cách phản danh", zh: "刑格反名" },
    "Kỷ|Tân": { vi: "Du hồn nhập mộ", zh: "游魂入墓" },
    "Kỷ|Nhâm": { vi: "Địa võng cao trương", zh: "地网高张" },
    "Kỷ|Quý": { vi: "Địa hình Huyền Vũ", zh: "地刑玄武" },
    "Canh|Ất": { vi: "Thái Bạch phùng tinh", zh: "太白逢星" },
    "Canh|Bính": { vi: "Thái Bạch nhập huỳnh (Tặc tất lai)", zh: "太白入荧" },
    "Canh|Đinh": { vi: "Đình đình chi cách", zh: "荧荧之格" },
    "Canh|Mậu": { vi: "Thiên Ất phục cung", zh: "天乙伏宫" },
    "Canh|Kỷ": { vi: "Quan phủ hình cách", zh: "官府刑格" },
    "Canh|Canh": { vi: "Thái Bạch đồng cung", zh: "太白同宫" },
    "Canh|Tân": { vi: "Bạch Hổ can cách", zh: "白虎干格" },
    "Canh|Nhâm": { vi: "Di đãng cách", zh: "荡漾格" },
    "Canh|Quý": { vi: "Đại cách", zh: "大格" },
    "Tân|Ất": { vi: "Bạch Hổ xương cuồng", zh: "白虎猖狂" },
    "Tân|Bính": { vi: "Can hợp bội sư", zh: "干合悖师" },
    "Tân|Đinh": { vi: "Ngục thần đắc kỳ", zh: "狱神得奇" },
    "Tân|Mậu": { vi: "Khốn Long bị thương", zh: "困龙被伤" },
    "Tân|Kỷ": { vi: "Nhập ngục tự hình", zh: "入狱自刑" },
    "Tân|Canh": { vi: "Bạch Hổ xuất lực", zh: "白虎出力" },
    "Tân|Tân": { vi: "Phục ngâm thiên đình", zh: "伏吟天庭" },
    "Tân|Nhâm": { vi: "Hung xà nhập ngục", zh: "凶蛇入狱" },
    "Tân|Quý": { vi: "Thiên Lao hoa cái", zh: "天牢华盖" },
    "Nhâm|Ất": { vi: "Tiểu xà đắc thế", zh: "小蛇得势" },
    "Nhâm|Bính": { vi: "Thủy xà nhập hỏa", zh: "水蛇入火" },
    "Nhâm|Đinh": { vi: "Can hợp xà hình", zh: "干合蛇刑" },
    "Nhâm|Mậu": { vi: "Tiểu xà hóa Long", zh: "小蛇化龙" },
    "Nhâm|Kỷ": { vi: "Phản ngâm xà hình", zh: "反吟蛇刑" },
    "Nhâm|Canh": { vi: "Thái Bạch cầm xà", zh: "太白擒蛇" },
    "Nhâm|Tân": { vi: "Đằng Xà tương triền", zh: "腾蛇相缠" },
    "Nhâm|Nhâm": { vi: "Thiên ngục tự hình", zh: "天狱自刑" },
    "Nhâm|Quý": { vi: "Ấu nữ gian dâm", zh: "幼女奸淫" },
    "Quý|Ất": { vi: "Hoa cái phùng tinh", zh: "华盖逢星" },
    "Quý|Bính": { vi: "Hoa cái bội sư", zh: "华盖悖师" },
    "Quý|Đinh": { vi: "Đằng Xà yêu kiều", zh: "腾蛇妖娇" },
    "Quý|Mậu": { vi: "Thiên Ất hội hợp", zh: "天乙会合" },
    "Quý|Kỷ": { vi: "Hoa cái địa hộ", zh: "华盖地户" },
    "Quý|Canh": { vi: "Thái Bạch nhập võng", zh: "太白入网" },
    "Quý|Tân": { vi: "Võng cái thiên lao", zh: "网盖天牢" },
    "Quý|Nhâm": { vi: "Phục kiến Đằng Xà", zh: "伏见腾蛇" },
    "Quý|Quý": { vi: "Thiên võng tứ trương", zh: "天网四张" },
};

/* Renders the "Thiên / Địa" pairs for the back side of each flip cell.
   Array format: [kyCanIfAny, mainCan1, mainCan2, ...]
   All combinations of (thiênBàn cans) × (địaBàn cans) are generated. */
function renderFlipContent(thienCanArr, diaCanArr) {
    // Both arrays are plain can lists (no ký-can sentinel); just dedup
    const getList = arr => {
        if (!arr || arr.length === 0) return [];
        return [...new Set(arr)];
    };
    const tList = getList(thienCanArr);
    const dList = getList(diaCanArr);
    if (!tList.length || !dList.length) return '';
    const toVI = c => {
        const iF = dataBase.zh.canFull.indexOf(c);
        if (iF !== -1) return dataBase.vi.canFull[iF];
        const iS = dataBase.zh.can.indexOf(c);
        if (iS !== -1) return dataBase.vi.can[iS];
        return c; // already VI or unknown
    };
    // Collect unique pairs first so we know total count before rendering
    const pairs = [];
    const seen = new Set();
    for (const tc of tList) {
        for (const dc of dList) {
            const key = `${tc}|${dc}`;
            if (seen.has(key)) continue;
            seen.add(key);
            pairs.push([tc, dc]);
        }
    }
    const totalRows = pairs.length;
    const rows = pairs.map(([tc, dc]) => {
        const viKey = `${toVI(tc)}|${toVI(dc)}`;
        const cachEntry = cachCucMap[viKey];
        const cach = cachEntry ? (currentLang === 'zh' ? cachEntry.zh : cachEntry.vi) : '';
        let cachMain = cach, cachSub = '';
        if (cach && currentLang !== 'zh') {
            const pi = cach.indexOf(' (');
            if (pi !== -1) {
                cachMain = cach.slice(0, pi);
                // Show parenthetical subtitle only when space allows (< 4 pairs)
                if (totalRows < 4) cachSub = cach.slice(pi);
            }
        }
        return (
            `<div class="flip-pair-row">` +
            `<span class="flip-can-pair">${getDisplayCan(tc)} / ${getDisplayCan(dc)}</span>` +
            (cach
                ? `<span class="flip-cach-cuc">${cachMain}` +
                  (cachSub ? `<span class="flip-cach-sub">${cachSub}</span>` : '') +
                  `</span>`
                : '') +
            `</div>`
        );
    });
    return `<div class="flip-cell-inner" data-rows="${totalRows}">${rows.join('')}</div>`;
}

/* getEquationOfTime() đã bỏ: phương trình thời gian nay chỉ còn MỘT bản, nằm
   trong astro.js, gọi qua Ephem. Hai bản song song từng lệch nhau tới 8,8
   giây — mà cả hai đều dùng để tính Chính Ngọ, thứ định ranh giới Chính Tý,
   tức định mùng 1. */

/* ======================================================================
   HÀM PHỤ MODULE-LEVEL (không tái tạo mỗi lần gọi processAll)
   ====================================================================== */

/**
 * Quy đổi giờ Bắc Kinh (UTC+8) sang múi giờ địa phương theo offset CỐ ĐỊNH
 * `targetTz` (số giờ lệch UTC), trả chuỗi "DD-MM-YYYY HH:MM".
 *
 * CHÚ Ý: dùng offset cố định — không tự tra DST theo từng thời điểm. Chỉ
 * dùng cho fallback khi không có `tzId` (IANA) để gọi
 * formatUTC8SolarToLocal()/_formatJdUTC8ToLocal() (hàm đó mới là cách tính
 * đúng, tự xử lý DST theo thời điểm của chính mốc đang quy đổi).
 */
function convertBeijingToLocal(solarObj, targetTz) {
    const jdLocal = solarObj.getJulianDay() + (targetTz - 8) / 24;
    const local   = Solar.fromJulianDay(jdLocal);
    return `${pad(local.getDay())}-${pad(local.getMonth())}-${local.getYear()} ${pad(local.getHour())}:${pad(local.getMinute())}`;
}

/** Chuyển Solar object thành số YYYYMMDDHHMM để so sánh thời điểm giao tiết */
function solarToCompareNum(s) {
    return parseInt(`${s.getYear()}${pad(s.getMonth())}${pad(s.getDay())}${pad(s.getHour())}${pad(s.getMinute())}`);
}

/* ======================================================================
   SÓC CHÍNH XÁC ĐẾN PHÚT (Precise New Moon time)
   ----------------------------------------------------------------------
   Lunar.fromYmd(...).getSolar() chỉ trả về ngày-tháng-năm của Sóc đã được
   LÀM TRÒN về 00:00 (Math.floor(t + 0.5) trong calcShuo) — KHÔNG có
   thông tin giờ/phút thực của thời điểm trăng non (hợp sóc).
   Hàm dưới đây tính lại giá trị "t" thô (trước khi làm tròn) bằng cùng
   thuật toán shuoHigh/msaLonT2 của thư viện, dựa trên Julian Day đã làm
   tròn làm điểm neo để xác định đúng kỳ trăng (k), sau đó trả về thời
   điểm Sóc chính xác (theo giờ Bắc Kinh UTC+8 — mốc tham chiếu cố định
   của lịch âm, giống cách tính ngày/tháng âm lịch ở processAll()).
   ====================================================================== */

/**
 * Tính thời điểm Sóc (trăng non) chính xác đến phút, theo giờ UTC+8.
 * @param {object} roundedSolar  Solar object từ Lunar.fromYmd(y, m, 1).getSolar()
 *                                (ngày đã làm tròn, giờ luôn = 00:00)
 * @returns {object} Solar object chính xác đến phút (giờ UTC+8)
 */
function getPreciseSocSolarUTC8(roundedSolar) {
    return Ephem.socSolar(roundedSolar, 8);
}

/**
 * Quy đổi thời điểm Sóc (UTC+8, chính xác đến phút) sang giờ địa phương,
 * trả chuỗi "DD-MM-YYYY HH:MM".
 * @param {object} roundedSolar  Solar object từ Lunar.fromYmd(y, m, 1).getSolar()
/**
 * Quy đổi Julian Day (mốc UTC+8 / giờ Bắc Kinh) sang offset UTC (giờ) thực
 * tế của múi giờ `tzId` TẠI CHÍNH THỜI ĐIỂM ĐÓ — dùng Intl.DateTimeFormat
 * nên tự động xử lý đúng DST (giờ mùa hè/đông).
 * @param {number} jdUTC8  Julian Day theo mốc UTC+8
 * @param {string} tzId    IANA timezone id (vd 'Europe/Paris')
 */
function _tzOffsetAtJdUTC8(jdUTC8, tzId) {
    const jdUTC  = jdUTC8 - 8 / 24;
    const unixMs = (jdUTC - 2440587.5) * 86400000;
    return getTimezoneOffset(tzId, new Date(unixMs));
}

/**
 * Quy đổi một Julian Day chính xác (mốc UTC+8) sang giờ địa phương `tzId`,
 * trả chuỗi "DD-MM-YYYY HH:MM".
 *
 * jdUTC8 biểu diễn một THỜI ĐIỂM UTC CỐ ĐỊNH (jdUTC = jdUTC8 - 8/24). Offset
 * DST của tzId tại thời điểm UTC đó được lấy bằng Intl.DateTimeFormat
 * (_tzOffsetAtJdUTC8) — tự động đúng cho mọi trường hợp, kể cả khi
 * Sóc/Tiết khí rơi đúng vào giờ chuyển DST (mùa hè/đông) của tzId.
 *
 * (Trước đây offset được tính từ NGÀY của socSolar/jqSolar theo mốc UTC+8 —
 * có thể sai 1h nếu offset DST tại "ngày UTC+8" khác với offset DST tại
 * thời điểm UTC thực mà jdUTC8 biểu diễn, lệch nhau ~6-9h múi giờ.)
 * @param {number} jdUTC8  Julian Day theo mốc UTC+8
 * @param {string} tzId    IANA timezone id
 */
function _formatJdUTC8ToLocal(jdUTC8, tzId) {
    const tz = _tzOffsetAtJdUTC8(jdUTC8, tzId);
    const jdLocal = jdUTC8 + (tz - 8) / 24;
    const local = Solar.fromJulianDay(jdLocal);
    return `${pad(local.getDay())}-${pad(local.getMonth())}-${local.getYear()} ${pad(local.getHour())}:${pad(local.getMinute())}`;
}

/**
 * Quy đổi thời điểm Sóc (UTC+8, chính xác đến phút) sang giờ địa phương,
 * trả chuỗi "DD-MM-YYYY HH:MM".
 * @param {object} roundedSolar  Solar object từ Lunar.fromYmd(y, m, 1).getSolar()
 * @param {string} tzId          IANA timezone id (vd 'Europe/Paris')
 */
function formatPreciseSocLocal(roundedSolar, tzId) {
    const preciseUTC8 = getPreciseSocSolarUTC8(roundedSolar);
    return _formatJdUTC8ToLocal(preciseUTC8.getJulianDay(), tzId);
}

/**
 * Quy đổi một Solar object đã CHÍNH XÁC ĐẾN PHÚT theo giờ Bắc Kinh (UTC+8)
 * sang giờ địa phương, trả chuỗi "DD-MM-YYYY HH:MM".
 * Dùng cho jqSolar (thời điểm giao Tiết Khí) — vốn đã có độ chính xác
 * phút/giây (qiAccurate2), không cần bước getPreciseSocSolarUTC8() như Sóc
 * (Lunar.fromYmd().getSolar() bị làm tròn về 00:00).
 * @param {object} solarUTC8  Solar object ở mốc UTC+8 (giờ Bắc Kinh)
 * @param {string} tzId       IANA timezone id
 */
function formatUTC8SolarToLocal(solarUTC8, tzId) {
    return _formatJdUTC8ToLocal(solarUTC8.getJulianDay(), tzId);
}


/* ══════════════════════════════════════════════════════════════════════
   RANH GIỚI NGÀY ÂM LỊCH: CHÍNH TÝ THIÊN VĂN
   ══════════════════════════════════════════════════════════════════════

   Mùng 1 là ngày CHỨA điểm Sóc. Ngày ở đây đếm từ **Chính Tý tới Chính Tý**
   — tức nửa đêm MẶT TRỜI THẬT (Chính Ngọ − 12h), chứ không phải 00:00 đồng hồ.

   Đây là chuyện QUY ƯỚC, không phải đúng/sai. Lịch pháp Trung–Việt định ngày
   từ nửa đêm đồng hồ tới nửa đêm đồng hồ tại KINH TUYẾN QUY CHIẾU, và mọi cuốn
   lịch in đều theo luật ấy. Ứng dụng này chọn nửa đêm THẬT tại nơi người dùng
   đứng, cùng hệ với Chính Ngọ mà nó vẫn hiển thị.

   Không lấy ranh giới đầu giờ Tý (Chính Ngọ − 13h): đó là quy ước của mệnh lý
   cho TRỤ NGÀY, và bản thân nó còn hai phái (早子時 / 夜子時). Nửa đêm thật thì
   chỉ có một.

   Chính Ngọ lệch khỏi 12:00 đồng hồ vì kinh độ, phương trình thời gian và giờ
   mùa hè cộng lại, nên Chính Tý cũng lệch khỏi 00:00 đúng chừng ấy:

     Hà Nội        Chính Ngọ 12:01 → Chính Tý 00:01  (lệch 1 phút)
     Paris (CEST)  Chính Ngọ 13:55 → Chính Tý 01:55  (lệch 115 phút)

   Nên ở Việt Nam gần như không đổi gì, còn nơi lệch xa kinh tuyến múi giờ của
   mình thì chừng 8% số tháng đổi mùng 1.
   ══════════════════════════════════════════════════════════════════════ */

/** Số phút kể từ 00:00 mà Chính Tý (nửa đêm mặt trời thật) rơi vào. */
function zi_midnightMinutes(y, m, d, lon, tz) {
    return Ephem.solarMidnightMinutes(y, m, d, lon, tz);   // có thể âm
}

/**
 * Ngày (đếm từ Chính Tý tới Chính Tý) chứa một thời điểm giờ địa phương.
 * @returns {{y:number,m:number,d:number}}
 */
function zi_dayOf(local, lon, tzId) {
    const y = local.getYear(), m = local.getMonth(), d = local.getDay();
    const tz = getTimezoneOffset(tzId, new Date(y, m - 1, d, 12));
    const t = local.getHour() * 60 + local.getMinute();
    const shift = Math.floor((t - zi_midnightMinutes(y, m, d, lon, tz)) / 1440);
    if (shift === 0) return { y: y, m: m, d: d };
    const dt = new Date(y, m - 1, d + shift);
    return { y: dt.getFullYear(), m: dt.getMonth() + 1, d: dt.getDate() };
}

/** Đưa một Solar ở mốc UTC+8 về giờ địa phương (Solar, chính xác tới phút). */
function _localSolarFromJdUTC8(jdUTC8, tzId) {
    const tz = _tzOffsetAtJdUTC8(jdUTC8, tzId);
    return Solar.fromJulianDay(jdUTC8 + (tz - 8) / 24);
}

/**
 * Mùng 1 của một tháng âm: ngày (Chính Tý → Chính Tý) chứa điểm Sóc.
 * @param {object} mo  LunarMonth từ LunarYear.getMonths()
 */
function zi_mung1(mo, lon, tzId) {
    return zi_mung1FromJd(mo.getFirstJulianDay(), lon, tzId);
}

/** Như zi_mung1 nhưng nhận thẳng số ngày Julius của mùng 1 đã làm tròn. */
function zi_mung1FromJd(firstJd, lon, tzId) {
    const socUTC8 = getPreciseSocSolarUTC8(Solar.fromJulianDay(firstJd));
    const local = _localSolarFromJdUTC8(socUTC8.getJulianDay(), tzId);
    return zi_dayOf(local, lon, tzId);
}

/** Số ngày Julius của một ngày dương lịch (Fliegel–Van Flandern). */
function zi_jdn(y, m, d) {
    const a = Math.floor((14 - m) / 12), yy = y + 4800 - a, mm = m + 12 * a - 3;
    return d + Math.floor((153 * mm + 2) / 5) + 365 * yy
        + Math.floor(yy / 4) - Math.floor(yy / 100) + Math.floor(yy / 400) - 32045;
}

/**
 * Kinh tuyến quy chiếu cho NHÃN tháng: UTC+7 cho lịch ta, UTC+8 cho lịch Tàu.
 * Chính chỗ khác nhau này làm Tết ta và Tết Tàu thỉnh thoảng lệch một ngày.
 */
function zi_labelBasis() {
    return (typeof currentLang !== 'undefined' && currentLang === 'zh') ? 8 : 7;
}

/**
 * Danh sách tháng âm quanh một năm dương: NHÃN lấy ở mốc quy chiếu, MỐC BẮT
 * ĐẦU neo theo Chính Tý địa phương. Trả mảng {jdn, month, year, leap} tăng dần.
 *
 * Vì sao tách đôi như vậy:
 *
 *   • "Tháng này là tháng mấy, tháng nào nhuận" là QUY ƯỚC LỊCH, không phải sự
 *     kiện thiên văn tại chỗ người dùng đứng. Nó do luật "tháng không có trung
 *     khí là tháng nhuận" quyết, và luật ấy được định tại kinh tuyến quy chiếu.
 *     Lấy nhãn ở đó thì số tháng khớp lịch in, và xác định — không trôi.
 *
 *   • "Mùng 1 rơi vào ngày dương nào" thì mới là chuyện địa phương: ngày chứa
 *     điểm Sóc, đếm từ Chính Tý.
 *
 * Trước đây hỏi lunar.js ở ngay mốc địa phương, tức để chính luật trung khí bị
 * đánh giá trên lưới nửa đêm ĐỒNG HỒ ở một offset nguyên giờ. Mà Chính Tý lại
 * xê dịch tới ~30 phút trong năm theo phương trình thời gian, nên một offset cố
 * định không diễn tả nổi nó: đo ra chừng 4–8 tháng mỗi thế kỷ đổi nhãn chỉ vì
 * mốc lệch 15–30 phút. Nay nhãn không còn phụ thuộc chuyện đó nữa.
 *
 * Ghép nhãn với mốc bắt đầu là an toàn: dãy tuần trăng giống hệt nhau ở mọi
 * mốc — đã kiểm 1900–2100, mọi mốc từ UTC−8 tới UTC+12 đều ra ĐÚNG 2486 tháng,
 * mốc bắt đầu lệch tối đa 1 ngày, không cặp nào lệch quá.
 */
const _ziMonthCache = new Map();
function zi_months(gregYear, lon, tzId, tz) {
    const basis = zi_labelBasis();
    const key = gregYear + '|' + tzId + '|' + lon + '|' + basis;
    if (_ziMonthCache.has(key)) return _ziMonthCache.get(key);
    const out = [];
    for (let ly = gregYear - 1; ly <= gregYear + 1; ly++) {
        // Ephem nhớ theo (năm, mốc) nên ba năm liền kề không còn đá văng nhau
        // như khi gọi thẳng LunarYear.fromYear — bộ nhớ đệm của nó chỉ giữ MỘT.
        for (const mo of Ephem.monthsAtBasis(ly, basis)) {
            const g = zi_mung1FromJd(mo.jd, lon, tzId);   // mốc bắt đầu: địa phương
            out.push({
                jdn: zi_jdn(g.y, g.m, g.d),
                month: mo.month,                          // nhãn: mốc quy chiếu
                year: ly, leap: mo.leap,
            });
        }
    }
    out.sort((a, b) => a.jdn - b.jdn);
    if (_ziMonthCache.size > 24) _ziMonthCache.clear();
    _ziMonthCache.set(key, out);
    return out;
}

/**
 * Ngày âm lịch của một ngày dương, theo ranh giới Chính Tý.
 * @returns {{day:number,month:number,year:number,leap:boolean}|null}
 */
function zi_lunarOf(y, m, d, lon, tzId, tz) {
    const list = zi_months(y, lon, tzId, tz);
    const j = zi_jdn(y, m, d);
    let at = -1;
    for (let i = 0; i < list.length; i++) {
        if (list[i].jdn <= j) at = i; else break;
    }
    if (at < 0) return null;
    return {
        day: j - list[at].jdn + 1, month: list[at].month,
        year: list[at].year, leap: list[at].leap,
    };
}
window.zi_lunarOf = zi_lunarOf;
window.zi_months = zi_months;
window.zi_mung1 = zi_mung1;

/** Định dạng chuỗi Không Vong để hiển thị theo ngôn ngữ */
function formatKVdisp(kv, lang) {
    if (!kv || kv === '-') return '-';
    if (lang === 'zh') return kv.split(' ').map(chi => chiMapping[chi] || chi).join('');
    return kv;
}

/* ======================================================================
   TRÍ NHUẬN & SÁCH BỔ PHÁP
   ====================================================================== */

// ── Hằng số dùng chung cho cả Trí Nhuận và Sách Bổ ──
// (TK_VI, TK_ZH đã khai báo ở khu vực DICTIONARIES & MAPPING phía trên)

/** Map tiếng Việt → chữ Hán */
const TK_VI_TO_ZH = Object.fromEntries(TK_VI.map((v, i) => [v, TK_ZH[i]]));

/** Số cục [Thượng, Trung, Hạ nguyên] theo tiết khí (dùng chung TN + SB) */
const TK_SO_CUC = {
    '冬至':[1,7,4],'小寒':[2,8,5],'大寒':[3,9,6],
    '立春':[8,5,2],'雨水':[9,6,3],'惊蛰':[1,7,4],
    '春分':[3,9,6],'清明':[4,1,7],'谷雨':[5,2,8],
    '立夏':[4,1,7],'小满':[5,2,8],'芒种':[6,3,9],
    '夏至':[9,3,6],'小暑':[8,2,5],'大暑':[7,1,4],
    '立秋':[2,5,8],'处暑':[1,4,7],'白露':[9,3,6],
    '秋分':[7,1,4],'寒露':[6,9,3],'霜降':[5,8,2],
    '立冬':[6,9,3],'小雪':[5,8,2],'大雪':[4,7,1]
};

/** Dương Độn: Đông Chí → Mang Chủng (index 0–11); Âm Độn: Hạ Chí → Đại Tuyết (12–23) */
const TK_DUONG_DON = new Set(TK_ZH.slice(0, 12));

const TN_PTTN_NAMES  = ['Giáp Tý','Kỷ Mão','Giáp Ngọ','Kỷ Dậu'];
const TN_THUONG_CHI  = new Set(['子','午','卯','酉']);
const TN_TRUNG_CHI   = new Set(['寅','申','巳','亥']);
const TN_NGUYEN_VI   = ['Thượng nguyên','Trung nguyên','Hạ nguyên'];

// ── Tiện ích Julian Day ──

function tn_dateToJD(y, m, d) {
    const a = Math.floor((14 - m) / 12), Y2 = y + 4800 - a, M = m + 12 * a - 3;
    return d + Math.floor((153 * M + 2) / 5) + 365 * Y2
        + Math.floor(Y2 / 4) - Math.floor(Y2 / 100) + Math.floor(Y2 / 400) - 32045;
}

function tn_jdToYMD(jd) {
    const z = Math.floor(jd);
    let a = z;
    if (z >= 2299161) { const al = Math.floor((z - 1867216.25) / 36524.25); a = z + 1 + al - Math.floor(al / 4); }
    const b = a + 1524, c = Math.floor((b - 122.1) / 365.25),
          d2 = Math.floor(365.25 * c), e = Math.floor((b - d2) / 30.6001);
    const day = b - d2 - Math.floor(30.6001 * e), month = e < 14 ? e - 1 : e - 13,
          year = month > 2 ? c - 4716 : c - 4715;
    return { year, month, day };
}

/** Số thứ tự can-chi trong lục thập hoa giáp (0–59) */
function tn_sex60(c, ch) {
    for (let n = 0; n < 60; n++) if (n % 10 === c && n % 12 === ch) return n;
    return -1;
}

// ── Tìm tiết khí (dùng chung TN + SB) ──

/**
 * Chuyển Solar object (tiết khí) thành local JD fractional.
 * OPT-3-FULL: với ShouXingUtil.setTzOffsetHours(tz) đã được set trong
 * processAll(), Solar object `s` (từ getNextJieQi/getPrevJieQi, dựa trên
 * LunarYear.getJieQiJulianDays()) đã là giờ ĐỊA PHƯƠNG — không còn UTC+8/CST.
 * Vì vậy không cần dịch chuyển `-8h + targetTz` nữa, chỉ đọc trực tiếp.
 */
function jqSolarToLocalJDFrac(s, tzId, defaultTz) {
    return {
        year: s.getYear(), month: s.getMonth(), day: s.getDay(),
        hour: s.getHour(), minute: s.getMinute(),
        jdFrac: tn_dateToJD(s.getYear(), s.getMonth(), s.getDay())
                + s.getHour() / 24 + s.getMinute() / 1440
    };
}

/**
 * Tìm tiết khí theo tên chữ Hán, bắt đầu tìm từ startY/startM.
 * Thay thế cả tn_findTerm (TN) và sb_getDongChiJD (SB) — cùng logic, cùng timezone handling.
 * @returns {{ year, month, day, hour, minute, jdFrac } | null}
 */
function findJieQi(zhName, startY, startM, tzId, defaultTz) {
    let ref = Solar.fromYmd(startY, startM || 1, 1);
    for (let i = 0; i < 30; i++) {
        const nxt = ref.getLunar().getNextJieQi(true);
        if (nxt.getName() === zhName) {
            return jqSolarToLocalJDFrac(nxt.getSolar(), tzId, defaultTz);
        }
        ref = nxt.getSolar().next(1);
    }
    return null;
}

// ── Trí Nhuận Pháp ──

/**
 * Giờ Tý thiên văn tại kinh độ lonDeg, DST-aware qua tzId.
 * @returns {{ calJD, dispJD, hour, minute }}
 */
function tn_getTyTimestamp(ymd, lonDeg, tzH, tzId) {
    const calJD = tn_dateToJD(ymd.year, ymd.month, ymd.day);
    // DST-aware: tính lại tzH chính xác cho ngày này
    const effectiveTzH = tzId
        ? getTimezoneOffset(tzId, new Date(ymd.year, ymd.month - 1, ymd.day, 12, 0, 0))
        : tzH;

    const T    = (calJD - 2451545.0) / 36525.0;
    const L0   = (280.46646 + 36000.76983 * T) % 360;
    const M    = ((357.52911 + 35999.05029 * T) % 360) * Math.PI / 180;
    const e    = 0.016708634 - 0.000042037 * T;
    const C    = (1.914602 - 0.004817 * T) * Math.sin(M) + 0.019993 * Math.sin(2 * M) + 0.000289 * Math.sin(3 * M);
    const omega = (125.04 - 1934.136 * T) * Math.PI / 180;
    const eps  = ((23.439291111 - 0.013004167 * T) + 0.00256 * Math.cos(omega)) * Math.PI / 180;
    const y2   = Math.tan(eps / 2) ** 2;
    const L0r  = (L0 + C) * Math.PI / 180;
    const EoT  = (y2 * Math.sin(2 * L0r) - 2 * e * Math.sin(M)
               + 4 * e * y2 * Math.sin(M) * Math.cos(2 * L0r)
               - 0.5 * y2 * y2 * Math.sin(4 * L0r)
               - 1.25 * e * e * Math.sin(2 * M)) * (1440 / (2 * Math.PI));

    // Giờ Tý = Chính Ngọ − 13h
    let tyMin = 720 - (lonDeg - effectiveTzH * 15) * 4 - EoT - 780;
    let dayOffset = 0;
    while (tyMin < 0)    { tyMin += 1440; dayOffset--; }
    while (tyMin >= 1440){ tyMin -= 1440; dayOffset++; }

    let h = Math.floor(tyMin / 60), mn = Math.round(tyMin % 60);
    if (mn === 60) { mn = 0; h++; }
    if (h  === 24) { h  = 0; }
    return { calJD, dispJD: calJD + dayOffset, hour: h, minute: mn };
}

/** Can chi ngày theo lunar.js (UTC+8 canonical) */
function tn_getDayGZ(dt) {
    const l = Solar.fromYmdHms(dt.year, dt.month, dt.day, dt.hour || 0, dt.minute || 0, 0).getLunar();
    return { can: l.getDayGanIndexExact(), chi: l.getDayZhiIndexExact() };
}

/** Tìm ngày PTTN trước nearDt có can-chi = pttnIdx60 */
function tn_findPTTNDay(pttnIdx60, nearDt, lonDeg, tzH, tzId) {
    const base = tn_dateToJD(nearDt.year, nearDt.month, nearDt.day);
    for (let off = -20; off <= 5; off++) {
        const dt = tn_jdToYMD(base + off);
        const l  = Solar.fromYmdHms(dt.year, dt.month, dt.day, 12, 0, 0).getLunar();
        if (tn_sex60(l.getDayGanIndexExact(), l.getDayZhiIndexExact()) === pttnIdx60)
            return tn_getTyTimestamp(dt, lonDeg, tzH, tzId);
    }
    return null;
}

/**
 * Tính toàn bộ bảng PTTN cho khoảng Đông Chí(Y) → Đông Chí(Y+1).
 * @returns {object | null}  data object với getSoCucAt(), fmtTyLocal(), pttnName()
 */
function tn_calcPTTN(Y, lonDeg, tzH, tzId) {
    // BUG FIX: dùng findJieQi (DST-aware, UTC+8 → local đúng) thay vì tn_findTerm cũ
    const tk1 = findJieQi('冬至', Y,   11, tzId, tzH);
    const tk2 = findJieQi('夏至', Y+1,  5, tzId, tzH);
    const tk3 = findJieQi('冬至', Y+1, 11, tzId, tzH);
    if (!tk1 || !tk2 || !tk3) return null;

    const gz1 = tn_getDayGZ(tk1), gz2 = tn_getDayGZ(tk2), gz3 = tn_getDayGZ(tk3);
    const p1Idx = Math.floor(tn_sex60(gz1.can, gz1.chi) / 15) * 15;
    const p2Idx = Math.floor(tn_sex60(gz2.can, gz2.chi) / 15) * 15;
    const p3Idx = Math.floor(tn_sex60(gz3.can, gz3.chi) / 15) * 15;

    const p1Day = tn_findPTTNDay(p1Idx, tk1, lonDeg, tzH, tzId);
    const p2Day = tn_findPTTNDay(p2Idx, tk2, lonDeg, tzH, tzId);
    const p3Day = tn_findPTTNDay(p3Idx, tk3, lonDeg, tzH, tzId);
    if (!p1Day || !p2Day || !p3Day) return null;

    const toJDFrac = pd => pd.calJD + pd.hour / 24 + pd.minute / 1440;
    const D1 = Math.ceil(Math.abs(Math.round((toJDFrac(p1Day) - tk1.jdFrac) * 10) / 10));
    const D2 = Math.ceil(Math.abs(Math.round((toJDFrac(p2Day) - tk2.jdFrac) * 10) / 10));
    const D3 = Math.ceil(Math.abs(Math.round((toJDFrac(p3Day) - tk3.jdFrac) * 10) / 10));
    const b1 = D1 >= 9, b2 = !b1 && D2 >= 9, b3 = !b1 && !b2 && D3 >= 9;

    const baseJD = p1Day.calJD;
    const i2 = Math.round((p2Day.calJD - baseJD) / 15);
    const i3 = Math.round((p3Day.calJD - baseJD) / 15);

    const tkAssign = [], dupSlots = new Set();
    if (!b1) {
        if (!b2) {
            for (let i = 0; i < i2; i++)  tkAssign.push(TK_VI[i]);
            tkAssign.push(TK_VI[12]);
            for (let i = i2 + 1; i < i3; i++) tkAssign.push(TK_VI[i - i2 + 12]);
            tkAssign.push(b3 ? TK_VI[23] : TK_VI[0]);
            if (b3) dupSlots.add(i3);
        } else {
            for (let i = 0; i < i2; i++)  tkAssign.push(TK_VI[i]);
            tkAssign.push(TK_VI[11]); dupSlots.add(i2);
            tkAssign.push(TK_VI[12]);
            for (let i = i2 + 2; i < i3; i++) tkAssign.push(TK_VI[i - i2 - 2 + 13]);
            tkAssign.push(TK_VI[23]);
        }
    } else {
        for (let i = 0; i < 25; i++) tkAssign.push(TK_VI[(23 + i) % 24]);
        dupSlots.add(0);
    }
    while (tkAssign.length < 25) tkAssign.push(TK_VI[tkAssign.length % 24]);

    const caseStr       = b1 ? 'Đại Tuyết đầu' : b2 ? 'Mang Chủng lặp' : b3 ? 'Đại Tuyết lặp' : 'Chuẩn';
    const caseThreshold = b1 ? D1 : b2 ? D2 : b3 ? D3 : null;
    const startPhase    = Math.floor(p1Idx / 15) % 4;

    // tyTimes[i] = JD frac ranh giới PTTN thứ i (giờ Tý local, DST-aware)
    const tyTimes = Array.from({ length: 25 }, (_, i) => {
        const ty = tn_getTyTimestamp(tn_jdToYMD(baseJD + i * 15), lonDeg, tzH, tzId);
        return ty.dispJD + ty.hour / 24 + ty.minute / 1440;
    });

    return {
        tk1, tk2, tk3, p1Day, p2Day, p3Day,
        D1, D2, D3, b1, b2, b3, caseStr, caseThreshold,
        baseJD, tkAssign, dupSlots, startPhase, tyTimes, lonDeg, tzH, tzId,

        getSoCucAt(jdFrac, dayGanIdx, dayZhiIdx) {
            let pttnIdx = -1;
            for (let i = 0; i < 24; i++) {
                if (jdFrac >= tyTimes[i] && jdFrac < tyTimes[i + 1]) { pttnIdx = i; break; }
            }
            if (pttnIdx === -1) return null;

            const tkVi    = tkAssign[pttnIdx];
            const tkZh    = TK_VI_TO_ZH[tkVi];
            const cucArr  = TK_SO_CUC[tkZh] || [1, 1, 1];
            const isDuong = TK_DUONG_DON.has(tkZh);

            const offset    = dayGanIdx % 5;
            const phuDauChi = arrZhiZH[(dayZhiIdx - offset + 12) % 12];
            const nguyenIdx = TN_THUONG_CHI.has(phuDauChi) ? 0 : TN_TRUNG_CHI.has(phuDauChi) ? 1 : 2;
            return {
                pttnNo: pttnIdx + 1,
                pttn:   TN_PTTN_NAMES[(startPhase + pttnIdx) % 4],
                tk: tkVi, isDuong,
                don:    isDuong ? 'Dương Độn' : 'Âm Độn',
                nguyen: TN_NGUYEN_VI[nguyenIdx],
                cuc:    cucArr[nguyenIdx],
            };
        },

        fmtTyLocal(i) {
            const dispJD  = Math.floor(tyTimes[i]);
            const minFrac = Math.round((tyTimes[i] - dispJD) * 1440);
            const d = tn_jdToYMD(dispJD);
            return `${pad(d.day)}/${pad(d.month)}/${d.year} ${pad(Math.floor(minFrac / 60))}h${pad(minFrac % 60)}`;
        },

        fmtDt(dt) {
            // dt có thể là {calJD, dispJD, hour, minute} (pDay) hoặc {year, month, day, ...} (tk)
            if (dt.dispJD !== undefined) {
                const ymd = tn_jdToYMD(dt.dispJD);
                return `${pad(ymd.day)}/${pad(ymd.month)}/${ymd.year} ${pad(dt.hour || 0)}h${pad(dt.minute || 0)}`;
            }
            return `${pad(dt.day)}/${pad(dt.month)}/${dt.year} ${pad(dt.hour || 0)}h${pad(dt.minute || 0)}`;
        },

        pttnName(i) { return TN_PTTN_NAMES[(startPhase + i) % 4]; },
    };
}

/** Cache PTTN — invalidate tự động nếu country thay đổi */
let _tnCache = null;

function tn_getOrCalc(Y, lonDeg, tzH, tzId, inputJD) {
    // Invalidate nếu country thay đổi
    if (_tnCache && (_tnCache.lonDeg !== lonDeg || _tnCache.tzId !== tzId)) _tnCache = null;

    if (_tnCache && inputJD >= _tnCache.tyTimes[0] && inputJD < _tnCache.tyTimes[24]) return _tnCache;

    for (let dY = 0; dY <= 2; dY++) {
        const d = tn_calcPTTN(Y - 1 + dY, lonDeg, tzH, tzId);
        if (d && inputJD >= d.tyTimes[0] && inputJD < d.tyTimes[24]) {
            return (_tnCache = d);
        }
    }
    // Fallback: tính năm Y
    return (_tnCache = tn_calcPTTN(Y, lonDeg, tzH, tzId));
}

/**
 * Tính số cục Trí Nhuận cho ngày/giờ cụ thể.
 * @returns {{ cuc, don, pttn, pttnNo, tk, nguyen } | null}
 */
function tn_getSoCuc(year, month, day, hour, minute, lonDeg, tzH, tzId, dayGanIdx, dayZhiIdx) {
    const inputJD = tn_dateToJD(year, month, day) + hour / 24 + minute / 1440;
    const data    = tn_getOrCalc(year, lonDeg, tzH, tzId, inputJD);
    return data ? data.getSoCucAt(inputJD, dayGanIdx, dayZhiIdx) : null;
}

/** Render bảng Trí Nhuận vào #trinhuanPanel */
// ── Helpers render dùng chung — định nghĩa trước để tn/sb/ab_renderPanel đều dùng được ──

/** Badge Dương/Âm dùng CSS class (không inline style). sm=true → don-badge-sm */
function _donBadge(isDuong, sm) {
    const cls = sm ? 'don-badge-sm' : 'don-badge';
    return `<span class="${cls} ${isDuong ? 'duong' : 'am'}">${isDuong ? 'Dương' : 'Âm'}</span>`;
}
function _donBadgeSm(isDuong) { return _donBadge(isDuong, true); }

function _mkRow(isActive, i, innerHtml) {
    const tr = document.createElement('tr');
    tr.className = isActive ? 'dp-row-active' : (i % 2 === 0 ? 'dp-row-alt' : '');
    tr.innerHTML = innerHtml;
    return tr;
}

/**
 * Render bảng Trí Nhuận vào #trinhuanPanel.
 * @param {object|null} res  Kết quả getSoCucAt đã tính sẵn từ processAll (tránh tính lại).
 */
function tn_renderPanel(Y, lonDeg, tzH, tzId, inputJDFrac, dayGanIdx, dayZhiIdx, res) {
    const data = tn_getOrCalc(Y, lonDeg, tzH, tzId, inputJDFrac);
    if (!data) return;

    const { fmtDt } = data;
    getDOM('trn-pttn1').textContent = fmtDt(data.p1Day);
    getDOM('trn-pttn2').textContent = fmtDt(data.p2Day);
    getDOM('trn-pttn3').textContent = fmtDt(data.p3Day);
    getDOM('trn-tk1').textContent   = fmtDt(data.tk1);
    getDOM('trn-tk2').textContent   = fmtDt(data.tk2);
    getDOM('trn-tk3').textContent   = fmtDt(data.tk3);
    getDOM('trn-d-summary').textContent =
        `D1=${data.D1}, D2=${data.D2}, D3=${data.D3}` +
        (data.caseThreshold !== null ? ` ≥9 → ${data.caseStr}` : ` → ${data.caseStr}`);

    // res truyền vào từ processAll (cache hit) — nếu thiếu thì tính lại
    if (res === undefined) res = data.getSoCucAt(inputJDFrac, dayGanIdx, dayZhiIdx);

    const tbody = getDOM('trn-tbody');
    tbody.innerHTML = '';
    for (let i = 0; i < 25; i++) {
        const tkVi    = data.tkAssign[i];
        const tkZh    = TK_VI_TO_ZH[tkVi];
        const isDup   = data.dupSlots.has(i);
        const isDuong = TK_DUONG_DON.has(tkZh);
        const cucArr  = TK_SO_CUC[tkZh] || ['-', '-', '-'];
        const isActive = res && i + 1 === res.pttnNo;
        tbody.appendChild(_mkRow(isActive, i,
            `<td style="font-weight:600;">${data.pttnName(i)}</td>` +
            `<td class="dp-num">${data.fmtTyLocal(i)}</td>` +
            `<td>${tkVi}${isDup ? ' <span class="dp-dup">(lặp)</span>' : ''}</td>` +
            `<td class="dp-c">${_donBadgeSm(isDuong)}</td>` +
            `<td class="dp-cuc">${cucArr.join(' · ')}</td>`
        ));
    }
}

// ── Sách Bổ Pháp ──

/** Cache tiết khí Sách Bổ theo (Y, tzId) */
let _sbCache = null;

/**
 * Lấy mảng 24 chuỗi ngày local cho bảng Sách Bổ (cache-aware).
 * Dùng LunarYear.getJieQiJulianDays() — chính xác và nhanh.
 */
function sb_getJieQiDates(Y, tzId, defaultTz) {
    if (_sbCache && _sbCache.Y === Y && _sbCache.tzId === tzId) return _sbCache.dates;

    // FIX (DST cho Sách Bổ): trước đây ly.getJieQiJulianDays() được tính
    // khi _tzOffsetHours = tz (một SỐ CỐ ĐỊNH, lấy theo DST của ngày hiện
    // tại đang xem) — áp dụng đồng nhất cho CẢ 24 mốc tiết khí trải dài cả
    // năm. Với các quốc gia có DST (vd Paris: mùa hè +2, mùa đông +1), nếu
    // đang xem app vào mùa hè (_tzOffsetHours=2) thì các tiết khí mùa đông
    // (Đông Chí, Tiểu Hàn...) cũng bị cộng +2 thay vì +1 đúng — lệch 1 giờ.
    //
    // → Tính jieQiJulianDays ở mốc UTC+8 cố định (_tzOffsetHours=null), rồi
    // mỗi mốc tự quy đổi sang giờ địa phương bằng formatUTC8SolarToLocal()
    // — hàm này tự tra DST đúng theo THỜI ĐIỂM CỦA CHÍNH MỐC ĐÓ (xem
    // _tzOffsetAtJdUTC8), không dùng offset cố định chung.
    // Ephem nhớ theo (năm, mốc) và tự trả mốc múi giờ toàn cục về nguyên trạng.
    const jds = Ephem.jieQiJdAtBasis(Y + 1, null); // UTC+8, index per JIE_QI_IN_USE

    // index 1..24 = 冬至(Y), 小寒, ..., 大雪(Y)
    const dates = Array.from({ length: 24 }, (_, i) => {
        const s = Solar.fromJulianDay(jds[i + 1]);
        return tzId
            ? formatUTC8SolarToLocal(s, tzId)
            : convertBeijingToLocal(s, defaultTz);
    });
    _sbCache = { Y, tzId, dates };
    return dates;
}

/**
 * Xác định năm Y (Đông Chí) mà thời điểm input thuộc về.
 * Dùng findJieQi (đã thống nhất) thay vì sb_getDongChiJD riêng.
 */
function sb_findY(y, m, d, h, min, tzId, defaultTz) {
    const inputJD = tn_dateToJD(y, m, d) + h / 24 + min / 1440;
    for (let dY = -1; dY <= 1; dY++) {
        const tryY = y + dY;
        const dc  = findJieQi('冬至', tryY,     11, tzId, defaultTz);
        const dc1 = findJieQi('冬至', tryY + 1, 11, tzId, defaultTz);
        if (dc && dc1 && inputJD >= dc.jdFrac && inputJD < dc1.jdFrac) return tryY;
    }
    return y - 1;
}

/** Render bảng Sách Bổ vào #sachboPanel */
function sb_renderPanel(jieQiZH, y, m, d, h, min, tz, tzId) {
    const Y        = sb_findY(y, m, d, h, min, tzId, tz);
    const dateStrs = sb_getJieQiDates(Y, tzId, tz);
    const tbody    = getDOM('sb-tbody');
    tbody.innerHTML = '';
    for (let i = 0; i < 24; i++) {
        const zhName  = TK_ZH[i];
        const isDuong = TK_DUONG_DON.has(zhName);
        const cucArr  = TK_SO_CUC[zhName] || ['-', '-', '-'];
        const isActive = zhName === jieQiZH;
        tbody.appendChild(_mkRow(isActive, i,
            `<td${isActive ? ' style="font-weight:700;"' : ''}>${TK_VI[i]}</td>` +
            `<td class="dp-num">${dateStrs[i] || ''}</td>` +
            `<td class="dp-c">${_donBadgeSm(isDuong)}</td>` +
            `<td class="dp-cuc">${cucArr.join(' · ')}</td>`
        ));
    }
}

// ── Âm Bàn Pháp ──

/* ======================================================================
   ÂM BÀN PHÁP — BẢNG SÓC 12 THÁNG ÂM
   Render danh sách điểm sóc 12 tháng âm của năm âm lịch hiện tại,
   điều chỉnh theo timezone của countryData.
   ====================================================================== */

/**
 * Tính thời điểm bắt đầu Mùng 1 theo Giờ Tý thiên văn (真子時) địa phương.
 *
 * Quy ước: Mùng 1 bắt đầu từ Giờ Tý thiên văn của ngày Sóc — tức là
 * Chính Ngọ (True Solar Noon) − 13h của ngày dương lịch tương ứng với
 * firstJulianDay (ngày sóc tính theo UTC+8).
 *
 * Trả về chuỗi "DD-MM-YYYY HH:MM" (giờ địa phương).
 *
 * @param {object} socSolar  Solar object của ngày Sóc (từ firstJulianDay, UTC+8)
 * @param {number} lonDeg    Kinh độ địa phương (độ, Đông dương)
 * @param {number} tzH       Múi giờ địa phương (giờ, vd: 7 cho VN, 2 cho Paris)
 * @param {string} tzId      IANA timezone id (để quy đổi DST chính xác)
 */
function ab_getTyStart(socSolar, lonDeg, tzH, tzId) {
    // Ngày dương lịch của Sóc theo UTC+8 (đây là ngày lịch cố định)
    const socY = socSolar.getYear(), socM = socSolar.getMonth(), socD = socSolar.getDay();

    // Chính Ngọ địa phương của NGÀY SÓC (phút từ nửa đêm).
    // FIX DST: tzH truyền vào là offset tại thời điểm nhập — có thể sai cho tháng khác DST
    // (vd user nhập tháng 6 CEST=+2, nhưng tháng 11/12 là CET=+1).
    // Dùng getTimezoneOffset(tzId, date) đúng với ngày Sóc cụ thể, giống tn_getTyTimestamp.
    const effectiveTzH = tzId
        ? getTimezoneOffset(tzId, new Date(socY, socM - 1, socD, 12, 0, 0))
        : tzH;
    const noonSoc = Ephem.solarNoonMinutes(socY, socM, socD, lonDeg, effectiveTzH);

    // Giờ Tý bắt đầu tại Chính Ngọ − 13h = noonMins − 780
    // Giá trị có thể âm (tức là trước nửa đêm, thuộc ngày hôm trước);
    // ta quy về phút dương mod 1440 để lấy giờ đồng hồ địa phương.
    const tyStartMins = ((noonSoc - 780) % 1440 + 1440) % 1440;

    // Ngày lịch địa phương của điểm Giờ Tý:
    // Nếu noonSoc − 780 < 0 → Giờ Tý bắt đầu trước nửa đêm ngày Sóc
    //   → lịch: ngày Sóc (vì tyStart = hhmm trong cùng ngày địa phương, trước 00:00 thuộc ngày trước)
    // Thực tế đơn giản hơn: chỉ cần thể hiện HH:MM của Giờ Tý; ngày lịch
    // chính là ngày của socSolar (firstJulianDay) vì chúng ta đã đảm bảo
    // lunarLocal dùng shift phù hợp.
    // Dùng Math.round (nhất quán với tn_getTyTimestamp dòng 10490)
    // để tránh lệch ±1 phút khi phần thập phân EoT ≥ 0.5.
    const rawStart = noonSoc - 780;
    const dispMins = ((rawStart % 1440) + 1440) % 1440;
    const tyH   = Math.floor(dispMins / 60);
    const tyMin = Math.round(dispMins % 60);

    // Nếu rawStart < 0 → Giờ Tý bắt đầu trước nửa đêm → thuộc ngày hôm trước ngày Sóc.
    let dispY = socY, dispM = socM, dispD = socD;
    if (rawStart < 0) {
        const prev = new Date(Date.UTC(socY, socM - 1, socD) - 86400000);
        dispY = prev.getUTCFullYear(); dispM = prev.getUTCMonth() + 1; dispD = prev.getUTCDate();
    }
    return `${pad(dispD)}-${pad(dispM)}-${dispY} ${pad(tyH)}:${pad(tyMin)}`;
}

/**
 * Render bảng sóc 12 tháng âm vào panel #ambanPanel.
 * @param {number} lunarYear   - Năm âm lịch (vd: 2025)
 * @param {number} lunarMonth  - Tháng âm lịch hiện tại (để highlight hàng, có dấu)
 * @param {string} tzId        - Timezone ID (vd: "Asia/Ho_Chi_Minh")
 * @param {number} lonDeg      - Kinh độ địa phương (độ)
 * @param {number} tzH         - Múi giờ địa phương (giờ)
 */
function ab_renderPanel(lunarYear, lunarMonth, tzId, lonDeg, tzH) {
    const tbody = getDOM('ab-tbody');
    if (!tbody) return;
    tbody.innerHTML = '';

    // Mốc múi giờ ĐỊA PHƯƠNG, không phải UTC+8.
    //
    // Chính hàng này là chỗ lỗi lộ rõ nhất: cột "Sóc" quy về giờ địa phương
    // (formatPreciseSocLocal) còn cột "Mùng 1" lại lấy từ mo.getFirstJulianDay()
    // tính ở UTC+8 — hai hệ quy chiếu nằm cạnh nhau trong CÙNG MỘT DÒNG. Ở
    // Paris tháng 7 âm 2026: Sóc ghi 12-08-2026 19:37 nhưng Mùng 1 ghi
    // 13-08-2026, trong khi quy tắc là mùng 1 phải là ngày CHỨA điểm Sóc.
    //
    // Đặt cùng một mốc cho cả hai thì cột Mùng 1 (và cột Rằm, vốn là mùng 1 +
    // 14) tự khớp với giờ Sóc đang hiện.
    // Mốc múi giờ ĐỊA PHƯƠNG cho cả khối (xem ghi chú dưới); Ephem giữ và trả
    // lại biến toàn cục giúp, nên không cần tự chụp ảnh nữa.
    const monthsOfYear = Ephem.monthsAtBasis(lunarYear, tzH);

    // FIX (tháng nhuận): trước đây loop `for mo=1..12` gọi
    // Lunar.fromYmd(lunarYear, mo, 1) — với năm có tháng nhuận (vd 2025 có
    // nhuận tháng 6, getMonth()=-6), loop này HOÀN TOÀN BỎ QUA tháng nhuận
    // (sóc 25/07/2025 không xuất hiện), và nếu ngày hiện tại rơi vào tháng
    // nhuận (lunarMonth = -6, lunarMonthNum = abs = 6), dòng được highlight
    // là "Tháng 6" thường (sóc 25/06/2025) — SAI, vì sóc đúng của tháng hiện
    // tại là 25/07/2025.
    // → Dùng LunarYear.fromYear(lunarYear).getMonths(), lọc các tháng thuộc
    // đúng lunarYear (kể cả tháng nhuận, getMonth() âm), giữ thứ tự thời
    // gian gốc của thư viện (đã đúng theo lịch), và so khớp active bằng
    // getMonth() có dấu (lunarMonth truyền vào nay là lunar.getMonth() có
    // dấu, không phải Math.abs()).
    for (const mo of monthsOfYear) {
        const isLeap   = mo.leap;
        const moAbs    = mo.month;
        const moNum    = isLeap ? -moAbs : moAbs;
        const socSolar = Solar.fromJulianDay(mo.jd);

        const socStr   = formatPreciseSocLocal(socSolar, tzId);
        // Mùng 1: ngày CHỨA điểm Sóc, đếm từ Chính Tý tới Chính Tý — không
        // phải ngày dương của firstJulianDay. Xem khối ghi chú về Chính Tý.
        const g1       = zi_mung1FromJd(mo.jd, lonDeg, tzId);
        const mung1Str = `${pad(g1.d)}-${pad(g1.m)}-${g1.y}`;
        // Rằm: ngày 15 âm = mùng 1 + 14
        const ram      = new Date(g1.y, g1.m - 1, g1.d + 14);
        const ramStr   = `${pad(ram.getDate())}-${pad(ram.getMonth() + 1)}-${ram.getFullYear()}`;
        const isActive = (moNum === lunarMonth);
        const fw = isActive ? ' style="font-weight:700;"' : '';
        const label = isLeap ? `Tháng ${moAbs} (Nhuận)` : `Tháng ${moAbs}`;
        tbody.appendChild(_mkRow(isActive, moAbs,
            `<td class="dp-c"${fw}>${label}</td>` +
            `<td class="dp-num-ab"${fw}>${socStr}</td>` +
            `<td class="dp-num-ab"${fw}>${mung1Str}</td>` +
            `<td class="dp-num-ab"${fw}>${ramStr}</td>`
        ));
    }
}

/* ======================================================================
   Điểm khởi đầu cho mọi phương pháp lập bàn Kỳ Môn Độn Giáp.
   Input thuần túy — không đọc DOM, không có side effect.

   @param {object} p
   @param {'amban'|'bophap'} p.method      Phương pháp lập bàn
   @param {string}  p.jieQiZH              Tiết khí hiện tại (chữ Hán, vd: "芒种")
   @param {string}  p.dayGanHan            Can ngày (chữ Hán)
   @param {string}  p.dayZhiHan            Chi ngày (chữ Hán)
   @param {string}  p.yearZhiHan           Chi năm âm lịch (chữ Hán)
   @param {number}  p.lunarMonthNum        Tháng âm lịch (số dương, đã abs())
   @param {number}  p.lunarDay             Ngày âm lịch
   @param {string}  p.timeZhiHan           Chi giờ (chữ Hán)
   @returns {{ cuc: number, don: 'duong'|'am' }}
   ====================================================================== */
// _BOPHAP_THUONG/_BOPHAP_TRUNG: dùng chung với TN — alias, không tạo Set mới
const _BOPHAP_THUONG = TN_THUONG_CHI;
const _BOPHAP_TRUNG  = TN_TRUNG_CHI;

function calculateCucDon({ method, jieQiZH,
                            dayGanHan, dayZhiHan,
                            yearZhiHan, lunarMonthNum, lunarDay, timeZhiHan,
                            trinhuanResult }) {
    // Âm/Dương Độn: mặc định theo Tiết Khí (TK_AM_DON là Set → O(1))
    let don = TK_AM_DON.has(jieQiZH) ? 'am' : 'duong';
    let cuc;

    if (method === 'trinhuan') {
        // Trí Nhuận Pháp: số cục từ kết quả Pháp Tiết Thiên Nhật
        if (trinhuanResult) {
            cuc = trinhuanResult.cuc;
            don = trinhuanResult.isDuong ? 'duong' : 'am';
        } else {
            cuc = 1; // fallback
        }
    } else if (method === 'bophap') {
        // Sách Bổ Pháp: tìm Phù Đầu → tra bảng TK_SO_CUC
        const ganIdx = arrGanZH.indexOf(dayGanHan);
        const chiIdx = arrZhiZH.indexOf(dayZhiHan);
        if (ganIdx === -1 || chiIdx === -1) return { cuc: 1, don };

        const phuDauChi = arrZhiZH[(chiIdx - ganIdx % 5 + 12) % 12];
        const nguyenIdx = _BOPHAP_THUONG.has(phuDauChi) ? 0
                        : _BOPHAP_TRUNG.has(phuDauChi)  ? 1 : 2;
        cuc = TK_SO_CUC[jieQiZH][nguyenIdx];
    } else {
        // Âm Bàn Pháp: cộng Tứ Trụ theo Âm Lịch
        cuc = (chiToStt[yearZhiHan] + lunarMonthNum + lunarDay + chiToStt[timeZhiHan]) % 9 || 9;
    }

    return { cuc, don };
}

/** Padding helper — dùng chung toàn file (đã khai báo ở đầu script) */

/* ─────────────────────────────────────────────────────────────────────────
   HELPERS nội bộ của processAll — định nghĩa 1 lần, tái dùng mỗi lần gọi
   ───────────────────────────────────────────────────────────────────────── */

/** Đọc thời gian UTC+8 từ input cục bộ */
function _readInputBJ(y, m, d, h, min, tz) {
    const dUTC = new Date(Date.UTC(y, m - 1, d, h, min) - tz * 3600000 + 8 * 3600000);
    const solar = Solar.fromYmdHms(
        dUTC.getUTCFullYear(), dUTC.getUTCMonth() + 1, dUTC.getUTCDate(),
        dUTC.getUTCHours(),    dUTC.getUTCMinutes(),   dUTC.getUTCSeconds()
    );
    return { solarBJ: solar, dUTC };
}

/** Tính Tiết Khí chuẩn xác (sửa lỗi nhảy độn sớm) */
function _getPrevJieQi(solarBJ, dUTC) {
    let jqObj   = solarBJ.getLunar().getPrevJieQi(true);
    let jqSolar = jqObj.getSolar();
    if (solarToCompareNum(solarBJ) < solarToCompareNum(jqSolar)) {
        const prev = new Date(dUTC.getTime() - 86400000);
        jqObj   = Solar.fromYmdHms(
            prev.getUTCFullYear(), prev.getUTCMonth() + 1, prev.getUTCDate(),
            prev.getUTCHours(),    prev.getUTCMinutes(),   prev.getUTCSeconds()
        ).getLunar().getPrevJieQi(true);
        jqSolar = jqObj.getSolar();
    }
    return { jqObj, jqSolar };
}

/** Tạo HTML cho một cell thường (không phải trung cung) */
function _buildCell(lt, res, isCoban, isZH,
                    noiBanSet, kvLtSet, dmLT, dmTxtStr,
                    canGioSoSanh, canNgaySoSanh, canThangSoSanh, canNamSoSanh,
                    hlCanThien, truPhuTinh, truSuMon) {
    const kvPresent = kvLtSet.has(lt);
    const dmPresent = dmLT === lt;
    const sameCell  = kvPresent && dmPresent;
    const noiBanClass = noiBanSet.has(lt) ? 'noi-ban' : '';

    const kvHtml  = kvPresent ? `<div class="${sameCell ? 'kv-mark' : 'kv-mark-solo'}"></div>` : '';
    const dmFontSz = isZH ? (sameCell ? '10px' : '11px') : '7px';
    const dmHtml  = dmPresent
        ? `<div class="${sameCell ? 'dichma-mark' : 'dichma-mark-solo'}" style="font-size:${dmFontSz}">${dmTxtStr}</div>`
        : '';

    // Marker Bát Tự
    const canListHT = Array.isArray(res.thienCan[lt]) ? res.thienCan[lt] : [res.thienCan[lt]];
    let markerStr = '';
    if (canListHT.includes(canGioSoSanh))   markerStr += 'H';
    if (canListHT.includes(canNgaySoSanh))  markerStr += 'D';
    if (canListHT.includes(canThangSoSanh)) markerStr += 'M';
    if (canListHT.includes(canNamSoSanh))   markerStr += 'Y';
    let baziMarkerHtml = '';
    if (markerStr) {
        if (!isCoban) {
            const cls = markerStr.length > 1 ? 'marker-multi' : 'marker-single';
            baziMarkerHtml = `<div class="sub-cell pos-tu-cuong ${cls}">${markerStr.split('').join('<br>')}</div>`;
        } else {
            const n = markerStr.length;
            baziMarkerHtml = `<div class="sub-cell pos-tu-cuong marker-coban-${n}">${
                markerStr.split('').map(c => `<span>${c}</span>`).join('')
            }</div>`;
        }
    }

    const canDiaArr = lt === 2 ? [...res.diaBan[5], ...res.diaBan[2]] : res.diaBan[lt];

    const numArr    = ltNumberArrMap[lt] || [];
    const ltDisplay = numArr.length === 4
        ? `<span><b>${numArr[0]}</b> ${numArr[1]}</span><span>${numArr[2]} ${numArr[3]}</span>`
        : lt;

    const isTrucPhuTinh = res.thienTinh[lt] === truPhuTinh ? 'bold-km' : '';
    const isTrucSu      = res.thienMon[lt]  === truSuMon   ? 'bold-km' : '';

    const cellHtml = `<div class="cell ${noiBanClass}">
        ${kvHtml}${dmHtml}
        <div class="sub-cell pos-lt-num">${ltDisplay}</div>
        <div class="sub-cell pos-than-thien ${getRC(res.thanTB[lt])}">${res.thanTB[lt]||''}</div>
        <div class="sub-cell pos-tinh ${getRC(res.thienTinh[lt])} ${isTrucPhuTinh}">${res.thienTinh[lt]||''}</div>
        <div class="sub-cell pos-mon ${getRC(res.thienMon[lt])} ${isTrucSu}">${res.thienMon[lt]||''}</div>
        <div class="sub-cell pos-than-dia ${getRC(res.thanDB[lt])}">${res.thanDB[lt]||''}</div>
        <div class="pos-que-cung-hau-thien">${hauThienMapCung[lt]||''}</div>
        <div class="pos-que-cung-tien-thien">${tienThienMapCung[lt]||''}</div>
        ${renderCanPair(res.thienCan[lt], hlCanThien,  'pos-can-thien')}
        ${renderCanPair(canDiaArr,        null,        'pos-can-dia')}
        ${renderCanPair(res.amCan[lt],    null,        'pos-can-am')}
        ${renderCanPair(res.anCan[lt],    null,        'pos-can-an')}
        ${baziMarkerHtml}
    </div>`;

    const backHtml = renderFlipContent(res.thienCan[lt], canDiaArr);
    return { cellHtml, backContent: { html: backHtml, noiBanClass } };
}

/* ======================================================================
   XỬ LÝ DỮ LIỆU TỪ INPUT VÀ RENDER
   ====================================================================== */

// ════════════════════════════════════════════════════════════════
// LỊCH TÂY TẠNG (Phugpa system) — port từ thư viện mã nguồn mở
// @hnw/date-tibetan (MIT License, https://github.com/hnw/date-tibetan),
// dựa trên công trình "Tibetan calendar mathematics" của Svante Janson
// (Đại học Uppsala, 2007/2014). Bản port này thay thư viện JD phụ
// thuộc "astronomia" bằng hàm Solar.getJulianDay()/Solar.fromJulianDay()
// đã có sẵn trong file (đã kiểm chứng cho kết quả giống hệt astronomia
// trên cùng bộ dữ liệu test, và khớp các ngày Losar thực tế 2024/2025/2026).
// ════════════════════════════════════════════════════════════════
function TibetanCalendar(cycle, year, month, leapMonth, day, leapDay) {
    // Epoch constants for Phugpa E806 (Year 806, Month 3)
    this._M0 = 2015501 + 4783 / 5656;
    this._M1 = 167025 / 5656;
    this._M2 = 11135 / 11312;
    this._S0 = 743 / 804;
    this._S1 = 65 / 804;
    this._S2 = 13 / 4824;
    this._A0 = 475 / 3528;
    this._A1 = 253 / 3528;
    this._A2 = 1 / 28;
    this._P0 = 139 / 180;
    this._EPOCH_YEAR = 806;
    this._JD_OFFSET_STD_TIME = (6 + 4 / 60) / 24;   // Lhasa Mean Time
    this._JD_OFFSET_DAY_START = (-5 + 12) / 24;
    this._IS_BHUTAN_LEAP = false;
    this._EPOCH_RAB_BYUNG = 1027;                    // start of first Rabjung cycle (AD)
    this._moon_tab_values = [0, 5, 10, 15, 19, 22, 24, 25];
    this._sun_tab_values = [0, 6, 10, 11];
    this.set(cycle, year, month, leapMonth, day, leapDay);
}
TibetanCalendar.prototype.set = function (cycle, year, month, leapMonth, day, leapDay) {
    if (cycle instanceof TibetanCalendar) {
        this.cycle = cycle.cycle; this.year = cycle.year; this.month = cycle.month;
        this.leapMonth = !!cycle.leapMonth; this.day = cycle.day; this.leapDay = !!cycle.leapDay;
    } else if (Array.isArray(cycle)) {
        this.cycle = cycle[0]; this.year = cycle[1]; this.month = cycle[2];
        this.leapMonth = !!cycle[3]; this.day = cycle[4]; this.leapDay = !!cycle[5];
    } else {
        this.cycle = cycle; this.year = year; this.month = month;
        this.leapMonth = !!leapMonth; this.day = day; this.leapDay = !!leapDay;
    }
    return this;
};
TibetanCalendar.prototype._getAlpha = function () {
    return 12 * (this._S0 - this._P0);
};
TibetanCalendar.prototype._getBeta = function () {
    var alpha = this._getAlpha();
    return Math.ceil(67 * alpha) - (this._IS_BHUTAN_LEAP ? 2 : 0);
};
TibetanCalendar.prototype._getGregorianYear = function () {
    return this._EPOCH_RAB_BYUNG + (this.cycle - 1) * 60 + (this.year - 1);
};
TibetanCalendar.prototype.epochCycleFromYear = function (gyear) {
    var cycle = Math.floor((gyear - this._EPOCH_RAB_BYUNG) / 60) + 1;
    var year = ((gyear - this._EPOCH_RAB_BYUNG) % 60) + 1;
    return { cycle: cycle, year: year };
};
TibetanCalendar.prototype._isLeapMonthFromYearAndMonth = function (year, month) {
    var beta = this._getBeta();
    var M_prime = 12 * (year - this._EPOCH_YEAR) + month;
    var mod_65_val = (M_prime * 2 - beta) % 65;
    return (mod_65_val == 0 || mod_65_val == 1);
};
TibetanCalendar.prototype._getTibetanMonthFromTrueMonthCount = function (trueMonthCount) {
    var n = trueMonthCount;
    var beta = this._getBeta();
    var x = Math.ceil((65 * n + beta) / 67);
    var M = x % 12;
    if (M == 0) M = 12;
    var Y = (x - M) / 12 + this._EPOCH_YEAR;
    var L = false;
    var leapX = Math.ceil((65 * (this._IS_BHUTAN_LEAP ? n - 1 : n + 1) + beta) / 67);
    if (x == leapX) L = true;
    var cy = this.epochCycleFromYear(Y);
    return { cycle: cy.cycle, year: cy.year, month: M, leapMonth: L };
};
TibetanCalendar.prototype._linearInterpolate = function (x_in, tableValues, halfSymmetryLen, periodLen) {
    var x = x_in % periodLen;
    if (x < 0) x += periodLen;
    var sign = 1;
    var symmetryPoint = halfSymmetryLen * 2;
    if (x >= symmetryPoint) { sign = -1; x -= symmetryPoint; }
    if (x > halfSymmetryLen) { x = symmetryPoint - x; }
    var i = Math.floor(x);
    var frac = x - i;
    var val;
    if (i < 0) { val = tableValues[0]; }
    else if (i >= tableValues.length - 1) { val = tableValues[tableValues.length - 1]; }
    else { val = tableValues[i] * (1 - frac) + tableValues[i + 1] * frac; }
    return sign * val;
};
TibetanCalendar.prototype.getTrueDate = function (trueMonthCount, lunarDay) {
    var n = trueMonthCount, d = lunarDay;
    var mean_date = n * this._M1 + d * this._M2 + this._M0;
    var mean_sun = n * this._S1 + d * this._S2 + this._S0;
    mean_sun = mean_sun - Math.floor(mean_sun);
    var anomaly_moon = n * this._A1 + d * this._A2 + this._A0;
    anomaly_moon = anomaly_moon - Math.floor(anomaly_moon);
    var moon_equ = this._linearInterpolate(28 * anomaly_moon, this._moon_tab_values, 7, 28);
    var anomaly_sun = mean_sun - 1 / 4;
    anomaly_sun = anomaly_sun - Math.floor(anomaly_sun);
    var sun_equ = this._linearInterpolate(12 * anomaly_sun, this._sun_tab_values, 3, 12);
    return mean_date + moon_equ / 60 - sun_equ / 60;
};
TibetanCalendar.prototype._from = function (jd) {
    var jdn = Math.trunc(parseFloat(jd.toFixed(7)));
    var solarDaysFromEpoch = jdn - this._M0;
    var n = Math.floor(solarDaysFromEpoch / this._M1);
    var day = Math.floor((solarDaysFromEpoch - n * this._M1) / this._M2);
    var leapDay = false;
    for (var i = 0; i < 3; i++) {
        var trueDate = this.getTrueDate(n, day);
        if (trueDate > jdn + 1) { leapDay = true; break; }
        else if (trueDate > jdn) { break; }
        day++;
    }
    if (day == 0) { n--; day = 30; }
    if (day > 30) { n++; day -= 30; }
    var r = this._getTibetanMonthFromTrueMonthCount(n);
    this.set(r.cycle, r.year, r.month, r.leapMonth, day, leapDay);
};
TibetanCalendar.prototype.fromGregorian = function (year, month, day) {
    // Solar.getJulianDay() đã được kiểm chứng cho cùng kết quả với
    // astronomia.julian.CalendarGregorian(...).toJD() (lệch 0 tuyệt đối
    // trên các ngày test), nên dùng trực tiếp thay cho dependency ngoài.
    var jdn = Solar.fromYmd(year, month, day).getJulianDay() + 0.5; // 12:00
    this._from(jdn);
    return this;
};

// Ngày vía Phật/Bồ Tát theo lịch Tây Tạng (Kim Cương Thừa), tính theo ngày
// trong tháng của lịch Tây Tạng (tib.day), không phụ thuộc lịch âm Trung Hoa.
const VIA_DAYS = {
    1:  { vi: 'Đức Phật A Súc Bệ',          zh: '阿閦佛' },
    8:  { vi: 'Đức Phật Dược Sư',           zh: '药师佛' },
    10: { vi: 'Đức Liên Hoa Sinh',          zh: '莲花生大士' },
    14: { vi: 'Chư Phật khắp mười phương',  zh: '十方诸佛' },
    15: { vi: 'Đức Phật A Di Đà',           zh: '阿弥陀佛' },
    18: { vi: 'Đức Quán Thế Âm Bồ Tát',     zh: '观世音菩萨' },
    23: { vi: 'Đức Phật Đại Nhật Như Lai',  zh: '大日如来' },
    24: { vi: 'Đức Phật Phổ Hiền Vương',    zh: '普贤王如来' },
    28: { vi: 'Ngũ Trí Phật Như Lai',       zh: '五智如来' },
    29: { vi: 'Kim Cương Hộ Pháp',          zh: '金刚护法' },
    30: { vi: 'Đức Phật Thích Ca',          zh: '释迦牟尼佛' },
};

function processAll() {
    try {
        if (typeof Solar === 'undefined') {
            alert('Lỗi tải thư viện lunar.js! Vui lòng đặt file lunar.js ở cùng thư mục hoặc dán code vào block được cung cấp.');
            return;
        }

        // ── 1. Đọc input ──
        const y      = parseInt(getDOM('inYear').value);
        const m      = parseInt(getDOM('inMonth').value);
        const d      = parseInt(getDOM('inDay').value);
        const h      = parseInt(getDOM('solarHour').value);
        const min    = parseInt(getDOM('solarMinute').value);
        const method = getDOM('methodSelect').value;

        // ── 2. Múi giờ & kinh độ ──
        const countryKey = getDOM('country').value;
        const info       = countryData[countryKey];
        const tz         = getTimezoneOffset(info.tzId, new Date(y, m - 1, d, h));
        const lon        = info.lon;

        // ── 3. Chính Ngọ (kinh độ + Equation of Time) ──
        // Chính Ngọ lấy thẳng từ Ephem để cả ứng dụng chung một con số;
        // eotMins suy ngược ra vì bước 4 (dịch sang giờ Mặt Trời thật) cần nó.
        const lonOffsetMins = (lon - tz * 15) * 4;
        const noonMins      = Ephem.solarNoonMinutes(y, m, d, lon, tz);
        const eotMins       = 720 - lonOffsetMins - noonMins;
        const chinhNgoStr   = `${pad(Math.floor(noonMins / 60))}:${pad(Math.floor(noonMins % 60))}`;
        const gmtStr        = `GMT${tz >= 0 ? '+' : ''}${tz}`;

        // ── 4. Bát Tự: Local (Chân Thái Dương) + Bắc Kinh (UTC+8) ──
        //
        // QUY ƯỚC THIÊN VĂN (真子時):
        // Giờ Tý bắt đầu tại Chính Ngọ − 13h (= noonMins − 780 phút).
        // Ngày âm lịch đổi tại thời điểm này, không phải tại nửa đêm đồng hồ.
        // Mùng 1 của tháng bắt đầu từ Giờ Tý thiên văn của ngày Sóc, không từ điểm Sóc.
        //
        // SHIFT CHO lunarLocal:
        // Để lunar.js (dùng ranh giới nửa đêm lịch) cho kết quả đúng, ta dịch
        // exactDate sao cho ranh giới Giờ Tý (tyStartMins = noonMins − 780) ánh xạ
        // về đúng 00:00 trong giờ đã dịch.
        //
        //   tyStartMins = noonMins − 780 = (720 − lonOffsetMins − eotMins) − 780
        //               = −60 − lonOffsetMins − eotMins
        //   shift cần = −tyStartMins (mod 1440) = 60 + lonOffsetMins + eotMins
        //
        // Chứng minh: tyStartMins + shift = (−60 − lonOff − eot) + (60 + lonOff + eot) = 0 ✓
        //
        // Lưu ý: shift này cũng tự động làm cho ranh giới "giờ Tý = 23:00 TST"
        // trong _computeDay.dayGanExact (hm >= '23:00') vẫn đúng — vì:
        //   TST(tyStart) = tyStart + lonOff + eot = −60 → 23:00 (mod 1440) ← bất biến.
        //
        // FIX: KHÔNG gọi ShouXingUtil.setTzOffsetHours(tz) trước bước này.
        // Việc set _tzOffsetHours toàn cục TRƯỚC khi tính lunarLocal làm
        // LunarYear.fromYear() tính lại điểm Sóc các tháng theo múi giờ
        // địa phương, có thể đẩy điểm Sóc qua ranh giới ngày dương lịch
        // (lệch so với mốc UTC+8 chuẩn của lịch Trung Hoa), khiến độ dài
        // tháng âm lịch bị tính sai 29/30 ngày → ngày âm lịch hiển thị
        // sai (ví dụ 14/6/2026 ra ngày 30 thay vì 29 khi chọn múi giờ
        // lệch xa UTC+8 như châu Âu).
        // → Đảm bảo _tzOffsetHours = null (mặc định UTC+8) khi tính
        // lunarLocal/ngày-tháng-năm âm lịch của ngày hiện tại; chỉ set tz
        // địa phương SAU bước tính baziBJ/Tiết Khí (vì solarBJ là giờ Bắc
        // Kinh — xem ghi chú ở bước 7), dành riêng cho bảng Sóc/Tiết Khí
        // (Sách Bổ / Âm Bàn) bên dưới.
        ShouXingUtil.setTzOffsetHours(null);
        const exactDate = new Date(y, m - 1, d, h, min);
        exactDate.setMinutes(exactDate.getMinutes() + lonOffsetMins + eotMins + 60);
        const lunarLocal = Solar.fromDate(exactDate).getLunar();
        const baziLocal  = lunarLocal.getEightChar();

        // NGÀY ÂM LỊCH tính ở mốc múi giờ ĐỊA PHƯƠNG, không phải UTC+8.
        //
        // Quy tắc của lịch âm: mùng 1 là ngày CHỨA điểm Sóc. "Ngày" nào thì
        // tuỳ mốc quy chiếu — và mốc ấy phải trùng với mốc dùng để HIỆN giờ
        // Sóc, nếu không một màn hình có hai hệ quy chiếu. Đúng lỗi này:
        // ở Paris, Sóc hiện 12/08/2026 19:37 (giờ Paris) trong khi mùng 1
        // lại là 13/08 (vì tính ở UTC+8, nơi Sóc rơi vào 13/08 00:37).
        //
        // Ghi chú cũ ở đây cảnh báo rằng đặt mốc địa phương làm hỏng độ dài
        // tháng 29/30. Đã kiểm lại: KHÔNG đúng. Quét 2020–2035 ở các mốc từ
        // UTC−8 tới UTC+12, mọi tháng đều 29 hoặc 30 ngày, số ngày âm liên
        // tục, và mùng 1 luôn chứa Sóc (198/198 tháng mỗi mốc).
        //
        // Chỉ đổi NGÀY ÂM LỊCH. Năm/Tháng Can Chi vẫn lấy từ baziBJ (đổi tại
        // Lập Xuân và các mốc Tiết, so với solarBJ ở giờ Bắc Kinh) và tiết khí
        // vẫn tính ở mốc UTC+8 — đó là lý do thật của ghi chú cũ, và nó vẫn
        // đúng: trộn solarBJ giờ Bắc Kinh với mốc Lập Xuân giờ địa phương thì
        // Can Chi năm/tháng lệch hẳn.
        ShouXingUtil.setTzOffsetHours(tz);
        const lunarDisp = Solar.fromDate(exactDate).getLunar();
        ShouXingUtil.setTzOffsetHours(null);

        // …và mốc bắt đầu tháng còn phải chỉnh theo CHÍNH TÝ nữa: mùng 1 là
        // ngày chứa điểm Sóc, đếm từ nửa đêm mặt trời thật chứ không phải
        // 00:00 đồng hồ (xem khối "RANH GIỚI NGÀY ÂM LỊCH" ở trên).
        // exactDate đã là giờ mặt trời thật nên hỏi bằng chính ngày dương đang
        // xét: trụ ngày của nó cũng lấy từ mốc giờ Tý ấy.
        const ziDay = zi_dayOf(Solar.fromYmdHms(y, m, d, h, min, 0), lon, info.tzId);
        const ziLunar = zi_lunarOf(ziDay.y, ziDay.m, ziDay.d, lon, info.tzId, tz);

        // FIX: _readInputBJ() trả về solarBJ ở GIỜ BẮC KINH (UTC+8) — Năm/Tháng
        // Can Chi (baziBJ.getYearGan/Zhi, getMonthGan/Zhi) phụ thuộc vào việc so
        // sánh solarBJ với mốc Lập Xuân (yearGanIndexByLiChun/Exact trong
        // _computeYear), và mốc Lập Xuân đó được lấy từ
        // LunarYear.getJieQiJulianDays() — vốn được tính theo _tzOffsetHours
        // toàn cục. Nếu _tzOffsetHours = tz (múi giờ địa phương, có thể lệch xa
        // UTC+8, ví dụ Paris +2 vs Bắc Kinh +8 = lệch 6h), mốc Lập Xuân/Tiết Khí
        // sẽ bị tính theo giờ địa phương trong khi solarBJ vẫn là giờ Bắc Kinh
        // → so sánh lệch múi giờ, có thể đẩy sai ngày-tháng Can Chi năm/tháng
        // hoặc chọn sai jieQiZH (tiết khí hiện tại) gần ranh giới giao tiết.
        // → Giữ _tzOffsetHours = null (UTC+8) cho cả baziBJ và _getPrevJieQi.
        const { solarBJ, dUTC } = _readInputBJ(y, m, d, h, min, tz);
        const baziBJ  = solarBJ.getLunar().getEightChar();
        // Ngày-tháng-năm âm lịch (hiển thị + cục Âm Bàn + bảng Sóc) lấy bản đã
        // chỉnh theo Chính Tý; baziLocal vẫn dùng lunarLocal (can chi ngày là chu
        // kỳ 60 ngày liên tục, không phụ thuộc mốc này).
        // ziLunar chỉ null khi lunar.js không dựng nổi danh sách tháng — lúc ấy
        // lùi về bản mốc địa phương còn hơn là hỏng cả trang.
        const lunar = ziLunar ? {
            getDay:   () => ziLunar.day,
            getMonth: () => ziLunar.leap ? -ziLunar.month : ziLunar.month,
            getYear:  () => ziLunar.year,
        } : lunarDisp;

        // Đối tượng bazi tổng hợp (Năm/Tháng từ BJ, Ngày/Giờ từ Local).
        // Giờ Can/Chi (getTimeGan/Zhi/getTime) được ghi đè bằng giá trị thiên văn
        // ở bước 5 bên dưới (sau khi tính astTimeZhiIdx).
        const bazi = {
            getYearGan:  () => baziBJ.getYearGan(),    getYearZhi:  () => baziBJ.getYearZhi(),
            getMonthGan: () => baziBJ.getMonthGan(),   getMonthZhi: () => baziBJ.getMonthZhi(),
            getDayGan:   () => baziLocal.getDayGan(),  getDayZhi:   () => baziLocal.getDayZhi(),
            getTimeGan:  () => bazi._astTimeGan,       getTimeZhi:  () => bazi._astTimeZhi,
            getYear:     () => baziBJ.getYear(),       getMonth:    () => baziBJ.getMonth(),
            getDay:      () => baziLocal.getDay(),     getTime:     () => bazi._astTime,
            _astTimeGan: null, _astTimeZhi: null, _astTime: null
        };

        // ── 5. Can/Chi cần dùng ──
        // FIX: Năm Can Chi (yearGanHan/yearZhiHan) và Tháng Can Chi
        // (monthGanHan/monthZhiHan) dùng theo Bát Tự CHUẨN — đổi năm/tháng
        // tại các mốc TIẾT (Lập Xuân cho năm; 12 Tiết trong 24 tiết khí cho
        // tháng) — KHÔNG đổi tại mốc Sóc (mùng 1 âm lịch/Tết).
        //
        // Trước đây yearGanHan/yearZhiHan = lunar.getYearGan/Zhi() (theo
        // Sóc/Tết) còn dayGanHan/dayZhiHan/timeGanHan/timeZhiHan lại lấy từ
        // `bazi` (đúng Bát Tự). Khoảng ~13 ngày mỗi năm — từ Lập Xuân
        // (~4/2) đến Tết (mùng 1 tháng Giêng) — hai mốc này LỆCH NHAU 1 năm
        // Can Chi (vd 2026: 04/02-16/02 Lập Xuân đã sang Bính Ngọ nhưng Sóc
        // vẫn còn Ất Tỵ), khiến canChiNam hiển thị sai VÀ cuc (Âm Bàn Pháp,
        // dùng chiToStt[yearZhiHan]) bị tính sai trong khoảng đó.
        //
        // Tương tự, monthGanHan/monthZhiHan trước đây tính từ lunarMonthNum
        // (số tháng âm lịch theo Sóc) qua công thức Ngũ Hổ Độn — sai khác
        // với chuẩn Bát Tự (đổi tháng tại mốc Tiết) trong ~23% số ngày mỗi
        // năm (ngay sau mỗi lần giao Tiết).
        // → Dùng trực tiếp baziBJ.getYearGan/Zhi() và
        // baziBJ.getMonthGan/Zhi() (đã có sẵn, tính theo Lập Xuân/Tiết).
        //
        // Lưu ý: lunar.getYear()/getMonth()/getDay() (Sóc-based) VẪN được
        // giữ nguyên cho phần "ngày-tháng-năm ÂM LỊCH" hiển thị
        // (lunarMonthNum, socStr, ab_renderPanel...) — đó là quy ước lịch âm
        // dân gian đúng, không đổi.
        const yearGanHan = baziBJ.getYearGan(), yearZhiHan = baziBJ.getYearZhi();
        const dayGanHan  = bazi.getDayGan(),    dayZhiHan  = bazi.getDayZhi();

        // ── Giờ Tý thiên văn (真子時) ──
        // Giờ Tý bắt đầu tại noonMins − 780 (Chính Ngọ − 13h).
        // Mỗi thời辰 = 120 phút thực, tính từ ranh giới Giờ Tý này.
        // Index: 0=子 Tý, 1=丑 Sửu, …, 11=亥 Hợi.
        const tyStartMins    = noonMins - 780;
        const inputMins      = h * 60 + min;
        const astTimeZhiIdx  = Math.floor(((inputMins - tyStartMins) % 1440 + 1440) % 1440 / 120) % 12;
        // Can giờ: công thức Ngũ Thử Độn — (dayGanIndexExact % 5 × 2 + zhiIdx) % 10
        const dayGanIdxExact = arrGanZH.indexOf(dayGanHan);
        const astTimeGanIdx  = (dayGanIdxExact % 5 * 2 + astTimeZhiIdx) % 10;
        const timeZhiHan     = arrZhiZH[astTimeZhiIdx];
        const timeGanHan     = arrGanZH[astTimeGanIdx];
        // Ghi vào bazi để Tứ Trụ panel và getTime() dùng đúng giá trị thiên văn
        bazi._astTimeGan = timeGanHan;
        bazi._astTimeZhi = timeZhiHan;
        bazi._astTime    = timeGanHan + timeZhiHan;

        const lunarMonthNum = Math.abs(lunar.getMonth());
        const monthGanHan   = baziBJ.getMonthGan();
        const monthZhiHan   = baziBJ.getMonthZhi();

        // ── 6. Tứ Trụ panel ──
        updateTuTru(bazi.getYear(), bazi.getMonth(), bazi.getDay(), bazi.getTime());
        getDOM('ttValNam').textContent   = y;
        getDOM('ttValThang').textContent = m;
        getDOM('ttValNgay').textContent  = d;
        getDOM('ttValGio').textContent   = `${pad(h)}:${pad(min)}`;

        // ── 7. Tiết Khí ──
        const { jqObj: jieQiObj, jqSolar } = _getPrevJieQi(solarBJ, dUTC);
        const jieQiZH = jieQiObj.getName();

        // OPT-3-FULL: thiết lập múi giờ toàn cục cho ShouXingUtil SAU khi đã
        // tính xong lunarLocal/baziBJ/jieQiZH (tất cả ở mốc UTC+8) — đảm bảo
        // bảng tiết khí (sách bổ) và bảng sóc (âm bàn) dùng múi giờ địa phương
        // thực, không còn cố định UTC+8.
        ShouXingUtil.setTzOffsetHours(tz);

        // ── 8. Số Cục + Âm/Dương Độn ──
        const inputJDFrac = tn_dateToJD(y, m, d) + h / 24 + min / 1440;
        const dayGanIdx   = arrGanZH.indexOf(dayGanHan);
        const dayZhiIdx   = arrZhiZH.indexOf(dayZhiHan);

        const trinhuanResult = method === 'trinhuan'
            ? tn_getSoCuc(y, m, d, h, min, lon, tz, info.tzId, dayGanIdx, dayZhiIdx)
            : null;

        const { cuc, don } = calculateCucDon({
            method, jieQiZH, dayGanHan, dayZhiHan,
            yearZhiHan, lunarMonthNum, lunarDay: lunar.getDay(), timeZhiHan,
            trinhuanResult
        });

        // ── 9. Tuần Thủ, Không Vong, Dịch Mã ──
        const canGioRaw   = mapToVi[timeGanHan];
        const canChiGio   = `${canGioRaw} ${mapToVi[timeZhiHan]}`;
        const canTuan     = dayToTuanThu[canChiGio];
        const canChiNam   = `${mapToVi[yearGanHan]} ${mapToVi[yearZhiHan]}`;
        const canChiThang = `${mapToVi[monthGanHan]} ${mapToVi[monthZhiHan]}`;
        const canChiNgay  = `${mapToVi[dayGanHan]} ${mapToVi[dayZhiHan]}`;
        const kvStr       = hoaGiapToKhongVong[canChiGio];
        const dmChi       = hoaGiapToDichMa[canChiGio];

        const isZH    = currentLang === 'zh';
        const isCoban = getDOM('mainBody').classList.contains('coban-mode');

        const tuanGiapVi     = thuToTuanGiap[canTuan];
        const displayTuanThu = isZH
            ? `${arrGanZH[0]}${chiMapping[tuanGiapVi.split(' ')[1]]} (${arrGanZH[dataBase.vi.canFull.indexOf(canTuan)]})`
            : `${tuanGiapVi} (${canTuan})`;

        // ── 10. Tính bàn KMDG ──
        const res = calculateQMDJ(cuc, don, canTuan, canGioRaw, currentLang);

        // ── 11. Info Panel ──
        // Lunar.fromYmd(...) phải chạy ở CÙNG mốc múi giờ với lunarMonthNum
        // (nay là mốc địa phương), nếu không nó tra ra mùng 1 của tháng âm
        // theo một mốc khác và giờ Sóc hiện ra lệch hẳn một ngày.
        // socSolar chỉ là ngày đã làm tròn; formatPreciseSocLocal() tính lại
        // thời điểm Sóc chính xác đến phút rồi quy sang giờ địa phương.
        const socSolar = Ephem.atBasis(tz, () =>
            Lunar.fromYmd(lunar.getYear(), lunarMonthNum, 1).getSolar());

        // FIX (DST): formatPreciseSocLocal/formatUTC8SolarToLocal nay nhận
        // tzId (IANA) và tự xác định offset DST đúng tại CHÍNH thời điểm
        // UTC của Sóc/Tiết khí (không còn tính offset từ "ngày UTC+8 đã làm
        // tròn", có thể sai 1h nếu rơi đúng ngày chuyển giờ DST).
        const socStr    = formatPreciseSocLocal(socSolar, info.tzId);
        // jqSolar (từ _getPrevJieQi) nay được tính ở mốc UTC+8 cố định (xem
        // FIX ở bước 7) — phải quy đổi sang giờ địa phương bằng JD shift,
        // KHÔNG dùng convertBeijingToLocal() (hàm đó dành cho dữ liệu của
        // sb_getJieQiDates(), vốn đã ở mốc local theo _tzOffsetHours=tz).
        const nhapTiet  = formatUTC8SolarToLocal(jqSolar, info.tzId);

        const tietKhiName = isZH ? jieQiZH : tietKhiMap[jieQiZH];
        const nhuanTxt    = (lunar.getMonth() < 0 && !isZH) ? ' (Nhuận)' : '';
        const lunarStr = isZH
            ? `${pad(lunar.getDay())} - ${pad(lunarMonthNum)} - ${lunar.getYear()}`
            : `${pad(lunar.getDay())} - ${pad(lunarMonthNum)}${nhuanTxt} - ${lunar.getYear()}`;

        // Ngày vía (lịch Tây Tạng) — chỉ tính ngày trong tháng, không hiển thị
        // ngày/tháng/năm lịch Tây Tạng, chỉ hiển thị tên ngày vía nếu trúng ngày.
        const tib = new TibetanCalendar().fromGregorian(y, m, d);
        const viaInfo = VIA_DAYS[tib.day] || null;
        const viaStr  = viaInfo ? (isZH ? viaInfo.zh : viaInfo.vi) : '';
        const donTxt    = isZH ? (don === 'duong' ? '阳' : '阴') : (don === 'duong' ? 'Dương' : 'Âm');
        const methodTxt = method === 'amban'    ? uiDict[currentLang].methodAmBan
                        : method === 'bophap'   ? uiDict[currentLang].methodBoPhap
                        : uiDict[currentLang].methodTriNhuan;

        const kvFmt = kv => {
            const v = formatKVdisp(kv, currentLang);
            if (!v || v === '-') return '-';
            return isZH
                ? (v.length >= 2 ? `${v[0]},${v.slice(1)}` : v)
                : v.replace(' ', ', ');
        };

        // Batch DOM updates — info panel
        const infoUpdates = {
            'out-chinhngo': `${chinhNgoStr} (${gmtStr})`,
            'out-lunar-table': lunarStr,
            'out-via':      viaStr,
            'out-tietkhi':  `${tietKhiName} ${nhapTiet}`,
            'out-tuan':     displayTuanThu,
        };
        for (const [id, val] of Object.entries(infoUpdates)) { const el = getDOM(id); if (el) el.innerText = val; }
        getDOM('out-cuc').innerHTML = `${donTxt} ${cuc} (${methodTxt})`;
        getDOM('out-tp').innerText  = isZH ? `天${dataBase.zh.tinh[res.idxTP]}` : `Thiên ${dataBase.vi.tinh[res.idxTP]}`;
        getDOM('out-ts').innerText  = isZH ? `${dataBase.zh.mon[res.idxTS]}门`  : `${dataBase.vi.mon[res.idxTS]} môn`;

        // Nạp Âm Ngũ Hành cho Tứ Trụ
        [['ttKVNam', canChiNam], ['ttKVThang', canChiThang], ['ttKVNgay', canChiNgay], ['ttKVGio', canChiGio]]
            .forEach(([id, chi]) => {
                const viName = canChiToNapAm[chi] || '-';
                getDOM(id).textContent = (isZH && viName !== '-') ? (napAmToZH[viName] || viName) : viName;
            });

        getDOM('baziInfoPanel').style.display = 'block';
        getDOM('viaWrap').style.display = viaInfo ? 'inline' : 'none';

        // ── Detail panels (Trí Nhuận / Sách Bổ / Âm Bàn) ──
        const notZH = !isZH;
        const panelCfg = [
            { id: 'trinhuanPanel', ...(_panelIds.trinhuan), active: method === 'trinhuan' && notZH },
            { id: 'sachboPanel',   ...(_panelIds.sachbo),   active: method === 'bophap'   && notZH },
            { id: 'ambanPanel',    ...(_panelIds.amban),    active: method === 'amban'    && notZH },
        ];
        for (const cfg of panelCfg) {
            const panel = getDOM(cfg.id);
            if (!panel) continue;
            panel.style.display = cfg.active ? 'block' : 'none';
            if (cfg.active) {
                const body = getDOM(cfg.bodyId), chev = getDOM(cfg.chevId);
                if (body && body.style.display === 'none' && chev) chev.style.transform = 'rotate(0deg)';
            }
        }
        if (method === 'trinhuan' && notZH) tn_renderPanel(y, lon, tz, info.tzId, inputJDFrac, dayGanIdx, dayZhiIdx, trinhuanResult);
        if (method === 'bophap'   && notZH) sb_renderPanel(jieQiZH, y, m, d, h, min, tz, info.tzId);
        if (method === 'amban'    && notZH) ab_renderPanel(lunar.getYear(), lunar.getMonth(), info.tzId, lon, tz);

        // ── 12. Vẽ bàn KMDG ──
        const canGioSoSanh   = mapToVi[timeGanHan]  === 'Giáp' ? canTuan                   : mapToVi[timeGanHan];
        const canNamSoSanh   = mapToVi[yearGanHan]  === 'Giáp' ? dayToTuanThu[canChiNam]   : mapToVi[yearGanHan];
        const canNgaySoSanh  = mapToVi[dayGanHan]   === 'Giáp' ? dayToTuanThu[canChiNgay]  : mapToVi[dayGanHan];
        const canThangSoSanh = mapToVi[monthGanHan] === 'Giáp' ? dayToTuanThu[canChiThang] : mapToVi[monthGanHan];
        const hlCanThien     = canGioRaw === 'Giáp' ? canTuan : canGioRaw;

        const kvLtSet   = new Set(kvStr ? kvStr.split(' ').map(chi => chiToLT[chi]) : []);
        const dmLT      = chiToLT[dmChi];
        const noiBanSet = new Set(don === 'duong' ? [1, 3, 4, 8] : [2, 6, 7, 9]);
        const dmTxtStr  = isZH ? '马' : 'Mã';
        const truPhuTinh = dataBase[currentLang].tinh[res.idxTP];
        const truSuMon   = dataBase[currentLang].mon[res.idxTS];

        const layout          = [4, 9, 2, 3, 5, 7, 8, 1, 6];
        const cells           = [];
        const backCellContents = [];

        for (const lt of layout) {
            if (lt === 5) {
                cells.push(`<div class="cell trung-cung">
                    ${kvLtSet.has(5) ? '<div class="kv-mark-solo"></div>' : ''}
                    ${renderCanPair(res.diaBan[5], null, 'pos-can-dia')}
                    ${renderCanPair(res.anCan[5],  null, 'pos-can-an')}
                </div>`);
                backCellContents.push({ html: '', noiBanClass: '' });
            } else {
                const { cellHtml, backContent } = _buildCell(
                    lt, res, isCoban, isZH,
                    noiBanSet, kvLtSet, dmLT, dmTxtStr,
                    canGioSoSanh, canNgaySoSanh, canThangSoSanh, canNamSoSanh,
                    hlCanThien, truPhuTinh, truSuMon
                );
                cells.push(cellHtml);
                backCellContents.push(backContent);
            }
        }

        const board = getDOM('board');
        board.innerHTML = cells.join('');
        board.style.display = 'grid';

        // ── Card flip 3D ──
        let flipInner = getDOM('boardFlipInner');
        if (!flipInner) {
            const scene = document.createElement('div');
            scene.id = 'boardFlipScene';

            flipInner = document.createElement('div');
            flipInner.id = 'boardFlipInner';
            DOM['boardFlipInner'] = flipInner;

            const back = document.createElement('div');
            back.id = 'boardBack';
            DOM['boardBack'] = back;
            for (let i = 0; i < 9; i++) {
                const c = document.createElement('div');
                c.className = 'cell';
                back.appendChild(c);
            }

            board.parentNode.insertBefore(scene, board);
            flipInner.appendChild(board);
            flipInner.appendChild(back);
            scene.appendChild(flipInner);

            let animating = false;
            function doFlip() {
                if (animating) return;
                animating = true;
                flipInner.classList.toggle('is-flipped');
                setTimeout(() => { animating = false; }, 680);
            }
            board.addEventListener('click', e => { if (e.target.closest('.cell')) doFlip(); });
            back.addEventListener('click',  () => doFlip());
        } else {
            flipInner.classList.remove('is-flipped');
            flipInner.style.transition = 'none';
            requestAnimationFrame(() => requestAnimationFrame(() => {
                flipInner.style.transition = '';
            }));
        }

        const boardBack = getDOM('boardBack');
        if (boardBack) {
            const backCellEls = boardBack.children;
            for (let i = 0; i < backCellContents.length && i < backCellEls.length; i++) {
                const { html, noiBanClass } = backCellContents[i];
                backCellEls[i].innerHTML = html;
                backCellEls[i].classList.toggle('noi-ban', noiBanClass === 'noi-ban');
            }
        }

    } catch (e) {
        console.error(e);
        alert('Đã xảy ra lỗi lập quẻ. Vui lòng tải lại trang.');
    }
}

/* ══════════════════════════════════════════════
   DRUM DATETIME PICKER  (5 columns)
══════════════════════════════════════════════ */
(function() {
    const IH = 40, PAD = 2 * IH;
    const tmp = { day: 0, month: 0, year: 0, hour: 0, minute: 0 };
    const COLS   = ['day', 'month', 'year', 'hour', 'minute'];
    const COUNTS = { day: 31, month: 12, year: 201, hour: 24, minute: 60 };

    function buildItems(col) {
        const items = [];
        for (let i = 0; i < COUNTS[col]; i++) {
            let label;
            if      (col === 'year')                    label = 1900 + i;
            else if (col === 'hour' || col === 'minute') label = String(i).padStart(2, '0');
            else                                         label = i + 1;
            items.push(`<div class="drum-item">${label}</div>`);
        }
        return items.join('');
    }
    function setY(col, y, smooth) {
        const list = getDOM('drumList_' + col);
        if (!list) return;
        list.style.transition = smooth ? 'transform 0.32s cubic-bezier(0.25,0.46,0.45,0.94)' : 'none';
        list.style.transform  = `translateY(${y}px)`;
    }
    function highlight(col) {
        const list = getDOM('drumList_' + col);
        if (!list) return;
        list.querySelectorAll('.drum-item').forEach((el,i) => el.classList.toggle('sel', i === tmp[col]));
    }
    function jumpTo(col, idx) {
        idx = Math.max(0, Math.min(COUNTS[col]-1, idx));
        tmp[col] = idx;
        setY(col, PAD - idx * IH, false);
        highlight(col);
    }
    function snapTo(col, y, velocity) {
        let idx = Math.round((PAD - (y + velocity * 80)) / IH);
        idx = Math.max(0, Math.min(COUNTS[col]-1, idx));
        tmp[col] = idx;
        setY(col, PAD - idx * IH, true);
        setTimeout(() => highlight(col), 340);
        highlight(col);
    }

    let drag = null;
    function currentY(col) {
        const list = getDOM('drumList_' + col);
        if (!list) return PAD;
        return new DOMMatrix(getComputedStyle(list).transform).m42;
    }
    function onStart(col, clientY) {
        const list = getDOM('drumList_' + col);
        if (list) list.style.transition = 'none';
        drag = { col, startY: clientY, translateY: currentY(col), lastY: clientY, lastT: Date.now(), velocity: 0 };
    }
    function onMove(clientY) {
        if (!drag) return;
        const now = Date.now(), dt = now - drag.lastT || 1;
        drag.velocity = (clientY - drag.lastY) / dt;
        drag.lastY = clientY; drag.lastT = now;
        let newY = drag.translateY + (clientY - drag.startY);
        const minY = PAD - (COUNTS[drag.col]-1)*IH, maxY = PAD;
        if (newY < minY) newY = minY - (minY-newY)*0.25;
        if (newY > maxY) newY = maxY + (newY-maxY)*0.25;
        setY(drag.col, newY, false);
        const li = Math.max(0, Math.min(COUNTS[drag.col]-1, Math.round((PAD-newY)/IH)));
        if (li !== tmp[drag.col]) { tmp[drag.col] = li; highlight(drag.col); }
    }
    function onEnd() {
        if (!drag) return;
        const { col, velocity } = drag; drag = null;
        snapTo(col, currentY(col), velocity);
    }
    function attachCol(col) {
        const el = getDOM('drumCol_' + col);
        if (!el) return;
        el.addEventListener('touchstart', e => onStart(col, e.touches[0].clientY), { passive: true });
        el.addEventListener('touchmove',  e => { e.preventDefault(); onMove(e.touches[0].clientY); }, { passive: false });
        el.addEventListener('touchend',   () => onEnd());
        el.addEventListener('mousedown',  e => { e.preventDefault(); onStart(col, e.clientY); });
        el.addEventListener('wheel', e => {
            e.preventDefault();
            const ni = Math.max(0, Math.min(COUNTS[col]-1, tmp[col]+(e.deltaY>0?1:-1)));
            jumpTo(col, ni); setY(col, PAD-ni*IH, true);
            setTimeout(() => highlight(col), 340);
        }, { passive: false });
    }
    document.addEventListener('mousemove', e => { if (drag) onMove(e.clientY); });
    document.addEventListener('mouseup',   () => { if (drag) onEnd(); });

    window.updateDateDisplay = function() {
        const d   = parseInt(getDOM('inDay').value)     || 1;
        const m   = parseInt(getDOM('inMonth').value)   || 1;
        const y   = parseInt(getDOM('inYear').value)    || 2026;
        const h   = parseInt(getDOM('solarHour').value) || 0;
        const min = parseInt(getDOM('solarMinute').value)|| 0;
        getDOM('dateDisplayText').textContent = `${pad(d)}-${pad(m)}-${y} ${pad(h)}:${pad(min)}`;
    };

    window.openDatePicker = function() {
        jumpTo('day',    (parseInt(getDOM('inDay').value)    ||1) - 1);
        jumpTo('month',  (parseInt(getDOM('inMonth').value)  ||1) - 1);
        jumpTo('year',   (parseInt(getDOM('inYear').value)   ||2026) - 1900);
        jumpTo('hour',    parseInt(getDOM('solarHour').value)   || 0);
        jumpTo('minute',  parseInt(getDOM('solarMinute').value) || 0);
        getDOM('drumOverlay').classList.add('open');
    };

    function closePicker(confirm) {
        getDOM('drumOverlay').classList.remove('open');
        if (confirm) {
            getDOM('inDay').value      = tmp.day    + 1;
            getDOM('inMonth').value    = tmp.month  + 1;
            getDOM('inYear').value     = 1900 + tmp.year;
            getDOM('solarHour').value  = tmp.hour;
            getDOM('solarMinute').value= tmp.minute;
            updateDateDisplay();
            if (typeof Solar !== 'undefined') processAll();
        }
    }

    document.addEventListener('DOMContentLoaded', function() {
        COLS.forEach(col => {
            getDOM('drumList_' + col).innerHTML = buildItems(col);
            attachCol(col);
        });
        getDOM('drumOkBtn')    .addEventListener('click', () => closePicker(true));
        getDOM('drumCancelBtn').addEventListener('click', () => closePicker(false));
        getDOM('drumOverlay')  .addEventListener('click', function(e) { if(e.target===this) closePicker(false); });

        const _orig = window.toggleLang;
        window.toggleLang = function() {
            _orig();
            const zh = getDOM('mainBody').classList.contains('lang-zh');
            const labels = {
                drumCancelBtn: zh ? '取消' : 'Hủy',
                drumOkBtn:     zh ? '选择' : 'Chọn',
                drumTitleLbl:  zh ? '时间' : 'Thời gian',
                lblDrumDay:    zh ? '日'   : 'Ngày',
                lblDrumMonth:  zh ? '月'   : 'Tháng',
                lblDrumYear:   zh ? '年'   : 'Năm',
                lblDrumHour:   zh ? '时'   : 'Giờ',
                lblDrumMinute: zh ? '分'   : 'Phút',
                ctryCancelBtn: zh ? '取消' : 'Hủy',
                ctryTitleLbl:  zh ? '国家' : 'Quốc gia',
                ctryOkBtn:     zh ? '选择' : 'Chọn',
            };
            for (const [id, text] of Object.entries(labels)) { const el = getDOM(id); if (el) el.textContent = text; }
            if (typeof updateCountryDisplay === 'function') updateCountryDisplay();
        };
    });
})();
