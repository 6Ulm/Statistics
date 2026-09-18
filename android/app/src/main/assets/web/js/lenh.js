/* ════════════════════════════════════════════════════════════════════
   lenh.js — tab "Lệnh": nhân nguyên tư lệnh (人元司令分野)

   Mỗi tháng khí (từ TIẾT này tới TIẾT sau) không do một can duy nhất nắm.
   Chi của tháng tàng 2–3 can, và chúng thay nhau "cầm lệnh" theo thứ tự
   dư khí → trung khí → bản khí. Bảng này nói: tại thời điểm đang xem, can
   nào đang cầm lệnh.

   RANH GIỚI ĐO BẰNG ĐỘ HOÀNG KINH, KHÔNG PHẢI SỐ NGÀY.

   Sách xưa chép phần chia theo NGÀY ("Mậu 7 ngày, Bính 7 ngày, Giáp 16
   ngày"), vì một ngày Mặt Trời đi xấp xỉ một độ. Nhưng "xấp xỉ" ấy lệch tới
   3,4%: quanh cận nhật (tháng Giêng) Mặt Trời đi 1,019°/ngày, quanh viễn
   nhật (tháng Bảy) chỉ 0,953°/ngày. Đếm ngày thì tổng 30 ngày của một tháng
   khí không khớp khoảng cách thật giữa hai tiết (29,44 ngày mùa đông,
   31,44 ngày mùa hè) — cộng dồn ba đoạn là lệch cả ngày rưỡi, và đoạn cuối
   không rơi đúng vào tiết sau.

   Đo bằng độ thì hết hẳn: 7° + 7° + 16° = 30° = đúng khoảng cách hai tiết,
   theo định nghĩa. Ảnh mẫu người dùng gửi cũng ghi cột "Hoàng kinh" chứ không
   ghi số ngày. (Cột ấy nay hiện SỐ ĐỘ của từng đoạn — 7 · 7 · 16 — chứ không
   hiện khoảng 285~294°: bộ số mới là thứ phân biệt ba sách bên dưới.)

   MỘT NGUỒN DUY NHẤT VỚI BẢNG TIẾT KHÍ.

   Mốc mở tháng (bội số của 15°) lấy THẲNG từ ShouXingUtil.qiAccurate — đúng
   hàm mà bảng tiết khí ở tab Lịch và bảng Sách Bổ ở tab Kỳ Môn đang dùng,
   nên nó tra bảng DE423 y hệt. Không có chuyện "Lập Xuân" ở tab Lịch một
   giờ mà "vào lệnh Mậu" ở tab này một giờ khác.

   Mốc giữa tháng (7°, 22°… không phải bội số của 15°) thì bảng DE423 không
   có — nó chỉ lưu 24 tiết khí. Chỗ ấy giải bằng chuỗi giải tích saLonT của
   chính lunar.js. Hai nguồn lệch nhau ≤ 2 giây tại các mốc dùng chung (đo
   trên cả năm 2026), tức không bao giờ đủ để đổi con số phút hiện ra.

   BA BỘ SỐ, BA CUỐN SÁCH — chọn bằng ô bên cạnh ô ngày giờ.

   Cổ thư không thống nhất phần chia này, và chênh nhau không phải vài phút
   mà là cả tuần. Tháng Dần: 渊海 chia 7·7·16 (Mậu tới ngày 7, Bính tới ngày
   14, rồi Giáp), 三命通会 chia 5·5·20 (Mậu tới ngày 5, Bính tới ngày 10, rồi
   Giáp) — hai mốc đổi can lệch nhau 2 và 4 ngày, nên có những ngày CÙNG MỘT
   người ra hai can khác nhau. Đo trên chính ứng dụng, Hà Nội, Lập Xuân 2026
   rơi 04/02 03:02:

     sinh 10/02 (≈6 ngày sau):  渊海 → Mậu,  三命通会 → Bính
     sinh 16/02 (≈12 ngày sau): 渊海 → Bính, 三命通会 → Giáp

   Không có cách nào "trung hoà" ba bộ số ấy thành một; chỉ có cách nói rõ
   đang dùng bộ nào.

   Xuất xứ từng bộ (xem RULES bên dưới, mỗi bộ kèm nguyên văn chữ Hán):

     • 三命通会 và 子平真诠 chép THẲNG từ nguyên văn, đối chiếu nhiều bản độc
       lập; cả 12 tháng của cả hai bộ đều cộng đúng 30.
     • 渊海子平 là BẢN THÔNG HÀNH mà giới mệnh lý ngày nay quy cho hệ Uyên
       Hải — cũng đúng bộ số trong ảnh mẫu người dùng gửi. Bản 渊海子平 tìm
       được chỉ có bài "论天地干支暗藏总诀" chia theo nửa tháng, không phải
       bảng ba đoạn này, nên chỗ quy cho ấy là theo tập quán chứ không phải
       một dòng đọc được trong sách.
   ════════════════════════════════════════════════════════════════════ */
(function () {
    'use strict';

    var CAN_VI = ['Giáp', 'Ất', 'Bính', 'Đinh', 'Mậu', 'Kỷ', 'Canh', 'Tân', 'Nhâm', 'Quý'];
    var CHI_VI = ['Tý', 'Sửu', 'Dần', 'Mão', 'Thìn', 'Tỵ', 'Ngọ', 'Mùi', 'Thân', 'Dậu', 'Tuất', 'Hợi'];
    var CAN_ZH = ['甲', '乙', '丙', '丁', '戊', '己', '庚', '辛', '壬', '癸'];
    var CHI_ZH = ['子', '丑', '寅', '卯', '辰', '巳', '午', '未', '申', '酉', '戌', '亥'];

    var T = {
        tabLenh:  { vi: 'Lệnh',        zh: '令' },
        title:    { vi: 'LỆNH NĂM',    zh: '司令' },
        now:      { vi: 'Lệnh',        zh: '司令' },
        colMonth: { vi: 'Tháng',       zh: '月' },
        colCan:   { vi: 'Can',         zh: '天干' },
        colLon:   { vi: 'Số độ',       zh: '度数' },
        colIn:    { vi: 'Vào lệnh',    zh: '入令' },
        colOut:   { vi: 'Hết lệnh',    zh: '退令' },
        pickRule: { vi: 'Quy tắc', zh: '选择流派' },
        daiVan:   { vi: 'Nhập vận',    zh: '起运' },
        pickGender: { vi: 'Giới tính', zh: '性别' },
        daiVanPillar: { vi: 'Đại Vận', zh: '大运' },
    };
    function isZH() { return typeof currentLang !== 'undefined' && currentLang === 'zh'; }
    function t(k) { return T[k][isZH() ? 'zh' : 'vi']; }
    function canName(i) { return isZH() ? CAN_ZH[i] : CAN_VI[i]; }
    function chiName(i) { return isZH() ? CHI_ZH[i] : CHI_VI[i]; }
    function jieName(k) {
        if (isZH()) return (typeof TK_ZH !== 'undefined') ? TK_ZH[k] : '';
        return (typeof TK_VI !== 'undefined') ? TK_VI[k] : '';
    }

    function esc(s) {
        return String(s).replace(/[&<>"]/g, function (c) {
            return { '&': '&amp;', '<': '&lt;', '>': '&gt;', '"': '&quot;' }[c];
        });
    }
    function pad2(n) { return (n < 10 ? '0' : '') + n; }

    /* ─────────────── Ba bảng phân dã ───────────────
     *
     * Khoá = chỉ số TIẾT mở tháng trong TK_VI/TK_ZH (0 = Đông Chí), luôn LẺ:
     * tháng khí mở bằng TIẾT (Tiểu Hàn, Lập Xuân…), còn KHÍ (Đại Hàn, Vũ
     * Thủy…) rơi vào giữa tháng và không phải ranh giới của gì cả.
     *
     * Giá trị = [chỉ số can, số độ] theo đúng thứ tự cầm lệnh. Tổng của mỗi
     * tháng phải bằng 30 — buildYear() kiểm lại và ném lỗi nếu không, vì gõ
     * nhầm một số ở đây thì mọi mốc phía sau trong tháng ấy trôi theo mà bảng
     * vẫn trông bình thường.
     *
     * Cổ thư chép bằng NGÀY ("Mậu 7 ngày"); ở đây đọc là ĐỘ hoàng kinh — xem
     * khối ghi chú đầu tệp về việc vì sao ngày không cộng ra 30.
     */

    /** Bản thông hành (hệ Uyên Hải): 寅 7·7·16, 巳 5·9·16, 申 7·7·16… */
    var FEN_YHZP = {
        1:  [[9, 9], [7, 3], [5, 18]],    // Sửu  (Tiểu Hàn):  Quý 9 · Tân 3 · Kỷ 18
        3:  [[4, 7], [2, 7], [0, 16]],    // Dần  (Lập Xuân):  Mậu 7 · Bính 7 · Giáp 16
        5:  [[0, 10], [1, 20]],           // Mão  (Kinh Trập): Giáp 10 · Ất 20
        7:  [[1, 9], [9, 3], [4, 18]],    // Thìn (Thanh Minh):Ất 9 · Quý 3 · Mậu 18
        9:  [[4, 5], [6, 9], [2, 16]],    // Tỵ   (Lập Hạ):    Mậu 5 · Canh 9 · Bính 16
        11: [[2, 10], [5, 9], [3, 11]],   // Ngọ  (Mang Chủng):Bính 10 · Kỷ 9 · Đinh 11
        13: [[3, 9], [1, 3], [5, 18]],    // Mùi  (Tiểu Thử):  Đinh 9 · Ất 3 · Kỷ 18
        15: [[4, 7], [8, 7], [6, 16]],    // Thân (Lập Thu):   Mậu 7 · Nhâm 7 · Canh 16
        17: [[6, 10], [7, 20]],           // Dậu  (Bạch Lộ):   Canh 10 · Tân 20
        19: [[7, 9], [3, 3], [4, 18]],    // Tuất (Hàn Lộ):    Tân 9 · Đinh 3 · Mậu 18
        21: [[4, 7], [0, 5], [8, 18]],    // Hợi  (Lập Đông):  Mậu 7 · Giáp 5 · Nhâm 18
        23: [[8, 10], [9, 20]],           // Tý   (Đại Tuyết): Nhâm 10 · Quý 20
    };

    /**
     * 三命通会 卷二「论人元司事」, nguyên văn:
     *
     *   如正月建寅，寅中有艮土用事五日，丙火长生五日，甲木二十日；二月建卯，卯中
     *   有甲木用事七日，乙木二十三日；三月建辰，辰中有乙木用事七日，壬水墓库五日，
     *   戊土一十八日；四月建巳，巳中有戊土七日，庚金长生五日，丙火一十八日；五月
     *   建午，午中丙火用事七日，丁火二十三日；六月建未，未中有丁火用事七日，甲木
     *   墓库五日，己土一十八日；七月建申，申中有坤土用事五日，壬水长生五日，庚金
     *   二十日；八月建酉，酉中有庚金用事七日，辛金二十三日；九月建戌，戌中有辛金
     *   用事七日，丙火墓库五日，戊土一十八日；十月建亥，亥中有戊土五日，甲木长生
     *   五日，壬水用事二十日；十一月建子，子中有壬水用事七日，癸水二十三日；十二
     *   月建丑，丑中有癸水用事七日，庚金墓库五日，己土一十八日。
     *
     * "艮土" và "坤土" đều là MẬU (hai quẻ ấy thuộc thổ, và chi Dần/Thân tàng
     * Mậu chứ không tàng Kỷ). Nét riêng dễ nhận của bộ này: bốn tháng mộ khố
     * (Thìn Mùi Tuất Sửu) lấy can DƯƠNG làm trung khí — Thìn dùng Nhâm chứ
     * không Quý, Mùi dùng Giáp chứ không Ất — ngược hẳn hai bộ kia.
     */
    var FEN_SMTH = {
        1:  [[9, 7], [6, 5], [5, 18]],    // Sửu:  Quý 7 · Canh 5 · Kỷ 18
        3:  [[4, 5], [2, 5], [0, 20]],    // Dần:  Mậu 5 · Bính 5 · Giáp 20
        5:  [[0, 7], [1, 23]],            // Mão:  Giáp 7 · Ất 23
        7:  [[1, 7], [8, 5], [4, 18]],    // Thìn: Ất 7 · Nhâm 5 · Mậu 18
        9:  [[4, 7], [6, 5], [2, 18]],    // Tỵ:   Mậu 7 · Canh 5 · Bính 18
        11: [[2, 7], [3, 23]],            // Ngọ:  Bính 7 · Đinh 23
        13: [[3, 7], [0, 5], [5, 18]],    // Mùi:  Đinh 7 · Giáp 5 · Kỷ 18
        15: [[4, 5], [8, 5], [6, 20]],    // Thân: Mậu 5 · Nhâm 5 · Canh 20
        17: [[6, 7], [7, 23]],            // Dậu:  Canh 7 · Tân 23
        19: [[7, 7], [2, 5], [4, 18]],    // Tuất: Tân 7 · Bính 5 · Mậu 18
        21: [[4, 5], [0, 5], [8, 20]],    // Hợi:  Mậu 5 · Giáp 5 · Nhâm 20
        23: [[8, 7], [9, 23]],            // Tý:   Nhâm 7 · Quý 23
    };

    /**
     * 子平真诠评注, bảng 「十二月令人元司令分野表」, nguyên văn:
     *
     *   寅月立春后戊土七日，丙火七日，甲木十六日
     *   卯月惊蛰后甲木十日，乙木二十日
     *   辰月清明后乙木九日，癸水三日，戊土十八日
     *   巳月立夏后戊土五日，庚金九日，丙火十六日
     *   午月芒种后丙火十日，己土九日，丁火十一日
     *   未月小暑后丁火九日，乙木三日，己土十八日
     *   申月立秋后戊己土十日，壬水三日，庚金十七日
     *   酉月白露后庚金十日，辛金二十日
     *   戌月寒露后辛金九日，丁火三日，戊土十八日
     *   亥月立冬后戊土七日，甲木五日，壬水十八日
     *   子月大雪后壬水十日，癸水二十日
     *   丑月小寒后癸水九日，辛金三日，己土十八日
     *
     * Chỉ khác bản thông hành ở ĐÚNG MỘT tháng — tháng Thân: 10·3·17 thay vì
     * 7·7·16. ("戊己土" gộp làm một đoạn; lấy MẬU vì chi Thân tàng Mậu.)
     */
    var FEN_ZPZQ = {
        1:  [[9, 9], [7, 3], [5, 18]],    // Sửu:  Quý 9 · Tân 3 · Kỷ 18
        3:  [[4, 7], [2, 7], [0, 16]],    // Dần:  Mậu 7 · Bính 7 · Giáp 16
        5:  [[0, 10], [1, 20]],           // Mão:  Giáp 10 · Ất 20
        7:  [[1, 9], [9, 3], [4, 18]],    // Thìn: Ất 9 · Quý 3 · Mậu 18
        9:  [[4, 5], [6, 9], [2, 16]],    // Tỵ:   Mậu 5 · Canh 9 · Bính 16
        11: [[2, 10], [5, 9], [3, 11]],   // Ngọ:  Bính 10 · Kỷ 9 · Đinh 11
        13: [[3, 9], [1, 3], [5, 18]],    // Mùi:  Đinh 9 · Ất 3 · Kỷ 18
        15: [[4, 10], [8, 3], [6, 17]],   // Thân: Mậu 10 · Nhâm 3 · Canh 17
        17: [[6, 10], [7, 20]],           // Dậu:  Canh 10 · Tân 20
        19: [[7, 9], [3, 3], [4, 18]],    // Tuất: Tân 9 · Đinh 3 · Mậu 18
        21: [[4, 7], [0, 5], [8, 18]],    // Hợi:  Mậu 7 · Giáp 5 · Nhâm 18
        23: [[8, 10], [9, 20]],           // Tý:   Nhâm 10 · Quý 20
    };

    /** Thứ tự trong ô chọn: theo niên đại sách (Tống → Minh → Thanh). */
    // Tên tiếng Việt là CHỮ ĐẦU viết tắt (UHTB/TMTH/TBCT), không phải tên
    // đầy đủ — ô hiện có 25% bề ngang hàng (một phần tư bảng Bát Tự), tên đầy
    // đủ dài nhất "Tam Mệnh Thông Hội" không lọt nổi ở cỡ chữ đọc được trên
    // máy 360px. Tên đầy đủ vẫn còn NGUYÊN VĂN trong README và trong khối ghi
    // chú xuất xứ ở đầu tệp — chữ viết tắt chỉ là CÁCH HIỆN, không đổi việc gì
    // khác. Tiếng Trung không rút gọn: 渊海子平/三命通会/子平真诠 vốn đã ngắn.
    var RULES = [
        { key: 'yhzp', fen: FEN_YHZP, vi: 'UHTB', zh: '渊海子平' },
        { key: 'smth', fen: FEN_SMTH, vi: 'TMTH', zh: '三命通会' },
        { key: 'zpzq', fen: FEN_ZPZQ, vi: 'TBCT', zh: '子平真诠' },
    ];
    var K_RULE = 'qmdj.lenhRule';
    var rule = RULES[0];
    function ruleByKey(k) {
        for (var i = 0; i < RULES.length; i++) if (RULES[i].key === k) return RULES[i];
        return RULES[0];
    }
    function ruleLabel(r) { return isZH() ? r.zh : r.vi; }

    /* ─────────────── Ô Giới tính ───────────────
     * Quyết CHIỀU đại vận (thuận/nghịch) khi chéo với âm dương của can năm —
     * xem isThuanHanh() ngay dưới hàm daiVanTuoi(). Không đụng gì tới bảng
     * Lệnh của cả năm (bảng ấy chỉ phụ thuộc bộ số ở RULES), chỉ đụng dòng
     * "Nhập vận: …" phía trên bảng. */
    var GENDERS = [
        { key: 'nam', vi: 'Nam', zh: '男' },
        { key: 'nu',  vi: 'Nữ',  zh: '女' },
    ];
    var K_GENDER = 'qmdj.lenhGender';
    var gender = GENDERS[0];
    function genderByKey(k) {
        for (var i = 0; i < GENDERS.length; i++) if (GENDERS[i].key === k) return GENDERS[i];
        return GENDERS[0];
    }
    function genderLabel(g) { return isZH() ? g.zh : g.vi; }

    /** Chi của tháng mà mỗi TIẾT mở ra. */
    var CHI_OF_JIE = { 1: 1, 3: 2, 5: 3, 7: 4, 9: 5, 11: 6, 13: 7, 15: 8, 17: 9, 19: 10, 21: 11, 23: 0 };
    /** Thứ tự 12 tháng trong một NĂM DƯƠNG LỊCH: Sửu (tháng 1) → Tý (tháng 12). */
    var JIE_ORDER = [1, 3, 5, 7, 9, 11, 13, 15, 17, 19, 21, 23];

    var RAD = Math.PI / 180;
    /** Số ngày trung bình của một tiết khí — chỉ dùng để ĐOÁN chỉ số, rồi dò lại. */
    var MEAN_TERM_DAYS = 365.2422 / 24;

    /* ─────────────── Nghiệm Mặt Trời ─────────────── */

    /**
     * Mốc tiết khí số `n` (hoàng kinh biểu kiến 15n°), ngày Julius ở UTC+8.
     *
     * TRUYỀN THẲNG 8 chứ không để trống: ShouXingUtil giữ mốc múi giờ trong
     * một biến TOÀN CỤC mà tab Lịch đặt về múi giờ địa phương trước mỗi lần
     * vẽ. Bỏ trống là con số ra theo mốc của người gọi cuối cùng.
     */
    function termJd(n) {
        return ShouXingUtil.qiAccurate(n * Math.PI / 12, 8) + Solar.J2000;
    }

    /**
     * Thời điểm Mặt Trời đạt hoàng kinh `deg` (ĐỘ, đã mở vòng — 315° của năm
     * sau là 675°), ngày Julius ở UTC+8.
     *
     * Đây là ruột của qiAccurate, bỏ đi bước tra bảng DE423: bảng chỉ có 24
     * tiết khí nên mốc giữa tháng không tra được. `dtT` quy TT về UT, rồi
     * cộng 8 giờ ra giờ Bắc Kinh — đúng chuỗi phép mà qiAccurate làm.
     */
    function lonJd(deg) {
        var t = ShouXingUtil.saLonT(deg * RAD) * 36525;
        return t - ShouXingUtil.dtT(t) + 8 / 24 + Solar.J2000;
    }

    /**
     * Chỉ số tiết khí TUYỆT ĐỐI (n = 0 tại Xuân Phân 1999) của mốc `jd`.
     *
     * Đoán bằng số ngày trung bình rồi DÒ LẠI ±2: quỹ đạo Trái Đất là ellip
     * nên khoảng cách hai tiết lệch khỏi trung bình tới một ngày, và sau vài
     * chục năm phép đoán suông có thể lệch hẳn một chỉ số ngay sát ranh giới.
     */
    function termIndexNear(jd) {
        var n = Math.round((jd - 2451259.0) / MEAN_TERM_DAYS);
        for (var k = -2; k <= 2; k++) {
            if (Math.abs(termJd(n + k) - jd) < 3) return n + k;
        }
        return n;
    }

    /**
     * Ranh giới THÁNG LỆNH chứa `active.part` — không phải ranh giới của
     * riêng ĐOẠN cầm lệnh (`part.jdFrom/jdTo`, có thể là mốc GIỮA tháng khi
     * đoạn ấy không phải đoạn đầu), mà là ranh giới của CẢ THÁNG: mốc mở
     * tháng (đoạn ĐẦU cũng bắt đầu ở đó) và mốc mở tháng SAU (đoạn CUỐI cũng
     * kết ở đó, vì buildYear() chốt cứng đoạn cuối bằng `termJd(n0 + 2)`).
     *
     * Hai mốc này KHÔNG phụ thuộc bộ số đang chọn: `n0` tính một lần cho mỗi
     * tháng, trước khi chia theo `rule.fen` — ba sách chỉ khác nhau ở chỗ
     * CHIA nhỏ 30° ấy ra sao, không khác ở chỗ 30° ấy bắt đầu/kết thúc khi
     * nào. Tuổi nhập đại vận vì thế giống nhau ở cả ba sách, dù can cầm lệnh
     * (Mậu/Bính/Giáp…) có thể khác nhau.
     */
    function monthBounds(month) {
        var parts = month.parts;
        return { start: parts[0].jdFrom, end: parts[parts.length - 1].jdTo };
    }

    /**
     * Thuận hành hay nghịch hành: chéo GIỚI TÍNH với ÂM DƯƠNG của can năm.
     *
     * Nam sinh năm can DƯƠNG (Giáp Bính Mậu Canh Nhâm — chỉ số CHẴN trong
     * CAN_VI/CAN_ZH) hoặc Nữ sinh năm can ÂM (Ất Đinh Kỷ Tân Quý — chỉ số
     * LẺ) thì THUẬN; còn lại (Nam+Âm hoặc Nữ+Dương) thì NGHỊCH.
     *
     * `yearGanIdx` do app.js lộ ra qua `window.__yearGanIdx` (0=Giáp…9=Quý),
     * tính lại mỗi lần `processAll()` chạy — cùng lúc bảng Bát Tự phía trên vẽ
     * lại, nên không lệch pha với can năm đang hiển thị. Chưa có (lần vẽ đầu
     * tiên, trước khi processAll từng chạy) thì mặc định THUẬN — chỉ để khỏi
     * ném lỗi, không phải một lựa chọn có ý nghĩa mệnh lý.
     */
    function isThuanHanh(genderKey, yearGanIdx) {
        if (typeof yearGanIdx !== 'number' || yearGanIdx < 0) return true;
        var duong = (yearGanIdx % 2 === 0);
        var nam = (genderKey === 'nam');
        return (nam && duong) || (!nam && !duong);
    }

    /**
     * Can chi của Đại Vận ĐẦU TIÊN: bước đúng MỘT nấc trong lục thập hoa
     * giáp kể từ trụ THÁNG — tới nếu THUẬN, lùi nếu NGHỊCH. Ví dụ đối chiếu
     * người dùng cho: tháng Đinh Dậu, thuận → Mậu Tuất; nghịch → Bính Thân.
     *
     * Can và chi CÙNG bước một nấc — không phải chọn can và chi độc lập rồi
     * ghép: lục thập hoa giáp chỉ có 60 cặp hợp lệ trong 120 cặp có thể (can
     * và chi phải cùng tính chẵn/lẻ), và bước cả hai chỉ số cùng lúc luôn giữ
     * đúng tính chất ấy — cách DUY NHẤT sinh ra cặp kế tiếp/trước đó hợp lệ mà
     * khỏi phải dò qua bảng 60 cặp.
     */
    function daiVanPillarOf(thuan, monthCanIdx, monthChiIdx) {
        var step = thuan ? 1 : -1;
        return {
            can: (monthCanIdx + step + 10) % 10,
            chi: (monthChiIdx + step + 12) % 12,
        };
    }

    /**
     * Tuổi nhập đại vận: quy ước "tam nhật nhất tuế" — 3 NGÀY cách mốc mở
     * tháng (kế tiếp nếu THUẬN, hiện tại nếu NGHỊCH) = 1 TUỔI.
     *
     * `diffDays` là khoảng cách ấy theo ngày thập phân (giờ/phút của giờ sinh
     * đã gộp vào phần lẻ, vì `diffDays` tự nó là hiệu hai mốc Julian Day).
     * Tuổi ra số thập phân; phần lẻ của TUỔI quy tiếp ra THÁNG (1 tuổi = 12
     * tháng), rồi phần lẻ của THÁNG quy ra NGÀY (1 tháng = 30 ngày) — dừng ở
     * ngày, không xuống giờ/phút.
     *
     * Ví dụ đối chiếu (thuận hành): cách mốc mở tháng kế tiếp 10,5 ngày →
     * 10,5 / 3 = 3,5 tuổi = 3 tuổi + (0,5 × 12 =) 6 tháng, không dư ngày →
     * "3 tuổi 6 tháng".
     */
    function daiVanTuoi(diffDays) {
        var ageY = diffDays / 3;
        var years = Math.floor(ageY + 1e-9);
        var monthsDec = (ageY - years) * 12;
        var months = Math.floor(monthsDec + 1e-9);
        var days = Math.round((monthsDec - months) * 30);
        // Số lẻ dấu phẩy động hoạ hiếm khi tròn NGAY 30/12 — dồn lên đơn vị
        // trên để không hiện "3 tuổi 11 tháng 30 ngày" cạnh "4 tuổi 0 tháng".
        if (days >= 30) { days -= 30; months += 1; }
        if (months >= 12) { months -= 12; years += 1; }
        return { years: years, months: months, days: days };
    }

    /**
     * "3 tuổi 6 tháng", bỏ đơn vị bằng 0 (kể cả ở giữa) — khớp đúng cách ví
     * dụ đối chiếu ở trên viết ra: không có "0 ngày" thừa khi không dư ngày.
     * Còn tất cả đều 0 (cực hiếm: sinh đúng thời khắc tiết khí) thì hiện
     * "0 ngày" chứ không bỏ trắng.
     */
    function fmtTuoi(dv) {
        var u = isZH() ? { y: '岁', m: '个月', d: '天' }
                        : { y: ' tuổi', m: ' tháng', d: ' ngày' };
        var parts = [];
        if (dv.years  > 0) parts.push(dv.years  + u.y);
        if (dv.months > 0) parts.push(dv.months + u.m);
        if (dv.days   > 0) parts.push(dv.days   + u.d);
        if (!parts.length) parts.push('0' + u.d);
        return parts.join(isZH() ? '' : ' ');
    }

    /**
     * Cộng lịch (không phải cộng ngày thô): thời điểm sinh (Y/M/D dương lịch,
     * đúng ngày người dùng đã chọn) + tuổi nhập vận (năm, tháng, ngày) =
     * thời điểm bắt đầu đại vận. Dùng Date gốc của JS để việc "tràn" tự
     * đúng — cộng tháng mà lố qua năm sau, cộng ngày mà lố qua tháng sau đều
     * do chính Date lo, khỏi tự viết lại lịch Gregory.
     *
     * Đây LÀ phép cộng người thường vẫn hiểu ("sinh năm nay, ba tuổi rưỡi
     * thì vào vận năm kia"), không phải cộng thẳng số-ngày-thập-phân đã dùng
     * để RA số năm/tháng/ngày ở trên — hai phép cộng cho kết quả gần nhau
     * nhưng không hệt nhau (tháng quy ước 30 ngày trong `daiVanTuoi`, còn ở
     * đây tháng dài ngắn thật). Đúng ý người dùng yêu cầu: cộng theo lịch.
     */
    function addYMD(y, m, d, dv) {
        var dt = new Date(y, m - 1, d);
        dt.setFullYear(dt.getFullYear() + dv.years);
        dt.setMonth(dt.getMonth() + dv.months);
        dt.setDate(dt.getDate() + dv.days);
        return { y: dt.getFullYear(), m: dt.getMonth() + 1, d: dt.getDate() };
    }

    /** "17/06/2028" — CỐ ĐỊNH dd/mm/yyyy, không đổi theo ngôn ngữ: mọi cột
     *  ngày tháng khác trong tab này (Vào lệnh/Hết lệnh) đã theo đúng quy
     *  ước ấy bất kể tiếng Việt hay tiếng Trung. */
    function fmtYMD(ymd) {
        return pad2(ymd.d) + '/' + pad2(ymd.m) + '/' + ymd.y;
    }

    /* ─────────────── Dựng bảng một năm ─────────────── */

    var _yearCache = {};
    function cacheKey(Y) { return rule.key + '|' + Y; }

    /**
     * 12 tháng lệnh của NĂM DƯƠNG LỊCH `Y`, từ Sửu (Tiểu Hàn, tháng 1) tới Tý
     * (Đại Tuyết, tháng 12). Mọi mốc là ngày Julius ở UTC+8 — không phụ thuộc
     * địa điểm, nên nhớ theo năm là đủ; việc quy sang giờ địa phương để LÀ
     * việc của lúc vẽ.
     *
     * Mỗi phần tử: { chi, jie, n, parts: [{ can, from, to, jdFrom, jdTo }] }
     * với `from`/`to` là hoàng kinh ĐÃ MỞ VÒNG (cộng dồn qua các năm).
     */
    function buildYear(Y) {
        var ck = cacheKey(Y);
        if (_yearCache[ck]) return _yearCache[ck];
        // Người dùng bấm tới bấm lui vài trăm năm thì bộ nhớ đệm phình mãi.
        // Dựng một năm chỉ mất chừng một mili giây, nên dọn sạch rồi dựng lại
        // là rẻ hơn hẳn so với việc giữ một danh sách LRU.
        if (Object.keys(_yearCache).length > 40) _yearCache = {};

        // Cùng dãy mốc mà bảng tiết khí của tab Lịch dùng: index 1..24 là
        // Đông Chí (tháng 12 năm Y−1) rồi Tiểu Hàn (tháng 1 năm Y)… tới Đại
        // Tuyết (tháng 12 năm Y). Nên TIẾT có chỉ số k nằm ở ô k+1.
        var jds = Ephem.jieQiJdAtBasis(Y, null);
        var out = [];
        for (var i = 0; i < JIE_ORDER.length; i++) {
            var k = JIE_ORDER[i];
            var n0 = termIndexNear(jds[k + 1]);
            var parts = [], off = 0;
            var fen = rule.fen[k];
            for (var j = 0; j < fen.length; j++) {
                var can = fen[j][0], span = fen[j][1];
                parts.push({
                    can: can,
                    from: 15 * n0 + off,
                    to: 15 * n0 + off + span,
                    // Mốc mở/đóng tháng là bội số của 15° → đi qua bảng DE423,
                    // trùng khít bảng tiết khí. Mốc giữa tháng thì giải chuỗi.
                    jdFrom: off === 0 ? termJd(n0) : lonJd(15 * n0 + off),
                    jdTo: (off + span) === 30 ? termJd(n0 + 2) : lonJd(15 * n0 + off + span),
                });
                off += span;
            }
            if (off !== 30) throw new Error('lenh.js: tháng ' + k + ' cộng ra ' + off + '°');
            out.push({ chi: CHI_OF_JIE[k], jie: k, n: n0, parts: parts });
        }
        _yearCache[ck] = out;
        return out;
    }

    /**
     * Can đang cầm lệnh tại thời điểm `jdUTC8`, hoặc null nếu tra không ra.
     *
     * Dò cả năm TRƯỚC nữa: tháng Tý mở ở Đại Tuyết (tháng 12) mà đóng ở Tiểu
     * Hàn năm sau, nên mọi ngày đầu tháng 1 đều thuộc bảng của năm trước.
     */
    function lenhAt(jdUTC8, y) {
        for (var dy = 0; dy >= -1; dy--) {
            var year = buildYear(y + dy);
            for (var i = 0; i < year.length; i++) {
                var parts = year[i].parts;
                for (var j = 0; j < parts.length; j++) {
                    if (jdUTC8 >= parts[j].jdFrom && jdUTC8 < parts[j].jdTo) {
                        return { month: year[i], part: parts[j] };
                    }
                }
            }
        }
        return null;
    }

    /* ─────────────── Đọc đầu vào ─────────────── */

    /** Ngày giờ đang chọn + địa điểm, hoặc null khi app.js chưa sẵn sàng. */
    function readInput() {
        if (typeof getDOM !== 'function' || typeof countryData === 'undefined') return null;
        var y = parseInt(getDOM('inYear').value, 10);
        var m = parseInt(getDOM('inMonth').value, 10);
        var d = parseInt(getDOM('inDay').value, 10);
        var h = parseInt(getDOM('solarHour').value, 10);
        var mi = parseInt(getDOM('solarMinute').value, 10);
        if (!isFinite(y) || !isFinite(m) || !isFinite(d)) return null;
        var info = countryData[getDOM('country').value];
        if (!info) return null;
        var tz = (typeof getTimezoneOffset === 'function')
            ? getTimezoneOffset(info.tzId, new Date(y, m - 1, d, h || 12)) : 7;
        return { y: y, m: m, d: d, h: h || 0, mi: mi || 0, info: info, tz: tz };
    }

    /**
     * Giờ địa phương của một mốc UTC+8, trả về đối tượng Solar.
     *
     * Dùng ĐÚNG đường mà bảng tiết khí dùng (_tzOffsetAtJdUTC8): offset lấy
     * tại CHÍNH THỜI ĐIỂM ấy chứ không phải theo ngày đang xem, nên mốc mùa
     * đông không bị cộng nhầm giờ mùa hè ở những nước có DST.
     */
    function toLocal(jdUTC8, tzId) {
        var tz = (typeof _tzOffsetAtJdUTC8 === 'function')
            ? _tzOffsetAtJdUTC8(jdUTC8, tzId) : 8;
        return Solar.fromJulianDay(jdUTC8 + (tz - 8) / 24);
    }

    /**
     * "05/01 16:23", hoặc "05/01/2027 22:09" khi mốc rơi ra ngoài năm của
     * bảng — đúng cách ảnh mẫu viết. Bỏ năm ở 31/33 hàng thì cột hẹp đi
     * chừng 30px, đủ để cả năm cột vừa màn hình 360px mà không phải kéo ngang.
     */
    function fmtLocal(jdUTC8, tzId, tableYear) {
        var s = toLocal(jdUTC8, tzId);
        var ymd = pad2(s.getDay()) + '/' + pad2(s.getMonth())
            + (s.getYear() === tableYear ? '' : '/' + s.getYear());
        return ymd + ' ' + pad2(s.getHour()) + ':' + pad2(s.getMinute());
    }

    /* ─────────────── Vẽ ─────────────── */

    var lastActive = null;
    /** Chỉ dùng cho bộ kiểm thử: kết quả tuổi nhập vận của lần vẽ gần nhất,
     *  kèm chiều (thuận/nghịch) đã dùng — đọc thẳng, khỏi phải giật ngược
     *  chuỗi hiển thị. */
    var lastDaiVan = null;

    function render() {
        var head = document.getElementById('lenhTitle');
        var nowBox = document.getElementById('lenhNow');
        var body = document.getElementById('lenhBody');
        if (!body) return;
        if (typeof Solar === 'undefined' || typeof ShouXingUtil === 'undefined' ||
            typeof Ephem === 'undefined' || typeof TK_VI === 'undefined') return;

        var inp = readInput();
        if (!inp) return;

        var year;
        try { year = buildYear(inp.y); } catch (e) { body.innerHTML = ''; return; }

        // Can đang cầm lệnh: hỏi ở ĐÚNG thời điểm UTC+8 mà trụ tháng dùng, nên
        // chi của tháng lệnh luôn khớp chi của trụ tháng trong bảng Bát Tự
        // ngay bên trên — hai con số cãi nhau trên cùng một màn hình là lỗi
        // nặng hơn cả sai vài phút.
        var active = null;
        var vanText = null;
        var pillarText = null;
        lastDaiVan = null;
        if (typeof _readInputBJ === 'function') {
            try {
                var bj = _readInputBJ(inp.y, inp.m, inp.d, inp.h, inp.mi, inp.tz);
                var birthJd = bj.solarBJ.getJulianDay();
                active = lenhAt(birthJd, inp.y);
                // Cùng mốc giờ Bắc Kinh mà lenhAt() vừa dùng, nên "đang cầm
                // lệnh can nào" và "tuổi nhập vận" luôn khớp cùng MỘT thời
                // điểm sinh, không lệch nguồn.
                if (active) {
                    var mb = monthBounds(active.month);
                    var yIdx = (typeof window !== 'undefined') ? window.__yearGanIdx : undefined;
                    var thuan = isThuanHanh(gender.key, yIdx);
                    // Thuận: mốc mở tháng KẾ TIẾP trừ giờ sinh.
                    // Nghịch: giờ sinh trừ mốc mở tháng HIỆN TẠI.
                    var diff = thuan ? (mb.end - birthJd) : (birthJd - mb.start);
                    var dv = daiVanTuoi(diff);
                    // Can chi Đại Vận đầu tiên: bước một nấc từ CHÍNH trụ
                    // tháng, cùng chiều thuận/nghịch vừa dùng ở trên — không
                    // phải một phép tính tách rời. monthCanIdx đọc từ app.js
                    // (window.__monthGanIdx); monthChiIdx dùng thẳng
                    // active.month.chi — đã có sẵn và đã BẢO ĐẢM khớp trụ
                    // tháng thật (xem khối ghi chú ngay trên `active`).
                    var mCanIdx = (typeof window !== 'undefined') ? window.__monthGanIdx : undefined;
                    var pillar = (typeof mCanIdx === 'number' && mCanIdx >= 0)
                        ? daiVanPillarOf(thuan, mCanIdx, active.month.chi) : null;
                    lastDaiVan = { dv: dv, thuan: thuan, diffDays: diff, mb: mb, pillar: pillar };
                    // Cộng lịch vào ĐÚNG ngày sinh người dùng đã chọn (inp.y/
                    // m/d, dương lịch địa phương) — không phải birthJd giờ
                    // Bắc Kinh, vì đó là ngày mà "tuổi nhập vận" phải được
                    // hiểu theo. Luôn CỘNG TỚI (không trừ) dù nghịch hành hay
                    // thuận hành: "tuổi nhập vận" luôn là một tuổi DƯƠNG, và
                    // ngày bắt đầu đại vận luôn ở SAU ngày sinh.
                    vanText = fmtTuoi(dv) + ' · ' + fmtYMD(addYMD(inp.y, inp.m, inp.d, dv));
                    // Tiếng Việt cần khoảng trắng ("Mậu Tuất" là hai tiếng);
                    // tiếng Trung thì không (ghép can chi liền nhau — "戊戌",
                    // không phải "戊 戌" — đúng cách mọi cặp can chi khác
                    // trong ứng dụng đã hiện, ví dụ cột "Tuần thủ" ở tab Kỳ
                    // Môn).
                    if (pillar) pillarText = canName(pillar.can) + (isZH() ? '' : ' ') + chiName(pillar.chi);
                }
            } catch (e2) { active = null; vanText = null; }
        }
        lastActive = active;

        if (head) {
            head.textContent = isZH()
                ? (inp.y + '年' + t('title'))
                : (t('title') + ' ' + inp.y);
        }
        if (nowBox) {
            var lệnhLine = esc(t('now')) + (isZH() ? '：' : ': ') +
                '<b id="lenhNowVal">' + esc(active ? canName(active.part.can) : '—') + '</b>';
            var vanLine = vanText
                ? '<div id="lenhDaiVan">' + esc(t('daiVan')) + (isZH() ? '：' : ': ') +
                  '<b id="lenhDaiVanVal">' + esc(vanText) + '</b></div>'
                : '';
            var pillarLine = pillarText
                ? '<div id="lenhDaiVanPillar">' + esc(t('daiVanPillar')) + (isZH() ? '：' : ': ') +
                  '<b id="lenhDaiVanPillarVal">' + esc(pillarText) + '</b></div>'
                : '';
            nowBox.innerHTML = lệnhLine + vanLine + pillarLine;
        }

        var tzId = inp.info.tzId;
        var rows = '', alt = false;
        for (var i = 0; i < year.length; i++) {
            var mo = year[i];
            for (var j = 0; j < mo.parts.length; j++) {
                var p = mo.parts[j];
                var on = active && active.part === p;
                rows += '<tr class="' + (alt ? 'dp-row-alt' : '') + (on ? ' lenh-on' : '') + '"' +
                    (on ? ' id="lenhActive"' : '') + '>';
                if (j === 0) {
                    rows += '<td class="lenh-mon" rowspan="' + mo.parts.length + '">' +
                        esc(chiName(mo.chi)) +
                        ' <span class="lenh-jie">(' + esc(jieName(mo.jie)) + ')</span></td>';
                }
                rows += '<td class="c lenh-can">' + esc(canName(p.can)) + '</td>' +
                    '<td class="c dp-num lenh-lon">' + (p.to - p.from) + '</td>' +
                    '<td class="c dp-num">' + esc(fmtLocal(p.jdFrom, tzId, inp.y)) + '</td>' +
                    '<td class="c dp-num lenh-last">' + esc(fmtLocal(p.jdTo, tzId, inp.y)) + '</td>' +
                    '</tr>';
            }
            alt = !alt;   // đổi nền theo THÁNG, không theo hàng: ba hàng của
                          // một tháng phải cùng nền thì ô gộp mới liền khối.
        }

        body.innerHTML =
            '<table class="dp-table lenh-tb">' +
            '<colgroup><col><col><col><col><col></colgroup>' +
            '<thead><tr class="lenh-head">' +
            '<th>' + esc(t('colMonth')) + '</th>' +
            '<th class="c">' + esc(t('colCan')) + '</th>' +
            '<th class="c">' + esc(t('colLon')) + '</th>' +
            '<th class="c">' + esc(t('colIn')) + '</th>' +
            '<th class="c lenh-last">' + esc(t('colOut')) + '</th>' +
            '</tr></thead><tbody>' + rows + '</tbody></table>';

        fit();
        setTimeout(scrollToActive, 40);
    }

    /**
     * Cuộn để hàng đang cầm lệnh nằm giữa khung — bảng dài 33 hàng, mở ra mà
     * phải tự đi tìm hàng của hôm nay thì bảng vô dụng một nửa.
     *
     * Chốt về đúng ranh giới hàng: hàng tiêu đề DÍNH ở mép trên, nên cuộn tới
     * một vị trí bất kỳ là để lại một hàng bị cắt ngang nằm ngay dưới nó.
     */
    function scrollToActive() {
        var box = document.getElementById('lenhBody');
        var row = document.getElementById('lenhActive');
        if (!box || !row) return;
        // Cùng lý do với fit(): khung còn đóng thì mọi phép đo ra số 0, cuộn
        // theo đó chỉ đặt scrollTop sai — bỏ qua, đợi lúc mở lại gọi.
        var sec = document.getElementById('lenhSec');
        if (sec && getComputedStyle(sec).display === 'none') return;
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var thead = box.querySelector('thead');
        var headH = thead ? thead.getBoundingClientRect().height / zoom : 0;

        // Bước 1 — cuộn thô, đo TƯƠNG ĐỐI: "hàng này đang cách đỉnh khung bao
        // nhiêu" cộng vào scrollTop hiện thời. Không dựng lại hệ toạ độ nội
        // dung từ scrollTop và các mép: viền, đệm và chính hàng tiêu đề DÍNH
        // (getBoundingClientRect của <thead> vẫn trả vị trí lúc CHƯA dính, vì
        // chỉ mấy ô <th> mới sticky chứ không phải cả <thead>) — ba thứ ấy đủ
        // để lệch cả chục pixel.
        var rBox = box.getBoundingClientRect();
        var rRow = row.getBoundingClientRect();
        var mid = Math.max(0, (box.clientHeight - headH - rRow.height / zoom) / 2);
        box.scrollTop = Math.max(0, Math.round(
            box.scrollTop + (rRow.top - rBox.top) / zoom - headH - mid));

        alignUnderHead(box, headH, zoom);
    }

    /**
     * ĐO rồi chỉnh: kéo khung xuống vừa đủ để hàng đang bị hàng tiêu đề cắt
     * ngang lộ ra trọn vẹn.
     *
     * Các hàng KHÔNG cao bằng nhau — ô tháng gộp 2 hàng chứa hai dòng chữ nên
     * kéo hai hàng ấy cao hơn ba hàng của tháng bên cạnh — nên không thể chốt
     * bằng cách chia cho một "chiều cao hàng". Sau khi đã cuộn thì mọi thứ đã
     * nằm trên trang, hỏi thẳng là xong; kéo xuống chỉ giấu thêm phần trên,
     * không sinh ra hàng cụt mới.
     */
    function alignUnderHead(box, headH, zoom) {
        var bt = parseFloat(getComputedStyle(box).borderTopWidth) || 0;
        var edge = box.getBoundingClientRect().top / zoom + bt + headH;
        var rows = box.querySelectorAll('tbody tr');

        // Chốt về đầu THÁNG, không phải đầu hàng.
        //
        // Ô tháng gộp 2–3 hàng, nên dừng ở giữa một tháng là hàng đầu khung
        // hiện "(Kinh Trập)" mà mất chữ "Mão" nằm trên nó — một cái tên tiết
        // mồ côi, không biết của tháng nào. Một tháng chỉ cao 2–3 hàng mà
        // khung thì hiện hơn 20 hàng, nên kéo lên tới đầu tháng gần nhất
        // không hề đẩy hàng đang cầm lệnh ra khỏi khung.
        var top = 0;
        for (var i = 0; i < rows.length; i++) {
            var q = rows[i].getBoundingClientRect();
            if (q.top / zoom > edge + 0.5) break;
            if (rows[i].querySelector('.lenh-mon')) top = q.top / zoom;
        }
        if (!top) return;
        box.scrollTop = Math.max(0, box.scrollTop - Math.round(edge - top));
    }

    /* ─────────────── Chia chiều cao ─────────────── */

    /** Đệm chống tràn — phép làm tròn nửa pixel, không phải phép cộng lề. */
    var CHROME = 10;
    /** Khung bảng thấp hơn chừng này thì thà cuộn cả trang còn hơn. */
    var BOX_MIN = 120;

    /**
     * Kéo bảng lệnh cho lấp đúng phần màn hình còn lại.
     *
     * Cùng bài toán của fitGrid bên calendar.js: khung trên (ô ngày giờ, bảng
     * Bát Tự, dòng "Lệnh:") cao bao nhiêu là do nội dung, còn bảng thì dài vô
     * hạn — nên ĐO phần cố định rồi cấp phần còn lại cho bảng, chứ không đoán.
     * Không có bước này thì bảng 33 hàng đẩy trang cao gấp rưỡi màn hình,
     * viewport.js thu nhỏ cả trang để chữa, và bảng Bát Tự bé lại vô cớ.
     */
    function fit() {
        var sec = document.getElementById('lenhSec');
        var box = document.getElementById('lenhBody');
        var bar = document.getElementById('bottomDock');
        if (!sec || !box || !bar) return;
        if (!document.body.classList.contains('view-lenh')) return;
        // Bảng đóng (xem #lenhSec{display:none} trong lenh.css): đo bây giờ
        // chỉ ra toàn số 0 (getBoundingClientRect của phần tử display:none),
        // và maxHeight tính từ đó là rác — bỏ qua, đợi lúc MỞ (xem chỗ bọc
        // toggleDetailPanel() gọi lại đúng hàm này).
        if (getComputedStyle(sec).display === 'none') return;

        // Khung bảng là khối CUỐI CÙNG của trang, nên chỗ nó được phép chiếm
        // chính là khoảng từ đỉnh nó tới thanh dưới — đo thẳng, khỏi phải cộng
        // lại chiều cao từng khối bên trên cùng mọi khe giữa chúng (bản đầu
        // làm vậy: mười mấy dòng, và sai ngay khi ai đó thêm một lề trong CSS).
        // Hạ chiều cao khung KHÔNG làm chính đỉnh nó nhúc nhích, nên con số đo
        // được vẫn đúng sau khi đặt.
        var zoom = parseFloat(getComputedStyle(document.body).zoom) || 1;
        var top = sec.getBoundingClientRect().top / zoom;
        var floorY = bar.getBoundingClientRect().top / zoom;
        var room = Math.max(BOX_MIN, Math.floor(floorY - top - CHROME));

        // Bảng ngắn hơn chỗ được cấp (năm nào cũng 33 hàng nên hiếm, nhưng máy
        // tính bảng thì có) thì kẹp theo chính nó, đừng chừa một khoảng trắng.
        var tb = box.querySelector('table');
        var nat = tb ? Math.ceil(tb.getBoundingClientRect().height / zoom) + 2 : 0;
        box.style.maxHeight = (nat ? Math.min(nat, room) : room) + 'px';

        // Mép dưới không được cắt hàng cuối đúng chỗ có DẤU: "Mậu" cụt dấu
        // nặng thành "Mâu", "Bạch Lộ" thành "Bạch Lô" — chữ khác hẳn, không
        // phải hàng cụt. Phép canh nằm ở calendar.js (xem snapCut ở đó), dùng
        // chung một bản cho cả hai tab.
        if (typeof window.__snapCutRows === 'function') {
            try { window.__snapCutRows(box); } catch (e) {}
        }

        // Canh lại mép TRÊN sau MỌI lần đổi chiều cao. Lượt cuộn tới hàng đang
        // cầm lệnh chạy 40ms sau khi vẽ, còn viewport.js chỉnh tỉ lệ rồi gọi
        // fit() lần nữa SAU đó — chia lại khung xong mà không canh lại thì cú
        // canh trước đó tính trên một khung đã không còn nữa, và hàng đầu lại
        // nằm cụt dưới hàng tiêu đề.
        var thead = box.querySelector('thead');
        if (thead) alignUnderHead(box, thead.getBoundingClientRect().height / zoom, zoom);
    }

    /* ─────────────── Nhãn + móc nối ─────────────── */

    /* ─────────────── Ô chọn quy tắc ─────────────── */

    function showRule() {
        var el = document.getElementById('lenhRuleText');
        if (el) el.textContent = ruleLabel(rule);
    }

    /**
     * Đổi bộ số thì NHỚ LẠI: người dùng theo một phái chứ không chọn lại mỗi
     * lần mở app. Cùng cách app.js nhớ ngôn ngữ và phái Kỳ Môn.
     */
    function setRule(key) {
        if (key === rule.key) return;
        rule = ruleByKey(key);
        if (typeof safeStorage !== 'undefined') safeStorage.setItem(K_RULE, rule.key);
        showRule();
        render();
    }

    window.openLenhRulePicker = function () {
        if (typeof openOptionPicker !== 'function') return;
        var opts = [];
        for (var i = 0; i < RULES.length; i++) {
            opts.push({ value: RULES[i].key, label: ruleLabel(RULES[i]) });
        }
        openOptionPicker(t('pickRule'), opts, rule.key, setRule);
    };

    function showGender() {
        var el = document.getElementById('lenhGenderText');
        if (el) el.textContent = genderLabel(gender);
    }

    function setGender(key) {
        if (key === gender.key) return;
        gender = genderByKey(key);
        if (typeof safeStorage !== 'undefined') safeStorage.setItem(K_GENDER, gender.key);
        showGender();
        // Đổi giới tính đổi CHIỀU đại vận (thuận/nghịch), nên "Nhập vận: …"
        // phải vẽ lại — xem isThuanHanh().
        render();
    }

    window.openLenhGenderPicker = function () {
        if (typeof openOptionPicker !== 'function') return;
        var opts = [];
        for (var i = 0; i < GENDERS.length; i++) {
            opts.push({ value: GENDERS[i].key, label: genderLabel(GENDERS[i]) });
        }
        openOptionPicker(t('pickGender'), opts, gender.key, setGender);
    };

    function refreshLabels() {
        var tab = document.getElementById('tabLenh');
        if (tab) tab.querySelector('.tab-lbl').textContent = t('tabLenh');
        showRule();
        showGender();
        if (document.body.classList.contains('view-lenh')) render();
    }

    window.__lenhRefreshLabels = refreshLabels;
    window.__lenhRender = render;
    window.__lenhFit = function () {
        if (!document.body.classList.contains('view-lenh')) return;
        try { fit(); } catch (e) {}
    };
    /** Chỉ dùng cho bộ kiểm thử: đọc thẳng bảng đã tính, khỏi phải đọc DOM. */
    window.__lenhData = function (Y) { return buildYear(Y); };
    window.__lenhActive = function () { return lastActive; };
    window.__lenhDaiVan = function () { return lastDaiVan; };
    window.__lenhRule = function (k) { if (k) setRule(k); return rule.key; };
    window.__lenhGender = function (k) { if (k) setGender(k); return gender.key; };

    document.addEventListener('DOMContentLoaded', function () {
        if (typeof safeStorage !== 'undefined') {
            var saved = safeStorage.getItem(K_RULE);
            if (saved) rule = ruleByKey(saved);
            var savedG = safeStorage.getItem(K_GENDER);
            if (savedG) gender = genderByKey(savedG);
        }
        refreshLabels();

        // Vẽ lại sau MỌI lần engine tính lại — đổi ngày, đổi địa điểm, đổi
        // ngôn ngữ đều đi qua processAll(). Cùng cách calendar.js móc vào, nên
        // ba tab luôn nói cùng một địa điểm và cùng một thời điểm.
        if (typeof processAll === 'function' && !processAll.__lenhWrapped) {
            var orig = processAll;
            var wrapped = function () {
                var r = orig.apply(this, arguments);
                if (document.body.classList.contains('view-lenh')) {
                    try { render(); } catch (err) { console.warn('lenh:', err); }
                }
                return r;
            };
            wrapped.__lenhWrapped = true;
            window.processAll = wrapped;
        }

        // Bảng đóng sẵn (xem #lenhSec{display:none} trong lenh.css) nên
        // fit() không đo được gì trong lúc đóng — mọi con số nó tính lúc ấy
        // đều dựa trên một khung rộng-cao 0 (getBoundingClientRect của phần
        // tử display:none). Vừa MỞ ra thì phải đo lại NGAY, không đợi tới
        // lần render() kế tiếp (có thể rất lâu sau, tới khi đổi ngày/địa
        // điểm). toggleDetailPanel() ở app.js không biết gì về Lệnh — bọc nó
        // ở đây, đúng cách processAll() vừa được bọc ở trên, để app.js vẫn
        // dùng chung một hàm cho cả bốn bảng gập/mở.
        if (typeof window.toggleDetailPanel === 'function' && !window.toggleDetailPanel.__lenhWrapped) {
            var origToggle = window.toggleDetailPanel;
            var wrappedToggle = function (which) {
                var r = origToggle.apply(this, arguments);
                if (which === 'lenh') {
                    var sec = document.getElementById('lenhSec');
                    var head = document.getElementById('lenhHead');
                    var open = sec && getComputedStyle(sec).display !== 'none';
                    if (head) head.classList.toggle('lenh-open', !!open);
                    if (open) {
                        // Đợi khung kịp lên 'block' rồi mới đo — cùng độ trễ
                        // toggleDetailPanel() tự dùng cho phép cuộn trang của
                        // chính nó (requestAnimationFrame), nên hai việc không
                        // giẫm lên nhau: nó cuộn TRANG, còn đây cuộn TRONG
                        // khung bảng — hai hộp cuộn khác nhau.
                        setTimeout(function () {
                            try { fit(); } catch (e) {}
                            scrollToActive();
                        }, 60);
                    }
                }
                return r;
            };
            wrappedToggle.__lenhWrapped = true;
            window.toggleDetailPanel = wrappedToggle;
        }
    });

    window.addEventListener('resize', function () {
        if (document.body.classList.contains('view-lenh')) setTimeout(fit, 180);
    });
})();
