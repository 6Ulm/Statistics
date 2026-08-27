/**
 * Đo tốc độ các đường nóng, và CANH cho chúng khỏi tụt lại.
 *
 *   node test_perf.mjs
 *
 * Mốc đo sau khi gom engine thiên văn vào ephem.js (xem README). Ngưỡng đặt
 * rộng gấp rưỡi số đo để khỏi đỏ vì máy chạy test bận, nhưng vẫn bắt được
 * kiểu hồi quy đã từng có: LunarYear.fromYear() chỉ nhớ MỘT năm, nên hỏi ba
 * năm liền là dựng lại ba lần.
 */
import fs from 'fs'; import http from 'http'; import path from 'path';
const WEB='/home/user/Statistics/android/app/src/main/assets/web';
const MIME={'.html':'text/html','.js':'text/javascript','.css':'text/css','.txt':'text/plain'};
const server=http.createServer((q,r)=>{const rel=decodeURIComponent(q.url.split('?')[0]).replace(/^\/+/,'')||'index.html';
 const f=path.join(WEB,rel); if(!f.startsWith(WEB)||!fs.existsSync(f)){r.writeHead(404);return r.end();}
 r.writeHead(200,{'Content-Type':MIME[path.extname(f)]||'application/octet-stream'});fs.createReadStream(f).pipe(r);});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const {chromium}=await import('playwright');
const b=await chromium.launch({executablePath:'/opt/pw-browsers/chromium'});
const ctx=await b.newContext({viewport:{width:412,height:900}});
await ctx.addInitScript(()=>{try{localStorage.setItem('defaultLang','vi')}catch(e){}});
const p=await ctx.newPage();
await p.goto(`http://127.0.0.1:${server.address().port}/index.html`,{waitUntil:'networkidle'});
await p.waitForTimeout(1000);

const out = await p.evaluate(() => {
  const t=f=>{const a=performance.now(); f(); return performance.now()-a;};
  const med=(f,n)=>{const xs=[];for(let i=0;i<n;i++)xs.push(t(f));xs.sort((a,b)=>a-b);return xs[n>>1];};
  // NGUỘI: xoá sạch đệm trước MỖI lần đo, nếu không phép đo lẫn lộn trúng đệm
  // với trượt đệm và con số nhảy gấp bốn giữa hai lần chạy.
  // sb_getJieQiDates còn bộ nhớ đệm RIÊNG (_sbCache) — không dọn thì phép đo
  // "nguội" của nó chỉ đo một lần so sánh khoá và ra 0,0 ms.
  const cold=(f,n)=>med(()=>{
    if (typeof Ephem !== 'undefined' && Ephem.__clearCaches) Ephem.__clearCaches();
    if(typeof _ziMonthCache!=='undefined') _ziMonthCache.clear();
    try { _sbCache = null; } catch(e) {}
    f();
  },n);
  const info=countryData[document.getElementById('country').value];
  const tz=getTimezoneOffset(info.tzId,new Date(2026,7,13,12));
  let d=1;
  return {
    processAll: med(()=>{d=d%28+1; getDOM('inDay').value=d; processAll();},15),
    calRender:  (window.showTab('cal'), med(()=>window.__calGoto(2026,(d++%12)+1,15),12)),
    ziMonths:   (window.showTab('qmdj'), cold(()=>zi_months(2026, info.lon, info.tzId, tz),8)),
    jieQiCold:  cold(()=>sb_getJieQiDates(2026, info.tzId, tz),8),
    jieQiWarm:  med(()=>sb_getJieQiDates(2026, info.tzId, tz),200),
    // Đường mà người dùng thật sự chạm vào: đổi ngày, lật tháng. Đo giống hệt
    // nhau ở cả hai bản nên so được.
  };
});
/* Ngưỡng là LƯỚI CHẶN HỒI QUY, không phải phép đo chính xác: máy chạy test lúc
   bận có thể chậm gấp rưỡi, nên để rộng tay. Đặt sát quá thì nó đỏ vì tiếng ồn
   chứ không vì code — đã gặp: render() dao động 8,3–11,7 ms giữa các lần chạy. */
const LIMITS = { processAll: 22, calRender: 16, ziMonths: 28, jieQiCold: 22, jieQiWarm: 0.05 };
const NAMES = { processAll:'processAll', calRender:'render() tab Lịch',
                ziMonths:'zi_months (nguội)', jieQiCold:'sb_getJieQiDates nguội',
                jieQiWarm:'sb_getJieQiDates ấm' };
let bad = 0;
console.log('\nNgưỡng canh hồi quy:');
for (const k of Object.keys(LIMITS)) {
  const ok = out[k] <= LIMITS[k];
  if (!ok) bad++;
  console.log(`  ${ok?'ok  ':'CHẬM'} ${NAMES[k].padEnd(20)} ${out[k].toFixed(1)} ms (ngưỡng ${LIMITS[k]} ms)`);
}
await b.close(); server.close();
console.log(bad ? `\n✗ ${bad} đường nóng vượt ngưỡng` : '\n✓ mọi đường nóng trong ngưỡng');
process.exit(bad ? 1 : 0);
