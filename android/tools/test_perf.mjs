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
console.log(await p.evaluate(()=>{
  const t=f=>{const a=performance.now(); f(); return performance.now()-a;};
  const med=(f,n)=>{const xs=[];for(let i=0;i<n;i++)xs.push(t(f));xs.sort((a,b)=>a-b);return xs[n>>1];};
  const out=[];
  // đổi ngày mỗi lần để không ăn cache
  let d=1;
  const nextDay=()=>{ d=d%28+1; getDOM('inDay').value=d; };
  out.push(`  processAll (đổi ngày)      ${med(()=>{nextDay(); processAll();},15).toFixed(1)} ms`);
  window.showTab('cal');
  out.push(`  render() tab Lịch          ${med(()=>{window.__calGoto(2026,(d++%12)+1,15);},12).toFixed(1)} ms`);
  window.showTab('qmdj');
  // đo riêng từng khối
  const info=countryData[document.getElementById('country').value];
  const tz=getTimezoneOffset(info.tzId,new Date(2026,7,13,12));
  out.push(`  zi_months (1 năm, nguội)   ${med(()=>{ _ziMonthCache.clear(); zi_months(2026+(d++%40), info.lon, info.tzId, tz);},8).toFixed(1)} ms`);
  out.push(`  zi_months (đã cache)       ${med(()=>zi_months(2026, info.lon, info.tzId, tz),200).toFixed(3)} ms`);
  out.push(`  sb_getJieQiDates           ${med(()=>sb_getJieQiDates(2020+(d++%40), info.tzId, tz),10).toFixed(1)} ms`);
  out.push(`  getPreciseSocSolarUTC8 ×30 ${med(()=>{for(let i=0;i<30;i++) getPreciseSocSolarUTC8(Solar.fromYmd(2026,(i%12)+1,1));},20).toFixed(2)} ms`);
  return out.join('\n');
}));
const out = await p.evaluate(() => {
  const t=f=>{const a=performance.now(); f(); return performance.now()-a;};
  const med=(f,n)=>{const xs=[];for(let i=0;i<n;i++)xs.push(t(f));xs.sort((a,b)=>a-b);return xs[n>>1];};
  const info=countryData[document.getElementById('country').value];
  const tz=getTimezoneOffset(info.tzId,new Date(2026,7,13,12));
  let d=1;
  return {
    processAll: med(()=>{d=d%28+1; getDOM('inDay').value=d; processAll();},15),
    calRender:  (window.showTab('cal'), med(()=>window.__calGoto(2026,(d++%12)+1,15),12)),
    ziMonths:   (window.showTab('qmdj'), med(()=>{_ziMonthCache.clear(); zi_months(2026+(d++%40), info.lon, info.tzId, tz);},8)),
    jieQi:      med(()=>sb_getJieQiDates(2020+(d++%40), info.tzId, tz),10),
  };
});
const LIMITS = { processAll: 16, calRender: 10, ziMonths: 17, jieQi: 7 };
const NAMES = { processAll:'processAll', calRender:'render() tab Lịch',
                ziMonths:'zi_months (nguội)', jieQi:'sb_getJieQiDates' };
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
