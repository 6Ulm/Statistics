/**
 * Chụp bảng chi tiết Âm Bàn pháp (Tháng âm | Thời điểm Sóc | Thời điểm Vọng).
 *
 *   node shot_amban.mjs            # Hà Nội
 *   CITY=Paris node shot_amban.mjs
 */
import fs from 'fs'; import http from 'http'; import path from 'path';
import { fileURLToPath } from 'url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const OUT = process.env.OUT || path.join(HERE, '..', '..', 'shots');
fs.mkdirSync(OUT, { recursive: true });

const MIME={'.html':'text/html','.js':'text/javascript','.css':'text/css','.txt':'text/plain'};
const server=http.createServer((q,r)=>{const rel=decodeURIComponent(q.url.split('?')[0]).replace(/^\/+/,'')||'index.html';
 const f=path.join(WEB,rel); if(!f.startsWith(WEB)||!fs.existsSync(f)){r.writeHead(404);return r.end();}
 r.writeHead(200,{'Content-Type':MIME[path.extname(f)]||'application/octet-stream'});fs.createReadStream(f).pipe(r);});
await new Promise(r=>server.listen(0,'127.0.0.1',r));
const {chromium}=await import('playwright');
const b=await chromium.launch(
  fs.existsSync('/opt/pw-browsers/chromium') ? {executablePath:'/opt/pw-browsers/chromium'} : {});

for (const want of (process.env.CITY ? [process.env.CITY] : ['Hà Nội','Paris'])) {
  const ctx=await b.newContext({viewport:{width:412,height:1500},deviceScaleFactor:2.5,isMobile:true,hasTouch:true});
  await ctx.addInitScript(()=>{try{localStorage.setItem('defaultLang','vi')}catch(e){}});
  const p=await ctx.newPage();
  p.on('pageerror',e=>console.error('LỖI:',e.message));
  await p.goto(`http://127.0.0.1:${server.address().port}/index.html`,{waitUntil:'networkidle'});
  await p.waitForTimeout(900);
  const name = await p.evaluate((want)=>{
    const opt=[...document.getElementById('country').options]
      .find(o=>(countryData[o.value]?.name_vi||'').includes(want));
    document.getElementById('country').value=opt.value;
    selectMethod('amban');
    getDOM('inYear').value=2026; getDOM('inMonth').value=8; getDOM('inDay').value=26;
    getDOM('solarHour').value=12; getDOM('solarMinute').value=0;
    processAll();
    // mở bảng chi tiết
    if (getDOM('ambanBody').style.display === 'none' ||
        !getDOM('ambanPanel').classList.contains('open')) toggleDetailPanel('amban');
    return countryData[opt.value].name_vi;
  }, want);
  // Thanh tab là lớp phủ cố định, không thuộc bảng — giấu đi để chụp trọn 12
  // dòng; và bỏ giới hạn chiều cao của khung cuộn.
  await p.evaluate(()=>{
    const bar=document.getElementById('tabBar'); if(bar) bar.style.display='none';
    const wrap=document.getElementById('ab-table-wrap');
    if(wrap){ wrap.style.maxHeight='none'; wrap.style.overflow='visible'; }
    const body=document.getElementById('ambanBody');
    if(body){ body.style.maxHeight='none'; body.style.overflow='visible'; }
  });
  await p.waitForTimeout(500);
  const el = await p.$('#ambanPanel');
  const file = path.join(OUT, `amban-${want.replace(/\s+/g,'')}.png`);
  await el.screenshot({ path: file });
  console.log(`✓ ${name} → ${path.basename(file)}`);
  await ctx.close();
}
await b.close(); server.close();
