/**
 * Đối chiếu MÀN HÌNH: dựng cây DOM chỉ gồm phần thực sự nhìn thấy (theo
 * computed style) của cả hai bản rồi so từng phần tử một. Bảo đảm bản Android
 * không thêm/bớt bất cứ thứ gì so với bản web gốc.
 *
 *   node test_visual_parity.mjs /đường/dẫn/QMDJ_1_1.html
 */
import fs from 'fs'; import path from 'path'; import { JSDOM } from 'jsdom';
import { fileURLToPath } from 'url';
const HERE = path.dirname(fileURLToPath(import.meta.url));
const WEB = path.join(HERE, '..', 'app', 'src', 'main', 'assets', 'web');
const ORIG = process.argv[2];
if (!ORIG || !fs.existsSync(ORIG)) {
  console.error('Dùng: node test_visual_parity.mjs <QMDJ_goc.html>');
  process.exit(2);
}
async function boot(html,url,native){
  const dom=new JSDOM(html,{url,runScripts:'dangerously',resources:'usable',pretendToBeVisual:true});
  Object.defineProperty(dom.window.Element.prototype,'innerText',{get(){return this.textContent;},set(v){this.textContent=v;},configurable:true});
  if(native) dom.window.QMDJNative=native;
  await new Promise(r=>dom.window.addEventListener('load',r));
  await new Promise(r=>setTimeout(r,500));
  return dom;
}
const domA=await boot(fs.readFileSync(ORIG,'utf8'),'file://'+ORIG,null);
const domB=await boot(fs.readFileSync(WEB+'/index.html','utf8'),'file://'+WEB+'/index.html',{
  readAsset:p=>fs.readFileSync(path.join(WEB,p),'utf8'),getPref:()=>null,setPref:()=>{},
  deviceTimeZone:()=>'Asia/Ho_Chi_Minh',hasLocationPermission:()=>false,requestLocation:()=>{},platform:()=>'android'});
function setup(dom,c){
  const d=dom.window.document,w=dom.window;
  if(w.currentLang!==c.lang) w.setLang(c.lang);
  for(const[i,v]of[['inYear',c.y],['inMonth',c.m],['inDay',c.d],['solarHour',c.h],['solarMinute',c.mi],['country',c.country],['methodSelect',c.method]]) d.getElementById(i).value=String(v);
  w.updateCountryDisplay();   // đặt .value trực tiếp thì nhãn nút không tự cập nhật
  w.processAll();
}
/** Cây DOM chỉ gồm phần THỰC SỰ nhìn thấy (theo computed style). */
function visible(dom){
  const w=dom.window,d=w.document,out=[];
  (function walk(el,depth){
    if(el.nodeType!==1) return;
    const tag=el.tagName.toLowerCase();
    if(tag==='script'||tag==='style'||tag==='link') return;
    const cs=w.getComputedStyle(el);
    if(cs.display==='none'||cs.visibility==='hidden') return;
    if(el.classList.contains('hidden-select')) return;
    const own=[...el.childNodes].filter(n=>n.nodeType===3).map(n=>n.textContent).join('').replace(/\s+/g,' ').trim();
    out.push('  '.repeat(depth)+`<${tag}${el.id?'#'+el.id:''}${el.className&&typeof el.className==='string'?'.'+el.className.trim().split(/\s+/).join('.'):''}>${own?' "'+own+'"':''}`);
    for(const ch of el.children) walk(ch,depth+1);
  })(d.body,0);
  return out;
}
const CASES=[
 {y:1988,m:3,d:15,h:7,mi:25,country:'VN-HCM',method:'amban',lang:'vi'},
 {y:2026,m:8,d:25,h:23,mi:40,country:'VN-HN',method:'trinhuan',lang:'zh'},
 {y:2000,m:2,d:4,h:12,mi:0,country:'FR',method:'bophap',lang:'vi'},
];
let bad=0;
for(const c of CASES){
  setup(domA,c); setup(domB,c);
  const A=visible(domA),B=visible(domB);
  const label=`${c.y}-${c.m}-${c.d} ${c.country}/${c.method}/${c.lang}`;
  if(A.join('\n')===B.join('\n')){ console.log(`  ok   ${label} — ${A.length} phần tử hiển thị, giống hệt`); }
  else{
    bad++;
    console.log(`  KHÁC ${label}`);
    const setA=new Set(A),setB=new Set(B);
    B.filter(x=>!setA.has(x)).slice(0,10).forEach(x=>console.log('    + '+x.trim()));
    A.filter(x=>!setB.has(x)).slice(0,10).forEach(x=>console.log('    - '+x.trim()));
  }
}
console.log(bad?`\n✗ ${bad} ca khác nhau`:'\n✓ MÀN HÌNH GIỐNG HỆT BẢN WEB GỐC');
process.exit(bad?1:0);
