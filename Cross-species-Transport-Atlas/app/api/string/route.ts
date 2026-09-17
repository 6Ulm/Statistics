import {ACCESS_COOKIE,requestCookie,validAccessSession} from '@/lib/access';

const hosts:Record<string,string>={preview:'https://preview.string-db.org',v12:'https://version-12-0.string-db.org'};
export async function POST(req:Request){
 try{
  if(!await validAccessSession(requestCookie(req,ACCESS_COOKIE)))return Response.json({error:'Access code required.'},{status:401});
  if(Number(req.headers.get('content-length')||0)>500000)return Response.json({error:'Input is too large.'},{status:413});
  const body=await req.json() as {version:string,action:string,species:string,type:string,score:number,identifiers:unknown,sources?:unknown};const {version,action,species,type,score}=body;const ids=body.identifiers;
  if(!hosts[version]||!['map','network','link','svg'].includes(action)||!/^\d+$/.test(String(species))||!['physical','functional'].includes(type)||!Number.isInteger(score)||score<0||score>1000||!Array.isArray(ids)||ids.length<2||ids.length>2000||ids.some((x:unknown)=>typeof x!=='string'||!x||x.length>200||/[\r\n]/.test(x)))return Response.json({error:'Use 2–2,000 protein identifiers, a numeric species ID and a confidence from 0 to 1000.'},{status:400});
  const allowedSources=type==='physical'?['textmining','experimental','database','transfer']:['neighborhood','fusion','cooccurrence','coexpression','experimental','database','textmining',...(version==='preview'?['homology']:[])];const sources=body.sources===undefined?allowedSources:body.sources;
  if(!Array.isArray(sources)||sources.some(source=>typeof source!=='string'||!allowedSources.includes(source)))return Response.json({error:'Invalid interaction source selection.'},{status:400});
  const endpoint=action==='map'?'json/get_string_ids':action==='network'?'json/network':action==='link'?'tsv-no-header/get_link':'svg/network';
  const params=new URLSearchParams({identifiers:ids.join('\r'),species:String(species),required_score:String(score),network_type:type,network_flavor:'confidence',add_nodes:'0',add_color_nodes:'0',add_white_nodes:'0',hide_disconnected_nodes:'0',caller_identity:'STRING_Cluster_Lens',echo_query:'1'});
  if(type==='physical')params.set('useTransferScores',sources.includes('transfer')?'1':'0');
  const res=await fetch(hosts[version]+'/api/'+endpoint,{method:'POST',headers:{'Content-Type':'application/x-www-form-urlencoded'},body:params,signal:AbortSignal.timeout(90000)});
  const text=await res.text();if(!res.ok) return Response.json({error:'STRING returned '+res.status+'. Check identifiers and species, or try again later.'},{status:502});
  if(action==='svg'){if(!text.includes('<svg'))throw Error('STRING did not return a network image.');return new Response(text,{headers:{'Content-Type':'image/svg+xml','Content-Disposition':'attachment; filename="filtered-string-network.svg"'}})}
  if(action==='link'){const url=text.trim();if(!/^https:\/\/(?:[a-z0-9-]+\.)?string-db\.org\//.test(url))throw Error('STRING did not return a valid network link.');return Response.json({url})}
  let data;try{data=JSON.parse(text)}catch{throw Error('STRING returned an unexpected response. Try again or import a TSV export.')}
  if(!Array.isArray(data))throw Error(data?.message||'STRING returned no data.');return Response.json({data});
 }catch(e){return Response.json({error:e instanceof Error?e.message:'Unable to reach STRING.'},{status:502})}
}
