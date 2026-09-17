import {ACCESS_COOKIE,accessIsConfigured,accessSessionToken,validAccessCode,validAccessSession,requestCookie} from '@/lib/access';

export async function GET(request:Request){
  if(!accessIsConfigured())return Response.json({error:'Access protection is not configured.'},{status:503});
  return Response.json({authorized:await validAccessSession(requestCookie(request,ACCESS_COOKIE))});
}

export async function POST(request:Request){
  if(!accessIsConfigured())return Response.json({error:'Access protection is not configured.'},{status:503});
  if(Number(request.headers.get('content-length')||0)>2048)return Response.json({error:'Invalid request.'},{status:413});
  let body:{code?:unknown};try{body=await request.json() as {code?:unknown}}catch{return Response.json({error:'Invalid request.'},{status:400})}
  if(typeof body.code!=='string'||body.code.length>200||!validAccessCode(body.code))return Response.json({error:'Incorrect access code.'},{status:401});
  const response=Response.json({authorized:true});
  response.headers.set('Set-Cookie',`${ACCESS_COOKIE}=${await accessSessionToken()}; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=2592000`);
  return response;
}

export async function DELETE(){
  const response=Response.json({authorized:false});
  response.headers.set('Set-Cookie',`${ACCESS_COOKIE}=; Path=/; HttpOnly; Secure; SameSite=Lax; Max-Age=0`);
  return response;
}
