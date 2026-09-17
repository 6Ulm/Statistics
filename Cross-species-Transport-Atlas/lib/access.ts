export const ACCESS_COOKIE='transport_atlas_access';

function accessCode(){return process.env.ACCESS_CODE?.trim()||''}

function constantTimeEqual(left:string,right:string){
  const length=Math.max(left.length,right.length);let difference=left.length^right.length;
  for(let index=0;index<length;index++)difference|=(left.charCodeAt(index)||0)^(right.charCodeAt(index)||0);
  return difference===0;
}

export function accessIsConfigured(){return Boolean(accessCode())}

export function validAccessCode(value:string){const configured=accessCode();return Boolean(configured)&&constantTimeEqual(value,configured)}

export async function accessSessionToken(){
  const configured=accessCode();if(!configured)return'';
  const bytes=new TextEncoder().encode(`cross-species-transport-atlas:${configured}`);
  const digest=await crypto.subtle.digest('SHA-256',bytes);
  return [...new Uint8Array(digest)].map(value=>value.toString(16).padStart(2,'0')).join('');
}

export async function validAccessSession(value:string|undefined){const expected=await accessSessionToken();return Boolean(expected&&value)&&constantTimeEqual(value!,expected)}

export function requestCookie(request:Request,name:string){
  const cookies=request.headers.get('cookie')||'';
  for(const item of cookies.split(';')){const separator=item.indexOf('=');if(separator<0)continue;const key=item.slice(0,separator).trim();if(key===name)return decodeURIComponent(item.slice(separator+1).trim())}
}
