'use client';

import {FormEvent,useState} from 'react';
import {Activity,LoaderCircle,LockKeyhole} from 'lucide-react';

export function AccessGate({configurationError=false}:{configurationError?:boolean}){
  const[code,setCode]=useState(''),[submitting,setSubmitting]=useState(false),[error,setError]=useState(configurationError?'Access protection is temporarily unavailable.':'');
  async function submit(event:FormEvent){
    event.preventDefault();if(!code||submitting||configurationError)return;setSubmitting(true);setError('');
    try{const response=await fetch('/api/access',{method:'POST',headers:{'Content-Type':'application/json'},body:JSON.stringify({code})});const payload=await response.json() as {error?:string};if(!response.ok)throw Error(payload.error||'Access denied.');window.location.reload()}
    catch(reason){setError(reason instanceof Error?reason.message:'Access denied.');setSubmitting(false)}
  }
  return <main className="access-gate"><section><div className="access-brand"><span><Activity size={24}/></span><div><b>Cross-species Transport Atlas</b><small>STRING physical networks · coclustering · optimal transport</small></div></div><div className="access-lock"><LockKeyhole size={24}/></div><h1>Enter the access code</h1><p>This research application is shared by link and protected with a code.</p><form onSubmit={submit}><label htmlFor="atlas-access-code">Access code</label><input id="atlas-access-code" type="password" value={code} onChange={event=>setCode(event.target.value)} autoComplete="current-password" autoFocus disabled={configurationError} aria-invalid={Boolean(error)}/>{error&&<div className="access-error" role="alert">{error}</div>}<button type="submit" disabled={!code||submitting||configurationError}>{submitting?<LoaderCircle className="spin" size={18}/>:<LockKeyhole size={18}/>} {submitting?'Checking…':'Open atlas'}</button></form></section></main>
}
