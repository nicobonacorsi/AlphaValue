#!/usr/bin/env python3
import json, urllib.parse, urllib.request
from datetime import datetime, timezone
API="https://api.toobit.com"; UA={"User-Agent":"AlphaValue-Toobit-kline-diag/1.0"}
SYM_SPOT="ADAUSDT"; SYM_PERP="ADA-SWAP-USDT"
A=int(datetime(2024,1,1,tzinfo=timezone.utc).timestamp()*1000)
B=int(datetime(2024,1,3,tzinfo=timezone.utc).timestamp()*1000)
def arr(o):
  if isinstance(o,list):return o
  if isinstance(o,dict):
    for k in ("data","list","rows"):
      if isinstance(o.get(k),list):return o[k]
  return []
def get(path,p):
  u=API+path+"?"+urllib.parse.urlencode(p)
  with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=30) as r:o=json.loads(r.read())
  xs=arr(o); ts=[]
  for x in xs:
    try:
      if isinstance(x,dict): ts.append(int(x.get("time") or x.get("timestamp") or x.get("t")))
      else: ts.append(int(x[0]))
    except: pass
  return {"rows":len(xs),"first":min(ts) if ts else None,"last":max(ts) if ts else None}
cases=[
 ("regular_latest","/quote/v1/klines",{"symbol":SYM_PERP,"interval":"1h","limit":1000}),
 ("regular_start_end","/quote/v1/klines",{"symbol":SYM_PERP,"interval":"1h","startTime":A,"endTime":B-1,"limit":1000}),
 ("regular_from_to","/quote/v1/klines",{"symbol":SYM_PERP,"interval":"1h","from":A,"to":B-1,"limit":1000}),
 ("regular_end","/quote/v1/klines",{"symbol":SYM_PERP,"interval":"1h","endTime":B-1,"limit":1000}),
 ("regular_start","/quote/v1/klines",{"symbol":SYM_PERP,"interval":"1h","startTime":A,"limit":1000}),
 ("spot_start_end","/quote/v1/klines",{"symbol":SYM_SPOT,"interval":"1h","startTime":A,"endTime":B-1,"limit":1000}),
 ("mark_latest","/quote/v1/markPrice/klines",{"symbol":SYM_PERP,"interval":"1h","limit":1000}),
 ("mark_start_end","/quote/v1/markPrice/klines",{"symbol":SYM_PERP,"interval":"1h","startTime":A,"endTime":B-1,"limit":1000}),
 ("mark_from_to","/quote/v1/markPrice/klines",{"symbol":SYM_PERP,"interval":"1h","from":A,"to":B-1,"limit":1000}),
 ("mark_end","/quote/v1/markPrice/klines",{"symbol":SYM_PERP,"interval":"1h","endTime":B-1,"limit":1000}),
]
out=[]
for name,path,p in cases:
  try:r=get(path,p);r.update(name=name,params=p)
  except Exception as e:r={"name":name,"params":p,"error":type(e).__name__+":"+str(e)[:200]}
  out.append(r); print("KLINE_DIAG:"+json.dumps(r,sort_keys=True),flush=True)
open("toobit_kline_diag.json","w").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
