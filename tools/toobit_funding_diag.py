#!/usr/bin/env python3
import json, urllib.parse, urllib.request
from datetime import datetime, timezone

API="https://api.toobit.com"
UA={"User-Agent":"AlphaValue-Toobit-funding-diag/1.0"}
SYMS=["BTC-SWAP-USDT","SOL-SWAP-USDT","DOGE-SWAP-USDT"]
WINS=[
 ("2024A","2024-01-01T00:00:00+00:00","2024-01-11T00:00:00+00:00"),
 ("2026A","2026-08-01T00:00:00+00:00","2026-08-11T00:00:00+00:00")
]
def ms(x): return int(datetime.fromisoformat(x).timestamp()*1000)
def get(sym,a,b):
    p=urllib.parse.urlencode({"symbol":sym,"startTime":ms(a),"endTime":ms(b)-1,"limit":1000})
    req=urllib.request.Request(API+"/api/v1/futures/historyFundingRate?"+p,headers=UA)
    with urllib.request.urlopen(req,timeout=30) as r:o=json.loads(r.read())
    xs=o if isinstance(o,list) else o.get("data",[])
    ys=[x for x in xs if isinstance(x,dict) and x.get("symbol")==sym]
    ts=sorted(int(x["settleTime"]) for x in ys)
    periods={}
    for x in ys: periods[str(x.get("period") or "UNKNOWN")]=periods.get(str(x.get("period") or "UNKNOWN"),0)+1
    return {"rows":len(ys),"first":ts[0] if ts else None,"last":ts[-1] if ts else None,"periods":periods}
out=[]
for s in SYMS:
  for tag,a,b in WINS:
    try:r=get(s,a,b);r.update(symbol=s,window=tag)
    except Exception as e:r={"symbol":s,"window":tag,"error":type(e).__name__+":"+str(e)[:200]}
    out.append(r);print("DIAG:"+json.dumps(r,sort_keys=True),flush=True)
open("toobit_funding_diag.json","w").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
