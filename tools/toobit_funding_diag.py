#!/usr/bin/env python3
import json, urllib.parse, urllib.request
API="https://api.toobit.com"; UA={"User-Agent":"AlphaValue-Toobit-cursor-diag/1.0"}
SYMS=["BTC-SWAP-USDT","SOL-SWAP-USDT"]
def call(sym,params):
    q={"symbol":sym,"limit":1000};q.update(params)
    u=API+"/api/v1/futures/historyFundingRate?"+urllib.parse.urlencode(q)
    with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=30) as r:o=json.loads(r.read())
    xs=o if isinstance(o,list) else o.get("data",[])
    ys=[x for x in xs if isinstance(x,dict) and x.get("symbol")==sym]
    ids=sorted(int(x["id"]) for x in ys);ts=sorted(int(x["settleTime"]) for x in ys)
    return {"rows":len(ys),"min_id":ids[0] if ids else None,"max_id":ids[-1] if ids else None,
            "first":ts[0] if ts else None,"last":ts[-1] if ts else None,
            "periods":sorted(set(str(x.get("period")) for x in ys))}
out=[]
for s in SYMS:
    base=call(s,{})
    out.append({"symbol":s,"mode":"latest",**base})
    cur=(base["min_id"] or 1)-1
    for mode in ["endId","fromId"]:
        try:r=call(s,{mode:cur});r.update(symbol=s,mode=mode,cursor=cur)
        except Exception as e:r={"symbol":s,"mode":mode,"cursor":cur,"error":type(e).__name__+":"+str(e)[:200]}
        out.append(r)
for x in out: print("CURSOR:"+json.dumps(x,sort_keys=True),flush=True)
open("toobit_cursor_diag.json","w").write(json.dumps(out,indent=2,sort_keys=True)+"\n")
