#!/usr/bin/env python3
from __future__ import annotations
import concurrent.futures,csv,gzip,hashlib,json,threading,time,urllib.parse,urllib.request
from datetime import datetime,timezone,timedelta
from decimal import Decimal
from pathlib import Path

API="https://api.toobit.com"
UA={"User-Agent":"AlphaValue-Toobit-public-data-collector/1.0"}
PRE=datetime(2023,6,1,tzinfo=timezone.utc)
START=datetime(2024,1,1,tzinfo=timezone.utc)-timedelta(hours=1)
END=datetime(2026,8,31,tzinfo=timezone.utc)
PRE_MS=int(PRE.timestamp()*1000); START_MS=int(START.timestamp()*1000); END_MS=int(END.timestamp()*1000)
TOP15=["ADA","CRV","DYDX","ATOM","1INCH","UNI","XRP","DASH","FIL","ARB","LDO","LTC","AAVE","GALA","LINK"]
_lock=threading.Lock(); _calls=[]

def throttle(rate=8):
    while True:
        with _lock:
            now=time.monotonic()
            while _calls and now-_calls[0]>=1:_calls.pop(0)
            if len(_calls)<rate:
                _calls.append(now);return
            delay=1-(now-_calls[0])+0.01
        time.sleep(max(delay,0.01))

def getj(path,params=None,retries=5):
    url=API+path
    if params:url+="?"+urllib.parse.urlencode(params)
    last=None
    for k in range(retries):
        try:
            throttle()
            with urllib.request.urlopen(urllib.request.Request(url,headers=UA),timeout=35) as r:
                b=r.read(16*1024*1024+1)
            if len(b)>16*1024*1024:raise RuntimeError("BODY_TOO_LARGE")
            return json.loads(b)
        except Exception as e:
            last=e;time.sleep(min(5,.5*(k+1)))
    raise last

def arr(o):
    if isinstance(o,list):return o
    if isinstance(o,dict):
        for k in ("data","list","rows"):
            if isinstance(o.get(k),list):return o[k]
    raise RuntimeError("ARRAY_SCHEMA")

def funding(root):
    sym=root+"-SWAP-USDT";out={};periods={};cursor=None;pages=0;prev=None
    while pages<12:
        p={"symbol":sym,"limit":1000}
        if cursor is not None:p["fromId"]=cursor
        xs=arr(getj("/api/v1/futures/historyFundingRate",p));pages+=1
        rows=[]
        for r in xs:
            if not isinstance(r,dict) or str(r.get("symbol"))!=sym:continue
            rid=int(r["id"]);t=int(r["settleTime"]);rate=Decimal(str(r["settleRate"]));period=str(r.get("period") or "UNKNOWN")
            if not rate.is_finite():raise RuntimeError("FUNDING_RATE")
            rows.append((rid,t,str(rate),period))
        if not rows:break
        rows.sort(key=lambda z:z[1]);earliest=rows[0][1];latest=rows[-1][1]
        if prev is not None and latest>=prev:raise RuntimeError("CURSOR_NOT_OLDER")
        for rid,t,rate,period in rows:
            periods[period]=periods.get(period,0)+1
            if PRE_MS<=t<END_MS:
                old=out.get(t);cur=(rid,rate,period)
                if old is not None and old!=cur:raise RuntimeError("FUNDING_CONFLICT")
                out[t]=cur
        if earliest<=PRE_MS:break
        cursor=min(z[0] for z in rows)-1;prev=earliest
    vals=[(t,*out[t]) for t in sorted(out)]
    return vals,{"pages":pages,"period_counts":periods,"rows":len(vals)}

def regular(sym):
    out={};calls=0;cur=START
    while cur<END:
        nxt=min(cur+timedelta(hours=900),END)
        xs=arr(getj("/quote/v1/klines",{"symbol":sym,"interval":"1h",
            "startTime":int(cur.timestamp()*1000),"endTime":int(nxt.timestamp()*1000)-1,"limit":1000}))
        calls+=1
        for r in xs:
            if not isinstance(r,(list,tuple)) or len(r)<5:continue
            try:
                t=int(r[0]);op,hi,lo,cl=map(float,r[1:5])
                if not START_MS<=t<END_MS:continue
                if min(op,hi,lo,cl)<=0 or hi<max(op,cl) or lo>min(op,cl):continue
                row=(op,hi,lo,cl,str(r[5]) if len(r)>5 else "",str(r[7]) if len(r)>7 else "")
                old=out.get(t)
                if old is not None and old!=row:raise RuntimeError("KLINE_CONFLICT")
                out[t]=row
            except RuntimeError:raise
            except Exception:continue
        cur=nxt
    return [(t,*out[t]) for t in sorted(out)],calls

def mark(sym):
    out={};calls=0;cur=START
    while cur<END:
        nxt=min(cur+timedelta(hours=1800),END)
        xs=arr(getj("/quote/v1/markPrice/klines",{"symbol":sym,"interval":"1h",
            "from":int(cur.timestamp()*1000),"to":int(nxt.timestamp()*1000)-1,"limit":2000}))
        calls+=1
        for r in xs:
            if not isinstance(r,dict):continue
            try:
                t=int(r["time"]);op=float(r["open"]);hi=float(r["high"]);lo=float(r["low"]);cl=float(r["close"])
                if not START_MS<=t<END_MS:continue
                if min(op,hi,lo,cl)<=0 or hi<max(op,cl) or lo>min(op,cl):continue
                row=(op,hi,lo,cl)
                old=out.get(t)
                if old is not None and old!=row:raise RuntimeError("MARK_CONFLICT")
                out[t]=row
            except RuntimeError:raise
            except Exception:continue
        cur=nxt
    return [(t,*out[t]) for t in sorted(out)],calls

def write_gz(path,header,rows):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with gzip.open(path,"wt",newline="",encoding="utf-8") as f:
        w=csv.writer(f);w.writerow(header);w.writerows(rows)
    return hashlib.sha256(path.read_bytes()).hexdigest()

def one(root):
    d=Path("toobit_raw")/root;d.mkdir(parents=True,exist_ok=False)
    audit={"root":root,"spot_symbol":root+"USDT","perp_symbol":root+"-SWAP-USDT",
           "outcome_blind":True,"policy_executed":False,"economic_outcomes_opened":False}
    try:
        f,fm=funding(root);s,sc=regular(root+"USDT");p,pc=regular(root+"-SWAP-USDT");m,mc=mark(root+"-SWAP-USDT")
        audit.update(status="COMPLETE",funding_meta=fm,spot_calls=sc,perp_calls=pc,mark_calls=mc,
                     funding_rows=len(f),spot_rows=len(s),perp_rows=len(p),mark_rows=len(m))
        audit["sha256"]={
          "funding":write_gz(d/"funding.csv.gz",["timestamp_ms","id","funding_rate","period"],f),
          "spot":write_gz(d/"spot.csv.gz",["timestamp_ms","open","high","low","close","volume","quote_volume"],s),
          "perp":write_gz(d/"perp.csv.gz",["timestamp_ms","open","high","low","close","volume","quote_volume"],p),
          "mark":write_gz(d/"mark.csv.gz",["timestamp_ms","open","high","low","close"],m)
        }
    except Exception as e:
        audit.update(status="ERROR",error_type=type(e).__name__,error=str(e)[:300])
    (d/"AUDIT.json").write_text(json.dumps(audit,indent=2,sort_keys=True)+"\n")
    print("TOOBIT_RAW_ASSET:"+json.dumps(audit,sort_keys=True),flush=True)
    return audit

def main():
    audits=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=5) as ex:
        futs={ex.submit(one,r):r for r in TOP15}
        for f in concurrent.futures.as_completed(futs):audits.append(f.result())
    audits.sort(key=lambda z:TOP15.index(z["root"]))
    summary={"roots":TOP15,"complete":sum(x["status"]=="COMPLETE" for x in audits),
             "errors":[x["root"] for x in audits if x["status"]!="COMPLETE"],
             "outcome_blind":True,"policy_executed":False,"economic_outcomes_opened":False}
    Path("toobit_raw/SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    print("TOOBIT_RAW_FINAL:"+json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__":main()
