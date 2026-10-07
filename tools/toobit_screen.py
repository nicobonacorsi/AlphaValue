#!/usr/bin/env python3
from __future__ import annotations
import bisect, concurrent.futures, csv, json, math, threading, time, urllib.parse, urllib.request
from collections import deque
from datetime import datetime, timezone, timedelta
from decimal import Decimal
from pathlib import Path

API="https://api.toobit.com"
UA={"User-Agent":"AlphaValue-Toobit-public-transport-screen/1.0"}
PRE=datetime(2023,6,1,tzinfo=timezone.utc)
START=datetime(2024,1,1,tzinfo=timezone.utc)
END=datetime(2026,8,31,tzinfo=timezone.utc)
PRE_MS=int(PRE.timestamp()*1000); START_MS=int(START.timestamp()*1000); END_MS=int(END.timestamp()*1000)
HOUR=3_600_000; STEP=8*HOUR; HIST=540; HORIZON=21; COOLDOWN=21
EXCLUDED={"BTC","ETH"}
_lock=threading.Lock(); _calls=[]

def throttle(rate=8):
    while True:
        with _lock:
            now=time.monotonic()
            while _calls and now-_calls[0]>=1:_calls.pop(0)
            if len(_calls)<rate:
                _calls.append(now); return
            delay=1-(now-_calls[0])+0.01
        time.sleep(max(delay,0.01))

def getj(path,params=None,retries=5):
    q=urllib.parse.urlencode(params or {})
    url=API+path+("?" + q if q else "")
    last=None
    for k in range(retries):
        try:
            throttle()
            req=urllib.request.Request(url,headers=UA)
            with urllib.request.urlopen(req,timeout=35) as r:
                body=r.read(16*1024*1024+1)
            if len(body)>16*1024*1024: raise RuntimeError("BODY_TOO_LARGE")
            return json.loads(body)
        except Exception as e:
            last=e; time.sleep(min(5,0.5*(k+1)))
    raise last

def arr(obj):
    if isinstance(obj,list): return obj
    if isinstance(obj,dict):
        for k in ("data","list","rows"):
            if isinstance(obj.get(k),list): return obj[k]
    raise ValueError("ARRAY_SCHEMA")

def universe():
    o=getj("/api/v1/exchangeInfo")
    spots={str(x.get("symbol")) for x in (o.get("symbols") or [])
           if str(x.get("status","")).upper()=="TRADING" and str(x.get("quoteAsset","")).upper()=="USDT"}
    out=[]
    for x in o.get("contracts") or []:
        if str(x.get("status","")).upper()!="TRADING": continue
        if bool(x.get("inverse",False)): continue
        if str(x.get("quoteAsset","")).upper()!="USDT" or str(x.get("marginToken","")).upper()!="USDT": continue
        root=str(x.get("underlying") or "").upper()
        sym=str(x.get("symbol") or "")
        if not root or root in EXCLUDED or root+"USDT" not in spots or not sym.endswith("-SWAP-USDT"): continue
        out.append({"root":root,"spot":root+"USDT","perp":sym})
    out.sort(key=lambda z:(z["root"],z["perp"]))
    seen=set(); ans=[]
    for z in out:
        if z["root"] in seen: continue
        seen.add(z["root"]); ans.append(z)
    return ans

def funding_history(sym):
    by_time={}; periods={}; calls=0
    cursor=None; pages=0; previous_earliest=None
    while pages<12:
        params={"symbol":sym,"limit":1000}
        if cursor is not None: params["fromId"]=cursor
        xs=arr(getj("/api/v1/futures/historyFundingRate",params)); calls+=1; pages+=1
        page=[]
        for r in xs:
            if not isinstance(r,dict) or str(r.get("symbol"))!=sym: continue
            rid=int(r["id"]); t=int(r["settleTime"]); rate=Decimal(str(r["settleRate"])); period=str(r.get("period") or "")
            if not rate.is_finite(): raise RuntimeError("FUNDING_RATE")
            page.append((rid,t,rate,period or "UNKNOWN"))
        if not page: break
        page.sort(key=lambda x:x[1])
        earliest=page[0][1]; latest=page[-1][1]
        if previous_earliest is not None and latest>=previous_earliest:
            raise RuntimeError("FUNDING_CURSOR_NOT_STRICTLY_OLDER")
        for rid,t,rate,period in page:
            periods[period]=periods.get(period,0)+1
            if not PRE_MS<=t<END_MS: continue
            if period!="8H": continue
            old=by_time.get(t)
            if old is not None and old!=rate: raise RuntimeError("FUNDING_CONFLICT")
            by_time[t]=rate
        if earliest<=PRE_MS: break
        min_id=min(x[0] for x in page)
        if min_id<=0: break
        cursor=min_id-1
        previous_earliest=earliest
    xs=sorted(by_time.items())
    return xs,calls,periods

def q95(vals):
    pos=Decimal(len(vals)-1)*Decimal("0.95")
    lo=int(pos); hi=lo if pos==lo else lo+1
    if lo==hi:return vals[lo]
    w=pos-Decimal(lo)
    return vals[lo]*(Decimal(1)-w)+vals[hi]*w

def monday(ms):
    d=datetime.fromtimestamp(ms/1000,timezone.utc)
    z=datetime(d.year,d.month,d.day,tzinfo=timezone.utc)-timedelta(days=d.weekday())
    return z.isoformat()

def scan_births(xs):
    n=len(xs)
    grids=[t//HOUR*HOUR for t,_ in xs]; rates=[r for _,r in xs]
    gaps=[None]+[grids[i]-grids[i-1] for i in range(1,n)]
    run=[0]*n
    for i in range(1,n): run[i]=run[i-1]+1 if gaps[i]==STEP else 0
    best=max(run,default=0)+1 if n else 0
    if n<HIST+HORIZON+1:
        return {"records":n,"longest_8h":best,"raw_triggers":0,"stable21_births":0,"weeks":[],"birth_times":[]}
    win=deque(); ordered=[]; next_start=HIST; stable=[]; raw=0
    for i in range(n):
        if i>=HIST and i>=next_start and run[i]>=HIST:
            cur=rates[i]; th=q95(ordered)
            if cur>0 and cur+Decimal("1e-18")>=th:
                next_start=i+COOLDOWN+1
                g=grids[i]
                if START_MS<=g<END_MS and g+7*86400000<END_MS:
                    raw+=1
                    if i+HORIZON<n and all(gaps[j]==STEP for j in range(i+1,i+HORIZON+1)):
                        stable.append(g)
        v=abs(rates[i]); bisect.insort(ordered,v); win.append(v)
        if len(win)>HIST:
            old=win.popleft(); ordered.pop(bisect.bisect_left(ordered,old))
    return {"records":n,"longest_8h":best,"raw_triggers":raw,"stable21_births":len(stable),
            "weeks":sorted({monday(t) for t in stable}),
            "birth_times":[datetime.fromtimestamp(t/1000,timezone.utc).isoformat() for t in stable]}

def regular_times(sym,start,end):
    xs=arr(getj("/quote/v1/klines",{"symbol":sym,"interval":"1h","startTime":start,"endTime":end-1,"limit":1000}))
    good=set()
    for r in xs:
        if not isinstance(r,(list,tuple)) or len(r)<5: continue
        try:
            t=int(r[0]); o,h,l,c=map(float,r[1:5])
            if min(o,h,l,c)>0 and h>=max(o,c) and l<=min(o,c): good.add(t)
        except Exception: pass
    return good

def mark_times(sym,start,end):
    xs=arr(getj("/quote/v1/markPrice/klines",{"symbol":sym,"interval":"1h","from":start,"to":end-1,"limit":2000}))
    good=set()
    for r in xs:
        if not isinstance(r,dict): continue
        try:
            t=int(r["time"]); o=float(r["open"]); h=float(r["high"]); l=float(r["low"]); c=float(r["close"])
            if min(o,h,l,c)>0 and h>=max(o,c) and l<=min(o,c): good.add(t)
        except Exception: pass
    return good

def audit(z):
    rec=dict(z)
    try:
        xs,calls,periods=funding_history(z["perp"])
        b=scan_births(xs)
        rec.update(status="COMPLETE",funding_calls=calls,funding_period_counts=periods,**b)
        rec["funding_depth_pass"]=bool(b["records"]>=HIST and b["longest_8h"]>=HIST)
        if rec["funding_depth_pass"]:
            anchor=min(END_MS-24*HOUR,max(START_MS,(xs[-1][0]-7*86400000)//(24*HOUR)*(24*HOUR)))
            exp=set(range(anchor,anchor+24*HOUR,HOUR))
            s=regular_times(z["spot"],anchor,anchor+24*HOUR)
            p=regular_times(z["perp"],anchor,anchor+24*HOUR)
            m=mark_times(z["perp"],anchor,anchor+24*HOUR)
            rec.update(price_probe_start_ms=anchor,spot_hours=len(s&exp),perp_hours=len(p&exp),mark_hours=len(m&exp),
                       spot_exact_24h=(s&exp)==exp,perp_exact_24h=(p&exp)==exp,mark_exact_24h=(m&exp)==exp)
            rec["transport_pass"]=bool(rec["spot_exact_24h"] and rec["perp_exact_24h"] and rec["mark_exact_24h"])
        else:
            rec["transport_pass"]=False
    except Exception as e:
        rec.update(status="ERROR",error_type=type(e).__name__,error=str(e)[:300],transport_pass=False,
                   funding_depth_pass=False,records=0,longest_8h=0,stable21_births=0,weeks=[],birth_times=[])
    return rec

def selftest():
    base=START_MS-HIST*STEP
    xs=[]
    for i in range(HIST+HORIZON+6):
        xs.append((base+i*STEP,Decimal("0.02") if i==HIST else Decimal("0.001")))
    z=scan_births(xs); assert z["stable21_births"]==1,z
    print("SELFTEST_PASS")

def main():
    selftest()
    u=universe()
    print("TOOBIT_UNIVERSE",len(u),flush=True)
    reports=[]
    with concurrent.futures.ThreadPoolExecutor(max_workers=6) as ex:
        futs={ex.submit(audit,z):z["root"] for z in u}
        for i,f in enumerate(concurrent.futures.as_completed(futs),1):
            r=f.result(); reports.append(r)
            print("TOOBIT_PROGRESS:"+json.dumps({"done":i,"total":len(u),"root":r["root"],"status":r["status"],
                "transport_pass":r.get("transport_pass"),"records":r.get("records"),"longest_8h":r.get("longest_8h"),
                "stable21_births":r.get("stable21_births")},sort_keys=True),flush=True)
    reports.sort(key=lambda r:r["root"])
    eligible=[r for r in reports if r.get("transport_pass") is True]
    births=[]
    for r in eligible:
        for t in r.get("birth_times",[]):
            ms=int(datetime.fromisoformat(t).timestamp()*1000)
            births.append((r["root"],r["perp"],t,monday(ms)))
    weeks=sorted({x[3] for x in births})
    ranking=sorted([{"root":r["root"],"perp":r["perp"],"stable21_births":int(r.get("stable21_births",0))}
                    for r in eligible if int(r.get("stable21_births",0))>0],
                   key=lambda x:(-x["stable21_births"],x["root"]))
    for i,x in enumerate(ranking,1):x["rank"]=i
    outdir=Path("toobit_screen"); outdir.mkdir()
    summary={"outcome_blind":True,"economic_outcomes_opened":False,"policy_executed":False,
             "current_survivor_candidates":len(u),"transport_complete_assets":len(eligible),
             "transport_errors":sum(r["status"]=="ERROR" for r in reports),
             "stable21_upper_births":len(births),"distinct_candidate_birth_weeks":len(weeks),
             "candidate_birth_weeks":weeks,"ranking_size":len(ranking)}
    (outdir/"SUMMARY.json").write_text(json.dumps(summary,indent=2,sort_keys=True)+"\n")
    (outdir/"ASSET_REPORTS.json").write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n")
    (outdir/"FROZEN_ACTION_RANKING.json").write_text(json.dumps(ranking,indent=2,sort_keys=True)+"\n")
    with (outdir/"UPPER_BIRTHS.csv").open("w",newline="") as fh:
        w=csv.writer(fh);w.writerow(["root","perp","birth_time","birth_week_utc"]);w.writerows(births)
    print("TOOBIT_SCREEN_FINAL:"+json.dumps(summary,sort_keys=True),flush=True)

if __name__=="__main__": main()

