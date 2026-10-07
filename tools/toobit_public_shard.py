#!/usr/bin/env python3
from __future__ import annotations
import json,os,time,urllib.error,urllib.parse,urllib.request
from datetime import datetime,timezone
from pathlib import Path

API="https://api.toobit.com"
UA={"User-Agent":"Toobit-public-sharded-fetch/1.0"}
PRE=int(datetime(2023,6,1,tzinfo=timezone.utc).timestamp()*1000)
END=int(datetime(2026,8,31,tzinfo=timezone.utc).timestamp()*1000)
PROBE_START=int(datetime(2026,8,1,tzinfo=timezone.utc).timestamp()*1000)
PROBE_END=PROBE_START+24*3600*1000
HOUR=3600*1000
times=[]

def throttle(n=4):
    while True:
        now=time.monotonic()
        while times and now-times[0]>=1: times.pop(0)
        if len(times)<n:
            times.append(now);return
        time.sleep(max(.02,1-(now-times[0])+.01))

def getj(path,p=None,retries=6):
    u=API+path+("?" + urllib.parse.urlencode(p or {}) if p else "")
    last=None
    for k in range(retries):
        try:
            throttle()
            with urllib.request.urlopen(urllib.request.Request(u,headers=UA),timeout=35) as r:
                return json.loads(r.read()),u
        except urllib.error.HTTPError as e:
            last=e
            if e.code not in (408,429,500,502,503,504):break
        except Exception as e:last=e
        time.sleep(min(8,.7*(k+1)))
    raise last if last else RuntimeError("TRANSPORT")

def arr(o):
    if isinstance(o,list):return o
    if isinstance(o,dict):
        for k in ("data","list","rows"):
            if isinstance(o.get(k),list):return o[k]
    raise ValueError("ARRAY_SCHEMA")

def universe():
    o,_=getj("/api/v1/exchangeInfo")
    spots=o.get("symbols") or [];contracts=o.get("contracts") or []
    spotset={str(x.get("symbol")) for x in spots if str(x.get("status","")).upper()=="TRADING" and str(x.get("quoteAsset","")).upper()=="USDT"}
    out=[]
    for x in contracts:
        if str(x.get("status","")).upper()!="TRADING":continue
        if bool(x.get("inverse",False)):continue
        if str(x.get("quoteAsset","")).upper()!="USDT" or str(x.get("marginToken","")).upper()!="USDT":continue
        perp=str(x.get("symbol") or "");suffix="-SWAP-USDT"
        if not perp.endswith(suffix):continue
        root=perp[:-len(suffix)];spot=root+"USDT"
        if root and spot in spotset:out.append({"root":root,"spot_symbol":spot,"perp_symbol":perp})
    best={}
    for r in sorted(out,key=lambda z:(z["root"],z["perp_symbol"])):best.setdefault(r["root"],r)
    return list(best.values())

def parse_fund(o,symbol):
    by={}
    for r in arr(o):
        if not isinstance(r,dict) or str(r.get("symbol"))!=symbol:continue
        try:z={"id":int(r["id"]),"settleTime":int(r["settleTime"]),"settleRate":str(r["settleRate"]),"period":str(r.get("period") or "")}
        except Exception:continue
        if 10**12<=z["settleTime"]<10**14:by[z["id"]]=z
    return sorted(by.values(),key=lambda r:r["settleTime"])

def funding(symbol):
    o,u=getj("/api/v1/futures/historyFundingRate",{"symbol":symbol,"limit":1000})
    first=parse_fund(o,symbol)
    if not first:return [],{"status":"EMPTY","pages":1,"urls":[u]}
    by={r["id"]:r for r in first};urls=[u];pages=1
    while min(r["settleTime"] for r in by.values())>PRE and pages<12:
        earliest=min(r["settleTime"] for r in by.values());cursor=min(by)-1
        oo,uu=getj("/api/v1/futures/historyFundingRate",{"symbol":symbol,"fromId":cursor,"limit":1000})
        rr=parse_fund(oo,symbol);older=[r for r in rr if r["settleTime"]<earliest]
        if not older:break
        before=len(by)
        for r in rr:by[r["id"]]=r
        urls.append(uu);pages+=1
        if len(by)==before:break
    rows=sorted((r for r in by.values() if PRE<=r["settleTime"]<END),key=lambda r:r["settleTime"])
    ts=[r["settleTime"] for r in rows]
    best=cur=1 if ts else 0
    for a,b in zip(ts,ts[1:]):
        if b-a==8*HOUR:cur+=1
        else:best=max(best,cur);cur=1
    best=max(best,cur)
    return rows,{"status":"OK","pages":pages,"records":len(rows),"first_ms":ts[0] if ts else None,"last_ms":ts[-1] if ts else None,
                 "periods":sorted({r["period"] for r in rows if r["period"]}),"longest_8h":best,"urls":urls}

def regular(symbol):
    o,u=getj("/quote/v1/klines",{"symbol":symbol,"interval":"1h","startTime":PROBE_START,"endTime":PROBE_END-1,"limit":24})
    return {int(r[0]) for r in arr(o) if isinstance(r,(list,tuple)) and len(r)>=5},u

def mark(symbol):
    o,u=getj("/quote/v1/markPrice/klines",{"symbol":symbol,"interval":"1h","from":PROBE_START,"to":PROBE_END-1,"limit":24})
    rows=[r for r in arr(o) if isinstance(r,dict) and r.get("time") is not None and str(r.get("klineType") or "1h")=="1h"]
    return {int(r["time"]) for r in rows},u

def main():
    shard=int(os.environ["SHARD"]);shards=int(os.environ.get("SHARDS","8"))
    uni=universe();assigned=[r for i,r in enumerate(uni) if i%shards==shard]
    out=Path("out");(out/"funding").mkdir(parents=True)
    exp=set(range(PROBE_START,PROBE_END,HOUR));reports=[]
    for i,r in enumerate(assigned,1):
        z=dict(r)
        try:
            fr,fm=funding(r["perp_symbol"]);(out/"funding"/(r["root"]+".json")).write_text(json.dumps(fr,separators=(",",":"))+"\n")
            s,su=regular(r["spot_symbol"]);p,pu=regular(r["perp_symbol"]);m,mu=mark(r["perp_symbol"])
            z.update(status="COMPLETE",funding=fm,price_probe={"spot_rows":len(s),"perp_rows":len(p),"mark_rows":len(m),
                "spot_exact":s==exp,"perp_exact":p==exp,"mark_exact":m==exp,
                "spot_missing":len(exp-s),"perp_missing":len(exp-p),"mark_missing":len(exp-m),
                "spot_url":su,"perp_url":pu,"mark_url":mu})
        except Exception as e:z.update(status="ERROR",error_type=type(e).__name__,error=str(e)[:300])
        reports.append(z);print("TOOBIT_SHARD_PROGRESS:"+json.dumps({"shard":shard,"i":i,"n":len(assigned),"root":r["root"],"status":z["status"],
            "records":(z.get("funding") or {}).get("records"),"longest_8h":(z.get("funding") or {}).get("longest_8h"),
            "price":z.get("price_probe")},sort_keys=True),flush=True)
    (out/"reports.json").write_text(json.dumps(reports,indent=2,sort_keys=True)+"\n")
    (out/"meta.json").write_text(json.dumps({"shard":shard,"shards":shards,"universe_count":len(uni),"assigned_count":len(assigned)},indent=2)+"\n")
    print("TOOBIT_SHARD_FINAL:"+json.dumps({"shard":shard,"assigned":len(assigned),"complete":sum(x.get("status")=="COMPLETE" for x in reports),"errors":sum(x.get("status")=="ERROR" for x in reports)},sort_keys=True),flush=True)
if __name__=="__main__":main()
