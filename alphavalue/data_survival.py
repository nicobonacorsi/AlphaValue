"""Data-driven Alpha Survival diagnostics.

The key identification boundary is deliberate: past realized returns alone do not
identify a *future* alpha half-life.  A finite-lifetime estimate therefore requires
an explicit structural decay model and an ex-ante edge path (expected return in the
same per-period units as the realized return), or a user-supplied half-life.

The exponential fit and moving-block bootstrap below are model-based diagnostics,
not an anytime-valid certificate.  They are intended to feed the theorem-driven
Alpha Survival Frontier without silently inventing a lifetime from a backtest.
"""
from __future__ import annotations

import csv
import html
import math
from pathlib import Path
from typing import Iterable

import numpy as np
from scipy.optimize import minimize_scalar, brentq
from scipy.stats import nct

from .survival_report import alpha_survival_report


def _as_finite_array(values: Iterable[float], name: str) -> np.ndarray:
    x=np.asarray(list(values),dtype=float)
    if x.ndim!=1 or x.size==0 or not np.all(np.isfinite(x)):
        raise ValueError(f"{name} must be a nonempty finite one-dimensional series")
    return x


def fit_exponential_edge(edge: Iterable[float], *, max_decay_rate: float | None=None) -> dict:
    """Least-squares fit edge_t = amplitude * exp(-lambda*t) + error.

    The amplitude is profiled analytically for every lambda>=0.  ``lambda`` is in
    inverse observation periods.  A boundary solution lambda=0 means that a finite
    decay half-life is not identified by the fitted model; it must not be silently
    converted into an infinite-certainty claim.
    """
    y=_as_finite_array(edge,"edge")
    n=y.size
    if n<8:
        raise ValueError("at least 8 finite edge observations are required")
    t=np.arange(n,dtype=float)
    if max_decay_rate is None:
        # permits half-lives down to one quarter of an observation period
        max_decay_rate=math.log(2.0)/0.25
    max_decay_rate=float(max_decay_rate)
    if not (max_decay_rate>0 and math.isfinite(max_decay_rate)):
        raise ValueError("max_decay_rate must be positive and finite")

    def profile(lam: float):
        w=np.exp(-lam*t)
        den=float(np.dot(w,w))
        amp=max(0.0,float(np.dot(w,y))/den) if den>0 else 0.0
        resid=y-amp*w
        return float(np.dot(resid,resid)),amp

    opt=minimize_scalar(lambda lam: profile(float(lam))[0],bounds=(0.0,max_decay_rate),method="bounded",options={"xatol":1e-12})
    lam=float(opt.x); rss,amp=profile(lam)
    # explicitly compare to boundary lambda=0 because bounded scalar minimizers do
    # not always return the exact endpoint.
    rss0,amp0=profile(0.0)
    if rss0 <= rss*(1.0+1e-10):
        lam=0.0; rss=rss0; amp=amp0
    fitted=amp*np.exp(-lam*t)
    resid=y-fitted
    sst=float(np.dot(y-y.mean(),y-y.mean()))
    r2=float(1.0-rss/sst) if sst>0 else (1.0 if rss==0 else float("nan"))
    # Gaussian quasi-BICs, used only as a descriptive model-diagnostics comparison.
    tiny=np.finfo(float).tiny
    bic_exp=n*math.log(max(rss/n,tiny))+2.0*math.log(n)
    const=float(np.mean(y)); rss_const=float(np.dot(y-const,y-const))
    bic_const=n*math.log(max(rss_const/n,tiny))+1.0*math.log(n)
    return {
        "n":int(n),"amplitude_per_period":float(amp),"decay_rate_per_period":lam,
        "half_life_periods":(math.log(2.0)/lam if lam>1e-12 else math.inf),
        "rss":rss,"r2":r2,"bic_exponential":float(bic_exp),"bic_constant":float(bic_const),
        "delta_bic_exp_vs_constant":float(bic_const-bic_exp),
        "fitted":fitted,"residuals":resid,
        "finite_decay_identified":bool(lam>1e-12),
    }


def _circular_block_indices(n:int, block_length:int, rng:np.random.Generator)->np.ndarray:
    if n<1 or block_length<1:
        raise ValueError("n and block_length must be positive")
    out=[]
    while len(out)<n:
        start=int(rng.integers(0,n))
        out.extend((start+j)%n for j in range(block_length))
    return np.asarray(out[:n],dtype=int)


def analyze_edge_return_series(
    returns: Iterable[float],
    edge: Iterable[float],
    *,
    periods_per_unit: float,
    alpha: float=0.05,
    power: float=0.90,
    multiplicity: int=1,
    correction: str="bonferroni",
    time_unit: str="years",
    bootstrap_samples: int=1000,
    bootstrap_level: float=0.95,
    block_length: int|None=None,
    seed: int=2718,
) -> dict:
    """Fit an ex-ante exponential edge path and map it to the Survival Frontier.

    ``edge`` must be a pre-outcome expected-return/edge path constructed without future outcome data, in the same *per-period*
    units as ``returns``.  The point model is

        edge_t = a exp(-lambda t) + noise,
        return_t = edge_t + innovation_t.

    The moving-block bootstrap is an approximate dependence-aware diagnostic.  It
    does not have the finite-sample coverage guarantee of the formal G2 certificate.
    """
    r=_as_finite_array(returns,"returns"); e=_as_finite_array(edge,"edge")
    if r.size!=e.size:
        raise ValueError("returns and edge must have equal length")
    n=r.size; P=float(periods_per_unit)
    if n<24:
        raise ValueError("at least 24 paired observations are required for data-driven survival analysis")
    if not (P>0 and math.isfinite(P)):
        raise ValueError("periods_per_unit must be positive and finite")
    if not (0<bootstrap_level<1):
        raise ValueError("bootstrap_level must lie in (0,1)")
    if int(bootstrap_samples)<0:
        raise ValueError("bootstrap_samples must be nonnegative")
    fit=fit_exponential_edge(e)
    eps=r-e
    sigma=float(np.std(eps,ddof=1))
    if not (sigma>0 and math.isfinite(sigma)):
        raise ValueError("realized return minus ex-ante edge must have positive finite sample standard deviation")
    amp=float(fit["amplitude_per_period"]); lam=float(fit["decay_rate_per_period"])
    S=amp/sigma*math.sqrt(P)
    h=(math.log(2.0)/lam/P) if lam>1e-12 else math.inf

    point=None
    if math.isfinite(h) and S>0:
        point=alpha_survival_report(sharpe=S,half_life=h,alpha=alpha,power=power,multiplicity=multiplicity,correction=correction,time_unit=time_unit)

    if block_length is None:
        block_length=max(2,int(round(n**(1.0/3.0))))
    L=int(block_length)
    if not (1<=L<=n):
        raise ValueError("block_length must be between 1 and n")

    bs=[]
    B=int(bootstrap_samples)
    if B>0:
        rng=np.random.default_rng(int(seed))
        fitted=np.asarray(fit["fitted"],float); u=e-fitted
        pair=np.column_stack([u,eps])
        for _ in range(B):
            idx=_circular_block_indices(n,L,rng)
            ub=pair[idx,0]; eb_noise=pair[idx,1]
            edge_b=fitted+ub
            try:
                fb=fit_exponential_edge(edge_b)
                lb=float(fb["decay_rate_per_period"])
                ab=float(fb["amplitude_per_period"])
                sigb=float(np.std(eb_noise,ddof=1))
                if not (lb>1e-12 and ab>0 and sigb>0 and math.isfinite(sigb)):
                    continue
                Sb=ab/sigb*math.sqrt(P)
                hb=math.log(2.0)/lb/P
                Ib=Sb*Sb*hb/(4.0*math.log(2.0))
                if math.isfinite(Ib) and Ib>=0:
                    bs.append((Sb,hb,Ib,lb,ab,sigb))
            except (ValueError,FloatingPointError,OverflowError):
                continue
    robust=None
    if bs:
        arr=np.asarray(bs,float)
        qlo=1.0-float(bootstrap_level)
        # direct lower quantile of lifetime information avoids pretending that
        # marginal parameter bounds combine into an exact joint confidence set.
        I_lower=float(np.quantile(arr[:,2],qlo))
        I_med=float(np.median(arr[:,2]))
        I_upper=float(np.quantile(arr[:,2],bootstrap_level))
        # Required information does not depend on observed inputs; obtain from a
        # unit report to keep one implementation of multiplicity semantics.
        ref=alpha_survival_report(sharpe=1.0,half_life=1.0,alpha=alpha,power=power,multiplicity=multiplicity,correction=correction,time_unit=time_unit)
        Icrit=float(ref["information"]["required_information_nats"])
        cr_lo=I_lower/Icrit
        robust={
            "method":"circular moving-block residual bootstrap",
            "formal_certificate":False,
            "bootstrap_samples_requested":B,"bootstrap_samples_usable":int(arr.shape[0]),
            "block_length":L,"level":float(bootstrap_level),"seed":int(seed),
            "lifetime_information_nats_lower":I_lower,
            "lifetime_information_nats_median":I_med,
            "lifetime_information_nats_upper":I_upper,
            "robust_certifiability_ratio_lower":float(cr_lo),
            "status":"MODEL_ROBUST_FEASIBLE" if cr_lo>=1.0 else "MODEL_ROBUST_INFEASIBLE",
            "sharpe_lower":float(np.quantile(arr[:,0],qlo)),
            "half_life_lower":float(np.quantile(arr[:,1],qlo)),
        }

    finite=bool(fit["finite_decay_identified"])
    observed_span=(n-1)/P
    extrapolation_ratio=(h/observed_span) if (math.isfinite(h) and observed_span>0) else None
    weak_decay_evidence=bool(finite and fit["delta_bic_exp_vs_constant"] <= 2.0)
    if amp<=0:
        data_status="NO_POSITIVE_EDGE"
    elif not finite:
        data_status="FINITE_LIFETIME_UNIDENTIFIED"
    elif weak_decay_evidence:
        data_status="DECAY_MODEL_WEAKLY_IDENTIFIED"
    elif point is None:
        data_status="UNRESOLVED"
    else:
        data_status="POINT_"+str(point["status"])
    return {
        "report":"DATA_DRIVEN_ALPHA_SURVIVAL",
        "data_status":data_status,
        "formal_certificate":False,
        "n_observations":int(n),
        "inputs":{"periods_per_unit":P,"time_unit":str(time_unit),"alpha":float(alpha),"power":float(power),"candidate_multiplicity":int(multiplicity),"correction":str(correction)},
        "decay_fit":{k:v for k,v in fit.items() if k not in {"fitted","residuals"}},
        "return_noise":{"innovation_sd_per_period":sigma},
        "implied":{"instantaneous_sharpe":float(S),"half_life":float(h) if math.isfinite(h) else None,
                   "observed_span":float(observed_span),"half_life_to_observed_span":float(extrapolation_ratio) if extrapolation_ratio is not None else None,
                   "extrapolation_warning":bool(extrapolation_ratio is not None and extrapolation_ratio>1.0)},
        "point_survival_report":point,
        "bootstrap_robustness":robust,
        "identification_boundary":{
            "returns_only_lifetime_identified":False,
            "statement":"Past realized returns alone do not identify a future alpha half-life without a structural continuation model. This report estimates lifetime only because an ex-ante edge path and exponential-decay model were declared.",
        },
        "interpretation":[
            "The edge column must be ex ante: it may not be reconstructed using the same future returns being evaluated.",
            "The exponential decay fit is a structural model, not a distribution-free fact about the strategy.",
            "The moving-block bootstrap is an approximate robustness diagnostic, not an anytime-valid confidence sequence or deployment certificate.",
            "A fitted half-life longer than the observed sample span is flagged as extrapolative rather than treated as established persistence.",
            "For formal data-adaptive inference use AlphaValue's G2 unknown-scale certificate with a declared economic null and lifetime model.",
        ],
    }


def load_edge_return_csv(path:Path, *, return_column:str="return", edge_column:str="edge") -> tuple[np.ndarray,np.ndarray]:
    path=Path(path)
    with path.open("r",encoding="utf-8-sig",newline="") as f:
        reader=csv.DictReader(f)
        if not reader.fieldnames or return_column not in reader.fieldnames or edge_column not in reader.fieldnames:
            raise ValueError(f"CSV must contain columns {return_column!r} and {edge_column!r}")
        r=[]; e=[]
        for row in reader:
            try:
                rv=float(row[return_column]); ev=float(row[edge_column])
            except (TypeError,ValueError):
                continue
            if math.isfinite(rv) and math.isfinite(ev):
                r.append(rv); e.append(ev)
    if not r:
        raise ValueError("CSV contains no finite paired return/edge rows")
    return np.asarray(r,float),np.asarray(e,float)


def gaussian_iid_sharpe_lower_bound(
    returns: Iterable[float],
    *,
    periods_per_unit: float,
    confidence: float = 0.95,
) -> dict:
    """Exact one-sided lower confidence bound for Sharpe under iid Gaussian returns.

    If T=sqrt(n)*mean/sd, then under iid Gaussian returns T is noncentral-t
    with noncentrality sqrt(n)*mu/sigma. Inverting that distribution gives
    an exact lower confidence bound under this declared model. It is not a
    dependence-robust or distribution-free statement.
    """
    r=_as_finite_array(returns,"returns")
    n=int(r.size); P=float(periods_per_unit)
    if n<3:
        raise ValueError("at least 3 returns are required for a Sharpe confidence bound")
    if not (P>0 and math.isfinite(P)):
        raise ValueError("periods_per_unit must be positive and finite")
    if not (0.5 < confidence < 1.0):
        raise ValueError("confidence must lie in (0.5,1)")
    sd=float(np.std(r,ddof=1))
    if not (sd>0 and math.isfinite(sd)):
        raise ValueError("returns must have positive finite sample standard deviation")
    mean=float(np.mean(r)); df=n-1
    t_obs=math.sqrt(n)*mean/sd
    target=float(confidence)
    def g(delta: float) -> float:
        val=float(nct.cdf(t_obs,df,delta))
        if not math.isfinite(val):
            # Symmetry fallback avoids an occasional SciPy nct.cdf NaN for
            # large negative noncentralities while preserving the same CDF.
            val=float(nct.sf(-t_obs,df,-delta))
        return val-target
    lo,hi=-8.0,8.0
    while g(lo)<0 and lo>-1e6:
        lo*=2.0
    while g(hi)>0 and hi<1e6:
        hi*=2.0
    if not (g(lo)>=0 and g(hi)<=0):
        raise ValueError("failed to bracket noncentral-t Sharpe confidence bound")
    delta_lo=float(brentq(g,lo,hi,xtol=1e-12,rtol=1e-12,maxiter=200))
    point=mean/sd*math.sqrt(P)
    lower=delta_lo/math.sqrt(n)*math.sqrt(P)
    return {
        "model":"iid Gaussian returns / exact noncentral-t inversion",
        "confidence":float(confidence),
        "annualized_sharpe_point":float(point),
        "annualized_sharpe_lower":float(lower),
        "t_statistic":float(t_obs),
        "df":int(df),
        "formal_under_declared_model":True,
        "dependence_robust":False,
        "statement":"Exact one-sided Sharpe lower confidence bound under iid Gaussian returns; it does not identify future alpha lifetime and is not robust to serial dependence or non-Gaussian misspecification.",
    }


def returns_only_identification_report(
    returns:Iterable[float],
    *,
    periods_per_unit:float,
    time_unit:str="years",
    strength_confidence:float=0.95,
) -> dict:
    r=_as_finite_array(returns,"returns")
    if r.size<8:
        raise ValueError("at least 8 returns are required")
    sd=float(np.std(r,ddof=1)); mean=float(np.mean(r)); P=float(periods_per_unit)
    if sd<=0 or P<=0:
        raise ValueError("positive return variation and periods_per_unit are required")
    strength=gaussian_iid_sharpe_lower_bound(r,periods_per_unit=P,confidence=strength_confidence)
    return {
        "report":"RETURNS_ONLY_SURVIVAL_IDENTIFICATION",
        "status":"LIFETIME_UNIDENTIFIED",
        "formal_certificate":False,
        "n_observations":int(r.size),
        "historical_sharpe":mean/sd*math.sqrt(P),
        "historical_strength":strength,
        "time_unit":str(time_unit),
        "statement":"Historical returns can summarize past strength, but without a structural continuation/decay model they do not identify the future alpha half-life required by the Alpha Survival Frontier.",
        "next_step":"Provide a pre-outcome edge column constructed without future outcome data, or supply a predeclared half-life/model rather than estimating future lifetime from realized returns alone.",
        "interpretation":[
            "The reported Sharpe lower bound is exact only under the declared iid Gaussian return model.",
            "No confidence statement for future half-life is produced from realized returns alone.",
            "Because lifetime is unidentified, no lifetime information or Certifiability Ratio is reported in returns-only mode.",
        ],
    }


def render_returns_only_survival_html(report:dict, output:Path)->Path:
    output=Path(output)
    hs=report["historical_strength"]
    def f(x,d=2):
        return f"{float(x):.{d}f}"
    doc=(
        '<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">'
        '<title>Alpha Survival - Lifetime unidentified</title>'
        '<style>body{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:#f5f6f8;color:#11151a;margin:0;padding:28px}'
        '.card{max-width:900px;margin:auto;background:#fff;border:1px solid #e2e5e9;border-radius:18px;padding:28px;box-shadow:0 8px 30px rgba(0,0,0,.06)}'
        'h1{margin:0}.sub{color:#5f6873;margin:6px 0 24px}.status{font-size:26px;font-weight:760;margin:18px 0;padding:16px;border:1px solid #e5e8ec;border-radius:12px}'
        '.grid{display:grid;grid-template-columns:repeat(auto-fit,minmax(210px,1fr));gap:12px}.m{border:1px solid #e5e8ec;border-radius:12px;padding:14px}'
        '.k{font-size:12px;text-transform:uppercase;color:#6b7480;letter-spacing:.05em}.v{font-size:23px;font-weight:730;margin-top:5px}'
        '.note{margin-top:22px;background:#f7f8fa;border-radius:12px;padding:15px;line-height:1.55;font-size:13px}</style></head><body><div class="card">'
        '<h1>Alpha Survival Report</h1><div class="sub">Historical strength can be estimated; future alpha lifetime cannot be inferred from returns alone.</div>'
        '<div class="status">LIFETIME_UNIDENTIFIED</div>'
        f'<div class="grid"><div class="m"><div class="k">Historical Sharpe</div><div class="v">{f(report["historical_sharpe"])}</div></div>'
        f'<div class="m"><div class="k">95% lower Sharpe</div><div class="v">{f(hs["annualized_sharpe_lower"])}</div></div>'
        f'<div class="m"><div class="k">Observations</div><div class="v">{report["n_observations"]}</div></div></div>'
        f'<div class="note"><strong>Identification boundary.</strong> {html.escape(report["statement"])}<br><br>'
        f'<strong>Strength bound.</strong> {html.escape(hs["statement"])}<br><br>'
        '<strong>What AlphaValue refuses to do.</strong> It does not convert this historical Sharpe into an invented future half-life, lifetime-information budget, or Certifiability Ratio.</div>'
        '</div></body></html>'
    )
    output.write_text(doc,encoding="utf-8")
    return output


def render_data_survival_html(report:dict, output:Path)->Path:
    output=Path(output)
    fit=report["decay_fit"]; imp=report["implied"]; rob=report.get("bootstrap_robustness"); point=report.get("point_survival_report")
    point_cr=(point or {}).get("information",{}).get("certifiability_ratio")
    point_I=(point or {}).get("information",{}).get("lifetime_information_nats")
    req_I=(point or {}).get("information",{}).get("required_information_nats")
    def f(x,d=3):
        if x is None:return "—"
        if isinstance(x,float) and math.isinf(x):return "∞"
        return f"{float(x):.{d}f}" if isinstance(x,(float,int,np.floating,np.integer)) else html.escape(str(x))
    robust_status=rob["status"] if rob else "UNAVAILABLE"
    robust_cr=rob["robust_certifiability_ratio_lower"] if rob else None
    doc=f'''<!doctype html><html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>Data-driven Alpha Survival Report</title>
<style>body{{font-family:Inter,system-ui,-apple-system,Segoe UI,sans-serif;background:#f5f6f8;color:#11151a;margin:0;padding:28px}}.card{{max-width:980px;margin:auto;background:#fff;border:1px solid #e2e5e9;border-radius:18px;padding:28px;box-shadow:0 8px 30px rgba(0,0,0,.06)}}h1{{margin:0}}.sub{{color:#5f6873;margin:6px 0 24px}}.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}}.m{{border:1px solid #e5e8ec;border-radius:12px;padding:14px}}.k{{font-size:12px;text-transform:uppercase;color:#6b7480;letter-spacing:.05em}}.v{{font-size:23px;font-weight:730;margin-top:5px}}table{{width:100%;border-collapse:collapse;margin-top:22px}}td{{padding:9px 0;border-bottom:1px solid #edf0f2}}td:last-child{{text-align:right;font-weight:650}}.note{{margin-top:22px;background:#f7f8fa;border-radius:12px;padding:15px;line-height:1.5;font-size:13px}}</style></head><body><div class="card">
<h1>Data-driven Alpha Survival Report</h1><div class="sub">Raw observations → declared decay model → survival feasibility</div>
<div class="grid"><div class="m"><div class="k">Point CR</div><div class="v">{f(point_cr,2)}×</div></div><div class="m"><div class="k">Bootstrap lower CR</div><div class="v">{f(robust_cr,2)}×</div></div><div class="m"><div class="k">Implied Sharpe</div><div class="v">{f(imp['instantaneous_sharpe'],2)}</div></div><div class="m"><div class="k">Implied half-life</div><div class="v">{f(imp['half_life'],2)} {html.escape(report['inputs']['time_unit'])}</div></div></div>
<table><tr><td>Observations</td><td>{report['n_observations']}</td></tr><tr><td>Decay rate / period</td><td>{f(fit['decay_rate_per_period'],5)}</td></tr><tr><td>Decay fit R²</td><td>{f(fit['r2'],3)}</td></tr><tr><td>ΔBIC exponential vs constant</td><td>{f(fit['delta_bic_exp_vs_constant'],2)}</td></tr><tr><td>Point lifetime information</td><td>{f(point_I,2)} nats</td></tr><tr><td>Required information</td><td>{f(req_I,2)} nats</td></tr><tr><td>Bootstrap robustness</td><td>{html.escape(robust_status)}</td></tr></table>
<div class="note"><strong>Identification boundary.</strong> Past realized returns alone cannot identify future alpha lifetime. This report is only available because the input contains a pre-outcome edge path constructed without future outcome data and the exponential-decay model was declared. The block bootstrap is a model-robustness diagnostic, not a formal CAC deployment certificate.</div>
</div></body></html>'''
    output.write_text(doc,encoding="utf-8")
    return output
