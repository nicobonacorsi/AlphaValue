"""PM-facing Alpha Survival Frontier report.

This module deliberately reports model-based *certifiability feasibility*, not a
realized-P&L prediction.  The benchmark is the zero-hurdle exponentially decaying
alpha model from the CAC survival theorem.
"""
from __future__ import annotations

import html
import json
import math
from pathlib import Path

from .cac import (
    certifiability_ratio,
    fixed_time_reliable_capacity_lower,
    minimum_exponential_half_life,
    multiplicity_adjusted_alpha,
    search_breadth_capacity,
)


def alpha_survival_report(
    *,
    sharpe: float,
    half_life: float,
    alpha: float = 0.05,
    power: float = 0.90,
    multiplicity: int = 1,
    correction: str = "bonferroni",
    time_unit: str = "years",
) -> dict:
    """Build a one-signal Alpha Survival report.

    ``sharpe`` and ``half_life`` must use consistent time units.  In the
    continuous-time benchmark, S=mu0/sigma and

        I_life = S^2 h / (4 log 2).

    This is a model-implied information budget, not an empirical estimate of
    future P&L and not a substitute for the full unknown-scale certificate.
    """
    S=abs(float(sharpe)); h=float(half_life); a=float(alpha); p=float(power)
    m=int(multiplicity); correction=str(correction).lower(); unit=str(time_unit)
    if not (S>0 and h>0 and 0<a<1 and 0<p<1 and m>=1):
        raise ValueError("require |sharpe|>0, half_life>0, alpha/power in (0,1), multiplicity>=1")
    beta=1.0-p
    I=S*S*h/(4.0*math.log(2.0))
    cr=certifiability_ratio(I,a,beta,multiplicity=m,correction=correction)
    hmin=minimum_exponential_half_life(S,a,beta,multiplicity=m,correction=correction)
    per_alpha=multiplicity_adjusted_alpha(a,m,method=correction)
    fixed=fixed_time_reliable_capacity_lower(per_alpha,beta,2.0*I)
    breadth=search_breadth_capacity(I,a,beta,method=correction)
    Icrit=float(cr["I_crit_nats"])
    gap=I-Icrit
    # Raw leftover information is not the same as optimal CAC; label it only as
    # a planning upper-budget fraction.
    leftover=max(0.0,gap/I) if I>0 else 0.0
    if cr["status"]=="INFEASIBLE":
        headline="CANNOT BE RELIABLY CERTIFIED BEFORE INFORMATION EXHAUSTION"
    elif cr["status"]=="BOUNDARY":
        headline="AT THE RELIABLE-CERTIFICATION BOUNDARY"
    else:
        headline="RELIABLE CERTIFICATION IS FEASIBLE IN THE CANONICAL MODEL"
    return {
        "report":"ALPHA_SURVIVAL_FRONTIER",
        "status":cr["status"],
        "headline":headline,
        "inputs":{
            "instantaneous_sharpe":S,
            "alpha_half_life":h,
            "time_unit":unit,
            "family_false_deployment_alpha":a,
            "required_power":p,
            "candidate_multiplicity":m,
            "multiplicity_correction":correction,
        },
        "information":{
            "lifetime_information_nats":I,
            "required_information_nats":Icrit,
            "information_surplus_nats":gap,
            "certifiability_ratio":float(cr["certifiability_ratio"]),
            "per_candidate_alpha":per_alpha,
        },
        "survival_frontier":{
            "minimum_half_life":float(hmin["half_life_min"]),
            "half_life_shortfall":max(0.0,float(hmin["half_life_min"])-h),
            "half_life_surplus":max(0.0,h-float(hmin["half_life_min"])),
            "max_candidate_breadth_supported":breadth["max_candidates"],
        },
        "economic_capacity_planning":{
            "constructive_fixed_time_certified_value_fraction_lower":float(fixed["fraction_lower"]),
            "constructive_test_information_time":fixed["test_time"],
            "constructive_test_power":float(fixed["power"]),
            "raw_information_left_after_threshold_fraction":leftover,
        },
        "interpretation":[
            "CR<1 is an impossibility statement only inside the declared canonical decay model and multiplicity rule.",
            "CR>1 means statistical feasibility, not guaranteed positive realized P&L.",
            "The constructive value fraction is a lower bound for one fixed-time Neyman-Pearson strategy, not the exact finite-horizon optimal CAC.",
            "Use the full AlphaValue unknown-scale certificate when inference must be learned from raw signal/return data rather than supplied Sharpe/half-life inputs.",
        ],
        "model":"zero-hurdle exponential alpha decay; deterministic total information budget",
    }


def _fmt(x, digits=3):
    if isinstance(x,int): return str(x)
    if isinstance(x,float):
        if math.isinf(x): return "∞"
        return f"{x:.{digits}f}"
    return str(x)


def render_survival_html(report: dict, output: Path) -> Path:
    """Write a self-contained PM-facing HTML card."""
    output=Path(output)
    i=report["inputs"]; inf=report["information"]; sf=report["survival_frontier"]; ec=report["economic_capacity_planning"]
    cr=float(inf["certifiability_ratio"])
    meter=max(0.0,min(100.0,100.0*cr))
    status=html.escape(report["status"]); headline=html.escape(report["headline"])
    unit=html.escape(i["time_unit"])
    # Deliberately no traffic-light promise language: the card describes model status.
    doc=f'''<!doctype html>
<html><head><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1">
<title>Alpha Survival Report</title>
<style>
body{{font-family:Inter,ui-sans-serif,system-ui,-apple-system,Segoe UI,sans-serif;background:#f6f7f9;color:#101318;margin:0;padding:32px}}
.card{{max-width:900px;margin:auto;background:white;border:1px solid #e4e7eb;border-radius:18px;padding:30px;box-shadow:0 8px 32px rgba(0,0,0,.06)}}
h1{{margin:0 0 4px;font-size:30px}} .sub{{color:#59636e;margin-bottom:26px}} .headline{{font-size:20px;font-weight:750;margin:18px 0}}
.grid{{display:grid;grid-template-columns:repeat(auto-fit,minmax(190px,1fr));gap:12px}} .metric{{border:1px solid #e7e9ed;border-radius:12px;padding:15px}}
.k{{font-size:12px;text-transform:uppercase;letter-spacing:.06em;color:#67717c}} .v{{font-size:24px;font-weight:720;margin-top:4px}}
.meter{{height:14px;border-radius:9px;background:#eceff2;overflow:hidden;margin:12px 0 4px}} .fill{{height:100%;width:{meter:.2f}%;background:#20242a}}
.small{{font-size:13px;color:#67717c;line-height:1.45}} table{{width:100%;border-collapse:collapse;margin-top:20px}} td{{padding:9px 0;border-bottom:1px solid #eceff2}} td:last-child{{text-align:right;font-weight:650}}
.note{{margin-top:24px;padding:16px;border-radius:12px;background:#f7f8fa;font-size:13px;line-height:1.5}} code{{background:#f2f4f6;padding:2px 5px;border-radius:5px}}
</style></head><body><div class="card">
<h1>Alpha Survival Report</h1><div class="sub">Will the alpha survive long enough to be certified?</div>
<div class="headline">{headline}</div>
<div class="grid">
<div class="metric"><div class="k">Certifiability ratio</div><div class="v">{_fmt(cr,2)}×</div></div>
<div class="metric"><div class="k">Lifetime information</div><div class="v">{_fmt(inf['lifetime_information_nats'],2)} nats</div></div>
<div class="metric"><div class="k">Required information</div><div class="v">{_fmt(inf['required_information_nats'],2)} nats</div></div>
<div class="metric"><div class="k">Minimum half-life</div><div class="v">{_fmt(sf['minimum_half_life'],2)} {unit}</div></div>
</div>
<div class="meter"><div class="fill"></div></div><div class="small">CR = lifetime information / required reliable-certification information. The bar caps visually at 1×.</div>
<table>
<tr><td>Instantaneous Sharpe</td><td>{_fmt(i['instantaneous_sharpe'],2)}</td></tr>
<tr><td>Observed/modelled half-life</td><td>{_fmt(i['alpha_half_life'],2)} {unit}</td></tr>
<tr><td>Family false-deployment budget</td><td>{100*i['family_false_deployment_alpha']:.2f}%</td></tr>
<tr><td>Required power</td><td>{100*i['required_power']:.1f}%</td></tr>
<tr><td>Candidate strategies screened</td><td>{i['candidate_multiplicity']}</td></tr>
<tr><td>Per-candidate alpha ({html.escape(i['multiplicity_correction'])})</td><td>{100*inf['per_candidate_alpha']:.4f}%</td></tr>
<tr><td>Constructive certified-value fraction lower bound</td><td>{100*ec['constructive_fixed_time_certified_value_fraction_lower']:.1f}%</td></tr>
<tr><td>Maximum candidate breadth supported at same information</td><td>{_fmt(sf['max_candidate_breadth_supported'])}</td></tr>
</table>
<div class="note"><strong>Scope.</strong> This is a theorem-driven feasibility report for the canonical zero-hurdle exponential-decay experiment. <strong>{status}</strong> is not a forecast of realized trading P&amp;L. A full data-driven AlphaValue certificate additionally estimates model uncertainty, persistence, noise scales and economic value.</div>
</div></body></html>'''
    output.write_text(doc,encoding="utf-8")
    return output


def write_report_json(report:dict,output:Path)->Path:
    output=Path(output); output.write_text(json.dumps(report,indent=2,allow_nan=False),encoding='utf-8'); return output
