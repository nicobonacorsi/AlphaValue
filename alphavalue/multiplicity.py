"""Multiplicity tools for prespecified AlphaValue research analyses.

The primary anomaly-zoo economic labels use simultaneous time-uniform confidence
sets with a conservative familywise budget.  This module adds a *secondary*,
fixed-cut e-BH discovery layer.  It does not replace the simultaneous coverage
claim and it must not be repeatedly stopped across locally constructed e-processes
without a separate global-filtration argument.
"""
from __future__ import annotations

import math
from typing import Iterable

import numpy as np

from .unknown_scale import nig_log_mixture_constant


def e_bh(evalues: Iterable[float], alpha: float = 0.05) -> dict:
    """Apply e-BH at one prespecified analysis time.

    For m nonnegative e-values sorted decreasingly, choose the largest k with
    e_(k) >= m/(alpha*k).  All hypotheses with e_i at least that threshold are
    rejected.  Fixed-time e-BH controls FDR under arbitrary dependence of valid
    input e-values.  This function makes no sequential/global-filtration claim.
    """
    if not math.isfinite(alpha) or not 0 < alpha < 1:
        raise ValueError("alpha must lie in (0,1)")
    e=np.asarray(list(evalues),dtype=float)
    if e.ndim!=1 or e.size<1:
        raise ValueError("evalues must be a nonempty one-dimensional sequence")
    if np.isnan(e).any() or (e<0).any():
        raise ValueError("evalues must be nonnegative and not NaN")
    m=int(e.size)
    order=np.argsort(-e,kind='mergesort')
    sorted_e=e[order]
    kstar=0
    for k,val in enumerate(sorted_e,1):
        if val >= m/(alpha*k):
            kstar=k
    if kstar==0:
        mask=np.zeros(m,dtype=bool); threshold=math.inf
    else:
        threshold=m/(alpha*kstar)
        mask=e>=threshold
    return {
        'alpha':float(alpha),'m':m,'k':int(mask.sum()),'k_star':int(kstar),
        'threshold':float(threshold) if math.isfinite(threshold) else None,
        'rejected':mask.tolist(),'order':order.tolist(),
        'scope':('Fixed-cut FDR discovery layer for valid e-values. It does not replace the '
                 'primary familywise time-uniform confidence labels and is not an anytime-valid '
                 'stopped e-BH claim without a verified global-filtration condition.'),
    }


def nig_fixed_cut_coefficient_evalue(predictor, response, beta0: float = 0.0, *, tuning=None) -> float:
    """Nuisance-variance-safe fixed-cut e-value for H0: beta=beta0.

    The numerator is the proper NIG marginal likelihood already used by AlphaValue.
    The denominator is the supremum of the Gaussian likelihood over the unknown
    positive variance with beta fixed at beta0.  Under every null variance the
    denominator dominates the true null likelihood, hence the ratio is an e-value
    at this deterministic cut.  It is deliberately *not* advertised as an e-process.
    """
    from .unknown_scale import NIGMixtureTuning
    if tuning is None:
        tuning=NIGMixtureTuning()
    if not math.isfinite(beta0):
        raise ValueError('beta0 must be finite')
    s=nig_log_mixture_constant(predictor,response,tuning=tuning)
    n=int(s['n']); q=float(s['design_energy']); bh=float(s['beta_hat']); rss=float(s['rss_min'])
    rss0=rss+q*(float(beta0)-bh)**2
    if rss0 <= 0:
        return 0.0
    log_e=float(s['log_mixture_constant'] + 0.5*n*(1.0+math.log(rss0/n)))
    # Avoid overflow while preserving the valid extended-real e-value.
    return math.exp(log_e) if log_e < 709 else math.inf
