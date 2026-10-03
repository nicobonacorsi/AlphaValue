"""Classical Sharpe-based validation diagnostics used as AlphaValue comparators.

These are comparator statistics, not AlphaValue certificates.  Formulas follow the
Probabilistic Sharpe Ratio / Minimum Track Record Length and Deflated Sharpe Ratio
literature.  Sharpe ratios are per-observation; callers are responsible for consistent
frequency units.
"""
from __future__ import annotations
import math
import numpy as np
from scipy.stats import norm, skew as scipy_skew, kurtosis as scipy_kurtosis

EULER_MASCHERONI = 0.5772156649015329


def sharpe_moments(returns) -> dict:
    r=np.asarray(returns,dtype=float)
    r=r[np.isfinite(r)]
    n=int(r.size)
    if n<4:
        raise ValueError("at least four finite observations are required")
    sd=float(np.std(r,ddof=1))
    if not math.isfinite(sd) or sd<=0:
        raise ValueError("returns must have positive finite sample standard deviation")
    sr=float(np.mean(r)/sd)
    sk=float(scipy_skew(r,bias=False))
    ku=float(scipy_kurtosis(r,fisher=False,bias=False))
    return {"n":n,"sharpe":sr,"skewness":sk,"kurtosis":ku}


def sharpe_variance_factor(sharpe: float, skewness: float, kurtosis: float) -> float:
    """Numerator in the asymptotic variance of the sample Sharpe ratio."""
    q=1.0-float(skewness)*float(sharpe)+((float(kurtosis)-1.0)/4.0)*float(sharpe)**2
    return float(max(q,0.0))


def probabilistic_sharpe_ratio(*, sharpe: float, n: int, benchmark: float=0.0,
                               skewness: float=0.0, kurtosis: float=3.0) -> float:
    if n<2:
        raise ValueError("n must be at least 2")
    q=sharpe_variance_factor(sharpe,skewness,kurtosis)
    if q<=0:
        return float(sharpe>benchmark)
    z=(float(sharpe)-float(benchmark))*math.sqrt(n-1)/math.sqrt(q)
    return float(norm.cdf(z))


def minimum_track_record_length(*, sharpe: float, benchmark: float=0.0,
                                confidence: float=0.95, skewness: float=0.0,
                                kurtosis: float=3.0) -> float:
    if not 0<confidence<1:
        raise ValueError("confidence must lie in (0,1)")
    gap=float(sharpe)-float(benchmark)
    if gap<=0:
        return math.inf
    q=sharpe_variance_factor(sharpe,skewness,kurtosis)
    return float(1.0+q*(float(norm.ppf(confidence))/gap)**2)


def expected_maximum_sharpe(*, n_trials: int, sharpe_variance: float) -> float:
    """Expected-maximum null benchmark used by DSR.

    For one prespecified trial we use 0 as a practical convention; the Gumbel
    approximation itself is defined for n_trials > 1.
    """
    if n_trials<1:
        raise ValueError("n_trials must be positive")
    if sharpe_variance<0 or not math.isfinite(sharpe_variance):
        raise ValueError("sharpe_variance must be finite and nonnegative")
    if n_trials==1 or sharpe_variance==0:
        return 0.0
    n=float(n_trials)
    g=EULER_MASCHERONI
    mx=(1-g)*norm.ppf(1-1/n)+g*norm.ppf(1-1/(n*math.e))
    return float(math.sqrt(sharpe_variance)*mx)


def deflated_sharpe_ratio(*, sharpe: float, n: int, n_trials: int,
                          trial_sharpe_variance: float, skewness: float=0.0,
                          kurtosis: float=3.0) -> dict:
    sr0=expected_maximum_sharpe(n_trials=n_trials,sharpe_variance=trial_sharpe_variance)
    dsr=probabilistic_sharpe_ratio(sharpe=sharpe,n=n,benchmark=sr0,
                                   skewness=skewness,kurtosis=kurtosis)
    return {"dsr":float(dsr),"expected_maximum_sharpe":float(sr0)}
