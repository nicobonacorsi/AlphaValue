"""Unknown-noise anytime-valid joint parameter and economic certificates.

This module extends the known-scale joint certificate without plugging in a
variance estimate.  For a predictable scalar regression

    z_{t+1} = beta * x_t + eps_{t+1},   eps_{t+1}|F_t ~ N(0, variance),

both ``beta`` and the positive constant ``variance`` are unknown.  We test each
point null (beta, variance) with a proper Normal--Inverse-Gamma likelihood
mixture in the numerator.  The resulting Bayes-factor process is a nonnegative
martingale under every point null, so inversion gives a genuinely time-uniform
joint confidence set for (beta, variance).  The exact curved set has convenient
closed-form projections onto beta and variance.

Two such sets, one for (theta, sigma_R^2) and one for (phi, sigma_X^2), are
combined by Bonferroni.  The rectangular product of their coordinate
projections is a conservative outer set for economic certification.

The economic value and oracle value are both proportional to sigma_X^2.  Hence
relative regret cancels the signal innovation variance exactly.  This yields an
anytime-valid ``eta``-efficiency certificate even when neither Gaussian noise
scale is known.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import mpmath as mp
import numpy as np
from scipy.optimize import brentq
from scipy.special import gammaln


@dataclass(frozen=True)
class NIGMixtureTuning:
    """Predeclared proper Normal--Inverse-Gamma alternative mixture.

    We use the parameterization

        variance ~ InvGamma(shape, scale),
        beta | variance ~ N(mean, variance / kappa).

    ``scale`` has the units of a residual sum of squares and all four tuning
    constants must be fixed before the monitored likelihood-ratio process is
    started.  They affect power/width, not validity.
    """

    mean: float = 0.0
    kappa: float = 1.0
    shape: float = 1.0
    scale: float = 1.0

    def validate(self) -> None:
        vals=(self.mean,self.kappa,self.shape,self.scale)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("NIG tuning parameters must be finite")
        if self.kappa<=0 or self.shape<=0 or self.scale<=0:
            raise ValueError("NIG kappa, shape and scale must be strictly positive")


@dataclass(frozen=True)
class UnknownScaleCertificateModel:
    """Economic parameters for an unknown-noise joint certificate."""

    gamma: float
    trading_cost: float
    horizon: int
    terminal_penalty: float = 0.0

    def validate(self) -> None:
        vals=(self.gamma,self.trading_cost,self.terminal_penalty)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("economic parameters must be finite")
        if self.gamma<=0 or self.trading_cost<0 or self.terminal_penalty<0:
            raise ValueError("require gamma>0, trading_cost>=0 and terminal_penalty>=0")
        if not isinstance(self.horizon,int) or isinstance(self.horizon,bool) or self.horizon<1:
            raise ValueError("horizon must be a positive integer")


def _sufficient_statistics(predictor, response) -> tuple[np.ndarray,np.ndarray,int,float,float,float]:
    x=np.asarray(predictor,dtype=float)
    y=np.asarray(response,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or x.shape!=y.shape or len(x)<1:
        raise ValueError("predictor and response must be equal nonempty one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("predictor and response must be finite")
    n=len(x)
    q=float(np.dot(x,x))
    xy=float(np.dot(x,y))
    yy=float(np.dot(y,y))
    if not all(math.isfinite(v) for v in (q,xy,yy)):
        raise ValueError("regression sufficient statistics exceed floating-point range")
    if q<=0:
        raise ValueError("predictor has zero design energy")
    return x,y,n,q,xy,yy


def nig_log_mixture_constant(predictor, response, *, tuning: NIGMixtureTuning=NIGMixtureTuning()) -> dict:
    """Closed-form conditional marginal likelihood ingredients.

    The integrated likelihood is

        p_mix(z_{1:n}|x_{1:n}) = (2*pi)^(-n/2) * exp(K_n),

    with ``K_n`` returned as ``log_mixture_constant``.
    """
    tuning.validate()
    _,_,n,q,xy,yy=_sufficient_statistics(predictor,response)
    k0=float(tuning.kappa); m0=float(tuning.mean)
    a0=float(tuning.shape); b0=float(tuning.scale)
    kn=k0+q
    mn=(k0*m0+xy)/kn
    an=a0+0.5*n
    bn=b0+0.5*(yy+k0*m0*m0-kn*mn*mn)
    # Roundoff can make the quadratic completion tiny-negative only at machine
    # precision.  A genuinely nonpositive posterior scale is invalid.
    if bn<=0 or not math.isfinite(bn):
        raise ValueError("posterior inverse-gamma scale is nonpositive or nonfinite")
    K=(0.5*(math.log(k0)-math.log(kn))
       +a0*math.log(b0)-float(gammaln(a0))
       +float(gammaln(an))-an*math.log(bn))
    beta_hat=xy/q
    rss_min=yy-xy*xy/q
    if rss_min<0 and rss_min>-1e-10*max(1.0,yy):
        rss_min=0.0
    if rss_min<0 or not math.isfinite(rss_min):
        raise ValueError("least-squares residual sum of squares is invalid")
    return {
        "n":n,"design_energy":q,"xy":xy,"yy":yy,
        "beta_hat":beta_hat,"rss_min":rss_min,
        "posterior_kappa":kn,"posterior_mean":mn,
        "posterior_shape":an,"posterior_scale":bn,
        "log_mixture_constant":K,
        "tuning":asdict(tuning),
    }


def nig_log_evalue(predictor, response, beta: float, variance: float, *,
                   tuning: NIGMixtureTuning=NIGMixtureTuning()) -> float:
    """Log likelihood-mixture e-value for the point null (beta, variance)."""
    if not math.isfinite(beta):
        raise ValueError("beta must be finite")
    if not math.isfinite(variance) or variance<=0:
        raise ValueError("variance must be positive and finite")
    s=nig_log_mixture_constant(predictor,response,tuning=tuning)
    q=float(s["design_energy"]); bh=float(s["beta_hat"]); rss=float(s["rss_min"])
    rss_beta=rss+q*(float(beta)-bh)**2
    return float(s["log_mixture_constant"] + 0.5*s["n"]*math.log(variance) + rss_beta/(2.0*variance))


def _variance_projection_roots(n: int, rss_min: float, budget: float) -> tuple[float,float]:
    """Solve n/2 log(v)+rss/(2v) <= budget for v>0.

    The returned closed interval is the closure of the exact projection.
    """
    if n<1 or rss_min<0 or not math.isfinite(rss_min) or not math.isfinite(budget):
        raise ValueError("invalid variance projection inputs")
    if rss_min==0.0:
        # f(v)=n/2 log(v) is strictly increasing and the lower endpoint of the
        # closure is zero.
        upper=math.exp(2.0*budget/n)
        return 0.0,float(upper)
    v0=rss_min/n
    fmin=0.5*n*(math.log(v0)+1.0)
    if budget<fmin:
        raise ValueError("empty confidence set")
    if math.isclose(budget,fmin,rel_tol=0,abs_tol=1e-14*max(1.,abs(fmin))):
        return float(v0),float(v0)

    def f_logv(z: float) -> float:
        # Work in log variance to avoid underflow/overflow over wide scales.
        ez=math.exp(z)
        return 0.5*n*z+rss_min/(2.0*ez)-budget

    z0=math.log(v0)
    zlo=z0-1.0
    while f_logv(zlo)<0:
        zlo-=2.0
        if zlo<-745:
            zlo=-745.0
            break
    zhi=z0+1.0
    while f_logv(zhi)<0:
        zhi+=2.0
        if zhi>709:
            raise ValueError("variance confidence projection exceeds floating-point range")
    lo=math.exp(brentq(f_logv,zlo,z0,xtol=1e-14,rtol=1e-14,maxiter=200))
    hi=math.exp(brentq(f_logv,z0,zhi,xtol=1e-14,rtol=1e-14,maxiter=200))
    return float(lo),float(hi)


def nig_mixture_cs_projection(predictor, response, *, alpha: float,
                              tuning: NIGMixtureTuning=NIGMixtureTuning(),
                              beta_bounds: tuple[float,float] | None=None) -> dict:
    """Coordinate projections of an exact anytime-valid (beta, variance) CS.

    Let ``E_t(beta,variance)`` be the proper NIG likelihood-mixture e-process.
    The exact two-dimensional confidence set is

        C_t = {(beta,v): log E_t(beta,v) < log(1/alpha)}.

    This function returns the closure of its beta and variance projections.  If
    ``beta_bounds`` is supplied, the beta projection is intersected with that
    prespecified structural set.  The variance projection is deliberately *not*
    narrowed after this intersection; it remains an outer projection and is
    therefore conservative for the downstream rectangular certificate.
    """
    tuning.validate()
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    s=nig_log_mixture_constant(predictor,response,tuning=tuning)
    n=int(s["n"]); q=float(s["design_energy"]); bh=float(s["beta_hat"])
    rss=float(s["rss_min"]); K=float(s["log_mixture_constant"])
    threshold=math.log(1.0/alpha)
    budget=threshold-K
    # Profile over v: min_v [n/2 log v + RSS_beta/(2v)]
    # = n/2 [1 + log(RSS_beta/n)].
    rss_cap=n*math.exp(2.0*budget/n-1.0)
    if not math.isfinite(rss_cap) or rss_cap<rss-1e-12*max(1.,rss):
        return {
            **s,"alpha":alpha,"empty":True,
            "beta_lower":None,"beta_upper":None,
            "variance_lower":None,"variance_upper":None,
            "rss_cap":rss_cap,
            "coverage":("Exact time-uniform two-parameter confidence set under the declared "
                        "conditional-Gaussian regression; returned intervals are coordinate projections."),
            "set_inequality":"log_mixture_constant + n/2*log(v) + RSS(beta)/(2v) < log(1/alpha)",
        }
    delta=max(0.0,rss_cap-rss)
    radius=math.sqrt(delta/q)
    raw_beta=(bh-radius,bh+radius)
    blo,bhi=raw_beta
    empty=False
    if beta_bounds is not None:
        sl,sh=map(float,beta_bounds)
        if not (math.isfinite(sl) and math.isfinite(sh) and sl<sh):
            raise ValueError("invalid beta_bounds")
        blo=max(blo,sl); bhi=min(bhi,sh)
        if blo>bhi:
            empty=True
    if empty:
        return {
            **s,"alpha":alpha,"empty":True,
            "beta_lower":None,"beta_upper":None,"raw_beta_projection":list(raw_beta),
            "variance_lower":None,"variance_upper":None,"rss_cap":rss_cap,
            "coverage":("Exact time-uniform two-parameter confidence set before structural intersection; "
                        "the supplied beta structural set is rejected at this monitoring time."),
            "set_inequality":"log_mixture_constant + n/2*log(v) + RSS(beta)/(2v) < log(1/alpha)",
        }
    try:
        vlo,vhi=_variance_projection_roots(n,rss,budget)
    except ValueError:
        # If the beta projection is nonempty, this should only be reachable from
        # extreme floating-point rounding.  Fail loudly rather than silently
        # issuing an invalid variance interval.
        raise ValueError("failed to compute the nonempty variance projection")
    return {
        **s,"alpha":alpha,"empty":False,
        "beta_lower":float(blo),"beta_upper":float(bhi),
        "raw_beta_projection":list(raw_beta),"beta_radius":float(radius),
        "variance_lower":float(vlo),"variance_upper":float(vhi),
        "rss_cap":float(rss_cap),
        "coverage":("Exact time-uniform two-parameter confidence set under the declared "
                    "conditional-Gaussian regression; returned intervals are coordinate projections."),
        "set_inequality":"log_mixture_constant + n/2*log(v) + RSS(beta)/(2v) < log(1/alpha)",
    }


def joint_unknown_scale_set(signal, next_signal, next_return, *, alpha: float=0.05,
                            return_tuning: NIGMixtureTuning=NIGMixtureTuning(),
                            signal_tuning: NIGMixtureTuning=NIGMixtureTuning(),
                            phi_bounds: tuple[float,float]=(-0.999,0.999)) -> dict:
    """Anytime-valid 4D set for (theta, phi, sigma_R^2, sigma_X^2)."""
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    x=np.asarray(signal,dtype=float); xn=np.asarray(next_signal,dtype=float); y=np.asarray(next_return,dtype=float)
    if x.ndim!=1 or xn.shape!=x.shape or y.shape!=x.shape or len(x)<1:
        raise ValueError("signal, next_signal and next_return must have the same nonempty one-dimensional shape")
    if not (np.isfinite(x).all() and np.isfinite(xn).all() and np.isfinite(y).all()):
        raise ValueError("input arrays must be finite")
    each=alpha/2.0
    ret=nig_mixture_cs_projection(x,y,alpha=each,tuning=return_tuning)
    sig=nig_mixture_cs_projection(x,xn,alpha=each,tuning=signal_tuning,beta_bounds=phi_bounds)
    empty=bool(ret["empty"] or sig["empty"])
    rectangle=None
    if not empty:
        rectangle={
            "theta":[float(ret["beta_lower"]),float(ret["beta_upper"])],
            "phi":[float(sig["beta_lower"]),float(sig["beta_upper"])],
            "return_noise_variance":[float(ret["variance_lower"]),float(ret["variance_upper"])],
            "signal_innovation_variance":[float(sig["variance_lower"]),float(sig["variance_upper"])],
        }
    return {
        "return_regression":ret,
        "signal_regression":sig,
        "alpha_joint":alpha,"alpha_each":each,"empty":empty,
        "projection_rectangle":rectangle,
        "coverage_statement":(
            f"Under the two declared conditional-Gaussian regressions, with probability at least {1-alpha:.6g}, "
            "the true tuple (theta, phi, sigma_R^2, sigma_X^2) belongs to the product of the two exact "
            "coefficient-scale confidence sequences at every monitoring time.  The reported four-dimensional "
            "rectangle is a conservative coordinate-projection outer set and is therefore also time-uniform."
        ),
        "method":"proper Normal-Inverse-Gamma likelihood-mixture e-processes plus Bonferroni",
        "independence_required_between_regressions":False,
    }


def _unit_gain_and_curvature(phi: float, *, model: UnknownScaleCertificateModel) -> tuple[np.ndarray,np.ndarray]:
    model.validate()
    if not math.isfinite(phi) or not -1<phi<1:
        raise ValueError("phi must lie in (-1,1)")
    T=model.horizon
    D=np.empty(T,dtype=float); c=np.empty(T,dtype=float)
    K=float(model.terminal_penalty); L=0.0
    for t in range(T-1,-1,-1):
        Dt=model.gamma+model.trading_cost+K
        ct=(1.0+phi*L)/Dt
        D[t]=Dt; c[t]=ct
        K=model.trading_cost*((model.gamma+K)/Dt)
        L=model.trading_cost*ct
    if not (np.isfinite(D).all() and np.isfinite(c).all()):
        raise ValueError("Bellman calculation exceeds floating-point range")
    return c,D


def world_value_unknown_scale(theta: float, phi: float, signal_innovation_variance: float,
                              gains: Iterable[float], *, model: UnknownScaleCertificateModel) -> float:
    """Exact expected objective for a deterministic gain vector."""
    model.validate(); theta=float(theta); phi=float(phi); vx=float(signal_innovation_variance)
    b=np.asarray(tuple(gains),dtype=float)
    if b.shape!=(model.horizon,) or not np.isfinite(b).all():
        raise ValueError("gains must contain one finite value per horizon period")
    if not (math.isfinite(theta) and math.isfinite(phi) and -1<phi<1):
        raise ValueError("invalid world parameter")
    if not math.isfinite(vx) or vx<=0:
        raise ValueError("signal innovation variance must be positive and finite")
    c,D=_unit_gain_and_curvature(phi,model=model)
    s2=vx/(1.0-phi*phi)
    return 0.5*s2*float(np.dot(D,2.0*theta*c*b-b*b))


def world_oracle_value_unknown_scale(theta: float, phi: float, signal_innovation_variance: float, *,
                                     model: UnknownScaleCertificateModel) -> float:
    model.validate(); theta=float(theta); phi=float(phi); vx=float(signal_innovation_variance)
    if not (math.isfinite(theta) and math.isfinite(phi) and -1<phi<1):
        raise ValueError("invalid world parameter")
    if not math.isfinite(vx) or vx<=0:
        raise ValueError("signal innovation variance must be positive and finite")
    c,D=_unit_gain_and_curvature(phi,model=model)
    s2=vx/(1.0-phi*phi)
    return 0.5*s2*float(np.dot(D,(theta*c)**2))


def relative_regret(theta: float, phi: float, gains: Iterable[float], *,
                    model: UnknownScaleCertificateModel) -> float:
    """Exact oracle-relative regret; independent of both Gaussian noise scales."""
    model.validate(); theta=float(theta); phi=float(phi); b=np.asarray(tuple(gains),dtype=float)
    if b.shape!=(model.horizon,) or not np.isfinite(b).all():
        raise ValueError("gains must contain one finite value per horizon period")
    if not math.isfinite(theta) or theta==0:
        raise ValueError("relative regret requires nonzero theta so the oracle value is positive")
    if not math.isfinite(phi) or not -1<phi<1:
        raise ValueError("phi must lie in (-1,1)")
    c,D=_unit_gain_and_curvature(phi,model=model)
    num=float(np.dot(D,(theta*c-b)**2))
    den=float(np.dot(D,(theta*c)**2))
    if den<=0 or not math.isfinite(den):
        raise ValueError("oracle normalization is nonpositive")
    return num/den


def _closed_form_positive_candidate(theta_interval: tuple[float,float],
                                    phi_interval: tuple[float,float], *,
                                    model: UnknownScaleCertificateModel) -> dict | None:
    tlo,thi=map(float,theta_interval); plo,phi=map(float,phi_interval)
    if not (0.0<=plo<=phi<1.0):
        return None
    if tlo<=0.0<=thi:
        theta_near=0.0; gains=np.zeros(model.horizon)
    else:
        theta_near=tlo if tlo>0 else thi
        c,_=_unit_gain_and_curvature(plo,model=model)
        gains=theta_near*c
    theta_far=tlo if abs(tlo)>=abs(thi) else thi
    return {
        "controller_theta":float(theta_near),"controller_phi":float(plo),
        "theta_farthest_from_zero":float(theta_far),"gains":np.asarray(gains,dtype=float),
        "method":"positive-persistence exact maximin gain over the rectangular outer set",
    }


def _generic_center_candidate(theta_interval: tuple[float,float],phi_interval: tuple[float,float], *,
                              model: UnknownScaleCertificateModel) -> dict:
    tlo,thi=map(float,theta_interval); plo,phi=map(float,phi_interval)
    if tlo<=0<=thi:
        ct=0.0; cp=0.5*(plo+phi); gains=np.zeros(model.horizon)
    else:
        ct=0.5*(tlo+thi); cp=0.5*(plo+phi)
        c,_=_unit_gain_and_curvature(cp,model=model); gains=ct*c
    return {"controller_theta":ct,"controller_phi":cp,"gains":np.asarray(gains,dtype=float),
            "method":"center-world oracle-shaped candidate; validity does not require optimality"}


def _iv_bounds(x) -> tuple[float,float]:
    return float(x.a),float(x.b)


def _iv_absolute_enclosures(theta_interval: tuple[float,float],phi_interval: tuple[float,float],
                            signal_variance_interval: tuple[float,float],gains: np.ndarray, *,
                            model: UnknownScaleCertificateModel,
                            theta_cells: int=20,phi_cells: int=80) -> tuple[float,float]:
    """Interval enclosure of candidate lower value and compatible oracle upper."""
    if theta_cells<1 or phi_cells<1:
        raise ValueError("interval cell counts must be positive")
    tlo,thi=theta_interval; plo,phi=phi_interval; vlo,vhi=signal_variance_interval
    if not (0<=vlo<=vhi and math.isfinite(vlo) and math.isfinite(vhi)):
        raise ValueError("invalid signal variance interval")
    gamma=mp.iv.mpf([np.nextafter(model.gamma,-np.inf),np.nextafter(model.gamma,np.inf)])
    lam=mp.iv.mpf([np.nextafter(model.trading_cost,-np.inf),np.nextafter(model.trading_cost,np.inf)])
    terminal=mp.iv.mpf([np.nextafter(model.terminal_penalty,-np.inf),np.nextafter(model.terminal_penalty,np.inf)])
    vx_iv=mp.iv.mpf([max(0.0,float(vlo)),float(vhi)])
    candidate_lower=math.inf; oracle_upper=-math.inf
    t_edges=np.linspace(tlo,thi,theta_cells+1); p_edges=np.linspace(plo,phi,phi_cells+1)
    for ti in range(theta_cells):
        theta_iv=mp.iv.mpf([float(t_edges[ti]),float(t_edges[ti+1])])
        for pi in range(phi_cells):
            phi_iv=mp.iv.mpf([float(p_edges[pi]),float(p_edges[pi+1])])
            K=terminal; L=mp.iv.mpf([0,0]); c=[None]*model.horizon; D=[None]*model.horizon
            for t in range(model.horizon-1,-1,-1):
                Dt=gamma+lam+K; ct=(1+phi_iv*L)/Dt
                D[t]=Dt; c[t]=ct
                K=lam*((gamma+K)/Dt); L=lam*ct
            s2=vx_iv/(1-phi_iv*phi_iv)
            total=mp.iv.mpf([0,0]); oracle=mp.iv.mpf([0,0])
            for t in range(model.horizon):
                b=mp.iv.mpf([float(gains[t]),float(gains[t])])
                total+=D[t]*(2*theta_iv*c[t]*b-b*b)
                og=theta_iv*c[t]; oracle+=D[t]*og*og
            val=mp.iv.mpf('0.5')*s2*total
            ov=mp.iv.mpf('0.5')*s2*oracle
            val_lo,_=_iv_bounds(val); _,ov_hi=_iv_bounds(ov)
            candidate_lower=min(candidate_lower,val_lo); oracle_upper=max(oracle_upper,ov_hi)
    return float(candidate_lower),float(oracle_upper)


def _iv_relative_regret_upper(theta_interval: tuple[float,float],phi_interval: tuple[float,float],
                              gains: np.ndarray, *, model: UnknownScaleCertificateModel,
                              theta_cells: int=20,phi_cells: int=80) -> float:
    """Directed interval upper bound on worst relative regret over theta/phi."""
    tlo,thi=theta_interval; plo,phi=phi_interval
    if tlo<=0<=thi:
        return math.inf
    gamma=mp.iv.mpf([np.nextafter(model.gamma,-np.inf),np.nextafter(model.gamma,np.inf)])
    lam=mp.iv.mpf([np.nextafter(model.trading_cost,-np.inf),np.nextafter(model.trading_cost,np.inf)])
    terminal=mp.iv.mpf([np.nextafter(model.terminal_penalty,-np.inf),np.nextafter(model.terminal_penalty,np.inf)])
    upper=0.0
    t_edges=np.linspace(tlo,thi,theta_cells+1); p_edges=np.linspace(plo,phi,phi_cells+1)
    for ti in range(theta_cells):
        theta_iv=mp.iv.mpf([float(t_edges[ti]),float(t_edges[ti+1])])
        for pi in range(phi_cells):
            phi_iv=mp.iv.mpf([float(p_edges[pi]),float(p_edges[pi+1])])
            K=terminal; L=mp.iv.mpf([0,0]); c=[None]*model.horizon; D=[None]*model.horizon
            for t in range(model.horizon-1,-1,-1):
                Dt=gamma+lam+K; ct=(1+phi_iv*L)/Dt
                D[t]=Dt;c[t]=ct
                K=lam*((gamma+K)/Dt);L=lam*ct
            num=mp.iv.mpf([0,0]);den=mp.iv.mpf([0,0])
            for t in range(model.horizon):
                b=mp.iv.mpf([float(gains[t]),float(gains[t])])
                target=theta_iv*c[t]
                num+=D[t]*(target-b)*(target-b)
                den+=D[t]*target*target
            dlo,_=_iv_bounds(den)
            if dlo<=0:
                return math.inf
            ratio=num/den
            _,rhi=_iv_bounds(ratio)
            upper=max(upper,rhi)
    return float(upper)


def unknown_scale_value_certificate(signal,next_signal,next_return, *,
                                    model: UnknownScaleCertificateModel,
                                    hurdle: float=0.0,alpha: float=0.05,
                                    efficiency_tolerance: float=0.05,
                                    return_tuning: NIGMixtureTuning=NIGMixtureTuning(),
                                    signal_tuning: NIGMixtureTuning=NIGMixtureTuning(),
                                    phi_bounds: tuple[float,float]=(-0.999,0.999),
                                    theta_cells: int=20,phi_cells: int=80) -> dict:
    """Anytime-valid unknown-scale economic and eta-efficiency certificate."""
    model.validate()
    if not math.isfinite(hurdle) or hurdle<0:
        raise ValueError("hurdle must be finite and nonnegative")
    if not math.isfinite(efficiency_tolerance) or not 0<efficiency_tolerance<1:
        raise ValueError("efficiency_tolerance must lie in (0,1)")
    cset=joint_unknown_scale_set(signal,next_signal,next_return,alpha=alpha,
                                 return_tuning=return_tuning,signal_tuning=signal_tuning,
                                 phi_bounds=phi_bounds)
    if cset["empty"]:
        return {"model":asdict(model),"parameter_set":cset,"decision":"model_rejected",
                "efficiency_decision":"not_certified","efficiency_tolerance":efficiency_tolerance,
                "limitations":["The structural/statistical confidence set is empty at this monitoring time."]}
    r=cset["projection_rectangle"]
    ti=tuple(map(float,r["theta"])); pi=tuple(map(float,r["phi"]));
    vxi=tuple(map(float,r["signal_innovation_variance"]))
    exact=_closed_form_positive_candidate(ti,pi,model=model)
    candidate=exact if exact is not None else _generic_center_candidate(ti,pi,model=model)
    gains=np.asarray(candidate["gains"],dtype=float)
    lower,upper=_iv_absolute_enclosures(ti,pi,vxi,gains,model=model,
                                         theta_cells=theta_cells,phi_cells=phi_cells)
    if lower>hurdle:
        decision="deploy_in_model"
    elif upper<hurdle:
        decision="economically_small_in_model"
    else:
        decision="insufficient_evidence"
    eta_upper=_iv_relative_regret_upper(ti,pi,gains,model=model,
                                        theta_cells=theta_cells,phi_cells=phi_cells)
    eta_cert=bool(math.isfinite(eta_upper) and eta_upper<=efficiency_tolerance)
    eff_decision=(f"certified_{efficiency_tolerance:.6g}_efficient_in_model" if eta_cert
                  else "not_certified")
    return {
        "model":asdict(model),"parameter_set":cset,
        "controller":{"theta":candidate["controller_theta"],"phi":candidate["controller_phi"],
                      "gains":[float(x) for x in gains],"method":candidate["method"]},
        "policy_value_lower":lower,"oracle_value_upper":upper,"hurdle":hurdle,"decision":decision,
        "relative_regret_upper":eta_upper if math.isfinite(eta_upper) else None,
        "efficiency_tolerance":efficiency_tolerance,
        "efficiency_decision":eff_decision,
        "certified_eta_efficient":eta_cert,
        "verification":{
            "theta_cells":theta_cells,"phi_cells":phi_cells,
            "absolute_method":"directed interval arithmetic over the 4D projection rectangle; return-noise variance drops out economically",
            "relative_method":"directed interval arithmetic over theta/phi; both Gaussian noise scales cancel from oracle-relative regret",
            "candidate_optimality_required_for_validity":False,
        },
        "coverage_statement":cset["coverage_statement"],
        "limitations":[
            "The time-uniform statement assumes constant conditional-Gaussian noise variances; the variances are unknown but are part of the confidence set.",
            "NIG mixture tuning must be fixed before monitoring; it affects width/power but not validity.",
            "The four-dimensional rectangle is an outer projection of the exact curved coefficient-scale confidence sets and can be conservative.",
            "A finite relative-efficiency certificate is not issued while the loading projection contains zero because the compatible oracle value can vanish.",
            "The objective is finite-horizon expected quadratic mean-risk value, not realized Sharpe or wealth growth.",
            "Impact, capacity, liquidation, structural breaks and uncertainty in economic coefficients remain outside this certificate.",
        ],
    }
