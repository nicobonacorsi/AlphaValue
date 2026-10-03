"""Joint anytime-valid parameter sets and economic value certificates.

The statistical model is deliberately narrow.  For predictable scalar ``x_t``::

    x_{t+1} = phi * x_t + xi_{t+1},     xi | F_t ~ N(0, sigma_x^2)
    y_{t+1} = theta * x_t + eps_{t+1},  eps | F_t ~ N(0, sigma_y^2)

with the stated conditional Gaussian variances known.  The confidence sequences
below are obtained by inverting normal-mixture test martingales.  They are
uniformly valid over time under this model and therefore remain valid at stopping
times.  Bonferroni combines the two one-parameter sequences into a joint
rectangle.  No claim is made under structural breaks or variance misspecification.

The economic model is the finite-horizon quadratic trading problem in
``adaptive.py``.  A candidate controller uses the parameter-free inventory
retention coefficients and a deterministic vector of signal gains.  Its expected
value in a world (theta, phi) has the exact Bellman-residual representation

    J(theta, phi; b) = .5 s_phi^2 sum_t D_t [2 theta c_t(phi)b_t - b_t^2].

A deterministic search chooses a candidate gain vector.  Independent interval
arithmetic then encloses its worst-world value over the entire confidence
rectangle and also encloses the largest oracle value.  The search is NOT itself
used as a proof of optimality: the lower certificate is valid for the returned
candidate whether or not the search found the maximin controller.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
from typing import Iterable

import mpmath as mp
import numpy as np
from scipy.optimize import differential_evolution

from .adaptive import FiniteWorldModel, bellman_coefficients


@dataclass(frozen=True)
class JointCertificateModel:
    gamma: float
    trading_cost: float
    signal_innovation_sd: float
    return_noise_sd: float
    horizon: int
    terminal_penalty: float = 0.0

    def validate(self) -> None:
        vals=(self.gamma,self.trading_cost,self.signal_innovation_sd,
              self.return_noise_sd,self.terminal_penalty)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("joint certificate parameters must be finite")
        if self.gamma<=0 or self.trading_cost<0:
            raise ValueError("require gamma>0 and trading_cost>=0")
        if self.signal_innovation_sd<=0 or self.return_noise_sd<=0:
            raise ValueError("noise standard deviations must be positive")
        if not isinstance(self.horizon,int) or isinstance(self.horizon,bool) or self.horizon<1:
            raise ValueError("horizon must be a positive integer")
        if self.terminal_penalty<0:
            raise ValueError("terminal_penalty must be nonnegative")


def gaussian_mixture_cs_interval(predictor, response, *, noise_sd: float,
                                 alpha: float, rho: float = 1.0,
                                 parameter_bounds: tuple[float,float] | None = None) -> dict:
    """Normal-mixture confidence sequence evaluated at the supplied stopping time.

    For ``response_t = beta * predictor_t + noise_t`` with predictor predictable
    and conditionally Gaussian ``noise_t ~ N(0, noise_sd**2)``, the set is valid
    simultaneously for every sample size.  ``rho`` is the positive normal-mixture
    precision; changing it changes width but not validity.
    """
    x=np.asarray(predictor,dtype=float); y=np.asarray(response,dtype=float)
    if x.ndim!=1 or y.ndim!=1 or x.shape!=y.shape or len(x)<1:
        raise ValueError("predictor and response must be equal nonempty one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("predictor and response must be finite")
    if not math.isfinite(noise_sd) or noise_sd<=0:
        raise ValueError("noise_sd must be positive and finite")
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    if not math.isfinite(rho) or rho<=0:
        raise ValueError("rho must be positive and finite")
    Q=float(np.dot(x,x))
    if not math.isfinite(Q) or Q<=0:
        raise ValueError("predictor has zero design energy")
    beta_hat=float(np.dot(x,y)/Q)
    intrinsic=noise_sd*noise_sd*Q
    log_factor=math.log(1.0/alpha)+0.5*math.log((rho+intrinsic)/rho)
    radius=math.sqrt(2.0*(rho+intrinsic)*log_factor)/Q
    lo,hi=beta_hat-radius,beta_hat+radius
    raw=(lo,hi)
    if parameter_bounds is not None:
        blo,bhi=map(float,parameter_bounds)
        if not (math.isfinite(blo) and math.isfinite(bhi) and blo<bhi):
            raise ValueError("invalid parameter_bounds")
        lo=max(lo,blo); hi=min(hi,bhi)
        if lo>hi:
            # The supplied structural parameter set has been rejected at this time.
            return {
                "estimate":beta_hat,"radius":radius,"lower":None,"upper":None,
                "raw_interval":list(raw),"design_energy":Q,"alpha":alpha,"rho":rho,
                "empty_after_structural_intersection":True,
                "coverage":"Time-uniform under the declared conditional-Gaussian regression model before structural intersection",
            }
    return {
        "estimate":beta_hat,"radius":radius,"lower":lo,"upper":hi,
        "raw_interval":list(raw),"design_energy":Q,"alpha":alpha,"rho":rho,
        "empty_after_structural_intersection":False,
        "coverage":"Time-uniform under the declared conditional-Gaussian regression model",
    }


def joint_parameter_rectangle(signal, next_signal, next_return, *, model: JointCertificateModel,
                              alpha: float=0.05, theta_rho: float=1.0,
                              phi_rho: float=1.0,
                              phi_bounds: tuple[float,float]=(-0.999,0.999)) -> dict:
    """Bonferroni joint anytime-valid rectangle for (theta, phi)."""
    model.validate()
    if not math.isfinite(alpha) or not 0<alpha<1:
        raise ValueError("alpha must lie in (0,1)")
    x=np.asarray(signal,dtype=float); xn=np.asarray(next_signal,dtype=float); y=np.asarray(next_return,dtype=float)
    if x.ndim!=1 or xn.shape!=x.shape or y.shape!=x.shape or len(x)<1:
        raise ValueError("signal, next_signal and next_return must have the same nonempty one-dimensional shape")
    each=alpha/2.0
    theta=gaussian_mixture_cs_interval(x,y,noise_sd=model.return_noise_sd,alpha=each,rho=theta_rho)
    phi=gaussian_mixture_cs_interval(x,xn,noise_sd=model.signal_innovation_sd,alpha=each,rho=phi_rho,
                                     parameter_bounds=phi_bounds)
    empty=theta["empty_after_structural_intersection"] or phi["empty_after_structural_intersection"]
    return {
        "theta":theta,"phi":phi,"alpha_joint":alpha,"alpha_each":each,
        "empty":empty,
        "coverage_statement":(
            f"Under the two declared conditional-Gaussian regressions, with probability at least {1-alpha:.6g}, "
            "both parameter sequences contain their true values simultaneously for every monitoring time; "
            "therefore the rectangle remains valid at arbitrary stopping times."
        ),
        "method":"normal-mixture e-processes plus Bonferroni union bound",
    }


def _gain_and_curvature(phi: float, *, theta: float, model: JointCertificateModel) -> tuple[np.ndarray,np.ndarray]:
    fm=FiniteWorldModel((float(phi),),gamma=model.gamma,trading_cost=model.trading_cost,
                        theta=float(theta),innovation_variance=model.signal_innovation_sd**2,
                        terminal_penalty=model.terminal_penalty)
    c=bellman_coefficients(fm,model.horizon)
    return c.gains[:,0].copy(),c.curvature.copy()


def world_value(theta: float, phi: float, gains: Iterable[float], *, model: JointCertificateModel) -> float:
    """Exact expected finite-horizon objective for a deterministic gain vector."""
    model.validate(); theta=float(theta); phi=float(phi); b=np.asarray(tuple(gains),dtype=float)
    if b.shape!=(model.horizon,) or not np.isfinite(b).all():
        raise ValueError("gains must contain one finite value per horizon period")
    if not math.isfinite(theta) or not math.isfinite(phi) or not -1<phi<1:
        raise ValueError("invalid world parameter")
    unit,D=_gain_and_curvature(phi,theta=1.0,model=model)
    s2=model.signal_innovation_sd**2/(1.0-phi*phi)
    return 0.5*s2*float(np.dot(D,2.0*theta*unit*b-b*b))


def world_oracle_value(theta: float, phi: float, *, model: JointCertificateModel) -> float:
    g,D=_gain_and_curvature(phi,theta=theta,model=model)
    s2=model.signal_innovation_sd**2/(1.0-phi*phi)
    return 0.5*s2*float(np.dot(D,g*g))



def _positive_persistence_exact_candidate(theta_interval: tuple[float,float],
                                          phi_interval: tuple[float,float], *,
                                          model: JointCertificateModel) -> dict | None:
    """Closed-form maximin candidate when persistence is known nonnegative.

    If ``phi`` lies in [p0,p1] subset [0,1), every unit-loading finite-horizon
    oracle coefficient c_H(phi) and the stationary signal variance are
    nondecreasing in phi.  If the loading interval excludes zero, the oracle
    for the loading endpoint closest to zero and the smallest persistence is
    therefore least-favourable and, remarkably, its policy is maximin over *all*
    causal policies: it attains that least-world oracle value in the least world
    and no policy can exceed that oracle value there.  If the loading interval
    crosses zero, the zero controller is exact maximin with value zero.

    Returns ``None`` outside the nonnegative-persistence regime so the generic
    interval-certified search can be used instead.
    """
    tlo,thi=map(float,theta_interval); plo,phi=map(float,phi_interval)
    if not (0.0 <= plo <= phi < 1.0):
        return None
    # The world theta=0 has oracle value zero, so no policy can guarantee a
    # positive value when zero is compatible.  The zero controller attains 0.
    if tlo <= 0.0 <= thi:
        theta_near=0.0
        gains=np.zeros(model.horizon)
        robust_value=0.0
    else:
        theta_near=tlo if tlo>0.0 else thi  # endpoint closest to zero
        gains,_=_gain_and_curvature(plo,theta=theta_near,model=model)
        robust_value=world_oracle_value(theta_near,plo,model=model)
    theta_far=tlo if abs(tlo)>=abs(thi) else thi
    oracle_upper=world_oracle_value(theta_far,phi,model=model)
    return {
        "controller_theta":float(theta_near),
        "controller_phi":float(plo),
        "gains":gains,
        "theoretical_maximin_value":float(robust_value),
        "theoretical_oracle_upper":float(oracle_upper),
        "theta_farthest_from_zero":float(theta_far),
        "method":"positive-persistence closed-form least-favourable-world theorem",
        "scope":"exact maximin over all causal policies for rectangles with 0<=phi_lower<=phi_upper<1",
    }

def _candidate_search(theta_interval: tuple[float,float], phi_interval: tuple[float,float], *,
                      model: JointCertificateModel, seed: int=1729) -> dict:
    """Deterministic numerical search for an oracle-shaped robust candidate.

    The ensuing interval certificate does not depend on search optimality.
    """
    tlo,thi=theta_interval; plo,phi=phi_interval
    if tlo<=0<=thi:
        return {"controller_theta":0.0,"controller_phi":0.5*(plo+phi),
                "gains":np.zeros(model.horizon),"search_grid_worst_value":0.0,
                "search_status":"zero controller: loading interval crosses zero"}
    theta_grid=np.linspace(tlo,thi,9); phi_grid=np.linspace(plo,phi,17)
    worlds=[(t,p) for t in theta_grid for p in phi_grid]
    def objective(z):
        ct,cp=float(z[0]),float(z[1])
        gains,_=_gain_and_curvature(cp,theta=ct,model=model)
        worst=min(world_value(t,p,gains,model=model) for t,p in worlds)
        return -worst
    bounds=[(min(tlo,0.0),max(thi,0.0)),(plo,phi)]
    res=differential_evolution(objective,bounds,seed=seed,tol=1e-10,polish=True,
                               popsize=12,maxiter=220,workers=1,updating="immediate")
    ct,cp=map(float,res.x); gains,_=_gain_and_curvature(cp,theta=ct,model=model)
    zero_worst=0.0
    found=-float(res.fun)
    if found<zero_worst:
        ct=0.0;cp=0.5*(plo+phi);gains=np.zeros(model.horizon);found=0.0
    return {"controller_theta":ct,"controller_phi":cp,"gains":gains,
            "search_grid_worst_value":found,"search_status":str(res.message)}


def _iv_bounds(x) -> tuple[float,float]:
    return float(x.a),float(x.b)


def _iv_world_enclosures(theta_interval: tuple[float,float], phi_interval: tuple[float,float],
                         gains: np.ndarray, *, model: JointCertificateModel,
                         theta_cells: int=24, phi_cells: int=96) -> tuple[float,float]:
    """Interval enclosure of candidate lower value and oracle upper value."""
    if theta_cells<1 or phi_cells<1:
        raise ValueError("interval cell counts must be positive")
    tlo,thi=theta_interval; plo,phi=phi_interval
    # Parameter-free Bellman curvature; evaluate once in ordinary arithmetic and
    # outward-pad each scalar by four ulps before converting to intervals.
    _,D_float=_gain_and_curvature(0.5*(plo+phi),theta=1.0,model=model)
    D_iv=[]
    for d in D_float:
        lo=np.nextafter(np.nextafter(float(d),-np.inf),-np.inf)
        hi=np.nextafter(np.nextafter(float(d),np.inf),np.inf)
        D_iv.append(mp.iv.mpf([lo,hi]))
    gamma=mp.iv.mpf([np.nextafter(model.gamma,-np.inf),np.nextafter(model.gamma,np.inf)])
    lam=mp.iv.mpf([np.nextafter(model.trading_cost,-np.inf),np.nextafter(model.trading_cost,np.inf)])
    sig2=model.signal_innovation_sd**2
    sig2_iv=mp.iv.mpf([np.nextafter(sig2,-np.inf),np.nextafter(sig2,np.inf)])
    terminal=mp.iv.mpf([np.nextafter(model.terminal_penalty,-np.inf),np.nextafter(model.terminal_penalty,np.inf)])
    # Recompute D and c inside each phi cell; D is independent of phi, but doing so
    # avoids relying on the float recurrence for the value enclosure.
    candidate_lower=math.inf; oracle_upper=-math.inf
    t_edges=np.linspace(tlo,thi,theta_cells+1); p_edges=np.linspace(plo,phi,phi_cells+1)
    for ti in range(theta_cells):
        theta_iv=mp.iv.mpf([float(t_edges[ti]),float(t_edges[ti+1])])
        for pi in range(phi_cells):
            p0,p1=float(p_edges[pi]),float(p_edges[pi+1])
            phi_iv=mp.iv.mpf([p0,p1])
            K=terminal; L=mp.iv.mpf([0,0]); c=[None]*model.horizon; D=[None]*model.horizon
            for t in range(model.horizon-1,-1,-1):
                Dt=gamma+lam+K
                ct=(1+phi_iv*L)/Dt
                D[t]=Dt;c[t]=ct
                K=lam*((gamma+K)/Dt)
                L=lam*ct
            denom=1-phi_iv*phi_iv
            s2=sig2_iv/denom
            total=mp.iv.mpf([0,0]); oracle=mp.iv.mpf([0,0])
            for t in range(model.horizon):
                b=mp.iv.mpf([float(gains[t]),float(gains[t])])
                total += D[t]*(2*theta_iv*c[t]*b-b*b)
                og=theta_iv*c[t]
                oracle += D[t]*og*og
            val=mp.iv.mpf('0.5')*s2*total
            ov=mp.iv.mpf('0.5')*s2*oracle
            vlo,_=_iv_bounds(val); _,ohi=_iv_bounds(ov)
            candidate_lower=min(candidate_lower,vlo);oracle_upper=max(oracle_upper,ohi)
    return float(candidate_lower),float(oracle_upper)


def joint_value_certificate(signal, next_signal, next_return, *, model: JointCertificateModel,
                            hurdle: float=0.0, alpha: float=0.05,
                            theta_rho: float=1.0, phi_rho: float=1.0,
                            phi_bounds: tuple[float,float]=(-0.999,0.999),
                            theta_cells: int=24, phi_cells: int=96,
                            search_seed: int=1729) -> dict:
    """Anytime-valid three-way economic certificate for joint (theta, phi) uncertainty.

    The returned lower value is for an explicit controller and is certified over
    the entire joint confidence rectangle by interval arithmetic.  The returned
    oracle upper value covers every oracle in that rectangle.  For nonnegative-persistence rectangles the exact maximin controller is available
    in closed form.  Outside that regime the controller is searched numerically,
    but validity of its lower value does not require global search optimality.
    """
    model.validate()
    if not math.isfinite(hurdle) or hurdle<0:
        raise ValueError("hurdle must be finite and nonnegative")
    rect=joint_parameter_rectangle(signal,next_signal,next_return,model=model,alpha=alpha,
                                   theta_rho=theta_rho,phi_rho=phi_rho,phi_bounds=phi_bounds)
    if rect["empty"]:
        return {"model":asdict(model),"parameter_rectangle":rect,"decision":"model_rejected",
                "hurdle":hurdle,"limitations":["The structural parameter set is empty after intersection; no economic certificate is issued."]}
    ti=(float(rect["theta"]["lower"]),float(rect["theta"]["upper"]))
    pi=(float(rect["phi"]["lower"]),float(rect["phi"]["upper"]))
    exact=_positive_persistence_exact_candidate(ti,pi,model=model)
    if exact is not None:
        candidate={
            "controller_theta":exact["controller_theta"],
            "controller_phi":exact["controller_phi"],
            "gains":np.asarray(exact["gains"],dtype=float),
            "search_grid_worst_value":exact["theoretical_maximin_value"],
            "search_status":exact["method"],
        }
        certificate_structure={
            "mode":"closed_form_positive_persistence",
            "theoretical_maximin_value":exact["theoretical_maximin_value"],
            "theoretical_oracle_upper":exact["theoretical_oracle_upper"],
            "theta_farthest_from_zero":exact["theta_farthest_from_zero"],
            "theorem_scope":exact["scope"],
        }
    else:
        candidate=_candidate_search(ti,pi,model=model,seed=search_seed)
        certificate_structure={
            "mode":"generic_interval_certified_candidate",
            "theorem_scope":"no closed-form maximin claim outside nonnegative persistence",
        }
    # Independently interval-check the actual floating-point gain vector over the
    # whole rectangle.  This numerical enclosure remains the reported lower/upper
    # certificate even when the closed-form theorem identifies the exact maximin
    # policy mathematically.
    lower,upper=_iv_world_enclosures(ti,pi,np.asarray(candidate["gains"]),model=model,
                                     theta_cells=theta_cells,phi_cells=phi_cells)
    if lower>hurdle:
        decision="deploy_in_model"
    elif upper<hurdle:
        decision="economically_small_in_model"
    else:
        decision="insufficient_evidence"
    return {
        "model":asdict(model),"parameter_rectangle":rect,
        "controller":{"theta":candidate["controller_theta"],"phi":candidate["controller_phi"],
                      "gains":[float(x) for x in candidate["gains"]],
                      "search_grid_worst_value":candidate["search_grid_worst_value"],
                      "search_status":candidate["search_status"]},
        "policy_value_lower":lower,"oracle_value_upper":upper,"hurdle":hurdle,"decision":decision,
        "certificate_structure":certificate_structure,
        "verification":{"theta_cells":theta_cells,"phi_cells":phi_cells,
                        "method":"mpmath interval arithmetic on a complete rectangular cover",
                        "search_optimality_required_for_validity":False},
        "coverage_statement":rect["coverage_statement"],
        "limitations":[
            "Conditional Gaussian noise scales are treated as known and correctly specified.",
            "The confidence rectangle is a model-based time-uniform guarantee, not a distribution-free market guarantee.",
            ("For nonnegative-persistence rectangles the controller is closed-form exact maximin; "
             "outside that regime candidate search is numerical and only the returned policy value, not global maximin optimality, is certified."),
            "The objective is finite-horizon expected quadratic mean-risk value, not realized Sharpe or wealth growth.",
            "Impact, capacity, liquidation, structural breaks and uncertainty in economic coefficients are outside this certificate.",
        ],
    }
