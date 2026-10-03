"""Canonical Certified Alpha Capacity utilities.

The canonical experiment is expressed in information time s. Under P1,
    Z_s = B_s + s/2,
and under P0,
    Z_s = B_s - s/2,
so exp(Z_s) is the likelihood ratio dP1/dP0 on F_s.

This module implements rigorous scalar consequences of the reduction:
* binary-KL converse for the certified fraction;
* finite-horizon likelihood-threshold lower bound;
* first-passage probabilities for the Brownian log-likelihood process;
* baseline-relative versus flat-wait economic price per information nat.

The exact free-boundary/dual problem is characterized in the research note but is
not numerically solved here; avoiding an unverified PDE solver is deliberate.
"""
from __future__ import annotations

import math
import numpy as np
from statistics import NormalDist
from typing import Callable

try:
    from scipy.integrate import quad
    from scipy.optimize import brentq
except Exception as exc:  # pragma: no cover - package already depends on scipy in research env
    quad = None
    brentq = None

_N = NormalDist()


def _Phi(x: float) -> float:
    return _N.cdf(float(x))


def binary_kl(p: float, q: float) -> float:
    """Binary relative entropy kl(p,q), with endpoint conventions."""
    p, q = float(p), float(q)
    if not (0.0 <= p <= 1.0 and 0.0 < q < 1.0):
        raise ValueError("require p in [0,1] and q in (0,1)")
    if p == 0.0:
        return -math.log1p(-q)
    if p == 1.0:
        return -math.log(q)
    return p * math.log(p / q) + (1.0 - p) * math.log((1.0 - p) / (1.0 - q))


def first_passage_cdf(t: float, h: float, drift: float) -> float:
    """P(tau_h <= t) for X_t=B_t+drift*t, tau_h=inf{t:X_t>=h}.

    Formula: Phi((mu t-h)/sqrt(t)) + exp(2 mu h)
             Phi((-mu t-h)/sqrt(t)).
    """
    t, h, drift = float(t), float(h), float(drift)
    if t <= 0.0:
        return 0.0
    if h <= 0.0:
        return 1.0
    rt = math.sqrt(t)
    ans = _Phi((drift * t - h) / rt) + math.exp(2.0 * drift * h) * _Phi((-drift * t - h) / rt)
    return min(1.0, max(0.0, ans))


def finite_horizon_threshold(alpha: float, A: float) -> float:
    """Unique h_A>0 with P0(tau_h < A)=alpha for drift -1/2.

    The strict/weak endpoint distinction is immaterial for continuous paths.
    """
    alpha, A = float(alpha), float(A)
    if not (0.0 < alpha < 1.0 and A > 0.0):
        raise ValueError("require alpha in (0,1), A>0")
    if brentq is None:
        raise RuntimeError("scipy is required")
    f: Callable[[float], float] = lambda h: first_passage_cdf(A, h, -0.5) - alpha
    hi = max(2.0, 2.0 * math.log(1.0 / alpha) + 2.0 * math.sqrt(A))
    while f(hi) > 0.0:
        hi *= 2.0
    return float(brentq(f, 1e-14, hi, xtol=1e-13, rtol=1e-13))


def threshold_capacity_lower(alpha: float, A: float, *, finite_horizon_calibrated: bool = True) -> dict:
    """Achievable information-time capacity for a constant LLR boundary.

    If finite_horizon_calibrated=True, choose h_A so the finite-horizon false
    deployment probability equals alpha. Otherwise use h=log(1/alpha), the
    anytime-valid Ville boundary.

    Returns c_lower = E_1[(A-tau_h)_+] = integral_0^A P_1(tau_h<=t) dt.
    """
    alpha, A = float(alpha), float(A)
    if not (0.0 < alpha < 1.0 and A > 0.0):
        raise ValueError("require alpha in (0,1), A>0")
    if quad is None:
        raise RuntimeError("scipy is required")
    h = finite_horizon_threshold(alpha, A) if finite_horizon_calibrated else math.log(1.0 / alpha)
    c = float(quad(lambda t: first_passage_cdf(t, h, 0.5), 0.0, A,
                   epsabs=1e-11, epsrel=1e-11, limit=400)[0])
    p0 = first_passage_cdf(A, h, -0.5)
    p1 = first_passage_cdf(A, h, 0.5)
    return {"h": h, "capacity_lower": c, "fraction_lower": c / A,
            "false_deploy": p0, "power_by_horizon": p1}


def certified_fraction_converse(alpha: float, A: float) -> float:
    """Unique rbar in [alpha,1) solving A(1-r)=2 kl(r,alpha).

    For the convexified alpha-only capacity c_alpha(A), r=c/A obeys r<=rbar.
    Equivalently with I=A/2, I(1-r)>=kl(r,alpha).
    """
    alpha, A = float(alpha), float(A)
    if not (0.0 < alpha < 1.0 and A > 0.0):
        raise ValueError("require alpha in (0,1), A>0")
    if brentq is None:
        raise RuntimeError("scipy is required")
    f = lambda r: A * (1.0 - r) - 2.0 * binary_kl(r, alpha)
    lo = alpha
    hi = 1.0 - 1e-14
    return float(brentq(f, lo, hi, xtol=1e-13, rtol=1e-13))


def fixed_power_capacity_upper(alpha: float, beta: float, A: float) -> float:
    """Information-time upper bound c <= [A-2 kl(1-beta,alpha)]_+.

    This applies when deployment before death has power at least 1-beta.
    """
    alpha, beta, A = float(alpha), float(beta), float(A)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0 and A > 0.0):
        raise ValueError("require alpha,beta in (0,1), A>0")
    p = 1.0 - beta
    if p <= alpha:
        # The KL expression remains valid, but the monotone 'safety/power gap'
        # interpretation is not the relevant one. Keep the exact expression.
        pass
    return max(0.0, A - 2.0 * binary_kl(p, alpha))


def economic_capacity_scale(sigma: float, gamma: float) -> float:
    """Multiply information-time capacity by sigma^2/(2 gamma)."""
    sigma, gamma = float(sigma), float(gamma)
    if sigma <= 0.0 or gamma <= 0.0:
        raise ValueError("sigma,gamma must be positive")
    return sigma * sigma / (2.0 * gamma)


def price_per_nat(sigma: float, gamma: float, theta: float, b: float) -> dict:
    """Economic regret per KL nat for two pre-certification policies.

    Comparing P_theta to P_b, KL accrues at rate
       (theta-b)^2 W^2/(2 sigma^2).

    * baseline_relative: use the boundary oracle q_b=bW/gamma before
      certification; oracle regret per nat is sigma^2/gamma.
    * flat_wait: stay at q=0; oracle regret per nat is
      (sigma^2/gamma)*(theta/(theta-b))^2, which diverges as b->theta.

    The second singularity is therefore policy-relative, not a singularity of
    the statistical experiment itself.
    """
    sigma, gamma, theta, b = map(float, (sigma, gamma, theta, b))
    if sigma <= 0.0 or gamma <= 0.0:
        raise ValueError("sigma,gamma must be positive")
    if theta == b:
        flat = math.inf if theta != 0.0 else math.nan
    else:
        flat = (sigma * sigma / gamma) * (theta / (theta - b)) ** 2
    return {"baseline_relative": sigma * sigma / gamma, "flat_wait": flat}


def gaussian_power_envelope(alpha: float, A: float) -> float:
    """Exact maximal P1 rejection/deployment probability by information time A.

    In the canonical Brownian experiment, the whole path likelihood ratio at A
    is exp(Z_A), so Neyman-Pearson reduces to the terminal Gaussian statistic.
    The level-alpha power envelope is Phi(sqrt(A)-z_{1-alpha}).
    """
    alpha, A = float(alpha), float(A)
    if not (0.0 < alpha < 1.0 and A >= 0.0):
        raise ValueError("require alpha in (0,1), A>=0")
    z = _N.inv_cdf(1.0 - alpha)
    return _Phi(math.sqrt(A) - z)


def reliable_information_threshold(alpha: float, beta: float) -> dict:
    """Exact lifetime information threshold for type-I alpha and power 1-beta.

    Returns A_crit in information-clock units and I_crit=A_crit/2 in KL nats.
    A strategy with deployment by death can have power >=1-beta at level alpha
    iff A >= A_crit (up to the conventional terminal-time equality issue).
    Positive *economic* capacity requires A>A_crit because a test using exactly
    A_crit units can spend the entire lifetime on statistical resolution.
    """
    alpha, beta = float(alpha), float(beta)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("require alpha,beta in (0,1)")
    z_alpha = _N.inv_cdf(1.0 - alpha)
    z_beta = _N.inv_cdf(1.0 - beta)
    Acrit = (z_alpha + z_beta) ** 2
    return {"A_crit": Acrit, "I_crit_nats": 0.5 * Acrit,
            "z_1_minus_alpha": z_alpha, "z_1_minus_beta": z_beta}


def fixed_time_reliable_capacity_lower(alpha: float, beta: float, A: float) -> dict:
    """Constructive lower bound from a fixed-time NP test.

    For t in [A_crit,A), test at t at level alpha. Its power is pi_alpha(t),
    and deploying then earns A-t information-time value when rejection occurs.
    We maximize (A-t)*pi_alpha(t). This is not claimed optimal; sequential
    stopping can improve on it.
    """
    alpha, beta, A = float(alpha), float(beta), float(A)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0 and A > 0.0):
        raise ValueError("require alpha,beta in (0,1), A>0")
    crit = reliable_information_threshold(alpha, beta)["A_crit"]
    if A <= crit:
        return {"capacity_lower": 0.0, "fraction_lower": 0.0,
                "test_time": None, "power": gaussian_power_envelope(alpha, A),
                "A_crit": crit}
    if brentq is None:
        raise RuntimeError("scipy is required")
    # Golden-section search implemented explicitly to avoid another scipy import.
    lo, hi = crit, A
    gr = (math.sqrt(5.0) - 1.0) / 2.0
    def obj(t: float) -> float:
        return (A - t) * gaussian_power_envelope(alpha, t)
    x1 = hi - gr * (hi - lo)
    x2 = lo + gr * (hi - lo)
    f1, f2 = obj(x1), obj(x2)
    for _ in range(120):
        if f1 < f2:
            lo = x1; x1 = x2; f1 = f2
            x2 = lo + gr * (hi - lo); f2 = obj(x2)
        else:
            hi = x2; x2 = x1; f2 = f1
            x1 = hi - gr * (hi - lo); f1 = obj(x1)
    tstar = 0.5 * (lo + hi)
    val = obj(tstar)
    return {"capacity_lower": val, "fraction_lower": val / A,
            "test_time": tstar, "power": gaussian_power_envelope(alpha, tstar),
            "A_crit": crit}


def economic_information_spectrum(noise_cov, risk_matrix) -> dict:
    """Extreme economic regret per KL nat in a multivariate Gaussian market.

    Observation drift covariance is ``noise_cov = Sigma`` and the quadratic
    position penalty is ``risk_matrix = Gamma``.  For a drift displacement
    Delta, the direction-specific conversion factor is

        Delta' Gamma^{-1} Delta / (Delta' Sigma^{-1} Delta).

    Its exact range is the spectrum of Sigma^{1/2} Gamma^{-1} Sigma^{1/2}.
    """
    import numpy as np

    Sigma = np.asarray(noise_cov, dtype=float)
    Gamma = np.asarray(risk_matrix, dtype=float)
    if Sigma.ndim != 2 or Gamma.ndim != 2 or Sigma.shape != Gamma.shape or Sigma.shape[0] != Sigma.shape[1]:
        raise ValueError("noise_cov and risk_matrix must be same-size square matrices")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12) or not np.allclose(Gamma, Gamma.T, atol=1e-12):
        raise ValueError("matrices must be symmetric")
    ws, Us = np.linalg.eigh(Sigma)
    wg = np.linalg.eigvalsh(Gamma)
    if np.min(ws) <= 0.0 or np.min(wg) <= 0.0:
        raise ValueError("matrices must be positive definite")
    S12 = (Us * np.sqrt(ws)) @ Us.T
    M = S12 @ np.linalg.solve(Gamma, S12)
    vals = np.linalg.eigvalsh((M + M.T) / 2.0)
    return {
        "min_price_per_nat": float(vals[0]),
        "max_price_per_nat": float(vals[-1]),
        "eigenvalues": vals,
    }


def directional_price_per_nat(delta, noise_cov, risk_matrix) -> float:
    """Exact economic regret / KL ratio for one nonzero drift direction."""
    import numpy as np

    d = np.asarray(delta, dtype=float).reshape(-1)
    Sigma = np.asarray(noise_cov, dtype=float)
    Gamma = np.asarray(risk_matrix, dtype=float)
    if Sigma.shape != (d.size, d.size) or Gamma.shape != (d.size, d.size):
        raise ValueError("matrix dimensions must match delta")
    if not np.allclose(Sigma, Sigma.T, atol=1e-12) or not np.allclose(Gamma, Gamma.T, atol=1e-12):
        raise ValueError("matrices must be symmetric")
    if np.min(np.linalg.eigvalsh(Sigma)) <= 0.0 or np.min(np.linalg.eigvalsh(Gamma)) <= 0.0:
        raise ValueError("matrices must be positive definite")
    if float(d @ d) == 0.0:
        raise ValueError("delta must be nonzero")
    econ = float(d @ np.linalg.solve(Gamma, d))
    info = float(d @ np.linalg.solve(Sigma, d))
    return econ / info


def multiplicity_adjusted_alpha(alpha_family: float, m: int, *, method: str = "bonferroni") -> float:
    """Per-candidate type-I budget under a declared FWER correction.

    ``bonferroni`` is valid under arbitrary dependence and returns alpha_family/m.
    ``sidak`` returns 1-(1-alpha_family)^(1/m) and should only be used when the
    independence assumptions justifying Sidak are part of the model.
    """
    alpha_family = float(alpha_family)
    m = int(m)
    if not (0.0 < alpha_family < 1.0) or m < 1:
        raise ValueError("require alpha_family in (0,1) and integer m>=1")
    method = str(method).lower()
    if method == "bonferroni":
        return alpha_family / m
    if method == "sidak":
        return -math.expm1(math.log1p(-alpha_family) / m)
    raise ValueError("method must be 'bonferroni' or 'sidak'")


def multiplicity_reliable_information_threshold(
    alpha_family: float,
    beta: float,
    m: int,
    *,
    method: str = "bonferroni",
) -> dict:
    """Reliable-information threshold after an explicit FWER correction.

    This is exact *for the chosen per-test alpha budget* in the canonical
    simple-vs-simple Gaussian experiment.  Bonferroni itself is only a valid
    (generally conservative) familywise allocation under arbitrary dependence;
    it is not claimed to be the globally optimal multiple-testing procedure.
    """
    per_alpha = multiplicity_adjusted_alpha(alpha_family, m, method=method)
    out = reliable_information_threshold(per_alpha, beta)
    return {
        **out,
        "family_alpha": float(alpha_family),
        "per_test_alpha": per_alpha,
        "multiplicity": int(m),
        "correction": str(method).lower(),
    }


def certifiability_ratio(
    lifetime_information_nats: float,
    alpha: float,
    beta: float,
    *,
    multiplicity: int = 1,
    correction: str = "bonferroni",
) -> dict:
    """Lifetime information divided by the reliable-certification threshold.

    CR<1 means the requested type-I/power target is infeasible in the canonical
    experiment under the declared multiplicity correction.  CR=1 is the
    terminal feasibility boundary; strictly positive post-certification value
    requires CR>1.
    """
    I = float(lifetime_information_nats)
    if I < 0.0:
        raise ValueError("lifetime_information_nats must be nonnegative")
    thr = multiplicity_reliable_information_threshold(
        alpha, beta, multiplicity, method=correction
    )
    Icrit = float(thr["I_crit_nats"])
    ratio = I / Icrit
    if ratio < 1.0 - 1e-12:
        status = "INFEASIBLE"
    elif ratio <= 1.0 + 1e-12:
        status = "BOUNDARY"
    else:
        status = "FEASIBLE"
    return {**thr, "I_life_nats": I, "certifiability_ratio": ratio, "status": status}


def exponential_lifetime_information(
    mu0: float,
    sigma: float,
    decay_rate: float,
    *,
    hurdle: float = 0.0,
) -> dict:
    """Total KL information before exponential alpha reaches its economic null.

    Model: mu(t)=mu0*exp(-lambda*t), with mu0>0, sigma>0, lambda>0.
    If hurdle b=0, economic death is only asymptotic and the finite total KL is

        I_life = mu0^2/(4 sigma^2 lambda).

    If 0<b<mu0, death occurs at T=log(mu0/b)/lambda and

        I_life = mu0^2/(2 sigma^2 lambda)
                 [1/2 - 2r + 3r^2/2 + r^2 log(1/r)], r=b/mu0.

    The comparison law is the boundary world with drift b over the useful life.
    """
    mu0, sigma, decay_rate, hurdle = map(float, (mu0, sigma, decay_rate, hurdle))
    if mu0 <= 0.0 or sigma <= 0.0 or decay_rate <= 0.0 or hurdle < 0.0:
        raise ValueError("require mu0,sigma,decay_rate>0 and hurdle>=0")
    if hurdle >= mu0:
        return {
            "I_life_nats": 0.0,
            "economic_death_time": 0.0,
            "hurdle_ratio": hurdle / mu0,
            "finite_calendar_death": True,
        }
    if hurdle == 0.0:
        I = mu0 * mu0 / (4.0 * sigma * sigma * decay_rate)
        return {
            "I_life_nats": I,
            "economic_death_time": math.inf,
            "hurdle_ratio": 0.0,
            "finite_calendar_death": False,
        }
    r = hurdle / mu0
    g = 0.5 - 2.0 * r + 1.5 * r * r + r * r * math.log(1.0 / r)
    I = (mu0 * mu0 / (2.0 * sigma * sigma * decay_rate)) * g
    T = math.log(1.0 / r) / decay_rate
    return {
        "I_life_nats": I,
        "economic_death_time": T,
        "hurdle_ratio": r,
        "finite_calendar_death": True,
    }


def exponential_remaining_information_from_death(
    delta: float,
    hurdle: float,
    sigma: float,
    decay_rate: float,
) -> float:
    """Exact KL remaining delta time units before a positive exponential hurdle.

    At economic death T, mu(T)=b>0.  Writing u=T-t, the signal-to-boundary
    gap is b(exp(lambda*u)-1).  Therefore the remaining path KL is

      b^2/(2 sigma^2) * [ (e^(2 lambda delta)-1)/(2 lambda)
                          -2(e^(lambda delta)-1)/lambda + delta ].
    """
    delta, hurdle, sigma, decay_rate = map(float, (delta, hurdle, sigma, decay_rate))
    if delta < 0.0 or hurdle <= 0.0 or sigma <= 0.0 or decay_rate <= 0.0:
        raise ValueError("require delta>=0 and hurdle,sigma,decay_rate>0")
    x = decay_rate * delta
    # expm1 is materially more stable near economic death.
    bracket = math.expm1(2.0 * x) / (2.0 * decay_rate) \
        - 2.0 * math.expm1(x) / decay_rate + delta
    return (hurdle * hurdle / (2.0 * sigma * sigma)) * bracket


def polynomial_terminal_information(
    delta: float,
    coefficient,
    noise_cov,
    order: float,
) -> float:
    """Exact remaining KL for Delta(T-u)=coefficient*u^order.

    This function implements the local polynomial benchmark exactly.  For a
    general smooth crossing with the same leading term it is the first-order
    asymptotic as delta -> 0, not an exact global formula.
    """
    import numpy as np

    delta, order = float(delta), float(order)
    c = np.asarray(coefficient, dtype=float).reshape(-1)
    Sigma = np.asarray(noise_cov, dtype=float)
    if delta < 0.0 or order <= -0.5:
        raise ValueError("require delta>=0 and order>-1/2")
    if Sigma.shape != (c.size, c.size) or not np.allclose(Sigma, Sigma.T, atol=1e-12):
        raise ValueError("noise_cov must be symmetric and match coefficient")
    if np.min(np.linalg.eigvalsh(Sigma)) <= 0.0 or float(c @ c) == 0.0:
        raise ValueError("noise_cov must be positive definite and coefficient nonzero")
    q = float(c @ np.linalg.solve(Sigma, c))
    return 0.5 * q * delta ** (2.0 * order + 1.0) / (2.0 * order + 1.0)


def polynomial_terminal_economic_value(
    delta: float,
    coefficient,
    risk_matrix,
    order: float,
) -> float:
    """Exact remaining oracle-vs-boundary quadratic value in the polynomial benchmark."""
    import numpy as np

    delta, order = float(delta), float(order)
    c = np.asarray(coefficient, dtype=float).reshape(-1)
    Gamma = np.asarray(risk_matrix, dtype=float)
    if delta < 0.0 or order <= -0.5:
        raise ValueError("require delta>=0 and order>-1/2")
    if Gamma.shape != (c.size, c.size) or not np.allclose(Gamma, Gamma.T, atol=1e-12):
        raise ValueError("risk_matrix must be symmetric and match coefficient")
    if np.min(np.linalg.eigvalsh(Gamma)) <= 0.0 or float(c @ c) == 0.0:
        raise ValueError("risk_matrix must be positive definite and coefficient nonzero")
    q = float(c @ np.linalg.solve(Gamma, c))
    return 0.5 * q * delta ** (2.0 * order + 1.0) / (2.0 * order + 1.0)


def polynomial_uncertifiable_terminal_layer(
    information_threshold_nats: float,
    coefficient,
    noise_cov,
    order: float,
) -> float:
    """Width of the terminal layer with insufficient remaining information.

    Exact when Delta(T-u)=c*u^p with constant covariance on the terminal
    interval.  For a general crossing it is the corresponding local asymptotic
    scale whenever the critical layer lies inside the local regime.
    """
    import numpy as np

    Icrit, order = float(information_threshold_nats), float(order)
    c = np.asarray(coefficient, dtype=float).reshape(-1)
    Sigma = np.asarray(noise_cov, dtype=float)
    if Icrit < 0.0 or order <= -0.5:
        raise ValueError("require information_threshold_nats>=0 and order>-1/2")
    if Icrit == 0.0:
        return 0.0
    if Sigma.shape != (c.size, c.size) or not np.allclose(Sigma, Sigma.T, atol=1e-12):
        raise ValueError("noise_cov must be symmetric and match coefficient")
    if np.min(np.linalg.eigvalsh(Sigma)) <= 0.0 or float(c @ c) == 0.0:
        raise ValueError("noise_cov must be positive definite and coefficient nonzero")
    q = float(c @ np.linalg.solve(Sigma, c))
    return (2.0 * (2.0 * order + 1.0) * Icrit / q) ** (1.0 / (2.0 * order + 1.0))


def minimum_exponential_half_life(
    sharpe: float,
    alpha: float,
    beta: float,
    *,
    multiplicity: int = 1,
    correction: str = "bonferroni",
) -> dict:
    """Minimum exponential alpha half-life for reliable certification.

    In the zero-hurdle model, with time measured in the same units used for the
    instantaneous Sharpe S=mu0/sigma,

        I_life = S^2 h/(4 log 2).

    The returned h_min is therefore exact within this canonical model and the
    declared multiplicity allocation.
    """
    S = abs(float(sharpe))
    if S <= 0.0:
        raise ValueError("sharpe must be nonzero")
    thr = multiplicity_reliable_information_threshold(
        alpha, beta, multiplicity, method=correction
    )
    h = 4.0 * math.log(2.0) * float(thr["I_crit_nats"]) / (S * S)
    return {**thr, "sharpe": S, "half_life_min": h}


def search_breadth_capacity(
    lifetime_information_nats: float,
    alpha_family: float,
    beta: float,
    *,
    method: str = "bonferroni",
) -> dict:
    """Maximum integer candidate breadth supported by the lifetime information.

    For Bonferroni this is an exact inversion of the canonical per-test
    feasibility condition alpha_family/M >= alpha_min(I,beta), while preserving
    FWER under arbitrary dependence.  For Sidak it is the analogous inversion
    under the independence model that justifies Sidak.
    """
    I, alpha_family, beta = map(float, (lifetime_information_nats, alpha_family, beta))
    if I < 0.0 or not (0.0 < alpha_family < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("require I>=0 and alpha_family,beta in (0,1)")
    method = str(method).lower()
    z_beta = _N.inv_cdf(1.0 - beta)
    q = math.sqrt(2.0 * I) - z_beta
    alpha_min = 0.5 * math.erfc(q / math.sqrt(2.0))
    if alpha_min <= 0.0:
        return {"max_candidates": math.inf, "max_candidates_real": math.inf,
                "minimum_per_test_alpha": 0.0, "correction": method}
    if method == "bonferroni":
        real = alpha_family / alpha_min
    elif method == "sidak":
        real = math.log1p(-alpha_family) / math.log1p(-alpha_min)
    else:
        raise ValueError("method must be 'bonferroni' or 'sidak'")
    max_int = max(0, int(math.floor(real + 1e-12)))
    return {
        "max_candidates": max_int,
        "max_candidates_real": real,
        "minimum_per_test_alpha": alpha_min,
        "correction": method,
    }


def minimum_type1_budget_for_information(lifetime_information_nats: float, beta: float) -> float:
    """Smallest marginal type-I level that can attain power 1-beta.

    Invert the canonical Gaussian power envelope with A=2I:

        alpha_min(I,beta) = barPhi(sqrt(2I)-z_{1-beta}).

    This is a property of one simple-vs-simple candidate experiment and does not
    by itself specify how familywise error is controlled across candidates.
    """
    I, beta = float(lifetime_information_nats), float(beta)
    if I < 0.0 or not (0.0 < beta < 1.0):
        raise ValueError("require lifetime_information_nats>=0 and beta in (0,1)")
    z_beta = _N.inv_cdf(1.0 - beta)
    q = math.sqrt(2.0 * I) - z_beta
    return 0.5 * math.erfc(q / math.sqrt(2.0))


def heterogeneous_bonferroni_frontier(
    lifetime_information_nats,
    alpha_family: float,
    beta,
) -> dict:
    """Exact error-budget frontier for separable Bonferroni certification.

    Candidate j with lifetime information I_j and power target 1-beta_j needs at
    least alpha_j^*=barPhi(sqrt(2I_j)-z_{1-beta_j}) marginal type-I budget.
    Under the dependence-robust Bonferroni architecture, all candidates can be
    certified at their targets iff sum_j alpha_j^* <= alpha_family.

    The maximum *number* of candidates that can be made feasible is obtained by
    sorting alpha_j^* increasingly and taking the longest prefix whose sum fits
    inside alpha_family.  This is exact for cardinality because every selected
    candidate has unit value; weighted selection becomes a 0-1 knapsack problem.
    """
    import numpy as np

    I = np.asarray(lifetime_information_nats, dtype=float).reshape(-1)
    if I.size == 0 or np.any(I < 0.0):
        raise ValueError("lifetime_information_nats must be a nonempty nonnegative vector")
    alpha_family = float(alpha_family)
    if not (0.0 < alpha_family < 1.0):
        raise ValueError("alpha_family must be in (0,1)")
    if np.isscalar(beta):
        B = np.full(I.size, float(beta), dtype=float)
    else:
        B = np.asarray(beta, dtype=float).reshape(-1)
        if B.size != I.size:
            raise ValueError("beta must be scalar or match lifetime_information_nats")
    if np.any((B <= 0.0) | (B >= 1.0)):
        raise ValueError("all beta values must lie in (0,1)")
    budgets = np.array([
        minimum_type1_budget_for_information(float(ii), float(bb))
        for ii, bb in zip(I, B)
    ])
    order = np.argsort(budgets, kind="stable")
    csum = np.cumsum(budgets[order])
    k = int(np.searchsorted(csum, alpha_family, side="right"))
    selected = order[:k]
    return {
        "minimum_type1_budgets": budgets,
        "total_budget_required_all": float(np.sum(budgets)),
        "all_feasible": bool(float(np.sum(budgets)) <= alpha_family + 1e-15),
        "max_cardinality": k,
        "max_cardinality_indices": selected,
        "budget_used_max_cardinality": float(np.sum(budgets[selected])) if k else 0.0,
        "family_alpha": alpha_family,
    }


def minimum_type1_budget_from_binary_kl(information_nats: float, beta: float) -> float:
    """Necessary marginal type-I budget implied by a KL information budget.

    Let p=1-beta be the requested power.  Data processing implies that any
    test/event with type-I probability q and power at least p must satisfy

        I >= kl(p,q).

    For fixed I and p, this function returns the unique q in (0,p] satisfying
    kl(p,q)=I.  Allocating a smaller type-I budget is information-theoretically
    impossible for that pair of path laws.  Unlike
    ``minimum_type1_budget_for_information``, this is a portable KL converse;
    it is necessary, not generally sufficient, and is not tied to the exact
    simple Gaussian power envelope.
    """
    I, beta = float(information_nats), float(beta)
    if I < 0.0 or not (0.0 < beta < 1.0):
        raise ValueError("require information_nats>=0 and beta in (0,1)")
    p = 1.0 - beta
    if I == 0.0:
        return p
    if brentq is None:
        raise RuntimeError("scipy is required")

    logp = math.log(p)
    # Solve in log q to remain stable when the required type-I budget is tiny.
    def f(logq: float) -> float:
        q = math.exp(logq)
        return binary_kl(p, q) - I

    lo = -50.0
    while f(lo) < 0.0 and lo > -740.0:
        lo = max(-745.0, lo * 1.5)
    if f(lo) < 0.0:
        # The mathematical root is below floating-point range.
        return 0.0
    root = brentq(f, lo, logp, xtol=1e-13, rtol=1e-13, maxiter=300)
    return float(math.exp(root))


def student_t_reliable_information_threshold(alpha: float, beta: float, df: int) -> dict:
    """Exact terminal one-sided Student-t power frontier for unknown scale.

    For a Gaussian no-intercept regression with fixed design energy Q and
    unknown constant variance sigma^2, the usual one-sided t statistic has a
    noncentral-t distribution with ``df`` residual degrees of freedom and
    noncentrality

        delta = (theta-b) * sqrt(Q) / sigma.

    This function solves for the smallest delta whose level-alpha t test has
    power 1-beta and reports I_equiv=delta^2/2.  The latter is the KL that the
    same coefficient displacement would generate if sigma were known; it is a
    convenient information-equivalent scale, not a claim that Studentization
    leaves the finite-sample likelihood experiment unchanged.
    """
    alpha, beta = float(alpha), float(beta)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("require alpha,beta in (0,1)")
    if not isinstance(df, int) or isinstance(df, bool) or df < 1:
        raise ValueError("df must be a positive integer")
    try:
        from scipy.stats import t as student_t, nct
    except Exception as exc:  # pragma: no cover
        raise RuntimeError("scipy.stats is required") from exc
    target = 1.0 - beta
    critical = float(student_t.ppf(1.0 - alpha, df))
    if target <= alpha:
        delta = 0.0
    else:
        def power(d: float) -> float:
            return float(nct.sf(critical, df, d))
        hi = 1.0
        while power(hi) < target:
            hi *= 2.0
            if hi > 1e6:
                raise RuntimeError("failed to bracket noncentral-t power threshold")
        delta = float(brentq(lambda d: power(d) - target, 0.0, hi,
                             xtol=1e-13, rtol=1e-13, maxiter=300))
    known = reliable_information_threshold(alpha, beta)["I_crit_nats"]
    equiv = 0.5 * delta * delta
    return {
        "df": int(df),
        "alpha": alpha,
        "beta": beta,
        "critical_t": critical,
        "noncentrality_crit": delta,
        "I_equivalent_nats": equiv,
        "known_scale_I_crit_nats": float(known),
        "finite_sample_unknown_scale_penalty": (equiv / known) if known > 0 else math.nan,
    }


def ar1_expected_design_energy(
    current_signal: float,
    phi: float,
    signal_innovation_variance: float,
    horizon: int,
) -> float:
    """E[sum_{k=0}^{H-1} X_k^2 | X_0=x] for a Gaussian AR(1).

    X_{k+1}=phi X_k+xi_{k+1}, Var(xi)=signal_innovation_variance.
    The recursion E[X_{k+1}^2]=phi^2 E[X_k^2]+variance is numerically stable
    even very near the unit-root boundary.
    """
    x, phi, vx = map(float, (current_signal, phi, signal_innovation_variance))
    if not (math.isfinite(x) and math.isfinite(phi) and math.isfinite(vx)):
        raise ValueError("inputs must be finite")
    if not -1.0 < phi < 1.0:
        raise ValueError("phi must lie in (-1,1)")
    if vx < 0.0:
        raise ValueError("signal_innovation_variance must be nonnegative")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")
    second = x * x
    total = second
    r = phi * phi
    for _ in range(1, horizon):
        second = r * second + vx
        total += second
    return float(total)


def _min_abs_on_interval(lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("invalid interval")
    if lo <= 0.0 <= hi:
        return 0.0
    return min(abs(lo), abs(hi))


def _max_abs_on_interval(lo: float, hi: float) -> float:
    if lo > hi:
        raise ValueError("invalid interval")
    return max(abs(lo), abs(hi))


def g2_expected_kl_bounds(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Exact expected pairwise-KL bounds over the G2 projection rectangle.

    The future model is

        X_{k+1}=phi X_k+xi_{k+1},
        Y_{k+1}=theta X_k+eps_{k+1},

    with innovation variances v_X and v_R.  For each compatible alternative
    world we compare to its *boundary companion* with coefficient ``b`` and the
    same nuisance parameters (phi,v_R,v_X).  Conditional on the predictor path,
    the KL is (theta-b)^2 sum X_k^2/(2 v_R), hence

        E KL = (theta-b)^2 E Q_H /(2 v_R).

    Because E Q_H is a polynomial in r=phi^2 and v_X with nonnegative
    coefficients, its exact extrema on a rectangular G2 projection occur at
    the smallest/largest |phi| and the variance endpoints.  ``direction=1``
    treats theta>b as the alternative; ``direction=-1`` treats theta<b.

    A zero lower bound when the coefficient interval touches/crosses the
    economic boundary is intentional: uniform robust power over that rectangle
    cannot be guaranteed while a boundary-compatible world remains.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if not isinstance(projection_rectangle, dict):
        raise ValueError("projection_rectangle must be a dictionary")
    try:
        tlo, thi = map(float, projection_rectangle["theta"])
        plo, phi = map(float, projection_rectangle["phi"])
        vrlo, vrhi = map(float, projection_rectangle["return_noise_variance"])
        vxlo, vxhi = map(float, projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing required G2 intervals") from exc
    b = float(economic_boundary)
    x = float(current_signal)
    if not all(math.isfinite(v) for v in (tlo, thi, plo, phi, vrlo, vrhi, vxlo, vxhi, b, x)):
        raise ValueError("rectangle endpoints and scalar inputs must be finite")
    if not (tlo <= thi and -1.0 < plo <= phi < 1.0):
        raise ValueError("invalid theta/phi intervals")
    if not (0.0 < vrlo <= vrhi and 0.0 <= vxlo <= vxhi):
        raise ValueError("invalid variance intervals")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 1:
        raise ValueError("horizon must be a positive integer")

    d1 = direction * (tlo - b)
    d2 = direction * (thi - b)
    dlo, dhi = min(d1, d2), max(d1, d2)
    gap_lower = max(0.0, dlo)
    gap_upper = max(0.0, dhi)

    amin = _min_abs_on_interval(plo, phi)
    amax = _max_abs_on_interval(plo, phi)
    # Preserve an interval endpoint sign only for reporting; E Q depends on phi^2.
    phi_min_abs = amin
    phi_max_abs = amax
    qmin = ar1_expected_design_energy(x, phi_min_abs, vxlo, horizon)
    qmax = ar1_expected_design_energy(x, phi_max_abs, vxhi, horizon)
    lower = (gap_lower * gap_lower) * qmin / (2.0 * vrhi)
    upper = (gap_upper * gap_upper) * qmax / (2.0 * vrlo)
    return {
        "expected_kl_lower_nats": float(lower),
        "expected_kl_upper_nats": float(upper),
        "expected_design_energy_lower": float(qmin),
        "expected_design_energy_upper": float(qmax),
        "coefficient_gap_lower": float(gap_lower),
        "coefficient_gap_upper": float(gap_upper),
        "phi_abs_lower": float(phi_min_abs),
        "phi_abs_upper": float(phi_max_abs),
        "return_variance_worst_for_lower": float(vrhi),
        "signal_variance_worst_for_lower": float(vxlo),
        "economic_boundary": b,
        "direction": int(direction),
        "horizon": int(horizon),
        "has_compatible_alternative": bool(gap_upper > 0.0),
        "uniform_alternative_separation": bool(gap_lower > 0.0),
        "interpretation": (
            "Expected path-law KL to each world's economic-boundary companion. "
            "Bounds are exact over the rectangular G2 projection for this expected-KL functional."
        ),
    }


def composite_g2_survival_frontier(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Portable composite survival veto from the G2 uncertainty rectangle.

    If a fresh-start deployment rule must have type-I probability <=alpha for
    every boundary companion and power >=1-beta for every compatible
    alternative world, data processing requires, for every such pair,

        D(P_alt || P_boundary) >= kl(1-beta, alpha).

    This function compares that universal necessary threshold with the exact
    lower/upper expected pairwise-KL bounds from ``g2_expected_kl_bounds``.

    Status meanings:
      * ALL_COMPATIBLE_WORLDS_KL_INSUFFICIENT: even the largest compatible
        expected KL is below the necessary threshold;
      * MIXED_OR_UNRESOLVED: some compatible worlds fail the pairwise barrier;
      * PAIRWISE_KL_BARRIER_CLEARED_UNIFORMLY: every compatible world clears
        this necessary pairwise barrier.  This is *not* a sufficient composite
        certification theorem; nuisance complexity and the chosen sequential
        test still matter.
    """
    alpha, beta = float(alpha), float(beta)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("require alpha,beta in (0,1)")
    p = 1.0 - beta
    if p <= alpha:
        raise ValueError("reliable-certification interpretation requires 1-beta > alpha")
    bounds = g2_expected_kl_bounds(
        projection_rectangle, current_signal, horizon,
        economic_boundary=economic_boundary, direction=direction,
    )
    threshold = binary_kl(p, alpha)
    lo = float(bounds["expected_kl_lower_nats"])
    hi = float(bounds["expected_kl_upper_nats"])
    if not bounds["has_compatible_alternative"]:
        status = "NO_ALTERNATIVE_SIDE_IN_SET"
    elif hi < threshold * (1.0 - 1e-12):
        status = "ALL_COMPATIBLE_WORLDS_KL_INSUFFICIENT"
    elif lo >= threshold * (1.0 - 1e-12):
        status = "PAIRWISE_KL_BARRIER_CLEARED_UNIFORMLY"
    else:
        status = "MIXED_OR_UNRESOLVED"
    return {
        **bounds,
        "alpha": alpha,
        "beta": beta,
        "binary_kl_threshold_nats": float(threshold),
        "robust_pairwise_ratio": (lo / threshold) if threshold > 0.0 else math.inf,
        "optimistic_pairwise_ratio": (hi / threshold) if threshold > 0.0 else math.inf,
        "status": status,
        "necessary_not_sufficient": True,
    }


def minimum_horizon_to_clear_g2_kl_barrier(
    projection_rectangle: dict,
    current_signal: float,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    max_horizon: int = 100000,
) -> dict:
    """Earliest H for which every compatible world clears the pairwise KL veto.

    The result is a *necessary-condition horizon*: at or beyond it the universal
    pairwise KL obstruction has disappeared over the frozen G2 rectangle.  It
    does not assert that the existing NIG e-process, or any other particular
    composite procedure, has already achieved the requested power.
    """
    if not isinstance(max_horizon, int) or isinstance(max_horizon, bool) or max_horizon < 1:
        raise ValueError("max_horizon must be a positive integer")
    first = composite_g2_survival_frontier(
        projection_rectangle, current_signal, 1, alpha, beta,
        economic_boundary=economic_boundary, direction=direction,
    )
    if not first["has_compatible_alternative"]:
        return {"minimum_horizon": None, "reason": "no_alternative_side_in_set", **first}
    if not first["uniform_alternative_separation"]:
        return {
            "minimum_horizon": None,
            "reason": "rectangle_touches_or_crosses_economic_boundary",
            **first,
        }
    if first["status"] == "PAIRWISE_KL_BARRIER_CLEARED_UNIFORMLY":
        return {"minimum_horizon": 1, "reason": "cleared", **first}

    def cleared(H: int) -> bool:
        o = composite_g2_survival_frontier(
            projection_rectangle, current_signal, H, alpha, beta,
            economic_boundary=economic_boundary, direction=direction,
        )
        return o["expected_kl_lower_nats"] >= o["binary_kl_threshold_nats"]

    hi = 2
    while hi <= max_horizon and not cleared(hi):
        hi *= 2
    if hi > max_horizon:
        hi = max_horizon
        if not cleared(hi):
            last = composite_g2_survival_frontier(
                projection_rectangle, current_signal, hi, alpha, beta,
                economic_boundary=economic_boundary, direction=direction,
            )
            return {"minimum_horizon": None, "reason": "not_cleared_within_max_horizon", **last}
    lo = max(1, hi // 2)
    while lo + 1 < hi:
        mid = (lo + hi) // 2
        if cleared(mid):
            hi = mid
        else:
            lo = mid
    out = composite_g2_survival_frontier(
        projection_rectangle, current_signal, hi, alpha, beta,
        economic_boundary=economic_boundary, direction=direction,
    )
    return {"minimum_horizon": int(hi), "reason": "cleared", **out}


def g2_stationary_kl_rate_bounds(
    projection_rectangle: dict,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Exact per-observation KL-rate bounds in the stationary G2 submodel.

    Under X_{t+1}=phi X_t+xi_{t+1} with stationary variance
    v_X/(1-phi^2), comparison of theta to its boundary companion b in the
    return regression gives expected KL per predictor/return observation

        kappa = (theta-b)^2 v_X / [2 v_R (1-phi^2)].

    The extrema use coefficient-gap, variance, and |phi| extrema; the
    smallest |phi| is zero when its interval contains zero.  The statement uses the orthogonal-
    innovations (or conditionally exogenous return-noise) submodel.  It is also
    a valid hard-world construction for a uniform guarantee over a larger G2
    class that contains this submodel.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    try:
        tlo, thi = map(float, projection_rectangle["theta"])
        plo, phi = map(float, projection_rectangle["phi"])
        vrlo, vrhi = map(float, projection_rectangle["return_noise_variance"])
        vxlo, vxhi = map(float, projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing required G2 intervals") from exc
    b = float(economic_boundary)
    if not all(math.isfinite(v) for v in (tlo, thi, plo, phi, vrlo, vrhi, vxlo, vxhi, b)):
        raise ValueError("rectangle endpoints and boundary must be finite")
    if not (tlo <= thi and -1.0 < plo <= phi < 1.0):
        raise ValueError("invalid theta/phi intervals")
    if not (0.0 < vrlo <= vrhi and 0.0 <= vxlo <= vxhi):
        raise ValueError("invalid variance intervals")
    d1, d2 = direction*(tlo-b), direction*(thi-b)
    dlo, dhi = min(d1,d2), max(d1,d2)
    gap_lo, gap_hi = max(0.0,dlo), max(0.0,dhi)
    a_lo = _min_abs_on_interval(plo,phi)
    a_hi = _max_abs_on_interval(plo,phi)
    rate_lo = gap_lo*gap_lo*vxlo/(2.0*vrhi*(1.0-a_lo*a_lo))
    rate_hi = gap_hi*gap_hi*vxhi/(2.0*vrlo*(1.0-a_hi*a_hi))
    return {
        "kl_rate_lower_nats_per_observation": float(rate_lo),
        "kl_rate_upper_nats_per_observation": float(rate_hi),
        "coefficient_gap_lower": float(gap_lo),
        "coefficient_gap_upper": float(gap_hi),
        "phi_abs_lower": float(a_lo),
        "phi_abs_upper": float(a_hi),
        "economic_boundary": b,
        "direction": int(direction),
        "uniform_alternative_separation": bool(gap_lo>0.0),
        "has_compatible_alternative": bool(gap_hi>0.0),
    }


def minimum_stationary_horizon_to_clear_g2_kl_barrier(
    projection_rectangle: dict,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Closed-form stationary horizon needed to clear the universal KL veto.

    If the G2 rectangle touches the economic boundary or permits zero signal
    innovation variance, the robust KL-rate floor is zero and no finite horizon
    can clear the pairwise obstruction uniformly over the frozen rectangle.
    Otherwise H_min=ceil(kl(1-beta,alpha)/kappa_lower).
    """
    alpha,beta = float(alpha),float(beta)
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    rates=g2_stationary_kl_rate_bounds(
        projection_rectangle,economic_boundary=economic_boundary,direction=direction
    )
    K=binary_kl(1.0-beta,alpha)
    k=float(rates["kl_rate_lower_nats_per_observation"])
    if not rates["has_compatible_alternative"]:
        H=None; reason="no_alternative_side_in_set"
    elif k<=0.0:
        H=None; reason="zero_robust_information_rate"
    else:
        H=max(1,int(math.ceil(K/k-1e-15))); reason="cleared"
    return {**rates,"binary_kl_threshold_nats":K,"minimum_horizon":H,"reason":reason}


def g2_discrete_information_economic_identity(
    theta: float,
    boundary: float,
    phi: float,
    return_noise_variance: float,
    signal_innovation_variance: float,
    *,
    gamma: float,
    trading_cost: float,
    horizon: int,
    terminal_penalty: float = 0.0,
) -> dict:
    """Discrete-time G2 boundary-regret / information conservation identity.

    Assume the stationary AR(1) signal and the finite-horizon quadratic trading
    model used by the G2 certificate.  Let c_t(phi),D_t be the Bellman gain
    shape and curvature.  Using the boundary-world oracle b*c_t in the true
    theta world incurs expected oracle regret

        R = v_X/(2(1-phi^2)) * (theta-b)^2 * sum_t D_t c_t^2.

    The path-law KL in H return observations between theta and the boundary
    companion b is

        I = H * v_X/(2(1-phi^2)) * (theta-b)^2 / v_R.

    Therefore, for theta != b,

        R/I = v_R * (1/H) sum_t D_t c_t^2.

    Signal scale v_X and coefficient separation theta-b cancel exactly.  With
    zero trading cost and zero terminal penalty, D_t*c_t^2=1/gamma and the
    conversion factor reduces to v_R/gamma, the discrete analogue of the
    scalar continuous-time CAC price-per-nat identity.
    """
    theta,boundary,phi,vR,vX,gamma,trading_cost,terminal_penalty = map(
        float,(theta,boundary,phi,return_noise_variance,signal_innovation_variance,
               gamma,trading_cost,terminal_penalty)
    )
    if not -1.0<phi<1.0:
        raise ValueError("phi must lie in (-1,1)")
    if vR<=0.0 or vX<=0.0 or gamma<=0.0 or trading_cost<0.0 or terminal_penalty<0.0:
        raise ValueError("require positive variances/gamma and nonnegative costs/terminal penalty")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    # Same backward Bellman recursion used in alphavalue.unknown_scale.
    D=[0.0]*horizon; c=[0.0]*horizon
    K=terminal_penalty; L=0.0
    for t in range(horizon-1,-1,-1):
        Dt=gamma+trading_cost+K
        ct=(1.0+phi*L)/Dt
        D[t]=Dt; c[t]=ct
        K=trading_cost*((gamma+K)/Dt)
        L=trading_cost*ct
    curvature_sum=sum(Dt*ct*ct for Dt,ct in zip(D,c))
    stationary_var=vX/(1.0-phi*phi)
    gap=theta-boundary
    regret=0.5*stationary_var*gap*gap*curvature_sum
    info=0.5*stationary_var*gap*gap*horizon/vR
    price=vR*curvature_sum/horizon
    return {
        "economic_boundary_regret": float(regret),
        "statistical_information_nats": float(info),
        "price_per_nat": float(price),
        "bellman_curvature_average": float(curvature_sum/horizon),
        "bellman_curvature_sum": float(curvature_sum),
        "stationary_signal_variance": float(stationary_var),
        "identity_residual": float(regret-price*info),
        "horizon": int(horizon),
    }


def g2_stationary_t_power_lower_bound(
    projection_rectangle: dict,
    horizon: int,
    alpha: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 121,
) -> dict:
    """Rigorous uniform lower bound for terminal one-sided t-test power in G2.

    Assumptions
    -----------
    * The signal is started in its stationary Gaussian AR(1) law.
    * Return innovations are Gaussian and conditionally independent of the
      signal path (the orthogonal-innovation G2 submodel).
    * We test the coefficient boundary ``theta=b`` versus the selected side.
    * The terminal test is the usual no-intercept Gaussian-regression t-test,
      using ``H`` predictor/return pairs and ``H-1`` residual degrees of
      freedom.

    Conditional on the realized design X=(X_0,...,X_{H-1}), the t-statistic is
    noncentral t under an alternative, with noncentrality

        delta=(theta-b)*sqrt(Q/v_R),   Q=sum X_t^2.

    For the stationary AR(1) covariance C, the precision matrix/Gershgorin
    bound gives

        lambda_min(C) >= v_X/(1+|phi|)^2.

    Hence uniformly over a rectangular G2 projection,

        Q >=_st c_design * U,   U ~ chi-square_H,
        c_design = vX_lower/(1+max|phi|)^2.

    For every eta in (0,1), on the event U >= chi2_ppf(eta;H), which has
    probability 1-eta, conditional power is at least the noncentral-t power at
    the corresponding noncentrality.  Therefore

        Power >= (1-eta) * Power_t(delta_eta).

    We maximize this *family of valid lower bounds* numerically over eta.  The
    returned value remains a lower bound even if the numerical maximization is
    imperfect, because every evaluated eta is individually valid.

    This is an achievability bound for a fixed terminal horizon.  It is not a
    claim about the power of the package's particular NIG mixture e-process.
    """
    if direction not in (-1, 1):
        raise ValueError("direction must be +1 or -1")
    if not isinstance(horizon, int) or isinstance(horizon, bool) or horizon < 2:
        raise ValueError("horizon must be an integer at least 2 for a t-test")
    alpha = float(alpha)
    if not (0.0 < alpha < 1.0):
        raise ValueError("alpha must lie in (0,1)")
    if not isinstance(eta_grid_size, int) or eta_grid_size < 9:
        raise ValueError("eta_grid_size must be an integer at least 9")
    try:
        tlo, thi = map(float, projection_rectangle["theta"])
        plo, phi = map(float, projection_rectangle["phi"])
        vrlo, vrhi = map(float, projection_rectangle["return_noise_variance"])
        vxlo, vxhi = map(float, projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing required G2 intervals") from exc
    b = float(economic_boundary)
    if not all(math.isfinite(v) for v in (tlo,thi,plo,phi,vrlo,vrhi,vxlo,vxhi,b)):
        raise ValueError("rectangle endpoints and boundary must be finite")
    if not (tlo <= thi and -1.0 < plo <= phi < 1.0):
        raise ValueError("invalid theta/phi intervals")
    if not (0.0 < vrlo <= vrhi and 0.0 <= vxlo <= vxhi):
        raise ValueError("invalid variance intervals")
    d1, d2 = direction*(tlo-b), direction*(thi-b)
    dlo, dhi = min(d1,d2), max(d1,d2)
    gap_lo, gap_hi = max(0.0,dlo), max(0.0,dhi)
    a_hi = _max_abs_on_interval(plo,phi)
    design_eigen_floor = vxlo / ((1.0 + a_hi)**2)

    from scipy.stats import chi2, nct, t as student_t

    df = horizon - 1
    crit = float(student_t.ppf(1.0-alpha, df))

    # Search on a logit-like grid so that very small and very large eta values
    # are both represented.  Each grid point itself gives a rigorous bound;
    # taking their maximum therefore preserves validity.
    L = 12.0
    zs = np.linspace(-L, L, eta_grid_size)
    etas = 1.0/(1.0+np.exp(-zs))
    best = (-1.0, None, None, None)
    for eta in etas:
        q = float(chi2.ppf(float(eta), horizon))
        if not math.isfinite(q) or q < 0.0:
            continue
        delta = gap_lo * math.sqrt(max(0.0, design_eigen_floor*q/vrhi))
        cond_power = float(nct.sf(crit, df, delta))
        bound = (1.0-float(eta))*cond_power
        if bound > best[0]:
            best = (bound, float(eta), q, delta)
    bound, eta_star, q_star, delta_star = best
    if eta_star is None:
        raise RuntimeError("failed to evaluate chi-square power lower bound")
    cond_power_star = float(nct.sf(crit, df, delta_star))
    return {
        "power_lower_bound": float(max(0.0,min(1.0,bound))),
        "eta": float(eta_star),
        "chi2_quantile": float(q_star),
        "noncentrality_floor_on_event": float(delta_star),
        "conditional_power_floor_on_event": float(cond_power_star),
        "t_critical": crit,
        "degrees_of_freedom": int(df),
        "coefficient_gap_lower": float(gap_lo),
        "coefficient_gap_upper": float(gap_hi),
        "max_abs_phi": float(a_hi),
        "stationary_design_eigenvalue_floor": float(design_eigen_floor),
        "return_noise_variance_upper": float(vrhi),
        "has_compatible_alternative": bool(gap_hi>0.0),
        "uniform_alternative_separation": bool(gap_lo>0.0),
        "assumption_scope": "stationary Gaussian AR(1), orthogonal/conditionally exogenous return innovations",
        "terminal_t_test_not_nig_eprocess": True,
    }


def minimum_stationary_horizon_for_g2_t_power(
    projection_rectangle: dict,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    max_horizon: int = 5000,
    eta_grid_size: int = 121,
) -> dict:
    """First horizon H whose rigorous t-power lower bound reaches 1-beta.

    This is a sufficient *terminal-certification* horizon in the stationary
    orthogonal-innovation G2 submodel.  It is deliberately distinguished from
    the earlier KL barrier, which is only necessary, and from the NIG
    likelihood-mixture e-process actually used by AlphaValue.
    """
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    if not isinstance(max_horizon,int) or isinstance(max_horizon,bool) or max_horizon<2:
        raise ValueError("max_horizon must be an integer at least 2")
    target=1.0-beta
    last=None
    for H in range(2,max_horizon+1):
        out=g2_stationary_t_power_lower_bound(
            projection_rectangle,H,alpha,
            economic_boundary=economic_boundary,direction=direction,
            eta_grid_size=eta_grid_size,
        )
        last=out
        if not out["has_compatible_alternative"]:
            return {**out,"minimum_horizon":None,"reason":"no_alternative_side_in_set","target_power":target}
        if not out["uniform_alternative_separation"] or out["stationary_design_eigenvalue_floor"]<=0.0:
            return {**out,"minimum_horizon":None,"reason":"zero_uniform_separation_or_design_floor","target_power":target}
        if out["power_lower_bound"] >= target*(1.0-1e-12):
            return {**out,"minimum_horizon":int(H),"reason":"cleared","target_power":target}
    return {**(last or {}),"minimum_horizon":None,"reason":"not_cleared_within_max_horizon","target_power":target}


def composite_g2_certifiability_sandwich(
    projection_rectangle: dict,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    max_horizon: int = 5000,
    eta_grid_size: int = 121,
) -> dict:
    """Necessary/sufficient horizon sandwich for stationary composite G2.

    Lower endpoint: binary-data-processing KL obstruction.  No uniformly valid
    procedure with the declared type-I/power guarantees can exist below this
    horizon in the orthogonal-innovation submodel.

    Upper endpoint: a concrete ordinary terminal Gaussian-regression t-test is
    guaranteed to attain the requested power uniformly over the frozen
    rectangle by ``g2_stationary_t_power_lower_bound``.

    Thus, when both endpoints are finite, the unknown optimal terminal horizon
    H_* obeys

        H_KL_necessary <= H_* <= H_t_sufficient.

    The upper endpoint does *not* certify the current NIG e-process power.  It
    proves existence of a statistically valid terminal procedure under the
    stated model assumptions.
    """
    nec=minimum_stationary_horizon_to_clear_g2_kl_barrier(
        projection_rectangle,alpha,beta,
        economic_boundary=economic_boundary,direction=direction,
    )
    suf=minimum_stationary_horizon_for_g2_t_power(
        projection_rectangle,alpha,beta,
        economic_boundary=economic_boundary,direction=direction,
        max_horizon=max_horizon,eta_grid_size=eta_grid_size,
    )
    Hn=nec.get("minimum_horizon")
    Hs=suf.get("minimum_horizon")
    if Hn is None:
        status="NO_FINITE_UNIFORM_KL_CLEARANCE_FROM_FROZEN_RECTANGLE"
    elif Hs is None:
        status="NECESSARY_BARRIER_CLEARS_BUT_T_SUFFICIENCY_NOT_PROVED_WITHIN_RANGE"
    else:
        status="CERTIFIABILITY_HORIZON_SANDWICHED"
    return {
        "status":status,
        "necessary_horizon_kl":Hn,
        "sufficient_horizon_terminal_t":Hs,
        "sandwich_width": (int(Hs-Hn) if Hn is not None and Hs is not None else None),
        "alpha":float(alpha),"beta":float(beta),
        "necessary":nec,"sufficient":suf,
        "scope":"stationary Gaussian G2 orthogonal-innovation submodel; terminal t upper bound, pairwise-KL lower bound",
    }


def g2_observed_design_certifiability_frontier(
    projection_rectangle: dict,
    design_energy: float,
    sample_size: int,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Exact conditional-design necessary/sufficient frontier for unknown v_R.

    Consider a fresh, no-intercept Gaussian return regression

        Y_i = theta X_i + eps_i,   eps_i ~ iid N(0,v_R),

    and condition on a fixed or conditionally exogenous design with

        Q = sum_i X_i^2.

    Over a *declared/frozen* rectangular uncertainty set for (theta,v_R), let
    d_- be the smallest coefficient distance to the economic boundary on the
    chosen alternative side and let v_R^+ be the largest return variance.
    Then the smallest conditional pairwise KL over compatible alternatives is

        I_- = d_-^2 Q / (2 v_R^+).

    Necessity: if I_- < kl(1-beta,alpha), at least one compatible world fails
    the binary data-processing requirement, so no test can uniformly have
    type-I <= alpha and power >= 1-beta.

    Sufficiency: the ordinary one-sided regression t-test has conditional
    noncentrality delta=(theta-b)sqrt(Q/v_R).  If

        I_- >= I_t(alpha,beta,nu),  nu=n-1,

    where I_t=delta_*^2/2 is the exact noncentral-t threshold, that concrete
    terminal t-test has power at least 1-beta for *every* compatible world and
    exact conditional level alpha at the boundary.

    Thus the interval between the binary-KL and noncentral-t thresholds is an
    explicit finite-sample unresolved zone.  The signal-dynamics nuisance
    (phi,v_X) disappears conditional on realized design energy: it controls how
    quickly Q is generated, not the return-regression evidence threshold.

    The result requires that conditional on the entire design vector the
    errors remain iid Gaussian with variance v_R. Predictability alone is
    insufficient for that conditional law. The null must contain the
    same-nuisance boundary companion of the least-information alternative.  If the rectangle itself is estimated from earlier
    data, its coverage error must be budgeted separately (sample splitting or
    another valid composition argument); this function does not silently spend
    that error budget.
    """
    if direction not in (-1,1):
        raise ValueError("direction must be +1 or -1")
    Q=float(design_energy); alpha=float(alpha); beta=float(beta)
    if not math.isfinite(Q) or Q<=0.0:
        raise ValueError("design_energy must be positive and finite")
    if not isinstance(sample_size,int) or isinstance(sample_size,bool) or sample_size<2:
        raise ValueError("sample_size must be an integer at least 2")
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    try:
        tlo,thi=map(float,projection_rectangle["theta"])
        vrlo,vrhi=map(float,projection_rectangle["return_noise_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing theta/return variance intervals") from exc
    b=float(economic_boundary)
    if not all(math.isfinite(v) for v in (tlo,thi,vrlo,vrhi,b)):
        raise ValueError("rectangle endpoints and boundary must be finite")
    if tlo>thi or not (0.0<vrlo<=vrhi):
        raise ValueError("invalid theta/return variance intervals")
    d1,d2=direction*(tlo-b),direction*(thi-b)
    dlo,dhi=min(d1,d2),max(d1,d2)
    gap_lo,gap_hi=max(0.0,dlo),max(0.0,dhi)
    I_floor=gap_lo*gap_lo*Q/(2.0*vrhi)
    K=binary_kl(1.0-beta,alpha)
    It=student_t_reliable_information_threshold(alpha,beta,sample_size-1)
    I_suf=float(It["I_equivalent_nats"])
    if gap_hi<=0.0:
        status="NO_ALTERNATIVE_SIDE_IN_SET"
    elif gap_lo<=0.0:
        status="NO_UNIFORM_SEPARATION_FROM_BOUNDARY"
    elif I_floor < K*(1.0-1e-12):
        status="UNIFORMLY_IMPOSSIBLE_BY_PAIRWISE_KL"
    elif I_floor >= I_suf*(1.0-1e-12):
        status="UNIFORMLY_TERMINAL_T_CERTIFIABLE"
    else:
        status="FINITE_SAMPLE_UNRESOLVED_ZONE"
    if gap_lo>0.0:
        Qnec=2.0*vrhi*K/(gap_lo*gap_lo)
        Qsuf=2.0*vrhi*I_suf/(gap_lo*gap_lo)
    else:
        Qnec=math.inf; Qsuf=math.inf
    return {
        "status":status,
        "design_energy":Q,
        "sample_size":int(sample_size),
        "degrees_of_freedom":int(sample_size-1),
        "coefficient_gap_lower":float(gap_lo),
        "coefficient_gap_upper":float(gap_hi),
        "return_noise_variance_upper":float(vrhi),
        "robust_conditional_information_floor_nats":float(I_floor),
        "binary_kl_necessary_threshold_nats":float(K),
        "student_t_sufficient_threshold_nats":float(I_suf),
        "design_energy_necessary":float(Qnec),
        "design_energy_sufficient_terminal_t":float(Qsuf),
        "necessary_ratio":float(I_floor/K),
        "sufficient_ratio":float(I_floor/I_suf),
        "student_t_frontier":It,
        "alpha":alpha,"beta":beta,
        "economic_boundary":b,"direction":int(direction),
        "conditional_on_design":True,
        "phi_and_signal_variance_only_control_clock_speed":True,
        "estimated_set_error_budget_must_be_composed_separately":True,
    }


def predictable_gaussian_eprocess_information_threshold(alpha: float, beta: float) -> float:
    """Constructive robust information threshold for a horizon-matched e-process.

    Let a predictable Gaussian regression satisfy, after a freeze time,

        Y_{t+1}=theta X_t+eps_{t+1},
        eps_{t+1}|F_t ~ N(0,v),   v <= vbar,

    and suppose the alternative is separated from the one-sided economic null
    theta<=b by theta>=b+d, d>0.  For a declared target design-energy budget q,
    set

        lambda = sqrt(2 log(1/alpha)/(vbar q))

    and use

        E_t = exp(lambda sum X_s(Y_{s+1}-b X_s)
                  - .5 lambda^2 vbar sum X_s^2).

    This is a nonnegative supermartingale under every theta<=b, v<=vbar, so
    crossing 1/alpha is anytime-valid.  If the design clock reaches q under an
    alternative with gap at least d, a Gaussian-martingale lower-tail bound
    yields

        P(no crossing by q) <= exp[-(sqrt(I)-sqrt(L))^2],
        I=d^2 q/(2 vbar), L=log(1/alpha),

    whenever I>L.  Therefore power >=1-beta is guaranteed by

        I >= (sqrt(log(1/alpha))+sqrt(log(1/beta)))^2.

    This is a constructive sufficient threshold, not an optimality claim.
    """
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<1.0 and 0.0<beta<1.0):
        raise ValueError("alpha and beta must lie in (0,1)")
    L=math.log(1.0/alpha); B=math.log(1.0/beta)
    return float((math.sqrt(L)+math.sqrt(B))**2)


def predictable_gaussian_eprocess_power_lower_bound(information: float, alpha: float) -> float:
    """Lower bound on crossing power of the horizon-matched robust e-process.

    ``information`` is I=d^2 q/(2 vbar).  The bound is zero below the point at
    which the proof makes the relevant lower-tail deviation negative; above it
    the bound is 1-exp[-(sqrt(I)-sqrt(log(1/alpha)))^2].
    """
    I=float(information); alpha=float(alpha)
    if I<0.0 or not math.isfinite(I):
        raise ValueError("information must be finite and nonnegative")
    if not 0.0<alpha<1.0:
        raise ValueError("alpha must lie in (0,1)")
    L=math.log(1.0/alpha)
    if I<=L:
        return 0.0
    exponent=(math.sqrt(I)-math.sqrt(L))**2
    return float(1.0-math.exp(-exponent))


def g2_postfreeze_eprocess_frontier(
    projection_rectangle: dict,
    remaining_design_energy: float,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
) -> dict:
    """Post-freeze robust CAC frontier using a variance-envelope e-process.

    The rectangle is frozen before the *future* certification stream begins.
    On the good-set event it supplies a minimum alternative gap d_- and an
    upper return-noise variance vbar.  The future predictable design may be
    random/adaptive and contemporaneous signal/return innovations need not be
    orthogonal; only the declared conditional Gaussian return regression and
    v<=vbar are used by the e-process proof.

    The procedure is genuinely anytime-valid after the freeze time.  It uses a
    predeclared remaining design-energy budget q and a fixed horizon-matched
    betting fraction lambda.  The returned sufficient threshold guarantees
    future crossing power >=1-beta *conditional on the frozen rectangle being
    correct and the design clock reaching q before economic death*.

    If the rectangle came from an earlier confidence sequence with failure
    probability delta, that delta must be composed separately with the future
    type-I/power budgets.  This routine deliberately does not hide that cost.
    """
    if direction not in (-1,1):
        raise ValueError("direction must be +1 or -1")
    q=float(remaining_design_energy); alpha=float(alpha); beta=float(beta)
    if not math.isfinite(q) or q<=0.0:
        raise ValueError("remaining_design_energy must be positive and finite")
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    try:
        tlo,thi=map(float,projection_rectangle["theta"])
        vrlo,vrhi=map(float,projection_rectangle["return_noise_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing theta/return variance intervals") from exc
    b=float(economic_boundary)
    if not all(math.isfinite(v) for v in (tlo,thi,vrlo,vrhi,b)):
        raise ValueError("rectangle endpoints and boundary must be finite")
    if tlo>thi or not (0.0<vrlo<=vrhi):
        raise ValueError("invalid theta/return variance intervals")
    d1,d2=direction*(tlo-b),direction*(thi-b)
    dlo,dhi=min(d1,d2),max(d1,d2)
    gap_lo,gap_hi=max(0.0,dlo),max(0.0,dhi)
    I=gap_lo*gap_lo*q/(2.0*vrhi)
    Knec=binary_kl(1.0-beta,alpha)
    Ksuf=predictable_gaussian_eprocess_information_threshold(alpha,beta)
    power_lb=predictable_gaussian_eprocess_power_lower_bound(I,alpha)
    L=math.log(1.0/alpha)
    if gap_hi<=0.0:
        status="NO_ALTERNATIVE_SIDE_IN_SET"
    elif gap_lo<=0.0:
        status="NO_UNIFORM_SEPARATION_FROM_BOUNDARY"
    elif I < Knec*(1.0-1e-12):
        status="UNIFORMLY_IMPOSSIBLE_BY_HARD_WORLD_KL"
    elif I >= Ksuf*(1.0-1e-12):
        status="ROBUST_ANYTIME_EPROCESS_POWER_GUARANTEED"
    else:
        status="NECESSARY_SUFFICIENT_GAP"
    if gap_lo>0.0:
        qnec=2.0*vrhi*Knec/(gap_lo*gap_lo)
        qsuf=2.0*vrhi*Ksuf/(gap_lo*gap_lo)
        lam=math.sqrt(2.0*L/(vrhi*q))
    else:
        qnec=math.inf; qsuf=math.inf; lam=None
    return {
        "status":status,
        "remaining_design_energy":q,
        "coefficient_gap_lower":float(gap_lo),
        "coefficient_gap_upper":float(gap_hi),
        "return_noise_variance_upper":float(vrhi),
        "robust_information_floor_nats":float(I),
        "binary_kl_necessary_threshold_nats":float(Knec),
        "robust_eprocess_sufficient_threshold_nats":float(Ksuf),
        "power_lower_bound":float(power_lb),
        "design_energy_necessary":float(qnec),
        "design_energy_sufficient_eprocess":float(qsuf),
        "horizon_matched_lambda": (float(lam) if lam is not None else None),
        "alpha":alpha,"beta":beta,
        "economic_boundary":b,"direction":int(direction),
        "anytime_valid_after_freeze":True,
        "requires_design_clock_to_reach_declared_budget":True,
        "frozen_set_coverage_error_not_included":True,
        "does_not_require_signal_return_innovation_orthogonality":True,
    }


def g2_correlated_information_economic_identity(
    theta: float,
    boundary: float,
    phi: float,
    return_noise_variance: float,
    signal_innovation_variance: float,
    innovation_correlation: float,
    *,
    gamma: float,
    trading_cost: float,
    horizon: int,
    terminal_penalty: float = 0.0,
) -> dict:
    """Full-joint G2 information/economic identity with correlated innovations.

    Let the contemporaneous return/signal innovations have covariance

        Omega = [[v_R, rho sqrt(v_R v_X)],
                 [rho sqrt(v_R v_X), v_X]],   |rho|<1.

    Between worlds that differ only in the return loading theta versus b, the
    conditional joint mean shift is ((theta-b) X_t, 0).  Consequently the full
    joint Gaussian KL rate is the return-only rate multiplied by 1/(1-rho^2).

    The LQ economic regret is unchanged, so the full-information price per nat
    is

        Pi_full = v_R (1-rho^2) * (1/H) sum_t D_t c_t(phi)^2.

    The orthogonal-innovation formula is recovered at rho=0.  Correlation can
    only add statistical information about the return innovation in this joint
    Gaussian model; the least-informative rho is therefore zero.
    """
    rho=float(innovation_correlation)
    if not math.isfinite(rho) or not -1.0<rho<1.0:
        raise ValueError("innovation_correlation must lie in (-1,1)")
    base=g2_discrete_information_economic_identity(
        theta,boundary,phi,return_noise_variance,signal_innovation_variance,
        gamma=gamma,trading_cost=trading_cost,horizon=horizon,
        terminal_penalty=terminal_penalty,
    )
    I_return=float(base["statistical_information_nats"])
    I_full=I_return/(1.0-rho*rho)
    R=float(base["economic_boundary_regret"])
    Pi_full=float(base["price_per_nat"])*(1.0-rho*rho)
    return {
        **base,
        "return_only_information_nats":I_return,
        "full_joint_information_nats":float(I_full),
        "full_joint_price_per_nat":Pi_full,
        "innovation_correlation":rho,
        "full_joint_identity_residual":float(R-Pi_full*I_full),
        "least_informative_correlation_is_zero":True,
    }


def student_t_information_threshold_asymptotic(alpha: float, beta: float, df: int) -> dict:
    """First-order large-df expansion of the unknown-scale Student-t frontier.

    Let a=z_{1-alpha}, b=z_{1-beta}, delta_0=a+b.  If delta_df is the
    noncentrality required by the exact one-sided t test with ``df`` residual
    degrees of freedom, then

        delta_df = delta_0 * [1 + a^2/(4 df)] + O(df^-2),

    and therefore, on the information-equivalent scale I=delta^2/2,

        I_t(df) = I_z * [1 + a^2/(2 df)] + O(df^-2).

    The expansion is a classical large-df Studentization calculation; its role
    here is to quantify the finite-sample nuisance-scale tax relative to the
    canonical known-scale CAC frontier.
    """
    alpha, beta = float(alpha), float(beta)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0):
        raise ValueError("require alpha,beta in (0,1)")
    if not isinstance(df, int) or isinstance(df, bool) or df < 1:
        raise ValueError("df must be a positive integer")
    a = _N.inv_cdf(1.0-alpha)
    b = _N.inv_cdf(1.0-beta)
    delta0 = a+b
    I0 = 0.5*delta0*delta0
    delta1 = delta0*(1.0 + a*a/(4.0*df))
    I1 = I0*(1.0 + a*a/(2.0*df))
    exact = student_t_reliable_information_threshold(alpha,beta,df)
    return {
        "df": int(df), "alpha": alpha, "beta": beta,
        "known_scale_noncentrality": float(delta0),
        "known_scale_I_crit_nats": float(I0),
        "noncentrality_first_order": float(delta1),
        "I_first_order_nats": float(I1),
        "relative_I_inflation_first_order": float(1.0+a*a/(2.0*df)),
        "exact_noncentrality": float(exact["noncentrality_crit"]),
        "exact_I_equivalent_nats": float(exact["I_equivalent_nats"]),
        "noncentrality_remainder": float(exact["noncentrality_crit"]-delta1),
        "information_remainder_nats": float(exact["I_equivalent_nats"]-I1),
    }


def predictable_gaussian_eprocess_exact_power_lower_bound(information: float, alpha: float) -> float:
    """Gaussian design-clock power bound for the horizon-matched e-process.

    Consider the post-freeze predictable Gaussian return regression used by
    ``g2_postfreeze_eprocess_frontier`` and suppose a predictable clipping rule
    stops betting when the variance/design clock reaches exactly q.  With
    v<=vbar and alternative gap d>0 define I=d^2 q/(2 vbar), and choose

        lambda = sqrt(2 log(1/alpha)/(vbar q)).

    When the exact clipped quadratic clock is reached almost surely, the
    martingale transform has Gaussian law with variance v q. A calendar
    deadline that can leave the clock unfinished requires independent
    Gaussian completion for the proof and subtraction of the probability
    of not reaching q from the resulting unconditional power bound. No
    Gaussian law conditional on the clock being reached is asserted.  Once I>log(1/alpha), the worst lower-tail case is v=vbar,
    which gives the rigorous bound

        P(cross by q) >= Phi(sqrt(2I)-sqrt(2 log(1/alpha))).

    Below I<=log(1/alpha) we return zero rather than use the Gaussian expression
    outside the monotonic worst-variance regime.  The clipping stake is
    predictable and therefore preserves null supermartingale validity.
    """
    I, alpha = float(information), float(alpha)
    if not math.isfinite(I) or I < 0.0:
        raise ValueError("information must be finite and nonnegative")
    if not 0.0 < alpha < 1.0:
        raise ValueError("alpha must lie in (0,1)")
    L = math.log(1.0/alpha)
    if I <= L:
        return 0.0
    return float(_Phi(math.sqrt(2.0*I)-math.sqrt(2.0*L)))


def predictable_gaussian_eprocess_exact_information_threshold(alpha: float, beta: float) -> float:
    """Sufficient information for power 1-beta of the clipped Gaussian e-process.

    For power targets above one half, inversion of

        Phi(sqrt(2I)-sqrt(2 log(1/alpha))) >= 1-beta

    yields

        I >= .5 [sqrt(2 log(1/alpha)) + z_{1-beta}]^2.

    This is strictly sharper than the generic Chernoff sufficient threshold
    ``predictable_gaussian_eprocess_information_threshold`` while using the
    declared conditional-Gaussian structure.  It remains an achievability
    result, not a minimax-optimality claim.
    """
    alpha, beta = float(alpha), float(beta)
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 0.5):
        raise ValueError("require alpha in (0,1) and beta in (0,0.5)")
    L = math.log(1.0/alpha)
    z = _N.inv_cdf(1.0-beta)
    return float(0.5*(math.sqrt(2.0*L)+z)**2)


def g2_conditional_design_energy_quantile_lower(
    current_signal: float,
    max_abs_phi: float,
    signal_variance_lower: float,
    horizon: int,
    eta: float,
    *,
    min_abs_phi: float = 0.0,
) -> dict:
    """Uniform lower quantile for future G2 design energy from the current state.

    Conditional on ``X_0=x``, for ``H`` return observations define

        Q_H = sum_{k=0}^{H-1} X_k^2.

    Two rigorous Gaussian lower-quantile bounds are computed and the sharper is
    retained.

    1. **Finite-H spectral/Anderson bound.**  For ``|phi|<=rho`` and
       ``v_X>=vX-``, the covariance of ``(X_1,...,X_{H-1})`` has minimum
       eigenvalue at least

           c_X = vX-/[1+rho^2+2 rho cos(pi/H)]       (H>=2).

       This follows by dominating the Gram matrix of the bidiagonal inverse-AR
       filter by an explicit Toeplitz tridiagonal matrix.  Anderson's Gaussian
       peakedness inequality then gives a chi-square lower quantile.

    2. **Trace/Frobenius (Laurent--Massart) bound.**  Put ``m=H-1`` and
       ``r=phi^2``.  With unit innovation variance, let ``C_m(r)`` be the
       conditional covariance of ``(X_1,...,X_m)``.  Its trace and squared
       Frobenius norm are

           T_m(r)=sum_i a_i(r),
           S_m(r)=sum_i a_i(r)^2[1+2 sum_{h=1}^{m-i} r^h],
           a_i(r)=sum_{j=0}^{i-1} r^j.

       Both increase in ``r``.  Therefore, for ``r_min=min|phi|^2`` and
       ``r_max=max|phi|^2``, the centered Gaussian quadratic-form lower-tail
       inequality yields

           Q_H >= x^2 + vX- [T_m(r_min)
                    -2 sqrt(S_m(r_max) log(1/eta))]_+

       with probability at least ``1-eta``.  Anderson's inequality makes the
       same bound valid with the nonzero conditional mean induced by ``X_0``.
       Unlike the eigenvalue bound, its leading large-H rate is the exact
       least-compatible expected signal energy rate
       ``vX-/(1-r_min)``; the uncertainty penalty is only order sqrt(H).

    Taking the larger of the two deterministic lower quantiles is valid because
    each individually has coverage at least ``1-eta``.  ``min_abs_phi`` should
    be the minimum absolute persistence allowed by the frozen rectangle; the
    default zero remains valid but can be conservative when that rectangle is
    bounded away from zero.
    """
    x,rho,vx,eta,amin=map(float,(current_signal,max_abs_phi,signal_variance_lower,eta,min_abs_phi))
    if not all(math.isfinite(v) for v in (x,rho,vx,eta,amin)):
        raise ValueError("inputs must be finite")
    if not (0.0<=amin<=rho<1.0):
        raise ValueError("require 0<=min_abs_phi<=max_abs_phi<1")
    if vx<0.0:
        raise ValueError("signal_variance_lower must be nonnegative")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    if not 0.0<eta<1.0:
        raise ValueError("eta must lie in (0,1)")

    if horizon==1:
        spectral_den=1.0; floor=vx; chi=0.0
        q_spectral=x*x; q_trace=x*x
        Tmin=0.0; Smax=0.0
    else:
        from scipy.stats import chi2
        m=horizon-1
        spectral_den=1.0+rho*rho+2.0*rho*math.cos(math.pi/horizon)
        floor=vx/spectral_den
        chi=float(chi2.ppf(eta,m))
        q_spectral=x*x+floor*chi

        def _trace_frob_unit(mm: int, r: float) -> tuple[float,float]:
            # a_i = 1+r+...+r^(i-1), built stably without cancellation near r=1.
            arr=[]; a=0.0
            for _i in range(1,mm+1):
                a=1.0+r*a
                arr.append(a)
            T=float(sum(arr)); S=0.0
            for idx,ai in enumerate(arr, start=1):
                rem=mm-idx
                if rem<=0 or r==0.0:
                    tail=0.0
                else:
                    # finite sum r+...+r^rem, stable for r<1.
                    tail=r*(1.0-r**rem)/(1.0-r)
                S += ai*ai*(1.0+2.0*tail)
            return T,float(S)

        Tmin,_=_trace_frob_unit(m,amin*amin)
        _,Smax=_trace_frob_unit(m,rho*rho)
        lm_core=max(0.0,Tmin-2.0*math.sqrt(max(0.0,Smax)*math.log(1.0/eta)))
        q_trace=x*x+vx*lm_core

    if q_trace>=q_spectral:
        q=q_trace; method="TRACE_FROBENIUS_LAURENT_MASSART"
    else:
        q=q_spectral; method="FINITE_H_SPECTRAL_CHI_SQUARE"
    return {
        "design_energy_quantile_lower":float(q),
        "design_energy_quantile_lower_spectral":float(q_spectral),
        "design_energy_quantile_lower_trace_frobenius":float(q_trace),
        "selected_quantile_method":method,
        "probability_at_least":float(1.0-eta),
        "eta":eta,
        "random_design_df":int(max(0,horizon-1)),
        "conditional_covariance_eigenvalue_floor":float(floor),
        "finite_h_spectral_denominator":float(spectral_den),
        "trace_unit_covariance_lower":float(Tmin),
        "frob_squared_unit_covariance_upper":float(Smax),
        "current_signal_energy":float(x*x),
        "min_abs_phi":amin,
        "max_abs_phi":rho,
        "signal_variance_lower":vx,
        "horizon":int(horizon),
    }


def g2_anytime_survival_power_lower_bound(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 161,
) -> dict:
    """Uniform post-freeze anytime power lower bound in the full predictable G2 model.

    This closes the main gap between the fixed-design Student-t benchmark and
    the actual G2 setting.  The frozen rectangle supplies

      d_-   : minimum coefficient gap to the economic null,
      vR+   : upper return-noise variance,
      vX-   : lower signal-innovation variance,
      rho+  : maximum |phi|.

    For each eta we obtain a design-energy level q_eta reached by economic death
    with probability at least 1-eta.  A predictable clipped betting strategy
    uses exactly q_eta units of variance clock if that level is reached.  Its
    level-alpha e-process is valid for every predictable design and does not
    require orthogonality between return and signal innovations.  Gaussianity
    sharpens its crossing-power bound to

        Phi(sqrt(2 I_eta)-sqrt(2 log(1/alpha))) - eta,

    where I_eta=d_-^2 q_eta/(2 vR+).  Maximizing over a deterministic eta grid
    preserves rigor because every grid point separately supplies a valid bound.

    The rectangle must be frozen before this future stream.  Its own coverage
    error is not included here; use ``g2_end_to_end_survival_frontier`` to
    compose it with the future error budgets.
    """
    if direction not in (-1,1):
        raise ValueError("direction must be +1 or -1")
    alpha=float(alpha)
    if not 0.0<alpha<1.0:
        raise ValueError("alpha must lie in (0,1)")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(eta_grid_size,int) or eta_grid_size<21:
        raise ValueError("eta_grid_size must be an integer at least 21")
    try:
        tlo,thi=map(float,projection_rectangle["theta"])
        plo,phi=map(float,projection_rectangle["phi"])
        vrlo,vrhi=map(float,projection_rectangle["return_noise_variance"])
        vxlo,vxhi=map(float,projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing required G2 intervals") from exc
    b=float(economic_boundary); x=float(current_signal)
    if not all(math.isfinite(v) for v in (tlo,thi,plo,phi,vrlo,vrhi,vxlo,vxhi,b,x)):
        raise ValueError("rectangle endpoints and scalar inputs must be finite")
    if not (tlo<=thi and -1.0<plo<=phi<1.0 and 0.0<vrlo<=vrhi and 0.0<=vxlo<=vxhi):
        raise ValueError("invalid G2 rectangle")
    d1,d2=direction*(tlo-b),direction*(thi-b)
    dlo,dhi=min(d1,d2),max(d1,d2)
    gap_lo,gap_hi=max(0.0,dlo),max(0.0,dhi)
    rho=max(abs(plo),abs(phi))
    L=math.log(1.0/alpha)

    if gap_hi<=0.0:
        return {
            "power_lower_bound":0.0,"status":"NO_ALTERNATIVE_SIDE_IN_SET",
            "coefficient_gap_lower":float(gap_lo),"coefficient_gap_upper":float(gap_hi),
            "alpha":alpha,"horizon":int(horizon),"economic_boundary":b,"direction":int(direction),
        }
    if gap_lo<=0.0 or vxlo<=0.0:
        return {
            "power_lower_bound":0.0,"status":"NO_UNIFORM_GAP_OR_DESIGN_VARIANCE_FLOOR",
            "coefficient_gap_lower":float(gap_lo),"coefficient_gap_upper":float(gap_hi),
            "signal_variance_lower":float(vxlo),"alpha":alpha,"horizon":int(horizon),
            "economic_boundary":b,"direction":int(direction),
        }

    # Search eta on a logit grid.  Endpoints are kept away from 0/1 to avoid
    # degenerate chi-square quantiles; every evaluated eta is individually safe.
    zs=np.linspace(-12.0,8.0,eta_grid_size)
    etas=1.0/(1.0+np.exp(-zs))
    best=None
    for eta in etas:
        qout=g2_conditional_design_energy_quantile_lower(x,rho,vxlo,horizon,float(eta),min_abs_phi=_min_abs_on_interval(plo,phi))
        q=float(qout["design_energy_quantile_lower"])
        I=gap_lo*gap_lo*q/(2.0*vrhi)
        if I<=L:
            p_at_clock=0.0
        else:
            p_at_clock=float(_Phi(math.sqrt(2.0*I)-math.sqrt(2.0*L)))
        bound=max(0.0,p_at_clock-float(eta))
        cand=(bound,float(eta),q,I,p_at_clock,qout)
        if best is None or cand[0]>best[0]:
            best=cand
    bound,eta_star,q_star,I_star,p_clock,qout=best
    lam=math.sqrt(2.0*L/(vrhi*q_star)) if q_star>0.0 else math.inf
    return {
        "power_lower_bound":float(bound),
        "status":"POWER_BOUND_COMPUTED",
        "eta":float(eta_star),
        "design_energy_clock":float(q_star),
        "robust_information_at_clock_nats":float(I_star),
        "power_at_exact_clock_lower_bound":float(p_clock),
        "coefficient_gap_lower":float(gap_lo),
        "coefficient_gap_upper":float(gap_hi),
        "return_noise_variance_upper":float(vrhi),
        "signal_variance_lower":float(vxlo),
        "max_abs_phi":float(rho),
        "conditional_covariance_eigenvalue_floor":float(qout["conditional_covariance_eigenvalue_floor"]),
        "horizon_matched_lambda":float(lam),
        "alpha":alpha,"horizon":int(horizon),
        "economic_boundary":b,"direction":int(direction),
        "anytime_valid_after_freeze":True,
        "requires_return_signal_orthogonality":False,
        "uses_conditional_gaussian_return_and_signal_regressions":True,
        "frozen_set_coverage_error_not_included":True,
    }


def minimum_horizon_for_g2_anytime_power(
    projection_rectangle: dict,
    current_signal: float,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    max_horizon: int = 5000,
    eta_grid_size: int = 161,
) -> dict:
    """First H whose full-G2 anytime lower bound reaches power 1-beta."""
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<1.0 and 0.0<beta<1.0):
        raise ValueError("alpha,beta must lie in (0,1)")
    if not isinstance(max_horizon,int) or isinstance(max_horizon,bool) or max_horizon<1:
        raise ValueError("max_horizon must be a positive integer")
    target=1.0-beta
    last=None
    # Monotonicity of the optimized analytical lower bound is expected but a
    # linear scan avoids assuming it in the public guarantee.
    for H in range(1,max_horizon+1):
        out=g2_anytime_survival_power_lower_bound(
            projection_rectangle,current_signal,H,alpha,
            economic_boundary=economic_boundary,direction=direction,
            eta_grid_size=eta_grid_size,
        )
        last=out
        if out["status"]!="POWER_BOUND_COMPUTED":
            return {**out,"minimum_horizon":None,"reason":out["status"],"target_power":target}
        if out["power_lower_bound"]>=target*(1.0-1e-12):
            return {**out,"minimum_horizon":int(H),"reason":"cleared","target_power":target}
    return {**(last or {}),"minimum_horizon":None,"reason":"not_cleared_within_max_horizon","target_power":target}


def g2_end_to_end_survival_frontier(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha_total: float,
    beta_total: float,
    frozen_set_failure: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 161,
) -> dict:
    """Compose frozen-G2 set error with the future anytime deployment test.

    If the frozen rectangle misses the true parameter with probability at most
    delta, and the post-freeze e-process uses level alpha_f, then total false
    deployment is at most delta+alpha_f.  Likewise if future conditional power
    on the coverage event is at least 1-beta_f, unconditional power is at least
    1-delta-beta_f.

    We therefore set

        alpha_f = alpha_total-delta,
        beta_f  = beta_total-delta,

    and test whether the rigorous future power lower bound reaches 1-beta_f.
    This produces an end-to-end guarantee without pretending that the estimated
    rectangle is known with certainty.
    """
    alpha_total,beta_total,delta=map(float,(alpha_total,beta_total,frozen_set_failure))
    if not (0.0<alpha_total<1.0 and 0.0<beta_total<1.0 and 0.0<=delta<min(alpha_total,beta_total)):
        raise ValueError("require 0<=frozen_set_failure<min(alpha_total,beta_total)<1")
    af=alpha_total-delta
    bf=beta_total-delta
    future=g2_anytime_survival_power_lower_bound(
        projection_rectangle,current_signal,horizon,af,
        economic_boundary=economic_boundary,direction=direction,
        eta_grid_size=eta_grid_size,
    )
    total_power_lb=max(0.0,float(future.get("power_lower_bound",0.0))-delta)
    target=1.0-beta_total
    if future.get("status")!="POWER_BOUND_COMPUTED":
        status=future.get("status","UNRESOLVED")
    elif total_power_lb>=target*(1.0-1e-12):
        status="END_TO_END_RELIABLE_CERTIFICATION_GUARANTEED"
    else:
        status="END_TO_END_POWER_NOT_YET_GUARANTEED"
    return {
        "status":status,
        "alpha_total":alpha_total,"beta_total":beta_total,
        "frozen_set_failure":delta,
        "future_alpha_budget":af,"future_beta_budget":bf,
        "total_false_deploy_upper_bound":float(delta+af),
        "total_power_lower_bound":float(total_power_lb),
        "target_power":float(target),
        "future":future,
        "horizon":int(horizon),
        "economic_boundary":float(economic_boundary),
        "direction":int(direction),
    }


def g2_positive_rectangle_economic_loading_boundary(
    projection_rectangle: dict,
    hurdle: float,
    *,
    gamma: float,
    trading_cost: float,
    economic_horizon: int,
    terminal_penalty: float = 0.0,
) -> dict:
    """Control-defined loading boundary for the positive-persistence G2 rectangle.

    In the finite-horizon LQ model the oracle value is

        V*(theta,phi,vX)=K(phi,vX) theta^2,
        K = vX/[2(1-phi^2)] sum_H D_H c_H(phi)^2.

    On a rectangle with 0<=phi_-<=phi_+<1 and vX>=vX_->0, K is minimized at
    (phi_-,vX_-).  Therefore the largest loading boundary compatible with the
    declared economic hurdle h is

        b_econ = sqrt(h/K_min).

    Testing theta>b_econ is sufficient to put every compatible nuisance world
    above the oracle-value hurdle.  This is the statistical/economic bridge used
    by the control-coupled survival frontier.
    """
    hurdle,gamma,trading_cost,terminal_penalty=map(float,(hurdle,gamma,trading_cost,terminal_penalty))
    if hurdle<0.0 or gamma<=0.0 or trading_cost<0.0 or terminal_penalty<0.0:
        raise ValueError("invalid economic parameters")
    if not isinstance(economic_horizon,int) or isinstance(economic_horizon,bool) or economic_horizon<1:
        raise ValueError("economic_horizon must be a positive integer")
    try:
        plo,phi=map(float,projection_rectangle["phi"])
        vxlo,vxhi=map(float,projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing phi/signal variance intervals") from exc
    if not (0.0<=plo<=phi<1.0 and 0.0<vxlo<=vxhi):
        raise ValueError("requires 0<=phi_lower<=phi_upper<1 and positive signal-variance floor")
    K=terminal_penalty; L=0.0; curvature_sum=0.0
    D=[0.0]*economic_horizon; c=[0.0]*economic_horizon
    for t in range(economic_horizon-1,-1,-1):
        Dt=gamma+trading_cost+K
        ct=(1.0+plo*L)/Dt
        D[t]=Dt; c[t]=ct
        K=trading_cost*((gamma+K)/Dt)
        L=trading_cost*ct
    curvature_sum=sum(Dt*ct*ct for Dt,ct in zip(D,c))
    multiplier=vxlo*curvature_sum/(2.0*(1.0-plo*plo))
    boundary=0.0 if hurdle==0.0 else math.sqrt(hurdle/multiplier)
    return {
        "economic_loading_boundary":float(boundary),
        "least_favourable_value_multiplier":float(multiplier),
        "least_favourable_phi":float(plo),
        "least_favourable_signal_variance":float(vxlo),
        "hurdle":float(hurdle),
        "economic_horizon":int(economic_horizon),
        "curvature_sum":float(curvature_sum),
        "scope":"positive-persistence rectangular G2 outer set",
    }


def g2_control_coupled_end_to_end_frontier(
    projection_rectangle: dict,
    current_signal: float,
    future_horizon: int,
    alpha_total: float,
    beta_total: float,
    frozen_set_failure: float,
    hurdle: float,
    *,
    gamma: float,
    trading_cost: float,
    economic_horizon: int,
    terminal_penalty: float = 0.0,
    eta_grid_size: int = 161,
) -> dict:
    """End-to-end G2 Survival Frontier with a control-defined economic null."""
    econ=g2_positive_rectangle_economic_loading_boundary(
        projection_rectangle,hurdle,gamma=gamma,trading_cost=trading_cost,
        economic_horizon=economic_horizon,terminal_penalty=terminal_penalty,
    )
    b=float(econ["economic_loading_boundary"])
    stat=g2_end_to_end_survival_frontier(
        projection_rectangle,current_signal,future_horizon,
        alpha_total,beta_total,frozen_set_failure,
        economic_boundary=b,direction=1,eta_grid_size=eta_grid_size,
    )
    return {**stat,"economic":econ,"control_defined_boundary":b}

def composite_unknown_scale_pinching(alpha: float, beta: float, df: int) -> dict:
    """Finite-df pinching of the robust unknown-scale information frontier.

    Consider a one-sided Gaussian regression test at a fixed design clock with
    unknown constant scale.  Let ``I_*`` denote the (generally unknown) minimax
    information threshold for uniform level ``alpha`` and power ``1-beta`` over
    a nuisance-scale family that contains the hard pair with the same variance.

    Neyman--Pearson applied to that hard *known-scale* pair gives the necessary
    threshold

        I_z = .5 (z_{1-alpha}+z_{1-beta})^2.

    The ordinary one-sided Student t test is a concrete scale-free procedure,
    giving the sufficient threshold ``I_t(df)``.  Consequently

        I_z <= I_* <= I_t(df).

    For fixed alpha,beta and df -> infinity,

        I_t/I_z = 1 + z_{1-alpha}^2/(2 df) + O(df^-2).

    This function reports the exact endpoints and the first-order pinching
    approximation.  It does not claim that the t test is minimax among all
    noninvariant tests at finite df.
    """
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    if not isinstance(df,int) or isinstance(df,bool) or df<1:
        raise ValueError("df must be a positive integer")
    iz=float(reliable_information_threshold(alpha,beta)["I_crit_nats"])
    it=student_t_reliable_information_threshold(alpha,beta,df)
    isu=float(it["I_equivalent_nats"])
    a=float(_N.inv_cdf(1.0-alpha))
    first_rel=1.0+a*a/(2.0*df)
    return {
        "known_scale_np_necessary_nats":iz,
        "student_t_sufficient_nats":isu,
        "minimax_threshold_lower_nats":iz,
        "minimax_threshold_upper_nats":isu,
        "exact_pinching_width_nats":float(isu-iz),
        "exact_relative_pinching_ratio":float(isu/iz),
        "first_order_relative_upper_ratio":float(first_rel),
        "first_order_width_nats":float(iz*(first_rel-1.0)),
        "alpha":alpha,"beta":beta,"df":int(df),
        "finite_df_exact_minimax_not_claimed":True,
    }


def predictable_gaussian_clock_power_lower_bound(information: float, alpha: float) -> float:
    """Power bound for a predeclared *information-clock* Gaussian test.

    In the predictable Gaussian regression

        Y_{t+1}=theta X_t+eps_{t+1},   eps|F_t ~ N(0,v),

    suppose a frozen formation set guarantees ``v<=vbar`` on its coverage
    event.  Use predictable nonnegative clipping weights c_t<=1 and stop when

        sum c_t^2 X_t^2 = q

    exactly.  At that clock, the noise martingale transform is exactly
    N(0,v q), while under theta>=b+d its drift is at least d q because
    c_t X_t^2 >= c_t^2 X_t^2.  Testing

        M_q >= z_{1-alpha} sqrt(vbar q)

    is therefore level alpha for every theta<=b and v<=vbar.  Writing
    I=d^2 q/(2 vbar), once the alternative mean lies above the critical
    threshold the worst scale is v=vbar and

        Power >= Phi(sqrt(2I)-z_{1-alpha}).

    Below that monotonicity region we return zero.  Unlike an e-process fixed
    boundary, this is a single predeclared test triggered by *information time*;
    it is not valid for arbitrary outcome-dependent stopping rules.
    """
    I,alpha=float(information),float(alpha)
    if not math.isfinite(I) or I<0.0:
        raise ValueError("information must be finite and nonnegative")
    if not 0.0<alpha<0.5:
        raise ValueError("alpha must lie in (0,0.5)")
    za=float(_N.inv_cdf(1.0-alpha))
    if math.sqrt(2.0*I) <= za:
        return 0.0
    return float(_Phi(math.sqrt(2.0*I)-za))


def predictable_gaussian_clock_information_threshold(alpha: float, beta: float) -> float:
    """Canonical-sharp sufficient information for the clock-triggered test.

    For alpha,beta<1/2 the preceding power lower bound reaches 1-beta exactly
    at the canonical Gaussian threshold

        I_z = .5 (z_{1-alpha}+z_{1-beta})^2.

    Thus a rigorous upper variance envelope recovers the known-scale CAC
    information threshold despite the underlying variance being unknown.
    """
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<0.5 and 0.0<beta<0.5):
        raise ValueError("require alpha,beta in (0,0.5)")
    return float(reliable_information_threshold(alpha,beta)["I_crit_nats"])


def g2_information_clock_survival_power_lower_bound(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 161,
) -> dict:
    """Calendar-horizon power bound for the post-freeze G2 clock test.

    The formation rectangle is frozen before the future stream.  On its
    coverage event it supplies a uniform coefficient gap d_-, an upper return
    variance vR+, a lower signal innovation variance vX-, and a stability bound
    rho=max|phi|.  For each deterministic eta, a lower quantile q_eta of future
    design energy is reached by ``horizon`` with probability at least 1-eta.

    At the predeclared clipped information clock q_eta, the variance-envelope
    Gaussian test has power at least

        Phi(sqrt(2 I_eta)-z_{1-alpha}),
        I_eta=d_-^2 q_eta/(2 vR+).

    Hence deployment by the calendar horizon has power at least that quantity
    minus eta.  Maximizing over a deterministic eta grid remains rigorous.

    This test is sequential in calendar time but only tests at its predeclared
    design clock.  It should not be described as an arbitrary-stopping
    e-process.  Its advantage is that it recovers the canonical Gaussian
    information threshold at the clock, rather than paying a fixed-boundary
    anytime-monitoring penalty.
    """
    if direction not in (-1,1):
        raise ValueError("direction must be +1 or -1")
    alpha=float(alpha)
    if not 0.0<alpha<0.5:
        raise ValueError("alpha must lie in (0,0.5)")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    if not isinstance(eta_grid_size,int) or eta_grid_size<21:
        raise ValueError("eta_grid_size must be an integer at least 21")
    try:
        tlo,thi=map(float,projection_rectangle["theta"])
        plo,phi=map(float,projection_rectangle["phi"])
        vrlo,vrhi=map(float,projection_rectangle["return_noise_variance"])
        vxlo,vxhi=map(float,projection_rectangle["signal_innovation_variance"])
    except Exception as exc:
        raise ValueError("projection_rectangle is missing required G2 intervals") from exc
    b=float(economic_boundary); x=float(current_signal)
    if not all(math.isfinite(v) for v in (tlo,thi,plo,phi,vrlo,vrhi,vxlo,vxhi,b,x)):
        raise ValueError("rectangle endpoints and scalar inputs must be finite")
    if not (tlo<=thi and -1.0<plo<=phi<1.0 and 0.0<vrlo<=vrhi and 0.0<=vxlo<=vxhi):
        raise ValueError("invalid G2 rectangle")
    d1,d2=direction*(tlo-b),direction*(thi-b)
    dlo,dhi=min(d1,d2),max(d1,d2)
    gap_lo,gap_hi=max(0.0,dlo),max(0.0,dhi)
    rho=max(abs(plo),abs(phi))
    if gap_hi<=0.0:
        return {"power_lower_bound":0.0,"status":"NO_ALTERNATIVE_SIDE_IN_SET",
                "coefficient_gap_lower":float(gap_lo),"coefficient_gap_upper":float(gap_hi),
                "alpha":alpha,"horizon":int(horizon),"economic_boundary":b,"direction":int(direction)}
    if gap_lo<=0.0 or vxlo<=0.0:
        return {"power_lower_bound":0.0,"status":"NO_UNIFORM_GAP_OR_DESIGN_VARIANCE_FLOOR",
                "coefficient_gap_lower":float(gap_lo),"coefficient_gap_upper":float(gap_hi),
                "signal_variance_lower":float(vxlo),"alpha":alpha,"horizon":int(horizon),
                "economic_boundary":b,"direction":int(direction)}
    za=float(_N.inv_cdf(1.0-alpha))
    zs=np.linspace(-12.0,8.0,eta_grid_size)
    etas=1.0/(1.0+np.exp(-zs))
    best=None
    for eta in etas:
        qout=g2_conditional_design_energy_quantile_lower(x,rho,vxlo,horizon,float(eta),min_abs_phi=_min_abs_on_interval(plo,phi))
        q=float(qout["design_energy_quantile_lower"])
        I=gap_lo*gap_lo*q/(2.0*vrhi)
        if math.sqrt(2.0*I)<=za:
            pclock=0.0
        else:
            pclock=float(_Phi(math.sqrt(2.0*I)-za))
        bd=max(0.0,pclock-float(eta))
        cand=(bd,float(eta),q,I,pclock,qout)
        if best is None or cand[0]>best[0]:
            best=cand
    bound,eta_star,q_star,I_star,p_clock,qout=best
    return {
        "power_lower_bound":float(bound),
        "status":"CLOCK_TRIGGERED_POWER_BOUND_COMPUTED",
        "eta":float(eta_star),
        "design_energy_clock":float(q_star),
        "robust_information_at_clock_nats":float(I_star),
        "power_at_exact_clock_lower_bound":float(p_clock),
        "coefficient_gap_lower":float(gap_lo),
        "coefficient_gap_upper":float(gap_hi),
        "return_noise_variance_upper":float(vrhi),
        "signal_variance_lower":float(vxlo),
        "max_abs_phi":float(rho),
        "conditional_covariance_eigenvalue_floor":float(qout["conditional_covariance_eigenvalue_floor"]),
        "alpha":alpha,"horizon":int(horizon),
        "economic_boundary":b,"direction":int(direction),
        "predeclared_information_clock":True,
        "arbitrary_outcome_dependent_stopping_allowed":False,
        "requires_return_signal_orthogonality":False,
        "uses_conditional_gaussian_return_and_signal_regressions":True,
        "frozen_set_coverage_error_not_included":True,
    }


def minimum_horizon_for_g2_clock_power(
    projection_rectangle: dict,
    current_signal: float,
    alpha: float,
    beta: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    max_horizon: int = 5000,
    eta_grid_size: int = 161,
) -> dict:
    """First calendar H for which the rigorous G2 clock bound reaches 1-beta."""
    alpha,beta=float(alpha),float(beta)
    if not (0.0<alpha<0.5 and 0.0<beta<0.5):
        raise ValueError("require alpha,beta in (0,0.5)")
    if not isinstance(max_horizon,int) or isinstance(max_horizon,bool) or max_horizon<1:
        raise ValueError("max_horizon must be a positive integer")
    target=1.0-beta
    last=None
    for H in range(1,max_horizon+1):
        out=g2_information_clock_survival_power_lower_bound(
            projection_rectangle,current_signal,H,alpha,
            economic_boundary=economic_boundary,direction=direction,
            eta_grid_size=eta_grid_size,
        )
        last=out
        if out["status"]!="CLOCK_TRIGGERED_POWER_BOUND_COMPUTED":
            return {**out,"minimum_horizon":None,"reason":out["status"],"target_power":target}
        if out["power_lower_bound"]>=target*(1.0-1e-12):
            return {**out,"minimum_horizon":int(H),"reason":"cleared","target_power":target}
    return {**(last or {}),"minimum_horizon":None,"reason":"not_cleared_within_max_horizon","target_power":target}


def g2_end_to_end_clock_frontier(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha_total: float,
    beta_total: float,
    frozen_set_failure: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 161,
) -> dict:
    """End-to-end formation+future guarantee for the G2 clock-triggered test."""
    alpha_total,beta_total,delta=map(float,(alpha_total,beta_total,frozen_set_failure))
    if not (0.0<alpha_total<0.5 and 0.0<beta_total<0.5 and 0.0<=delta<min(alpha_total,beta_total)):
        raise ValueError("require 0<=frozen_set_failure<min(alpha_total,beta_total)<0.5")
    af=alpha_total-delta
    bf=beta_total-delta
    future=g2_information_clock_survival_power_lower_bound(
        projection_rectangle,current_signal,horizon,af,
        economic_boundary=economic_boundary,direction=direction,
        eta_grid_size=eta_grid_size,
    )
    total_power=max(0.0,float(future.get("power_lower_bound",0.0))-delta)
    target=1.0-beta_total
    if future.get("status")!="CLOCK_TRIGGERED_POWER_BOUND_COMPUTED":
        status=future.get("status","UNRESOLVED")
    elif total_power>=target*(1.0-1e-12):
        status="END_TO_END_CLOCK_CERTIFICATION_GUARANTEED"
    else:
        status="END_TO_END_CLOCK_POWER_NOT_YET_GUARANTEED"
    return {
        "status":status,
        "alpha_total":alpha_total,"beta_total":beta_total,
        "frozen_set_failure":delta,
        "future_alpha_budget":af,"future_beta_budget":bf,
        "total_false_deploy_upper_bound":float(delta+af),
        "total_power_lower_bound":float(total_power),
        "target_power":float(target),
        "future":future,"horizon":int(horizon),
        "economic_boundary":float(economic_boundary),"direction":int(direction),
    }


def g2_control_coupled_end_to_end_clock_frontier(
    projection_rectangle: dict,
    current_signal: float,
    future_horizon: int,
    alpha_total: float,
    beta_total: float,
    frozen_set_failure: float,
    hurdle: float,
    *,
    gamma: float,
    trading_cost: float,
    economic_horizon: int,
    terminal_penalty: float = 0.0,
    eta_grid_size: int = 161,
) -> dict:
    """Control-defined economic null plus end-to-end clock certification."""
    econ=g2_positive_rectangle_economic_loading_boundary(
        projection_rectangle,hurdle,gamma=gamma,trading_cost=trading_cost,
        economic_horizon=economic_horizon,terminal_penalty=terminal_penalty,
    )
    b=float(econ["economic_loading_boundary"])
    stat=g2_end_to_end_clock_frontier(
        projection_rectangle,current_signal,future_horizon,
        alpha_total,beta_total,frozen_set_failure,
        economic_boundary=b,direction=1,eta_grid_size=eta_grid_size,
    )
    return {**stat,"economic":econ,"control_defined_boundary":b}


def two_stage_error_composition(
    alpha_future: float,
    beta_future: float,
    null_stage1_failure: float,
    alt_stage1_failure: float,
    *,
    gate_implies_alternative_on_good_event: bool = False,
) -> dict:
    """Rigorous two-stage error composition.

    Let G0/G1 denote stage-1 calibration/resolution events under a null/alternative
    world.  Assume

        P0(G0^c) <= delta0,   P1(G1^c) <= delta1.

    Conditional on the corresponding good event and the frozen formation history,
    suppose the future test has type-I error at most ``alpha_future`` and type-II
    error at most ``beta_future``.  Then

        type-I <= delta0 + (1-delta0) alpha_future,
        power  >= (1-delta1) (1-beta_future).

    If final deployment additionally requires a stage-1 gate which, whenever the
    stage-1 confidence statement is correct, is impossible under the null (for
    example a confidence rectangle lying wholly above the economic boundary),
    then false deployment is bounded by ``delta0`` alone.  This sharper gated
    bound does not by itself provide power: alternative-stage resolution still
    needs its own probability guarantee.
    """
    af,bf,d0,d1=map(float,(alpha_future,beta_future,null_stage1_failure,alt_stage1_failure))
    if not all(0.0 <= x < 1.0 for x in (af,bf,d0,d1)):
        raise ValueError("all error probabilities must lie in [0,1)")
    generic_type1=d0+(1.0-d0)*af
    gated_type1=d0 if gate_implies_alternative_on_good_event else generic_type1
    power=(1.0-d1)*(1.0-bf)
    return {
        "generic_false_deploy_upper_bound":float(generic_type1),
        "false_deploy_upper_bound":float(gated_type1),
        "power_lower_bound":float(power),
        "type2_upper_bound":float(1.0-power),
        "alpha_future":af,"beta_future":bf,
        "null_stage1_failure":d0,"alt_stage1_failure":d1,
        "gate_implies_alternative_on_good_event":bool(gate_implies_alternative_on_good_event),
    }


def future_error_budgets_for_two_stage_targets(
    alpha_total: float,
    beta_total: float,
    null_stage1_failure: float,
    alt_stage1_failure: float,
) -> dict:
    """Largest future-stage errors compatible with total two-stage targets.

    This is the multiplicatively sharp algebraic inversion of

        delta0 + (1-delta0) alpha_f <= alpha_total,
        (1-delta1)(1-beta_f) >= 1-beta_total.

    It is meaningful only after ``delta1`` bounds the probability that stage 1
    is *bad or unresolved* under the alternative.  A confidence-set coverage
    error alone is generally not such a power-resolution bound.
    """
    a,b,d0,d1=map(float,(alpha_total,beta_total,null_stage1_failure,alt_stage1_failure))
    if not (0.0<a<1.0 and 0.0<b<1.0 and 0.0<=d0<a and 0.0<=d1<b):
        raise ValueError("require d0<alpha_total and d1<beta_total, all in [0,1)")
    af=(a-d0)/(1.0-d0)
    bf=(b-d1)/(1.0-d1)
    return {
        "alpha_future_max":float(af),
        "beta_future_max":float(bf),
        "target_false_deploy":a,"target_type2":b,
        "null_stage1_failure":d0,"alt_stage1_failure":d1,
    }


def g2_realized_rectangle_future_power_certificate(
    projection_rectangle: dict,
    current_signal: float,
    horizon: int,
    alpha_future: float,
    formation_coverage_failure: float,
    *,
    economic_boundary: float = 0.0,
    direction: int = 1,
    eta_grid_size: int = 161,
) -> dict:
    """A valid *reported-bound* interpretation for a realized G2 rectangle.

    The random formation rectangle is not treated as though its nominal coverage
    were a posterior probability after it is observed.  Instead the statement is
    repeated-sampling valid:

      with probability at least 1-delta over the formation experiment, the
      reported lower bound is a valid lower bound on the conditional future
      deployment probability of every true world contained in the rectangle.

    This is deliberately weaker than an unconditional end-to-end power theorem.
    Such a theorem additionally needs a bound on the probability that formation
    is unresolved under the alternative; coverage alone does not supply it.
    """
    delta=float(formation_coverage_failure)
    if not 0.0<=delta<1.0:
        raise ValueError("formation_coverage_failure must lie in [0,1)")
    future=g2_information_clock_survival_power_lower_bound(
        projection_rectangle,current_signal,horizon,alpha_future,
        economic_boundary=economic_boundary,direction=direction,
        eta_grid_size=eta_grid_size,
    )
    return {
        "status":"HIGH_CONFIDENCE_FUTURE_POWER_BOUND",
        "formation_confidence":float(1.0-delta),
        "formation_coverage_failure":delta,
        "reported_future_power_lower_bound":float(future.get("power_lower_bound",0.0)),
        "future_alpha":float(alpha_future),
        "future":future,
        "unconditional_power_claimed":False,
        "note":"Coverage validates the reported conditional-future power bound with high probability over formation; it is not itself a lower bound on formation resolution probability.",
    }


# ---------------------------------------------------------------------------
# Direct compact-G2 certification-time frontier
# ---------------------------------------------------------------------------

def g2_structural_clock_power_lower_bound(
    current_signal: float,
    horizon: int,
    alpha: float,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds: tuple[float, float],
    signal_variance_lower: float,
    *,
    eta_grid_size: int = 161,
) -> dict:
    """Uniform power lower bound for a direct G2 economic-null certificate.

    This is the structural-envelope counterpart of
    :func:`g2_information_clock_survival_power_lower_bound`.  No previously
    estimated confidence rectangle is needed.  The parameter class is

        theta <= b                    under the null,
        theta >= b + d                under the alternative,
        v_R <= V,
        phi in [phi_lo, phi_hi] subset (-1,1),
        v_X >= vX_min > 0.

    The economic boundary ``b`` cancels from the score after centering, so the
    bound depends only on the declared positive gap ``d``.  At a predictable
    clipped design clock q, the Gaussian martingale transform has deterministic
    quadratic variation v_R q.  Hence the one-sided clock test has power at
    least Phi(d sqrt(q/V)-z_{1-alpha}).  A uniform lower quantile for the AR(1)
    design energy converts this information-time guarantee into calendar time.

    The result remains valid when contemporaneous return/signal innovations are
    correlated, because the score coefficient at time t is F_t-measurable and
    the return innovation is conditionally Gaussian given F_t.  What is needed
    is the stated conditional return-variance envelope, not exogeneity of the
    full future design.
    """
    x=float(current_signal); alpha=float(alpha); d=float(coefficient_gap)
    V=float(return_variance_upper); vx=float(signal_variance_lower)
    if not math.isfinite(x) or not math.isfinite(d) or not math.isfinite(V) or not math.isfinite(vx):
        raise ValueError("scalar inputs must be finite")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    if not 0.0<alpha<0.5:
        raise ValueError("alpha must lie in (0,0.5)")
    if d<=0.0 or V<=0.0 or vx<=0.0:
        raise ValueError("require coefficient_gap>0, return_variance_upper>0, signal_variance_lower>0")
    if not isinstance(eta_grid_size,int) or eta_grid_size<21:
        raise ValueError("eta_grid_size must be an integer at least 21")
    plo,phi=map(float,phi_bounds)
    if not (math.isfinite(plo) and math.isfinite(phi) and -1.0<plo<=phi<1.0):
        raise ValueError("phi_bounds must lie inside (-1,1)")
    rho=max(abs(plo),abs(phi)); amin=_min_abs_on_interval(plo,phi)
    za=float(_N.inv_cdf(1.0-alpha))
    zs=np.linspace(-12.0,8.0,eta_grid_size)
    etas=1.0/(1.0+np.exp(-zs))
    best=None
    for eta in etas:
        qout=g2_conditional_design_energy_quantile_lower(
            x,rho,vx,horizon,float(eta),min_abs_phi=amin
        )
        q=float(qout["design_energy_quantile_lower"])
        I=d*d*q/(2.0*V)
        pclock=predictable_gaussian_clock_power_lower_bound(I,alpha)
        bound=max(0.0,pclock-float(eta))
        cand=(bound,float(eta),q,I,pclock,qout)
        if best is None or cand[0]>best[0]:
            best=cand
    bound,eta_star,q_star,I_star,pclock,qout=best
    return {
        "power_lower_bound":float(bound),
        "status":"STRUCTURAL_G2_CLOCK_POWER_BOUND_COMPUTED",
        "eta":float(eta_star),
        "design_energy_clock":float(q_star),
        "robust_information_at_clock_nats":float(I_star),
        "power_at_exact_clock_lower_bound":float(pclock),
        "coefficient_gap":d,
        "return_noise_variance_upper":V,
        "signal_variance_lower":vx,
        "phi_lower":plo,"phi_upper":phi,
        "min_abs_phi":float(amin),"max_abs_phi":float(rho),
        "alpha":alpha,"horizon":int(horizon),
        "current_signal":x,
        "requires_frozen_confidence_rectangle":False,
        "requires_return_signal_orthogonality":False,
        "predeclared_information_clock":True,
        "arbitrary_outcome_dependent_stopping_allowed":False,
        "interpretation":"Uniform direct certificate over a compact structural G2 nuisance class.",
    }


def g2_structural_hard_pair_kl(
    current_signal: float,
    horizon: int,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds: tuple[float,float],
    signal_variance_lower: float,
) -> dict:
    """Expected KL of the least-informative orthogonal hard pair in the class.

    The class contains the boundary world theta=b and the alternative
    theta=b+d with the same nuisance parameters.  Choosing the largest allowed
    return variance, the smallest allowed signal innovation variance and the
    persistence with smallest absolute value minimizes E sum X_t^2 over the
    rectangular structural class.  With zero contemporaneous innovation
    correlation this gives a hard pair with

        KL_H = d^2 E Q_H /(2 V).

    Therefore any level-alpha, power-(1-beta) procedure based on the full path
    is impossible whenever this quantity is below kl(1-beta,alpha), provided
    the declared nuisance class contains this orthogonal hard pair.  If a model
    fixes a nonzero innovation correlation away from zero, the corresponding
    full-joint KL should be used instead.
    """
    x=float(current_signal); d=float(coefficient_gap); V=float(return_variance_upper); vx=float(signal_variance_lower)
    if not all(math.isfinite(v) for v in (x,d,V,vx)):
        raise ValueError("scalar inputs must be finite")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    if d<=0.0 or V<=0.0 or vx<0.0:
        raise ValueError("require d>0, V>0 and vX_min>=0")
    plo,phi=map(float,phi_bounds)
    if not (-1.0<plo<=phi<1.0):
        raise ValueError("phi_bounds must lie inside (-1,1)")
    amin=_min_abs_on_interval(plo,phi)
    q=ar1_expected_design_energy(x,amin,vx,horizon)
    I=d*d*q/(2.0*V)
    return {
        "hard_pair_expected_kl_nats":float(I),
        "hard_pair_expected_design_energy":float(q),
        "hard_pair_phi_abs":float(amin),
        "hard_pair_return_variance":V,
        "hard_pair_signal_variance":vx,
        "coefficient_gap":d,
        "horizon":int(horizon),
        "orthogonal_hard_pair_required_for_this_converse":True,
    }


def g2_direct_certification_horizon_sandwich(
    current_signal: float,
    alpha: float,
    beta: float,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds: tuple[float,float],
    signal_variance_lower: float,
    *,
    max_horizon: int = 5000,
    eta_grid_size: int = 161,
) -> dict:
    """Necessary/sufficient calendar-time sandwich for direct compact-G2 certification.

    The lower endpoint is the first horizon at which the orthogonal hard pair no
    longer violates the universal binary-KL data-processing obstruction.  The
    upper endpoint is the first horizon at which the explicit predictable
    information-clock test reaches power 1-beta uniformly over the compact
    nuisance class.  Thus, whenever both endpoints are finite,

        H_nec <= H_opt <= H_suf,

    for the minimax reliable-deployment horizon H_opt of this declared class.

    The result is an existence/achievability theorem.  It does not assert that
    AlphaValue's proper-NIG rectangle hits the economic boundary at H_suf; the
    direct clock test is a separate, lifetime-matched certifier.
    """
    alpha,beta=map(float,(alpha,beta))
    if not (0.0<alpha<0.5 and 0.0<beta<0.5 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,0.5) and 1-beta>alpha")
    if not isinstance(max_horizon,int) or isinstance(max_horizon,bool) or max_horizon<1:
        raise ValueError("max_horizon must be a positive integer")
    threshold=binary_kl(1.0-beta,alpha)
    h_nec=None; nec_last=None
    h_suf=None; suf_last=None
    for H in range(1,max_horizon+1):
        if h_nec is None:
            nec_last=g2_structural_hard_pair_kl(
                current_signal,H,coefficient_gap,return_variance_upper,
                phi_bounds,signal_variance_lower,
            )
            if nec_last["hard_pair_expected_kl_nats"]>=threshold*(1.0-1e-12):
                h_nec=H
        if h_suf is None:
            suf_last=g2_structural_clock_power_lower_bound(
                current_signal,H,alpha,coefficient_gap,return_variance_upper,
                phi_bounds,signal_variance_lower,eta_grid_size=eta_grid_size,
            )
            if suf_last["power_lower_bound"]>=(1.0-beta)*(1.0-1e-12):
                h_suf=H
        if h_nec is not None and h_suf is not None:
            break
    status=("DIRECT_G2_HORIZON_SANDWICHED" if h_nec is not None and h_suf is not None
            else "DIRECT_G2_HORIZON_NOT_FULLY_RESOLVED_WITHIN_MAX")
    return {
        "status":status,
        "necessary_horizon_pairwise_kl":h_nec,
        "sufficient_horizon_clock":h_suf,
        "binary_kl_threshold_nats":float(threshold),
        "target_power":float(1.0-beta),
        "alpha":alpha,"beta":beta,
        "max_horizon":int(max_horizon),
        "necessary_endpoint":nec_last,
        "sufficient_endpoint":suf_last,
        "same_class_minimax_ordering_certified":bool(h_nec is not None and h_suf is not None and h_nec<=h_suf),
    }


def g2_resolution_time_bound(
    current_signal: float,
    null_gate_error: float,
    unresolved_alt_probability: float,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds: tuple[float,float],
    signal_variance_lower: float,
    *,
    max_horizon: int = 5000,
    eta_grid_size: int = 161,
) -> dict:
    """Constructive bound for the economic-null resolution time.

    Define tau_res as the first predeclared design-clock gate produced by the
    direct structural G2 test.  This routine returns a deterministic H such that

        P_alt(tau_res <= H) >= 1-delta_1

    uniformly over the declared alternative class, while the probability that
    a null world ever passes this *single* gate is at most ``null_gate_error``.

    Unlike confidence-set coverage, ``unresolved_alt_probability`` is a genuine
    power-resolution error.  It is therefore the quantity needed in an
    end-to-end survival calculation.
    """
    a=float(null_gate_error); d1=float(unresolved_alt_probability)
    if not (0.0<a<0.5 and 0.0<d1<0.5):
        raise ValueError("gate and unresolved probabilities must lie in (0,0.5)")
    out=g2_direct_certification_horizon_sandwich(
        current_signal,a,d1,coefficient_gap,return_variance_upper,
        phi_bounds,signal_variance_lower,max_horizon=max_horizon,
        eta_grid_size=eta_grid_size,
    )
    return {
        **out,
        "resolution_horizon_upper":out["sufficient_horizon_clock"],
        "null_gate_error":a,
        "unresolved_alt_probability":d1,
        "resolution_probability_lower":float(1.0-d1),
        "interpretation":"Constructive hitting-time bound for a direct economic-null gate, not a posterior statement about a realized confidence set.",
    }


def g2_weak_gap_asymptotic_frontier_constant(
    alpha: float,
    beta: float,
    return_variance_upper: float,
    phi_bounds: tuple[float,float],
    signal_variance_lower: float,
) -> dict:
    """First-order weak-gap constant for the compact-G2 survival frontier.

    Let d be the uniform loading gap above a fixed economic boundary and let
    d -> 0 while the nuisance class stays compact.  Write

        s_min = vX_min / (1-phi_*^2),
        phi_* = argmin_{phi in Phi} |phi|.

    In the least-informative stationary hard world the per-observation KL rate
    is asymptotically d^2 s_min/(2 V).  The direct information-clock test has
    the same first-order rate uniformly over the class.  Consequently the
    minimax reliable-certification horizon satisfies

        d^2 H_*(d) -> C_*,
        C_* = V (z_{1-alpha}+z_{1-beta})^2 / s_min,

    equivalently r_min(d) H_*(d) -> I_z.

    The theorem behind this reported constant assumes the nuisance class
    contains the orthogonal least-informative hard pair for the converse and
    uses the AR(1) design-quantile bound for achievability.  This helper returns
    the constant; it is not a finite-d guarantee by itself.
    """
    alpha,beta,V,vx=map(float,(alpha,beta,return_variance_upper,signal_variance_lower))
    if not (0.0<alpha<0.5 and 0.0<beta<0.5 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,0.5) and 1-beta>alpha")
    if V<=0.0 or vx<=0.0 or not (math.isfinite(V) and math.isfinite(vx)):
        raise ValueError("variance envelope and signal variance floor must be positive finite")
    plo,phi=map(float,phi_bounds)
    if not (-1.0<plo<=phi<1.0):
        raise ValueError("phi_bounds must lie inside (-1,1)")
    amin=_min_abs_on_interval(plo,phi)
    smin=vx/(1.0-amin*amin)
    iz=float(reliable_information_threshold(alpha,beta)["I_crit_nats"])
    zsum2=2.0*iz
    C=V*zsum2/smin
    return {
        "worst_stationary_signal_variance":float(smin),
        "min_abs_phi":float(amin),
        "canonical_information_threshold_nats":float(iz),
        "weak_gap_horizon_constant":float(C),
        "limit_statement":"coefficient_gap^2 * H_opt -> weak_gap_horizon_constant",
        "equivalent_information_limit":"worst_case_information_rate * H_opt -> I_z",
        "assumptions":[
            "compact nuisance class with vR<=V and vX>=vX_min>0",
            "stable AR(1) persistence interval inside (-1,1)",
            "orthogonal least-informative hard pair belongs to the class for the converse",
            "fixed alpha,beta with coefficient gap d tending to zero",
        ],
    }

# ---------------------------------------------------------------------------
# Random information deadlines and shadow-Snell integration
# ---------------------------------------------------------------------------

def random_deadline_expected_information_obstruction(
    mean_information_horizon: float,
    alpha: float,
    beta: float,
) -> dict:
    """Portable necessary condition under a random information deadline.

    In the canonical likelihood experiment, let H be an integrable stopping
    time in *information-clock units*. Under P1,

        Z_H = B_H + H/2,

    and, under the usual optional-sampling/UI assumptions,

        D(P1^H || P0^H) = E_1[H]/2.

    Any deployment decision measurable no later than H with type-I <= alpha and
    power >= 1-beta therefore requires

        E_1[H]/2 >= kl(1-beta, alpha).

    This is necessary only; the distribution and predictability of H can make
    certification much harder than its mean suggests.
    """
    m, alpha, beta = map(float, (mean_information_horizon, alpha, beta))
    if m < 0.0 or not math.isfinite(m):
        raise ValueError("mean_information_horizon must be finite and nonnegative")
    if not (0.0 < alpha < 1.0 and 0.0 < beta < 1.0 and 1.0-beta > alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    avail = 0.5*m
    req = binary_kl(1.0-beta, alpha)
    return {
        "mean_information_horizon": m,
        "expected_kl_nats": avail,
        "binary_kl_required_nats": req,
        "clears_necessary_mean_information_barrier": bool(avail + 1e-15 >= req),
        "necessary_not_sufficient": True,
    }


def random_deadline_frechet_power_lower(
    certification_power_by_horizon: float,
    deadline_survival_probability: float,
) -> float:
    """Dependence-robust lower bound for certification before a deadline.

    If F(h) <= P_1(tau_cert <= h) and S(h) <= P_1(H > h), then

        P_1(tau_cert < H) >= [F(h) + S(h) - 1]_+.

    No independence between the certification time and the deadline is needed.
    """
    F, S = map(float, (certification_power_by_horizon, deadline_survival_probability))
    if not (0.0 <= F <= 1.0 and 0.0 <= S <= 1.0):
        raise ValueError("probabilities must lie in [0,1]")
    return max(0.0, F + S - 1.0)


def g2_random_deadline_power_lower_bound(
    horizons,
    deadline_survival_lower,
    current_signal: float,
    alpha: float,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds,
    signal_variance_lower: float,
    *,
    eta_grid_size: int = 81,
) -> dict:
    """Compose the direct G2 certifier with a stochastic economic deadline.

    ``deadline_survival_lower[j]`` must be a valid lower bound on
    P_1(T_death > horizons[j]) uniformly over the declared alternative class.
    For each deterministic h we combine the compact-G2 lower bound
    F(h) <= P_1(tau_cert<=h) with the survival lower bound by the Frechet
    inequality.  Maximizing over a *predeclared deterministic grid* preserves
    validity and requires no independence between certification and death.
    """
    hs = [int(h) for h in horizons]
    surv = [float(s) for s in deadline_survival_lower]
    if len(hs) == 0 or len(hs) != len(surv):
        raise ValueError("horizons and deadline_survival_lower must have equal nonzero length")
    if any(h < 1 for h in hs) or any(not (0.0 <= s <= 1.0) for s in surv):
        raise ValueError("horizons must be positive integers and survival bounds in [0,1]")
    rows=[]
    best=None
    for h,s in zip(hs,surv):
        det = g2_structural_clock_power_lower_bound(
            current_signal, h, alpha, coefficient_gap,
            return_variance_upper, phi_bounds, signal_variance_lower,
            eta_grid_size=eta_grid_size,
        )
        F=float(det["power_lower_bound"])
        joint=random_deadline_frechet_power_lower(F,s)
        row={
            "horizon":h,
            "deadline_survival_lower":s,
            "deterministic_certification_power_lower":F,
            "certify_before_death_power_lower":joint,
            "clock_details":det,
        }
        rows.append(row)
        if best is None or joint > best["certify_before_death_power_lower"] + 1e-15:
            best=row
    return {
        "power_lower_bound": float(best["certify_before_death_power_lower"]),
        "best_horizon": int(best["horizon"]),
        "best_deadline_survival_lower": float(best["deadline_survival_lower"]),
        "best_deterministic_certification_power_lower": float(best["deterministic_certification_power_lower"]),
        "grid": rows,
        "requires_deadline_certifier_independence": False,
        "bound_type": "FRECHET_DEPENDENCE_ROBUST",
    }


def exponential_information_deadline_frontier(
    alpha: float,
    beta: float,
    hazard: float,
) -> dict:
    """Exact canonical frontier for an independent exponential info deadline.

    Let H~Exp(rho) in information time, independent of the canonical LLR
    Brownian motion Z.  The opportunity disappears without advance warning.
    A constant likelihood threshold h has killed hitting probabilities

        p_i(h) = E_i[e^{-rho tau_h}]
               = exp[-a_i h],

    with
        a_1 = sqrt(1/4+2rho)-1/2,   under P1,
        a_0 = sqrt(1/4+2rho)+1/2,   under P0.

    The infinite-horizon discounted optimal-stopping problem has a constant
    threshold, so the exact level-alpha power envelope is

        pi_{alpha,rho} = alpha**(a_1/a_0).

    Memorylessness gives E[(H-S)1_{S<H,D=1}] = P1(D=1)/rho, hence the
    alpha-only certified fraction of expected perfect-information value equals
    this power envelope.  Reliable (alpha,beta) certification is feasible iff
    pi_{alpha,rho} >= 1-beta.

    Exponential random horizons are classical optimal-stopping/sequential-
    testing objects; this helper records their CAC specialization rather than
    claiming the killed-Brownian calculation itself as new.
    """
    alpha,beta,rho=map(float,(alpha,beta,hazard))
    if not (0.0<alpha<1.0 and 0.0<beta<1.0 and 1.0-beta>alpha):
        raise ValueError("require alpha,beta in (0,1) and 1-beta>alpha")
    if not (rho>0.0 and math.isfinite(rho)):
        raise ValueError("hazard must be finite and positive")
    s=math.sqrt(0.25+2.0*rho)
    a1=s-0.5
    a0=s+0.5
    kappa=a1/a0
    h=math.log(1.0/alpha)/a0
    power=math.exp(-a1*h)
    # Algebraically identical; retain both as a useful regression check.
    power_roc=alpha**kappa
    p0=math.exp(-a0*h)
    mean_H=1.0/rho
    mean_KL=0.5/rho
    ptarget=1.0-beta
    r=math.log(ptarget)/math.log(alpha)
    rho_crit=r/(2.0*(1.0-r)**2)
    mean_KL_crit=(1.0-r)**2/r
    feasible=power + 1e-14 >= ptarget
    return {
        "alpha":alpha,
        "beta":beta,
        "hazard":rho,
        "threshold":h,
        "null_killed_hitting_probability":p0,
        "power_envelope":power,
        "power_envelope_roc_form":power_roc,
        "roc_exponent":kappa,
        "expected_information_horizon":mean_H,
        "expected_lifetime_kl_nats":mean_KL,
        "information_time_capacity_alpha_only":power/rho,
        "certified_fraction_of_expected_PI_value_alpha_only":power,
        "reliable_target_feasible":bool(feasible),
        "critical_hazard_for_reliable_target":rho_crit,
        "critical_expected_lifetime_kl_nats":mean_KL_crit,
    }


def constant_information_rate_exponential_calendar_deadline(
    alpha: float,
    beta: float,
    kl_rate_per_calendar_time: float,
    calendar_hazard: float,
) -> dict:
    """Translate an exponential *calendar* deadline into information time.

    If KL accumulates deterministically at rate ``iota`` nats per calendar unit,
    the canonical information clock A accumulates at rate 2*iota.  For an
    independent calendar deadline T~Exp(lambda), H=A_T is exponential in
    information time with hazard rho=lambda/(2*iota).

    The result is exact when the information rate is deterministic.  It is the
    local weak-gap approximation used by the G2 corollary below when the random
    design clock self-averages to its least-compatible stationary rate.
    """
    iota, lam = map(float, (kl_rate_per_calendar_time, calendar_hazard))
    if not (iota > 0.0 and math.isfinite(iota)):
        raise ValueError("kl_rate_per_calendar_time must be finite and positive")
    if not (lam > 0.0 and math.isfinite(lam)):
        raise ValueError("calendar_hazard must be finite and positive")
    rho=lam/(2.0*iota)
    out=exponential_information_deadline_frontier(alpha,beta,rho)
    return {
        **out,
        "calendar_hazard":lam,
        "kl_rate_per_calendar_time":iota,
        "information_clock_rate":2.0*iota,
        "mean_calendar_lifetime":1.0/lam,
        "mean_lifetime_kl_nats":iota/lam,
        "information_time_hazard":rho,
    }


def g2_weak_gap_exponential_death_frontier(
    alpha: float,
    beta: float,
    coefficient_gap: float,
    return_variance_upper: float,
    phi_bounds,
    signal_variance_lower: float,
    calendar_hazard: float,
) -> dict:
    """Weak-gap G2 frontier under an independent exponential death hazard.

    The compact-G2 result shows that the problem canonicalizes locally at the
    least-compatible stationary KL rate

        iota_*(d) = d^2 s_* / (2 v_R^+),
        s_* = v_X^-/(1-phi_*^2).

    Combining that local limit with the exact exponential information-deadline
    benchmark gives the hazard-adjusted weak-alpha frontier.  The returned
    feasibility label is therefore an *asymptotic weak-gap* statement, not an
    exact finite-d G2 theorem.
    """
    d,V,vx,lam=map(float,(coefficient_gap,return_variance_upper,signal_variance_lower,calendar_hazard))
    if d <= 0.0 or V <= 0.0 or vx <= 0.0 or lam <= 0.0:
        raise ValueError("gap, variances, and hazard must be positive")
    plo,phi=map(float,phi_bounds)
    if not (-1.0 < plo <= phi < 1.0):
        raise ValueError("phi_bounds must lie in (-1,1)")
    amin=_min_abs_on_interval(plo,phi)
    sstar=vx/(1.0-amin*amin)
    iota=d*d*sstar/(2.0*V)
    base=constant_information_rate_exponential_calendar_deadline(alpha,beta,iota,lam)
    p=1.0-float(beta)
    rr=math.log(p)/math.log(float(alpha))
    Jcrit=(1.0-rr)**2/rr
    dcrit=math.sqrt(2.0*V*lam*Jcrit/sstar)
    return {
        **base,
        "coefficient_gap":d,
        "critical_coefficient_gap_weak_limit":dcrit,
        "worst_stationary_signal_variance":sstar,
        "weak_gap_kl_rate_nats_per_calendar_time":iota,
        "weak_gap_feasible":bool(d + 1e-14 >= dcrit),
        "asymptotic_weak_gap_statement":True,
    }


def canonical_random_deadline_power_upper_at_cut(
    alpha: float,
    information_cut: float,
    deadline_survival_probability: float,
) -> float:
    """Distribution-aware necessary upper bound on power before random death.

    In the canonical simple-vs-simple information-time experiment, any level-
    alpha deployment rule that succeeds before random deadline H satisfies, for
    every deterministic cut h,

        P_1(deploy before H)
        <= P_1(deploy by h) + P_1(H>h)
        <= pi_alpha(h) + P_1(H>h).

    No independence between H and the likelihood path is needed.  The function
    returns min(1, pi_alpha(h)+S_H(h)).  Taking an infimum over h gives a valid
    power upper bound and hence an impossibility certificate when it falls below
    the target power.
    """
    alpha,h,S=float(alpha),float(information_cut),float(deadline_survival_probability)
    if not (0.0<alpha<1.0 and h>=0.0 and 0.0<=S<=1.0):
        raise ValueError("require alpha in (0,1), information_cut>=0, survival in [0,1]")
    return min(1.0, gaussian_power_envelope(alpha,h)+S)


def canonical_random_deadline_power_sandwich(
    alpha: float,
    information_cuts,
    deadline_survival,
    certification_power_lower=None,
) -> dict:
    """Power sandwich for a random information deadline on a fixed cut grid.

    ``deadline_survival[j]`` is S(h_j)=P_1(H>h_j).  The upper endpoint uses the
    canonical deterministic NP envelope and is valid without independence.  If
    a certified lower bound F_j<=P_1(tau_cert<=h_j) is supplied, the lower
    endpoint is max_j [F_j+S(h_j)-1]_+.
    """
    hs=[float(x) for x in information_cuts]
    ss=[float(x) for x in deadline_survival]
    if not hs or len(hs)!=len(ss) or any(h<0 for h in hs) or any(not 0<=s<=1 for s in ss):
        raise ValueError("cuts/survival must be equal-length nonempty valid sequences")
    ups=[canonical_random_deadline_power_upper_at_cut(alpha,h,s) for h,s in zip(hs,ss)]
    upper=min(ups)
    out={
        "power_upper_bound":float(upper),
        "upper_best_cut":float(hs[ups.index(upper)]),
        "upper_grid":list(zip(hs,ups)),
    }
    if certification_power_lower is not None:
        fs=[float(x) for x in certification_power_lower]
        if len(fs)!=len(hs) or any(not 0<=f<=1 for f in fs):
            raise ValueError("certification_power_lower must match cuts and lie in [0,1]")
        lows=[random_deadline_frechet_power_lower(f,s) for f,s in zip(fs,ss)]
        lower=max(lows)
        out.update({
            "power_lower_bound":float(lower),
            "lower_best_cut":float(hs[lows.index(lower)]),
            "lower_grid":list(zip(hs,lows)),
            "sandwich_ordered":bool(lower <= upper + 1e-12),
        })
    return out
