"""Certification-constrained equilibrium helpers used by the paper and tests."""
from __future__ import annotations

import math
from typing import Callable

from .cac import reliable_information_threshold


def cac_threshold_nats(alpha: float, beta: float) -> float:
    return float(reliable_information_threshold(alpha, beta)["I_crit_nats"])


def constructive_cac_surplus(
    I: float,
    alpha: float,
    beta: float,
    value_per_nat: float = 1.0,
) -> float:
    """Rigorous lower envelope for optimal reliable CAC economic surplus.

    Test at the critical information time and, on rejection, use the residual
    lifetime.  The rule has power exactly 1-beta at the critical point.
    """
    I = float(I)
    v = float(value_per_nat)
    if v <= 0:
        raise ValueError("value_per_nat must be positive")
    Istar = cac_threshold_nats(alpha, beta)
    return v * (1.0 - float(beta)) * max(0.0, I - Istar)


def exponential_crowding_information(I0: float, chi: float, x: float) -> float:
    I0, chi, x = map(float, (I0, chi, x))
    if I0 <= 0 or chi <= 0 or x < 0:
        raise ValueError("require I0>0, chi>0, x>=0")
    return I0 / (1.0 + chi * x)


def inverse_exponential_crowding(I0: float, chi: float, I: float) -> float:
    I0, chi, I = map(float, (I0, chi, I))
    if I0 <= 0 or chi <= 0 or not (0 < I <= I0):
        raise ValueError("require I0>0, chi>0, 0<I<=I0")
    return (I0 / I - 1.0) / chi


def constructive_equilibrium_information(
    I0: float,
    alpha: float,
    beta: float,
    research_cost: float,
    crowding_strength: float,
    value_per_nat: float = 1.0,
) -> float:
    """Closed-form residual information using the constructive CAC lower envelope."""
    I0 = float(I0)
    c = float(research_cost)
    chi = float(crowding_strength)
    v = float(value_per_nat)
    if I0 <= 0 or c < 0 or chi <= 0 or v <= 0:
        raise ValueError("invalid parameters")
    Istar = cac_threshold_nats(alpha, beta)
    if I0 <= Istar:
        return I0
    if c == 0:
        return Istar
    rho = c / (chi * v * (1.0 - float(beta)))
    disc = (Istar - rho) ** 2 + 4.0 * rho * I0
    return 0.5 * (Istar - rho + math.sqrt(disc))


def generic_equilibrium_information(
    I0: float,
    Istar: float,
    research_cost: float,
    inverse_crowding: Callable[[float], float],
    certified_surplus: Callable[[float], float],
    *,
    tol: float = 1e-12,
    max_iter: int = 300,
) -> float:
    """Solve C(I)=c X(I) on (Istar,I0) by bisection.

    Assumes C is continuous/increasing with C(Istar)=0 and X is continuous/
    decreasing with X(I0)=0.  These are exactly the theorem hypotheses.
    """
    I0, Istar, c = map(float, (I0, Istar, research_cost))
    if not (I0 > 0 and Istar > 0 and c >= 0):
        raise ValueError("invalid parameters")
    if I0 <= Istar:
        return I0
    if c == 0:
        return Istar
    lo, hi = Istar, I0
    flo = certified_surplus(lo) - c * inverse_crowding(lo)
    fhi = certified_surplus(hi) - c * inverse_crowding(hi)
    if flo > 1e-10 or fhi < -1e-10:
        raise ValueError("root is not bracketed by theorem endpoints")
    for _ in range(max_iter):
        mid = 0.5 * (lo + hi)
        fm = certified_surplus(mid) - c * inverse_crowding(mid)
        if abs(fm) <= tol or hi - lo <= tol:
            return mid
        if fm > 0:
            hi = mid
        else:
            lo = mid
    return 0.5 * (lo + hi)


def optimal_equilibrium_wedge_upper_bound(
    I0: float,
    alpha: float,
    beta: float,
    research_cost: float,
    crowding_strength: float,
    value_per_nat: float = 1.0,
) -> float:
    """Rigorous upper bound on I_eq^* - I_crit for optimal CAC.

    Uses C^*(I) >= v(1-beta)(I-I*) and X(I_eq)<=X(I*).
    """
    I0 = float(I0)
    c = float(research_cost)
    chi = float(crowding_strength)
    v = float(value_per_nat)
    if I0 <= 0 or c < 0 or chi <= 0 or v <= 0:
        raise ValueError("invalid parameters")
    Istar = cac_threshold_nats(alpha, beta)
    if I0 <= Istar or c == 0:
        return 0.0
    xstar = inverse_exponential_crowding(I0, chi, Istar)
    return c * xstar / (v * (1.0 - float(beta)))


def linear_information_impact(
    I0: float,
    Istar: float,
    kappa: float,
    aggregate_effort: float,
) -> float:
    """Linear information-destruction benchmark G(E)=max(I*, I0-kappa E)."""
    I0, Istar, kappa, E = map(float, (I0, Istar, kappa, aggregate_effort))
    if not (I0 > 0 and Istar > 0 and I0 >= Istar and kappa > 0 and E >= 0):
        raise ValueError("require I0>=Istar>0, kappa>0, E>=0")
    return max(Istar, I0 - kappa * E)


def strategic_linear_contest_equilibrium(
    I0: float,
    Istar: float,
    n_arbitrageurs: int,
    research_cost: float,
    info_impact: float,
    certified_slope: float,
) -> dict:
    """Exact symmetric Nash benchmark for strategic certification crowding.

    The primitive game is

        u_i(e) = (e_i/E) * a * (I0-Istar-kappa E)_+ - c e_i,

    with E=sum e_i, N>=2, a>0 and kappa>0.  In the active region this is
    exactly a proportional-share (Tullock r=1) contest with fixed prize
    a(I0-Istar) and effective marginal effort cost a*kappa+c.

    Returns the unique symmetric equilibrium with positive gross rent.
    At zero research cost, additional zero-rent equilibria can exist;
    no uniqueness claim for those profiles is made.
    """
    I0 = float(I0)
    Istar = float(Istar)
    c = float(research_cost)
    kappa = float(info_impact)
    a = float(certified_slope)
    N = int(n_arbitrageurs)
    if not (I0 > Istar > 0 and N >= 2 and c >= 0 and kappa > 0 and a > 0):
        raise ValueError("require I0>Istar>0, N>=2, c>=0, kappa>0, slope>0")
    gap = I0 - Istar
    effective_cost = a * kappa + c
    prize = a * gap
    effort_each = prize * (N - 1.0) / (effective_cost * N * N)
    aggregate_effort = N * effort_each
    residual_information = I0 - kappa * aggregate_effort
    # Closed-form wedge decomposition.
    cost_wedge_fraction = c / effective_cost
    concentration_wedge_fraction = (a * kappa) / (N * effective_cost)
    residual_wedge_fraction = (residual_information - Istar) / gap
    return {
        "effort_each": effort_each,
        "aggregate_effort": aggregate_effort,
        "residual_information": residual_information,
        "residual_wedge": residual_information - Istar,
        "residual_wedge_fraction": residual_wedge_fraction,
        "cost_wedge_fraction": cost_wedge_fraction,
        "concentration_wedge_fraction": concentration_wedge_fraction,
        "effective_marginal_cost": effective_cost,
        "fixed_prize_equivalent": prize,
    }


def strategic_linear_contest_continuum_information(
    I0: float,
    Istar: float,
    research_cost: float,
    info_impact: float,
    certified_slope: float,
) -> float:
    """Atomistic/large-N limit of the linear strategic crowding benchmark."""
    I0, Istar, c, kappa, a = map(
        float, (I0, Istar, research_cost, info_impact, certified_slope)
    )
    if not (I0 > Istar > 0 and c >= 0 and kappa > 0 and a > 0):
        raise ValueError("require I0>Istar>0, c>=0, kappa>0, slope>0")
    gap = I0 - Istar
    return Istar + gap * c / (a * kappa + c)


def strategic_share_contest_foc(
    total_effort: float,
    n_arbitrageurs: int,
    research_cost: float,
    rent_pool: Callable[[float], float],
    rent_pool_derivative: Callable[[float], float],
) -> float:
    """Symmetric finite-N FOC residual for an endogenous-prize share contest.

    For u_i=(e_i/E)R(E)-c e_i and E=N e_i at a symmetric interior point,
    the first-order condition is

        (1-1/N) R(E)/E + (1/N) R'(E) - c = 0.

    The atomistic limit is R(E)/E-c=0, which is exactly the reduced-form
    capacity-sharing condition used by the CAC equilibrium theorem.
    """
    E = float(total_effort)
    N = int(n_arbitrageurs)
    c = float(research_cost)
    if E <= 0 or N < 2 or c < 0:
        raise ValueError("require E>0, N>=2, c>=0")
    return (1.0 - 1.0 / N) * float(rent_pool(E)) / E + (
        1.0 / N
    ) * float(rent_pool_derivative(E)) - c



def strategic_arbitrageability_fraction(
    n_arbitrageurs: int,
    research_cost: float,
    info_impact: float,
    certified_slope: float,
) -> float:
    """Fraction of excess information destroyed in the linear strategic benchmark.

    If X=I0-I* denotes excess lifetime information, the exact symmetric
    finite-N benchmark satisfies

        D = m X,
        m = a*kappa*(1-1/N)/(a*kappa+c),

    where D is information destroyed by equilibrium arbitrage.  The fraction
    lies in [0,1) for N>=2, c>=0, kappa>0 and a>0.
    """
    N = int(n_arbitrageurs)
    c = float(research_cost)
    kappa = float(info_impact)
    a = float(certified_slope)
    if N < 2 or c < 0 or kappa <= 0 or a <= 0:
        raise ValueError("require N>=2, c>=0, info_impact>0, certified_slope>0")
    return (a * kappa * (1.0 - 1.0 / N)) / (a * kappa + c)


def heterogeneous_cross_section_slope(
    excess_information,
    arbitrageability,
) -> dict:
    """Exact pooled-slope decomposition for heterogeneous arbitrage technology.

    For observations satisfying D_i = m_i X_i, with X_i>=0 and m_i>=0,
    the population/sample OLS slope of D on X (with intercept) obeys

      beta = E[m] + {Cov(m,X^2)-E[X]Cov(m,X)}/Var(X).

    Hence positive within-technology comparative statics need not imply a
    positive pooled cross-sectional slope when m and X are dependent.
    """
    import numpy as np

    x = np.asarray(excess_information, dtype=float)
    m = np.asarray(arbitrageability, dtype=float)
    if x.ndim != 1 or m.ndim != 1 or len(x) != len(m) or len(x) < 2:
        raise ValueError("excess_information and arbitrageability must be equal-length vectors")
    if not np.all(np.isfinite(x)) or not np.all(np.isfinite(m)):
        raise ValueError("inputs must be finite")
    if np.any(x < 0) or np.any(m < 0):
        raise ValueError("require nonnegative excess information and arbitrageability")
    vx = float(np.var(x, ddof=0))
    if vx <= 0:
        raise ValueError("excess information must have positive variance")
    d = m * x
    cov_x_d = float(np.mean((x - x.mean()) * (d - d.mean())))
    beta = cov_x_d / vx
    cov_m_x = float(np.mean((m - m.mean()) * (x - x.mean())))
    x2 = x * x
    cov_m_x2 = float(np.mean((m - m.mean()) * (x2 - x2.mean())))
    heterogeneity_term = (cov_m_x2 - float(x.mean()) * cov_m_x) / vx
    decomposition = float(m.mean()) + heterogeneity_term
    return {
        "slope": beta,
        "mean_arbitrageability": float(m.mean()),
        "heterogeneity_term": heterogeneity_term,
        "decomposition": decomposition,
        "cov_m_x": cov_m_x,
        "cov_m_x2": cov_m_x2,
    }


def conditional_cross_section_margin(x: float, mean_arbitrageability: float, derivative: float) -> float:
    """Derivative of E[D|X=x]=x*mubar(x).

    A positive value is the exact local condition for a positive conditional
    certifiability-decay comparative static.  When mubar>0 this is equivalent
    to an elasticity of mean arbitrageability with respect to X greater than
    -1.
    """
    x = float(x)
    mu = float(mean_arbitrageability)
    dmu = float(derivative)
    if x < 0 or mu < 0:
        raise ValueError("require x>=0 and mean_arbitrageability>=0")
    return mu + x * dmu
