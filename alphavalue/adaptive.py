"""Finite-world Bayesian trading in a specified exogenous Gaussian model.

Only signal persistence is unknown. Costs, loading and innovation variance are
common and known. The criterion is a quadratic mean-risk objective, not a Sharpe
ratio, utility certainty equivalent, or confidence guarantee for market data.
"""
from __future__ import annotations

from dataclasses import dataclass
import math
from typing import Literal

import numpy as np
from scipy.special import logsumexp


@dataclass(frozen=True)
class FiniteWorldModel:
    """X[k+1] = phi*X[k] + Gaussian innovation, initially stationary.

    Trading reward is theta*X*q - gamma*q²/2 - trading_cost*(q-qprev)²/2,
    with initial inventory zero and terminal_penalty*q_last²/2 charged once.
    The current signal is observed before each action. The return noise, if any,
    is mean zero and independent of the entire signal path and world conditional
    on that path; hence observing returns supplies no extra world information.
    """
    persistences: tuple[float, ...] = (0.97, 0.94)
    gamma: float = 1.0
    trading_cost: float = 100.0
    theta: float = 1.0
    innovation_variance: float = 0.03
    terminal_penalty: float = 0.0

    def __post_init__(self):
        phis = tuple(float(p) for p in self.persistences)
        object.__setattr__(self, "persistences", phis)
        scalars = (self.gamma, self.trading_cost, self.theta,
                   self.innovation_variance, self.terminal_penalty)
        if not all(math.isfinite(v) for v in scalars + phis):
            raise ValueError("model parameters must be finite")
        if (not phis or len(set(phis)) != len(phis)
                or any(abs(p) >= 1 for p in phis)):
            raise ValueError("persistences must be distinct and strictly between -1 and 1")
        if (self.gamma <= 0 or self.trading_cost < 0 or self.theta == 0
                or self.innovation_variance <= 0 or self.terminal_penalty < 0):
            raise ValueError("require gamma>0, cost>=0, theta!=0, innovation variance>0, terminal penalty>=0")
        scale = self.trading_cost + self.terminal_penalty
        if not math.isfinite(scale + self.gamma) or scale + self.gamma == scale:
            raise ValueError("cost scale exceeds supported floating-point precision")
        if not np.isfinite(self.stationary_variances).all():
            raise ValueError("stationary variance exceeds supported floating-point range")

    @property
    def stationary_variances(self) -> np.ndarray:
        phis = np.asarray(self.persistences)
        return self.innovation_variance / ((1.0 - phis) * (1.0 + phis))


def _positive_int(value, name, *, allow_zero=False):
    if (not isinstance(value, (int, np.integer)) or isinstance(value, (bool, np.bool_))
            or value < (0 if allow_zero else 1)):
        raise ValueError(f"{name} must be an integer >= {0 if allow_zero else 1}")


@dataclass(frozen=True)
class BellmanCoefficients:
    curvature: np.ndarray  # D[t]
    retention: np.ndarray  # a[t]
    gains: np.ndarray      # g[t, world]
    oracle_values: np.ndarray


def bellman_coefficients(model: FiniteWorldModel, horizon: int) -> BellmanCoefficients:
    """Exact finite-horizon recurrences, evaluated in floating-point arithmetic.

    oracle_values[i] is the expected optimal total objective in world i, from
    stationary X and zero initial inventory. It includes the terminal charge.
    """
    _positive_int(horizon, "horizon")
    phis = np.asarray(model.persistences)
    D = np.empty(horizon)
    a = np.empty(horizon)
    g = np.empty((horizon, len(phis)))
    K = model.terminal_penalty
    L = np.zeros(len(phis))
    for t in range(horizon - 1, -1, -1):
        D[t] = model.gamma + model.trading_cost + K
        a[t] = model.trading_cost / D[t]
        g[t] = (model.theta + phis * L) / D[t]
        # This avoids subtracting two almost equal terms at large trading cost.
        K = model.trading_cost * ((model.gamma + K) / D[t])
        L = model.trading_cost * g[t]
    values = 0.5 * model.stationary_variances * np.sum(D[:, None] * g * g, axis=0)
    if not (np.isfinite(D).all() and np.isfinite(g).all()
            and np.isfinite(values).all() and (values > 0).all()):
        raise ValueError("Bellman calculation exceeds supported floating-point range")
    return BellmanCoefficients(D, a, g, values)


def _prior_array(prior, worlds):
    p = np.full(worlds, 1.0 / worlds) if prior is None else np.asarray(prior, dtype=float)
    if (p.shape != (worlds,) or not np.isfinite(p).all() or (p < 0).any()
            or not math.isfinite(float(p.sum())) or p.sum() <= 0):
        raise ValueError("prior must contain one finite nonnegative weight per world, with positive sum")
    return p / p.sum()


def _normalize_log_weights(log_weights):
    z = logsumexp(log_weights)
    if not math.isfinite(float(z)):
        raise ValueError("signal scale exceeds supported likelihood precision")
    return log_weights - z


def posterior_path(signal, model: FiniteWorldModel, prior=None) -> np.ndarray:
    """Posterior[i] uses X[0],...,X[i] only, including stationary X[0] density.

    A zero prior weight remains zero. There is no parameter estimation from the
    complete path before filtering. Each transition updates the previous filter.
    """
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1 or len(x) < 1 or not np.isfinite(x).all():
        raise ValueError("signal must be a nonempty finite one-dimensional array")
    p = _prior_array(prior, len(model.persistences))
    with np.errstate(divide="ignore"):
        logp = np.log(p)
    variances = model.stationary_variances
    phis = np.asarray(model.persistences)
    with np.errstate(over="ignore", invalid="ignore"):
        logp = _normalize_log_weights(logp - 0.5 * np.log(variances) - 0.5 * (x[0] / np.sqrt(variances)) ** 2)
    out = np.empty((len(x), len(p)))
    out[0] = np.exp(logp)
    for k in range(1, len(x)):
        with np.errstate(over="ignore", invalid="ignore"):
            standardized = (x[k] - phis * x[k - 1]) / math.sqrt(model.innovation_variance)
            logp = _normalize_log_weights(logp - 0.5 * standardized ** 2)
        out[k] = np.exp(logp)
    return out


@dataclass(frozen=True)
class PolicyRun:
    positions: np.ndarray
    effective_gains: np.ndarray
    posterior: np.ndarray
    decision_weights: np.ndarray
    oracle_values: np.ndarray
    residual_score_by_world: np.ndarray
    policy: str
    objective: str


def two_world_fixed_blend(model: FiniteWorldModel, horizon: int) -> dict:
    """Exact optimal fixed gain blend and an upper bound on minimax regret.

    This minimizes worst-world relative regret over policies with a constant
    mixture weight between the two oracle gains. It is NOT asserted to minimize
    regret over all adaptive policies. The value is an upper bound on that harder
    minimax problem, and its formula does not use training observations.
    """
    if len(model.persistences) != 2:
        raise ValueError("fixed blend requires exactly two candidate worlds")
    coeff = bellman_coefficients(model, horizon)
    difference = coeff.gains[:, 0] - coeff.gains[:, 1]
    scale = 0.5 * float(np.dot(coeff.curvature, difference*difference))
    A = model.stationary_variances * scale / coeff.oracle_values
    roots = np.sqrt(A)
    if roots.sum() == 0:
        weight, upper = 0.5, 0.0
    else:
        weight = float(roots[0]/roots.sum())
        upper = float((roots[0]*roots[1]/roots.sum())**2)
    return {"world0_gain_weight": weight,
            "relative_regret_by_world": [float(A[0]*(1-weight)**2), float(A[1]*weight**2)],
            "minimax_relative_regret_upper": upper,
            "scope": "Exact in the stated model for the fixed blend; upper bound on unrestricted causal minimax regret"}


def adaptive_policy(signal, *, model=FiniteWorldModel(), training_transitions=0,
                    horizon=None, prior=None,
                    objective: Literal["absolute", "relative"] = "absolute",
                    policy: Literal["bayes", "frozen_bayes", "posterior_parameter", "fixed_minimax_blend", "zero"] = "bayes") -> PolicyRun:
    """Evaluate one causal policy on a signal path with a fixed deployment horizon.

    Deployment starts at X[training_transitions]. The first action has observed
    exactly that many transitions and training_transitions+1 signal values.
    Extra input observations beyond the fixed horizon are ignored.

    'bayes' averages oracle *gains*. It minimizes prior expected absolute regret,
    or prior expected world-normalized regret when objective='relative'. In the
    latter case decision weights are posterior[i]/oracle_values[i], normalized.
    These weights are a loss adjustment, not the ordinary world posterior.

    residual_score_by_world[i] = sum D/2*(q-a*qprev-g_i*X)^2. Its EXPECTATION in
    world i equals expected regret. It is not realized-profit regret on one path.
    """
    _positive_int(training_transitions, "training_transitions", allow_zero=True)
    x = np.asarray(signal, dtype=float)
    if x.ndim != 1:
        raise ValueError("signal must be a one-dimensional array")
    if horizon is None:
        horizon = len(x) - training_transitions
    _positive_int(horizon, "horizon")
    if len(x) < training_transitions + horizon:
        raise ValueError("insufficient observations for training and deployment horizon")
    x = x[:training_transitions + horizon]
    if not np.isfinite(x).all():
        raise ValueError("observed signal must be finite")
    if objective not in ("absolute", "relative"):
        raise ValueError("objective must be 'absolute' or 'relative'")
    if policy not in ("bayes", "frozen_bayes", "posterior_parameter", "fixed_minimax_blend", "zero"):
        raise ValueError("unknown policy")
    coeff = bellman_coefficients(model, horizon)
    ordinary = posterior_path(x, model, prior)[training_transitions:]
    # Compute the tilted filter directly so tiny posterior probabilities remain
    # representable when their world value is very small.
    if objective == "relative":
        p = _prior_array(prior, len(model.persistences))
        with np.errstate(divide="ignore"):
            tilted_log = _normalize_log_weights(np.log(p) - np.log(coeff.oracle_values))
        weights = posterior_path(x, model, np.exp(tilted_log))[training_transitions:]
    else:
        weights = ordinary.copy()
    if policy == "frozen_bayes":
        weights = np.repeat(weights[:1], horizon, axis=0)
    if policy == "fixed_minimax_blend":
        if objective != "relative":
            raise ValueError("fixed_minimax_blend requires objective='relative'")
        w = two_world_fixed_blend(model, horizon)["world0_gain_weight"]
        weights = np.tile([w, 1-w], (horizon, 1))
    gains = np.sum(weights * coeff.gains, axis=1)
    if policy == "posterior_parameter":
        estimated_phis = weights @ np.asarray(model.persistences)
        # A comparison heuristic: solve the known-parameter problem at the
        # posterior mean parameter. This generally differs from averaging gains.
        for t, phi in enumerate(estimated_phis):
            local = FiniteWorldModel((float(phi),), model.gamma, model.trading_cost,
                                     model.theta, model.innovation_variance, model.terminal_penalty)
            gains[t] = bellman_coefficients(local, horizon - t).gains[0, 0]
    positions = np.empty(horizon)
    score = np.zeros(len(model.persistences))
    previous = 0.0
    for t, xt in enumerate(x[training_transitions:]):
        positions[t] = 0.0 if policy == "zero" else coeff.retention[t] * previous + gains[t] * xt
        residual = positions[t] - coeff.retention[t] * previous - coeff.gains[t] * xt
        score += 0.5 * coeff.curvature[t] * residual * residual
        previous = positions[t]
    if not (np.isfinite(positions).all() and np.isfinite(score).all()):
        raise ValueError("policy evaluation exceeds supported floating-point range")
    if policy == "zero":
        gains = np.zeros(horizon)
    return PolicyRun(positions, gains, ordinary, weights, coeff.oracle_values,
                     score, policy, objective)


def phase_order_proxy(*, horizon: int, training_transitions: int, epsilon: float) -> float:
    """Uncalibrated order proxy, NOT a confidence bound or exact finite-n rate.

    Uses H=T-t, b_H=min(H*epsilon,1), c_H=min((H-1)*epsilon,1), and
    sum b_H²*c_H²/[1+(n+t)*epsilon] / sum b_H².
    H=1 gives zero because the last decision is independent of persistence.
    This proxy concerns the local positive near-unit-root regime only.
    """
    _positive_int(horizon, "horizon")
    _positive_int(training_transitions, "training_transitions", allow_zero=True)
    if not math.isfinite(epsilon) or not 0 < epsilon <= 0.25:
        raise ValueError("epsilon must be in (0, 0.25]")
    t = np.arange(horizon, dtype=float)
    remaining = horizon - t
    b = np.minimum(remaining * epsilon, 1.0)
    c = np.minimum((remaining - 1.0) * epsilon, 1.0)
    denominator = float(np.sum(b * b))
    if denominator == 0:
        raise ValueError("epsilon is too small for supported floating-point range")
    return float(np.sum(b * b * c * c / (1.0 + (training_transitions + t) * epsilon)) / denominator)


def general_phase_order_proxy(n: int, T: int, a: float, delta: float) -> float:
    """Dimensionless matching-order functional with separate memory scales.

    n counts training transitions, T counts deployment decisions, and a is the
    infinite-horizon inventory retention, lambda/gamma=a/(1-a)**2. The theorem
    has persistence interval [1-2*delta, 1-delta], known innovation variance delta
    and known nonzero loading theta. Initial inventory and terminal charge are
    zero, and signals start stationary.

    h=1-a+a*delta, b_H=min(H*h,1), c_H=min((H-1)*h,1); the return value is
    (a*delta/h)**2 * sum b_H**2*c_H**2/(1+(n+t)*delta) / sum b_H**2.
    This is NOT calibrated regret, a profit prediction, or a confidence bound.
    At a=0 or T=1, persistence-learning loss is zero.
    """
    _positive_int(n, "n", allow_zero=True)
    _positive_int(T, "T")
    if not math.isfinite(a) or not 0 <= a < 1:
        raise ValueError("a must be finite and in [0, 1)")
    if not math.isfinite(delta) or not 0 < delta <= 0.5:
        raise ValueError("delta must be finite and in (0, 0.5]")
    if a == 0 or T == 1:
        return 0.0
    try:
        n_float = float(n)
    except OverflowError as error:
        raise ValueError("n exceeds supported floating-point range") from error
    if not math.isfinite(n_float):
        raise ValueError("n exceeds supported floating-point range")
    h = (1.0-a) + a*delta
    t = np.arange(T, dtype=float)
    remaining = T-t
    b = np.minimum(remaining*h, 1.0)
    c = np.minimum((remaining-1.0)*h, 1.0)
    # Common scaling preserves the ratio without squaring tiny b directly.
    b_scaled = b / b[0]
    with np.errstate(over="ignore"):
        information_scale = 1.0 + (n_float+t)*delta
    weighted_ratio = np.sum(b_scaled*b_scaled*c*c/information_scale) / np.sum(b_scaled*b_scaled)
    return float((a*delta/h)**2 * weighted_ratio)
