"""Finite-horizon certificates for an unknown predictive loading.

Model
-----
X follows a stationary Gaussian AR(1) with *known* persistence ``phi`` and
innovation variance. Conditional on the entire signal path,

    Y[t+1] = theta * X[t] + epsilon[t+1],
    epsilon iid N(0, sigma_y^2),

with return noise independent of the full signal path. The trading objective is

    E sum_t [theta X_t q_t - gamma q_t^2/2
             - lambda (q_t-q_{t-1})^2/2],

with q_{-1}=0 and the same finite-horizon conventions as ``adaptive.py``.

The certificate is exact conditional on the observed training signal under these
assumptions when sigma_y is known. It is not a distribution-free market claim.
"""
from __future__ import annotations

from dataclasses import dataclass, asdict
import math
import numpy as np
from scipy.stats import norm

from .adaptive import FiniteWorldModel, bellman_coefficients


@dataclass(frozen=True)
class LoadingCertificateModel:
    phi: float
    gamma: float
    trading_cost: float
    signal_innovation_variance: float
    return_noise_sd: float
    horizon: int
    confidence: float = 0.95
    terminal_penalty: float = 0.0

    def validate(self) -> None:
        vals = (self.phi, self.gamma, self.trading_cost,
                self.signal_innovation_variance, self.return_noise_sd,
                self.confidence, self.terminal_penalty)
        if not all(math.isfinite(v) for v in vals):
            raise ValueError("certificate parameters must be finite")
        if not -1 < self.phi < 1:
            raise ValueError("phi must lie strictly between -1 and 1")
        if self.gamma <= 0 or self.trading_cost < 0:
            raise ValueError("require gamma>0 and trading_cost>=0")
        if self.signal_innovation_variance <= 0 or self.return_noise_sd <= 0:
            raise ValueError("variances must be positive")
        if not isinstance(self.horizon, int) or isinstance(self.horizon, bool) or self.horizon < 1:
            raise ValueError("horizon must be a positive integer")
        if not 0 < self.confidence < 1:
            raise ValueError("confidence must lie in (0,1)")
        if self.terminal_penalty < 0:
            raise ValueError("terminal_penalty must be nonnegative")


def _unit_value_scale(model: LoadingCertificateModel) -> tuple[float, np.ndarray, np.ndarray]:
    """Return A where oracle value equals A*theta^2, plus D and unit gains."""
    model.validate()
    m = FiniteWorldModel(
        persistences=(model.phi,), gamma=model.gamma,
        trading_cost=model.trading_cost, theta=1.0,
        innovation_variance=model.signal_innovation_variance,
        terminal_penalty=model.terminal_penalty,
    )
    coeff = bellman_coefficients(m, model.horizon)
    unit_gain = coeff.gains[:, 0]
    A = float(coeff.oracle_values[0])
    if not math.isfinite(A) or A <= 0:
        raise ArithmeticError("nonpositive or nonfinite model value scale")
    return A, coeff.curvature.copy(), unit_gain.copy()


def theta_interval(signal, next_return, *, return_noise_sd: float, confidence: float = 0.95) -> dict:
    """Exact conditional Gaussian interval for theta.

    ``signal[i]`` must be known before ``next_return[i]``. The result is exact
    conditional on the supplied signal vector under independent Gaussian return
    noise with known standard deviation.
    """
    x = np.asarray(signal, dtype=float)
    y = np.asarray(next_return, dtype=float)
    if x.ndim != 1 or y.ndim != 1 or x.shape != y.shape or len(x) < 1:
        raise ValueError("signal and next_return must be equal nonempty one-dimensional arrays")
    if not np.isfinite(x).all() or not np.isfinite(y).all():
        raise ValueError("signal and next_return must be finite")
    if not math.isfinite(return_noise_sd) or return_noise_sd <= 0:
        raise ValueError("return_noise_sd must be positive and finite")
    if not math.isfinite(confidence) or not 0 < confidence < 1:
        raise ValueError("confidence must lie in (0,1)")
    Q = float(np.dot(x, x))
    if not math.isfinite(Q) or Q <= 0:
        raise ValueError("signal has zero design energy")
    theta_hat = float(np.dot(x, y) / Q)
    se = return_noise_sd / math.sqrt(Q)
    z = float(norm.ppf((1.0 + confidence) / 2.0))
    radius = z * se
    return {
        "theta_hat": theta_hat,
        "standard_error": se,
        "z": z,
        "radius": radius,
        "lower": theta_hat - radius,
        "upper": theta_hat + radius,
        "design_energy": Q,
        "confidence": confidence,
        "scope": "Exact conditional Gaussian interval under known independent return-noise variance",
    }


def robust_theta(interval_lower: float, interval_upper: float) -> float:
    """Maximin loading for the finite-horizon LQ model when only theta is uncertain."""
    lo, hi = float(interval_lower), float(interval_upper)
    if not math.isfinite(lo) or not math.isfinite(hi) or lo > hi:
        raise ValueError("invalid loading interval")
    if lo > 0:
        return lo
    if hi < 0:
        return hi
    return 0.0


def loading_certificate(signal, next_return, *, model: LoadingCertificateModel,
                        hurdle: float = 0.0) -> dict:
    """Return a three-way economic certificate under the declared model.

    With probability ``confidence`` (conditional on the observed signal path),
    the true model value of the robust controller is at least ``policy_value_lower``
    and the oracle value is at most ``oracle_value_upper``.

    Decisions:
    - ``deploy_in_model`` if the robust policy lower bound exceeds ``hurdle``;
    - ``economically_small_in_model`` if even the oracle upper bound is below it;
    - otherwise ``insufficient_evidence``.
    """
    model.validate()
    if not math.isfinite(hurdle) or hurdle < 0:
        raise ValueError("hurdle must be finite and nonnegative")
    ci = theta_interval(signal, next_return,
                        return_noise_sd=model.return_noise_sd,
                        confidence=model.confidence)
    lo, hi = ci["lower"], ci["upper"]
    A, D, unit_gain = _unit_value_scale(model)
    t_rob = robust_theta(lo, hi)
    policy_lower = A * t_rob * t_rob
    oracle_upper = A * max(abs(lo), abs(hi)) ** 2
    oracle_lower = 0.0 if lo <= 0 <= hi else A * min(abs(lo), abs(hi)) ** 2
    plugin_regret_upper = A * ci["radius"] ** 2
    if policy_lower > hurdle:
        decision = "deploy_in_model"
    elif oracle_upper < hurdle:
        decision = "economically_small_in_model"
    else:
        decision = "insufficient_evidence"
    return {
        "model": asdict(model),
        "theta_interval": ci,
        "value_scale_A": A,
        "robust_theta": t_rob,
        "robust_gain_sequence": (t_rob * unit_gain).tolist(),
        "bellman_curvature": D.tolist(),
        "policy_value_lower": policy_lower,
        "oracle_value_lower": oracle_lower,
        "oracle_value_upper": oracle_upper,
        "plugin_regret_upper_on_interval": plugin_regret_upper,
        "hurdle": hurdle,
        "decision": decision,
        "coverage_statement": (
            f"Conditional on the supplied signal path, under the stated Gaussian model with known "
            f"return-noise SD, simultaneous value statements hold with probability {model.confidence:.6g}."
        ),
        "limitations": [
            "Signal persistence and both noise variances are treated as known.",
            "Coverage is model-conditional, not distribution-free or a future-market guarantee.",
            "The objective is finite-horizon expected quadratic mean-risk value, not Sharpe ratio.",
            "No allowance is made for parameter search, structural breaks, impact, capacity or execution misspecification.",
        ],
    }


def policy_value(theta: float, controller_theta: float, *, model: LoadingCertificateModel) -> float:
    """Exact expected value for the controller using controller_theta in the known-phi model."""
    A, _, _ = _unit_value_scale(model)
    return A * (2.0 * float(theta) * float(controller_theta) - float(controller_theta) ** 2)


def oracle_value(theta: float, *, model: LoadingCertificateModel) -> float:
    A, _, _ = _unit_value_scale(model)
    return A * float(theta) ** 2


def joint_misspecification_regret(*, true_theta: float, true_phi: float,
                                  controller_theta: float, controller_phi: float,
                                  gamma: float, trading_cost: float,
                                  signal_innovation_variance: float, horizon: int,
                                  terminal_penalty: float = 0.0) -> dict:
    """Exact expected regret of an oracle-shaped controller with wrong theta/phi.

    The controller uses the finite-horizon gain sequence optimal for
    ``(controller_theta, controller_phi)`` while the true world is
    ``(true_theta, true_phi)``. Risk/cost coefficients and signal innovation
    variance are common and known. The identity follows directly from the
    Bellman residual representation.
    """
    vals=(true_theta,true_phi,controller_theta,controller_phi,gamma,trading_cost,
          signal_innovation_variance,terminal_penalty)
    if not all(math.isfinite(v) for v in vals):
        raise ValueError("parameters must be finite")
    if not -1<true_phi<1 or not -1<controller_phi<1:
        raise ValueError("persistences must lie in (-1,1)")
    if gamma<=0 or trading_cost<0 or signal_innovation_variance<=0 or terminal_penalty<0:
        raise ValueError("invalid economic parameters")
    if not isinstance(horizon,int) or isinstance(horizon,bool) or horizon<1:
        raise ValueError("horizon must be a positive integer")
    true_model=FiniteWorldModel((true_phi,),gamma,trading_cost,1.,signal_innovation_variance,terminal_penalty)
    proxy_model=FiniteWorldModel((controller_phi,),gamma,trading_cost,1.,signal_innovation_variance,terminal_penalty)
    tc=bellman_coefficients(true_model,horizon)
    pc=bellman_coefficients(proxy_model,horizon)
    # Curvature is parameter-free, hence the two arrays must agree numerically.
    if not np.allclose(tc.curvature,pc.curvature,rtol=1e-13,atol=1e-14):
        raise ArithmeticError("parameter-free Bellman curvature mismatch")
    residual=controller_theta*pc.gains[:,0]-true_theta*tc.gains[:,0]
    s2=float(true_model.stationary_variances[0])
    regret=.5*s2*float(np.dot(tc.curvature,residual*residual))
    oracle=.5*s2*float(np.dot(tc.curvature,(true_theta*tc.gains[:,0])**2))
    return {
        "regret":regret,
        "oracle_value":oracle,
        "relative_regret":regret/oracle if oracle>0 else None,
        "true_theta":true_theta,"true_phi":true_phi,
        "controller_theta":controller_theta,"controller_phi":controller_phi,
        "scope":"Exact expected Bellman regret for the declared finite-horizon stationary Gaussian signal model",
    }
