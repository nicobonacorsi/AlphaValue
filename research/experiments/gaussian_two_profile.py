"""Exact Gaussian two-profile terminal benchmark (deterministic quadrature).

Both drift templates have the same L2 norm. Swapping the two templates is an
orthogonal symmetry preserving the null; therefore the equal-mixture NP test
has equal power at both alternatives and is minimax among level-alpha tests.
This is a model calculation, not a backtest or a claim of a new testing theorem.
"""
from pathlib import Path
import json
import numpy as np
from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.stats import norm


def integral_exp(rate, horizon):
    return -np.expm1(-rate * horizon) / rate


def profile_correlation(lam1, lam2, horizon):
    return integral_exp(lam1 + lam2, horizon) / np.sqrt(
        integral_exp(2 * lam1, horizon) * integral_exp(2 * lam2, horizon)
    )


def minimax_power(amplitude_norm, correlation, alpha=0.05, tol=2e-11):
    """Return power and size of the symmetric mixture likelihood test.

Under the null, S,D are independent N(0,1). Under alternative one their
means are a,b, where a=A sqrt((1+r)/2), b=A sqrt((1-r)/2).
The log mixture LR is a*S-A**2/2+log(cosh(b*D)).
"""
    if not -1 < correlation <= 1:
        raise ValueError("correlation outside the admitted range")
    if correlation == 1:
        return {"size": alpha, "power": float(norm.cdf(amplitude_norm - norm.isf(alpha))),
                "log_threshold": float(amplitude_norm * norm.isf(alpha) - amplitude_norm**2/2)}
    a = amplitude_norm * np.sqrt((1 + correlation)/2)
    b = amplitude_norm * np.sqrt((1 - correlation)/2)
    def tail(log_k, alternative=False):
        mean_d = b if alternative else 0.0
        mean_s = a if alternative else 0.0
        def integrand(centered_d):
            d = centered_d + mean_d
            log_cosh = np.logaddexp(b*d, -b*d) - np.log(2)
            s_cut = (log_k + amplitude_norm**2/2 - log_cosh)/a
            return norm.pdf(centered_d) * norm.sf(s_cut - mean_s)
        # Missing standard-Gaussian mass is < 4e-33 on [-12,12].
        return quad(integrand, -12, 12, epsabs=tol, epsrel=tol, limit=200)[0]
    threshold = brentq(lambda k: tail(k) - alpha, -100, 100, xtol=tol)
    return {"size": float(tail(threshold)), "power": float(tail(threshold, True)),
            "log_threshold": float(threshold)}


def main():
    alpha, target = 0.05, 0.90
    horizon, lambdas = 5.0, (0.2, 2.0)
    rho = float(profile_correlation(*lambdas, horizon))
    norm_crit = float(norm.isf(alpha) + norm.ppf(target))
    rows = []
    for cr in (0.8, 1.0, 1.05, 1.1, 1.2, 1.5):
        n = norm_crit * np.sqrt(cr)
        result = minimax_power(n, rho, alpha)
        rows.append({"individual_CR": cr, "known_profile_power": float(norm.cdf(n - norm.isf(alpha))), **result})
    robust_norm = brentq(lambda n: minimax_power(n, rho, alpha)["power"] - target,
                         norm_crit, 2*norm_crit, xtol=2e-10)
    repeat = minimax_power(norm_crit, rho, alpha, tol=2e-13)
    at_critical = rows[1]
    assert abs(at_critical["size"] - alpha) < 1e-9
    assert at_critical["power"] < target
    assert abs(repeat["power"] - at_critical["power"]) < 1e-9
    assert abs(minimax_power(norm_crit, 1.0, alpha)["power"] - target) < 1e-12
    assert abs(minimax_power(norm_crit, 0.999999, alpha)["power"] - target) < 1e-6
    payload = {
        "contract": "Full Gaussian observation; null zero; two positive nonordered exponential drift templates; same individual L2 information; uniform alternative power.",
        "alpha": alpha, "target_power": target, "horizon": horizon,
        "decays": lambdas, "template_correlation": rho,
        "individual_critical_A": norm_crit**2,
        "critical_template_amplitudes": [float(norm_crit/np.sqrt(integral_exp(2*l,horizon))) for l in lambdas],
        "robust_critical_A": float(robust_norm**2),
        "robust_to_individual_information_ratio": float((robust_norm/norm_crit)**2),
        "power_shortfall_at_individual_CR1": float(target - at_critical["power"]),
        "quadrature_repeat_power_difference": float(repeat["power"] - at_critical["power"]),
        "rows": rows,
        "interpretation": "A geometry effect of unknown alternative profile. Neither economic P&L nor evidence that CAC beats a forecasting model. Minimax optimality follows by symmetry and the mixture NP lemma; numeric values use floating-point quadrature, not interval arithmetic."
    }
    out=Path(__file__).with_name("gaussian_two_profile_results.json")
    out.write_text(json.dumps(payload, indent=2)+"\n")
    print(json.dumps(payload, indent=2))


if __name__ == "__main__":
    main()
