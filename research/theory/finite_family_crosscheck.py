"""Independent Gaussian-family check using Gauss--Hermite quadrature.

This does not optimize sequential capacity. It independently evaluates the
two-profile terminal minimax test and checks the ordered least-profile case.
"""
from pathlib import Path
import json
import numpy as np
from scipy.optimize import brentq
from scipy.special import roots_hermitenorm
from scipy.stats import norm


def integral_exp(rate, horizon):
    return -np.expm1(-rate * horizon) / rate


def hermite_power(information_time, correlation, alpha, n_nodes):
    nodes, weights = roots_hermitenorm(n_nodes)
    weights = weights / np.sqrt(2 * np.pi)
    a = np.sqrt(information_time * (1 + correlation) / 2)
    b = np.sqrt(information_time * (1 - correlation) / 2)

    def tail(c, alternative=False):
        d = nodes + (b if alternative else 0)
        logcosh = np.logaddexp(b*d, -b*d) - np.log(2)
        cutoff = (c + information_time / 2 - logcosh) / a
        return np.dot(weights, norm.sf(cutoff - (a if alternative else 0)))

    cutoff = brentq(lambda c: tail(c) - alpha, -30, 30, xtol=1e-13)
    return float(tail(cutoff, True)), float(tail(cutoff)), float(cutoff)


def main():
    alpha, target, horizon = .05, .90, 5.
    rates = (.2, 2.)
    A0 = float((norm.isf(alpha) + norm.ppf(target))**2)
    corr = float(integral_exp(sum(rates), horizon) / np.sqrt(
        integral_exp(2*rates[0], horizon) * integral_exp(2*rates[1], horizon)))
    evaluations = []
    for nodes in (64, 128, 256):
        power, size, cutoff = hermite_power(A0, corr, alpha, nodes)
        evaluations.append({"nodes": nodes, "power": power,
                            "size": size, "log_threshold": cutoff})
    assert abs(evaluations[-1]["power"] - evaluations[-2]["power"]) < 1e-9
    robust_A = brentq(lambda A: hermite_power(A, corr, alpha, 256)[0] - target,
                     A0, 2*A0, xtol=1e-11)
    reference_path = Path(__file__).parents[1] / "experiments" / "gaussian_two_profile_results.json"
    reference = json.loads(reference_path.read_text())
    reference_power = next(row["power"] for row in reference["rows"]
                           if row["individual_CR"] == 1)
    power_difference = evaluations[-1]["power"] - reference_power
    A_difference = robust_A - reference["robust_critical_A"]
    assert abs(power_difference) < 1e-9
    assert abs(A_difference) < 1e-8

    # Same amplitude across decay rates: the fastest profile is pointwise least.
    theta = np.sqrt(A0 / integral_exp(2*rates[1], horizon))
    ordered_rows = []
    for rate in (.2, .6, 1., 2.):
        noncentrality = theta**2 * integral_exp(rate + rates[1], horizon) / np.sqrt(A0)
        power = norm.cdf(noncentrality - norm.isf(alpha))
        assert noncentrality >= np.sqrt(A0) - 1e-12
        ordered_rows.append({"decay": rate, "amplitude": float(theta),
                             "least_template_test_power": float(power)})
    assert abs(ordered_rows[-1]["least_template_test_power"] - target) < 1e-12
    output = {
        "scope": "Terminal Gaussian tests only; no empirical backtest or sequential-capacity optimizer.",
        "critical_A": A0,
        "template_correlation": corr,
        "nonordered_hermite_evaluations": evaluations,
        "nonordered_robust_critical_A": float(robust_A),
        "difference_from_adaptive_quadrature_power": float(power_difference),
        "difference_from_adaptive_quadrature_critical_A": float(A_difference),
        "ordered_fixed_amplitude_members": ordered_rows,
        "all_checks_passed": True,
        "numerical_status": "Floating-point quadrature agreement; not interval arithmetic."
    }
    path = Path(__file__).with_suffix(".json")
    path.write_text(json.dumps(output, indent=2) + "\n")
    print(json.dumps(output, indent=2))


if __name__ == "__main__":
    main()
