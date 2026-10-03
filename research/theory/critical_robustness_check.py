"""Deterministic high-precision checks for critical_robustness.tex.

Run with Python and mpmath installed. No external data are downloaded.
The script checks identities and margins, not optimal stopping numerics.
"""
from pathlib import Path
import csv
import json
import mpmath as mp

mp.mp.dps = 80
ROOT = Path(__file__).resolve().parent


def Phi(x):
    return mp.erfc(-x / mp.sqrt(2)) / 2


def phi(x):
    return mp.exp(-x*x/2) / mp.sqrt(2*mp.pi)


def z(u):
    return mp.sqrt(2) * mp.erfinv(2*u-1)


def frontier(alpha, power):
    return (z(1-alpha)+z(power))**2


def one_contract(alpha, power):
    alpha, power = mp.mpf(alpha), mp.mpf(power)
    a, b = z(1-alpha), z(power)
    s = a+b
    Astar = s*s
    K = 2*s*(1/phi(a)+1/phi(b))
    H = 2*s*(power-alpha)/phi(b)
    rows = []
    worst_residual = mp.mpf(0)
    for estr in ["1e-2", "1e-3", "1e-4", "1e-5", "1e-6"]:
        eps = mp.mpf(estr)
        A = Astar+eps
        f = lambda tau: frontier(alpha-tau, power+tau)-A
        tau = mp.findroot(f, (eps/K/2, eps/K*mp.mpf("1.1")), solver="secant")
        assert 0 < tau < min(alpha, 1-power)
        terminal_power = Phi(mp.sqrt(A)-a)
        rho = (terminal_power-power)/(terminal_power-alpha)
        mixture_power_target = (power-rho*alpha)/(1-rho)
        residuals = [
            abs(f(tau)),
            abs(Phi(mp.sqrt(A)-z(1-alpha+tau))-(power+tau)),
            abs(frontier(alpha, mixture_power_target)-A),
            abs((1-rho)*terminal_power+rho*alpha-power),
        ]
        worst_residual = max(worst_residual, *residuals)
        assert max(residuals) < mp.mpf("1e-65")
        # A margin just beyond either frontier must be infeasible.
        assert f(tau*mp.mpf("1.001")) > 0
        rho_over = rho*mp.mpf("1.001")
        assert (1-rho_over)*terminal_power+rho_over*alpha < power
        # Initial policy mixing satisfies the tighter size/power exactly
        # in the binding case, for three distinct weights.
        for w in [mp.mpf(".1"), mp.mpf(".5"), mp.mpf(".9")]:
            tau_used = w*tau
            assert abs((1-w)*alpha+w*(alpha-tau)-(alpha-tau_used)) < mp.mpf("1e-70")
            assert abs((1-w)*power+w*(power+tau)-(power+tau_used)) < mp.mpf("1e-70")
        rows.append({
            "alpha": mp.nstr(alpha, 30), "power": mp.nstr(power, 30),
            "epsilon": estr, "tau_max": mp.nstr(tau, 50),
            "xi_max_for_double_tightening": mp.nstr(tau/2, 50),
            "rho_max": mp.nstr(rho, 50),
            "tau_over_linear_approx": mp.nstr(tau/(eps/K), 30),
            "rho_over_linear_approx": mp.nstr(rho/(eps/H), 30),
        })
    return rows, {
        "alpha": str(alpha), "power": str(power),
        "A_star": mp.nstr(Astar, 60), "K": mp.nstr(K, 60),
        "H": mp.nstr(H, 60),
        "max_identity_residual": mp.nstr(worst_residual, 20),
    }


def main():
    rows, summaries = [], []
    # Covers ordinary, multiple-testing, and low-power interior contracts.
    for alpha, power in [(".05", ".9"), (mp.mpf(".05")/101, ".9"), (".1", ".2")]:
        r, s = one_contract(alpha, power)
        rows.extend(r)
        summaries.append(s)
    with (ROOT / "critical_robustness_numbers.csv").open("w", newline="") as out:
        writer = csv.DictWriter(out, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)
    metadata = {
        "precision_decimal_digits": mp.mp.dps,
        "contracts": summaries,
        "checked_horizons": len(rows),
        "checks": ["frontier inversion", "tightened terminal power", "mixture frontier",
                   "mixture power", "beyond-frontier infeasibility", "policy mixing margins"],
        "scope": "Numerical identity checks only; no optimal stopping solver or empirical data.",
    }
    (ROOT / "critical_robustness_checks.json").write_text(json.dumps(metadata, indent=2)+"\n")
    print(json.dumps(metadata, indent=2))


if __name__ == "__main__":
    main()
