#!/usr/bin/env python3
"""Numerical lower-policy search; no claim of globally optimal capacity.

Search a finite family: early upper-threshold deployment at t=r*A; otherwise
use a calibrated terminal upper threshold. The terminal decision counts toward
power but earns zero post-certification reward. All event probabilities use
one-dimensional positive Gaussian integrals. SciPy error estimates and an
independent mpmath check are numerical evidence, not directed-rounding bounds.
"""
from pathlib import Path
import csv
import json
import math

from scipy.integrate import quad
from scipy.optimize import brentq
from scipy.special import ndtr
from scipy.stats import norm


ALPHA = 0.05
POWER = 0.90
FRACTIONS = (0.5, 0.8, 0.9, 0.95, 0.98)
EPSILONS = (1e-1, 1e-2, 1e-3, 1e-4, 1e-5, 1e-6)
ROOT = Path(__file__).resolve().parent


def solve_policy(eps, fraction):
    critical = (norm.isf(ALPHA) + norm.ppf(POWER)) ** 2
    total = critical + eps
    early = fraction * total
    lead = total - early
    sd = math.sqrt(total)
    bridge_sd = math.sqrt(early * (1.0 - early / total))
    size_target = ALPHA - 1e-4 * eps
    power_target = POWER + 1e-4 * eps
    terminal_np = -total / 2 + sd * norm.isf(size_target)
    integration_errors = []

    def cross(terminal, threshold, drift):
        """P_drift(Z_early>=threshold, Z_total<terminal)."""
        cutoff = (terminal - drift * total) / sd

        def integrand(x):
            z = drift * total + sd * x
            conditional = ndtr((early / total * z - threshold) / bridge_sd)
            return math.exp(-x * x / 2) / math.sqrt(2 * math.pi) * conditional

        val, error = quad(integrand, -math.inf, cutoff,
                          epsabs=2e-14, epsrel=2e-12, limit=250)
        integration_errors.append(error)
        return val

    def terminal_for(early_threshold):
        def size_residual(terminal):
            return (norm.sf((terminal + total / 2) / sd)
                    + cross(terminal, early_threshold, -0.5) - size_target)
        # The root approaches the terminal NP threshold as the early event
        # vanishes. Avoid a floating-point sign failure at that endpoint.
        if cross(terminal_np, early_threshold, -0.5) < 2e-17:
            return terminal_np
        low = terminal_np - 1e-12
        high = terminal_np + 20 * sd
        while size_residual(high) > 0:
            high += 20 * sd
        return brentq(size_residual, low, high, xtol=1e-12, rtol=1e-14)

    def power_for(early_threshold):
        terminal = terminal_for(early_threshold)
        return (norm.sf((terminal - total / 2) / sd)
                + cross(terminal, early_threshold, 0.5))

    low = -early / 2 + math.sqrt(early) * norm.isf(size_target) + 1e-5
    high = max(low + 10, 10)
    while power_for(high) < power_target:
        high += 10
    early_threshold = brentq(lambda b: power_for(b) - power_target,
                            low, high, xtol=1e-11, rtol=1e-13)
    terminal = terminal_for(early_threshold)
    p0 = norm.sf((terminal + total / 2) / sd) + cross(terminal, early_threshold, -0.5)
    p1 = norm.sf((terminal - total / 2) / sd) + cross(terminal, early_threshold, 0.5)
    early_power = norm.sf((early_threshold - early / 2) / math.sqrt(early))
    capacity = lead * early_power
    return dict(epsilon=eps, early_fraction=fraction, A_crit=critical, A=total,
                early_time=early, early_threshold=early_threshold,
                terminal_threshold=terminal, size_target=size_target,
                power_target=power_target, size=p0, power=p1,
                size_margin=ALPHA-p0, power_margin=p1-POWER,
                early_deployment_probability=early_power,
                policy_capacity=capacity,
                scipy_max_quadrature_error_estimate=max(integration_errors))


def independent_mpmath_check(row):
    import mpmath as mp
    mp.mp.dps = 60
    conv = lambda x: mp.mpf(str(x))
    A = conv(row['A']); t = conv(row['early_time'])
    b = conv(row['early_threshold']); k = conv(row['terminal_threshold'])
    sd = mp.sqrt(A); bridge = mp.sqrt(t*(1-t/A))
    sf = lambda x: mp.erfc(x/mp.sqrt(2))/2
    phi = lambda x: mp.exp(-x*x/2)/mp.sqrt(2*mp.pi)
    def probability(drift):
        cutoff = (k-drift*A)/sd
        def integrand(x):
            z = drift*A + sd*x
            return phi(x)*sf((b-t/A*z)/bridge)
        cross = mp.quad(integrand, [-mp.inf, min(mp.mpf(-2),cutoff-1), cutoff])
        return sf(cutoff)+cross
    p0 = probability(mp.mpf('-.5')); p1 = probability(mp.mpf('.5'))
    value = (A-t)*sf((b-t/2)/mp.sqrt(t))
    return dict(method='independent mpmath positive Gaussian integrals',
                decimal_precision=mp.mp.dps,
                rigorous_interval_arithmetic=False,
                input_interpretation='policy thresholds interpreted as exact decimal strings in CSV',
                epsilon=row['epsilon'], early_fraction=row['early_fraction'],
                size=mp.nstr(p0,55), power=mp.nstr(p1,55),
                size_margin=mp.nstr(mp.mpf('.05')-p0,55),
                power_margin=mp.nstr(p1-mp.mpf('.9'),55),
                policy_capacity=mp.nstr(value,55),
                agreement_size=mp.nstr(p0-conv(row['size']),25),
                agreement_power=mp.nstr(p1-conv(row['power']),25),
                agreement_capacity=mp.nstr(value-conv(row['policy_capacity']),25))


def main():
    rows=[solve_policy(eps,r) for eps in EPSILONS for r in FRACTIONS]
    with (ROOT/'two_look_results.csv').open('w',newline='') as stream:
        writer=csv.DictWriter(stream,fieldnames=list(rows[0]))
        writer.writeheader();writer.writerows(rows)
    final=max((r for r in rows if r['epsilon']==1e-6),
              key=lambda r:r['policy_capacity'])
    check=independent_mpmath_check(final)
    (ROOT/'two_look_high_precision_check.json').write_text(json.dumps(check,indent=2)+'\n')
    for eps in EPSILONS:
        best=max((r for r in rows if r['epsilon']==eps),key=lambda r:r['policy_capacity'])
        print(f"epsilon={eps:g}, best grid fraction={best['early_fraction']}, "
              f"policy value={best['policy_capacity']:.12g}")
    print(json.dumps(check,indent=2))


if __name__=='__main__':
    main()
