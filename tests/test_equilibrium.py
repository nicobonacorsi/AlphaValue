import numpy as np

from alphavalue.equilibrium import (
    conditional_cross_section_margin,
    heterogeneous_cross_section_slope,
    strategic_arbitrageability_fraction,
)


def test_strategic_arbitrageability_matches_exact_destroyed_fraction():
    N, c, kappa, a = 7, 0.4, 0.8, 1.7
    m = strategic_arbitrageability_fraction(N, c, kappa, a)
    expected = a * kappa * (1 - 1 / N) / (a * kappa + c)
    assert abs(m - expected) < 1e-14
    assert 0 < m < 1


def test_constant_technology_pooled_slope_equals_arbitrageability():
    x = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    m = np.full_like(x, 0.37)
    out = heterogeneous_cross_section_slope(x, m)
    assert abs(out['slope'] - 0.37) < 1e-14
    assert abs(out['decomposition'] - out['slope']) < 1e-14
    assert abs(out['heterogeneity_term']) < 1e-14


def test_exact_heterogeneity_decomposition():
    x = np.array([0.5, 1.0, 2.0, 4.0, 8.0])
    m = np.array([0.8, 0.6, 0.5, 0.25, 0.1])
    out = heterogeneous_cross_section_slope(x, m)
    assert abs(out['decomposition'] - out['slope']) < 1e-14


def test_simpson_reversal_is_possible_under_technology_heterogeneity():
    # Every individual technology has D=mX with m>0, but sorting high-X
    # opportunities into sufficiently low-arbitrageability technologies can
    # reverse the pooled cross-sectional comparative static.
    x = np.array([1.0, 2.0])
    m = np.array([1.0, 0.1])
    out = heterogeneous_cross_section_slope(x, m)
    assert out['slope'] < 0


def test_inverse_elasticity_boundary():
    # mubar(x)=K/x has derivative -K/x^2, so x*mubar(x) is locally flat.
    x, K = 4.0, 3.0
    mu = K / x
    dmu = -K / (x * x)
    assert abs(conditional_cross_section_margin(x, mu, dmu)) < 1e-14
    # A slower decline gives positive conditional decay; faster gives negative.
    assert conditional_cross_section_margin(x, mu, 0.5 * dmu) > 0
    assert conditional_cross_section_margin(x, mu, 1.5 * dmu) < 0
