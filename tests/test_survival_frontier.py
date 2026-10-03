import math

import numpy as np
from scipy.integrate import quad

from alphavalue.cac import (
    certifiability_ratio,
    exponential_lifetime_information,
    exponential_remaining_information_from_death,
    minimum_exponential_half_life,
    multiplicity_adjusted_alpha,
    multiplicity_reliable_information_threshold,
    polynomial_terminal_economic_value,
    polynomial_terminal_information,
    polynomial_uncertifiable_terminal_layer,
    search_breadth_capacity,
)


def test_exponential_positive_hurdle_closed_form_matches_quadrature():
    mu0, b, sigma, lam = 1.7, 0.6, 1.3, 0.4
    out = exponential_lifetime_information(mu0, sigma, lam, hurdle=b)
    T = math.log(mu0 / b) / lam
    direct = quad(lambda t: (mu0 * math.exp(-lam*t) - b)**2 / (2*sigma*sigma), 0, T)[0]
    assert abs(out["economic_death_time"] - T) < 1e-13
    assert abs(out["I_life_nats"] - direct) < 1e-12


def test_exponential_zero_hurdle_finite_information_despite_infinite_calendar_life():
    mu0, sigma, lam = 2.0, 1.5, 0.25
    out = exponential_lifetime_information(mu0, sigma, lam, hurdle=0.0)
    expected = mu0**2 / (4*sigma**2*lam)
    assert math.isinf(out["economic_death_time"])
    assert abs(out["I_life_nats"] - expected) < 1e-13


def test_cubic_certifiability_collapse_at_positive_exponential_hurdle():
    b, sigma, lam = 0.8, 1.2, 0.35
    coeff = b*b*lam*lam/(6*sigma*sigma)
    for delta in (1e-1, 5e-2, 1e-2, 5e-3):
        exact = exponential_remaining_information_from_death(delta, b, sigma, lam)
        approx = coeff * delta**3
        # Relative error vanishes linearly with delta.
        assert abs(exact/approx - 1.0) < 0.08


def test_general_2p_plus_1_terminal_law_and_value_information_ratio():
    Sigma = np.array([[2.0, 0.2], [0.2, 1.1]])
    Gamma = np.array([[1.4, 0.1], [0.1, 2.5]])
    c = np.array([0.7, -1.2])
    delta, p = 0.3, 2.0
    I = polynomial_terminal_information(delta, c, Sigma, p)
    V = polynomial_terminal_economic_value(delta, c, Gamma, p)
    qI = c @ np.linalg.solve(Sigma, c)
    qV = c @ np.linalg.solve(Gamma, c)
    assert abs(I - 0.5*qI*delta**5/5.0) < 1e-14
    assert abs(V/I - qV/qI) < 1e-13


def test_uncertifiable_terminal_layer_inverts_information_exactly():
    Sigma = np.array([[1.7]])
    c = np.array([0.9])
    p, Icrit = 1.0, 4.2819236753
    delta = polynomial_uncertifiable_terminal_layer(Icrit, c, Sigma, p)
    recovered = polynomial_terminal_information(delta, c, Sigma, p)
    assert abs(recovered - Icrit) < 1e-11


def test_multiplicity_half_life_frontier_reproduces_reference_numbers():
    alpha, beta = 0.05, 0.10
    h1 = minimum_exponential_half_life(1.0, alpha, beta, multiplicity=1)
    h100 = minimum_exponential_half_life(1.0, alpha, beta, multiplicity=100)
    assert abs(h1["half_life_min"] - 11.872) < 0.01
    assert abs(h100["half_life_min"] - 28.98) < 0.03
    # 1/S^2 scaling.
    h2 = minimum_exponential_half_life(2.0, alpha, beta, multiplicity=100)
    assert abs(h2["half_life_min"] - h100["half_life_min"]/4.0) < 1e-12


def test_bonferroni_and_sidak_are_declared_distinct_allocations():
    alpha = 0.05
    b = multiplicity_adjusted_alpha(alpha, 100, method="bonferroni")
    s = multiplicity_adjusted_alpha(alpha, 100, method="sidak")
    assert abs(b - 0.0005) < 1e-15
    assert s > b


def test_certifiability_ratio_phase_labels():
    thr = multiplicity_reliable_information_threshold(0.05, 0.10, 1)
    Icrit = thr["I_crit_nats"]
    assert certifiability_ratio(0.99*Icrit, 0.05, 0.10)["status"] == "INFEASIBLE"
    assert certifiability_ratio(Icrit, 0.05, 0.10)["status"] == "BOUNDARY"
    assert certifiability_ratio(1.01*Icrit, 0.05, 0.10)["status"] == "FEASIBLE"


def test_search_breadth_capacity_is_exact_inverse_of_bonferroni_frontier():
    alpha, beta = 0.05, 0.10
    for M in (1, 10, 100, 1000):
        thr = multiplicity_reliable_information_threshold(alpha, beta, M, method="bonferroni")
        cap = search_breadth_capacity(thr["I_crit_nats"], alpha, beta, method="bonferroni")
        assert cap["max_candidates"] >= M
        if M > 1:
            prev = search_breadth_capacity(thr["I_crit_nats"]*(1-1e-10), alpha, beta, method="bonferroni")
            assert prev["max_candidates"] <= M


def test_minimum_type1_budget_inverts_information_threshold():
    from alphavalue.cac import minimum_type1_budget_for_information, reliable_information_threshold
    beta = 0.10
    for alpha in (0.05, 0.01, 0.001):
        I = reliable_information_threshold(alpha, beta)["I_crit_nats"]
        amin = minimum_type1_budget_for_information(I, beta)
        assert abs(amin - alpha) < 2e-14


def test_heterogeneous_bonferroni_frontier_exact_cardinality_rule():
    from alphavalue.cac import heterogeneous_bonferroni_frontier
    # Construct four information budgets whose required marginal alpha budgets
    # differ materially. The exact max-cardinality subset is the cheapest prefix.
    I = np.array([4.3, 7.5, 10.5, 14.0])
    out = heterogeneous_bonferroni_frontier(I, 0.05, 0.10)
    b = out["minimum_type1_budgets"]
    order = np.argsort(b)
    csum = np.cumsum(b[order])
    expected = int(np.searchsorted(csum, 0.05, side="right"))
    assert out["max_cardinality"] == expected
    assert np.array_equal(out["max_cardinality_indices"], order[:expected])
    assert out["budget_used_max_cardinality"] <= 0.05 + 1e-15
