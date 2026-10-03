import math

from alphavalue.cac import (
    binary_kl,
    certified_fraction_converse,
    finite_horizon_threshold,
    first_passage_cdf,
    fixed_power_capacity_upper,
    price_per_nat,
    threshold_capacity_lower,
)


def test_first_passage_finite_horizon_calibration():
    alpha, A = 0.05, 10.0
    h = finite_horizon_threshold(alpha, A)
    assert abs(first_passage_cdf(A, h, -0.5) - alpha) < 1e-10
    assert h < math.log(1 / alpha)  # finite horizon can use a lower boundary


def test_anytime_ville_boundary_is_conservative_at_finite_horizon():
    alpha, A = 0.05, 5.0
    h = math.log(1 / alpha)
    assert first_passage_cdf(A, h, -0.5) < alpha


def test_nonasymptotic_fraction_converse_identity():
    alpha, A = 0.05, 20.0
    r = certified_fraction_converse(alpha, A)
    assert alpha < r < 1.0
    assert abs(A * (1 - r) - 2 * binary_kl(r, alpha)) < 1e-9


def test_threshold_lower_is_below_converse():
    for A in (1.0, 2.0, 5.0, 10.0, 20.0, 50.0):
        lower = threshold_capacity_lower(0.05, A)["capacity_lower"]
        upper = A * certified_fraction_converse(0.05, A)
        assert 0.0 <= lower <= upper + 1e-8 <= A + 1e-8


def test_asymptotic_certification_tax_matches_two_log_alpha():
    alpha, A = 0.05, 1000.0
    # The anytime threshold h=log(1/alpha) has E tau=2h under P1;
    # truncation error is negligible at A=1000.
    lower = threshold_capacity_lower(alpha, A, finite_horizon_calibrated=False)["capacity_lower"]
    tax = A - lower
    target = 2 * math.log(1 / alpha)
    assert abs(tax - target) < 2e-5


def test_fixed_power_barrier_zero_when_information_insufficient():
    alpha, beta = 0.05, 0.10
    required_A = 2 * binary_kl(1 - beta, alpha)
    assert fixed_power_capacity_upper(alpha, beta, required_A * 0.99) == 0.0
    assert fixed_power_capacity_upper(alpha, beta, required_A * 1.01) > 0.0


def test_flat_wait_singularity_but_baseline_price_constant():
    sigma, gamma, theta = 2.0, 3.0, 1.0
    p1 = price_per_nat(sigma, gamma, theta, 0.5)
    p2 = price_per_nat(sigma, gamma, theta, 0.99)
    assert abs(p1["baseline_relative"] - sigma**2 / gamma) < 1e-15
    assert abs(p2["baseline_relative"] - sigma**2 / gamma) < 1e-15
    assert p2["flat_wait"] > 1000 * p1["baseline_relative"]


def test_exact_gaussian_reliable_information_threshold():
    from alphavalue.cac import gaussian_power_envelope, reliable_information_threshold
    alpha, beta = 0.05, 0.10
    crit = reliable_information_threshold(alpha, beta)
    assert abs(gaussian_power_envelope(alpha, crit["A_crit"]) - (1 - beta)) < 1e-12
    assert gaussian_power_envelope(alpha, crit["A_crit"] * 0.99) < 1 - beta
    assert gaussian_power_envelope(alpha, crit["A_crit"] * 1.01) > 1 - beta


def test_reliable_capacity_phase_transition_constructive_lower_bound():
    from alphavalue.cac import fixed_time_reliable_capacity_lower, reliable_information_threshold
    alpha, beta = 0.05, 0.10
    Acrit = reliable_information_threshold(alpha, beta)["A_crit"]
    assert fixed_time_reliable_capacity_lower(alpha, beta, Acrit)["capacity_lower"] == 0.0
    out = fixed_time_reliable_capacity_lower(alpha, beta, Acrit + 2.0)
    assert out["capacity_lower"] > 0.0
    assert out["power"] >= 1 - beta - 1e-12
    assert Acrit <= out["test_time"] < Acrit + 2.0


def test_multivariate_spectrum_recovers_scalar_constant():
    import numpy as np
    from alphavalue.cac import economic_information_spectrum, directional_price_per_nat

    sigma2, gamma = 4.0, 2.0
    Sigma = sigma2 * np.eye(3)
    Gamma = gamma * np.eye(3)
    spec = economic_information_spectrum(Sigma, Gamma)
    assert np.allclose(spec["eigenvalues"], sigma2 / gamma)
    d = np.array([1.0, -2.0, 0.5])
    assert abs(directional_price_per_nat(d, Sigma, Gamma) - sigma2 / gamma) < 1e-12


def test_directional_price_lies_in_exact_spectral_bounds():
    import numpy as np
    from alphavalue.cac import economic_information_spectrum, directional_price_per_nat

    Sigma = np.array([[2.0, 0.35], [0.35, 1.0]])
    Gamma = np.array([[1.5, 0.2], [0.2, 3.0]])
    spec = economic_information_spectrum(Sigma, Gamma)
    for d in (np.array([1.0, 0.0]), np.array([0.0, 1.0]), np.array([1.0, -3.0])):
        r = directional_price_per_nat(d, Sigma, Gamma)
        assert spec["min_price_per_nat"] - 1e-12 <= r <= spec["max_price_per_nat"] + 1e-12
