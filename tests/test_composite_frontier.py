import math

import numpy as np
from scipy.stats import nct, t as student_t
import pytest

from alphavalue.cac import (
    ar1_expected_design_energy,
    binary_kl,
    composite_g2_survival_frontier,
    g2_expected_kl_bounds,
    minimum_horizon_to_clear_g2_kl_barrier,
    minimum_type1_budget_from_binary_kl,
    student_t_reliable_information_threshold,
)


def _rect(theta=(0.4, 0.8), phi=(0.2, 0.7), vr=(0.8, 1.4), vx=(0.3, 0.9)):
    return {
        "theta": list(theta),
        "phi": list(phi),
        "return_noise_variance": list(vr),
        "signal_innovation_variance": list(vx),
    }


def test_ar1_expected_design_energy_matches_direct_second_moment_sum():
    x, phi, vx, H = 1.3, -0.72, 0.41, 17
    r = phi * phi
    direct = 0.0
    for k in range(H):
        mean2 = x*x*(r**k)
        var = vx * sum(r**j for j in range(k))
        direct += mean2 + var
    got = ar1_expected_design_energy(x, phi, vx, H)
    assert abs(got - direct) < 1e-12


def test_expected_design_energy_monotone_in_abs_phi_and_signal_variance():
    x, H = 0.8, 30
    q1 = ar1_expected_design_energy(x, 0.1, 0.2, H)
    q2 = ar1_expected_design_energy(x, -0.6, 0.2, H)
    q3 = ar1_expected_design_energy(x, -0.6, 0.7, H)
    assert q1 <= q2 <= q3


def test_g2_expected_kl_rectangle_bounds_are_exact_against_dense_grid():
    rect = _rect(theta=(0.35, 0.75), phi=(-0.8, 0.55), vr=(0.7, 1.3), vx=(0.15, 0.8))
    x, H, b = 0.6, 12, 0.1
    out = g2_expected_kl_bounds(rect, x, H, economic_boundary=b)
    vals = []
    for th in np.linspace(*rect["theta"], 31):
        for ph in np.linspace(*rect["phi"], 81):
            for vr in rect["return_noise_variance"]:
                for vx in rect["signal_innovation_variance"]:
                    q = ar1_expected_design_energy(x, ph, vx, H)
                    vals.append((th-b)**2*q/(2*vr))
    assert out["expected_kl_lower_nats"] <= min(vals) + 1e-11
    assert out["expected_kl_upper_nats"] >= max(vals) - 1e-11
    # The exact lower corner is phi=0 because the interval crosses zero; the
    # dense grid above need not hit zero exactly.  Check both analytic corners.
    qlo = ar1_expected_design_energy(x, 0.0, rect["signal_innovation_variance"][0], H)
    exact_lo = (rect["theta"][0]-b)**2*qlo/(2*rect["return_noise_variance"][1])
    qhi = ar1_expected_design_energy(x, -0.8, rect["signal_innovation_variance"][1], H)
    exact_hi = (rect["theta"][1]-b)**2*qhi/(2*rect["return_noise_variance"][0])
    assert abs(out["expected_kl_lower_nats"] - exact_lo) < 1e-12
    assert abs(out["expected_kl_upper_nats"] - exact_hi) < 1e-12


def test_g2_floor_is_zero_when_loading_rectangle_touches_boundary():
    out = g2_expected_kl_bounds(_rect(theta=(-0.1, 0.5)), 1.0, 24, economic_boundary=0.0)
    assert out["expected_kl_lower_nats"] == 0.0
    assert out["uniform_alternative_separation"] is False


def test_composite_status_three_zones():
    alpha, beta = 0.05, 0.10
    weak = _rect(theta=(0.02, 0.03), phi=(0.0, 0.1), vr=(1.0, 1.1), vx=(0.01, 0.02))
    a = composite_g2_survival_frontier(weak, 0.0, 2, alpha, beta)
    assert a["status"] == "ALL_COMPATIBLE_WORLDS_KL_INSUFFICIENT"

    strong = _rect(theta=(1.5, 2.0), phi=(0.7, 0.9), vr=(0.7, 0.9), vx=(0.8, 1.0))
    b = composite_g2_survival_frontier(strong, 1.0, 30, alpha, beta)
    assert b["status"] == "PAIRWISE_KL_BARRIER_CLEARED_UNIFORMLY"

    mixed = _rect(theta=(-0.05, 1.5), phi=(0.4, 0.8), vr=(0.7, 1.0), vx=(0.5, 0.9))
    c = composite_g2_survival_frontier(mixed, 1.0, 20, alpha, beta)
    assert c["status"] == "MIXED_OR_UNRESOLVED"


def test_minimum_horizon_finds_first_integer_clearing_pairwise_barrier():
    rect = _rect(theta=(0.5, 0.7), phi=(0.3, 0.6), vr=(0.9, 1.1), vx=(0.4, 0.6))
    out = minimum_horizon_to_clear_g2_kl_barrier(rect, 0.5, 0.05, 0.10, max_horizon=10000)
    H = out["minimum_horizon"]
    assert H is not None and H >= 1
    now = composite_g2_survival_frontier(rect, 0.5, H, 0.05, 0.10)
    assert now["expected_kl_lower_nats"] >= now["binary_kl_threshold_nats"]
    if H > 1:
        prev = composite_g2_survival_frontier(rect, 0.5, H-1, 0.05, 0.10)
        assert prev["expected_kl_lower_nats"] < prev["binary_kl_threshold_nats"]


def test_minimum_horizon_is_undefined_if_current_rectangle_contains_boundary_world():
    out = minimum_horizon_to_clear_g2_kl_barrier(
        _rect(theta=(-0.2, 0.7)), 1.0, 0.05, 0.10, max_horizon=100
    )
    assert out["minimum_horizon"] is None
    assert out["reason"] == "rectangle_touches_or_crosses_economic_boundary"


def test_binary_kl_type1_budget_is_exact_inverse():
    beta = 0.10
    p = 1-beta
    for I in (0.1, 1.0, 2.376205, 6.0, 12.0):
        q = minimum_type1_budget_from_binary_kl(I, beta)
        assert 0.0 < q < p
        assert abs(binary_kl(p, q) - I) < 1e-10


def test_student_t_frontier_has_exact_target_power_and_converges_to_z_frontier():
    alpha, beta = 0.05, 0.10
    small = student_t_reliable_information_threshold(alpha, beta, 11)
    critical = student_t.ppf(1-alpha, 11)
    power = nct.sf(critical, 11, small["noncentrality_crit"])
    assert abs(power - (1-beta)) < 1e-11
    assert small["finite_sample_unknown_scale_penalty"] > 1.0

    large = student_t_reliable_information_threshold(alpha, beta, 10000)
    assert abs(large["finite_sample_unknown_scale_penalty"] - 1.0) < 1e-3


def test_student_t_reference_penalties_are_stable():
    alpha, beta = 0.05, 0.10
    ref = {
        11: 4.8888418554,   # n=12 => df=11
        23: 4.5511038669,   # n=24
        59: 4.3826528782,   # n=60
        119: 4.3312208245,  # n=120
    }
    for df, Iref in ref.items():
        got = student_t_reliable_information_threshold(alpha, beta, df)["I_equivalent_nats"]
        assert abs(got - Iref) < 2e-9


def test_stationary_kl_rate_bounds_match_closed_form_corners():
    from alphavalue.cac import g2_stationary_kl_rate_bounds
    rect=_rect(theta=(0.4,0.9),phi=(-0.5,0.8),vr=(0.6,1.4),vx=(0.2,0.7))
    out=g2_stationary_kl_rate_bounds(rect,economic_boundary=0.1)
    lo=(0.4-0.1)**2*0.2/(2*1.4)  # min |phi|=0
    hi=(0.9-0.1)**2*0.7/(2*0.6*(1-0.8**2))
    assert abs(out["kl_rate_lower_nats_per_observation"]-lo)<1e-14
    assert abs(out["kl_rate_upper_nats_per_observation"]-hi)<1e-13


def test_stationary_minimum_horizon_is_closed_form_first_integer():
    from alphavalue.cac import minimum_stationary_horizon_to_clear_g2_kl_barrier
    rect=_rect(theta=(0.5,0.8),phi=(0.2,0.6),vr=(0.9,1.1),vx=(0.3,0.6))
    out=minimum_stationary_horizon_to_clear_g2_kl_barrier(rect,0.05,0.10)
    H=out["minimum_horizon"]
    assert H is not None
    k=out["kl_rate_lower_nats_per_observation"]
    K=out["binary_kl_threshold_nats"]
    assert H*k>=K-1e-13
    if H>1:
        assert (H-1)*k<K


def test_discrete_information_economic_identity_exactly_matches_regret_formula():
    from alphavalue.cac import g2_discrete_information_economic_identity
    out=g2_discrete_information_economic_identity(
        0.8,0.2,0.65,0.12,0.18,gamma=1.3,trading_cost=0.7,horizon=15
    )
    assert abs(out["identity_residual"])<1e-14
    assert out["economic_boundary_regret"]>0
    assert out["statistical_information_nats"]>0


def test_discrete_price_per_nat_reduces_to_variance_over_gamma_without_costs():
    from alphavalue.cac import g2_discrete_information_economic_identity
    vR,gamma=0.17,1.4
    for phi in (-0.8,0.0,0.75):
        out=g2_discrete_information_economic_identity(
            1.0,0.3,phi,vR,0.2,gamma=gamma,trading_cost=0.0,horizon=20
        )
        assert abs(out["price_per_nat"]-vR/gamma)<1e-14


def test_stationary_t_power_bound_is_genuine_lower_bound_against_monte_carlo_reference():
    # Monte Carlo is seeded and used only as a regression sanity check with a
    # generous tolerance; theorem validity comes from the chi-square event
    # argument, not from simulation.
    from alphavalue.cac import g2_stationary_t_power_lower_bound
    rng=np.random.default_rng(2718)
    rect=_rect(theta=(0.55,0.8),phi=(-0.4,0.65),vr=(0.8,1.2),vx=(0.25,0.7))
    H=24; alpha=0.05
    out=g2_stationary_t_power_lower_bound(rect,H,alpha,eta_grid_size=101)
    assert 0.0 <= out["power_lower_bound"] <= 1.0
    # Simulate the worst corner used by the spectral bound.  Its actual power
    # should exceed our conservative analytical lower bound.
    theta=rect["theta"][0]; phi=0.65; vR=rect["return_noise_variance"][1]; vX=rect["signal_innovation_variance"][0]
    nmc=20000
    x=np.empty((nmc,H))
    x[:,0]=rng.normal(scale=math.sqrt(vX/(1-phi*phi)),size=nmc)
    for j in range(1,H):
        x[:,j]=phi*x[:,j-1]+rng.normal(scale=math.sqrt(vX),size=nmc)
    eps=rng.normal(scale=math.sqrt(vR),size=(nmc,H))
    y=theta*x+eps
    q=np.sum(x*x,axis=1)
    bh=np.sum(x*y,axis=1)/q
    rss=np.sum((y-bh[:,None]*x)**2,axis=1)
    shat=np.sqrt(rss/(H-1))
    T=bh*np.sqrt(q)/shat
    crit=student_t.ppf(1-alpha,H-1)
    mc=float(np.mean(T>crit))
    assert mc + 0.015 >= out["power_lower_bound"]


def test_t_power_lower_bound_increases_enough_for_strong_rect_and_yields_sufficient_horizon():
    from alphavalue.cac import minimum_stationary_horizon_for_g2_t_power
    rect=_rect(theta=(0.8,1.0),phi=(0.1,0.4),vr=(0.8,1.0),vx=(0.6,0.9))
    out=minimum_stationary_horizon_for_g2_t_power(rect,0.05,0.10,max_horizon=300,eta_grid_size=81)
    assert out["minimum_horizon"] is not None
    assert out["power_lower_bound"] >= 0.90-1e-12


def test_composite_g2_horizon_sandwich_orders_necessary_and_sufficient_bounds():
    from alphavalue.cac import composite_g2_certifiability_sandwich
    rect=_rect(theta=(0.8,1.0),phi=(0.1,0.4),vr=(0.8,1.0),vx=(0.6,0.9))
    out=composite_g2_certifiability_sandwich(rect,0.05,0.10,max_horizon=400,eta_grid_size=81)
    assert out["status"] == "CERTIFIABILITY_HORIZON_SANDWICHED"
    assert out["necessary_horizon_kl"] <= out["sufficient_horizon_terminal_t"]


def test_t_sufficient_horizon_is_none_when_theta_rectangle_touches_boundary():
    from alphavalue.cac import minimum_stationary_horizon_for_g2_t_power
    out=minimum_stationary_horizon_for_g2_t_power(
        _rect(theta=(-0.1,0.8)),0.05,0.10,max_horizon=50,eta_grid_size=41
    )
    assert out["minimum_horizon"] is None
    assert out["reason"] == "zero_uniform_separation_or_design_floor"


def test_observed_design_frontier_has_three_exact_zones_and_ordered_energy_thresholds():
    from alphavalue.cac import g2_observed_design_certifiability_frontier
    rect=_rect(theta=(0.5,0.8),vr=(0.8,1.2))
    # Thresholds are returned independent of the chosen Q.
    base=g2_observed_design_certifiability_frontier(rect,1.0,30,0.05,0.10)
    qn=base["design_energy_necessary"]
    qs=base["design_energy_sufficient_terminal_t"]
    assert 0.0 < qn < qs < math.inf
    low=g2_observed_design_certifiability_frontier(rect,0.9*qn,30,0.05,0.10)
    mid=g2_observed_design_certifiability_frontier(rect,0.5*(qn+qs),30,0.05,0.10)
    high=g2_observed_design_certifiability_frontier(rect,1.01*qs,30,0.05,0.10)
    assert low["status"] == "UNIFORMLY_IMPOSSIBLE_BY_PAIRWISE_KL"
    assert mid["status"] == "FINITE_SAMPLE_UNRESOLVED_ZONE"
    assert high["status"] == "UNIFORMLY_TERMINAL_T_CERTIFIABLE"


def test_observed_design_t_sufficiency_matches_noncentral_t_at_worst_corner():
    from alphavalue.cac import g2_observed_design_certifiability_frontier
    rect=_rect(theta=(0.45,0.9),vr=(0.7,1.3))
    alpha,beta,n=0.05,0.10,50
    tmp=g2_observed_design_certifiability_frontier(rect,1.0,n,alpha,beta)
    Q=tmp["design_energy_sufficient_terminal_t"]
    out=g2_observed_design_certifiability_frontier(rect,Q,n,alpha,beta)
    delta=rect["theta"][0]*math.sqrt(Q/rect["return_noise_variance"][1])
    crit=student_t.ppf(1-alpha,n-1)
    p=nct.sf(crit,n-1,delta)
    assert abs(p-(1-beta)) < 2e-10
    assert out["status"] == "UNIFORMLY_TERMINAL_T_CERTIFIABLE"


def test_observed_design_frontier_refuses_uniform_claim_if_rectangle_touches_boundary():
    from alphavalue.cac import g2_observed_design_certifiability_frontier
    out=g2_observed_design_certifiability_frontier(
        _rect(theta=(-0.1,0.7)),1000.0,40,0.05,0.10
    )
    assert out["status"] == "NO_UNIFORM_SEPARATION_FROM_BOUNDARY"
    assert math.isinf(out["design_energy_sufficient_terminal_t"])


def test_predictable_eprocess_threshold_inverts_power_bound_exactly():
    from alphavalue.cac import (
        predictable_gaussian_eprocess_information_threshold,
        predictable_gaussian_eprocess_power_lower_bound,
    )
    for alpha,beta in [(0.05,0.10),(0.01,0.20),(0.1,0.05)]:
        I=predictable_gaussian_eprocess_information_threshold(alpha,beta)
        p=predictable_gaussian_eprocess_power_lower_bound(I,alpha)
        assert abs(p-(1-beta)) < 1e-12


def test_postfreeze_eprocess_frontier_has_impossible_gap_and_sufficient_zones():
    from alphavalue.cac import g2_postfreeze_eprocess_frontier
    rect=_rect(theta=(0.5,0.9),vr=(0.8,1.2))
    base=g2_postfreeze_eprocess_frontier(rect,1.0,0.05,0.10)
    qn=base["design_energy_necessary"]
    qs=base["design_energy_sufficient_eprocess"]
    assert qn < qs
    a=g2_postfreeze_eprocess_frontier(rect,0.9*qn,0.05,0.10)
    b=g2_postfreeze_eprocess_frontier(rect,0.5*(qn+qs),0.05,0.10)
    c=g2_postfreeze_eprocess_frontier(rect,1.01*qs,0.05,0.10)
    assert a["status"] == "UNIFORMLY_IMPOSSIBLE_BY_HARD_WORLD_KL"
    assert b["status"] == "NECESSARY_SUFFICIENT_GAP"
    assert c["status"] == "ROBUST_ANYTIME_EPROCESS_POWER_GUARANTEED"
    assert c["power_lower_bound"] >= 0.90


def test_postfreeze_eprocess_sufficient_threshold_reference_value():
    from alphavalue.cac import predictable_gaussian_eprocess_information_threshold
    I=predictable_gaussian_eprocess_information_threshold(0.05,0.10)
    ref=(math.sqrt(math.log(20.0))+math.sqrt(math.log(10.0)))**2
    assert abs(I-ref) < 1e-14
    assert 10.5 < I < 10.6


def test_correlated_information_identity_recovers_rho_zero_and_adds_information():
    from alphavalue.cac import g2_correlated_information_economic_identity
    kw=dict(theta=0.9,boundary=0.2,phi=0.5,return_noise_variance=0.2,
            signal_innovation_variance=0.3,gamma=1.1,trading_cost=0.4,horizon=12)
    z=g2_correlated_information_economic_identity(**kw,innovation_correlation=0.0)
    r=g2_correlated_information_economic_identity(**kw,innovation_correlation=0.8)
    assert abs(z["full_joint_information_nats"]-z["return_only_information_nats"])<1e-14
    assert r["full_joint_information_nats"] > r["return_only_information_nats"]
    assert abs(r["full_joint_information_nats"] / r["return_only_information_nats"] - 1/(1-0.8**2)) < 1e-13
    assert abs(r["full_joint_identity_residual"]) < 1e-13
    assert r["full_joint_price_per_nat"] < z["full_joint_price_per_nat"]


def test_student_t_first_order_information_tax_matches_exact_to_second_order_scale():
    from alphavalue.cac import student_t_information_threshold_asymptotic
    alpha,beta=0.05,0.10
    # Remainder times df^2 should remain bounded/stable as predicted by O(df^-2).
    vals=[]
    for df in (59,119,311,999):
        out=student_t_information_threshold_asymptotic(alpha,beta,df)
        vals.append(abs(out["information_remainder_nats"])*df*df)
        assert out["relative_I_inflation_first_order"]>1.0
    assert max(vals) < 12.0
    assert min(vals) > 0.1


def test_exact_gaussian_eprocess_threshold_is_sharper_than_chernoff_and_inverts_power():
    from alphavalue.cac import (
        predictable_gaussian_eprocess_exact_information_threshold,
        predictable_gaussian_eprocess_exact_power_lower_bound,
        predictable_gaussian_eprocess_information_threshold,
    )
    alpha,beta=0.05,0.10
    I= predictable_gaussian_eprocess_exact_information_threshold(alpha,beta)
    old=predictable_gaussian_eprocess_information_threshold(alpha,beta)
    p=predictable_gaussian_eprocess_exact_power_lower_bound(I,alpha)
    assert abs(p-(1-beta)) < 2e-13
    assert I < old
    assert 6.9 < I < 7.0


def test_conditional_design_quantile_lower_bound_is_empirically_conservative_from_nonstationary_state():
    from alphavalue.cac import g2_conditional_design_energy_quantile_lower
    rng=np.random.default_rng(271801)
    x0=1.7; phi=0.7; vx=0.35; H=18; eta=0.10
    out=g2_conditional_design_energy_quantile_lower(x0,abs(phi),vx,H,eta)
    q0=out["design_energy_quantile_lower"]
    nmc=30000
    x=np.full(nmc,x0)
    q=np.full(nmc,x0*x0)
    for _ in range(1,H):
        x=phi*x+rng.normal(scale=math.sqrt(vx),size=nmc)
        q+=x*x
    frac=float(np.mean(q>=q0))
    assert frac >= 1-eta-0.01


def test_full_g2_anytime_power_bound_is_below_seeded_monte_carlo_power():
    # Regression sanity check.  The theorem is analytical; simulation only
    # checks that implementation is on the conservative side in one world.
    from alphavalue.cac import g2_anytime_survival_power_lower_bound
    rng=np.random.default_rng(271802)
    rect=_rect(theta=(1.20,1.40),phi=(-0.35,0.55),vr=(0.7,1.0),vx=(0.45,0.8))
    H=28; x0=0.6; alpha=0.05
    out=g2_anytime_survival_power_lower_bound(rect,x0,H,alpha,eta_grid_size=101)
    assert 0.0 < out["power_lower_bound"] < 1.0
    # Simulate a deliberately hard compatible corner with independent innovations.
    theta=rect["theta"][0]; phi=0.55; vR=rect["return_noise_variance"][1]; vX=rect["signal_innovation_variance"][0]
    qtarget=out["design_energy_clock"]; lam=out["horizon_matched_lambda"]; L=math.log(1/alpha)
    nmc=20000; crossed=0
    for _ in range(nmc):
        x=x0; clock=0.0; loge=0.0; hit=False
        for _j in range(H):
            if clock < qtarget:
                rem=qtarget-clock
                if x*x <= rem:
                    c=1.0
                else:
                    c=math.sqrt(rem/(x*x)) if x!=0.0 else 0.0
                eps=rng.normal(scale=math.sqrt(vR))
                y=theta*x+eps
                w=c*x
                loge += lam*w*y - 0.5*lam*lam*vR*w*w
                clock += w*w
                if loge >= L:
                    hit=True
            else:
                eps=rng.normal(scale=math.sqrt(vR))
            xi=rng.normal(scale=math.sqrt(vX))
            x=phi*x+xi
        crossed += hit
    mc=crossed/nmc
    assert mc + 0.015 >= out["power_lower_bound"]


def test_end_to_end_error_composition_hits_declared_budgets_when_future_bound_clears():
    from alphavalue.cac import g2_end_to_end_survival_frontier
    rect=_rect(theta=(1.1,1.3),phi=(0.0,0.2),vr=(0.6,0.8),vx=(0.8,1.0))
    out=g2_end_to_end_survival_frontier(
        rect,1.0,80,0.05,0.10,0.01,eta_grid_size=101
    )
    assert out["total_false_deploy_upper_bound"] <= 0.05+1e-15
    assert out["future_alpha_budget"] == pytest.approx(0.04)
    assert out["future_beta_budget"] == pytest.approx(0.09)
    if out["status"]=="END_TO_END_RELIABLE_CERTIFICATION_GUARANTEED":
        assert out["total_power_lower_bound"] >= 0.90-1e-12


def test_control_defined_boundary_reproduces_least_world_hurdle_exactly():
    from alphavalue.cac import g2_positive_rectangle_economic_loading_boundary
    from alphavalue.unknown_scale import UnknownScaleCertificateModel, world_oracle_value_unknown_scale
    rect=_rect(theta=(0.4,1.0),phi=(0.2,0.7),vr=(0.8,1.2),vx=(0.25,0.6))
    hurdle=0.015; gamma=1.2; cost=0.7; H=9; term=0.1
    out=g2_positive_rectangle_economic_loading_boundary(
        rect,hurdle,gamma=gamma,trading_cost=cost,economic_horizon=H,terminal_penalty=term
    )
    b=out["economic_loading_boundary"]
    model=UnknownScaleCertificateModel(gamma,cost,H,terminal_penalty=term)
    v=world_oracle_value_unknown_scale(b,rect["phi"][0],rect["signal_innovation_variance"][0],model=model)
    assert abs(v-hurdle) < 1e-12


def test_control_coupled_frontier_uses_positive_economic_boundary():
    from alphavalue.cac import g2_control_coupled_end_to_end_frontier
    rect=_rect(theta=(0.8,1.1),phi=(0.15,0.45),vr=(0.6,0.9),vx=(0.5,0.8))
    out=g2_control_coupled_end_to_end_frontier(
        rect,0.8,80,0.05,0.10,0.01,0.01,
        gamma=1.0,trading_cost=0.5,economic_horizon=10,eta_grid_size=81,
    )
    assert out["control_defined_boundary"]>0.0
    assert out["economic"]["least_favourable_value_multiplier"]>0.0


def test_unknown_scale_pinching_uses_np_lower_and_t_upper_and_shrinks():
    from alphavalue.cac import composite_unknown_scale_pinching
    a=composite_unknown_scale_pinching(0.05,0.10,23)
    b=composite_unknown_scale_pinching(0.05,0.10,311)
    assert a["minimax_threshold_lower_nats"] < a["minimax_threshold_upper_nats"]
    assert b["exact_pinching_width_nats"] < a["exact_pinching_width_nats"]
    assert abs(b["exact_relative_pinching_ratio"]-1.0) < 0.01


def test_finite_h_design_spectral_floor_improves_classic_one_plus_rho_bound():
    from alphavalue.cac import g2_conditional_design_energy_quantile_lower
    rho=0.7; vx=0.4; H=8
    out=g2_conditional_design_energy_quantile_lower(0.2,rho,vx,H,0.1)
    old=vx/(1+rho)**2
    assert out["conditional_covariance_eigenvalue_floor"] > old
    assert out["finite_h_spectral_denominator"] < (1+rho)**2


def test_clock_threshold_recovers_canonical_reliable_information_threshold():
    from alphavalue.cac import (
        predictable_gaussian_clock_information_threshold,
        predictable_gaussian_clock_power_lower_bound,
        reliable_information_threshold,
    )
    alpha,beta=0.05,0.10
    I=predictable_gaussian_clock_information_threshold(alpha,beta)
    assert I == pytest.approx(reliable_information_threshold(alpha,beta)["I_crit_nats"])
    assert predictable_gaussian_clock_power_lower_bound(I,alpha) == pytest.approx(1-beta,abs=2e-13)


def test_clock_g2_bound_dominates_fixed_boundary_eprocess_bound_same_rectangle():
    from alphavalue.cac import (
        g2_anytime_survival_power_lower_bound,
        g2_information_clock_survival_power_lower_bound,
    )
    rect=_rect(theta=(0.70,0.95),phi=(-0.35,0.55),vr=(0.7,1.0),vx=(0.45,0.8))
    for H in (40,80,120):
        c=g2_information_clock_survival_power_lower_bound(rect,0.6,H,0.05,eta_grid_size=121)
        e=g2_anytime_survival_power_lower_bound(rect,0.6,H,0.05,eta_grid_size=121)
        assert c["power_lower_bound"] >= e["power_lower_bound"]-1e-14


def test_clock_g2_bound_is_conservative_under_correlated_innovations_mc():
    from alphavalue.cac import g2_information_clock_survival_power_lower_bound
    from scipy.stats import norm
    rng=np.random.default_rng(271803)
    rect=_rect(theta=(1.20,1.40),phi=(-0.35,0.55),vr=(0.7,1.0),vx=(0.45,0.8))
    H=28; x0=0.6; alpha=0.05
    out=g2_information_clock_survival_power_lower_bound(rect,x0,H,alpha,eta_grid_size=101)
    assert 0.0 < out["power_lower_bound"] < 1.0
    theta=rect["theta"][0]; phi=0.55; vR=rect["return_noise_variance"][1]; vX=rect["signal_innovation_variance"][0]
    rho_eps=0.8; qtarget=out["design_energy_clock"]
    threshold=float(norm.ppf(1-alpha))*math.sqrt(vR*qtarget)
    nmc=20000
    x=np.full(nmc,x0); clock=np.zeros(nmc); score=np.zeros(nmc); done=np.zeros(nmc,dtype=bool); reject=np.zeros(nmc,dtype=bool)
    cov=np.array([[vR,rho_eps*math.sqrt(vR*vX)],[rho_eps*math.sqrt(vR*vX),vX]])
    for _j in range(H):
        draws=rng.multivariate_normal([0.0,0.0],cov,size=nmc)
        eps,xi=draws[:,0],draws[:,1]
        y=theta*x+eps
        active=~done
        rem=np.maximum(0.0,qtarget-clock)
        xx=x*x
        c=np.zeros(nmc)
        whole=active & (xx<=rem)
        c[whole]=1.0
        part=active & (~whole) & (xx>0.0)
        c[part]=np.sqrt(rem[part]/xx[part])
        w=c*x
        score += w*y
        clock += w*w
        newly=active & (clock>=qtarget*(1-1e-12))
        reject[newly]=score[newly]>=threshold
        done[newly]=True
        x=phi*x+xi
    mc=float(np.mean(reject))
    assert mc + 0.015 >= out["power_lower_bound"]

def test_end_to_end_clock_composition_respects_total_error_budget():
    from alphavalue.cac import g2_end_to_end_clock_frontier
    rect=_rect(theta=(1.1,1.3),phi=(0.0,0.2),vr=(0.6,0.8),vx=(0.8,1.0))
    out=g2_end_to_end_clock_frontier(rect,1.0,80,0.05,0.10,0.01,eta_grid_size=101)
    assert out["total_false_deploy_upper_bound"] <= 0.05+1e-15
    assert out["future_alpha_budget"] == pytest.approx(0.04)
    assert out["future_beta_budget"] == pytest.approx(0.09)
    if out["status"]=="END_TO_END_CLOCK_CERTIFICATION_GUARANTEED":
        assert out["total_power_lower_bound"] >= 0.90-1e-12


def test_trace_frobenius_design_bound_is_valid_and_asymptotically_sharper():
    from alphavalue.cac import g2_conditional_design_energy_quantile_lower
    # Rectangle includes phi=0, so the least-compatible long-run design rate is vX-.
    vx=0.45; rho=0.55; eta=0.05
    small=g2_conditional_design_energy_quantile_lower(0.6,rho,vx,40,eta,min_abs_phi=0.0)
    large=g2_conditional_design_energy_quantile_lower(0.6,rho,vx,1000,eta,min_abs_phi=0.0)
    assert large["design_energy_quantile_lower_trace_frobenius"] > large["design_energy_quantile_lower_spectral"]
    rate=(large["design_energy_quantile_lower_trace_frobenius"]-0.36)/(999.0)
    assert 0.30 < rate < vx
    assert small["design_energy_quantile_lower"] >= small["design_energy_quantile_lower_spectral"]


def test_trace_frobenius_quantile_mc_coverage_at_persistent_nonstationary_world():
    from alphavalue.cac import g2_conditional_design_energy_quantile_lower
    rng=np.random.default_rng(271804)
    x0=1.1; phi=0.75; vx=0.3; H=180; eta=0.10
    out=g2_conditional_design_energy_quantile_lower(x0,abs(phi),vx,H,eta,min_abs_phi=abs(phi))
    q0=out["design_energy_quantile_lower"]
    nmc=30000
    x=np.full(nmc,x0); q=np.full(nmc,x0*x0)
    for _ in range(1,H):
        x=phi*x+rng.normal(scale=math.sqrt(vx),size=nmc)
        q+=x*x
    assert float(np.mean(q>=q0)) >= 1-eta-0.01


def test_two_stage_composition_is_multiplicative_and_gated_bound_is_sharper():
    from alphavalue.cac import two_stage_error_composition
    out=two_stage_error_composition(0.04,0.08,0.01,0.02)
    assert out["generic_false_deploy_upper_bound"] == pytest.approx(0.01+0.99*0.04)
    assert out["power_lower_bound"] == pytest.approx(0.98*0.92)
    gated=two_stage_error_composition(0.40,0.08,0.01,0.02,gate_implies_alternative_on_good_event=True)
    assert gated["false_deploy_upper_bound"] == pytest.approx(0.01)


def test_two_stage_target_budget_inversion_hits_targets_exactly():
    from alphavalue.cac import future_error_budgets_for_two_stage_targets, two_stage_error_composition
    b=future_error_budgets_for_two_stage_targets(0.05,0.10,0.01,0.02)
    out=two_stage_error_composition(b["alpha_future_max"],b["beta_future_max"],0.01,0.02)
    assert out["generic_false_deploy_upper_bound"] == pytest.approx(0.05)
    assert out["type2_upper_bound"] == pytest.approx(0.10)


def test_realized_rectangle_certificate_does_not_overclaim_unconditional_power():
    from alphavalue.cac import g2_realized_rectangle_future_power_certificate
    rect=_rect(theta=(1.1,1.3),phi=(0.0,0.2),vr=(0.6,0.8),vx=(0.8,1.0))
    out=g2_realized_rectangle_future_power_certificate(rect,1.0,80,0.04,0.01,eta_grid_size=81)
    assert out["formation_confidence"] == pytest.approx(0.99)
    assert out["unconditional_power_claimed"] is False
    assert 0.0 <= out["reported_future_power_lower_bound"] <= 1.0
