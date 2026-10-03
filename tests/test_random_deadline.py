import math

from alphavalue.cac import (
    binary_kl,
    exponential_information_deadline_frontier,
    g2_random_deadline_power_lower_bound,
    random_deadline_expected_information_obstruction,
    random_deadline_frechet_power_lower,
)


def test_random_deadline_mean_information_obstruction_matches_optional_sampling_identity():
    out=random_deadline_expected_information_obstruction(10.0,0.05,0.10)
    assert out['expected_kl_nats']==5.0
    assert abs(out['binary_kl_required_nats']-binary_kl(0.9,0.05))<1e-14


def test_frechet_power_lower_needs_no_independence():
    assert abs(random_deadline_frechet_power_lower(0.9,0.8)-0.7)<1e-15
    assert random_deadline_frechet_power_lower(0.2,0.3)==0.0


def test_exponential_deadline_threshold_uses_full_alpha_budget():
    out=exponential_information_deadline_frontier(0.05,0.10,0.02)
    assert abs(out['null_killed_hitting_probability']-0.05)<1e-12
    assert abs(out['power_envelope']-out['power_envelope_roc_form'])<1e-13


def test_exponential_deadline_power_decreases_with_hazard():
    a=exponential_information_deadline_frontier(0.05,0.10,0.005)
    b=exponential_information_deadline_frontier(0.05,0.10,0.05)
    assert a['power_envelope']>b['power_envelope']
    assert a['certified_fraction_of_expected_PI_value_alpha_only']==a['power_envelope']


def test_exponential_deadline_critical_hazard_hits_target_power():
    base=exponential_information_deadline_frontier(0.05,0.10,0.02)
    rho=base['critical_hazard_for_reliable_target']
    at=exponential_information_deadline_frontier(0.05,0.10,rho)
    assert abs(at['power_envelope']-0.9)<2e-12
    assert at['reliable_target_feasible']


def test_exponential_deadline_mean_kl_can_be_far_above_deterministic_threshold():
    out=exponential_information_deadline_frontier(0.05,0.10,0.02)
    # This is a structural regression: the critical mean-KL under unannounced
    # exponential death is > the deterministic-horizon Gaussian threshold.
    assert out['critical_expected_lifetime_kl_nats'] > 4.28


def test_g2_random_deadline_composition_is_bounded_by_both_marginal_constraints():
    out=g2_random_deadline_power_lower_bound(
        horizons=[10,30,60],
        deadline_survival_lower=[0.95,0.80,0.55],
        current_signal=0.0,
        alpha=0.05,
        coefficient_gap=0.8,
        return_variance_upper=1.0,
        phi_bounds=(0.1,0.4),
        signal_variance_lower=0.6,
        eta_grid_size=41,
    )
    assert 0.0 <= out['power_lower_bound'] <= 1.0
    assert out['requires_deadline_certifier_independence'] is False
    best=[r for r in out['grid'] if r['horizon']==out['best_horizon']][0]
    assert out['power_lower_bound'] <= best['deterministic_certification_power_lower']+1e-15
    assert out['power_lower_bound'] <= best['deadline_survival_lower']+1e-15


def test_constant_information_rate_calendar_translation_matches_mean_kl():
    from alphavalue.cac import constant_information_rate_exponential_calendar_deadline
    out=constant_information_rate_exponential_calendar_deadline(0.05,0.10,0.5,0.02)
    assert abs(out['information_time_hazard']-0.02)<1e-14
    assert abs(out['mean_lifetime_kl_nats']-25.0)<1e-14


def test_g2_weak_gap_exponential_death_frontier_inverts_gap():
    from alphavalue.cac import g2_weak_gap_exponential_death_frontier
    base=g2_weak_gap_exponential_death_frontier(
        0.05,0.10,0.8,1.0,(0.1,0.4),0.6,0.01
    )
    dcrit=base['critical_coefficient_gap_weak_limit']
    at=g2_weak_gap_exponential_death_frontier(
        0.05,0.10,dcrit,1.0,(0.1,0.4),0.6,0.01
    )
    assert abs(at['power_envelope']-0.9)<3e-12
    assert at['weak_gap_feasible']


def test_canonical_random_deadline_upper_bound_and_sandwich_order():
    from alphavalue.cac import canonical_random_deadline_power_sandwich
    cuts=[1.0,4.0,9.0]
    surv=[0.95,0.75,0.4]
    lower=[0.15,0.55,0.82]
    out=canonical_random_deadline_power_sandwich(0.05,cuts,surv,lower)
    assert 0.0 <= out['power_lower_bound'] <= out['power_upper_bound'] <= 1.0
    assert out['sandwich_ordered']
