import math

from alphavalue.cac import (
    binary_kl,
    g2_structural_clock_power_lower_bound,
    g2_structural_hard_pair_kl,
    g2_direct_certification_horizon_sandwich,
    g2_resolution_time_bound,
)


def test_structural_clock_power_increases_with_horizon_and_is_bounded():
    kw=dict(current_signal=0.0,alpha=0.05,coefficient_gap=0.8,
            return_variance_upper=1.0,phi_bounds=(0.1,0.4),
            signal_variance_lower=0.6,eta_grid_size=81)
    a=g2_structural_clock_power_lower_bound(horizon=10,**kw)
    b=g2_structural_clock_power_lower_bound(horizon=80,**kw)
    assert 0.0 <= a['power_lower_bound'] <= 1.0
    assert 0.0 <= b['power_lower_bound'] <= 1.0
    assert b['power_lower_bound'] >= a['power_lower_bound'] - 1e-12
    assert b['requires_frozen_confidence_rectangle'] is False
    assert b['requires_return_signal_orthogonality'] is False


def test_hard_pair_kl_matches_closed_form_at_phi_zero():
    # phi=0, x0=0 gives E Q_H=(H-1)vX because X0=0 and each later X is a fresh innovation.
    H=17; d=0.7; V=1.3; vx=0.4
    out=g2_structural_hard_pair_kl(0.0,H,d,V,(0.0,0.0),vx)
    q=(H-1)*vx
    assert abs(out['hard_pair_expected_design_energy']-q)<1e-13
    assert abs(out['hard_pair_expected_kl_nats']-d*d*q/(2*V))<1e-13


def test_direct_horizon_sandwich_orders_endpoints():
    out=g2_direct_certification_horizon_sandwich(
        0.0,0.05,0.10,0.9,1.0,(0.1,0.4),0.7,
        max_horizon=1000,eta_grid_size=81,
    )
    assert out['status']=='DIRECT_G2_HORIZON_SANDWICHED'
    assert out['necessary_horizon_pairwise_kl'] is not None
    assert out['sufficient_horizon_clock'] is not None
    assert out['necessary_horizon_pairwise_kl'] <= out['sufficient_horizon_clock']
    assert out['same_class_minimax_ordering_certified']


def test_resolution_time_bound_meets_declared_power():
    out=g2_resolution_time_bound(
        0.0,0.05,0.10,0.9,1.0,(0.1,0.4),0.7,
        max_horizon=1000,eta_grid_size=81,
    )
    assert out['resolution_horizon_upper'] is not None
    assert out['resolution_probability_lower']==0.9
    assert out['sufficient_endpoint']['power_lower_bound'] >= 0.9-1e-12


def test_unrestricted_scale_obstruction_is_visible_from_hard_pair():
    # For any fixed H, the hard-pair KL can be made arbitrarily small by taking V large.
    H=100; target=binary_kl(0.9,0.05)
    small=g2_structural_hard_pair_kl(0.0,H,0.5,1e12,(0.0,0.3),0.5)
    assert small['hard_pair_expected_kl_nats'] < target


def test_vanishing_signal_variance_obstruction_from_zero_state():
    # With x0=0 and vX=0 the predictor is identically zero in the hard world.
    out=g2_structural_hard_pair_kl(0.0,100,1.0,1.0,(0.0,0.4),0.0)
    assert out['hard_pair_expected_design_energy']==0.0
    assert out['hard_pair_expected_kl_nats']==0.0


def test_weak_gap_constant_matches_information_rate_identity():
    from alphavalue.cac import g2_weak_gap_asymptotic_frontier_constant, reliable_information_threshold
    a,b,V,vx=0.05,0.10,1.3,0.6
    ph=(0.2,0.7)
    out=g2_weak_gap_asymptotic_frontier_constant(a,b,V,ph,vx)
    s=vx/(1-0.2**2)
    iz=reliable_information_threshold(a,b)['I_crit_nats']
    assert abs(out['worst_stationary_signal_variance']-s)<1e-14
    assert abs(out['weak_gap_horizon_constant']-2*V*iz/s)<1e-13


def test_weak_gap_clock_bound_moves_toward_asymptotic_constant():
    # This is a deterministic regression check of the theorem's numerical side,
    # not the proof.  A smaller gap at a fixed normalized horizon gives a bound
    # closer to the weak-gap limit because the O(sqrt(H)) design penalty is lower order.
    from alphavalue.cac import g2_structural_clock_power_lower_bound, g2_weak_gap_asymptotic_frontier_constant
    alpha,beta,V,vx=0.05,0.10,1.0,0.7
    ph=(0.1,0.4)
    C=g2_weak_gap_asymptotic_frontier_constant(alpha,beta,V,ph,vx)['weak_gap_horizon_constant']
    # Use a modest >1 safety factor; asymptotically the power lower bound must clear 0.9.
    factor=1.6
    powers=[]
    for d in (0.20,0.10,0.06):
        H=max(2,int(math.ceil(factor*C/(d*d))))
        out=g2_structural_clock_power_lower_bound(0.0,H,alpha,d,V,ph,vx,eta_grid_size=101)
        powers.append(out['power_lower_bound'])
    assert powers[-1] >= 0.90
    assert powers[-1] >= powers[0]-1e-12
