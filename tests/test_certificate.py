import math
import numpy as np
import pytest
from alphavalue.certificate import (
    LoadingCertificateModel, loading_certificate, theta_interval,
    robust_theta, policy_value, oracle_value,
)


def model(**kw):
    d=dict(phi=.8,gamma=1.,trading_cost=10.,signal_innovation_variance=.36,
           return_noise_sd=1.,horizon=24,confidence=.95)
    d.update(kw); return LoadingCertificateModel(**d)


def test_robust_theta_cases():
    assert robust_theta(.2,.5)==pytest.approx(.2)
    assert robust_theta(-.5,-.2)==pytest.approx(-.2)
    assert robust_theta(-.1,.2)==0


def test_value_identity_and_regret_square():
    m=model(); th=.3; c=-.1
    A=oracle_value(1.,model=m)
    assert oracle_value(th,model=m)-policy_value(th,c,model=m)==pytest.approx(A*(th-c)**2,rel=1e-12)


def test_interval_matches_closed_form():
    x=np.array([1.,2.,-1.]); theta=.4
    y=theta*x
    ci=theta_interval(x,y,return_noise_sd=2.,confidence=.95)
    assert ci['theta_hat']==pytest.approx(theta)
    assert ci['standard_error']==pytest.approx(2/math.sqrt(6))


def test_certificate_bounds_every_theta_in_interval():
    x=np.linspace(-2,2,80)
    y=.35*x
    m=model(return_noise_sd=.4,horizon=18)
    c=loading_certificate(x,y,model=m,hurdle=.001)
    lo,hi=c['theta_interval']['lower'],c['theta_interval']['upper']
    for th in np.linspace(lo,hi,101):
        assert policy_value(th,c['robust_theta'],model=m) >= c['policy_value_lower']-1e-12
        assert oracle_value(th,model=m) <= c['oracle_value_upper']+1e-12


def test_three_decisions_are_reachable():
    x=np.ones(200)
    m=model(return_noise_sd=.05,horizon=12)
    deploy=loading_certificate(x,.5*x,model=m,hurdle=.01)
    small=loading_certificate(x,0*x,model=m,hurdle=1.)
    uncertain=loading_certificate(x,0*x,model=m,hurdle=.0001)
    assert deploy['decision']=='deploy_in_model'
    assert small['decision']=='economically_small_in_model'
    assert uncertain['decision']=='insufficient_evidence'


def test_gaussian_coverage_monte_carlo():
    rng=np.random.default_rng(1729)
    x=np.linspace(-1,1,60); theta=.2; sd=.7; reps=20000
    Q=x@x; estimates=theta+sd/math.sqrt(Q)*rng.standard_normal(reps)
    r=1.959963984540054*sd/math.sqrt(Q)
    frac=np.mean((estimates-r<=theta)&(theta<=estimates+r))
    assert .944 < frac < .956


@pytest.mark.parametrize('bad',[float('nan'),float('inf'),0.,-1.])
def test_bad_noise_rejected(bad):
    with pytest.raises(ValueError): theta_interval([1.],[1.],return_noise_sd=bad)


def test_persistence_has_zero_price_without_trading_cost():
    from alphavalue.certificate import joint_misspecification_regret
    r=joint_misspecification_regret(true_theta=.3,true_phi=.9,controller_theta=.3,controller_phi=-.4,
        gamma=2.,trading_cost=0.,signal_innovation_variance=.2,horizon=30)
    assert r['regret']==pytest.approx(0.,abs=1e-14)


def test_loading_error_still_costs_without_trading_cost():
    from alphavalue.certificate import joint_misspecification_regret
    r=joint_misspecification_regret(true_theta=.3,true_phi=.9,controller_theta=.1,controller_phi=.9,
        gamma=2.,trading_cost=0.,signal_innovation_variance=.2,horizon=30)
    assert r['regret']>0
