import json
from pathlib import Path

import numpy as np
import pytest

from alphavalue.joint import (
    JointCertificateModel, gaussian_mixture_cs_interval, joint_parameter_rectangle,
    joint_value_certificate, world_oracle_value, world_value,
)


def simulate(seed=11,n=160,phi=.65,theta=.35,sx=.5,sy=.7):
    rng=np.random.default_rng(seed)
    x=np.empty(n+1)
    x[0]=rng.normal(scale=sx/np.sqrt(1-phi*phi))
    for t in range(n):
        x[t+1]=phi*x[t]+rng.normal(scale=sx)
    y=theta*x[:-1]+rng.normal(scale=sy,size=n)
    return x[:-1],x[1:],y


def test_normal_mixture_cs_is_nested_in_model_statement_not_fixed_ci():
    x,xn,y=simulate()
    r=gaussian_mixture_cs_interval(x,y,noise_sd=.7,alpha=.025,rho=1.)
    assert r['lower']<.35<r['upper']
    assert r['radius']>0
    assert 'Time-uniform' in r['coverage']


def test_joint_rectangle_contains_true_parameters_in_seeded_case():
    x,xn,y=simulate(seed=18)
    m=JointCertificateModel(1.,2.,.5,.7,8)
    r=joint_parameter_rectangle(x,xn,y,model=m,alpha=.05,phi_bounds=(-.95,.95))
    assert r['theta']['lower']<.35<r['theta']['upper']
    assert r['phi']['lower']<.65<r['phi']['upper']
    assert not r['empty']


def test_interval_economic_bounds_cover_true_world_values():
    x,xn,y=simulate(seed=21,n=220)
    m=JointCertificateModel(1.,2.,.5,.7,7)
    r=joint_value_certificate(x,xn,y,model=m,hurdle=0.,alpha=.05,
                              phi_bounds=(-.95,.95),theta_cells=5,phi_cells=10)
    gains=np.asarray(r['controller']['gains'])
    true_policy=world_value(.35,.65,gains,model=m)
    true_oracle=world_oracle_value(.35,.65,model=m)
    assert r['policy_value_lower']<=true_policy+1e-12
    assert r['oracle_value_upper']>=true_oracle-1e-12
    assert r['decision'] in {'deploy_in_model','insufficient_evidence','economically_small_in_model'}


def test_zero_trading_cost_removes_persistence_from_oracle_gain():
    m=JointCertificateModel(1.,0.,.5,.7,6)
    from alphavalue.joint import _gain_and_curvature
    g1,_=_gain_and_curvature(.1,theta=.4,model=m)
    g2,_=_gain_and_curvature(.9,theta=.4,model=m)
    np.testing.assert_allclose(g1,g2,rtol=0,atol=1e-14)


def test_crossing_zero_returns_zero_candidate_lower_bound():
    # Directly construct weak data so the loading CS crosses zero.
    rng=np.random.default_rng(4);x=rng.normal(size=80);xn=.5*x+rng.normal(scale=.5,size=80);y=rng.normal(scale=1.,size=80)
    m=JointCertificateModel(1.,1.,.5,1.,5)
    r=joint_value_certificate(x,xn,y,model=m,hurdle=.01,phi_bounds=(-.9,.9),theta_cells=3,phi_cells=6)
    assert r['parameter_rectangle']['theta']['lower']<0<r['parameter_rectangle']['theta']['upper']
    assert r['controller']['theta']==0
    assert r['policy_value_lower']<=0+1e-12


def test_sequential_joint_cs_empirical_crossing_rate_is_controlled():
    # Validation, not proof: one fixed seed and a deliberately moderate Monte Carlo size.
    rng=np.random.default_rng(7781)
    reps=1200;n=80;phi=.55;theta=.25;sx=.7;sy=.9;alpha=.10
    failures=0
    m=JointCertificateModel(1.,1.,sx,sy,4)
    for _ in range(reps):
        x=np.empty(n+1);x[0]=rng.normal(scale=sx/np.sqrt(1-phi*phi))
        for t in range(n):x[t+1]=phi*x[t]+rng.normal(scale=sx)
        y=theta*x[:-1]+rng.normal(scale=sy,size=n)
        crossed=False
        for k in (10,20,40,80):
            rect=joint_parameter_rectangle(x[:k],x[1:k+1],y[:k],model=m,alpha=alpha,
                                           phi_bounds=(-.95,.95))
            if not (rect['theta']['lower']<=theta<=rect['theta']['upper'] and
                    rect['phi']['lower']<=phi<=rect['phi']['upper']):
                crossed=True;break
        failures+=crossed
    # A generous statistical regression guard. The theorem, not this MC, provides coverage.
    assert failures/reps < .13


def test_positive_persistence_closed_form_is_exact_least_world_maximin():
    from alphavalue.joint import _positive_persistence_exact_candidate, _gain_and_curvature
    m=JointCertificateModel(1.3,7.0,.4,.6,9,terminal_penalty=.8)
    ti=(.18,.51); pi=(.35,.91)
    e=_positive_persistence_exact_candidate(ti,pi,model=m)
    assert e is not None
    assert e['controller_theta']==pytest.approx(ti[0])
    assert e['controller_phi']==pytest.approx(pi[0])
    # Oracle unit gains must be coordinatewise nondecreasing in positive persistence.
    g0,_=_gain_and_curvature(pi[0],theta=1.,model=m)
    g1,_=_gain_and_curvature(pi[1],theta=1.,model=m)
    assert np.all(g1>=g0)
    # The proposed controller attains the least-world oracle and never does worse
    # on a dense independent rectangle check.
    least=world_oracle_value(ti[0],pi[0],model=m)
    assert e['theoretical_maximin_value']==pytest.approx(least,rel=1e-13)
    gains=np.asarray(e['gains'])
    vals=[world_value(th,ph,gains,model=m)
          for th in np.linspace(*ti,31) for ph in np.linspace(*pi,47)]
    assert min(vals)>=least-2e-12
    # Any causal policy is upper bounded in the least world by that world's oracle.
    assert min(vals)==pytest.approx(least,abs=2e-12)
    assert e['theoretical_oracle_upper']==pytest.approx(world_oracle_value(ti[1],pi[1],model=m),rel=1e-13)


def test_negative_loading_rectangle_has_same_closed_form_geometry():
    from alphavalue.joint import _positive_persistence_exact_candidate
    m=JointCertificateModel(1.,3.,.5,.8,6)
    ti=(-.7,-.2);pi=(.1,.8)
    e=_positive_persistence_exact_candidate(ti,pi,model=m)
    assert e['controller_theta']==pytest.approx(-.2)
    assert e['controller_phi']==pytest.approx(.1)
    gains=np.asarray(e['gains'])
    least=world_oracle_value(-.2,.1,model=m)
    vals=[world_value(th,ph,gains,model=m) for th in np.linspace(*ti,29) for ph in np.linspace(*pi,33)]
    assert min(vals)>=least-2e-12


def test_crossing_zero_positive_persistence_has_exact_zero_maximin():
    from alphavalue.joint import _positive_persistence_exact_candidate
    m=JointCertificateModel(1.,4.,.5,.7,5)
    e=_positive_persistence_exact_candidate((-.2,.3),(.2,.9),model=m)
    assert e['theoretical_maximin_value']==0
    assert np.all(np.asarray(e['gains'])==0)
