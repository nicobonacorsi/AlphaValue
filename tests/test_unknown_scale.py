import math
import numpy as np
import pytest

from alphavalue.unknown_scale import (
    NIGMixtureTuning, UnknownScaleCertificateModel,
    nig_log_evalue, nig_mixture_cs_projection, joint_unknown_scale_set,
    relative_regret, unknown_scale_value_certificate,
    world_value_unknown_scale, world_oracle_value_unknown_scale,
    _closed_form_positive_candidate,
)


def simulate(seed=18,n=500,phi=.65,theta=.35,sx=.5,sy=.7):
    rng=np.random.default_rng(seed)
    x=np.empty(n+1)
    x[0]=rng.normal(scale=sx/np.sqrt(1-phi*phi))
    for t in range(n):
        x[t+1]=phi*x[t]+rng.normal(scale=sx)
    y=theta*x[:-1]+rng.normal(scale=sy,size=n)
    return x[:-1],x[1:],y


def test_nig_projection_inverts_point_eprocess():
    x,_,y=simulate(seed=7,n=180)
    alpha=.025
    r=nig_mixture_cs_projection(x,y,alpha=alpha)
    assert not r['empty']
    # The true seeded world lies inside this realized projection.
    assert r['beta_lower'] < .35 < r['beta_upper']
    assert r['variance_lower'] < .7**2 < r['variance_upper']
    # The least-squares coefficient with the variance minimizing its null
    # e-value is accepted whenever the projected set is nonempty.
    vhat=r['rss_min']/r['n']
    assert nig_log_evalue(x,y,r['beta_hat'],vhat) < math.log(1/alpha)
    # A coefficient just beyond the exact profiled projection cannot be accepted
    # for any variance.  Its profile minimum exceeds the threshold.
    beta_out=r['raw_beta_projection'][1]+1e-6
    rss_beta=r['rss_min']+r['design_energy']*(beta_out-r['beta_hat'])**2
    profile=0.5*r['n']*(1+math.log(rss_beta/r['n']))+r['log_mixture_constant']
    assert profile > math.log(1/alpha)


def test_joint_unknown_scale_set_contains_seeded_true_tuple():
    x,xn,y=simulate(seed=18,n=600)
    r=joint_unknown_scale_set(x,xn,y,alpha=.05,phi_bounds=(-.95,.95))
    assert not r['empty']
    q=r['projection_rectangle']
    assert q['theta'][0] < .35 < q['theta'][1]
    assert q['phi'][0] < .65 < q['phi'][1]
    assert q['return_noise_variance'][0] < .7**2 < q['return_noise_variance'][1]
    assert q['signal_innovation_variance'][0] < .5**2 < q['signal_innovation_variance'][1]
    assert 'every monitoring time' in r['coverage_statement']


def test_relative_regret_exactly_cancels_signal_variance():
    model=UnknownScaleCertificateModel(1.2,5.0,9,terminal_penalty=.4)
    # Use a deliberately misspecified fixed gain vector.
    from alphavalue.unknown_scale import _unit_gain_and_curvature
    c,_=_unit_gain_and_curvature(.52,model=model)
    b=.31*c
    rr=relative_regret(.4,.71,b,model=model)
    for vx in (.03,.25,4.0):
        V=world_oracle_value_unknown_scale(.4,.71,vx,model=model)
        J=world_value_unknown_scale(.4,.71,vx,b,model=model)
        assert (V-J)/V == pytest.approx(rr,rel=3e-13,abs=3e-13)


def test_unknown_scale_certificate_bounds_true_world_and_can_certify_efficiency():
    x,xn,y=simulate(seed=18,n=5000)
    model=UnknownScaleCertificateModel(1.,2.,8)
    r=unknown_scale_value_certificate(
        x,xn,y,model=model,alpha=.05,hurdle=.001,
        efficiency_tolerance=.20,phi_bounds=(-.95,.95),
        theta_cells=5,phi_cells=12)
    assert r['decision']=='deploy_in_model'
    assert r['certified_eta_efficient']
    assert r['relative_regret_upper'] <= .20
    gains=np.asarray(r['controller']['gains'])
    trueJ=world_value_unknown_scale(.35,.65,.5**2,gains,model=model)
    trueV=world_oracle_value_unknown_scale(.35,.65,.5**2,model=model)
    trueR=(trueV-trueJ)/trueV
    assert r['policy_value_lower'] <= trueJ+1e-11
    assert r['oracle_value_upper'] >= trueV-1e-11
    assert r['relative_regret_upper'] >= trueR-1e-11


def test_efficiency_not_issued_if_loading_projection_crosses_zero():
    rng=np.random.default_rng(9)
    n=120
    x=rng.normal(size=n)
    xn=.4*x+rng.normal(scale=.5,size=n)
    y=rng.normal(scale=1.,size=n)
    model=UnknownScaleCertificateModel(1.,1.,5)
    r=unknown_scale_value_certificate(x,xn,y,model=model,alpha=.05,
                                      efficiency_tolerance=.5,phi_bounds=(-.9,.9),
                                      theta_cells=4,phi_cells=8)
    ti=r['parameter_set']['projection_rectangle']['theta']
    assert ti[0] < 0 < ti[1]
    assert r['relative_regret_upper'] is None
    assert not r['certified_eta_efficient']


def test_sequential_unknown_scale_joint_cs_empirical_crossing_rate_is_controlled():
    # Regression guard only; the martingale proof provides the theorem.
    rng=np.random.default_rng(1729)
    reps=350;n=80;phi=.55;theta=.25;sx=.7;sy=.9;alpha=.10
    failures=0
    for _ in range(reps):
        x=np.empty(n+1);x[0]=rng.normal(scale=sx/np.sqrt(1-phi*phi))
        for t in range(n):
            x[t+1]=phi*x[t]+rng.normal(scale=sx)
        y=theta*x[:-1]+rng.normal(scale=sy,size=n)
        crossed=False
        for k in (10,20,40,80):
            r=joint_unknown_scale_set(x[:k],x[1:k+1],y[:k],alpha=alpha,
                                      phi_bounds=(-.95,.95))
            if r['empty']:
                crossed=True;break
            q=r['projection_rectangle']
            if not (q['theta'][0]<=theta<=q['theta'][1] and
                    q['phi'][0]<=phi<=q['phi'][1] and
                    q['return_noise_variance'][0]<=sy*sy<=q['return_noise_variance'][1] and
                    q['signal_innovation_variance'][0]<=sx*sx<=q['signal_innovation_variance'][1]):
                crossed=True;break
        failures+=crossed
    assert failures/reps < .16


def test_bundled_unknown_scale_efficiency_example_reproduces():
    data=np.genfromtxt('examples/unknown_scale_efficiency.csv',delimiter=',',names=True)
    model=UnknownScaleCertificateModel(1.,2.,8)
    r=unknown_scale_value_certificate(
        data['signal'],data['next_signal'],data['next_return'],model=model,
        alpha=.05,hurdle=.001,efficiency_tolerance=.10,
        phi_bounds=(-.95,.95),theta_cells=6,phi_cells=16)
    assert r['decision']=='deploy_in_model'
    assert r['efficiency_decision']=='certified_0.1_efficient_in_model'
    assert r['relative_regret_upper']==pytest.approx(0.09456378208504121,rel=2e-12,abs=2e-12)



def test_unknown_scale_positive_rectangle_corner_is_exact_on_grid():
    model=UnknownScaleCertificateModel(1.1,3.0,7,terminal_penalty=.2)
    ti=(.35,.8); pi=(.2,.75); vi=(.04,.36)
    candidate=_closed_form_positive_candidate(ti,pi,model=model)
    assert candidate is not None
    assert 'exact maximin' in candidate['method']
    gains=np.asarray(candidate['gains'])
    least=world_oracle_value_unknown_scale(ti[0],pi[0],vi[0],model=model)
    worst=math.inf
    oracle_max=-math.inf
    for theta in np.linspace(*ti,8):
        for phi in np.linspace(*pi,9):
            for vx in np.linspace(*vi,5):
                worst=min(worst,world_value_unknown_scale(theta,phi,vx,gains,model=model))
                oracle_max=max(oracle_max,world_oracle_value_unknown_scale(theta,phi,vx,model=model))
    far=ti[1]
    largest=world_oracle_value_unknown_scale(far,pi[1],vi[1],model=model)
    assert worst == pytest.approx(least,rel=2e-12,abs=2e-12)
    assert oracle_max == pytest.approx(largest,rel=2e-12,abs=2e-12)
