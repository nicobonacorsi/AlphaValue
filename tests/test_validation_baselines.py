import math
import numpy as np
from scipy.stats import norm
from alphavalue.validation_baselines import (
    probabilistic_sharpe_ratio, minimum_track_record_length,
    expected_maximum_sharpe, deflated_sharpe_ratio, sharpe_moments,
)

def test_psr_normal_reduces_to_z_formula():
    sr=.1;n=101
    got=probabilistic_sharpe_ratio(sharpe=sr,n=n,benchmark=0,skewness=0,kurtosis=3)
    den=math.sqrt(1+.5*sr*sr)
    assert abs(got-norm.cdf(sr*math.sqrt(n-1)/den))<1e-12

def test_mintrl_inverts_psr():
    sr=.12;sk=.2;ku=4.0;c=.95
    n=minimum_track_record_length(sharpe=sr,confidence=c,skewness=sk,kurtosis=ku)
    # At the continuous n solution PSR equals confidence.
    got=probabilistic_sharpe_ratio(sharpe=sr,n=n,skewness=sk,kurtosis=ku)
    assert abs(got-c)<1e-12

def test_mintrl_infinite_below_benchmark():
    assert math.isinf(minimum_track_record_length(sharpe=-.01,benchmark=0))

def test_expected_max_one_trial_convention():
    assert expected_maximum_sharpe(n_trials=1,sharpe_variance=.2)==0

def test_expected_max_increases_with_trials():
    a=expected_maximum_sharpe(n_trials=10,sharpe_variance=.02)
    b=expected_maximum_sharpe(n_trials=100,sharpe_variance=.02)
    assert b>a>0

def test_dsr_is_psr_against_expected_max():
    kw=dict(sharpe=.2,n=200,n_trials=50,trial_sharpe_variance=.005,skewness=.1,kurtosis=3.5)
    out=deflated_sharpe_ratio(**kw)
    direct=probabilistic_sharpe_ratio(sharpe=kw['sharpe'],n=kw['n'],benchmark=out['expected_maximum_sharpe'],skewness=kw['skewness'],kurtosis=kw['kurtosis'])
    assert abs(out['dsr']-direct)<1e-15

def test_sharpe_moments_finite():
    r=np.array([-.02,.01,.03,.04,-.01,.02,.00,.05])
    m=sharpe_moments(r)
    assert m['n']==8 and all(math.isfinite(m[k]) for k in ['sharpe','skewness','kurtosis'])
