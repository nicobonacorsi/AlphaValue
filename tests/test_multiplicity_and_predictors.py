import math
import numpy as np
import pytest

from alphavalue.multiplicity import e_bh, nig_fixed_cut_coefficient_evalue
from alphavalue.predictors import fixed_ewma


def test_e_bh_known_threshold_and_arbitrary_input_order():
    # m=5, alpha=.1. For k=2 the cutoff is 25; two e-values clear it.
    r=e_bh([1,40,2,30,4],alpha=.1)
    assert r['k']==2
    assert r['threshold']==pytest.approx(25.)
    assert r['rejected']==[False,True,False,True,False]
    assert 'Fixed-cut' in r['scope']


def test_e_bh_no_rejections():
    r=e_bh([1,2,3],alpha=.05)
    assert r['k']==0 and r['threshold'] is None
    assert r['rejected']==[False,False,False]


def test_fixed_cut_composite_variance_evalue_is_nonnegative():
    rng=np.random.default_rng(19)
    x=rng.normal(size=200)
    y=.45*x+rng.normal(scale=.8,size=200)
    e=nig_fixed_cut_coefficient_evalue(x,y,beta0=0.)
    assert e>=0 and not math.isnan(e)
    assert e>1


def test_fixed_ewma_has_declared_half_life_recursion():
    x=np.array([1.,0.,0.,0.,0.,0.,0.])
    y=fixed_ewma(x,half_life=6.)
    decay=2**(-1/6)
    assert y[1]==pytest.approx(decay)
    assert y[6]==pytest.approx(.5)
