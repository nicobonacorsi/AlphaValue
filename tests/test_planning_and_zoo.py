import csv
import json
from pathlib import Path

import numpy as np
import pytest

from alphavalue.planning import canonical_one_percent_history_floor
from alphavalue.zoo import ZooColumns, run_zoo, write_zoo_csv


def _series(seed, n, phi, theta, sx, sy):
    rng=np.random.default_rng(seed)
    x=np.empty(n+1)
    x[0]=rng.normal(scale=sx/np.sqrt(1-phi*phi))
    for t in range(n):
        x[t+1]=phi*x[t]+rng.normal(scale=sx)
    y=theta*x[:-1]+rng.normal(scale=sy,size=n)
    return x[:-1],x[1:],y


def test_canonical_history_floor_is_explicitly_necessary_not_sufficient():
    p=canonical_one_percent_history_floor(12,20,period_unit='months')
    assert p['necessary_history_per_deployment_horizon']==pytest.approx(3.670443075025288)
    assert p['necessary_history_floor_integer']==45
    assert p['additional_periods_to_floor']==25
    assert p['status']=='necessary_not_sufficient'
    assert 'not a sufficient' in p['scope']


def test_zoo_runs_multiple_frozen_signals_and_writes_table(tmp_path):
    path=tmp_path/'zoo.csv'
    specs=[('A',11,.62,.36,.45,.65,2.0),('B',12,.35,.12,.65,.90,1.5)]
    with path.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['name','signal','next_signal','next_return','trading_cost'])
        for name,seed,phi,theta,sx,sy,cost in specs:
            x,xn,y=_series(seed,900,phi,theta,sx,sy)
            for a,b,c in zip(x,xn,y):
                w.writerow([name,a,b,c,cost])
    r=run_zoo(path,gamma=1.,horizon=8,alpha=.10,hurdle=.0001,efficiency_tolerance=.5,
              phi_bounds=(-.95,.95),theta_cells=3,phi_cells=6,period_unit='months')
    assert r['n_signals']==2
    assert {x['name'] for x in r['results']}=={'A','B'}
    assert sum(r['decision_counts'].values())==2
    assert 'necessary barrier' in r['lifetime_scope']
    out=tmp_path/'results.csv'; write_zoo_csv(r,out)
    rows=list(csv.DictReader(out.open()))
    assert len(rows)==2
    assert 'canonical_1pct_history_floor' in rows[0]


def test_zoo_rejects_cost_that_changes_inside_signal(tmp_path):
    p=tmp_path/'bad.csv'
    p.write_text('name,signal,next_signal,next_return,trading_cost\nA,1,1,1,1\nA,1,1,1,2\n')
    with pytest.raises(ValueError,match='constant within'):
        run_zoo(p,gamma=1.,horizon=2,theta_cells=2,phi_cells=2)
