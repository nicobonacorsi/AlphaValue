import math
import numpy as np
import pytest

from alphavalue.data_survival import (
    analyze_edge_return_series,
    fit_exponential_edge,
    returns_only_identification_report,
)


def test_exact_exponential_decay_recovery():
    n=96
    lam=math.log(2.0)/24.0
    amp=0.01
    edge=amp*np.exp(-lam*np.arange(n))
    fit=fit_exponential_edge(edge)
    assert fit["amplitude_per_period"] == pytest.approx(amp,rel=1e-6)
    assert fit["decay_rate_per_period"] == pytest.approx(lam,rel=1e-6)
    assert fit["half_life_periods"] == pytest.approx(24.0,rel=1e-6)
    assert fit["r2"] > .999999


def test_returns_only_does_not_invent_lifetime():
    r=np.array([.01,-.005,.004,.012,-.008,.006,.003,-.002,.005,.007,.001,-.004])
    out=returns_only_identification_report(r,periods_per_unit=12)
    assert out["status"] == "LIFETIME_UNIDENTIFIED"
    assert "do not identify" in out["statement"]


def test_data_report_maps_consistent_units():
    n=120; P=12.0
    amp=.01; sigma=.02; half_life_periods=24.0
    lam=math.log(2.0)/half_life_periods
    edge=amp*np.exp(-lam*np.arange(n))
    # deterministic alternating innovation with exact positive variance
    eps=sigma*np.tile(np.array([-1.0,1.0]),n//2)
    ret=edge+eps
    out=analyze_edge_return_series(ret,edge,periods_per_unit=P,bootstrap_samples=0,multiplicity=1)
    S_expected=amp/np.std(eps,ddof=1)*math.sqrt(P)
    assert out["implied"]["instantaneous_sharpe"] == pytest.approx(S_expected,rel=1e-6)
    assert out["implied"]["half_life"] == pytest.approx(2.0,rel=1e-6)
    assert out["point_survival_report"] is not None


def test_bootstrap_is_deterministic_with_seed():
    rng=np.random.default_rng(123)
    n=100; P=12
    lam=math.log(2.0)/36
    fit=.012*np.exp(-lam*np.arange(n))
    edge=fit+rng.normal(0,.0007,n)
    ret=edge+rng.normal(0,.02,n)
    a=analyze_edge_return_series(ret,edge,periods_per_unit=P,bootstrap_samples=80,seed=7)
    b=analyze_edge_return_series(ret,edge,periods_per_unit=P,bootstrap_samples=80,seed=7)
    assert a["bootstrap_robustness"] == b["bootstrap_robustness"]
    assert a["bootstrap_robustness"]["formal_certificate"] is False


def test_constant_edge_marks_finite_lifetime_unidentified():
    n=60
    edge=np.full(n,.01)
    ret=edge+np.linspace(-.02,.02,n)
    out=analyze_edge_return_series(ret,edge,periods_per_unit=12,bootstrap_samples=0)
    assert out["decay_fit"]["finite_decay_identified"] is False
    assert out["data_status"] == "FINITE_LIFETIME_UNIDENTIFIED"
    assert out["point_survival_report"] is None


def test_survival_data_cli_end_to_end(tmp_path):
    import csv, json, subprocess, sys
    p=tmp_path/'x.csv'
    n=48; lam=math.log(2)/18; amp=.008
    edge=amp*np.exp(-lam*np.arange(n))
    eps=.015*np.tile(np.array([-1.0,1.0]),n//2)
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['edge','return'])
        for e,z in zip(edge,edge+eps): w.writerow([e,z])
    outj=tmp_path/'r.json'; outh=tmp_path/'r.html'
    cmd=[sys.executable,'-m','alphavalue','survival-data',str(p),'--edge-column','edge','--return-column','return','--periods-per-unit','12','--bootstrap-samples','20','--output-json',str(outj),'--output-html',str(outh)]
    q=subprocess.run(cmd,capture_output=True,text=True,check=True)
    obj=json.loads(q.stdout)
    assert obj['report']=='DATA_DRIVEN_ALPHA_SURVIVAL'
    assert outj.exists() and outh.exists()
    assert 'Identification boundary' in outh.read_text()


def test_survival_data_cli_returns_only_refuses_lifetime(tmp_path):
    import csv, json, subprocess, sys
    p=tmp_path/'x.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['return'])
        for x in [.01,-.01,.02,-.005,.003,.004,-.002,.006,.001,.005]: w.writerow([x])
    cmd=[sys.executable,'-m','alphavalue','survival-data',str(p),'--return-column','return','--periods-per-unit','12','--bootstrap-samples','0']
    q=subprocess.run(cmd,capture_output=True,text=True,check=True)
    obj=json.loads(q.stdout)
    assert obj['status']=='LIFETIME_UNIDENTIFIED'


def test_returns_only_has_explicit_gaussian_sharpe_lower_bound():
    from alphavalue.data_survival import gaussian_iid_sharpe_lower_bound
    r=np.array([.012,.004,-.003,.009,.015,-.002,.006,.011,.005,.008,.001,.007]*4)
    out=gaussian_iid_sharpe_lower_bound(r,periods_per_unit=12,confidence=.95)
    assert out["annualized_sharpe_lower"] < out["annualized_sharpe_point"]
    assert out["formal_under_declared_model"] is True
    assert out["dependence_robust"] is False


def test_returns_only_html_is_generated(tmp_path):
    from alphavalue.data_survival import render_returns_only_survival_html
    r=np.array([.01,-.004,.008,.003,-.002,.006,.009,.001,.004,.005,.002,.007])
    report=returns_only_identification_report(r,periods_per_unit=12)
    p=tmp_path/'returns.html'
    render_returns_only_survival_html(report,p)
    txt=p.read_text()
    assert 'LIFETIME_UNIDENTIFIED' in txt
    assert '95% lower Sharpe' in txt


def test_survival_alias_returns_only_with_html(tmp_path):
    import csv, json, subprocess, sys
    p=tmp_path/'x.csv'
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['return'])
        for x in [.01,-.01,.02,-.005,.003,.004,-.002,.006,.001,.005,.007,.002]: w.writerow([x])
    outh=tmp_path/'r.html'
    q=subprocess.run([sys.executable,'-m','alphavalue','survival',str(p),'--return-column','return','--periods-per-unit','12','--bootstrap-samples','0','--output-html',str(outh)],capture_output=True,text=True,check=True)
    obj=json.loads(q.stdout)
    assert obj['status']=='LIFETIME_UNIDENTIFIED'
    assert 'historical_strength' in obj
    assert outh.exists()


def test_survival_alias_edge_mode(tmp_path):
    import csv, json, subprocess, sys
    p=tmp_path/'x.csv'; n=48; lam=math.log(2)/18; amp=.008
    edge=amp*np.exp(-lam*np.arange(n)); eps=.015*np.tile(np.array([-1.0,1.0]),n//2)
    with p.open('w',newline='') as f:
        w=csv.writer(f); w.writerow(['edge','return'])
        for e,z in zip(edge,edge+eps): w.writerow([e,z])
    q=subprocess.run([sys.executable,'-m','alphavalue','survival',str(p),'--edge-column','edge','--return-column','return','--periods-per-unit','12','--bootstrap-samples','0'],capture_output=True,text=True,check=True)
    obj=json.loads(q.stdout)
    assert obj['report']=='DATA_DRIVEN_ALPHA_SURVIVAL'
    assert obj['point_survival_report'] is not None
