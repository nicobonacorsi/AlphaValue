import math
from pathlib import Path
from alphavalue.survival_report import alpha_survival_report, render_survival_html


def test_boundary_matches_minimum_half_life():
    base=alpha_survival_report(sharpe=2,half_life=1,alpha=.05,power=.9,multiplicity=100)
    h=base['survival_frontier']['minimum_half_life']
    at=alpha_survival_report(sharpe=2,half_life=h,alpha=.05,power=.9,multiplicity=100)
    assert abs(at['information']['certifiability_ratio']-1)<1e-12


def test_more_search_makes_certification_harder():
    a=alpha_survival_report(sharpe=2,half_life=5,multiplicity=1)
    b=alpha_survival_report(sharpe=2,half_life=5,multiplicity=100)
    assert b['information']['required_information_nats']>a['information']['required_information_nats']
    assert b['information']['certifiability_ratio']<a['information']['certifiability_ratio']


def test_stronger_sharpe_increases_lifetime_information():
    a=alpha_survival_report(sharpe=1,half_life=3)
    b=alpha_survival_report(sharpe=2,half_life=3)
    assert abs(b['information']['lifetime_information_nats']/a['information']['lifetime_information_nats']-4)<1e-12


def test_feasible_report_has_constructive_nonnegative_fraction():
    r=alpha_survival_report(sharpe=3,half_life=5)
    assert r['status']=='FEASIBLE'
    assert 0<=r['economic_capacity_planning']['constructive_fixed_time_certified_value_fraction_lower']<=1


def test_html_is_self_contained(tmp_path:Path):
    r=alpha_survival_report(sharpe=2,half_life=4,multiplicity=10)
    p=render_survival_html(r,tmp_path/'report.html')
    s=p.read_text()
    assert 'Alpha Survival Report' in s and 'Certifiability ratio' in s and '<html>' in s
