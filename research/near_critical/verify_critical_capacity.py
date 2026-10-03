"""Deterministic checks of an admissible two-look rule, not an optimal solver.

Run: python verify_critical_capacity.py
Outputs: numerical_checks.csv, numerical_metadata.json, construction_check.pdf
The theorem is proved analytically; quadrature checks only this explicit rule.
"""
from pathlib import Path
import csv
import json
import math
import numpy as np
from scipy.integrate import quad
from scipy.stats import norm


def rule(alpha, beta, eps, d=1.0, tol=1e-13):
    if not (0 < alpha < 1-beta < 1 and 0 < eps < 1 and d > 0):
        raise ValueError("Require 0<alpha<1-beta<1, 0<eps<1, d>0")
    za, zb = norm.isf(alpha), norm.isf(beta)
    astar = (za + zb)**2
    a = astar + eps
    h = d*d/(4*math.log(1/eps))
    t = a-h
    if t <= 0:
        raise ValueError("Asymptotic construction needs t=A-h>0")
    ca = -a/2 + math.sqrt(a)*za
    early, terminal = ca+2*d, ca+eps/4
    vals, err, extras = [], [], []
    for drift in [-0.5, 0.5]:
        term = norm.sf((terminal-drift*a)/math.sqrt(a))
        def integrand(z):
            return norm.pdf(z, loc=drift*t, scale=math.sqrt(t))*norm.cdf(
                (terminal-z-drift*h)/math.sqrt(h))
        extra, error = quad(integrand, early, np.inf, epsabs=tol,
                            epsrel=tol, limit=300)
        vals.append(float(term+extra))
        err.append(float(error))
        extras.append(float(extra))
    early_power = float(norm.sf((early-t/2)/math.sqrt(t)))
    reward = h*early_power
    p_limit = norm.cdf(zb-2*d/math.sqrt(astar))
    return dict(alpha=alpha, beta=beta, epsilon=eps, d=d, A_star=astar,
                A=a, h=h, early_time=t, early_threshold=early,
                terminal_threshold=terminal, size=vals[0], power=vals[1],
                quadrature_error_size=err[0], quadrature_error_power=err[1],
                extra_size=extras[0], extra_power=extras[1],
                early_probability=early_power, reward_lower=reward,
                fixed_time_reward=(1-beta)*eps,
                scaled_reward=reward*math.log(1/eps),
                construction_limit=float(d*d*p_limit/4))


def main():
    root=Path(__file__).resolve().parent
    rows=[]
    # Different error contracts check the construction, not only the headline pair.
    for alpha,beta in [(0.05,0.10),(0.05/101,0.10),(0.10,0.20)]:
        for eps in [1e-1,1e-2,1e-3,1e-4,1e-5,1e-6]:
            row=rule(alpha,beta,eps)
            tighter=rule(alpha,beta,eps,tol=1e-15)
            row['tolerance_change']=max(abs(row[k]-tighter[k]) for k in ['size','power','reward_lower'])
            # QUADPACK errors are numerical diagnostics, not certified intervals.
            assert row['size'] < alpha
            assert row['power'] > 1-beta
            assert row['quadrature_error_size'] < (alpha-row['size'])/100
            assert row['quadrature_error_power'] < (row['power']-(1-beta))/100
            assert row['tolerance_change'] < 1e-12
            if eps <= 1e-3:
                assert row['reward_lower'] > row['fixed_time_reward']
            rows.append(row)
    with (root/'numerical_checks.csv').open('w',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0]));w.writeheader();w.writerows(rows)
    metadata=dict(method='one-dimensional deterministic Gaussian quadrature',
                  objective='admissible two-look construction only; not the optimum',
                  total_checks=len(rows),alpha_default=.05,beta_default=.10,
                  eps_min=1e-6,eps_max=.1,
                  caveat='Floating-point quadrature diagnostics are not interval-arithmetic certificates.',
                  theorem='This feasible construction does not compute the leading optimal constant; the revised paper establishes it analytically.',
                  capacity_solver_available_in_attached_release=False)
    (root/'numerical_metadata.json').write_text(json.dumps(metadata,indent=2)+'\n')
    import matplotlib
    matplotlib.use('Agg')
    import matplotlib.pyplot as plt
    plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,
                         'axes.spines.right':False,'pdf.fonttype':42})
    selected=sorted([r for r in rows if r['alpha']==.05],key=lambda r:r['epsilon'])
    x=np.array([r['epsilon'] for r in selected])
    y=np.array([r['reward_lower'] for r in selected])
    f=np.array([r['fixed_time_reward'] for r in selected])
    fig,ax=plt.subplots(1,2,figsize=(9,3.1))
    ax[0].loglog(x,y,'o-',color='#9b0000',label='Two-look achievable value')
    ax[0].loglog(x,f,'s--',color='#555555',label='Test once at critical time')
    ax[0].set_xlabel(r'$\varepsilon=A-A_*$');ax[0].set_ylabel('Expected information time saved')
    ax[0].legend(frameon=False,fontsize=8)
    ax[1].semilogx(x,y*np.log(1/x),'o-',color='#9b0000')
    ax[1].axhline(selected[0]['construction_limit'],color='#555555',linestyle='--',label='Limit for this construction')
    ax[1].set_xlabel(r'$\varepsilon=A-A_*$');ax[1].set_ylabel(r'$R_{\mathrm{rule}}\log(1/\varepsilon)$')
    ax[1].legend(frameon=False,fontsize=8)
    fig.tight_layout()
    fig.savefig(root/'construction_check.pdf',bbox_inches='tight')
    print(json.dumps(metadata,indent=2))
    for r in selected:
        print(f"eps={r['epsilon']:.0e}, size={r['size']:.12g}, power={r['power']:.12g}, R={r['reward_lower']:.12g}, Rlog={r['scaled_reward']:.12g}")


if __name__=='__main__':
    main()
