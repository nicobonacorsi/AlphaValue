"""Reproduce the full-Gaussian composite-profile power figure."""
from pathlib import Path
import json
import numpy as np
from scipy.stats import norm
import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
from gaussian_two_profile import minimax_power

root=Path(__file__).parent
data=json.loads((root/'gaussian_two_profile_results.json').read_text())
cr=np.linspace(.7,1.5,33)
a=np.sqrt(data['individual_critical_A'])
known=norm.cdf(a*np.sqrt(cr)-norm.isf(data['alpha']))
common=np.array([minimax_power(a*np.sqrt(c),data['template_correlation'],data['alpha'])['power'] for c in cr])
plt.rcParams.update({'font.family':'serif','font.size':10,'axes.spines.top':False,'axes.spines.right':False,
                     'pdf.fonttype':42,'ps.fonttype':42})
fig,ax=plt.subplots(figsize=(6.8,3.4),layout='constrained')
ax.plot(cr,known,color='#333333',lw=2,label='Known profile: Neyman-Pearson power')
ax.plot(cr,common,color='#9c1625',lw=2,label='Unknown profile: optimal worst-case power')
ax.axhline(.9,color='#777777',lw=.8,ls=':')
ax.axvline(1,color='#777777',lw=.8,ls=':')
cross=data['robust_to_individual_information_ratio']
ax.scatter([1,cross],[common[np.argmin(abs(cr-1))],.9],s=22,color='#9c1625',zorder=5)
ax.annotate('86.55% at individual CR = 1',(1,.8655488924403708),xytext=(.91,.78),
            arrowprops={'arrowstyle':'-','color':'#9c1625'},color='#9c1625',fontsize=9)
ax.annotate('Uniform 90% requires CR = 1.1268',(cross,.9),xytext=(1.04,.70),
            arrowprops={'arrowstyle':'-','color':'#9c1625'},fontsize=9)
ax.set(xlabel='Information / individual-profile critical information',ylabel='Terminal certification probability',
       ylim=(.67,.995),xlim=(.7,1.5))
ax.legend(loc='upper left',frameon=False,fontsize=8.5)
ax.grid(axis='y',alpha=.15)
fig.savefig(root/'profile_geometry.pdf')
fig.savefig(root/'profile_geometry.png',dpi=160)
np.savetxt(root/'profile_geometry.csv',np.column_stack([cr,known,common]),delimiter=',',
           header='individual_CR,known_profile_power,common_minimax_power',comments='')
