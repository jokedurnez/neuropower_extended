import nibabel as nib
import cluster
import BUM
import neuropower
import peakdistribution
from nipy.algorithms.statistics.empirical_pvalue import NormalEmpiricalNull
import matplotlib.pyplot as plt
from palettable.colorbrewer.qualitative import Paired_12
import numpy as np
import scipy.stats as stats
import math

spm = nib.load("/Users/Joke/Documents/Onderzoek/Studie_7_neuropower_improved/WORKDIR/zstat1.nii").get_data()

ps = spm.flatten()
ps = [x for x in ps if x!=0]
enn = NormalEmpiricalNull(ps)
enn.learn()
spm[spm==0] = 'nan'
spm = spm-enn.mu

ps = spm.flatten()
ps = [x for x in ps if x == x]


xn = np.arange(-10,10,0.01)
twocol = Paired_12.mpl_colors
plt.figure(figsize=(7,5))
plt.hist(ps,lw=0,facecolor=twocol[0],normed=True,bins=np.arange(-2,10,0.3),label="observed distribution")
plt.xlim([-2,10])
plt.ylim([0,0.5])
plt.plot(xn,stats.norm.pdf(xn),color=twocol[1],lw=3,label="null distribution")
plt.show()

peaks = cluster.cluster(spm)
peaks['pval'] = peakdistribution.peakp(peaks.peak.tolist())
bum = BUM.bumOptim(peaks["pval"].tolist(),starts=10)
modelfit = neuropower.TFpeakfit(peaks['peak'].tolist(),bum['pi1'])


xn = np.arange(-10,10,0.01)

twocol = Paired_12.mpl_colors
plt.figure(figsize=(7,5))
plt.hist(peaks['peak'].tolist(),lw=0,facecolor=twocol[0],normed=True,bins=np.arange(-2,10,0.3),label="observed distribution")
plt.xlim([-2,10])
plt.ylim([0,0.5])
plt.plot(xn,[(1-bum["pi1"])*peakdistribution.peakdens3D(p,1) for p in xn],color=twocol[3],lw=3,label="null distribution")
plt.plot(xn,[bum["pi1"]*peakdistribution.peakdens3D(p-modelfit['delta'],1) for p in xn],color=twocol[5],lw=3,label="alternative distribution")
plt.plot(xn,neuropower.mixprobdens(modelfit["delta"],bum["pi1"],xn),color=twocol[1],lw=3,label="fitted distribution")
plt.title("histogram")
plt.xlabel("peak height")
plt.ylabel("density")
plt.legend(loc="upper right",frameon=False)
plt.show()
