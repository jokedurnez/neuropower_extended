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

# example
'''
xn = np.arange(-10,10,0.01).tolist()
yn1a = [0.7*nulPDF(x,2,method="RFT")[0] for x in xn]
yn1b = [0.3*altPDF(x,4,1,2,method="RFT") for x in xn]
yn1t = mixprobdens(xn,0.3,4,1,2,method="RFT")
plt.plot(xn,yn1a);plt.plot(xn,yn1b);plt.ylim(0,0.3);plt.plot(xn,yn1t); plt.show()

yn2a= [0.7*nulPDF(x,method="CS")[0] for x in xn]
yn2b = [0.3*altPDF(x,4,method="CS")[0] for x in xn]
yn2t = mixprobdens(xn,0.3,4,method="CS")
plt.plot(xn,yn2a); plt.ylim(0,1);plt.plot(xn,yn2b); plt.plot(xn,yn2t);plt.show()
'''

# example
'''
 example
Load data for examples
peaks_CS = pd.read_csv("/Users/Joke/Documents/Onderzoek/Studie_7_neuropower_improved/WORKDIR/locmax2.txt",sep="\t")
peaks_CS['pval'] = [1-nulCDF(p,method="CS")[0] for p in peaks_CS.Value]
peaks_RFT = peaks_CS[peaks_CS.Value>2]
peaks_RFT['pval'] = [1-nulCDF(p,2,method="RFT")[0] for p in peaks_RFT.Value]
bumCS = BUM.bumOptim(peaks_CS.pval)
bumRFT = BUM.bumOptim(peaks_RFT.pval)

sll_RFT = mixPDF_SLL_RFT([3,1],peaks_RFT.Value.tolist(),bumRFT['pi1'],2)
sll_CS = mixPDF_SLL_CS(3,peaks_CS.Value.tolist(),bumCS['pi1'])

mix_rft = modelfit(peaks_RFT.Value.tolist(),bumRFT['pi1'],exc=2,starts=2,method="RFT")
mix_cs = modelfit(peaks_CS.Value.tolist(),bumCS['pi1'],starts=2,method="CS")
'''








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
