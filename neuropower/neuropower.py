import scipy
import numpy as np
import neuropower
import peakdistribution

"""
Fit a exponential-truncated normal mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

def altprobdens(delta,sigma,peaks):
	out = scipy.stats.norm(delta,sigma).pdf(peaks)
	return out

def mixprobdens(delta,pi1,peaks):
        f0=[(1-pi1)*peakdistribution.peakdens3D(p,1) for p in peaks]
        fa=[pi1*(peakdistribution.peakdens3D(p-delta,1)) for p in peaks]
        f=[x + y for x, y in zip(f0, fa)]
	return(f)

def mixprobdensSLL(delta,pi1,peaks):
	f = mixprobdens(delta,pi1,peaks)
	LL = -sum(np.log(f))
	return(LL)

def TFpeakfit(peaks,pi1):
	"""Searches the maximum likelihood estimator for the mixture distribution of null and alternative"""
	start = [5]
	opt = scipy.optimize.minimize(mixprobdensSLL,start,method='L-BFGS-B',args=(pi1,peaks),bounds=((0.5,50),))
	out={'maxloglikelihood': opt.fun,
		'delta': opt.x}
	return out
