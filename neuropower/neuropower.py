import scipy
import scipy.stats as sst 
import numpy as np
import peakdistribution
import matplotlib.pyplot as plt
import pandas as pd

"""
Fit a exponential-truncated normal mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

def altPDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	"""
	altPDF: Returns probability density using a truncated normal
	distribution that we define as the distribution of local maxima in a
	GRF under the alternative hypothesis of activation
	parameters
	----------
	peaks: float or list of floats
		list of peak heigths
	mu: 
	sigma:

	returns
	-------
	fa: float or list 
		probability density of the peaks heights under Ha  


	!!! todo : change lists to numpy arrays
	"""
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		# assert type(sigma) is in [float, int]
		# assert sigma is not None 
		normal = sst.norm(mu,sigma)
		num = 1/sigma * normal.pdf(peaks)
		den = 1. - normal.cdf(exc)
		fa = num/den
	elif method == "CS":
		fa = [peakdistribution.peakdens3D(y-mu,1) for y in peaks]
	return fa

def nulPDF(peaks,exc=None,method="RFT"):
	#Returns probability density using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		f0 = [exc*np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS":
		f0 = [peakdistribution.peakdens3D(x,1) for x in peaks]
	return f0

def altCDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	if method == "RFT":
		ksi = (peaks-mu)/sigma
		alpha = (exc-mu)/sigma
		Fa = (scipy.stats.norm(mu,sigma).cdf(peaks) - scipy.stats.norm(mu,sigma).cdf(exc))/(1-scipy.stats.norm(mu,sigma).cdf(exc))
	elif method == "CS":
		Fa = [1-peakdistribution.peakp(y-mu)[0] for y in peaks]
	return Fa

def nulCDF(peaks,exc=None,method="RFT"):
	"""Returns cumulative  density (p-values) using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		F0 = [1-np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS":
		F0 = [1-peakdistribution.peakp(y)[0] for y in peaks]
	return F0

def mixprobdens(peaks,pi1,mu,sigma=None,exc=None,method="RFT"):
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		f0=[nulPDF(p,exc=exc,method="RFT")[0] for p in peaks]
		fa=[altPDF(p,mu,sigma=sigma,exc=exc,method="RFT") for p in peaks]
	elif method == "CS":
		f0 = [nulPDF(p,method="CS")[0] for p in peaks]
		fa = [altPDF(p,mu,method="CS")[0] for p in peaks]
	f=[(1-pi1)*x + pi1*y for x, y in zip(f0, fa)]
	return(f)

# show distributions
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


##############################I'm here ##############
'''Load data for examples
peaks_CS = pd.read_csv("/Users/Joke/Documents/Onderzoek/Studie_7_neuropower_improved/WORKDIR/locmax2.txt",sep="\t")
peaks_CS['pval'] = [1-nulCDF(p,method="CS")[0] for p in peaks_CS.Value]
peaks_RFT = peaks_CS[peaks_CS.Value>2]
peaks_RFT['pval'] = [1-nulCDF(p,2,method="RFT")[0] for p in peaks_RFT.Value]
bumCS = BUM.bumOptim(peaks_CS.pval)
bumRFT = BUM.bumOptim(peaks_RFT.pval)
'''

def mixPDF_SLL_RFT(pars,peaks,pi1,exc):
	mu = pars[0]
	sigma = pars[1]
	f = mixprobdens(peaks,pi1,mu,sigma,exc,method="RFT")
	LL = -sum(np.log(f))
	return(LL)

def mixPDF_SLL_CS(mu,peaks,pi1):
	f = [mixprobdens(x,pi1,mu,method="CS") for x in peaks]
	LL = -sum([np.log(x) for x in f])
	return(LL)

# example
'''
sll_RFT = mixPDF_SLL_RFT([3,1],peaks_RFT.Value.tolist(),bumRFT['pi1'],2)
sll_CS = mixPDF_SLL_CS(3,peaks_CS.Value.tolist(),bumCS['pi1'])
'''

def modelfit(peaks,pi1,exc=None,starts=1,method="RFT"):
	"""Searches the maximum likelihood estimator for the mixture distribution of null and alternative"""
	if method == "RFT":
		mus = np.random.uniform(exc+0.5,10,(starts,))
		sigmas = np.random.uniform(0.1,5,(starts,))
		best = []
		par = []
		for i in range(0,starts):
			opt = scipy.optimize.minimize(mixPDF_SLL_RFT,[mus[i],sigmas[i]],method='L-BFGS-B',args=(peaks,pi1,exc),bounds=((exc+0.5,50),(0.1,50)))
			best.append(opt.fun)
			par.append(opt.x)
		minind=best.index(np.nanmin(best))
		out={'maxloglikelihood': best[minind],
				'mu': par[minind][0],
				'sigma': par[minind][0]}
		if method == "CS":
			mus = np.random.uniform(2,10,(starts,))
		best = []
		par = []
		for i in range(0,starts):
			opt = scipy.optimize.minimize(mixPDF_SLL_CS,mus[i],method='L-BFGS-B',args=(peaks,pi1),bounds=((0.5,50),))
			best.append(opt.fun)
			par.append(opt.x)
		minind=best.index(np.nanmin(best))
		out={'maxloglikelihood': best[minind],
				'delta': par[minind]}
		return out

# example
'''
mix_rft = modelfit(peaks_RFT.Value.tolist(),bumRFT['pi1'],exc=2,starts=2,method="RFT")
mix_cs = modelfit(peaks_CS.Value.tolist(),bumCS['pi1'],starts=2,method="CS")
'''

def threshold():
	""" Compute the significance threshold for a given Multiple comparison procedure"""


#   
#   	thresh <- seq(from=u,to=15,length=100)
#   	cdfN <- exp(-u*(thresh-u))
#   	cdfN_RFT <- resels*exp(-thresh^2/2)*thresh^2
#   	ps <- estimates$peaks$pvalue
#   	pvalms <- sort(ps)
#   	orderpvalms <- rank(ps)
#   	FDRqval <- (orderpvalms/length(ps))*alpha
#   	pr <- ifelse(pvalms[orderpvalms] < FDRqval,1,0)
#   	FDRc <- ifelse(sum(pr)==0,0,max(FDRqval[pr==1]))
#   	cutoff.BH <- ifelse(FDRc==0,NA,thresh[min(which(cdfN<FDRc))])
#   	# compute Qvalue threshold
#   	Q <- qvalue(ps,fdr.level=alpha)
#   	cutoff.Q <- ifelse(!is.list(Q),NA,ifelse(sum(Q$significant)==0,NA,min(estimates$peaks$peaks[Q$significant==TRUE])))
#   	# compute threshold for uncorrected, FWE and RFT
#   	cutoff.UN <- thresh[min(which(cdfN<alpha))]
#   	cutoff.FWE <- thresh[min(which(cdfN<(alpha/length(estimates$peaks))))]
#   	cutoff.RFT <- thresh[min(which(cdfN_RFT<alpha))]
