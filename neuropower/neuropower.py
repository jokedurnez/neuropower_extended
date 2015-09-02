import scipy
import scipy.stats as sst 
import numpy as np
import peakdistribution
import matplotlib.pyplot as plt
import pandas as pd

"""
Fit a mixture model to a list of peak height T-values.
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
	#Returns probability density of the alternative peak distribution
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
	#Returns probability density of the null peak distribution
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		f0 = [exc*np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS"
		f0 = [peakdistribution.peakdens3D(x,1) for x in peaks]
	return f0

def altCDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	# Returns the CDF of the alternative peak distribution
	if method == "RFT":
		ksi = (peaks-mu)/sigma
		alpha = (exc-mu)/sigma
		Fa = (scipy.stats.norm(mu,sigma).cdf(peaks) - scipy.stats.norm(mu,sigma).cdf(exc))/(1-scipy.stats.norm(mu,sigma).cdf(exc))
	elif method == "CS":
		Fa = [1-peakdistribution.peakp(y-mu)[0] for y in peaks]
	return Fa

def nulCDF(peaks,exc=None,method="RFT"):
	# Returns the CDF of the null peak distribution
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		F0 = [1-np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS":
		F0 = [1-peakdistribution.peakp(y)[0] for y in peaks]
	return F0

def mixprobdens(peaks,pi1,mu,sigma=None,exc=None,method="RFT"):
	# returns the PDF of the mixture of null and alternative distribution
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		f0=[nulPDF(p,exc=exc,method="RFT")[0] for p in peaks]
		fa=[altPDF(p,mu,sigma=sigma,exc=exc,method="RFT") for p in peaks]
	elif method == "CS":
		f0 = [nulPDF(p,method="CS")[0] for p in peaks]
		fa = [altPDF(p,mu,method="CS")[0] for p in peaks]
	f=[(1-pi1)*x + pi1*y for x, y in zip(f0, fa)]
	return(f)

def mixPDF_SLL_RFT(pars,peaks,pi1,exc):
	# Returns the negative sum of the loglikelihood of the PDF with RFT
	mu = pars[0]
	sigma = pars[1]
	f = mixprobdens(peaks,pi1,mu,sigma,exc,method="RFT")
	LL = -sum(np.log(f))
	return(LL)

def mixPDF_SLL_CS(mu,peaks,pi1):
	# Returns the negative sum of the loglikelihood of the PDF with Cheng & Schwartzmans peak distribution
	f = [mixprobdens(x,pi1,mu,method="CS") for x in peaks]
	LL = -sum([np.log(x) for x in f])
	return(LL)

def modelfit(peaks,pi1,exc=None,starts=1,method="RFT"):
	# Searches the maximum likelihood estimator for the mixture distribution of null and alternative
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

'''
def threshold():
	# Compute the significance threshold for a given Multiple comparison procedure



	thresh <- seq(from=u,to=15,length=100)
	cdfN <- exp(-u*(thresh-u))
	cdfN_RFT <- resels*exp(-thresh^2/2)*thresh^2
	ps <- estimates$peaks$pvalue
	pvalms <- sort(ps)
	orderpvalms <- rank(ps)
	FDRqval <- (orderpvalms/length(ps))*alpha
	pr <- ifelse(pvalms[orderpvalms] < FDRqval,1,0)
	FDRc <- ifelse(sum(pr)==0,0,max(FDRqval[pr==1]))
	cutoff.BH <- ifelse(FDRc==0,NA,thresh[min(which(cdfN<FDRc))])
	# compute Qvalue threshold
	Q <- qvalue(ps,fdr.level=alpha)
	cutoff.Q <- ifelse(!is.list(Q),NA,ifelse(sum(Q$significant)==0,NA,min(estimates$peaks$peaks[Q$significant==TRUE])))
	# compute threshold for uncorrected, FWE and RFT
	cutoff.UN <- thresh[min(which(cdfN<alpha))]
	cutoff.FWE <- thresh[min(which(cdfN<(alpha/length(estimates$peaks))))]
	cutoff.RFT <- thresh[min(which(cdfN_RFT<alpha))]
'''
