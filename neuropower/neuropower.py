import scipy
import numpy as np
import neuropower
import peakdistribution

"""
Fit a exponential-truncated normal mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

def altPDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	#Returns probability density using a truncated normal distribution that we define as the distribution of local maxima in a GRF under the alternative hypothesis of activation
	if method == "RFT":
		num = 1/sigma*scipy.stats.norm(mu,sigma).pdf(peaks)
		den = 1-scipy.stats.norm(mu,sigma).cdf(exc)
		fa = num/den
	elif method == "CS":
		fa = [peakdistribution.peakdens3D(y-mu,1) for y in peaks]
	return fa

# example
'''
xn = np.arange(-10,10,0.01)
yn=altPDF(xn,2,method="CS")
yn2 = altPDF(xn,3,1,2,method="RFT")
plt.plot(xn,yn); plt.ylim(0,2);plt.plot(xn,yn2); plt.show()

'''

def nulPDF(peaks,exc=None,method="RFT"):
	#Returns probability density using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005
	if method == "RFT":
		f0 = [exc*np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS":
		f0 = [peakdistribution.peakdens3D(x,1) for x in peaks]
	return f0

# example
'''
xn = np.arange(-10,10,0.01)
yn=nulPDF(xn,method="CS")
yn2 = nulPDF(xn,2,method="RFT")
plt.plot(xn,yn); plt.ylim(0,2);plt.plot(xn,yn2); plt.show()
'''

def altCDF(peaks,mu,sigma=None,exc=None,method="RFT"):
	if method == "RFT":
		ksi = (peaks-mu)/sigma
		alpha = (exc-mu)/sigma
		Fa = (scipy.stats.norm(mu,sigma).cdf(peaks) - scipy.stats.norm(mu,sigma).cdf(exc))/(1-scipy.stats.norm(mu,sigma).cdf(exc))
	elif method == "CS":
		Fa = [1-peakdistribution.peakp(y-mu)[0] for y in peaks]
	return Fa

# example
'''
xn = np.arange(-10,10,0.01)
yn=altCDF(xn,2,method="CS")
yn2 = altCDF(xn,2,1,2,method="RFT")
plt.plot(xn,yn); plt.ylim(0,2);plt.plot(xn,yn2); plt.show()
'''

def nulCDF(peaks,exc=None,method="RFT"):
	"""Returns cumulative  density (p-values) using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	peaks = (peaks,) if not isinstance(peaks, (tuple, list)) else peaks
	if method == "RFT":
		F0 = [1-np.exp(-exc*(x-exc)) for x in peaks]
	elif method == "CS":
		F0 = [1-peakdistribution.peakp(y)[0] for y in peaks]
	return F0

# example
'''
xn = np.arange(-10,10,0.01).tolist()
yn = nulCDF(xn,method="CS")
yn2 = nulCDF(xn,2,method="RFT")
plt.plot(xn,yn); plt.ylim(0,2);plt.plot(xn,yn2); plt.show()
'''

##############################I'm here ##############


def mixprobdens(peaks,pi1,mu,sigma=None,exc=None,method="RFT"):
	if method == "RFT":
		f0=[(1-pi1)*nulPDF(peaks,exc=exc,method="RFT") for p in peaks]
		fa=[pi1*altPDF(peaks,mu,sigma=sigma,exc=exc,method="RFT") for p in peaks]
	elif method == "CS":
		f0 = [(1-pi1)*nulPDF(peaks,method="CS") for p in peaks]
		fa = [pi1*altPDF(peaks,mu,method="CS") for p in peaks]
	f=[x + y for x, y in zip(f0, fa)]
	return(f)

def mixPDF_SLL_RFT(mu,sigma,peaks,pi1,exc):
	f = mixprobdens(peaks,pi1,mu,sigma,exc,method="RFT")
	LL = -sum(np.log(f))
	return(LL)

def mixPDF_SLL_CS(mu,peaks,pi1):
	f = [mixprobdens(x,pi1,mu,method="CS") for x in y]
	LL = -sum([np.log(x) for x in f])
	return(LL)

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

modelfit(peaks.pval.tolist(),0.5,method="CS")


def threshold():
	""" Compute the significance threshold for a given Multiple comparison procedure"""



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
