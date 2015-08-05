import scipy
import numpy as np

"""
Fit a exponential-truncated normal mixture model to a list of peak height T-values.
The model is introduced in the HBM poster:
http://www2.warwick.ac.uk/fac/sci/statistics/staff/academic-research/nichols/presentations/ohbm2015/Durnez-PeakPower-OHBM2015.pdf
"""

def nulcumdens(exc,peaks):
	"""Returns cumulative  density (p-values) using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	v = exc
	u = peaks - v
	F0 = 1-((u+v)**2*np.exp(-(u+v)**2/2)/(v**2*np.exp(-v**2/2)))
	return F0

def nulprobdens(exc,peaks):
	"""Returns probability density using the exponential function that defines the distribution of local maxima in a GRF under the null hypothesis of no activation as introduced in Cheng & Schwartzman, 2005"""
	v = exc
	u = peaks - v
	f0 = (2+(u+v)**2)*(u+v)*np.exp(-(u+v)**2/2)/(v**2*np.exp(-v**2/2))
	return f0

def altprobdens(mu,sigma,exc,peaks):
	"""Returns probability density using a truncated normal distribution that we define as the distribution of local maxima in a GRF under the alternative hypothesis of activation"""
	num = 1/sigma*scipy.stats.norm(mu,sigma).pdf(peaks)
	den = 1-scipy.stats.norm(mu,sigma).cdf(exc)
	fa = num/den
	return fa

def altcumdens(mu,sigma,exc,peaks):
	ksi = (peaks-mu)/sigma
	alpha = (exc-mu)/sigma
	Fa = (scipy.stats.norm(mu,sigma).cdf(peaks) - scipy.stats.norm(mu,sigma).cdf(exc))/(1-scipy.stats.norm(mu,sigma).cdf(exc))
	return Fa

def mixprobdens(mu,sigma,pi1,exc,peaks):
	"""Returns probability density using mixture of an exponential function and a truncated normal distribution, weighted by pi1 (=1-pi0)"""
	f0 = nulprobdens(exc,peaks)
	fa = altprobdens(mu,sigma,exc,peaks)
	f = (1-pi1)*f0 + pi1*fa
	return(f)

def mixprobdensSLL(pars,pi1,exc,peaks):
	"""Returns the negative of the sum of the loglikelihood of the mixture distribution"""
	mu=pars[0]
	sigma=pars[1]
	f = mixprobdens(mu,sigma,pi1,exc,peaks)
	LL = -sum(np.log(f))
	return(LL)

def peakmixmodfit(peaks,pi1,exc):
	"""Searches the maximum likelihood estimator for the mixture distribution of null and alternative"""
	start = [5,0.5]
	opt = scipy.optimize.minimize(mixprobdensSLL,start,method='L-BFGS-B',args=(pi1,exc,peaks),bounds=((2.5,50),(0.1,50)))
	out={'maxloglikelihood': opt.fun,
		'mu': opt.x[0],
		'sigma': opt.x[1]}
	return out

def peakcutoff(peaks,exc,resels,alpha):
	thrange = np.arange(exc,15,0.0001)
	pUN = 1-nulcumdens(thrange)
	pRFT = resels*np.exp(-exc**2/2)*exc**2
	# cutoff uncorrected
	cutUN = thrange[np.min(np.where(pUN<0.05))]
