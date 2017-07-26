'''A short statistics tutorial

This script was used to demonstrate some basic statistical techniques
to students in the 2017 ASIAA stummer program. The focus was on simple
robust statistics and on fitting a line to data, under conditions that
are typical of astronomical datasets.

This leans very heavily on previous works and tutorials. Of particular
importance are:

    Hogg, Bovy and Lang, 2010, https://arxiv.org/abs/1008.4686
        Data analysis recipes: Fitting a model to data
        This is a key resource for any astronomer who cares about statistics

    Foreman-Mackey et al, 2012, https://arxiv.org/abs/1202.3665 emcee:
        The MCMC Hammer documentation for emcee is at: http://emcee.readthedocs.io/en/latest/ 
        and a quickstart guide which covers much of the latter part of 
        this tutorial can be found at: http://emcee.readthedocs.io/en/latest/user/line/



'''

import numpy as np
from astropy.table import Table
from scipy.optimize import minimize,curve_fit

#First, we read the data into a Table object for ease:
dataTable=Table.read("xses_Orich_Srinivasanetal2009.csv",format="ascii.csv")

#now extract important parameters into numpy arrays
L = dataTable['L(Lsun)'].data


#then compute basic statistics for the data


mean = np.mean()
median=np.median()

sigma=np.stddev()


def mad(arr):
    """ Median Absolute Deviation: a "Robust" version of standard deviation.
        
        Indicates variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return np.median(np.abs(arr - med))

sig_mad=mad()

#Now we attempt to fit a line to the data, in a few different ways
#Before we do, we will help out the fitting algorithm by taking the log of the data.
#This reduces the dynamic range, making the fit more stable over the whole range of the data.

#First try - simple chi-squared minimisation
def chisqfunc((a, b)):
    model = a + b*y
    chisq = numpy.sum(((y - model)/yerr)**2)
    return chisq

p0=np.array([0,0]) #some initial guess
result=minimise(chisqfunc,p0)

#Linear least-square fit:
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

print("""Least-squares results:
    m = {0} ± {1} (truth: {2})
    b = {3} ± {4} (truth: {5})
""".format(m_ls, np.sqrt(cov[1, 1]), m_true, b_ls, np.sqrt(cov[0, 0]), b_true))




#Second, LM algorithm (curve_fit) - non-linear least square fit
def line(x,m,b):
    return m*x+b

p0=np.array([1,1]) #see what happens if you play with these values
popt,pcov=curve_fit(line,x,y,p0=p0,sigma=yerr,absolute_sigma=True)


#Third, the MCMC hammer for a few variations
import emcee

ndim=2
nwalkers=100

pos=
#emcee depends on you having defined a reasonable likelihood and prior, everything else is just brute force. We will assume flat priors today, but I suggest you read the relevant sections of Hogg 2010 to understand why this assumption should be discarded whenever possible.
#For comparison with the previous fits, we will start just by using the uncertainties on the excesses.
def lnlike(theta,x,y,yerr):
    m,b=theta
    model=m*x + b 
    return 0.5 * np.sum((y - model)**2/yerr**2 + np.log(2*np.pi*yerr**2))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, yerr))


#But the real power of MCMC comes from its ability to do much more complicated things. It is possible to assume that there is some additional source of scatter that the uncertainties don't properly convey (so called "Intrinsic Scatter"). All that is required is that the likelihood is different
ndim=3
def lnlike(theta,x,y,yerr):
    m,b,lnfy=theta
    model=m*x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, yerr))

#And it is also possible to use the uncertainties on both parameters. In this case, however, we completely transform our approach - our likelihood now depends on the displacement of the points from the line. This is most easily described in terms of the angle between the x-axis and the line we are interested in.
ndim=2
def lnlike(theta,x,y,xerr,yerr):
    m,b=theta
    model=m*x + b #no longer necessary, but I've left it in so you can still see what it is.
    angle=np.atan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y + b*np.cos(angle)
    sigmasq=np.sin(angle)**2 *xerr + np.cos(angle)**2 *yerr
    return np.sum(0.5* delta / sigmasq)#0.5 * (y - model)**2/yerr**2 + np.log(2*np.pi*yerr**2)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y,xerr, yerr))

#Finally, we can include intrinsic scatter on *both* of the parameters.
def lnlike(theta,x,y,yerr):
    m,b,scatter=theta
    model=m*x + b
    angle=np.atan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y + b*np.cos(angle)
    sigmasq=np.sin(angle)**2 *xerr + np.cos(angle)**2 *yerr
    return np.sum(0.5*(sigmasq + V)) - np.sum(0.5* delta / (sigmasq + V))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnlike, args=(x, y, yerr))

