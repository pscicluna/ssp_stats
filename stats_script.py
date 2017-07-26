#!/usr/bin/env python
# -*- coding: utf-8 -*-

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
        Referred to below as Hogg+2010

    Foreman-Mackey et al, 2012, https://arxiv.org/abs/1202.3665 
        emcee: The MCMC Hammer 
        documentation for emcee is at: http://emcee.readthedocs.io/en/latest/ 
        and a quickstart guide which covers much of the latter part of 
        this tutorial can be found at: http://emcee.readthedocs.io/en/latest/user/line/



'''
from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
from astropy.table import Table
from scipy.optimize import minimize,curve_fit


def z(x,centre,spread):
    return (x-centre(x))/spread(x)
def madm(arr):
    """ Median Absolute Deviation from the Median: a "Robust" version of standard deviation.
        
        Indicates variabililty of the sample.
        https://en.wikipedia.org/wiki/Median_absolute_deviation 
    """
    arr = np.ma.array(arr).compressed() # should be faster to not use masked arrays.
    med = np.median(arr)
    return 1.4826* np.median(np.abs(arr - med))

x = np.array([2.30, 2.20, 2.35, 2.25, 2.30, 23.0, 2.25])

print("Input array: ",x)
print("Centre/Spread:")
print("Non-robust: ",np.mean(x),np.std(x))
print("Robust: ",np.median(x),madm(x))
print("z (non-robust): ",z(x,np.mean,np.std))
print("Outlier detection using z>3: ",x[z(x,np.mean,np.std) > 3.] )
print("z (robust): ",z(x,np.median,madm))
print("Outlier detection using z>3: ",x[z(x,np.median,madm) > 3.])
#exit()

#First, we read the data into a Table object for ease:
dataTable=Table.read("xses_Orich_Srinivasanetal2009.csv",format="ascii.csv")

#now extract important parameters into numpy arrays
L = dataTable['L'].data
dL= dataTable['dL'].data

x24=dataTable['x24'].data
dx24=dataTable['dx24'].data

#compute bolometric magnitudes from data
mbol=-2.5*np.log10(L) + 4.72
#print(mbol)
bins=np.arange(-8,-3,0.1)
plt.hist(mbol,bins,log=True)
bins=np.arange(-8.05,-3.05,0.1)
plt.hist(mbol,bins,log=True)
bins=np.arange(-8.0,-3.0,0.03)
plt.hist(mbol,bins,log=True)
#plt.set_yscale('log')
plt.show()
exit()
#x24=dataTable['x8'].data
#dx24=dataTable['dx8'].data

#then compute basic statistics for the data


mean = np.mean(L)
median=np.median(L)

sigma=np.std(L)




sig_madm=madm(L)


print(mean,median)
print(sigma,sig_madm)

#Now we attempt to fit a line to the data, in a few different ways
#Before we do, we will help out the fitting algorithm by taking the log of the data.
#This reduces the dynamic range, making the fit more stable over the whole range of the data.
a=x24 > 0.
xerr=dL[a]/L[a]
x=np.log10(L[a])
yerr=dx24[a]/x24[a]
y=np.log10(x24[a])
print(x,xerr)
print(y,yerr)

#First try - simple chi-squared minimisation
#def chisqfunc((a, b)):
#    model = a + b*x
#    chisq = numpy.sum(((y - model)/yerr)**2)
#    return chisq

#p0=np.array([0,0]) #some initial guess
#result=minimise(chisqfunc,p0)

#Linear least-square fit:
A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

print("""Least-squares results:
    m = {0} ± {1}
    b = {2} ± {3}
""".format(m_ls, np.sqrt(cov[1, 1]),b_ls, np.sqrt(cov[0, 0])))


#Second, LM algorithm (curve_fit) - non-linear least square fit
#def line(x,m,b):
#    return m*x+b

#p0=np.array([1,1]) #see what happens if you play with these values
#popt,pcov=curve_fit(line,x,y,p0=p0,sigma=yerr,absolute_sigma=True)

#print(popt)
#print(np.sqrt(pcov))
#print(pcov)

plt.plot(x,y,'o')
xplot=np.arange(np.min(x),np.max(x),0.3)
plt.plot(xplot,xplot*m_ls + b_ls,'--')
plt.show()

#exit()
#Third, the MCMC hammer for a few variations
import emcee

ndim=2
nwalkers=100

pos2=[[m_ls,b_ls] + np.random.randn(ndim) for i in range(nwalkers)]
pos=pos2
#emcee depends on you having defined a reasonable likelihood and prior, everything else is just brute force. We will assume flat priors today, but I suggest you read the relevant sections of Hogg+2010 to understand why this assumption should be discarded whenever possible.
#For comparison with the previous fits, we will start just by using the uncertainties on the excesses.
def lnprior(theta):
    m,b=theta
    if -10. < m < 10. and -100 < b < 100:
        return 0.0
    return -np.inf

def lnlike(theta,x,y,yerr):
    m,b=theta
    model=m*x + b 
    return 0.5 * np.sum((y - model)**2/yerr**2 + np.log(2*np.pi*yerr**2))

def lnprob(theta,x,y,yerr):
    lp=lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, yerr)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
def run_emcee(sampler,pos,ndim,labels,steps=500,prefix=""):
    print("Running MCMC...")
    sampler.run_mcmc(pos,steps, rstate0=np.random.get_state())
    print("Done.")

    plt.clf()
    fig, axes = plt.subplots(ndim, 1, sharex=True, figsize=(8, 9))
    
    
    for i in range(ndim):
        axes[i].plot(sampler.chain[:, :, i].T, color="k", alpha=0.4)
        #axes[i].yaxis.set_major_locator(MaxNLocator(5))
        #axes[i].axhline(m_true, color="#888888", lw=2)
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time.png")
    return sampler

labels=["$m$","$b$"]
results=run_emcee(sampler,pos,ndim,labels,1000,prefix="2par")

#exit()
#burnin=50

#samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
#fig = corner.corner(samples, labels=labels[0:ndim])
#fig.savefig("line-triangle.png")

#m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
#                             zip(*np.percentile(samples, [16, 50, 84],
#                                                axis=0)))
#, f_mcmc
#print("""MCMC result:
#    m = {0[0]} +{0[1]} -{0[2]}
#    b = {1[0]} +{1[1]} -{1[2]}""".format(m_mcmc, b_mcmc))
#    f = {4[0]} +{4[1]} -{4[2]} (truth: {5})



#But the real power of MCMC comes from its ability to do much more complicated things. It is possible to assume that there is some additional source of scatter that the uncertainties don't properly convey (so called "Intrinsic Scatter"). All that is required is that the likelihood is different
ndim=3
labels=["$m$","$b$","$f$"]
pos3=[[m_ls,b_ls,0.] + np.random.randn(ndim) for i in range(nwalkers)]
pos=pos3
def lnprior(theta):
    m,b,lnf=theta
    if -10. < m < 10. and -100 < b < 100 and -10. < lnf < 10.:
        return 0.0
    return -np.inf

def lnlike(theta,x,y,yerr):
    m,b,lnf=theta
    model=m*x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
results=run_emcee(sampler,pos,ndim,labels,500,prefix="3par")
#exit()
#And it is also possible to use the uncertainties on both parameters. In this case, however, we completely transform our approach - our likelihood now depends on the displacement of the points from the line. This is most easily described in terms of the angle between the x-axis and the line we are interested in.

ndim=2

def lnprior(theta):
    m,b=theta
    if -10. < m < 10. and -100 < b < 100:
        return 0.0
    return -np.inf
def lnlike(theta,x,y,xerr,yerr):
    m,b=theta
    model=m*x + b #no longer necessary, but I've left it in so you can still see what it is.
    angle=np.arctan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y + b*np.cos(angle)
    sigmasq=np.sin(angle)**2 *xerr + np.cos(angle)**2 *yerr
    return np.sum(0.5* delta / sigmasq)#0.5 * (y - model)**2/yerr**2 + np.log(2*np.pi*yerr**2)

def lnprob(theta,x,y,xerr,yerr):
    lp=lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, xerr, yerr)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y,xerr, yerr))
results=run_emcee(sampler,pos2,ndim,labels,5000,prefix="2d")

#Finally, we can include intrinsic scatter on *both* of the parameters.
def lnprior(theta):
    m,b,V=theta
    if -10. < m < 10. and -100 < b < 100 and -10. < V < 10.:
        return 0.0
    return -np.inf
def lnlike(theta,x,y,xerr,yerr):
    m,b,V=theta
    model=m*x + b
    angle=np.arctan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y + b*np.cos(angle)
    sigmasq=np.sin(angle)**2 *xerr + np.cos(angle)**2 *yerr
    return np.sum(0.5*(sigmasq + V)) - np.sum(0.5* delta / (sigmasq + V))
ndim=3
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, xerr, yerr))
results=run_emcee(sampler,pos3,ndim,labels,5000,prefix="scatter")
