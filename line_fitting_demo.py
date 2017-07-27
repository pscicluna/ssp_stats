#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function


import numpy as np
import matplotlib.pyplot as plt
import corner
from astropy.table import Table,Column
from scipy.optimize import minimize,curve_fit

data=Table.read("test_data.csv",format="ascii.csv")
x=data['x']
xerr=data['sigma_x']
y=data['y']
yerr=data['sigma_y']
plt.errorbar(x,y,xerr=xerr,yerr=yerr,fmt='o')
plt.savefig("data.png")

A = np.vstack((np.ones_like(x), x)).T
C = np.diag(yerr * yerr)
cov = np.linalg.inv(np.dot(A.T, np.linalg.solve(C, A)))
b_ls, m_ls = np.dot(cov, np.dot(A.T, np.linalg.solve(C, y)))

print("""Least-squares results:
    m = {0} ± {1}
    b = {2} ± {3}
""".format(m_ls, np.sqrt(cov[1, 1]),b_ls, np.sqrt(cov[0, 0])))

xplot=np.arange(np.min(x),np.max(x),0.3)
plt.plot(xplot,xplot*m_ls + b_ls,'--',linewidth=3)
plt.savefig("least_square.png")
import emcee

ndim=2
nwalkers=100

pos2=[[m_ls,b_ls] + np.random.randn(ndim) for i in range(nwalkers)]
pos=pos2
#emcee depends on you having defined a reasonable likelihood and prior, everything else is just brute force. We will assume flat priors today, but I suggest you read the relevant sections of Hogg+2010 to understand why this assumption should be discarded whenever possible.
#For comparison with the previous fits, we will start just by using the uncertainties on the excesses.
def lnprior(theta):
    m,b=theta
    if -10. < m < 10. and -20 < b < 0:
        return 0.0
    return -np.inf

def lnlike(theta,x,y,yerr):
    m,b=theta
    model=m*x + b 
    return -0.5 * np.sum((((y - model)**2)/(yerr**2)))

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
        axes[i].set_ylabel(labels[i])

    fig.tight_layout(h_pad=0.0)
    fig.savefig(prefix+"line-time.png")
    return sampler

labels=["$m$","$b$"]
results=run_emcee(sampler,pos,ndim,labels,1000,prefix="2par")

#exit()
def mcmc_results(sampler,ndim,percentiles=[16, 50, 84],burnin=200,labels="",prefix=""):

    samples = sampler.chain[:, burnin:, :].reshape((-1, ndim))
    print(samples.shape)

    fig = corner.corner(samples, labels=labels[0:ndim])
    fig.savefig(prefix+"line-triangle.png")
    credible_interval=[]
    for i in range(ndim):
        credible_interval.append(np.percentile(samples[:,i], percentiles))
        credible_interval[i][2] -= credible_interval[i][1]
        credible_interval[i][0] = credible_interval[i][1] - credible_interval[i][0]
        #m_mcmc, b_mcmc = map(lambda v: (v[1], v[2]-v[1], v[1]-v[0]),
        #                 zip(*np.percentile(samples, percentiles,
        #                                    axis=0)
        #                     )
        #                 )
    print(quantiles)
    #exit()
    print("MCMC results:")
    for i in range(ndim):
        print("{0}  = {1[1]} + {1[2]} - {1[0]}".format(labels[i],credible_interval[i]))

    #now produce output plots of the distribution of lines

    fig=plt.figure()
    ax=fig.add_subplot(111)
    xplot=np.arange(-1,7,0.3)
    try:
        for m,b in samples[np.random.randint(len(samples),size=1000),0:2]:
            ax.plot(xplot,m*xplot+b,color="k",alpha=0.02)
    except:
        for m,b in samples[np.random.randint(len(samples),size=1000)]:
            ax.plot(xplot,m*xplot+b,color="k",alpha=0.02)
    ax.errorbar(x,y,xerr=xerr,yerr=yerr,fmt="ob")
    ax.set_xlim([-1,7])
    ax.set_ylim([-10,10])
    fig.savefig(prefix+"line-mcmc.png")

mcmc_results(results,ndim,labels=labels,prefix="2par")

#But the real power of MCMC comes from its ability to do much more complicated things. It is possible to assume that there is some additional source of scatter that the uncertainties don't properly convey (e.g. that the uncertainties are underestimated by some unknown amount). All that is required is that the likelihood is different
ndim=3
labels=["$m$","$b$","$f$"]
pos3=[[m_ls,b_ls,0.] + np.random.randn(ndim) for i in range(nwalkers)]
pos=pos3
def lnprior(theta):
    m,b,lnf=theta
    if -10. < m < 10. and -20 < b < 0 and -10. < lnf < 10.:
        return 0.0
    return -np.inf

def lnlike(theta,x,y,yerr):
    m,b,lnf=theta
    model=m*x + b
    inv_sigma2 = 1.0/(yerr**2 + model**2*np.exp(2*lnf))
    return -0.5*(np.sum((y-model)**2*inv_sigma2 - np.log(inv_sigma2)))

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, yerr))
results=run_emcee(sampler,pos,ndim,labels,1000,prefix="3par")
mcmc_results(results,ndim,labels=labels,prefix="3par")

#And it is also possible to use the uncertainties on both parameters. In this case, however, we completely transform our approach - our likelihood now depends on the displacement of the points from the line. This is most easily described in terms of the angle between the x-axis and the line we are interested in.

ndim=2

def lnprior(theta):
    m,b=theta
    if -10. < m < 10. and -20 < b < 0:
        return 0.0
    return -np.inf
def lnlike(theta,x,y,xerr,yerr):
    m,b=theta
    model=m*x + b #no longer necessary, but I've left it in so you can still see what it is.
    angle=np.arctan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y - b*np.cos(angle)
    sigmasq=xerr*np.sin(angle)**2 + yerr*np.cos(angle)**2
    return -np.sum(0.5* delta**2 / sigmasq)#0.5 * (y - model)**2/yerr**2 + np.log(2*np.pi*yerr**2)

def lnprob(theta,x,y,xerr,yerr):
    lp=lnprior(theta)
    if not np.isfinite(lp):
        return -np.inf
    return lp + lnlike(theta, x, y, xerr, yerr)

sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y,xerr, yerr))
results=run_emcee(sampler,pos2,ndim,labels,1000,prefix="2d")
mcmc_results(results,ndim,labels=labels,prefix="2d")

#Finally, we can include intrinsic scatter on the data. This is used to capture e.g. that there is some other physical process that is not included in the model than contributes to the data.
def lnprior(theta):
    m,b,V=theta
    if -10. < m < 10. and -20 < b < 0 and 0. < V < 5:# < 100.:
        return 0.0
    return -np.inf
def lnlike(theta,x,y,xerr,yerr):
    m,b,V=theta
    model=m*x + b
    angle=np.arctan(m)
    delta=-1.*np.sin(angle)*x + np.cos(angle)*y - b*np.cos(angle)
    sigmasq=xerr*np.sin(angle)**2 + yerr*np.cos(angle)**2
    return -np.sum(0.5*np.log(sigmasq + V)) - np.sum(0.5* delta**2 / (sigmasq + V))

ndim=3
sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob, args=(x, y, xerr, yerr))
results=run_emcee(sampler,pos3,ndim,labels,1000,prefix="scatter")
mcmc_results(results,ndim,labels=labels,prefix="scatter")

#plt.show()

