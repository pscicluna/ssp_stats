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

#First, we read the data into a Table object for ease:
dataTable=Table.read("xses_Orich_Srinivasanetal2009.csv",format="ascii.csv")

#now extract important parameters into numpy arrays
L = dataTable['L(Lsun)'].data


#then compute basic statistics for the data




#Now we attempt to fit a line to the data, in a few different ways


#First try - simple chi-squared minimisation


#Second, LM algorithm (curve_fit)


#Third, the MCMC hammer for a few variations
import emcee



#emcee depends on you having defined a reasonable likelihood and prior, everything else is just brute force. We will assume flat priors today, but I suggest you read the relevant sections of Hogg 2010 to understand why this assumption should be discarded whenever possible.
#For comparison with the previous fits, we will start just by using hte uncertainties on the excesses.



#But the real power of MCMC comes from its ability to do much more complicated things. It is possible to assume that there is some additional source of scatter that the uncertainties don't properly convey (so called "Intrinsic Scatter"). All that is required is that the likelihood is different


#And it is also possible to use the uncertainties on both parameters




#Finally, we can include intrinsic scatter on *both* of the parameters.



