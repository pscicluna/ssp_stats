from __future__ import print_function
import numpy as np
import matplotlib.pyplot as plt

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
