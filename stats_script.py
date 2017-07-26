'''A short statistics tutorial

This script was used to demonstrate some basic statistical techniques
to students in the 2017 ASIAA stummer program. The focus was on simple
robust statistics and on fitting a line to data, under conditions that
are typical of astronomical datasets.

'''

import numpy as np
from astropy.table import Table

#First, we read the data into a Table object for ease:
dataTable=Table.read("xses_Orich_Srinivasanetal2009.csv",format="ascii.csv")

#now extract important parameters into numpy arrays
L = dataTable['L(Lsun)'].data
