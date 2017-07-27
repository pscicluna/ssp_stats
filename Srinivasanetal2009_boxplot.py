#Adapted from http://blog.bharatbhole.com/creating-boxplots-with-matplotlib/
from __future__ import print_function

from astropy.table import Table
import numpy as np
import matplotlib as mpl

filename='xses_Orich_Srinivasanetal2009.csv'

data=Table.read(filename,format="ascii.csv")

#now extract important parameters into numpy arrays
lum = data['L'].data

#bolometric magnitude for entire population
mbolall=-2.5*np.log10(lum)+4.72
#for just the subset with Mbol < -5.5 mag
mbolbright=mbolall[np.where(mbolall<=-5.5)]
#these are the two arrays to plot
mbol=[mbolall,mbolbright]

mpl.use('agg')
import matplotlib.pyplot as plt


fig=plt.figure(figsize=(9,6),dpi=300)
ax=fig.add_subplot(111)
#ax.set_ylim([-9,-2.7])
bins=np.arange(-9,-3,0.1)
ax.hist(mbolall,bins,log=True)
ax.set_xlabel("M$_{bol}$ (mag)")
ax.set_ylabel("Number")
fig.savefig("MbolHist.png",bbox_inches="tight")


fig=plt.figure(figsize=(9,6))
ax=fig.add_subplot(111)
ax.set_ylim([-2.7,-9])
bp=ax.boxplot(mbol,patch_artist=True,showmeans=True)

## change outline colour, fill colour, and linewidth of the boxes
for box in bp['boxes']:
    #change outline colour
    box.set(color='#7570b3',linewidth=2)
    #change fill colour
    box.set(facecolor='#1b9e77')

## change colour and linewidth of the whiskers
for whisker in bp['whiskers']:
    whisker.set(color='#7570b3',linewidth=2)

##change colour and linewidth of the caps
for cap in bp['caps']:
    cap.set(color='#7570b3',linewidth=2)

#change colour and linewidth of the medians
for median in bp['medians']:
    median.set(color='#b2df8a',linewidth=2)

## change the style of fliers and their fill
for flier in bp['fliers']:
    flier.set(marker='o',markerfacecolor='k',alpha=0.2)

ax.set_xticklabels(['All O-AGB','Bright O-AGB'])
ax.set_ylabel("M$_{bol}$ (mag)")

fig.savefig('Srinivasanetal2009_boxplot.png',bbox_inches='tight')

