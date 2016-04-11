#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr 11 09:32:59 2016

@author: mints
"""

import pylab as plt
import numpy as np
from astropy.io import fits
import joblib
from itertools import cycle

full_data = fits.open('RAVE_ready.fits')
data = full_data[1].data
spectra = full_data[2].data
smooth_spectra = full_data[3].data
lambdas = full_data[5].data

rfr = joblib.load('rave_rfr.dump')

colors = cycle('krbmc')

def plot_pixel(ipixel, value):
    plt.scatter(spectra[:, ipixel], data[value], color=colors.next(),
                label=ipixel)
    #plt.scatter(smooth_spectra[:, ipixel], data[value], color='red')

#for isp , sp in enumerate(spectra):
#    print isp, min(sp), max(sp)
#for x in [-5, -3, -2, -1]:
    #data[x]
    #print data[x][0]
    #sp = spectra[x]
    #plt.plot(range(len(sp)), sp)

#plot_pixel(484, 'teff')
for ifeature, feature in enumerate(rfr.feature_importances_):
    if feature > 0.02:
        print ifeature, feature
        plot_pixel(ifeature, 'logg')
#plt.subplot(211)
#plt.scatter(smooth_spectra[:, 634], smooth_spectra[:, 182], c=data['teff'], s=50)
#plt.colorbar().set_label('T')
#plt.subplot(212)
#plt.scatter(smooth_spectra[:, 634], smooth_spectra[:, 182], c=data['logg'], s=50)
#plt.colorbar().set_label('logg')
plt.xlabel('Pixel value')
plt.ylabel('logg')
plt.legend(loc='lower right')
#plt.show()
plt.savefig('pixels_logg.png')


