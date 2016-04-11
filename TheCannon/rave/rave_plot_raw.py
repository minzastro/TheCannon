#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 10:40:38 2016

@author: mints
"""

import pylab as plt
import cPickle as pi
import numpy as np
from astropy.io import fits

full_data = fits.open('RAVE_ready.fits')
data = full_data[1].data

fit1 = pi.load(open('order_1.pickle', 'r'))
fit2 = pi.load(open('order_2.pickle', 'r'))
t = data['teff']
logg = data['logg']
met = data['param_m_h']
spec = full_data[3].data
lambdas = full_data[5].data
c = 'rgbkc'

dataall, metaall, features, offsets, coeffs, covs, scatters, chis, chisqs = fit1
_, _, _, offsets2, coeffs2, _, _, _, _ = fit2

print coeffs
c2 = np.zeros((4, 4))
for icurrent in [10]:
    flux1 = np.zeros(len(chis))
    flux2 = np.zeros(len(chis))
    for item in xrange(len(chis)):
        flux1[item] = coeffs[item][0] + \
                      coeffs[item][1] * (t[icurrent] - offsets[0]) + \
                      coeffs[item][2] * (logg[icurrent] - offsets[1]) + \
                      coeffs[item][3] * (met[icurrent] - offsets[2])

        c2[np.triu_indices(4)] = coeffs2[item]
        par = np.array([1., t[icurrent] - offsets2[0],
                        logg[icurrent] - offsets2[1],
                        met[icurrent] - offsets2[2]])
        flux2[item] = np.sum(np.outer(par, par) * c2)
    #plt.plot(lambdas, spec[icurrent], 'r')
    plt.plot(lambdas, flux1-spec[icurrent], 'b')
    plt.plot(lambdas, flux2-spec[icurrent], 'g')
#    #for ix, x in enumerate([10, 20, 50, 100]):
#    #    plt.scatter(spec[:, x], logg, color=c[ix])
#plt.show()
#plt.plot(lambdas, coeffs[:, 1] / np.mean(coeffs[:, 1]), 'r')
#plt.plot(lambdas, coeffs2[:, 1] / np.mean(coeffs2[:, 1]), 'g')
#plt.plot(lambdas, coeffs2[:, 4] / np.mean(coeffs2[:, 4]), 'k')
#plt.plot(lambdas, coeffs[:, 1] / np.mean(coeffs[:, 1]), 'g')
#plt.plot(lambdas, coeffs[:, 2] / np.mean(coeffs[:, 2]), 'b')
#plt.plot(lambdas, coeffs[:, 3] / np.mean(coeffs[:, 3]), 'k')
plt.show()
