#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 14:11:21 2016

@author: mints
"""
from astropy.io import fits
import numpy as np
from TheCannon.code.rave.fitspectra_rave import train
import pylab as plt

full_data = fits.open('RAVE_ready.fits')
data = full_data[1].data

"""
#0 rave_id
#1 stn
#2 snr
#3 teff
#4 eteff
#5 logg
#6 elogg
#7 meta
#8 emeta
#9 new_meta
#10 algo_conv
#11 chi2
#12 obs_sp
#13 synth_sp
#14 lambda
"""

lambdas = full_data[5].data
spec = full_data[3].data
npix = len(lambdas)
nmeta = 3
meta_names = ['Teff', 'logg', '[Fe/H]']

dataall = np.zeros((npix, len(data), 3))
metaall = np.ones((len(data), nmeta))
Ametaall = np.ones((len(data), nmeta))

metaall[:, 0] = data['Teff_K']
metaall[:, 1] = data['logg_K']
metaall[:, 2] = data['Met_N_K']
Ametaall[:, 0] = data['teff']
Ametaall[:, 1] = data['logg']
Ametaall[:, 2] = data['param_m_h']

for irow, row in enumerate(data):
    dataall[:, irow, 0] = lambdas
    dataall[:, irow, 1] = spec[irow]
    dataall[:, irow, 2] = 1./(row[2] / spec[irow])**2

for i in xrange(3):
    plt.plot(metaall[:, i], Ametaall[:, i] - metaall[:, i], '.')
    plt.title(meta_names[i])
    plt.xlabel('RAVE')
    plt.ylabel('APOGEE - RAVE')
    plt.savefig('RAVE_%d.png' % i)
    plt.clf()


train(dataall, metaall, 1, 'order_1.pickle', Ametaall,
      'Noname')
import ipdb; ipdb.set_trace()
train(dataall, metaall, 2, 'order_2.pickle', Ametaall,
      'Noname')
