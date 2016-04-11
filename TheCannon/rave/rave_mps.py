#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:14:26 2016

@author: mints
"""

from astropy.io import fits
from scipy.io import readsav
import numpy as np
import cPickle as pickle

#spectra = readsav('2009K1_parameters.save').items()[0][1]
spectra = pickle.load(open('RAVE_APOGEE_GCS.pickle', 'r'))

matches = fits.open('RAVE_APOGEE_GCS.fits')[1]
rave_ids = matches.data['RAVE_OBS_ID']
rave_ids = np.asarray(rave_ids, dtype='|S20')
indexes = np.in1d(spectra['rave_id'], rave_ids)

#import ipdb; ipdb.set_trace()
subset = []
for ind, _ in enumerate(indexes):
    m = matches.data[matches.data['RAVE_OBS_ID'] == spectra[ind][0]][0]
    #print m, type(spectra[ind]
    subset.append(tuple(spectra[ind]) + tuple(m))

pickle.dump(subset, open('RAVE_APOGEE_GCS.pickle', 'w'))


def read_rave_data(filename):
    inputf = readsav(filename)
    items = inputf.items()
    data = items[0][1]
    wl = data['lambda'][0]  # assuming they're all the same...
    sp = data['obs_sp']  # (75437, 839)
    test_flux = np.zeros((len(sp), len(sp[0])))
    for jj in xrange(len(sp)):
        test_flux[jj, :] = sp[jj]
    snr = np.array(data['snr'])
    test_ivar = (snr[:, None] / test_flux)**2
    bad = np.logical_or(np.isnan(test_ivar), np.isnan(test_flux))
    test_ivar[bad] = 0.
    test_flux[bad] = 0.
    return (test_flux, test_ivar, wl)
