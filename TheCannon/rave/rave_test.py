#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Mon Apr  4 15:01:12 2016

@author: mints
"""
from astropy.io import fits
import numpy as np
import cPickle as pi
from TheCannon.code.rave.fitspectra_rave import infer_labels, infer_labels_nonlinear

data = pi.load(open('RAVE_APOGEE_GCS.pickle', 'r'))

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
fit1 = pi.load(open('order_1.pickle', 'r'))
fit2 = pi.load(open('order_2.pickle', 'r'))
t = np.array([row[3] for row in data])
logg = np.array([row[5] for row in data])
met = np.array([row[7] for row in data])
spec = np.array([row[12] for row in data])
lambdas = data[0][14]
c = 'rgbkc'

N = 10
npix = len(data[0][14])

dataall = np.zeros((npix, N, 3))
result = np.zeros((N, 3))
result_apo = np.zeros((N, 3))
for irow, row in enumerate(data[:N]):
    dataall[:, irow, 0] = row[14]
    dataall[:, irow, 1] = row[12]
    dataall[:, irow, 2] = (row[2] / row[12])**2
    result[irow, 0] = row[3]
    result[irow, 1] = row[5]
    result[irow, 2] = row[7]
    result_apo[irow, 0] = row[19]
    result_apo[irow, 1] = row[21]
    result_apo[irow, 2] = row[25]

out = infer_labels('order_1.pickle', dataall, 'zx.pickle', 0., 2.)
out2 = infer_labels_nonlinear('order_2.pickle', dataall, 'zx2.pickle', 0., 2.)
for i in xrange(N):
    print result[i], result_apo[i], out[0][i], out2[0][i]
