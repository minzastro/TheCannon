#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 16:10:44 2016

@author: mints
"""

import pylab as plt
import cPickle as pi
from scipy.ndimage.filters import gaussian_filter1d
c = 'rgbkc'

def smooth(x, y, width_angstrom=50):
    width = width_angstrom / (x[1] - x[0])
    return gaussian_filter1d(y, width)

data = pi.load(open('RAVE_APOGEE_GCS.pickle', 'r'))
x = data[0][14]
for i in xrange(5):
    plt.plot(x, data[i][12], color=c[i], linestyle='solid')
    sm = smooth(x, data[i][12])
    plt.plot(x, sm, color=c[i], linestyle='dashed')
    plt.plot(x, data[i][12]/sm - 1., color=c[i]) #, linestyle='dashed')
plt.show()