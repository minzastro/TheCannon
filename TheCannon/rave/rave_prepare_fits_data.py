#!/usr/bin/env python
# -*- coding: utf-8 -*-
"""
Created on Fri Apr  1 10:14:26 2016

@author: mints
"""

from astropy.io import fits
from astropy.table import Table
from scipy.io import readsav
import numpy as np
import cPickle as pickle
from scipy.ndimage.filters import gaussian_filter1d

#spectra = readsav('2009K1_parameters.save').items()[0][1]
spectra = pickle.load(open('RAVE_APOGEE_GCS.pickle', 'r'))

colnames = """rave_id  stn  snr  teff_0  eteff_0  logg_0  elogg_0
meta  emeta  new_meta  algo_conv  chi2  obs_sp  synth_sp  lambda
apogee_id  ra  dec  snr_APOGEE  teff  teff_err  logg  logg_err
param_alpha_m  param_alpha_m_err
param_m_h  param_m_h_err  param_m_h_flag
param_c_m  param_c_m_flag  param_n_m  param_n_m_flag
al_h  al_h_err  al_h_flag  c_h  c_h_err  c_h_flag
ca_h  ca_h_err  ca_h_flag  fe_h  fe_h_err  fe_h_flag
k_h  k_h_err  k_h_flag  mg_h  mg_h_err  mg_h_flag
mn_h  mn_h_err  mn_h_flag  na_h  na_h_err  na_h_flag
ni_h  ni_h_err  ni_h_flag  n_h  n_h_err  n_h_flag
o_h  o_h_err  o_h_flag  si_h  si_h_err  si_h_flag
s_h  s_h_err  s_h_flag  ti_h  ti_h_err  ti_h_flag
v_h  v_h_err  v_h_flag  is_cluster
RAVE_OBS_ID  RAVEID  Teff_K  eTeff_K  logg_K  elogg_K
Met_K  Met_N_K  eMet_K  SNR_K  Algo_Conv_K""".split()

hdu = fits.PrimaryHDU()
data = []
obs_sp = []
synth_sp = []
smooth_sp = []
lambdas = spectra[0][14]
colnames_out = colnames[:12] + colnames[15:]

width = 50./ (lambdas[1] - lambdas[0])

for row in spectra:
    # Some spectra are just invalid...
    if row[12][0] > 0.2:
        data.append(row[:12] + row[15:])
        obs_sp.append(row[12])
        smooth_sp.append(row[12] / gaussian_filter1d(row[12], width))
        synth_sp.append(row[13])

data_table = Table(rows=data, names=colnames_out, dtype=[type(x) for x in data[0]])
header = fits.Header()
header.update(data_table.meta)
main = fits.BinTableHDU(data_table.as_array(), header)
lambda_column = fits.Column('lambda', 'E', array=lambdas)
lambdas = fits.BinTableHDU.from_columns([lambda_column])
header = fits.Header()
header['CONTENT'] = 'Observed spectra'
obs_sp = fits.ImageHDU(obs_sp, header, do_not_scale_image_data=True)
header['CONTENT'] = 'Smoothed spectra'
smooth_sp = fits.ImageHDU(smooth_sp, header, do_not_scale_image_data=True)
header['CONTENT'] = 'Synthetic spectra'
synth_sp = fits.ImageHDU(synth_sp, header, do_not_scale_image_data=True)
hdulist = fits.HDUList([hdu, main, obs_sp, smooth_sp, synth_sp, lambdas])
hdulist.writeto('RAVE_ready.fits')
