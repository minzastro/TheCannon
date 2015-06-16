""" Functions for reading in APOGEE spectra and training labels """

from __future__ import (absolute_import, division, print_function,)
import numpy as np
import scipy.optimize as opt
import os
import sys
import matplotlib.pyplot as plt

# python 3 special
PY3 = sys.version_info[0] > 2
if not PY3:
    range = xrange

try:
    from astropy.io import fits as pyfits
except ImportError:
    import pyfits

def get_pixmask(fluxes, flux_errs):
    """ Create and return a bad pixel mask for an APOGEE spectrum

    Bad pixels are defined as follows: fluxes or errors are not finite, or 
    reported errors are <= 0

    Parameters
    ----------
    fluxes: ndarray
        Flux data values 

    flux_errs: ndarray
        Flux uncertainty data values 

    Returns
    -------
    mask: ndarray
        Bad pixel mask, value of True corresponds to bad pixels
    """
    bad_flux = (~np.isfinite(fluxes)) 
    bad_err = (~np.isfinite(flux_errs)) | (flux_errs <= 0)
    bad_pix = bad_err | bad_flux
    return bad_pix


def get_starmask(ids, labels, aspcapflag, paramflag):
    """ Identifies which APOGEE objects have unreliable physical parameters,
    as laid out in Holzman et al 2015 and on the APOGEE DR12 website

    Parameters
    ----------
    data: np array
        all APOGEE DR12 IDs and labels

    Returns
    -------
    bad: np array
        mask where 1 corresponds to a star with unreliable parameters
    """
    # teff outside range (4000,6000) K and logg < 0
    teff = labels[0,:]
    bad_teff = np.logical_or(teff < 4000, teff > 6000)
    logg = labels[1,:]
    bad_logg = logg < 0
    cuts = bad_teff | bad_logg

    # STAR_WARN flag set (TEFF, LOGG, CHI2, COLORTE, ROTATION, SN)
    # M_H_WARN, ALPHAFE_WARN not included in the above, so do them separately
    star_warn = np.bitwise_and(aspcapflag, 2**7) != 0
    star_bad = np.bitwise_and(aspcapflag, 2**23) != 0
    feh_warn = np.bitwise_and(aspcapflag, 2**3) != 0
    alpha_warn = np.bitwise_and(aspcapflag, 2**4) != 0
    aspcapflag_bad = star_warn | star_bad | feh_warn | alpha_warn

    # separate element flags
    teff_flag = paramflag[:,0] != 0
    logg_flag = paramflag[:,1] != 0
    feh_flag = paramflag[:,3] != 0
    alpha_flag = paramflag[:,4] != 0
    paramflag_bad = teff_flag | logg_flag | feh_flag | alpha_flag

    return cuts | aspcapflag_bad | paramflag_bad 

def make_apogee_label_file():
    hdulist = pyfits.open("allStar-v603.fits")
    datain = hdulist[1].data
    apstarid= datain['APSTAR_ID']
    aspcapflag = datain['ASPCAPFLAG']
    paramflag =datain['PARAMFLAG']
    nstars = len(apstarid)
    ids = np.array([element.split('.')[-1] for element in apstarid])
    t = np.array(datain['TEFF'], dtype=float)
    g = np.array(datain['LOGG'], dtype=float)
    # according to Holzman et al 2015, the most reliable values
    f = np.array(datain['PARAM_M_H'], dtype=float)
    a = np.array(datain['PARAM_ALPHA_M'], dtype=float)
    labels = np.vstack((t, g, f, a))

    # 1 if object would be an unsuitable training object
    star_mask = get_starmask(ids, labels, aspcapflag, paramflag)

    outputf = open("apogee_dr12_labels.csv", "w")
    header = "id,teff,logg,feh,alpha,bad\n"
    outputf.write(header)
    for i in range(nstars):
        line = str(ids[i])+','+str(t[i])+','+str(g[i])+','+str(f[i])+','+\
                ','+str(a[i])+','+str(star_mask[i])+'\n'
        outputf.write(line)
    outputf.close()


def load_spectra(data_dir):
    """ Reads wavelength, flux, and flux uncertainty data from apogee fits files

    Parameters
    ----------
    data_dir: str
        Name of the directory containing all of the data files

    Returns
    -------
    wl: ndarray
        Rest-frame wavelength vector

    fluxes: ndarray
        Flux data values

    ivars: ndarray
        Inverse variance values corresponding to flux values
    """
    print("Loading spectra from directory %s" %data_dir)
    files = list(sorted([data_dir + "/" + filename
             for filename in os.listdir(data_dir) if filename.endswith('fits')]))
    nstars = len(files)  
    for jj, fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        flux = np.array(file_in[1].data)
        if jj == 0:
            npixels = len(flux)
            fluxes = np.zeros((nstars, npixels), dtype=float)
            ivars = np.zeros(fluxes.shape, dtype=float)
            start_wl = file_in[1].header['CRVAL1']
            diff_wl = file_in[1].header['CDELT1']
            val = diff_wl * (npixels) + start_wl
            wl_full_log = np.arange(start_wl,val, diff_wl)
            wl_full = [10 ** aval for aval in wl_full_log]
            wl = np.array(wl_full)
        flux_err = np.array((file_in[2].data))
        badpix = get_pixmask(flux, flux_err)
        flux = np.ma.array(flux, mask=badpix)
        flux_err = np.ma.array(flux_err, mask=badpix)
        ones = np.ma.array(np.ones(npixels), mask=badpix)
        ivar = ones / flux_err**2
        ivar = np.ma.filled(ivar, fill_value=0.)
        fluxes[jj,:] = flux
        ivars[jj,:] = ivar
    print("Spectra loaded")
    return files, wl, fluxes, ivars


def load_labels(filename):
    """ Extracts reference labels from a file

    Parameters
    ----------
    filename: str
        Name of the file containing the table of reference labels

    Returns
    -------
    labels: ndarray
        Reference label values for all reference objects
    """
    print("Loading reference labels from file %s" %filename)
    data = Table(filename)
    data.sort('id')
    label_names = data.keys()[1:] # ignore id
    nlabels = len(label_names)
    print('%s labels:' %nlabels)
    print(label_names)
    labels = np.array([data[k] for k in label_names], dtype=float).T
    return labels 

if  __name__ =='__main__':
    make_apogee_label_file()