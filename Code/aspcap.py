"""Contains useful methods for reading and manipulating ASPCAP data.

The user will have the option to use these, or to just write his/her own. 

Methods
-------
get_spectra
continuum_normalize
get_training_labels (assumes a standard ASCII table)
    can choose all labels, can specify some labels with either name or column #

"""

import pyfits
import numpy as np
import os

def get_spectra(self, files):
    """
    Extracts spectra (wavelengths, fluxes, fluxerrs) from aspcap fits files

    Input: a list of data file names of length nstars
    Returns: a 3D float array of shape (nstars, npixels, 3)
    with spectra[:,:,0] = pixel wavelengths
    spectra[:,:,1] = flux values
    spectra[:,:,2] = flux err values
    """
    for jj,fits_file in enumerate(files):
        file_in = pyfits.open(fits_file)
        fluxes = np.array(file_in[1].data)
        if jj == 0: 
            nstars = len(files) 
            npixels = len(fluxes)
            spectra = np.zeros((nstars, npixels, 3))
        flux_errs = np.array((file_in[2].data))
        start_wl = file_in[1].header['CRVAL1']
        diff_wl = file_in[1].header['CDELT1']
        val = diff_wl*(npixels) + start_wl
        wl_full_log = np.arange(start_wl,val, diff_wl)
        wl_full = [10**aval for aval in wl_full_log]
        pixels = np.array(wl_full) 
        spectra[jj, :, 0] = pixels
        spectra[jj, :, 1] = fluxes
        spectra[jj, :, 2] = flux_errs
    return spectra

def continuum_normalize(self, spectra):
    """
    Continuum-normalizes the spectra.

    Fit a 2nd order Chebyshev polynomial to each segment 
    and divide each segment by its corresponding polynomial 

    Input: spectra array, 2D float shape nstars,npixels,3
    Returns: 3D continuum-normalized spectra (nstars, npixels,3)
            2D continuum array (nstars, npixels)
    """
    nstars = spectra.shape[0]
    npixels = spectra.shape[1]
    continua = np.zeros((nstars, npixels))
    normalized_spectra = np.ones((nstars, npixels, 3))
    # list of "true" continuum pix, det. here by the Cannon
    pixlist = list(np.loadtxt("pixtest4.txt", usecols = (0,), unpack =1))
    # We discard the edges of the fluxes: 10 Angstroms, which is ~50 pixels
    ## I found the regions with flux to be: 
    ## [321, 3242] [3647, 6047], [6411, 8305]
    ## With edge cuts: [371, 3192], [3697, 5997], [6461, 8255]
    ## Corresponding to: [15218, 15743] [15931, 16367] [16549, 16887] 
    ranges = [[371,3192], [3697,5997], [6461,8255]]
    LARGE = 200.
    for jj in range(nstars):
        # There must be a better way to do this, but right now I'm stumped
        # Mask unphysical pixels
        bad1 = np.invert(np.logical_and(np.isfinite(spectra[jj,:,1]),  
            np.isfinite(spectra[jj,:,2])))
        bad = bad1 | (spectra[jj,:,2] <= 0)
        spectra[jj,:,1][bad] = 0.
        spectra[jj,:,2][bad] = np.Inf
        var_array = 100**2*np.ones(npixels)
        var_array[pixlist] = 0.000
        ivar = 1. / ((spectra[jj, :, 2] ** 2) + var_array)
        ivar = (np.ma.masked_invalid(ivar)).filled(0)
        for i in range(len(ranges)):
            start, stop = ranges[i][0], ranges[i][1]
            spectrum = spectra[jj,start:stop,:]
            ivar1 = ivar[start:stop]
            fit = np.polynomial.chebyshev.Chebyshev.fit(x=spectrum[:,0], 
                    y=spectrum[:,1], w=ivar1, deg=3)
            continua[jj,start:stop] = fit(spectrum[:,0])
            normalized_fluxes = spectrum[:,1]/fit(spectra[0,start:stop,0])
            bad = np.invert(np.isfinite(normalized_fluxes))
            normalized_fluxes[bad] = 1.
            normalized_flux_errs = spectrum[:,2]/fit(spectra[0,start:stop,0])
            bad = np.logical_or(np.invert(np.isfinite(normalized_flux_errs)),
                    normalized_flux_errs <= 0)
            normalized_flux_errs[bad] = LARGE
            normalized_spectra[jj,:,0] = spectra[jj,:,0]
            normalized_spectra[jj,start:stop,1] = normalized_fluxes 
            normalized_spectra[jj,start:stop,2] = normalized_flux_errs
        # One last check for unphysical pixels
        bad = spectra[jj,:,2] > LARGE
        normalized_spectra[jj,np.logical_or(bad, bad1),1] = 1.
        normalized_spectra[jj,np.logical_or(bad, bad1),2] = LARGE
    return normalized_spectra, continua

def get_training_labels(self):
    """
    Extracts training labels from file

    Input: name(string) of the data file containing the labels
    Returns: a 2D np.array (size=numtrainingstars, nlabels)
    consisting of all of the training labels
    """
    input1 = "starsin_SFD_Pleiades.txt"
    input2 = "ages_2.txt"
    #input2 = "logAges.txt"
    T_est,g_est,feh_est,T_A, g_A, feh_A = np.loadtxt(input1, 
            usecols = (4,6,8,3,5,7), unpack =1)
    age_est = np.loadtxt(input2, usecols = (0,), unpack =1)
    training_labels = np.array([T_est, g_est, feh_est, age_est])
    # This is a cheat...not sure how to deal with this generally.
    # this should probably be dealt with somehow by discard_stars
    apogee_labels = np.array([T_A, g_A, feh_A])
    return training_labels.T, apogee_labels.T
