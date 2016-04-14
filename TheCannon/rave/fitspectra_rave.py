"""
This file is part of The Cannon analysis project.
Copyright 2014 Melissa Ness.

# urls
- http://iopscience.iop.org/1538-3881/146/5/133/suppdata/aj485195t4_mrt.txt for calibration stars
- http://data.sdss3.org/irSpectrumDetail?locid=4330&commiss=0&apogeeid=2M17411636-2903150&show_aspcap=True object explorer
- http://data.sdss3.org/basicIRSpectra/searchStarA
- http://data.sdss3.org/sas/dr10/apogee/spectro/redux/r3/s3/a3/ for the data files

# to-do
- need to add a test that the wavelength range is the same - and if it isn't interpolate to the same range
- format PEP8-ish (four-space tabs, for example)
- take logg_cut as an input
- extend to perform quadratic fitting
"""

from astropy.io import fits as pyfits
import glob
import pickle
from scipy import optimize as opt
import numpy as np
import pylab as plt
normed_training_data = 'normed_data.pickle'


def weighted_median(values, weights, quantile):
    """weighted_median

    keywords
    --------

    values: ndarray
        input values

    weights: ndarray
        weights to apply to each value in values

    quantile: float
        quantile selection

    returns
    -------
    val: float
        median value
    """
    sindx = np.argsort(values)
    cvalues = np.cumsum(weights[sindx])
    cvalues = cvalues / cvalues[-1]
    foo = sindx[cvalues > quantile]
    if len(foo) == 0:
        return values[0]
    return values[foo[0]]


def do_one_regression_at_fixed_scatter(data, features, scatter):
    """
    Parameters
    ----------
    data: ndarray, [nobjs, 3]
        wavelengths, fluxes, invvars

    meta: ndarray, [nobjs, nmeta]
        Teff, Feh, etc, etc

    scatter:


    Returns
    -------
    coeff: ndarray
        coefficients of the fit

    MTCinvM: ndarray
        inverse covariance matrix for fit coefficients

    chi: float
        chi-squared at best fit

    logdet_Cinv: float
        inverse of the log determinant of the cov matrice
        :math:`\sum(\log(Cinv))`
    """
    # least square fit
    Cinv = 1. / (data[:, 2] ** 2 + scatter ** 2)  # invvar slice of data
    M = features
    MTCinvM = np.dot(M.T, Cinv[:, None] * M)
    x = data[:, 1]  # intensity slice of data
    MTCinvx = np.dot(M.T, Cinv * x)
    try:
        coeff = np.linalg.solve(MTCinvM, MTCinvx)
    except np.linalg.linalg.LinAlgError:
        print MTCinvM, MTCinvx, data[:, 0], data[:, 1], data[:, 2]
        print features
    assert np.all(np.isfinite(coeff))
    chi = np.sqrt(Cinv) * (x - np.dot(M, coeff))
    logdet_Cinv = np.sum(np.log(Cinv))
    return (coeff, MTCinvM, chi, logdet_Cinv)


def do_one_regression(data, metadata):
    """
    does a regression at a single wavelength to fit
    calling the fixed scatter routine
    # inputs:
    """
    ln_s_values = np.arange(np.log(0.0001), 0., 0.5)
    chis_eval = np.zeros_like(ln_s_values)
    for ii, ln_s in enumerate(ln_s_values):
        _, _, chi, logdet_Cinv = do_one_regression_at_fixed_scatter(
            data, metadata, scatter=np.exp(ln_s))
        chis_eval[ii] = np.sum(chi * chi) - logdet_Cinv
    if np.any(np.isnan(chis_eval)):
        s_best = np.exp(ln_s_values[-1])
        return do_one_regression_at_fixed_scatter(data, metadata,
                                                  scatter=s_best) + (s_best,)
    lowest = np.argmin(chis_eval)
    if lowest == 0 or lowest == len(ln_s_values)-1:
        s_best = np.exp(ln_s_values[lowest])
        return do_one_regression_at_fixed_scatter(data, metadata,
                                                  scatter=s_best) + (s_best,)
    ln_s_values_short = ln_s_values[np.array([lowest-1, lowest, lowest+1])]
    chis_eval_short = chis_eval[np.array([lowest-1, lowest, lowest+1])]
    z = np.polyfit(ln_s_values_short, chis_eval_short, 2)
    s_best = np.exp(np.roots(np.polyder(z))[0])
    return do_one_regression_at_fixed_scatter(data, metadata,
                                              scatter=s_best) + (s_best,)


def do_regressions(dataall, features):
    """
    """
    nlam, nobj, ndata = dataall.shape
    nobj, npred = features.shape
    featuresall = np.zeros((nlam, nobj, npred))
    featuresall[:, :, :] = features[None, :, :]
    return map(do_one_regression, dataall, featuresall)


def train(dataall, metaall, order, fn, Ametaall, cluster_name,
          logg_cut=100., teff_cut=0., leave_out=None):
    """
    - `leave out` must be in the correct form to be an input to `np.delete`
    """
    if leave_out is not None:
        dataall = np.delete(dataall, [leave_out], axis=1)
        metaall = np.delete(metaall, [leave_out], axis=0)
        Ametaall = np.delete(Ametaall, [leave_out], axis=0)

    diff_t = np.abs(np.array(metaall[:, 0] - Ametaall[:, 0]))
    good = np.logical_and((metaall[:, 1] > 0.2), (diff_t < 6000.))
    dataall = dataall[:, good]
    metaall = metaall[good]
    nstars, nmeta = metaall.shape
    offsets = np.mean(metaall, axis=0)
    features = np.ones((nstars, 1))
    if order >= 1:
        features = np.hstack((features, metaall - offsets))
    if order >= 2:
        newfeatures = np.array([np.outer(m, m)[np.triu_indices(nmeta)]
                                for m in (metaall - offsets)])
        features = np.hstack((features, newfeatures))
    blob = do_regressions(dataall, features)
    coeffs = np.array([b[0] for b in blob])
    covs = np.array([np.linalg.inv(b[1]) for b in blob])
    chis = np.array([b[2] for b in blob])
    chisqs = np.array([np.dot(b[2], b[2]) - b[3] for b in blob])  # holy crap be careful
    scatters = np.array([b[4] for b in blob])

    fd = open(fn, "w")
    pickle.dump((dataall, metaall, features, offsets, coeffs, covs,
                 scatters, chis, chisqs), fd)
    fd.close()



def get_goodness_fit(fn_pickle, filein, Params_all, MCM_rotate_all):
    with open(fn_pickle, 'r') as fd:
        dataall, metaall, labels, offsets, coeffs, covs, \
            scatters, chis, chisq = pickle.load(fd)
    file_with_star_data = filein
    file_normed = normed_training_data.split('.pickle')
    if filein != file_normed:
        f_flux = open(file_with_star_data, 'r')
        flux = pickle.load(f_flux)
    if filein == file_normed:
        f_flux = open('self_2nd_order.pickle', 'r')
        flux, metaall, labels, Ametaall, cluster_name, ids = pickle.load(f_flux)
    f_flux.close()
    labels = Params_all
    nlabels = labels.shape[1]
    nstars = labels.shape[0]
    features_data = np.ones((nstars, 1))
    offsets = np.mean(labels, axis=0)
    features_data = np.hstack((features_data, labels - offsets))
    newfeatures_data = np.array([np.outer(m, m)[np.triu_indices(nlabels)] for m in labels - offsets])
    features_data = np.hstack((features_data, newfeatures_data))
    chi2_all = np.zeros(nstars)
    for jj in range(nstars):
        model_gen = np.dot(coeffs, features_data.T[:, jj])
        data_star = flux[:, jj, 1]
        Cinv = 1. / (flux[:, jj, 2] ** 2 + scatters ** 2)  # invvar slice of data
        chi2 = sum((Cinv) * (data_star - np.dot(coeffs, features_data.T[:, jj]))**2)
        chi2_all[jj] = chi2
        fig = plt.figure()
        ax1 = fig.add_subplot(211)
        ax2 = fig.add_subplot(212)
        noises = (flux[:, jj, 2]**2 + scatters**2)**0.5
        ydiff_norm = 1. / noises * (data_star - model_gen)
        bad = flux[:, jj, 2] > 0.1
        ydiff_norm[bad] = None
        data_star[bad] = None
        model_gen[bad] = None
        ax1.plot(flux[:, jj, 0], data_star, 'k')
        ax1.plot(flux[:, jj, 0], model_gen, 'r')
        ax2.plot(flux[:, jj, 0], ydiff_norm, 'r')
        ax1.set_xlim(15200, 16000)
        ax1.set_ylim(0.5, 1.2)
        ax2.set_xlim(15200, 16000)
        ax2.set_ylim(-10.2, 10.2)
        prefix = str('check'+str(filein)+"_"+str(jj))
        savefig2(fig, prefix, transparent=False, bbox_inches='tight',
                 pad_inches=0.5)
        plt.close()
    return chi2_all


def savefig2(fig, prefix, **kwargs):
    suffix = ".png"
    print "writing %s" % (prefix + suffix)
    fig.savefig(prefix + suffix, **kwargs)


## non linear stuff below ##
# returns the non linear function
def func(coeff, a, b, c):
    return np.sum(np.dot(coeff, np.outer([1., a, b, c], [1., a, b, c])), axis=(1,2))


# thankyou stack overflow for the example below on how to use the optimse function
def nonlinear_invert(f, coeff, sigmavals):
    def wrapped_func(observation_points, a, b, c):
        return func(observation_points, a, b, c)
    model, cov = opt.curve_fit(wrapped_func, coeff, f,
                               sigma=sigmavals, maxfev=2000)#absolute_sigma = True)  is not an option in my version of scipy will upgrade scipy
    return model, cov


def infer_labels_nonlinear(fn_pickle, testdata, fout_pickle,
                           weak_lower, weak_upper):
    """
    best log g = weak_lower = 0.95, weak_upper = 0.98
    best teff = weak_lower = 0.95, weak_upper = 0.99
    best_feh = weak_lower = 0.935, weak_upper = 0.98
    this returns the parameters for a field of data -
    and normalises if it is not already normalised
    this is slow because it reads a pickle file
    """
    file_in = open(fn_pickle, 'r')
    dataall, metaall, labels, offsets, coeffs, covs, \
        scatters, chis, chisq = pickle.load(file_in)
    file_in.close()
    nstars = (testdata.shape)[1]
    nlabels = offsets.shape[0]
    import ipdb; ipdb.set_trace()
    Params_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, np.shape(coeffs)[1]-1,
                               np.shape(coeffs)[1]-1.))
    covs_all = np.zeros((nstars, nlabels, nlabels))
    for jj in xrange(nstars):
        if np.any(abs(testdata[:, jj, 0] - dataall[:, 0, 0]) > 0.0001):
            print testdata[range(5), jj, 0], dataall[range(5), 0, 0]
            assert False
        ydata = testdata[:, jj, 1]
        ysigma = testdata[:, jj, 2]
        coeff = np.zeros((coeffs.shape[0], 4, 4))
        for icoe, coe in enumerate(coeffs):
            coeff[icoe][np.triu_indices(4)] = coe
            #coeff[icoe] = coeff[icoe] + coeff[icoe].T
            #coeff[icoe][np.diag_indices(4)] = coeff[icoe][np.diag_indices(4)]*0.5
        Cinv = 1. / (ysigma ** 2 + scatters ** 2)
        Params, covs = nonlinear_invert(ydata, coeff, 1/Cinv**0.5)
        Params = Params + offsets
        coeffs_slice = coeffs[:, -9:]
        MCM_rotate = np.dot(coeffs_slice.T, Cinv[:, None] * coeffs_slice)
        Params_all[jj, :] = Params
        MCM_rotate_all[jj, :, :] = MCM_rotate
        covs_all[jj, :, :] = covs
    filein = fout_pickle.split('_tags')[0]
    file_in = open(fout_pickle, 'w')
    if filein == 'self_2nd_order':
        file_normed = normed_training_data.split('.pickle')[0]
        chi2 = get_goodness_fit(fn_pickle, file_normed,
                                Params_all, MCM_rotate_all)
    #else:
    #    chi2 = get_goodness_fit(fn_pickle, filein, Params_all, MCM_rotate_all)
    chi2_def = None #chi2
    pickle.dump((Params_all, covs_all, chi2_def), file_in)
    file_in.close()
    return Params_all, MCM_rotate_all


def infer_labels(fn_pickle, testdata, fout_pickle,
                 weak_lower, weak_upper):
    """
    best log g = weak_lower = 0.95, weak_upper = 0.98
    best teff = weak_lower = 0.95, weak_upper = 0.99
    best_feh = weak_lower = 0.935, weak_upper = 0.98
    this returns the parameters for a field of data  -
    and normalises if it is not already normalised
    this is slow because it reads a pickle file
    """
    with open(fn_pickle, 'r') as file_in:
        dataall, metaall, labels, offsets, coeffs, \
            covs, scatters, chis, chisqs = pickle.load(file_in)
    nstars = testdata.shape[1]
    nlabels = labels.shape[1] - 1
    Params_all = np.zeros((nstars, nlabels))
    MCM_rotate_all = np.zeros((nstars, nlabels, nlabels))
    for jj in xrange(nstars):
        if np.any(testdata[:, jj, 0] != dataall[:, 0, 0]):
            print testdata[range(5), jj, 0], dataall[range(5), 0, 0]
            assert False
        xdata = testdata[:, jj, 0]
        ydata = testdata[:, jj, 1]
        ysigma = testdata[:, jj, 2]
        ydata_norm = ydata  - coeffs[:, 0] # subtract the mean
        coeffs_slice = coeffs[:, -3:]
        ind1 =  np.logical_and(ydata > weak_lower, ydata < weak_upper)
        Cinv = 1. / (ysigma ** 2 + scatters ** 2)
        MCM_rotate = np.dot(coeffs_slice[ind1].T,
                            Cinv[:, None][ind1] * coeffs_slice[ind1])
        MCy_vals = np.dot(coeffs_slice[ind1].T, Cinv[ind1] * ydata_norm[ind1])
        Params = np.linalg.solve(MCM_rotate, MCy_vals)
        Params = Params + offsets
        #print Params
        Params_all[jj, :] = Params
        MCM_rotate_all[jj, :, :] = MCM_rotate
    file_in = open(fout_pickle, 'w')
    pickle.dump((Params_all, MCM_rotate_all), file_in)
    file_in.close()
    return Params_all, MCM_rotate_all
