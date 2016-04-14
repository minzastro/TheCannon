# coding: utf-8

from sklearn.ensemble import RandomForestRegressor as RFR
from sklearn import cross_validation as cv
from astropy.io import fits
import numpy as np
import pylab as plt
import joblib
data = fits.open('RAVE_ready.fits')
sp = data[2].data
sp.shape
target = data[1].data
ta = np.array([target['teff'], target['logg'], target['param_m_h']]).T
mask = ~np.isnan(ta[:, 1])
sp = sp[mask]
ta = ta[mask]
rfr = RFR(n_estimators=20, oob_score=True)

print cv.cross_val_predict(rfr, sp, ta, n_jobs=2)
#rfr.fit(sp, ta)

#joblib.dump(rfr, 'rave_rfr.dump', compress=3)

fig = plt.figure(figsize=(6, 9))
ax = plt.subplot(3, 1, 1)
ax.set_title('Temperature')
ax.set_xlabel('APOGEE')
ax.set_ylabel('Predict')
ax.scatter(ta[:, 0], rfr.oob_prediction_[:, 0])
ax = plt.subplot(3, 1, 2)
ax.set_title('logg')
ax.set_xlabel('APOGEE')
ax.set_ylabel('Predict')
ax.scatter(ta[:, 1], rfr.oob_prediction_[:, 1])
ax = plt.subplot(3, 1, 3)
ax.set_title('[Fe/H]')
ax.set_xlabel('APOGEE')
ax.set_ylabel('Predict')
ax.scatter(ta[:, 2], rfr.oob_prediction_[:, 2])
plt.tight_layout()
plt.savefig('rave_rfr.png')

for i in xrange(3):
    diff = ta[:, i] - rfr.oob_prediction_[:, i]
    print np.mean(diff), np.median(diff), np.std(diff)