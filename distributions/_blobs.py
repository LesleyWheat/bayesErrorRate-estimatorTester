# Imports

# Packages
import numpy as np

# Import individual functions for speedup
from numpy import asarray, atleast_2d, int64
from sklearn.neighbors import KernelDensity
from scipy.stats import multivariate_normal

# Module
from ._base import abstractDistribution

# --------------------------------------------------------------------


class blob(abstractDistribution):

    def generate(self, rng: np.random.Generator, samples: int) -> np.ndarray:
        data = asarray(self._kde.tree_.data)
        u = rng.uniform(0, 1, size=samples)
        i = (u * data.shape[0]).astype(int64)
        x = atleast_2d(rng.normal(data[i], self._bandwidth))
        return x

    def logPdf(self, x: np.ndarray) -> np.ndarray:
        # Use KDE to create the pdf, faster for more blobs
        if x.size == self._nFeatures:
            x = x.reshape(1, -1)
        score = self._kde.score_samples(x)
        return score

    def _initKde(self):
        ker = "gaussian"
        bw = self._bandwidth
        self._kde = KernelDensity(kernel=ker, bandwidth=bw).fit(self._centers)


class oneBlob(blob):
    """
    A Guassian distribution object of two.
    Both are offset.

    Parameters
    ----------
    nFeatures: int
        the number of features
    mean: float
        the offset of the distribution
    var: float
        Parameter to control variance.
    singleD: bool, Optional

    Notes
    -------
    May fail at low variance and high dimensionality
    due to low floating point values.

    """
    _minFeatures = 1
    _minVar = 0.001

    def __init__(self, nFeatures: int, mean: float, var: float,
                 singleD=False):
        self._useCustomValues = False
        self._bandwidth = var**(1/2)

        if nFeatures > self._minFeatures:
            self._nFeatures = nFeatures
        else:
            raise ValueError((f'd must be greater than {self._minFeatures}. '
                             f'Given: {nFeatures}'))

        if var > self._minVar:
            self._var = var
            self._covar = var*np.identity(nFeatures)
        else:
            raise ValueError((f'var must be greater than {self._minVar}. '
                              f'Given: {var}'))

        if singleD:
            self._mean = np.zeros(nFeatures)
            self._mean[0] = mean
        else:
            self._mean = mean*np.ones(nFeatures)

        self._centers = atleast_2d([self._mean])

        self._initKde()

    def set_mean(self, mean: float):
        # For tests
        self._mean = mean
        self._centers = atleast_2d([self._mean])
        self._initKde()

    def set_covar(self, covar: np.ndarray):
        # For tests
        from warnings import warn

        # KDE will no longer be used
        self._useCustomValues = True
        self._covar = covar
        self._var = np.max(covar)

        msg = ('Custom covariance is set. Scipy will be used. '
               'Be aware there may be accuracy errors '
               'when using more than 14 features.')
        warn(msg)

    def generate(self, rng: np.random.Generator, samples: int) -> np.ndarray:
        if self._useCustomValues:
            return rng.multivariate_normal(self._mean,
                                           self._covar,
                                           size=(samples))
        else:
            return super().generate(rng, samples)

    def logPdf(self, x: np.ndarray) -> np.ndarray:
        if self._useCustomValues:
            eval = multivariate_normal.logpdf(x,
                                              mean=self._mean,
                                              cov=self._covar)
            return eval
        else:
            return super().logPdf(x)


class twoBlob(blob):
    """
    A Guassian mixture distribution object of three.
    One is placed at the center, the other two are offset.

    Parameters
    ----------
    nFeatures: int
        the number of features
    mean: float
        the offset of the distribution
    var: float
        Parameter to control variance.

    Notes
    -------
    May fail at low variance and high dimensionality
    due to low floating point values.

    """
    _minOffset = 0

    def __init__(self, nFeatures: int, offset: float, var: float):
        self._nFeatures = nFeatures
        self._offset = (offset**2/nFeatures)**(1/2)
        self._covar = var*np.identity(nFeatures)
        self._var = var
        self._bandwidth = var**(1/2)

        if offset > self._minOffset:
            self._offset = (offset**2/nFeatures)**(1/2)
        else:
            raise ValueError((f'twoBlob: offset must be more than {self._minOffset}.'
                             f'Given: {offset}'))

        offsetList = [1*self._offset*np.ones(self._nFeatures),
                      -1*self._offset*np.ones(self._nFeatures)]

        self._centers = np.vstack(offsetList)
        self._initKde()


class threeBlob(blob):
    """
    A Guassian mixture distribution object.

    Parameters
    ----------
    nFeatures: int
        the number of features
    mean: float
        the offset of the distribution
    var: float
        Parameter to control variance.

    Notes
    -------
    May fail at low variance and high dimensionality
    due to low floating point values.

    """
    _minOffset = 0

    def __init__(self, nFeatures: int, offset: float, var: float):
        self._nFeatures = nFeatures
        self._covar = var*np.identity(nFeatures)
        self._bandwidth = var**(1/2)

        if offset > self._minOffset:
            self._offset = (offset**2/nFeatures)**(1/2)
        else:
            raise ValueError(f'threeBlob: offset must be more than {self._minOffset}.'
                             f'Given: {offset}')

        offsetList = [1*self._offset*np.ones(self._nFeatures),
                      -1*self._offset*np.ones(self._nFeatures),
                      np.zeros(self._nFeatures)]

        self._centers = np.vstack(offsetList)
        self._initKde()
