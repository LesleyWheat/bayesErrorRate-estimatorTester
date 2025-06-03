# Imports
import warnings

# Packages
import numpy as np

# --------------------------------------------------------------------


class lowDensityFinder():
    """
    Finds new points in low density areas.

    Parameters
    ----------
    rng: numpy.random.Generator object, optional
        Pass a preseeded generator for reproducible results.
    nMinK: int, optional
        Number of low density points to find.
    nNew: int, optional
        Number of new points to generate for each found low
        density point.
    ymin: float, optional
        Minimum y value to use.
    ymax: float, optional
        Maximum y value to use.
    xMinLimits: list of floats
        Points under these limits will be deleted.
        Size must match number of features.
    xMaxLimits: list of floats
        Points over these limits will be deleted.
        Size must match number of features.
    """

    def __init__(self, rng: np.random.Generator | None = None,
                 nMinK: int = 3, nNew: int = 2,
                 ymin: float = 0, ymax: float = 0.5,
                 xMinLimits=[], xMaxLimits=[]):

        self._nMinK = nMinK
        self._nNew = nNew
        self._yMin = ymin
        self._yMax = ymax
        self._xMinLimits = xMinLimits
        self._xMaxLimits = xMaxLimits
        self._rng = np.random.default_rng() if rng is None else rng

    def _covarEstimate(self, X: np.ndarray, covarRatio: float):
        covar = np.identity(X.shape[1])
        for idx in range(X.shape[1]):
            xMax = np.max(X[:, idx])
            xMin = np.min(X[:, idx])
            covar[idx, idx] = covarRatio * (xMax - xMin)
        return covar

    def _deleteOutsideLimits(self, new_X: np.ndarray):
        if self._xMinLimits:
            for i in range(new_X.shape[1]):
                idxs = np.argwhere(new_X[:, i] < self._xMinLimits[i])
                new_X = np.delete(new_X, idxs, axis=0)

        if self._xMaxLimits:
            for i in range(new_X.shape[1]):
                idxs = np.argwhere(new_X[:, i] > self._xMaxLimits[i])
                new_X = np.delete(new_X, idxs, axis=0)

        return new_X

    def _noDuplicates(self, newPoints: np.ndarray, oldPoints: np.ndarray):
        # Check for existing
        dupBool = (np.isclose(newPoints[:, None], oldPoints)).all(-1).any(-1)
        if (dupBool).any():
            msg = ('Duplicate points detected in new points generation. '
                   'Removing and moving on.')
            warnings.warn(msg)
        nonDupIndxs = np.where(~dupBool)[0]
        genPoints = newPoints[nonDupIndxs, :]
        return genPoints

    def newPointsByDensity(self, X: np.ndarray, Y: np.ndarray, covarRatio=0.1):
        """
        Generates new points in low density areas.

        Parameters
        ----------
        X: numpy array of shape (n, m)
            Location of data in x-space.
        Y: numpy array of shape (n)
            Location of data in y-space.
        covarRatio: float, optional
            Ratio for noise covariance for new points generation.

        Returns
        -------
        newX: numpy array of sample(s)
            Has size (nN, m) where nN is the number of new
            samples. nN is equal or less than nMinK*nNew.

        """

        if X.shape[0] != Y.shape[0]:
            msg = (f'X with shape {X.shape} does not match '
                   f'Y with shape: {Y.shape}')
            raise Exception(msg)

        covar = self._covarEstimate(X, covarRatio)
        new = []

        # Find lowest density X points based on Y
        muX = self._oldPointsByGap(X, Y)

        # generate some new points with noise
        for idx in range(muX.shape[0]):
            mu = muX[idx, :]
            noisyPoints = self._rng.multivariate_normal(mu, covar,
                                                        size=(self._nNew))
            new.append(self._noDuplicates(noisyPoints, muX))

        return self._deleteOutsideLimits(np.vstack(new))

    def _oldPointsByGap(self, X: np.ndarray, Y: np.ndarray) -> np.ndarray:
        """
        Finds old points in/closest to low density areas.

        Parameters
        ----------
        X: numpy array of shape (n, m)
            Location of data in x-space.
        Y: numpy array of shape (n)
            Location of data in y-space.

        Returns
        -------
        lonelyX: numpy array
            Contains old low density samples.
            Length is less than nMinK*m.

        """

        sortedY = np.sort(Y)
        gapList = np.diff(sortedY)
        indGap = np.argpartition(gapList, -self._nMinK)[-self._nMinK:]
        indGapSorted = indGap[np.argsort(gapList[indGap])][::-1]
        lonelyY = []

        # Check ends
        if np.min(Y)-self._yMin > gapList[indGapSorted[-1]]:
            indGapSorted = indGapSorted[:-1]
            lonelyY.append((np.min(Y)+self._yMin)/2)

        if self._yMax-np.max(Y) > gapList[indGapSorted[-1]]:
            indGapSorted = indGapSorted[:-1]
            lonelyY.append((self._yMax+np.max(Y))/2)

        # Take midpoints
        for idx in range(len(indGapSorted)):
            jdx = indGapSorted[idx]
            lonelyY.append((sortedY[jdx]+sortedY[jdx+1])/2)

        lonelyY = np.array(lonelyY)

        # Find closestest uniform Y to existing Y
        closestYIdx = np.unique(np.argmin(np.abs(lonelyY-Y[:, None]), axis=0))

        # uniques points only, in case of X duplicates
        lonelyX = np.unique(X[closestYIdx, :], axis=0)

        # return points
        return lonelyX
