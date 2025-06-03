# Imports
import warnings
import os
import glob
import copy
from typing import List

# Packages
import pandas as pd
import scipy
import numpy as np

# Import individual functions for speedup
from numpy import array, arange, ones, zeros, identity
from numpy import concatenate, vstack, linspace, argwhere
from numpy import absolute, amin, argmin, amax
from numpy import sqrt, square, power, cumsum, cumprod, roll
from numpy import arctan2, sign, sin, cos, pi
from numpy import log, logaddexp
from numpy import sum as npSum

# Project
import config as param

# Module
from ._blobs import blob, oneBlob

# --------------------------------------------------------------------
# Functions for coordinate conversion


def angular_coordinates(x: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """
    Calculate angular coordinates of the vectors in `x` along the first axis.

    Parameters
    ----------
    x: numpy matrix (n, m)

    Returns
    ----------
    r: numpy array
    phi: numpy matrix

    Notes
    -------
    Source:
    https://en.wikipedia.org/wiki/N-sphere#Spherical_coordinates

    """
    if len(x.shape) < 2:
        x = x.reshape(1, -1)

    # Compute radius
    r = sqrt(npSum(x**2, axis=1))

    # Compute spherical angles
    a = x[:, 1:] ** 2
    b = sqrt(cumsum(a[:, ::-1], axis=1))
    phi = arctan2(b[:, ::-1], x[:, :-1])
    phi[:, -1] = phi[:, -1]*sign(x[:, -1])
    return r, phi


def cartesian_coordinates(r: np.ndarray, phi: np.ndarray) -> np.ndarray:
    """
    Returns cartesian coordinates with n coordinates in m dimensions
    given angular coordinates.

    Parameters
    ----------
    r: numpy array
        Size: (n)
    phi: numpy array
        Size: (n, m-1)

    Returns
    ----------
    x: numpy matrix
        Transformed cartesian coordinates.
        Size: (n, m)
    """
    r = r.reshape(-1, 1)
    if len(phi.shape) < 2:
        phi = phi.reshape(1, -1)

    a = concatenate(((2*pi)*ones((phi.shape[0], 1)), phi), axis=1)
    si = sin(a)
    si[:, 0] = 1
    si = cumprod(si, axis=1)
    co = cos(a)
    co = roll(co, -1, axis=1)
    x = si*co*r
    return x

# --------------------------------------------------------------------
# Functions


def _addSlackNumbers(oldList: List[int], slack: int):
    if slack == 0:
        return oldList

    newList = []
    for n in oldList:
        newList.append(n)
        for idx in range(slack):
            # Add positive and negative
            plus = n+(1+idx)
            minus = n-(1+idx)
            newList.append(plus)

            if minus > 1:
                newList.append(minus)

    # Convert to set to remove duplicates
    return set(newList)

# --------------------------------------------------------------------
# Classes


class _evenlySpacedPointsOnSphere():
    """
    Returns a set of points approximately evenly distributed
    over the surface of a hypersphere in d dimensions with
    dist spacing. This is done with a hyperspiral.

    Parameters
    ----------
    maxLoops: int, optional
        The maximum number of loops allowed to search for a
        number of points before continuing. When this number is hit,
        it will trigger an exception.
    atol: float, optional
        Tolerance for distance in dropping points.
    maxPoints: int, optional
        When this number of points is hit, it will trigger an
        exception.
    """

    def __init__(self, maxLoops: int = 100, atol: float = 1e-6,
                 maxPoints: int = 1000):
        self._maxLoops = maxLoops
        self._atol = atol
        self._maxPoints = maxPoints

    def _pointsCalculator(self, nFeatures=None, dist=None, nPoints=None,
                          reverse=False) -> int | float:
        """
        Approximate calculation for number of points given target
        distance of hyperspiral or distance predicted to get a number
        of points.

        Parameters
        ----------
        nFeatures: int, optional
            Number of dimensions
        dist: float, optional
            Distance between the points
            Required if reverse is False
        nPoints: int, optional
            Target number of points
            Required if reverse is True
        reverse: bool, optional
            True indicates number of points is to be calculated
            False indicates distance is to be calculated

        Returns
        ----------
        points: int
            Approximate expected number of points to distribute over sphere

        OR 

        dist: float
            Target distance


        Notes
        -------
        This calculcation is not exact and tends to underestimate the number of
        points, but it works for the generator.

        See: https://en.wikipedia.org/wiki/N-sphere

        """
        if nFeatures is None:
            nFeatures = self._nFeatures

        gamma = scipy.special.gamma((nFeatures+1)/2)
        surfaceArea = 2 * pi**((nFeatures+1)/2) / gamma
        factor = surfaceArea / 2

        if reverse:
            dist = power(factor/nPoints, (1./(nFeatures-1)))
            return dist
        else:
            points = factor/(dist**(nFeatures-1))
            return points

    def generate(self, nFeatures: int, dist: float,
                 factor: float | None = None) -> np.ndarray:
        """
        Returns a set of points approximately evenly distributed
        over the surface of a hypersphere in d dimensions with
        dist spacing. This is done with a hyperspiral.

        Parameters
        ----------
        nFeatures: int
            Number of dimensions
        dist: float
            Target distance between the points

        Returns
        ----------
        x: numpy matrix (n, nFeatures)
            Returns n points in nFeatures dimensions
            in cartesian coordinates.

        Notes
        -------


        """
        self._nFeatures = nFeatures
        self._dd = dist**2
        screws = pi/dist

        if not (factor is None):
            # for testing different parameters
            self._points = factor/dist**(self._nFeatures-1)
        else:
            self._points = self._pointsCalculator(dist=dist)

        # Set up angles
        sPower = power(screws, arange(self._nFeatures-1))
        self._angBase = ones(self._nFeatures-1)*pi*sPower
        # Last one is full
        self._angBase[-1] *= 2.0

        self._runLoop()

        x = vstack(self._pointsList)
        return x

    def _runLoop(self):
        # Setup variables
        dt = (1/10)*(1.0/self._points)
        lastCoor = zeros(self._nFeatures)
        lastCoor[0] = 1.0
        pointsList = []
        pointsList.append(lastCoor)

        t = float()
        while True:
            dtt = dt
            j = 0
            while j < self._maxLoops:
                t += dtt
                ang = t*self._angBase
                coor = cartesian_coordinates(array([1], dtype=np.float64), ang)
                # Square is faster than power
                distanceToLast = npSum(square(coor - lastCoor))
                if not j and distanceToLast < self._dd:
                    j -= 1
                    t -= dtt
                    dtt *= 4.0
                    continue
                if distanceToLast > self._dd:
                    t -= dtt
                if abs(distanceToLast-self._dd) < self._atol:
                    # close enough, go to next point
                    break
                j += 1
                dtt *= 0.5
            if (t > 1.0):
                break
            ang = t*self._angBase
            coor = cartesian_coordinates(array([1], dtype=np.float64), ang)
            lastCoor = coor
            pointsList.append(coor)

            if (j == self._maxLoops) or (len(pointsList) > self._maxPoints):
                raise RuntimeError(
                    'Calculating points on sphere hit max loops.')

        self._pointsList = pointsList

# --------------------------------------------------------------------
# Classes


class _spherePointFinder():
    def __init__(self, nFeatures: int,
                 pointsRange: List[int] | set[int] | None = None,
                 maxLoops: int = 100, intialStep: float = None,
                 minGuess: float = 0.1, intialGuess: float = 1.5,
                 nSlack: int = 1):

        self._maxLoops = maxLoops
        self._minGuess = minGuess
        self._intialGuess = intialGuess
        self._nFeatures = nFeatures
        self._allowedSlack = nSlack

        if intialStep is None:
            if nFeatures < 7:
                self._intialStep = 0.1
            elif nFeatures < 10:
                self._intialStep = 0.01
            elif nFeatures < 13:
                self._intialStep = 0.001
            else:
                self._intialStep = 0.1
        else:
            self._intialStep = intialStep

        self._nList = []
        self._pointsList = []
        self._pointsListKeep = []

        if pointsRange is None:
            self._pointsRange = {3, 4, 5, 6, 8, 9, 10, 11, 13, 14,
                                 15, 16, 18, 19, 20, 25, 50, 90,
                                 100, 150, 190, 200}
        else:
            self._pointsRange = pointsRange

        self._pointsRangeArray = array(list(self._pointsRange))

    def fillFromDf(self, df: pd.DataFrame):
        self._pointsList = df.to_dict('records')
        self._pointsListKeep = copy.copy(self._pointsList)
        self._nList = df['nPoints'].tolist()

    def addPointToSearch(self, nPoints: int, newSlack: int | None = None):
        if newSlack:
            self._allowedSlack = newSlack

        self._pointsRange.add(nPoints)
        self._pointsRangeArray = array(list(self._pointsRange))

        # check if points need to be added to keep
        if self._pointsListKeep:
            nKept = array([pDict['nPoints']
                          for pDict in self._pointsListKeep])
            if absolute(nKept[argmin(absolute(nKept-nPoints))]-nPoints) <= self._allowedSlack:
                # valid point already inside
                return

        if self._pointsList:
            nArr = array(self._nList)
            minIdx = argmin(absolute(nArr-nPoints))
            minVal = nArr[minIdx]
            if absolute(minVal-nPoints) <= self._allowedSlack:
                self._pointsListKeep.append(self._pointsList[minIdx])
            else:
                self._lookForMissingPoint(nPoints)

    def _gridSearch(self):
        # Grid search
        t = self._intialGuess

        grid = linspace(self._intialGuess, self._minGuess,
                        retstep=-1*self._intialStep)[0]
        gen = _evenlySpacedPointsOnSphere()

        for idx in range(len(grid)):
            if idx != 0 and max(self._nList) > max(self._pointsRange):
                break

            # Get points
            t = grid[idx]
            for jdx in range(self._maxLoops):
                try:
                    x = gen.generate(self._nFeatures, t)
                    self._checkAndSavePoint(x, t)
                    break
                except RuntimeError:
                    t -= self._intialStep/(2**(jdx+1))

    def _addPointToFound(self, xLoc: np.ndarray, tLoc: float,
                         saveToFile=False):
        r, phiLoc = angular_coordinates(xLoc)
        pointsDict = {'nFeatures': xLoc.shape[1],
                      'nPoints': xLoc.shape[0],
                      'angles': phiLoc,
                      'distance': tLoc
                      }
        self._pointsList.append(pointsDict)
        self._nList.append(xLoc.shape[0])
        if saveToFile:
            self._pointsListKeep.append(pointsDict)

    def _checkAndSavePoint(self, xLoc: np.ndarray, tLoc: float):
        nPoints = xLoc.shape[0]
        if not (nPoints in self._nList):
            # Accidentally find another missing number?
            # Closest n
            closestDiff = amin(absolute(self._pointsRangeArray - nPoints))
            if closestDiff <= self._allowedSlack:
                self._addPointToFound(xLoc, tLoc, saveToFile=True)
            else:
                self._addPointToFound(xLoc, tLoc, saveToFile=False)

    def _lookForMissingPoint(self, findN: int):
        # Get guess from closest smaller n
        nArray = array(self._nList)
        lastN = amax(nArray[nArray < findN])
        lastIdx = argwhere(nArray == lastN)[0][0]

        # Look for larger N
        if npSum(nArray > findN):
            nextN = amax(nArray[nArray > findN])
            nextIdx = argwhere(nArray == nextN)[0][0]
            # Start halfway
            lastT = self._pointsList[lastIdx]['distance']
            nextT = self._pointsList[nextIdx]['distance']
            dt = abs(lastT-nextT)/2
            t = lastT-dt
        else:
            # Start with step
            dt = self._intialStep
            t = self._pointsList[lastIdx]['distance']-dt

        # Step until n is found or maxLoops is hit
        loopIdx = 0
        gen = _evenlySpacedPointsOnSphere()
        while loopIdx < self._maxLoops:
            try:
                x = gen.generate(self._nFeatures, t)
            except RuntimeError:
                t -= dt
                loopIdx += 1
                continue

            nPoints = x.shape[0]
            dt *= 0.75

            if (nPoints == findN) | (abs(nPoints-findN) <= self._allowedSlack):
                # found, can move on
                self._addPointToFound(x, t, saveToFile=True)
                break
            else:
                self._checkAndSavePoint(x, t)

            # Step distance
            if nPoints > findN:
                t += dt
            else:
                t -= dt

            loopIdx += 1

    def findPoints(self):
        pointsSet = set(self._pointsRange)

        if not self._pointsList:
            self._gridSearch()

        # Find missing points, if they aren't in list
        if list(pointsSet - set(self._nList)):
            for findN in self._pointsRange:
                if (findN in self._nList):
                    # Already have those points, move on
                    # Needed because nList can update
                    continue

                self._lookForMissingPoint(findN)

        # Check if any numbers are missing after all loops completed
        self.checkPointsFound()

    def checkPointsFound(self):
        # Check if any numbers are missing after all loops completed
        nArr = array(self._nList)
        missingList = []
        for idx in range(len(self._pointsRangeArray)):
            targetN = self._pointsRangeArray[idx]
            if amin(absolute(nArr-targetN)) > self._allowedSlack:
                missingList.append(targetN)

        if missingList:
            msg = (f'Unable to find the following points '
                   f'with maximum number of loops {self._maxLoops} '
                   f'for dimensions {self._nFeatures}: {missingList} '
                   f'but did find: {self._nList}')
            warnings.warn(msg)

    def getPoints(self):
        return self._pointsListKeep

    def getValidN(self):
        # Take only allowed
        nValid = [pDict['nPoints'] for pDict in self._pointsListKeep]
        final = _addSlackNumbers(nValid, self._allowedSlack)
        return final


class _sphereParamManager():
    """
    Stores angles of points to be used later.

    Parameters
    ----------
    pathCache: string, optional
        Cache location. If None, caching is disabled. This is not
        recommended.
    clear: bool, optional
        If True, clear cache on initialization.
    """

    def __init__(self, pathCache: str | None = None, clear=False):
        self._cacheFilename = 'nBlobOverSphere'

        if pathCache is None:
            self._pathSave = None
            msg = ('Cache will not be used for spherical distributions'
                   'This may slow things down significantly.')
            warnings.warn(msg)
        else:
            subpath = os.path.join(pathCache, 'distributionParam')
            self._pathSave = os.path.abspath(subpath)
            os.makedirs(self._pathSave, exist_ok=True)

            if clear:
                searchName = f'*{self._cacheFilename}*'
                fileList = glob.glob(searchName, root_dir=self._pathSave)
                for f in fileList:
                    os.remove(os.path.join(self._pathSave, f))

    def buildParams(self, nFeatures: int,
                    pointsRange: None | set[int] | List[int] = None,
                    maxLoops: int = 10, intialStep: None | float = None,
                    minGuess: float = 0.1, intialGuess: float = 1.5):
        """
        Stores angles of points to be used later.

        Parameters
        ----------
        nFeatures: int
        pointsRange: list of ints
        maxLoops: int, optional
            The maximum number of allowed loops to find numbers of points.
        intialStep: float, optional
        minGuess: float, optional
        intialGuess: float, optional

        Notes
        -------
        This function is only called from test_nBlobsOverSphere_buildParam
        and the cache is built using those test parameters.
        Rebuild by running pytest.

        """
        finder = _spherePointFinder(nFeatures, pointsRange=pointsRange,
                                    maxLoops=maxLoops, intialStep=intialStep,
                                    minGuess=minGuess, intialGuess=intialGuess)

        finder.findPoints()
        df = pd.DataFrame(finder.getPoints())
        self._saveToCache(df, nFeatures)
        return df

    def checkValidParamInCache(self, dList: List[int], slack: int = 0):
        sharedN = set()
        for nFeatures in dList:
            filename = self._cacheFilename + f'-{nFeatures}d'
            filepath = os.path.join(self._pathSave, filename)

            try:
                df = pd.read_hdf(filepath, "table", index_col=None, header=0)
                havePoints = set(df['nPoints'].unique())
                points = _addSlackNumbers(havePoints, slack)
                if not sharedN:
                    sharedN = points
                else:
                    sharedN = sharedN & points
            except Exception:
                pass

        return sharedN

    def getCache(self, nFeatures: int):
        if self._pathSave is None:
            raise OSError('No cache set')

        filename = self._cacheFilename + f'-{nFeatures}d'
        filepath = os.path.join(self._pathSave, filename)

        # Check if cache has been built
        if not os.path.isfile(filepath):
            msg = ('Cache file for nBlobOverSphere does not exist. '
                   'Try running pytest to rebuild '
                   'or run test_nBlobsOverSphere_buildParam.')
            raise OSError(msg)

        # Get params from file
        df = pd.read_hdf(filepath, "table", index_col=None, header=0)
        return df

    def _saveToCache(self, df: pd.DataFrame, nFeatures: int):
        if self._pathSave is None:
            return
        # ignore pandas warning about pickling performance
        # Arrays of points are all different sizes, so this is easiest
        warnings.simplefilter(action='ignore',
                              category=pd.errors.PerformanceWarning)

        filename = self._cacheFilename + f'-{nFeatures}d'
        filepath = os.path.join(self._pathSave, filename)
        df.to_hdf(filepath, key="table", index=False)

    def _getParamFromDataframe(self, df: pd.DataFrame, nFeatures: int,
                               nPoints: int, slack: int = 0) -> tuple[float, float]:
        reducedDf = df[(df['nPoints'] == nPoints)]

        notFound = False
        if len(reducedDf) < 1:
            if slack == 0:
                # If no matching parameters found, raise error
                notFound = True
            else:
                # look for close
                nArr = df['nPoints'].to_numpy()
                indxMin = np.argmin(np.absolute(nArr-nPoints), keepdims=True)
                minVal = nArr[indxMin[0]]
                if abs(minVal-nPoints) <= slack:
                    # Found one
                    reducedDf = df[(df['nPoints'] == minVal)]
                    self._nBlobs = minVal
                else:
                    # If no matching parameters found given slack
                    notFound = True
        elif len(reducedDf) > 1:
            reducedDf.sort_values(by=['distance'], ascending=False)
            # Too many parameters, take first and warn user
            msg = (f'Found {len(reducedDf)} matching parameters '
                   f'for {nFeatures} dimensions with {nPoints} points. '
                   'Taking option with largest distance.')
            warnings.warn(msg)

        if notFound:
            msg = (f'No parameters found matching {nFeatures} '
                   f'dimensions with {nPoints} points '
                   f'with {slack} slack')
            raise ValueError(msg)

        # Save angles
        phi = reducedDf['angles'].values[0]
        distance = reducedDf['distance'].values[0]

        return phi, distance

    def _addToCache(self, nFeatures: int, nPoints: int, slack: int):
        cachedf = self.getCache(nFeatures)
        finder = _spherePointFinder(nFeatures, nSlack=slack)
        finder.fillFromDf(cachedf)
        finder.addPointToSearch(nPoints)
        newDf = pd.DataFrame(finder.getPoints())
        self._saveToCache(newDf, nFeatures)
        foundN = self._getParamFromDataframe(
            newDf, nFeatures, nPoints, slack=slack)
        return foundN

    def getParam(self, nFeatures: int, nPointsOverSphere: int, slack: int = 0):
        try:
            df = self.getCache(nFeatures)
            prm = self._getParamFromDataframe(df, nFeatures,
                                              nPointsOverSphere, slack=slack)
        except OSError as e:
            # No cache exists, make cache
            newDf = self.buildParams(nFeatures)
            prm = self._getParamFromDataframe(newDf, nFeatures,
                                              nPointsOverSphere, slack=slack)
        except ValueError as e:
            # have cache just need to add point
            prm = self._addToCache(nFeatures, nPointsOverSphere, slack)

        return prm


class nBlobOverSphere(blob):
    """
    A Guassian mixture distribution object where the points are distributed
    on the surface of a hypersphere approximately evenly using a hyperspiral.

    Parameters
    ----------
    nFeatures: int
        The number of features
        Minimum of 2
        Maximum of 14
    offset: float
        Offset between center blob and outer sphere
    var: float
        Variance of gaussian mixture
    nBlobs: int, optional
        Number of points used to construct the sphere approximation
    slackInN: int, optional
        Allowed +/- for number of points
        n will be set to closest available valid value within this range
    buildRun: bool, optional
        Used only to initalize object for building cache
    pathCache: str
        path to cache

    Notes
    -------
    Exact numbers of points over the spherical surface can not always be
    found by the point finder, and this is very common when large numbers
    of points are requested. The "slackInN" parameter should be adjusted
    accordingly.

    """
    _minVar = 0.01
    _minOffset = 0

    def __init__(self, nFeatures: int, offset: float, var: float,
                 nBlobs: int = 3, slackInN: int = 0, buildRun=False,
                 pathCache: str = param.pathCache):
        self._nFeatures = nFeatures
        self._nBlobs = nBlobs
        self._bandwidth = var**(1/2)
        self._slackInN = slackInN
        self._var = var
        self._offset = offset

        self._verify()

        if nFeatures == 2:
            self._phi = linspace(0, 2*pi, num=self._nBlobs).reshape((-1, 1))
            r = self._offset*ones(self._nBlobs)
            self._centers = cartesian_coordinates(r, self._phi)
            self._initKde()
        else:
            # Get from cache
            if not buildRun:
                # this can change self._nBlobs value
                try:
                    self._getFromCache(pathCache)
                except ValueError as e:
                    # Try again
                    warnings.warn(f'Retrying points search due to: {e}')
                    self._getFromCache(pathCache)

    def _verify(self):
        if self._var > self._minVar:
            self._covar = self._var*identity(self._nFeatures)
        else:
            raise ValueError((f'nBlobOverSphere: var must be greater than '
                             f'{self._minVar}. Given: {self._var}'))

        if self._offset <= self._minOffset:
            raise ValueError((f'nBlobOverSphere: offset must be more than '
                             f'{self._minOffset}. Given: {self._offset}'))

    def _getFromCache(self, pathCache: str):
        man = _sphereParamManager(pathCache=pathCache)
        phi, t = man.getParam(self._nFeatures, self._nBlobs,
                              slack=self._slackInN)
        self._phi = phi
        self._distance = t
        self._nBlobs = self._phi.shape[0]
        r = self._offset*ones(self._nBlobs)
        self._centers = cartesian_coordinates(r, self._phi)
        self._initKde()


class sphereAndBlob(blob):
    """
    A Guassian mixture distribution object where half the points
    are distributed on the surface of a hypersphere while half
    are drawn from a Guassian in the center.

    Parameters
    ----------
    nFeatures: int
        The number of features
        Minimum of 2
        Maximum of 14
    offset: float
        Offset between center blob and outer sphere
    var: float
        Variance of gaussian mixture
    nBlobs: int, optional
        Number of points used to construct the sphere approximation
    slackInN: int, optional
        Allowed +/- for number of points
        n will be set to closest available valid value within this range

    Notes
    -------
    Exact numbers of points over the spherical surface can not always be
    found by the point finder, and this is very common when large numbers
    of points are requested. The "slackInN" parameter should be adjusted
    accordingly.

    """

    def __init__(self, nFeatures: int, offset: float, var: float,
                 nBlobs: int = 3, slackInN: int = 0):
        self._sphereDist = nBlobOverSphere(nFeatures, offset, var,
                                           nBlobs=nBlobs, slackInN=slackInN)
        self._blobDist = oneBlob(nFeatures, 0, var, singleD=False)

    def generate(self, rng: np.random.Generator, samples: int) -> np.ndarray:
        ran = rng.integers(0, high=2, size=samples)
        a = self._sphereDist.generate(rng, int(np.sum(ran == 0)))
        b = self._blobDist.generate(rng, int(np.sum(ran == 1)))
        return vstack([a, b])

    def logPdf(self, x: np.ndarray) -> np.ndarray:
        a = self._sphereDist.logPdf(x)
        b = self._blobDist.logPdf(x)
        r = logaddexp(a, b)-log(2)
        return r
