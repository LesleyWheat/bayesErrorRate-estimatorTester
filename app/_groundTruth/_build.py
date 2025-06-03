# Imports
import multiprocessing
import os
import warnings
from typing import List

# Packages
import pandas as pd
import numpy as np

# Project
import config as param
from .._tools import findCache, saveCache, lowDensityFinder
from .._setup import childLogSetup

# Module
from ._calc import berCalc

# --------------------------------------------------------------------


def baseBerCalc(codename: str, nFeatures: int,
                pathCache=param.pathCache,
                seed=param.seed):
    """
    Generates a list of BER values.

    Parameters
    ----------
    codename: string
        Must be a valid distribution pair for
        distributions.tools.get_distPair with a dictionary in
        simBaseParameters in config.py.
    nFeatures: int
        Number of features.
    pathCache: cache location, optional
        Path used for caching
    seed: int, optional
        For setting custom seed

    Returns
    -------
    dfBer: pandas dataframe
        BER values.
    """
    # Check cache
    generator = _berGenerator(codename, nFeatures, seed=seed,
                              pathCache=os.path.join(pathCache, 'BER_GT'))
    runCalc, dfCache = generator.checkCache()

    if runCalc:
        # Build the parameter-BER list
        simDict = param.simBaseParameters[codename]
        generator.berGridCalc(simDict['offsetList'], simDict['varianceList'])
        generator.berIterCalc(nNew=param.berNewPointsPerLoop,
                              ymin=simDict['minBER'], ymax=simDict['maxBER'],
                              breakGap=param.maxBerGap,
                              maxLoops=param.maxBerLoops,
                              maxSamples=param.maximumBerSample)

        # Save found parameters
        dfBer = generator.saveToCache()
    else:
        dfBer = dfCache

    return dfBer


# --------------------------------------------------------------------


class _berGenerator():
    """
    Generates new BER points.

    Parameters
    ----------
    codename: string
        Must be a valid distribution pair for
        distributions.tools.get_distPair with a dictionary
        in simBaseParameters in config.py.
    nFeatures: int
        Number of features.
    seed: int, optional
        Seed value for random number generator, if None, then random
        default is used.
    noPrint: bool, optional
        Don't print out anything to console
    pathCache: string, optional
        Location of cache. If None, caching is disabled. Caching is
        recommended.
    """

    def __init__(self, codename: str, nFeatures: int, seed=None,
                 noPrint=False, pathCache=None):
        self._codename = codename
        self._nFeatures = nFeatures
        self._seed = seed
        self._noPrint = noPrint
        self._parameterBerList = []
        self._logger = multiprocessing.get_logger()
        if pathCache is None:
            self._pathCache = None
            msg = ('BER ground truth cache not used. May be much slower.')
            warnings.warn(msg)
        else:
            self._pathCache = os.path.join(os.getcwd(), pathCache)

    def _makeFilename(self):
        targetName = self._codename+'_'+str(self._nFeatures)+'d_baseBER'
        return targetName

    def checkCache(self):
        """
        Looks for existing cache file. If one is found, it is returned.
        """
        if self._pathCache is None:
            logger = childLogSetup(logTitle)
            logger.info("Not using cache")
            runCalc = True
            dfCache = None
            return runCalc, dfCache

        targetName = self._makeFilename()
        dfCache = findCache(self._pathCache, targetName)
        strId = f'{self._codename}-{self._nFeatures}d'
        logTitle = f"subproc-berBuild-"+strId

        runCalc = True
        if dfCache is None:
            logger = childLogSetup(logTitle)
            logger.info("No cached files found")
        else:
            Y = dfCache['pdfMonteCarlo'].to_numpy()
            minBER = param.simBaseParameters[self._codename]['minBER']
            maxBER = param.simBaseParameters[self._codename]['maxBER']
            berGap = calcGap(Y, minBER, maxBER)
            if berGap > param.maxBerGap:
                logger = childLogSetup(logTitle)
                msg = (strId+f': Need {param.maxBerGap} or '
                       f'less BER gap, got: {berGap}. '
                       f'Add more samples to cache.')
                logger.warning(msg)
            else:
                runCalc = False

        return runCalc, dfCache

    def _runBerCalc(self, offset: float, var1: float, var2: float):
        try:
            berCalc(self._parameterBerList, self._nFeatures,
                    self._codename, offset=offset,
                    var1=var1, var2=var2, noPrint=self._noPrint,
                    seed=self._seed)
        except Exception as e:
            msg = (f'Encountered error {e} on BER Calculation: '
                   f'{self._codename} {self._nFeatures} '
                   f'{offset} {var1} {var2} {self._seed} '
                   'Moving on to next value.')
            warnings.warn(msg)

    def getBerParameters(self):
        """
        Gets stored list.

        Returns
        -------
        parameterBerList: List of dictionaries
            Contains calculated BER values.

        Notes
        -------


        """
        return self._parameterBerList

    def saveToCache(self):
        """
        Saves BER values as file.

        Parameters:
        -------
        pathCache: path
            Where to cache files.

        Returns
        -------
        dfBer: Pandas Dataframe
            BER data

        Notes
        -------


        """
        dfBer = pd.DataFrame(self._parameterBerList)
        if not (self._pathCache is None):
            filename = self._makeFilename()+'.h5'
            saveCache(self._pathCache, filename, dfBer)
        return dfBer

    def berGridCalc(self, offsetList: List[float], varianceList: List[float]):
        """
        Calculates the BER values over a grid of parameters.

        Parameters
        ----------
        offsetList: iterable of floats of size (n)
            List of initial offsets to use.
        varianceList: iterable of floats of size (m)
            List of initial variances to use.

        Notes
        -------


        """
        # Build up the parameter-BER list
        self._parameterBerList = []

        for offset in offsetList:
            for var1 in varianceList:
                for var2 in varianceList:
                    if var1 == var2:
                        # Do not allow duplicates
                        continue
                    self._runBerCalc(offset, var1, var2)

    def _runCalcForNewPoints(self, newPoints: np.ndarray):
        # run simulation for new points
        for idx in range(newPoints.shape[0]):
            var1 = newPoints[idx, 0]
            var2 = newPoints[idx, 1]
            offset = newPoints[idx, 2]

            self._runBerCalc(offset, var1, var2)

    def _checkBerGap(self, ymin: float, ymax: float, breakGap: float, idxLoop: int):
        nBerSamples = len(self._parameterBerList)

        # Check spacing
        dfBer = pd.DataFrame(self._parameterBerList)
        Y = dfBer['pdfMonteCarlo'].to_numpy()
        maxGap = calcGap(Y, ymin, ymax)
        idealGap = (ymax-ymin)/nBerSamples

        # Warn if largest spacing is too large
        msg = (f'BER ground truth iterative sampling completed for '
               f'{self._codename} {self._nFeatures} with {nBerSamples} '
               f'samples in {idxLoop} loops. Min: '
               f'{np.min(Y)} Max: {np.max(Y)}. Resulting gap: {maxGap} '
               f'Ideal: {idealGap} Allowed: {breakGap}')
        if maxGap > breakGap:
            warnings.warn(msg, RuntimeWarning)
        else:
            self._logger.info(msg)

    def berIterCalc(self, nNew=2, ymin=0.01, ymax=0.49,
                    breakGap=0.01, nMinK=3,
                    maxLoops=10, maxSamples=100):
        """
        Generates new BER points.

        Parameters
        ----------
        nNew: int, optional
            Number of random points generated per found low density point.
            Maximum number of new points to use in each loop is nNew*nMinK.
        ymin: float, optional
            Minimum y value for search.
        ymax: float, optional
            Maximum y value for search.
        breakGap: float, optional
            Stop search when largest gap is less than breakGap.
        nMinK: int, optional
            Number of low density points to find.
        maxLoops: int, optional
            Maximum number of loops.
        maxSamples: int, optional
            Maximum number of samples of BER values and parameters to be
            generated.

        Notes
        -------


        """
        nBerSamples = len(self._parameterBerList)
        idxLoop = 0

        # Keep rng so different values are generated when the
        # same points are selected
        rng = np.random.default_rng(seed=self._seed)

        # Run batches of new points
        finder = lowDensityFinder(nMinK=nMinK, nNew=nNew,
                                  rng=rng, ymin=ymin, ymax=ymax,
                                  xMinLimits=[0.01, 0.01, 0.01])

        while (nBerSamples < maxSamples) & (idxLoop < maxLoops):
            tempDf = pd.DataFrame(self._parameterBerList)
            Y = tempDf['pdfMonteCarlo'].to_numpy()

            # Check every few loops if gaps are within spec
            gapSize = calcGap(Y, ymin, ymax)
            if gapSize < breakGap:
                # Break out early if there are no big spaces
                self._logger.info('Breaking BER calculation early.')
                break
            else:
                msg = (f'The current gap size is: {gapSize}. '
                       f'Goal: {breakGap}')
                self._logger.info(msg)

            # Fetch some new parameters
            X_var1 = tempDf['var1'].to_numpy()
            X_var2 = tempDf['var2'].to_numpy()
            X_offset = tempDf['offset'].to_numpy()
            cX = np.array([X_var1, X_var2, X_offset]).T
            xNew = finder.newPointsByDensity(cX, Y, covarRatio=gapSize)

            if not (xNew is None):
                # run simulation for new points
                self._runCalcForNewPoints(xNew)

            # Update for next loop
            nBerSamples = len(self._parameterBerList)
            idxLoop = idxLoop + 1

        # Check spacing
        self._checkBerGap(ymin, ymax, breakGap, idxLoop)

        # Hit maximum allowed number of loops
        if idxLoop >= maxLoops:
            # Use the warnings library so they can be caught by pytest
            warnings.warn('Hit max loops for BER iterative sampling')

# --------------------------------------------------------------------


def calcGap(Y: np.ndarray, ymin: float, ymax: float) -> float:
    """
    Function to calculate total gap over a range of values.

    Parameters
    ----------
    Y: numpy array of size (n)
        Points in y-space.
    ymin: float
        Minimum value of range. Should be between 0-0.5.
    ymax: float
        Maximum value of range. Should be between 0-0.5.

    Returns
    -------
    maxOfArr: float
        The largest gap within the range of interest.
    """
    maxInside = np.max(np.diff(np.sort(Y)))
    maxOfArr = max([maxInside, np.min(Y)-ymin, ymax-np.max(Y)])
    return maxOfArr
