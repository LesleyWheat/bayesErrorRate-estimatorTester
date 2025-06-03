
# Imports
import warnings
from typing import List

# Packages
import pandas as pd
import numpy as np
import sklearn.metrics as metrics
import statsmodels.stats.weightstats as weightstats

# Project
import config as param

# --------------------------------------------------------------------


def errorCalc(df: pd.DataFrame, resultDict: dict, yName: str,
              xName='pdfMonteCarlo',
              lowerBound=2.5, upperbound=97.5,
              reweightBins=10, skewThresholdRatio=0.1):
    """
    Calculates MSE and bounds in percentage for a column of a dataframe
    compared to a second column.

    Parameters
    ----------
    df: pandas dataframe
        Contains error data.
    resultDict: dictionary
        Dictionary to store output values.
    yName: string
        Column becomes "Y_predict" array.
    xName: string, optional
        Default is the first monte carlo BER value for the
        expected value.
        Column becomes "Y_true" array.
    lowerBound: float, optional
        Percentile used for lower bound.
        Must be between 0 to 100 inclusive.
    upperBound: float, optional
        Percentile used for upper bound.
        Must be between 0 to 100 inclusive.
    reweightBins: int, optional
        Number of bins to use for reweighting.
    skewThresholdRatio: float, optional
        Ratio used to detect significant weighting differences.
        Will warn if the difference is above weighted MSE multipled
        by skewThresholdRatio.

    Notes
    -----
    MSE and percentile are reweighted evenly over the range of the
    Y_true or 'X' column. If values were evenly distributed orginally,
    and there are enough samples, difference in output should be minimal.

    """
    Y_true = 100*df[xName].to_numpy().reshape(-1)
    Y_pred = 100*df[yName].to_numpy().reshape(-1)
    unweightedMSE = metrics.mean_squared_error(Y_true, Y_pred)

    # Reweight so even across Y_true
    sortedTrue, sortedPred, weights = reweightErrors(Y_true, Y_pred,
                                                     reweightBins)
    MSE = metrics.mean_squared_error(sortedTrue, sortedPred,
                                     sample_weight=weights)

    # Weighted percentile
    m = weightstats.DescrStatsW(sortedTrue-sortedPred, weights=weights)
    per = m.quantile([lowerBound/100, upperbound/100], return_pandas=False)

    if abs(unweightedMSE - MSE) > MSE*skewThresholdRatio:
        warnings.warn(('Significant skew detected. '
                       f'Unweighted MSE: {unweightedMSE}. '
                       f'Weighted MSE: {MSE}'))

    # Update dictionary
    resultDict.update({yName+'-MSE': MSE})
    resultDict.update({yName+'-EB_2.5': per[0]})
    resultDict.update({yName+'-EB_97.5': per[1]})


def reweightErrors(yTrue: np.ndarray, yPred: np.ndarray,
                   nBins: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Weights errors evenly by calculating the number of values over
    a range of bins.

    Parameters
    ----------
    yTrue: Numpy Vector
        True values, to be used for weighting.
    yPred: Numpy Vector
        Associated estimator values.
    nBins: int
        Number of bins, should be smaller enough that no bin has zero
        yTrue values.

    Returns
    ----------
    sortedTrue: Numpy vector
        Sorted version of yTrue
    sortedPred: Numpy vector
        Sorted version of yPred
    weights: numpy vector
        Associated weight for each value in yTrue/yPred to get uniform
        distribution over yTrue.

    """
    sortIdxs = np.argsort(yTrue)
    sortedTrue = yTrue[sortIdxs]
    sortedPred = yPred[sortIdxs]

    hist, bin_edges = np.histogram(sortedTrue, bins=nBins)
    if np.min(hist) == 0:
        warnings.warn(('Bin with zero samples detected '
                       f'with {nBins} bins. '
                       'Suggested fix: reduce number of bins.'))
    avgInBin = np.average(hist)
    w = 1/(hist/avgInBin)
    weights = np.repeat(w, hist)

    return sortedTrue, sortedPred, weights


def _filterDf(df: pd.DataFrame, nFeatures: int, nClassSamples: int):
    mask = (df['nFeatures'] == nFeatures)
    mask = mask & (df['nClassSamples'] == nClassSamples)
    selected = df[mask].copy().reset_index()
    selected.fillna(0, inplace=True)
    return selected


def _getSimName(df: pd.DataFrame) -> str:
    simList = df['simType'].unique()

    if not len(simList) == 1:
        raise ValueError('Only one simulation/distribution type expected '
                         'in dataframe.')
    else:
        simType = simList[0]

    return simType


def collectErrors(df: pd.DataFrame, measuresList: List[str],
                  minSamples: int = param.baseSamples*param.simulationBatches):
    """
    Calculates error values from raw results data.

    Parameters
    ----------
    df: pandas dataframe
        Contains error data.
    measuresList: list of strings
        Names of the columns to have their errors calculated.
    minSamples: int, optional
        Ignore simulations sets with less than this number of completed
        results (ignores partial/incomplete simulation sets).

    Returns
    -----
    errorDf: Pandas Dataframe
        Calculated error values by simtype, nClassSamples, nFeatures and estimator.

    """
    simType = _getSimName(df)

    dList = df['nFeatures'].unique()
    nList = df['nClassSamples'].unique()
    errorList = []

    for nFeatures in dList:
        for nClassSamples in nList:
            selected = _filterDf(df, nFeatures, nClassSamples)
            if selected.empty or (selected.shape[0] < minSamples):
                continue

            collectedDict = dict()
            collectedDict.update({'nFeatures': nFeatures})
            collectedDict.update({'nClassSamples': nClassSamples})
            collectedDict.update({'simType': simType})

            for keyName in measuresList:
                try:
                    errorCalc(selected, collectedDict, keyName,
                              reweightBins=11)
                except Exception as e:
                    print(keyName)
                    print(e)

            errorList.append(collectedDict)

    errorDf = pd.DataFrame(errorList)

    if len(errorDf) < 1:
        raise Exception('No valid estimator error data found')

    return errorDf


class errorExtractor():
    """
    Calculates best estimator using mean square error (MSE).

    Parameters
    ----------
    errorDf: pandas dataframe
        Contains error data.
    sufixUB: string, optional
        Upper bound suffix
    sufixLB: string, optional
        Lower bound suffix

    """

    def __init__(self, errorDf: pd.DataFrame, sufixUB: str = '-EB_97.5',
                 sufixLB: str = '-EB_2.5'):
        self._errorDf = errorDf
        simList = self._errorDf['simType'].unique()
        self._sufixErr = '-MSE'
        self._sufixUB = sufixUB
        self._sufixLB = sufixLB

        if not len(simList) == 1:
            print(simList)
            raise ValueError(('Only one simulation/distribution type '
                             'expected in dataframe.'))
        else:
            self._simType = simList[0]

    def _fillVariables(self, dList: List[int], nList: List[int]):
        if not dList:
            self._dList = self._errorDf['nFeatures'].unique()
        else:
            self._dList = dList

        if not nList:
            self._nList = self._errorDf['nClassSamples'].unique()
        else:
            self._nList = nList

    def _fillEstimatorNamesFromSelf(self):
        cNames = self._errorDf.columns
        namesErr = [s for s in cNames if self._sufixErr in s]
        self._estimateNames = [s.split(self._sufixErr)[0] for s in namesErr]
        self._estimateNamesUB = [s for s in cNames if self._sufixUB in s]
        self._estimateNamesLB = [s for s in cNames if self._sufixLB in s]
        self._estimateNamesErr = namesErr

    def _fillEstimators(self, useEstimators: List[str]):
        if not useEstimators:
            self._fillEstimatorNamesFromSelf()
        else:
            namesErr = []
            estNamesUB = []
            estNamesLB = []
            estNames = useEstimators
            cNames = self._errorDf.columns
            for name in estNames:
                if name+self._sufixErr in cNames:
                    namesErr.append(name+self._sufixErr)
                    estNamesUB.append(name+self._sufixUB)
                    estNamesLB.append(name+self._sufixLB)

            self._estimateNamesErr = namesErr
            self._estimateNames = estNames
            self._estimateNamesUB = estNamesUB
            self._estimateNamesLB = estNamesLB

    def _filterDf(self, nFeatures: int, nClassSamples: int):
        maskFeatures = (self._errorDf['nFeatures'] == nFeatures)
        maskSamples = (self._errorDf['nClassSamples'] == nClassSamples)
        mask = maskFeatures & maskSamples
        return self._errorDf[mask]

    def _makeBasicDict(self, nFeatures: int, nClassSamples: int):
        collected = dict()
        collected.update({'nFeatures': nFeatures})
        collected.update({'nClassSamples': nClassSamples})
        collected.update({'simType': self._simType})
        return collected

    def _fillNaDict(self, collected: dict):
        collected.update({'lowestEstimateName': 'None'})
        collected.update({'lowestEstimateVal': np.nan})
        collected.update({'lowestEstimateLowerBound': np.nan})
        collected.update({'lowestEstimateUpperBound': np.nan})

    def _appendInvalidCollected(self, collected: dict):
        self._fillNaDict(collected)
        self._errorBestList.append(collected)

    def _getBounds(self, filteredDf: pd.DataFrame, lowestColName: str):
        selectedUb1 = filteredDf[self._estimateNamesUB]
        selectedLb1 = filteredDf[self._estimateNamesLB]
        selectedLb2 = selectedLb1.iloc[0]
        selectedUb2 = selectedUb1.iloc[0]
        lowBoundName = lowestColName.split(self._sufixErr)[0]+self._sufixLB
        upperBoundName = lowestColName.split(self._sufixErr)[0]+self._sufixUB
        lowerBound = selectedLb2[lowBoundName]
        upperBound = selectedUb2[upperBoundName]

        return lowerBound, upperBound

    def _findlowestbyMse(self, filteredDf: pd.DataFrame, collected: dict):
        selected = filteredDf[self._estimateNamesErr]

        if selected.empty:
            # Not enough to calculate
            self._appendInvalidCollected(collected)
            return

        rowValues = selected.iloc[0].to_numpy()
        lowestIdx = np.argmin(rowValues)

        lowestValue = rowValues[lowestIdx]
        lowestColName = selected.columns[lowestIdx]
        lowerBound, upperBound = self._getBounds(filteredDf, lowestColName)

        collected.update({'lowestEstimateName': lowestColName})
        collected.update({'lowestEstimateVal': lowestValue})
        collected.update({'lowestEstimateLowerBound': lowerBound})
        collected.update({'lowestEstimateUpperBound': upperBound})
        self._errorBestList.append(collected)

    def _collectForSimSet(self, nFeatures: int, nClassSamples: int):
        filteredDf = self._filterDf(nFeatures, nClassSamples)
        collected = self._makeBasicDict(nFeatures, nClassSamples)
        self._findlowestbyMse(filteredDf, collected)

    def _runLoop(self):
        self._errorBestList = []

        for nFeatures in self._dList:
            for nClassSamples in self._nList:
                self._collectForSimSet(nFeatures, nClassSamples)

    def calc(self, useEstimators: List[str] | None = None,
             dList: List[int] | None = None, nList: List[int] | None = None):
        """
        Calculates best estimator using mean square error (MSE).

        Parameters
        ----------
        useEstimators: list of strings, optional
            If None, all estimators are compared.
        dList: list of ints, optional
            If None, then all found d values are used.
        nList: list of ints, optional
            If None, then all found n values are used.

        Returns
        -----
        errorDf: Pandas Dataframe
            Calculated best error values and respective estimator.

        """
        self._fillVariables(dList, nList)
        self._fillEstimators(useEstimators)
        self._runLoop()
        return pd.DataFrame(self._errorBestList)
