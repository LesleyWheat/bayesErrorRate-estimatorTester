# Imports
from typing import List

# Packages
from numpy import zeros, average, var
from numpy import where, absolute, maximum
import numpy as np

# Project
from distributions import abstractDistribution

# --------------------------------------------------------------------
# Functions


def _calculateError(probabilityA: np.ndarray, probabilityB: np.ndarray,
                    nClasses: int, tol: float):
    # Class idx Error vs other
    labels1 = where(probabilityB > probabilityA, 1, 0)
    tooCloseA = absolute(probabilityB - probabilityA) < tol
    err = where(tooCloseA, 1/nClasses, labels1)
    return err


def bayesClassifierError(distList: List[abstractDistribution],
                         setList: List[np.ndarray],
                         tol: float = 1e-20) -> tuple[np.ndarray, np.ndarray]:
    """
    Runs the Bayes classifier over a set of points and class distributions
    to return the error.

    Parameters
    ----------
    distList: list of abstractDistribution objects
        Probability distributions to be used for calculation.
        Must be two or longer.
    setList: list of numpy arrays
        setList[i] should be associated with distList[i] 
    tol: float, optional
        Tolerance for idenfying random chance points. Chance of
        classification will be 1/(number of classes).

    Returns
    ----------
    error: double
        The expected error over the samples.
    var: double
        The variance of the error.

    Notes
    -------
    Assumes classes are balanced.

    """
    nClasses = len(distList)
    errorArr = zeros(nClasses)
    varArr = zeros(nClasses)

    for idx in range(nClasses):
        distA = distList[idx]
        dataA = setList[idx]
        scoreA = distA.logPdf(dataA)
        otherList = []

        for jdx in range(nClasses):
            if idx == jdx:
                continue

            distB = distList[jdx]
            otherList.append(distB.logPdf(dataA))

        scoreB = maximum.reduce(otherList)
        err = _calculateError(scoreA, scoreB, nClasses, tol)
        errorArr[idx] = average(err)
        varArr[idx] = var(err)

    error = average(errorArr)
    variance = average(varArr)

    return error, variance
