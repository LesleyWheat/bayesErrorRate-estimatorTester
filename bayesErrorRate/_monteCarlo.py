# Desc

# Imports
import functools
from typing import List

# Packages

# Import individual functions for speedup
from numpy import concatenate, asarray, zeros, maximum
from numpy import isnan, isneginf, average, var
from numpy import log, logaddexp, exp
import numpy as np

# Project
from distributions import abstractDistribution

# Module
from ._classifer import bayesClassifierError

# --------------------------------------------------------------------
# Functions


def _mcBerClasses(rng: np.random.Generator,
                  distList: List[abstractDistribution],
                  nPoints: int):
    """
    Estimates the true Bayes Error over a set of class distributions
    using Monte Carlo simulation.

    Parameters
    ----------
    rng: numpy.random.Generator object
        Random number generator for reproducible results.
    distList: list of distribution objects
        Must be two or longer.
    nPoints: int
        Target number of points to be generated for the estimate.
        Actual number of points used may be +/- number of classes.
        Must be greater than zero.

    Returns
    ----------
    error: double
        The expected error over the distributions.
    var: double
        The variance of the error.

    Notes
    -------
    Assumes classes are balanced.

    For further information see: S. Y. Sekeh, B. Oselio, and A. O. Hero,
    ‘Learning to Bound the Multi-Class Bayes Error’, IEEE Trans. Signal Process.,
    vol. 68, pp. 3793–3807, 2020, doi: 10.1109/TSP.2020.2994807.

    External function imports are done outside because it speds up the excution
    speed as it reduces the number of lookups.


    """
    samples = round(nPoints/len(distList))
    sampleList = [asarray(dist.generate(rng, samples)) for dist in distList]

    x = concatenate(sampleList, axis=0)
    logNC = log(len(distList))

    logpdfList = [dist.logPdf(x) for dist in distList]
    pdList = [pd-logNC for pd in logpdfList]
    log_fm = functools.reduce(logaddexp, pdList)
    if any(isnan(log_fm)):
        msg = (f'Nan in probability.')
        raise Exception(msg)

    pList = [(p - log_fm) for p in logpdfList]

    zeroProbBool = any(isneginf(log_fm))
    errArr = (1 - exp(functools.reduce(maximum, pList)-logNC))
    error = average(errArr)
    variance = var(errArr)

    if zeroProbBool:
        # Should be impossible since generated from distributions
        msg = (f'Probability of zero pdf detected for at least one point. '
               f'This should not be possible.')
        raise FloatingPointError(msg)

    return error, variance


def mcBerBatch(rng: np.random.Generator,
               distList: List[abstractDistribution],
               nLoops: int = 1000, nBatch: int = 1024, brute: bool = False):
    """
    Runs batchs to estimate the Bayes Error over all given distributions.
    Total number of points used is nLoops*nBatch.

    Parameters
    ----------
    rng: numpy.random.Generator object
        Random number generator for reproducible results.
    distList: list of distribution objects
        Must be two or longer.
    nLoops: int, optional
        Number of batchs to do.
    nBatch: int, optional
        Number of points to use per batch.
    brute: bool, optional
        If true, uses only classification result from Bayes Classifier
        which is generally less accurate.

    Returns
    ----------
    eT: double
        The expected Bayes Error Rate over the distributions.
    vT: double
        The variance of the error.

    Notes
    -------
    Assumes classes are balanced.

    """
    eList = zeros(nLoops)
    varList = zeros(nLoops)

    for i in range(nLoops):
        if brute:
            samples = round(nBatch/len(distList))
            setList = [dist.generate(rng, samples) for dist in distList]
            error, variance = bayesClassifierError(distList, setList)
        else:
            error, variance = _mcBerClasses(rng, distList, nBatch)
        eList[i] = asarray(error)
        varList[i] = asarray(variance)

    eT = average(eList)
    vT = var(eList)

    return eT, vT
