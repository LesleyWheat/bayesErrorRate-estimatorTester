# Imports
from typing import List

# Packages
import numpy as np
from numpy import array, repeat, concatenate, arange
import tensorflow as tf
from sklearn.naive_bayes import GaussianNB

# Project
from bayesErrorRate import bayesClassifierError
from distributions import get_distPair

# Module
from ._estimators import *

# --------------------------------------------------------------------


def _naiveBayes(results: dict, X: np.ndarray, Y: np.ndarray):
    # Naive bayes
    gnb = GaussianNB()
    y_pred = gnb.fit(X, Y).predict(X)
    nb_acc = (Y != y_pred).sum() / X.shape[0]
    results.update({'NaiveBayesError': nb_acc})

# --------------------------------------------------------------------


def runEstimates(results: dict, setList: List[np.ndarray],
                 useGpuId: int | None = None):
    """
    Adds the results of the estimators to a dictionary.

    Parameters
    ----------
    results : dictionary
        Output dictionary
    setList: list of numpy arrays
        Arrays must be of shape (n, m) where n is the number of samples and
        m is the number of features. m must match for every array.

    Notes
    -----
    CLAKDE uses pytorch while other estimators use tensorflow.

    """
    # Some estimators take data points and labels
    nClasses = len(setList)
    nList = array([s.shape[0] for s in setList], dtype=np.int64)
    X = concatenate(setList, axis=0)
    Y = repeat(arange(nClasses), nList)

    if useGpuId is None:
        tfDevice = f'/CPU:0'
    else:
        tfDevice = f'/GPU:{useGpuId}'

    runCLAKDE(results, X, Y, useGpuId=useGpuId)
    _naiveBayes(results, X, Y)
    runGKDE(results, X, Y)

    with tf.device(tfDevice):
        X_tf = tf.convert_to_tensor(X, tf.float64)
        runGHP(results, X_tf, Y)
        runKNN(results, X_tf, Y, kmin=1, kmax=199, useLogger=True)

    j = (results[f'CKDE_LAKDE-LOO_ES'] + results['GHP_L'])/2
    results.update({f'GHP(L)-CKDE(LL)(ES)': j})

# --------------------------------------------------------------------


def runSingleSim(resultsList: List[dict], resultsDict: dict,
                 rng: np.random.Generator, useGpuId: int | None = None):
    """
    Preforms a simulation.

    Parameters
    ----------
    resultsList : list of dictionaries
        Output list.
    resultsDict : dictionary
        Contains parameters for the simulations.
    rng: numpy.random.Generator object

    """
    # Get distribution objects for both classes
    distList = get_distPair(resultsDict['simType'],
                            nFeatures=resultsDict['nFeatures'],
                            var1=resultsDict['var1'],
                            var2=resultsDict['var2'],
                            offset=resultsDict['offset'])

    # Create new dictionary and update, so original resultsDict is not modified
    results = dict()
    results.update(resultsDict)

    # Get samples from classes
    n = resultsDict['nClassSamples']
    setList = [np.asarray(dist.generate(rng, n)) for dist in distList]

    # Run the Bayes Classifier
    bc_e, be_v = bayesClassifierError(distList, setList)
    results.update({'bayesClassifierError': bc_e})

    # Run the BER estimators
    runEstimates(results, setList, useGpuId=useGpuId)

    # Append dictionary with estimator results to list
    resultsList.append(results)
