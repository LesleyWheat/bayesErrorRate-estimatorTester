# Imports
import time
import warnings
import multiprocessing

# Packages
import numpy as np

# Project
import feebee_methods
import bayesErrorRate

# --------------------------------------------------------------------
# Functions


def runKNN(results: dict, X: np.ndarray, Y: np.ndarray,
           kmin=1, kmax=100, useLogger=False):
    """
    Adds the results of the KNN estimator to a dictionary.

    Parameters
    ----------
    results : dictionary
        Output dictionary
    X : numpy array
        The data array.
        Must be of shape (n, m) where n is the number of samples and
        m is the number of features.
    Y : numpy array
        The label array.
        Must be of shape (n) where n is the number of samples.
    kmin : int, optional
        Minimum number of nearest neighbours used.
    kmax: int, optional
        Maximum number of nearest neighbours used.
    """
    import feebee_methods.knn as knn

    distM = ["squared_l2"]
    if useLogger:
        logger = multiprocessing.get_logger()

    # Check kmax is possible,
    # If not then reduce
    if kmax > len(Y)-1:
        msg = (f"k-NN range is too large with range {kmin}-{kmax} "
               f"so kmax will be set to {len(Y)-1}.")
        if useLogger:
            logger.warning(msg)
        else:
            warnings.warn(msg)  # For pytest to detect
        kmax = len(Y)-1

    # Use only odd
    n_list = list(range(kmin, kmax+1))
    knn_list = [k for k in n_list if k % 2]

    start_time = time.time()
    for dist in distM:
        knn_r = knn.eval_from_matrix_loo(
            X, Y, knn_k=knn_list, knn_measure=dist)

        knn_H = np.empty(len(knn_list))
        knn_L = np.empty(len(knn_list))
        for idx in range(len(knn_list)):
            n = knn_list[idx]
            knn_UB = knn_r[f'measure={dist}, k={n}'][0]
            knn_LB = knn_r[f'measure={dist}, k={n}'][1]

            knn_H[idx] = knn_UB
            knn_L[idx] = knn_LB

        # Find lowest upper bound
        minIdx = np.argmin(knn_H, keepdims=True)[0]

        # Give warning if KNN minimum is at the end of the range
        if minIdx == len(knn_list)-1:
            optimalK = knn_list[minIdx]
            msg = (f"Optimal KNN {dist} k value is {optimalK} "
                   f"which is the maximum of range {kmin}-{kmax}. "
                   f"Extending KNN range may improve results. "
                   f"Dict: {str(results)}")
            if useLogger:
                logger.info(msg)
            else:
                warnings.warn(msg)  # For pytest to detect

        results.update({f'KNN({dist})_N': knn_list[minIdx]})
        results.update({f'KNN({dist})_H': knn_H[minIdx]})
        results.update({f'KNN({dist})_L': knn_L[minIdx]})
        results.update({f'KNN({dist})_M': (knn_H[minIdx]+knn_L[minIdx])/2})

    results.update({'time:KNN': round(time.time() - start_time, 2)})


def runGHP(results: dict, X: np.ndarray, Y: np.ndarray):
    """
    Adds the results of the GHP estimator from Feebee to a dictionary.

    Parameters
    ----------
    results : dictionary
        Output dictionary
    X : numpy array
        The data array.
        Must be of shape (n, m) where n is the number of samples and
        m is the number of features.
    Y : numpy array
        The label array.
        Must be of shape (n) where n is the number of samples.
    """
    import feebee_methods.ghp as ghp

    start_time = time.time()
    ghp_r = ghp.eval_from_matrix(X, Y)
    GHP_UB = ghp_r[list(ghp_r.keys())[0]][0]
    GHP_LB = ghp_r[list(ghp_r.keys())[0]][1]

    # Write to results dictionary
    results.update({'GHP_M': (GHP_UB+GHP_LB)/2})
    results.update({'GHP_H': GHP_UB})
    results.update({'GHP_L': GHP_LB})
    results.update({'time:GHP': round(time.time() - start_time, 2)})


def runGKDE(results: dict, X: np.ndarray, Y: np.ndarray):
    """
    Adds the results of the GKDE estimator from Feebee to a dictionary.

    Parameters
    ----------
    results : dictionary
        Output dictionary
    X : numpy array
        The data array.
        Must be of shape (n, m) where n is the number of samples and
        m is the number of features.
    Y : numpy array
        The label array.
        Must be of shape (n) where n is the number of samples.

    Notes
    -----
    Bandwidth list from Feebee is used with the addition of the
    silverman factor.
    """
    import feebee_methods.kde as fb

    # KDE
    bandwidthList = ['silverman', 0.0025, 0.05, 0.1, 0.25, 0.5]
    start_time = time.time()

    for b in bandwidthList:
        kde_r1 = fb.eval_from_matrix_kde(X, Y, kde_bandwidth=b)
        GKDE = kde_r1[list(kde_r1.keys())[0]][0]
        results.update({f'GKDE_{b}': GKDE})

    results.update({'time:GKDE': round(time.time() - start_time, 2)})


def runCLAKDE(results: dict, X: np.ndarray, Y: np.ndarray,
              useGpuId: int | None = None):
    """
    Adds the results of the CKDE estimator to a dictionary.

    Parameters
    ----------
    results : dictionary
        Output dictionary
    setA : numpy array
        Must be of shape (nA, m) where nA is the number of samples and
        m is the number of features.
    setB : numpy array
        Must be of shape (nB, m) where nB is the number of samples and
        m is the number of features.

    Notes
    -----
    Assumes balanced classes.

    """
    from bayesErrorRate import CLAKDE

    start_time = time.time()
    jDict = CLAKDE(X, Y, useGpuId=useGpuId, rtol=1e-5)
    results.update(jDict)
    totalTime = round(time.time() - start_time, 2)
    results.update({f'time:CKDE_LAKDE-LOO_ES': totalTime})
