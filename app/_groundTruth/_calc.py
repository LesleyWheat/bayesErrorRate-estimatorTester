# Imports
import time
import multiprocessing
from typing import List

# Packages
import numpy as np

# Project
import config as param
from distributions import get_distPair, abstractDistribution
from bayesErrorRate import mcBerBatch

# --------------------------------------------------------------------


def berCalc(berList: List[dict], nFeatures: int, codename: str,
            var1: float = 1, var2: float = 1, offset: float = 1,
            seed: int | None = None, noPrint=False,
            nLoops: int = 100, nBatch: int = 1024):
    """
    Function to calculate total gap over the desired BER range.

    Parameters
    ----------
    berList: List of dictionaries
        Dictionary is appended to this list.
    nFeatures: int
        Number of features.
    codename: string
        Must be a valid distribution pair for
        distributions.tools.get_distPair with a dictionary in
        simBaseParameters in config.py.
    var1: float, optional
        Variance one value.
    var2: float, optional
        Variance two value.
    offset: float, optional
        Offset value for distributions.
    seed: int, optional
        For reproducibility.
    noPrint: Bool, optional
        Don't print out anything to console.
    nLoops: int, optional
        Number of loops to use.
    nBatch: int, optional
        Batch size per loop.
    """
    logger = multiprocessing.get_logger()

    def logBerCalcStart():
        import datetime
        import time

        # Log the start
        logStr = (f"{datetime.datetime.now()} BER calc {codename} "
                  f"var: {var1} var2: {var2} offset: {offset} d: {nFeatures}")
        logger.info(logStr)
        if not noPrint:
            if param.reducedPrint:
                if (round(time.time() * 1000) % 100) == 0:
                    print(logStr)
            else:
                print(logStr)

    # Log the start
    logBerCalcStart()

    # Set seed
    rng = np.random.default_rng(seed=seed)

    # Setup results
    resultsPDF = dict()
    resultsPDF.update({'simType': codename})
    resultsPDF.update({'nFeatures': nFeatures})
    resultsPDF.update({'var1': var1})
    resultsPDF.update({'var2': var2})
    resultsPDF.update({'offset': offset})

    # Get samples from the distributions
    distList = get_distPair(codename, nFeatures=nFeatures, var1=var1, var2=var2,
                            offset=offset)

    # Calculate the ground truth BER
    try:
        _berEval(resultsPDF, rng, distList, nLoops=nLoops, nBatch=nBatch)
        berList.append(resultsPDF)
        logger.info('Completed: '+str(resultsPDF))
    except Exception as e:
        logger.exception(e)
        logger.warning('Skipping due to error: '+str(resultsPDF))


def _berEval(resPdf: dict, rng: np.random.Generator,
             distList: List[abstractDistribution],
             nLoops: int = 2000, nBatch: int = 512):
    """
    Calculates the BER ground truth for a set of distributions
    and adds it to a dictionary.

    Parameters
    ----------
    resPdf : dictionary
        Output dictionary, values are saved to this dictionary.
    rng: numpy.random.Generator object
        Random number generator for consistent results.
    distList: list of distribution objects
        Object representing the class distributions from which
        points can be drawn.
    nLoops: int, optional
        Number of batches to use in calculating BER.
        Controls calculation matrix size, may need to be optimized
        for different hardware.
    nBatch: int, optional
        Number of points to use in each batch for calculating BER.
        Controls calculation matrix size, may need to be optimized
        for different hardware.

    Notes
    ----------
    Total number of points used to caluclate BER: nint*nBatch.
    Calculation is preformed twice for validation.

    """

    start_time = time.time()
    e, v = mcBerBatch(rng, distList, brute=False,
                      nLoops=nLoops, nBatch=nBatch)
    resPdf.update({'pdfMonteCarlo': e})
    resPdf.update({'pdfMonteCarlo_var': v})
    runTime = round(time.time() - start_time, 3)
    resPdf.update({'time:pdfMonteCarlo': runTime})
