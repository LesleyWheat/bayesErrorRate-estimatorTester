# Imports
from typing import List, Literal

# Module
from ._base import abstractDistribution
from ._blobs import oneBlob, twoBlob, threeBlob
from ._radial import nBlobOverSphere, sphereAndBlob

# --------------------------------------------------------------------
# Get Distributions


def _getGvG(nFeatures, offset, var1, var2):
    distA = oneBlob(nFeatures, 0, var1)
    distB = oneBlob(nFeatures, offset, var2)
    distList = [distA, distB]
    return distList


def _getTvT(nFeatures, offset, var1, var2):
    distA = threeBlob(nFeatures, offset, var1)
    distB = twoBlob(nFeatures, offset/2, var2)
    distList = [distA, distB]
    return distList


def _getTvS(nFeatures, offset, var1, var2):
    nB = 200
    slack = 2
    distA = threeBlob(nFeatures, offset, var1)
    distB = nBlobOverSphere(nFeatures, offset/2, var2,
                            nBlobs=nB, slackInN=slack)
    distList = [distA, distB]
    return distList


def _getSvS(nFeatures, offset, var1, var2):
    slack = 2
    nB1 = 190
    nB2 = 200
    distA = sphereAndBlob(nFeatures, offset, var1,
                          nBlobs=nB1, slackInN=slack)
    distB = nBlobOverSphere(nFeatures, offset/2, var2,
                            nBlobs=nB2, slackInN=slack)
    distList = [distA, distB]
    return distList

# --------------------------------------------------------------------
# Functions


def get_distPair(simName: Literal['gvg', 'tvt', 'tvs', 'svs'],
                 nFeatures: int = 2,
                 var1: float = 1,
                 var2: float = 1,
                 offset: float = 1) -> List[abstractDistribution]:
    """
    Returns two distributions for a simulation type.

    Parameters
    ----------
    simName: string
        Key of distribution to be retrieved
    nFeatures: int, optional
        Number of features
    var1: float, optional
        Variance of first distribution
    var2: float, optional
        Variance of second distribution
    offset: float, optional
        Offset factor to be used

    Returns
    ----------
    distList: List of distribution objects
        At least two distributions
    """

    if simName == 'gvg':
        distList = _getGvG(nFeatures, offset, var1, var2)
    elif simName == 'tvt':
        distList = _getTvT(nFeatures, offset, var1, var2)
    elif simName == 'tvs':
        distList = _getTvS(nFeatures, offset, var1, var2)
    elif simName == 'svs':
        distList = _getSvS(nFeatures, offset, var1, var2)
    else:
        raise ValueError(f'get_distPair: Invalid sim type {simName}')

    return distList
