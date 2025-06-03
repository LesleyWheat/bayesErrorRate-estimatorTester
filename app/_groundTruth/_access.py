# Imports
from typing import List

# Packages
import numpy as np
import pandas as pd

# Project
import config as param

# Module
from ._build import baseBerCalc

# --------------------------------------------------------------------


def getBerList(runDict: dict, returnList=True,
               pathCache=param.pathCache,
               baseSamples=param.baseSamples) -> List[dict]:
    """
    Provides a list of dictionaries of parameters and their BER values
    for simulation.

    Parameters
    ----------
    runDict: dictionary
        Parameters for set of simulations.

    returnList: bool, optional
        If list of values should be returned.

    pathCache: string, optional
        Location of cache

    baseSamples: int, optional
        If returnList is True, how many samples to return

    Returns
    -------
    berDictList: list of dictionaries
    """
    codename = runDict['codename']
    nFeatures = runDict['nFeatures']
    df_ber = baseBerCalc(codename, nFeatures, pathCache=pathCache)

    if returnList:
        berMin = param.simBaseParameters[codename]['minBER']
        berMax = param.simBaseParameters[codename]['maxBER']
        berDictList = sampleBerList(df_ber, baseSamples,
                                    berMin=berMin, berMax=berMax)

        return berDictList

# --------------------------------------------------------------------


def sampleBerList(dfBer: pd.DataFrame, nSamples: int, berMin: float = 0,
                  berMax: float = 0.5, gtColName='pdfMonteCarlo') -> List[dict]:
    """
    Sample the BER dataframe to generate a uniformly spaced
    set of distribution parameters.

    Parameters
    ----------
    dfBer: pandas dataframe
        Dataframe of BER values and parameters.
    nSamples: int
        Number of samples to drawn from list of BER values and parameters.
    berMin: float, optional
        Greater than 0.
    berMax: float, optional
        Less than 0.5.
    gtColName: string, optional
        Ground truth column name in dataframe dfBer.

    Returns
    -------
    indxs: numpy array (n)
        A set of indicies of dfBer that create a near-uniform
        spacing of distribution parameters across the BER space.
    """
    # Extract from dataframe
    Y = dfBer[gtColName].to_numpy()

    Y_uniform = np.linspace(berMin, berMax, nSamples)

    # Look for closest values
    similarity = np.abs(Y.reshape((1, -1)) - Y_uniform.reshape((-1, 1)))

    # Select closeset values
    berIdxs = np.argmin(similarity, axis=1)

    berDictList = []

    for idx in range(len(berIdxs)):
        indx = berIdxs[idx]
        berDict = dfBer.loc[indx].to_dict()
        berDict.update({'idxSample': idx})

        berDictList.append(berDict)

    return berDictList
