# Imports
from typing import List

# Packages
import pandas as pd

# Project
import config as param

# Module
from ._groundTruth import getBerList
from ._tools import runFuncPool

# --------------------------------------------------------------------
#  Simulation Parameters


def _buildSimList() -> List[dict]:
    expDictList = []
    batchRange = list(range(param.simulationBatches))
    seedList = [i * param.baseSamples for i in batchRange]
    for idx in range(len(seedList)):
        seed = seedList[idx]
        expIdx = idx
        for codename in param.simNameList:
            dList = param.simBaseParameters[codename]['dList']
            nList = param.simBaseParameters[codename]['nList']
            for nClassSamples in nList:
                for nFeatures in dList:
                    expDictList.append({'codename': codename,
                                        'nFeatures': nFeatures,
                                        'nClassSamples': nClassSamples,
                                        'idx': expIdx,
                                        'seed': seed})
                    expIdx = expIdx+1

    return expDictList


def _makeListFromParam() -> List[dict]:
    runList = []
    for codename in param.simNameList:
        dList = param.simBaseParameters[codename]['dList']

        # Create run parameters
        for nFeatures in dList:
            runParam = {'codename': codename,
                        'nFeatures': nFeatures}
            runList.append(runParam)
    return runList


def buildSimParamList() -> List[dict]:
    """
    Build list of simulation parameters.

    Returns
    ----------
    expDictList: List of dictionaries
        List of simulation parameters to be done.

    Notes
    -----
    Uses values from config.py

    """
    timeout = param.timeLimitSec*(param.baseSamples+1)

    # For each simulation type
    # Build all in multiproccess
    runList = _makeListFromParam()
    reportedFailDictList = runFuncPool(getBerList, runList,
                                       timeout, param.maxCores,
                                       returnList=False)

    print('Finished creating BER ground truth')

    # Create the list of simulation parameters
    expDictList = _buildSimList()

    # Order list
    df = pd.DataFrame(expDictList)
    df = df.sort_values(['nFeatures', 'seed', 'idx'],
                        ascending=[False, False, False])
    expDictList = df.to_dict('records')

    return expDictList
