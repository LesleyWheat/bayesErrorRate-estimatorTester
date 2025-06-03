# Imports
import os
import logging
from typing import List

# Packages
import pandas as pd

# Project
import config as param

# Module
from ._results import findCommonResults, getResultsUsingDict
from ._taskLists import buildSimParamList
from ._groundTruth import getBerList
from ._tools import makeFullDictList

# --------------------------------------------------------------------


def verifyAll():
    """
    Checks all simulation results based on config.py parameters.
    """
    expDictList = buildSimParamList()
    incompleteList = checkExistingResults(expDictList)
    print(incompleteList)

# --------------------------------------------------------------------


def checkExistingResults(dictList: List[dict]) -> List[dict]:
    """
    Checks for simulation scenarios that don't have enough results
    based on the requested parameters.

    Parameters
    ----------
    dictList : List of dictionaries
        Simulation parameters to check.

    Returns
    ----------
    incompleteDictList: List of dictionaries
        List of incomplete simulation parameters to be done.

    Notes
    -----
    It only checks for the number of results, not if those results
    would have been the requested simulations.

    """
    pathSave = os.path.abspath(os.path.join(os.getcwd(), param.pathSave))
    incompleteDictList = []

    for idx in range(len(dictList)):
        expDict = dictList[idx]
        completed = _verifyCompleted(expDict, pathSave)
        if not completed:
            incompleteDictList.append(expDict)

    return incompleteDictList


def _verifyCompleted(expDict: dict, pathSave: str):
    # BER List should already be cached, just needs to be retrieved
    parameterBerList = getBerList(expDict)
    fullList = makeFullDictList(parameterBerList, expDict)
    parameterBerDf = pd.DataFrame(fullList)
    existingResultsDf = getResultsUsingDict(pathSave, expDict)

    if (existingResultsDf is None) or (len(existingResultsDf) < 1):
        completed = False
        msg = "No completed files found, starting from scratch"
    else:
        rawNum = len(existingResultsDf.index)
        noDupNum = len(existingResultsDf.drop_duplicates().index)

        if rawNum > noDupNum:
            logging.error(f'Oh no! Duplicates detected in {expDict}')

        completed, msg = _verifySim(parameterBerDf, existingResultsDf)

    msgLog = f"{expDict} "+msg
    print(msgLog)

    if completed:
        logging.info(msgLog)
    else:
        logging.warning(msgLog)

    return completed


def _verifySim(searchDf: pd.DataFrame, existingDf: pd.DataFrame):
    df = findCommonResults(searchDf, existingDf)

    existingSimNum = len(existingDf.index)
    doneSimNum = len(df.index)
    searchSimNum = len(searchDf.index)
    completed = False
    msg = ''
    msgA = ''

    msgB = (f'Searching for: {searchSimNum}. '
            f'Found: {existingSimNum}. '
            f'Common: {doneSimNum}. ')

    if doneSimNum == searchSimNum:
        completed = True
        msg = 'Found all simulation results. Skipping.'
        return completed, msg
    elif existingSimNum >= searchSimNum:
        msgA = ('Found enough results but they were not '
                'the ones that were expected. Signalling complete. ')
        completed = True
    elif existingSimNum < searchSimNum:
        msgA = ('Found less results than expected. '
                'Missing results. ')
    elif doneSimNum < searchSimNum:
        msgA = ('Missing results. ')

    if msgA:
        msg = msgA+msgB

    return completed, msg
