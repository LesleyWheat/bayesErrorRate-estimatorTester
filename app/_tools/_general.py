# Imports
import os
from typing import List

# --------------------------------------------------------------------


def makeUniquePath(path: str):
    """
    Finds a unique name if a file with that name already exists.
    Appends a uniques identifier in the form "filename (X).ext"

    Parameters
    ----------
    path: path string
        Includes filename

    Returns
    -------
    newPath: path string
        Returns the original path if it's unique
    """
    filename, ext = os.path.splitext(path)
    n = 1
    newPath = path

    while os.path.exists(newPath):
        newPath = f'{filename} ({str(n)}){ext}'
        n += 1

    return newPath


def clearUniquePath(path: str):
    """
    Deletes all files with the base name of "filename (X).ext"

    Parameters
    ----------
    path: path string
        Includes filename.
    """
    filename, ext = os.path.splitext(path)
    n = 1
    newPath = path

    while os.path.exists(newPath):
        os.remove(newPath)
        newPath = f'{filename} ({str(n)}){ext}'
        n += 1

# --------------------------------------------------------------------


def makeFullDictList(dictList: List[dict], expDict: dict) -> List[dict]:
    """
    Combines lists for simulations.

    Parameters
    ----------
    dictList: list of dictionaries
        Individual simulation parameters.
    expDict: dictionary
        Global simulation parameters.

    Returns
    -------
    fullList: list of dictionaries
        Combined list.

    """
    fullList = []
    for idx in range(len(dictList)):
        berDict = dictList[idx]
        berDict.update({'nClassSamples': expDict['nClassSamples']})
        simSeed = expDict['seed'] + berDict['idxSample']
        berDict.update({'seed': simSeed})
        fullList.append(berDict)
    return fullList
