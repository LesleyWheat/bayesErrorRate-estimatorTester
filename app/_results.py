# Imports
import os
import glob
import multiprocessing
import copy
from typing import List

# Packages
import pandas as pd

# Module
from ._tools import clearUniquePath, makeUniquePath

# --------------------------------------------------------------------


def _makeFilename(expDict: dict):
    codename = expDict['codename']
    d = expDict['nFeatures']
    n = expDict['nClassSamples']
    seed = expDict['seed']

    base = f'results_{codename}_{d}d_{n}n'

    if seed is None:
        name = (base+'.h5')
    else:
        name = (base+f'_{seed}seed.h5')

    return name


def _filelistToDataframe(pathSaveAll: str, fileList: List[str]):
    import os.path as path

    dataframeList = []
    for filename in fileList:
        filepath = path.join(pathSaveAll, filename)
        df = pd.read_hdf(filepath, "table", index_col=None, header=0)
        df.rename(columns={"d": "nFeatures"}, inplace=True, errors='ignore')
        df.rename(columns={"n": "nClassSamples"},
                  inplace=True, errors='ignore')
        dataframeList.append(df)

    df = pd.concat(dataframeList, axis=0, ignore_index=True)
    df = df.drop_duplicates()

    # Get subdf
    df.fillna(0, inplace=True)

    return df

# --------------------------------------------------------------------


def clearResultsUsingDict(pathSave: str, expDict: dict):
    """
    Removes results from output folder.

    Parameters
    ----------
    pathSave : absolute path
        Path to folder with files.
    expDict : dictionary
        Must have keys: nFeatures, nClassSamples, codename, seed

    Notes
    -----

    """
    import os.path as path

    if not path.exists(pathSave):
        return

    name = _makeFilename(expDict)
    basePath = path.join(pathSave, name)

    clearUniquePath(basePath)


def saveResultsUsingDict(pathSave: str, expDict: dict,
                         resultsDf: pd.DataFrame):
    """
    Save dataframe to file in h5 format.

    Parameters
    ----------
    pathSave : absolute path
        Path to folder to save file in.
    expDict : dictionary
        Must have keys: nFeatures, nClassSamples, codename, seed
    resultsDf : Pandas Dataframe
        Output data

    Notes
    -----

    """
    import os.path as path

    os.makedirs(pathSave, exist_ok=True)

    name = _makeFilename(expDict)
    basePath = path.join(pathSave, name)

    # save data in h5 format (better than csv)
    pathSaveResults = makeUniquePath(basePath)
    resultsDf.to_hdf(pathSaveResults, key="table", index=False)


def getResultsUsingDict(pathSave: str, expDict: dict, all=False):
    """
    Get results from files.

    Parameters
    ----------
    pathSave : absolute path
        Path to folder that save files are in.
    expDict : dictionary
        Must have keys: nFeatures, nClassSamples, codename, seed
    all: bool, optional
        If true, then get everything under codename, nFeatures,
        nClassSamples and ignores seeds.

    Returns
    ----------
    df: Pandas Dataframe
        Table with results.

    Notes
    -----

    """
    logger = multiprocessing.get_logger()

    os.makedirs(pathSave, exist_ok=True)

    baseName = ('results_'+str(expDict['codename'])+'_' +
                str(expDict['nFeatures'])+'d_' +
                str(expDict['nClassSamples'])+'n')

    # Collect results files
    if all:
        searchName = '*'+baseName+'*'
        searchPath = os.path.join(pathSave, searchName)
        fileList = glob.glob(searchPath, root_dir=pathSave)
    elif expDict['seed'] is None:
        searchName = '*'+baseName
        searchPath1 = os.path.join(pathSave, searchName+'.*')
        searchPath2 = os.path.join(pathSave, searchName+' *')
        fileList1 = glob.glob(searchPath1, root_dir=pathSave)
        fileList2 = glob.glob(searchPath2, root_dir=pathSave)
        fileList = fileList1 + fileList2
    else:
        searchName = '*'+baseName+'_'+str(expDict['seed'])+'seed*'
        searchPath = os.path.join(pathSave, searchName)
        fileList = glob.glob(searchPath, root_dir=pathSave)

    print(fileList)

    if not fileList:
        logger.info(f'No files found in {pathSave} for {searchName}')
        return None

    df = _filelistToDataframe(pathSave, fileList)

    return df


def getResultsAll(pathSaveAll: str, codename: str):
    """
    Gets all results.

    Parameters
    ----------
    pathSaveAll: absolute path
        Location where results are saved.
    simType: string
        Name of simulation type/codename.

    Returns
    ----------
    df: Pandas Dataframe
        Results in table format.

    Notes
    -----

    """

    # Collect results files
    fileList = glob.glob("*results_"+codename+"*",
                         root_dir=pathSaveAll)

    if not fileList:
        raise Exception('No files found')

    df = _filelistToDataframe(pathSaveAll, fileList)
    return df


def getResultsOneSim(pathSaveAll: str, simType: str, nFeatures: int,
                     nClassSamples: int):
    """
    Find results for only one set of simulations.

    Parameters
    ----------
    pathSaveAll: absolute path
        Location where results are saved.
    simType: string
        Name of simulation type/codename.
    nFeatures: int
        Number of features.
    n: int
        Number of samples per class.

    Returns
    ----------
    subdf: Pandas Dataframe
        Results in table format.

    Notes
    -----

    """
    expDict = {'codename': simType,
               'nFeatures': nFeatures,
               'nClassSamples': nClassSamples}
    subdf = getResultsUsingDict(pathSaveAll, expDict, all=True)

    if subdf is None:
        raise Exception(f'No files found in {pathSaveAll} for: {expDict}')

    if len(subdf) < 1:
        raise Exception(f'No data found in {pathSaveAll} for: {expDict}')

    return subdf


def findCommonResults(dfSearch: pd.DataFrame, dfExisting: pd.DataFrame):
    """
    Extract common columns and their data from a dataframe based
    on another dataframe.

    Parameters
    ----------
    dfSearch: Pandas Dataframe
        Take data from this dataframe.
    dfExisting: Pandas Dataframe
        Check for columns in this dataframe.

    Returns
    ----------
    df: Pandas Dataframe
        Results in table format.

    Notes
    -----

    """
    searchList = list(dfSearch.columns)
    existingList = list(dfExisting.columns)
    commonCol = list(set(searchList).intersection(existingList))
    df = pd.merge(dfSearch, dfExisting, how='inner', on=commonCol)[commonCol]
    return df


def removeCompletedResults(dfMain: pd.DataFrame, dfExisting: pd.DataFrame,
                           logger: object):
    """
    Returns simulations that have not been done.

    Parameters
    ----------
    dfMain: Pandas Dataframe
        Results that are wanted.
    dfExisting: Pandas Dataframe
        Finished results.
    logger: log object
        For generating warning and info messages.

    Returns
    ----------
    reducedList: list of dictionaries
        Simulations that have not been completed.

    Notes
    -----

    """

    df = findCommonResults(dfMain, dfExisting)

    if len(dfExisting.index) > len(df.index):
        msg = ('More results than requested already exist. '
               f'Found: {len(dfExisting.index)} simulation results '
               f'but only {len(df.index)} were asked for. '
               'Program may have been run previously with different parameters. '
               'Results may need to be cleared.')
        logger.warning(msg)

    if len(df.index) == len(dfMain):
        # Completed all parameter combos
        return []
    elif len(df.index) == 0:
        # No previously finished results
        return dfMain.to_dict('records')

    # Else remove existing results from sims to do list
    dfReduced = dfMain.loc[~dfMain.index.isin(
        dfMain.merge(df.assign(a='key'), how='left').dropna().index)]
    reducedList = dfReduced.to_dict('records')

    originalLen = len(dfMain.index)
    newLen = len(dfReduced.index)
    msg = (f'Deleted {originalLen-newLen} simulation parameters '
           'because they were already completed.')
    logger.info(msg)

    return reducedList


def combineResultsUsingDict(pathSave: str, expDict: dict):
    """
    Combines multiple results files into single files. Clears old files
    and saves new ones.

    Parameters
    ----------
    pathSave : absolute path
        Path to folder that save files are in.
    expDict : string
        Must have keys: nFeatures, nClassSamples, codename, seed

    Notes
    -----
    Saves memory. Cleans up lots of temp save files.

    """
    import warnings
    # Switch no seed to zero seed
    if expDict['seed'] == 0:
        tempDict = copy.deepcopy(expDict)
        tempDict['seed'] = None
        tempDf = getResultsUsingDict(pathSave, tempDict)
        if not (tempDf is None or tempDf.empty):
            rmdup = tempDf.drop_duplicates()
            if len(tempDf) > len(rmdup):
                warnings.warn((f"Duplicates detected in results: "
                               f"{len(tempDf)-len(rmdup)}"))
            clearResultsUsingDict(pathSave, tempDict)
            saveResultsUsingDict(pathSave, expDict, tempDf)

    tempDf = getResultsUsingDict(pathSave, expDict)
    if not (tempDf is None or tempDf.empty):
        rmdup = tempDf.drop_duplicates()
        if len(tempDf) > len(rmdup):
            warnings.warn((f"Duplicates detected in results: "
                           f"{len(tempDf)-len(rmdup)}"))
        clearResultsUsingDict(pathSave, expDict)
        saveResultsUsingDict(pathSave, expDict, tempDf)
