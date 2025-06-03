# Imports
import os
import glob

# Packages
import pandas as pd

# Module
from ._general import makeUniquePath

# --------------------------------------------------------------------


def findCache(pathCache: str, targetName: str):
    """
    Finds a file if it exists in the cache.

    Parameters
    ----------
    pathCache: path string
        Path to cache.
    targetName: string
        Search string for filenames, used to
        match filenames to be collected.

    Returns
    -------
    None or df: Pandas Dataframe
        If no files are found, None is returned.
        Else, df is returned, a dataframe containing
        the found data from the cache.
    """
    if not os.path.exists(pathCache):
        os.makedirs(pathCache)
        return None

    fileList = glob.glob("*"+targetName+"*", root_dir=pathCache)

    if not fileList:
        print(f'No matching files found in cache for {targetName}')
        return None

    dataframeList = []
    for filename in fileList:
        filepath = os.path.join(pathCache, filename)
        df = pd.read_hdf(filepath, "table", index_col=None, header=0)
        dataframeList.append(df)

    df = pd.concat(dataframeList, axis=0, ignore_index=True)
    df = df.drop_duplicates()

    return df


def saveCache(pathCache: str, filename: str, df: pd.DataFrame):
    """
    Saves a dataframe to cache as a file.

    Parameters
    ----------
    pathCache: path string
        Path to cache.
    filename: string
        Filename to use for df.
    df: Pandas Dataframe
        The data to be saved to the cache.
    """
    os.makedirs(pathCache, exist_ok=True)

    pathFile = makeUniquePath(os.path.join(pathCache, filename))
    df.to_hdf(pathFile, key="table", index=False)
