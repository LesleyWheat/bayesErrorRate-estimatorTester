# Imports
import functools
import logging
from concurrent.futures import TimeoutError
from typing import List
from collections.abc import Callable

# Packages
from pebble import ProcessPool

# --------------------------------------------------------------------


def runFuncPool(workerFunc: Callable, iterList: List,
                timeout: int, nMaxProc: int,
                *args, **kwargs):
    """
    Run a pool of functions on a single iterable with timeout.

    Parameters
    ----------
    workerFunc: function
        Function to run. (arg and kwargs must match function)
    iterList: List or similar iterable
        Inputs for workerFunc.
    timeout: int
        Seconds for each task to run before timing out.
    nMaxProc: int
        Number of workers to run.
    *args
        Passed on to workerFunc.
    **kwargs
        Passed on to workerFunc.

    Returns
    -----
    failedDictList: List
        Items of iterList which did not complete successfully

    """
    resultList = []
    failedDictList = []

    # Max_tasks = 1 means workers restart after each completed task
    # Needed to free up gpu memory
    with ProcessPool(max_workers=nMaxProc, max_tasks=1) as pool:
        future = pool.map(functools.partial(workerFunc, *args, **kwargs),
                          iterList, timeout=timeout)
        iterator = future.result()

        # go over everything, check for timeout
        idx = 0
        while True:
            try:
                result = next(iterator)
                resultList.append(result)
                if idx < len(iterList):
                    msg = (f'Completed {idx+1} of {len(iterList)}: '
                           f'{iterList[idx]}')
                    logging.info(msg)
            except StopIteration:
                break
            except TimeoutError as error:
                # Log error and move to next result
                problemDict = iterList[idx]
                msg = (f"Timeout on iterable index: {idx} "
                       f"with value: {problemDict}. "
                       "Skipping and will be placed in retry list.")
                logging.warning(msg)
                failedDictList.append(problemDict)
            except Exception as error:
                problemDict = iterList[idx]
                msg = (f"Error on iterable index: {idx} of "
                       f"{len(iterList)} with value: {problemDict}. "
                       f"Error: {error}. Skipping.")
                logging.error(msg)
                failedDictList.append(problemDict)
            finally:
                idx = idx+1

    msg = (f'Completed: all of {len(iterList)}')
    logging.info(msg)
    print(msg)

    return failedDictList
