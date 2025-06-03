# Imports
import os
import time
import datetime
import multiprocessing
from typing import List

# Packages
import numpy as np
import pandas as pd
import tensorflow as tf
import torch

# Project
import config as param

# Module
from ._tools import resourceSelector, runFuncPool, makeFullDictList
from ._verify import checkExistingResults
from ._results import removeCompletedResults
from ._results import clearResultsUsingDict, getResultsUsingDict
from ._results import combineResultsUsingDict, saveResultsUsingDict
from ._simulation import runSingleSim
from ._setup import childLogSetup
from ._groundTruth import getBerList

# --------------------------------------------------------------------


def runSimPool(dictList: List[dict], nMaxProc: int, *args, **kwargs):
    """
    Runs a pool of simulations.

    Parameters
    ----------
    dictList: list of dictionaries
        Simulation parameters to be dones.
    nMaxProc: int
        Maximum number of parallel proccesses.
    *args:
        Passed on to simworker.
    **kwargs:
        Passed on to simworker.

    Notes
    -----
    Uses values from config.py

    """
    allSimTimeout = param.timeLimitSec*(param.baseSamples+1)

    if param.clearOldResultsOnStartup:
        # Clear here
        runList = dictList
        fullPath = os.path.join(os.getcwd(), param.pathSave)
        for expDict in dictList:
            clearResultsUsingDict(fullPath, expDict)
    else:
        runList = checkExistingResults(dictList)

    _overwriteIndicies(runList)

    reportedFailDictList = runFuncPool(_runSimWorker, runList,
                                       allSimTimeout, nMaxProc,
                                       *args, **kwargs)

    return reportedFailDictList


def _overwriteIndicies(runList: List[dict]):
    newIdx = 0
    for idx in range(len(runList)):
        runList[idx]['idx'] = newIdx
        newIdx = newIdx + 1


def _runSimWorker(expDict: dict, **kwargs):
    worker = _simWorker(expDict, **kwargs)
    worker.run()

# --------------------------------------------------------------------


class _simWorker():
    def __init__(self, expDict: dict,
                 baseSamples: int = param.baseSamples,
                 pathSave: str = param.pathSave,
                 pathCache: str = param.pathCache,
                 pathLog: str = param.pathLog,
                 maxErrorsAllowed: int = 0,
                 checkpointInterval: int = param.checkpointInterval,
                 simTimeLimit: int = param.timeLimitSec,
                 consoleReducedPrint: bool = param.reducedPrint):
        """
        Starts a set of simulations.

        Parameters
        ----------
        expDict: dictionary
            Dictionary with simulation parameters.
        baseSamples: int, optional
            Number of samples to use.
        pathSave: path, optional
            Folder location to save results.
        pathCache: path, optional
            Cache folder location.
        pathLog: path, optional
            Log folder location.
        maxErrorsAllowed: int, optional
            Number of errors allowed per simulation set before
            abandonment.
        checkpointInterval: int, optional
            How often to save results.
        simTimeLimit: int, optional
            Time limit for how long a simulation can go before it
            is killed if not completed.
        consoleReducedPrint: bool, optional
            Occassionally print out values to console to let user know
            things are still running.
        """

        self._expDict = expDict
        self._baseSamples = baseSamples
        self._maxErrorsAllowed = maxErrorsAllowed
        self._checkpointInterval = checkpointInterval
        self._simTimeLimit = simTimeLimit
        self._consoleReducedPrint = consoleReducedPrint

        self._strSimId = (f"{expDict['codename']}-{expDict['nFeatures']}-"
                          f"{expDict['nClassSamples']}-{expDict['seed']}")

        # Sim setup
        self._initFolders(pathSave, pathLog, pathCache)
        self._initSimVariables()

        self._resourceFinder = resourceSelector(self._expDict)

    def _initFolders(self, pathSave: str, pathLog: str, pathCache: str):
        self._pathSave = os.path.abspath(os.path.join(os.getcwd(), pathSave))
        self._pathCache = os.path.abspath(os.path.join(os.getcwd(), pathCache))
        self._pathLog = os.path.abspath(os.path.join(os.getcwd(), pathLog))
        os.makedirs(self._pathSave, exist_ok=True)
        os.makedirs(self._pathCache, exist_ok=True)
        os.makedirs(self._pathLog, exist_ok=True)

    def run(self):
        # Check for resources and pick which to use
        self._useGpuId = self._resourceFinder.getResource()

        self._logSetup()
        self._fullBerList()
        self._loadExistingResults()

        # Run simulation for each set of parameters
        redoNum = self._runSimulationLoop()

        if redoNum > 0:
            msg = (f'Unable to complete {redoNum} simulations')
            self._logger.warning(msg)
        else:
            # Done, save all results into a single file
            combineResultsUsingDict(self._pathSave, self._expDict)

        self._logger.info(f"PID: {os.getpid()} Shutting down.")

    def _saveResultsLocal(self):
        if self._tempResults:
            saveResultsUsingDict(self._pathSave, self._expDict,
                                 pd.DataFrame(self._tempResults))
        self._tempResults = []

    def _simErrorCheck(self):
        # Check if there's too many errors
        nSimTotal = len(self._berList)
        if self._errorCount > self._maxErrorsAllowed:
            # Save results, if any
            self._saveResultsLocal()
            errStr = (f'{self._strSimId}: Exceeded max allowed '
                      f'errors ({self._maxErrorsAllowed}) '
                      f'for simulation with {nSimTotal} tasks. '
                      'More information is in the task log. '
                      'Will retry if allowed.')
            self._logger.warning(errStr)
            raise Exception(errStr)

    def _checkCheckpoint(self):
        # Save results periodically
        if (time.time() - self._checkpointTime > self._checkpointInterval):
            self._logger.info(f"Hit checkpoint. Saving results.")
            self._saveResultsLocal()
            self._resourceFinder.logMemoryUsage(self._logger)
            self._simTimeTakenList = []
            self._checkpointTime = time.time()

    def _initSimVariables(self):
        # Sim setup
        self._tempResults = []
        self._simTimeTakenList = []
        self._errorCount = 0
        self._completeCount = 0
        self._checkpointTime = time.time()

    def _hitSimTimeout(self, simIdx: int, maxIdx: int):
        errStr = (f"{self._strSimId}: Time limit of "
                  f"{self._simTimeLimit}s reached for simulation "
                  f"on {simIdx} out of {maxIdx}")
        self._logger.warning(errStr)
        self._errorCount = self._errorCount+1

    def _hitSimError(self, e: Exception):
        errStr = f'{self._strSimId}: {e}'
        self._logger.exception(errStr)
        self._errorCount = self._errorCount+1

    def _finishedOneSim(self, timeTaken: float, simIdx: int, maxIdx: int):
        self._simTimeTakenList.append(timeTaken)
        self._logger.info((f"{self._strSimId}: time taken was "
                          f"{timeTaken} "
                           f"for {simIdx} out of {maxIdx}"))
        self._completeCount += 1

    def _runInPool(self, idx: int, nSimTotal: int, berDict: dict,
                   simKwargs: dict):
        # Set random number generator for every simulation
        rng = np.random.default_rng(seed=berDict['seed'])

        with multiprocessing.pool.ThreadPool() as pool:
            startSimTime = time.time()
            simArgs = (self._tempResults, berDict, rng)
            res = pool.apply_async(runSingleSim, simArgs, simKwargs)
            try:
                out = res.get(timeout=self._simTimeLimit)
                simTime = round(time.time() - startSimTime, 2)
                self._finishedOneSim(simTime, idx+1, nSimTotal)
            except multiprocessing.TimeoutError:
                self._hitSimTimeout(idx+1, nSimTotal)
            except Exception as e:
                self._hitSimError(e)
                raise Exception(e)
            finally:
                pool.terminate()
                pool.join()

    def _runSimulationLoop(self):
        self._initSimVariables()
        nSimTotal = len(self._berList)
        simKwargs = {'useGpuId': self._useGpuId}

        # Run simulation for each set of parameters
        for idx in range(nSimTotal):
            berDict = self._berList[idx]
            self._logSimStart(berDict)

            # Run in pool of one to allow for timeout
            try:
                self._runInPool(idx, nSimTotal, berDict, simKwargs)
                # Check if there's too many errors
                self._simErrorCheck()
            except Exception:
                break

            # Save results periodically
            self._checkCheckpoint()

        # Save remaining results
        self._saveResultsLocal()
        redoNum = nSimTotal - self._completeCount
        return redoNum

    def _logSetup(self):
        logHeader = 'subthread-'+self._strSimId
        self._logger = childLogSetup(logHeader, path=self._pathLog)
        self._resourceFinder.resourceUtilLogger(logger=self._logger)

    def _logSimStart(self, berDict: dict, printFreq=10):
        # Extract parameters from dictionary
        var1 = berDict['var1']
        var2 = berDict['var2']
        offset = berDict['offset']

        logStr = (f"{datetime.datetime.now()} Simulating "
                  f"var: {var1} var2: {var2} "
                  f"offset: {offset}. " + self._strSimId)
        self._logger.info(logStr)

        if self._consoleReducedPrint:
            if (round(time.time() * 1000) % printFreq) == 0:
                print(logStr)

    def _fullBerList(self):
        # BER List should already be cached, just needs to be retrieved
        parameterBerList = getBerList(self._expDict,
                                      baseSamples=self._baseSamples,
                                      pathCache=self._pathCache)
        self._berList = makeFullDictList(parameterBerList, self._expDict)

    def _loadExistingResults(self):
        # Look for existing values
        parameterBerDf = pd.DataFrame(self._berList)
        existingResultsDf = getResultsUsingDict(self._pathSave,
                                                self._expDict)
        if not existingResultsDf is None:
            self._berList = removeCompletedResults(parameterBerDf,
                                                   existingResultsDf,
                                                   self._logger)
