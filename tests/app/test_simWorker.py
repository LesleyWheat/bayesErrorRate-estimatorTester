import os
import pytest

# ------------------------------------------------------------------------
pathSave = os.path.join('local', 'testOutput')
pathSaveData = os.path.join(pathSave, 'estimatorData')
pathCache = os.path.join('local', 'testCache')
pathLogs = os.path.join('local', 'testLogs')
# ------------------------------------------------------------------------


@pytest.mark.basic
def test_worker():
    from app._dispatcher import _simWorker
    from app._results import clearResultsUsingDict
    from app import runSimPool
    from .utils import basicRunList

    expDictList = basicRunList(nLen=1)
    nMaxProc = 1
    testSamples = 10

    clearResultsUsingDict(os.path.join(os.getcwd(), pathSaveData),
                          expDictList[0])

    worker = _simWorker(expDictList[0],
                        pathSave=pathSaveData,
                        pathLog=pathLogs,
                        baseSamples=testSamples,
                        pathCache=pathCache)
    worker.run()

    # Run - should do nothing since results are finished
    runSimPool(expDictList, nMaxProc,
               {'pathSave': pathSaveData})

    # Clear final
    clearResultsUsingDict(os.path.join(os.getcwd(), pathCache),
                          expDictList[0])


@pytest.mark.basic
def test_twoWorker():
    from app._dispatcher import _runSimWorker
    from app._tools._parallel import runFuncPool
    from app._results import clearResultsUsingDict
    from .utils import basicRunList

    runList = basicRunList(nLen=2)
    timeout = 120
    nMaxProc = 2

    for eDict in runList:
        clearResultsUsingDict(pathSaveData, eDict)

    failedDictList = runFuncPool(_runSimWorker, runList, timeout,
                                 nMaxProc, baseSamples=10,
                                 simTimeLimit=10,
                                 pathCache=pathCache,
                                 pathLog=pathLogs,
                                 pathSave=pathSaveData)
    print(failedDictList)

    assert len(failedDictList) == 0
