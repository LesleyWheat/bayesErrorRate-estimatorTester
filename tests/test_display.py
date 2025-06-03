import pytest
import os

# ------------------------------------------------------------------------
# global variables

pathCache = os.path.join(os.getcwd(), 'local', 'testCache')
pathLogs = os.path.join(os.getcwd(), 'local', 'testLogs')
pathSave = os.path.join(os.getcwd(), 'local', 'testOutput')
pathSaveData = os.path.join(pathSave, 'estimatorData')


# ------------------------------------------------------------------------

class Test_plots():
    nFeatures = 2
    codename = 'gvg'
    nClassSamples = 50
    testSamples = 20

    def _start(self):
        import os
        from app import getResultsOneSim
        self._runSim()
        filename = f'{self.codename}_{self.nFeatures}d_{self.nClassSamples}n'
        self.pathSaveSim = os.path.join(pathSave, 'singleSimGraphs', filename)
        self.resultsDf = getResultsOneSim(pathSaveData, self.codename,
                                          self.nFeatures, self.nClassSamples)

    def _runSim(self):
        from app._dispatcher import _simWorker

        expDict = dict()
        expDict.update({'codename': self.codename})
        expDict.update({'nFeatures': self.nFeatures})
        expDict.update({'idx': 0})
        expDict.update({'seed': 10})
        expDict.update({'nClassSamples': self.nClassSamples})

        worker = _simWorker(expDict,
                            pathSave=pathSaveData,
                            pathLog=pathLogs,
                            baseSamples=self.testSamples,
                            pathCache=pathCache)

        worker.run()

    @pytest.mark.plots
    def test_gt(self):
        from display._plots._simGraphs import _berGroundTruthPlot
        self._start()
        _berGroundTruthPlot(self.resultsDf, self.pathSaveSim).make()

    @pytest.mark.plots
    def test_time(self):
        from display._plots._simGraphs import _performaceTimePlot
        self._start()
        _performaceTimePlot(self.resultsDf, self.pathSaveSim).make()

    @pytest.mark.plots
    def test_singles(self):
        from display._plots._simGraphs import _plotEstimators
        self._start()
        _plotEstimators(self.resultsDf, self.pathSaveSim, singlePlots=True)

    @pytest.mark.plots
    def test_plotCombo(self):
        from display._plots._simGraphs import _plotEstimators
        self._start()
        _plotEstimators(self.resultsDf, self.pathSaveSim, singlePlots=False)
