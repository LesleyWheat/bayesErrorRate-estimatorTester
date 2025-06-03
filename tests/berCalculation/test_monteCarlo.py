

import pytest

# ------------------------------------------------------------------------
# Pytest


@pytest.fixture
def groupParam1():
    pass


def pytest_generate_tests(metafunc):
    if 'groupParam1' in metafunc.fixturenames:
        metafunc.parametrize("nFeatures", [(2), (10), (20)])
        metafunc.parametrize("typeC", [('mc'), ('mc1')])
        metafunc.parametrize("seed", [(1), (2)])

# ------------------------------------------------------------------------
# Functions


def runBERC(typeC, distA, distB, seed):
    import numpy as np
    from bayesErrorRate import mcBerBatch
    rng = np.random.default_rng(seed=seed)
    distList = [distA, distB]

    if typeC == 'mc':
        ber, v = mcBerBatch(rng, distList, brute=False)
    elif typeC == 'mc1':
        ber, v = mcBerBatch(rng, distList, brute=False,
                            nLoops=10, nBatch=2048)
    elif typeC == 'mc2':
        ber, v = mcBerBatch(rng, distList, brute=False,
                            nLoops=50, nBatch=1024)
    elif typeC == 'mc3':
        ber, v = mcBerBatch(rng, distList, brute=False,
                            nLoops=100, nBatch=1024)
    elif typeC == 'mc4':
        ber1, v = mcBerBatch(rng, distList, brute=False,
                             nLoops=20, nBatch=1024)
        ber2, v = mcBerBatch(rng, distList, brute=False,
                             nLoops=20, nBatch=1024)
        ber = ber1/2 + ber2/2
    elif typeC == 'mcBrute':
        ber, v = mcBerBatch(rng, distList, brute=True,
                            nLoops=50, nBatch=1024)
    else:
        raise Exception('Unsupported test')

    return ber, v


def compareAccuracy(pathSave, typeList, distA, distB, target):
    import numpy as np
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    mpl.use('agg')  # Use backend
    fig, axs = plt.subplots(1, len(typeList), squeeze=False)
    stdList = []
    seedList = range(100)

    for idx in range(len(typeList)):
        typeC = typeList[idx]
        berList = []

        for seed in seedList:
            ber, v = runBERC(typeC, distA, distB, seed)
            e = 100*ber - target
            berList.append(e)

        X = np.array(berList)
        std = np.std(X)
        stdList.append(std)
        axs[0, idx].hist(X)
        axs[0, idx].set_title(typeC)
        axs[0, idx].annotate("std = {:.3f}".format(std),  xy=(0, 0))

    fig.set_size_inches(32, 12)
    plt.tight_layout()
    plt.savefig(pathSave)
    plt.close()

    assert max(stdList)*2 < 0.1
# ------------------------------------------------------------------------
# Classes


class Test_bayesErrorGroundTruth():
    @pytest.mark.same
    @pytest.mark.ber
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_sameDistOne(self, groupParam1, nFeatures, typeC, seed, tol=0.1):
        from distributions import oneBlob
        distA = oneBlob(nFeatures, 0, 1)
        distB = oneBlob(nFeatures, 0, 1)

        ber, v = runBERC(typeC, distA, distB, seed)

        assert 100*ber == pytest.approx(50, abs=tol)

    @pytest.mark.same
    @pytest.mark.ber
    @pytest.mark.filterwarnings("ignore::UserWarning")
    def test_sameDistTwo(self, groupParam1, nFeatures, typeC, seed, tol=0.1):
        from distributions import twoBlob
        distA = twoBlob(nFeatures, 1, 0.04)
        distB = twoBlob(nFeatures, 1, 0.04)

        ber, v = runBERC(typeC, distA, distB, seed)

        assert 100*ber == pytest.approx(50, abs=tol)

    @pytest.mark.same
    @pytest.mark.ber
    @pytest.mark.xfail
    @pytest.mark.parametrize("seed", [(1), (2)])
    @pytest.mark.parametrize("nFeatures", [(20), (30)])
    @pytest.mark.parametrize("typeC", [('mc3'), ('mcBrute')])
    def test_sameDistTwo_xfail(self, nFeatures, typeC, seed, tol=0.1):
        # May fail at high dimensionalities with mc
        # Due to values being too small
        from distributions import twoBlob
        distA = twoBlob(nFeatures, 1, 0.01)
        distB = twoBlob(nFeatures, 1, 0.01)

        ber, v = runBERC(typeC, distA, distB, seed)

        assert 100*ber == pytest.approx(50, abs=tol)

    def compareToFormula(self, nFeatures, seed, typeC, m1, m2, v2, tol):
        import numpy as np
        from distributions import oneBlob
        from tests.berCalculation.test_literature import complexBayesErrorCalc

        # First distribution to have a variance of 1 so no exceptions
        v1 = 1

        cv1 = v1*np.identity(nFeatures)
        cv2 = v2*np.identity(nFeatures)

        distA = oneBlob(nFeatures, m1, v1)
        distB = oneBlob(nFeatures, m2, v2)

        ber, v = runBERC(typeC, distA, distB, seed)

        estimate = 100*ber

        c = 100*complexBayesErrorCalc(m1*np.ones(nFeatures), m2*np.ones(nFeatures), cv1, cv2)

        assert c == pytest.approx(estimate, abs=tol)

    @pytest.mark.slow
    @pytest.mark.ber
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
    @pytest.mark.parametrize("nFeatures", [(2), (5)])
    @pytest.mark.parametrize("seed", [(1), (2)])
    @pytest.mark.parametrize("typeC", [('mc3'), ('mcBrute')])
    @pytest.mark.parametrize("v2", [(1), (3)])
    @pytest.mark.parametrize("m1", [(0), (5)])
    @pytest.mark.parametrize("m2", [(0), (1), (5)])
    @pytest.mark.parametrize("tol", [(0.5)])
    def test_vsFormulas(self, nFeatures, seed, typeC, m1, m2, v2, tol):
        # Expect less than 5% failure (+/- 0.05% bounds is less than 95%)
        self.compareToFormula(nFeatures, seed, typeC, m1, m2, v2, tol)

    @pytest.mark.slow
    @pytest.mark.ber
    @pytest.mark.xfail
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
    @pytest.mark.parametrize("nFeatures", [(10), (20)])
    @pytest.mark.parametrize("seed", [(1), (2)])
    @pytest.mark.parametrize("typeC", [('mc3'), ('mcBrute')])
    @pytest.mark.parametrize("v2", [(1), (3)])
    @pytest.mark.parametrize("m1", [(0), (5)])
    @pytest.mark.parametrize("m2", [(0), (1), (5)])
    @pytest.mark.parametrize("tol", [(0.01)])
    def test_vsFormulas_xfail(self, nFeatures, seed, typeC, m1, m2, v2, tol):
        # Tends to fail at higher dimensions
        self.compareToFormula(nFeatures, seed, typeC, m1, m2, v2, tol)

    @pytest.mark.slow
    @pytest.mark.ber
    @pytest.mark.filterwarnings("ignore::UserWarning")
    @pytest.mark.filterwarnings("ignore:divide by zero encountered in double_scalars")
    @pytest.mark.parametrize("nFeatures", [(2), (5), (10)])
    @pytest.mark.parametrize("seed", [(10), (20)])
    @pytest.mark.parametrize("v2", [(1), (2), (3)])
    @pytest.mark.parametrize("m1", [(0), (5)])
    @pytest.mark.parametrize("m2", [(0), (5)])
    @pytest.mark.parametrize("tol", [(0.1)])
    def test_vsFormulasActualMethod(self, nFeatures, seed, m1, m2, v2, tol):
        # Expect less than 5% failure (+/- 0.05% bounds is less than 95%)
        import numpy as np
        from distributions import oneBlob
        from tests.berCalculation.test_literature import complexBayesErrorCalc
        from bayesErrorRate import mcBerBatch

        # First distribution to have a variance of 1 so no exceptions
        v1 = 1
        rng = np.random.default_rng(seed=seed)

        cv1 = v1*np.identity(nFeatures)
        cv2 = v2*np.identity(nFeatures)

        distA = oneBlob(nFeatures, m1, v1)
        distB = oneBlob(nFeatures, m2, v2)

        c = 100*complexBayesErrorCalc(m1*np.ones(nFeatures), m2*np.ones(nFeatures), cv1, cv2)

        ber, v = mcBerBatch(rng, [distA, distB])
        estimate = ber*100

        assert c == pytest.approx(estimate, abs=tol)

# ------------------------------------------------------------------------
# Multiclass
@pytest.mark.ber
@pytest.mark.parametrize("nClasses", [3, 4, 5, 6, 7, 8, 9, 10])
def test_berMultiClass(nClasses):
    import numpy as np
    from distributions._blobs import oneBlob
    from bayesErrorRate import mcBerBatch

    d = 8
    seed = 1
    nint = 100
    nBatch = 1000

    rng = np.random.default_rng(seed=seed)
    distList = []

    for idx in range(nClasses):
        distA = oneBlob(d, 0, 1)
        distList.append(distA)

    e, v = mcBerBatch(rng, distList, brute=False, nLoops=nint, nBatch=nBatch)

    expected = (nClasses-1)/nClasses

    print(f'{e} {v}')
    assert expected == pytest.approx(e)

# ------------------------------------------------------------------------
@pytest.mark.ber
@pytest.mark.slow
@pytest.mark.parametrize("codename", ['gvg', 'tvt', 'tvs', 'svs'])
@pytest.mark.parametrize("nFeatures", [2, 10])
def test_sim_berBatchImpact(codename, nFeatures):
    import os
    import time

    import numpy as np
    import matplotlib as mpl
    from matplotlib import pyplot as plt

    from bayesErrorRate import mcBerBatch
    from distributions import get_distPair

    # Use backend
    mpl.use('agg')

    # Save path
    pathSave = os.path.abspath(os.path.join(os.getcwd(), 'local',
                                            'testOutput', 'monteCarloBatchTest'))
    name = f'mcBatchTest_{codename}_{nFeatures}d_'
    os.makedirs(pathSave, exist_ok=True)

    nList = np.repeat(np.arange(10, 101, 50), 3)
    distList = get_distPair(codename, nFeatures=nFeatures)
    seed = 1
    nBatch = 2048
    rng = np.random.default_rng(seed=seed)

    runTimeList = []
    errorList = []
    varList = []
    for idx in range(len(nList)):
        start_time = time.time()
        error, var = mcBerBatch(rng, distList,
                                nLoops=nList[idx],
                                nBatch=nBatch)
        runTime = round(time.time() - start_time, 3)
        errorList.append(error*100)
        varList.append(var*100**2)
        runTimeList.append(runTime)

    mean = np.mean(np.array(errorList))

    fig, axs = plt.subplots(1, 4, squeeze=False)

    axs[0, 0].scatter(nList*nBatch, np.array(errorList))
    axs[0, 0].axhline(y=mean, color='r', linestyle='-')
    axs[0, 0].set_title('Bayses Error Estimate')

    axs[0, 1].scatter(nList*nBatch, mean-np.array(errorList))
    axs[0, 1].set_title('Error in BER Estimate')

    axs[0, 2].scatter(nList*nBatch, np.array(varList))
    axs[0, 2].axhline(y=np.mean(np.array(varList)), color='r', linestyle='-')
    axs[0, 2].set_title('Var')

    axs[0, 3].scatter(nList*nBatch, np.array(runTimeList))
    axs[0, 3].set_title('Time')

    fig.set_size_inches(12, 6)
    plt.tight_layout()
    plt.savefig(os.path.join(pathSave, name+'.png'))
    plt.close()