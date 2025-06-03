# These tests should always work

import pytest

# ------------------------------------------------------------------------
# Estimators


@pytest.mark.knn
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_knn():
    from ..utils import dist_one
    from app._estimators import runKNN

    X, Y = dist_one(seperate=False)
    results = dict()
    runKNN(results, X, Y, kmin=1, kmax=10)
    print(results)

    results.pop('time:KNN')
    expected = {'KNN(squared_l2)_N': pytest.approx(5),
                'KNN(squared_l2)_H': pytest.approx(0.355),
                'KNN(squared_l2)_L': pytest.approx(0.24529896699689366),
                'KNN(squared_l2)_M': pytest.approx(0.30014948349844683)}

    assert results == expected


@pytest.mark.gkde
def test_GKDE():
    from ..utils import dist_one
    from app._estimators import runGKDE
    X, Y = dist_one(seperate=False)
    results = dict()
    runGKDE(results, X, Y)
    print(results)

    test = results['GKDE_silverman']
    expected = pytest.approx(0.3654406068948828)

    assert test == expected


@pytest.mark.ghp
def test_GHP():
    from ..utils import dist_one
    from app._estimators import runGHP

    X, Y = dist_one(seperate=False)
    results = dict()

    runGHP(results, X, Y)
    print(results)

    results.pop('time:GHP')
    expected = {'GHP_M': pytest.approx(0.32773960600441426),
                'GHP_H': pytest.approx(0.39),
                'GHP_L': pytest.approx(0.2654792120088285)}

    assert results == expected


@pytest.mark.clakde
@pytest.mark.gpu
@pytest.mark.parametrize("expectedVal", [0.2397688289951661])
def test_CKDE_gpu(expectedVal):
    from ..utils import dist_one
    from app._estimators import runCLAKDE
    from ..app import checkGpu

    checkGpu()

    X, Y = dist_one(seperate=False)
    results = dict()
    runCLAKDE(results, X, Y)
    print(results)

    test = results['CKDE_LAKDE-LOO_ES']
    expected = pytest.approx(expectedVal)

    assert test == expected


@pytest.mark.clakde
@pytest.mark.cpu
@pytest.mark.parametrize("expectedVal, seed, nClassSamples",
                         [(0.2094586193561554, 2, 1000),
                          (0.20080555975437164, 3, 1000)])
def test_CKDE_cpu(expectedVal, seed, nClassSamples):
    import time
    from ..utils import dist_one
    from bayesErrorRate.CKDE import CLAKDE

    X, Y = dist_one(seperate=False,
                    nClassSamples=nClassSamples, seed=seed)

    start_time = time.time()
    results = CLAKDE(X, Y, useGpuId=None)
    print(round(time.time() - start_time, 2))
    print(results)

    expected = pytest.approx(expectedVal)
    test = results['CKDE_LAKDE-LOO_ES']

    assert test == expected


