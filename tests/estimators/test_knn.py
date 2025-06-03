import pytest

@pytest.mark.knn
def test_knn1():
    from app._estimators import runKNN
    import numpy as np
    from distributions import oneBlob

    d = 2
    seed = 1
    n = 500

    rng = np.random.default_rng(seed=seed)

    distA = oneBlob(d, 2, 1)
    distB = oneBlob(d, 0, 1)

    setA = distA.generate(rng, n)
    setB = distB.generate(rng, n)

    X = np.concatenate((setA, setB), axis=0)
    Y = np.repeat([0, 1], [setA.shape[0], setB.shape[0]])

    results = dict()

    with pytest.warns(UserWarning):
        runKNN(results, X, Y, kmin=1, kmax=7)
        print(results)
