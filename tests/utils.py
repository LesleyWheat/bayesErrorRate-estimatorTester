
# ------------------------------------------------------------------------
def compareSamples(distA, seed, n=10):
    import numpy as np

    rng = np.random.default_rng(seed=seed)
    setA = distA.generate(rng, samples=n)

    rng = np.random.default_rng(seed=seed)
    setB = distA.generate(rng, samples=n)

    assert (setA == setB).all()
# ------------------------------------------------------------------------


def dist_one(seperate=True, nClassSamples=100, nFeatures=2, seed=1, offset=1):
    import numpy as np
    from distributions._blobs import oneBlob
    from app._groundTruth._calc import _berEval

    rng = np.random.default_rng(seed=seed)

    distA = oneBlob(nFeatures, offset, 1)
    distB = oneBlob(nFeatures, 0, 1)

    setA = distA.generate(rng, nClassSamples)
    setB = distB.generate(rng, nClassSamples)

    resultsPDF = dict()
    _berEval(resultsPDF, rng, [distA, distB])
    print(f'Ground truth: {resultsPDF}')

    if seperate:
        return setA, setB
    else:
        X = np.concatenate((setA, setB), axis=0)
        Y = np.repeat([0, 1], [setA.shape[0], setB.shape[0]])
        return X, Y

