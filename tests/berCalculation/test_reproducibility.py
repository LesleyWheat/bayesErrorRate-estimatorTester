# These tests should always work

import pytest

# ------------------------------------------------------------------------
# Distribution Generators


@pytest.mark.parametrize("typeC", [('mc'), ('mcBrute')])
@pytest.mark.parametrize("d", [(2), (10)])
def test_berc1(typeC, d):
    # Test bayes error rate calculators
    from distributions._blobs import oneBlob
    from .test_monteCarlo import runBERC

    seed = 2

    distA = oneBlob(d, 0, 1)
    distB = oneBlob(d, 0, 1)

    ber1, v1 = runBERC(typeC, distA, distB, seed)
    ber2, v2 = runBERC(typeC, distA, distB, seed)

    assert ber1 == ber2


def _runBercSame(typeC, expected, d):
    # Test bayes error rate calculators
    from distributions._blobs import oneBlob
    from .test_monteCarlo import runBERC

    seed = 2

    distA = oneBlob(d, 0, 1)
    distB = oneBlob(d, 0, 1)

    ber1, v1 = runBERC(typeC, distA, distB, seed)

    assert ber1 == expected


@pytest.mark.ber
@pytest.mark.parametrize("typeC,expected", [('mc', 0.5),
                                            ('mcBrute', 0.5)])
def test_berc1(typeC, expected):
    d = 2
    _runBercSame(typeC, expected, d)


@pytest.mark.parametrize("typeD", ['oneBlob', 'twoBlob', 'sphere'])
@pytest.mark.parametrize("d", [2, 10])
def test_distGen(typeD, d):
    from distributions._radial import nBlobOverSphere
    from distributions._blobs import oneBlob, twoBlob
    from ..utils import compareSamples
    # Test distribution samplers
    seed = 2

    if typeD == 'oneBlob':
        distA = oneBlob(d, 0.1, 1)
    elif typeD == 'twoBlob':
        distA = twoBlob(d, 0.1, 1)
    elif typeD == 'sphere':
        distA = nBlobOverSphere(d, 1, 1)

    compareSamples(distA, seed)

# ------------------------------------------------------------------------


@pytest.mark.basic
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("nFeatures", [(4), (10)])
@pytest.mark.parametrize("maxLoops", [(1), (10), (100)])
def test_sim_ber_repro(nFeatures, maxLoops):
    import numpy as np
    from .test_generator import runBerList
    from app._groundTruth._build import calcGap

    codename = 'gvg'
    seed = 1
    breakGap = 0.01
    nNew = 5
    minBER = 0.01
    maxBER = 0.49
    minSamples = 1000
    offsetList = np.round(np.arange(0.01, 4, 0.5), 3)
    varianceList = np.round(np.arange(1.0, 2.1, 0.4), 3)

    YA = runBerList(codename, nFeatures, offsetList, varianceList, seed,
                    nNew, minBER, maxBER, maxLoops, minSamples, breakGap)

    YB = runBerList(codename, nFeatures, offsetList, varianceList, seed,
                    nNew, minBER, maxBER, maxLoops, minSamples, breakGap)

    maxGapA = calcGap(YA, minBER, maxBER)
    maxGapB = calcGap(YB, minBER, maxBER)

    assert maxGapA == maxGapB

    assert (YA == YB).all()
