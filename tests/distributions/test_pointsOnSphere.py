import os
import pytest

# ------------------------------------------------------------------------
# global variables
GL_targetPoints = {10, 20, 50, 100, 150, 200, 210}
GL_dimensions = {3, 4, 5, 6, 7, 8, 9, 10, 11, 12}
pathCache = os.path.join(os.getcwd(), 'local', 'testCache')

# ------------------------------------------------------------------------


def _buildParam(manager, nFeatures):
    r1 = list(range(3, 26))
    r2 = list(range(45, 55))
    r3 = list(range(90, 110))
    r4 = list(range(145, 155))
    r5 = list(range(190, 210))
    r6 = list(range(245, 255))

    pointsRange = r1+r2+r3+r4+r5+r6

    manager.buildParams(nFeatures, pointsRange, maxLoops=15,
                        intialStep=0.05*(3/4),
                        minGuess=0.1, intialGuess=1.5)


def _checkParamBuilt(nFeatures, nPoints, slack):
    from distributions._radial import _sphereParamManager
    manager = _sphereParamManager(pathCache=pathCache)
    try:
        # Get from cache
        prm = manager.getParam(nFeatures, nPoints, slack=slack)
    except OSError:
        _buildParam(manager, nFeatures)
        prm = manager.getParam(nFeatures, nPoints, slack=slack)


def _closestThreeError(dist, x):
    # For calculating error of points in hypersphere
    # Takes three closest points, distance should match dist
    import numpy as np
    from scipy.spatial import distance_matrix

    distance = distance_matrix(x, x)
    np.fill_diagonal(distance, np.inf)
    minDist = np.min(distance, axis=1)
    distRemoved = distance.copy()

    minSet1 = np.argmin(distRemoved, axis=1)

    for idx in range(minSet1.shape[0]):
        distRemoved[idx, minSet1[idx]] = np.inf

    minDist2 = np.min(distRemoved, axis=1)

    distRemoved2 = distRemoved.copy()
    minSet2 = np.argmin(distRemoved2, axis=1)
    for idx in range(minSet2.shape[0]):
        distRemoved2[idx, minSet2[idx]] = np.inf

    distRemoved2[:, np.argmin(distRemoved2, axis=1)] = np.inf
    minDist3 = np.min(distRemoved2, axis=1)

    minD = np.hstack((minDist, minDist2, minDist3))
    # Remove inf
    minD = minD[minD != np.inf]

    error = np.mean(np.power(dist-minD, 2))
    return error

# ------------------------------------------------------------------------
# Check that parameters can be build and with a specific number of points


@pytest.mark.sphere
def test_nBlobsOverSphere_clearCache():
    from distributions._radial import _sphereParamManager
    man = _sphereParamManager(clear=True, pathCache=pathCache)


@pytest.mark.sphere
def test_nBlobsOverSphere_warnCache():
    from distributions._radial import _sphereParamManager
    with pytest.warns(UserWarning):
        man = _sphereParamManager(clear=True)


@pytest.mark.sphere
@pytest.mark.slow
@pytest.mark.parametrize("nFeatures", GL_dimensions)
def test_nBlobsOverSphere_buildParam(nFeatures):
    # This function rebuilds the parameters
    import pandas as pd
    import warnings
    from distributions._radial import _sphereParamManager

    slack = 2
    missingNumSet = set()

    # ignore pandas warnings
    warnings.simplefilter(action='ignore',
                          category=pd.errors.PerformanceWarning)

    man = _sphereParamManager(pathCache=pathCache)
    _buildParam(man, nFeatures)

    for nPoints in list(GL_targetPoints):
        try:
            # Get from cache
            man.getParam(nFeatures, nPoints, slack=slack)
        except Exception:
            missingNumSet.add(nPoints)

    print(missingNumSet)

    assert not missingNumSet

    goodPointsSet = man.checkValidParamInCache([nFeatures], slack=slack)
    missingFromCheck = (GL_targetPoints - goodPointsSet)

    assert not missingFromCheck


@pytest.mark.sphere
@pytest.mark.slow
@pytest.mark.parametrize("nFeatures", GL_dimensions)
def test_nBlobsOverSphere_findPoints(nFeatures):
    # This function rebuilds the parameters
    import pandas as pd
    import warnings
    from distributions._radial import _spherePointFinder

    # ignore pandas warnings
    warnings.simplefilter(action='ignore',
                          category=pd.errors.PerformanceWarning)

    basicPoints = {5, 10, 20, 50, 100, 200}
    extraPoints = [150]

    finder = _spherePointFinder(nFeatures, basicPoints,
                                maxLoops=100, intialStep=0.05*(3/4),
                                minGuess=0.1, intialGuess=1.5,
                                nSlack=1)
    finder.findPoints()

    for nPoints in extraPoints:
        finder.addPointToSearch(nPoints, newSlack=0)
        valid = finder.getValidN()
        print(valid)
        print(finder._pointsRange)
        assert nPoints in valid


@pytest.mark.sphere
def test_nBlobsOverSphere_checkParam():
    # function check existing cached parameters
    from distributions._radial import _sphereParamManager

    maxAllowedMissing = 1
    s = 4

    man = _sphereParamManager(pathCache=pathCache)
    goodPointsSet = man.checkValidParamInCache(list(GL_dimensions), slack=s)
    missingNumSet = GL_targetPoints - goodPointsSet
    goodPointsList = list(goodPointsSet)
    goodPointsList.sort()
    print(goodPointsList)

    for nPoints in missingNumSet:
        for nFeatures in GL_dimensions:
            p, t = man.getParam(nFeatures, nPoints, slack=s)

    goodPointsSet = man.checkValidParamInCache(list(GL_dimensions), slack=s)
    missingNumSet = GL_targetPoints - goodPointsSet
    goodPointsList = list(goodPointsSet)
    goodPointsList.sort()
    print(goodPointsList)
    print(missingNumSet)

    # Basic check not too many numbers are missing
    assert len(missingNumSet) <= maxAllowedMissing

# ------------------------------------------------------------------------


@pytest.mark.sphere
@pytest.mark.parametrize("d", GL_dimensions)
@pytest.mark.parametrize("dist", [1.1])
def test_nPointsOnSphere_error(d, dist, mseLimit=1e-1):
    from distributions._radial import _evenlySpacedPointsOnSphere

    gen = _evenlySpacedPointsOnSphere()
    x = gen.generate(d, dist)
    mse = _closestThreeError(dist, x)

    assert mse < mseLimit


@pytest.mark.sphere
@pytest.mark.parametrize("d, rnge", [(3, [0.25, 0.7]),
                                     (4, [0.3, 0.7]),
                                     (5, [0.5, 0.8]),
                                     (6, [0.7, 1.1]),
                                     (7, [0.7, 1.1]),
                                     (8, [0.8, 1.0])])
@pytest.mark.parametrize("factor", [None])
def test_nPointsOnSphere_close(d, rnge, factor):
    import scipy
    import numpy as np
    from distributions._radial import _evenlySpacedPointsOnSphere

    distList = (np.linspace(rnge[0], rnge[1], 20)).tolist()
    MSE_per_list = []
    varList = []
    nList = []
    MSE_limit = 5
    gen = _evenlySpacedPointsOnSphere()

    for dist in distList:
        x = gen.generate(d, dist, factor=factor)
        nList.append(x.shape[0])
        distance = scipy.spatial.distance_matrix(x, x)
        np.fill_diagonal(distance, np.inf)
        minDist = np.min(distance, axis=1)
        MSE = np.average(np.power(dist - minDist, 2))
        MSE_per_list.append(MSE/dist*100)
        varList.append(np.var(minDist))

    MMSE = sum(MSE_per_list)/len(MSE_per_list)

    print(f'MMSE: {MMSE} VAR: {sum(varList)/len(varList)} n: {nList}')
    print(f'n: {nList} MSE: {MSE_per_list}')

    assert MMSE < MSE_limit


@pytest.mark.sphere
@pytest.mark.parametrize("nFeatures", GL_dimensions)
@pytest.mark.parametrize("nPoints", GL_targetPoints)
def test_nBlobsOverSphere_paramError_withSlack(nFeatures, nPoints,
                                               mseLimitRel=0.2):
    # Check retrieval from cache
    import numpy as np
    from distributions._radial import nBlobOverSphere
    from distributions._radial import cartesian_coordinates

    # Check retrieval from cache
    slack = 2
    offset = 1
    var = 0.1
    _checkParamBuilt(nFeatures, nPoints, slack)
    dist = nBlobOverSphere(nFeatures, offset, var, nBlobs=nPoints,
                           slackInN=slack, pathCache=pathCache)
    phi = dist._phi
    distance = dist._distance
    x = cartesian_coordinates(np.ones(phi.shape[0]), phi)
    mse = _closestThreeError(distance, x)
    mseLimit = mseLimitRel*distance

    assert mse < mseLimit
