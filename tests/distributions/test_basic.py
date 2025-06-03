import pytest
# ------------------------------------------------------------------------

DIST_NAME_LIST = ['gvg', 'tvt', 'tvs', 'svs']

# ------------------------------------------------------------------------


@pytest.mark.basic
@pytest.mark.dist
@pytest.mark.parametrize("codename, expectedAvgScore",
                         [('gvg', -2.938516705870735),
                          ('tvt', -3.1087778985290564),
                          ('tvs', -3.1391450040413194),
                          ('svs', -3.0567782760302054)])
def test_distBasic(codename, expectedAvgScore):
    import numpy as np
    from distributions import get_distPair

    distList = get_distPair(codename)
    seed = 2
    n = 100
    scoreArr = np.zeros(len(distList))

    for idx in range(len(distList)):
        dist = distList[idx]
        rng = np.random.default_rng(seed=seed)
        x = dist.generate(rng, n)
        score = dist.logPdf(x)
        scoreArr[idx] = np.average(score)

    assert np.average(scoreArr) == pytest.approx(expectedAvgScore)

# ------------------------------------------------------------------------


@pytest.mark.parametrize("codename", ['tvt', 'tvs', 'svs'])
@pytest.mark.basic
@pytest.mark.dist
def test_invalidOffset(codename):
    from distributions import get_distPair
    with pytest.raises(ValueError):
        distList = get_distPair(codename, offset=-1)


@pytest.mark.parametrize("codename", DIST_NAME_LIST)
@pytest.mark.basic
@pytest.mark.dist
def test_invalidVar1(codename):
    from distributions import get_distPair
    with pytest.raises(ValueError):
        distList = get_distPair(codename, var1=-1)


@pytest.mark.parametrize("codename", DIST_NAME_LIST)
@pytest.mark.basic
@pytest.mark.dist
def test_invalidVar2(codename):
    from distributions import get_distPair
    with pytest.raises(ValueError):
        distList = get_distPair(codename, var2=-1)


@pytest.mark.parametrize("codename", '-')
@pytest.mark.basic
@pytest.mark.dist
def test_invalidType(codename):
    from distributions import get_distPair
    with pytest.raises(ValueError):
        distList = get_distPair(codename)
