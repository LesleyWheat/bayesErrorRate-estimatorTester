# Imports
import os

# Packages
import numpy as np
from matplotlib import pyplot as plt

# Project
from distributions import get_distPair, abstractDistribution

# --------------------------------------------------------------------


def _plotTwoD(pathSave: str, name: str, distA: abstractDistribution,
              distB: abstractDistribution, num: int, seed: int | None = 5):
    rng = np.random.default_rng(seed=seed)
    fig, axs = plt.subplots(1, 1, squeeze=False,)
    a = distA.generate(rng, num)
    b = distB.generate(rng, num)
    axs[0, 0].scatter(a[:, 0], a[:, 1], color='b')
    axs[0, 0].scatter(b[:, 0], b[:, 1], color='r', marker='^')

    # Remove tick values
    axs[0, 0].axes.xaxis.set_ticklabels([])
    axs[0, 0].axes.yaxis.set_ticklabels([])

    fig.set_size_inches(4, 4)
    plt.tight_layout()
    plt.savefig(os.path.join(pathSave, name+'.png'))
    plt.close()


def _plotThreeD(pathSave: str, name: str, distA: abstractDistribution,
                distB: abstractDistribution, num: int, seed: int | None = 5):
    rng = np.random.default_rng(seed=seed)
    fig = plt.figure()
    ax = fig.add_subplot(1, 1, 1, projection='3d')
    a = distA.generate(rng, num)
    b = distB.generate(rng, num)
    ax.scatter(a[:, 0], a[:, 1], a[:, 2], color='b')
    ax.scatter(b[:, 0], b[:, 1], b[:, 2], color='r', marker='^')

    # Remove tick values
    ax.xaxis.set_tick_params(color='white')
    ax.yaxis.set_tick_params(color='white')
    ax.zaxis.set_tick_params(color='white')

    fig.set_size_inches(4, 4)
    plt.tight_layout()

    # Keep grid
    ax.xaxis.set_ticklabels([])
    ax.yaxis.set_ticklabels([])
    ax.zaxis.set_ticklabels([])

    plt.savefig(os.path.join(pathSave, name+'.png'))
    plt.close()


def displayTwoD(pathSave: str):
    """
    Creates examples of distributions in 2D and saves as images.

    Parameters
    ----------
    pathSave: path
        Location to save plots.
    """

    # Use Backend
    import matplotlib as mpl
    mpl.use('agg')

    nSamples = 100
    os.makedirs(pathSave, exist_ok=True)

    # 2D not noisy
    distList = get_distPair(simName='gvg', nFeatures=2,
                            var1=0.1, var2=0.15, offset=1)
    _plotTwoD(pathSave, '2d_GvG', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvt', nFeatures=2,
                            var1=0.1, var2=0.15, offset=4)
    _plotTwoD(pathSave, '2d_TvT', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvs', nFeatures=2,
                            var1=0.1, var2=0.15, offset=4)
    _plotTwoD(pathSave, '2d_TvS', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='svs', nFeatures=2,
                            var1=0.1, var2=0.15, offset=4)
    _plotTwoD(pathSave, '2d_SvS', distList[0], distList[1], nSamples)

    # 2D Noisy
    distList = get_distPair(simName='gvg', nFeatures=2,
                            var1=0.1, var2=0.15, offset=0.6)
    _plotTwoD(pathSave, '2d_GvG_noise', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvt', nFeatures=2,
                            var1=0.1, var2=0.15, offset=1.7)
    _plotTwoD(pathSave, '2d_TvT_noise', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvs', nFeatures=2,
                            var1=0.1, var2=0.15, offset=2)
    _plotTwoD(pathSave, '2d_TvS_noise', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='svs', nFeatures=2,
                            var1=0.1, var2=0.15, offset=2.5)
    _plotTwoD(pathSave, '2d_SvS_noise', distList[0], distList[1], nSamples)


def displayThreeD(pathSave: str):
    """
    Creates examples of distributions in 3D and saves as images.

    Parameters
    ----------
    pathSave: path
        Location to save plots.
    """

    # Use backend
    import matplotlib as mpl
    mpl.use('agg')

    nSamples = 100
    os.makedirs(pathSave, exist_ok=True)

    # 3D not noisy
    distList = get_distPair(simName='gvg', nFeatures=3,
                            var1=0.1, var2=0.15, offset=1)
    _plotThreeD(pathSave, '3d_GvG', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvt', nFeatures=3,
                            var1=0.1, var2=0.15, offset=4)
    _plotThreeD(pathSave, '3d_TvT', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='tvs', nFeatures=3,
                            var1=0.1, var2=0.15, offset=4)
    _plotThreeD(pathSave, '3d_TvS', distList[0], distList[1], nSamples)

    distList = get_distPair(simName='svs', nFeatures=3,
                            var1=0.1, var2=0.15, offset=4)
    _plotThreeD(pathSave, '3d_SvS', distList[0], distList[1], nSamples)
