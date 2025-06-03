
# Imports
import os
import copy

# Packages
from scipy import stats
import numpy as np
from matplotlib import pyplot as plt
from matplotlib.axes import Axes
import matplotlib as mpl
from mpl_toolkits.axes_grid1.axes_divider import make_axes_locatable
import pandas as pd

# Project
from app import calcGap

# Module
from ._utils import plotWrapper, basicPlot
from ._singles import plotLinearFit, plotCurveFit_annot, plotCurveFit_noAnnot

# --------------------------------------------------------------------
# Classes


class _berGroundTruthPlot(basicPlot):
    def __init__(self, resultsDf: pd.DataFrame, pathSave: str,
                 minBER: float = 0.01, maxBER: float = 0.49):
        self._resultsDf = resultsDf
        self._xSize = 32
        self._ySize = 12
        self._minBER = minBER
        self._maxBER = maxBER
        self._pathImg = os.path.join(pathSave, 'ber_est')
        self._mcName = 'pdfMonteCarlo'
        self._berValues = self._resultsDf[self._mcName].to_numpy()
        self._varArray = self._resultsDf[self._mcName+'_var'].to_numpy()
        self._upperlabel = self._mcName+'_upper'
        self._lowerLabel = self._mcName+'_lower'

    def _middlePlotSplit(self, ax: Axes):
        divider = make_axes_locatable(ax)
        axHistx = divider.append_axes("bottom", size=2, pad=0.1,
                                      sharex=ax)
        kdex = divider.append_axes("bottom", size=1.5, pad=0.1,
                                   sharex=ax)
        axHistx.hist(self._berValues)

        try:
            X_uni = np.linspace(0, 0.5, num=50)
            kernel_X = stats.gaussian_kde(np.transpose(self._berValues))
            pdf_X = kernel_X.pdf(X_uni)
            kdex.plot(X_uni, pdf_X, label='pdf_Y')
        except Exception as e:
            print(f'Problem plotting: {e}')

    def _addLegends(self):
        self._axs[0, 0].legend()
        self._axs[0, 1].legend()
        self._axs[0, 2].legend()

    def _annotateGap(self, ax: Axes, x: float, y: float):
        berGap = calcGap(self._berValues, self._minBER, self._maxBER)
        ax.annotate(f"Max gap: {berGap}",  xy=(x, y))

    def _annotateSamples(self, ax: Axes, x: float, y: float):
        n = len(self._berValues)
        ax.annotate(f"Number of samples: {n}", xy=(x, y))

    def _leftPlot(self, ax: Axes, low_mc: np.ndarray, high_mc: np.ndarray):
        ax.plot(self._berValues, label=self._mcName)
        ax.plot(low_mc, label=self._upperlabel)
        ax.plot(high_mc, label=self._lowerLabel)

    def _midPlot(self, ax: Axes, low_mc: np.ndarray, high_mc: np.ndarray):
        ax.scatter(self._berValues, low_mc, label=self._upperlabel)
        ax.scatter(self._berValues, high_mc, label=self._lowerLabel)

    def make(self):
        self._fig, self._axs = plt.subplots(1, 3, squeeze=False)

        mc_min = np.min(self._berValues)
        mc_max = np.max(self._berValues)
        std = np.sqrt(self._varArray)
        low_mc = self._berValues-std
        high_mc = self._berValues+std

        self._leftPlot(self._axs[0, 0], low_mc, high_mc)
        self._middlePlotSplit(self._axs[0, 1])
        self._midPlot(self._axs[0, 1], low_mc, high_mc)

        self._axs[0, 2].scatter(self._berValues, self._varArray,
                                label=self._mcName+'_var')

        self._addLegends()
        self._annotateGap(self._axs[0, 1], 0, mc_max*0.9)
        self._annotateSamples(self._axs[0, 1], 0, mc_max*0.8)

        self._axs[0, 1].annotate(f"Min: {mc_min}",  xy=(0, mc_max*0.7))
        self._axs[0, 1].annotate(f"Max: {mc_max}",  xy=(0, mc_max*0.6))

        self._savePNG()
        plt.close()


class _performaceTimePlot(basicPlot):
    def __init__(self, resultsDf: pd.DataFrame, pathSave: str):
        self._resultsDf = resultsDf
        self._xSize = 32
        self._ySize = 12
        self._pathImg = os.path.join(pathSave, 'timing')
        self._timeNames = ['time:GKDE', 'time:KNN', 'time:GHP',
                           'time:CKDE_LAKDE-LOO_ES']

    def _plotTime(self, ax: Axes, name: str):
        ax.plot(self._resultsDf[name].to_numpy(), label='time:'+name)

    def make(self):
        self._fig, axs = plt.subplots(1, 2, squeeze=False)

        for name in self._timeNames:
            try:
                self._plotTime(axs[0, 0], name)
            except Exception as e:
                print(e)

        mcTime = self._resultsDf['time:pdfMonteCarlo'].to_numpy()
        axs[0, 1].plot(mcTime, label='time:pdfMonteCarlo')

        axs[0, 0].legend()
        axs[0, 1].legend()

        self._savePNG()
        plt.close()

# --------------------------------------------------------------------
# Functions


def _plotEstimators(resultsDf: pd.DataFrame, pathSave: str, singlePlots=True):
    nameList = ['bayesClassifierError', 'NaiveBayesError', 'GKDE_silverman',
                'KNN(squared_l2)_H', 'KNN(squared_l2)_M', 'KNN(squared_l2)_L',
                'GHP_H', 'GHP_M', 'GHP_L',
                'CKDE_LAKDE-LOO_ES', 'GHP(L)-CKDE(LL)(ES)',
                ]

    pathImg = os.path.join(pathSave, 'group_linFit')
    plotWrapper(resultsDf, pathImg, nameList, plotLinearFit).make()

    pathImg = os.path.join(pathSave, 'group_curveFit')
    plotWrapper(resultsDf, pathImg, nameList, plotCurveFit_annot).make()

    if singlePlots:
        for name in nameList:
            newName = copy.copy(name)
            newName = newName.replace('(', '0')
            newName = newName.replace(')', '0')
            # Line fit
            pathImg = os.path.join(pathSave, f'fit-linear-{name}')
            plotWrapper(resultsDf, pathImg, [name], plotLinearFit).make()

            # Curve fit
            pathImg = os.path.join(pathSave, f'fit-curve-annot-{name}')
            plotWrapper(resultsDf, pathImg, [
                        name], plotCurveFit_annot).make()

            # Curve fit
            pathImg = os.path.join(pathSave, f'fit-curve-{newName}')
            plotWrapper(resultsDf, pathImg, [name],
                        plotCurveFit_noAnnot).make(saveType='pdf')


def makePlots(resultsDf: pd.DataFrame, pathSave: str):
    """
    Runs the CKDE estimator using LAKDE.

    Parameters
    ----------
    resultsDf: pandas dataframe
        Estimator results.
    pathsave: path
        Folder to save files.
    """
    mpl.use('agg')

    # Mote carlo ground truth examine
    _berGroundTruthPlot(resultsDf, pathSave).make()

    # Timing
    _performaceTimePlot(resultsDf, pathSave).make()

    # correlation plots
    _plotEstimators(resultsDf, pathSave)
