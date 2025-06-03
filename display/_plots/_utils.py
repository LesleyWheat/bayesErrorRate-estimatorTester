# Imports
import math
import os
from typing import List, Literal
from collections.abc import Callable

# Packages
from matplotlib import pyplot as plt
import matplotlib as mpl
from matplotlib.axes import Axes
import pandas as pd

# Project
from .._formatting import formatEstimatorName

# --------------------------------------------------------------------
# Functions


def _deleteAnnotionsFromAxis(ax: Axes):
    for child in ax.get_children():
        if isinstance(child, mpl.text.Annotation):
            child.remove()


def _loopFunc(applyFunc: Callable, firstIdxLen: int, secondIdxLen: int,
              maxIdx: int):
    idx = 0
    for xIdx in range(firstIdxLen):
        for yIdx in range(secondIdxLen):
            if idx >= maxIdx:
                break
            else:
                applyFunc(xIdx, yIdx, idx)
                idx = idx+1

# --------------------------------------------------------------------
# Classes


class basicPlot():
    def __init__(self):
        self._fig = plt.figure
        self._xSize = 1
        self._ySize = 1

    def _savePNG(self):
        os.makedirs(os.path.dirname(self._pathImg), exist_ok=True)
        self._fig.set_size_inches(self._xSize, self._ySize)
        plt.tight_layout()
        plt.savefig(self._pathImg+'.png')

    def _savePDF(self):
        os.makedirs(os.path.dirname(self._pathImg), exist_ok=True)
        self._fig.set_size_inches(self._xSize, self._ySize)
        plt.tight_layout()
        plt.savefig(self._pathImg+'.pdf')


class plotWrapper(basicPlot):
    def __init__(self, resultsDf: pd.DataFrame, pathImg: str,
                 nameList: List[str], plotfunc: Callable):
        self._resultsDf = resultsDf
        self._pathImg = pathImg
        self._nameList = nameList
        self._plotfunc = plotfunc

        self._xName = 'pdfMonteCarlo'
        self._xLabel = r'BER (\%)'
        self._yLabel = r'Estimated BER (\%)'

        self._initSize()
        allX = self._resultsDf[self._xName].to_numpy()
        self._X = 100*allX.reshape(-1)

    def _initSize(self):
        if len(self._nameList) == 1:
            self._xPlots = 1
            self._yPlots = 1
            self._xSize = 5
            self._ySize = 3
        else:
            self._xPlots = 3
            self._yPlots = math.ceil(len(self._nameList)/3)
            self._xSize = self._xPlots*8
            self._ySize = self._yPlots*4

    def _applyAxes(self, xIdx: int, yIdx: int, idx: int):
        name = self._nameList[idx]
        ax = self._axs[xIdx, yIdx]
        try:
            ax.plot(self._X, self._X, c='k', label='Target')
            Y = 100*self._resultsDf[name].to_numpy()
            self._plotfunc(ax, self._X, Y)
            ax.set_xlabel(self._xLabel)
            ax.set_ylabel(self._yLabel)
            ax.set_title(formatEstimatorName(name))
            ax.legend()
        except Exception as e:
            print((f'Problem with: {name}. '
                  f'Valid names: {self._resultsDf.columns}'))

    def _removeAxesInfo(self, xIdx: int, yIdx: int, idx):
        ax = self._axs[xIdx, yIdx]
        # Delete annotations and title
        _deleteAnnotionsFromAxis(ax)
        ax.set_title("")

    def make(self, saveType: Literal['pdf', 'png'] = 'png'):
        maxIdx = len(self._nameList)
        self._fig, self._axs = plt.subplots(self._yPlots, self._xPlots,
                                            squeeze=False)

        _loopFunc(self._applyAxes, self._yPlots, self._xPlots, maxIdx)

        if saveType == 'pdf':
            self._savePDF()
        elif saveType == 'png':
            self._savePNG()
        else:
            raise Exception(f'Unsupported plot type: {saveType}')

        plt.close()
