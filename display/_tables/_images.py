# Imports
from abc import ABC, abstractmethod
from typing import List

# Packages
import numpy as np
from matplotlib import pyplot as plt
import matplotlib as mpl
import pandas as pd

# Project
from .._formatting import roundSigDig, formatEstimatorName

# Module
from ._laTex import singleRow, singleRowBrackets
from ._laTex import doubleRow, doubleRowBrackets

# --------------------------------------------------------------------
# Image Functions


class _generateTableImg(ABC):
    def __init__(self, mainDf: pd.DataFrame, nCellRows: int):
        self._nRows = len(mainDf)
        self._nCols = len(mainDf.columns)
        self._nColLabels = mainDf.columns
        self._nRowLabels = mainDf.index
        self._nCellRows = nCellRows

        # Backend
        mpl.use('agg')

    def _formatVal(self, val: str | float, sigDig: int = 3):
        if isinstance(val, str):
            return formatEstimatorName(val)
        else:
            return roundSigDig(val, sigDig)

    @abstractmethod
    def _getValsFromDf(self, idx: int, jdx: int):
        pass

    @abstractmethod
    def asLaTex(self, filepath: str, highlightMax: float = 5):
        """
        Makes .tex file for LaTex.

        Parameters
        ----------
        filepath: path
            Location to save file, with filename, not extension.
        highlightMax: float
            Bold values or ranges under this number.
        """
        pass

    def _insertVal(self, cellValues: np.ndarray, rowIdx: int, jdx: int,
                   valList: List):
        for idx in range(self._nCellRows):
            cellValues[rowIdx+idx, jdx] = valList[idx]

    def _tableToImage(self, cellValues: np.ndarray,
                      rowlabels: List[str], ncolLabels: List[str],
                      filepath: str):
        fig, axs = plt.subplots(1, 1, squeeze=False)
        ax = axs[0, 0]
        ax.table(cellText=cellValues, rowLabels=rowlabels,
                 colLabels=ncolLabels, loc='center')
        ax.axis('off')
        ax.axis('tight')
        plt.tight_layout()
        plt.savefig(filepath+'.png')
        plt.close()

    def asImage(self, filepath: str):
        """
        Makes .png file to view table.

        Parameters
        ----------
        filepath: path
            Location to save file, with filename, not extension.
        """
        rowlabels = []
        cellValues = np.empty((self._nRows*self._nCellRows, self._nCols),
                              dtype=object)

        rowIdx = 0
        for idx in range(self._nRows):
            nFeatures = self._nRowLabels[idx]
            rowlabels.append(nFeatures)

            for jdx in range(self._nCellRows-1):
                rowlabels.append("-")

            for jdx in range(self._nCols):
                valList = self._getValsFromDf(idx, jdx)
                self._insertVal(cellValues, rowIdx, jdx, valList)

            rowIdx = rowIdx+self._nCellRows

        self._tableToImage(cellValues, rowlabels, self._nColLabels, filepath)


class singleLineTable(_generateTableImg):
    def __init__(self, df: pd.DataFrame):
        nCellRows = 1
        self._df = df
        super().__init__(df, nCellRows)

    def _getValsFromDf(self, idx: int, jdx: int):
        return [self._formatVal(self._df.iloc[idx, jdx])]

    def asLaTex(self, filepath: str, highlightMax: float = 5):
        latexMaker = singleRow(self._df)
        latexMaker.make(filepath, highlightMax=highlightMax)


class singleLineRangeTable(_generateTableImg):
    def __init__(self, firstDf: pd.DataFrame, secondDf: pd.DataFrame):
        nCellRows = 1
        self._firstDf = firstDf
        self._secondDf = secondDf
        super().__init__(firstDf, nCellRows)

    def _getValsFromDf(self, idx: int, jdx: int):
        t1 = self._formatVal(self._firstDf.iloc[idx, jdx])
        t2 = self._formatVal(self._secondDf.iloc[idx, jdx])
        return [f'({t1}, {t2})']

    def asLaTex(self, filepath: str, highlightMax: float = 5):
        latexMaker = singleRowBrackets(self._firstDf, self._secondDf)
        latexMaker.make(filepath, highlightMax=highlightMax)


class twoLinesTwoValuesTable(_generateTableImg):
    def __init__(self, topDf: pd.DataFrame, botDf: pd.DataFrame):
        nCellRows = 2
        self._brackets = False
        self._topDf = topDf
        self._botDf = botDf
        super().__init__(topDf, nCellRows)

    def _getValsFromDf(self, idx: int, jdx: int):
        valt = self._formatVal(self._topDf.iloc[idx, jdx])
        valb = self._formatVal(self._botDf.iloc[idx, jdx])
        return [valt, valb]

    def asLaTex(self, filepath: str, highlightMax: float = 5):
        latexMaker = doubleRow(self._topDf, self._botDf)
        latexMaker.make(filepath, highlightMax=highlightMax)


class twoLinesValueRangeTable(_generateTableImg):
    def __init__(self, topDf: pd.DataFrame, firstBotDf: pd.DataFrame,
                 secondBotDf: pd.DataFrame):
        nCellRows = 2
        self._brackets = True
        self._topDf = topDf
        self._firstBotDf = firstBotDf
        self._secondBotDf = secondBotDf
        super().__init__(topDf, nCellRows)

    def _getValsFromDf(self, idx: int, jdx: int):
        valt = self._formatVal(self._topDf.iloc[idx, jdx])
        b1 = self._formatVal(self._firstBotDf.iloc[idx, jdx])
        b2 = self._formatVal(self._secondBotDf.iloc[idx, jdx])
        valb = f'({b1}, {b2})'
        return [valt, valb]

    def asLaTex(self, filepath: str, highlightMax: float = 5):
        latexMaker = doubleRowBrackets(self._topDf, self._firstBotDf,
                                       self._secondBotDf)
        latexMaker.make(filepath, highlightMax=highlightMax)
