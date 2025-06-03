# Imports
from abc import ABC, abstractmethod
import pandas as pd
from typing import List

# Project
from .._formatting import roundSigDig, formatEstimatorName

# --------------------------------------------------------------------
# Expanded Latex Table Classes


class _latexTableBase(ABC):
    def __init__(self, mainDf: pd.DataFrame, xLabel: str, yLabel: str):
        self._nRows = len(mainDf)
        self._nCols = len(mainDf.columns)
        self._topLabel = xLabel
        self._leftLabel = yLabel
        self._rowLabelList = mainDf.index.to_list()
        self._fullSpanTable = False

    @abstractmethod
    def _getCellVal(self, idx: int, jdx: int, highlightMax: float):
        pass

    @abstractmethod
    def make(self, filepath: str, highlightMax: float = 5):
        """
        Creates and exports table as LaTex. (.tex file)

        Parameters
        ----------
        filepath: str
            Location to save file, with filename, not extension.
        highlightMax: float
            Bold values or ranges under this number.
        """
        pass

    def _newFile(self, textStr):
        with open(self._filepath, 'w') as f:
            f.write(textStr)

    def _appendToFile(self, textStr):
        with open(self._filepath, 'a') as f:
            f.write(textStr)

    def _makeHeader(self, colLabelList: List[str]):
        # Headers
        if self._fullSpanTable:
            tableType = r"\begin{tabular*}{\linewidth}"
        else:
            tableType = r"\begin{tabular}"

        if self._leftLabel is None:
            leftSep = ''
            leftCol = 'c'
        else:
            leftSep = '&'
            leftCol = 'cc'

        ht = tableType+r"{@{\extracolsep{\fill}}"+leftCol
        h_over = leftSep + \
            r"&\multicolumn{"+str(self._nCols)+r"}{c}{"+self._topLabel+r"}"
        h_over = h_over + r'\\' + r'\cline{3-'+str(self._nCols+2)+r'}'
        h = ""+leftSep
        for idx in range(self._nCols):
            colLabel = colLabelList[idx]
            h = h+"&"+""+str(colLabel)
            ht = ht+'c'
        h = h+r'\\ \cline{2-'+str(self._nCols+2)+r'}'
        ht = ht+r'}'+r'\\'  # +r'\cline{3-'+str(ncols+2)+r'}'

        self._newFile(f"{ht}\n")
        self._appendToFile(f"{h_over}\n")
        self._appendToFile(f"{h}\n")

    def _closeTable(self):
        if self._fullSpanTable:
            end = r'\end{tabular*}'
        else:
            end = r'\end{tabular}'
        self._appendToFile(end)

    def _getLeftLabel(self):
        if self._leftLabel is None:
            return ''
        else:
            fullRows = str(self._nRows*self._nCellRows)
            width = '8mm' if self._nCellRows == 1 else '3mm'
            l1 = r"\parbox[t]{"+width+r"}{\multirow{"+fullRows
            l2 = r"}{*}{\rotatebox[origin=c]{90}{"+self._leftLabel+r"}}}"
            return l1+l2

    def _colourCell(self, idx: int, jdx: int):
        if (idx % 2 == 0) & (jdx % 2 == 0):
            coloring = r"\cellcolor{gray!25}"
        elif (idx % 2 == 1) & (jdx % 2 == 1):
            coloring = r"\cellcolor{gray!25}"
        else:
            coloring = ""

        return coloring


class singleRow(_latexTableBase):
    """
    Creates and exports table as LaTex.
    With one value per cell.

    Parameters
    ----------
    df: dataframe
        Table to export.
    xLabel: str
        Label on left.
    yLabel: str
        Label on top.
    """

    def __init__(self, df: pd.DataFrame, xLabel='Number of Features',
                 yLabel=r'\makecell{Number of Samples\\Per Class}'):
        super().__init__(df, xLabel, yLabel)
        self._df = df
        self._nCellRows = 1
        self._fullSpanTable = False

    def _getCellVal(self, idx, jdx, highlightMax):
        try:
            valb1 = self._df.iloc[idx, jdx]
            if abs(valb1) < highlightMax:
                valb1 = roundSigDig(self._df.iloc[idx, jdx], 3)
                valb = r"\textbf{"+valb1+"}"
            else:
                valb1 = roundSigDig(self._df.iloc[idx, jdx], 3)
                valb = f'{valb1}'
        except Exception as e:
            valb = 'err'

        return valb

    def make(self, filepath: str, highlightMax: float = 5):
        self._filepath = filepath+'.tex'
        self._makeHeader(self._df.columns)

        for idx in range(self._nRows):
            rowLabel = self._rowLabelList[idx]

            # Left Hand label
            b = self._getLeftLabel() if idx == 0 else ""
            b = b + "&" if not (self._leftLabel is None) else b
            b = b+str(rowLabel)

            for jdx in range(self._nCols):
                valb = self._getCellVal(idx, jdx, highlightMax)
                b = b+"&"+self._colourCell(idx, jdx)+valb

            b = b+r'\\ \cline{2-'+str(self._nCols+2)+r'}'

            self._appendToFile(f"{b}\n")

        self._closeTable()


class singleRowMse(singleRow):
    """
    Creates and exports table as LaTex.
    With one value per cell, taken by mean
    to be used for overall MSE table.

    Parameters
    ----------
    df: dataframe
        Table to export.
    """

    def __init__(self, df: pd.DataFrame):
        xLabel = 'Number of Samples Per Class'
        yLabel = None
        super().__init__(df, xLabel=xLabel, yLabel=yLabel)
        self._df = df
        self._nCellRows = 1
        self._fullSpanTable = True
        self._rowLabelList = ['MSE']
        self._nRows = 1

    def _colourCell(self, idx, jdx):
        return ""

    def _getCellVal(self, idx, jdx: int, highlightMax):
        colName = self._df.columns[jdx]
        # Remove sig digs for this
        try:
            val = roundSigDig(self._df[colName].mean(), 2)
        except ValueError:
            val = self._df[colName].mean()
        return val


class singleRowBrackets(singleRow):
    """
    Creates and exports table as LaTex.
    With two values per cell in the format: (left, right)

    Parameters
    ----------
    firstDf: dataframe
        Left cell values.
    secondDf: dataframe
        Right cell values.
    xLabel: str
        Label on left.
    """

    def __init__(self, firstDf: pd.DataFrame, secondDf: pd.DataFrame,
                 xLabel='Number of Features'):
        super().__init__(firstDf, xLabel=xLabel)
        self._firstDf = firstDf
        self._secondDf = secondDf
        self._fullSpanTable = False

    def _getCellVal(self, idx: int, jdx: int, highlightMax: float):
        try:
            valTemp1 = self._firstDf.iloc[idx, jdx]
            valTemp2 = self._secondDf.iloc[idx, jdx]
            valb1 = roundSigDig(valTemp1, 3)
            valb2 = roundSigDig(valTemp2, 3)
            if abs(valTemp1-valTemp2) < highlightMax:
                valb = r"\textbf{("+valb1+", "+valb2+")}"
            else:
                valb = f'({valb1}, {valb2})'
        except Exception as e:
            valb = 'err'

        return valb


class doubleRow(_latexTableBase):
    """
    Creates and exports table as LaTex.
    With two values per cell in the format:

    top
    bottom

    Parameters
    ----------
    topDf: dataframe
        Top cell values.
    botDf: dataframe
        Bottom cell values.
    xLabel: str
        Label on left.
    yLabel: str
        Label on top.
    """

    def __init__(self, topDf: pd.DataFrame, botDf: pd.DataFrame,
                 xLabel='Number of Features',
                 yLabel='Number of Samples Per Class'):
        super().__init__(topDf, xLabel, yLabel=yLabel)
        self._topDf = topDf
        self._botDf = botDf
        self._nCellRows = 2
        self._fullSpanTable = True
        self._rowLabelList = topDf.index

    def _getCellVal(self, idx: int, jdx: int, highlightMax: float):
        valTempT = self._topDf.iloc[idx, jdx]
        valTempB = self._botDf.iloc[idx, jdx]

        if isinstance(valTempT, str):
            valt = formatEstimatorName(valTempT)
            valb = roundSigDig(valTempB, 3)
            return valt, valb
        else:
            try:
                valtFormat = roundSigDig(valTempT, 3)
                valbFormat = roundSigDig(valTempB, 3)
                if abs(valTempT-valTempB) < highlightMax:
                    valt = r"\textbf{"+valtFormat+"}"
                    valb = r"\textbf{"+valbFormat+"}"
                else:
                    valt = valtFormat
                    valb = valbFormat
            except Exception:
                valt = 'err'
                valb = 'err'

        return valt, valb

    def make(self, filepath: str, highlightMax: float = 5):
        self._filepath = filepath+'.tex'
        self._makeHeader(self._topDf.columns)

        for idx in range(self._nRows):
            rowLabel = self._topDf.index[idx]
            a = self._getLeftLabel() if idx == 0 else ""
            a = a + "&" if not (self._leftLabel is None) else a
            a = a+r"\multirow{2}{*}{"+str(rowLabel)+"}"
            b = "&" if not (self._leftLabel is None) else ""
            for jdx in range(self._nCols):
                valt, valb = self._getCellVal(idx, jdx, highlightMax)
                coloring = self._colourCell(idx, jdx)
                a = a+"&"+coloring+valt
                b = b+"&"+coloring+valb
            a = a+r'\\'
            b = b+r'\\ \cline{2-'+str(self._nCols+2)+r'}'

            self._appendToFile(f"{a}\n")
            self._appendToFile(f"{b}\n")

        self._closeTable()


class doubleRowBrackets(doubleRow):
    """
    Creates and exports table as LaTex.
    With three values per cell in the format:

    top
    (left, right)

    Parameters
    ----------
    topDf: dataframe
        Top cell values.
    firstDf: dataframe
        Bottom left cell values.
    secondDf: dataframe
        Bottom right cell values.
    xLabel: str
        Label on left.
    yLabel: str
        Label on top.
    """

    def __init__(self, topDf: pd.DataFrame, firstDf: pd.DataFrame,
                 secondDf: pd.DataFrame, xLabel='Number of Features',
                 yLabel='Number of Samples Per Class'):
        super().__init__(topDf, firstDf, xLabel=xLabel, yLabel=yLabel)
        self._topDf = topDf
        self._firstDf = firstDf
        self._secondDf = secondDf
        self._fullSpanTable = True

    def _getCellVal(self, idx: int, jdx: int, highlightMax: float,
                    sigDigs: int = 3):
        try:
            valTemp1 = self._firstDf.iloc[idx, jdx]
            valTemp2 = self._secondDf.iloc[idx, jdx]
            valb1 = roundSigDig(valTemp1, sigDigs)
            valb2 = roundSigDig(valTemp2, sigDigs)
            name = self._topDf.iloc[idx, jdx]

            if abs(valTemp1-valTemp2) < highlightMax:
                valt = formatEstimatorName(name, bold=True)
                valb = r"\textbf{("+valb1+", "+valb2+")}"
            else:
                valt = formatEstimatorName(name)
                valb = f'({valb1}, {valb2})'
        except Exception as e:
            valt = str(self._topDf.iloc[idx, jdx])
            valb = 'err'

        return valt, valb
