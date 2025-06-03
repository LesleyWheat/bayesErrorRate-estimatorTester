# Imports
import os
import pandas as pd
from typing import List

# Project
from app import errorExtractor

# Module
from ._tables import singleLineTable, singleLineRangeTable
from ._tables import twoLinesTwoValuesTable, twoLinesValueRangeTable

# --------------------------------------------------------------------


class summaryGenerator():
    """
    Makes tables.

    Parameters
    ----------
    errorDf: Pandas dataframe
        Error data from estimators.
    pathSaveErr: path
        Folder location.
    rowName: str, optional
        Name of values for rows.
    colName: str, optional
        Name of values for columns.

    """

    def __init__(self, errorDf: pd.DataFrame, pathSaveErr: str,
                 rowName='nClassSamples',
                 colName='nFeatures'):
        self._errors = errorDf
        self._pathSave = pathSaveErr
        self._rowName = rowName
        self._colName = colName
        self._simType = errorDf['simType'].unique()[0]
        self._nameLabel = 'lowestEstimateName'
        self._valLabel = 'lowestEstimateVal'
        self._lowLabel = 'lowestEstimateLowerBound'
        self._upLabel = 'lowestEstimateUpperBound'
        self._makeFolders(pathSaveErr)

    def _makeFolders(self, pathSaveErr: str):
        name = 'texTables'
        self._pathSaveTables = os.path.join(pathSaveErr, name)
        self._pathSaveTablesImg = os.path.join(pathSaveErr, name+'-images')
        os.makedirs(self._pathSaveTables, exist_ok=True)
        os.makedirs(self._pathSaveTablesImg, exist_ok=True)

    def _doBounds(self, errorBestDf: pd.DataFrame, subDfTop: pd.DataFrame,
                  estimatorList: List[str], saveKey: str):
        pathImg = os.path.join(self._pathSaveTablesImg, saveKey)
        pathTbl = os.path.join(self._pathSaveTables, saveKey)

        subdf_BL = errorBestDf.pivot(index=self._rowName,
                                     columns=self._colName,
                                     values=self._lowLabel)
        subdf_BU = errorBestDf.pivot(index=self._rowName,
                                     columns=self._colName,
                                     values=self._upLabel)

        if len(estimatorList) == 1:
            tbl = singleLineRangeTable(subdf_BL, subdf_BU)
            tbl.asImage(pathImg)
            tbl.asLaTex(pathTbl)
        else:
            tbl = twoLinesValueRangeTable(subDfTop, subdf_BL, subdf_BU)
            tbl.asImage(pathImg)
            tbl.asLaTex(pathTbl)

    def _doMse(self, errorBestDf: pd.DataFrame, subDfTop: pd.DataFrame,
               estimatorList: List[str], saveKey: str):
        pathImg = os.path.join(self._pathSaveTablesImg, saveKey)
        pathTbl = os.path.join(self._pathSaveTables, saveKey)

        subDfBot = errorBestDf.pivot(index=self._rowName,
                                     columns=self._colName,
                                     values=self._valLabel)

        if len(estimatorList) == 1:
            tbl = singleLineTable(subDfBot)
            tbl.asImage(pathImg)
            tbl.asLaTex(pathTbl)
        else:
            tbl = twoLinesTwoValuesTable(subDfTop, subDfBot)
            tbl.asImage(pathImg)
            tbl.asLaTex(pathTbl)

    def _makeSummary(self, code: str, estimatorList: List[str] | None,
                     dList: List[int] | None = None, nList: List[int] | None = None,
                     exportMSE=True, exportBounds=True):
        extractor = errorExtractor(self._errors)
        errorBestDf = extractor.calc(useEstimators=estimatorList,
                                     dList=dList, nList=nList)

        subDfTop = errorBestDf.pivot(index=self._rowName,
                                     columns=self._colName,
                                     values=self._nameLabel)

        fStr = f'summary-table-{code}'
        bounStr = f'-bounds-{self._simType}'
        mseKey = fStr+f'-{self._simType}'
        bKey = fStr+bounStr

        if exportBounds:
            self._doBounds(errorBestDf, subDfTop, estimatorList, bKey)

        if exportMSE:
            self._doMse(errorBestDf, subDfTop, estimatorList, mseKey)

    def summaryAllEstimators(self,
                             dList: List[int] | None = None,
                             nList: List[int] | None = None):
        estimatorList = ['NaiveBayesError', 'CKDE_LAKDE-LOO_ES',
                         'GHP(L)-CKDE(LL)(ES)', 'GKDE_0.5',
                         'GKDE_0.1', 'GKDE_0.25', 'GKDE_0.05', 'GKDE_0.0025',
                         'KNN(squared_l2)_H', 'KNN(squared_l2)_M',
                         'KNN(squared_l2)_L', 'GHP_M', 'GHP_H', 'GHP_L']
        code = 'bestEst'
        self._makeSummary(code, estimatorList, nList=nList, dList=dList)

    def summaryNoNaiveBayes(self,
                            dList: List[int] | None = None,
                            nList: List[int] | None = None):
        estimatorList = ['CKDE_LAKDE-LOO_ES', 'GHP(L)-CKDE(LL)(ES)', 'GKDE_0.5',
                         'GKDE_0.1', 'GKDE_0.25', 'GKDE_0.05', 'GKDE_0.0025',
                         'KNN(squared_l2)_H', 'KNN(squared_l2)_M',
                         'KNN(squared_l2)_L', 'GHP_M', 'GHP_H', 'GHP_L']
        code = 'noNaiveBayes'
        self._makeSummary(code, estimatorList, nList=nList, dList=dList)

    def summaryGKDE(self,
                    dList: List[int] | None = None,
                    nList: List[int] | None = None):
        code = 'GKDE'
        estimatorList = ['GKDE_0.5', 'GKDE_0.1', 'GKDE_0.25',
                         'GKDE_0.05', 'GKDE_0.0025']
        self._makeSummary(code, estimatorList, nList=nList, dList=dList,
                          exportBounds=False)

    def summaryCLAKDE(self,
                      dList: List[int] | None = None,
                      nList: List[int] | None = None):
        code = 'CLAKDE'
        estimatorList = ['CKDE_LAKDE-LOO_ES']
        self._makeSummary(code, estimatorList, nList=nList, dList=dList,
                          exportBounds=False)

    def summaryGHP(self,
                   dList: List[int] | None = None,
                   nList: List[int] | None = None):
        code = 'GHP'
        estimatorList = ['GHP_M', 'GHP_H', 'GHP_L']
        self._makeSummary(code, estimatorList, dList=dList, nList=nList)

    def summaryKNN(self,
                   dList: List[int] | None = None,
                   nList: List[int] | None = None):
        code = 'KNN'
        estimatorList = ['KNN(squared_l2)_H', 'KNN(squared_l2)_M',
                         'KNN(squared_l2)_L']
        self._makeSummary(code, estimatorList, dList=dList, nList=nList)

    def summaryGHPCLAKDE(self,
                         dList: List[int] | None = None,
                         nList: List[int] | None = None):
        code = 'GC'
        estimatorList = ['GHP(L)-CKDE(LL)(ES)', 'CKDE_LAKDE-LOO_ES', 'GHP_L']
        self._makeSummary(code, estimatorList, dList=dList, nList=nList,
                          exportBounds=False)
