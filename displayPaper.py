'''
Script to generate graphs, plots and tables for the paper.
'''

# Core
import os

# Libraries
import matplotlib as mpl
from matplotlib import pyplot as plt

# Project
import config as param
from display import displayTwoD, displayThreeD, makePlots
from display import summaryGenerator, singleRowMse
from app import getResultsOneSim, collectErrors, getResultsAll

# --------------------------------------------------------------------
PATH_SAVE = os.path.join(os.getcwd(), param.pathSave)
PATH_DATA = os.path.join(os.getcwd(), param.pathData)
# --------------------------------------------------------------------


def displayDistributions():
    # create plots
    pathImg = os.path.abspath(os.path.join(PATH_SAVE, 'distributionImgs'))

    # Seed is set inside functions
    displayTwoD(pathImg)
    displayThreeD(pathImg)


def displaySingleSimulation():
    pathSaveFolder = os.path.join(PATH_SAVE, 'singleSimGraphs')

    # For paper
    paramList = [{'codename': "tvt", 'nClassSamples': 1000, 'nFeatures': 20},
                 {'codename': "tvt", 'nClassSamples': 2500, 'nFeatures': 2},
                 {'codename': "tvt", 'nClassSamples': 2500, 'nFeatures': 10}]

    for paramDict in paramList:
        nFeatures = paramDict['nFeatures']
        nClassSamples = paramDict['nClassSamples']
        codename = paramDict['codename']
        pathSave = os.path.join(pathSaveFolder,
                                f'{codename}_{nFeatures}d_{nClassSamples}n')
        resultsDf = getResultsOneSim(PATH_DATA, codename, nFeatures,
                                     nClassSamples)
        makePlots(resultsDf, pathSave)

# --------------------------------------------------------------------


def displaySummaryResults():
    pathSaveErr = os.path.join(PATH_SAVE, 'errorCalc')
    pathSaveErrTables = os.path.join(pathSaveErr, 'texTables')
    pathSaveErrTablesPNG = os.path.join(pathSaveErr, 'texTables-images')

    displayList = ['bayesClassifierError', 'NaiveBayesError',
                   'CKDE_LAKDE-LOO_ES', 'GHP(L)-CKDE(LL)(ES)',
                   'GHP_L', 'GHP_M', 'GHP_H',
                   'GKDE_silverman',
                   'GKDE_0.5', 'GKDE_0.1', 'GKDE_0.25',
                   'GKDE_0.05', 'GKDE_0.0025',
                   'KNN(squared_l2)_H', 'KNN(squared_l2)_M',
                   'KNN(squared_l2)_L',
                   ]

    # setup
    os.makedirs(pathSaveErrTables, exist_ok=True)
    os.makedirs(pathSaveErrTablesPNG, exist_ok=True)

    simList = ['tvs', 'svs', 'tvt', 'gvg']
    rowName = 'nClassSamples'
    colName = 'nFeatures'
    nList = [500, 1000, 1500, 2000, 2500]

    for simType in simList:
        print(f'Working on: {simType}')

        if simType == 'gvg':
            dList = [2, 3, 4, 10, 20, 30]
        elif simType == 'tvt':
            dList = [2, 4, 6, 8, 10, 12, 20]
        elif simType == 'tvs':
            dList = [2, 4, 6, 8, 10, 12, 14]
        elif simType == 'svs':
            dList = [2, 3, 4, 6, 8, 10, 12]
        else:
            dList = [2, 4, 6, 8, 10, 12]

        try:
            try:
                # Get subdf
                subdf = getResultsAll(PATH_DATA, simType)
                subdf.drop_duplicates(inplace=True)
            except Exception as e:
                print(e)
                print('Skipping to next')
                continue

            errorDf = collectErrors(subdf, displayList)

            gen = summaryGenerator(errorDf, pathSaveErr)
            gen.summaryAllEstimators(dList=dList, nList=nList)
            gen.summaryGHP(dList=dList, nList=nList)
            gen.summaryKNN(dList=dList, nList=nList)
            gen.summaryGHPCLAKDE(dList=dList, nList=nList)

            if simType == 'gvg':
                gen.summaryNoNaiveBayes(dList=dList, nList=nList)

                # Bayes Classifier Table MSE
                bceDf = errorDf.pivot(index=colName, columns=rowName,
                                      values='bayesClassifierError-MSE')
                p = os.path.join(pathSaveErrTables, 'summary-BC-MSE-'+simType)
                singleRowMse(bceDf).make(p)

            if simType == 'tvt':
                dList = [2, 4, 6, 8, 10, 12]
                gen.summaryGKDE(dList=dList, nList=nList)
                gen.summaryCLAKDE(dList=dList, nList=nList)
            else:
                gen.summaryGKDE(dList=dList, nList=nList)
                gen.summaryCLAKDE(dList=dList, nList=nList)

        except OSError as e:
            print(e)
            print('Skipping to next')

# --------------------------------------------------------------------


if __name__ == '__main__':
    mpl.use('agg')
    plt.rcParams['text.usetex'] = True  # Use Latex formatting

    displayDistributions()
    displaySummaryResults()
    displaySingleSimulation()
