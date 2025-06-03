# Imports
import sys
import os

# Packages
from matplotlib import pyplot as plt

# Project
import config as param
from display import singleRowMse, summaryGenerator
from app import collectErrors, getResultsAll

# --------------------------------------------------------------------
rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(rootDir)
os.chdir(rootDir)

pathSave = os.path.join(os.getcwd(), param.pathSave)
pathData = os.path.join(os.getcwd(), param.pathData)
# --------------------------------------------------------------------

# Main

if __name__ == '__main__':
    pathSaveErr = os.path.join(pathSave, 'errorCalc')
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

    plt.rcParams['text.usetex'] = True  # Use Latex formatting

    for simType in simList:
        print(f'Working on: {simType}')

        try:
            try:
                # Get subdf
                subdf = getResultsAll(pathData, simType)
                subdf.drop_duplicates(inplace=True)
            except Exception as e:
                print(e)
                print('Skipping to next')
                continue

            errorDf = collectErrors(subdf, displayList)

            gen = summaryGenerator(errorDf, pathSaveErr)
            gen.summaryAllEstimators(dList=None, nList=None)
            gen.summaryGKDE(dList=None, nList=None)
            gen.summaryCLAKDE(dList=None, nList=None)
            gen.summaryGHP(dList=None, nList=None)
            gen.summaryKNN(dList=None, nList=None)
            gen.summaryGHPCLAKDE(dList=None, nList=None)

            if simType == 'gvg':
                gen.summaryNoNaiveBayes(dList=None, nList=None)

                # Bayes Classifier Table MSE
                bceDf = errorDf.pivot(index=colName, columns=rowName,
                                      values='bayesClassifierError-MSE')
                p = os.path.join(pathSaveErrTables, 'summary-BC-MSE-'+simType)
                singleRowMse(bceDf).make(p)

        except OSError as e:
            print(e)
            print('Skipping to next')

    print('Done!')
