# Imports
import sys
import os

# Project files
from display import makePlots
from app import getResultsOneSim
import config as param

# --------------------------------------------------------------------
rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(rootDir)
os.chdir(rootDir)
pathSave = os.path.join(os.getcwd(), param.pathSave)
pathData = os.path.join(os.getcwd(), param.pathData)
# --------------------------------------------------------------------
# Functions
if __name__ == '__main__':
    pathSaveFolder = os.path.join(pathSave, 'singleSimGraphs')

    # For paper
    paramList = [{'codename': "gvg", 'nClassSamples': 100, 'nFeatures': 2}]

    for paramDict in paramList:
        print(paramDict)
        nFeatures = paramDict['nFeatures']
        nClassSamples = paramDict['nClassSamples']
        codename = paramDict['codename']
        filename = f'{codename}_{nFeatures}d_{nClassSamples}n'
        pathSave = os.path.join(pathSaveFolder, filename)
        resultsDf = getResultsOneSim(pathData, codename, nFeatures,
                                     nClassSamples)
        makePlots(resultsDf, pathSave)
