# Imports
import sys
import os

# Libraries
import numpy as np

# Project
from display import displayTwoD, displayThreeD
import config as param

# --------------------------------------------------------------------

rootDir = os.path.dirname(os.path.dirname(os.path.realpath(__file__)))
sys.path.append(rootDir)
os.chdir(rootDir)

pathSave = os.path.join(os.getcwd(), param.pathSave)

# --------------------------------------------------------------------

if __name__ == '__main__':
    # parameters
    seed = 7
    rng = np.random.default_rng(seed=seed)

    # create plots
    displayTwoD(pathSave)
    displayThreeD(pathSave)
