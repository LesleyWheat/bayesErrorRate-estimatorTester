"""
Application
=====

Provides files to support and enable the testing of BER estimators.

"""
from ._setup import appSetup
from ._groundTruth import getBerList, calcGap
from ._dispatcher import runSimPool
from ._taskLists import buildSimParamList
from ._results import getResultsOneSim, getResultsAll

from ._analysis import errorCalc, collectErrors
from ._analysis import reweightErrors, errorExtractor