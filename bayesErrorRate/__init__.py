"""
BER Estimation and Approximation
=====

Supports BER high-accuracy approximation using monte carlo simulation
and contains the new estimator CLAKDE.

"""
from ._classifer import bayesClassifierError
from .CKDE import CLAKDE
from ._monteCarlo import mcBerBatch