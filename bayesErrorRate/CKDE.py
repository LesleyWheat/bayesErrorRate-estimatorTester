# Imports
import os
from typing import List

# Packages
import torch
import numpy as np

# Import individual functions for speedup
from numpy import unique
from torch import from_numpy, tensor, zeros
from torch import reshape, arange
from torch import logsumexp, log

# Conditional and internal imports:
# lakde: Due to environment variables need to be set first

# --------------------------------------------------------------------
# Functions


def CLAKDE(X: np.ndarray, Y: np.ndarray, iterations: int = 200,
           useGpuId: int | None = None, rtol: float = 1e-3) -> dict:
    """
    Runs the CKDE estimator using LAKDE.

    Parameters
    ----------
    X: numpy array
        Must be of shape (n, m) where n is the number of samples and
        m is the number of features.
    Y: numpy array
        Must be of shape (n) where n is the number of samples.
        Should contain two classes. Class labels should be integers.
    iterations: int, optional
        Maximum number of iterations for LAKDE.
    useGpuId: int, optional
        Identifier for which GPU to use.
        None is CPU only.
    rtol: float
        Tolerance for LAKDE early stop.

    Returns
    -------
    results: dictionary
        Dictionary containing estimate: 'CKDE_LAKDE-LOO_ES' and number
        of steps needed.

    Notes
    -------
    Number of points for each class should be similar for best results.

    """

    device = _selectTorchDevice(useGpuId)
    classLabels = unique(Y)

    if not (len(classLabels) == 2):
        msg = (f'CLAKE given {len(classLabels)} when only binary '
               'classification is supported.')
        raise ValueError(msg)

    # train KDE bandwidth matrix
    kdeList = []
    classDataList = []
    for className in classLabels:
        classData = X[(Y == className), :]
        classDataOnDevice = from_numpy(classData).float().to(device)
        classDataList.append(classDataOnDevice)
        trainedKde = _trainKde(classDataOnDevice, iterations, rtol)
        kdeList.append(trainedKde)

    # No need to recalculate
    otherDist = [kdeList[1], kdeList[0]]
    otherClassData = [classDataList[1], classDataList[0]]

    # Calculate similarity
    crossData = _kdeCross(classDataList, otherClassData, otherDist)
    sameData = _kdeSameLoo(classDataList, kdeList)
    j = (crossData.sum() - sameData.sum()).exp().cpu().detach().numpy()*(1/2)

    results = {
        'CKDE_LAKDE-LOO_ES': j,
        'CKDE_LAKDE_A-iter_steps': kdeList[0].iter_steps,
        'CKDE_LAKDE_B-iter_steps': kdeList[1].iter_steps
    }

    return results


def _selectTorchDevice(useGpuId: int | None) -> torch.device:
    if useGpuId is None:
        device = torch.device("cpu")
    else:
        device = torch.device(f"cuda:{useGpuId}")
    return device


def _kdeCross(classDataList: List[torch.FloatTensor],
              otherClassData: List[torch.FloatTensor],
              otherDist: object) -> torch.FloatTensor:
    nClasses = len(classDataList)
    crossData = zeros(nClasses)

    for idx in range(nClasses):
        kdeA = otherDist[idx]
        dataA = otherClassData[idx]
        dataB = classDataList[idx]
        n = dataB.shape[0]
        n = dataB.shape[0]
        p = kdeA.log_pred_density(dataA, dataB)
        crossData[idx] = logsumexp(p, 0) - log(tensor(n))

    return crossData


def _kdeSameLoo(classDataList: List[torch.FloatTensor],
                kdeList: List[object]) -> torch.FloatTensor:
    nClasses = len(classDataList)
    sameData = zeros(nClasses)

    for idx in range(nClasses):
        classData = classDataList[idx]
        kde = kdeList[idx]
        n = classData.shape[0]
        p = zeros(n)

        # Do multiple calculations while data is on GPU
        for jdx in range(n):
            x = reshape(classData[jdx], (1, classData.shape[1]))
            if jdx == 0:
                # Only allocate data once
                A_LOO = classData[arange(0, classData.shape[0]) != jdx, ...]
            else:
                A_LOO[jdx-1, :] = xLast
                A_LOO[jdx-1, :] = xLast

            p[jdx] = kde.log_pred_density(A_LOO, x)
            xLast = x

        sameData[idx] = logsumexp(p, 0) - log(tensor(n))

    return sameData


def _trainKde(classDataOnDevice: torch.FloatTensor, iterations: int,
              rtol: float):
    os.environ['LAKDE_NO_EXT'] = '1'

    from lakde import SharedFullKDE
    from lakde.callbacks import LikelihoodCallback

    kde = SharedFullKDE(verbose=False)

    # Early stop callback
    early_stop_cond = LikelihoodCallback(
        classDataOnDevice,
        rtol=rtol,
        verbose=False,
    )

    kde.fit(classDataOnDevice, iterations=iterations,
            callbacks=[early_stop_cond])

    # Store
    return kde
