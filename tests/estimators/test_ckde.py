import warnings
import pytest

def checkCuda():
    import numpy as np
    import torch
    useGpuId = 0
    arraySize = 100

    device = torch.device(f"cuda:{useGpuId}")

    t1 = torch.ones(arraySize, device=device)
    t2 = np.ones((arraySize))

    arrGpu = torch.from_numpy(t2).float().to(device)

@pytest.mark.clakde
@pytest.mark.lakde
@pytest.mark.gpu
@pytest.mark.parametrize("nFeatures", [3, 9, 10])
@pytest.mark.parametrize("nClassSamples", [100, 1000, 1500])
def test_CKDE4(nClassSamples, nFeatures, tol=1e-3):
    import time
    from bayesErrorRate.CKDE import CLAKDE
    from ..utils import dist_one
    from ..app import checkGpu

    checkGpu()
    checkCuda()

    X, Y = dist_one(seperate=False, nClassSamples=nClassSamples,
                    nFeatures=nFeatures)
    iterations = 10

    # J kde
    start_time = time.time()
    ja = CLAKDE(X, Y, iterations=iterations, useGpuId=None)
    ja = ja[f'CKDE_LAKDE-LOO_ES']
    cpu_runtime = round(time.time() - start_time, 2)
    print(f'{ja} {round(time.time() - start_time,2)}')

    start_time = time.time()
    jb = CLAKDE(X, Y, iterations=iterations, useGpuId=0)
    jb = jb[f'CKDE_LAKDE-LOO_ES']
    gpu_runtime = round(time.time() - start_time, 2)
    print(f'{jb} {round(time.time() - start_time,2)}')

    print(f"GPU1: {gpu_runtime} CPU: {cpu_runtime}")
    print(f"CPU: {ja} GPU1: {jb}")

    if gpu_runtime > cpu_runtime:
        warnings.warn((f'GPU is slower than CPU. '
                      f'GPU: {gpu_runtime} CPU: {cpu_runtime} '
                      f'n: {nClassSamples}'))

    assert ja == pytest.approx(jb, abs=tol)