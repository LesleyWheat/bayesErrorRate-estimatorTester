import pytest

# ------------------------------------------------------------------------

@pytest.mark.gpu
def test_gpuUtil():
    from .utils import checkGpu
    checkGpu()

    import nvidia_smi

    nvidia_smi.nvmlInit()
    deviceCount = nvidia_smi.nvmlDeviceGetCount()
    for i in range(deviceCount):
        handle = nvidia_smi.nvmlDeviceGetHandleByIndex(i)
        util = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        mem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        print(f"Device {i}")
        print(f"Mem Free: {mem.free/1024**2:5.2f}MB / {mem.total/1024**2:5.2f}MB")
        print(f"gpu-util: {util.gpu/100.0:3.1%}")
        print(f"gpu-mem: {util.memory/100.0:3.1%}")


@pytest.mark.cpu
def test_cpuUtil():
    import psutil

    print(psutil.cpu_percent())


# ------------------------------------------------------------------------

@pytest.mark.basic
def test_cacheSave():
    import os
    import pandas as pd
    from .utils import basicRunList
    from app._tools import findCache, saveCache, clearUniquePath

    pathCache = os.path.join('local', 'testCache')
    fullPath = os.path.join(os.getcwd(), pathCache)
    os.makedirs(fullPath, exist_ok=True)

    filename = 'testSave'
    runList = basicRunList(nLen = 2)
    df1 = pd.DataFrame(runList)

    clearUniquePath(os.path.join(fullPath, filename))
    saveCache(pathCache, filename, df1)
    df2 = findCache(pathCache, filename)

    print(df1)
    print(df2)

    pd.testing.assert_frame_equal(df1, df2)

    clearUniquePath(os.path.join(fullPath, filename))