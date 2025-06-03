
# Imports
import os
import time
import logging
import random

# Packages
import psutil

# Project
import config as param

# Conditional and internal imports:
# nvidia_smi
# tensorflow
# torch

# --------------------------------------------------------------------


def _getCpuUtil(sampleTimes: int = 3):
    # Get CPU utilization
    util = 0
    for idx in range(sampleTimes):
        util = util + psutil.cpu_percent()
        time.sleep(0.2)
    return round(util/sampleTimes)


def _getGpuUtil(gpuIdx: int, nSamples: int = 3):
    # Get GPU utilization and memory
    # https://docs.nvidia.com/deploy/nvml-api/structnvmlUtilization__t.html#structnvmlUtilization__t
    import nvidia_smi
    handle = nvidia_smi.nvmlDeviceGetHandleByIndex(gpuIdx)
    util = 0
    mem = 0
    for idx in range(nSamples):
        res = nvidia_smi.nvmlDeviceGetUtilizationRates(handle)
        util = res.gpu + util
        resMem = nvidia_smi.nvmlDeviceGetMemoryInfo(handle)
        mem = mem + resMem.used/resMem.total*100

        if idx < nSamples-1:
            time.sleep(1)  # Refresh rate may take up to 1 second

    return round(util/nSamples), round(mem/nSamples)

# --------------------------------------------------------------------


class resourceSelector():
    """
    Selects resource for simulation use based on avaliability.

    Parameters
    ----------
    expDict: dictionary
        Dictionary with simulation parameters.
    cpuThreshold: float, optional
        0-100 % range for starting new simulations.
    gpuComThreshold: float, optional
        0-100 % range for starting new simulations.
    gpuMemThreshold: float, optional
        0-100 % range for starting new simulations.
    maxCpuSampleSize: int, optional
        If number of samples per class is under this size,
        then use CPU automatically.
    useGpu: bool, optional
        If GPU is available

    Notes
    -----
    Some estimators can not use the GPU (GKDE), so even GPU enabled
    simulations will use some CPU.

    Tensorflow will automatically preallocate a small amount of
    memory on all avalaible GPUs, even if CPU is selected for that task.

    When CPU-selected, estimators will be using multiple cores. For CPU-only,
    setting the maximum number of tasks lower than the number of cores
    is recommended.

    Defaults are taken from config.py.

    """

    def __init__(self, expDict: dict,
                 cpuThreshold: float = param.cpuUseThreshold,
                 gpuComThreshold: float = param.gpuUseThreshold,
                 gpuMemThreshold: float = param.gpuMemThreshold,
                 maxCpuSampleSize: int = param.maxCpuSampleSize,
                 useGpu: bool = param.useGPU):

        self._expDict = expDict
        self._useGpu = useGpu
        self._maxCpuSampleSize = maxCpuSampleSize
        self._cpuThreshold = cpuThreshold
        self._gpuComThreshold = gpuComThreshold
        self._gpuMemThreshold = gpuMemThreshold
        self._proccess = psutil.Process(os.getpid())

        if self._useGpu:
            self._initGpu()

    def getResource(self,
                    resourceFreeCheckTime: int = param.resourceFreeCheckTime,
                    allowedRetries: int = param.resourceAllowedRetries,
                    maxParallelSim: int = param.maxParallelSim,
                    delayStart: int = 10,
                    logger: logging.Logger = logging):
        """
        Checks for resource availability and selects resource for use.
        If no resources available, continues checking until time limit
        is reached, then raises exception.

        Parameters
        ----------
        resourceFreeCheckTime: int, optional
            Time between checks.
        allowedRetries: int, optional
            Number of times allowed to recheck if resources are taken.
        maxParallelSim: int, optional
            Maximum number of parallel process, used for staggered start.
        sampleTimes: int, optional
            Number of times to sample for average.
        delayStart: int, optional
            Time to delay first processes.
        logger: logger object, optional
            For messages.

        Returns
        -------
        useGpuId: int or None
            None indicates CPU to be used, an integer represents the identifier
            of the GPU to be used.

        """

        maxTimeMin = round(allowedRetries*resourceFreeCheckTime/60)
        self._delayStartup(maxParallelSim, delayStart, logger)

        # All resources taken -> Wait
        retries = 0
        resourceFree, self._useGpuId = self._resourceCheck()
        while not resourceFree:
            if retries > allowedRetries:
                msg = (f'All resources in use. '
                       f'Waited for {maxTimeMin} min '
                       'Trying another task. '
                       'Will retry later if allowed.')
                logger.warning(msg)
                raise Exception(msg)

            time.sleep(resourceFreeCheckTime)
            retries = retries+1
            resourceFree, self._useGpuId = self._resourceCheck()

        # Resource free, green light
        self.resourceUtilLogger()

        return self._useGpuId

    def _initGpu(self):
        # Tensorflow allocates all GPU memory by default
        # disable to run parallel simulations
        import nvidia_smi
        from tensorflow import config as config

        nvidia_smi.nvmlInit()
        gpuList = config.experimental.list_physical_devices(device_type='GPU')
        for gpu in gpuList:
            config.experimental.set_memory_growth(gpu, True)
        self._nGPU = len(gpuList)

    def _delayStartup(self, maxParallelSim: int, delayStart: int,
                      logger: logging.Logger):
        # Stagger startup so threads have a chance to grab resources
        # Allow some gpu/cpu growth
        if self._expDict['idx'] < maxParallelSim:
            random.seed(self._expDict['idx'])
            sleepStartTime = round(random.uniform(0, 60))
            sleepStartTime = self._expDict['idx']*delayStart
            logStr = (f"{self._expDict} Delaying startup for: "
                      f"{sleepStartTime} s")
            logger.info(logStr)
            time.sleep(sleepStartTime)

    def _resourceCheck(self):
        # Note: CPU-selected proccesses will still try to allocate GPU memory
        # If it exists. Likewise, GPU-selected proccess will use some CPU.
        # This can cause slowdown even if function says resource is free.
        cpuSampleCheck = (self._expDict['idx'] <= self._maxCpuSampleSize)
        cpuUtilCheck = (_getCpuUtil() < self._cpuThreshold)

        if cpuSampleCheck & cpuUtilCheck:
            # in case of small arrays, use CPU
            # FOr low numbers of points, it is faster
            return True, None

        # Check for gpu
        if self._useGpu:
            # Alternate list order
            if self._expDict['idx'] % 2 == 0:
                idxList = reversed(range(self._nGPU))
            else:
                idxList = range(self._nGPU)

            for idx in idxList:
                util, mem = _getGpuUtil(idx)
                if (util < self._gpuComThreshold):
                    if (mem < self._gpuMemThreshold):
                        return True, idx

        return False, None

    def resourceUtilLogger(self, logger: logging.Logger = logging):
        """
        Output memory usage to log file.

        Parameters
        ----------
        logger: logging object
            Needs to be previously initalized.

        Notes
        -------


        """
        if self._useGpuId == None:
            cpuUtil = _getCpuUtil(sampleTimes=1)
            msg = (f'Resources sufficent. '
                   f'Using CPU with utilzation {cpuUtil}%. '
                   f'Starting: {self._expDict}')
            logger.info(msg)
        else:
            gpuUtil, gpuMem = _getGpuUtil(self._useGpuId, nSamples=1)
            msg = (f'Resources sufficent. '
                   f'Using GPU: {self._useGpuId} '
                   f' with utilzation {gpuUtil}% and used memory {gpuMem}%. '
                   f'Starting: {self._expDict}')
            logger.info(msg)

        self.logMemoryUsage(logger)

    def logMemoryUsage(self, logger):
        """
        Output memory usage to log file.

        Parameters
        ----------
        logger: logger object
            Needs to be previously initalized.

        Notes
        -------


        """
        memPre = self._proccess.memory_info().rss // (1024*1024)
        memAva = psutil.virtual_memory().available // (1024*1024)

        logger.info((f"Currently used memory: {memPre} MB. "
                    f"Available: {memAva} MB"))
