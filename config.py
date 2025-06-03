import os

# Parameters

useGPU = True
"""
System setting, bool

Controls whether GPU should be used at all.

Notes
-----
Needs to be set before running install.py.
GPU is only supported on linux.
Depending on drivers, package versions may need to be changed.

"""

maxCores = 20
"""
System setting, int >= 1

Maximum number of cores to use.
"""

maxParallelSim = 6
"""
System setting, int >= 1

Maximum number of parallel simulations.

Notes
-----
Recommended to be set below the number of cores as many
estimators have parallesim built in.

"""

clearOldResultsOnStartup = False
"""
System setting, bool

Controls whether GPU should be used at all.

Notes
-----
Needs to be set before running install.py.
GPU is only supported on linux.
Depending on drivers, package versions may need to be changed.

"""


simNameList = ['gvg', 'tvt', 'tvs', 'svs']
"""
Simulation setting, list of strings

List of simulation types to run.

Notes
-----
Must be valid distribution types 

"""

seed = 1
"""
Simulation setting, int

Global seed.

Notes
-----
Individual seeds are still set by simulation as well.

"""

baseSamples = 10
"""
Simulation setting, int > 1

Controls the number of samples per batch.

Notes
-----
Total number of simulations = baseSamples*simulationBatches

"""

simulationBatches = 5
"""
Simulation setting, int >= 1

Controls the number of batches.

Notes
-----
Total number of simulations = baseSamples*simulationBatches.
Each batch will approximately evenly cover the BER range but
will start with different initial seed values.

"""

maximumBerSample = 2000
"""
BER ground truth setting, int > 1

Controls the maximum number of samples.

"""

maxBerGap = 0.01
"""
BER ground truth setting, float > 0

Maximum allowable gap in 

"""

berNewPointsPerLoop = 5
"""
BER ground truth setting, int >= 1

Number of random new points to draw when looking to close gaps.

"""

maxBerLoops = 2000
"""
BER ground truth setting, int >= 1

Maximum number of loops can be used when trying to meet the minimum
gap size.
"""

simBaseParameters = {
    "gvg": {"varianceList": [4.9, 5.1],
            "offsetList": [0.01, 1, 4, 10, 20],
            "maxBER": 0.49,
            "minBER": 0.01,
            "dList": [2, 3, 4, 10, 20, 30],
            "nList": [500, 1000, 1500, 2000, 2500]
            },
    "tvt": {"varianceList": [4.9, 5.1],
            "offsetList": [0.01, 1, 4, 10, 20],
            "maxBER": 0.49,
            "minBER": 0.01,
            "dList": [2, 4, 6, 8, 10, 12, 20],
            "nList": [500, 1000, 1500, 2000, 2500]
            },
    "tvs": {"varianceList": [4.9, 5.1],
            "offsetList": [0.01, 1, 4, 10, 20],
            "maxBER": 0.49,
            "minBER": 0.01,
            "dList": [2, 4, 6, 8, 10, 12, 14],
            "nList": [500, 1000, 1500, 2000, 2500]
            },
    "svs": {"varianceList": [4.9, 5.1],
            "offsetList": [0.01, 1, 4, 10, 20],
            "maxBER": 0.49,
            "minBER": 0.01,
            "dList": [2, 3, 4, 6, 8, 10, 12],
            "nList": [500, 1000, 1500, 2000, 2500]
            },
}
"""
Simulation setting: List of dictionaries that contain specific
parameters for different simulation types.

Parameters
-----
varianceList: list of floats
        Initial grid search list for creating a range of BER
        ground truth values. Minimum of two values.
offsetList: list of floats
        Initial grid search list for creating a range of BER
        ground truth values. Minimum of two values.
maxBER: float
        Note for any problem, the maximum is (number of classes - 1)/
        (number of classes).
minBER: float
        Minimum BER value
        Minimum: 0.
dList: list of ints
        List of number of features to run simulations for.
nList: list of ints
        List of number of samples per class to run simulations for.

Notes
-----
Simulation parameters are only used if simulation type is
included in simNameList.
"""

timeLimitSec = 200
"""
System setting, int > 1

Maximum timeout for each individual simulation
(points draw + running estimators).

Notes
-----
This prevents hanging and resource crowding.
May need to be set higher for more points.
System-dependant.

If simulations are killed by timeout, they will be retried.
If simulations can still not be completed at end of the program,
set clearOldResultsOnStartup to false, edit settings and run the
program again. It will only work on the unfinished simulations.

"""

checkpointInterval = 30*60
"""
System setting, int > 1

How often to save results in seconds.

Notes
-----
Simulations can be resumed if unfinished if clearOldResultsOnStartup
is set to false.

"""

reducedPrint = True
"""
System setting, bool

Print less to console.
"""

cpuUseThreshold = 50
"""
System resource setting, float, 0-100

Start new CPU tasks if usage is under this threshold.
"""

gpuUseThreshold = 50
"""
System resource setting, float, 0-100

Don't start new GPU tasks until utilization is under this threshold.
"""

gpuMemThreshold = 60
"""
System resource setting, float, 0-100

Don't start new GPU tasks until memory usage is under this threshold.
"""

resourceFreeCheckTime = 30
"""
System resource setting, int >= 1

How often to check for free resources to start new tasks in seconds.
"""
resourceAllowedRetries = 2*4*60
"""
System resource setting, int >= 1

Maximum time (in seconds) resource checking can fail before task
is set aside and another is attempted. Tasks that fail under this case
will be automatically retried.
"""

maxCpuSampleSize = 1000
"""
System resource setting, int >= 0

Use CPU by default (instead of GPU) if simulation sample size is under
this value. CPU is often faster for small arrays.
"""

pathSave = os.path.join('local', 'output')
pathData = os.path.join(pathSave, 'estimatorData')
"""
System setting, local directory

Where to store results
"""

pathCache = os.path.join('local', 'cache')
"""
System setting, local directory

Caching location
"""

pathLog = os.path.join('local', 'logs')
"""
System setting, local directory

Logging location
"""
