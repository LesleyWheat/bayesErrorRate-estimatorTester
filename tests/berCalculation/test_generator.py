import os
import pytest
# ------------------------------------------------------------------------
pathCache = os.path.join('local', 'testCache', 'berGT')
# ------------------------------------------------------------------------


def runBerList(codename, nFeatures, offsetList, varianceList, seed,
               nNew, minBER, maxBER, maxLoops, maxSamples, breakGap):
    import pandas as pd
    from app._groundTruth._build import _berGenerator
    from app._groundTruth._access import sampleBerList

    generator = _berGenerator(codename, nFeatures, seed=seed, noPrint=True,
                              pathCache=pathCache)

    # Build up the parameter-BER list
    generator.berGridCalc(offsetList, varianceList)
    generator.berIterCalc(nNew=nNew, ymin=minBER, ymax=maxBER,
                          breakGap=breakGap, maxLoops=maxLoops,
                          maxSamples=maxSamples)
    parameterBerList = generator.getBerParameters()

    df_ber = pd.DataFrame(parameterBerList)

    berDictList = sampleBerList(df_ber, maxSamples,
                                berMin=minBER, berMax=maxBER)

    df_berSampled = pd.DataFrame(berDictList)
    Y = df_berSampled['pdfMonteCarlo'].to_numpy()

    return Y


# ------------------------------------------------------------------------
# Simulation
@pytest.mark.basic
@pytest.mark.filterwarnings("ignore::UserWarning")
def test_sim_berWarn():
    import numpy as np
    from app._groundTruth._build import _berGenerator
    codename = 'gvg'
    nFeatures = 8
    seed = 1
    nNew = 3
    minBER = 0.01
    maxBER = 0.49
    maxLoops = 10
    maxSamples = 100
    breakGap = 0.01
    offsetList = np.round(np.arange(0.1, 2.2, 0.5), 3)
    varianceList = np.round(np.arange(0.4, 2.1, 0.4), 3)

    # Build up the parameter-BER list
    generator = _berGenerator(codename, nFeatures, seed=seed, noPrint=True,
                              pathCache=pathCache)
    generator.berGridCalc(offsetList, varianceList)

    # Make parameter-BER list more uniform
    with pytest.warns(RuntimeWarning):
        generator.berIterCalc(nNew=nNew, ymin=minBER, ymax=maxBER,
                              breakGap=breakGap, maxLoops=maxLoops,
                              maxSamples=maxSamples)

    parameterBerList = generator.getBerParameters()


@pytest.mark.basic
@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.filterwarnings("error::RuntimeWarning")
@pytest.mark.parametrize("nFeatures", [2, 10, 15])
def test_sim_berGood(nFeatures):
    import numpy as np
    from app._groundTruth._build import calcGap

    codename = 'gvg'
    seed = 1
    tl = 10
    nNew = 5
    minBER = 0.01
    maxBER = 0.49
    maxLoops = 200
    maxSamples = 1000
    breakGap = 0.01
    offsetList = np.round(np.arange(0.01, 4, 0.5), 3)
    varianceList = np.round(np.arange(1.0, 2.6, 0.4), 3)

    Y = runBerList(codename, nFeatures, offsetList, varianceList, seed,
                   nNew, minBER, maxBER, maxLoops, maxSamples, breakGap)

    maxGap = calcGap(Y, minBER, maxBER)

    print(maxGap)

    assert maxGap < 0.01

# ------------------------------------------------------------------------
# Test iterative calculation


@pytest.mark.filterwarnings("ignore::UserWarning")
@pytest.mark.parametrize("codename, nFeatures, maxBER", [('tvt', 10, 0.49)])
def test_berGenWarn(codename, nFeatures, maxBER):
    # Should produce warnings
    import numpy as np
    from app._groundTruth._build import _berGenerator

    maxLoops = 1
    maxSamples = 100
    seed = 2
    breakGap = 0.01
    offsetList = np.round(np.arange(0.1, 2.2, 0.6), 3).tolist()
    varianceList = np.round(np.arange(0.4, 2.1, 0.5), 3).tolist()
    nNew = 3
    minBER = 0.001

    with pytest.warns(RuntimeWarning):
        generator = _berGenerator(codename, nFeatures, seed=seed,
                                  pathCache=pathCache)
        generator.berGridCalc(offsetList, varianceList)

        generator.berIterCalc(nNew=nNew, ymin=minBER, ymax=maxBER,
                              breakGap=breakGap, maxLoops=maxLoops,
                              maxSamples=maxSamples)

        parameterBerList = generator.getBerParameters()

    assert not (parameterBerList is None)


@pytest.mark.slow
@pytest.mark.filterwarnings("error")
@pytest.mark.parametrize("codename, nFeatures, maxBER", [('tvt', 10, 0.49)])
def test_berGenGood(codename, nFeatures, maxBER):
    # Don't want a warning on this one
    # Make it an easy solve
    import numpy as np
    from app._groundTruth._build import _berGenerator

    seed = 2
    offsetList = np.round(np.arange(0.1, 2.2, 0.6), 3).tolist()
    varianceList = np.round(np.arange(0.4, 2.1, 0.5), 3).tolist()
    breakGap = 0.01
    nNew = 3
    minBER = 0.1
    maxLoops = 100
    maxSamples = 1000

    generator = _berGenerator(codename, nFeatures, seed=seed,
                              pathCache=pathCache)
    generator.berGridCalc(offsetList, varianceList)
    generator.berIterCalc(nNew=nNew, ymin=minBER, ymax=maxBER,
                          breakGap=breakGap, maxLoops=maxLoops,
                          maxSamples=maxSamples)
    parameterBerList = generator.getBerParameters()

    assert not (parameterBerList is None)


