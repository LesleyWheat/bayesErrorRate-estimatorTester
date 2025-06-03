import tensorflow as tf

# ------------------------------------------------------------------------

def checkGpu():
    nGPU = len(tf.config.list_physical_devices('GPU'))
    if nGPU < 1:
        raise Exception('Told to use GPU but none are available')
    
# ------------------------------------------------------------------------
# Generic parameters

def basicRunList(nLen = 1):
    from app._tools import makeFullDictList

    expDict = dict()
    expDict.update({'nClassSamples': 50})
    expDict.update({'seed':10})

    dictList = []

    for idx in range(nLen):
        berDict = dict()
        berDict.update({'codename': 'gvg'})
        berDict.update({'nFeatures': 2})
        berDict.update({'idxSample': idx})
        berDict.update({'idx': idx})
        dictList.append(berDict)

    return makeFullDictList(dictList, expDict)