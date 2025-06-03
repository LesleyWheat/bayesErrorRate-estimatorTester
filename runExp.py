# Imports
import os
import logging

# Project Files
import config as param
from app import appSetup, buildSimParamList, runSimPool

# --------------------------------------------------------------------
# Simulation

# Main

if __name__ == '__main__':
    pathSave = os.path.join(os.getcwd(), param.pathSave)
    pathData = os.path.join(os.getcwd(), param.pathData)

    appSetup()

    # For each simulation type
    # Build everthing
    print('Generating ground truth...')
    expDictList = buildSimParamList()

    # Run the list of simulations once
    print('Starting simulations, look for details in local->logs.')
    print('This may take a while...')
    nMaxProc = min(param.maxParallelSim, len(expDictList))
    remainingDictList = runSimPool(expDictList, nMaxProc,
                                   {'pathSave': pathData})

    # Retry remaining dictionaries
    retries = 0
    allowedRetries = min(10, nMaxProc - 1)
    while (len(remainingDictList) > 0) & (retries < allowedRetries):
        # dictionaries left
        # Reduce number of subproc
        nMaxProc = max(nMaxProc - 1, 1)
        msg = (f'Beginning retry of {retries+1} '
               f'out of a maximum of {allowedRetries}. '
               f'Working on {len(remainingDictList)} '
               f'items of {len(expDictList)} total '
               f'with {nMaxProc} subproccesses.')
        print(msg)
        logging.info(msg)
        remainingDictList = runSimPool(remainingDictList, nMaxProc,
                                       {'pathSave': pathData})
        retries = retries+1

    print('Finished, check log files for errors.')
    print('Run displayPaper.py to generate tables and plots.')
