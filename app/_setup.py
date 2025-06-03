# Imports
import logging
import os
import glob
import warnings
import multiprocessing

# Project
import config as param

# --------------------------------------------------------------------
# Suppress tensorflow logs to console
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'
# --------------------------------------------------------------------


def appSetup():
    """
    Runs startup functions.

    Notes
    -----
    Tensorflow logs are suppressed to show only errors.

    """

    # check files and folders
    checkAll()

    # Clear logs
    logClear()

    # Setup logger
    logSetup()

    # Inform
    logging.info('App setup complete')


def checkAll():
    """
    Checks folders and hardware.

    Notes
    -----
    Torch or tensorflow check can not be done here (makes cuda errors).

    """
    # Check folders

    # Temp and datasets
    os.makedirs(param.pathSave, exist_ok=True)
    os.makedirs(param.pathLog, exist_ok=True)
    os.makedirs(param.pathCache, exist_ok=True)

# --------------------------------------------------------------------
#  Logs


def logClear():
    """
    Clear existing logs.

    Notes
    -----
    Log files that can not be deleted produce a warning.

    """
    logPath = os.path.join(param.pathLog, '*')
    files = glob.glob(logPath)
    for f in files:
        try:
            os.remove(f)
        except Exception as e:
            warnings.warn(f'Could not remove: {f}')


def logSetup(path: str = param.pathLog):
    """
    Sets up the main logger.

    Parameters
    -----
    path: path, optional
        Uses default from config.py if not set.

    Notes
    -----
    Note that on Windows child processes will only inherit the
    level of the parent process's logger, any other customization
    of the logger will not be inherited.

    """
    # Set log location
    os.makedirs(path, exist_ok=True)
    path_logfile = os.path.join(path, 'app.log')

    logging.basicConfig(
        format='%(asctime)s %(levelname)-8s %(name)s: %(message)s',
        filename=path_logfile,
        level=logging.INFO,
        datefmt='%Y-%m-%d %H:%M:%S')
    logging.info('Logger configured by: '+str(os.getpid()))
    logging.captureWarnings(True)

    # Use multiproccess logger
    logger = multiprocessing.get_logger()

    # Set log location
    path_logfile = os.path.join(path, 'app_MT.log')

    # Add timestamp
    formatter = logging.Formatter('%(asctime)s %(levelname)-8s %(name)s: %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')

    fileHandler = logging.FileHandler(path_logfile, mode='w')

    fileHandler.setFormatter(formatter)
    logger.addHandler(fileHandler)
    logger.setLevel(logging.INFO)


def childLogSetup(childName: str, path: str = param.pathLog):
    """
    Sets up thread/subproccess loggers.

    Parameters
    -----
    path: path, optional
        Uses default from config.py if not set.

    Notes
    -----
    Sets custom format.

    """
    logSetup(path=path)

    names = childName.split("_")

    if len(names) > 1:
        topName = 'AT_'+names[0]
        subName = childName+'_'+str(os.getpid())
    else:
        topName = 'app'
        subName = childName

    # Set log location
    path_logfile = os.path.join(path, str(subName)+'.log')

    # Use multiproccess logger
    logger = multiprocessing.get_logger()
    logger.handlers = []  # remove all handlers

    # Add timestamp
    formatter = logging.Formatter('%(asctime)s %(process)d %(levelname)-8s %(name)s: %(message)s',
                                  datefmt='%Y/%m/%d %H:%M:%S')

    fileHandler = logging.FileHandler(path_logfile, mode='w')
    fileHandler.setFormatter(formatter)
    fileHandler.setLevel(logging.INFO)
    logger.addHandler(fileHandler)

    # Parent logging
    path_logfileTop = os.path.join(path, str(topName)+'.log')
    fileHandlerTop = logging.FileHandler(path_logfileTop, mode='a')
    fileHandlerTop.setFormatter(formatter)
    fileHandlerTop.setLevel(logging.WARN)
    logger.addHandler(fileHandlerTop)

    # Log PID
    logger.info(f"PID: {os.getpid()}")

    return logger
