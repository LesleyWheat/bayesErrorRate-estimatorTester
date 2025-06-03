'''

Installs packages

'''
# Imports
import sys
import subprocess

# Project
import config as param

# --------------------------------------------------------------------
minTestedVersion = (3, 10, 2)
reqTestedVersion = (3, 11, 7)
# --------------------------------------------------------------------
# Functions

def _installFromFile(filename):
        call = 'pip' + ' ' +'install'+' '+ '-r'+' '+str(filename)
        theproc = subprocess.Popen(call, shell=True, text=True)
        theproc.communicate()

def _checkPlatform():
        if sys.platform.startswith('win'):
               if param.useGPU:
                       print(('GPU on windows is not supported. '
                              'Check settings in config.py.'))
                       exit(1)
        elif sys.platform.startswith('linux'):
                pass
        else:
                raise Exception('Unsupported platform: '+str(sys.platform))
        
def _checkVersion():
        pythonVersion = sys.version_info[:3]

        if pythonVersion > reqTestedVersion:
                print(('Notice: This program was tested with '
                       f'Python {minTestedVersion} and you are using: '
                       f'{sys.version_info}. Python {reqTestedVersion} is '
                       'recommended. Packages will attempt to install'))
        elif pythonVersion < minTestedVersion:
                print(('Error: This program was tested with Python '
                       f'{minTestedVersion} and you are using: '
                       f'{sys.version_info}. Using Python {reqTestedVersion} '
                       f'or later is recommended. Best to upgrade Python.'
                        'Packages were not installed.'))
                exit(1)

# --------------------------------------------------------------------
# Main
if __name__ == '__main__':
        _checkPlatform()
        _checkVersion()

        if param.useGPU:
                _installFromFile("requirements-gpu.txt")
        else:
                _installFromFile("requirements-cpu.txt")

        # Install rest of pacakges
        _installFromFile("requirements.txt")