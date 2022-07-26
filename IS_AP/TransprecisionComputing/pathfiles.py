import os
import specificenv as se
from pathlib import Path

def GetDataFileFullPath(name):
    return os.path.join(se.datafolder, name)

def GetInputDataFileFullPath(name):
    return os.path.join(se.dataInputFolder, name)

def GetOutputDataFileFullPath(name):
    if not os.path.exists(se.dataOutputFolder):
        os.makedirs(se.dataOutputFolder)
    path = Path(se.dataOutputFolder)
    return os.path.join(path, name)


