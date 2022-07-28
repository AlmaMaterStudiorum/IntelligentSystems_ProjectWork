import os
import specificenv as se
from pathlib import Path


def SetRunDataOutputFolder(runFolder):
    se.runDataOutputFolder = runFolder
    
def GetRunDataOutputFolder():
    return se.runDataOutputFolder

def GetRunDataOutputFolderFullPath():
    folder = os.path.join(se.dataOutputFolder,se.runDataOutputFolder) 
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def GetRunDataOutputFileFullPath(name):
    folder = GetRunDataOutputFolderFullPath()
    if not os.path.exists(folder):
        os.makedirs(folder)
    path = Path(folder)
    return os.path.join(path, name)
    
def GetDataFileFullPath(name):
    return os.path.join(se.datafolder, name)

def GetInputDataFileFullPath(name):
    return os.path.join(se.dataInputFolder, name)

def GetOutputDataFileFullPath(name):
    if not os.path.exists(se.dataOutputFolder):
        os.makedirs(se.dataOutputFolder)
    path = Path(se.dataOutputFolder)
    return os.path.join(path, name)


