import os
import specificenv as se

def GetInputDataFileFullPath(name):
    return os.path.join(se.dataInputFolder, name)



