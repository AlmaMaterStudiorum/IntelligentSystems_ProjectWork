import pandas as pd
import support as sup

def loadInputData(name):
    fullPath = sup.GetInputDataFileFullPath(name)
    data_import = pd.read_csv(fullPath,skiprows = 1)    # Read pandas DataFrame from CSV
    print(data_import)                                  # Print imported pandas DataFrame




