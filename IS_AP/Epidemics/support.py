import numpy as np
from sklearn import metrics
import os
import globaldata as gd
import specificenv as se

def get_metrics(x, y):

    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, x))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, x)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, x)
    return r2, mae, rmse

def printMetrics(x, y):
    r2, mae, rmse = get_metrics(x,y)
    printMetrics2(r2, mae, rmse)
    
def getStringMetrics(x, y):
    r2, mae, rmse = get_metrics(x,y)
    return getStringMetrics2(r2, mae, rmse)
    
def getStringMetrics2(r2, mae, rmse):
    return f'RMSE: {rmse} , MAE:{mae} , R2:{r2}'
    
def printMetrics2(r2, mae, rmse):
    print(getStringMetrics2(r2, mae, rmse))
    
def GetDataFileFullPath(name):
    return os.path.join(se.datafolder, name)
    

def saveTFLiteFile(fileObj,name,functionName):
    fullPath = GetDataFileFullPath('{name}_{functionName}.tflite'.format(name=name,functionName=functionName))

    # Open file in binary write mode
    binary_file = open(fullPath, "wb")
  
    # Write bytes to file
    binary_file.write(fileObj)
  
    # Close file
    binary_file.close()
    
def deleteMetrics():
    fullPath = GetDataFileFullPath(gd.metricsFile)
    if os.path.exists(fullPath):
        os.remove(fullPath)
        print("The file has been deleted successfully")
    else:
        print("The file does not exist!")
       
    
def appendMetrics(message):
    # Open a file with access mode 'a'
    file_object = open(GetDataFileFullPath(gd.metricsFile), 'a+')
    # Append 'hello' at the end of file
    file_object.write(f'{message}\n')
    # Close the file
    file_object.close()



