#from itertools import Predicate
import numpy as np
from sklearn import metrics
import os
import tensorflow as tf
from tensorflow.keras import models
import globaldata as gd
import specificenv as se
import datetime
import evaluatemetricsfile as emf
import trainmetricsfile as tmf


def get_metrics(expecteds, predicteds):

    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(expecteds, predicteds))
    # Compute the MAE
    mae = metrics.mean_absolute_error(expecteds, predicteds)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(expecteds, predicteds)
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

def getMLMetrics(model, X, Y):

    starttime = datetime.datetime.now()
    # Obtain the predictions
    predicteds = model.predict(X)
    elapsed = datetime.datetime.now() - starttime
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(Y, predicteds))
    # Compute the MAE
    mae = metrics.mean_absolute_error(Y, predicteds)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(Y, predicteds)

    accuracy = getIntAccuracy(Y, predicteds,Y.shape[0])

    m2 = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    m2.update_state(Y,predicteds)
    binaccuracy = m2.result().numpy() * 100

    r2, mae, rmse = get_metrics(Y,predicteds)

    executionTime = int(elapsed.total_seconds() * 1000)

    return r2, mae, rmse, accuracy,binaccuracy,executionTime

def getStringMLMetrics(model, X, y, label=None):
    # Obtain the predictions
    pred = model.predict(X)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(y, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(y, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(y, pred)
    lbl = '' if label is None else f' ({label})'
    return getStringMetrics2(r2, mae, rmse)
    #print(f'R2: {r2:.2f}, MAE: {mae:.2}, RMSE: {rmse:.2f}{lbl}')
    
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

    return fullPath
    
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

def deleteMetricsCSV(name):
    realname = getMetricsCSVName(name)
    fullPath = GetDataFileFullPath(realname)
    if os.path.exists(fullPath):
        os.remove(fullPath)
        print("The file has been deleted successfully")
    else:
        print("The file does not exist!")
           
def appendMetricsCSV(name,message):
    realname = getMetricsCSVName(name)
    # Open a file with access mode 'a'
    file_object = open(GetDataFileFullPath(realname), 'a+')
    # Append 'hello' at the end of file
    file_object.write(f'{message}\n')
    # Close the file
    file_object.close()

def appendMetricsCSVList(name,functionName,predicteds,expecteds):
    #m1 = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    #m1.update_state(predicteds,expecteds)
    #accuracy = m1.result().numpy()


    accuracy = getIntAccuracy(predicteds,expecteds,3)

    m2 = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    m2.update_state(predicteds,expecteds)
    binaccuracy = m2.result().numpy() * 100

    r2, mae, rmse = get_metrics(predicteds,expecteds)

    row = getStringMetricsCSVRow(name,functionName,accuracy,binaccuracy,rmse,mae,r2)
    appendMetricsCSV(name,row)

def getMetricsCSVName(name):
    return f'{name}_Metrics.csv'

def getStringMetricsCSVHeader():
    return 'name;functionName;accuracy;binaccuracy;rmse;mse;r2'

def getStringMetricsCSVRow(nnName,functionName,accuracy,binaccuracy,rmse,mse,r2,executionTime,sizeFile,executionTimeRatio,sizeTimeRatio):
    # Root Mean Square Error
    # Mean absolute error
    # Coefficient of Discrimination, R-Squared (R2)
    return  f'{nnName};{functionName};{accuracy:4f};{binaccuracy:4f};{rmse:.4f};{mse:.4f};{r2:.4f};{executionTime};{sizeFile};{executionTimeRatio};{sizeTimeRatio}'

def initMetricsCSV(name):
    deleteMetricsCSV(name)
    appendMetricsCSV(name,getStringMetricsCSVHeader())

def getIntAccuracy(expecteds,predictions,dim):
    ibinsum = 0
    isum = 0
    accuracy = 0
    for i, j in zip(predictions, expecteds):
        for d in range(i.shape[0]):
            if i[d] == j[d] :
                ibinsum+=1
        if ibinsum == dim :
            isum+=1
            
        ibinsum = 0
    accuracy = isum / len(predictions)

    return accuracy

def getIntBinaryAccuracy(expecteds,predictions,dim):
    ibinsum = 0
    accuracy = 0
    for i, j in zip(predictions, expecteds):
        for d in range(i.shape[0]):
            if i[d] == j[d] :
                ibinsum+=1

    accuracy = ibinsum / (len(predictions) * i.shape[0])

    return accuracy

def appendOriginalModelMetricsCSV(name,X,Y):
    folder = se.datafolder
    fullpathh5 = "{folder}\{name}.h5".format(folder = folder, name = name)
    fullpathjson = "{folder}\{name}.json".format(folder = folder, name = name)
    with open(fullpathjson) as f:
        model = models.model_from_json(f.read())
        model.load_weights(fullpathh5)

    r2, mse, rmse, accuracy,binaccuracy,executionTime = getMLMetrics(model,X,Y)

    sizeFile = os.path.getsize(fullpathh5)
    executionTimeRatio = 1
    sizeTimeRatio = 1
    row = getStringMetricsCSVRow(name,"original",accuracy,binaccuracy,rmse,mse,r2,executionTime,sizeFile,executionTimeRatio,sizeTimeRatio)
    appendMetricsCSV(name,row)


def appendCompressModelMetricsCSV(name,functionName,executionTime,sizeFile,expecteds,predicteds):
    folder = se.datafolder
    fullpathh5 = "{folder}\{name}.h5".format(folder = folder, name = name)
    fullpathjson = "{folder}\{name}.json".format(folder = folder, name = name)
    with open(fullpathjson) as f:
        model = models.model_from_json(f.read())
        model.load_weights(fullpathh5)

    accuracy = getIntAccuracy(predicteds,expecteds,3)

    m2 = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    m2.update_state(predicteds,expecteds)
    binaccuracy = m2.result().numpy() * 100

    r2, mae, rmse = get_metrics(expecteds,predicteds)

    executionTimeRatio = 1
    sizeTimeRatio = sizeFile / os.path.getsize(fullpathh5)
    row = getStringMetricsCSVRow(name,functionName,accuracy,binaccuracy,rmse,mae,r2,executionTime,sizeFile,executionTimeRatio,sizeTimeRatio)
    appendMetricsCSV(name,row)


def saveTrainingMetrics(name,functionName,fullPath):
    now=datetime.datetime.now()
    size = os.path.getsize(fullPath)
    row=tmf.getString(name,functionName,now.strftime("%Y-%m-%d %H:%M:%S"),now.strftime("%Y-%m-%d %H:%M:%S"),0,size)
    realName = f'{name}_{functionName}'
    tmf.save(GetDataFileFullPath(realName),row)

def saveEvaluationMetrics(name,functionName,expecteds,predictions):
    starttime = datetime.datetime.now()
    
    intaccuracy = getIntAccuracy(expecteds,predictions,3) * 100
    intbinaccuracy = getIntBinaryAccuracy(expecteds,predictions,3) * 100

    
    m1 = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    m1.update_state(expecteds,predictions)
    mlaccuracy = m1.result().numpy() * 100
    
    m2 = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    m2.update_state(expecteds,predictions)
    mlbinaccuracy = m2.result().numpy() * 100

    r2, mae, rmse = get_metrics(expecteds,predictions)

    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)
    row=emf.getString(name,functionName,[0,0],[float(mlbinaccuracy)],[float(intaccuracy),float(intbinaccuracy)],float(rmse),float(mae),float(r2),starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    realName = f'{name}_{functionName}'
    emf.save(GetDataFileFullPath(realName),row)