import json
import copy 
import os
import metricsfile as mf
import pathfiles as pf


structConst={
    "modelName": "", 
    "functionName":"", 
    "startTrainingDate": "", 
    "endTrainingDate": "",
    "trainingTime": "",
    "size" : ""
}

fileNameExtension = 'train.metrics.json'

def getFileName(modelName,netType,benchmark):
    return f'{modelName}.{netType}.{benchmark}.{fileNameExtension}'

def getFullPath(modelName,netType,benchmark):
    fileName = getFileName(modelName,netType,benchmark)
    fullpath = pf.GetOutputDataFileFullPath(fileName)
    return fullpath

def save(modelName,netType,benchmark,content):
    fullpath = getFullPath(modelName,netType,benchmark)
    mf.save(fullpath,content)

def getString(modelName,functionName,startTrainingDate,endTrainingDate,trainingTime,size):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["startTrainingDate"]=startTrainingDate
    content["endTrainingDate"]=endTrainingDate
    content["trainingTime"]=trainingTime
    content["size"]=size
    return content






