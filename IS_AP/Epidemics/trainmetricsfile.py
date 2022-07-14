import json
import copy 
import os
import metricsfile as mf


structConst={
    "modelName": "", 
    "functionName":"", 
    "startTrainingDate": "", 
    "endTrainingDate": "",
    "trainingTime": "",
    "size" : ""
}

fileNameExtension = 'train.metrics.json'

def getOperationName(fileName):
    return f'{fileName}.{fileNameExtension}'

def save(fileName,content):
    realName = getOperationName(fileName)
    mf.save(realName,content)

def getString(modelName,functionName,startTrainingDate,endTrainingDate,trainingTime,size):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["startTrainingDate"]=startTrainingDate
    content["endTrainingDate"]=endTrainingDate
    content["trainingTime"]=trainingTime
    content["size"]=size
    return content






