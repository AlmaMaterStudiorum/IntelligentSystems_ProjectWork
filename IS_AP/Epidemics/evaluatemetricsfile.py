import json
import copy 
import os
import metricsfile as mf

structConst={
    "modelName": "", 
    "functionName":"", 
    "accuracy": "",
    "binaccuracy": "", 
    "rmse": "", 
    "mae": "",
    "r2": "", 
    "startEvaluateDate": "", 
    "endEvaluateDate": "",
    "evaluateTime": ""
}
fileNameExtension = 'evaluate.metrics.json'

def getOperationName(fileName):
    return f'{fileName}.{fileNameExtension}'

def save(fileName,content):
    realName = getOperationName(fileName)
    mf.save(realName,content)

def getString(modelName,functionName,accuracy,binaccuracy,rmse,mae,r2,startEvaluateDate,endEvaluateDate,evaluateTime):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["accuracy"]=accuracy
    content["binaccuracy"]=binaccuracy
    content["rmse"]=rmse
    content["mae"]=mae
    content["r2"]=r2
    content["startEvaluateDate"]=startEvaluateDate
    content["endEvaluateDate"]=endEvaluateDate
    content["evaluateTime"]=evaluateTime
    return content




