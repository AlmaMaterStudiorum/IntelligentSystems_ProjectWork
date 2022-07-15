import json
import copy 
import os
import metricsfile as mf

structConst={
    "modelName": "", 
    "functionName":"", 
    "loss": 0,
    "accuracy": "",
    "k_binaccuracy": "",
    "intaccuracy": "", 
    "intbinaccuracy": "",
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

def getString(modelName,functionName,mlaccuracy,kerasaccuracy,intaccuracy,rmse,mae,r2,startEvaluateDate,endEvaluateDate,evaluateTime):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["loss"]=mlaccuracy[0]
    content["accuracy"]=mlaccuracy[1]
    content["k_binaccuracy"]=kerasaccuracy[0]
    content["intaccuracy"]=intaccuracy[0]
    content["intbinaccuracy"]=intaccuracy[1]
    content["rmse"]=rmse
    content["mae"]=mae
    content["r2"]=r2
    content["startEvaluateDate"]=startEvaluateDate
    content["endEvaluateDate"]=endEvaluateDate
    content["evaluateTime"]=evaluateTime
    return content




