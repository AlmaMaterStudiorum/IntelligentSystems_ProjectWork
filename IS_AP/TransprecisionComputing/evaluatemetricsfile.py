import json
import copy 
import os
import metricsfile as mf
import pathfiles as pf

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
    "startDate": "", 
    "endDate": "",
    "elapsed": ""
}
fileNameExtension = 'evaluate.metrics.json'

def getFileName(modelName,netType,benchmark):
    return f'{modelName}.{netType}.{benchmark}.{fileNameExtension}'

def getFullPath(modelName,netType,benchmark):
    fileName = getFileName(modelName,netType,benchmark)
    fullpath = pf.GetOutputDataFileFullPath(fileName)
    return fullpath

def save(modelName,netType,benchmark,content):
    fullpath = getFullPath(modelName,netType,benchmark)
    mf.save(fullpath,content)

def getString(modelName,functionName,mlaccuracy,kerasaccuracy,intaccuracy,rmse,mae,r2,startDate,endDate,elapsed):
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
    content["startDate"]=startDate
    content["endDate"]=endDate
    content["elapsed"]=elapsed
    return content




