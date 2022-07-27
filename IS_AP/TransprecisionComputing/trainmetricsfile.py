import json
import copy 
import os
import metricsfile as mf
import pathfiles as pf


structConst={
    "modelName": "", 
    "functionName":"", 
    "startDate" : "", 
    "endDate"   : "",
    "elapsed"   : "",
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

def getString(modelName,functionName,startDate,endDate,elapsed,size):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["startDate"]=startDate
    content["endDate"]=endDate
    content["elapsed"]=elapsed
    content["size"]=size
    return content






