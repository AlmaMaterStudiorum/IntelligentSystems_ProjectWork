import json
import copy 
import os
import metricsfile as mf
import pathfiles as pf

structConstEncoder={
    "modelName" : "", 
    "netType"   : "", 
    "benchmark" : "",   
    "startDate" : "", 
    "endDate"   : "",
    "elapsed"   : ""
}

structConstSolver={
    "modelName" : "", 
    "netType"   :"", 
    "benchmark" :"",
    "boundaries": "",
    "status"    : "",
    "normalizedvalue": "",
    "denormalizedvalue": "", 
    "closed"    : "",
    "startDate" : "", 
    "endDate"   : "",
    "elapsed"   : ""
}

fileNameExtensionEncoder = 'coencoder.metrics.json'
fileNameExtensionSolver  = 'cosolver.metrics.json'

def getFileNameEncoder(modelName,netType,benchmark):
    return f'{modelName}.{netType}.{benchmark}.{fileNameExtensionEncoder}'

def getFullPathEncoder(modelName,netType,benchmark):
    fileName = getFileNameEncoder(modelName,netType,benchmark)
    fullpath = pf.GetRunDataOutputFileFullPath(fileName)
    return fullpath

def saveEncoder(modelName,netType,benchmark,content):
    fullpath = getFullPathEncoder(modelName,netType,benchmark)
    mf.save(fullpath,content)

def getStringEncoder(modelName,netType,benchmark,startDate,endDate,elapsed):
    content =copy.deepcopy(structConstSolver)
    content["modelName"]=modelName
    content["netType"]=netType
    content["benchmark"]=benchmark
    content["startDate"]=startDate
    content["endDate"]=endDate
    content["elapsed"]=elapsed
    return content


def getFileNameSolver(modelName,netType,benchmark):
    return f'{modelName}.{netType}.{benchmark}.{fileNameExtensionSolver}'


def getFullPathSolver(modelName,netType,benchmark):
    fileName = getFileNameSolver(modelName,netType,benchmark)
    fullpath = pf.GetRunDataOutputFileFullPath(fileName)
    return fullpath
    
def saveSolver(modelName,netType,benchmark,content):
    fullpath = getFullPathSolver(modelName,netType,benchmark)
    mf.save(fullpath,content)

def getStringSolver(modelName,netType,benchmark,boundaries,status,normalizedvalue,denormalizedvalue,closed,startDate,endDate,elapsed):
    content =copy.deepcopy(structConstSolver)
    content["modelName"]=modelName
    content["netType"]=netType
    content["benchmark"]=benchmark
    content["boundaries"]=boundaries
    content["status"]=status
    content["normalizedvalue"]=normalizedvalue
    content["denormalizedvalue"]=denormalizedvalue
    content["closed"]=closed
    content["startDate"]=startDate
    content["endDate"]=endDate
    content["elapsed"]=elapsed
    return content



