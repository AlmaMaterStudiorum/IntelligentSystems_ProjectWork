import json
import copy 
import os
import metricsfile as mf

structConst={
    "modelName": "", 
    "functionName":"", 
    "benckmark":"",
    "boundaries": "",

    "status": "",
    "normalizedvalue": "",
    "denormalizedvalue": "", 
    "closed": "",
    "startEvaluateDate": "", 
    "endEvaluateDate": "",
    "evaluateTime": ""
}
fileNameExtension = 'copti.metrics.json'

def getOperationName(fileName,functionName,benckmark):
    return f'{fileName}.{functionName}.{benckmark}.{fileNameExtension}'

def save(fileName,functionName,benckmark,content):
    realName = getOperationName(fileName,functionName,benckmark)
    mf.save(realName,content)

def getString(modelName,functionName,benckmark,boundaries,status,normalizedvalue,denormalizedvalue,closed,startEvaluateDate,endEvaluateDate,evaluateTime):
    content =copy.deepcopy(structConst)
    content["modelName"]=modelName
    content["functionName"]=functionName
    content["benckmark"]=benckmark
    content["boundaries"]=boundaries
    content["status"]=status
    content["normalizedvalue"]=normalizedvalue
    content["denormalizedvalue"]=denormalizedvalue
    content["closed"]=closed
    content["startEvaluateDate"]=startEvaluateDate
    content["endEvaluateDate"]=endEvaluateDate
    content["evaluateTime"]=evaluateTime
    return content



