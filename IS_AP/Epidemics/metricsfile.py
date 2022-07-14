import json
import copy 
import os

def save(fileName,content):
    with open(fileName, 'w') as outfile:
        json.dump(content, outfile, indent=4)

def read(fileName):
    with open(fileName) as json_file:
        data = json.load(json_file)
    return data

def getItem(fileName,itemName):
      data = read(fileName)
      return data[itemName]

def setItem(fileName,itemName,value):
      data = read(fileName)
      data[itemName]= value
      save(fileName,data)

def delete(fileName):
    if os.path.exists(fileName):
        os.remove(fileName)




