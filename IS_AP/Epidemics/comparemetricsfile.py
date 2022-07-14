import metricsfile as mf


def appendRow(fullPath,row):
    # Open a file with access mode 'a'
    file_object = open(fullPath, 'a+')
    # Append 'hello' at the end of file
    file_object.write(f'{row}\n')
    # Close the file
    file_object.close()

def getRow(modelName,functionName,accuracy,binaccuracy,rmse,mae,r2,referenceTrainingTime,targetTrainigTime,trainingRatio,referenceExecutionTime,targetExecutionTime,executionRatio,referenceSize,targetSize,sizeRatio):
    return f'{modelName};{functionName};{accuracy};{binaccuracy};{rmse};{mae};{r2};{referenceTrainingTime};{targetTrainigTime};{trainingRatio};{referenceExecutionTime};{targetExecutionTime};{executionRatio};{referenceSize};{targetSize};{sizeRatio}'
   
    pass

def create(fileName,referenceTrainingModelFilename,referenceEvaluateModelFilename,listOfTargetTrainingModels,listOfTargetEvaluateModels):
    
    # write Header
    header = 'modelName;functionName;accuracy;binaccuracy;rmse;mae;r2;referenceTrainingTime;targetTrainigTime;trainingRatio;referenceExecutionTime;targetExecutionTime;executionRatio;referenceSize;targetSize;sizeRatio'
    
    appendRow(header)
       
    # load referenceTrainigModel
    # load referenceExecutionModel    
    dataT = mf.read(referenceTrainingModelFilename)
    dataE = mf.read(referenceEvaluateModelFilename)
    
    modelName = dataT['modelName']
    functionName = dataT['functionName']
    accuracy = dataE['accuracy']
    binaccuracy = dataE['binaccuracy']
    rmse = dataE['rmse']
    mae = dataE['mae']
    r2 = dataE['r2']
    referenceTrainingTime = dataT['trainingTime']
    targetTrainigTime=0
    trainingRatio=0
    referenceExecutionTime=dataE['executionTime']
    targetExecutionTime=0
    executionRatio=0
    referenceSize=dataT['size']
    targetSize = 0
    sizeRatio=0
            
    # write referenceModel
    row = getRow(modelName,functionName,accuracy,binaccuracy,rmse,mae,r2,referenceTrainingTime,targetTrainigTime,trainingRatio,referenceExecutionTime,targetExecutionTime,executionRatio,referenceSize,targetSize,sizeRatio)
    appendRow(row)
    
    # for each targetModel 
    #   load targetTrainingModel
    #   load targetExecutionModel
    #   calculate ratio
    #   write targetModel data + ratio

    for tt,te in zip(listOfTargetTrainingModels,listOfTargetEvaluateModels):
        modelName               = tt['modelName']
        functionName            = tt['functionName']
        accuracy                = te['accuracy']
        binaccuracy             = te['binaccuracy']
        rmse                    = te['rmse']
        mae                     = te['mae']
        r2                      = te['r2']
        referenceTrainingTime   = dataT['trainingTime']
        targetTrainigTime       = tt['trainigTime']
        trainingRatio           = targetTrainigTime/referenceTrainingTime
        referenceExecutionTime  = dataE['executionTime']
        targetExecutionTime     = te['executionTime']
        executionRatio          = targetExecutionTime/referenceExecutionTime
        referenceSize           = dataT['size']
        targetSize              = tt['size']
        sizeRatio               = targetSize/referenceSize
        row = getRow(modelName,functionName,accuracy,binaccuracy,rmse,mae,r2,referenceTrainingTime,targetTrainigTime,trainingRatio,referenceExecutionTime,targetExecutionTime,executionRatio,referenceSize,targetSize,sizeRatio)
        appendRow(row)
    
    pass




