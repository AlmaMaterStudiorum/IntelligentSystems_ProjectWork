import os
import specificenv as se
import datetime
from os.path import exists
from util import util
from sklearn import metrics
from pathlib import Path
import inspect
import pandas as pd
import numpy as np
from ortools.linear_solver import pywraplp
import numpy as np
from matplotlib import pyplot as plt
from scipy.integrate import odeint
from scipy import stats
import pandas as pd
from skopt.space import Space
from skopt.sampler import Lhs
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers
from tensorflow.keras import callbacks
from tensorflow.keras import models
from tensorflow.keras import activations
from sklearn import metrics
from eml.net.embed import encode
import eml.backend.ortool_backend as ortools_backend
from eml.net.reader import keras_reader
from eml.net.process import ibr_bounds
from ortools.linear_solver import pywraplp
# from livelossplot.inputs.tf_keras import PlotLossesCallback
import warnings
from sklearn.gaussian_process import GaussianProcessRegressor
from sklearn.gaussian_process.kernels import RBF, WhiteKernel
import time
import skopt
import tensorflow_model_optimization as tfmot
import tempfile

import metricsfile as mf
import evaluatemetricsfile as emf
import trainmetricsfile as tmf
import coptimetricsfile as cmf
import pathfiles as pf

overwriteModel = True
splittingData = 0.8
epochConfig = 100
batch_sizeConfig = 32
patienceConfig=10
learning_rateConfig=0.001


initial_sparsityConfig = 0.5
final_sparsityConfig = 0.9

PRUNING = 'pruning'
REFERENCE ='reference'


# CONFIG
# Control figure size
interactive_figures = False
if interactive_figures:
    # Normal behavior
    #
    #%matplotlib widget
    figsize=(9, 3)
else:
    # PDF export behavior
    figsize=(14, 5)


    
    

def CleanFolder(fullPath):
    for root, dirs, files in os.walk(fullPath):
        for f in files:
            os.unlink(os.path.join(root, f))
        for d in dirs:
            shutil.rmtree(os.path.join(root, d))

def CreateOutputRunDataFolder():
    folder = pf.GetRunDataOutputFolderFullPath()
    if not os.path.exists(folder):
        os.makedirs(folder)
    return folder

def PrintNow():
    now = datetime.datetime.now()
    snow = now.strftime("%Y-%m-%d %H:%M:%S")
    print ("Current date and time : {snow}".format(snow=snow))

def BeginMethod(methodName):
    global starttime
    print('------------------------------------------------------')
    print('Begin {methodName}'.format(methodName=methodName))
    PrintNow()
    starttime = datetime.datetime.now()

def EndMethod(methodName):
    print('End {methodName}'.format(methodName=methodName))
    PrintNow()
    elapsed = datetime.datetime.now() - starttime
    print('elapsed {elapsed}'.format(elapsed=int(elapsed.total_seconds() * 1000)))
    print('------------------------------------------------------')

    

def BuildTrainPrintSaveModelNeuralNetwork(name,hidden,benchmark, size_in ,size_out, tr_in,tr_out,ts_in,ts_out):
    functionName=inspect.stack()[0][3]
    netType = REFERENCE
    LogMessage = '{functionName} : {name} {netType} {benchmark}'.format(functionName=functionName ,name=name, netType=netType,benchmark=benchmark)
    BeginMethod(LogMessage)
    coreName = f'{name}.{netType}.{benchmark}'
    nameHF='{coreName}.{extension}'.format(coreName=coreName,extension='h5')
    if (not exists( pf.GetRunDataOutputFileFullPath(nameHF)) or (overwriteModel == True)):    
        modelNeuralNework = util.build_ml_model2(input_size=size_in, output_size=size_out, hidden=hidden, name=name,activation='relu')

        starttime = datetime.datetime.now()
        opt = keras.optimizers.Adam(learning_rate=learning_rateConfig)
        historyNeuralNework,accuracyNeuralNetwork = util.train_ml_model2(modelNeuralNework, tr_in, tr_out, ts_in, ts_out, 
                                                                         verbose='auto', 
                                                                         epochs=epochConfig,
                                                                         patience=patienceConfig,
                                                                         opt=opt,
                                                                         batch_size=batch_sizeConfig)
        endtime = datetime.datetime.now()
        elapsed = int((endtime - starttime).total_seconds() * 1000)
        print("###################")
        print("Print Model")
        iLayer = 0
        for layer in modelNeuralNework.layers:
          print('Layer {layer}'.format(layer=iLayer))
          iLayer =+ 1
          print("Print weight ")
          print(layer.get_weights()[0]) # weights
          print("Print biases ")
          print(layer.get_weights()[1]) # biases
        print("###################")
        print("# # # # # # # # # # # # # # # # # # #")
        print("###################")
        print("modelNeuralNework.summary()")
        print(modelNeuralNework.summary())
        print("###################")


        fullpathh5,fullpathjson,fullPath = util.save_ml_model_with_winfolder(pf.GetRunDataOutputFolderFullPath(),modelNeuralNework, coreName)
        size = os.path.getsize(fullpathh5)
        row=tmf.getString(name,hidden,netType,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed,size)
        #realMetricsName = f'{name}.{netType}.{benchmark}'
        tmf.save(name,netType,benchmark,row)
        
        starttime = datetime.datetime.now()

        # Obtain the predictions
        pred = modelNeuralNework.predict(ts_in)
        # Compute the root MSE
        rmse = np.sqrt(metrics.mean_squared_error(ts_out, pred))
        # Compute the MAE
        mae = metrics.mean_absolute_error(ts_out, pred)
        # Compute the coefficient of determination
        r2 = metrics.r2_score(ts_out, pred)

        endtime = datetime.datetime.now()
        elapsed = int((endtime - starttime).total_seconds() * 1000)
        row=emf.getString(name,netType,[accuracyNeuralNetwork[0],accuracyNeuralNetwork[1]],[0],[0,0],rmse,mae,r2,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
        emf.save(name,netType,benchmark,row)

    else:
        print('No execution')
    
    EndMethod(LogMessage)

def Normalize(value,lb,ub):
     n = (value - lb) / (ub - lb)
     return n
 
def Denormalize(value,lb,ub):
     d = (value *  (ub - lb)) + lb 
     return d

    
def GetTheData(benchmark):

    fileNameg100 = f'{benchmark}_g100.csv'
    fileNamepc = f'{benchmark}_pc.csv'
    fileNamevm = f'{benchmark}_vm.csv'

    dfg100 = pd.read_csv(pf.GetInputDataFileFullPath(fileNameg100))
    dfg100['g100'] = 1
    dfg100['pc'] = 0
    dfg100['vm'] = 0
    dfpc = pd.read_csv(pf.GetInputDataFileFullPath(fileNamepc))
    dfpc['g100'] = 0
    dfpc['pc'] = 1
    dfpc['vm'] = 0
    dfvm = pd.read_csv(pf.GetInputDataFileFullPath(fileNamevm))
    dfvm['g100'] = 0
    dfvm['pc'] = 0
    dfvm['vm'] = 1

    dfAll = pd.DataFrame()


    df = dfAll.append([dfg100, dfpc,dfvm])
    

    boundaries = {}

    if benchmark == 'Convolution' :
        # var_0	var_1	var_2	var_3	error	time	memory_mean	memory_peak
        variables = ['var_0','var_1','var_2','var_3','g100','pc','vm','error','time','memory_mean','memory_peak']

        boundaries ={   'var_0'         :  {'lb' : 0  , 'ub' : 0},
                        'var_1'         :  {'lb' : 0  , 'ub' : 0},
                        'var_2'         :  {'lb' : 0  , 'ub' : 0},                        
                        'var_3'         :  {'lb' : 0  , 'ub' : 0},                        
                        'g100'          :  {'lb' : 0  , 'ub' : 0},
                        'pc'            :  {'lb' : 0  , 'ub' : 0},
                        'vm'            :  {'lb' : 0  , 'ub' : 0},
                        'error'         :  {'lb' : 0  , 'ub' : 0},
                        'time'          :  {'lb' : 0  , 'ub' : 0},   
                        'memory_mean'   :  {'lb' : 0  , 'ub' : 0},  
                        'memory_peak'   :  {'lb' : 0  , 'ub' : 0}  
        }

        column = 'error'
        df[column] = np.where(df[column] == 0, 58, -np.log10(df[column]))
       
        for v in variables:
            boundaries[v]['lb'] = df[v].min()
            boundaries[v]['ub'] = df[v].max()

        for v in variables:
            df[v] = (df[v] -  boundaries[v]['lb']) / ( boundaries[v]['ub'] -  boundaries[v]['lb'] )

        # Creating a dataframe with 70% values of original dataframe
        tr = df.sample(frac = splittingData)

        # Creating dataframe with rest of the 30% values
        ts = df.drop(tr.index)
                   
        tr_in = pd.DataFrame(tr,columns = ['var_0', 'var_1', 'var_2', 'var_3' ,'g100','pc','vm'])
        tr_out = pd.DataFrame(tr,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        ts_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'var_3' ,'g100','pc','vm'])
        ts_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        
    elif  benchmark == 'Correlation' :
        # var_0	var_1	var_2	var_3	var_4	var_5	var_6	error	time	memory_mean	memory_peak

        variables = ['var_0','var_1','var_2','var_3','var_4','var_5','var_6','g100','pc','vm','error','time','memory_mean','memory_peak']

        boundaries ={   'var_0'         :  {'lb' : 0  , 'ub' : 0},
                        'var_1'         :  {'lb' : 0  , 'ub' : 0},
                        'var_2'         :  {'lb' : 0  , 'ub' : 0},                        
                        'var_3'         :  {'lb' : 0  , 'ub' : 0},      
                        'var_4'         :  {'lb' : 0  , 'ub' : 0},   
                        'var_5'         :  {'lb' : 0  , 'ub' : 0},   
                        'var_6'         :  {'lb' : 0  , 'ub' : 0},                                                                                                  
                        'g100'          :  {'lb' : 0  , 'ub' : 0},
                        'pc'            :  {'lb' : 0  , 'ub' : 0},
                        'vm'            :  {'lb' : 0  , 'ub' : 0},
                        'error'         :  {'lb' : 0  , 'ub' : 0},
                        'time'          :  {'lb' : 0  , 'ub' : 0},   
                        'memory_mean'   :  {'lb' : 0  , 'ub' : 0},  
                        'memory_peak'   :  {'lb' : 0  , 'ub' : 0}  
        }

        column = 'error'
        df[column] = np.where(df[column] == 0, 58, -np.log10(df[column]))
        
        for v in variables:
            boundaries[v]['lb'] = df[v].min()
            boundaries[v]['ub'] = df[v].max()


        for v in variables:
            df[v] = (df[v] -  boundaries[v]['lb']) / ( boundaries[v]['ub'] -  boundaries[v]['lb'] )

        # Creating a dataframe with 70% values of original dataframe
        tr = df.sample(frac = splittingData)

        # Creating dataframe with rest of the 30% values
        ts = df.drop(tr.index)

        
        tr_in = pd.DataFrame(tr,columns = ['var_0', 'var_1', 'var_2', 'var_3','var_4', 'var_5', 'var_6' ,'g100','pc','vm'])
        tr_out = pd.DataFrame(tr,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        ts_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'var_3','var_4', 'var_5', 'var_6' ,'g100','pc','vm'])
        ts_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

    elif  benchmark == 'Saxpy' :
        # var_0	var_1	var_2	var_3	error	time	memory_mean	memory_peak
        variables = ['var_0','var_1','var_2','g100','pc','vm','error','time','memory_mean','memory_peak']

        boundaries ={   'var_0'         :  {'lb' : 0  , 'ub' : 0},
                        'var_1'         :  {'lb' : 0  , 'ub' : 0},
                        'var_2'         :  {'lb' : 0  , 'ub' : 0},                                              
                        'g100'          :  {'lb' : 0  , 'ub' : 0},
                        'pc'            :  {'lb' : 0  , 'ub' : 0},
                        'vm'            :  {'lb' : 0  , 'ub' : 0},
                        'error'         :  {'lb' : 0  , 'ub' : 0},
                        'time'          :  {'lb' : 0  , 'ub' : 0},   
                        'memory_mean'   :  {'lb' : 0  , 'ub' : 0},  
                        'memory_peak'   :  {'lb' : 0  , 'ub' : 0}  
        }

        column = 'error'
        df[column] = np.where(df[column] == 0, 58, -np.log10(df[column]))
       
        for v in variables:
            boundaries[v]['lb'] = df[v].min()
            boundaries[v]['ub'] = df[v].max()

        for v in variables:
            df[v] = (df[v] -  boundaries[v]['lb']) / ( boundaries[v]['ub'] -  boundaries[v]['lb'] )

        # Creating a dataframe with 70% values of original dataframe
        tr = df.sample(frac = splittingData)

        # Creating dataframe with rest of the 30% values
        ts = df.drop(tr.index)
                   
        tr_in = pd.DataFrame(tr,columns = ['var_0', 'var_1', 'var_2','g100','pc','vm'])
        tr_out = pd.DataFrame(tr,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        ts_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'g100','pc','vm'])
        ts_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])
    print(boundaries)

    return tr_in,tr_out,ts_in,ts_out,boundaries


# TensorFlow Model Optimization Toolkit - TMOT    
# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
def PruningNeuralNetwork(name,topology,benchmark,tr_in,tr_out,ts_in,ts_out):
    functionName=inspect.stack()[0][3]
    netType = 'pruning'
    LogMessage = '{functionName} : {name} {netType} {benchmark}'.format(functionName=functionName ,name=name, netType=netType,benchmark=benchmark)
    BeginMethod(LogMessage)

    
    numrows = tr_in.shape[0]
    end_stepConfig = np.ceil(numrows / batch_sizeConfig).astype(np.int32) * epochConfig
    # Compress
    
    # BASELINE
    coreName = f'{name}.{netType}.{benchmark}'
    model = util.load_ml_model_with_winfolder(pf.GetRunDataOutputFolderFullPath() ,f'{name}.{REFERENCE}.{benchmark}')

    model.summary()

    # PRUNED MODEL
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                            initial_sparsity=initial_sparsityConfig, 
                            final_sparsity=final_sparsityConfig,
                            begin_step=0, 
                            end_step=end_stepConfig)

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    opt = keras.optimizers.Adam(learning_rate=learning_rateConfig)
    model_for_pruning.compile(optimizer=opt,loss="mse",metrics=['accuracy'])

    model_for_pruning.summary()
    
    logdir = tempfile.mkdtemp()

    cb = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
      callbacks.EarlyStopping(patience=patienceConfig,restore_best_weights=True)
    ]
  
    # train metrics
    starttime = datetime.datetime.now()
    model_for_pruning.fit(tr_in, tr_out, batch_size=batch_sizeConfig, epochs=epochConfig, validation_split=0.2,callbacks=cb)
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)


    print("Print weight Model")
    iLayer = 0
    for layer in model_for_pruning.layers:
      print('Layer {layer}'.format(layer=iLayer))
      iLayer =+ 1
      print(layer.get_weights()[0]) # weights
      print(layer.get_weights()[1]) # biases
    print("###################")

    nameHF='{coreName}.{extension}'.format(coreName=coreName,extension='h5')
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    fullpathh5 = pf.GetRunDataOutputFileFullPath(nameHF)
    tf.keras.models.save_model(model_for_export, fullpathh5, include_optimizer=False)

    size = os.path.getsize(fullpathh5)
    row=tmf.getString(name,topology,netType,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed,size)
    tmf.save(name,netType,benchmark,row)
    
     
    # evaluate metrics
    starttime = datetime.datetime.now()

    accuracy = model_for_pruning.evaluate(ts_in, ts_out, verbose=0)

    # Obtain the predictions
    pred = model_for_pruning.predict(ts_in)
    # Compute the root MSE
    rmse = np.sqrt(metrics.mean_squared_error(ts_out, pred))
    # Compute the MAE
    mae = metrics.mean_absolute_error(ts_out, pred)
    # Compute the coefficient of determination
    r2 = metrics.r2_score(ts_out, pred)

    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)
    row=emf.getString(name,netType,[accuracy[0],accuracy[1]],[0],[0,0],rmse,mae,r2,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    emf.save(name,netType,benchmark,row)
    
    EndMethod(LogMessage)

def ExecuteCombinatorialOptimizationConvolution(keras_model,modelName,netType,boundaries,constrains,size_in):
    functionName=inspect.stack()[0][3]
    benchmark = 'Convolution'

    LogMessage = '{functionName} : {modelName} {netType} {benchmark}'.format(functionName=functionName ,modelName=modelName, netType=netType,benchmark=benchmark)
    BeginMethod(LogMessage)
   
    
    variables = ['var_0','var_1','var_2','var_3','g100','pc','vm','error','time','memory_mean','memory_peak']

    ErrMax_log = -np.log10(constrains['error'])

    normTimeMax = Normalize(constrains['time'],boundaries['time']['lb'],boundaries['time']['ub'])
    normErrMax_log = Normalize(ErrMax_log,boundaries['error']['lb'],boundaries['error']['ub'])
    
    tlim=100
    
    slv = pywraplp.Solver.CreateSolver('CBC')
    # Define the variables
    X = {}
      
    X['var_0'] = slv.NumVar(0, 1, 'var_0')
    X['var_1'] = slv.NumVar(0, 1, 'var_1')
    X['var_2'] = slv.NumVar(0, 1, 'var_2')
    X['var_3'] = slv.NumVar(0, 1, 'var_3')
    X['g100']  = slv.IntVar(0, 1, 'g100')
    X['pc']    = slv.IntVar(0, 1, 'pc')
    X['vm']    = slv.IntVar(0, 1, 'vm')
    X['error']  = slv.NumVar(0, 1, 'error')
    X['time']    = slv.NumVar(0, 1, 'time')
    X['memory_mean']    = slv.NumVar(0, 1, 'memory_mean')
    X['memory_peak']    = slv.NumVar(0, 1, 'memory_peak')
    X['bitsum'] = X['var_0'] + X['var_1'] + X['var_2'] + X['var_3'] 
    # Constrains
    slv.Add(X['time'] <= normTimeMax)
    # slv.Add(X['error'] <= errorMax)
    slv.Add(X['error'] >= normErrMax_log)
    slv.Add(X['g100'] + X['pc'] + X['vm']  == 1)

    slv.Minimize(X['bitsum'])

    # Build a backend object
    bkd = ortools_backend.OrtoolsBackend()
    # Convert the keras model in internal format
    nn = keras_reader.read_keras_sequential(keras_model)
    # Set bounds
    nn.layer(0).update_lb(np.zeros(size_in))
    nn.layer(0).update_ub(np.ones(size_in))
    # Propagate bounds
    ibr_bounds(nn)
    # Build the encodings

    vin = [X['var_0'], X['var_1'], X['var_2'], X['var_3'], X['g100'] ,X['pc'], X['vm'] ]
    vout = [X['error'], X['time'], X['memory_mean'] , X['memory_peak'] ]
    starttime = datetime.datetime.now()
    # Encode in optimization problem
    util.encode(bkd, nn, slv, vin, vout, 'nn')
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)

    content  = cmf.getStringEncoder(modelName,netType,benchmark,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    cmf.saveEncoder(modelName,netType,benchmark,content)

    SolveCombinatorialOptimization(slv,X,boundaries,modelName,netType,variables,benchmark)

    EndMethod(LogMessage)    


def ExecuteCombinatorialOptimizationCorrelation(keras_model,modelName,netType,boundaries,constrains,size_in):
    functionName=inspect.stack()[0][3]
    benchmark = 'Correlation'
    LogMessage = '{functionName} : {modelName} {netType} {benchmark}'.format(functionName=functionName ,modelName=modelName, netType=netType,benchmark=benchmark)
    BeginMethod(LogMessage)

    variables = ['var_0','var_1','var_2','var_3','var_4','var_5','var_6','g100','pc','vm','error','time','memory_mean','memory_peak']

    ErrMax_log = -np.log10(constrains['error'])

    normTimeMax = Normalize(constrains['time'],boundaries['time']['lb'],boundaries['time']['ub'])
    normErrMax_log = Normalize(ErrMax_log,boundaries['error']['lb'],boundaries['error']['ub'])
    
    tlim=100
    
    slv = pywraplp.Solver.CreateSolver('CBC')
    # Define the variables
    X = {}
      
    X['var_0'] = slv.NumVar(0, 1, 'var_0')
    X['var_1'] = slv.NumVar(0, 1, 'var_1')
    X['var_2'] = slv.NumVar(0, 1, 'var_2')
    X['var_3'] = slv.NumVar(0, 1, 'var_3')
    X['var_4'] = slv.NumVar(0, 1, 'var_4')
    X['var_5'] = slv.NumVar(0, 1, 'var_5')
    X['var_6'] = slv.NumVar(0, 1, 'var_6')
    X['g100']  = slv.IntVar(0, 1, 'g100')
    X['pc']    = slv.IntVar(0, 1, 'pc')
    X['vm']    = slv.IntVar(0, 1, 'vm')
    X['error']  = slv.NumVar(0, 1, 'error')
    X['time']    = slv.NumVar(0, 1, 'time')
    X['memory_mean']    = slv.NumVar(0, 1, 'memory_mean')
    X['memory_peak']    = slv.NumVar(0, 1, 'memory_peak')
    X['bitsum'] = X['var_0'] + X['var_1'] + X['var_2'] + X['var_3'] + X['var_4'] + X['var_5'] + X['var_6']
    # Constrains
    slv.Add(X['time'] <= normTimeMax)
    # slv.Add(X['error'] <= errorMax)
    slv.Add(X['error'] >= normErrMax_log)
    slv.Add(X['g100'] + X['pc'] + X['vm']  == 1)

    slv.Minimize(X['bitsum'])

    # Build a backend object
    bkd = ortools_backend.OrtoolsBackend()
    # Convert the keras model in internal format
    nn = keras_reader.read_keras_sequential(keras_model)
    # Set bounds
    nn.layer(0).update_lb(np.zeros(size_in))
    nn.layer(0).update_ub(np.ones(size_in))
    # Propagate bounds
    ibr_bounds(nn)
    # Build the encodings

    vin = [X['var_0'], X['var_1'], X['var_2'], X['var_3'],X['var_4'], X['var_5'], X['var_6'], X['g100'] ,X['pc'], X['vm'] ]
    vout = [X['error'], X['time'], X['memory_mean'] , X['memory_peak'] ]
    starttime = datetime.datetime.now()
    # Encode in optimization problem
    util.encode(bkd, nn, slv, vin, vout, 'nn')
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)

    content  = cmf.getStringEncoder(modelName,netType,benchmark,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    cmf.saveEncoder(modelName,netType,benchmark,content)

    SolveCombinatorialOptimization(slv,X,boundaries,modelName,netType,variables,benchmark)

    EndMethod(LogMessage) 
    

def ExecuteCombinatorialOptimizationSaxpy(keras_model,modelName,netType,boundaries,constrains,size_in):
    functionName=inspect.stack()[0][3]
    benchmark = 'Saxpy'
    LogMessage = '{functionName} : {modelName} {netType} {benchmark}'.format(functionName=functionName ,modelName=modelName, netType=netType,benchmark=benchmark)
    BeginMethod(LogMessage)

    variables = ['var_0','var_1','var_2','g100','pc','vm','error','time','memory_mean','memory_peak']

    ErrMax_log = -np.log10(constrains['error'])

    normTimeMax = Normalize(constrains['time'],boundaries['time']['lb'],boundaries['time']['ub'])
    normErrMax_log = Normalize(ErrMax_log,boundaries['error']['lb'],boundaries['error']['ub'])
    
    tlim=1000
    
    slv = pywraplp.Solver.CreateSolver('CBC')
    # Define the variables
    X = {}
      
    X['var_0'] = slv.NumVar(0, 1, 'var_0')
    X['var_1'] = slv.NumVar(0, 1, 'var_1')
    X['var_2'] = slv.NumVar(0, 1, 'var_2')
    X['g100']  = slv.IntVar(0, 1, 'g100')
    X['pc']    = slv.IntVar(0, 1, 'pc')
    X['vm']    = slv.IntVar(0, 1, 'vm')
    X['error']  = slv.NumVar(0, 1, 'error')
    X['time']    = slv.NumVar(0, 1, 'time')
    X['memory_mean']    = slv.NumVar(0, 1, 'memory_mean')
    X['memory_peak']    = slv.NumVar(0, 1, 'memory_peak')
    X['bitsum'] = X['var_0'] + X['var_1'] + X['var_2'] 
    # Constrains
    slv.Add(X['time'] <= normTimeMax)
    # slv.Add(X['error'] <= errorMax)
    slv.Add(X['error'] >= normErrMax_log)
    slv.Add(X['g100'] + X['pc'] + X['vm']  == 1)

    slv.Minimize(X['bitsum'])

    # Build a backend object
    bkd = ortools_backend.OrtoolsBackend()
    # Convert the keras model in internal format
    nn = keras_reader.read_keras_sequential(keras_model)
    # Set bounds
    nn.layer(0).update_lb(np.zeros(size_in))
    nn.layer(0).update_ub(np.ones(size_in))
    # Propagate bounds
    ibr_bounds(nn)
    # Build the encodings

    vin = [X['var_0'], X['var_1'], X['var_2'], X['g100'] ,X['pc'], X['vm'] ]
    vout = [X['error'], X['time'], X['memory_mean'] , X['memory_peak'] ]

    starttime = datetime.datetime.now()
    # Encode in optimization problem
    util.encode(bkd, nn, slv, vin, vout, 'nn')
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)

    content  = cmf.getStringEncoder(modelName,netType,benchmark,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    cmf.saveEncoder(modelName,netType,benchmark,content)
  
    SolveCombinatorialOptimization(slv,X,boundaries,modelName,netType,variables,benchmark)

    EndMethod(LogMessage) 
    

def SolveCombinatorialOptimization(solver,X,boundaries,modelName,netType,variables,benchmark):

    tlim=1000
    # Set a time limit
    if tlim is not None:
        solver.SetTimeLimit(tlim * 1000)
    # Solve the problem
    starttime = datetime.datetime.now()
    status = solver.Solve()
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)
    # Return the result

    closed = False
    normalizedvalue = None
    denormalizedvalue={}

    if status == pywraplp.Solver.OPTIMAL :
        sstatus = 'OPTIMAL'
        normalizedvalue = {}
        for k, x in X.items():
            normalizedvalue[k] = x.solution_value()
        closed = True   
    elif status == pywraplp.Solver.FEASIBLE :
        sstatus = 'FEASIBLE'
        normalizedvalue = {}
        for k, x in X.items():
            normalizedvalue[k] = x.solution_value()
        closed = True
    elif status == pywraplp.Solver.INFEASIBLE :
        sstatus = 'INFEASIBLE'
        
    elif status == pywraplp.Solver.UNBOUNDED :
        sstatus = 'UNBOUNDED'
        
    elif status == pywraplp.Solver.ABNORMAL :
        sstatus = 'ABNORMAL'
        
    elif status == pywraplp.Solver.NOT_SOLVED :
        sstatus = 'NOT_SOLVED'
        
    elif status == pywraplp.Solver.UNKNOWN :
        sstatus = 'UNKNOWN'
        
    else :
        raise Exception('Unknown status')

    if normalizedvalue is not None :  
        print('Normalized values')
        print(normalizedvalue)
        

        for v in variables:
            denormalizedvalue[v] = Denormalize(normalizedvalue[v],boundaries[v]['lb'],boundaries[v]['ub'])
            
        print('Denormalized values')
        print(denormalizedvalue)     


    print(f'Problem closed: {closed} , status : {sstatus} ')

    content  = cmf.getStringSolver(modelName,netType,benchmark,boundaries,sstatus,normalizedvalue,denormalizedvalue,closed,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    cmf.saveSolver(modelName,netType,benchmark,content)

    EndMethod('{functionName} '.format(functionName=netType ))
    return normalizedvalue, closed


def CreateSummary(nns, netTypes,benchmarks):

    df = pd.DataFrame()
    row = 0
    metrics = ['topology','train','size','evaluate','accuracy','rmse','encode','solver','objective','status']
    numberofdigit = 4

    for i in range(len(metrics)) :
        df.loc[i,'metrics'] = metrics[i]
    
    for modelName in nns:
        for benchmark in benchmarks:
            for netType in netTypes:            
                    column = f'{modelName}.{netType}.{benchmark}'
                    row = 0

                    fullPath = tmf.getFullPath(modelName,netType,benchmark)

                    item = mf.getItem(fullPath,'topology')                       
                    df.loc[row,column] = " ".join(str(x) for x in item)
                    row+=1

                    item = mf.getItem(fullPath,'elapsed')               
                    df.loc[row,column] = item
                    row+=1

                    item = mf.getItem(fullPath,'size')               
                    df.loc[row,column] = item
                    row+=1

                    fullPath = emf.getFullPath(modelName,netType,benchmark)
                    item = mf.getItem(fullPath,'elapsed')                
                    df.loc[row,column] = item
                    row+=1

                    #fullPath = emf.getFullPath(modelName,netType,benchmark)
                    item = mf.getItem(fullPath,'accuracy')                
                    df.loc[row,column] = round(item, numberofdigit)
                    row+=1

                    #fullPath = emf.getFullPath(modelName,netType,benchmark)
                    item = mf.getItem(fullPath,'rmse')                
                    df.loc[row,column] = round(item, numberofdigit)
                    row+=1

                    fullPath = cmf.getFullPathEncoder(modelName,netType,benchmark)
                    item = mf.getItem(fullPath,'elapsed')               
                    df.loc[row,column] = item
                    row+=1


                    fullPath = cmf.getFullPathSolver(modelName,netType,benchmark)
                    item = mf.getItem(fullPath,'elapsed')               
                    df.loc[row,column] = item
                    row+=1

                    normalizedvalue = mf.getItem(fullPath,'normalizedvalue')     
                    if normalizedvalue != None :
                        item = normalizedvalue['bitsum']
                        df.loc[row,column] = round(item, numberofdigit)
                    row+=1
                    
                    item = mf.getItem(fullPath,'status')               
                    df.loc[row,column] = item
                    row+=1
    
         
    filePath = pf.GetRunDataOutputFileFullPath('run.summary.csv')               

    df.to_csv(filePath)  