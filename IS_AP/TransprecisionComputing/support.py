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


import evaluatemetricsfile as emf
import trainmetricsfile as tmf

overwriteModel = True
splittingData = 0.8
epochConfig = 100
batch_sizeConfig = 32

PRUNING = 'pruning'


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

def GetDataFileFullPath(name):
    return os.path.join(se.datafolder, name)

def GetInputDataFileFullPath(name):
    return os.path.join(se.dataInputFolder, name)

def GetOutputDataFileFullPath(name):

    if not os.path.exists(se.dataOutputFolder):
        os.makedirs(se.dataOutputFolder)
    path = Path(se.dataOutputFolder)

    return os.path.join(path, name)

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



def BuildTrainPrintSaveModelNeuralNetwork(hidden,name, size_in ,size_out, tr_in,tr_out,ts_in,ts_out):
    functionName=inspect.stack()[0][3]
    BeginMethod('{functionName} : {name}'.format(functionName=functionName ,name=name))
    nameHF='{name}.{extension}'.format(name=name,extension='h5')
    if (not exists( GetOutputDataFileFullPath(nameHF)) or (overwriteModel == True)):    
        modelNeuralNework = util.build_ml_model(input_size=size_in, output_size=size_out, hidden=hidden, name='MLP')

        starttime = datetime.datetime.now()
        historyNeuralNework,accuracyNeuralNetwork = util.train_ml_model2(modelNeuralNework, tr_in, tr_out, ts_in, ts_out, verbose=0, epochs=epochConfig)
        endtime = datetime.datetime.now()
        elapsed = int((endtime - starttime).total_seconds() * 1000)

        #util.plot_training_history(historyNeuralNework, figsize=figsize)
        #util.print_ml_metrics(modelNeuralNework, tr_in, tr_out, 'training')
        #util.print_ml_metrics(modelNeuralNework, ts_in, ts_out, 'test')
        fullpathh5,fullpathjson,fullPath = util.save_ml_model_with_winfolder(se.dataOutputFolder ,modelNeuralNework, name)
        size = os.path.getsize(fullpathh5)
        row=tmf.getString(name,"reference",starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed,size)
        realName = f'{name}_reference'
        tmf.save(GetOutputDataFileFullPath(realName),row)

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
        row=emf.getString(name,"reference",[accuracyNeuralNetwork[0],accuracyNeuralNetwork[1]],[0],[0,0],rmse,mae,r2,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
        emf.save(GetOutputDataFileFullPath(realName),row)

    else:
        print('No execution')
    
    EndMethod('{functionName} : {name}'.format(functionName=functionName ,name=name))

def GetTheData(benchmark):

    IsNormalize = True

    fileNameg100 = f'{benchmark}_g100.csv'
    fileNamepc = f'{benchmark}_pc.csv'
    fileNamevm = f'{benchmark}_vm.csv'

    dfg100 = pd.read_csv(GetInputDataFileFullPath(fileNameg100))
    dfg100['g100'] = 1
    dfg100['pc'] = 0
    dfg100['vm'] = 0
    dfpc = pd.read_csv(GetInputDataFileFullPath(fileNamepc))
    dfpc['g100'] = 0
    dfpc['pc'] = 1
    dfpc['vm'] = 0
    dfvm = pd.read_csv(GetInputDataFileFullPath(fileNamevm))
    dfvm['g100'] = 0
    dfvm['pc'] = 0
    dfvm['vm'] = 1

    dfAll = pd.DataFrame()


    df = dfAll.append([dfg100, dfpc,dfvm])

    if IsNormalize == True :
        # https://www.geeksforgeeks.org/normalize-a-column-in-pandas/#:~:text=Using%20The%20min%2Dmax%20feature,max()%20methods.
        dfNormalize = (df - np.min(df)) / (np.max(df) - np.min(df))

        print(dfNormalize)

        # Creating a dataframe with 70% values of original dataframe
        tr = dfNormalize.sample(frac = splittingData)

        # Creating dataframe with rest of the 30% values
        ts = dfNormalize.drop(tr.index)
    else:
        # Creating a dataframe with 70% values of original dataframe
        tr = df.sample(frac = splittingData)

        # Creating dataframe with rest of the 30% values
        ts = df.drop(tr.index)




    if benchmark == 'Convolution' :
        # var_0	var_1	var_2	var_3	error	time	memory_mean	memory_peak
        tr_in = pd.DataFrame(tr,columns = ['var_0', 'var_1', 'var_2', 'var_3' ,'g100','pc','vm'])
        tr_out = pd.DataFrame(tr,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        ts_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'var_3' ,'g100','pc','vm'])
        ts_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        
    else :
        # var_0	var_1	var_2	var_3	var_4	var_5	var_6	error	time	memory_mean	memory_peak
        tr_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'var_3','var_4', 'var_5', 'var_6' ,'g100','pc','vm'])
        tr_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])

        ts_in = pd.DataFrame(ts,columns = ['var_0', 'var_1', 'var_2', 'var_3','var_4', 'var_5', 'var_6' ,'g100','pc','vm'])
        ts_out = pd.DataFrame(ts,columns = ['error', 'time', 'memory_mean', 'memory_peak' ])



    return tr_in,tr_out,ts_in,ts_out


# TensorFlow Model Optimization Toolkit - TMOT    
# https://www.tensorflow.org/model_optimization/guide/pruning/pruning_with_keras
def PruningNeuralNetwork(name,tr_in,tr_out,ts_in,ts_out):
    functionName=inspect.stack()[0][3]
    BeginMethod('{functionName} : {name}'.format(functionName=functionName ,name=name))


    numrows = tr_in.shape[0]
    end_stepConfig = np.ceil(numrows / batch_sizeConfig).astype(np.int32) * epochConfig
    # Compress
    
    # BASELINE
    model = util.load_ml_model_with_winfolder(se.dataOutputFolder,name)

    model.summary()

    # PRUNED MODEL
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                            initial_sparsity=0.2, 
                            final_sparsity=0.9,
                            begin_step=0, 
                            end_step=end_stepConfig)

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    model_for_pruning.compile(optimizer='Adam',loss="mse",metrics=['accuracy'])

    model_for_pruning.summary()
    
    logdir = tempfile.mkdtemp()

    cb = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
      callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    ]
  
    # train metrics
    starttime = datetime.datetime.now()
    model_for_pruning.fit(tr_in, tr_out, batch_size=batch_sizeConfig, epochs=epochConfig, validation_split=0.2,callbacks=cb)
    endtime = datetime.datetime.now()
    elapsed = int((endtime - starttime).total_seconds() * 1000)


    print("Print weight Model")
    for layer in model_for_pruning.layers:
      print(layer.get_weights()[0]) # weights
      print(layer.get_weights()[1]) # biases
    print("###################")

    realName = f'{name}_{PRUNING}'
    model_for_export = tfmot.sparsity.keras.strip_pruning(model_for_pruning)
    fullpathh5 = GetOutputDataFileFullPath(f'{realName}.h5')
    tf.keras.models.save_model(model_for_export, fullpathh5, include_optimizer=False)

    size = os.path.getsize(fullpathh5)
    row=tmf.getString(name,f'{PRUNING}',starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed,size)
    tmf.save(GetOutputDataFileFullPath(realName),row)
    
     
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
    row=emf.getString(name,f'{PRUNING}',[accuracy[0],accuracy[1]],[0],[0,0],rmse,mae,r2,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    emf.save(GetOutputDataFileFullPath(realName),row)
    
    EndMethod('{functionName} : {name}'.format(functionName=functionName ,name=name))

def ExecuteCombinatorialOptimizationConvolution(keras_model):
    functionName=inspect.stack()[0][3]
    BeginMethod('{functionName} '.format(functionName=functionName))
    timeMax = 0.9
    errorMax = 0.1
    tlim=100

    slv = pywraplp.Solver.CreateSolver('CBC')
    # Define the variables
    X = {}
    """
    for r in range(rows):
        # Build the Convolution variables
        X['var_0', r] = slv.NumVar(0, 1, f'var_0_{r}')
        X['var_1', r] = slv.NumVar(0, 1, f'var_1_{r}')
        X['var_2', r] = slv.NumVar(0, 1, f'var_2_{r}')
        X['var_3', r] = slv.NumVar(0, 1, f'var_3_{r}')
        X['g100', r]  = slv.IntVar(0, 1, f'g100_{r}')
        X['pc', r]    = slv.IntVar(0, 1, f'pc_{r}')
        X['vm', r]    = slv.IntVar(0, 1, f'vm_{r}')
        X['error', r]  = slv.NumVar(0, 1, f'error_{r}')
        X['time', r]    = slv.NumVar(0, 1, f'time_{r}')
        X['memory_mean', r]    = slv.NumVar(0, 1, f'memory_mean_{r}')
        X['memory_peak', r]    = slv.NumVar(0, 1, f'memory_peak{r}')
        X['bitsum',r] = X['var_0', r] + X['var_1', r] + X['var_2', r] + X['var_3', r] 
        # Constrains
        slv.Add(X['time', r] <= timeMax)
        slv.Add(X['error', r] <= errorMax)

    # slv.Minimize(X['bitsum'])
    """
    X['var_0'] = slv.NumVar(0, 1, f'var_0')
    X['var_1'] = slv.NumVar(0, 1, f'var_1')
    X['var_2'] = slv.NumVar(0, 1, f'var_2')
    X['var_3'] = slv.NumVar(0, 1, f'var_3')
    X['g100']  = slv.IntVar(0, 1, f'g100')
    X['pc']    = slv.IntVar(0, 1, f'pc')
    X['vm']    = slv.IntVar(0, 1, f'vm')
    X['error']  = slv.NumVar(0, 1, f'error')
    X['time']    = slv.NumVar(0, 1, f'time')
    X['memory_mean']    = slv.NumVar(0, 1, f'memory_mean')
    X['memory_peak']    = slv.NumVar(0, 1, f'memory_peak')
    X['bitsum'] = X['var_0'] + X['var_1'] + X['var_2'] + X['var_3'] 
    # Constrains
    slv.Add(X['time'] <= timeMax)
    slv.Add(X['error'] <= errorMax)
    slv.Add(X['g100'] + X['pc'] + X['vm']  == 1)

    slv.Minimize(X['bitsum'])

    # Build a backend object
    bkd = ortools_backend.OrtoolsBackend()
    # Convert the keras model in internal format
    nn = keras_reader.read_keras_sequential(keras_model)
    # Set bounds
    nn.layer(0).update_lb(np.zeros(7))
    nn.layer(0).update_ub(np.ones(7))
    # Propagate bounds
    ibr_bounds(nn)
    # Build the encodings

    vin = [X['var_0'], X['var_1'], X['var_2'], X['var_3'], X['g100'] ,X['pc'], X['vm'] ]
    vout = [X['error'], X['time'], X['memory_mean'] , X['memory_peak'] ]
    util.encode(bkd, nn, slv, vin, vout, f'nn')


        # Set a time limit
    if tlim is not None:
        slv.SetTimeLimit(tlim * 1000)
    # Solve the problem
    status = slv.Solve()
    # Return the result
    res = None
    closed = False
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.FEASIBLE):
        res = {}
        for k, x in X.items():
            res[k] = x.solution_value()
    if status in (pywraplp.Solver.OPTIMAL, pywraplp.Solver.INFEASIBLE):
        closed = True


    print(res)
    print(f'Problem closed: {closed}' )
    EndMethod('{functionName} '.format(functionName=functionName ))
    return res, closed