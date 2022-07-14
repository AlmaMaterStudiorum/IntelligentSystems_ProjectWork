
#from matplotlib import pyplot as plt
from glob import glob
from lib2to3.pgen2.token import NAME
from util import util
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from skopt.space import Space
# Regression Tree
from sklearn.tree import DecisionTreeRegressor
from sklearn.tree import plot_tree
import matplotlib.pyplot as plt
import os
from os.path import exists
import datetime
import gc
import tensorflow as tf
import tensorflow_model_optimization as tfmot
from keras.models import Sequential
from keras.layers import Dense
import pydot
import graphviz
from keras.utils.vis_utils import plot_model
import inspect
import tempfile
from tensorflow.keras import callbacks
import support
import globaldata as gd
import specificenv as se
import trainmetricsfile as tmf
import evaluatemetricsfile as emf
from sklearn import metrics




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


seedConfig=42
modeConfig='max_min'
max_beta=0.4
n_tr, n_ts = 10000, 2000
gammaConfig = 1/14
tmaxConfig=7
sir_tr_in=None
sir_ts_in=None
sir_tr_out=None
sir_ts_out=None
nameRegTree='regtree'
nameNN1='nn1'
nameNN2='nn2'
nameNN3='nn3'
nameNN4='nn4'

overwriteModel=True
overwriteData=False
modelRegressionTree=[]
solutionGlobal= {
    nameRegTree : {},
    nameNN1 : {},
    nameNN2 : {},
    nameNN3 : {},
    nameNN4 : {}
}

closedGlobal= {
    nameRegTree : False,
    nameNN1 : False,
    nameNN2 : False,
    nameNN3 : False,
    nameNN4 : False
}

accuracyGlobal = {
    nameRegTree : 0,
    nameNN1 : 0,
    nameNN2 : 0,
    nameNN3 : 0,
    nameNN4 : 0
}

executionTimeGlobal = {
    nameRegTree : 0,
    nameNN1 : 0,
    nameNN2 : 0,
    nameNN3 : 0,
    nameNN4 : 0
}

sizeGlobal = {
    nameRegTree : 0,
    nameNN1 : 0,
    nameNN2 : 0,
    nameNN3 : 0,
    nameNN4 : 0
}

npis = [
    util.NPI('masks-indoor', effect=0.75, cost=1),
    util.NPI('masks-outdoor', effect=0.9, cost=1),
    util.NPI('dad', effect=0.7, cost=3),
    util.NPI('bar-rest', effect=0.6, cost=3),
    util.NPI('transport', effect=0.6, cost=4)
]
    
    
S0, I0, R0 = 0.99, 0.01, 0.00
nweeks = 3
tlim = 3600
beta_base = 0.35
budget = 20
starttime=0


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


 

def CreateInput():
    BeginMethod('generate_SIR_input')
    global sir_tr_in 
    global sir_ts_in 
    sir_tr_in = util.generate_SIR_input(max_samples=n_tr, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta)
    sir_ts_in = util.generate_SIR_input(max_samples=n_ts, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta)    
     
    if (not exists(support.GetDataFileFullPath('sir_tr_in.pkl'))) or (overwriteData == True): 
        sir_tr_in.to_pickle(support.GetDataFileFullPath('sir_tr_in.pkl'))   

    if (not exists(support.GetDataFileFullPath('sir_ts_in.pkl'))) or (overwriteData== True):         
        sir_ts_in.to_pickle(support.GetDataFileFullPath('sir_ts_in.pkl')) 
    
    sir_tr_in.head()
    
    EndMethod('generate_SIR_input')

def CreateOutput():
    BeginMethod('generate_SIR_output')
    global sir_tr_out
    global sir_ts_out
    sir_tr_out = util.generate_SIR_output(sir_tr_in, gammaConfig, tmaxConfig)
    sir_ts_out = util.generate_SIR_output(sir_ts_in, gammaConfig, tmaxConfig)

    if (not exists(support.GetDataFileFullPath('sir_tr_out.pkl'))) or (overwriteData == True): 
        sir_tr_out.to_pickle(support.GetDataFileFullPath('sir_tr_out.pkl'))  

    if (not exists(support.GetDataFileFullPath('sir_ts_out.pkl'))) or (overwriteData== True):         
        sir_ts_out.to_pickle(support.GetDataFileFullPath('sir_ts_out.pkl'))  
       
    sir_tr_out.head()
    EndMethod('generate_SIR_output')

def LoadData():
    BeginMethod(inspect.stack()[0][3])
    global sir_tr_in 
    global sir_ts_in 
    global sir_tr_out
    global sir_ts_out
    if sir_tr_in is None :
        sir_tr_in = pd.read_pickle(support.GetDataFileFullPath('sir_tr_in.pkl'))

    if sir_tr_out is None :
        sir_tr_out = pd.read_pickle(support.GetDataFileFullPath('sir_tr_out.pkl'))

    if sir_ts_in is None :
        sir_ts_in = pd.read_pickle(support.GetDataFileFullPath('sir_ts_in.pkl'))            
            
    if sir_ts_out is None :
        sir_ts_out = pd.read_pickle(support.GetDataFileFullPath('sir_ts_out.pkl'))   
    
    EndMethod(inspect.stack()[0][3])   
      
def ManageData():
    BeginMethod(inspect.stack()[0][3])
    global sir_tr_in 
    global sir_ts_in 
    global sir_tr_out
    global sir_ts_out

    # INPUT
    if (exists(support.GetDataFileFullPath('sir_tr_in.pkl'))) and (overwriteData == False):        
        sir_tr_in = pd.read_pickle(support.GetDataFileFullPath('sir_tr_in.pkl'))
    else:
        sir_tr_in = util.generate_SIR_input(max_samples=n_tr, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta)
        sir_tr_in.to_pickle(support.GetDataFileFullPath('sir_tr_in.pkl'))  
   
    if (exists(support.GetDataFileFullPath('sir_ts_in.pkl'))) and (overwriteData == False): 
        sir_ts_in = pd.read_pickle(support.GetDataFileFullPath('sir_ts_in.pkl'))
    else:
        sir_ts_in = util.generate_SIR_input(max_samples=n_ts, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta) 
        sir_ts_in.to_pickle(support.GetDataFileFullPath('sir_ts_in.pkl'))  
                
    #OUTPUT
    if (exists(support.GetDataFileFullPath('sir_tr_out.pkl'))) and (overwriteData == False): 
        sir_tr_out = pd.read_pickle(support.GetDataFileFullPath('sir_tr_out.pkl'))
    else:
        sir_tr_out = util.generate_SIR_output(sir_tr_in, gammaConfig, tmaxConfig)
        sir_tr_out.to_pickle(support.GetDataFileFullPath('sir_tr_out.pkl'))  
   
    if (exists(support.GetDataFileFullPath('sir_ts_out.pkl'))) and (overwriteData == False): 
        sir_ts_out = pd.read_pickle(support.GetDataFileFullPath('sir_ts_out.pkl'))  
    else:
        sir_ts_out = util.generate_SIR_output(sir_ts_in, gammaConfig, tmaxConfig)
        sir_ts_out.to_pickle(support.GetDataFileFullPath('sir_ts_out.pkl'))  
        
    support.deleteMetrics()    
        
    EndMethod(inspect.stack()[0][3]) 



def BuildTrainPrintModelRegressionTree():
    BeginMethod('BuildTrainPrintModelRegressionTree')
    print('Before train')
    global modelRegressionTree
    modelRegressionTree = DecisionTreeRegressor(random_state=44)

    historyRegressionTree = modelRegressionTree.fit(sir_tr_in, sir_tr_out)

    print('After train')

    predictRegressionTree=modelRegressionTree.predict(sir_ts_in[:1])
    scoreRegressionTree=modelRegressionTree.score(sir_ts_in, sir_ts_out)
    accuracyGlobal[nameRegTree] = scoreRegressionTree
    print('predict = {predict}'.format(predict = predictRegressionTree))
    print('score = {score}'.format(score = scoreRegressionTree))
    print('After train')

    # plt.figure(figsize=(10,8), dpi=150)
    # plot_tree(modelRegressionTree, feature_names=sir_tr_in.columns)
    # plt.savefig('C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\fst.svg',format='svg',bbox_inches = "tight")
    # ax = modelRegressionTree.create_tree_digraph(clf)
    # with open('C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\fst.svg', 'w') as f:
    #    f.write(ax._repr_svg_())
    
    EndMethod('BuildTrainPrintModelRegressionTree')

def BuildTrainPrintSaveModelNeuralNetwork(hidden,name): 
    BeginMethod('BuildTrainPrintSaveModelNeuralNetwork: {name}'.format(name=name))
    nameHF='{name}.{extension}'.format(name=name,extension='h5')
    if (not exists( support.GetDataFileFullPath(nameHF)) or (overwriteModel == True)):    
        modelNeuralNework = util.build_ml_model(input_size=4, output_size=3, hidden=hidden, name='MLP')


        starttime = datetime.datetime.now()
        historyNeuralNework,accuracyNeuralNetwork = util.train_ml_model2(modelNeuralNework, sir_tr_in, sir_tr_out, sir_ts_in,sir_ts_out , verbose=0, epochs=100)
        endtime = datetime.datetime.now()
        elapsed = int((endtime - starttime).total_seconds() * 1000)

        accuracyGlobal[name] = accuracyNeuralNetwork
        message = f'Baseline accuracy {name}:{accuracyNeuralNetwork}'
        print(message)
        support.appendMetrics(message)
        message = support.getStringMLMetrics(modelNeuralNework,sir_ts_in,sir_ts_out)
        print(message)
        support.appendMetrics(message)
        util.plot_training_history(historyNeuralNework, figsize=figsize)
        util.print_ml_metrics(modelNeuralNework, sir_tr_in, sir_tr_out, 'training')
        util.print_ml_metrics(modelNeuralNework, sir_ts_in, sir_ts_out, 'test')
        fullpathh5,fullpathjson,fullPath = util.save_ml_model_with_winfolder(se.datafolder,modelNeuralNework, name)
        size = os.path.getsize(fullpathh5)
        row=tmf.getString(name,"",starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed,size)
        tmf.save(support.GetDataFileFullPath(name),row)

        starttime = datetime.datetime.now()

        # Obtain the predictions
        pred = modelNeuralNework.predict(sir_ts_in)
        # Compute the root MSE
        rmse = np.sqrt(metrics.mean_squared_error(sir_ts_out, pred))
        # Compute the MAE
        mae = metrics.mean_absolute_error(sir_ts_out, pred)
        # Compute the coefficient of determination
        r2 = metrics.r2_score(sir_ts_out, pred)

        endtime = datetime.datetime.now()
        elapsed = int((endtime - starttime).total_seconds() * 1000)
        row=emf.getString(name,"",accuracyNeuralNetwork,0,rmse,mae,r2,starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
        emf.save(support.GetDataFileFullPath(name),row)

    else:
        print('No execution')
    
    EndMethod('BuildTrainPrintSaveModelNeuralNetwork: {name}'.format(name=name))
      
    
def BuildTrainPrintSaveModelNeuralNetwork1():
    BuildTrainPrintSaveModelNeuralNetwork([8,8,8,8],nameNN1)
                
def BuildTrainPrintSaveModelNeuralNetwork2():
    BuildTrainPrintSaveModelNeuralNetwork([16,16],nameNN2)

def BuildTrainPrintSaveModelNeuralNetwork3():
    BuildTrainPrintSaveModelNeuralNetwork([16,16,16,16,16,16,16,16,16,16,16,16,16,16,16,16],nameNN3)

def BuildTrainPrintSaveModelNeuralNetwork4():
    BuildTrainPrintSaveModelNeuralNetwork([1024,1024],nameNN4)
         
def ExecuteOptimizationProblem(name):
    BeginMethod('ExecuteOptimizationProblem: {name}'.format(name=name))
    gc.collect()
    if name==nameRegTree :
        model = modelRegressionTree
    else:
        model = util.load_ml_model_with_winfolder(se.datafolder,name)
    
    sol, closed = util.solve_sir_planning(model, npis, S0, I0, R0, beta_base=beta_base, budget=budget,nweeks=nweeks, tlim=tlim)
       
    cost = 0
    if closed == True:
        cost = sol['cost']
        sol_df = util.sol_to_dataframe(sol, npis, nweeks)
        print('sol_df')
        pd.set_option('display.max_columns',20)
        print(sol_df)
    else:
        cost = 0
        
    solutionGlobal[name] = sol_df
    closedGlobal[name] = closed
    gc.collect()       
    print(f'Problem closed: {closed} , Cost = {cost}')
    EndMethod('ExecuteOptimizationProblem: {name}'.format(name=name))
 
def ExecuteOptimizationProblemRegTree():
    ExecuteOptimizationProblem(nameRegTree)

def ExecuteOptimizationProblemNN1():
    ExecuteOptimizationProblem(nameNN1)

def ExecuteOptimizationProblemNN2():
    ExecuteOptimizationProblem(nameNN2)
    
def ExecuteOptimizationProblemNN3():
    ExecuteOptimizationProblem(nameNN3)

def SimulateProblemByName(name):
    if closedGlobal[name] == True:
        SimulateProblem(name,solutionGlobal[name],S0,I0,R0,gammaConfig)


def SimulateProblem(name,sol,s,i,r,g):
    BeginMethod('SimulateProblem {name}'.format(name=name))
    sim = util.simulate_SIR_NPI(s, i, r, sol.iloc[:-1]['b'], g, steps_per_day=1)
    print('sim')
    pd.set_option('display.max_columns',20)
    print(sim)
    EndMethod('SimulateProblem {name}'.format(name=name))
 
def SimulateProblemRegTree():
    SimulateProblemByName(nameRegTree)
    
def SimulateProblemNN1():
    SimulateProblemByName(nameNN1)
    
def SimulateProblemNN2():
    SimulateProblemByName(nameNN2)
    
def SimulateProblemNN3():
    SimulateProblemByName(nameNN3)

def BruteForce():
    BeginMethod(inspect.stack()[0][3])

    best_S, best_sched = util.solve_sir_brute_force(npis, S0, I0, R0, beta_base, gammaConfig, nweeks, budget)

    print('best_S')
    print(best_S)
    EndMethod(inspect.stack()[0][3])



# TensorFlow Model Optimization Toolkit - TMOT    
def TMOTCompressPredictNeuralNetwork(name):
    BeginMethod('CompressPredictNeuralNetwork {name}'.format(name=name))
    # Compress
    
    # BASELINE
    model = util.load_ml_model_with_winfolder(se.datafolder,name)

    model.summary()

    # PRUNED MODEL
    pruning_schedule = tfmot.sparsity.keras.PolynomialDecay(
                            initial_sparsity=0.0, final_sparsity=0.5,
                            begin_step=2000, end_step=4000)

    model_for_pruning = tfmot.sparsity.keras.prune_low_magnitude(model, pruning_schedule=pruning_schedule)

    model_for_pruning.compile(optimizer='Adam',loss="mse",metrics=['accuracy'])
    
    logdir = tempfile.mkdtemp()

    cb = [
      tfmot.sparsity.keras.UpdatePruningStep(),
      tfmot.sparsity.keras.PruningSummaries(log_dir=logdir),
      callbacks.EarlyStopping(patience=10,restore_best_weights=True)
    ]
  
    model_for_pruning.fit(sir_tr_in, sir_tr_out,
                      batch_size=32, epochs=20, validation_split=0.2,
                      callbacks=cb)
    model_for_pruning_accuracy = model_for_pruning.evaluate(sir_ts_in, sir_ts_out, verbose=0)
    
    
    message = f'Pruned test accuracy {name}: {model_for_pruning_accuracy}'
    print(message)
    support.appendMetrics(message)

    # Predict
    EndMethod('CompressPredictNeuralNetwork {name}'.format(name=name))


def TMOTCompressPredictNeuralNetwork1():
    TMOTCompressPredictNeuralNetwork(nameNN1)
    
def TMOTCompressPredictNeuralNetwork2():
    TMOTCompressPredictNeuralNetwork(nameNN2)
    
def TMOTCompressPredictNeuralNetwork3():
    TMOTCompressPredictNeuralNetwork(nameNN3)

def TMOTCompressPredictNeuralNetwork4():
    TMOTCompressPredictNeuralNetwork(nameNN4)


def representative_data_gen():
  for input_value in tf.data.Dataset.from_tensor_slices(sir_ts_in.astype('float32')).batch(1).take(100):
    yield [input_value]
    


#
# https://www.tensorflow.org/lite/performance/post_training_quantization
def TMOTQuantizationPostTrainDynamicRangeQuantization(name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]

    # sono bytes
    tflite_model_quant = converter.convert()

    fullPathTFLiteModel = support.saveTFLiteFile(tflite_model_quant,name,functionName)   

    # training metrics
    now=datetime.datetime.now()
    size = os.path.getsize(fullPathTFLiteModel)
    row=tmf.getString(name,"",now.strftime("%Y-%m-%d %H:%M:%S"),now.strftime("%Y-%m-%d %H:%M:%S"),0,size)
    tmf.save(support.GetDataFileFullPath(name),row)
    
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_path=fullPathTFLiteModel)
    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    #print(input_details)
    output_details = interpreter.get_output_details()[0]
    #print(output_details)
    
    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]
        
        test_in_type = np.expand_dims(test_in, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)


    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)

    #starttime = datetime.datetime.now()
    #accuracy = support.getAccuracy(expecteds,predictions,3)

    #m2 = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    #m2.update_state(expecteds,predictions)
    #binaccuracy = m2.result().numpy() * 100

    #r2, mae, rmse = support.get_metrics(expecteds,predictions)

    #endtime = datetime.datetime.now()
    #elapsed = int((endtime - starttime).total_seconds() * 1000)
    #row=emf.getString(name,functionName,accuracy,float(binaccuracy),float(rmse),float(mae),float(r2),starttime.strftime("%Y-%m-%d %H:%M:%S"),endtime.strftime("%Y-%m-%d %H:%M:%S"),elapsed)
    #compressedMetric = f'{name}_{functionName}'
    #emf.save(support.GetDataFileFullPath(compressedMetric),row)


    EndMethod(logTag)


# quantize weight , quantize activation 
# https://www.tensorflow.org/lite/performance/post_training_quantization#full_integer_quantization
def TMOTQuantizationPostTrainFullIntegerQuantizationIntegerOnly(name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen
    # Ensure that if any ops can't be quantized, the converter throws an error
    converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
    # Set the input and output tensors to uint8 (APIs added in r2.3)
    converter.inference_input_type = tf.int8  
    converter.inference_output_type = tf.int8  

    # sono bytes
    tflite_model_quant = converter.convert()
    
    fullPathTFLiteModel = support.saveTFLiteFile(tflite_model_quant,name,functionName)

    interpreter = tf.lite.Interpreter(model_path=fullPathTFLiteModel)

    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    input_scale, input_zero_point = input_details["quantization"]
    output_scale, output_zero_point = output_details["quantization"]

    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]

        # Check if the input type is quantized, then rescale input data to uint8
        if input_details['dtype'] == np.int8 :          
            test_in_s = test_in / input_scale
            test_in_s_z = test_in_s + input_zero_point

        if output_details['dtype'] == np.int8 :                      
            test_out_s = test_out / output_scale
            test_out_s_z = test_out_s + output_zero_point

        test_in_type = np.expand_dims(test_in_s_z, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out_s_z, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)

    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)

    EndMethod(logTag)
    
    
# quantize weight , float activation 
# https://www.tensorflow.org/lite/performance/post_training_quantization#integer_with_float_fallback_using_default_float_inputoutput
# In order to fully integer quantize a model, but use float operators when they don't have an integer implementation (to ensure conversion occurs smoothly)
def TMOTQuantizationPostTrainFullIntegerQuantizationIntegerWithFloatFallback (name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.representative_dataset = representative_data_gen

    # sono bytes
    tflite_model_quant = converter.convert()
    
    support.saveTFLiteFile(tflite_model_quant,name,functionName)
    
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]
   
    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]

        test_in_type = np.expand_dims(test_in, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)


    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)
    # support.appendMetricsCSVList(name,functionName,predictions,expecteds)
    #m = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_accuracy = m.result().numpy()

    #m = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_binaccuracy = m.result().numpy()
                                          
    #message = support.getStringMetrics(predictions,expecteds)
    #support.appendMetrics(logTag)
    #support.appendMetrics(message)
    #print('Quantization test accuracy:', tflite_model_quant_accuracy)
    #print('Quantization test bin accuracy:', tflite_model_quant_binaccuracy)
    EndMethod(logTag)


# 
# https://www.tensorflow.org/lite/performance/post_training_quantization#float16_quantization
# You can reduce the size of a floating point model by quantizing the weights to float16, the IEEE standard for 16-bit floating point numbers
def TMOTQuantizationPostTrainFloat16Quantization (name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_types = [tf.float16]

    # sono bytes
    tflite_model_quant = converter.convert()
    
    support.saveTFLiteFile(tflite_model_quant,name,functionName)
          
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]

        test_in_type = np.expand_dims(test_in, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)

    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)
    # support.appendMetricsCSVList(name,functionName,predictions,expecteds)
    #m = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_accuracy = m.result().numpy()

    #m = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_binaccuracy = m.result().numpy()
                                          
    #message = support.getStringMetrics(predictions,expecteds)
    #support.appendMetrics(logTag)
    #support.appendMetrics(message)
    #print('Quantization test accuracy:', tflite_model_quant_accuracy)
    #print('Quantization test bin accuracy:', tflite_model_quant_binaccuracy)
    EndMethod(logTag)


# 
# https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only_16-bit_activations_with_8-bit_weights_experimental
# This is an experimental quantization scheme. It is similar to the "integer only" scheme, but activations are quantized based on their range to 16-bits, 
# weights are quantized in 8-bit integer and bias is quantized into 64-bit integer
def TMOTQuantizationPostTrainIntegerOnly16BitActivationsWith8BitWeightsV1(name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8]

    # sono bytes
    tflite_model_quant = converter.convert()
    
    support.saveTFLiteFile(tflite_model_quant,name,functionName)
          
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]

        test_in_type = np.expand_dims(test_in, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)


    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)
    # support.appendMetricsCSVList(name,functionName,predictions,expecteds)
    #m = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_accuracy = m.result().numpy()

    #m = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_binaccuracy = m.result().numpy()
                                          
    #message = support.getStringMetrics(predictions,expecteds)
    #support.appendMetrics(logTag)
    #support.appendMetrics(message)
    #print('Quantization test accuracy:', tflite_model_quant_accuracy)
    #print('Quantization test bin accuracy:', tflite_model_quant_binaccuracy)
    EndMethod(logTag)


# 
# https://www.tensorflow.org/lite/performance/post_training_quantization#integer_only_16-bit_activations_with_8-bit_weights_experimental
# This is an experimental quantization scheme. It is similar to the "integer only" scheme, but activations are quantized based on their range to 16-bits, 
# weights are quantized in 8-bit integer and bias is quantized into 64-bit integer.
# If 16x8 quantization is not supported for some operators in the model, then the model still can be quantized, but unsupported operators kept in float
def TMOTQuantizationPostTrainIntegerOnly16BitActivationsWith8BitWeightsV2(name):
    functionName=inspect.stack()[0][3]
    logTag = '{functionName} {name}'.format(name=name,functionName=functionName)
    BeginMethod(logTag)
    model = util.load_ml_model_with_winfolder(se.datafolder,name)
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    converter.optimizations = [tf.lite.Optimize.DEFAULT]
    converter.target_spec.supported_ops = [tf.lite.OpsSet.EXPERIMENTAL_TFLITE_BUILTINS_ACTIVATIONS_INT16_WEIGHTS_INT8,tf.lite.OpsSet.TFLITE_BUILTINS]

    # sono bytes
    tflite_model_quant = converter.convert()
    
    support.saveTFLiteFile(tflite_model_quant,name,functionName)
          
    # Initialize the interpreter
    interpreter = tf.lite.Interpreter(model_content=tflite_model_quant)
    interpreter.allocate_tensors()

    test_indices = range(sir_ts_in.shape[0])
    
    input_details = interpreter.get_input_details()[0]
    output_details = interpreter.get_output_details()[0]

    predictions = []
    expecteds = []
    predictionsum=0
    predictionBinarysum=0
    for test_index in range(len(test_indices)):
        test_in  = sir_ts_in.iloc[test_index]
        test_out = sir_ts_out.iloc[test_index]

        test_in_type = np.expand_dims(test_in, axis=0).astype(input_details["dtype"])
        test_out_type = np.expand_dims(test_out, axis=0).astype(output_details["dtype"])[0]
        interpreter.set_tensor(input_details["index"], test_in_type)
        interpreter.invoke()
        output = interpreter.get_tensor(output_details["index"])[0]
        
        predictions.append(output)
        expecteds.append(test_out_type)


    support.saveEvaluationMetrics(name,functionName,expecteds,predictions)
    # support.appendMetricsCSVList(name,functionName,predictions,expecteds)
    #m = tf.keras.metrics.Accuracy(name="accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_accuracy = m.result().numpy()

    #m = tf.keras.metrics.Accuracy(name="binary_accuracy", dtype=None)
    #m.update_state(predictions,expecteds)
    #tflite_model_quant_binaccuracy = m.result().numpy()
                                          
    #message = support.getStringMetrics(predictions,expecteds)
    #support.appendMetrics(logTag)
    #support.appendMetrics(message)
    #print('Quantization test accuracy:', tflite_model_quant_accuracy)
    #print('Quantization test bin accuracy:', tflite_model_quant_binaccuracy)
    EndMethod(logTag)
 

def TMOTQuantizationPostTrain1():    
    TMOTQuantizationPostTrain(nameNN1)

def TMOTQuantizationPostTrain2():    
    TMOTQuantizationPostTrain(nameNN2)

def TMOTQuantizationPostTrain3():    
    TMOTQuantizationPostTrain(nameNN3)

def TMOTQuantizationPostTrain4():    
    TMOTQuantizationPostTrain(nameNN4)


def TMOTQuantizationPostTrain(name):
    TMOTQuantizationPostTrainDynamicRangeQuantization(name)
    TMOTQuantizationPostTrainFullIntegerQuantizationIntegerWithFloatFallback(name)
    TMOTQuantizationPostTrainFullIntegerQuantizationIntegerOnly(name)
    TMOTQuantizationPostTrainFloat16Quantization(name)
    TMOTQuantizationPostTrainIntegerOnly16BitActivationsWith8BitWeightsV1(name)
    TMOTQuantizationPostTrainIntegerOnly16BitActivationsWith8BitWeightsV2(name)
    # load original training model metrics
    # load original evaluate model metrics
    # load compresseds training model metrics
    # load compresseds evaluate model metrics


        
def Run(): 
    
    # CREATE INPUT
    CreateInput()


    # CREATE OUTPUT
    CreateOutput()


    # BUILD + TRAIN + PRINT MODEL Regression Tree
    BuildTrainPrintModelRegressionTree()
    

    # BUILD + TRAIN + PRINT + SAVE MODEL NN1
    BuildTrainPrintSaveModelNeuralNetwork1()

    
    # BUILD + TRAIN + PRINT + SAVE MODEL NN2
    BuildTrainPrintSaveModelNeuralNetwork2()


    # BUILD + TRAIN + PRINT + SAVE MODEL NN3
    BuildTrainPrintSaveModelNeuralNetwork3()

    # EXECUTE OPTIMIZATION MODEL Regression Tree
    # ExecuteOptimizationProblemRegTree()
    
    # EXECUTE OPTIMIZATION MODEL NN1
    ExecuteOptimizationProblemNN1()

    # EXECUTE OPTIMIZATION MODEL NN2
    ExecuteOptimizationProblemNN2()

    # EXECUTE OPTIMIZATION MODEL NN3
    ExecuteOptimizationProblemNN3()


    # SIMULATE MODEL Regression Tree
    SimulateProblemRegTree()
    
    # SIMULATE MODEL NN1
    SimulateProblemNN1()

    # SIMULATE MODEL NN2
    SimulateProblemNN2()

    # SIMULATE MODEL NN2
    SimulateProblemNN3()

    # BRUTE FORCE
    BruteForce()

def RunTest():
    # CREATE INPUT
    # CreateInput()

    
    # CREATE OUTPUT
    # CreateOutput()

    ManageData()

    # BUILD + TRAIN + PRINT + SAVE MODEL NN1
    #BuildTrainPrintSaveModelNeuralNetwork1()

    #TMOTCompressPredictNeuralNetwork1()

    TMOTQuantizationPostTrain1()    
    #TMOTQuantizationPostTrain2()  
    #TMOTQuantizationPostTrain3()
    #TMOTQuantizationPostTrain4()

# START    
# Run()
RunTest()

















