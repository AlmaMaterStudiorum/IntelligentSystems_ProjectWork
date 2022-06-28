
#from matplotlib import pyplot as plt
from glob import glob
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
sir_tr_in=[]
sir_ts_in=[]
sir_tr_out=[]
sir_ts_out=[]
nameRegTree='regtree'
nameNN1='nn1'
nameNN2='nn2'
nameNN3='nn3'
nameNN1HF='{name}.{extension}'.format(name=nameNN1,extension='h5')
nameNN2HF='{name}.{extension}'.format(name=nameNN2,extension='h5')
nameNN3HF='{name}.{extension}'.format(name=nameNN3,extension='h5')
datafolder = 'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data'
overwriteModel=False
modelRegressionTree=[]




def CreateInput():
    print('Before generate_SIR_input')
    global sir_tr_in 
    global sir_ts_in 
    sir_tr_in = util.generate_SIR_input(max_samples=n_tr, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta)
    sir_ts_in = util.generate_SIR_input(max_samples=n_ts, mode=modeConfig, seed=seedConfig, normalize=True, max_beta=max_beta)
    
    
    sir_tr_in.head()
    print('After generate_SIR_input')

def CreateOutput():
    print('Before generate_SIR_output')
    global sir_tr_out
    global sir_ts_out
    sir_tr_out = util.generate_SIR_output(sir_tr_in, gammaConfig, tmaxConfig)
    sir_ts_out = util.generate_SIR_output(sir_ts_in, gammaConfig, tmaxConfig)
    sir_tr_out.head()
    print('After generate_SIR_output')

def BuildTrainPrintModelRegressionTree():
    print('Start BuildTrainPrintModelRegressionTree')
    print('Before train')
    global modelRegressionTree
    modelRegressionTree = DecisionTreeRegressor(random_state=44)

    historyRegressionTree = modelRegressionTree.fit(sir_tr_in, sir_tr_out)

    print('After train')

    predictRegressionTree=modelRegressionTree.predict(sir_ts_in[:1])
    scoreRegressionTree=modelRegressionTree.score(sir_ts_in, sir_ts_out)
    print('predict = {predict}'.format(predict = predictRegressionTree))
    print('score = {score}'.format(score = scoreRegressionTree))
    print('After train')

    # plt.figure(figsize=(10,8), dpi=150)
    # plot_tree(modelRegressionTree, feature_names=sir_tr_in.columns)
    # plt.savefig('C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\fst.svg',format='svg',bbox_inches = "tight")
    # ax = modelRegressionTree.create_tree_digraph(clf)
    # with open('C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\fst.svg', 'w') as f:
    #    f.write(ax._repr_svg_())

    print('End BuildTrainPrintModelRegressionTree')

def BuildTrainPrintSaveModelNeuralNetwork(hidden,name): 
    print('------------------------------------------------------')
    print('BuildTrainPrintSaveModelNeuralNetwork: {name}'.format(name=name))
    
    modelNeuralNework = util.build_ml_model(input_size=4, output_size=3, hidden=hidden, name='MLP')
    historyNeuralNework = util.train_ml_model(modelNeuralNework, sir_tr_in, sir_tr_out, verbose=0, epochs=100)
    util.plot_training_history(historyNeuralNework, figsize=figsize)
    util.print_ml_metrics(modelNeuralNework, sir_tr_in, sir_tr_out, 'training')
    util.print_ml_metrics(modelNeuralNework, sir_ts_in, sir_ts_out, 'test')

    util.save_ml_model_with_winfolder(datafolder,modelNeuralNework, name)
    print('------------------------------------------------------')
    
def BuildTrainPrintSaveModelNeuralNetwork1():
    if (not exists(os.path.join(datafolder, nameNN1HF))) or (overwriteModel == True):
        BuildTrainPrintSaveModelNeuralNetwork([8,8,8,8],nameNN1)
                
def BuildTrainPrintSaveModelNeuralNetwork2():
    if (not exists(os.path.join(datafolder, nameNN2HF))) or (overwriteModel == True):
        BuildTrainPrintSaveModelNeuralNetwork([16,16,16,16],nameNN2)

def BuildTrainPrintSaveModelNeuralNetwork3():
    if (not exists(os.path.join(datafolder, nameNN3HF))) or (overwriteModel == True):
        BuildTrainPrintSaveModelNeuralNetwork([16,16],nameNN3)
         
def ExecuteOptimizationProblem(name):
    print('------------------------------------------------------')
    print('ExecuteOptimizationProblem: {name}'.format(name=name))
    npis = [
        util.NPI('masks-indoor', effect=0.75, cost=1),
        util.NPI('masks-outdoor', effect=0.9, cost=1),
        util.NPI('dad', effect=0.7, cost=3),
        util.NPI('bar-rest', effect=0.6, cost=3),
        util.NPI('transport', effect=0.6, cost=4)
    ]
    
    
    S0, I0, R0 = 0.99, 0.01, 0.00
    nweeks = 3
    tlim = 800
    beta_base = 0.35
    budget = 20
    
    starttime = datetime.datetime.now()
    
    if name==nameRegTree :
        model = modelRegressionTree
    else:
        model = util.load_ml_model_with_winfolder(datafolder,name)
    
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
    endtime = datetime.datetime.now()
    elapsed = endtime - starttime
    print('elapsed {elapsed}'.format(elapsed=int(elapsed.total_seconds() * 1000)))

    print(f'Problem closed: {closed} , Cost = {cost}')
    

    print('------------------------------------------------------')
 
def ExecuteOptimizationProblemRegTree():
    ExecuteOptimizationProblem(nameRegTree)

def ExecuteOptimizationProblemNN1():
    ExecuteOptimizationProblem(nameNN1)

def ExecuteOptimizationProblemNN2():
    ExecuteOptimizationProblem(nameNN2)
    
def ExecuteOptimizationProblemNN3():
    ExecuteOptimizationProblem(nameNN3)
    
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


    # DEFINE CONSTRAINS



    # DEFINE OPTIZATION MODEL 



    # EXECUTE OPTIMIZATION MODEL Regression Tree
    # ExecuteOptimizationProblemRegTree()
    
    # EXECUTE OPTIMIZATION MODEL NN1
    ExecuteOptimizationProblemNN1()

    # EXECUTE OPTIMIZATION MODEL NN2
    ExecuteOptimizationProblemNN2()

    # EXECUTE OPTIMIZATION MODEL NN3
    ExecuteOptimizationProblemNN3()


    # SIMULATE MODEL Regression Tree

    # SIMULATE MODEL NN1

    # SIMULATE MODEL NN2

    

# START    
Run()

















