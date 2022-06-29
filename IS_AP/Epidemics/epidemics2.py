
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
datafolder = 'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data'
overwriteModel=False
modelRegressionTree=[]
solutionGlobal= {
    nameRegTree : {},
    nameNN1 : {},
    nameNN2 : {},
    nameNN3 : {}
}

closedGlobal= {
    nameRegTree : False,
    nameNN1 : False,
    nameNN2 : False,
    nameNN3 : False
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
    sir_tr_in.head()
    EndMethod('generate_SIR_input')

def CreateOutput():
    BeginMethod('generate_SIR_output')
    global sir_tr_out
    global sir_ts_out
    sir_tr_out = util.generate_SIR_output(sir_tr_in, gammaConfig, tmaxConfig)
    sir_ts_out = util.generate_SIR_output(sir_ts_in, gammaConfig, tmaxConfig)
    sir_tr_out.head()
    EndMethod('generate_SIR_output')

def BuildTrainPrintModelRegressionTree():
    BeginMethod('BuildTrainPrintModelRegressionTree')
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

    EndMethod('BuildTrainPrintModelRegressionTree')

def BuildTrainPrintSaveModelNeuralNetwork(hidden,name): 
    BeginMethod('BuildTrainPrintSaveModelNeuralNetwork: {name}'.format(name=name))
    nameHF='{name}.{extension}'.format(name=name,extension='h5')
    if (not exists(os.path.join(datafolder, nameHF))) or (overwriteModel == True):    
        modelNeuralNework = util.build_ml_model(input_size=4, output_size=3, hidden=hidden, name='MLP')
        historyNeuralNework = util.train_ml_model(modelNeuralNework, sir_tr_in, sir_tr_out, verbose=0, epochs=100)
        util.plot_training_history(historyNeuralNework, figsize=figsize)
        util.print_ml_metrics(modelNeuralNework, sir_tr_in, sir_tr_out, 'training')
        util.print_ml_metrics(modelNeuralNework, sir_ts_in, sir_ts_out, 'test')
        util.save_ml_model_with_winfolder(datafolder,modelNeuralNework, name)
    else:
        print('No execution')
    

    EndMethod('BuildTrainPrintSaveModelNeuralNetwork: {name}'.format(name=name))
      
    
def BuildTrainPrintSaveModelNeuralNetwork1():
    BuildTrainPrintSaveModelNeuralNetwork([8,8,8,8],nameNN1)
                
def BuildTrainPrintSaveModelNeuralNetwork2():
    BuildTrainPrintSaveModelNeuralNetwork([16,16],nameNN2)

def BuildTrainPrintSaveModelNeuralNetwork3():
    BuildTrainPrintSaveModelNeuralNetwork([16,16,16,16],nameNN3)
         
def ExecuteOptimizationProblem(name):
    BeginMethod('ExecuteOptimizationProblem: {name}'.format(name=name))
    gc.collect()
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
    BeginMethod('BruteForce')

    best_S, best_sched = util.solve_sir_brute_force(npis, S0, I0, R0, beta_base, gammaConfig, nweeks, budget)

    print('best_S')
    print(best_S)
    EndMethod('BruteForce')
    
        
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

# START    
Run()

















