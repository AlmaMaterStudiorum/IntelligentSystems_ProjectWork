import pickle as pkl
import pandas as pd

import pathfiles as pf
from keras.models import Sequential
from keras.layers import Dense
import tensorflow as tf

filein = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\exp_results_correlation_10000.pickle'
fileout = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\exp_results_correlation_10000.csv'

if False :

    with open(filein, "rb") as f:
        object = pkl.load(f)
    
    df = pd.DataFrame(object)
    df.to_csv(fileout)

def func001():

    df = pd.read_csv(fileout)

    # iterating the columns
    for col in df.columns:
        row0 = df[col][0]
        row1 = df[col][1]
        message = '{col} : {row0}  {row1}'.format(col=col,row0=row0,row1=row1)
        print(message)


    print(df.iloc[0])

    print(df.iloc[1])


def TransformCorrelation():
    correlationnewmod = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\exp_results_correlation_10000_mod.csv'
    correlationOutput = r'C:\RepositoryGIT\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\tpc\Correlation_newhw.csv'
    benchmark = 'Correlation'
    fileNameg100 = f'{benchmark}_g100.csv'
    fileNamepc = f'{benchmark}_pc.csv'
    fileNamevm = f'{benchmark}_vm.csv'

    dfg100 = pd.read_csv(pf.GetInputDataFileFullPath(fileNameg100))

    dfpc = pd.read_csv(pf.GetInputDataFileFullPath(fileNamepc))

    dfvm = pd.read_csv(pf.GetInputDataFileFullPath(fileNamevm))

    dfNew = pd.read_csv(correlationnewmod)

    
    for indexOld, rowOld in dfg100.iterrows():
        for indexNew, rowNew in dfNew.iterrows():
            needCopy = True
            for i in [0 , 1, 2, 3, 4, 5, 6]:
                varX = f'var_{i}'
                if rowOld[varX] != rowNew[varX]:
                    needCopy = False
                    break

            if needCopy == True:
                dfg100.iloc[indexOld]['error'] = rowNew['error']

    for indexOld, rowOld in dfpc.iterrows():
        for indexNew, rowNew in dfNew.iterrows():
            needCopy = True
            for i in [0 , 1, 2, 3, 4, 5, 6]:
                varX = f'var_{i}'
                if rowOld[varX] != rowNew[varX]:
                    needCopy = False
                    break

            if needCopy == True:
                dfpc.iloc[indexOld]['error'] = rowNew['error']


    for indexOld, rowOld in dfvm.iterrows():
        for indexNew, rowNew in dfNew.iterrows():
            needCopy = True
            for i in [0 , 1, 2, 3, 4, 5, 6]:
                varX = f'var_{i}'
                if rowOld[varX] != rowNew[varX]:
                    needCopy = False
                    break

            if needCopy == True:
                dfvm.iloc[indexOld]['error'] = rowNew['error']


    dfAll = pd.DataFrame()


    df = dfAll.append([dfg100, dfpc,dfvm])
    df.to_csv(correlationOutput)


def func002():
    model = Sequential(
        [
            Dense(3, activation="relu"),
            Dense(5, activation="relu")
        ]
    )  # No weights to be addded here

    # Here we cannot check for weights
    # model.weights

    # Neither we can look at the summary
    # model.summary()

    # First we must call the model and evaluate it on test data
    x = tf.ones((5, 2))
    y = model(x)
    print("Number of weights after calling the model:", len(model.weights))
    model.summary()

    print("Print weight Model")
    iLayer = 0
    for layer in model.layers:
      print('Layer {layer}'.format(layer=iLayer))
      iLayer =+ 1
      l = layer.get_weights()[0]
      print(l) # weights

      for ws in l:
          for w in ws:
              print(w)
      print(layer.get_weights()[1]) # biases
    print("###################")

#TransformCorrelation()

func002()