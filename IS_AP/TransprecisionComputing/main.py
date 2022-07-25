import tensorflow as tf
from util import util

import specificenv as se
import support


def RunTPC001():
    """
    GetTheData
    Create NN
    Train NN
    Evaluate NN
    Pruning NN -> NN.P
    Executute OptComb NN
    Executute OptComb NN.P

    """
    thick = 16

    nns = { 
                'nn1' : [thick,thick,thick,thick] ,
                'nn2' : [thick,thick,thick,thick,thick,thick,thick,thick]       
          }

    nameNN = 'nn1'
    benchmarks = ['Convolution','Correlation','Saxpy']
    benchmarksMod = ['Convolution','Saxpy']

    topologies ={ 'Convolution'  :  {'size_in' : 7  , 'size_out' : 4},
                  'Correlation'  :  {'size_in' : 10 , 'size_out' : 4},
                  'Saxpy'        :  {'size_in' : 6 , 'size_out' : 4}
    }

    # currentBenchmark = 0
    # benchmark = benchmarks[currentBenchmark]
    timeMax = 100
    errorMax = 0.01

    startFromZero = True

    if startFromZero == True:
        support.CleanFolder(se.dataOutputFolder)


    for  benchmark in benchmarksMod:

        print(f"Benchmark {benchmark}")    

        tr_in,tr_out,ts_in,ts_out,boundaries = support.GetTheData(benchmark)

        for nn in nns:
            nameNN = f'{nn}_{benchmark}'
            hidden = nns[nn]

            if startFromZero == True:

                support.BuildTrainPrintSaveModelNeuralNetwork(hidden,nameNN,topologies[benchmark]['size_in'],topologies[benchmark]['size_out'],tr_in,tr_out,ts_in,ts_out)

                support.PruningNeuralNetwork(nameNN,tr_in,tr_out,ts_in,ts_out)
    
     
            modelReference = util.load_ml_model_with_winfolder(se.dataOutputFolder,nameNN)

            modelPruning = tf.keras.models.load_model(support.GetOutputDataFileFullPath(f'{nameNN}_{support.PRUNING}.h5'))


            if benchmark == 'Convolution':
                support.ExecuteCombinatorialOptimizationConvolution(modelReference,nn,'reference',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationConvolution(modelPruning,nn,'pruning',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])
            elif benchmark == 'Correlation':
                support.ExecuteCombinatorialOptimizationCorrelation(modelReference,nn,'reference',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationCorrelation(modelPruning,nn,'pruning',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])
            elif benchmark == 'Saxpy':
                support.ExecuteCombinatorialOptimizationSaxpy(modelReference,nn,'reference',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationSaxpy(modelPruning,nn,'pruning',boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])





# RUNNING AREA

RunTPC001()
