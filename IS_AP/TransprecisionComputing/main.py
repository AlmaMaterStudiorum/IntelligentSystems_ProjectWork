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
    hidden = [thick,thick,thick,thick]
    nameNN = 'nn1'
    benchmarks = ['Convolution','Correlation']
    topologies ={ 'Convolution'  :  {'size_in' : 7  , 'size_out' : 4},
                  'Correlation'  :  {'size_in' : 10 , 'size_out' : 4}
    }

    currentBenchmark = 0
    benchmark = benchmarks[currentBenchmark]
 
    tr_in,tr_out,ts_in,ts_out,boundaries = support.GetTheData(benchmark)

    """
    support.BuildTrainPrintSaveModelNeuralNetwork(hidden,nameNN,topologies[benchmark]['size_in'],topologies[benchmark]['size_out'],tr_in,tr_out,ts_in,ts_out)

    support.PruningNeuralNetwork(nameNN,tr_in,tr_out,ts_in,ts_out)
    """
     
    modelReference = util.load_ml_model_with_winfolder(se.dataOutputFolder,nameNN)

    modelPruning = tf.keras.models.load_model(support.GetOutputDataFileFullPath(f'{nameNN}_{support.PRUNING}.h5'))



    support.ExecuteCombinatorialOptimizationConvolution(modelReference,boundaries)

    support.ExecuteCombinatorialOptimizationConvolution(modelPruning,boundaries)



# RUNNING AREA

RunTPC001()
