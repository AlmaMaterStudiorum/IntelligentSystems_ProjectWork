import tensorflow as tf
from util import util

import specificenv as se
import pathfiles as pf
import support

# GLOBAL
thick = 16

nns = { 
            'nn1' : [thick,thick,thick,thick] ,
            'nn2' : [thick,thick,thick,thick,thick,thick,thick,thick]       
        }

benchmarks = ['Convolution','Correlation','Saxpy']
benchmarksMod = ['Convolution','Saxpy']
netTypes = [support.REFERENCE,support.PRUNING]

dataStructure = { 'Convolution' : {  
                                    'var' : { 'var_0','var_1','var_2','var_3' },
                                    'hw'  : { 'g100','pc','vm'} ,
                                    'out' : { 'error','time','memory_mean','memory_peak'}
                                  },
                  'Correlation' : {  
                                    'var' : { 'var_0','var_1','var_2','var_3','var_4','var_5','var_6' },
                                    'hw'  : { 'g100','pc','vm'} ,
                                    'out' : { 'error','time','memory_mean','memory_peak'}
                                  },
                  'Saxpy'       : {  
                                    'var' : { 'var_0','var_1','var_2' },
                                    'hw'  : { 'g100','pc','vm'} ,
                                    'out' : { 'error','time','memory_mean','memory_peak'}
                                  }                                     
                }

topologies ={   'Convolution'  :  {'size_in' :  7 , 'size_out' : 4},
                'Correlation'  :  {'size_in' : 10 , 'size_out' : 4},
                'Saxpy'        :  {'size_in' :  6 , 'size_out' : 4}
}

# currentBenchmark = 0
# benchmark = benchmarks[currentBenchmark]
timeMax = 100
errorMax = 0.01
    
startFromZero = True

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


    if startFromZero == True:
        support.CleanFolder(se.dataOutputFolder)


    for  benchmark in benchmarksMod:

        print(f"Benchmark {benchmark}")    

        tr_in,tr_out,ts_in,ts_out,boundaries = support.GetTheData(benchmark)

        for nn in nns:

            hidden = nns[nn]

            if startFromZero == True:

                support.BuildTrainPrintSaveModelNeuralNetwork(hidden,nn,benchmark,topologies[benchmark]['size_in'],topologies[benchmark]['size_out'],tr_in,tr_out,ts_in,ts_out)

                support.PruningNeuralNetwork(nn,benchmark,tr_in,tr_out,ts_in,ts_out)
    
     
            modelReference = util.load_ml_model_with_winfolder(se.dataOutputFolder,f'{nn}.{support.REFERENCE}.{benchmark}')

            modelPruning = tf.keras.models.load_model(pf.GetOutputDataFileFullPath(f'{nn}.{support.PRUNING}.{benchmark}.h5'))


            if benchmark == 'Convolution':
                support.ExecuteCombinatorialOptimizationConvolution(modelReference,nn,support.REFERENCE,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationConvolution(modelPruning,nn,support.PRUNING,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])
            elif benchmark == 'Correlation':
                support.ExecuteCombinatorialOptimizationCorrelation(modelReference,nn,support.REFERENCE,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationCorrelation(modelPruning,nn,support.PRUNING,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])
            elif benchmark == 'Saxpy':
                support.ExecuteCombinatorialOptimizationSaxpy(modelReference,nn,support.REFERENCE,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

                support.ExecuteCombinatorialOptimizationSaxpy(modelPruning,nn,support.PRUNING,boundaries,timeMax,errorMax,topologies[benchmark]['size_in'])

    support.CreateSummary(nns,netTypes,benchmarksMod)
    
def RunTPC002():
    support.CreateSummary(nns,netTypes,benchmarksMod)

# RUNNING AREA

#RunTPC001()
RunTPC002()
