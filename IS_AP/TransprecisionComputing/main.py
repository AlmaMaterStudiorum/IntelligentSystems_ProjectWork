import tensorflow as tf
import torch
import pandas as pd
from util import util

from nncf import NNCFConfig
from nncf.tensorflow import create_compressed_model, register_default_init_args



import tpcsupport as sup
import specificenv as se
import support


def RunTPC001():
    """
    GetTheData
    Create NN
    Train NN
    Evaluate NN
    Compress NN -> NN.C
    Executute OptComb NN
    Execute OptComb NN.C (? non sono sicuro che sia possibile ?)


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



def RunNNCF001Tensorflow():
    # Instantiate your uncompressed model
    from tensorflow.keras.applications import ResNet50
    model = ResNet50()

    # Load a configuration file to specify compression
    configFile = r"D:\Sviluppo\GitHub\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\nncf\resnet_default.json"
    datasetPattern = r"D:\Sviluppo\GitHub\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\*.pkl"
    nncf_config = NNCFConfig.from_json(configFile)

    # Provide dataset for compression algorithm initialization
    representative_dataset = tf.data.Dataset.list_files(datasetPattern)
    nncf_config = register_default_init_args(nncf_config, representative_dataset, batch_size=256)

    # Apply the specified compression algorithms to the model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    # Now use compressed_model as a usual Keras model
    # to fine-tune compression parameters along with the model weights

    # ... the rest of the usual TensorFlow-powered training pipeline

    # Export to Frozen Graph, TensorFlow SavedModel or .h5  when done fine-tuning 
    compression_ctrl.export_model("compressed_model.pb", save_format='frozen_graph')

def RunNNCF001PyTorch():
    # Instantiate your uncompressed model
    from torchvision.models.resnet import resnet50
    model = resnet50()

    configFile = r"D:\Sviluppo\GitHub\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\input\nncf\resnet_default.json"
    datasetPattern = r"D:\Sviluppo\GitHub\AlmaMaterStudiorum\IntelligentSystems_ProjectWork\IS_AP\data\sir_tr_in.pkl"

    # Load a configuration file to specify compression
    nncf_config = NNCFConfig.from_json(configFile)

    # Provide data loaders for compression algorithm initialization, if necessary
    import torchvision.datasets as datasets
    #representative_dataset = datasets.ImageFolder(datasetPattern)

    representative_dataset = pd.read_pickle(datasetPattern)

    init_loader = torch.utils.data.DataLoader(representative_dataset)
    nncf_config = register_default_init_args(nncf_config=nncf_config,batch_size=256,data_loader= init_loader)

    print(list(model.parameters()))

    for name, param in model.named_parameters() :
        param.requires_grad = False
        if name.startswith('built') : 
            param.requires_grad = True

    # Apply the specified compression algorithms to the model
    compression_ctrl, compressed_model = create_compressed_model(model, nncf_config)

    # Now use compressed_model as a usual torch.nn.Module 
    # to fine-tune compression parameters along with the model weights

    # ... the rest of the usual PyTorch-powered training pipeline

    # Export to ONNX or .pth when done fine-tuning
    compression_ctrl.export_model("compressed_model.onnx")
    torch.save(compressed_model.state_dict(), "compressed_model.pth")



# RUNNING AREA


#RunNNCF001Tensorflow()
#RunNNCF001PyTorch()
RunTPC001()
