import argparse
import pandas as pd
import pickle 
import docplex
from eml.backend import cplex_backend
from eml.tree.reader.sklearn_reader import read_sklearn_tree
from eml.tree import embed 
import os

####### LOGGER #######
def logger(x):
    with open("builder.log", "a") as file_log:
        file_log.write(f"{x}\n")
    print(x)

####### METADATA ########
HW = ["pc", "vm", "g100"]
TARGET = ["time", "error", "memory"]
    
def build_model(var_bounds, mlmodel_files):
    """
    Build basic HADA (user-independent components):
        1. declare basic variables and constraints
        2. embed predictive models 

    PARAMETERS
    ---------
    var_bounds [pd.DataFrame]: lower and upper bound (along columns) of each declarative variable (along rows)
    mlmodel_files [dict]: for each pair hw-target (keys), the path to the corresponding predictive model (values) 

    RETURN
    ------
    basic_hada.lp [.lp file]: basic HADA, exported into an .lp file
    """

    logger("Start building model")

    ####### MODEL #######
    bkd = cplex_backend.CplexBackend()
    mdl = docplex.mp.model.Model("HADA")

    ####### VARIABLES #######
    # A binary variable for each hw, specifying whether this hw is selected or not
    for hw in HW:
        mdl.binary_var(name = f"b_{hw}")

    # A continuous variable for each pair hw-target, representing the algorithm performance, 
    # in terms of this target, when running on this hw 
    for hw in HW:
        for target in TARGET:
            mdl.continuous_var(name = f"y_{hw}_{target}", 
                    lb = var_bounds.loc[f"{target}", "min"], ub = var_bounds.loc[f"{target}", "max"])

    # An integer variable and an auxiliary continuous variable for each integer parameter of the algorithm, 
    # representing the value assigned to this parameter: the auxiliary variables are used as input to the 
    # predictive models (emllib accepts solely continuous variables), then converted into integer variables
    # through an equality constraint
    for i in [0,1,2]:
        mdl.integer_var(name = f"var_{i}", 
                lb = var_bounds.loc[f"var_{i}", "min"], ub = var_bounds.loc[f"var_{i}", "max"])
        mdl.continuous_var(name = f"auxiliary_var_{i}", 
                lb = var_bounds.loc[f"var_{i}", "min"], ub = var_bounds.loc[f"var_{i}", "max"])

    ####### CONSTRAINTS ######
    # HW Selection Constraint, forcing the selection of a single hw platform
    mdl.add_constraint(mdl.sum(mdl.get_var_by_name(f"b_{hw}") for hw in HW) == 1, ctname = "hw_selection")

    # Integrality Constraints, enabling the convertion of the auxiliary variables from continuous to integer
    for i in [0,1,2]:
        mdl.add_constraint(mdl.get_var_by_name(f"var_{i}") == mdl.get_var_by_name(f"auxiliary_var_{i}"), 
                ctname = f"var_{i}_integrality_constraint")
    
    logger("\nDeclarative component size")
    logger(f"   #Variables: {mdl.number_of_variables}") 
    logger(f"   #Constraints: {mdl.number_of_constraints}")
                
    logger("\nStart embedding predective models")
    # Empirical Constraints: embed the predictive models into the optimization system (through emllib)
    for hw in HW:
        for target in TARGET:

            model = pickle.load(open(mlmodel_files[(f"{hw}", f"{target}")], 'rb'))
            model = read_sklearn_tree(model)
             
            for i in [0,1,2]:
                model.update_lb(i, var_bounds.loc[f"var_{i}", "min"])
                model.update_ub(i, var_bounds.loc[f"var_{i}", "max"])

            embed.encode_backward_implications(
                    bkd = bkd, mdl = mdl,
                    tree = model, 
                    tree_in = [mdl.get_var_by_name(f"auxiliary_var_{i}") for i in [0,1,2]], 
                    tree_out = mdl.get_var_by_name(f"y_{hw}_{target}"),
                    name = f"DT_{hw}_{target}"
                    )

            logger(f"   Predictive model corresponding to ({hw}, {target}) embedded")

    logger("\nFull model size")
    logger(f"   #Variables: {mdl.number_of_variables}") 
    logger(f"   #Constraints: {mdl.number_of_constraints}")
    
    logger("\nExporting model")
    mdl.export_as_lp("basic_hada.lp")
    logger("\nModel exported to basic_hada.lp")

def Run(data_folder= "datasets",model_folders="DTs"):
    ###### DATA #####
    # extract, from the training dataset of the empirical models, upper/lower bound of each declarative varibale 
    bounds_min = {}
    bounds_max = {}


    # Get the current working directory
    cwd = os.getcwd()

    # Print the current working directory
    print("Current working directory: {0}".format(cwd))
    
    for hw in HW:
        fullpath = fr".\vertical-matchmaking\{data_folder}\Saxpy_{hw}.csv"
        dataset = pd.read_csv(fullpath,
                usecols = ["var_0", "var_1", "var_2", "err_mean", "ex_time", "peak_mem_used"])
        dataset = dataset.rename({"ex_time" : "time", "err_mean" : "error", "peak_mem_used": "memory"}, axis = 1)
        bounds_min[hw] = dataset.min()
        bounds_max[hw] = dataset.max()
    var_bounds = pd.DataFrame({
            "min" : pd.DataFrame(bounds_min).transpose().min(), 
            "max" : pd.DataFrame(bounds_max).transpose().max()
            })
    var_bounds.loc[["var_0", "var_1", "var_2"], "max"] = [53, 53, 53]
 
    ##### TREES ######
    # load the trained empirical model to embed into the HADA
    mlmodel_files = {(f"{hw}", f"{target}") : fr".\vertical-matchmaking\{model_folders}\saxpy_{hw}_{target}_DecisionTree_10" for hw in HW for target in TARGET}

    ##### BUILD ######
    build_model(var_bounds, mlmodel_files)



if __name__ == "__main__":
    logger("===========================================================")

    ####### INPUT #######
    parser = argparse.ArgumentParser()
    parser.add_argument('--data_folder', type = str, default = "datasets", required = False)
    parser.add_argument('--models_folder', type = str, default = "DTs", required = False)
    
    args = parser.parse_args()

    Run(args.data_folder,args.models_folder)


