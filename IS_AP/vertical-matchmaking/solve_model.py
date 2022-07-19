import os
import csv
import argparse
import pandas as pd
import docplex
from docplex.mp.model_reader import ModelReader

####### LOGGER #######
def logger(x):
    with open("solver.log", "a") as file_log:
        file_log.write(f"{x}\n")
    print(x)

####### METADATA #######
HW = ["pc", "vm", "g100"]
TARGET = ["time", "error", "memory"]

def solve_model(mdl, cost, user_constraint):
    '''
    Select optimal hardware platform and algorithm configuration
        1. Add objective and user-defined constraints on top of basic HADA
        2. Solve the complemented model and output an optimal matching hw-platform/alg-configuration 

    PARAMETERS
    ---------
    mdl [docplex.mp.Model]: basic HADA model
    cost [dict]: cost of each hw platform (objective parameters) 
    user_constraint [dict]: type (<=, ==, >=) and right-hand side of each user-defined constraint  

    RETURN
    ------
    sol: optimal solution, stored into a .csv file
    hada.lp [.lp file]: final model, exported into an .lp file
    '''

    ####### CONSTRAINTS ######
    # User-Defined Constraints, bounding the performance of the algorithm as required by the user
    for target in user_constraint.keys():
        for hw in HW:
            if user_constraint[target]["type"] == "<=":
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"), 
                        mdl.get_var_by_name(f"y_{hw}_{target}") <= user_constraint[f"{target}"]["bound"],
                        1, name = f"user_constraint_{target}_{hw}")
            elif user_constraint[target]["type"] == ">=":
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"), 
                        mdl.get_var_by_name(f"y_{hw}_{target}") >= user_constraint[f"{target}"]["bound"],
                        1, name = f"user_constraint_{target}_{hw}")
            else:
                mdl.add_indicator(mdl.get_var_by_name(f"b_{hw}"), 
                        mdl.get_var_by_name(f"y_{hw}_{target}") == user_constraint[f"{target}"]["bound"],
                        1, name = f"user_constraint_{target}_{hw}")
    
    logger("User requirements:")
    for target in user_constraint:
        logger("    {} {} {}".format(target, user_constraint[target]["type"], user_constraint[target]["bound"]))
            
    ##### OBJECTIVE #####
    #Select the hw platform that minimizes the cost for running the algorithm 
    mdl.minimize(mdl.sum(cost[hw]*mdl.get_var_by_name(f"b_{hw}") for hw in HW))
    
    logger("\nObjective costs:")
    for hw in HW: 
        logger("    {} = {}".format(hw, cost[hw]))

    logger("\nExporting model")
    mdl.export_as_lp("hada.lp")
    logger("Model exported into hada.lp")

    ##### SOLVE #####
    logger("\nStart solve")
    sol = mdl.solve()

    if sol is None:
        solution = dict(
               {target : (user_constraint[target]["type"], user_constraint[target]["bound"]) if target in user_constraint.keys() else None for target in TARGET}, 
                **{f"b_{hw}" : None for i in HW},
                **{f"var_{i}" : None for i in [0,1,2]},
                **{f"y_{hw}_{target}" : None for hw in HW for target in TARGET}
                )
    else:
       solution = dict(
               {target : (user_constraint[target]["type"], user_constraint[target]["bound"]) if target in user_constraint.keys() else None for target in TARGET}, 
               **{f"b_{hw}" : sol[f"b_{hw}"] for hw in HW},
               **{f"var_{i}" : sol[f"var_{i}"] for i in [0,1,2]},
               **{f"y_{hw}_{target}" : sol[f"y_{hw}_{target}"] for hw in HW for target in TARGET}
               )
    
    # Print optimal solution    
    logger("\nSolution")
    for key in solution.keys():
        logger(f"   {key} = {solution[key]}")

    # Append the optimal solution as a row of a (just created/already existing) .csv file 
    if not os.path.exists("./results.csv"):
        results = pd.DataFrame({k : [] for k in 
            TARGET + 
            [f"b_{hw}" for hw in HW] + 
            [f"var_{i}" for i in [0,1,2]] + 
            [f"y_{hw}_{target}" for hw in HW for target in TARGET]
            })
        results.to_csv("results.csv", index = False)

    with open('results.csv','a') as f:
        writer=csv.writer(f)
        writer.writerow(list(solution.values()))


if __name__ == "__main__":
    logger("===========================================================")

    ####### INPUT #######
    parser = argparse.ArgumentParser()
    # model 
    parser.add_argument('--basic_model', type = str, required = True)
    # hw costs
    parser.add_argument('--cost_pc', type = float, default = 1, required = False)
    parser.add_argument('--cost_vm', type = float, default = 3, required = False)
    parser.add_argument('--cost_g100', type = float, default = 2, required = False)
    # target bounds
    parser.add_argument('--constraint_time', type = str, required = False, nargs = 2)
    parser.add_argument('--constraint_error', type = str, required = False, nargs = 2)
    parser.add_argument('--constraint_memory', type = str, required = False, nargs = 2)

    args = parser.parse_args()

    ####### MODEL  #######
    logger("\nLoading basic HADA")
    mdl = ModelReader.read(args.basic_model)

    ###### DATA #####
    cost = {"pc" : args.cost_pc, "vm" : args.cost_vm, "g100" : args.cost_g100}
    user_constraint = {"time" : args.constraint_time, "error" : args.constraint_error, "memory" : args.constraint_memory}
    user_constraint = {target : {"type" : constraint[0], "bound" : float(constraint[1])} 
            for target, constraint in user_constraint.items() if constraint is not None} 
    
    solve_model(mdl, cost, user_constraint)
