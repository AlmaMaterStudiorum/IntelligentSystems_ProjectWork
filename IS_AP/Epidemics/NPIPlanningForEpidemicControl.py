# Control figure size
interactive_figures = False
if interactive_figures:
    # Normal behavior
    #%matplotlib widget
    figsize=(9, 3)
else:
    # PDF export behavior
    figsize=(14, 5)

#from matplotlib import pyplot as plt
from util import util
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from skopt.space import Space
from eml.net.reader import keras_reader
import datetime

# Some parameters
#S0, I0, R0 = 0.99, 0.01, 0.0
#gamma = 1/14

# knn0 = util.load_ml_model('nn0')
# knn1 = util.load_ml_model('nn1')

pd.set_option('display.max_rows', None)
pd.set_option('display.max_columns', None)
pd.set_option('display.width', 1000)
pd.set_option('display.colheader_justify', 'center')
pd.set_option('display.precision', 3)

datafolder = 'C:\Sviluppo\Lab\DATA\DesktopSandBox\IS_AP'

knn0 = util.load_ml_model_with_winfolder(datafolder,'nn0')
knn1 = util.load_ml_model_with_winfolder(datafolder,'nn1')


nn0 = keras_reader.read_keras_sequential(knn0)
nn1 = keras_reader.read_keras_sequential(knn1)

nn0

nn0.layer(0).update_lb(np.zeros(4))
nn0.layer(0).update_ub(np.ones(4));

from eml.net.process import ibr_bounds
ibr_bounds(nn0)
nn0

npis = [
    util.NPI('masks-indoor', effect=0.75, cost=1),
    util.NPI('masks-outdoor', effect=0.9, cost=1),
    util.NPI('dad', effect=0.7, cost=3),
    util.NPI('bar-rest', effect=0.6, cost=3),
    util.NPI('transport', effect=0.6, cost=4)
]

S0, I0, R0 = 0.99, 0.01, 0.00
nweeks = 3
tlim = 30
beta_base = 0.35
budget = 20
gamma = 1/14


# knn1
starttime = datetime.datetime.now()

sol, closed = util.solve_sir_planning(knn1, npis, S0, I0, R0, beta_base=beta_base, budget=budget,
                                      nweeks=nweeks, tlim=tlim)
endtime = datetime.datetime.now()
elapsed = endtime - starttime
print('elapsed {elapsed}'.format(elapsed=int(elapsed.total_seconds() * 1000)))

print(f'Problem closed: {closed}')
sol_df = util.sol_to_dataframe(sol, npis, nweeks)

print('sol_df')
print(sol_df)

beta_sched = sol_df.iloc[:-1]['b']
sim_df = util.simulate_SIR_NPI(S0, I0, R0, beta_sched, gamma, steps_per_day=1)
print('sim_df')
print(sim_df)




# knn0
starttime = datetime.datetime.now()
sol2, closed2 = util.solve_sir_planning(knn0, npis, S0, I0, R0, beta_base=beta_base, budget=budget,
                                      nweeks=nweeks, tlim=tlim)
endtime = datetime.datetime.now()
elapsed = endtime - starttime
print('elapsed {elapsed}'.format(elapsed=int(elapsed.total_seconds() * 1000)))
print(f'Problem closed: {closed}')
sol_df2 = util.sol_to_dataframe(sol2, npis, nweeks)
print('sol_df2')
print(sol_df2)


beta_sched2 = sol_df2.iloc[:-1]['b']
sim_df = util.simulate_SIR_NPI(S0, I0, R0, beta_sched2, gamma, steps_per_day=1)
print('sim_df 2')
print(sim_df)

# Brute Force
starttime = datetime.datetime.now()
best_S, best_sched = util.solve_sir_brute_force(npis, S0, I0, R0, beta_base, gamma, nweeks, budget)
endtime = datetime.datetime.now()
elapsed = endtime - starttime
print('elapsed {elapsed}'.format(elapsed=int(elapsed.total_seconds() * 1000)))
print('best_S')
print(best_S)

input('Premi un tasto per uscire ...')






