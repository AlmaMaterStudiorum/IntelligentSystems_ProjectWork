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

#from matplotlib import pyplot as plt
from util import util
from scipy.integrate import odeint
import numpy as np
import pandas as pd
from skopt.space import Space
#from eml.net.reader import keras_reader

# Latin Hypercube Sampling

# Let's see a practical example
# Here is the result of uniform sampling, for reference
test_nsamples, test_ranges = 12, [(0., 1.), (0., 1.)]
X = util.sample_points(test_ranges, test_nsamples, mode='uniform', seed=42)
util.plot_2D_samplespace(X, figsize=figsize)


# Let's see a practical example
# ...And here is the result of classical LHS:
test_nsamples, test_ranges = 12, [(0., 1.), (0., 1.)]
X = util.sample_points(test_ranges, test_nsamples, mode='lhs', seed=42)
util.plot_2D_samplespace(X, figsize=figsize)


# The process can be further improved
# E.g. after sampling we can try to maximize the minimum distance
test_nsamples, test_ranges = 12, [(0., 1.), (0., 1.)]
X = util.sample_points(test_ranges, test_nsamples, mode='max_min', seed=42)
util.plot_2D_samplespace(X, figsize=figsize)


# Dataset Input
# We are now ready to generate our dataset input

n_tr, n_ts = 10000, 2000
sir_tr_in = util.generate_SIR_input(max_samples=n_tr, mode='max_min', seed=42, normalize=True, max_beta=0.4)
sir_ts_in = util.generate_SIR_input(max_samples=n_ts, mode='max_min', seed=42, normalize=True, max_beta=0.4)
sir_tr_in.head()

# Dataset Output
# We obtain the corresponding output via simulation

# %%time
gamma = 1/14
sir_tr_out = util.generate_SIR_output(sir_tr_in, gamma, 7)
sir_ts_out = util.generate_SIR_output(sir_ts_in, gamma, 7)
sir_tr_out.head()

# Training a Model
# We try with Linear Regression

nn0 = util.build_ml_model(input_size=4, output_size=3, hidden=[], name='LR')
history0 = util.train_ml_model(nn0, sir_tr_in, sir_tr_out, verbose=0, epochs=100)
util.plot_training_history(history0, figsize=figsize)
util.print_ml_metrics(nn0, sir_tr_in, sir_tr_out, 'training')
util.print_ml_metrics(nn0, sir_ts_in, sir_ts_out, 'test')

# Training a Model
# ...And with a shallow Neural Network

nn1 = util.build_ml_model(input_size=4, output_size=3, hidden=[8], name='MLP')
history1 = util.train_ml_model(nn1, sir_tr_in, sir_tr_out, verbose=0, epochs=100)
util.plot_training_history(history1, figsize=figsize)
util.print_ml_metrics(nn1, sir_tr_in, sir_tr_out, 'training')
util.print_ml_metrics(nn1, sir_ts_in, sir_ts_out, 'test')

# Considerations and Next Steps
# We will save both models for later

# util.save_ml_model(nn0, 'nn0')
# util.save_ml_model(nn1, 'nn1')

datafolder = 'C:\Sviluppo\Lab\DATA\DesktopSandBox\IS_AP'
util.save_ml_model_with_winfolder(datafolder,nn0, 'nn0')
util.save_ml_model_with_winfolder(datafolder,nn1, 'nn1')

input('Premi un tasto per uscire ...')

