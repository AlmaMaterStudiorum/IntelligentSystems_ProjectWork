from util import util

# Control figure size
interactive_figures = False
if interactive_figures:
    # Normal behavior
    # %matplotlib widget
    figsize=(9, 3)
else:
    # PDF export behavior
    figsize=(14, 5)

S0, I0, R0 = 0.99, 0.01, 0.0
beta, gamma = 0.1, 1/14
tmax = 365

data = util.simulate_SIR(S0, I0, R0, beta, gamma, tmax=tmax, steps_per_day=1)

# Plot S + I * R
util.plot_df_cols(data, figsize=figsize)

# Plot only I
util.plot_df_cols(data[['I']], figsize=figsize)

input('Premi un tasto per uscire ...')






