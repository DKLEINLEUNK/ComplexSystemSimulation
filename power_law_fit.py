
"""
Power law fitting for EPS and failures, only for the best DATASET/number 5 (3000 simulations and 0.3 failure)
Only for EPS and failures.
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw as powerlaw

EPS = [0.3, 0.6, 0.7, 0.85, 0.95]
FAIL = [0.1,0.4,0.6,0.8]
FILE_NUMBER = 5
VARIABLE_TO_PLOT = "EPS"
SHOW_PLOT = True
SAVE_PLOTS = True

EPS_SAMPLES = 2 ## EPS specifications to plot counting backwards, must be less or equal than 5
FAIL_SAMPLES = 3 ## FAIL specifications to plot counting forwards, must be less or equal than 4

p_value = []
alpha = []

def specs():
    """
    Sets specifications for the plot
    """
    if VARIABLE_TO_PLOT == "failure":
        plt.title(f"Threshold change: Log- Log PDF of Failure Sizes", fontsize = 15)
    else:
        plt.title(f"{VARIABLE_TO_PLOT} change: Log- Log PDF of Failure Sizes", fontsize = 15)
    plt.grid(True)
    plt.xlabel("Failure size", fontsize = 15)
    plt.ylabel("P(s)", fontsize = 15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()

def fitting(fit):
    """
    Gets, fitting values, power_law p_value done vs exponential distribution
    """
    alpha.append(fit.power_law.alpha)
    p_value.append(fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)[0])


if __name__ == "__main__":

    if VARIABLE_TO_PLOT == "EPS":
        for i in range(EPS_SAMPLES):
            data_eps = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p0.1-fail0.3-recov0.1-loss{EPS[-1-i]}.npy")
            fit = powerlaw.Fit(data_eps*100, discrete = True)
            fig = fit.plot_pdf(color=f'C{5-i}', linewidth=2, label=f'EPS={EPS[-1-i]}', linestyle = '-') # 5 is the number of EPS I can choose from
            fit.power_law.plot_pdf(color=f'C{5-i}', linewidth = 1, linestyle='--', label = f'EPS = {EPS[-1-i]} (alpha = {np.round(fit.power_law.alpha,2)})', ax=fig)
            fitting(fit)

    if VARIABLE_TO_PLOT == "failure":
        for i in range(FAIL_SAMPLES):
            data_thr  = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p0.1-fail{FAIL[i+1]}-recov0.1-loss0.85.npy")
            fit = powerlaw.Fit(data_thr*100, discrete = True)
            fig = fit.plot_pdf(color=f'C{i+1}', linewidth=2, label=f'Threshold={FAIL[i+1]}', linestyle = '-')
            fit.power_law.plot_pdf(color=f'C{i+1}', linewidth = 1, linestyle='--', label = f'Threshold ={FAIL[i+1]} (alpha = {np.round(fit.power_law.alpha,2)})', ax=fig)
            fitting(fit)

    specs()

    if SAVE_PLOTS == True:
        plt.savefig(f'figures\\{VARIABLE_TO_PLOT}\\dataset_{FILE_NUMBER}_pw_p0.1.png')
    if SHOW_PLOT == True:
        plt.show()
    
    print(alpha)
    print(p_value)
