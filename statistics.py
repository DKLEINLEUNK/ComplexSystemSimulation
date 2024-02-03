import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import powerlaw as powerlaw




EPS = [0.3, 0.6, 0.7, 0.85, 0.95]
rec = [0.1,0.4,0.6,1.0]
fail = [0.1,0.4,0.6,0.8]

value = "EPS"
p_value = []
alpha = []

if value == "EPS":
    for i in range(2):
        data_eps = np.load(f"data\{value}5\p0.1-fail0.3-recov0.1-loss{EPS[-1-i]}.npy")

        fit = powerlaw.Fit(data_eps*100, discrete = True)
        fig = fit.plot_pdf(color=f'C{5-i}', linewidth=2, label=f'EPS={EPS[-1-i]}', linestyle = '-')
        fit.power_law.plot_pdf(color=f'C{5-i}', linewidth = 1, linestyle='--', label = f'EPS = {EPS[-1-i]} (alpha = {np.round(fit.power_law.alpha,2)})', ax=fig)

        alpha.append(fit.power_law.alpha)

        p_value.append(fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)[0])

    plt.title("EPS change: Log- Log PDF of Failure Sizes", fontsize = 15)

    # Use ScalarFormatter for log scales


    plt.grid(True)
    plt.xlabel("Failure size", fontsize = 15)
    plt.ylabel("P(s)", fontsize = 15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    #plt.show()
    print(alpha)
    print(p_value)


if value == "failure":
    data_thr = []
    for i in range(3):
        data_thr  = np.load(f"data\{value}5\p0.1-fail{fail[i+1]}-recov0.1-loss0.85.npy")

        fit = powerlaw.Fit(data_thr*100, discrete = True)
        fig = fit.plot_pdf(color=f'C{i+1}', linewidth=2, label=f'Threshold={fail[i+1]}', linestyle = '-')
        fit.power_law.plot_pdf(color=f'C{i+1}', linewidth = 1, linestyle='--', label = f'Threshold ={fail[i+1]} (alpha = {np.round(fit.power_law.alpha,2)})', ax=fig)

        alpha.append(fit.power_law.alpha)

        p_value.append(fit.distribution_compare('power_law', 'exponential', normalized_ratio=True)[1])

    plt.title("Threshold change: Log- Log PDF of Failure Sizes", fontsize = 15)

    # Use ScalarFormatter for log scales


    plt.grid(True)
    plt.xlabel("Failure size", fontsize = 15)
    plt.ylabel("P(s)", fontsize = 15)
    plt.xscale('log')
    plt.yscale('log')
    plt.legend()
    plt.show()
    print(alpha)
    print(p_value)
