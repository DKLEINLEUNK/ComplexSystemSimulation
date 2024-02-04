"""
This file makes the plots of the results of the data of simulate failures of network.py file
"""

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import powerlaw as powerlaw

VARIABLE_TO_PLOT = "failure"
SAVE_PLOTS = True
SHOW_PLOT = False
FILE_NUMBER = 5 ## Which data set to use. 
FAIL = 0.3 ##See in data folder which fail was set for the dataset used
PROB = 0 ## Which probability to plot from probs list, 0-> 0.1, 1-> 0.2. 2-> 0.4 for EPS and Failures plots
PLOT_TYPE = "multi_plot" ## Define plot_type, multi_plot or single_plot


probs = [0.1,0.2,0.4]
color = ["green", "yellow", "red"]

def specs(axs):
    """
    Set specifications of single plots and save the,
    This function must not be executed, is comes derived by single_plot
    Inputs:
        - axs: figure to plot
    """
    axs.set_yscale("log")
    axs.set_xscale('log')
    axs.set_xlabel("s (fraction of failures)", fontsize = 15)
    axs.set_ylabel("$P(s)$", fontsize = 15)
    axs.legend(fontsize = 15)


def multi_plot():
    """
    Creates a multi plot with the three probabilities
    """

    fig, axs = plt.subplots(1, 3, figsize=(25, 5),sharex=True,sharey = True)
    axs = axs.flatten()

    if VARIABLE_TO_PLOT == "EPS":
        var = [0.3, 0.6, 0.7, 0.85, 0.95]
        for j in range(len(probs)):
            axs[j].set_title(f"Erdos-Renyi with p = {probs[j]}")
            for i in  range(len(var)):
                data = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[j]}-fail{FAIL}-recov0.1-loss{var[i]}.npy")
                sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[j], marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")
            specs(axs[j])
    
    elif VARIABLE_TO_PLOT == "recovery":
        var = [0.1,0.4,0.6,1.0]
        for j in range(len(probs)):
            axs[j].set_title(f"Erdos-Renyi with p = {probs[j]}")
            for i in  range(len(var)):
                data = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[j]}-fail{FAIL}-recov{var[i]}-loss0.85.npy")
                sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[j], marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")
            specs(axs[j])

    elif VARIABLE_TO_PLOT == "failure":
        var = [0.1,0.4,0.6,0.8]
        for j in range(len(probs)):
            axs[j].set_title(f"Erdos-Renyi with p = {probs[j]}")
            for i in  range(len(var)):
                data = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[j]}-fail{var[i]}-recov0.1-loss0.85.npy")
                sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[j], marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")
            specs(axs[j])
    
    if SAVE_PLOTS == True:
        plt.savefig(f'figures\\{VARIABLE_TO_PLOT}\\dataset_{FILE_NUMBER}_multi_p.png')
    if SHOW_PLOT == True:
        plt.show()

       
def single_plot():
    """
    Saves and plots a single plot with one porbability
    """
    fig, axs = plt.subplots(1, 1, figsize=(10, 15))
    axs.set_title(f"Erdos-Renyi with p = {probs[PROB]}", fontsize = 15)
    
    if VARIABLE_TO_PLOT == "EPS":
        var = [0.3, 0.6, 0.7, 0.85, 0.95]
        for i in  range(len(var)):
            data = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[PROB]}-fail{FAIL}-recov0.1-loss{var[i]}.npy")
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs, marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")

    elif VARIABLE_TO_PLOT == "recovery":
        var = [0.1,0.4,0.6,1.0]
        for i in  range(len(var)):
            data = np.load( f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[PROB]}-fail{FAIL}-recov{var[i]}-loss0.85.npy")
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs, marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")

    elif VARIABLE_TO_PLOT == "failure":
        var = [0.1,0.4,0.6,0.8]
        for i in  range(len(var)):
            data = np.load(f"data\{VARIABLE_TO_PLOT}{FILE_NUMBER}\p{probs[PROB]}-fail{var[i]}-recov0.1-loss0.85.npy")
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs, marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")
    
    specs(axs)

    if SAVE_PLOTS == True:
        plt.savefig(f'figures\\{VARIABLE_TO_PLOT}\\dataset_{FILE_NUMBER}_single_p{probs[PROB]}.png')
    if SHOW_PLOT == True:
        plt.show()



if __name__ == "__main__":
    if PLOT_TYPE == "multi_plot":
        multi_plot()
    elif PLOT_TYPE == "single_plot":
        single_plot()
    