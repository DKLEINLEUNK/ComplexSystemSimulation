import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import powerlaw as powerlaw

VARIABLE_TO_PLOT = "recovery"


probs = [0.1,0.2,0.4]
color = ["green", "yellow", "red"]

if VARIABLE_TO_PLOT == "EPS":
    fig, axs = plt.subplots(1, 1, figsize=(5, 15))
    var = [0.3, 0.6, 0.7, 0.85, 0.95]
    axs.set_title(f"Erdos-Renyi with p = {probs[0]}", fontsize = 15)
    for i in  range(5):
        data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[0]}-fail0.3-recov0.1-loss{var[i]}.npy")
        sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs, marker='^', linestyle='', markersize=5, label = f"{VARIABLE_TO_PLOT} = {var[i]*100}%")

        axs.set_yscale("log")
        axs.set_xscale('log')
        axs.set_xlabel("s(fraction of failures)", fontsize = 15)
        axs.set_ylabel("$P(s)$", fontsize = 15)
        axs.legend(fontsize = 15)
    plt.show()

elif VARIABLE_TO_PLOT == "recovery":
    fig, axs = plt.subplots(1, 3, figsize=(25, 5),sharex=True,sharey = True)
    var = [0.1,0.4,0.6,1.0]

    for j in range(3):
        axs[j].set_title(f"Erdos-Renyi with p = {probs[j]}")
        for i in  range(4):
            data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[j]}-fail0.3-recov{var[i]}-loss0.85.npy") 
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[j], marker='^', linestyle='', markersize=5, label = f"Recovery = {var[i]*100}%")
            #axs[i].plot(midpoints, proportion, 'v', color= color[j], markersize=7, label='e=0.01')
        axs[j].set_yscale("log")
        axs[j].set_xscale('log')
        axs[j].set_xlabel("s(fraction of failures)", fontsize = 15)
        axs[j].set_ylabel("$P(s)$", fontsize = 15)
        axs[j].legend(fontsize = 15)
        

    plt.show()

elif VARIABLE_TO_PLOT == "failure":
    var = [0.1,0.4,0.6,0.8]
    fig, axs = plt.subplots(1, 1, figsize=(5, 15))
    axs.set_title(f"Erdos-Renyi with p = {probs[0]}",fontsize = 15)
    for i in  range(4):
        data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[0]}-fail{var[i]}-recov0.1-loss0.85.npy")
        sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs,  marker='^', linestyle='', markersize=5, label = f"Threshold = {var[i]*100}%")

    axs.set_yscale("log")
    axs.set_xticks([10**i for i in range(-2, 1)])
    axs.set_xticklabels([f"${tick}$" for tick in [10**i for i in range(-2, 1)]])
    axs.set_xscale('log')
    axs.set_xlabel("s (Fraction of failure)", fontsize = 15)
    axs.set_ylabel("$P(s)$", fontsize = 15)
    plt.legend(fontsize = 15)

  
    plt.show()

