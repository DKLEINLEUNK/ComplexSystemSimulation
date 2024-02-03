import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import powerlaw as powerlaw

VARIABLE_TO_PLOT = "recovery"

fig, axs = plt.subplots(2, 2, figsize=(15, 15),sharex=True)
axs = axs.flatten()
probs = [0.1,0.2,0.4]
color = ["green", "yellow", "red"]

if VARIABLE_TO_PLOT == "EPS":
    var = [0.3, 0.6, 0.7, 0.85, 0.9]

    for i in  range(4):
        axs[i].set_title(f"{VARIABLE_TO_PLOT} = {var[i]*100}%")
        for j in range(3):
            data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[j]}-fail0.3-recov0.1-loss{var[i]}.npy")
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[i], marker='^', linestyle='', markersize=2, label = f"p = {probs[j]}")

        axs[i].set_yscale("log")
        axs[i].set_xscale('log')
        axs[i].set_xlabel("$s$")
        axs[i].set_ylabel("$P(s)$")
        axs[i].legend()
    plt.show()

elif VARIABLE_TO_PLOT == "recovery":
    var = [0.1,0.4,0.6,1.0]

    for i in  range(4):
        axs[i].set_title(f"Recovery rate = {var[i]*100}%")
        for j in range(3):
            data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[j]}-fail0.3-recov{var[i]}-loss0.85.npy")
            fig = fit.plot_pdf(color='b', linewidth=2)
            fit.power_law.plot_pdf(color='b', linestyle='--', ax=axs[i])
            print("Power-law alpha:", fit.power_law.alpha)
            print("Power-law D (KS test):", fit.power_law.D)
            print("P-value (KS test):", fit.power_law.p)
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[i], marker='^', linestyle='', markersize=2, label = f"p = {probs[j]}")
            #axs[i].plot(midpoints, proportion, 'v', color= color[j], markersize=7, label='e=0.01')

        axs[i].set_yscale("log")
        axs[i].set_xscale('log')
        axs[i].set_xlabel("$s$")
        axs[i].set_ylabel("$P(s)$")
        axs[i].legend()
        

    plt.show()

elif VARIABLE_TO_PLOT == "failure":
    var = [0.1,0.4,0.6,0.8]

    for i in  range(4):
        axs[i].set_title(f"Threshold = {var[i]*100} %")
        for j in range(3):
            data = np.load(f"data\{VARIABLE_TO_PLOT}5\p{probs[j]}-fail{var[i]}-recov0.1-loss0.85.npy")
            sns.kdeplot(data, bw_adjust=0.5, cut=0, ax = axs[i], marker='^', linestyle='', markersize=2, label = f"p = {probs[j]}")
        axs[i].set_yscale("log")
        axs[i].set_xscale('log')
        axs[i].set_xlabel("$s$")
        axs[i].set_ylabel("$P(s)$")
        axs[i].legend()
    plt.show()

