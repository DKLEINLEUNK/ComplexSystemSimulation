import json

import matplotlib.pyplot as plt
import numpy as np


def plot_results(n_nodes=10_000):
    # Initialize the plot
    plt.figure()
    plt.xlabel('s')
    plt.ylabel('Pr(s)')
    # plt.xscale('log')
    plt.yscale('log')
    plt.title('Cascade Size Distribution')
    plt.grid(True)

    # Read the data from the file
    with open('exports/results.txt', 'r') as f:
        for line in f:
            try:
                data = json.loads(line)
                # plt.plot(np.array(list(data.values()))* n_nodes, list(data.values()))
                plt.plot(dict(data).keys(), data.values())
                # print(dict(data).keys())
            except:
                # a zero entry is found
                pass
    
    plt.show()
            

def plot_algorithm_efficiency():
    # Initialize the plot
    plt.figure()
    plt.xlabel('number of nodes')
    plt.ylabel('time per step')
    plt.xscale('log')
    # plt.yscale('log')
    plt.title('Algorithm Efficiency')
    plt.grid(True)

    nodes = []
    times = []

    # Read the data from the file
    filename = ['execution_times_3R.txt', 'execution_times_ER.txt', 'execution_times_BA.txt']
    color = ['blue', 'red', 'green']
    names = ['3R', 'ER', 'BA']
    for i in range(3):
        with open(filename[i], 'r') as f:
            for line in f:
                data = line.rstrip('\n').split(', ')
                nodes.append(int(data[0]))
                times.append(float(data[2]))
            
        plt.plot(nodes, times, color=color[i], marker='x', label=names[i])
        nodes = []
        times = []

    plt.legend()    
    plt.show()

# plot_algorithm_efficiency()


def plot_failure_size_distribution(S, Pr_S):
    # Initialize the plot
    plt.figure()
    plt.xlabel('s')
    plt.ylabel('Pr(s)')
    plt.xscale('log')
    plt.yscale('log')

    plt.plot(S, Pr_S, marker='x')
    plt.show()


if __name__ == '__main__':

    N = np.linspace(0, 100_000, 100)
    