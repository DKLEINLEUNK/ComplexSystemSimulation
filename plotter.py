import json

import matplotlib.pyplot as plt
import numpy as np

from self_organized_dragon_king import Inoculation


def run_simulation(n_timesteps=15_000, n_nodes=1000, export_path='exports/first_results.txt'):

    sim = Inoculation(
        n_steps=n_timesteps,
        n_nodes=n_nodes,
        d_regular = 3,
        epsilon = 0.001,
        complex_contagion=True,
        visualize=False,
        export_path=export_path
    )

    sim.run()


def plot_results(path='exports/first_results.txt', time=True, size=False):
    
    plt.figure()
    plt.xlabel('$t$')
    plt.ylabel('$p$')
    
    # plt.xscale('log')
    # plt.yscale('log')
    
    plt.title('Weak Nodes over Time') if time else plt.title('Failure Size Distribution')
    
    plt.grid(True)

    # Read the data from the file
    with open(path, 'r') as f:
        line_counter = 0
        for line in f:
            line_counter += 1
            try:
                data = json.loads(line)
                
                time = line_counter
                
                failures = []
                for measurements in list(data.values()):
                    failures.append(measurements[0])
                                
                weaklings = data['0'][1]
                
                plt.plot(time, weaklings, 'o', color='blue', markersize=1)

            except:
                # a zero entry is found do nothing
                pass
    
    plt.show()


if __name__ == '__main__':

    run_simulation(export_path='exports/CC_first_run.txt')
    # plot_results()
