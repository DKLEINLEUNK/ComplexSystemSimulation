import json

import matplotlib.pyplot as plt
import numpy as np

import powerlaw

from self_organized_dragon_king import Inoculation


def run_simulation(n_timesteps=15_000, n_nodes=1000, epsilon=0.001, CC=True, export_path='exports/first_results.txt'):

    sim = Inoculation(
        n_steps = n_timesteps,
        n_nodes = n_nodes,
        d_regular = 3,
        epsilon = epsilon,
        complex_contagion = CC,
        visualize = False,
        export_path = export_path
    )

    sim.run()


def plot_results(path='exports/first_results.txt', time=False, size=False, exclude=0):

    # plt.figure()
    # plt.xlabel('$t$')
    # plt.ylabel('$p$')

    # plt.xscale('log')
    # plt.yscale('log')

    # plt.title('Weak Nodes over Time') if time else plt.title('Failure Size Distribution')
    # plt.grid(True)

    # Read the data from the file
    with open(path, 'r') as f:
        line_counter = 0
        failures = []
        weaklings = []
        times = []
        for line in f:
            line_counter += 1

            if line_counter > exclude:
                try:
                    data = json.loads(line)

                    if size:
                        last_entry = list(data.values())[-1]
                        failures.append(last_entry[0])
                        # failures = failures.append(last_entry[0])
                        # for measurements in list(data.values()):
                            # failures.append(measurements[0])

                    if time:
                        weaklings.append(data['0'][1])
                        times.append(line_counter)

                except:
                    # a zero entry is found do nothing
                    pass

    if time:
        plt.figure(figsize=(6, 6))
        plt.plot(times, weaklings, '--', color='green', label='e=0.01')
        plt.ylim(.4, .8)
        plt.xlabel('Time')
        plt.ylabel('Weak Nodes')
        plt.legend()
        plt.grid(True)

    if size:
        fit = powerlaw.Fit(failures)
        # plt.figure(figsize=(12, 6))

        # PDF
        plt.figure(figsize=(6, 6))
        # plt.subplot(1, 2, 1)

        # TODO only plot pdf for a range of the data
        fit.power_law.plot_pdf(color='blue', linestyle='--', label=f'a={fit.power_law.alpha:.2f}')

        bins, proportion = fit.pdf(original_data=True)
        midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

        plt.plot(midpoints, proportion, 'x', color='orange', markersize=5, label='e=0.01')
        plt.xlabel('Proportions')
        plt.ylabel('PDF')
        plt.legend()
        plt.grid(True)

        # # CDF
        # plt.subplot(1, 2, 2)
        # fit.power_law.plot_cdf(color='b', linestyle='--', label='Power law fit')
        # fit.plot_cdf(color='r', label='Empirical Data')
        # plt.xlabel('Proportions')
        # plt.ylabel('CDF')
        # plt.legend()

        plt.tight_layout()
        plt.show()
    # plt.show()


if __name__ == '__main__':

    name = 'exports/IN_n_5e3_t_5e4_e_1e-2.txt'
    run_simulation(n_nodes=5_000, n_timesteps=50_000, epsilon=0.001, CC=False, export_path=name)

    plot_results(path=name, time=True, size=False, exclude=0)
    plot_results(path=name, time=False, size=True, exclude=0)
    # plot_results(path='exports/first_results.txt', time=False, size=True)
