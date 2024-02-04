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
        plt.figure(figsize=(3, 3))

        times = np.array(times) / 50_000

        plt.plot(times, weaklings, '-', color='purple', label='e=0.01')
        plt.ylim(0, 1)
        plt.xlabel('$t$', fontsize=12)
        plt.ylabel('p(t)', fontsize=12)
        plt.tight_layout()
        plt.show()

    if size:

        failures = np.array(failures) * 5_000

        xmin = 10
        xmax = 200

        fit = powerlaw.Fit(failures, discrete=True, xmin=xmin, xmax=xmax)

        plt.figure(figsize=(3, 3))
        plt.xscale('log')
        plt.yscale('log')
        alpha = fit.power_law.alpha
        xmin = fit.xmin

        # Generate x-values from xmin to the maximum value in your data or beyond, as needed
        x = np.linspace(xmin, 500, 1000)

        # Calculate the PDF of the power law using the formula: C*x^(-alpha)
        # Where C is a normalization constant. For simplicity, we'll focus on the shape,
        # so the exact value of C is not critical for visual representation.
        pdf = (x ** (-alpha))

        # Offset the PDF by a small factor to "lift" the line up
        offset_factor = 1.2
        offset_pdf = pdf * offset_factor

        # fit.power_law.plot_pdf(color='purple', linestyle='--', label=f'a={fit.power_law.alpha:.2f}')

        bins, proportion = fit.pdf(original_data=True)
        midpoints = [(bins[i] + bins[i+1]) / 2 for i in range(len(bins) - 1)]

        plt.plot(midpoints, proportion, 'v', color='yellow', markersize=7, label='e=0.01')

        # Plot the adjusted fitted power law
        plt.plot(x, offset_pdf, color='purple', linestyle='--', label=f'Adjusted a={alpha:.2f}')

        x_text = np.max(x) * 0.5 - 100
        y_text = np.max(offset_pdf) * 0.4

        # Add text for alpha
        plt.text(x_text, y_text, f'$\\alpha = {alpha:.2f}$', fontsize=12, verticalalignment='top')

        plt.xlabel('$s$', fontsize=12)
        plt.ylabel('$p(s)$', fontsize=12)
        plt.tight_layout()
        plt.show()


if __name__ == '__main__':

    name = 'exports/CC_n_5e3_t_5e4_e_1e-2.txt'
    run_simulation(n_nodes=5_000, n_timesteps=50_000, epsilon=0.001, CC=False, export_path=name)

    plot_results(path=name, time=True, size=False, exclude=0)
    plot_results(path=name, time=False, size=True, exclude=0)
