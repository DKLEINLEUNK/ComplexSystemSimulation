'''
This module contains an implementation of the 'self-organized dragon king' model by Lin et al (2018).

The model consists of two versions:
    1. Inoculation or IN:
        a. Nodes of status 1 (weak) fail if 1 or more neighbors fail.
        b. Nodes of status 2 (strong) cannot fail.
    
    2. TODO Complex contagion or CC:
        a. Nodes of status 1 (weak) fail if 1 or more neighbors fail.
        b. Nodes of status 2 (strong) fail if 2 or more neighbors fail.

Running this module as a script run an example.
'''

import json
import time

import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import powerlaw

from network import Network
from network_modifier import degrade, fail, save_state, load_state, reinforce


class Inoculation:
    
    '''
    Class to simulate the inoculation version of the self-organized dragon king model.
    '''

    def __init__(self, n_steps, n_nodes, n_edges=None, pr_edge=None, d_regular=None, BA=False, epsilon=0.2, complex_contagion=False, verbose=False, visualize=False, export_path=None):
        '''
        Description
        -----------
        Initializes the model and simulation environment.

        Parameters
        ----------
        `n_steps` : int
            The number of steps in a trial.
        `n_nodes` : int
            The number of nodes in the network.
        `n_edges` : int
            The number of edges in the network.
        `pr_edge` : float
            The probability that two nodes will be connected.
        `d_regular` : int
            The degree of each node in the regular random network.
        `epsilon` : float
            The probability that a weak node will be repaired as strong.
        `verbose` : bool
            Whether whether, per step, progression should be broadcasted. Replaces the default progress bar. 
        `visualize` : 
            Whether, per step, progression should be plotted. CAUTION only use with very small `n_nodes`.
        `export_path` : str
            Whether the resulting failure size distributions should be exported. If `None` (default), no export will take place. 
            Otherwise, the string should contain the path to the export directory (e.g. 'exports/example.txt').
        '''

        # Initialize the network
        self.n_nodes = n_nodes
        self.n_edges = n_edges
        self.pr_edge = pr_edge
        self.network = Network(n_nodes, n_edges, pr_edge, d_regular, BA)
        
        # Initialize the state
        self.network.set_all_statuses(1)
        
        # Initialize the parameters
        self.epsilon = epsilon
        self.n_steps = n_steps
        
        # Initialize the current step
        self.time = 0

        # Initialize flags
        self.complex_contagion = complex_contagion
        self.verbose = verbose
        self.visualize = visualize
        self.export_path = export_path
        self.exporting = False

        # Visualize initial network
        self._visualize_network() if self.visualize else None

        # Initialize the export directory
        if export_path is not None:
            self._initialize_results()


    def run(self):
        '''
        Description
        -----------
        Runs the simulation.
        '''
            
        # Loop through the number of steps
        while self.time < self.n_steps:
            
            if not self.verbose:
                # If verbose is False (default), display a progress bar
                print(f"t = {self.time} of {self.n_steps}              ", end='\r')
            
            # Execute a single step
            self.step()

        print('\nSimulation completed.')
        
        if self.exporting:
            self._export_results()
            print(f"Exported results for S to '{self.export_path}'.")


    def step(self):
        '''
        Description
        -----------
        Runs a single step of the simulation.
        '''
        
        # Save the current state (only used in cases of failure during degradation)
        before_degrade = save_state(self.network)

        # Degrade a node at random
        degrade(self.network, random=True)

        if self.contains_failed_nodes():
            
            # Store first failure size
            # self._store_results(1, self.trial, step) if self.exporting else None

            # Cascade failures until no more failures occur
            failed_nodes = self.cascade_failures()

            # Repair nodes
            load_state(self.network, before_degrade)

            # Reinforce some of the failed nodes with status 1 given epsilon
            reinforce(self.network, failed_nodes, self.epsilon)

            self.time += 1
            if self.visualize:
                return self._visualize_network()
            

        # Otherwise (i.e. no failure), end the step
        else:

            self.time += 1
            
            return self._visualize_network() if self.visualize else None
    

    def cascade_failures(self):
        '''
        Description
        -----------
        Cascades failures until no more failures occur.

        Returns
        -------
        An array containing all failed nodes of status 1.
        '''
        # Visualize network
        self._visualize_network() if self.visualize else None

        # Get current statuses as values
        current_state = nx.get_node_attributes(self.network.graph, 'status')

        # Initialize previous state to save
        previous_state = None
        
        # Start tracking cascade
        self._initialize_cascade()  if self.exporting  else None
        
        # Cycle until no more changes in status occur
        while not previous_state == current_state:

            previous_state = current_state

            # Find vulnerable neighbors of failed nodes
            failed_nodes = {key:val for key, val in current_state.items() if val == 0}
            neighbors = set().union(*(self.network.graph.neighbors(n) for n in failed_nodes))
            statuses = nx.get_node_attributes(self.network.graph, 'status')
            vulnerables = {key for key in neighbors if statuses[key] == 1}
            
            if self.complex_contagion:
                strong_neighbors = {key for key in neighbors if statuses[key] == 2}
                for node in strong_neighbors:
                    if len(set(self.network.graph.neighbors(node)).intersection(failed_nodes)) >= 2:
                        vulnerables.add(node)

            fail(self.network, vulnerables)

            # Visualize network
            self._visualize_network() if self.visualize else None

            # Update current_state
            current_state = nx.get_node_attributes(self.network.graph, 'status')

            # Store current cascade iteration
            self._store_cascade()  if self.exporting  else None

        # Store results of step
        self._store_step_results() if self.exporting else None

        return failed_nodes
    

    def contains_failed_nodes(self):
        '''
        Description
        -----------
        Checks if the network contains failed nodes.

        Returns
        -------
        `True` if the network contains failed nodes, `False` otherwise.
        '''
        print(f'Result of contains_failed_nodes method: {0 in self.network.get_all_statuses().values()}') if self.verbose else None
        print(f'Found the following failured nodes: {np.argwhere(np.array(self.network.get_all_statuses().values()) == 0)}') if self.verbose else None
        return 0 in self.network.get_all_statuses().values()
    

    def _initialize_cascade(self):
        '''
        Description
        -----------
        Initialize a failure size array.
        '''
        self.cascade_dict = {}
        self.cascade_counter = -1
        self._store_cascade()


    def _get_cascade(self):
        '''
        Description
        -----------
        Returns the failure size and counter of current cascade.
        '''
        status_values = list(self.network.get_all_statuses().values())

        # TODO the below value error gets raised
        # if len(status_values) != self.n_nodes:
        #     raise ValueError("Length of all statuses does not match the specified number of nodes.")

        failures = status_values.count(0)
        weaklings = status_values.count(1)

        return self.cascade_counter, failures / len(status_values), weaklings / len(status_values)


    def _store_cascade(self):
        '''
        Description
        -----------
        Stores a value in the cascade dict.
        '''
        self.cascade_counter += 1
        t, s, w = self._get_cascade()
        self.cascade_dict[t] = s, w


    def _visualize_network(self):
        '''
        Description
        -----------
        Visualize the network.
        
        red: failed nodes.
        yellow: weak nodes.  
        green: strong nodes.
        '''
        # Draw the network with labels

        nx.draw(
                self.network.graph, 
                node_color=['red' if self.network.graph.nodes[node]['status'] == 0 else ('yellow' if self.network.graph.nodes[node]['status'] == 1 else 'green') for node in self.network.graph.nodes],
                with_labels=True
                )
        plt.title(f"Time Step: {self.time}")
        plt.show()


    def _initialize_results(self):
        '''
        Description
        -----------
        Initializes the results array. 

        Used to track failure sizes over time, starting from 1st node failure.

        TODO switch to a pickle/file based approach to secure intermittent results.
        '''
        self.exporting = True
        self.results = [0]*self.n_steps

    def _store_step_results(self):
        '''
        Description
        -----------
        Stores results of a step.
        '''
        self.results[self.time] = self.cascade_dict


    def _export_results(self):
        '''
        Description
        -----------
        Exports stored results to csv file.
        '''
        with open(f'{self.export_path}', 'w') as file:
            for step in np.arange(self.n_steps):
                json.dump(self.results[step], file)
                file.write('\n')


def complex_contagion():
    pass


if __name__ == "__main__":

    complex_simulation = Inoculation(
        n_steps=3,
        n_nodes=10,
        d_regular = 3,
        epsilon = 0.1,
        complex_contagion=False,
        visualize=True,
        export_path='exports/trial.txt'
    )

    complex_simulation.run()

    # def format_e(n):
    #     a = '%E' % n
    #     a = a.split('E')[0].rstrip('0').rstrip('.') + 'E' + a.split('E')[1]
    #     a = a.replace('+', '')
    #     a = a.replace('E', 'e')
    #     return a

    
    # start = time.time()

    ### EXAMPLE USAGE ###
    # N = np.logspace(1, 5, 10)  # For different network sizes
    
    # for n in N:
    #     if n < 10:
    #         continue 
    #     name_N = format_e(n)
        
    #     print(f'Running simulation for n = {name_N}')
        
    #     simulation = Inoculation(
    #         n_steps=10_000,
    #         n_trials=1,
    #         n_nodes=n,
    #         d_regular = 3,
    #         epsilon = 0.001,
    #         export_path=f'data/N{str(name_N)}.txt'
    #     )

    #     simulation.run()

    # # execution_time = time.time() - start

    # # with open('execution_times.txt', 'a') as file:
    # #     file.write( f'{n_nodes}, {n_edges}, {execution_time / n_steps}\n' )
