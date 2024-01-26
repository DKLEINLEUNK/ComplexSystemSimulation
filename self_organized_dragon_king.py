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

from network import Network
from network_modifier import degrade, fail, save_state, load_state, reinforce


class Inoculation:
    
    '''
    Class to simulate the inoculation version of the self-organized dragon king model.
    '''

    def __init__(self, n_steps, n_trials, n_nodes, n_edges=None, pr_edge=None, d_regular=None, BA=False, epsilon=0.2, verbose=False, visualize=False, export_dir=None):
        '''
        Description
        -----------
        Initializes the model and simulation environment.

        Parameters
        ----------
        `n_steps` : int
            The number of steps in a trial.
        `n_trials` : int
            The number of trials.
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
        `export_dir` : str
            Whether the resulting failure size distributions should be exported. If `None` (default), no export will take place. 
            Otherwise, the string should contain the path to the export directory (e.g. 'exports/').
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
        self.n_trials = n_trials
        
        # Initialize the current step
        self.time = 0

        # Initialize flags
        self.verbose = verbose
        self.visualize = visualize
        self.export_dir = export_dir
        self.exporting = False

        # Initialize the export directory
        if export_dir is not None:
            self._initialize_results()


    def run(self):
        '''
        Description
        -----------
        Runs the simulation.
        '''
        # Loop through trials
        for self._itrial in np.arange(1, self.n_trials + 1):
            
            # Loop through the number of steps
            for self._istep in np.arange(1, self.n_steps + 1):
                
                if not self.verbose:
                    # If verbose is False (default), display a progress bar
                    print(f"Starting trial {self._itrial} of {self.n_trials}  |  Step {self._istep} of {self.n_steps}               ", end='\r')
                
                # Execute a single step
                self.step()

            # # Re-initialize the network
            # self.network = Network(self.n_nodes, self.n_edges, self.pr_edge)

        print('\nSimulation completed.')
        
        if self.exporting:
            self._export_results()
            print(f'Exported results to {self.export_dir}results.csv')


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

            if self.visualize:
                return self._visualize_network()

        # Otherwise (i.e. no failure), end the step
        else:
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
        # Get current statuses as values
        current_state = np.array(list(self.network.get_all_statuses().values()))

        # Initialize previous state to save
        previous_state = np.zeros_like(current_state)
        
        # Start tracking cascade
        self._initialize_cascade()  if self.exporting  else None
        
        ### NOTE ###
        # 
        # The following cascading mechanism is a bit of a mess.
        # 
        # I think it can be simplified by circumventing the while loop
        # and just focusing on the final state of the network.
        #
        # In particular, as we know the neighbors of each node, we can 
        # concatenate all the neighbors and immediately adjust their 
        # statuses.
        # 
        ############

        # Cycle until no more changes in status occur
        while not (previous_state == current_state).all():
            
            previous_state = current_state
            
            # Find failed nodes
            failed_nodes = np.where(current_state==0)[0]
            
            # Locate neighbors
            all_neighbors = self.network.get_multiple_neighbors(failed_nodes)
            
            # Cycle through neighbors (multiple neighbors of multiple nodes) and locate weaklings
            for neighbors in all_neighbors:

                fail(self.network, list(neighbors))

            # Update current_state
            current_state = np.array(list(self.network.get_all_statuses().values()))

            # Visualize network
            self._visualize_network() if self.visualize else None

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
        if len(status_values) != self.n_nodes:
            raise ValueError("Length of all statuses does not match the specified number of nodes.")

        failures = status_values.count(0)

        return self.cascade_counter, failures / len(status_values)


    def _store_cascade(self):
        '''
        Description
        -----------
        Stores a value in the cascade dict.
        '''
        self.cascade_counter += 1
        t, x = self._get_cascade()
        self.cascade_dict[t] = x


    def _visualize_network(self):
        '''
        Description
        -----------
        Visualize the network.
        
        red: failed nodes.
        yellow: weak nodes.  
        green: strong nodes.
        '''
        nx.draw(
                self.network.graph, 
                node_color=['red' if self.network.graph.nodes[node]['status'] == 0 else ('yellow' if self.network.graph.nodes[node]['status'] == 1 else 'green') for node in self.network.graph.nodes]
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
        self.results = [[0]*self.n_steps]*self.n_trials


    def _store_step_results(self):
        '''
        Description
        -----------
        Stores results of a step.
        '''
        self.results[self._itrial - 1][self._istep - 1] = self.cascade_dict
        print(f"Stored result for trial {self._itrial} and step {self._istep}.") if self.verbose else None


    def _export_results(self):
        '''
        Description
        -----------
        Exports stored results to csv file.
        '''
        with open(f'{self.export_dir}results.txt', 'w') as file:
            for trial in np.arange(self.n_trials):
                for step in np.arange(self.n_steps):
                    # print(f"\nExporting result for trial {trial} and step {step}.")
                    json.dump(self.results[trial][step], file)
                    file.write('\n')


def complex_contagion():
    pass


if __name__ == "__main__":

    start = time.time()

    ### EXAMPLE USAGE ###
    n_steps = 10
    n_nodes = 100_000  # 10^5
    n_edges = 300_000

    simulation = Inoculation(
        n_steps=n_steps, 
        n_trials=1, 
        n_nodes=n_nodes,
        n_edges=None,
        pr_edge=None,
        d_regular=None,
        BA=True,
        epsilon=0.001, 
        verbose=False, 
        visualize=False,
        export_dir=None
    )

    # # Visualize the network at t = 0
    # simulation._visualize_network()
    
    simulation.run()

    execution_time = time.time() - start

    with open('execution_times.txt', 'a') as file:
        file.write( f'{n_nodes}, {n_edges}, {execution_time / n_steps}\n' )
