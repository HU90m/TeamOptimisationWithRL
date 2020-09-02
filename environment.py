"""A module for simulation a collaborative problem solving environment."""

###############################################################################
# Imports
###############################################################################
#
from multiprocessing import sharedctypes
import numpy as np
from numpy import random

from actions import ACTION_STR, ACTION_NUM
from actions import OUTCOME_STR, OUTCOME_NUM


###############################################################################
# Types
###############################################################################
#
# Simulation Record
#
class SimulationRecord():
    '''
    Stores the positions,
    fitnesses and actions of each agent/node at each time.
    '''
    def __init__(self, num_bits, num_nodes, deadline,
                 fitness_func, fitness_func_norm, num_processes=1):
        self.num_bits = num_bits
        self.num_nodes = num_nodes
        self.deadline = deadline

        self.fitness_func = fitness_func
        self.fitness_func_norm = fitness_func_norm

        if num_processes < 2:
            self.positions = np.empty((num_nodes, deadline), dtype='I')
            self.fitnesses = np.empty((num_nodes, deadline))
            self.actions = np.empty((num_nodes, deadline), dtype='I')
            self.outcomes = np.empty((num_nodes, deadline), dtype='I')

        else:
            # make shared memory
            positions_shared = sharedctypes.RawArray('I', num_nodes*deadline)
            fitnesses_shared = sharedctypes.RawArray('d', num_nodes*deadline)
            actions_shared = sharedctypes.RawArray('I', num_nodes*deadline)
            outcomes_shared = sharedctypes.RawArray('I', num_nodes*deadline)

            # abstract shared memory to numpy array
            self.positions = np.frombuffer(
                positions_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

            self.fitnesses = np.frombuffer(
                fitnesses_shared,
                dtype='d',
            ).reshape((num_nodes, deadline))

            self.actions = np.frombuffer(
                actions_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

            self.outcomes = np.frombuffer(
                outcomes_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

    def set_random_initial_position(self):
        '''Sets a random position for each node at time 0.'''
        for node in range(self.num_nodes):
            self.positions[node, 0] = random.randint(2**self.num_bits)
            self.actions[node, 0] = ACTION_NUM['teleport']
            self.outcomes[node, 0] = OUTCOME_NUM['teleported']

    def fill_fitnesses(self):
        """Fills the fitnesses for each node and time."""
        for time in range(self.deadline):
            for node in range(self.num_nodes):
                self.fitnesses[node, time] = \
                    self.fitness_func[self.positions[node, time]]

    def save(self, file_name):
        """Saves the simulation record as a npz archive."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                num_bits=self.num_bits,
                num_nodes=self.num_nodes,
                deadline=self.deadline,
                positions=self.positions,
                fitnesses=self.fitnesses,
                actions=self.actions,
                outcomes=self.outcomes,
                fitness_func=self.fitness_func,
                fitness_func_norm=self.fitness_func_norm,
            )

    def load(self, file_name):
        """Loads a saved simulation record."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)

            self.num_bits = int(file_content['num_bits'])
            self.num_nodes = int(file_content['num_nodes'])
            self.deadline = int(file_content['deadline'])
            self.positions = file_content['positions']
            self.fitnesses = file_content['fitnesses']
            self.actions = file_content['actions']
            self.outcomes = file_content['outcomes']
            self.fitness_func = file_content['fitness_func']
            self.fitness_func_norm = file_content['fitness_func_norm']

    def draw_outcomes_stack_plot(self, axis):
        """
        Plots a stack plot of the proportion of different outcomes over time.
        """
        outcome_frequencies = {}
        for outcome_name in OUTCOME_NUM:
            outcome_frequencies[outcome_name] = \
                    np.zeros(self.deadline, dtype='I')

        for time in range(self.deadline):
            for node in range(self.num_nodes):
                outcome_str = OUTCOME_STR[self.outcomes[node, time]]
                outcome_frequencies[outcome_str][time] += 1

        axis.stackplot(
            range(self.deadline),
            outcome_frequencies.values(),
            labels=outcome_frequencies.keys(),
        )

    def draw_outcomes_bar_plot(self, axis):
        """
        Plots a bar plot of the proportion of different outcomes over time.
        """
        outcome_frequencies = {}
        for outcome_name in OUTCOME_NUM:
            outcome_frequencies[outcome_name] = \
                    np.zeros(self.deadline, dtype='I')

        for time in range(self.deadline):
            for node in range(self.num_nodes):
                outcome_str = OUTCOME_STR[self.outcomes[node, time]]
                outcome_frequencies[outcome_str][time] += 1

        cumulative_values = np.zeros(self.deadline)
        for label, values in outcome_frequencies.items():
            axis.bar(
                range(self.deadline),
                values,
                bottom=cumulative_values,
                label=label,
            )
            cumulative_values = cumulative_values + values

    def draw_actions_stack_plot(self, axis):
        """
        Plots a stack plot of the proportion of different actions over time.
        """
        action_frequencies = {}
        for action_name in ACTION_NUM:
            action_frequencies[action_name] = \
                    np.zeros(self.deadline, dtype='I')

        for time in range(self.deadline):
            for node in range(self.num_nodes):
                action_str = ACTION_STR[self.actions[node, time]]
                action_frequencies[action_str][time] += 1

        axis.stackplot(
            range(self.deadline),
            action_frequencies.values(),
            labels=action_frequencies.keys(),
        )

    def draw_actions_bar_plot(self, axis):
        """
        Plots a bar plot of the proportion of different actions over time.
        """
        action_frequencies = {}
        for action_name in ACTION_NUM:
            action_frequencies[action_name] = \
                    np.zeros(self.deadline, dtype='I')

        for time in range(self.deadline):
            for node in range(self.num_nodes):
                action_str = ACTION_STR[self.actions[node, time]]
                action_frequencies[action_str][time] += 1

        cumulative_values = np.zeros(self.deadline)
        for label, values in action_frequencies.items():
            axis.bar(
                range(self.deadline),
                values,
                bottom=cumulative_values,
                label=label,
            )
            cumulative_values = cumulative_values + values


###############################################################################
# Main Function
###############################################################################
#
def run_episode(
        graph,
        sim_record,
        action_func,
        neighbour_sample_size=None,
):
    """A single episode of the simulation."""

    num_nodes = len(graph)

    sim_record.set_random_initial_position()

    for time in range(sim_record.deadline -1):
        for node in range(num_nodes):
            # Find the node's neighbours
            neighbours = list(graph.neighbors(node))
            if neighbour_sample_size is not None:
                if len(neighbours) > neighbour_sample_size:
                    neighbours = random.choice(
                        neighbours,
                        size=neighbour_sample_size,
                        replace=False,
                    )

            # carry out action
            action_func(time, node, sim_record, neighbours)

    return sim_record
