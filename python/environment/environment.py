"""Provides a collaborative problem solving environment."""

###############################################################################
# Imports
###############################################################################
#
import numpy as np
from numpy import random

import environment.nklandscapes as nkl
from environment.actions import ACTION_STR, ACTION_NUM, ACTION_FUNC
from environment.actions import OUTCOME_STR, OUTCOME_NUM


###############################################################################
# Types
###############################################################################
#
# Environment
#
class Environment():
    '''The Environment in which episodes are run.'''
    def __init__(self, num_bits, num_components, graph,
                 deadline, neighbour_sample_size=None, max_processes=1):
        # settings
        self.num_bits = num_bits
        self.num_components = num_components
        self.deadline = deadline

        self.num_nodes = len(graph)
        self.graph = graph

        self.neighbour_sample_size = neighbour_sample_size
        self.max_processes = max_processes

        # Trace Arrays
        self._positions = \
                np.empty((self.num_nodes, self.deadline +1), dtype='I')
        self._actions = np.zeros((self.num_nodes, self.deadline), dtype='I')
        self._outcomes = np.empty((self.num_nodes, self.deadline), dtype='I')

        self._set_random_initial_position()

        # Generate fitness function
        self._fitness_func = None
        self._fitness_func_norm = None
        self.generate_new_fitness_func()

    def generate_new_fitness_func(self):
        """Generates a new fitness function for the environment."""
        (
            self._fitness_func,
            self._fitness_func_norm,
        ) = nkl.generate_fitness_func(
            self.num_bits,
            self.num_components,
            num_processes=self.max_processes,
        )

    def reset(self):
        """Resets the environment.
        Prepares the environment for a new episode.
        """
        self._reset_trace_arrays()
        self._set_random_initial_position()

    def _reset_trace_arrays(self):
        """Change all values in environment's trace arrays to zero."""
        self._positions *= 0
        self._actions *= 0
        self._outcomes *= 0

    def _set_random_initial_position(self):
        '''Sets a random position for each node at time 0.'''
        for node in range(self.num_nodes):
            self._positions[node, 0] = random.randint(2**self.num_bits)

    def get_node_action(self, node, time):
        """Returns the action taken or to be taken by the given node at the
        given time.
        """
        return self._actions[node, time]

    def get_node_position(self, node, time):
        """Returns the given node's position at the given time."""
        return self._positions[node, time]

    def get_fitness(self, position):
        """Returns the fitness of a given position."""
        return self._fitness_func[position]

    def get_node_fitness(self, node, time):
        """Returns the given node's fitness at the given time."""
        return self._fitness_func[self._positions[node, time]]

    def get_node_fitness_norm(self, node, time):
        """Returns the given node's fitness norm at the given time.
        The fitness norm is the fitness value after being normalised
        but before being passed through a monotonic function.
        """
        return self._fitness_func_norm[self._positions[node, time]]

    def get_all_positions(self):
        """Returns the position of each node at each time."""
        return self._positions

    def get_all_fitnesses(self):
        """Returns the fitness of each node at each time."""
        fitnesses = np.empty((self.num_nodes, self.deadline +1))
        for time in range(self.deadline +1):
            for node in range(self.num_nodes):
                fitnesses[node, time] = \
                    self._fitness_func[self._positions[node, time]]
        return fitnesses

    def get_mean_fitnesses(self):
        """Returns the mean fitness across all nodes at each time step."""
        return np.mean(self.get_all_fitnesses(), axis=0)

    def get_neighbours(self, node):
        """Returns a node's neighbours."""
        neighbours = list(self.graph.neighbors(node))
        if self.neighbour_sample_size is not None:
            if len(neighbours) > self.neighbour_sample_size:
                neighbours = random.choice(
                    neighbours,
                    size=self.neighbour_sample_size,
                    replace=False,
                )
        return neighbours

    def get_neighbours_fitnesses(self, node, time):
        """Returns a node's neighbours."""
        neighbours = self.get_neighbours(node)

        neighbours_fitnesses = [
            self._fitness_func[self._positions[neighbour, time]]
            for neighbour in neighbours
        ]
        return neighbours, neighbours_fitnesses

    def set_action(self, node, time, action_num):
        """Sets the given action for the given node to perform
        at the given time.
        """
        self._actions[node, time] = action_num

    def set_all_actions(self, action_num):
        """Sets the given action for every node at every time."""
        self._actions = (self._actions * 0) + action_num

    def run_time_step(self, time_step):
        """Run a single time step of the episode."""
        for node in range(self.num_nodes):
            action = self._actions[node, time_step]

            # carry out action
            (
                    self._outcomes[node, time_step], # action outcome
                    self._positions[node, time_step +1], # next position
                    _,
            ) = ACTION_FUNC[action](self, node, time_step)

    def run_episode(self):
        """Run the whole episode"""
        for time in range(self.deadline):
            self.run_time_step(time)


###############################################################################
# Simulation Environment Functions
###############################################################################
#
def get_action_num(action_name):
    """Returns the action number of the action with the given name."""
    return ACTION_NUM[action_name]

def draw_outcomes_stack_plot(num_nodes, deadline, outcomes, axis):
    """
    Plots a stack plot of the proportion of different outcomes over time.
    """
    outcome_frequencies = {}
    for outcome_name in OUTCOME_NUM:
        outcome_frequencies[outcome_name] = \
                np.zeros(deadline, dtype='I')

    for time in range(deadline):
        for node in range(num_nodes):
            outcome_str = OUTCOME_STR[outcomes[node, time]]
            outcome_frequencies[outcome_str][time] += 1

    axis.stackplot(
        range(deadline),
        outcome_frequencies.values(),
        labels=outcome_frequencies.keys(),
    )

def draw_outcomes_bar_plot(num_nodes, deadline, outcomes, axis):
    """
    Plots a bar plot of the proportion of different outcomes over time.
    """
    outcome_frequencies = {}
    for outcome_name in OUTCOME_NUM:
        outcome_frequencies[outcome_name] = \
                np.zeros(deadline, dtype='I')

    for time in range(deadline):
        for node in range(num_nodes):
            outcome_str = OUTCOME_STR[outcomes[node, time]]
            outcome_frequencies[outcome_str][time] += 1

    cumulative_values = np.zeros(deadline)
    for label, values in outcome_frequencies.items():
        axis.bar(
            range(deadline),
            values,
            bottom=cumulative_values,
            label=label,
        )
        cumulative_values = cumulative_values + values

def draw_actions_stack_plot(num_nodes, deadline, actions, axis):
    """
    Plots a stack plot of the proportion of different actions over time.
    """
    action_frequencies = {}
    for action_name in ACTION_NUM:
        action_frequencies[action_name] = \
                np.zeros(deadline, dtype='I')

    for time in range(deadline):
        for node in range(num_nodes):
            action_str = ACTION_STR[actions[node, time]]
            action_frequencies[action_str][time] += 1

    axis.stackplot(
        range(deadline),
        action_frequencies.values(),
        labels=action_frequencies.keys(),
    )

def draw_actions_bar_plot(num_nodes, deadline, actions, axis):
    """
    Plots a bar plot of the proportion of different actions over time.
    """
    action_frequencies = {}
    for action_name in ACTION_NUM:
        action_frequencies[action_name] = \
                np.zeros(deadline, dtype='I')

    for time in range(deadline):
        for node in range(num_nodes):
            action_str = ACTION_STR[actions[node, time]]
            action_frequencies[action_str][time] += 1

    cumulative_values = np.zeros(deadline)
    for label, values in action_frequencies.items():
        axis.bar(
            range(deadline),
            values,
            bottom=cumulative_values,
            label=label,
        )
        cumulative_values = cumulative_values + values
