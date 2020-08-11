"""A module for simulation a collaborative problem solving environment."""

###############################################################################
# Imports
###############################################################################
#
import multiprocessing as mp
from multiprocessing import sharedctypes
from collections import Counter
import numpy as np
from numpy import random
import bitmanipulation as bitm



###############################################################################
# Constants
###############################################################################
#
# Actions
#
# action index -> action name
ACTION_STR = [
        'step',
        'teleport',
        'best',
        'modal',
        'best_then_step',
        'step_then_best',
        'modal_then_step',
]
# action name -> action index
ACTION_NUM = {
        'step' : 0,
        'teleport' : 1,
        'best' : 2,
        'modal' : 3,
        'best_then_step' : 4,
        'step_then_best' : 5,
        'modal_then_step' : 6,
}
# action index -> action function
ACTION_FUNC = {}

#
# Outcomes
#
# outcome index -> outcome name
OUTCOME_STR = [
        'no change',
        'stepped',
        'teleported',
        'copied best',
        'copied a modal',
]
# outcome name -> outcome index
OUTCOME_NUM = {
        'no change' : 0,
        'stepped' : 1,
        'teleported' : 2,
        'copied best' : 3,
        'copied a modal' : 4,
}


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
    def __init__(self, num_nodes, deadline, num_processes=1):
        self.num_nodes = num_nodes
        self.deadline = deadline

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

    def set_random_initial_position(self, num_bits, fitness_func):
        '''Sets a random position for each node at time 0.'''
        for node in range(self.num_nodes):
            self.positions[node, 0] = random.randint(2**num_bits)
            self.fitnesses[node, 0] = \
                    fitness_func[self.positions[node, 0]]
            self.actions[node, 0] = ACTION_NUM['teleport']
            self.outcomes[node, 0] = OUTCOME_NUM['teleported']

    def save(self, file_name):
        """Saves the simulation record as a npz archive."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                num_nodes=self.num_nodes,
                deadline=self.deadline,
                positions=self.positions,
                fitnesses=self.fitnesses,
                actions=self.actions,
                outcomes=self.outcomes,
            )

    def load(self, file_name):
        """Loads a saved simulation record."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)

            self.num_nodes = int(file_content['num_nodes'])
            self.deadline = int(file_content['deadline'])
            self.positions = file_content['positions']
            self.fitnesses = file_content['fitnesses']
            self.actions = file_content['actions']
            self.outcomes = file_content['outcomes']

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
# Action Function Components
###############################################################################
#
def find_best_neighbour(time, sim_record, fitness_func, neighbours):
    """Finds the position of the neighbour with the highest fitness."""
    neighbours_fitness = [
        fitness_func[sim_record.positions[neighbour, time]]
        for neighbour in neighbours
    ]
    return sim_record.positions[np.argmax(neighbours_fitness), time]


def random_step_search(time, node, sim_record, num_bits):
    """Flips a random bit of the previous position."""
    last_position = sim_record.positions[node, time]
    return bitm.flip_random_bit(num_bits, last_position)


def random_teleport_search(num_bits):
    """Returns a random integer from 0 to N-1."""
    return random.randint(num_bits)


###############################################################################
# Action Functions
###############################################################################
#
def action_best_then_step(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    best_neighbour_position = \
            find_best_neighbour(time, sim_record, fitness_func, neighbours)
    best_neighbour_fitness = fitness_func[best_neighbour_position]

    # search for a possible next position
    search_position = random_step_search(time, node, sim_record, num_bits)

    current_node_fitness = fitness_func[sim_record.positions[node, time]]
    if best_neighbour_fitness > current_node_fitness:
        next_position = best_neighbour_position
        outcome = OUTCOME_NUM['copied best']
    elif fitness_func[search_position] > current_node_fitness:
        next_position = search_position
        outcome = OUTCOME_NUM['stepped']
    else:
        next_position = sim_record.positions[node, time]
        outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['best_then_step']
    sim_record.outcomes[node, time +1] = outcome

ACTION_FUNC[ACTION_NUM["best_then_step"]] = action_best_then_step


def action_step_then_best(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, keep last position.
    """
    # search for a possible next position
    step_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if fitness_func[step_position] > current_node_fitness:
        next_position = step_position
        outcome = OUTCOME_NUM['stepped']

    else:
        best_neighbour_position = \
                find_best_neighbour(time, sim_record, fitness_func, neighbours)
        best_neighbour_fitness = fitness_func[best_neighbour_position]

        if best_neighbour_fitness > current_node_fitness:
            next_position = best_neighbour_position
            outcome = OUTCOME_NUM['copied best']
        else:
            next_position = sim_record.positions[node, time]
            outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['step_then_best']
    sim_record.outcomes[node, time +1] = outcome

ACTION_FUNC[ACTION_NUM["step_then_best"]] = action_step_then_best


def action_modal_then_step(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Try to find a fitness_func occurring more than once among neighbours.
    If successful, take the position of one of these neighbours at random.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    # finds a neighbour with modal fitness_func
    neighbours_fitness = np.empty(len(neighbours))
    for neighbour_idx, neighbour in enumerate(neighbours):
        neighbours_fitness[neighbour_idx] = \
                fitness_func[sim_record.positions[neighbour, time]]

    neighbours_fitness_freq = Counter(neighbours_fitness)

    min_freq = 1
    modal_fitness = 0
    for key, value in neighbours_fitness_freq.items():
        if value > min_freq:
            min_freq = value
            modal_fitness = key

    # search for a possible next position
    step_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if modal_fitness > current_node_fitness:
        modal_neighbour_idxs = np.where(neighbours_fitness == modal_fitness)[0]
        rand_idx = random.randint(len(modal_neighbour_idxs))
        neighbour = neighbours[modal_neighbour_idxs[rand_idx]]

        next_position = sim_record.positions[neighbour, time]
        outcome = OUTCOME_NUM['copied a modal']

    elif fitness_func[step_position] > current_node_fitness:
        next_position = step_position
        outcome = OUTCOME_NUM['stepped']

    else:
        next_position = sim_record.positions[node, time]
        outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['modal_then_step']
    sim_record.outcomes[node, time +1] = outcome

ACTION_FUNC[ACTION_NUM["modal_then_step"]] = action_modal_then_step


###############################################################################
# Strategy Classes
###############################################################################
#
class SimpleMCAgent():
    """
    Agent which employs Monte Carlo prediction to make decisions
    using only time as a state.
    """
    def __init__(
            self,
            deadline,

            possible_actions=(
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
            ),
    ):
        self.deadline = deadline

        # set up actions
        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        # generate q table
        self.q_table = np.ones((self.deadline -1, len(possible_actions)))

    def _choose_greedy_action(self, current_state):
        """
        Chooses the best action according the Q table.
        """
        return self.possible_actions[np.argmax(self.q_table[current_state])]

    def perform_greedy_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """Performs the learned best action for a given state."""
        # decide on next action
        action = self._choose_greedy_action(time)

        # perform action
        ACTION_FUNC[action](
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
        )

    def learn_and_perform_random_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """
        Performs the epsilon greedy action for a given state
        and if at the end of an episode,
        learns using Monte Carlo Prediction.
        """
        # perform random action
        action = random.choice(self.possible_actions)


        # perform action
        ACTION_FUNC[action](
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
        )

        # if at the end of an episode
        if time == self.deadline -2:
            reward = fitness_func[sim_record.positions[node, time]]

            for step in range(1, self.deadline -1):
                action_idx = sim_record.actions[node, step]
                self.q_table[
                        step,
                        self.possible_actions.index(action_idx),
                ] += reward

    def save_q_table(self, file_name):
        """Save the q_table with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.save(file_handle, self.q_table)

    def load_q_table(self, file_name):
        """Load a q_table with the given file name."""
        with open(file_name, 'rb') as file_handle:
            self.q_table = np.load(file_handle)

    def plot_q_table(self, axis, normalise=True):
        """
        Plots the perceived utility of the actions
        for the all values of time.
        """
        time_actions = self.q_table
        if normalise:
            for idx, _ in enumerate(time_actions):
                time_actions[idx] = \
                        time_actions[idx] /np.sum(time_actions[idx])

        actions_time = np.swapaxes(time_actions, 0, 1)

        cumulative_values = np.zeros(self.deadline -1)
        for idx, action_idx in enumerate(self.possible_actions):
            axis.bar(
                range(self.deadline -1),
                actions_time[idx],
                bottom=cumulative_values,
                label=ACTION_STR[action_idx],
            )
            cumulative_values = cumulative_values + actions_time[idx]


class SimpleQLearningAgent():
    """
    Agent which employs Q learning to make decisions
    using only time as a state.
    """
    def __init__(
            self,
            deadline,
            learning_rate=0.6,
            discount_factor=0.1,

            epsilon=1,
            epsilon_decay=1e-6,

            possible_actions=(
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
            ),
    ):
        # learning variables
        self.deadline = deadline
        self.learning_rate = learning_rate
        self.discount = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # set up actions
        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        # generate q table
        self.q_table = random.uniform(
            size=(self.deadline -1, len(possible_actions)),
        )

    def _update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_action = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_action]

        action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][action_idx]

        self.q_table[state][action_idx] += self.learning_rate * td_delta

    def _choose_greedy_action(self, current_state):
        """
        Chooses the best action according the Q table.
        """
        return self.possible_actions[np.argmax(self.q_table[current_state])]

    def _choose_epsilon_greedy_action(self, current_state):
        """
        Chooses either the best action according the Q table
        or a random action
        depending on the current epsilon value.
        """
        if random.rand() <= self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                np.argmax(self.q_table[current_state])
            ]
        self.epsilon -= self.epsilon_decay * self.epsilon
        return action

    def perform_greedy_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """Performs the learned best action for a given state."""
        # decide on next action
        action = self._choose_greedy_action(time)

        # perform action
        ACTION_FUNC[action](
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
        )

    def learn_and_perform_epsilon_greedy_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """
        Learns from the previous time step
        and performs the epsilon greedy action for a given state.
        """
        # if not first run learn from the last decision
        if time > 1:
            reward = fitness_func[sim_record.positions[node, time]] \
                    - fitness_func[sim_record.positions[node, time -1]]

            self._update_q_table(
                time -1,
                sim_record.actions[node, time-1],
                time,
                reward,
            )

        # decide on next action
        action = self._choose_epsilon_greedy_action(time)

        # perform action
        ACTION_FUNC[action](
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
        )

    def save_q_table(self, file_name):
        """Save the q_table with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.save(file_handle, self.q_table)

    def load_q_table(self, file_name):
        """Load a q_table with the given file name."""
        with open(file_name, 'rb') as file_handle:
            self.q_table = np.load(file_handle)

    def plot_q_table(self, axis, normalise=True):
        """
        Plots the perceived utility of the actions
        for the all values of time.
        """
        time_actions = self.q_table
        if normalise:
            for idx, _ in enumerate(time_actions):
                time_actions[idx] = \
                        time_actions[idx] /np.sum(time_actions[idx])

        actions_time = np.swapaxes(time_actions, 0, 1)

        cumulative_values = np.zeros(self.deadline -1)
        for idx, action_idx in enumerate(self.possible_actions):
            axis.bar(
                range(self.deadline -1),
                actions_time[idx],
                bottom=cumulative_values,
                label=ACTION_STR[action_idx],
            )
            cumulative_values = cumulative_values + actions_time[idx]


class QLearningAgent():
    """Agent which employs Q learning to make decisions."""
    def __init__(
            self,
            deadline,
            learning_rate=0.6,
            discount_factor=0.1,

            epsilon=1,
            epsilon_decay=1e-6,

            quantisation_levels=100,
            state_components=(
                'time',
                'score',
                #'best_neighbour',
            ),

            possible_actions=(
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
            ),
    ):
        # learning variables
        self.deadline = deadline
        self.learning_rate = learning_rate
        self.discount = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        # set up actions
        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        # set up states
        self.quantisation_levels = quantisation_levels
        self.state_components = state_components
        self.state_dimensions = []
        for state_component in self.state_components:
            if state_component == 'time':
                self.state_dimensions += [deadline]
            elif state_component == 'score':
                self.state_dimensions += [quantisation_levels]
            elif state_component == 'best neighbour score':
                self.state_dimensions += [quantisation_levels]

        # generate q table
        self.q_table = random.uniform(
            size=(list(self.state_dimensions) + [len(possible_actions)]),
        )

    def _find_state(self, time, node, neighbours, fitness_func, sim_record):
        """Find a node's state at the given time."""
        state = []
        for state_component in self.state_components:
            if state_component == 'time':
                state += [self.deadline - time -1]
            elif state_component == 'score':
                current_fitness = \
                        fitness_func[sim_record.positions[node, time]]
                state += [
                    int(round(current_fitness * (self.quantisation_levels -1)))
                ]
            elif state_component == 'best neighbour score':
                best_neighbour_fitness = fitness_func[
                    find_best_neighbour(time, sim_record,
                                        fitness_func, neighbours)
                ]
                state += [int(round(
                    best_neighbour_fitness * (self.quantisation_levels -1),
                ))]
        return tuple(state)

    def _update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_action = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_action]

        action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][action_idx]

        self.q_table[state][action_idx] += self.learning_rate * td_delta

    def _choose_greedy_action(self, current_state):
        """
        Chooses the best action according the Q table.
        """
        return self.possible_actions[np.argmax(self.q_table[current_state])]

    def _epsilon_greedy_choose_action(self, current_state):
        """
        Chooses either the best action according the Q table
        or a random action
        depending on the current epsilon value.
        """
        if random.rand() <= self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                np.argmax(self.q_table[current_state])
            ]
        self.epsilon -= self.epsilon_decay * self.epsilon
        return action

    def _perform_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
            action,
    ):
        """Calls the action function corresponding to the given action."""
        if action == ACTION_NUM['best_then_step']:
            action_best_then_step(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
        elif action == ACTION_NUM['step_then_best']:
            action_step_then_best(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
        elif action == ACTION_NUM['modal_then_step']:
            action_modal_then_step(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )

    def save_q_table(self, file_name):
        """Save the q_table with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.save(file_handle, self.q_table)

    def load_q_table(self, file_name):
        """Load a q_table with the given file name."""
        with open(file_name, 'rb') as file_handle:
            self.q_table = np.load(file_handle)

    def perform_greedy_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """Performs the learned best action for a given state."""
        # find current state
        current_state = self._find_state(
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )

        # decide on next action
        action = self._choose_greedy_action(current_state)

        # perform action
        self._perform_action(num_bits, time, node, neighbours,
                             fitness_func, sim_record, action)

    def learn_and_perform_epsilon_greedy_action(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """
        Learns from the previous time step
        and performs the epsilon greedy action for a given state.
        """
        # find current state
        current_state = self._find_state(
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(
                time-1,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
            reward = fitness_func[sim_record.positions[node, time]] \
                    - fitness_func[sim_record.positions[node, time -1]]

            self._update_q_table(
                last_state,
                sim_record.actions[node, time-1],
                current_state,
                reward,
            )

        # decide on next action
        action = self._greedy_epsilon_choose_action(current_state)

        # perform action
        self._perform_action(num_bits, time, node, neighbours,
                             fitness_func, sim_record, action)

    def plot_q_table(self, axis, state_component_name, normalise=True):
        """
        Plots the perceived utility of the actions
        for the all values of a state component.
        """

        comp_idx = self.state_components.index(state_component_name)
        comp_q_table = np.swapaxes(self.q_table, 0, comp_idx)

        comp_actions = np.sum(
            comp_q_table,
            axis=tuple(range(1, len(self.state_dimensions))),
            dtype='d',
        )
        if normalise:
            for idx, _ in enumerate(comp_actions):
                comp_actions[idx] = \
                        comp_actions[idx] /np.sum(comp_actions[idx])

        actions_comp = np.swapaxes(comp_actions, 0, 1)

        cumulative_values = np.zeros(self.state_dimensions[comp_idx])
        for idx, action_idx in enumerate(self.possible_actions):
            axis.bar(
                range(self.state_dimensions[comp_idx]),
                actions_comp[idx],
                bottom=cumulative_values,
                label=ACTION_STR[action_idx],
            )
            cumulative_values = cumulative_values + actions_comp[idx]

        axis.set_title(state_component_name)


###############################################################################
# Main Function
###############################################################################
#
def run_episode(
        graph,
        num_bits,
        deadline,
        fitness_func,
        neighbour_sample_size=None,
        num_processes=1,
        strategy=action_best_then_step,
):
    """A single episode of the simulation."""

    num_nodes = len(graph)

    sim_record = SimulationRecord(num_nodes, deadline, num_processes=4)
    sim_record.set_random_initial_position(num_bits, fitness_func)

    for time in range(deadline -1):
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

            # carry out the strategy
            strategy(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )

    return sim_record
