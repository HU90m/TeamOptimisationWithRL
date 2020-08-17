"""A module containing the agents which can learn strategies for workers."""

import numpy as np
from numpy import random

from actions import ACTION_NUM, ACTION_FUNC


###############################################################################
# Agents
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
    num_actions = 2 # the number of actions n_greedy_actions can perform

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

        current_state = time
        # perform action with the highest perceived value
        action = self.possible_actions[np.argmax(self.q_table[current_state])]

        # perform action
        ACTION_FUNC[action](
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )

    def perform_n_greedy_actions(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """
        Performs the learned N best actions for a given state in order.
        """
        current_state = time

        # The possible action indices in order of highest to lowest value
        action_ids_in_order = np.flip(np.argsort(self.q_table[current_state]))

        # for N best actions
        for action_id in action_ids_in_order[:self.num_actions]:
            action = self.possible_actions[action_id]

            # attempt action, if successful finish
            if ACTION_FUNC[action](
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            ):
                break

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
        action = self._epsilon_greedy_choose_action(current_state)

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
