"""A module containing the agents which can learn strategies for workers."""

import json
from os import path
import numpy as np
from numpy import random

from actions import ACTION_NUM, ACTION_STR, ACTION_FUNC


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

            learning_rate=0.6,

            possible_actions=(
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
            ),
    ):
        self.deadline = deadline
        self.learning_rate = learning_rate

        # set up actions
        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        # generate q table
        self.q_table = np.zeros((self.deadline -1, len(possible_actions)))

    def _choose_greedy_action(self, current_state):
        """
        Chooses the best action according the Q table.
        """
        return self.possible_actions[np.argmax(self.q_table[current_state])]

    def perform_greedy_action(self, time, node, sim_record, neighbours):
        """Performs the learned best action for a given state."""
        # decide on next action
        action = self._choose_greedy_action(time)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_and_perform_random_action(
            self,
            time,
            node,
            sim_record,
            neighbours,
    ):
        """
        Performs the epsilon greedy action for a given state
        and if at the end of an episode,
        learns using Monte Carlo Prediction.
        """
        # perform random action
        action = random.choice(self.possible_actions)


        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

        # if at the end of an episode
        if time == self.deadline -2:
            reward = sim_record.fitness_func[sim_record.positions[node, time]]

            for step in range(1, self.deadline -1):
                action_idx = sim_record.actions[node, step]
                current_value = self.q_table[
                    step,
                    self.possible_actions.index(action_idx),
                ]
                update = self.learning_rate * (reward - current_value)

                self.q_table[
                    step,
                    self.possible_actions.index(action_idx),
                ] += update

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
        time_actions = np.copy(self.q_table)
        if normalise:
            for idx, _ in enumerate(time_actions):
                total = np.sum(time_actions[idx])
                if total:
                    time_actions[idx] = \
                            time_actions[idx] / total
                else:
                    time_actions[idx] = \
                            time_actions[idx] * 0

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
            random_initialisation=False,
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
        if random_initialisation:
            self.q_table = random.uniform(
                size=(self.deadline -1, len(possible_actions)),
            )
        else:
            self.q_table = np.zeros((self.deadline -1, len(possible_actions)))

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

    def perform_greedy_action(self, time, node, sim_record, neighbours):
        """Performs the learned best action for a given state."""

        current_state = time
        # perform action with the highest perceived value
        action = self.possible_actions[np.argmax(self.q_table[current_state])]

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def perform_n_greedy_actions(self, time, node, sim_record, neighbours):
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
            if ACTION_FUNC[action](time, node, sim_record, neighbours):
                break

    def learn_all_rewards_and_perform_epsilon_greedy_action(
            self,
            time,
            node,
            sim_record,
            neighbours,
    ):
        """
        Learns from the previous time step
        (with a reward available at each step)
        and performs the epsilon greedy action for a given state.
        """
        # if not first run learn from the last decision
        if time > 1:
            current_fitness = \
                sim_record.fitness_func[sim_record.positions[node, time]]
            last_fitness = \
                sim_record.fitness_func[sim_record.positions[node, time -1]]
            reward = current_fitness - last_fitness

            self._update_q_table(
                time -1,
                sim_record.actions[node, time-1],
                time,
                reward,
            )

        # decide on next action
        action = self._choose_epsilon_greedy_action(time)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_end_reward_and_perform_epsilon_greedy_action(
            self,
            time,
            node,
            sim_record,
            neighbours,
    ):
        """
        Learns from the previous time step
        (with a reward only available at the end)
        and performs the epsilon greedy action for a given state.
        """
        # if not first run learn from the last decision
        if time > 1:
            if time == self.deadline -2:
                reward = \
                    sim_record.fitness_func[sim_record.positions[node, time]]
            else:
                reward = 0

            self._update_q_table(
                time -1,
                sim_record.actions[node, time-1],
                time,
                reward,
            )

        # decide on next action
        action = self._choose_epsilon_greedy_action(time)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

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
        time_actions = np.copy(self.q_table)

        if normalise:
            for idx, _ in enumerate(time_actions):
                total = np.sum(time_actions[idx])
                if total:
                    time_actions[idx] = \
                            time_actions[idx] / total
                else:
                    time_actions[idx] = \
                            time_actions[idx] * 0

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
            ),

            possible_actions=(
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
            ),
            random_initialisation=False,
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
                self.state_dimensions += [deadline-1]
            elif state_component == 'score':
                self.state_dimensions += [quantisation_levels]

        # generate q table
        if random_initialisation:
            self.q_table = random.uniform(
                size=(list(self.state_dimensions) + [len(possible_actions)]),
            )
        else:
            self.q_table = np.zeros(
                self.state_dimensions + [len(possible_actions)],
            )

        # the count of the number of times each state has been updated
        self.update_count = np.zeros(self.state_dimensions)

    def _find_state(self, time, node, sim_record):
        """Find a node's state at the given time."""
        state = []
        for state_component in self.state_components:
            if state_component == 'time':
                state += [time]
            elif state_component == 'score':
                current_fitness = sim_record.fitness_func_norm[
                    sim_record.positions[node, time]
                ]
                state += [
                    int(current_fitness * (self.quantisation_levels -1))
                ]
        return tuple(state)

    def _update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_action = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_action]

        action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][action_idx]

        self.q_table[state][action_idx] += self.learning_rate * td_delta
        self.update_count[state] += 1

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

    def save_tables(self, file_name):
        """Save the q_table and update_count with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.savez(file_handle, q_table=self.q_table, update_count=self.update_count)

    def load_tables(self, file_name):
        """Load a q_table and update_count with the given file name."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)
            self.q_table = file_content["q_table"]
            self.update_count = file_content["update_count"]

    def perform_greedy_action(self, time, node, sim_record, neighbours):
        """Performs the learned best action for a given state."""
        # find current state
        current_state = self._find_state(
            time,
            node,
            sim_record,
        )

        # decide on next action
        action = self._choose_greedy_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_all_rewards_and_perform_epsilon_greedy_action(
            self,
            time,
            node,
            sim_record,
            neighbours,
    ):
        """
        Learns from the previous time step
        (with a reward available at each step)
        and performs the epsilon greedy action for a given state.
        """
        # find current state
        current_state = self._find_state(
            time,
            node,
            sim_record,
        )

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(
                time-1,
                node,
                sim_record,
            )
            current_fitness = \
                sim_record.fitness_func[sim_record.positions[node, time]]
            last_fitness = \
                sim_record.fitness_func[sim_record.positions[node, time -1]]

            reward = current_fitness - last_fitness

            self._update_q_table(
                last_state,
                sim_record.actions[node, time-1],
                current_state,
                reward,
            )

        # decide on next action
        action = self._epsilon_greedy_choose_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_end_reward_and_perform_epsilon_greedy_action(
            self,
            time,
            node,
            sim_record,
            neighbours,
    ):
        """
        Learns from the previous time step
        (with a reward only available at the end)
        and performs the epsilon greedy action for a given state.
        """
        # find current state
        current_state = self._find_state(
            time,
            node,
            sim_record,
        )

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(
                time-1,
                node,
                sim_record,
            )

            if time == self.deadline -2:
                reward = \
                    sim_record.fitness_func[sim_record.positions[node, time]]
            else:
                reward = 0

            self._update_q_table(
                last_state,
                sim_record.actions[node, time-1],
                current_state,
                reward,
            )

        # decide on next action
        action = self._epsilon_greedy_choose_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def plot_q_table(self, axis, state_component_name, normalise=True):
        """
        Plots the perceived utility of the actions
        for the all values of a state component.
        """
        comp_idx = self.state_components.index(state_component_name)

        local_table = np.copy(self.q_table)
        comp_q_table = np.swapaxes(local_table, 0, comp_idx)

        comp_actions = np.sum(
            comp_q_table,
            axis=tuple(range(1, len(self.state_dimensions))),
            dtype='d',
        )
        if normalise:
            for idx, _ in enumerate(comp_actions):
                total = np.sum(comp_actions[idx])
                if total:
                    comp_actions[idx] = \
                            comp_actions[idx] / total
                else:
                    comp_actions[idx] = \
                            comp_actions[idx] * 0

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
# Functions
###############################################################################
#
def load_agent_and_settings(config_location, training=False, episodes=None):
    """Constructs an agent according to it's configuration file"""

    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

    np.random.seed(config["seed"])

    # initialise agent
    possible_actions = [
        ACTION_NUM[possible_action]
        for possible_action in config["agent"]["possible actions"]
    ]

    if config["agent"]["type"] == "QLearningAgent":
        agent = QLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            quantisation_levels=config["agent"]["quantisation levels"],
            state_components=config["agent"]["state components"],
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )

    elif config["agent"]["type"] == "SimpleQLearningAgent":
        agent = SimpleQLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )

    elif config["agent"]["type"] == "SimpleMCAgent":
        agent = SimpleMCAgent(
            config["deadline"],
            learning_rate=config["agent"]["learning rate"],
            possible_actions=possible_actions,
        )

    # load the appropriate q table
    if training:
        if config["agent"]["load table"]:
            agent.load_tables(
                path.join(config_dir, config["agent"]["load table"]),
            )
    else:
        name = config['name']

        if episodes:
            agent.load_tables(
                path.join(config_dir, f'{name}-{episodes}.npz'),
            )
        else:
            agent.load_tables(
                path.join(config_dir, f'{name}.npz'),
            )

    return agent, config, config_dir
