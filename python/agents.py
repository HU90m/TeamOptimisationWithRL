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

        self.quantisation_levels = quantisation_levels

        # set up actions
        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        # generate q table
        q_table_shape = [
            deadline-1, # time
            quantisation_levels, # score
            len(possible_actions), # possible actions
        ]
        self.q_table = np.zeros(q_table_shape)

        # the count of the number of times each state has been updated
        self.update_count = np.zeros(q_table_shape[:-1])

    def _find_state(self, time, node, sim_record):
        """Find a worker's state at the given time."""
        current_fitness = \
            sim_record.fitness_func_norm[sim_record.positions[node, time]]
        return (time, int(current_fitness * (self.quantisation_levels -1)))

    def _update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_p_action_idx = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_p_action_idx]

        p_action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][p_action_idx]

        self.q_table[state][p_action_idx] += self.learning_rate * td_delta
        self.update_count[state] += 1

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
        # choose action
        if random.rand() <= self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                np.argmax(self.q_table[current_state])
            ]
        # decay epsilon
        self.epsilon -= self.epsilon_decay * self.epsilon
        return action

    def test(self, time, node, sim_record, neighbours):
        """Performs the learned best action for a given state."""
        # find current state
        current_state = self._find_state(time, node, sim_record)

        # decide on next action
        action = self._choose_greedy_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def train(self, time, node, sim_record, neighbours):
        """
        Learns from the previous time step
        (with a reward only given at the end)
        and performs the epsilon greedy action for a given state.
        """
        # find current state
        current_state = self._find_state(time, node, sim_record)

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(time-1, node, sim_record)

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
        action = self._choose_epsilon_greedy_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def save(self, file_name):
        """Save the q_table and update_count with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                q_table=self.q_table,
                update_count=self.update_count,
            )

    def load(self, file_name):
        """Load a q_table and update_count with the given file name."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)
            self.q_table = file_content["q_table"]
            self.update_count = file_content["update_count"]

    def get_possible_actions(self):
        """Returns possible actions as a list of their names."""
        return [ACTION_STR[action] for action in self.possible_actions]


class PolicyGradientAgent():
    """Agent which employs Reinforce Policy Gradient to make decisions."""
    def __init__(
            self,
            deadline,
            learning_rate=1e-13,
            baseline_learning_rate=0.05,
            action_1=ACTION_NUM["best"],
            action_2=ACTION_NUM["step"],
    ):
        self.deadline = deadline
        self.lr = learning_rate

        self.w_0 = 0.5
        self.w_1 = 0

        self.action_1 = action_1
        self.action_2 = action_2

        self.baseline = 0.5
        self.blr = baseline_learning_rate

    def policy(self, time):
        """Policy Function"""
        return self.w_0 + (self.w_1 * time)

    def train(self, time, node, sim_record, neighbours):
        """Makes decisions and learns from them."""

        policy = self.policy(time)

        # perform action according to policy
        if random.random() > policy:
            ACTION_FUNC[self.action_1](time, node, sim_record, neighbours)
        else:
            ACTION_FUNC[self.action_2](time, node, sim_record, neighbours)

        # if last state in episode learn
        if time == self.deadline - 2:
            reward = \
                sim_record.fitness_func[sim_record.positions[node, time]] \
                - self.baseline

            self.baseline = \
                    ((1 - self.blr) * self.baseline) \
                    + (self.blr * reward)

            self.w_0 += self.lr * reward * (1 / policy)
            self.w_1 += self.lr * reward * (time / policy)

    def test(self, time, node, sim_record, neighbours):
        """Makes decisions but doesn't learn from them."""
        if random.random() > self.policy(time):
            ACTION_FUNC[self.action_1](time, node, sim_record, neighbours)
        else:
            ACTION_FUNC[self.action_2](time, node, sim_record, neighbours)

    def save(self, file_name):
        """Saves the agent's weights."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                w_0=self.w_0,
                w_1=self.w_1,
                baseline=self.baseline,
            )

    def load(self, file_name):
        """Loads the agent's weights."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)

            self.w_0 = float(file_content['w_0'])
            self.w_1 = float(file_content['w_1'])
            self.baseline = float(file_content['baseline'])

    def plot(self):
        return


###############################################################################
# Functions
###############################################################################
#
def load_agent_and_settings(config_location, training=False, episode=None):
    """Constructs an agent according to it's configuration file"""

    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

    # initialise agent
    agent_type = config["agent"]["type"]
    if agent_type == "QLearningAgent":
        possible_actions = [
            ACTION_NUM[possible_action]
            for possible_action in config["agent"]["possible actions"]
        ]
        agent = QLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            quantisation_levels=config["agent"]["quantisation levels"],
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )
    elif agent_type == "PolicyGradientAgent":
        agent = PolicyGradientAgent(
            config["deadline"],
            learning_rate=config["agent"]["learning rate"],
        )
    else:
        raise ValueError(f"{agent_type} is not a supported agent.")

    # load the appropriate q table
    if training:
        if config["agent"]["load table"]:
            agent.load(
                path.join(config_dir, config["agent"]["load table"]),
            )
    else:
        name = config['name']

        if episode:
            agent.load(
                path.join(config_dir, f'{name}-{episode}.npz'),
            )
        else:
            agent.load(
                path.join(config_dir, f'{name}.npz'),
            )

    return agent, config, config_dir
