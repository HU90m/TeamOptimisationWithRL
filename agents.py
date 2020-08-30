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

    def perform_greedy_action(self, time, node, sim_record, neighbours):
        """Performs the learned best action for a given state."""
        # find current state
        current_state = self._find_state(time, node, sim_record)

        # decide on next action
        action = self._choose_greedy_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_using_all_rewards(self, time, node, sim_record, neighbours):
        """
        Learns from the previous time step
        (with a reward given at each step)
        and performs the epsilon greedy action for a given state.
        """
        # find current state
        current_state = self._find_state(time, node, sim_record)

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(time-1, node, sim_record)
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
        action = self._choose_epsilon_greedy_action(current_state)

        # perform action
        ACTION_FUNC[action](time, node, sim_record, neighbours)

    def learn_using_end_rewards(self, time, node, sim_record, neighbours):
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

    def save_tables(self, file_name):
        """Save the q_table and update_count with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                q_table=self.q_table,
                update_count=self.update_count,
            )

    def load_tables(self, file_name):
        """Load a q_table and update_count with the given file name."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)
            self.q_table = file_content["q_table"]
            self.update_count = file_content["update_count"]

    def get_possible_actions(self):
        """Returns possible actions as a list of their names."""
        return [ACTION_STR[action] for action in self.possible_actions]


###############################################################################
# Functions
###############################################################################
#
def load_agent_and_settings(config_location, training=False, episodes=None):
    """Constructs an agent according to it's configuration file"""

    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

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
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )
    else:
        raise ValueError("QLearningAgent is the only supported agent.")

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
