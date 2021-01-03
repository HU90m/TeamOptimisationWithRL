"""A module containing the agents which can learn strategies for workers."""

from os import path
import numpy as np
from numpy import random


###############################################################################
# Agents
###############################################################################
#
class QLearningAgent:
    """Agent which employs Q learning to make decisions."""
    def __init__(self, config, config_dir, get_action_num_func):
        # settings
        self.name = config["name"]
        self.config_dir = config_dir
        self.deadline = config["deadline"]

        # learning variables
        self.learning_rate = config["learning rate"]
        self.discount = config["discount factor"]

        self.epsilon = config["epsilon"]
        self.epsilon_decay = config["epsilon decay"]

        self.quantisation_levels = config["quantisation levels"]

        # set up actions
        self.possible_actions = []
        for action_name in config["possible actions"]:
            # convert the actions names to numbers, with the given function
            self.possible_actions.append(get_action_num_func(action_name))

        # this maps the external action number to internal action indicies
        self.action_num2idx = {}
        for action_idx, action_num in enumerate(self.possible_actions):
            self.action_num2idx[action_num] = action_idx

        # generate q table
        q_table_shape = [
            self.deadline, # time
            self.quantisation_levels, # score
            len(self.possible_actions), # possible actions
        ]
        self.q_table = np.zeros(q_table_shape)

        # the count of the number of times each state has been updated
        self.update_count = np.zeros(q_table_shape[:-1])

    def _find_state(self, time, current_fitness):
        """Find a worker's state at the given time."""
        return (time, int(current_fitness * (self.quantisation_levels -1)))

    def _update_q_table(self, prior_state, action_num, post_state, reward):
        """Updates the Q table using the given transition."""
        # the action index of the best action in the post transition state.
        post_best_action_idx = np.argmax(self.q_table[post_state])
        td_target = reward \
            + self.discount * self.q_table[post_state][post_best_action_idx]

        action_idx = self.action_num2idx[action_num]
        td_delta = td_target - self.q_table[prior_state][action_idx]

        self.q_table[prior_state][action_idx] += self.learning_rate * td_delta
        self.update_count[prior_state] += 1

    def choose_greedy_action(self, time, current_fitness_norm):
        """Returns the best action according the Q table."""
        current_state = self._find_state(time, current_fitness_norm)
        return self.possible_actions[np.argmax(self.q_table[current_state])]

    def choose_epsilon_greedy_action(self, time, current_fitness_norm):
        """Returns either the best action according the Q table
        or a random action depending on the current epsilon value.
        """
        current_state = self._find_state(time, current_fitness_norm)
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

    def learn(self, prior_time, prior_fitness_norm,
              chosen_action, post_time, post_fitness_norm, post_fitness):
        """Learns from a transition."""
        prior_state = self._find_state(prior_time, prior_fitness_norm)
        post_state = self._find_state(post_time, post_fitness_norm)

        # each node only receives a reward at the deadline
        if post_time == self.deadline:
            reward = post_fitness
        else:
            reward = 0

        self._update_q_table(prior_state, chosen_action, post_state, reward)

    def _find_file_name(self, suffix):
        """Finds the file name for a given suffix."""
        if suffix:
            return path.join(self.config_dir, f'{self.name}_{suffix}.npz')
        return path.join(self.config_dir, f'{self.name}.npz')

    def save(self, suffix=None):
        """Saves the agent's learnings.
        Saves the q_table and update_count in a file with the the given suffix.
        """
        file_name = self._find_file_name(suffix)
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                q_table=self.q_table,
                update_count=self.update_count,
            )

    def load(self, suffix=None):
        """Loads learnings.
        Loads the q_table and update_count from a file with the the given
        suffix.
        """
        file_name = self._find_file_name(suffix)
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)
            self.q_table = file_content["q_table"]
            self.update_count = file_content["update_count"]
