"""A module containing the agents which can learn strategies for workers."""

from os import path
import numpy as np
from numpy import random
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

import environment.bitmanipulation as bitm

###############################################################################
# Agent
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

        # set up actions
        self._possible_actions_str = config["possible actions"]
        self.possible_actions = []
        for action_name in self._possible_actions_str:
            # convert the actions names to numbers, with the given function
            self.possible_actions.append(get_action_num_func(action_name))

        # this maps the external action number to internal action indicies
        self.action_num2idx = {}
        for action_idx, action_num in enumerate(self.possible_actions):
            self.action_num2idx[action_num] = action_idx

        self.state_space_type = config["state space"]["type"]
        if self.state_space_type == "time fitness":
            self.quantisation_levels = \
                    config["state space"]["quantisation levels"]

            self._find_state = self._find_state_time_fitness

            q_table_shape = [
                self.deadline +1, # time
                self.quantisation_levels, # fitness
                len(self.possible_actions), # possible actions
            ]
        elif self.state_space_type == "time memory":
            assert len(self.possible_actions) == 2
            self.history = \
                    config["state space"]["history"]

            self._find_state = self._find_state_time_memory

            q_table_shape = [
                self.deadline +1, # time
                1 << self.history, # memory
                len(self.possible_actions), # possible actions
            ]
        else:
            raise ValueError("State space type not valid")

        self._q_table = np.zeros(q_table_shape)

        # the count of the number of times each state has been updated
        self._update_count = np.zeros(q_table_shape[:-1])


        # exploration type
        if config["exploration"]["type"] == "epsilon greedy":
            self.epsilon = config["exploration"]["epsilon start"]
            self.epsilon_decay = config["exploration"]["epsilon decay"]

            self.best_action = self._choose_greedy_action
            self.explore_action = self._choose_epsilon_greedy_action
            self.episode_end = self._decay_epsilon

        elif config["exploration"]["type"] == "boltzmann":
            self.best_action = self._choose_greedy_action
            self.explore_action = self._choose_boltzmann_action
            self.episode_end = lambda *args : None

        else:
            raise ValueError("Exploration type not valid")

    def _find_state_time_fitness(self, node, time, environment):
        """Find a worker's state at the given time.
        The state space used is the time and the quantised normalised fitness.
        """
        current_fitness = environment.get_node_fitness_norm(node, time)
        return (time, int(current_fitness * (self.quantisation_levels -1)))

    def _find_state_time_memory(self, node, time, environment):
        """Find a worker's state at the given time.
        The state space used is the time and the binary memory.

        Binary memory is an integer with bit-length of the given history.

        The value of each bit is the action
        which was taken in a given time-step of the last few time-steps.

        For example, the number 11 (1011 in binary),
        when there is a history of 4, would mean
        that action 1 was taken taken at time -1, time -2 and time -4,
        and so action 0 was only taken at time -3.

        | time -4 | time -3 | time -2 | time -1 |
        |    1    |    0    |    1    |    1    |
        """
        end = time if time < self.history else self.history

        memory = 0
        for idx in range(end):
            action_idx = self.action_num2idx[
                    environment.get_node_action(node, time - idx - 1)
            ]
            memory = bitm.set_bit(memory, idx, action_idx)

        return (time, memory)

    def _update_q_table(self, prior_state, action_num, post_state, reward):
        """Updates the Q table using the given transition."""
        # the action index of the best action in the post transition state.
        post_best_action_idx = np.argmax(self._q_table[post_state])
        td_target = reward \
            + self.discount * self._q_table[post_state][post_best_action_idx]

        action_idx = self.action_num2idx[action_num]
        td_delta = td_target - self._q_table[prior_state][action_idx]

        self._q_table[prior_state][action_idx] += self.learning_rate * td_delta
        self._update_count[prior_state] += 1

    def _choose_greedy_action(self, node, time, environment):
        """Returns the best action according the Q table."""
        current_state = self._find_state(node, time, environment)
        return self.possible_actions[np.argmax(self._q_table[current_state])]

    def _choose_epsilon_greedy_action(self, node, time, environment):
        """Returns either the best action according the Q table
        or a random action depending on the current epsilon value.
        """
        current_state = self._find_state(node, time, environment)
        # choose action
        if random.rand() <= self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                np.argmax(self._q_table[current_state])
            ]
        return action

    def _choose_boltzmann_action(self, node, time, environment):
        """Returns either the action using the softmax
        as a probability distribution"""
        current_state = self._find_state(node, time, environment)
        action_values = self._q_table[current_state]

        action_values_exp = \
                np.array([np.exp(action) for action in action_values])

        probs = action_values_exp / np.sum(action_values_exp)

        action_num = np.random.choice(
                self.possible_actions, 1, replace=False, p=probs,
        )
        return action_num

    def _decay_epsilon(self):
        """Decays the value of epsilon by one unit."""
        self.epsilon -= self.epsilon_decay * self.epsilon

    def learn(self, node, prior_time, post_time, environment):
        """Learns from a transition."""
        prior_state = self._find_state(node, prior_time, environment)
        post_state = self._find_state(node, post_time, environment)

        # each node only receives a reward at the deadline
        reward = environment.get_node_fitness(node, post_time) \
                if post_time == self.deadline else 0

        chosen_action = environment.get_node_action(node, prior_time)
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
                q_table=self._q_table,
                update_count=self._update_count,
            )

    def load(self, suffix=None):
        """Loads learnings.
        Loads the q_table and update_count from a file with the the given
        suffix.
        """
        file_name = self._find_file_name(suffix)
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)
            self._q_table = file_content["q_table"]
            self._update_count = file_content["update_count"]

    def plot(self, figure):
        """Plots the q table and update count.
        Plots the q table and update count of an agent in the provided figure.
        """
        if self.state_space_type == "time memory":
            bin_labels = True
            y_stride = 1
            print(
                "Binary memory values:\n\t0 -", self._possible_actions_str[0],
                "\n\t1 -", self._possible_actions_str[1],
            )
        else:
            y_stride = 2
            bin_labels = False

        num_possible_actions = len(self.possible_actions)
        if num_possible_actions == 2:
            axs = figure.subplots(3, 1)

            plot_q_table_image(
                self._possible_actions_str,
                self._q_table,
                axis=axs[0],
                ytick_stride=y_stride,
                binary_yticklabels=bin_labels,
            )
            plot_q_table_action_image(
                self._possible_actions_str,
                0,
                self._q_table,
                axis=axs[1],
                ytick_stride=y_stride,
                binary_yticklabels=bin_labels,
            )
            plot_update_count_image(
                self._update_count,
                axis=axs[2],
                ytick_stride=y_stride,
                binary_yticklabels=bin_labels,
            )

            for axis in axs:
                axis.set_ylabel("Fitness")
                axis.set_xlabel("Time Step")

        elif num_possible_actions == 3:

            axs = figure.subplots(3, 2)
            plot_q_table_image(
                self._possible_actions_str, self._q_table, axis=axs[0, 0]
            )
            plot_q_table_action_image(
                self._possible_actions_str, 0, self._q_table, axis=axs[1, 0]
            )
            plot_q_table_action_image(
                self._possible_actions_str, 1, self._q_table, axis=axs[2, 0]
            )
            plot_update_count_image(self._update_count, axis=axs[0, 1])
            plot_q_table_action_image(
                self._possible_actions_str, 2, self._q_table, axis=axs[1, 1]
            )
            axs[2, 1].remove()

            for i in range(3):
                for j in range(2):
                    axs[i, j].set_ylabel("Fitness")
                    axs[i, j].set_xlabel("Time Step")

        else:
            raise ValueError("Can only plot tables for agents with"
                             " two or three possible actions."
                             " This agent has {num_possible_actions}.")

        plt.tight_layout()
        plt.show()

###############################################################################
# Functions
###############################################################################
#
def plot_update_count_image(
        update_count,
        axis=None,
        ytick_stride=2,
        xtick_stride=2,
        binary_yticklabels=False,
):
    """Plots the given update_count as an image."""
    if not axis:
        axis = plt.gca()

    update_count = np.swapaxes(update_count, 0, 1)
    update_count = update_count[:, :-1]

    image = axis.imshow(update_count, norm=colors.LogNorm())
    cbar = axis.figure.colorbar(image, ax=axis)
    cbar.ax.set_ylabel("Number of Updates")

    # disable splines
    for _, spine in axis.spines.items():
        spine.set_visible(False)

    axis.set_xticks(np.arange(0.5, update_count.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(0.5, update_count.shape[0], 1), minor=True)

    yticks = np.arange(0, update_count.shape[0], ytick_stride)
    axis.set_yticks(yticks)
    axis.set_xticks(np.arange(0, update_count.shape[1], xtick_stride))

    if binary_yticklabels:
        axis.set_yticklabels([bin(i)[2:] for i in yticks])

    axis.set_xlim(0.5, update_count.shape[1])
    axis.set_ylim(-0.5, update_count.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_title("Experience")


def plot_q_table_image(
        possible_actions,
        q_table,
        axis=None,
        ytick_stride=2,
        xtick_stride=2,
        binary_yticklabels=False,
):
    """Plots the preferred action in each state for a given q table."""

    if not axis:
        axis = plt.gca()

    # the actions with the highest perceived value
    best_actions = np.argmax(q_table, axis=2)

    best_actions = np.swapaxes(best_actions, 0, 1)
    best_actions = best_actions[:, :-1]

    colour_tuple = ("g", "b", "m")

    fmt = ticker.FixedFormatter(possible_actions)
    cbar_ticks_len = len(possible_actions)
    cbar_ticks = np.arange(cbar_ticks_len, dtype="f")

    if cbar_ticks_len == 3:
        cbar_ticks[0] = 1./3
        cbar_ticks[cbar_ticks_len -1] = cbar_ticks_len - 1 - 1./3
    elif cbar_ticks_len == 2:
        cbar_ticks[0] = 1./4
        cbar_ticks[cbar_ticks_len -1] = cbar_ticks_len - 1 - 1./4

    cmap = colors.ListedColormap(colour_tuple[:cbar_ticks_len])

    image = axis.imshow(best_actions, cmap=cmap)
    axis.figure.colorbar(image, ax=axis, format=fmt, ticks=cbar_ticks)

    # disable splines
    for _, spine in axis.spines.items():
        spine.set_visible(False)

    axis.set_xticks(np.arange(0.5, best_actions.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(0.5, best_actions.shape[0], 1), minor=True)

    yticks = np.arange(0, best_actions.shape[0], ytick_stride)
    axis.set_yticks(yticks)
    axis.set_xticks(np.arange(0, best_actions.shape[1], xtick_stride))

    if binary_yticklabels:
        axis.set_yticklabels([bin(i)[2:] for i in yticks])

    axis.set_xlim(0.5, best_actions.shape[1])
    axis.set_ylim(-0.5, best_actions.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_title("Learnt Policy")


def plot_q_table_action_image(
        possible_actions,
        action_idx,
        q_table,
        axis=None,
        ytick_stride=2,
        xtick_stride=2,
        binary_yticklabels=False,
):
    """Plots the relative value of an action at each time step."""

    if not axis:
        axis = plt.gca()

    diff_actions = q_table[:, :, action_idx]/np.sum(q_table, axis=2)

    diff_actions = np.swapaxes(diff_actions, 0, 1)
    diff_actions = diff_actions[:, :-1]

    image = axis.imshow(diff_actions, cmap="gist_rainbow", vmin=0, vmax=1)
    cbar = axis.figure.colorbar(image, ax=axis)
    cbar.ax.set_ylabel("Relative Value of Action")

    # disable splines
    for _, spine in axis.spines.items():
        spine.set_visible(False)

    axis.set_xticks(np.arange(0.5, diff_actions.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(0.5, diff_actions.shape[0], 1), minor=True)

    yticks = np.arange(0, diff_actions.shape[0], ytick_stride)
    axis.set_yticks(yticks)
    axis.set_xticks(np.arange(0, diff_actions.shape[1], xtick_stride))

    if binary_yticklabels:
        axis.set_yticklabels([bin(i)[2:] for i in yticks])

    axis.set_xlim(0.5, diff_actions.shape[1])
    axis.set_ylim(-0.5, diff_actions.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_title(f"Relative Value of '{possible_actions[action_idx]}'")
