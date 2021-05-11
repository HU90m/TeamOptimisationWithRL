"""A module containing the RandomTableAgent.
An agent which can follows a randomly generated tabular policy.
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

import environment.bitmanipulation as bitm

###############################################################################
# Agent
###############################################################################
#
class RandomTableAgent:
    """An agent which follows a random tabular policy."""
    def __init__(self, config, _config_file, get_action_num_func):
        # settings
        self.deadline = config["deadline"]

        # set up actions
        self._possible_actions_str = config["possible actions"]
        self.possible_actions = []
        for action_name in self._possible_actions_str:
            # convert the actions names to numbers, with the given function
            self.possible_actions.append(get_action_num_func(action_name))

        # this maps the external action number to internal action indices
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

        rng = np.random.default_rng(config["table seed"])
        self._q_table = rng.uniform(size=q_table_shape)

        # for the agent interface
        self.episode_end = lambda : None

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

    def best_action(self, node, time, environment):
        """Returns the best action according the Q table."""
        current_state = self._find_state(node, time, environment)
        return self.possible_actions[np.argmax(self._q_table[current_state])]

    def explore_action(self, node, time, environment):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError(
                "A RandomTableAgent is kinda dumb, "
                "so doesn't explore and learn."
        )

    def save(self, suffix=None):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError(
                "A RandomTableAgent doesn't learn anything "
                "so has nothing to save and load."
        )

    def load(self, suffix=None):
        """This method is included for interface compatibility.
        It does not load anything."""
        pass

    def plot(self, figure):
        """Plots the q table and update count.
        Plots the q table and update count of an agent in the provided figure.
        """
        if self.state_space_type == "time memory":
            len_bin_labels = self.history
            y_stride = 1
            print(
                "Binary memory values:\n\t0 -", self._possible_actions_str[0],
                "\n\t1 -", self._possible_actions_str[1],
            )
            ylabel = "Binary Memory"
        else:
            len_bin_labels = None
            y_stride = 2
            ylabel = "Fitness"

        num_possible_actions = len(self.possible_actions)
        if num_possible_actions == 2:
            axs = figure.subplots(2, 1)

            plot_q_table_image(
                self._possible_actions_str,
                self._q_table,
                axis=axs[0],
                ytick_stride=y_stride,
                len_binary_yticklabels=len_bin_labels,
            )
            plot_q_table_action_image(
                self._possible_actions_str,
                0,
                self._q_table,
                axis=axs[1],
                ytick_stride=y_stride,
                len_binary_yticklabels=len_bin_labels,
            )

            for axis in axs:
                axis.set_ylabel(ylabel)
                axis.set_xlabel("Time Step")

        elif num_possible_actions == 3:

            axs = figure.subplots(2, 2)
            plot_q_table_image(
                self._possible_actions_str, self._q_table, axis=axs[0, 0]
            )
            plot_q_table_action_image(
                self._possible_actions_str, 0, self._q_table, axis=axs[1, 0]
            )
            plot_q_table_action_image(
                self._possible_actions_str, 1, self._q_table, axis=axs[0, 1]
            )
            plot_q_table_action_image(
                self._possible_actions_str, 2, self._q_table, axis=axs[1, 1]
            )
            for i in range(2):
                for j in range(2):
                    axs[i, j].set_ylabel("Fitness")
                    axs[i, j].set_xlabel("Time Step")

        else:
            raise ValueError("Can only plot tables for agents with"
                             " two or three possible actions."
                             " This agent has {num_possible_actions}.")

        plt.tight_layout()

###############################################################################
# Functions
###############################################################################
#
def plot_q_table_image(
        possible_actions,
        q_table,
        axis=None,
        ytick_stride=2,
        xtick_stride=2,
        len_binary_yticklabels=None,
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

    if len_binary_yticklabels:
        axis.set_yticklabels(
                [format(i, f'0{len_binary_yticklabels}b') for i in yticks]
        )

    axis.set_xlim(-0.5, best_actions.shape[1])
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
        len_binary_yticklabels=None,
        colour_map="plasma",
):
    """Plots the relative value of an action at each time step."""

    if not axis:
        axis = plt.gca()

    diff_actions = q_table[:, :, action_idx]/np.sum(q_table, axis=2)

    diff_actions = np.swapaxes(diff_actions, 0, 1)
    diff_actions = diff_actions[:, :-1]

    image = axis.imshow(diff_actions, cmap=colour_map, vmin=0, vmax=1)
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

    if len_binary_yticklabels:
        axis.set_yticklabels(
                [format(i, f'0{len_binary_yticklabels}b') for i in yticks]
        )

    axis.set_xlim(-0.5, diff_actions.shape[1])
    axis.set_ylim(-0.5, diff_actions.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_title(f"Relative Value of '{possible_actions[action_idx]}'")
