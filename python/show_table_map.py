"""A script for generating a visual representation of an agent's table."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

from agents import load_agent_and_settings


def plot_update_count_image(update_count, axis=None):
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

    axis.set_xticks(np.arange(0, update_count.shape[1], 2))
    axis.set_yticks(np.arange(0, update_count.shape[0], 2))

    axis.set_xlim(0.5, update_count.shape[1])
    axis.set_ylim(-0.5, update_count.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_ylabel("Score")
    axis.set_xlabel("Time Step")
    axis.set_title("Experience")


def plot_q_table_image(possible_actions, q_table, axis=None):
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

    axis.set_xticks(np.arange(0, best_actions.shape[1], 2))
    axis.set_yticks(np.arange(0, best_actions.shape[0], 2))

    axis.set_xlim(0.5, best_actions.shape[1])
    axis.set_ylim(-0.5, best_actions.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_ylabel("Score")
    axis.set_xlabel("Time Step")
    axis.set_title("Learnt Policy")


def plot_q_table_action_image(
        possible_actions, p_action_idx, q_table, axis=None
):
    """Plots the relative value of an action at each time step."""

    if not axis:
        axis = plt.gca()

    diff_actions = q_table[:, :, p_action_idx]/np.sum(q_table, axis=2)

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

    axis.set_xticks(np.arange(0, diff_actions.shape[1], 2))
    axis.set_yticks(np.arange(0, diff_actions.shape[0], 2))

    axis.set_xlim(0.5, diff_actions.shape[1])
    axis.set_ylim(-0.5, diff_actions.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_ylabel("Score")
    axis.set_xlabel("Time Step")
    axis.set_title(f"Relative Value of '{possible_actions[p_action_idx]}'")


def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a config json file.\n'
            'You can also provide a specific episode to view.'
        )
        sys.exit(0)

    if len(sys.argv) > 2:
        return load_agent_and_settings(sys.argv[1], episodes=sys.argv[2])

    return load_agent_and_settings(sys.argv[1])


if __name__ == '__main__':
    agent, config, _ = get_args()

    if len(agent.possible_actions) == 2:
        fig, axs = plt.subplots(3, 1)

        plot_q_table_image(
            agent.get_possible_actions(), agent.q_table, axis=axs[0]
        )
        plot_q_table_action_image(
            agent.get_possible_actions(), 0, agent.q_table, axis=axs[1]
        )
        plot_update_count_image(agent.update_count, axis=axs[2])

    elif len(agent.possible_actions) == 3:

        fig, axs = plt.subplots(3, 2)
        plot_q_table_image(
            agent.get_possible_actions(), agent.q_table, axis=axs[0, 0]
        )
        plot_q_table_action_image(
            agent.get_possible_actions(), 0, agent.q_table, axis=axs[1, 0]
        )
        plot_q_table_action_image(
            agent.get_possible_actions(), 1, agent.q_table, axis=axs[2, 0]
        )
        plot_update_count_image(agent.update_count, axis=axs[0, 1])
        plot_q_table_action_image(
            agent.get_possible_actions(), 2, agent.q_table, axis=axs[1, 1]
        )
        axs[2, 1].remove()

    plt.tight_layout()
    plt.show()
