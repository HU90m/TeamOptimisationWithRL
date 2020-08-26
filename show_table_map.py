"""A script for generating a visual representation of an agent's table."""

import sys
import numpy as np
import matplotlib.pyplot as plt
from matplotlib import colors, ticker

from actions import ACTION_STR
from agents import load_agent_and_settings


def plot_q_table_map(possible_actions, q_table, axis=None):

    if not axis:
        axis = plt.gca()

    # the actions with the highest perceived value
    best_actions = np.argmax(q_table, axis=2)

    shape = best_actions.shape

    if best_actions.shape[0] > best_actions.shape[1]:
        best_actions = np.swapaxes(best_actions, 0, 1)

    best_actions = best_actions[:,:-1]

    colour_tuple = ["green", "blue", "purple"]


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

    im = axis.imshow(best_actions, cmap=cmap)
    cbar = axis.figure.colorbar(im, ax=axis, format=fmt, ticks=cbar_ticks)


    # disable splines
    for edge, spine in axis.spines.items():
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



def plot_update_count_map(possible_actions, update_count, axis=None):

    if not axis:
        axis = plt.gca()


    if update_count.shape[0] > update_count.shape[1]:
        update_count = np.swapaxes(update_count, 0, 1)

    update_count = update_count[:,:-1]

    im = axis.imshow(update_count, norm=colors.LogNorm())
    cbar = axis.figure.colorbar(im, ax=axis)
    cbar.ax.set_ylabel("Number of Updates")

    # disable splines
    for edge, spine in axis.spines.items():
        spine.set_visible(False)

    axis.set_xticks(np.arange(0.5, update_count.shape[1], 1), minor=True)
    axis.set_yticks(np.arange(0.5, update_count.shape[0], 1), minor=True)

    axis.set_xticks(np.arange(0, update_count.shape[1], 3))
    axis.set_yticks(np.arange(0, update_count.shape[0], 2))

    axis.set_xlim(0.5, update_count.shape[1])
    axis.set_ylim(-0.5, update_count.shape[0])

    axis.grid(which="minor", color="w", linestyle="-", linewidth=2)
    axis.tick_params(which="minor", bottom=False, left=False)

    axis.set_ylabel("Score")
    axis.set_xlabel("Time Step")
    axis.set_title("Agent Experience")



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

    fig, axs = plt.subplots(2, 1)

    plot_update_count_map(config["agent"]["possible actions"], agent.update_count, axis=axs[1])

    plot_q_table_map(config["agent"]["possible actions"], agent.q_table, axis=axs[0])

    plt.show()
