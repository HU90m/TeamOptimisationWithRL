#!/usr/bin/python3
"""A script for plotting a visual representation of an agent's policy."""

import sys
import matplotlib.pyplot as plt

import agents
from environment import get_action_num


def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a config json file.\n'
            'You can also provide a specific episode to view.'
        )
        sys.exit(0)

    agent, _ = agents.from_config(sys.argv[1], get_action_num)

    if len(sys.argv) > 2:
        agent.load(suffix=sys.argv[2])
    else:
        agent.load(suffix="final")

    if len(sys.argv) > 3:
        save_location = sys.argv[3]
    else:
        save_location = None

    return agent, save_location


if __name__ == '__main__':
    agent, save_location = get_args()

    figure = plt.figure(figsize=(8, 10))

    agent.plot(figure)

    if save_location:
        plt.savefig(save_location)
    else:
        plt.show()
