"""A script for generating a visual representation of an agent's table."""

import sys
import json
from time import time
import numpy as np
import matplotlib.pyplot as plt

import nklandscapes as nkl
import environment as env


def get_config():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a config json file.\n'
        )
        sys.exit(0)

    with open(sys.argv[1], 'rb') as file_handle:
        return json.load(file_handle)

if __name__ == '__main__':
    config = get_config()

    np.random.seed(config["seed"])

    smart_agent = env.QLearningAgent(
        config["deadline"],
        epsilon_decay=config["agent"]["epsilon decay"],
        quantisation_levels=config["agent"]["quantisation levels"],
    )
    name = config['name']

    smart_agent.load_q_table(f"{name}.np")

    fig, axs = plt.subplots(1, 2, sharey=True)

    for idx, comp in enumerate(['time', 'score']):
        smart_agent.plot_q_table(axs[idx], comp)
        axs[idx].legend()

    plt.show()
