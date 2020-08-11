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

    if config["agent"]["type"] == "QLearningAgent":
        smart_agent = env.QLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            quantisation_levels=config["agent"]["quantisation levels"],
            state_components=config["agent"]["state components"],
        )
        name = config['name']
        smart_agent.load_q_table(f"{name}.np")

        fig, axs = plt.subplots(2, len(config["agent"]["state components"]),
                                sharey=False)

        if len(config["agent"]["state components"]) < 2:
            comp = config["agent"]["state components"][0]
            smart_agent.plot_q_table(axs[0], comp)
            axs[0].legend()
            smart_agent.plot_q_table(axs[1], comp, normalise=False)
            axs[1].legend()
        else:
            for idx, comp in enumerate(config["agent"]["state components"]):
                smart_agent.plot_q_table(axs[0, idx], comp)
                axs[0, idx].legend()
                smart_agent.plot_q_table(axs[1, idx], comp, normalise=False)
                axs[1, idx].legend()

    else:
        smart_agent = env.SimpleMCAgent(
            config["deadline"],
        )
        name = config['name']
        smart_agent.load_q_table(f"{name}.np")

        fig, axs = plt.subplots(2, 1, sharey=False)

        smart_agent.plot_q_table(axs[0])
        axs[0].legend()
        smart_agent.plot_q_table(axs[1], normalise=False)
        axs[1].legend()

    plt.show()
