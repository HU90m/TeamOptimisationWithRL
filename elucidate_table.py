"""A script for generating a visual representation of an agent's table."""

import sys
import matplotlib.pyplot as plt

from agents import load_agent_and_settings


def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a config json file.\n'
            'You can also provide a specific episode to view.'
        )
        sys.exit(0)

    if len(sys.argv) > 2:
        return load_agent_and_settings(sys.argv[1], episodes=int(sys.argv[2]))

    return load_agent_and_settings(sys.argv[1])


if __name__ == '__main__':

    agent, config, config_dir = get_args()

    if config["agent"]["type"] == "QLearningAgent":
        fig, axs = plt.subplots(2, len(config["agent"]["state components"]),
                                sharey=False)

        if len(config["agent"]["state components"]) < 2:
            state_comp = config["agent"]["state components"][0]
            agent.plot_q_table(axs[0], state_comp)
            axs[0].legend()
            agent.plot_q_table(axs[1], state_comp, normalise=False)
            axs[1].legend()
        else:
            for idx, state_comps in enumerate(
                    config["agent"]["state components"],
            ):
                agent.plot_q_table(axs[0, idx], state_comps)
                axs[0, idx].legend()
                agent.plot_q_table(axs[1, idx], state_comps, normalise=False)
                axs[1, idx].legend()
    else:
        fig, axs = plt.subplots(2, 1, sharey=False)

        agent.plot_q_table(axs[0])
        axs[0].legend()
        agent.plot_q_table(axs[1], normalise=False)
        axs[1].legend()


    print(agent.q_table)
    plt.show()
