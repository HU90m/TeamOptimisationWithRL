
import sys
import numpy as np
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
        return load_agent_and_settings(sys.argv[1], episode=sys.argv[2])

    return load_agent_and_settings(sys.argv[1])

if __name__ == "__main__":
    agent, config, _ = get_args()

    print(f"f(t) = {agent.w_0} + {agent.w_1} * t")
    print(f"baseline: {agent.baseline}")

    t = np.arange(0, 1, 0.01)

    p = [agent.policy(time) for time in t]

    fig, ax = plt.subplots()

    ax.plot(t, p)

    ax.set_ylim(0, 1)

    ax.grid()

    plt.show()
