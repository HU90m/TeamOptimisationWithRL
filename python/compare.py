"""A script for comparing different policies."""

import json
import sys
from os import path
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolours

from environment import Environment, get_action_num

import agents


def line_and_error(axis, x_values, y_values, y_err, label, colour, alpha):
    "Plots a line and its error"
    axis.fill_between(x_values, y_values + y_err, y_values - y_err,
                      color=colour, alpha=alpha*0.1)
    axis.plot(x_values, y_values + y_err, linewidth=0.5,
              linestyle=":", color=colour, alpha=alpha)
    axis.plot(x_values, y_values - y_err, linewidth=0.5,
              linestyle=":", color=colour, alpha=alpha)
    axis.plot(x_values, y_values, label=label,
              color=colour, alpha=alpha, linewidth=1)


def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a compare json file.\n'
        )
        sys.exit(0)

    return sys.argv[1]


if __name__ == '__main__':
    config_location = get_args()
    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

    random.seed(config["seed"])
    np.random.seed(random.getrandbits(32))

    # generate graph
    if config["graph"]["type"] == "regular":
        num_nodes = config["graph"]["num_nodes"]
        degree = config["graph"]["degree"]
        graph = nx.circulant_graph(num_nodes, range(degree//2 +1))

    elif config["graph"]["type"] == "full":
        graph = nx.complete_graph(config["graph"]["num_nodes"])

    DRAW_GRAPH = False
    if DRAW_GRAPH:
        nx.draw_circular(graph, node_size=10, width=0.5)
        plt.show()

    # load strategies
    strategies = {}
    for strategy in config["strategies"]:
        if strategy["type"] == "heuristic":
            strategies[strategy["name"]] = {
                "action num" : get_action_num(strategy["action"]),
                "type" : "constant",
                "alpha" : strategy["alpha"],
            }
        elif strategy["type"] == "learnt":
            agent, _, _ = agents.load_agent_and_settings(
                path.join(config_dir, strategy["agent config"]),
                episode=strategy["episode"],
            )
            strategies[strategy["name"]] = {
                "action function" : agent.test,
                "alpha" : strategy["alpha"],
            }

    # the environment
    environment = Environment(
            config["nk landscape"]["N"],
            config["nk landscape"]["K"],
            graph,
            config["deadline"],
            max_processes=config["max processes"],
    )

    # fitnesses holds the mean fitness across all nodes at each time step
    # for each strategy and episode run.
    fitnesses = {}
    for strategy_name in strategies:
        fitnesses[strategy_name] = []

    for _ in range(config["episodes"]):
        environment.generate_new_fitness_func()

        for strategy_name, strategy in strategies.items():
            environment.reset()

            if strategy["type"] == "constant":
                # if a constant action strategy, set all action to
                # the constant action and run the full episode.
                environment.set_all_actions(strategy["action num"])
                environment.run_episode()

            elif strategy["type"] == "learnt":

                for time in range(config["deadline"]):
                    for node in range(config["graph"]["num_nodes"]):
                        environment.set_action(node, time, 1)

                    environment.run_time_step(time)

            fitnesses[strategy_name].append(environment.get_mean_fitnesses())

    # plot fitness comparison over these episodes
    colour_iterator = iter(pltcolours.TABLEAU_COLORS)

    fitness_means = {}
    fitness_95confidence = {}
    for strategy_name in fitnesses:
        # concatonate the list of numpy arrays into a two 2D numpy array
        fitnesses[strategy_name] = np.column_stack(fitnesses[strategy_name])

        fitness_means[strategy_name] = \
            np.mean(fitnesses[strategy_name], axis=1)

        fitness_95confidence[strategy_name] = 1.96 \
                * np.std(fitnesses[strategy_name], axis=1) \
                / np.sqrt(config["episodes"])

        if config["95confidence"]:
            line_and_error(
                plt,
                range(config["deadline"] +1),
                fitness_means[strategy_name],
                fitness_95confidence[strategy_name],
                strategy_name,
                next(colour_iterator),
                strategies[strategy_name]["alpha"],
            )
        else:
            plt.plot(
                range(config["deadline"] +1),
                fitness_means[strategy_name],
                label=strategy_name,
                color=next(colour_iterator),
                alpha=strategies[strategy_name]["alpha"],
                linewidth=1,
            )

    plt.xlabel("Time Step")
    plt.ylabel("Average Score")
    plt.title(config["title"])
    plt.grid(True)
    plt.legend()
    plt.show()
