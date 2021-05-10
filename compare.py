#!/usr/bin/python3
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
            'You can provide a save location for the figure'
            ' as a second argument.\n'
        )
        sys.exit(0)

    save_location = None
    if len(sys.argv) > 2:
        save_location = sys.argv[2]

    return sys.argv[1], save_location


if __name__ == '__main__':
    config_location, save_location = get_args()
    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

    random.seed(config["seed"])
    np.random.seed(random.getrandbits(32))

    # generate graph
    graph_type = config["graph"]["type"]
    if graph_type == "regular":
        num_nodes = config["graph"]["num nodes"]
        degree = config["graph"]["degree"]
        graph = nx.circulant_graph(num_nodes, range(degree//2 +1))

    elif graph_type == "full":
        graph = nx.complete_graph(config["graph"]["num nodes"])

    else:
        raise ValueError(f"Graph type '{graph_type}' is not supported")

    DRAW_GRAPH = False
    if DRAW_GRAPH:
        nx.draw_circular(graph, node_size=10, width=0.5)
        plt.show()

    # load strategies
    strategies = {}
    for strategy_cfg in config["strategies"]:
        strategy_type = strategy_cfg["type"]
        if strategy_type == "constant":
            # add strategy to strategies dictionary
            strategies[strategy_cfg["name"]] = {
                "action num" : get_action_num(strategy_cfg["action"]),
                "type" : strategy_cfg["type"],
                "alpha" : strategy_cfg["alpha"],
            }
        elif strategy_type in ("learnt", "variable"):
            agent, agent_config = agents.from_config(
                    strategy_cfg["config file"],
                    get_action_num,
            )
            assert(agent_config["deadline"] == config["deadline"])
            if strategy_type == "learnt":
                if strategy_cfg["episode"]:
                    agent.load(suffix=strategy_cfg["episode"])
                else:
                    agent.load(suffix="final")

                agent_env_config = agent_config["training environment"]
                if agent_env_config["graph"] != config["graph"]:
                    print("Warning: '" + strategy_cfg["name"] + \
                          "' was trained on a different graph configuration.")
                if agent_env_config["nk landscape"] != config["nk landscape"]:
                    print("Warning: '" + strategy_cfg["name"] + \
                          "' was trained on a different",
                          "nk landscape configuration.")

            # add strategy to strategies dictionary
            strategies[strategy_cfg["name"]] = {
                "agent" : agent,
                "type" : strategy_cfg["type"],
                "alpha" : strategy_cfg["alpha"],
            }

        else:
            raise ValueError(f"Strategy type '{strategy_type}'"
                              " is not supported")

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

        for strategy_name, strategy_cfg in strategies.items():
            environment.reset()

            if strategy_cfg["type"] == "constant":
                # if a constant action strategy, set all action to
                # the constant action and run the full episode.
                environment.set_all_actions(strategy_cfg["action num"])
                environment.run_episode()

            elif strategy_cfg["type"] in ("learnt", "variable"):
                for time in range(config["deadline"]):
                    for node in range(config["graph"]["num nodes"]):
                        action = strategy_cfg["agent"].best_action(
                                node,
                                time,
                                environment,
                                )
                        environment.set_action(node, time, action)

                    environment.run_time_step(time)

            fitnesses[strategy_name].append(environment.get_mean_fitnesses())


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

        # plot fitness comparison over these episodes
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
    if save_location:
        plt.savefig(save_location)
    plt.show()
