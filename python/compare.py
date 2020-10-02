"""A script for comparing different policies."""

import json
import sys
from os import path
import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.colors as pltcolours

import nklandscapes as nkl
import environment as env

from actions import ACTION_NUM, ACTION_FUNC
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
                "action function" : ACTION_FUNC[ACTION_NUM[
                    strategy["action"]
                ]],
                "alpha" : strategy["alpha"],
            }
        elif strategy["type"] == "learnt":
            agent, _, _ = agents.load_agent_and_settings(
                path.join(config_dir, strategy["agent config"]),
                episode=strategy["episode"],
            )
            strategies[strategy["name"]] = {
                "action function" : agent.perform_greedy_action,
                "alpha" : strategy["alpha"],
            }

    # run episodes
    sim_records = {}
    for strategy_name in strategies:
        sim_records[strategy_name] = []

    for _ in range(config["episodes"]):
        fitness_func, fitness_func_norm = nkl.rusty_generate_fitness_func(
            config["nk landscape"]["N"],
            config["nk landscape"]["K"],
        )
        for strategy_name in strategies:
            sim_record = env.SimulationRecord(
                config["nk landscape"]["N"],
                config["graph"]["num_nodes"],
                config["deadline"],
                fitness_func,
                fitness_func_norm,
            )
            env.run_episode(
                graph,
                sim_record,
                strategies[strategy_name]["action function"],
            )
            sim_record.fill_fitnesses()
            sim_records[strategy_name].append(sim_record)


    # plot fitness comparison over these episodes
    colour_iterator = iter(pltcolours.TABLEAU_COLORS)

    fitnesses = {}
    fitness_means = {}
    fitness_95confidence = {}
    for strategy_name, strategy_sim_records in sim_records.items():
        fitnesses[strategy_name] = \
            np.empty((config["episodes"], config["deadline"]))

        for episode, sim_record in enumerate(strategy_sim_records):
            # mean nodes
            fitnesses[strategy_name][episode] = \
                    np.mean(sim_record.fitnesses, axis=0)

        fitness_means[strategy_name] = \
            np.mean(fitnesses[strategy_name], axis=0)

        fitness_95confidence[strategy_name] = 1.96 \
                * np.std(fitnesses[strategy_name], axis=0) \
                / np.sqrt(config["episodes"])

        if config["95confidence"] == True:
            line_and_error(
                plt,
                range(config["deadline"]),
                fitness_means[strategy_name],
                fitness_95confidence[strategy_name],
                strategy_name,
                next(colour_iterator),
                strategies[strategy_name]["alpha"],
            )
        else:
            plt.plot(
                range(config["deadline"]),
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
