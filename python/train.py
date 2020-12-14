"""A script for training a Agents."""

import sys
from os import path
from time import time
import random
import networkx as nx
import numpy as np

import nklandscapes as nkl
import environment as env
from agents import load_agent_and_settings

def file_write(name, line):
    """Write the given line to the file with the given name."""
    file_handle = open(name, 'w')
    file_handle.write(line)
    file_handle.close()


def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide a config json file.\n'
        )
        sys.exit(0)

    return sys.argv[1]


if __name__ == '__main__':
    config_file = get_args()

    # load agent
    agent, config, config_dir = \
        load_agent_and_settings(config_file, training=True)

    # seed random number generator
    random.seed(config["seed"])
    np.random.seed(random.getrandbits(32))


    # generate graph
    if config["graph"]["type"] == "regular":
        num_nodes = config["graph"]["num_nodes"]
        degree = config["graph"]["degree"]
        graph = nx.circulant_graph(num_nodes, range(degree//2 +1))

    elif config["graph"]["type"] == "full":
        graph = nx.complete_graph(config["graph"]["num_nodes"])


    # train agent
    name = config['name']
    output_file = path.join(config_dir, f"{name}.txt")
    max_time = config["max training time"] * 60
    t0 = time()

    for episode in range(config["episodes"]):
        if not episode % 100:
            mins_passed = (time() -t0)/60
            file_write(output_file,
                       f'episodes = {episode}\n'
                       f'time = {mins_passed} minutes\n')

            agent.save(
                path.join(config_dir, f"{name}-{episode}.npz"),
            )

        if config["use rust"]:
            fitness_func, fitness_func_norm = nkl.rusty_generate_fitness_func(
                config["nk landscape"]["N"],
                config["nk landscape"]["K"],
            )
        else:
            fitness_func, fitness_func_norm = nkl.generate_fitness_func(
                config["nk landscape"]["N"],
                config["nk landscape"]["K"],
                num_processes=config["max processes"],
            )

        sim_record = env.SimulationRecord(
            config["nk landscape"]["N"],
            config["graph"]["num_nodes"],
            config["deadline"],
            fitness_func,
            fitness_func_norm,
        )
        env.run_episode(graph, sim_record, agent.train)

        if time() - t0 > max_time:
            break


    # final file writes
    mins_passed = (time() -t0)/60
    file_write(output_file,
               f'episodes = {episode}\n'
               f'time = {mins_passed} minutes\n')

    agent.save(path.join(config_dir, f"{name}.npz"))
