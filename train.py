"""A script for training a Q Learning Agent."""

import sys
import json
from time import time
import numpy as np
import igraph as ig

import nklandscapes as nkl
import environment as env

from actions import ACTION_NUM
from agents import QLearningAgent, SimpleQLearningAgent, SimpleMCAgent


def file_write(name, line):
    """Write the given line to the file with the given name."""
    file_handle = open(name, 'w')
    file_handle.write(line)
    file_handle.close()


def file_append(name, line):
    """Appends the given line to the file with the given name."""
    file_handle = open(name, 'a')
    file_handle.write(line)
    file_handle.close()


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

    if config["graph"]["type"] == "regular":
        graph = ig.Graph.K_Regular(config["graph"]["num_nodes"],
                                   config["graph"]["degree"])
    elif config["graph"]["type"] == "full":
        graph = ig.Graph.Full(config["graph"]["num_nodes"])


    # make agent
    possible_actions = [
            ACTION_NUM[possible_action]
            for possible_action in config["agent"]["possible actions"]
    ]

    if config["agent"]["type"] == "QLearningAgent":
        smart_agent = QLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            quantisation_levels=config["agent"]["quantisation levels"],
            state_components=config["agent"]["state components"],
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )
        strategy_func = smart_agent.learn_and_perform_epsilon_greedy_action

    elif config["agent"]["type"] == "SimpleQLearningAgent":
        smart_agent = SimpleQLearningAgent(
            config["deadline"],
            epsilon_decay=config["agent"]["epsilon decay"],
            learning_rate=config["agent"]["learning rate"],
            discount_factor=config["agent"]["discount factor"],
            possible_actions=possible_actions,
        )
        strategy_func = smart_agent.learn_and_perform_epsilon_greedy_action

    elif config["agent"]["type"] == "SimpleMCAgent":
        smart_agent = SimpleMCAgent(
            config["deadline"],
            learning_rate=config["agent"]["learning rate"],
            possible_actions=possible_actions,
        )
        strategy_func=smart_agent.learn_and_perform_random_action

    if config["agent"]["load table"]:
        smart_agent.load_q_table(config["agent"]["load table"])


    # train agent
    name = config['name']
    output_file = f"{name}.txt"
    max_time = config["max training time"] * 60
    t0 = time()

    for episode in range(config["episodes"]):
        if not episode % 100:
            mins_passed = (time() -t0)/60
            file_write(output_file,
                       f'episodes = {episode}\n'
                       f'time = {mins_passed} minutes\n')
            if not config["agent"]["type"] == "SimpleMCAgent":
                file_append(output_file, f'epsilon = {smart_agent.epsilon}\n')
            smart_agent.save_q_table(f"{name}-{episode}.np")

        fitness_func = nkl.generate_fitness_func(
                config["nk landscape"]["N"],
                config["nk landscape"]["K"],
                num_processes=config["num processes"],
        )
        sim_record = env.run_episode(
            graph,
            config["nk landscape"]["N"],
            config["deadline"],
            fitness_func,
            strategy=strategy_func,
        )
        if time() - t0 > max_time:
            break


    # final file writes
    mins_passed = (time() -t0)/60
    file_write(output_file,
               f'episodes = {episode}\n'
               f'time = {mins_passed} minutes\n')
    if not config["agent"]["type"] == "SimpleMCAgent":
        file_append(output_file, f'epsilon = {smart_agent.epsilon}\n')

    smart_agent.save_q_table(f"{name}.np")
