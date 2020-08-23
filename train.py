"""A script for training a Q Learning Agent."""

import sys
from os import path

from time import time
import igraph as ig

import nklandscapes as nkl
import environment as env

from agents import load_agent_and_settings


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

    agent, config, config_dir = \
        load_agent_and_settings(config_file, training=True)


    if config["graph"]["type"] == "regular":
        graph = ig.Graph.K_Regular(config["graph"]["num_nodes"],
                                   config["graph"]["degree"])
    elif config["graph"]["type"] == "full":
        graph = ig.Graph.Full(config["graph"]["num_nodes"])

    # provide appropriate training function
    if (config["agent"]["type"] == "QLearningAgent" or
            config["agent"]["type"] == "SimpleQLearningAgent"):

        if config["agent"]["rewards"] == "all":
            strategy_func = \
                agent.learn_all_rewards_and_perform_epsilon_greedy_action

        elif config["agent"]["rewards"] == "end":
            strategy_func = \
                agent.learn_end_reward_and_perform_epsilon_greedy_action

    elif config["agent"]["type"] == "SimpleMCAgent":
        strategy_func = agent.learn_and_perform_random_action


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

            if not config["agent"]["type"] == "SimpleMCAgent":
                file_append(output_file, f'epsilon = {agent.epsilon}\n')

            agent.save_q_table(
                path.join(config_dir, f"{name}-{episode}.np"),
            )

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
        file_append(output_file, f'epsilon = {agent.epsilon}\n')

    agent.save_q_table(path.join(config_dir, f"{name}.np"))
