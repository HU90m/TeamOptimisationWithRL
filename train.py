"""A script for training a Agents."""

import sys
import random
import networkx as nx
import numpy as np

from environment import Environment, get_action_num
import agents

def get_args():
    """Loads the config file given in the arguments."""
    if len(sys.argv) < 2:
        print(
            'Please provide an agent config json file.\n'
        )
        sys.exit(0)

    return sys.argv[1]


if __name__ == '__main__':
    config_file = get_args()

    # load agent
    agent, config = agents.from_config(config_file, get_action_num)

    train_env_config = config["training environment"]

    # seed random number generator
    random.seed(train_env_config["seed"])
    np.random.seed(random.getrandbits(32))


    # generate graph
    if train_env_config["graph"]["type"] == "regular":
        num_nodes = train_env_config["graph"]["num nodes"]
        degree = train_env_config["graph"]["degree"]
        graph = nx.circulant_graph(num_nodes, range(degree//2 +1))

    elif train_env_config["graph"]["type"] == "full":
        graph = nx.complete_graph(train_env_config["graph"]["num_nodes"])

    # the environment
    environment = Environment(
            train_env_config["nk landscape"]["N"],
            train_env_config["nk landscape"]["K"],
            graph,
            config["deadline"],
            max_processes=train_env_config["max processes"],
    )

    # train agent
    deadline = config["deadline"]
    num_nodes = train_env_config["graph"]["num nodes"]
    save_interval = train_env_config["save interval"]

    for episode in range(1, train_env_config["episodes"]):
        if not episode % save_interval:
            agent.save(suffix=episode)

        # subsiquent time steps
        for time in range(deadline):
            for node in range(num_nodes):
                # choose action to be taken by this node at this time
                action = agent.choose_epsilon_greedy_action(
                        time,
                        environment.get_node_fitness_norm(node, time),
                        )
                # take selected action
                environment.set_action(node, time, action)

            # run time step of environment
            environment.run_time_step(time)

            # learn from the last transition
            for node in range(num_nodes):
                agent.learn(
                        time,
                        environment.get_node_fitness_norm(node, time),
                        environment.get_node_action(node, time),
                        time +1,
                        environment.get_node_fitness_norm(node, time +1),
                        environment.get_node_fitness(node, +1),
                        )


        # generate a new fitness function and reset for the next iteraction
        environment.generate_new_fitness_func()
        environment.reset()


    agent.save(suffix="final")
