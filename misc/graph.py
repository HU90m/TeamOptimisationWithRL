#!/usr/bin/python3
"""Script for exploring graphs."""
import json
from argparse import ArgumentParser
import networkx as nx
from networkx.algorithms.shortest_paths.astar import astar_path_length
import matplotlib.pyplot as plt


def parse_args():
    """Parses the CLI arguments."""
    parser = ArgumentParser(description='Find longest path length of a graph.')

    parser.add_argument('-c', '--config', dest='config', type=str,
            required=True, help='number of episodes which will be run.')

    parser.add_argument('-t', '--config-type',
            dest='config_type',
            type=str,
            default="comparison",
            help="Whether the config is a 'comparison' or 'agent' config.")

    parser.add_argument('-d', '--draw',
            dest='draw',
            action="store_true",
            help="When provided the graph will be drawn.")

    parser.add_argument('-p', '--propagation-delay',
            dest='prop_delay',
            action="store_true",
            help="Find the largest propagation delay between a pair of nodes.")

    return parser.parse_args()

def load_graph(graph_config):
    """Loads graph from provided configuration dictionary."""
    print("Graph:")
    if graph_config["type"] == "regular":
        num_nodes = graph_config["num nodes"]
        degree = graph_config["degree"]
        graph = nx.circulant_graph(num_nodes, range(degree//2 +1))
        print("\tType: regular")
        print("\tNumber of Nodes: " + str(num_nodes))
        print("\tDegree: " + str(degree))

    elif graph_config["type"] == "full":
        graph = nx.complete_graph(graph_config["num nodes"])
        print("\tType: fully connected")
        print("\tNumber of Nodes: " + str(graph_config["num nodes"]))

    else:
        raise ValueError(
                "Graph type '" + graph_config["type"] + "' is not supported"
                )
    return graph


def propagation_delay(graph):
    """Finds the largest propagation delay between a pair of nodes."""
    largest_delay = 0
    for starting_node in graph.nodes:
        other_nodes = list(graph.nodes)
        other_nodes.remove(starting_node)
        for ending_node in other_nodes:
            delay = astar_path_length(graph, starting_node, ending_node)
            largest_delay = max(delay, largest_delay)

    return largest_delay


if __name__ == '__main__':
    args = parse_args()

    with open(args.config, 'rb') as file_handle:
        raw_config = json.load(file_handle)

    # extract graph config
    if args.config_type == "agent":
        graph_config = raw_config["training environment"]["graph"]

    elif args.config_type == "comparison":
        graph_config = raw_config["graph"]

    else:
        raise ValueError(
                "There are only two config types 'comparison' and 'agent'."
                )

    graph = load_graph(graph_config)

    if args.prop_delay:
        print("\tPropagation Delay:", propagation_delay(graph))

    if args.draw:
        nx.draw_circular(graph, node_size=10, width=0.5)
        plt.show()
