"""A script for comparing different policies."""

import random
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt

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


if __name__ == '__main__':
    random.seed(8574058)
    np.random.seed(random.getrandbits(32))
    N, K = 12, 5

    NUM_NODES = 40
    DEGREE = 6

    DRAW_GRAPH = False

    DEADLINE = 40
    ITERATIONS = 100

    NUM_PROCESSES = 4

    graph = nx.circulant_graph(NUM_NODES, range(DEGREE//2 +1))

    if DRAW_GRAPH:
        nx.draw_circular(graph, node_size=10, width=0.5)
        plt.show()

    randy = agents.QLearningAgent(
        DEADLINE,
        epsilon_decay=1e-6,
        possible_actions=(
            ACTION_NUM['step'],
            ACTION_NUM['best'],
            ACTION_NUM['modal'],
        )
    )


    comp1, _, _ = agents.load_agent_and_settings('agent/basic3/basic3.json', episodes=3500)
    comp2, _, _ = agents.load_agent_and_settings('agent/basic3/basic3.json')

    policies = {
        'conformity imitation then step' : {
            "strategy" : ACTION_FUNC[ACTION_NUM['modal_then_step']],
            "sample" : None,
            "colour" : "purple",
            "alpha" : 1,
        },
        'best member imitation then step' : {
            "strategy" : ACTION_FUNC[ACTION_NUM['best_then_step']],
            "sample" : None,
            "colour" : "red",
            "alpha" : 1,
        },
        'step then best member imitation' : {
            "strategy" : ACTION_FUNC[ACTION_NUM['step_then_best']],
            "sample" : None,
            "colour" : "orange",
            "alpha" : 1,
        },
        #'Random' : {
        #    "strategy" : randy.perform_greedy_action,
        #    "sample" : None,
        #    "colour" : "purple",
        #    "alpha" : 1,
        #},
        'Q learning Agent (3500 episodes)' : {
            "strategy" : comp1.perform_greedy_action,
            "sample" : None,
            "colour" : "blue",
            "alpha" : 1,
        },
    }

    policies['Q learning Agent (10000 episodes)'] = {
            "strategy" : comp2.perform_greedy_action,
            "sample" : None,
            "colour" : "green",
            "alpha" : 1,
    }

    sim_records = {}
    for policy_name in policies:
        sim_records[policy_name] = []

    for iteration in range(ITERATIONS):
        fitness_func, fitness_func_norm = \
            nkl.generate_fitness_func(N, K, num_processes=NUM_PROCESSES)
        for policy_name in policies:
            sim_record = env.SimulationRecord(
                N,
                NUM_NODES,
                DEADLINE,
                fitness_func,
                fitness_func_norm,
            )
            env.run_episode(
                graph,
                sim_record,
                policies[policy_name]["strategy"],
                neighbour_sample_size=policies[policy_name]["sample"],
            )
            sim_record.fill_fitnesses()
            sim_records[policy_name].append(sim_record)


    fitnesses = {}
    fitness_means = {}
    fitness_95confidence = {}
    for policy_name, policy_sim_records in sim_records.items():
        fitnesses[policy_name] = np.empty((ITERATIONS, DEADLINE))

        for iteration, sim_record in enumerate(policy_sim_records):
            # mean nodes
            fitnesses[policy_name][iteration] = \
                    np.mean(sim_record.fitnesses, axis=0)

        fitness_means[policy_name] = np.mean(fitnesses[policy_name], axis=0)
        fitness_95confidence[policy_name] = 1.96 \
                * np.std(fitnesses[policy_name], axis=0) / np.sqrt(ITERATIONS)


        line_and_error(
            plt,
            range(DEADLINE),
            fitness_means[policy_name],
            fitness_95confidence[policy_name],
            policy_name,
            policies[policy_name]["colour"],
            policies[policy_name]["alpha"],
        )

    plt.xlabel("Time Step")
    plt.ylabel("Average Score")
    plt.grid(True)
    plt.legend()
    plt.show()
