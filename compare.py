"""A script for comparing different policies."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt

import nklandscapes as nkl
import environment as env


def line_and_error(axis, x, y, y_std, label):
    axis.fill_between(x, y + y_std, y - y_std, alpha=0.2)
    axis.plot(x, y, label=label)


if __name__ == '__main__':
    np.random.seed(42)
    N, K = 12, 5

    NUM_NODES = 60
    DEGREE = 4

    DEADLINE = 50
    ITERATIONS = 20

    graph = nx.complete_graph(NUM_NODES)
    #graph = nx.random_regular_graph(DEGREE, NUM_NODES)


    smart_agent = env.QLearningAgent(
        DEADLINE,
        epsilon_decay=1e-6,
        quantisation_levels=50,
        use_best_neighbour=False,
    )
    smart_agent.load_q_table('run2.np')


    policies = {
            'best then step' : (
                env.action_best_then_step,
                5,
            ),
            'step then best' : (
                env.action_step_then_best,
                5,
            ),
            'modal then step' : (
                env.action_modal_then_step,
                5,
            ),
            'smarty pants' : (
                smart_agent.perform_best_action,
                None,
            ),
    }

    sim_records = {}
    for policy_name in policies:
        sim_records[policy_name] = []

    for iteration in range(ITERATIONS):
        fitness_func = nkl.generate_fitness_func(N, K, num_processes=3)
        for policy_name in policies:
             sim_records[policy_name].append(env.run_episode(
                graph,
                N,
                DEADLINE,
                fitness_func,
                strategy=policies[policy_name][0],
                neighbour_sample_size=policies[policy_name][1],
            ))


    fitnesses = {}
    fitness_means = {}
    fitness_stds = {}
    for policy_name, policy_sim_records in sim_records.items():
        fitnesses[policy_name] = np.empty((ITERATIONS, DEADLINE))

        for iteration, sim_record in enumerate(policy_sim_records):
            # mean nodes
            fitnesses[policy_name][iteration] = \
                    np.mean(sim_record.fitnesses, axis=0)

        fitness_means[policy_name] = np.mean(fitnesses[policy_name], axis=0)
        fitness_stds[policy_name] = np.std(fitnesses[policy_name], axis=0)


        line_and_error(
                plt,
                range(DEADLINE),
                fitness_means[policy_name],
                fitness_stds[policy_name],
                policy_name,
        )

    plt.legend()
    plt.show()
