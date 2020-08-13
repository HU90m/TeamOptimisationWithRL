"""A script for comparing different policies."""

import numpy as np
import igraph as ig
import matplotlib.pyplot as plt

import nklandscapes as nkl
import environment as env


def line_and_error(axis, x, y, y_err, label, colour, alpha):
    axis.fill_between(x, y + y_err, y - y_err, color=colour, alpha=alpha*0.1)
    axis.plot(x, y + y_err, linewidth=0.5,
              linestyle=":", color=colour, alpha=alpha)
    axis.plot(x, y - y_err, linewidth=0.5,
              linestyle=":", color=colour, alpha=alpha)
    axis.plot(x, y, label=label, color=colour, alpha=alpha, linewidth=1)


if __name__ == '__main__':
    np.random.seed(789543218)
    N, K = 12, 5

    NUM_NODES = 60
    DEGREE = 4

    DEADLINE = 50
    ITERATIONS = 500

    #graph = ig.Graph.Full(NUM_NODES)
    graph = ig.Graph.K_Regular(NUM_NODES, DEGREE)


    time_only = env.SimpleQLearningAgent(
        DEADLINE,
        epsilon_decay=1e-6,
    )
    time_only.load_q_table('trained/time_only.np')

    policies = {
            'conformity imitation then step' : {
                "strategy" : env.action_modal_then_step,
                "sample" : None,
                "colour" : "green",
                "alpha" : 1,
            },
            'best member imitation then step' : {
                "strategy" : env.action_best_then_step,
                "sample" : None,
                "colour" : "blue",
                "alpha" : 1,
            },
            'step then best member imitation' : {
                "strategy" : env.action_step_then_best,
                "sample" : None,
                "colour" : "orange",
                "alpha" : 1,
            },
            'Q learning agent' : {
                "strategy" : time_only.perform_greedy_action,
                "sample" : None,
                "colour" : "purple",
                "alpha" : 1,
            },
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
                strategy=policies[policy_name]["strategy"],
                neighbour_sample_size=policies[policy_name]["sample"],
            ))


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
