"""A script for training a Q Agent."""

import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from time import time

import nklandscapes as nkl
import environment as env


def file_write(name, line):
    file_handle = open(name, 'w')
    file_handle.write(line)
    file_handle.close()

def file_append(name, line):
    file_handle = open(name, 'a')
    file_handle.write(line)
    file_handle.close()


if __name__ == '__main__':
    np.random.seed(42)
    N, K = 12, 5

    NUM_NODES = 60
    DEGREE = 4

    DEADLINE = 50
    ITERATIONS = 3


    # Fully connected graph
    print('generating graph')
    graph = nx.complete_graph(NUM_NODES)
    #graph = nx.random_regular_graph(DEGREE, NUM_NODES)

    #nx.draw_circular(graph)
    #plt.show()


    # make and train agent
    smart_agent = env.QLearningAgent(
        DEADLINE,
        epsilon_decay=1e-7,
        quantisation_levels=50,
        use_best_neighbour=False,
    )

    t0 = time()
    count = 0

    smart_agent.load_q_table('run1.np')

    while time() - t0 < 1800:
        fitness_func = nkl.generate_fitness_func(N, K, num_processes=3)

        env.run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            strategy=smart_agent.learn_and_perform_greedy_epsilon_action,
        )

        if not count % 100:
            mins_passed = (time() -t0)/60
            file_write('run2.txt', f'count, time = {count}, {mins_passed} minutes\n')
        count += 1


    smart_agent.save_q_table('run2.np')

    fitness_func = nkl.generate_fitness_func(N, K, num_processes=3)
    sim_record = env.run_episode(
        graph,
        N,
        DEADLINE,
        fitness_func,
        strategy=smart_agent.perform_best_action,
    )

    final_score = np.mean(sim_record.fitnesses[:, DEADLINE-1])
    file_append('run1.txt', f'final epsilon = {smart_agent.epsilon}\n')
    file_append('run1.txt', f'final score = {final_score}\n')

    sim_record.draw_actions_stack_plot(plt)
    plt.show()
