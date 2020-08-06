'''Modules for generating NK Landscapes.'''

import numpy as np
from numpy import random
from bitmanipulation import get_bit, set_bit


def generate_interaction_lists(num_bits, num_components):
    '''For each index, a list of K other random indices are generated.'''
    interaction_lists = np.empty((num_bits, num_components+1), dtype=int)

    for idx in range(num_bits):
        idx_list = list(range(num_bits))
        idx_list.remove(idx)
        random.shuffle(idx_list)

        chosen_indicies = [idx] + idx_list[0:num_components]
        interaction_lists[idx,] = np.sort(chosen_indicies)

    return interaction_lists


def generate_fitness_func(num_bits, num_components, fitness_8=True):
    '''
    Returns a list of the fitness function's output
    for each of the 2^N possible system configurations.
    '''
    interaction_lists = generate_interaction_lists(num_bits, num_components)

    # the output of each of the N fitness component functions
    # for each of their 2^(K+1) possible component configurations
    fitness_component = random.rand(2**(num_components+1), num_bits)

    # the output of each of the N fitness component functions
    # for each of the 2^N system configurations
    local_fitness = np.empty((2**num_bits, num_bits))

    # for each possible system configuration
    for system_config in range(2**num_bits):

        # for each bit of the system configuration
        for bit_idx in range(num_bits):

            # indices of this bits's dependant bits
            dep_bit_idxs = interaction_lists[bit_idx,]

            # creates the current component config
            # from the dependant bits' current states in the system config
            comp_config = 0
            for comp_bit_idx, dep_bit_idx in enumerate(dep_bit_idxs):
                comp_config = set_bit(
                    comp_config,
                    comp_bit_idx,
                    get_bit(system_config, dep_bit_idx),
                )

            # update the local fitnesses
            local_fitness[system_config, bit_idx] = \
                fitness_component[comp_config, bit_idx]

    # The fitness functions for each of the 2^N system configurations
    fitness = np.mean(local_fitness, axis=1)
    fitness_norm = fitness/max(fitness)

    if fitness_8:
        return fitness_norm**8

    return fitness_norm
