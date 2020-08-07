'''Modules for generating NK Landscapes.'''

import multiprocessing as mp
import numpy as np

from bitmanipulation import get_bit, set_bit


def generate_interaction_lists(num_bits, num_components):
    '''For each index, a list of K other random indices are generated.'''
    interaction_lists = np.empty((num_bits, num_components+1), dtype=int)

    for idx in range(num_bits):
        idx_list = list(range(num_bits))
        idx_list.remove(idx)
        np.random.shuffle(idx_list)

        chosen_indicies = [idx] + idx_list[0:num_components]
        interaction_lists[idx,] = np.sort(chosen_indicies)

    return interaction_lists


def find_solution_fitnesses(
        num_bits,
        interaction_lists,
        component_fitness_funcs,

        # solutions to work on
        start_solution,
        end_solution,

        # output
        fitness_func,
):
    '''
    Finds the output of the fitness function,
    in the range of solutions from start_solution to end_solution.
    '''
    # for each possible solution
    for solution in range(start_solution, end_solution):

        sum_of_component_functions = 0

        # for each bit of the solution
        for bit_idx in range(num_bits):

            # indices of this bit's dependant bits
            dep_bit_idxs = interaction_lists[bit_idx,]

            # creates the current component config
            # from the dependant bits' current states in the solution
            comp_config = 0
            for comp_bit_idx, dep_bit_idx in enumerate(dep_bit_idxs):
                comp_config = set_bit(
                    comp_config,
                    comp_bit_idx,
                    get_bit(solution, dep_bit_idx),
                )

            sum_of_component_functions += \
                component_fitness_funcs[bit_idx, comp_config]

        # return fitness value
        fitness_func[solution] = sum_of_component_functions / num_bits


def generate_fitness_func(
        num_bits,
        num_components,
        fitness_8=True,
        num_processes=1,
):
    '''
    Returns a list of the fitness function's output
    for each of the 2^N possible solutions
    '''
    if num_bits < num_components +1:
        raise ValueError('The following must be true:'
                         ' num_bits >= num_components +1')

    num_solutions = 1<<num_bits

    interaction_lists = generate_interaction_lists(num_bits, num_components)

    # the output of each of the N fitness component functions
    # for each of their 2^(K+1) possible component configurations
    component_fitness_funcs = np.random.rand(num_bits, 1<<(num_components+1))


    # finds the fitness_func
    # the output of each of the fitness function
    # for each of the 2^N solutions
    fitness_func = np.empty(num_solutions)

    if num_processes > 1:
        # setting up shared memory
        fitness_func_shared = mp.Array('d', fitness_func)

        processes = []
        for process_idx in range(num_processes):
            start = int(num_solutions * process_idx/num_processes)
            end = int(num_solutions * (process_idx+1)/num_processes)
            process = mp.Process(target=find_solution_fitnesses, args=(
                num_bits,
                interaction_lists,
                component_fitness_funcs,
                start,
                end,
                fitness_func_shared,
            ))
            process.start()
            processes.append(process)

        for process in processes:
            process.join()

        fitness_func = np.array(fitness_func_shared)

    else:
        fitness_func = np.empty(num_solutions)

        find_solution_fitnesses(
            num_bits,
            interaction_lists,
            component_fitness_funcs,
            0,
            num_solutions,
            fitness_func,
        )

    # The normalised fitness function for each of the 2^N possible solutions
    fitness_func_norm = fitness_func/max(fitness_func)

    if fitness_8:
        return fitness_func_norm**8

    return fitness_func_norm


if __name__ == '__main__':
    from time import time

    N, K = 10, 5

    np.random.seed(42)
    t0 = time()
    fitnesses1 = generate_fitness_func(N, K, num_processes=4)
    t1 = time()

    np.random.seed(42)
    t2 = time()
    fitnesses2 = generate_fitness_func(N, K)
    t3 = time()

    assert np.array_equal(fitnesses1, fitnesses2)

    print(f'num_processes=4: {t1-t0}s')
    print(f'num_processes=1: {t3-t2}s')
