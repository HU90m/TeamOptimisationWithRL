"""A module containing all the actions that can be attempted by a worker."""

from collections import Counter
import numpy as np
from numpy import random

import bitmanipulation as bitm


###############################################################################
# Constants
###############################################################################
#
# Actions
#
# action index -> action name
ACTION_STR = [
        'step',
        'teleport',
        'best',
        'modal',
        'best_then_step',
        'step_then_best',
        'modal_then_step',
]
# action name -> action index
ACTION_NUM = {
        'step' : 0,
        'teleport' : 1,
        'best' : 2,
        'modal' : 3,
        'best_then_step' : 4,
        'step_then_best' : 5,
        'modal_then_step' : 6,
}
# action index -> action function
ACTION_FUNC = {}

#
# Outcomes
#
# outcome index -> outcome name
OUTCOME_STR = [
        'no change',
        'stepped',
        'teleported',
        'copied best',
        'copied a modal',
]
# outcome name -> outcome index
OUTCOME_NUM = {
        'no change' : 0,
        'stepped' : 1,
        'teleported' : 2,
        'copied best' : 3,
        'copied a modal' : 4,
}


###############################################################################
# Action Function
###############################################################################
#
def action_step(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """Worker takes a random step, if it leads to a better position"""
    current_pos = sim_record.positions[node, time]
    random_step_pos = bitm.flip_random_bit(num_bits, current_pos)

    # if better, take random step
    if fitness_func[random_step_pos] > fitness_func[current_pos]:
        sim_record.positions[node, time +1] = random_step_pos
        sim_record.outcomes[node, time+1] = OUTCOME_NUM['stepped']
        return True

    # no change but this may be overwritten if another action is attempted
    sim_record.positions[node, time +1] = sim_record.positions[node, time]
    sim_record.outcomes[node, time+1] = OUTCOME_NUM['no change']
    return False

ACTION_FUNC[ACTION_NUM["step"]] = action_step


def action_teleport(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """Randomly teleports worker, if better than worker's current position"""
    current_pos = sim_record.positions[node, time]
    teleport_pos = random.randint(num_bits)

    # if better, randomly teleports
    if fitness_func[teleport_pos] > fitness_func[current_pos]:
        sim_record.positions[node, time +1] = teleport_pos
        sim_record.outcomes[node, time+1] = OUTCOME_NUM['teleported']
        return True

    # no change but this may be overwritten if another action is attempted
    sim_record.positions[node, time +1] = sim_record.positions[node, time]
    sim_record.outcomes[node, time+1] = OUTCOME_NUM['no change']
    return False

ACTION_FUNC[ACTION_NUM["teleport"]] = action_teleport


def action_best(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Worker copies their best colleague, if the colleague has a better position.
    """
    neighbour_fitnesses = [
        fitness_func[sim_record.positions[neighbour, time]]
        for neighbour in neighbours
    ]
    best_neighbour_pos = \
        sim_record.positions[np.argmax(neighbour_fitnesses), time]

    current_fitness = fitness_func[sim_record.positions[node, time]]

    # if better, copy best neighbour
    if fitness_func[best_neighbour_pos] > current_fitness:
        sim_record.positions[node, time +1] = best_neighbour_pos
        sim_record.outcomes[node, time+1] = OUTCOME_NUM['copied best']
        return True

    # no change but this may be overwritten if another action is attempted
    sim_record.positions[node, time +1] = sim_record.positions[node, time]
    sim_record.outcomes[node, time+1] = OUTCOME_NUM['no change']
    return False

ACTION_FUNC[ACTION_NUM["best"]] = action_best


def action_modal(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    neighbour_fitnesses = [
        fitness_func[sim_record.positions[neighbour, time]]
        for neighbour in neighbours
    ]

    # finds a neighbour with modal fitness
    neighbour_fitnesses_freq = Counter(neighbour_fitnesses)

    min_freq = 1
    modal_fitness = 0
    for key, value in neighbour_fitnesses_freq.items():
        if value > min_freq:
            min_freq = value
            modal_fitness = key

    # if better, copy a random modal neighbour
    if modal_fitness > fitness_func[sim_record.positions[node, time]]:
        modal_neighbour_idxs = \
                np.where(neighbour_fitnesses == modal_fitness)[0]
        neighbour = neighbours[random.choice(modal_neighbour_idxs)]

        sim_record.positions[node, time +1] = \
            sim_record.positions[neighbour, time]
        sim_record.outcomes[node, time+1] = OUTCOME_NUM['copied a modal']
        return True

    # no change but this may be overwritten if another action is attempted
    sim_record.positions[node, time +1] = sim_record.positions[node, time]
    sim_record.outcomes[node, time+1] = OUTCOME_NUM['no change']
    return False

ACTION_FUNC[ACTION_NUM["modal"]] = action_modal


def action_best_then_step(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    success = action_best(num_bits, time, node,
                          neighbours, fitness_func, sim_record)
    if not success:
        success = action_step(num_bits, time, node,
                              neighbours, fitness_func, sim_record)
    return success

ACTION_FUNC[ACTION_NUM["best_then_step"]] = action_best_then_step


def action_step_then_best(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, keep last position.
    """
    success = action_step(num_bits, time, node,
                          neighbours, fitness_func, sim_record)
    if not success:
        success = action_best(num_bits, time, node,
                              neighbours, fitness_func, sim_record)
    return success

ACTION_FUNC[ACTION_NUM["step_then_best"]] = action_step_then_best


def action_modal_then_step(
        num_bits,
        time,
        node,
        neighbours,
        fitness_func,
        sim_record,
):
    """
    Try to find a fitness_func occurring more than once among neighbours.
    If successful, take the position of one of these neighbours at random.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    success = action_modal(num_bits, time, node,
                           neighbours, fitness_func, sim_record)
    if not success:
        success = action_step(num_bits, time, node,
                              neighbours, fitness_func, sim_record)
    return success

ACTION_FUNC[ACTION_NUM["modal_then_step"]] = action_modal_then_step
