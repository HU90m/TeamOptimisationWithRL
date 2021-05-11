"""A module containing all the actions that can be attempted by a worker."""

from collections import Counter
import numpy as np
from numpy import random

import environment.bitmanipulation as bitm


###############################################################################
# Constants
###############################################################################
#
# Actions
#
# action index -> action name
ACTION_STR = [
        'null',
        'step',
        'teleport',
        'best',
        'modal',
        'best_then_step',
        'step_then_best',
        'modal_then_step',
        '40_step_60_best',
        '50_step_50_best',
        '60_step_40_best',
        '80_step_20_best',
]
# action name -> action index
ACTION_NUM = {}
for number, name in enumerate(ACTION_STR):
    ACTION_NUM[name] = number
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
OUTCOME_NUM = {}
for number, name in enumerate(OUTCOME_STR):
    OUTCOME_NUM[name] = number


###############################################################################
# Action Function
###############################################################################
#
def action_null(env, node, time):
    """Carries out null action.
    The null action raises an error to catch when no action has been selected.
    The null action has the default action number of 0.
    """
    raise RuntimeError(f"No action selected for node {node} at time {time}.")

ACTION_FUNC[ACTION_NUM["null"]] = action_null


def action_step(env, node, time):
    """Worker takes a random step, if it leads to a better position"""
    current_pos = env.get_node_position(node, time)
    current_fitness = env.get_fitness(current_pos)
    random_step_pos = bitm.flip_random_bit(env.num_bits, current_pos)

    # if better, take random step
    if env.get_fitness(random_step_pos) > current_fitness:
        return (
                OUTCOME_NUM['stepped'],
                random_step_pos,
                True,
        )
    return (
            OUTCOME_NUM['no change'], # action outcome
            current_pos, # next position
            False, # action success boolean
    )

ACTION_FUNC[ACTION_NUM["step"]] = action_step


def action_teleport(env, node, time):
    """Randomly teleports worker, if better than worker's current position"""
    current_fitness = env.get_node_fitness(node, time)
    teleport_pos = random.randint(env.num_bits)

    # if better, randomly teleports
    if env.get_fitness(teleport_pos) > current_fitness:
        return (
                OUTCOME_NUM['teleported'],
                teleport_pos,
                True,
        )
    return (
            OUTCOME_NUM['no change'], # action outcome
            env.get_node_position(node, time), # next position
            False, # action success boolean
    )

ACTION_FUNC[ACTION_NUM["teleport"]] = action_teleport


def action_best(env, node, time):
    """
    Worker copies their best colleague, if the colleague has a better position.
    """
    neighbours, neighbour_fitnesses = env.get_neighbours_fitnesses(node, time)

    # find the neighbour with the highest fitness
    best_neighbour = neighbours[np.argmax(neighbour_fitnesses)]
    # find this neighbour's current position
    best_neighbour_pos = env.get_node_position(best_neighbour, time)
    # find node's current fitness
    current_fitness = env.get_node_fitness(node, time)

    # if the best neighbour has a better fitness, copy this neighbour
    if env.get_fitness(best_neighbour_pos) > current_fitness:
        return (
                OUTCOME_NUM['copied best'],
                best_neighbour_pos,
                True,
        )
    return (
            OUTCOME_NUM['no change'], # action outcome
            env.get_node_position(node, time), # next position
            False, # action success boolean
    )

ACTION_FUNC[ACTION_NUM["best"]] = action_best


def action_modal(env, node, time):
    """
    If more than one colleague has the same fitness,
    the worker copies one of these colleagues.
    """
    neighbours, neighbour_fitnesses = env.get_neighbours_fitnesses(node, time)

    # finds a neighbour with modal fitness
    neighbour_fitnesses_freq = Counter(neighbour_fitnesses)

    min_freq = 1
    modal_fitness = 0
    for key, value in neighbour_fitnesses_freq.items():
        if value > min_freq:
            min_freq = value
            modal_fitness = key

    current_fitness = env.get_node_fitness(node, time)

    # if better, copy a random modal neighbour
    if modal_fitness > current_fitness:
        modal_neighbour_idxs = \
                np.where(neighbour_fitnesses == modal_fitness)[0]
        neighbour = neighbours[random.choice(modal_neighbour_idxs)]

        return (
                OUTCOME_NUM['copied a modal'],
                env.get_node_position(neighbour, time),
                True,
        )
    return (
            OUTCOME_NUM['no change'], # action outcome
            env.get_node_position(node, time), # next position
            False, # action success boolean
    )

ACTION_FUNC[ACTION_NUM["modal"]] = action_modal


def action_best_then_step(env, node, time):
    """
    Find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    outcome, next_position, success = action_best(env, node, time)
    if not success:
        outcome, next_position, success = action_step(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["best_then_step"]] = action_best_then_step


def action_step_then_best(env, node, time):
    """
    Check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, find the neighbour with the best performance.
    If their performance is better, take their position.
    Otherwise, keep last position.
    """
    outcome, next_position, success = action_step(env, node, time)
    if not success:
        outcome, next_position, success = action_best(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["step_then_best"]] = action_step_then_best


def action_modal_then_step(env, node, time):
    """
    Try to find a fitness_func occurring more than once among neighbours.
    If successful, take the position of one of these neighbours at random.
    Otherwise, check to see if the search position has better performance.
    If it does, take the search position.
    Otherwise, keep last position.
    """
    outcome, next_position, success = action_modal(env, node, time)
    if not success:
        outcome, next_position, success = action_step(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["modal_then_step"]] = action_modal_then_step


def action_40_step_60_best(env, node, time):
    """
    40% chance of stepping, 60% chance of best imitation
    """
    if random.rand() < 0.4:
        outcome, next_position, success = action_step(env, node, time)
    else:
        outcome, next_position, success = action_best(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["40_step_60_best"]] = action_40_step_60_best


def action_50_step_50_best(env, node, time):
    """
    50% chance of stepping, 50% chance of best imitation
    """
    if random.rand() < 0.5:
        outcome, next_position, success = action_step(env, node, time)
    else:
        outcome, next_position, success = action_best(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["50_step_50_best"]] = action_50_step_50_best


def action_60_step_40_best(env, node, time):
    """
    60% chance of stepping, 40% chance of best imitation
    """
    if random.rand() < 0.6:
        outcome, next_position, success = action_step(env, node, time)
    else:
        outcome, next_position, success = action_best(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["60_step_40_best"]] = action_60_step_40_best


def action_80_step_20_best(env, node, time):
    """
    80% chance of stepping, 20% chance of best imitation
    """
    if random.rand() < 0.8:
        outcome, next_position, success = action_step(env, node, time)
    else:
        outcome, next_position, success = action_best(env, node, time)
    return outcome, next_position, success

ACTION_FUNC[ACTION_NUM["80_step_20_best"]] = action_80_step_20_best
