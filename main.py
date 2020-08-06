"""Script for exploring collaborative problem solving strategies."""

###############################################################################
# Imports
###############################################################################
#
from collections import Counter
from enum import Enum
import numpy as np
import networkx as nx
#import matplotlib.pyplot as plt
from numpy import random

from nklandscapes import generate_fitness_func
import bitmanipulation as bitm


###############################################################################
# Types
###############################################################################
#
# Actions
#
class Action(Enum):
    """The possible actions that can be taken by an agent."""
    NO_CHANGE = 0
    RANDOM_STEP = 1
    COPY_BEST_NEIGHBOUR = 2
    COPY_A_MODAL_NEIGHBOUR = 3
    RANDOM_TELEPORT = 4

#
# Simulation Record
#
class SimulationRecord():
    '''
    Stores the positions,
    fitnesses and actions of each agent/node at each time.
    '''
    def __init__(self, num_nodes, deadline):
        self.num_nodes = num_nodes
        self.deadline = deadline

        self.positions = np.empty((num_nodes, deadline), dtype=int)
        self.fitnesses = np.empty((num_nodes, deadline))
        self.actions = np.empty((num_nodes, deadline), dtype=Action)

    def set_random_initial_position(self, num_bits, fitness_func):
        '''Sets a random position for each node at time 0.'''
        for node in range(self.num_nodes):
            self.positions[node, 0] = random.randint(2**num_bits)
            self.fitnesses[node, 0] = \
                    fitness_func[self.positions[node, 0]]
            self.actions[node, 0] = Action.RANDOM_TELEPORT


###############################################################################
# Innovate and Imitate Functions
###############################################################################
#
def find_best_neighbour(time, sim_record, fitness_func, neighbours):
    """Finds the position of the neighbour with the highest fitness."""
    neighbours_fitness = [
        fitness_func[sim_record.positions[neighbour, time]]
        for neighbour in neighbours
    ]
    return sim_record.positions[np.argmax(neighbours_fitness), time]


def random_step_search(time, node, sim_record, num_bits):
    """Flips a random bit of the previous position."""
    last_position = sim_record.positions[node, time]
    return bitm.flip_random_bit(num_bits, last_position)


def random_teleport_search(num_bits):
    """Returns a random integer from 0 to N-1."""
    return random.randint(num_bits)


###############################################################################
# Strategy Classes
###############################################################################
#
class QLearningAgent():
    """Agent which employs Q learning to make decisions."""

    def __init__(
            self,
            deadline,
            learning_rate=0.6,
            discount_factor=0.1,

            epsilon=1,
            epsilon_decay=0.10,

            quantisation_levels=100,
            use_best_neighbour=False,

            possible_actions=(
                Action.RANDOM_STEP,
                Action.COPY_BEST_NEIGHBOUR,
                Action.RANDOM_TELEPORT,
            ),
    ):
        self.deadline = deadline
        self.learning_rate = learning_rate
        self.discount = discount_factor

        self.epsilon = epsilon
        self.epsilon_decay = epsilon_decay

        self.possible_actions = possible_actions
        self.action2idx = {}
        for action_idx, possible_action in enumerate(possible_actions):
            self.action2idx[possible_action] = action_idx

        self.quantisation_levels = quantisation_levels
        self.use_best_neighbour = use_best_neighbour

        state_dimensions = [deadline, quantisation_levels+1]
        if use_best_neighbour:
            state_dimensions += [quantisation_levels+1]
        self.q_table = random.uniform(
            size=(list(state_dimensions) + [len(possible_actions)]),
        )


    def __update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_action = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_action]

        action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][action_idx]

        self.q_table[state][action_idx] += self.learning_rate * td_delta


    def __find_state(
            self,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """Find a node's state at the given time."""
        current_fitness = fitness_func[sim_record.positions[node, time]]
        state = [
            self.deadline - time -1,
            int(round(current_fitness * self.quantisation_levels)),
        ]
        if self.use_best_neighbour:
            best_neighbour_fitness = fitness_func[
                find_best_neighbour(time, sim_record, fitness_func, neighbours)
            ]
            state += [
                int(round(best_neighbour_fitness * self.quantisation_levels))
            ]
        return tuple(state)


    def __choose_action(self, current_state):
        """
        Chooses either the best action according the Q table
        or a random action
        depending on the current epsilon value.
        """
        if random.rand() <= self.epsilon:
            action = random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                np.argmax(self.q_table[current_state])
            ]
        self.epsilon -= self.epsilon_decay * self.epsilon
        return action


    def decide_action_and_move(
            self,
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
    ):
        """Function run at each time step of an episode."""
        # find current state
        current_state = self.__find_state(
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )

        if time > 0:
            # learn from the last decision
            last_state = self.__find_state(
                time-1,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
            self.__update_q_table(
                last_state,
                sim_record.actions[node, time-1],
                current_state,
                fitness_func[sim_record.positions[node, time]],
            )

        # decide on next action
        action = self.__choose_action(current_state)

        # carry out action
        if action == Action.RANDOM_STEP:
            proposed_position = \
                    random_step_search(time, node, sim_record, num_bits)

        elif action == Action.RANDOM_TELEPORT:
            proposed_position = random_teleport_search(num_bits)

        elif action == Action.COPY_BEST_NEIGHBOUR:
            proposed_position = find_best_neighbour(time, sim_record,
                                                    fitness_func, neighbours)

        current_position = sim_record.positions[node, time]
        if fitness_func[proposed_position] > fitness_func[current_position]:
            next_position = proposed_position
        else:
            next_position = current_position

        return next_position, action



###############################################################################
# Strategy Functions
###############################################################################
#
def strategy_best_then_search(
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
    best_neighbour_position = \
            find_best_neighbour(time, sim_record, fitness_func, neighbours)
    best_neighbour_fitness = fitness_func[best_neighbour_position]

    # search for a possible next position
    search_position = random_step_search(time, node, sim_record, num_bits)

    current_node_fitness = fitness_func[sim_record.positions[node, time]]
    if best_neighbour_fitness > current_node_fitness:
        new_position = best_neighbour_position
        action = Action.COPY_BEST_NEIGHBOUR
    elif fitness_func[search_position] > current_node_fitness:
        new_position = search_position
        action = Action.RANDOM_STEP
    else:
        new_position = sim_record.positions[node, time]
        action = Action.NO_CHANGE
    return new_position, action


def strategy_search_then_best(
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
    # search for a possible next position
    search_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if fitness_func[search_position] > current_node_fitness:
        new_position = search_position
        action = Action.RANDOM_STEP
    else:
        best_neighbour_position = \
                find_best_neighbour(time, sim_record, fitness_func, neighbours)
        best_neighbour_fitness = fitness_func[best_neighbour_position]

        if best_neighbour_fitness > current_node_fitness:
            new_position = best_neighbour_position
            action = Action.COPY_BEST_NEIGHBOUR
        else:
            new_position = sim_record.positions[node, time]
            action = Action.NO_CHANGE
    return new_position, action


def strategy_modal_then_search(
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
    # finds a neighbour with modal fitness_func
    neighbours_fitness = np.empty(len(neighbours))
    for neighbour_idx, neighbour in enumerate(neighbours):
        neighbours_fitness[neighbour_idx] = \
                fitness_func[sim_record.positions[neighbour, time]]

    neighbours_fitness_freq = Counter(neighbours_fitness)

    min_freq = 1
    modal_fitness = 0
    for key, value in neighbours_fitness_freq.items():
        if value > min_freq:
            min_freq = value
            modal_fitness = key

    # search for a possible next position
    search_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if modal_fitness > current_node_fitness:
        modal_neighbour_idxs = np.where(neighbours_fitness == modal_fitness)[0]
        rand_idx = random.randint(len(modal_neighbour_idxs))
        neighbour = neighbours[modal_neighbour_idxs[rand_idx]]

        new_position = sim_record.positions[neighbour, time]
        action = Action.COPY_A_MODAL_NEIGHBOUR

    elif fitness_func[search_position] > current_node_fitness:
        new_position = search_position
        action = Action.RANDOM_STEP
    else:
        new_position = sim_record.positions[node, time]
        action = Action.NO_CHANGE
    return new_position, action


###############################################################################
# Main Functions
###############################################################################
#
def run_episode(
        graph,
        num_bits,
        deadline,
        fitness_func,
        neighbour_sample_size=None,
        strategy=strategy_best_then_search,
):
    """A single episode of the simulation."""

    num_nodes = len(graph)

    sim_record = SimulationRecord(num_nodes, deadline)
    sim_record.set_random_initial_position(num_bits, fitness_func)


    for time in range(deadline-1):
        for node in range(num_nodes):

            # Find the node's neighbours
            neighbours = list(graph.neighbors(node))
            if neighbour_sample_size is not None:
                if len(neighbours) > neighbour_sample_size:
                    neighbours = random.choice(
                        neighbours,
                        size=neighbour_sample_size,
                        replace=False,
                    )

            # Find the next position and action used to determine it
            next_position, action = strategy(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )

            # Update step tracking variables
            sim_record.positions[node, time +1] = next_position
            sim_record.fitnesses[node, time +1] = fitness_func[next_position]
            sim_record.actions[node, time +1] = action

    return sim_record


###############################################################################
# Main
###############################################################################
#
if __name__ == '__main__':
    np.random.seed(42)
    N, K = 7, 6

    NUM_NODES = 60
    DEGREE = 4

    DEADLINE = 100
    ITERATIONS = 2

    SAMPLE = None

    # Fully connected graph
    print('generating graph')
    #graph = nx.complete_graph(NUM_NODES)
    graph = nx.random_regular_graph(DEGREE, NUM_NODES)

    #nx.draw_circular(graph)
    #plt.show()


    # make and train agent
    smart_agent = QLearningAgent(
        DEADLINE,
        epsilon_decay=1e-5,
        quantisation_levels=50,
    )
    while smart_agent.epsilon > 0.60:
        fitness_func = generate_fitness_func(N, K)

        run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            strategy=smart_agent.decide_action_and_move,
        )
        print(f'epsilon = {smart_agent.epsilon}')


    out_q = []
    out_bts = []
    out_stb = []
    out_mts = []

    for iteration in range(ITERATIONS):
        fitness_func = generate_fitness_func(N, K)

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            strategy=smart_agent.decide_action_and_move,
        ).fitnesses
        out_q.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=strategy_best_then_search,
        ).fitnesses
        out_bts.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=strategy_search_then_best,
        ).fitnesses
        out_stb.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=strategy_modal_then_search,
        ).fitnesses
        out_mts.append(np.mean(fitnesses[:, DEADLINE-1]))

        print(f'iteration: {iteration}')
        print(f'epsilon: {smart_agent.epsilon}')
        print()

        print(f'Q Learning Score: {out_q[iteration]}')
        print(f'Best Then Search Score: {out_bts[iteration]}')
        print(f'Search Then Best Score: {out_stb[iteration]}')
        print(f'Modal Then Search Score: {out_mts[iteration]}')
        print()

        print(f'Q Learning Average Score: {np.mean(out_q)}')
        print(f'Best Then Search Average Score: {np.mean(out_bts)}')
        print(f'Search Then Best Average Score: {np.mean(out_stb)}')
        print(f'Modal Then Search Average Score: {np.mean(out_mts)}')
        print()

        print(f'Q Learning Score Variance: {np.var(out_q)}')
        print(f'Best Then Search Score Variance: {np.var(out_bts)}')
        print(f'Search Then Best Score Variance: {np.var(out_stb)}')
        print(f'Modal Then Search Score Variance: {np.var(out_mts)}')
        print()
