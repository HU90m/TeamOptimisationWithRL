"""Script for exploring collaborative problem solving strategies."""

###############################################################################
# Imports
###############################################################################
#
import multiprocessing as mp
from multiprocessing import sharedctypes
from collections import Counter
import networkx as nx
import numpy as np
from numpy import random
import bitmanipulation as bitm
import nklandscapes as nkl


#import matplotlib.pyplot as plt

###############################################################################
# Constants
###############################################################################
#
# Actions
#
ACTION_STR = [
        'step',
        'teleport',
        'best',
        'modal',
        'best_then_step',
        'modal_then_step',
]
ACTION_NUM = {
        'step' : 0,
        'teleport' : 1,
        'best' : 2,
        'modal' : 3,
        'best_then_step' : 4,
        'step_then_best' : 4,
        'modal_then_step' : 6,
}

#
# Outcomes
#
OUTCOME_STR = [
        'no change',
        'stepped',
        'teleported',
        'copied best',
        'copied a modal',
]
OUTCOME_NUM = {
        'no change' : 0,
        'stepped' : 1,
        'teleported' : 2,
        'copied best' : 3,
        'copied a modal' : 4,
}

###############################################################################
# Types
###############################################################################
#
# Simulation Record
#
class SimulationRecord():
    '''
    Stores the positions,
    fitnesses and actions of each agent/node at each time.
    '''
    def __init__(self, num_nodes, deadline, num_processes=1):
        self.num_nodes = num_nodes
        self.deadline = deadline

        if num_processes < 2:
            self.positions = np.empty((num_nodes, deadline), dtype=int)
            self.fitnesses = np.empty((num_nodes, deadline))
            self.actions = np.empty((num_nodes, deadline), dtype=int)
            self.outcomes = np.empty((num_nodes, deadline), dtype=int)

        else:
            # make shared memory
            positions_shared = sharedctypes.RawArray('I', num_nodes*deadline)
            fitnesses_shared = sharedctypes.RawArray('d', num_nodes*deadline)
            actions_shared = sharedctypes.RawArray('I', num_nodes*deadline)
            outcomes_shared = sharedctypes.RawArray('I', num_nodes*deadline)

            # abstract shared memory to numpy array
            self.positions = np.frombuffer(
                positions_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

            self.fitnesses = np.frombuffer(
                fitnesses_shared,
                dtype='d',
            ).reshape((num_nodes, deadline))

            self.actions = np.frombuffer(
                actions_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

            self.outcomes = np.frombuffer(
                outcomes_shared,
                dtype='I',
            ).reshape((num_nodes, deadline))

    def set_random_initial_position(self, num_bits, fitness_func):
        '''Sets a random position for each node at time 0.'''
        for node in range(self.num_nodes):
            self.positions[node, 0] = random.randint(2**num_bits)
            self.fitnesses[node, 0] = \
                    fitness_func[self.positions[node, 0]]
            self.actions[node, 0] = ACTION_NUM['teleport']
            self.outcomes[node, 0] = OUTCOME_NUM['teleported']

    def save(self, file_name):
        """Saves the simulation record as a npz archive."""
        with open(file_name, 'wb') as file_handle:
            np.savez(
                file_handle,
                num_nodes=self.num_nodes,
                deadline=self.deadline,
                positions=self.positions,
                fitnesses=self.fitnesses,
                actions=self.actions,
            )

    def load(self, file_name):
        """Loads a saved simulation record."""
        with open(file_name, 'rb') as file_handle:
            file_content = np.load(file_handle)

        self.num_nodes = int(file_content['num_nodes'])
        self.deadline = int(file_content['deadline'])
        self.positions = file_content['positions']
        self.fitnesses = file_content['fitnesses']
        self.actions = file_content['actions']


###############################################################################
# Action Function Components
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
# Action Functions
###############################################################################
#
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
    best_neighbour_position = \
            find_best_neighbour(time, sim_record, fitness_func, neighbours)
    best_neighbour_fitness = fitness_func[best_neighbour_position]

    # search for a possible next position
    search_position = random_step_search(time, node, sim_record, num_bits)

    current_node_fitness = fitness_func[sim_record.positions[node, time]]
    if best_neighbour_fitness > current_node_fitness:
        next_position = best_neighbour_position
        outcome = OUTCOME_NUM['copied best']
    elif fitness_func[search_position] > current_node_fitness:
        next_position = search_position
        outcome = OUTCOME_NUM['stepped']
    else:
        next_position = sim_record.positions[node, time]
        outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['best_then_step']
    sim_record.outcomes[node, time +1] = outcome


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
    # search for a possible next position
    step_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if fitness_func[step_position] > current_node_fitness:
        next_position = step_position
        outcome = OUTCOME_NUM['stepped']

    else:
        best_neighbour_position = \
                find_best_neighbour(time, sim_record, fitness_func, neighbours)
        best_neighbour_fitness = fitness_func[best_neighbour_position]

        if best_neighbour_fitness > current_node_fitness:
            next_position = best_neighbour_position
            outcome = OUTCOME_NUM['copied best']
        else:
            next_position = sim_record.positions[node, time]
            outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['step_then_best']
    sim_record.outcomes[node, time +1] = outcome


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
    step_position = random_step_search(time, node, sim_record, num_bits)

    # makes decision
    current_node_fitness = fitness_func[sim_record.positions[node, time]]

    if modal_fitness > current_node_fitness:
        modal_neighbour_idxs = np.where(neighbours_fitness == modal_fitness)[0]
        rand_idx = random.randint(len(modal_neighbour_idxs))
        neighbour = neighbours[modal_neighbour_idxs[rand_idx]]

        next_position = sim_record.positions[neighbour, time]
        outcome = OUTCOME_NUM['copied a modal']

    elif fitness_func[step_position] > current_node_fitness:
        next_position = step_position
        outcome = OUTCOME_NUM['stepped']

    else:
        next_position = sim_record.positions[node, time]
        outcome = OUTCOME_NUM['no change']

    # update simulation record
    sim_record.positions[node, time +1] = next_position
    sim_record.fitnesses[node, time +1] = fitness_func[next_position]
    sim_record.actions[node, time +1] = ACTION_NUM['modal_then_step']
    sim_record.outcomes[node, time +1] = outcome


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
                ACTION_NUM['best_then_step'],
                ACTION_NUM['step_then_best'],
                ACTION_NUM['modal_then_step'],
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

        self.state_dimensions = [deadline, quantisation_levels+1]

        self.use_best_neighbour = use_best_neighbour
        if use_best_neighbour:
            self.state_dimensions += [quantisation_levels+1]

        self.q_table = random.uniform(
            size=(list(self.state_dimensions) + [len(possible_actions)]),
        )

    def _update_q_table(self, state, action, next_state, reward):
        """Updates the Q table after an action has been taken."""
        next_best_action = np.argmax(self.q_table[next_state])
        td_target = reward \
            + self.discount * self.q_table[next_state][next_best_action]

        action_idx = self.action2idx[action]
        td_delta = td_target - self.q_table[state][action_idx]

        self.q_table[state][action_idx] += self.learning_rate * td_delta

    def _find_state(
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

    def _choose_action(self, current_state):
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

    def save_q_table(self, file_name):
        """Save the q_table with the given file name."""
        with open(file_name, 'wb') as file_handle:
            np.save(file_handle, self.q_table)

    def load_q_table(self, file_name):
        """Load a q_table with the given file name."""
        with open(file_name, 'rb') as file_handle:
            self.q_table = np.load(file_handle)

    def learn_and_perform_action(
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
        current_state = self._find_state(
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )

        # if not first run learn from the last decision
        if time > 1:
            last_state = self._find_state(
                time-1,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
            self._update_q_table(
                last_state,
                sim_record.actions[node, time-1],
                current_state,
                fitness_func[sim_record.positions[node, time]],
            )

        # decide on next action
        action = self._choose_action(current_state)

        # carry out action
        if action == ACTION_NUM['best_then_step']:
            action_best_then_step(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
        elif action == ACTION_NUM['step_then_best']:
            action_step_then_best(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )
        elif action == ACTION_NUM['modal_then_step']:
            action_modal_then_step(
                num_bits,
                time,
                node,
                neighbours,
                fitness_func,
                sim_record,
            )


###############################################################################
# Main Functions
###############################################################################
#
def run_time_step(
        num_bits,
        time,
        graph,
        neighbour_sample_size,
        fitness_func,
        strategy,

        start_node,
        end_node,

        sim_record,
):
    """Runs the nodes from start_node to end_node, for a time step."""
    for node in range(start_node, end_node):
        # Find the node's neighbours
        neighbours = list(graph.neighbors(node))
        if neighbour_sample_size is not None:
            if len(neighbours) > neighbour_sample_size:
                neighbours = random.choice(
                    neighbours,
                    size=neighbour_sample_size,
                    replace=False,
                )

        # carry out the strategy
        strategy(
            num_bits,
            time,
            node,
            neighbours,
            fitness_func,
            sim_record,
        )


def run_episode(
        graph,
        num_bits,
        deadline,
        fitness_func,
        neighbour_sample_size=None,
        num_processes=1,
        strategy=action_best_then_step,
):
    """A single episode of the simulation."""

    num_nodes = len(graph)

    sim_record = SimulationRecord(num_nodes, deadline, num_processes=4)
    sim_record.set_random_initial_position(num_bits, fitness_func)

    for time in range(deadline-1):
        if num_processes < 2:
            run_time_step(
                num_bits,
                time,
                graph,
                neighbour_sample_size,
                fitness_func,
                strategy,
                0,
                num_nodes,
                sim_record,
            )
        else:
            processes = []
            for process_idx in range(num_processes):
                start_node = int(num_nodes * process_idx/num_processes)
                end_node = int(num_nodes * (process_idx+1)/num_processes)
                process = mp.Process(target=run_time_step, args=(
                    num_bits,
                    time,
                    graph,
                    neighbour_sample_size,
                    fitness_func,
                    strategy,
                    start_node,
                    end_node,
                    sim_record,
                ))
                process.start()
                processes.append(process)

            for process in processes:
                process.join()

    return sim_record


###############################################################################
# Main
###############################################################################
#
if __name__ == '__main__':
    np.random.seed(42)
    N, K = 13, 6

    NUM_NODES = 60
    DEGREE = 4

    DEADLINE = 50
    ITERATIONS = 30

    SAMPLE = 5

    # Fully connected graph
    print('generating graph')
    graph = nx.complete_graph(NUM_NODES)
    #graph = nx.random_regular_graph(DEGREE, NUM_NODES)

    #nx.draw_circular(graph)
    #plt.show()


    # make and train agent
    smart_agent = QLearningAgent(
        DEADLINE,
        epsilon_decay=1e-6,
        quantisation_levels=50,
    )

    print(f'epsilon = {smart_agent.epsilon}', end='')
    while smart_agent.epsilon > 0.05:
        fitness_func = nkl.generate_fitness_func(N, K, num_processes=4)

        run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            strategy=smart_agent.learn_and_perform_action,
        )
        print(f'\repsilon = {smart_agent.epsilon}', end='')

    print()


    out_q = []
    out_bts = []
    out_stb = []
    out_mts = []

    for iteration in range(ITERATIONS):
        fitness_func = nkl.generate_fitness_func(N, K, num_processes=4)

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            strategy=smart_agent.learn_and_perform_action,
        ).fitnesses
        out_q.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=action_best_then_step,
        ).fitnesses
        out_bts.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=action_step_then_best,
        ).fitnesses
        out_stb.append(np.mean(fitnesses[:, DEADLINE-1]))

        fitnesses = run_episode(
            graph,
            N,
            DEADLINE,
            fitness_func,
            neighbour_sample_size=SAMPLE,
            strategy=action_modal_then_step,
        ).fitnesses
        out_mts.append(np.mean(fitnesses[:, DEADLINE-1]))

        print(f'iteration: {iteration}')
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
