"""A module containing the agents which can learn strategies for workers."""

from os import path
import numpy as np
import torch
import torch.nn.functional as F
import matplotlib.pyplot as plt
from matplotlib import colors, ticker


###############################################################################
# Agent
###############################################################################
#
class PolicyGradientAgent:
    """Agent which employs Policy Gradient based learning."""
    def __init__(self, config, config_dir, get_action_num_func):
        # settings
        self.name = config["name"]
        self.config_dir = config_dir
        self.deadline = config["deadline"]

        # set up actions
        self._possible_actions_str = config["possible actions"]
        self.possible_actions = []
        for action_name in self._possible_actions_str:
            # convert the actions names to numbers, with the given function
            self.possible_actions.append(get_action_num_func(action_name))

        # this maps the external action number to internal action indicies
        self.action_num2idx = {}
        for action_idx, action_num in enumerate(self.possible_actions):
            self.action_num2idx[action_num] = action_idx

        # network choice
        if config["network type"] == "linear":
            network_type = LinearNet
        elif config["network type"] == "simple":
            network_type = SimpleNet
        else:
            raise ValueError("Network type not valid.")

        # state space choice
        if config["state space"] == "time":
            self._find_state = self._find_state_time
            self.num_states = 1
        elif config["state space"] == "time and score":
            self._find_state = self._find_state_time_score
            self.num_states = 2
        else:
            ValueError("State space not valid.")

        # exploration type
        if config["exploration"]["type"] == "epsilon greedy":
            self.epsilon = config["exploration"]["epsilon start"]
            self.epsilon_decay = config["exploration"]["epsilon decay"]

            self.best_action = self._choose_greedy_action
            self.explore_action = self._choose_epsilon_greedy_action
            self.episode_end = self._episode_count_learn_decay

        elif config["exploration"]["type"] == "boltzmann":
            self.best_action = self._choose_greedy_action
            self.explore_action = self._choose_boltzmann_action
            self.episode_end = self._episode_count_learn

        else:
            raise ValueError("Exploration type not valid.")

        # DQN setup
        self._qnet = DQN(
                num_states=self.num_states,
                num_actions=len(self.possible_actions),
                network_type=network_type,
                discount=config["discount factor"],
                learning_rate=config["learning rate"],
                buffer_capacity=config["buffer capacity"],
                max_sample_size=config["sample size"],
                target_update_frequency=config["target update frequency"],
        )
        self.learn = self._save_transition

        # number of episodes of filling the replay buffer without learning
        self.delay_learning = config["delay learning"]
        self.episode_count = 0

    def _choose_greedy_action(self, time, current_fitness_norm):
        """Returns the best action according the Q net."""
        current_state = self._find_state(time, current_fitness_norm)
        action_idx = self._qnet.choose_best_action(current_state)
        return self.possible_actions[action_idx]

    def _choose_epsilon_greedy_action(self, time, current_fitness_norm):
        """Returns either the best action according the Q net
        or a random action depending on the current epsilon value.
        """
        current_state = self._find_state(time, current_fitness_norm)
        # choose action
        if np.random.rand() <= self.epsilon:
            action = np.random.choice(self.possible_actions)
        else:
            action = self.possible_actions[
                self._qnet.choose_best_action(current_state)
            ]
        return action

    def _choose_boltzmann_action(self, time, current_fitness_norm):
        """Returns an action using the softmax
        as a probability distribution"""
        current_state = self._find_state(time, current_fitness_norm)
        action_probs = self._qnet.action_values_softmax(current_state)

        action_num = np.random.choice(
                self.possible_actions, 1, replace=False, p=action_probs,
        )
        return action_num

    def _episode_count_learn(self):
        """Count episodes and after delay start learning from each episode."""
        if self.episode_count > self.delay_learning:
            self._qnet.train_q_network()
        self.episode_count += 1

    def _episode_count_learn_decay(self):
        """Count episodes and after delay start learning from each episode.
        Also, decay epsilon after each episode."""
        if self.episode_count > self.delay_learning:
            self._qnet.train_q_network()
        self.episode_count += 1
        self.epsilon -= self.epsilon_decay * self.epsilon

    def _find_state_time(self, time, current_fitness):
        """Find a worker's state at the given time."""
        return (time,)

    def _find_state_time_score(self, time, current_fitness):
        """Find a worker's state at the given time."""
        return (time, current_fitness)

    def _save_transition(self, prior_time, prior_fitness_norm,
              chosen_action, post_time, post_fitness_norm, post_fitness):
        """Learns from a transition."""
        prior_state = self._find_state(prior_time, prior_fitness_norm)
        post_state = self._find_state(post_time, post_fitness_norm)

        # each node only receives a reward at the deadline
        reward = post_fitness if post_time == self.deadline else 0

        action_idx = self.action_num2idx[chosen_action]

        self._qnet.add_to_buffer(
                (prior_state, action_idx, reward, post_state)
        )

    def _find_file_name(self, suffix):
        """Finds the file name for a given suffix."""
        if suffix:
            return path.join(self.config_dir, f'{self.name}_{suffix}.pt')
        return path.join(self.config_dir, f'{self.name}.pt')

    def save(self, suffix=None):
        """Saves the agent's learnings.
        Saves the q_table and update_count in a file with the the given suffix.
        """
        file_name = self._find_file_name(suffix)
        self._qnet.save(file_name)

    def load(self, suffix=None):
        """Loads learnings.
        Loads the q_table and update_count from a file with the the given
        suffix.
        """
        file_name = self._find_file_name(suffix)
        self._qnet.load(file_name)

###############################################################################
# Other Classes
###############################################################################
#
class LinearNet(torch.nn.Module):
    """A Network consisting of one linear layer."""
    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()
        self.only_layer = torch.nn.Linear(input_dimensions, output_dimensions)

    def forward(self, state):
        """Returns the network's output, when x is the input."""
        return self.only_layer(state)

class SimpleNet(torch.nn.Module):
    """A Simple Network consisting of three layers."""

    def __init__(self, input_dimensions, output_dimensions):
        super().__init__()

        self.layer_1 = torch.nn.Linear(in_features=input_dimensions,
                                       out_features=100)
        self.layer_2 = torch.nn.Linear(in_features=100, out_features=100)
        self.output_layer = torch.nn.Linear(in_features=100,
                                            out_features=output_dimensions)

    def forward(self, state):
        """Returns the network's output, when x is the input.
        A ReLU activation function is used for both hidden layers,
        but the output layer has no activation function (just linear).
        """
        layer_1_output = torch.nn.functional.relu(self.layer_1(state))
        layer_2_output = torch.nn.functional.relu(self.layer_2(layer_1_output))
        output = self.output_layer(layer_2_output)
        return output

class DQN:
    """Determines how to train the above neural network."""

    def __init__(
            self,
            num_states=2,
            num_actions=2,
            network_type=LinearNet,
            discount=0.1,
            learning_rate=0.001,
            buffer_capacity=500,
            max_sample_size=100,
            target_update_frequency=10,
            priority_alpha=1,
            priority_e=0.001,
        ):
        # Create a Q-network, which predicts the q-value for a particular state.
        self.q_network = network_type(
                input_dimensions=num_states, output_dimensions=num_actions
        )
        # Define the optimiser which is used when updating the Q-network.
        # The learning rate determines
        # how big each gradient step is during backpropagation.
        self.optimiser = torch.optim.Adam(self.q_network.parameters(),
                                          lr=learning_rate)
        self.discount = discount
        # Create a target Q network
        self.target_net = network_type(
                input_dimensions=num_states, output_dimensions=num_actions
        )
        # set target network = Q network
        self.update_target()
        self.train_count = 0
        self.target_update_frequency = target_update_frequency

        # put target network into evaluation mode (as oppose to train mode)
        self.target_net.eval()

        # replay buffer
        self.priority_alpha = priority_alpha
        self.priority_e = priority_e
        self.max_sample_size = max_sample_size
        self.capacity = buffer_capacity
        self.buf = ReplayBuffer(buffer_capacity)

    def train_q_network(self):
        """Function that is called whenever we want to train the Q-network.
        Each call to this function takes in a transition tuple
        containing the data we use to update the Q-network.
        """
        # Set all the gradients stored in the optimiser to zero.
        self.optimiser.zero_grad()
        # Calculate the loss for this transition.
        loss = self._calculate_loss()
        # Compute the gradients based on this loss, i.e. the gradients
        # of the loss with respect to the Q-network parameters.
        loss.backward()
        # Take one gradient step to update the Q-network.
        self.optimiser.step()
        # update target after 'update_frequency' many training loops
        self.train_count += 0
        if self.train_count % self.target_update_frequency == 0:
            self.update_target()
        # Return the loss as a scalar
        return loss.item()

    def choose_best_action(self, state):
        """Choose the greedy action for a given state."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        best_action = torch.argmax(self.q_network.forward(state_tensor)[0])
        return int(best_action)

    def action_values_softmax(self, state):
        """Return the action values (logits) for each state."""
        state_tensor = torch.tensor(state, dtype=torch.float32).unsqueeze(0)
        action_values = self.q_network.forward(state_tensor).squeeze().detach()
        action_probs = torch.softmax(action_values, 0)
        return action_probs.numpy()

    def _calculate_loss(self):
        """Function to calculate the loss for a particular transition."""

        if self.max_sample_size < len(self.buf):
            sample_size = self.max_sample_size
        else:
            sample_size = len(self.buf)

        # update buffer probabilities
        self.buf.update_probs(self.priority_alpha)
        # sample from buffer
        transitions, sample_idxs = self.buf.prioritised_sample(sample_size)

        states, actions, rewards, next_states = zip(*transitions)

        state_tensor = torch.tensor(states, dtype=torch.float32)
        actions_tensor = torch.tensor(actions).unsqueeze(1)

        q_value_tensor = self.q_network.forward(state_tensor)
        # predicted state action values
        pred_state_action_values = q_value_tensor.gather(1, actions_tensor)

        reward_tensor = torch.tensor(rewards, dtype=torch.float32)
        next_state_tensor = torch.tensor(next_states, dtype=torch.float32)
        # max value in the next state
        # (according to the selected 'next state' network)
        next_state_values = \
                self.target_net.forward(next_state_tensor).max(1)[0]

        # expected state action values
        exp_state_action_values = (
                (next_state_values * self.discount) + reward_tensor
        ).unsqueeze(1)

        # convert to numpy arrays
        np_pred_state_action_values = pred_state_action_values.detach().numpy()
        np_exp_state_action_values = exp_state_action_values.detach().numpy()

        # find delta for each transition
        deltas = (
                np_pred_state_action_values - np_exp_state_action_values
        ).swapaxes(0, 1).squeeze()

        # update weights for sample
        new_weights = np.abs(deltas) + self.priority_e
        self.buf.update_weights(sample_idxs, new_weights)

        return F.mse_loss(pred_state_action_values, exp_state_action_values)

    def update_target(self):
        self.target_net.load_state_dict(self.q_network.state_dict())

    def add_to_buffer(self, transition):
        self.buf.append(transition)

    def save(self, file_location):
        torch.save(self.q_network.state_dict(), file_location)

    def load(self, file_location):
        self.q_network.load_state_dict(torch.load(file_location))
        self.update_target()

class ReplayBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.buffer = np.empty(capacity, dtype=object)
        self.weights = np.zeros(capacity)
        self.probs = np.zeros(capacity)

        self.full = False
        self.pointer = 0

        self.weights[0] = 1

    def __len__(self):
        if self.full:
            return self.capacity
        else:
            return self.pointer

    def append(self, transition):
        self.buffer[self.pointer] = transition
        self.weights[self.pointer] = np.max(self.weights)

        self.pointer += 1
        if self.pointer == self.capacity:
            self.pointer = 0
            self.full = True

    def random_sample(self, sample_size):
        sample_space = len(self)
        sample_idxs = np.random.choice(
                range(sample_space), sample_size, replace=False,
        )
        return(
                np.take(self.buffer, sample_idxs),
                sample_idxs,
        )

    def prioritised_sample(self, sample_size):
        sample_space = len(self)
        sample_idxs = np.random.choice(
                range(sample_space), sample_size, replace=False,
                p=self.probs[:sample_space],
        )
        return(
                np.take(self.buffer, sample_idxs),
                sample_idxs,
        )

    def update_weights(self, sample_idxs, sample_weights):
        self.weights[sample_idxs] = sample_weights

    def update_probs(self, alpha):
        weights_alpha = self.weights**alpha
        self.probs = weights_alpha/np.sum(weights_alpha)
