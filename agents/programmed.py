"""A module containing the ProgrammedAgent.
An agent which can follow a program setting out actions to be taken at
certain timesteps.
"""

import numpy as np

###############################################################################
# Agent
###############################################################################
#
class ProgrammedAgent:
    """Agent which follows a given program."""
    def __init__(self, config, _config_file, get_action_num_func):
        # settings
        self.deadline = config["deadline"]

        self.program = np.zeros(self.deadline) \
                + get_action_num_func(config["default action"])

        for program_item in config["program"]:
            for idx in program_item["time steps"]:
                if idx < 0:
                    self.program[self.deadline + idx] = \
                            get_action_num_func(program_item["action"])
                else:
                    self.program[idx] = \
                            get_action_num_func(program_item["action"])


        # for the agent interface
        self.episode_end = lambda : None

    def best_action(self, _node, time, _environment):
        """Returns the action number from the program."""
        return self.program[time]

    def explore_action(self, node, time, environment):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError(
                "A ProgrammedAgent is kinda dumb, "
                "so doesn't explore and learn."
        )

    def save(self, suffix=None):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError(
                "A ProgrammedAgent doesn't learn anything "
                "so has nothing to save and load."
        )

    def load(self, suffix=None):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError(
                "A ProgrammedAgent doesn't learn anything "
                "so has nothing to save and load."
        )

    def plot(self, figure):
        """Method included so agent fails nicely when not used properly."""
        raise NotImplementedError("This agent has nothing to plot.")
