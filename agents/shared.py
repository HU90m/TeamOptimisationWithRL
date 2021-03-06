"""Contains a shared function for loading any agent from it's config."""
import json
from os import path
from agents.qlearning import QLearningAgent
from agents.policygrad import PolicyGradientAgent
from agents.programmed import ProgrammedAgent
from agents.randomtable import RandomTableAgent

def from_config(config_location, get_action_num_func):
    """Loads any agent from a config file."""

    config_dir, _ = path.split(config_location)

    with open(config_location, 'rb') as file_handle:
        config = json.load(file_handle)

    agent_type = config["type"]
    if agent_type == "QLearningAgent":
        agent = QLearningAgent(config, config_dir, get_action_num_func)
    elif agent_type == "PolicyGradientAgent":
        agent = PolicyGradientAgent(config, config_dir, get_action_num_func)
    elif agent_type == "ProgrammedAgent":
        agent = ProgrammedAgent(config, config_dir, get_action_num_func)
    elif agent_type == "RandomTableAgent":
        agent = RandomTableAgent(config, config_dir, get_action_num_func)
    else:
        raise ValueError(f"{agent_type} is not a supported agent.")

    return agent, config
