# Using Reinforcement Learning to Learn Novel Strategies for Collective Decision Making

## What is this?
This repository holds the code to used to explore using Reinforcement Learning
to Learn Novel Strategies for Collective Decision Making.
This README was written for a future me so isn't a complete explanation.
If you are not me in the future,
please reach out and ask me questions.


## Structure
The three main scripts are `train.py`, `policy.py` and `compare.py`.

| Name       | Description                |
| :--------- | :------------------------- |
|`train.py`  | Trains an agent            |
|`policy.py` | Visualises a learnt policy |
|`compare.py`| Compares collective decision making strategies. Used to test agents |


There is also the `lint` shell script for linting code.

There are two modules `environment` and `agent`, which hold the environment code and the code of the agents respectively.

The there is also the `configs` which holds configurations of experiments.


## Example
All the scripts take a configuration file as their input.
Let's run through an example, using the provided example configurations,
to see how this all works.

Before we start, you may want to follow the instructions in the 'Using the rustylandscapes' which will significantly speed up experiments.

There are two types of configurations comparison configurations and agent configurations.
Comparison configurations are used by `compare.py` and agent configurations are used by `train.py` and `policy.py`.
For an example of a configuration look at `configs/example/heuristics.json`, which can be run with the following command.
```bash
python compare.py configs/example/heuristics.json
```

I put agent configurations in their own directory because when in training the configurations directory will be populated with saved instances of the agent.
Before training the agent, have a look at it's configuration in `configs/example/qlearn_agent/qlearn_agent.json` for an example of an agent configuration.
To train an agent with this configuration
```bash
python train.py configs/example/qlearn_agent/qlearn_agent.json
```

Once trained a visualisation of an agent's policy can be viewed.
```bash
python policy.py configs/example/qlearn_agent/qlearn_agent.json
```
If you don't want to view the agent's final policy, but a policy at a particular episode append the episode as an argument,
n.b. this episode has to be a multiple of the 'save interval' in the agent's configuration.
```bash
python policy.py configs/example/qlearn_agent/qlearn_agent.json 2500
```

Then to compare this trained agent's strategy with some heuristic strategies
```bash
python compare.py configs/example/compare_qlearn.json
```


## Using the rustylandscapes

The rustylandscapes module is a reimplementation of the NK Landscape generator
rewritten in rust for improved speed. To build it's wheel file
(the python package format) run install `maturin`.
This can be done with the following command.
```bash
pip install manturin
```

Then to build go into the rustylandscapes directory and run manturin.
```bash
cd environment/rustylandscapes
maturin build
```

The resulting wheel file will be shown in the last line of matuin's output.
To install this run the following.
```bash
pip install --user path/to/wheel.whl
```
