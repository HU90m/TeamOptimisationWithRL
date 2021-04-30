# Experiment 05: Agent Type
Compares policy gradient agents to Q-learning agents.


## general settings
- deadline of 30
- regular graph
    * 30 nodes
    * 5 degrees per node
- nk landscape
    * n = 14
    * k = 3
- trained over 10,000 episodes
- tested over 1,000 episodes


## Agents

### qlearn
- type : Q Learning
- learning rate : 0.05
- quantisation levels : 20
- exploration type : epsilon greedy
- epsilon decay : 6.91e-4


### policygrad
- type : Policy Gradient
- learning rate : 0.001

- exploration type : epsilon greedy
- epsilon decay : 6.91e-4

- buffer capacity : 6,000
- sample size : 600
- target update frequency : 5
- delay learning : 1

### policygrad\_boltzmann
- type : Policy Gradient
- learning rate : 0.001

- exploration type : boltzmann

- buffer capacity : 6,000
- sample size : 600
- target update frequency : 5
- delay learning : 1

### policygrad\_simple
- type : Policy Gradient
- learning rate : 0.008

- exploration type : epsilon greedy
- epsilon decay : 6.91e-4

- buffer capacity : 6,000
- sample size : 600
- target update frequency : 10
- delay learning : 1
