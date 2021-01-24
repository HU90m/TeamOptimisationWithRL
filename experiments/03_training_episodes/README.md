# Experiment 03: Training Episodes
Explores how the number of training episodes affects the agent's ability.

## General Settings
- q learning agent
- deadline of 30
- 20 quantisation levels
- regular graph
    * 30 nodes
    * 5 degrees per node
- nk landscape
    * n = 14
    * k = 3
- tested over 1,000 episodes

## Agents
| Name   | number of episodes | learning rate | epsilon decay |
|:------:|:------------------:|:-------------:|:-------------:|
|  10000 |              10000 |          0.05 |       6.91e-4 |
|  20000 |              20000 |         0.025 |       3.45e-4 |
|  40000 |              40000 |        0.0125 |       1.74e-4 |
|  80000 |              80000 |       0.00625 |       8.63e-5 |
| 160000 |             160000 |      0.003125 |       4.31e-5 |

## Notes
Scaled learning rate and epsilond decay over the number of episodes.
