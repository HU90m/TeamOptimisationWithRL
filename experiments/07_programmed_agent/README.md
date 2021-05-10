# Experiment 07: Testing new programmed Heuristics

## How the programmed agent is configured

The program is an array of items which contain an action to be taken
and the time steps inwhich to do it.
Negative time steps are taken as the number of timestep from the deadline.
If there are no actions specified for a timestep,
the agent will fall back on a default action.

## Agents

### copy\_last

This agent steps untill the last eight steps then performs best imitation.

## Experiment Settings

- deadline of 30
- regular graph
    * 30 nodes
    * 5 degrees per node
- nk landscape
    * n = 14
    * k = 4
- tested over 1,000 episodes
