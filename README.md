# DQN Algorithm for Lunar Lander Problem


This repository contains an implementation of the Deep Q-Network (DQN) algorithm to solve the Lunar Lander problem. The Lunar Lander problem is a classic reinforcement learning task where the goal is to safely land a spacecraft on the surface of the moon, using minimal fuel. More details about the environment: [OpenAI gym](https://gymnasium.farama.org/environments/box2d/lunar_lander/#lunar-lander).


## Usage
Clone the repository  
Create a conda env by running:
```
conda env create -f env.yml
```
Train DQN agent:
```
python lunar_lander.py
```

## Files
- DQN_agent.py: Contains the implementation of the DQN agent.
- lunar_lander.py: Script for training the DQN agent.
- DQN_check_solution.py: Script for evaluating the trained DQN agent.
- env.yml: Lists the required Python packages for this project.


NOTE: This code was written as a part of assignment for course (EL2805) Reinforcement Learning at KTH.  
Copyright [2020] [KTH Royal Institute of Technology] Licensed under the Educational Community License, Version 2.0 (the "License")  
Code structure by [Alessio Russo - alessior@kth.se]
