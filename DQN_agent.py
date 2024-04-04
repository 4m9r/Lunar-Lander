# Copyright [2020] [KTH Royal Institute of Technology] Licensed under the
# Educational Community License, Version 2.0 (the "License"); you may
# not use this file except in compliance with the License. You may
# obtain a copy of the License at http://www.osedu.org/licenses/ECL-2.0
# Unless required by applicable law or agreed to in writing,
# software distributed under the License is distributed on an "AS IS"
# BASIS, WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express
# or implied. See the License for the specific language governing
# permissions and limitations under the License.
#
# Course: EL2805 - Reinforcement Learning - Lab 2 Problem 1
# Code author: [Alessio Russo - alessior@kth.se]
# Last update: 20th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim

class FFNetwork(nn.Module):
    def __init__(self, input, output):
        super().__init__()
        self.layer1 = nn.Linear(input, 64)
        self.activation1= nn.ReLU()

        self.layer2 = nn.Linear(64, 64)
        self.activation2= nn.ReLU()

        self.output_layer = nn.Linear(64, output)

        

    def forward(self, x):
        x = self.layer1(x)
        x = self.activation1(x)

        x = self.layer2(x)
        x = self.activation2(x)

        out = self.output_layer(x)
        return out


# class Agent(object):
#     ''' Base agent class, used as a parent class

#         Args:
#             n_actions (int): number of actions

#         Attributes:
#             n_actions (int): where we store the number of actions
#             last_action (int): last action taken by the agent
#     '''
#     def __init__(self, n_actions: int, state_dim):
#         self.n_actions = n_actions
#         self.last_action = None

#         self.network = FFNetwork(state_dim, n_actions)
#         self.target_network = FFNetwork(state_dim, n_actions)

#         self.optimizer = optim.Adam(self.network.parameters(), lr = 5e-4)

#     def forward(self, state, eps):
#         ''' Performs a forward computation '''
#         actions = self.network(state)

#         # greedy epsilon
#         if np.random.uniform(0, 1) <= eps:
#             action = np.random.choice(self.n_actions)
#         else:
#             _, action = actions.max(1)
#             action = action.item()
#         return action

    def backward(self, states, actions, targets, N):
        ''' Performs a backward pass on the network '''
        self.optimizer.zero_grad()
        
        q_values = self.network(states)
        q_a = torch.zeros(N, requires_grad=False,dtype=torch.float32)
        
        for i in range(N):
            q_a[i] = q_values[i,actions[i]]

        loss = nn.functional.mse_loss(q_a, targets)

        loss.backward()

        nn.utils.clip_grad_norm_(self.network.parameters(), max_norm=1)

        self.optimizer.step()


# class RandomAgent(Agent):
#     ''' Agent taking actions uniformly at random, child of the class Agent'''
#     def __init__(self, n_actions: int, state_dim):
#         super(RandomAgent, self).__init__(n_actions, state_dim)

#     def forward(self, state: np.ndarray) -> int:
#         ''' Compute an action uniformly at random across n_actions possible
#             choices

#             Returns:
#                 action (int): the random action
#         '''
        
#         self.last_action = np.random.randint(0, self.n_actions)
#         return self.last_action
