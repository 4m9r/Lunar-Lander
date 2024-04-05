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
# Last update: 6th October 2020, by alessior@kth.se
#

# Load packages
import numpy as np
import gym
import torch
import matplotlib.pyplot as plt
from tqdm import trange
# from DQN_agent import RandomAgent
from DQN_agent import FFNetwork
from collections import deque
import torch.nn as nn
import torch.optim as optim
import copy


class ExperienceReplayBuffer(object):
    def __init__(self, max_len=20000):
        self.buffer = deque(maxlen=max_len)

    def append(self, experience):
        self.buffer.append(experience)

    def __len__(self):
        return len(self.buffer)

    def sample_batch(self, n):
        if n > len(self.buffer):
            print("Error! too many element to retrieve")

        random_indices = np.random.choice(
            a=len(self.buffer), size=n, replace=False)

        batch = [self.buffer[i] for i in random_indices]

        # there are list of n elements

        # states, actions, rewards, next_states, dones = zip(*batch)
        return zip(*batch)


def plot_env(state):
    plt.imshow(state)
    plt.pause(.01)
    plt.clf()


def running_average(x, N):
    ''' Function used to compute the running average
        of the last N elements of a vector x
    '''
    if len(x) >= N:
        y = np.copy(x)
        y[N-1:] = np.convolve(x, np.ones((N, )) / N, mode='valid')
    else:
        y = np.zeros_like(x)
    return y


# Import and initialize the discrete Lunar Laner Environment , render_mode='rgb_array'
env = gym.make('LunarLander-v2')
env.reset()

# Parameters
N_episodes = 400                       # Number of episodes
discount_factor = 1                     # Value of the discount factor
n_ep_running_average = 50                    # Running average of 50 episodes
n_actions = env.action_space.n               # Number of available actions
dim_state = len(env.observation_space.high)  # State dimensionality
L = 30000
N = 128
C = L // N
eps_min = 0.05
eps_max = 0.99
# We will use these variables to compute the average episodic reward and
# the average number of steps per episode
episode_reward_list = []       # this list contains the total reward per episode
episode_number_of_steps = []   # this list contains the number of steps per episode

# Random agent initialization

# Training process

# trange is an alternative to range in python, from the tqdm library
# It shows a nice progression bar that you can update with useful information
EPISODES = trange(N_episodes, desc='Episode: ', leave=True)
buffer = ExperienceReplayBuffer(max_len=L)

network = FFNetwork(dim_state, n_actions)
target_network = copy.deepcopy(network)

optimizer = optim.Adam(network.parameters(), lr=5e-4)
counter = 0
# model = torch.load('neural-network-1.pth')
for i in EPISODES:
    # Reset enviroment data and initialize variables
    done = False
    state = env.reset()[0]
    total_episode_reward = 0.
    t = 0
    eps = max(eps_min, eps_max - ((eps_max - eps_min)
              * (i - 1) / (N_episodes * 0.9 - 1)))
    while not done and t < 1000:
        # epsilon decay

        # Take a random action
        s_tensor = torch.tensor(
            [state], requires_grad=False, dtype=torch.float32)
        # action = random_agent.forward(state)
        q_values = network(s_tensor)
        # action = my_agent.forward(s_tensor, eps)
        if np.random.uniform(0, 1) <= eps:
            action = env.action_space.sample()
        else:
            _, action = q_values.max(1)
            action = action.item()

        # x = env.render()
        # plot_env(x)

        # Get next state and reward.  The done variable
        # will be True if you reached the goal position,
        # False otherwise
        next_state, reward, done, _, _ = env.step(action)

        exp = (state, action, reward, next_state, done)

        buffer.append(exp)
        if len(buffer) > 200:
            states, actions, rewards, next_states, dones = buffer.sample_batch(
                N)
            states_tensor = torch.tensor(
                states, requires_grad=True, dtype=torch.float32)

            next_states_values = target_network(torch.tensor(
                next_states, requires_grad=False, dtype=torch.float32))
            target_values = torch.zeros(
                N, requires_grad=False, dtype=torch.float32)

            # fix the targets
            for j in range(len(target_values)):
                if dones[j]:
                    target_values[j] = rewards[j]
                else:
                    # pdb.set_trace()
                    max_target_value = torch.max(next_states_values[j]).item()
                    discounted_value = discount_factor * max_target_value
                    target_values[j] = rewards[j] + discounted_value

            q_values = network(states_tensor)
            q_a = torch.zeros(N, requires_grad=False, dtype=torch.float32)

            for i in range(N):
                q_a[i] = q_values[i, actions[i]]

            optimizer.zero_grad()

            loss = nn.functional.mse_loss(q_a, target_values)

            loss.backward()

            nn.utils.clip_grad_norm_(network.parameters(), max_norm=1)

            optimizer.step()

        # Update episode reward
        total_episode_reward += reward

        counter += 1
        if counter == C:
            target_network.load_state_dict(network.state_dict())
            counter = 0

        # Update state for next iteration
        state = next_state
        t += 1

    # Append episode reward and total number of steps
    episode_reward_list.append(total_episode_reward)
    episode_number_of_steps.append(t)

    # Close environment
    env.close()

    # Updates the tqdm update bar with fresh information
    # (episode number, total reward of the last episode, total number of Steps
    # of the last episode, average reward, average number of steps)
    EPISODES.set_description(
        "Episode {} - Reward/Steps: {:.1f}/{} - Avg. Reward/Steps: {:.1f}/{}".format(
            i, total_episode_reward, t,
            running_average(episode_reward_list, n_ep_running_average)[-1],
            running_average(episode_number_of_steps, n_ep_running_average)[-1]))


torch.save(network, 'neural-network-1.pth')

# Plot Rewards and steps
fig, ax = plt.subplots(nrows=1, ncols=2, figsize=(16, 9))
ax[0].plot([i for i in range(1, N_episodes+1)],
           episode_reward_list, label='Episode reward')
ax[0].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_reward_list, n_ep_running_average), label='Avg. episode reward')
ax[0].set_xlabel('Episodes')
ax[0].set_ylabel('Total reward')
ax[0].set_title('Total Reward vs Episodes')
ax[0].legend()
ax[0].grid(alpha=0.3)

ax[1].plot([i for i in range(1, N_episodes+1)],
           episode_number_of_steps, label='Steps per episode')
ax[1].plot([i for i in range(1, N_episodes+1)], running_average(
    episode_number_of_steps, n_ep_running_average), label='Avg. number of steps per episode')
ax[1].set_xlabel('Episodes')
ax[1].set_ylabel('Total number of steps')
ax[1].set_title('Total number of steps vs Episodes')
ax[1].legend()
ax[1].grid(alpha=0.3)
plt.show()
