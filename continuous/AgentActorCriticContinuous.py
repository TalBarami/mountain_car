import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.distributions.normal import Normal
from torch.optim import Adam
import gym
import numpy as np
from tqdm import tqdm
env = gym.envs.make('MountainCarContinuous-v0')

class Policy(nn.Module):
        def __init__(self):
            super(Policy, self).__init__()
            self.fully1 = nn.Linear(2, 40)
            self.fully2 = nn.Linear(40, 40)
            self.mu     = nn.Linear(40, 1)
            self.sigma  = nn.Linear(40, 1)

        def forward(self, x):
            x = self.fully1(x)
            x = F.relu(x)

            x = self.fully2(x)
            x = F.relu(x)

            mu = self.mu(x)
            mu = F.relu(mu)

            sigma = self.sigma(x)
            sigma = F.relu(sigma) + 1e-5

            dist = Normal(mu, sigma)
            return dist

class PolicyLoss(nn.Module):
    def forward(self, x):
        dist = x[0]
        action = x[1]
        target = x[2]
        loss = -dist.log_prob(action) * target
        loss -= 1e-1 * dist.entropy()
        return loss


class ValueFunction(nn.Module):
    def __init__(self):
        super(ValueFunction, self).__init__()
        self.fully1 = nn.Linear(2, 40)
        self.fully2 = nn.Linear(40, 40)
        self.fully3 = nn.Linear(40, 1)

    def forward(self, x):
        x = self.fully1(x)
        x = F.relu(x)

        x = self.fully2(x)
        x = F.relu(x)

        x = self.fully3(x)
        return x


def train(policy, value_function, n_episodes, discount_factor=1.0):
    # stats = plotting.EpisodeStats(
    # 	episode_lengths = np.zeros(n_episodes),
    # 	episode_rewards = np.zeros(n_episodes))

    discount_factor = torch.tensor(discount_factor)

    episode_lengths = np.zeros(n_episodes)
    episode_rewards = np.zeros(n_episodes)
    max_steps = 1000
    policy_optimizer = Adam(policy.parameters(), lr=0.001)
    policy_criterion = PolicyLoss()
    value_optimizer = Adam(value_function.parameters(), lr=0.1)

    for ith_episode in tqdm(range(n_episodes)):
        state = env.reset()
        state = torch.tensor([state], dtype=torch.float)

        for t in range(max_steps):
            if ith_episode % 10 == 9:
                env.render()

            policy_dist = policy(state)
            action = policy_dist.sample()
            action = torch.clamp(action, env.action_space.low[0], env.action_space.high[0])

            next_state, reward, done, _ = env.step(action) #Todo convert to number
            episode_rewards[ith_episode] += reward
            reward = torch.tensor(reward, dtype=torch.float)
            next_state = torch.tensor([next_state], dtype=torch.float)

            value_next = value_function(next_state)
            td_target = reward + discount_factor * value_next
            td_loss = td_target - value_function(state)

            value_optimizer.zero_grad()
            td_loss.backward(retain_graph=True)
            value_optimizer.step()

            policy_optimizer.zero_grad()
            loss = policy_criterion([policy_dist, action, td_target])
            loss.backward(retain_graph=True)
            policy_optimizer.step()

            print("\rStep {} @ Episode {}/{} ({})".format(
                t, ith_episode + 1, n_episodes, episode_rewards[ith_episode - 1]), end="")

            if done:
                break

            state = next_state

    return episode_lengths, episode_rewards

policy = Policy()
value_function = ValueFunction()

train(policy, value_function, 100)