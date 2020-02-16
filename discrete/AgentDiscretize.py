import random

import numpy as np

from BaseAgent import BaseAgent


class AgentDiscretize(BaseAgent):
    def __init__(self):
        super().__init__("MountainCar-v0")
        self.name = "AgentDiscretize"
        self.env._max_episode_steps = 400
        self.max_epochs = 5000
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 0.8

        self.n_actions = self.env.action_space.n
        self.state_space = ((self.env.high - self.env.low) * np.array([10, 100])).astype(int) + 1

        self.w = np.zeros((self.n_actions, np.prod(self.state_space)))

    def featurize(self, state):
        i = np.round((state - self.env.observation_space.low) * np.array([10, 100]), 0).astype(int)
        s = np.zeros(self.state_space)
        s[i[0], i[1]] = 1

        return s.reshape(1, np.prod(self.state_space))

    def Q(self, state, action):
        return state.dot(self.w[action])

    def policy(self, state, epsilon=0):
        a = np.argmax([self.Q(state, a) for a in range(self.n_actions)])
        return self.env.action_space.sample() if random.uniform(0, 1) < epsilon else a

    def perform_step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)

        if done and next_state[0] >= 0.5:
            mask = (state.dot(reward) != 0).reshape((state.shape[1]))
            self.w[action][mask != 0] = reward
        else:
            next_state = self.featurize(next_state)
            next_action = self.policy(next_state)
            current_q = self.Q(state, action)
            next_q = self.Q(next_state, next_action)

            self.w[action] += self.alpha * (reward + self.gamma * next_q - current_q).dot(state)
        return next_state, done
