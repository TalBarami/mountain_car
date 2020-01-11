import random

import numpy as np

from BaseAgent import Agent


class QAgent(Agent):
    def __init__(self):
        super().__init__()
        self.env._max_episode_steps = 600
        self.max_epochs = 800
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 0.2

        self.state_space = ((self.env.high - self.env.low) * np.array([10, 100])).astype(int) + 1
        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(self.state_space[0], self.state_space[1],
                                               self.env.action_space.n))

    def featurize(self, state):
        state = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state, 0).astype(int)

    def policy(self, state, epsilon=0):
        return self.env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(self.q_table[state[0], state[1]])

    def perform_step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.featurize(next_state)

        current_q = self.q_table[state[0], state[1], action]
        next_q = np.max(self.q_table[next_state[0], next_state[1]])

        self.q_table[state[0], state[1], action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
        return next_state, done
