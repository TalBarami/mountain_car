import random

import numpy as np

from BaseAgent import BaseAgent


class AgentQTableContinuous(BaseAgent):
    def __init__(self):
        super().__init__("MountainCarContinuous-v0")
        self.name = "AgentQTableContinuous"
        self.env._max_episode_steps = 800
        self.max_epochs = 15000
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 0.8

        self.state_space = ((self.env.observation_space.high - self.env.observation_space.low) * np.array([10, 100])).astype(int) + 1
        self.action_space = ((self.env.action_space.high - self.env.action_space.low) * np.array([10])).astype(int) + 1
        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(self.state_space[0], self.state_space[1],
                                               self.action_space[0]))

    def featurize(self, state):
        state = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state, 0).astype(int)

    def policy(self, state, epsilon=0):
        if random.uniform(0, 1) < epsilon:
            return self.env.action_space.sample()
        else:
            return self.index_to_action(np.argmax(self.q_table[state[0], state[1]]))

    def Q(self, state, action):
        return self.q_table[state[0], state[1], self.action_to_index(action)]

    def action_to_index(self, action):
        int((action - self.env.action_space.low) * 10)

    def index_to_action(self, index):
        return index/10 + self.env.action_space.low

    def perform_step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)

        if done and next_state[0] >= 0.5:
            self.q_table[state[0], state[1], action] = reward
        else:
            next_state = self.featurize(next_state)
            next_action = self.policy(next_state)
            current_q = self.Q(state, action)
            next_q = self.Q(next_state, next_action)

            self.q_table[state[0], state[1], self.action_to_index(action)] += self.alpha * (reward + self.gamma * next_q - current_q)
        self.epsilon -= self.epsilon / self.max_epochs
        return next_state, done
