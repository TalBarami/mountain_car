import numpy as np
import random

from discrete.BaseAgent import BaseAgent


class ActionDiscretizeAgent(BaseAgent):
    def __init__(self):
        super().__init__("MountainCarContinuous-v0")
        self.name = "ActionDiscretizeAgent"

        self.max_epochs = 600
        self.alpha = 0.5
        self.gamma = 0.7
        self.epsilon = 0.5
        self.c = 100

        self.state_space = ((self.env.observation_space.high - self.env.observation_space.low) * np.array([10, 100])).astype(int) + 1
        self.actions = [np.round(x, 2) for x in np.arange(self.env.action_space.low, self.env.action_space.high + 0.1, 0.1)]

        self.q_table = np.zeros([self.state_space[0] * self.state_space[1], len(self.actions)])

    def featurize(self, observation):
        state = np.floor((observation - self.env.observation_space.low) * np.array([10, 100])).astype(int)
        return self.state_space[1] * state[0] + state[1]

    def featurize_action(self, action):
        return int(np.round((action[0] - self.env.action_space.low[0]) * 10, 2))

    def policy(self, state, epsilon=0):
        if random.uniform(0, 1) < epsilon:
            return np.round(self.env.action_space.sample(), 2)
        else:
            return [self.actions[int(np.argmax(self.q_table[state]))]]

    def perform_step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        if done and next_state[0] >= 0.5:
            self.q_table[state[0], state[1], action] = reward
        else:
            action = self.featurize_action(action)

            reward += self.c * np.abs(next_state[1])
            next_state = self.featurize(next_state)

            self.q_table[state, action] = (1 - self.alpha) * self.q_table[state, action] + \
                                          self.alpha * (reward + self.gamma * np.max(self.q_table[next_state]) - self.q_table[state, action])
        return next_state, done

    def shutdown(self):
        self.env.close()
