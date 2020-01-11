import numpy as np
import random
import gym
import matplotlib.pyplot as plt
from tqdm import tqdm


class Agent:
    def __init__(self):
        self.env = gym.make("MountainCar-v0")
        self.env._max_episode_steps = 600

        self.state_space = ((self.env.high - self.env.low) * np.array([10, 100])).astype(int) + 1

        self.q_table = np.random.uniform(low=-1, high=1,
                                         size=(self.state_space[0], self.state_space[1],
                                               self.env.action_space.n))

        self.max_epochs = 800
        self.alpha = 0.2
        self.gamma = 1.0
        self.epsilon = 0.2

    def discretsize(self, state):
        state = (state - self.env.observation_space.low) * np.array([10, 100])
        return np.round(state, 0).astype(int)

    def policy(self, state, epsilon=0):
        return self.env.action_space.sample() if random.uniform(0, 1) < epsilon else np.argmax(self.q_table[state[0], state[1]])

    def train(self, max_epochs=None, alpha=None, gamma=None, epsilon=None, verbose=20):
        self.max_epochs = max_epochs or self.max_epochs
        self.alpha = alpha or self.alpha
        self.gamma = gamma or self.gamma
        self.epsilon = epsilon or self.epsilon

        scatter_x = []
        scatter_y = []
        scatter_s = []

        for e in tqdm(range(self.max_epochs)):
            state = self.env.reset()
            state = self.discretsize(state)

            while True:
                action = self.policy(state, self.epsilon)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.discretsize(next_state)

                current_q = self.q_table[state[0], state[1], action]
                next_q = np.max(self.q_table[next_state[0], next_state[1]])

                self.q_table[state[0], state[1], action] = current_q + self.alpha * (reward + self.gamma * next_q - current_q)
                state = next_state

                if done:
                    break

            if e % verbose == 0:
                y, s = self.evaluate()
                scatter_x.append(e)
                scatter_y.append(y)
                scatter_s.append(s)

        print("Training completed.")
        plt.errorbar(scatter_x, scatter_y, scatter_s, linestyle='None', marker='^')

        plt.show()

    def evaluate(self):
        epochs = 25
        r = np.zeros(epochs)
        for e in range(epochs):
            state = self.env.reset()
            state = self.discretsize(state)
            c = 0
            while True:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                state = self.discretsize(next_state)

                c += 1
                if done:
                    r[e] = c
                    break

        return r.mean(), r.std()

    def simulate(self, state=None):
        env = self.env.env

        state = state or env.reset()
        state = self.discretsize(state)

        while True:
            self.env.render()
            action = self.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = self.discretsize(next_state)

            if done:
                break

    def shutdown(self):
        self.env.close()


if __name__ == "__main__":
    agent = Agent()
    agent.train()

    for i in range(100):
        agent.simulate()
    agent.shutdown()
