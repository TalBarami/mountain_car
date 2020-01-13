import numpy as np
import gym
import abc
from tqdm import tqdm
from os.path import join
import matplotlib.pyplot as plt


class BaseAgent(object):
    __metaclass__ = abc.ABCMeta

    def __init__(self):
        self.result_folder = 'results'
        self.env = gym.make("MountainCar-v0")
        self.env._max_episode_steps = None
        self.max_epochs = None
        self.alpha = None
        self.gamma = None
        self.epsilon = None
        self.result_name = None

    @abc.abstractmethod
    def featurize(self, state):
        raise NotImplementedError

    @abc.abstractmethod
    def policy(self, state, epsilon=0):
        raise NotImplementedError

    @abc.abstractmethod
    def perform_step(self, state, action):
        raise NotImplementedError

    def train(self, max_epochs=None, alpha=None, gamma=None, epsilon=None, verbose=None):
        self.max_epochs = max_epochs or self.max_epochs
        self.alpha = alpha or self.alpha
        self.gamma = gamma or self.gamma
        self.epsilon = epsilon or self.epsilon
        verbose = verbose or self.max_epochs / 20

        scatter_x = []
        scatter_y = []
        scatter_s = []

        for e in tqdm(range(self.max_epochs)):
            state = self.env.reset()
            state = self.featurize(state)

            while True:
                action = self.policy(state, self.epsilon)
                state, done = self.perform_step(state, action)

                if done:
                    break

            if e % verbose == 0:
                y, s = self.evaluate()
                scatter_x.append(e)
                scatter_y.append(y)
                scatter_s.append(s)

        print("Training completed.")
        plt.errorbar(scatter_x, scatter_y, scatter_s, linestyle='None', marker='^')
        plt.savefig(join(self.result_folder, self.result_name))
        plt.show()

    def evaluate(self):
        epochs = 20
        r = np.zeros(epochs)
        for e in range(epochs):
            state = self.env.reset()
            state = self.featurize(state)
            c = 0
            while True:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                state = self.featurize(next_state)

                c += 1
                if done:
                    r[e] = c
                    break

        return r.mean(), r.std()

    def simulate(self, state=None, render=False):
        env = self.env.env
        state = state or env.reset()
        state = self.featurize(state)

        c = 0
        while True:
            if render:
                env.render()
            action = self.policy(state)
            next_state, reward, done, _ = env.step(action)
            state = self.featurize(next_state)

            c += 1
            if done:
                return True
            if c > 800:
                print('Simulation failed!')
                return False

    def shutdown(self):
        self.env.close()
