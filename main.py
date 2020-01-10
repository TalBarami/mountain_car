import numpy as np
import gym
import matplotlib.pyplot as plt
from sklearn.kernel_approximation import RBFSampler
from sklearn.preprocessing import StandardScaler
from sklearn.pipeline import FeatureUnion

from tqdm import tqdm


class Agent:
    def __init__(self):
        self.env = gym.make("MountainCar-v0")

        self.max_epochs = 200
        self.alpha = 0.1
        self.gamma = 1.0
        self.epsilon = 0.1

        gammas = [0.5, 1.0, 2.0, 3.5, 5.0]
        self.n_actions = self.env.action_space.n
        self.n_components = 30
        features = [(f'rbf{i}', RBFSampler(gamma=g, n_components=self.n_components, random_state=1)) for i, g in
                    enumerate(gammas)]
        samples = np.array([self.env.observation_space.sample() for _ in range(10000)])

        self.scaler = StandardScaler()
        self.scaler.fit(samples)
        self.featurizer = FeatureUnion(features)
        self.featurizer.fit(self.scaler.transform(samples))

        self.w = np.zeros((self.env.action_space.n, self.n_components * len(features)))

    def featurize_state(self, state):
        return self.featurizer.transform(self.scaler.transform([state]))

    def Q(self, state, action):
        return state.dot(self.w[action])

    def policy(self, state):
        A = np.ones(self.n_actions, dtype=float) * self.epsilon / self.n_actions
        a = np.argmax([self.Q(state, a) for a in range(self.n_actions)])
        A[a] += (1.0 - self.epsilon)
        return np.random.choice(self.n_actions, p=A)

    def train(self, max_epochs=None, alpha=None, gamma=None, epsilon=None, verbose=10):
        self.max_epochs = max_epochs or self.max_epochs
        self.alpha = alpha or self.alpha
        self.gamma = gamma or self.gamma
        self.epsilon = epsilon or self.epsilon

        scatter_x = []
        scatter_y = []
        scatter_s = []

        for e in tqdm(range(self.max_epochs)):
            state = self.env.reset()
            state = self.featurize_state(state)
            c = 0
            while True:
                action = self.policy(state)

                next_state, reward, done, _ = self.env.step(action)
                next_state = self.featurize_state(next_state)
                next_action = self.policy(next_state)

                current_q = self.Q(state, action)
                next_q = self.Q(next_state, next_action)

                t = reward + self.gamma * next_q
                err = current_q - t
                dw = err.dot(state)

                self.w[action] -= self.alpha * dw

                c += 1
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
        plt.savefig('result.png')

    def evaluate(self):
        epochs = 25
        r = np.zeros(epochs)
        for e in range(epochs):
            state = self.env.reset()
            state = self.featurize_state(state)
            c = 0
            while True:
                action = self.policy(state)
                next_state, reward, done, _ = self.env.step(action)
                state = self.featurize_state(next_state)

                c += 1
                if done:
                    r[e] = c
                    break

        return r.mean(), r.std()

    def simulate(self, state=None):
        state = state or self.env.reset()
        state = self.featurize_state(state)

        while True:
            self.env.render()
            action = self.policy(state)
            next_state, reward, done, _ = self.env.step(action)
            state = self.featurize_state(next_state)

            if done:
                break

    def shutdown(self):
        self.env.close()


if __name__ == "__main__":
    agent = Agent()
    agent.train()

    for i in range(10):
        agent.simulate()
    agent.shutdown()
