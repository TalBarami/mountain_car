import numpy as np
from sklearn.kernel_approximation import RBFSampler
from sklearn.pipeline import FeatureUnion
from sklearn.preprocessing import StandardScaler

from BaseAgent import Agent


class SAgent(Agent):
    def __init__(self):
        super().__init__()
        self.env._max_episode_steps = 300
        self.max_epochs = 200
        self.alpha = 0.1
        self.gamma = 1.0
        self.epsilon = 0.1
        self.result_path = 'result_S.png'

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

    def featurize(self, state):
        return self.featurizer.transform(self.scaler.transform([state]))

    def Q(self, state, action):
        return state.dot(self.w[action])

    def policy(self, state, epsilon=0):
        A = np.ones(self.n_actions, dtype=float) * epsilon / self.n_actions
        a = np.argmax([self.Q(state, a) for a in range(self.n_actions)])
        A[a] += (1.0 - epsilon)
        return np.random.choice(self.n_actions, p=A)

    def perform_step(self, state, action):
        next_state, reward, done, _ = self.env.step(action)
        next_state = self.featurize(next_state)
        next_action = self.policy(next_state)
        current_q = self.Q(state, action)
        next_q = self.Q(next_state, next_action)

        self.w[action] += self.alpha * (reward + self.gamma * next_q - current_q).dot(state)
        return next_state, done
