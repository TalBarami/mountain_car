import tensorflow as tf
import gym
import numpy as np
import sklearn
import sklearn.preprocessing
import matplotlib.pyplot as plt
from os.path import join
from tqdm import tqdm


class ActorCriticAgent:
    def __init__(self, actor_lr=0.00001, critic_lr=0.0005, gamma=0.99):
        self.env = gym.envs.make("MountainCarContinuous-v0")
        self.name = 'AgentActorCritic'

        self.model_path = 'continuous/model/model.ckpt'
        self.result_folder = 'continuous/results'

        self.actor_lr = actor_lr
        self.critic_lr = critic_lr
        self.gamma = gamma
        self.epsilon = 1e-5
        self.state_placeholder = tf.placeholder(tf.float32, [None, 2])
        self.xavier_initializer = tf.contrib.layers.xavier_initializer()

        self.tf_value_function = self.init_value_function()
        self.tf_policy, distribution = self.init_policy()
        self.scaler = self.init_scaler()

        self.action_placeholder = tf.placeholder(tf.float32)
        self.delta_placeholder = tf.placeholder(tf.float32)
        self.target_placeholder = tf.placeholder(tf.float32)

        self.loss_actor = -tf.log(distribution.prob(self.action_placeholder) + self.epsilon) * self.delta_placeholder
        self.training_op_actor = tf.train.AdamOptimizer(self.actor_lr, name='actor_optimizer').minimize(self.loss_actor)

        self.loss_critic = tf.reduce_mean(tf.squared_difference(tf.squeeze(self.tf_value_function), self.target_placeholder))
        self.training_op_critic = tf.train.AdamOptimizer(self.critic_lr, name='critic_optimizer').minimize(self.loss_critic)

        self.saver = tf.train.Saver()

    def init_scaler(self):
        state_space_samples = np.array([self.env.observation_space.sample() for _ in range(10000)])
        scaler = sklearn.preprocessing.StandardScaler()
        scaler.fit(state_space_samples)
        return scaler

    def init_value_function(self):
        layer_size_1 = 512
        layer_size_2 = 512
        output_size = 1

        with tf.variable_scope("value_function_network"):
            layer_1 = tf.layers.dense(self.state_placeholder, layer_size_1, tf.nn.elu, self.xavier_initializer)
            layer_2 = tf.layers.dense(layer_1, layer_size_2, tf.nn.elu, self.xavier_initializer)
            tf_value_function = tf.layers.dense(layer_2, output_size, None, self.xavier_initializer)
        return tf_value_function

    def init_policy(self):
        layer_size_1 = 64
        layer_size_2 = 64
        output_size = 1

        with tf.variable_scope("policy_network"):
            layer_1 = tf.layers.dense(self.state_placeholder, layer_size_1, tf.nn.elu, self.xavier_initializer)
            layer_2 = tf.layers.dense(layer_1, layer_size_2, tf.nn.elu, self.xavier_initializer)
            mu = tf.layers.dense(layer_2, output_size, None, self.xavier_initializer)
            sigma = tf.layers.dense(layer_2, output_size, None, self.xavier_initializer)
            sigma = tf.nn.softplus(sigma) + self.epsilon
            distribution = tf.contrib.distributions.Normal(mu, sigma)
            tf_policy = tf.squeeze(distribution.sample(1), axis=0)
            tf_policy = tf.clip_by_value(tf_policy, self.env.action_space.low[0], self.env.action_space.high[0])
        return tf_policy, distribution

    def scale_state(self, state):
        scaled = self.scaler.transform([state])
        return scaled

    def value_function(self, sess, state):
        return np.squeeze(sess.run(self.tf_value_function,
                                   feed_dict={self.state_placeholder: self.scale_state(state)}))

    def policy(self, sess, state):
        return np.squeeze(sess.run(self.tf_policy,
                                   feed_dict={self.state_placeholder: self.scale_state(state)}), axis=0)

    def train(self, max_epochs=250, verbose=10):
        scatter_x = []
        scatter_y = []
        scatter_s = []
        self.env.seed(0)

        with tf.Session() as sess:
            sess.run(tf.global_variables_initializer())
            rewards = []
            for e in tqdm(range(max_epochs)):
                state = self.env.reset()
                episode_reward = 0
                while True:
                    action = self.policy(sess, state)
                    next_state, reward, done, _ = self.env.step(action)
                    episode_reward += reward
                    current_value = self.value_function(sess, state)
                    next_value = self.value_function(sess, next_state)

                    target = reward + self.gamma * next_value
                    error = target - current_value

                    sess.run([self.training_op_actor, self.loss_actor],
                             feed_dict={self.action_placeholder: action,
                                        self.state_placeholder: self.scale_state(state),
                                        self.delta_placeholder: error})

                    sess.run([self.training_op_critic, self.loss_critic],
                             feed_dict={self.state_placeholder: self.scale_state(state),
                                        self.target_placeholder: target})

                    state = next_state
                    if done:
                        break

                rewards.append(episode_reward)

                if e % verbose == 0:
                    y, s = self.evaluate(sess)
                    scatter_x.append(e)
                    scatter_y.append(y)
                    scatter_s.append(s)

                if (e % 5 == 4) and (np.mean(rewards) <= 0):
                    sess.run(tf.global_variables_initializer())
                    rewards = []
                elif np.mean(rewards[-100:]) > 90 and len(rewards) >= 101:
                    self.saver.save(sess, self.model_path)
                    break

        print("Training completed.")
        plt.errorbar(scatter_x, scatter_y, scatter_s, linestyle='None', marker='^')
        plt.savefig(join(self.result_folder, f'{self.name}.png'))
        plt.show()

    def evaluate(self, sess):
        epochs = 20
        r = np.zeros(epochs)
        for e in range(epochs):
            state = self.env.reset()
            c = 0
            while True:
                action = self.policy(sess, state)
                next_state, reward, done, _ = self.env.step(action)
                state = next_state
                c += 1
                if done:
                    r[e] = c
                    break

        return r.mean(), r.std()

    def simulate(self, state=None, render=False):
        env = self.env.env
        state = state or env.reset()

        c = 0
        with tf.Session() as sess:
            self.saver.restore(sess, self.model_path)
            while True:
                if render:
                    env.render()
                action = self.policy(sess, state)
                next_state, reward, done, _ = env.step(action)
                state = next_state

                c += 1
                if done:
                    return True
                if c > 800:
                    print('Simulation failed!')
                    return False

    def shutdown(self):
        self.env.close()
