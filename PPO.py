import random
import tensorflow as tf
import numpy as np
import gym
from tqdm import tqdm

random.seed(202)
tf.random.set_seed(202)


"""
PPO Implementation for CartPole-v0
"""


obs_size = 4
action_size = 2


class PPO:
    def __init__(self, critic_optimizer, policy_optimizer, lmbda, gamma, epochs, epsilon, batch_size):
        self.policy_optimizer = policy_optimizer
        self.critic_optimizer = critic_optimizer
        self.lmbda = lmbda
        self.gamma = gamma
        self.epochs = epochs
        self.epsilon = epsilon
        self.batch_size = batch_size

        self.actor = None
        self.critic = None
        self.build_actor()
        self.build_critic()

    def build_actor(self):
        inp = tf.keras.layers.Input((obs_size,))
        x = tf.keras.layers.Dense(64, activation=tf.keras.activations.swish, kernel_initializer=tf.keras.initializers.Orthogonal)(inp)
        x = tf.keras.layers.Dense(32, activation=tf.keras.activations.swish, kernel_initializer=tf.keras.initializers.Orthogonal)(x)
        policy = tf.keras.layers.Dense(action_size, activation='softmax')(x)
        self.actor = tf.keras.models.Model(inp, policy)

    def build_critic(self):
        inp = tf.keras.layers.Input((obs_size,))
        x = tf.keras.layers.Dense(64, activation=tf.keras.activations.swish, kernel_initializer=tf.keras.initializers.Orthogonal)(inp)
        x = tf.keras.layers.Dense(32, activation=tf.keras.activations.swish, kernel_initializer=tf.keras.initializers.Orthogonal)(x)
        value = tf.keras.layers.Dense(1)(x)
        self.critic = tf.keras.models.Model(inp, value)

    def act(self, state):  # sample action from policy distribution
        policy = self.actor(np.array([state]))
        action = random.choices([i for i in range(action_size)], weights=policy[0], k=1)
        return action, policy[0]

    def test(self, env, tests=100):
        """
        :param env: environment to evaluate in
        :param tests: number of episodes to run
        :return: average return of self.actor in env
        """
        returns = []
        for _ in range(tests):
            state = env.reset()
            rtg = 0
            done = False
            while not done:
                policy = self.actor(np.array([state]))
                action = np.argmax(policy)
                state, reward, done, _ = env.step(action)
                rtg += reward
            returns.append(rtg)
        return np.mean(returns)

    def batch_generator(self, time_steps):
        """
        :param time_steps: list or tuple of (returns, advantages, states, actions, probabilities)
        yields batch_size batches of time_steps for training
        """
        returns, advantages, states, actions, probs = time_steps
        returns_batch, adv_batch, states_batch, actions_batch, probs_batch = [], [], [], [], []
        idxs = np.arange(len(returns))
        np.random.shuffle(idxs)
        for k in idxs:
            returns_batch.append(returns[k])
            states_batch.append(states[k])
            actions_batch.append(actions[k])
            probs_batch.append(probs[k])
            adv_batch.append(advantages[k])
            if len(states_batch) == self.batch_size:
                yield np.array(returns_batch), np.array(adv_batch), np.array(states_batch), np.array(actions_batch), np.array(probs_batch)
                returns_batch, adv_batch, states_batch, actions_batch, probs_batch = [], [], [], [], []

    def train(self, time_steps):
        """
        :param time_steps: time steps collected from environment interaction
        :return: mean of critic and actor losses
        """
        policy_losses = []
        critic_losses = []
        for _ in range(self.epochs):
            for returns, advantages, states, actions, old_probs in self.batch_generator(time_steps):
                old_logprobs_dist = np.log(old_probs + 1e-10)
                old_logprobs = tf.reduce_sum(old_logprobs_dist * tf.one_hot(actions, depth=action_size), axis=-1)

                with tf.GradientTape() as tape1:
                    values = tf.reshape(self.critic(states), (self.batch_size,))
                    critic_loss = 0.5 * tf.reduce_mean(tf.square(returns - values))

                with tf.GradientTape() as tape2:
                    policy = self.actor(states)
                    logprobs = tf.reduce_sum(tf.math.log(policy) * tf.one_hot(actions, depth=action_size), axis=-1)

                    entropy = tf.reduce_mean(tf.math.negative(tf.math.multiply(policy, tf.math.log(policy))))

                    ratios = tf.exp(logprobs - old_logprobs)
                    s1 = ratios * advantages
                    s2 = tf.clip_by_value(ratios, 1 - self.epsilon, 1 + self.epsilon) * advantages
                    policy_loss = -tf.reduce_mean(tf.minimum(s1, s2)) - 0.001 * entropy

                grads = tape2.gradient(target=policy_loss, sources=self.actor.trainable_variables)
                self.policy_optimizer.apply_gradients(zip(grads, self.actor.trainable_variables))

                grads = tape1.gradient(target=critic_loss, sources=self.critic.trainable_variables)
                self.critic_optimizer.apply_gradients(zip(grads, self.critic.trainable_variables))

                policy_losses.append(policy_loss)
                critic_losses.append(critic_loss)
        print(np.mean(policy_losses), np.mean(critic_losses))

    def collect_data(self, time_steps, env):
        """
        :param time_steps: number of time steps to collect
        :param env: environment to interact with
        :return: returns, advantages, states, actions, probabilities
        """
        rewards = np.empty((time_steps, ), dtype=np.float32)
        dones = np.empty((time_steps, ), dtype=bool)
        states = np.empty((time_steps, obs_size), dtype=np.float32)
        actions = np.empty((time_steps, ), dtype=np.int32)
        probs = np.empty((time_steps, action_size), dtype=np.float32)
        values = np.empty((time_steps + 1, ), dtype=np.float32)

        state = env.reset()
        for i in tqdm(range(time_steps)):
            action, policy = self.act(state)
            value = self.critic(np.array([state]))[0]
            next_state, reward, done, _ = env.step(action[0])
            states[i] = state
            probs[i] = policy
            dones[i] = done
            values[i] = value
            actions[i] = action[0]
            rewards[i] = reward
            state = next_state
            if done:
                state = env.reset()
        values[time_steps] = self.critic(np.array([state]))[0]

        returns = []
        gae = 0
        for i in reversed(range(len(rewards))):
            delta = rewards[i] + self.gamma * values[i + 1] * (1 - dones[i]) - values[i]
            gae = delta + self.gamma * self.lmbda * gae * (1 - dones[i])
            returns.insert(0, gae + values[i])

        advantages = np.array(returns) - values[:-1]
        advantages = (advantages - np.mean(advantages)) / (np.std(advantages) + 1e-8)
        return returns, advantages, states, actions, probs


p_opt = tf.keras.optimizers.Adam(learning_rate=0.007, beta_1=0.9)
c_opt = tf.keras.optimizers.Adam(learning_rate=0.007, beta_1=0.9)

agent = PPO(c_opt, p_opt, gamma=1, lmbda=0.95, epochs=10, epsilon=0.2, batch_size=512)

env = gym.make('CartPole-v0')

print(agent.test(env))
for i in range(100):
    data = agent.collect_data(4096, env)
    agent.train(data)
    score = agent.test(env)
    print(score)
    if score == 200.0:
        exit()











