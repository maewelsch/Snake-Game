"""
dqn_agent.py

Deep Q-Network (DQN) implemented from scratch with NumPy.

Why NumPy?
-----------------------------------------
Because this project's state space is small (only 11 features) and the network itself is small (two hidden layers),
a full deep-learning framework wasn't necessary to get good results. 
Writing the forward pass, backpropagation, and the Adam optimizer also has a nice side benefit: every step of the 
Deep Q-Learning algo is visible in the code instead of hidden inside a library. That said, the agent still includes 
every piece of a real DQN, as described by Mnih et al. (2015): a neural network that approximates Q-values, an 
experience replay buffer, a separate target network for stable learning targets, and epsilon-greedy exploration that 
decays over time.

Class overview
--------------
QNetwork — a small neural network (state size → hidden → hidden → action size) with a hand-written forward pass, 
backward pass, and Adam optimizer.
ReplayBuffer — a fixed-size buffer that stores past experiences (state, action, reward, next state, done) and 
samples them in random batches.
DQNAgent — brings the two together: it picks actions with epsilon-greedy exploration, stores each experience as 
it happens, and runs the training step that computes the target Q-value using the Bellman equation: y = r + gamma 
* max_a' Q_target(s', a')
"""

import numpy as np
from collections import deque
import random


def relu(x):
    return np.maximum(0, x)


def relu_grad(x):
    return (x > 0).astype(x.dtype)


class QNetwork:

    def __init__(self, input_size, hidden_size, output_size, lr=1e-3, seed=None):
        rng = np.random.default_rng(seed)
        # He initialization, good default for ReLU networks
        self.W1 = rng.normal(0, np.sqrt(2.0 / input_size), (input_size, hidden_size)).astype(np.float32)
        self.b1 = np.zeros(hidden_size, dtype=np.float32)
        self.W2 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, hidden_size)).astype(np.float32)
        self.b2 = np.zeros(hidden_size, dtype=np.float32)
        self.W3 = rng.normal(0, np.sqrt(2.0 / hidden_size), (hidden_size, output_size)).astype(np.float32)
        self.b3 = np.zeros(output_size, dtype=np.float32)

        self.lr = lr
        # Adam optimizer state
        self._adam_state = {}
        for name in ["W1", "b1", "W2", "b2", "W3", "b3"]:
            self._adam_state[name] = {
                "m": np.zeros_like(getattr(self, name)),
                "v": np.zeros_like(getattr(self, name)),
            }
        self._t = 0
        self.beta1, self.beta2, self.eps = 0.9, 0.999, 1e-8

    def forward(self, x):
        """x: (batch, input_size). Returns Q-values (batch, output_size) and a cache for backprop."""
        z1 = x @ self.W1 + self.b1
        a1 = relu(z1)
        z2 = a1 @ self.W2 + self.b2
        a2 = relu(z2)
        z3 = a2 @ self.W3 + self.b3  # linear output layer -> raw Q-values
        cache = (x, z1, a1, z2, a2, z3)
        return z3, cache

    def predict(self, x):
        q, _ = self.forward(x)
        return q

    def backward(self, cache, grad_output):
        """grad_output: dLoss/dQ, shape (batch, output_size)."""
        x, z1, a1, z2, a2, z3 = cache
        batch = x.shape[0]

        dW3 = a2.T @ grad_output / batch
        db3 = grad_output.mean(axis=0)

        da2 = grad_output @ self.W3.T
        dz2 = da2 * relu_grad(z2)
        dW2 = a1.T @ dz2 / batch
        db2 = dz2.mean(axis=0)

        da1 = dz2 @ self.W2.T
        dz1 = da1 * relu_grad(z1)
        dW1 = x.T @ dz1 / batch
        db1 = dz1.mean(axis=0)

        grads = {"W1": dW1, "b1": db1, "W2": dW2, "b2": db2, "W3": dW3, "b3": db3}
        self._adam_step(grads)

    def _adam_step(self, grads):
        self._t += 1
        for name, grad in grads.items():
            state = self._adam_state[name]
            state["m"] = self.beta1 * state["m"] + (1 - self.beta1) * grad
            state["v"] = self.beta2 * state["v"] + (1 - self.beta2) * (grad ** 2)
            m_hat = state["m"] / (1 - self.beta1 ** self._t)
            v_hat = state["v"] / (1 - self.beta2 ** self._t)
            update = self.lr * m_hat / (np.sqrt(v_hat) + self.eps)
            param = getattr(self, name)
            setattr(self, name, param - update)

    def get_weights(self):
        return {n: getattr(self, n).copy() for n in ["W1", "b1", "W2", "b2", "W3", "b3"]}

    def set_weights(self, weights):
        for n, v in weights.items():
            setattr(self, n, v.copy())


class ReplayBuffer:

    def __init__(self, capacity=20000):
        self.buffer = deque(maxlen=capacity)

    def push(self, state, action, reward, next_state, done):
        self.buffer.append((state, action, reward, next_state, done))

    def sample(self, batch_size):
        batch = random.sample(self.buffer, batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        return (
            np.array(states, dtype=np.float32),
            np.array(actions, dtype=np.int64),
            np.array(rewards, dtype=np.float32),
            np.array(next_states, dtype=np.float32),
            np.array(dones, dtype=np.float32),
        )

    def __len__(self):
        return len(self.buffer)


class DQNAgent:

    def __init__(
        self,
        state_size=11,
        action_size=3,
        hidden_size=128,
        lr=1e-3,
        gamma=0.95,
        epsilon_start=1.0,
        epsilon_min=0.01,
        epsilon_decay=0.995,
        buffer_capacity=20000,
        batch_size=64,
        target_sync_every=200,
        seed=42,
    ):
        self.state_size = state_size
        self.action_size = action_size
        self.gamma = gamma
        self.epsilon = epsilon_start
        self.epsilon_min = epsilon_min
        self.epsilon_decay = epsilon_decay
        self.batch_size = batch_size
        self.target_sync_every = target_sync_every

        self.q_network = QNetwork(state_size, hidden_size, action_size, lr=lr, seed=seed)
        self.target_network = QNetwork(state_size, hidden_size, action_size, lr=lr, seed=seed)
        self.target_network.set_weights(self.q_network.get_weights())

        self.replay_buffer = ReplayBuffer(buffer_capacity)
        self._train_steps = 0

    def act(self, state, explore=True):
        if explore and np.random.rand() < self.epsilon:
            return np.random.randint(self.action_size)
        q_values = self.q_network.predict(state.reshape(1, -1))
        return int(np.argmax(q_values[0]))

    def remember(self, state, action, reward, next_state, done):
        self.replay_buffer.push(state, action, reward, next_state, done)

    def decay_epsilon(self):
        self.epsilon = max(self.epsilon_min, self.epsilon * self.epsilon_decay)

    def train_step(self):
        """One gradient step of Deep Q-Learning using a sampled minibatch."""
        if len(self.replay_buffer) < self.batch_size:
            return None

        states, actions, rewards, next_states, dones = self.replay_buffer.sample(self.batch_size)

        q_values, cache = self.q_network.forward(states)

        next_q_target = self.target_network.predict(next_states)
        max_next_q = np.max(next_q_target, axis=1)
        targets = rewards + (1.0 - dones) * self.gamma * max_next_q

        grad_output = np.zeros_like(q_values)
        batch_idx = np.arange(len(actions))
        td_error = q_values[batch_idx, actions] - targets
        grad_output[batch_idx, actions] = 2.0 * td_error  # d/dq (q - target)^2

        self.q_network.backward(cache, grad_output)

        self._train_steps += 1
        if self._train_steps % self.target_sync_every == 0:
            self.target_network.set_weights(self.q_network.get_weights())

        return float(np.mean(td_error ** 2))

    def save(self, path):
        np.savez(path, **self.q_network.get_weights(), epsilon=self.epsilon)

    def load(self, path):
        data = np.load(path)
        weights = {k: data[k] for k in ["W1", "b1", "W2", "b2", "W3", "b3"]}
        self.q_network.set_weights(weights)
        self.target_network.set_weights(weights)
        if "epsilon" in data:
            self.epsilon = float(data["epsilon"])
