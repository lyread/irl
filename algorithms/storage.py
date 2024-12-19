import random
from collections import deque

import numpy as np


class ReplayBuffer:
    """
    Experience replay buffer for storing and sampling past experiences.

    Args:
        buffer_capacity (int): Maximum capacity of the replay buffer.
    """

    def __init__(self, buffer_capacity):
        self.buffer_capacity = buffer_capacity
        self.buffer = deque(maxlen=buffer_capacity)

    def is_full(self):
        return len(self.buffer) == self.buffer_capacity

    def add(self, state, action, reward, next_state, next_action, terminated):
        """
        Add a transition to the replay buffer.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            next_state: Next state.
            next_action: Action taken in the next state.
            terminated: Whether the episode terminated after this transition.
        """
        self.buffer.append((state, action, reward, next_state, next_action, terminated))

    def sample(self, batch_size):
        """
        Sample a batch of transitions from the replay buffer.

        Args:
            batch_size: Size of the batch to sample.

        Returns:
            list: List of sampled transitions.
        """
        return random.sample(self.buffer, batch_size)

    def clear(self):
        self.buffer = deque(maxlen=self.buffer_capacity)


class Rollout:
    """
    Class for storing rollouts during training used by PPO.

    Args:
        size (int): Size of the rollout buffer.
    """

    def __init__(self, size):
        self.size = size
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.log_probs = []
        self.values = []
        self.advantages = np.zeros(size, dtype=np.float32)

    def collect_experience(self, state, action, reward, terminated, log_probs, values):
        """
        Collect experience by interacting with the environment.

        Args:
            state: Current state.
            action: Action taken.
            reward: Reward received.
            terminated: Whether the episode terminated after this transition.
            log_probs: Log probabilities of actions.
            values: Estimated values of states.
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)
        self.terminated.append(terminated)
        self.log_probs.append(log_probs)
        self.values.append(values)

    def is_full(self) -> bool:
        """
        Check if the rollout buffer is full.

        Returns:
            bool: True if the buffer is full, False otherwise.
        Note:
            Need one more transition to compute advantages correctly.
        """
        return len(self.states) == self.size + 1

    def compute_advantage(
        self, gamma: float, gae_lambda: float, normalize: bool = False
    ):
        """
        Compute advantage using Generalized Advantage Estimation (GAE).

        Args:
            gamma: Discount factor.
            gae_lambda: Lambda parameter for Generalized Advantage Estimation.
            normalize: Whether to normalize advantages.
        """
        gae = 0
        for t in reversed(range(self.size)):
            delta = (
                self.rewards[t]
                + gamma * self.values[t + 1] * (1 - self.terminated[t])
                - self.values[t]
            )
            gae = delta + gamma * gae_lambda * (1 - self.terminated[t]) * gae
            self.advantages[t] = gae
        if normalize:
            self.advantages = (self.advantages - np.mean(self.advantages)) / (
                np.std(self.advantages)
            )
        self._to_numpy()

    def _to_numpy(self):
        """
        Convert collected experiences to numpy arrays.
        And remove redundant transitions.
        """
        if not self.is_full():
            raise ValueError("Buffer is not full yet")
        self.states = np.array(self.states[:-1], dtype=np.float32)
        self.actions = np.array(self.actions[:-1], dtype=np.int64)
        self.rewards = np.array(self.rewards[:-1], dtype=np.float32)
        self.terminated = np.array(self.terminated[:-1], dtype=np.int64)
        self.log_probs = np.array(self.log_probs[:-1], dtype=np.float32)
        self.values = np.array(self.values[:-1], dtype=np.float32)

    def sample(self, batch_size: int):
        """
        Sample a batch of transitions from the rollout buffer.

        Args:
            batch_size: Size of the batch to sample.

        Returns:
            list: List of batches containing sampled transitions.
        """
        # FIXME: Sample randomly from the buffer?
        if batch_size > self.size:
            raise ValueError("Batch size must be smaller than buffer size")
        batch_start = np.arange(0, self.size, batch_size)
        indices = np.arange(self.size, dtype=np.int64)
        np.random.shuffle(indices)
        batch_indices = [indices[i : i + batch_size] for i in batch_start]
        batches = [self._get_batch(batch_index) for batch_index in batch_indices]
        return batches

    def _get_batch(self, indices: tuple) -> tuple:
        """
        Get a batch of transitions from the rollout buffer.

        Args:
            indices: Indices of transitions to include in the batch.

        Returns:
            tuple: Tuple containing batches of states, actions, log probabilities,
                values, and advantages.
        """
        state_batch = []
        action_batch = []
        old_log_prob_batch = []
        value_batch = []
        advantage_batch = []
        for index in indices:
            state_batch.append(self.states[index])
            action_batch.append(self.actions[index])
            old_log_prob_batch.append(self.log_probs[index])
            value_batch.append(self.values[index])
            advantage_batch.append(self.advantages[index])
        return (
            np.array(state_batch),
            np.array(action_batch, dtype=np.int64),
            np.array(old_log_prob_batch),
            np.array(value_batch),
            np.array(advantage_batch),
        )

    def clear(self):
        self.advantages = np.zeros(self.size, dtype=np.float32)
        self.states = []
        self.actions = []
        self.rewards = []
        self.terminated = []
        self.log_probs = []
        self.values = []
