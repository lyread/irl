import gymnasium as gym
import numpy as np


class FreezeActions(gym.Wrapper):
    def __init__(self, env, mask):
        super().__init__(env)
        self.mask = np.array(mask, dtype=np.float32)
        self.non_zero_indices = np.where(self.mask != 0)[0]

        low = self.env.action_space.low[self.non_zero_indices]
        high = self.env.action_space.high[self.non_zero_indices]
        self.action_space = gym.spaces.Box(low, high, dtype=self.env.action_space.dtype)

    def _mask_obs(self, obs):
        goal_coordinates = obs[: self.task_space_dims]
        agent_joints_pos = obs[self.task_space_dims :]
        agent_joints_pos *= self.mask
        masked_obs = np.concatenate((agent_joints_pos, goal_coordinates), axis=0)
        return masked_obs

    def step(self, action):
        full_action = np.zeros_like(self.mask, dtype=self.env.action_space.dtype)
        full_action[self.non_zero_indices] = action

        obs, reward, term, trunc, info = self.env.step(full_action)
        masked_obs = self._mask_obs(obs)
        return masked_obs, reward, term, trunc, info

    def reset(self, test=False):
        obs = self.env.reset(test=test)
        masked_obs = self._mask_obs(obs)
        return masked_obs
