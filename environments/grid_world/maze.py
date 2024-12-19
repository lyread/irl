import gym
from gym import spaces
import numpy as np


class Maze(gym.Env):
    """A Maze Environment class to create a maze with rewards and punishments."""

    def __init__(
        self,
        height=6,
        width=6,
    ):
        """Initialize the MazeEnv with default or given configurations."""
        self.height = height
        self.width = width
        # Set the start, goal, mine and agent coordinates.
        self._start_location = np.array([5, 4])
        self._agent_location = self._start_location
        self._target_location = np.array([2, 1])

        self._maze = np.zeros((self.height, self.width), dtype=bool)
        self._maze[0:2, 1] = True
        self._maze[4:6, 1] = True
        self._maze[2:6, 3] = True

        self.reward_target = 0
        self.reward_barrier = -100
        self.reward_transition = -1

        # Define observation and action spaces
        self.observation_space = spaces.Dict(
            {
                "agent": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.height - 1, self.width - 1]),
                    shape=(2,),
                    dtype=int,
                ),
                "target": spaces.Box(
                    low=np.array([0, 0]),
                    high=np.array([self.height - 1, self.width - 1]),
                    shape=(2,),
                    dtype=int,
                ),
            }
        )
        self.action_space = spaces.Discrete(4)

        # Define action-to-direction mappin
        self._action_to_direction = {
            # up
            0: np.array([-1, 0]),
            # right
            1: np.array([0, 1]),
            # down
            2: np.array([1, 0]),
            # left
            3: np.array([0, -1]),
        }

    def reset(self):
        """Reset the maze with default or given dimensions."""

        self._agent_location = self._start_location
        observation = self._get_obs()

        return observation

    def _get_obs(self):
        return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def step(self, action, render=False):
        """Perform one step in the environment, updating the agent's position."""
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction,
            0,
            np.array([self.height - 1, self.width - 1]),
        )

        # An episode is done if the agent has reached the target
        done = np.array_equal(self._agent_location, self._target_location)

        reward = self.reward_target if done else self.reward_transition
        if self._maze[tuple(self._agent_location)]:
            reward = self.reward_barrier
            self._agent_location = self._start_location

        observation = self._get_obs()["agent"]
        info = self._get_info()

        return observation, reward, done, info
