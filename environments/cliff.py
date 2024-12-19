import gymnasium as gym
import numpy as np
from gymnasium import spaces


class CliffActionWrapper(gym.ActionWrapper):
    def __init__(self, env, low=-1, high=1):
        super().__init__(env)
        self.low = low
        self.high = high
        self.num_actions = self.env.action_space.n

    def action(self, action) -> int:
        action = np.clip(action, self.low, self.high)
        boundaries = np.linspace(self.low, self.high, num=self.num_actions + 1)
        if action == self.high:
            return self.num_actions - 1
        for i, boundary in enumerate(boundaries[1::]):
            if action < boundary:
                return i


class Cliff(gym.Env):
    """
    A grid world: Cliff Walking as described in
    Sutton, R. S., & Barto, A. G. (2018).
    Reinforcement Learning: An Introduction (2nd ed.). The MIT Press.
    https://mitpress.mit.edu/books/reinforcement-learning-second-edition
    """

    def __init__(
        self,
        height: int = 4,
        width: int = 12,
        punishment: bool = False,
        normalize: bool = False,
        sequential: bool = False,
    ):
        """
        Initializing the cliff environment

        Parameters:
        - height (int): Number of possible y positions
        - width (int): Number of posible x positions
        - punishment (bool): Should the agent be punished for taking a step?
        - normalize (bool): Should the states be normalized? If yes the states will be between [-1,-1] and [1,1]
        """
        self.height = height
        self.width = width
        self.name = "Cliff"
        self.num_states = self.height * self.width
        self._start_location = np.array([self.height - 1, 0])
        self._agent_location = self._start_location
        self._target_location = np.array([self.height - 1, self.width - 1])

        # Cliff Location
        self._cliff = np.zeros((self.height, self.width), dtype=bool)
        self._cliff[self.height - 1, 1:-1] = True
        # Rewards
        self.punishment = punishment
        self.punish_value = -1
        self.reward_target = 1
        self.reward_cliff = -100
        self.reward_transition = 0
        if self.punishment:
            self.reward_target += self.punish_value
            self.reward_transition += self.punish_value
        # should the environment normalize outputs?
        self.normalize = normalize
        # should the state be returned in sequential form?
        self.sequential = sequential
        if self.normalize and self.sequential:
            raise ValueError(
                "Normalization and sequential representation cannot be used at the same time"
            )
        # Observations are dictionaries with the agent's and the target's loc.
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

        # We have 4 actions, corresponding to "right", "up", "left", "down"
        self.action_space = spaces.Discrete(4)

        """
        The following dictionary maps abstract actions from `self.action_space`
        to the direction we will walk in if that action is taken.
        I.e., 0 corresponds to "right", 1 to "up" etc.
        """
        self._action_to_direction = {
            0: np.array([-1, 0]),  # Up
            1: np.array([0, 1]),  # Right
            2: np.array([1, 0]),  # Down
            3: np.array([0, -1]),  # Left
        }

    def _get_obs(self):
        if self.normalize:
            return {
                "agent": self.normalize_state(self._agent_location),
                "target": self.normalize_state(self._target_location),
            }
        elif self.sequential:
            return {
                "agent": self.from_coordinate_to_index(self._agent_location),
                "target": self.from_coordinate_to_index(self._target_location),
            }
        else:
            return {"agent": self._agent_location, "target": self._target_location}

    def _get_info(self):
        return {
            "distance": np.linalg.norm(
                self._agent_location - self._target_location, ord=1
            )
        }

    def normalize_state(self, state):
        return np.array(
            [
                (2 * state[0] / (self.height - 1)) - 1,
                (2 * state[1] / (self.width - 1)) - 1,
            ]
        )

    def denormalize_state(self, state):
        return np.array(
            [
                ((state[0] + 1) * (self.height - 1)) / 2,
                ((state[1] + 1) * (self.width - 1)) / 2,
            ],
            dtype=np.int64,
        )

    def from_coordinate_to_index(self, coordinate):
        return [coordinate[0] * self.width + coordinate[1]]

    def from_index_to_coordinate(self, index):
        return np.array([index[0] // self.width, index[0] % self.width])

    def convert_q_table(self, q_table):
        num_states = self.height * self.width
        ret_q_table = np.zeros((self.height, self.width, self.action_space.n))
        for i in range(num_states):
            coordinate = self.from_index_to_coordinate([i])
            ret_q_table[coordinate[0], coordinate[1]] = q_table[i]
        return ret_q_table

    """
    reset: is called at the beginning of every episode.
    It's possible to force the initial state.
    """

    def reset(self, state=None, seed=None, return_info=False, options=None):
        self._state = state
        self._agent_location = state

        if state is None:
            state = self._start_location
            self._agent_location = self._start_location

        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation["agent"]

    def valid_move(self, state, action):
        """
        Check if a move is valid given a state and action.

        Parameters:
        - state (tuple): Current state of the agent.
        - action (int): Action to be taken in the current state.

        Returns:
        - bool: True if the move is valid, False otherwise.
        """
        movement = self._action_to_direction[action]
        new_position = state + movement
        # Use np.clip() to avoid stepping out of the grid
        clipped_position = np.clip(
            new_position, np.array([0, 0]), np.array([self.height - 1, self.width - 1])
        )

        # If clipping occurred, the move was not valid
        if np.any(new_position != clipped_position):
            return False
        else:
            return True

    def valid_actions(self, state):
        """
        Get a list of valid actions from the current state.

        Args:
            state (tuple): Current state of the agent.

        Returns:
            list: List of valid action indices.
        """

        return [
            action
            for action in range(self.action_space.n)
            if self.valid_move(state, action)
        ]

    """
    step: specifies the transition function of the environment,
    computes the reward, and signal absorbing states, i.e. states where
    every action keeps you in the same state
    """

    def step(self, action):
        # Map the action (element of {0,1,2,3}) to the direction we walk in
        direction = self._action_to_direction[action]
        # We use `np.clip` to make sure we don't leave the grid
        self._agent_location = np.clip(
            self._agent_location + direction,
            0,
            np.array([self.height - 1, self.width - 1]),
        )
        # An episode is terminated if the agent has reached the target
        truncated = False
        terminated = np.array_equal(self._agent_location, self._target_location)
        reward = self.reward_target if terminated else self.reward_transition
        if self._cliff[tuple(self._agent_location)]:
            reward = self.reward_cliff
            terminated = True
        observation = self._get_obs()["agent"]
        info = self._get_info()

        return observation, reward, terminated, truncated, info


# if __name__ == "__main__":
#     env = Cliff()
#     state_norm = env.normalize_state([0, 2])
#     print(env.denormalize_state(state_norm))
