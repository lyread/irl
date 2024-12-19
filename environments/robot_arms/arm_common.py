from abc import ABC, abstractmethod

import gymnasium as gym
import numpy as np
import pandas as pd
from gymnasium import spaces
from sklearn.utils import shuffle


class ArmCommon(gym.Env, ABC):
    """
    Abstract base class for arm-related Gym environments.

    Args:
        name (str): Name of the environment.
        joint_ranges: Range of joint angles for the arm.
        link_lengths: Lengths of arm links.
        task_space_dims (int, optional): Dimensionality of task space. Defaults to 3.
        goal_zone_radius (int, optional): Radius of the goal zone. Defaults to 50.
        max_step_length (float, optional): Maximum length of each step. Defaults to np.pi / 10.
        dataset_path (str, optional): Path to a dataset. Defaults to None.
        training_episodes (int, optional): Number of training episodes. Defaults to 100.
        test_episodes (int, optional): Number of test episodes. Defaults to 100.
        rescale_state (bool, optional): Whether to rescale state. Defaults to False.
        shuffle_mode (str, optional): Mode for shuffling data. Defaults to "keep_pairings".
    """

    def __init__(
        self,
        name: str,
        joint_ranges,
        link_lengths,
        task_space_dims=3,
        goal_zone_radius=50,
        max_step_length=np.pi / 10,
        dataset_path=None,
        training_episodes=100,
        test_episodes=100,
        rescale_state=False,
        shuffle_mode="keep_pairings",
        max_punishment=1,
        safety_margins=None,
    ):
        # The robot
        self.name = name
        self.joint_ranges = np.array(joint_ranges)
        self.link_lengths = np.array(link_lengths)
        self.dof = self.joint_ranges.shape[0]

        # gym environment attributes
        obs_shape = (self.dof + task_space_dims,)
        obs_low = np.min(joint_ranges)
        obs_high = np.max(joint_ranges)
        self.observation_space = spaces.Box(
            low=obs_low,
            high=obs_high,
            shape=obs_shape,
            dtype=np.float32,
        )
        self.action_space = spaces.Box(
            low=-max_step_length,
            high=max_step_length,
            shape=(self.dof,),
            dtype=np.float32,
        )
        self.reward_range = (0, 1)
        # The task
        self.task_space_dims = task_space_dims
        self.goal_zone_radius = goal_zone_radius
        self.max_step_length = max_step_length
        self.dataset_path = dataset_path
        self.num_training_episodes = training_episodes
        self.num_test_episodes = test_episodes
        self.rescale_state = rescale_state
        self.shuffle_mode = shuffle_mode
        self.max_punishment = max_punishment

        if safety_margins is None:
            safety_margins = np.array([0.1] * self.dof)
        self.safety_margins = safety_margins

        # Counter for load initial positions
        self.n_training = 0
        self.n_test = 0
        self.reward = 1

        # Load dataset or create a data set if dataset_path is not given
        if dataset_path:
            self.dataset = pd.read_pickle(dataset_path)
            self.num_training_episodes = self.num_test_episodes = len(self.dataset) // 2
        else:
            self.dataset = self.create_dataset()
        # Split dataset into training and test set
        (
            self.training_agent_pos,
            self.training_goal_pos,
            self.test_agent_pos,
            self.test_goal_pos,
        ) = self._split_dataset()
        # if scale state in range [-1, 1], get the max and min values of joint angles and coordinates
        if self.rescale_state:
            (
                self.max_joint_scale,
                self.min_joint_scale,
                self.max_coordinate_scale,
                self.min_coordinate_scale,
            ) = self._get_scale_range()
        # Initialize agent and goal position
        self.agent_position = self.training_agent_pos[self.n_training]
        self.goal_position = self.training_goal_pos[self.n_training]
        self.distance = self.Euclidean_distance(self.agent_position, self.goal_position)
        self.last_agent_position = self.agent_position

    def _get_scale_range(self):
        """
        Calculate the scale range for rescaling the state base on the dataset.
        """
        agent_pos = self.dataset["agent"].to_numpy()
        goal_pos = self.dataset["goal"].to_numpy()
        all_pos = np.concatenate((agent_pos, goal_pos), axis=0)
        max_joint_scale = np.max(
            [pos[self.task_space_dims :] for pos in all_pos], axis=0
        )
        min_joint_scale = np.min(
            [pos[self.task_space_dims :] for pos in all_pos], axis=0
        )
        max_coordinate_scale = np.max(
            [pos[: self.task_space_dims] for pos in all_pos], axis=0
        )
        min_coordinate_scale = np.min(
            [pos[: self.task_space_dims] for pos in all_pos], axis=0
        )
        return (
            max_joint_scale,
            min_joint_scale,
            max_coordinate_scale,
            min_coordinate_scale,
        )

    @staticmethod
    def scale_vector(vector, min_value, max_value, back=False):
        if not back:
            return 2 * (vector - min_value) / (max_value - min_value) - 1
        else:
            return (vector + 1) * (max_value - min_value) / 2 + min_value

    def _rescale_state(self, state, back=False):
        """
        Rescale state to range [-1, 1].

        Args:
            state: The state to be rescaled.
            back (bool, optional): Whether to unscale the state. Defaults to False.

        Returns:
            np.ndarray: Rescaled or unscaled state.
        """
        # Rescale joint positions
        goal_coordinates = state[self.dof :]
        joint_pos = state[: self.dof]
        if back:
            joint_pos = self.scale_vector(
                joint_pos, self.min_joint_scale, self.max_joint_scale, back=True
            )
            goal_coordinates = self.scale_vector(
                goal_coordinates,
                self.min_coordinate_scale,
                self.max_coordinate_scale,
                back=True,
            )
        else:
            joint_pos = self.scale_vector(
                joint_pos, self.min_joint_scale, self.max_joint_scale
            )
            goal_coordinates = self.scale_vector(
                goal_coordinates, self.min_coordinate_scale, self.max_coordinate_scale
            )
        return np.concatenate((joint_pos, goal_coordinates), axis=0)

    def _split_dataset(self):
        """
        Split the dataset into training and test sets.
        """
        training_set = self.dataset.loc[: self.num_training_episodes - 1]
        test_set = self.dataset.loc[
            self.num_training_episodes : self.num_training_episodes
            + self.num_test_episodes
            - 1
        ]
        training_agent_pos = training_set["agent"].to_numpy()
        training_goal_pos = training_set["goal"].to_numpy()
        test_agent_pos = test_set["agent"].to_numpy()
        test_goal_pos = test_set["goal"].to_numpy()
        if self.shuffle_mode != "keep_pairings":
            np.random.shuffle(training_agent_pos)
            np.random.shuffle(training_goal_pos)
            np.random.shuffle(test_agent_pos)
            np.random.shuffle(test_goal_pos)

        return training_agent_pos, training_goal_pos, test_agent_pos, test_goal_pos

    def create_init_positions(self, n):
        """
        Creates n initial positions for the agent and goal. Makes sure, that
        the target is not reached directly by these configurations.

        Args:
            n (int): Number of initial positions.

        Returns:
            np.ndarray: Array of initial agent positions.
            np.ndarray: Array of initial goal positions.
            np.ndarray: Array of distances.
        """

        agents = np.zeros((n, self.task_space_dims + self.dof))
        goals = np.zeros((n, self.task_space_dims + self.dof))
        distances = np.zeros(n)
        for i in range(n):
            # I actually want a Do-while loop, but python does not provide it :'(
            agent_position = self.get_random_position()
            goal_position = self.get_random_position()
            distance = self.Euclidean_distance(agent_position, goal_position)
            while self.is_target_reached(distance):
                agent_position = self.get_random_position()
                distance = self.Euclidean_distance(agent_position, goal_position)
            if n == 1:
                return agent_position, goal_position
            agents[i] = agent_position
            goals[i] = goal_position
            distances[i] = distance
        return agents, goals, distances

    def create_dataset(self, save: bool = False, path: str = ""):
        agents, goals, distances = self.create_init_positions(
            self.num_training_episodes + self.num_test_episodes
        )
        data = zip(agents, goals, distances)
        dataset = pd.DataFrame(data, columns=["agent", "goal", "distance"])
        if save:
            dataset.to_pickle(path)
        return dataset

    def get_random_position(self):
        random_increment = (
            self.joint_ranges[:, 1] - self.joint_ranges[:, 0]
        ) * np.random.random(self.dof)
        q = self.joint_ranges[:, 0] + random_increment

        angles = self.validate_angles(q)
        end_effector_coordinates = self.end_effector_coordinates(angles)

        return np.concatenate((end_effector_coordinates, angles), axis=0)

    def validate_angles(self, angles):
        """
        Checks that the angles given in parameters are within the specified
        ranges. If not, we give them their min/max value
        """
        corrected_angles = np.clip(
            angles, self.joint_ranges[:, 0], self.joint_ranges[:, 1]
        )
        return corrected_angles

    @abstractmethod
    def end_effector_coordinates(self, angles):
        """
        Returns the coordinates of O8 in base coordinates (R0) using the given angles.
        Parameters:
            angles -- the angles to use for the transformations, cannot be
                      outside the specified ranges
            returns -- the position of O8 (Origin of the End Effector) in base coordinates (R0)
        """
        # Overwrite in Child Classes !
        end_effector_coordinates = None
        return end_effector_coordinates

    def epoch_reset(self, shuffle_data=False):
        """
        Reset the environment at the start of a new epoch.

        Args:
            shuffle_data (bool, optional): Whether to shuffle the data. Defaults to False.
        """
        if shuffle_data:
            self.dataset = shuffle(self.dataset).reset_index(drop=True)
            (
                self.training_agent_pos,
                self.training_goal_pos,
                self.test_agent_pos,
                self.test_goal_pos,
            ) = self._split_dataset()
        # reset episode counter
        self.n_training = 0
        self.n_test = 0

    def reset(self, return_info=False, options=None, test: bool = False):
        if self.dataset_path:
            if test:
                self.agent_position = self.test_agent_pos[self.n_test]
                self.goal_position = self.test_goal_pos[self.n_test]
                self.n_test += 1
            else:
                self.agent_position = self.training_agent_pos[self.n_training]
                self.goal_position = self.training_goal_pos[self.n_training]
                self.n_training += 1
        else:
            # TODO: if use dataset ?
            self.agent_position, self.goal_position = self.create_init_positions(1)
        self.distance = self.Euclidean_distance(self.agent_position, self.goal_position)
        observation = self._get_obs()
        info = self._get_info()
        return (observation, info) if return_info else observation

    def undo(self):
        """
        Undoes the last action. This is only used for some interactive
        reinforcement learning implementations.
        """
        self.agent_position = self.last_agent_position
        self.distance = self.Euclidean_distance(self.agent_position, self.goal_position)

    def Euclidean_distance(self, agent, goal, space="task"):
        if space == "task":
            goal = goal[0 : self.task_space_dims]  # baseCoordinates
            agent = agent[0 : self.task_space_dims]
        else:
            goal = goal[self.task_space_dims :]  # joint space
            agent = agent[self.task_space_dims :]
        return np.linalg.norm(goal - agent)

    def is_target_reached(self, distance):
        return distance < self.goal_zone_radius

    def _get_obs(self):
        goal_coordinates = self.goal_position[: self.task_space_dims]
        agent_joints_pos = self.agent_position[self.task_space_dims :]
        obs = np.concatenate((agent_joints_pos, goal_coordinates), axis=0)
        if self.rescale_state:
            return self._rescale_state(obs)
        return obs

    def _get_info(self):
        return {
            "distance": self.Euclidean_distance(self.agent_position, self.goal_position)
        }

    def _get_punishment(self, joint_values):
        rt = np.zeros_like(joint_values, dtype=float)

        condition1 = (self.joint_ranges[:, 0] + self.safety_margins < joint_values) & (
            joint_values < self.joint_ranges[:, 1] - self.safety_margins
        )
        condition2 = self.joint_ranges[:, 0] >= self.safety_margins + joint_values
        condition3 = joint_values >= self.joint_ranges[:, 1] - self.safety_margins

        rt[condition1] = 0
        rt[condition2] = np.exp(
            -0.5
            * (
                (joint_values[condition2] - self.joint_ranges[condition2, 0])
                / self.safety_margins[condition2]
            )
            ** 2
        )
        rt[condition3] = np.exp(
            -0.5
            * (
                (joint_values[condition3] - self.joint_ranges[condition3, 1])
                / self.safety_margins[condition3]
            )
            ** 2
        )

        rt = -self.max_punishment / self.dof * rt

        return rt.sum()

    def _get_reward(self):
        return self.reward if self.is_target_reached(self.distance) else 0

    def step(self, action):
        self.last_agent_position = self.agent_position

        # calculate new positions
        joint_values = self.agent_position[self.task_space_dims :]
        new_joint_values = joint_values + (action * self.max_step_length)
        new_joint_values = self.validate_angles(new_joint_values)
        new_position = self.end_effector_coordinates(new_joint_values)
        self.agent_position = np.concatenate((new_position, new_joint_values), axis=0)
        self.distance = self.Euclidean_distance(self.agent_position, self.goal_position)

        observation = self._get_obs()
        pos_reward = self._get_reward()
        punishment = self._get_punishment(new_joint_values)
        terminated = self.is_target_reached(self.distance)
        truncated = False
        info = self._get_info()

        reward = np.array([pos_reward, punishment])

        return observation, reward, terminated, truncated, info
