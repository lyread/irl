import gymnasium as gym
from tqdm import tqdm

from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from loggers.basic_logger import BasicLogger
from loggers.logger_robotarm import LoggerRobotArm


class Agent:
    """
    An agent that interacts with an environment using a reinforcement learning algorithm.

    Args:
        env (gym.Env): The environment to interact with.
        algorithm (ReinforcementLearningAlgorithm): The reinforcement learning algorithm used by the agent.
        logger (Logger, optional): The logger to record training and testing information.
            Defaults to None, in which case a BasicLogger is used.
        max_episode_length (int, optional): Maximum length of an episode. Defaults to 30.
        test_every (int, optional): Frequency of testing, specified in terms of episodes. Defaults to 1.
    """

    def __init__(
        self,
        env: gym.Env,
        algorithm: ReinforcementLearningAlgorithm,
        logger: BasicLogger | LoggerRobotArm | None = None,
        max_episode_length: int = 30,
        test_every: int = 1,
    ) -> None:
        self.env = env
        self.algorithm = algorithm
        self.logger = logger if logger else BasicLogger(self.algorithm)
        self.max_episode_length = max_episode_length
        self.test_every = test_every

    def _stopping_criteria(
        self, counter: int, terminated: bool = False, truncated: bool = False
    ) -> bool:
        """
        Training episodes are stopped when target reached or the maximum episode length is reached, or the episode is truncated.

        Args:
            counter (int): Current step counter of the episode.
            terminated (bool): Terminated signal returned by step function.
            truncated (bool): Truncated signal returned by step function.

        Returns:
            bool: True if the episode should stop, False otherwise.
        """

        return (
            False
            if (not terminated and counter < self.max_episode_length and not truncated)
            else True
        )

    def _testing_criteria(self, counter: int) -> bool:
        """
        Determine whether to test or not based on the set self.test_every

        Args:
            counter (int): Current episode counter.

        Returns:
            bool: True if testing should be performed, False otherwise.
        """

        return True if (counter % self.test_every == 0) else False

    def train(self, num_episodes: int) -> None:
        """
        Train the agent for a specified number of episodes.

        Args:
            num_episodes (int): Number of episodes to train the agent.
        """

        for episode in tqdm(range(num_episodes), desc="Training", unit="episode"):
            self.logger.init_counter()
            stop = False
            test = self._testing_criteria(episode)
            # Get initial state
            state_current = self.env.reset()
            action_current = self.algorithm.select_action(
                state_current, method=self.algorithm.exploration
            )
            while not stop:
                state_next, reward, terminated, truncated, info = self.env.step(
                    action_current
                )
                action_next = self.algorithm.select_action(
                    state_next, method=self.algorithm.exploration
                )
                self.algorithm.learn(
                    state=state_current,
                    action=action_current,
                    next_state=state_next,
                    next_action=action_next,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                self.logger.update_learning_log_during_episode(
                    state=state_current,
                    action=action_current,
                    next_state=state_next,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                action_current = action_next
                state_current = state_next
                stop = self._stopping_criteria(
                    self.logger.step_counter, terminated, truncated
                )

            self.logger.update_learning_log_after_episode()
            if test:
                self._test()
                if self.algorithm.tensorboard:
                    self.algorithm.writer.add_scalar(
                        "test_steps",
                        self.logger.single_run_categories["test_steps"][-1],
                        episode,
                    )
                    self.algorithm.writer.add_scalar(
                        "test_rewards",
                        self.logger.single_run_categories["test_rewards"][-1],
                        episode,
                    )

    def _test(self) -> None:
        """
        Perform testing of the agent's performance using the greedy policy.
        """
        state_current = self.env.reset()
        self.logger.init_counter()
        stop = False
        while not stop:
            action = self.algorithm.select_action(state_current, method="greedy")
            state_next, reward, terminated, truncated, info = self.env.step(action)
            self.logger.update_test_log_during_episode(
                state=state_current,
                action=action,
                next_state=state_next,
                reward=reward,
                terminated=terminated,
                truncated=truncated,
                info=info,
            )
            state_current = state_next
            stop = self._stopping_criteria(
                self.logger.step_counter, terminated, truncated
            )
        self.logger.update_test_log_after_episode()
