import random

import numpy as np
from tqdm import tqdm

from algorithms.agent import Agent
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from environments.robot_arms.arm_common import ArmCommon
from loggers.logger_robotarm import LoggerRobotArm


class InteractiveAgent(Agent):
    logger: LoggerRobotArm

    def __init__(
        self,
        env: ArmCommon,
        algorithm: ReinforcementLearningAlgorithm,
        logger: LoggerRobotArm,
        num_epochs: int = 50,
        max_episode_length: int = 60,
        ask_likelihood: float = 0.4,
    ):
        super().__init__(
            env=env,
            algorithm=algorithm,
            logger=logger,
            max_episode_length=max_episode_length,
        )
        self.num_epochs = num_epochs
        self.ask_likelihood = ask_likelihood
        print(f"Ask likelihood: {self.ask_likelihood}")
        print(f"env.name: {self.env.name}")

    def _ask_teacher(self):
        """
        Determine whether to ask the teacher base on ask likelihood.

        Returns:
            bool: True if asking the teacher, False otherwise.
        """
        return random.random() < self.ask_likelihood

    def _is_good_action(self, old_distance, new_distance):
        """
        Check if the new action leads to a better outcome based on distances.

        Args:
            old_distance: Distance before taking the action.
            new_distance: Distance after taking the action.

        Returns:
            bool: True if the action leads to a better outcome, False otherwise.
        """
        return old_distance > new_distance

    def train(self, num_episodes: int) -> None:
        for epoch in tqdm(
            range(self.num_epochs), desc="Epochs", unit="epoch", colour="yellow"
        ):
            self.logger.epoch_reset()
            for _ in tqdm(
                range(self.env.num_training_episodes),
                desc=f"Epoch{epoch}: Training",
                unit="episode",
                colour="red",
            ):
                self.logger.init_counter()
                stop = False
                state_current = self.env.reset()
                while not stop:
                    good_action = True
                    old_distance = self.env.distance
                    action = self.algorithm.select_action(
                        state_current, method=self.algorithm.exploration
                    )
                    state_next, reward, terminated, truncated, info = self.env.step(
                        action
                    )
                    new_distance = self.env.distance
                    if self._ask_teacher():
                        good_action = self._is_good_action(old_distance, new_distance)
                    if good_action:
                        self.algorithm.learn(
                            state=state_current,
                            action=action,
                            next_state=state_next,
                            next_action=None,  # we don't need this for this algorithm
                            reward=reward,
                            terminated=terminated,
                            truncated=truncated,
                            info=info,
                        )
                        info["undo"] = False
                    else:  # if the action is not good, undo the action reset next state to current state
                        self.env.undo()
                        state_next = state_current
                        info["undo"] = True
                    self.logger.update_learning_log_during_episode(
                        state=state_current,
                        action=action,
                        next_state=state_next,
                        reward=reward,
                        terminated=terminated,
                        truncated=truncated,
                        info=info,
                    )
                    state_current = state_next
                    stop = self._stopping_criteria(self.logger.step_counter, terminated)
                self.logger.update_learning_log_after_episode()
            # Test algorithm performance using greedy exploration
            self._test(self.env.num_test_episodes, epoch)
            self.logger.update_epoch_log(epoch)
            self.env.epoch_reset()

    def _test(self, num_episodes: int, epoch: int) -> None:
        for _ in tqdm(
            range(num_episodes),
            desc=f"Epoch{epoch}: Test",
            unit="episode",
            colour="green",
        ):
            state_current = self.env.reset(test=True)
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
                stop = self._stopping_criteria(self.logger.step_counter, terminated)
            self.logger.update_test_log_after_episode()
        if self.algorithm.tensorboard:
            success_rate = np.mean(self.logger.test_episode_categories["successes"])
            failure_rate = 1 - success_rate
            self.algorithm.writer.add_scalar("Success Rate", success_rate, epoch)
            self.algorithm.writer.add_scalar("Failure Rate", failure_rate, epoch)
