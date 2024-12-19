from collections import defaultdict

import numpy as np
from overrides import overrides

import algorithms.utils.file_manipulation as fm
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from loggers.logger import Logger


class BasicLogger(Logger):
    def __init__(
        self,
        algorithm: ReinforcementLearningAlgorithm,
        logdir: str = "tmp",
    ):
        self.algorithm = algorithm
        self.logdir = fm.standardize_folder(logdir)
        self.single_run_categories = defaultdict(list)
        self.multiple_run_categories = defaultdict(list)
        self.run_counter = 1
        self.info = {
            "algorithm": algorithm.name,
            "exploration_method": algorithm.exploration,
        }
        self.suffix = ""

    @staticmethod
    def compute_discounted_reward(discount_factor, rewards):
        discounted_rewards = [
            discount_factor**t * reward for t, reward in enumerate(rewards)
        ]
        return sum(discounted_rewards)

    def init_logger(self):
        # call before experiment start
        self.single_run_categories = defaultdict(list)

    def init_counter(self):
        self.one_step_reward = []
        self.step_counter = 0
        self.episode_trajectory = []

    def _update_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info, learning
    ):
        self.one_step_reward.append(reward)
        self.step_counter += 1
        self.episode_trajectory.append((state, action, reward))

    def _update_after_episode(self, learning):
        prefix = "learning_" if learning else "test_"
        self.single_run_categories[prefix + "rewards"].append(sum(self.one_step_reward))
        self.single_run_categories[prefix + "steps"].append(self.step_counter)
        self.single_run_categories[prefix + "discounted_rewards"].append(
            self.compute_discounted_reward(
                self.algorithm.discount_factor, self.one_step_reward
            )
        )
        self.single_run_categories[prefix + "state_action_pairs"].append(
            self.episode_trajectory
        )

    @overrides
    def update_learning_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        self._update_during_episode(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
            info,
            learning=True,
        )

    def update_learning_log_after_episode(self):
        self._update_after_episode(learning=True)

    @overrides
    def update_test_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        self._update_during_episode(
            state,
            action,
            reward,
            next_state,
            terminated,
            truncated,
            info,
            learning=False,
        )

    def update_test_log_after_episode(self):
        self._update_after_episode(learning=False)
        self.single_run_categories["cumulated_learning_steps"].append(
            sum(self.single_run_categories["learning_steps"])
        )
        self.single_run_categories["cumulated_test_steps"].append(
            sum(self.single_run_categories["test_steps"])
        )

    def update_multiple_run_log(self, time_stampt=False):
        # call after a whole experiment is terminated
        for category in self.single_run_categories.keys():
            self.multiple_run_categories[category].append(
                self.single_run_categories[category]
            )
            self.algorithm.save(
                file_name=f"run{self.run_counter}",
                suffix=self.suffix,
                time_stampt=time_stampt,
            )

        self.run_counter += 1

    def _split_and_export_numerical_log(
        self,
        categories,
        log_type,
        folder_name,
        file_name,
        file_format,
        time_stampt,
        multiple_run,
    ):
        category_names = [
            f"{log_type}_rewards",
            f"{log_type}_discounted_rewards",
            f"{log_type}_steps",
        ]
        if log_type == "test":
            category_names.extend(["cumulated_learning_steps", "cumulated_test_steps"])

        log = {
            key: np.mean(value, axis=0) if multiple_run else value
            for key, value in categories.items()
            if key in category_names
        }

        fm.export_data(
            log,
            columns=None,
            folder_name=folder_name,
            suffix=self.suffix,
            file_name=file_name,
            file_format=file_format,
            time_stampt=time_stampt,
        )

    def export_single_run_data(self, time_stampt=False):
        for log_type in ["learning", "test"]:
            self._split_and_export_numerical_log(
                categories=self.single_run_categories,
                log_type=log_type,
                folder_name=f"{self.logdir}logs/{log_type}_log/",
                file_name=self.algorithm.name,
                file_format="csv",
                time_stampt=time_stampt,
                multiple_run=False,
            )

            self.algorithm.save(
                file_name=self.algorithm.name,
                suffix=self.suffix,
                time_stampt=False,
            )

    def export_multiple_run_data(self, time_stampt=False):
        for log_type in ["learning", "test"]:
            self._split_and_export_numerical_log(
                categories=self.multiple_run_categories,
                log_type=log_type,
                folder_name=f"{self.logdir}logs/multiple_run_{log_type}_log/",
                file_name=self.algorithm.name,
                file_format="csv",
                time_stampt=time_stampt,
                multiple_run=True,
            )

    def export_plot_data(self, time_stampt=False):
        plot_data = []
        metrics = []
        needed_data = {
            "learning_steps": "Learning Steps",
            "learning_rewards": "Learning Cumulated Rewards",
            "learning_discounted_rewards": "Learning Discounted Cumulated Rewards",
            "test_steps": "Test Steps",
            "test_rewards": "Test Cumulated Rewards",
            "test_discounted_rewards": "Test Discounted Cumulated Rewards",
            "cumulated_learning_steps": "Cumulated Learning Steps",
            "cumulated_test_steps": "Cumulated Test Steps",
        }
        for data in needed_data:
            if data in self.multiple_run_categories.keys():
                plot_data.append(self.multiple_run_categories[data])
                metrics.append(needed_data[data])
        data = zip(*plot_data)
        fm.export_data(
            data,
            columns=metrics,
            folder_name=f"{self.logdir}logs/plot_data/",
            suffix=self.suffix,
            file_name=self.algorithm.name,
            file_format="pkl",
            time_stampt=time_stampt,
        )
