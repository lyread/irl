from copy import deepcopy

import numpy as np
import pandas as pd
from overrides import overrides
from algorithms.utils import file_manipulation as fm
#import utils.file_manipulation as fm
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from environments.cliff import Cliff
from loggers.logger import Logger


class LoggerCliff(Logger):
    def __init__(
        self,
        env: Cliff,
        algorithm: ReinforcementLearningAlgorithm,
        logdir: str,
        categories: dict[str, list],
        multiple_run_categories: dict[str, list],
    ):
        self.env = env
        self.algorithm = algorithm
        self.logdir = fm.standardize_folder(logdir)
        self.categories = categories
        self.single_run_categories = deepcopy(categories)
        self.multiple_run_categories = multiple_run_categories
        self.matrix_size = self.categories.get(
            "learning_state_visit_matrix"
        ) or self.categories.get("test_state_visit_matrix")
        self.run_counter = 1

        self.info = {
            "environment": env.name,
            "punishment": "p" + str(env.punishment),
            "algorithm": algorithm.name,
            "init_q_method": algorithm.init_method,
            "exploration_method": algorithm.exploration,
        }
        self.suffix = self.info["punishment"]
        if self.info["init_q_method"] is not None:
            self.suffix += "_" + self.info["init_q_method"]
        if self.info["exploration_method"] is not None:
            self.suffix += "_" + self.info["exploration_method"]

    @staticmethod
    def compute_discounted_reward(discount_factor, rewards):
        discounted_rewards = [
            discount_factor**t * reward for t, reward in enumerate(rewards)
        ]
        return sum(discounted_rewards)

    def init_logger(self):
        # call before experiment start
        self.single_run_categories = deepcopy(self.categories)
        if "learning_state_visit_matrix" in self.categories.keys():
            self.single_run_categories["learning_state_visit_matrix"] = np.zeros(
                tuple(self.matrix_size)
            )
        if "test_state_visit_matrix" in self.categories.keys():
            self.single_run_categories["test_state_visit_matrix"] = np.zeros(
                tuple(self.matrix_size)
            )
        self.stored_q_tables = (
            None
            if self.algorithm.init_method is None or self.algorithm.init_method == "nn"
            else {}
        )

    def init_counter(self):
        self.one_step_reward = []
        self.step_counter = 0
        self.episode_trajectory = []
        self.success_counter = 0
        self.collision_counter = 0
        self.cliff_counter = 0

    def _update_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info, learning
    ):
        prefix = "learning_" if learning else "test_"
        if prefix + "rewards" in self.categories.keys():
            self.one_step_reward.append(reward)
        if prefix + "steps" in self.categories.keys():
            self.step_counter += 1
        if prefix + "state_action_pairs" in self.categories.keys():
            if self.env.sequential:
                coordinate = self.env.from_index_to_coordinate(state)
                self.episode_trajectory.append((coordinate, action, reward))
            elif self.env.normalize:
                coordinate = self.env.denormalize_state(state)
                self.episode_trajectory.append((coordinate, action, reward))
            else:
                self.episode_trajectory.append((state, action, reward))

        if prefix + "state_visit_matrix" in self.categories.keys():
            if self.env.sequential:
                coordinate = self.env.from_index_to_coordinate(state)
                self.single_run_categories[prefix + "state_visit_matrix"][
                    tuple(coordinate)
                ] += 1
            elif self.env.normalize:
                coordinate = self.env.denormalize_state(state)
                self.single_run_categories[prefix + "state_visit_matrix"][
                    tuple(coordinate)
                ] += 1
            else:
                self.single_run_categories[prefix + "state_visit_matrix"][
                    tuple(state)
                ] += 1
        if prefix + "success" in self.categories.keys():
            if self.env.reward_target is not None:
                self.success_counter += (
                    1 if terminated and reward == self.env.reward_target else 0
                )
        if prefix + "collision" in self.categories.keys():
            self.collision_counter += 1 if np.all(state == next_state) else 0
        if prefix + "fell_into_cliff" in self.categories.keys():
            if self.env.reward_cliff is not None:
                self.cliff_counter += (
                    1 if terminated and reward == self.env.reward_cliff else 0
                )

    def _update_after_episode(self, learning):
        prefix = "learning_" if learning else "test_"
        if prefix + "rewards" in self.categories.keys():
            self.single_run_categories[prefix + "rewards"].append(
                sum(self.one_step_reward)
            )
        if prefix + "steps" in self.categories.keys():
            self.single_run_categories[prefix + "steps"].append(self.step_counter)
        if prefix + "discounted_rewards" in self.categories.keys():
            self.single_run_categories[prefix + "discounted_rewards"].append(
                self.compute_discounted_reward(
                    self.algorithm.discount_factor, self.one_step_reward
                )
            )
        if prefix + "state_action_pairs" in self.categories.keys():
            self.single_run_categories[prefix + "state_action_pairs"].append(
                self.episode_trajectory
            )
        if prefix + "success" in self.categories.keys():
            if self.env.reward_target is not None:
                self.single_run_categories[prefix + "success"].append(
                    self.success_counter
                )
        if prefix + "fell_into_cliff" in self.categories.keys():
            if self.env.reward_cliff is not None:
                self.single_run_categories[prefix + "fell_into_cliff"].append(
                    self.cliff_counter
                )
        if prefix + "collision" in self.categories.keys():
            self.single_run_categories[prefix + "collision"].append(
                self.collision_counter
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
        if "cumulated_learning_steps" in self.categories.keys():
            self.single_run_categories["cumulated_learning_steps"].append(
                sum(self.single_run_categories["learning_steps"])
            )
        if "cumulated_test_steps" in self.categories.keys():
            self.single_run_categories["cumulated_test_steps"].append(
                sum(self.single_run_categories["test_steps"])
            )
        if self.stored_q_tables is not None:
            q_table = (
                self.env.convert_q_table(self.algorithm.q_table.copy())
                if self.env.sequential
                else self.algorithm.q_table.copy()
            )
            test_counter = len(self.single_run_categories["test_steps"])
            self.stored_q_tables[f"{test_counter}"] = q_table

    def update_multiple_run_log(self, time_stampt=False):
        # call after a whole experiment is terminated
        for category in self.multiple_run_categories:
            if category.endswith("final_q_table"):
                if self.env.sequential:
                    self.multiple_run_categories[category].append(
                        self.env.convert_q_table(self.algorithm.q_table.copy())
                    )
                else:
                    self.multiple_run_categories[category].append(
                        self.algorithm.q_table.copy()
                    )
            else:
                self.multiple_run_categories[category].append(
                    self.single_run_categories[category]
                )
            # save q tables for each run
            if self.stored_q_tables is not None:
                folder_name = f"{self.logdir}logs/q_table/{self.algorithm.name}/"
                fm.standardize_folder(folder_name)
                folder = fm.create_folder(folder_name)
                file_name = fm.create_filename(
                    folder=folder,
                    filename=f"run{self.run_counter}",
                    suffix=self.suffix,
                    file_format="",
                    time_stampt=time_stampt,
                )
                np.savez(file_name, **self.stored_q_tables)
            else:
                self.algorithm.save(
                    file_name=f"run{self.run_counter}",
                    suffix=self.suffix,
                    time_stampt=time_stampt,
                )

        for log_type in ["learning", "test"]:
            self._split_and_export_matrix_log(
                categories=self.single_run_categories,
                log_type=log_type,
                folder_name=f"{self.logdir}logs/multiple_run_{log_type}_log/",
                file_name=self.algorithm.name,
                file_format="csv",
                time_stampt=time_stampt,
                multiple_run=True,
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
            f"{log_type}_success",
            f"{log_type}_fell_into_cliff",
            f"{log_type}_collision",
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

    def _split_and_export_matrix_log(
        self,
        categories,
        log_type,
        folder_name,
        file_name,
        file_format,
        time_stampt,
        multiple_run=False,
    ):
        num_run = f"_run{self.run_counter}" if multiple_run else ""
        if log_type + "_state_action_pairs" in categories.keys():
            fm.export_data(
                categories[f"{log_type}_state_action_pairs"],
                columns=None,
                folder_name=folder_name + "state_action_pairs/",
                suffix=self.suffix,
                file_name=f"{file_name}_state_action_pairs" + num_run,
                file_format=file_format,
                time_stampt=time_stampt,
            )
        if log_type + "_state_visit_matrix" in categories.keys():
            folder_matrix = fm.create_folder(folder_name + "state_visit_matrix/")
            matrix_fn = fm.create_filename(
                folder=folder_matrix,
                filename=f"{file_name}_state_visit_matrix" + num_run,
                suffix=self.suffix,
                file_format="",
                time_stampt=time_stampt,
            )
            np.save(matrix_fn, categories[f"{log_type}_state_visit_matrix"])

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

            self._split_and_export_matrix_log(
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
        # TODO: add plot data for single runs
        if self.run_counter == 1:
            return
        plot_data = []
        metrics = []
        needed_data = {
            "learning_steps": "Learning Steps",
            "learning_rewards": "Learning Cumulated Rewards",
            "learning_discounted_rewards": "Learning Discounted Cumulated Rewards",
            "learning_success": "Learning Success",
            "learning_collision": "Learning Collision",
            "learning_fell_into_cliff": "Learning Fell Into Cliff",
            "learning_state_visit_matrix": "Learning State Visit Matrix",
            "test_steps": "Test Steps",
            "test_rewards": "Test Cumulated Rewards",
            "test_discounted_rewards": "Test Discounted Cumulated Rewards",
            "test_success": "Test Success",
            "test_collision": "Test Collision",
            "test_fell_into_cliff": "Test Fell Into Cliff",
            "test_state_visit_matrix": "Test State Visit Matrix",
            "cumulated_learning_steps": "Cumulated Training Steps",
            "cumulated_test_steps": "Cumulated Test Steps",
            "final_q_table": "Final Q Tables",
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
