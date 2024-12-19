from collections import defaultdict

import numpy as np

import algorithms.utils.file_manipulation as fm
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm as RLAlgorithm
from environments.robot_arms.arm_common import ArmCommon
from loggers.logger import Logger


class LoggerRobotArm(Logger):
    def __init__(
        self,
        env: ArmCommon,
        algorithm: RLAlgorithm,
        logdir: str = "tmp/",
        **kwargs,
    ):
        self.env = env
        self.algorithm = algorithm
        self.logdir = logdir
        self.suffix = kwargs.get("suffix", "")
        self.run_counter = 0
        self.cumulated_training_steps = 0
        self.cumulated_test_steps = 0
        self.multiple_run_training_categories = defaultdict(list)
        self.multiple_run_test_categories = defaultdict(list)

    def init_logger(self):
        # I added this to make sure the steps are not cumulated across runs
        self.cumulated_training_steps = 0
        self.cumulated_test_steps = 0

        self.epoch_training_categories = defaultdict(list)
        self.epoch_test_categories = defaultdict(list)

    def epoch_reset(self):
        self.training_episode_categories = defaultdict(list)
        self.test_episode_categories = defaultdict(list)

    def init_counter(self):
        self.one_step_reward = []
        # self.episode_trajectory = []
        self.step_counter = 0
        self.undo = 0
        self.success_counter = 0

    def update_learning_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        # self.episode_trajectory.append((state, action, reward, info["distance"]))
        self.one_step_reward.append(reward)
        self.step_counter += 1
        if info["undo"]:
            self.undo += 1
        if terminated:
            self.success_counter += 1

    def update_learning_log_after_episode(self):
        # self.training_episode_categories["trajectory"].append(self.episode_trajectory)
        self.training_episode_categories["steps"].append(self.step_counter)
        self.training_episode_categories["undos"].append(self.undo)
        self.training_episode_categories["successes"].append(self.success_counter)

        pos_rewards = sum([r[0] for r in self.one_step_reward])
        neg_rewards = sum([r[1] for r in self.one_step_reward])
        self.training_episode_categories["pos_rewards"].append(pos_rewards)
        self.training_episode_categories["neg_rewards"].append(neg_rewards)
        self.training_episode_categories["rewards"].append(pos_rewards + neg_rewards)

    def update_test_log_during_episode(
        self, state, action, reward, next_state, terminated, truncated, info
    ):
        # self.episode_trajectory.append((state, action, reward, info["distance"]))
        self.one_step_reward.append(reward)
        self.step_counter += 1
        if terminated:
            self.success_counter += 1

    def update_test_log_after_episode(self):
        # self.test_episode_categories["trajectory"].append(self.episode_trajectory)
        self.test_episode_categories["steps"].append(self.step_counter)
        self.test_episode_categories["successes"].append(self.success_counter)

        pos_rewards = sum([r[0] for r in self.one_step_reward])
        neg_rewards = sum([r[1] for r in self.one_step_reward])
        self.test_episode_categories["pos_rewards"].append(pos_rewards)
        self.test_episode_categories["neg_rewards"].append(neg_rewards)
        self.test_episode_categories["rewards"].append(pos_rewards + neg_rewards)

    def update_epoch_log(self, epoch: int):
        """
                                                                  shape
        steps, rewards, successes, undos               | (num_epochs, num_episodes)
        --------------------------------------------------------------------------------
        cumulated_training_steps, cumulated_test_steps |      (num_epochs,)
        """
        self.cumulated_training_steps += sum(self.training_episode_categories["steps"])
        self.cumulated_test_steps += sum(self.test_episode_categories["steps"])
        self.epoch_training_categories["steps"].append(
            self.training_episode_categories["steps"]
        )
        self.epoch_training_categories["rewards"].append(
            self.training_episode_categories["rewards"]
        )
        self.epoch_training_categories["successes"].append(
            self.training_episode_categories["successes"]
        )
        self.epoch_training_categories["undos"].append(
            self.training_episode_categories["undos"]
        )
        # self.epoch_training_categories["trajectory"].append(self.training_episode_categories["trajectory"])
        self.epoch_test_categories["steps"].append(
            self.test_episode_categories["steps"]
        )
        self.epoch_test_categories["rewards"].append(
            self.test_episode_categories["rewards"]
        )
        self.epoch_test_categories["successes"].append(
            self.test_episode_categories["successes"]
        )
        self.epoch_test_categories["cumulated_training_steps"].append(
            self.cumulated_training_steps
        )
        self.epoch_test_categories["cumulated_test_steps"].append(
            self.cumulated_test_steps
        )
        # self.epoch_test_categories["trajectory"].append(self.test_episode_categories["trajectory"])

        self.epoch_training_categories["pos_rewards"].append(
            self.training_episode_categories["pos_rewards"]
        )
        self.epoch_training_categories["neg_rewards"].append(
            self.training_episode_categories["neg_rewards"]
        )
        self.epoch_test_categories["pos_rewards"].append(
            self.test_episode_categories["pos_rewards"]
        )
        self.epoch_test_categories["neg_rewards"].append(
            self.test_episode_categories["neg_rewards"]
        )

    def update_multiple_run_log(self):
        for log, categories in zip(
            [self.multiple_run_training_categories, self.multiple_run_test_categories],
            [self.epoch_training_categories, self.epoch_test_categories],
        ):
            for key, value in categories.items():
                if key == "cumulated_training_steps" or key == "cumulated_test_steps":
                    log[key].append(value)
                elif key == "trajectory":
                    continue
                else:
                    log[key].append(np.mean(value, axis=1))
                    if key == "successes":
                        log["failure_rate"].append(1 - np.mean(value, axis=1))
        self.run_counter += 1

        if self.run_counter % 2 == 0:
            self.export_multiple_run_data(check_point=True)
            self.export_plot_data(check_point=True)
            self.algorithm.save(
                file_name=self.env.name,
                suffix=self.suffix,
                chkpt=True,
                time_stampt=False,
            )
            print(f"Check point saved at run {self.run_counter}....")

    def export_single_run_data(self, check_point: bool = False):
        training_log, test_log = {}, {}
        for log, categories in zip(
            [training_log, test_log],
            [self.epoch_training_categories, self.epoch_test_categories],
        ):
            for key, value in categories.items():
                if key == "cumulated_training_steps" or key == "cumulated_test_steps":
                    log[key] = value
                elif key == "trajectory":
                    continue
                else:
                    log[key] = np.mean(value, axis=1)
                    if key == "successes":
                        log["failure_rate"] = 1 - log[key]
        folder_name_training = (
            "logs/training_logs/single_run/"
            if not check_point
            else "logs/training_logs/check_point/"
        )
        folder_name_test = (
            "logs/test_logs/single_run/"
            if not check_point
            else "logs/test_logs/check_point/"
        )
        fm.export_data(
            data=training_log,
            folder_name=self.logdir + folder_name_training,
            suffix=self.suffix,
            file_name=self.env.name,
            file_format="csv",
            time_stampt=False,
        )
        fm.export_data(
            data=test_log,
            folder_name=self.logdir + folder_name_test,
            suffix=self.suffix,
            file_name=self.env.name,
            file_format="csv",
            time_stampt=False,
        )

        if not check_point:
            self.algorithm.save(
                file_name=self.env.name, suffix=self.suffix, time_stampt=False
            )

            # fm.export_data(
            #     data=self.epoch_training_categories["trajectory"],
            #     folder_name=self.logdir + "logs/training_logs/",
            #     suffix=self.suffix,
            #     file_name=self.env.name + "_trajectory",
            #     file_format="csv",
            #     time_stampt=False
            # )
            # fm.export_data(
            #     data=self.epoch_test_categories["trajectory"],
            #     folder_name=self.logdir + "logs/test_logs/",
            #     suffix=self.suffix,
            #     file_name=self.env.name + "_trajectory",
            #     file_format="csv",
            #     time_stampt=False
            # )

    def export_multiple_run_data(
        self, time_stampt: bool = False, check_point: bool = False
    ):
        training_log, test_log = {}, {}
        for log, categories in zip(
            [training_log, test_log],
            [self.multiple_run_training_categories, self.multiple_run_test_categories],
        ):
            for key, value in categories.items():
                log[key] = np.mean(value, axis=0)

        folder_name_training = (
            "logs/training_logs/multiple_run/"
            if not check_point
            else "logs/training_logs/multiple_run/check_point/"
        )
        folder_name_test = (
            "logs/test_logs/multiple_run/"
            if not check_point
            else "logs/test_logs/multiple_run/check_point/"
        )
        fm.export_data(
            data=training_log,
            folder_name=self.logdir + folder_name_training,
            suffix=self.suffix,
            file_name=self.env.name,
            file_format="csv",
            time_stampt=time_stampt,
        )
        fm.export_data(
            data=test_log,
            folder_name=self.logdir + folder_name_test,
            suffix=self.suffix,
            file_name=self.env.name,
            file_format="csv",
            time_stampt=time_stampt,
        )

        if not check_point:
            self.algorithm.save(
                file_name=self.env.name, suffix=self.suffix, time_stampt=False
            )

    def export_plot_data(self, check_point: bool = False):
        needed_data = {
            "rewards": "Rewards",
            "steps": "Steps",
            "successes": "Success Rate",
            "failure_rate": "Failure Rate",
            "cumulated_training_steps": "Cumulated Training Steps",
            "cumulated_test_steps": "Cumulated Test Steps",
        }
        needed_data["pos_rewards"] = "Positive Rewards"
        needed_data["neg_rewards"] = "Negative Rewards"

        data = {}

        if self.run_counter > 1:
            export_target = self.multiple_run_test_categories
            folder_name = self.logdir + "plot_data/multiple_run/"
            if check_point:
                folder_name = self.logdir + "plot_data/check_point/"
            for key, value in needed_data.items():
                data[value] = export_target[key]
        else:
            export_target = self.epoch_test_categories
            folder_name = self.logdir + "plot_data/single_run/"
            if check_point:
                folder_name = self.logdir + "plot_data/check_point/"
            for key, value in needed_data.items():
                if key == "cumulated_training_steps" or key == "cumulated_test_steps":
                    data[value] = export_target[key]
                elif key == "trajectory":
                    continue
                elif key == "failure_rate":
                    data[value] = 1 - np.mean(export_target["successes"], axis=1)
                else:
                    data[value] = np.mean(export_target[key], axis=1)
        fm.export_data(
            data=data,
            folder_name=folder_name,
            suffix=self.suffix,
            file_name=self.env.name,
            file_format="pkl",
            time_stampt=False,
        )
