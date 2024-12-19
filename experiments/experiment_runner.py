#!/usr/bin/env python
import json
import random

import numpy as np
import torch
from tqdm import tqdm

import algorithms.utils.file_manipulation as fm
from algorithms.agent import Agent


class ExperimentRunner:
    def __init__(self, config_file: str):
        with open(config_file) as f:
            self.config_file: dict = json.load(f)
        self.algorithm_paras = self._load_algorithm_parameters()
        self.environment_paras = self._load_environment_parameters()
        self.logger_paras = self._load_logger_parameters()
        self.agent_paras = self._load_agent_parameters()
        self.num_episodes = self.config_file["Experiment"].get("num_episodes")
        self.num_runs = self.config_file["Experiment"].get("num_runs")

    def _load_algorithm_parameters(self):
        return self.config_file.get("Algorithm", {})

    def _load_environment_parameters(self):
        return self.config_file.get("Environment", {})

    def _load_agent_parameters(self):
        return self.config_file.get("Agent", {})

    def _load_logger_parameters(self):
        if logger_paras := self.config_file.get("Logger"):
            return logger_paras
        else:
            print("No logger config found in the config file")
            print("Using basic logger instead")

    def _set_random_seed(self, seed: int = 0):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)

    def run_experiment(self, agent: Agent, seeds: int | list[int] | None = None):
        if self.num_runs == 1:
            if seeds:
                if isinstance(seeds, int):
                    self._set_random_seed(seeds)
                else:
                    raise ValueError(
                        "Only one seed is allowed for single run experiment!"
                    )
            agent.logger.init_logger()
            agent.train(self.num_episodes)
            agent.logger.export_single_run_data()
        else:
            for run in tqdm(range(self.num_runs), desc="Runs", unit="run"):
                if seeds:
                    if not isinstance(seeds, list):
                        raise ValueError(
                            "A list of seeds is required for multiple run experiment!"
                        )
                    elif len(seeds) < self.num_runs:
                        raise ValueError(
                            "Not enough seeds for multiple run experiment!"
                        )
                    else:
                        self._set_random_seed(seeds[run])
                agent.logger.init_logger()
                agent.train(self.num_episodes)
                agent.logger.update_multiple_run_log()
                agent.algorithm.reset()
            agent.logger.export_multiple_run_data()
        agent.logger.export_plot_data()
        print("Experiment completed!")

    def save_experiment_config(
        self,
        folder_name,
        file_name,
        suffix,
        time_stampt=False,
    ):
        folder_name = fm.standardize_folder(folder_name)
        folder = fm.create_folder(folder_name)
        filename = fm.create_filename(
            folder,
            filename=file_name,
            file_format="csv",
            suffix=suffix,
            time_stampt=time_stampt,
        )
        with open(filename, "w") as f:
            json.dump(self.config_file, f, indent=4)
