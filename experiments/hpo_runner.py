import time
from collections import defaultdict

from ConfigSpace import Configuration, ConfigurationSpace
from smac import Callback, HyperparameterOptimizationFacade, RunHistory, Scenario

import algorithms.utils.file_manipulation as fm
from algorithms.hpoagent import HPOAgent


class BasicCallback(Callback):
    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.trials_counter = 0
        self.time = time.time()

    def on_start(self, smbo):
        print(f"Start! {self.num_trials - self.trials_counter} trials ready to go!")

    def on_tell_end(self, smbo, info, value) -> bool | None:
        self.trials_counter += 1
        sec = time.time() - self.time
        time_consumption = time.strftime(
            "%H hours %M minutes %S seconds", time.gmtime(sec)
        )
        if self.trials_counter % 5 == 0:
            print(
                f"Evaluated {self.trials_counter} trials so far, {self.num_trials - self.trials_counter} trials left."
            )
            print(f"Time consumption:{time_consumption}")

            incumbent = smbo.intensifier.get_incumbent()
            print("*" * 50)
            assert incumbent is not None
            print("Current incumbent: ")
            for key, hp in incumbent.items():
                print(f"{key}: {hp}")
            print(f"Current incumbent AUC: {smbo.runhistory.get_cost(incumbent)}")
            print("*" * 50)


class HPORunner:
    def __init__(
        self,
        config_space: ConfigurationSpace,
        n_trials: int = 100,
        n_workers=1,
        callback: Callback = None,
    ) -> None:
        self.config_space = config_space
        self.n_trials = n_trials
        self.n_workers = n_workers
        self.callback = callback if callback else BasicCallback(n_trials)

    def run_hpo(
        self,
        agent: HPOAgent,
        save_result: bool = True,
        folder_name: str = "hpo_result/",
        file_name: str = "result",
        file_format: str = "pkl",
    ) -> Configuration:
        scenario = Scenario(
            self.config_space,
            deterministic=True,
            n_trials=self.n_trials,
            n_workers=self.n_workers,
        )
        smac = HyperparameterOptimizationFacade(
            scenario=scenario,
            target_function=agent.train,
            callbacks=[self.callback],
        )
        incumbent = smac.optimize()
        if save_result:
            self.save_result(smac.runhistory, folder_name, file_name, file_format)
        return incumbent

    def save_result(
        self, run_history: RunHistory, folder_name, file_name, file_format
    ) -> None:
        result = defaultdict(list)
        configs = run_history.get_configs()
        for config in configs:
            for key in config.keys():
                result[key].append(config.get(key))
            result["score"].append(run_history.get_cost(config))
        fm.export_data(
            result,
            folder_name=folder_name,
            file_name=file_name,
            file_format=file_format,
        )
