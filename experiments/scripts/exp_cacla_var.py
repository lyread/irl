from argparse import ArgumentParser

from algorithms.agent import Agent
from algorithms.cacla_var import CaclaVar
from environments.cliff import Cliff, CliffActionWrapper
from experiments.experiment_runner import ExperimentRunner
from loggers.logger_cliff import LoggerCliff

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--punishment", type=int, help="if time step punishment is used"
    )
    cml_args = parser.parse_args()

    config_file = "configs/CaclaVar_config.json"
    runner = ExperimentRunner(config_file)
    algorithm = CaclaVar(**runner.algorithm_paras)
    env = Cliff(**runner.environment_paras)
    env = CliffActionWrapper(env, low=-1, high=1)
    logger = LoggerCliff(env, algorithm, **runner.logger_paras)
    agent = Agent(env, algorithm, logger, **runner.agent_paras)
    runner.run_experiment(agent)
