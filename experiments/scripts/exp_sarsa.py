from argparse import ArgumentParser

from algorithms.agent import Agent
from algorithms.sarsa import Sarsa
from environments.cliff import Cliff
from experiments.experiment_runner import ExperimentRunner
from loggers.logger_cliff import LoggerCliff

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--algo", type=str, help="algorithm to use : sarsa or dsn", default="sarsa"
    )
    parser.add_argument(
        "--punishment",
        type=int,
        help=" --punishment=1: time step punishment is used; --punishment=0: time step punishment is not used",
    )
    cml_args = parser.parse_args()

    config_file = (
        "configs/Sarsa_config.json"
        if cml_args.algo == "sarsa"
        else "configs/DSN_config.json"
    )
    runner = ExperimentRunner(config_file)
    if cml_args.punishment is not None:
        runner.environment_paras["punishment"] = bool(cml_args.punishment)
    algorithm = Sarsa(**runner.algorithm_paras)
    env = Cliff(**runner.environment_paras)
    logger = LoggerCliff(env, algorithm, **runner.logger_paras)
    agent = Agent(env, algorithm, logger, **runner.agent_paras)
    runner.run_experiment(agent)
