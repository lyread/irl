from argparse import ArgumentParser

from algorithms.agent import Agent
from algorithms.ppo import PPO
from environments.cliff import Cliff
from experiments.experiment_runner import ExperimentRunner
from loggers.logger_cliff import LoggerCliff

if __name__ == "__main__":
    parser = ArgumentParser()
    parser.add_argument(
        "--punishment", type=int, help="if time step punishment is used"
    )
    cml_args = parser.parse_args()

    config_file = "configs/PPO_config.json"
    runner = ExperimentRunner(config_file)
    algorithm = PPO(**runner.algorithm_paras)
    env = Cliff(**runner.environment_paras)
    logger = LoggerCliff(env, algorithm, **runner.logger_paras)
    agent = Agent(env, algorithm, logger, **runner.agent_paras)
    runner.run_experiment(agent)
