import sys
import os

root_directory = os.path.abspath(os.path.join(os.path.dirname(__file__), "../../"))
sys.path.append(root_directory)

from algorithms.interactive_agent import InteractiveAgent
from algorithms.cacla2c import Cacla2C
from environments.robot_arms.nao_4dof import NAO4DoF
from experiments.experiment_runner import ExperimentRunner
from loggers.logger_robotarm import LoggerRobotArm


if __name__ == "__main__":
    config_file = "experiments/configs/Cacla2C_nao4dof_config_0.json"
    runner = ExperimentRunner(config_file)
    algorithm = Cacla2C(**runner.algorithm_paras)
    env = NAO4DoF(**runner.environment_paras)
    # env = FreezeActions(env=env, mask=[1, 0, 0, 0])
    logger = LoggerRobotArm(env, algorithm, **runner.logger_paras)
    agent = InteractiveAgent(env, algorithm, logger, **runner.agent_paras)
    runner.run_experiment(agent)
