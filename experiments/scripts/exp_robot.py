from argparse import ArgumentParser

from algorithms.cacla_var import CaclaVar
from algorithms.interactive_agent import InteractiveAgent
from environments.robot_arms.kuka_lbr_iiwa import KUKA_LBR_IIWA
from environments.robot_arms.nao_4dof import NAO4DoF
from experiments.experiment_runner import ExperimentRunner
from loggers.logger_robotarm import LoggerRobotArm


def get_args():
    arg_parser = ArgumentParser()
    arg_parser.add_argument(
        "--nao",
        action="store_true",
    )
    arg_parser.add_argument(
        "--kuka",
        action="store_true",
    )
    args = arg_parser.parse_args()

    return args


if __name__ == "__main__":
    args = get_args()
    if args.nao:
        config_file = "configs/nao4_config.json"
    elif args.kuka:
        config_file = "configs/kuka7_config.json"
    else:
        raise ValueError("Please specify a robot arm to run the experiment with.")
    runner = ExperimentRunner(config_file)
    if args.nao:
        env = NAO4DoF(**runner.environment_paras)
    elif args.kuka:
        env = KUKA_LBR_IIWA(**runner.environment_paras)
    else:
        raise ValueError(f"No environment found for {args}.")

    # config_file = "configs/kuka7_config.json"
    # runner = ExperimentRunner(config_file)
    # runner.algorithm_paras["tensorboard"] = False
    # runner.environment_paras["dataset_path"] = None
    # runner.environment_paras["training_episodes"] = 100
    # runner.environment_paras["test_episodes"] = 100
    # runner.logger_paras["suffix"] = "test"
    # env = KUKA_LBR_IIWA(**runner.environment_paras)

    algorithm = CaclaVar(**runner.algorithm_paras)
    logger = LoggerRobotArm(env, algorithm, **runner.logger_paras)
    agent = InteractiveAgent(env, algorithm, logger, **runner.agent_paras)
    runner.run_experiment(agent)
