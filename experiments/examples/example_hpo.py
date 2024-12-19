from functools import partial

from ConfigSpace import ConfigurationSpace, Float

from algorithms.hpoagent import HPOAgent
from algorithms.qlearning import QLearning
from environments.cliff import Cliff
from experiments.hpo_runner import HPORunner

if __name__ == "__main__":
    # Define the configuration space
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("learning_rate", (0.01, 0.99)))
    cs.add_hyperparameter(Float("discount_factor", (0.1, 0.9)))
    cs.add_hyperparameter(Float("epsilon", (0.1, 0.99)))
    # Initialize environment
    env = Cliff(punishment=True)
    # Make a partial class with fixed hyperparameters that you do not want to optimize
    # Algorithm will be initialized inside the HPOAgent
    partial_algorithm = partial(
        QLearning,
        name="Q_LEARNING",
        qtable_shape=[4, 12],
        num_actions=4,
        init_method="zero",
        exploration="e-greedy",
    )
    # Build agent
    agent = HPOAgent(
        env, partial_algorithm, num_episodes=10000, max_episode_length=30, test_every=10
    )
    runner = HPORunner(config_space=cs, n_trials=50, n_workers=5)
    runner.run_hpo(
        agent,
        save_result=True,
        folder_name="hpo_result/",
        file_name="hpo_qlearning",
        file_format="pkl",
    )
