import time
from collections import defaultdict

import pandas as pd
from ConfigSpace import (
    CategoricalHyperparameter,
    Configuration,
    ConfigurationSpace,
    Float,
    Integer,
)
from sklearn.metrics import auc
from smac import Callback, HyperparameterOptimizationFacade, RunHistory, Scenario

import utils.file_manipulation as fm
from algorithms.sarsa import Sarsa
from environments.cliff import Cliff

EPISODE_LENGTH = 30
NUM_EPISODES = 8000  # 1000
TEST_EVERY = 100
NUM_TRIALS = 200

result = defaultdict(list)


class MyCallback(Callback):
    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.trials_counter = 0

    def on_start(self, smbo):
        print(f"Start! {self.num_trials - self.trials_counter} trials ready to go!")

    def on_tell_end(self, smbo, info, value) -> bool | None:
        self.trials_counter += 1
        if self.trials_counter % 10 == 0:
            print(
                f"Evaluated {self.trials_counter} trials so far, {self.num_trials - self.trials_counter} trials left."
            )

            incumbent = smbo.intensifier.get_incumbent()
            print("*" * 50)
            assert incumbent is not None
            print("Current incumbent: ")
            for key, value in incumbent.items():
                print(f"{key}: {value}")
            print(f"Current incumbent value: {smbo.runhistory.get_cost(incumbent)}")
            print("*" * 50)


def train(config: Configuration, seed=0):
    env = Cliff(punishment=False)
    algorithm = Sarsa(
        name="SARSA",
        state_size_vector=[env.height, env.width],
        num_actions=env.action_space.n,
        init_method="zero",  # alternative: "zero" for tabular
        policy="e-greedy",
        discount_factor=config["discount_factor"],
        learning_rate=config["learning_rate"],
        epsilon=config["epsilon"],
        random_seed=seed,
    )
    algorithm.set_random_seed()
    # -----------------train-----------------
    test_every = TEST_EVERY
    episode_steps = 0
    cumulated_reward = []
    cumulated_training_steps = []
    # for episode in tqdm(range(NUM_EPISODES), desc=f"Training",total=NUM_EPISODES):
    for episode in range(NUM_EPISODES):
        state_current = env.reset()
        # Choose action for the first iteration
        action_current = algorithm.select_action(state_current, method=algorithm.policy)
        step = 0
        done = False
        while not done and step < EPISODE_LENGTH:
            # Get to be updated Q value and corresponding state and action
            # Take the action according to the to be updated Q
            state_next, reward, done, _ = env.step(action_current)
            action_next = algorithm.select_action(state_next, method=algorithm.policy)
            algorithm.learn(
                state=state_current,
                action=action_current,
                next_state=state_next,
                next_action=action_next,
                reward=reward,
                done=done,
            )
            # Update next action and state for next iteration
            action_current = action_next
            state_current = state_next
            step += 1
        episode_steps += step
        # -----------------test-----------------
        if episode % test_every == 0:
            cumulated_training_steps.append(episode_steps)
            step = 0
            reward_ls = []
            done = False
            state = env.reset()
            while not done and step < EPISODE_LENGTH:
                action = algorithm.select_action(state, method="greedy")
                state_next, reward, done, _ = env.step(action)
                state = state_next
                reward_ls.append(reward)
                step += 1
            # print(f"cummulated rewards: {sum(reward_ls)}; steps: {step}")
            cumulated_reward.append(sum(reward_ls))

    # After training and testing:
    # calculate area
    max_training_steps = EPISODE_LENGTH * NUM_EPISODES
    if cumulated_training_steps[-1] < max_training_steps:
        cumulated_training_steps.append(max_training_steps)  # the very end
        cumulated_reward.append(
            cumulated_reward[-1]
        )  # project last point to the very end
    if not env.punishment:
        cumulated_reward = [reward - 1 for reward in cumulated_reward]
    # special_rewards = True
    # if special_rewards:
    #     interval_size = 3 # look interval_size steps to the left and to the right
    #     for i, reward in enumerate(cumulated_reward):
    #         lower_border = max(0, i-interval_size)
    #         upper_border = min(len(cumulated_reward-1), i+interval_size)
    #         interval = cumulated_reward[lower_border:upper_border]

    area = auc(cumulated_training_steps, cumulated_reward)
    return -area


def save_result_as_pickle(result, folder_name, file_name):
    # normalize score
    # max_value = max(auc_list["score"])
    # min_value = min(auc_list["score"])
    # auc_list["score"] = [(value - min_value) / (max_value - min_value) for value in auc_list["score"]]
    # save
    folder_name = fm.standardize_folder(folder_name)
    folder = fm.create_folder(folder_name)
    file_name = fm.create_filename(
        folder,
        filename=file_name,
        suffix="",
        file_format="pkl",
        time_stampt=False,
    )
    df = pd.DataFrame(result)
    df.to_pickle(file_name)


if __name__ == "__main__":
    cs = ConfigurationSpace()
    cs.add_hyperparameter(Float("learning_rate", (0.01, 0.99)))
    cs.add_hyperparameter(Float("discount_factor", (0.1, 0.99)))
    cs.add_hyperparameter(Float("epsilon", (0.01, 0.99)))
    scenario = Scenario(cs, deterministic=True, n_trials=NUM_TRIALS, n_workers=2)
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train,
        overwrite=True,
        callbacks=[MyCallback(NUM_TRIALS)],
        logging_level=99999,
    )
    tic = time.perf_counter()
    incumbent = smac.optimize()
    toc = time.perf_counter()
    run_history: RunHistory = smac.runhistory
    configs = run_history.get_configs()
    for config in configs:
        for key in config.keys():
            result[key].append(config.get(key))
        result["score"].append(run_history.get_cost(config))
    save_result_as_pickle(result, "smac_tmp", "smac_results_sarsa_pF")

    print("-" * 50)
    print("best configuration :")
    for key, value in incumbent.items():
        print(f"{key}: {value}")
    print(f"best score: {run_history.get_cost(incumbent)}")
    print(f"Time: {toc - tic} seconds")
