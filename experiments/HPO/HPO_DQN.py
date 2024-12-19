#!/bin/env python
import time
from collections import defaultdict

import numpy as np
import pandas as pd
from ConfigSpace import CategoricalHyperparameter, ConfigurationSpace
from sklearn.metrics import auc
from smac import Callback, HyperparameterOptimizationFacade, RunHistory, Scenario

import utils.file_manipulation as fm
from algorithms.qlearning import QLearning
from environments.cliff import Cliff

EPISODE_LENGTH = 30
NUM_EPISODES = 8000  # 1000
TEST_EVERY = 100
NUM_TRIALS = 500
RUN_ON_SERVER = True

best_reward = [-200]
result = defaultdict(list)


class MyCallback(Callback):
    def __init__(self, num_trials):
        self.num_trials = num_trials
        self.trials_counter = 0
        self.time = time.time()

    def on_start(self, smbo):
        print(f"Start! {self.num_trials - self.trials_counter} trials ready to go!")

    def on_tell_end(self, smbo, info, value) -> bool | None:
        self.trials_counter += 1
        if self.trials_counter % 10 == 0:
            print(
                f"Evaluated {self.trials_counter} trials so far, {self.num_trials - self.trials_counter} trials left."
            )
            print(f"Time consumption:{time.time() - self.time} seconds")

            incumbent = smbo.intensifier.get_incumbent()
            print("*" * 50)
            assert incumbent is not None
            print("Current incumbent: ")
            for key, hp in incumbent.items():
                print(f"{key}: {hp}")
            print(f"Current incumbent AUC: {smbo.runhistory.get_cost(incumbent)}")
            print("*" * 50)


def train(config, seed=0):
    env = Cliff(punishment=True)
    algorithm = QLearning(
        name="DQN",
        state_size_vector=[env.height, env.width],
        num_actions=env.action_space.n,
        init_method="nn",
        exploration="e-greedy",
        discount_factor=config["discount_factor"],
        learning_rate=config["learning_rate"],
        epsilon=config["epsilon"],
        hidden_dim=config["hidden_dim"],
        buffer_capacity=config["buffer_capacity"],
        batch_size=config["batch_size"],
        random_seed=seed,
    )
    algorithm.set_random_seed()
    # -----------------train-----------------
    test_every = TEST_EVERY
    episode_steps = 0
    cumulated_reward = []
    cumulated_training_steps = []
    collect_experience(env, algorithm)
    for episode in range(NUM_EPISODES):
        # for episode in tqdm(range(NUM_EPISODES), desc=f"Training",total=NUM_EPISODES):
        state_current = env.reset()
        # Choose action for the first iteration
        action_current = algorithm.select_action(
            state_current, method=algorithm.exploration
        )
        step = 0
        done = False
        while not done and step < EPISODE_LENGTH:
            # Get to be updated Q value and corresponding state and action
            # Take the action according to the to be updated Q
            state_next, reward, done, _ = env.step(action_current)
            action_next = algorithm.select_action(
                state_next, method=algorithm.exploration
            )

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
            # if episode > 1200:
            #     test_every = 800
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

            sum_reward_les = sum(reward_ls)
            best_reward[0] = (
                sum_reward_les if sum_reward_les > best_reward[0] else best_reward[0]
            )
            # print(
            #     f"cumulated rewards: {sum_reward_les}; steps: {step}; best_reward: {best_reward[0]}"
            # )
            cumulated_reward.append(sum_reward_les)

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
    result = auc(cumulated_training_steps, cumulated_reward)
    return -result


def save_result_as_pickle(result, folder_name, file_name):
    # normalize score
    # max_value = max(result["score"])
    # min_value = min(result["score"])
    # result["score"] = [
    #     (value - min_value) / (max_value - min_value) for value in result["score"]
    # ]
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


def collect_experience(env, algorithm: QLearning):
    """
    fill replay buffer without learning
    """
    while len(algorithm.replay_buffer) < algorithm.buffer_capacity:
        state = env.reset()
        done = False
        counter = 0
        while not done and counter <= 30:
            action = algorithm.select_action(state, method=algorithm.exploration)
            state_next, reward, done, _ = env.step(action)
            counter += 1
            algorithm.replay_buffer.append((state, action, reward, state_next, 0, done))
            state = state_next


if __name__ == "__main__":
    tic = time.perf_counter()
    cs = ConfigurationSpace()

    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "learning_rate", np.arange(0.001, 0.11, 0.001).tolist(), default_value=0.001
        )
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "discount_factor", np.arange(0.9, 0.009, -0.01).tolist(), default_value=0.9
        )
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "epsilon", np.arange(0.1, 0.7, 0.01).tolist(), default_value=0.1
        )
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "buffer_capacity", [i for i in range(2000, 10001, 1000)], default_value=2000
        )
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "batch_size", [i for i in range(16, 257, 8)], default_value=64
        )
    )
    cs.add_hyperparameter(
        CategoricalHyperparameter(
            "hidden_dim", [i for i in range(16, 129, 8)], default_value=64
        )
    )

    scenario = Scenario(cs, deterministic=True, n_trials=NUM_TRIALS, n_workers=1)
    smac = HyperparameterOptimizationFacade(
        scenario=scenario,
        target_function=train,
        overwrite=True,
        callbacks=[MyCallback(NUM_TRIALS)],
        logging_level=99999,
    )
    incumbent = smac.optimize()
    toc = time.perf_counter()
    run_history: RunHistory = smac.runhistory
    configs = run_history.get_configs()
    for config in configs:
        for key in config.keys():
            result[key].append(config.get(key))
        result["score"].append(run_history.get_cost(config))
    folder_name = (
        "/home/kaixing.xiao/RLFramework/smac_results"
        if RUN_ON_SERVER
        else "./smac_results"
    )
    save_result_as_pickle(result, folder_name, "smac_result_dqn_pT")

    print("-" * 50)
    print("best configuration :")
    for key, value in incumbent.items():
        print(f"{key}: {value}")
    print(f"best score: {run_history.get_cost(incumbent)}")
    print(f"Time: {toc - tic} seconds")
