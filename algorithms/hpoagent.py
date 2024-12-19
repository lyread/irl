from functools import partial

from ConfigSpace import Configuration
from sklearn.metrics import auc

from algorithms.agent import Agent
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm


class HPOAgent(Agent):
    def __init__(
        self,
        env,
        partial_algorithm: partial[ReinforcementLearningAlgorithm],
        num_episodes: int,
        max_episode_length: int = 30,
        test_every: int = 10,
    ):
        self.env = env
        self.partial_algorithm = partial_algorithm
        self.max_episode_length = max_episode_length
        self.test_every = test_every
        self.num_episodes = num_episodes

    def _stopping_criteria(self, counter: int, terminated: bool = False) -> bool:
        return False if (not terminated and counter < self.max_episode_length) else True

    def _testing_criteria(self, episode: int) -> bool:
        return episode % self.test_every == 0

    def train(self, config: Configuration, seed: int = 0):
        self.algorithm: ReinforcementLearningAlgorithm = self.partial_algorithm(
            **config
        )
        self.algorithm.set_random_seed(seed)
        cumulated_rewards = []
        cumulated_training_steps = []
        total_steps = 0
        for episode in range(self.num_episodes):
            counter = 0
            stop = False
            test = self._testing_criteria(episode)
            state_current = self.env.reset()
            action_current = self.algorithm.select_action(
                state_current, method=self.algorithm.exploration
            )
            while not stop:
                state_next, reward, terminated, truncated, info = self.env.step(
                    action_current
                )
                counter += 1
                total_steps += 1
                action_next = self.algorithm.select_action(
                    state_next, method=self.algorithm.exploration
                )
                self.algorithm.learn(
                    state=state_current,
                    action=action_current,
                    next_state=state_next,
                    next_action=action_next,
                    reward=reward,
                    terminated=terminated,
                    truncated=truncated,
                    info=info,
                )
                # Update next action and state for next iteration
                action_current = action_next
                state_current = state_next
                stop = self._stopping_criteria(counter, terminated)
                if test:
                    cumulated_training_steps.append(total_steps)
                    test_reward = self._test()
                    cumulated_rewards.append(test_reward)
        return self._get_score(cumulated_training_steps, cumulated_rewards)

    def _test(self):
        state_current = self.env.reset()
        stop = False
        counter = 0
        rewards = 0
        while not stop:
            action = self.algorithm.select_action(state_current, method="greedy")
            state_next, reward, terminated, truncated, info = self.env.step(action)
            counter += 1
            rewards += reward
            state_current = state_next
            stop = self._stopping_criteria(counter, terminated)
        return rewards

    def _get_score(self, cumulated_training_steps, cumulated_rewards):
        max_training_steps = self.max_episode_length * self.num_episodes
        if cumulated_training_steps[-1] < max_training_steps:
            # extend the training steps and rewards to the max_training_steps
            cumulated_training_steps.append(max_training_steps)
            cumulated_rewards.append(cumulated_rewards[-1])
        return -auc(cumulated_training_steps, cumulated_rewards)
