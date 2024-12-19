import numpy as np
import torch
import torch.nn as nn
from overrides import overrides
from tensorboardX import SummaryWriter

import algorithms.utils.file_manipulation as fm
from algorithms.neural_network import MLP, ActorCritic, NeuralNetwork
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm


class CaclaAC(ActorCritic):
    def __init__(
        self,
        actor: NeuralNetwork,
        critic: NeuralNetwork,
        learning_rate_a: float = 0.0001,
        learning_rate_c: float = 0.0001,
    ):
        super().__init__(actor, critic, learning_rate_a, learning_rate_c)
        self.loss_fn = nn.MSELoss()

    def update_actor(self, state, action):
        action_estimate = self.act(state)
        actor_loss = self.loss_fn(action_estimate, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def update_critic(self, state, target):
        estimate = self.evaluate(state)
        critic_loss = self.loss_fn(estimate, target)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss


class Cacla(ReinforcementLearningAlgorithm):
    """
    Continuous Actor-Critic Learning Automaton (CACLA) reinforcement learning algorithm.

    Args:
        actor_config (dict): Configuration dictionary for the actor neural network.
        critic_config (dict): Configuration dictionary for the critic neural network.
        name (str): Name of the algorithm. Defaults to "Cacla".
        policy (str): Policy used by the algorithm. Defaults to "MLP".
        exploration (str): Exploration method. Defaults to "gaussian".
        action_range (tuple): Range of valid actions. Defaults to (-1, 1).
        discount_factor (float): Discount factor for future rewards. Defaults to 0.99.
        learning_rate_a (int): Learning rate for the actor. Defaults to 0.001.
        learning_rate_c (int): Learning rate for the critic. Defaults to 0.001.
        exploration_rate (float): Exploration rate. Defaults to 0.1.
        tensorboard (bool): Whether to use TensorBoard for logging. Defaults to False.
        device (str): Device to use for computations ("cpu" or "cuda"). Defaults to "cpu".
        model_path (str): Path to save/load model files. Defaults to "models/".
    """

    def __init__(
        self,
        actor_config: dict,
        critic_config: dict,
        name: str = "Cacla",
        policy: str = "MLP",
        exploration: str = "gaussian",
        action_range: tuple = (-1, 1),
        discount_factor: float = 0.99,
        learning_rate_a: int = 0.001,
        learning_rate_c: int = 0.001,
        exploration_rate: float = 0.1,
        tensorboard: bool = False,
        device: str = "cpu",
        model_path: str = "models/",
    ):
        super().__init__(
            name=name,
            exploration=exploration,
            tensorboard=tensorboard,
        )
        self.policy = policy
        self.min_action_value = action_range[0]
        self.max_action_value = action_range[1]
        self.discount_factor = discount_factor
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c = learning_rate_c
        self.exploration_rate = exploration_rate
        self.device = torch.device(device)
        self.model_path = fm.standardize_folder(model_path)
        self.actor_critic = self.make_actor_critic()

    def reset(self):
        """
        Reset the algorithm's internal state, including the actor-critic model and any logging-related components.
        """
        if self.tensorboard:
            self.writer = SummaryWriter()
            self.learning_iter = 0
        self.actor_critic = self.make_actor_critic()

    def make_actor_critic(self):
        """
        Create the actor-critic model based on the provided configurations.

        Returns:
            CaclaAC: The initialized actor-critic model.
        """
        if self.policy == "MLP":
            return CaclaAC(
                actor=MLP(**self.actor_config, device=self.device),
                critic=MLP(**self.critic_config, device=self.device),
                learning_rate_a=self.learning_rate_a,
                learning_rate_c=self.learning_rate_c,
            )
        else:
            raise NotImplementedError("Not implemented yet!")

    def save(
        self,
        file_name,
        suffix: str = "",
        chkpt: bool = False,
        time_stampt: bool = False,
    ):
        """
        Save the actor-critic model to a file.

        Args:
            file_name (str): The base name of the file to save.
            suffix (str, optional): Additional suffix for the file name. Defaults to "".
            chkpt (bool, optional): Whether to save as a checkpoint. Defaults to False.
            time_stampt (bool, optional): Whether to include a timestamp in the file name. Defaults to False.
        """
        folder_name = fm.standardize_folder(self.model_path)
        folder = fm.create_folder(folder_name)
        filename = fm.create_filename(
            folder,
            filename=file_name,
            suffix=suffix,
            file_format="",
            time_stampt=time_stampt,
        )
        self.actor_critic.save(filename, chkpt)

    def load(self, file_name: str, resume: bool = False):
        """
        Load the actor-critic model from a file.

        Args:
            file_name (str): The name of the file to load.
            resume (bool, optional): Whether to resume training from the loaded model. Defaults to False.
        """
        self.actor_critic.load(file_name, resume)

    @torch.no_grad()
    @overrides
    def select_action(self, state, method="gaussian"):
        """
        Select an action based on the current state.

        Args:
            state: The current state of the environment.
            method (str, optional): The method used for action selection. Defaults to "gaussian".

        Returns:
            The selected action.
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        action = self.actor_critic.act(state).cpu().numpy()
        action = np.clip(action, self.min_action_value, self.max_action_value)
        if method == "gaussian":
            return self._gaussian_action(action)
        elif method == "e-greedy":
            return self._epsilon_greedy_action(action)
        elif method == "greedy":
            return self._greedy_action(action)
        else:
            return self._random_action()

    def _gaussian_action(self, action) -> float:
        action = np.random.normal(action, self.exploration_rate)
        return np.clip(action, self.min_action_value, self.max_action_value)

    def _epsilon_greedy_action(self, action) -> float:
        if np.random.rand() < self.exploration_rate:
            return self._random_action()
        else:
            return self._greedy_action(action)

    def _random_action(self) -> float:
        return np.random.uniform(self.min_action_value, self.max_action_value)

    def _greedy_action(self, action) -> float:
        return action

    @overrides
    def learn(
        self,
        state,
        action,
        next_state,
        next_action,
        reward,
        terminated,
        truncated,
        info,
    ):
        # Sum up positive reward and punishment
        reward = sum(reward)

        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        reward = torch.tensor(reward, dtype=torch.float32).to(self.device)
        terminated = (
            torch.tensor(terminated, dtype=torch.int64).to(self.device).unsqueeze(0)
        )
        V = self.actor_critic.evaluate(state)
        V_prime = self.actor_critic.evaluate(next_state)
        td_target = (reward + (1 - terminated) * self.discount_factor * V_prime).to(
            self.device
        )
        # update critic using TD error
        critic_loss = self.actor_critic.update_critic(state, td_target)
        # update actor if TD error is positive
        delta = (td_target - V).to(self.device)
        if delta > 0:
            actor_loss = self.actor_critic.update_actor(state, action)
            if self.tensorboard:
                self.writer.add_scalar("Actor/Loss", actor_loss, self.learning_iter)
        if self.tensorboard:
            self.writer.add_scalar("Critic/Loss", critic_loss, self.learning_iter)
            self.learning_iter += 1
