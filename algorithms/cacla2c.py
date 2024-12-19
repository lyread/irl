import numpy as np
import torch
import torch.nn as nn
from overrides import overrides
from tensorboardX import SummaryWriter

import algorithms.utils.file_manipulation as fm
from algorithms.neural_network import MLP, ActorCritic2, NeuralNetwork
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm


class CaclaAC2C(ActorCritic2):
    def __init__(
        self,
        actor: NeuralNetwork,
        critic0: NeuralNetwork,
        critic1: NeuralNetwork,
        learning_rate_a: float = 0.0001,
        learning_rate_c0: float = 0.0001,
        learning_rate_c1: float = 0.0001,
    ):
        super().__init__(
            actor, critic0, critic1, learning_rate_a, learning_rate_c0, learning_rate_c1
        )
        self.loss_fn = nn.MSELoss()

    def update_actor(self, state, action):
        action_estimate = self.act(state)
        actor_loss = self.loss_fn(action_estimate, action)
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def update_critic(self, state, target0, target1):
        estimate0, estimate1 = self.evaluate(state)

        critic0_loss = self.loss_fn(estimate0, target0)
        critic1_loss = self.loss_fn(estimate1, target1)

        self.critic0_optimizer.zero_grad()
        self.critic1_optimizer.zero_grad()

        critic0_loss.backward()
        critic1_loss.backward()

        self.critic0_optimizer.step()
        self.critic1_optimizer.step()

        return critic0_loss, critic1_loss


class Cacla2C(ReinforcementLearningAlgorithm):
    """
    Continuous Actor-Critic Learning Automaton (CACLA) reinforcement learning algorithm with two critic networks.

    Args:
        actor_config (dict): Configuration dictionary for the actor neural network.
        critic0_config (dict): Configuration dictionary for the first critic neural network.
        critic1_config (dict): Configuration dictionary for the second critic neural network.
        name (str): Name of the algorithm. Defaults to "Cacla".
        policy (str): Policy used by the algorithm. Defaults to "MLP".
        exploration (str): Exploration method. Defaults to "gaussian".
        action_range (tuple): Range of valid actions. Defaults to (-1, 1).
        discount_factor (float): Discount factor for future rewards. Defaults to 0.99.
        learning_rate_a (float): Learning rate for the actor. Defaults to 0.001.
        learning_rate_c0 (float): Learning rate for the critic. Defaults to 0.001.
        learning_rate_c1 (float): Learning rate for the critic. Defaults to 0.001.
        exploration_rate (float): Exploration rate. Defaults to 0.1.
        tensorboard (bool): Whether to use TensorBoard for logging. Defaults to False.
        device (str): Device to use for computations ("cpu" or "cuda"). Defaults to "cpu".
        model_path (str): Path to save/load model files. Defaults to "models/".
    """

    def __init__(
        self,
        actor_config: dict,
        critic0_config: dict,
        critic1_config: dict,
        name: str = "Cacla",
        policy: str = "MLP",
        exploration: str = "gaussian",
        action_range: tuple = (-1, 1),
        discount_factor: float = 0.99,
        learning_rate_a: float = 0.001,
        learning_rate_c0: float = 0.001,
        learning_rate_c1: float = 0.001,
        default_scale: float = 1.0,
        pos_intercept: float = 3.0,
        pos_slope: float = 0.0,
        neg_intercept: float = 1.19,
        neg_slope: float = 0.4,
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
        self.critic0_config = critic0_config
        self.critic1_config = critic1_config
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c0 = learning_rate_c0
        self.learning_rate_c1 = learning_rate_c1
        self.default_scale = default_scale
        self.pos_intercept = pos_intercept
        self.pos_slope = pos_slope
        self.neg_intercept = neg_intercept
        self.neg_slope = neg_slope

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
            return CaclaAC2C(
                actor=MLP(**self.actor_config, device=self.device),
                critic0=MLP(**self.critic0_config, device=self.device),
                critic1=MLP(**self.critic1_config, device=self.device),
                learning_rate_a=self.learning_rate_a,
                learning_rate_c0=self.learning_rate_c0,
                learning_rate_c1=self.learning_rate_c1,
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

    def _get_scale(self, v_pos: torch.Tensor, v_neg: torch.Tensor) -> float:
        def pos_func(x):
            return self.pos_intercept + self.pos_slope * x

        def neg_func(x):
            return self.neg_intercept + self.neg_slope * x

        # Tensor to scalar and invert negativity
        v_pos_value = float(v_pos.item())
        v_neg_value = float(v_neg.item())

        if v_neg_value < 0:
            # Apparently, we have not learned the right value yet
            # so we should not scale the action
            return 1

        # Get positive and negativity values
        positivity = pos_func(v_pos_value)
        negativity = neg_func(v_neg_value)

        # Since positivity == negativity for the intersection point,
        # we can just use the difference between the two values
        # to determine the scaling factor
        diff = negativity - positivity

        scale = 1 + diff

        # Ensure minimum scale
        if scale < 0.05:
            scale = 0.05

        return scale

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
        action = self.actor_critic.act(state).numpy()
        V_pos, V_neg = self.actor_critic.evaluate(state)
        V_pos, V_neg = V_pos.numpy(), V_neg.numpy()

        scale = self._get_scale(V_pos, V_neg)

        action = np.clip(action, self.min_action_value, self.max_action_value)
        if method == "gaussian":
            return self._gaussian_action(action, scale)
        elif method == "greedy":
            return self._greedy_action(action)
        else:
            return self._random_action()

    def _gaussian_action(self, action, scale) -> float:
        action = np.random.normal(action, self.default_scale * scale)
        return np.clip(action, self.min_action_value, self.max_action_value)

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
        state = torch.tensor(state, dtype=torch.float32).to(self.device)
        action = torch.tensor(action, dtype=torch.float32).to(self.device)
        next_state = torch.tensor(next_state, dtype=torch.float32).to(self.device)
        pos_reward = torch.tensor(reward[0], dtype=torch.float32).to(self.device)
        neg_reward = torch.tensor(-reward[1], dtype=torch.float32).to(self.device)
        terminated = (
            torch.tensor(terminated, dtype=torch.int64).to(self.device).unsqueeze(0)
        )

        V_pos, V_neg = self.actor_critic.evaluate(state)
        V_pos_prime, V_neg_prime = self.actor_critic.evaluate(next_state)

        td_pos_target = (
            pos_reward + (1 - terminated) * self.discount_factor * V_pos_prime
        ).to(self.device)
        td_neg_target = (
            neg_reward + (1 - terminated) * self.discount_factor * V_neg_prime
        ).to(self.device)

        # update critic using TD error
        critic_pos_loss, critic_neg_loss = self.actor_critic.update_critic(
            state, td_pos_target, td_neg_target
        )

        # update actor if TD error is positive
        delta_pos = (td_pos_target - V_pos).to(self.device)
        delta_neg = (td_neg_target - V_neg).to(self.device)

        if delta_pos - delta_neg > 0:
            actor_loss = self.actor_critic.update_actor(state, action)
            if self.tensorboard:
                self.writer.add_scalar("Actor/Loss", actor_loss, self.learning_iter)
        if self.tensorboard:
            self.writer.add_scalar(
                "CriticPos/Loss", critic_pos_loss, self.learning_iter
            )
            self.writer.add_scalar(
                "CriticNeg/Loss", critic_neg_loss, self.learning_iter
            )
            self.learning_iter += 1
