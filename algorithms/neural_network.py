from abc import ABC, abstractmethod
from typing import Iterable

import torch
import torch.nn as nn


class NeuralNetwork(nn.Module):
    """
    Base class representing a neural network.

    Args:
        input_dim (int): Dimension of the input.
        output_dim (int): Dimension of the output.
        activation_fn (str, optional): Activation function to use. Defaults to "relu".
    """

    def __init__(self, input_dim, output_dim, activation_fn: str = "relu"):
        super().__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim
        self.model = nn.Sequential()
        self.activation_map = {
            "relu": nn.ReLU(),
            "sigmoid": nn.Sigmoid(),
            "tanh": nn.Tanh(),
            "softplus": nn.Softplus(),
            "softmax": nn.Softmax(dim=-1),
            "selu": nn.SELU(),
        }
        self.activation_fn = self.get_activation_function(activation_fn)

    def get_activation_function(self, activation_fn: str):
        return self.activation_map.get(activation_fn, nn.ReLU())

    def forward(self, x) -> torch.Tensor:
        return self.model(x)

    def save(self, file_name):
        torch.save(self.state_dict(), file_name)

    def load(self, file_name):
        self.load_state_dict(torch.load(file_name))


class MLP(NeuralNetwork):
    """
    Class representing a Multi-Layer Perceptron (MLP) neural network.

    Args:
        input_dim (int): Dimension of the input.
        hidden_dim (Iterable or int): Dimension(s) of the hidden layer(s).
        output_dim (int): Dimension of the output.
        activation_fn (str, optional): Activation function to use. Defaults to "relu".
        output_probs (bool, optional): Whether to output probabilities. Defaults to False.
        device (str, optional): Device to run the network on. Defaults to "cpu".
    """

    def __init__(
        self,
        input_dim: int,
        hidden_dim: Iterable | int,
        output_dim: int,
        activation_fn: str = "relu",
        output_probs: bool = False,
        device: str = "cpu",
    ):
        super().__init__(input_dim, output_dim, activation_fn)
        self.model = nn.Sequential()
        self.output_probs = output_probs
        self.device = torch.device(device)
        try:  # multiple hidden layers
            layout = [input_dim, *hidden_dim, output_dim]
            for i in range(len(layout) - 1):
                self.model.add_module(f"fc_{i}", nn.Linear(layout[i], layout[i + 1]))
                if i != len(layout) - 2:
                    self.model.add_module(f"activation_fn_{i}", self.activation_fn)
        except:  # single hidden layer
            self.model.add_module("fc_IH", nn.Linear(self.input_dim, hidden_dim))
            self.model.add_module("activation_fn_0", self.activation_fn)
            self.model.add_module("fc_HH", nn.Linear(hidden_dim, hidden_dim))
            self.model.add_module("activation_fn_1", self.activation_fn)
            self.model.add_module("fc_HO", nn.Linear(hidden_dim, self.output_dim))
        if self.output_probs:
            self.model.add_module("softmax", nn.Softmax(dim=-1))
        self.to(self.device)


class ActorCritic(ABC):
    """
    Abstract base class representing an Actor-Critic algorithm.

    Args:
        actor (NeuralNetwork): Actor neural network.
        critic (NeuralNetwork): Critic neural network.
        learning_rate_a (float, optional): Learning rate for the actor. Defaults to 0.0001.
        learning_rate_c (float, optional): Learning rate for the critic. Defaults to 0.0001.
    """

    def __init__(
        self,
        actor: NeuralNetwork,
        critic: NeuralNetwork,
        learning_rate_a: float = 0.0001,
        learning_rate_c: float = 0.0001,
    ):
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c = learning_rate_c
        self.actor = actor
        self.critic = critic
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_a
        )
        self.critic_optimizer = torch.optim.Adam(
            self.critic.parameters(), lr=self.learning_rate_c
        )

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def evaluate(self, state: torch.Tensor) -> torch.Tensor:
        return self.critic(state)

    def save(self, file_name, chkpt: bool = False):
        if chkpt:
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic": self.critic.state_dict(),
                    "critic_optimizer": self.critic_optimizer.state_dict(),
                },
                file_name + "_chkpt.pth",
            )
        else:
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "critic": self.critic.state_dict(),
                },
                file_name + ".pth",
            )

    def load(self, file_name: str, resume: bool = False):
        if resume:
            checkpoint = torch.load(file_name)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic.load_state_dict(checkpoint["critic"])
            self.critic_optimizer.load_state_dict(checkpoint["critic_optimizer"])
        else:
            model = torch.load(file_name)
            self.actor.load_state_dict(model["actor"])
            self.critic.load_state_dict(model["critic"])

    @abstractmethod
    def update_actor(self):
        pass

    @abstractmethod
    def update_critic(self):
        pass

class ActorCritic2(ABC):
    """
    Abstract base class representing an Actor-Critic algorithm containing two Critic networks.

    Args:
        actor (NeuralNetwork): Actor neural network.
        critic0 (NeuralNetwork): Critic0 neural network, default value function approximator.
        critic1 (NeuralNetwork): Critic1 neural network, for meta exploration.
        learning_rate_a (float, optional): Learning rate for the actor. Defaults to 0.0001.
        learning_rate_c0 (float, optional): Learning rate for the critic0. Defaults to 0.0001.
        learning_rate_c1 (float, optional): Learning rate for the critic1. Defaults to 0.0001.
    """

    def __init__(
        self,
        actor: NeuralNetwork,
        critic0: NeuralNetwork,
        critic1: NeuralNetwork,
        learning_rate_a: float = 0.0001,
        learning_rate_c1: float = 0.0001,
        learning_rate_c0: float = 0.0001,
    ):
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c0 = learning_rate_c0
        self.learning_rate_c1 = learning_rate_c1
        self.actor = actor
        self.critic0 = critic0
        self.critic1 = critic1
        self.actor_optimizer = torch.optim.Adam(
            self.actor.parameters(), lr=self.learning_rate_a
        )
        self.critic0_optimizer = torch.optim.Adam(
            self.critic0.parameters(), lr=self.learning_rate_c0
        )
        self.critic1_optimizer = torch.optim.Adam(
            self.critic1.parameters(), lr=self.learning_rate_c1
        )

    def act(self, state: torch.Tensor) -> torch.Tensor:
        return self.actor(state)

    def evaluate(self, state: torch.Tensor) -> tuple[torch.Tensor, torch.Tensor]:
        return self.critic0(state), self.critic1(state)

    def save(self, file_name, chkpt: bool = False):
        if chkpt:
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "actor_optimizer": self.actor_optimizer.state_dict(),
                    "critic0": self.critic0.state_dict(),
                    "critic1": self.critic1.state_dict(),
                    "critic0_optimizer": self.critic0_optimizer.state_dict(),
                    "critic1_optimizer": self.critic1_optimizer.state_dict(),
                },
                file_name + "_chkpt.pth",
            )
        else:
            torch.save(
                {
                    "actor": self.actor.state_dict(),
                    "critic0": self.critic0.state_dict(),
                    "critic1": self.critic1.state_dict(),
                },
                file_name + ".pth",
            )

    def load(self, file_name: str, resume: bool = False):
        if resume:
            checkpoint = torch.load(file_name)
            self.actor.load_state_dict(checkpoint["actor"])
            self.actor_optimizer.load_state_dict(checkpoint["actor_optimizer"])
            self.critic0.load_state_dict(checkpoint["critic0"])
            self.critic1.load_state_dict(checkpoint["critic1"])
            self.critic0_optimizer.load_state_dict(checkpoint["critic0_optimizer"])
            self.critic1_optimizer.load_state_dict(checkpoint["critic1_optimizer"])
        else:
            model = torch.load(file_name)
            self.actor.load_state_dict(model["actor"])
            self.critic0.load_state_dict(model["critic0"])
            self.critic1.load_state_dict(model["critic1"])

    @abstractmethod
    def update_actor(self):
        pass

    @abstractmethod
    def update_critic(self):
        pass
