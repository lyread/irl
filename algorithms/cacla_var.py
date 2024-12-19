import numpy as np
import torch

from algorithms.cacla import Cacla, CaclaAC
from algorithms.neural_network import MLP, NeuralNetwork


class CaclaVarAC(CaclaAC):
    def __init__(
        self,
        actor: NeuralNetwork,
        critic: NeuralNetwork,
        learning_rate_a: float = 0.0001,
        learning_rate_c: float = 0.0001,
    ):
        super().__init__(actor, critic, learning_rate_a, learning_rate_c)

    def update_actor(self, state, action, n_update: int):
        losses = []
        for _ in range(n_update):
            action_estimate = self.act(state)
            actor_loss = self.loss_fn(action_estimate, action)
            losses.append(actor_loss.item())
            self.actor_optimizer.zero_grad()
            actor_loss.backward()
            self.actor_optimizer.step()
        return np.mean(losses)


class CaclaVar(Cacla):
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
        beta: float = 0.1,
        var: float = 0.1,
        tensorboard: bool = False,
        device: str = "cpu",
        model_path: str = "models/",
    ):
        super().__init__(
            actor_config,
            critic_config,
            name,
            policy,
            exploration,
            action_range,
            discount_factor,
            learning_rate_a,
            learning_rate_c,
            exploration_rate,
            tensorboard,
            device,
            model_path,
        )
        self.beta = beta
        self.var = var

    def make_actor_critic(self):
        if self.policy == "MLP":
            return CaclaVarAC(
                actor=MLP(**self.actor_config, device=self.device),
                critic=MLP(**self.critic_config, device=self.device),
                learning_rate_a=self.learning_rate_a,
                learning_rate_c=self.learning_rate_c,
            )
        else:
            raise NotImplementedError("Not implemented yet!")

    def _update_variance(self, delta: float):
        self.var = (1 - self.beta) * self.var + self.beta * delta**2

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
        delta = td_target.item() - V.item()
        if delta > 0:
            n_update = np.ceil(delta / np.sqrt(self.var))
            actor_loss = self.actor_critic.update_actor(state, action, int(n_update))
            if self.tensorboard:
                self.writer.add_scalar("Actor/Loss", actor_loss, self.learning_iter)
        self._update_variance(delta)
        if self.tensorboard:
            self.writer.add_scalar("Critic/Loss", critic_loss, self.learning_iter)
            self.learning_iter += 1
