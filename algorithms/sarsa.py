import torch

from algorithms.qlearning import QLearning


class Sarsa(QLearning):
    def __init__(
        self,
        name: str = "Sarsa",
        qtable_shape: int | list = [4, 12],
        num_actions: int = 4,
        init_method: str = "zero",
        magnitude: float = 0.01,
        discount_factor: float = 0.6,
        learning_rate: float = 0.1,
        temperature: float = 0.1,
        epsilon: float = 0.1,
        qnet_config: dict = None,
        buffer_capacity: int = 1,
        batch_size: int = 1,
        device: str = "cpu",
        exploration: str = "e-greedy",
        model_path: str = "tmp/logs/q_table/",
        tensorboard: bool = False,
    ):
        super().__init__(
            name=name,
            qtable_shape=qtable_shape,
            num_actions=num_actions,
            init_method=init_method,
            magnitude=magnitude,
            discount_factor=discount_factor,
            learning_rate=learning_rate,
            temperature=temperature,
            epsilon=epsilon,
            qnet_config=qnet_config,
            buffer_capacity=buffer_capacity,
            batch_size=batch_size,
            device=device,
            exploration=exploration,
            model_path=model_path,
            tensorboard=tensorboard,
        )

    def get_q(self, state, action):
        if self.init_method == "nn":
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(action, torch.Tensor):
                action = torch.tensor(action, dtype=torch.int64).to(self.device)
            q_values = self.q_network(state)
            return q_values.gather(-1, action).squeeze(-1)
        else:
            return self.q_table[*state, action]

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
        if self.init_method == "nn":
            self.replay_buffer.add(
                state, action, reward, next_state, next_action, terminated
            )
            if self.replay_buffer.is_full():
                self.learning_iter += 1
                mini_batch = self.replay_buffer.sample(self.batch_size)
                (
                    state_tensor,
                    action_tensor,
                    reward_tensor,
                    next_state_tensor,
                    next_action_tensor,
                    terminated_tensor,
                ) = self._get_tensor(mini_batch)
                Q = self.get_q(state_tensor, action_tensor)
                Q_prime = self.get_q(next_state_tensor, next_action_tensor)
                # Update Q-Netz
                target = (
                    reward_tensor
                    + (1 - terminated_tensor) * self.discount_factor * Q_prime
                )
                loss = self.q_network.update_qnet(Q, target)
                self.writer.add_scalar("Loss", loss, self.learning_iter)

        else:
            Q = self.get_q(state, action)
            # recompute Q_prime
            Q_prime = self.get_q(next_state, next_action)
            # Update Q value
            td_target = reward + (1 - terminated) * self.discount_factor * Q_prime
            td_error = td_target - Q
            self.q_table[*state, action] += self.learning_rate * td_error
