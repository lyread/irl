import numpy as np
import torch
from overrides import overrides
from tensorboardX import SummaryWriter

from algorithms.utils import file_manipulation as fm

#import utils.file_manipulation as fm
from algorithms.neural_network import MLP
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from algorithms.storage import ReplayBuffer


class QNet:
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        output_dim: int,
        activation_fn: str,
        output_probs: bool = False,
        learning_rate: float = 0.001,
        device: str = "cpu",
    ):
        self.model = MLP(
            input_dim=input_dim,
            hidden_dim=hidden_dim,
            output_dim=output_dim,
            activation_fn=activation_fn,
            output_probs=output_probs,
            device=device,
        )
        self.learning_rate = learning_rate
        self.loss_fn = torch.nn.MSELoss()
        self.optimizer = torch.optim.Adam(
            self.model.parameters(), lr=self.learning_rate
        )

    def __call__(self, state):
        return self.model(state)

    def save(self, file_name, chkpt=False):
        if chkpt:
            torch.save(
                {
                    "model": self.model.model.state_dict(),
                    "optimizer": self.optimizer.state_dict(),
                },
                file_name,
            )
        else:
            self.model.save(file_name)

    def load(self, file_name, resume=False):
        if resume:
            checkpoint = torch.load(file_name)
            self.model.load_state_dict(checkpoint["model"])
            self.optimizer.load_state_dict(checkpoint["optimizer"])
        else:
            self.model.load(file_name)

    def update_qnet(self, Q, target):
        self.optimizer.zero_grad()
        loss = self.loss_fn(Q, target)
        loss.backward()
        self.optimizer.step()
        return loss


class QLearning(ReinforcementLearningAlgorithm):
    def __init__(
        self,
        name: str = "Q_LEARNING",
        qtable_shape: int | list = [4, 12],
        num_actions: int = 4,
        init_method: str = "random_positive",
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
            init_method=init_method,
            exploration=exploration,
            tensorboard=tensorboard,
        )
        self.qtable_shape = qtable_shape
        self.num_actions = num_actions  # Number of possible actions
        self.model_path = model_path
        self.discount_factor = discount_factor  # Discount factor for future rewards
        self.learning_rate = learning_rate  # Learning rate for updating Q-values
        # Tabular Q-learning
        self.temperature = temperature  # Temperature parameter for softmax policy
        self.epsilon = epsilon  # Probability of taking a random action in e-greedy
        self.magnitude = magnitude  # Magnitude for random Q-table
        # DQN
        self.device = torch.device(device)
        self.qnet_config = qnet_config
        self._make_q()
        self.buffer_capacity = buffer_capacity
        self.batch_size = batch_size
        self.replay_buffer = ReplayBuffer(self.buffer_capacity)
        self.action_space_tensor = torch.tensor(
            [[i for i in range(self.num_actions)] for _ in range(self.batch_size)],
            dtype=torch.long,
        ).to(self.device)

    def _make_q(self):
        if self.init_method == "nn":
            self.q_network = QNet(
                **self.qnet_config, learning_rate=self.learning_rate, device=self.device
            )
            self.q_table = None
        else:
            self.q_network = None
            if isinstance(self.qtable_shape, int):
                self.qtable_shape = [self.qtable_shape]
            if (
                self.init_method == "random_positive"
                or self.init_method == "random_negative"
            ):
                self.q_table = self._random_q_table()
            else:
                self.q_table = np.zeros((*self.qtable_shape, self.num_actions))

    def reset(self):
        """
        Initialize the Q-table based on the chosen method.

        Returns:
        - None
        """
        if self.tensorboard:
            self.writer = SummaryWriter()
            self.learning_iter = 0

        self._make_q()
        if self.init_method == "nn":
            self.replay_buffer.clear()

    def _random_q_table(self):
        """
        Create a random Q-table with small random values.

        Parameters:
        - size (tuple): Size of the Q-table.

        Returns:
        - numpy.ndarray: Randomly initialized Q-table.
        """
        size = (*self.qtable_shape, self.num_actions)
        return (
            np.random.uniform(0, self.magnitude, size)
            if self.init_method == "random_positive"
            else np.random.uniform(-self.magnitude, self.magnitude, size)
        )

    def save(self, file_name="", suffix=None, time_stampt=True):
        folder_name = fm.standardize_folder(self.model_path)
        folder = fm.create_folder(folder_name)
        filename = fm.create_filename(
            folder,
            filename=file_name,
            suffix=suffix,
            file_format="",
            time_stampt=time_stampt,
        )
        if self.init_method == "nn":
            self.q_network.save(filename + ".pt")
        else:
            np.save(filename, self.q_table)

    def load(self, file_name=""):
        self.q_network.load(file_name)

    @overrides
    def select_action(self, state, method="e-greedy"):
        """
        Select an action based on the chosen exploration method.

        Parameters:
        - state (tuple): Current state of the agent.
        - method (str): Exploration method ('e-greedy', 'softmax', 'greedy').

        Returns:
        - int: Selected action.
        """
        valid_actions = np.arange(self.num_actions)
        if self.init_method == "nn":
            state_tensor = torch.FloatTensor(state).to(self.device)
            q_values = self.q_network(state_tensor)
            q_values = q_values.cpu().detach().numpy()
        else:
            q_values = self.q_table[*state]

        if method == "e-greedy":
            return self._epsilon_greedy_action(q_values, valid_actions)
        elif method == "softmax":
            return self._softmax_action(q_values, valid_actions)
        elif method == "random":
            return self._random_action(valid_actions)
        else:
            return self._greedy_action(q_values, valid_actions)

    def _epsilon_greedy_action(self, q_value, actions):
        if np.random.uniform(0, 1) < self.epsilon:
            return self._random_action(actions)
        else:
            return self._greedy_action(q_value, actions)

    def _greedy_action(self, q_value, actions):
        max_indices = np.where(q_value == np.max(q_value))[0]
        return actions[np.random.choice(max_indices)]

    def _random_action(self, actions):
        return np.random.choice(actions)

    def _softmax_action(self, q_value, actions):
        """
        Select an action using the softmax exploration strategy.

        Parameters:
        - q_value (numpy.ndarray): Q-values for valid actions.
        - actions (list): List of valid actions.

        Returns:
        - int: Selected action.
        """
        # To avoid overflow or underflow
        q_value -= np.max(q_value)
        exp_values = np.exp(q_value / self.temperature)
        probabilities = exp_values / np.sum(exp_values)
        # Choose an action based on the probabilities
        return actions[np.random.choice(len(q_value), p=probabilities)]

    def _get_tensor(
        self, mini_batch: list[tuple[np.array, int, float, np.array, int, bool]]
    ):
        """
        Convert a mini-batch of transitions into tensors.

        Args:
            mini_batch (list): List of transition tuples, each containing:
                - State (np.array): The current state.
                - Action (int): The action taken in the current state.
                - Reward (float): The reward received.
                - Next State (np.array): The next state.
                - Next Action (int): The action taken in the next state.
                - Done (bool): Whether the episode terminated after this transition.

        Returns:
            Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor]:
                Tuple of tensors containing:
                - State tensor
                - Action tensor
                - Reward tensor
                - Next state tensor
                - Next action tensor
                - Done tensor
        """
        transitions = list(zip(*mini_batch))
        state_tensor = torch.FloatTensor(np.array(transitions[0])).to(self.device)
        action_tensor = (
            torch.tensor(np.array(transitions[1]), dtype=torch.int64)
            .unsqueeze(-1)
            .to(self.device)
        )
        reward_tensor = torch.FloatTensor(np.array(transitions[2])).to(self.device)
        next_state_tensor = torch.FloatTensor(np.array(transitions[3])).to(self.device)
        next_action_tensor = (
            torch.tensor(np.array(transitions[4]), dtype=torch.int64)
            .unsqueeze(-1)
            .to(self.device)
        )
        done_tensor = torch.FloatTensor(np.array(transitions[5])).to(self.device)
        return (
            state_tensor,
            action_tensor,
            reward_tensor,
            next_state_tensor,
            next_action_tensor,
            done_tensor,
        )

    def get_q(self, state, action):
        """
        Get the Q-value for a given state-action pair.

        Args:
            state: The current state.
            action: The action taken in the current state.

        Returns:
            Union[torch.Tensor, float]: The Q-value for the given state-action pair.
                If using a neural network for Q-value approximation, returns a tensor.
                If using a Q-table, returns a float.
        """
        if self.init_method == "nn":
            if not isinstance(state, torch.Tensor):
                state = torch.FloatTensor(state).to(self.device)
            if not isinstance(action, torch.Tensor):
                action = (
                    torch.tensor(action, dtype=torch.int64)
                    .unsqueeze(-1)
                    .to(self.device)
                )
            q_values = self.q_network(state)
            return torch.max(q_values.gather(-1, action), -1)[0]
        else:
            return np.max(self.q_table[*state, action])

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
                    _,
                    done_tensor,
                ) = self._get_tensor(mini_batch)
                Q = self.get_q(state_tensor, action_tensor)
                Q_prime = self.get_q(next_state_tensor, self.action_space_tensor)
                # Update Q-Netz
                target = (
                    reward_tensor + (1 - done_tensor) * self.discount_factor * Q_prime
                )
                loss = self.q_network.update_qnet(Q, target)
                self.writer.add_scalar("Loss", loss, self.learning_iter)

        else:
            valid_actions = np.arange(self.num_actions)
            Q = self.get_q(state, action)
            Q_prime = self.get_q(next_state, valid_actions)
            td_target = reward + (1 - terminated) * self.discount_factor * Q_prime
            td_error = td_target - Q
            self.q_table[*state, action] += (
                self.learning_rate * td_error
            )  # [*state, ...] required python >= 3.11
