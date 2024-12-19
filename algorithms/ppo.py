import torch
from tensorboardX import SummaryWriter
from torch.distributions.categorical import Categorical

import utils.file_manipulation as fm
from algorithms.neural_network import MLP, ActorCritic, NeuralNetwork
from algorithms.rl_algorithm import ReinforcementLearningAlgorithm
from algorithms.storage import Rollout


class PPOAC(ActorCritic):
    def __init__(
        self,
        actor: NeuralNetwork,
        critic: NeuralNetwork,
        learning_rate_a: float = 0.0003,
        learning_rate_c: float = 0.0003,
    ):
        super().__init__(actor, critic, learning_rate_a, learning_rate_c)

    def update_actor(self, states, actions, policy_clip, old_log_probs, advantages):
        # compute ppo loss
        new_probs = self.act(states)
        dist = Categorical(new_probs)
        new_log_prob = dist.log_prob(actions)
        prob_ratio = (new_log_prob - old_log_probs).exp()
        weighted_probs = advantages * prob_ratio
        weighted_probs_clipped = (
            torch.clamp(prob_ratio, 1 - policy_clip, 1 + policy_clip) * advantages
        )
        # compute policy entropy
        entropy = dist.entropy().mean()
        # compute actor loss
        actor_loss = -(
            torch.min(weighted_probs, weighted_probs_clipped).mean() + 0.001 * entropy
        )
        self.actor_optimizer.zero_grad()
        actor_loss.backward()
        self.actor_optimizer.step()
        return actor_loss

    def update_critic(self, states, values, advantages):
        critic_values = self.evaluate(states).squeeze()
        returns = values.squeeze() + advantages
        critic_loss = torch.nn.MSELoss()(returns, critic_values)
        self.critic_optimizer.zero_grad()
        critic_loss.backward()
        self.critic_optimizer.step()
        return critic_loss


class PPO(ReinforcementLearningAlgorithm):
    """
    Proximal Policy Optimization (PPO) algorithm implementation.

    Parameters:
        policy (str): The policy architecture to use. Currently supports only "MLP" (Multi-Layer Perceptron).
        actor_config (dict): Configuration parameters for the actor neural network.
        critic_config (dict): Configuration parameters for the critic neural network.
        name (str): Name of the algorithm instance.
        exploration (str): Exploration strategy for action selection.
        learning_rate_a (float): Learning rate for the actor network.
        learning_rate_c (float): Learning rate for the critic network.
        discount_factor (float): Discount factor for future rewards.
        gae_lambda (float): Lambda parameter for Generalized Advantage Estimation (GAE).
        policy_clip (float): Policy clipping parameter for PPO.
        rollout_capacity (int): Capacity of the rollout buffer.
        batch_size (int): Batch size for training.
        n_epochs (int): Number of epochs to train the networks on each rollout.
        device (str): Device to run the computations on (e.g., "cpu", "cuda").
        tensorboard (bool): Whether to use Tensorboard for logging.
        model_path (str): Path to save and load the trained models.
    """

    def __init__(
        self,
        policy: str,
        actor_config: dict,
        critic_config: dict,
        name: str = "PPO",
        exploration: str = "epsilon_greedy",
        learning_rate_a: float = 0.0003,
        learning_rate_c: float = 0.0003,
        discount_factor: float = 0.99,
        gae_lambda: float = 0.95,
        policy_clip: float = 0.2,
        rollout_capacity: int = 2048,
        batch_size: int = 512,
        n_epochs: int = 10,
        device="cpu",
        tensorboard: bool = False,
        model_path: str = "model",
    ):
        super().__init__(
            name=name,
            exploration=exploration,
            tensorboard=tensorboard,
        )
        self.policy = policy
        self.learning_rate_a = learning_rate_a
        self.learning_rate_c = learning_rate_c
        self.discount_factor = discount_factor
        self.gae_lambda = gae_lambda
        self.policy_clip = policy_clip
        self.rollout = Rollout(rollout_capacity)
        self.batch_size = batch_size
        self.n_epochs = n_epochs
        self.device = torch.device(device)
        self.actor_config = actor_config
        self.critic_config = critic_config
        self.actor_critic = self._make_actor_critic(actor_config, critic_config)
        self.model_path = model_path

    def _make_actor_critic(
        self, actor_config: dict, critic_config: dict
    ) -> ActorCritic:
        if self.policy == "MLP":
            actor = MLP(**actor_config, device=self.device)
            critic = MLP(**critic_config, device=self.device)
        else:
            raise NotImplementedError("Policy not supported")

        return PPOAC(actor, critic, self.learning_rate_a, self.learning_rate_c)

    def reset(self):
        self.actor_critic = self._make_actor_critic(
            self.actor_config, self.critic_config
        )
        if self.tensorboard:
            self.writer.close()
            self.writer = SummaryWriter()
            self.learning_iter = 0
        self.rollout.clear()

    def save(
        self,
        file_name,
        suffix: str = "",
        chkpt: bool = False,
        time_stampt: bool = False,
    ):
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
        self.actor_critic.load(file_name, resume)

    def select_action(self, state: torch.Tensor, method: str = "ppo") -> int:
        state = torch.as_tensor(state, dtype=torch.float32).unsqueeze(0).to(self.device)
        probs = self.actor_critic.act(state)
        dist = Categorical(probs)
        if method == "greedy":
            action = torch.argmax(probs)
        else:
            action = dist.sample()
        return int(action)

    def _collect_experience(self, state, action, reward, terminated):
        """
        Collect experience tuples and add them to the rollout buffer.

        This method collects experience tuples (state, action, reward, terminated) from
        the environment, computes log probability and current value estimation and
        adds them to the rollout buffer for training.

        Parameters:
            state: Current state of the environment.
            action: Action taken in the current state.
            reward: Reward received for the action.
            terminated: Whether the episode terminated after this step.

        Returns:
            None
        """
        if not isinstance(state, torch.Tensor):
            state = torch.as_tensor(state, dtype=torch.float32).to(self.device)
        if not isinstance(action, torch.Tensor):
            action = torch.as_tensor(action, dtype=torch.int64).to(self.device)
        probs = self.actor_critic.act(state)
        dist = Categorical(probs)
        log_prob = dist.log_prob(action).cpu().item()
        value = self.actor_critic.evaluate(state).cpu().item()
        self.rollout.collect_experience(
            state, action, reward, terminated, log_prob, value
        )

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
        self._collect_experience(state, action, reward, terminated)
        if self.rollout.is_full():
            avg_actor_loss = 0
            avg_critic_loss = 0
            self.rollout.compute_advantage(self.discount_factor, self.gae_lambda)
            self.learning_iter += 1
            for _ in range(self.n_epochs):
                batches = self.rollout.sample(self.batch_size)
                for i in range(len(batches)):
                    # get batch
                    states, actions, old_log_probs, values, advantages = batches[i]
                    states = torch.as_tensor(states, dtype=torch.float32).to(
                        self.device
                    )
                    actions = torch.as_tensor(actions, dtype=torch.int64).to(
                        self.device
                    )
                    old_log_probs = torch.as_tensor(
                        old_log_probs, dtype=torch.float32
                    ).to(self.device)
                    advantages = torch.as_tensor(advantages, dtype=torch.float32).to(
                        self.device
                    )
                    values = (
                        torch.as_tensor(values, dtype=torch.float32)
                        .squeeze()
                        .to(self.device)
                    )
                    # update actor and critic
                    actor_loss = self.actor_critic.update_actor(
                        states, actions, self.policy_clip, old_log_probs, advantages
                    )
                    critic_loss = self.actor_critic.update_critic(
                        states, values, advantages
                    )
                    # log losses
                    avg_actor_loss += actor_loss.item()
                    avg_critic_loss += critic_loss.item()
            avg_actor_loss /= len(batches) * self.n_epochs
            avg_critic_loss /= len(batches) * self.n_epochs
            if self.tensorboard:
                self.writer.add_scalar("Actor Loss", avg_actor_loss, self.learning_iter)
                self.writer.add_scalar(
                    "Critic Loss", avg_critic_loss, self.learning_iter
                )
            self.rollout.clear()
